# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.
# TorchNEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# TorchNEP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with TorchNEP.  If not, see <http://www.gnu.org/licenses/>.

"""Version-3 order/compression descriptor for local magnetic states.

``nep4_spin3`` keeps the validated spin2 grammar and closes its missing
angle-resolved fourth-order exchange sector.  The new sector is evaluated as
Cartesian density Gram contractions, not as explicit spherical harmonics or
neighbor-pair loops.  For radial channels ``a`` and ``b`` it represents

    C_l^{ab} = sum_{j != k} w_a(r_ij) w_b(r_ik)
               (S_i . S_j) (S_i . S_k) P_l(rhat_ij . rhat_ik),

for ``l = 0, 1, 2``.  The ``l=0`` block is exactly the spin2 distinct-neighbor
block.  ``l=1`` uses a three-vector density and ``l=2`` uses the existing
five-component orthonormal symmetric-traceless (STF) representation.  This
keeps work linear in the number of directed neighbor edges.

The class deliberately owns the version-3 schema.  Its spin2 core is an
implementation detail used to preserve the already validated contractions;
callers only see the common descriptor interface.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from . import ops
from .magnetic_polynomial import (
    MagneticPolynomialChannel,
    MagneticPolynomialDescriptor,
    _index_sum,
    _stf5,
)
from .oc_cuda_bank import edge_anchored_sector
from .oc_cuda_bank_v3 import angular_center_v3, density_bank_v3


class MagneticPolynomialDescriptorV3(torch.nn.Module):
    """Version-3 magnetic polynomial descriptor resolved through ``l=2``.

    The public constructor mirrors the spin2 descriptor so the model layer has
    one stable seam.  Version selection is made by the model format, never by
    a research feature flag inside this module.
    """

    def __init__(
        self,
        spin_compress: int = 2,
        spin_l_max: int = 2,
        spin_order: int = 3,
        spin_soc: bool = True,
        basis_size_radial: int = 8,
        cutoff_radial: float = 6.0,
        eps: float = 1.0e-12,
    ):
        super().__init__()
        self.core = MagneticPolynomialDescriptor(
            spin_compress=spin_compress,
            spin_l_max=spin_l_max,
            spin_order=spin_order,
            spin_soc=spin_soc,
            basis_size_radial=basis_size_radial,
            cutoff_radial=cutoff_radial,
            eps=eps,
        )
        self.spin_compress = self.core.spin_compress
        self.spin_l_max = self.core.spin_l_max
        self.spin_order = self.core.spin_order
        self.spin_soc = self.core.spin_soc
        self.basis_size_radial = self.core.basis_size_radial
        self.cutoff_radial = self.core.cutoff_radial
        self.eps = self.core.eps
        self.radial_pairs = self.core.radial_pairs
        self.channel_grammar = self._build_channel_grammar()
        self.schema = self.channel_grammar
        self.dim = self.schema[-1].stop
        self.order_dims = {
            order: max(
                channel.stop
                for channel in self.schema
                if channel.order <= order
            )
            for order in range(1, self.spin_order + 1)
        }

    @property
    def radial_leg_mix(self):
        """The shared learned radial-leg projection used by O2/O3 sectors."""
        return self.core.radial_leg_mix

    @property
    def radial_pair_left(self):
        return self.core.radial_pair_left

    @property
    def radial_pair_right(self):
        return self.core.radial_pair_right

    def _build_channel_grammar(self):
        channels = []
        position = 0
        for channel in self.core.schema:
            if channel.name != "correlation_distinct_neighbor":
                channels.append(replace(
                    channel,
                    start=position,
                    stop=position + channel.width,
                ))
                position += channel.width
                continue
            angular_ranks = range(self.spin_l_max + 1)
            for angular_rank in angular_ranks:
                name = f"correlation_distinct_l{angular_rank}"
                channels.append(MagneticPolynomialChannel(
                    name=name,
                    group="order2_angular_exchange",
                    start=position,
                    stop=position + channel.width,
                    order=channel.order,
                    geometry_rank=angular_rank,
                    spin_degree=channel.spin_degree,
                    topology=channel.topology,
                    minimum_radial_rank=channel.minimum_radial_rank,
                    inversion_parity=channel.inversion_parity,
                    time_reversal_parity=channel.time_reversal_parity,
                ))
                position += channel.width
        return tuple(channels)

    def split_by_token(self, descriptor: torch.Tensor):
        self._validate_dim(descriptor)
        return {
            token.name: descriptor[..., token.start:token.stop]
            for token in self.schema
        }

    def split_by_group(self, descriptor: torch.Tensor):
        self._validate_dim(descriptor)
        grouped = {}
        for token in self.schema:
            grouped.setdefault(token.group, []).append(
                descriptor[..., token.start:token.stop]
            )
        return {name: torch.cat(parts, -1) for name, parts in grouped.items()}

    def _validate_dim(self, descriptor):
        if descriptor.shape[-1] != self.dim:
            raise ValueError(
                f"expected descriptor last dimension {self.dim}, "
                f"got {descriptor.shape[-1]}"
            )

    def _angular_exchange(
        self,
        rij,
        pair_i,
        pair_j,
        spins,
        n_atoms,
        atom_types,
        coefficient,
        spin_dof_type_active,
        spin_env_type_active,
    ):
        pair_width = len(self.radial_pairs)
        extra_ranks = self.spin_l_max
        if pair_i.numel() == 0:
            return spins.new_zeros((n_atoms, pair_width * extra_ranks))

        dof_atoms, env_atoms = self.core._active_masks(
            atom_types,
            spins,
            spin_dof_type_active,
            spin_env_type_active,
        )
        distance = torch.linalg.vector_norm(rij, dim=-1).clamp_min(self.eps)
        rhat = rij / distance.unsqueeze(-1)
        weights, _ = self.core._radial_weights(
            distance, pair_i, pair_j, atom_types, coefficient
        )
        if dof_atoms is not None or env_atoms is not None:
            center_active = (
                torch.ones_like(pair_i, dtype=torch.bool)
                if dof_atoms is None
                else dof_atoms.index_select(0, pair_i)
            )
            neighbor_active = (
                torch.ones_like(pair_j, dtype=torch.bool)
                if env_atoms is None
                else env_atoms.index_select(0, pair_j)
            )
            weights = weights * (
                center_active & neighbor_active
            ).to(weights).unsqueeze(-1)

        si = spins.index_select(0, pair_i)
        sj = spins.index_select(0, pair_j)
        dot = (si * sj).sum(-1)
        weighted_dot = weights * dot.unsqueeze(-1)
        left = self.radial_pair_left
        right = self.radial_pair_right
        angular_basis = [rhat]
        widths = [3]
        if self.spin_l_max >= 2:
            angular_basis.append(_stf5(rhat, rhat))
            widths.append(5)
        joined_basis = torch.cat(angular_basis, -1)
        edge_density = (
            weighted_dot.unsqueeze(-1) * joined_basis.unsqueeze(1)
        ).reshape(pair_i.shape[0], -1)
        same_edge = (
            weights.index_select(1, left)
            * weights.index_select(1, right)
            * dot.square().unsqueeze(-1)
        )
        # One scatter supplies every added angular density and the shared
        # j=k subtraction.  This is the linear-edge-cost Cartesian equivalent
        # of the spherical-harmonic addition theorem.
        accumulated = _index_sum(
            edge_density,
            pair_i,
            torch.cat((edge_density, same_edge), -1),
            n_atoms,
        )
        density_width = self.spin_compress * sum(widths)
        density = accumulated[:, :density_width].reshape(
            n_atoms, self.spin_compress, sum(widths)
        )
        same = accumulated[:, density_width:]
        blocks = []
        start = 0
        for angular_rank, width in enumerate(widths, start=1):
            value = density[..., start:start + width]
            full = (
                value.index_select(1, left)
                * value.index_select(1, right)
            ).sum(-1)
            if angular_rank == 2:
                # For the orthonormal STF representation used by ``_stf5``,
                # (3/2) Q(r):Q(r') = P_2(r.r').
                full = 1.5 * full
            blocks.append(full - same)
            start += width
        return torch.cat(blocks, -1)

    def _splice_angular(self, core, angular):
        legacy = next(
            token for token in self.core.schema
            if token.name == "correlation_distinct_neighbor"
        )
        pair_width = legacy.width
        if angular.shape[-1] != pair_width * self.spin_l_max:
            raise RuntimeError("spin3 angular exchange schema mismatch")
        result = torch.cat((
            core[..., :legacy.start],
            core[..., legacy.start:legacy.stop],
            angular,
            core[..., legacy.stop:],
        ), -1)
        self._validate_dim(result)
        return result

    def _forward_shared_cuda(
        self,
        rij,
        pair_i,
        pair_j,
        spins,
        atom_types,
        coefficient,
        spin_dof_type_active,
        spin_env_type_active,
    ):
        fused = density_bank_v3(
            self.core,
            rij,
            pair_i,
            pair_j,
            spins,
            atom_types,
            coefficient,
            spin_dof_type_active,
            spin_env_type_active,
        )
        if fused is None:
            return None
        primitive, angular_moment, auxiliary = fused
        center = ops.oc_cuda_center_sector(self.core, primitive, spins)
        edge_anchor = edge_anchored_sector(
            self.core, primitive, auxiliary, pair_i
        )
        if center is None or edge_anchor is None:
            return None

        dof_atoms, _ = self.core._active_masks(
            atom_types,
            spins,
            spin_dof_type_active,
            spin_env_type_active,
        )
        active = (
            torch.ones(spins.shape[0], dtype=spins.dtype,
                       device=spins.device)
            if dof_atoms is None
            else dof_atoms.to(spins)
        )
        local = spins.square().sum(-1).mul(active).unsqueeze(-1)
        core = self.core._forward_fused_center(
            primitive, spins, local, center, edge_anchor
        )

        same_offset = 38 * self.spin_compress
        pair_width = len(self.radial_pairs)
        same = primitive[:, same_offset:same_offset + pair_width]
        angular = angular_center_v3(angular_moment, same)
        if angular is None:
            moment = angular_moment.reshape(spins.shape[0], 2, 8)
            left = self.radial_pair_left
            right = self.radial_pair_right
            first = (
                moment[:, :, :3].index_select(1, left)
                * moment[:, :, :3].index_select(1, right)
            ).sum(-1)
            second = 1.5 * (
                moment[:, :, 3:].index_select(1, left)
                * moment[:, :, 3:].index_select(1, right)
            ).sum(-1)
            angular = torch.cat((first - same, second - same), -1)
        return self._splice_angular(core, angular)

    def forward(
        self,
        rij: torch.Tensor,
        pair_i: torch.Tensor,
        pair_j: torch.Tensor,
        spins: torch.Tensor,
        N: int,
        atom_types: torch.Tensor | None = None,
        c_spin_2body: torch.Tensor | None = None,
        c_spin_3body: torch.Tensor | None = None,
        spin_dof_type_active: torch.Tensor | None = None,
        spin_env_type_active: torch.Tensor | None = None,
        rij_coupled: torch.Tensor | None = None,
        pair_i_coupled: torch.Tensor | None = None,
        pair_j_coupled: torch.Tensor | None = None,
    ):
        if (
            pair_i.numel()
            and c_spin_3body is None
            and rij_coupled is None
            and pair_i_coupled is None
            and pair_j_coupled is None
        ):
            fused = self._forward_shared_cuda(
                rij,
                pair_i,
                pair_j,
                spins,
                atom_types,
                c_spin_2body,
                spin_dof_type_active,
                spin_env_type_active,
            )
            if fused is not None:
                return fused
        core = self.core(
            rij,
            pair_i,
            pair_j,
            spins,
            N,
            atom_types,
            c_spin_2body,
            c_spin_3body,
            spin_dof_type_active,
            spin_env_type_active,
            rij_coupled,
            pair_i_coupled,
            pair_j_coupled,
        )
        if self.spin_order < 2 or self.spin_l_max == 0:
            return core
        angular = self._angular_exchange(
            rij,
            pair_i,
            pair_j,
            spins,
            N,
            atom_types,
            c_spin_2body,
            spin_dof_type_active,
            spin_env_type_active,
        )
        return self._splice_angular(core, angular)
