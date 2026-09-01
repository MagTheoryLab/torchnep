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

"""Order/compression (O/C) descriptor for local magnetic states.

The implementation builds a shared bank of irreducible edge densities and
contracts that bank through an order-nested grammar.  ``spin_order`` controls
the maximum nested primitive-contraction level O and ``spin_compress`` controls
the shared radial rank C.  O is a hierarchy of explicitly supplied primitive
invariants, not the body order of the final energy: a nonlinear ANN may combine
descriptor coordinates across levels.  The same interface is used by training,
inference, serialization and future generated CUDA implementations.

The descriptor is a symmetry-respecting local approximation to a DFT energy
surface.  It is not a fixed spin Hamiltonian and it does not claim exact DFT
equivalence at finite cutoff, radial rank, angular rank or contraction level.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from . import ops
from .oc_cuda_bank import density_bank as _oc_cuda_density_bank
from .oc_cuda_center import center_sector as _oc_cuda_center_sector
from .oc_cuda_full import full_descriptor as _oc_cuda_full_descriptor
from .oc_cuda_train import moment_gate as _oc_cuda_moment_gate


@dataclass(frozen=True)
class MagneticPolynomialChannel:
    """Metadata for one generated scalar contraction family.

    ``minimum_radial_rank`` is the algebraic lower bound needed to keep the
    contraction from vanishing solely because nominally independent legs
    collapse onto the same rank-one projection.  Tokens above the configured
    C are removed from both the schema and the forward computation.
    """

    name: str
    group: str
    start: int
    stop: int
    order: int
    geometry_rank: int
    spin_degree: int
    topology: str
    minimum_radial_rank: int = 1
    inversion_parity: int = 1
    time_reversal_parity: int = 1

    @property
    def width(self) -> int:
        return self.stop - self.start


def _index_sum(
    reference: torch.Tensor,
    index: torch.Tensor,
    values: torch.Tensor,
    size: int,
) -> torch.Tensor:
    out = reference.new_zeros((size, values.shape[-1]))
    if index.numel():
        out.index_add_(0, index, values)
    return out


def _stf5(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Five orthonormal components of STF(sym(left outer right))."""

    xx = left[..., 0] * right[..., 0]
    yy = left[..., 1] * right[..., 1]
    zz = left[..., 2] * right[..., 2]
    xy = 0.5 * (
        left[..., 0] * right[..., 1]
        + left[..., 1] * right[..., 0]
    )
    yz = 0.5 * (
        left[..., 1] * right[..., 2]
        + left[..., 2] * right[..., 1]
    )
    zx = 0.5 * (
        left[..., 2] * right[..., 0]
        + left[..., 0] * right[..., 2]
    )
    return torch.stack(
        (
            (2.0 * zz - xx - yy) / math.sqrt(6.0),
            (xx - yy) / math.sqrt(2.0),
            math.sqrt(2.0) * xy,
            math.sqrt(2.0) * yz,
            math.sqrt(2.0) * zx,
        ),
        dim=-1,
    )


def _stf5_to_matrix(value: torch.Tensor) -> torch.Tensor:
    q0, q1, q2, q3, q4 = value.unbind(-1)
    xx = -q0 / math.sqrt(6.0) + q1 / math.sqrt(2.0)
    yy = -q0 / math.sqrt(6.0) - q1 / math.sqrt(2.0)
    zz = 2.0 * q0 / math.sqrt(6.0)
    xy = q2 / math.sqrt(2.0)
    yz = q3 / math.sqrt(2.0)
    zx = q4 / math.sqrt(2.0)
    return torch.stack(
        (xx, xy, zx, xy, yy, yz, zx, yz, zz), dim=-1
    ).reshape(value.shape[:-1] + (3, 3))


class MagneticPolynomialDescriptor(torch.nn.Module):
    """Order-nested local magnetic polynomial descriptor.

    Parameters have deliberately narrow semantics:

    ``spin_compress``
        Latent radial/CP rank.  Higher rank increases radial resolution of
        every contraction without exposing individual physics blocks.
    ``spin_l_max``
        Maximum geometry-spin irreducible rank (currently 0, 1 or 2).
    ``spin_order``
        Maximum nested primitive-contraction level.  Order 1 contains
        local/direct-edge response, order 2 adds two-density and environment
        response, and order 3 adds triple-density and dot-gated response.
        Lower orders are literal output prefixes of higher orders at fixed C.
        This is not a body-order label for the final nonlinear ANN energy.
    ``spin_soc``
        If false, space and spin may rotate independently.  If true, joint
        O(3) contractions with even inversion and time-reversal parity are
        enabled.

    The forward interface mirrors the production spin descriptor seam, but
    the implementation and channel layout are independent.
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
        if spin_compress < 1:
            raise ValueError("spin_compress must be >= 1")
        if spin_l_max not in (0, 1, 2):
            raise ValueError("spin_l_max must be 0, 1 or 2")
        if spin_order not in (1, 2, 3):
            raise ValueError("spin_order must be 1, 2 or 3")
        if basis_size_radial + 1 < spin_compress:
            raise ValueError("basis_size_radial must cover spin_compress")
        if cutoff_radial <= 0.0:
            raise ValueError("cutoff_radial must be > 0")
        self.spin_compress = int(spin_compress)
        self.spin_l_max = int(spin_l_max)
        self.spin_order = int(spin_order)
        self.spin_soc = bool(spin_soc)
        self.basis_size_radial = int(basis_size_radial)
        self.cutoff_radial = float(cutoff_radial)
        self.eps = float(eps)
        self.radial_pairs = tuple(
            (left, right)
            for left in range(self.spin_compress)
            for right in range(left, self.spin_compress)
        )
        self.register_buffer(
            "radial_pair_left",
            torch.tensor(
                [pair[0] for pair in self.radial_pairs], dtype=torch.long
            ),
            persistent=False,
        )
        self.register_buffer(
            "radial_pair_right",
            torch.tensor(
                [pair[1] for pair in self.radial_pairs], dtype=torch.long
            ),
            persistent=False,
        )

        # Four CP legs are shared by all contraction paths.  Identity/cyclic
        # initialization gives deterministic, non-collinear legs at C >= 3.
        leg_mix = torch.zeros(4, self.spin_compress, self.spin_compress)
        for leg in range(4):
            for channel in range(self.spin_compress):
                source = (channel + leg) % self.spin_compress
                leg_mix[leg, channel, source] = 1.0
        self.radial_leg_mix = torch.nn.Parameter(leg_mix)
        if not (
            self.spin_soc and self.spin_order >= 2 and self.spin_l_max >= 1
        ):
            self.radial_leg_mix.requires_grad_(False)

        self.channel_grammar = self._build_channel_grammar()
        self.schema = self.channel_grammar
        self._channel_names = frozenset(
            channel.name for channel in self.channel_grammar
        )
        self.dim = self.channel_grammar[-1].stop
        self.order_dims = {
            order: max(
                channel.stop
                for channel in self.channel_grammar
                if channel.order <= order
            )
            for order in range(1, self.spin_order + 1)
        }

    def _build_channel_grammar(self):
        channels = []
        position = 0

        def add(
            name,
            group,
            width,
            order,
            geometry_rank,
            spin_degree,
            topology,
            minimum_radial_rank=1,
        ):
            nonlocal position
            if minimum_radial_rank < 1:
                raise ValueError("minimum_radial_rank must be >= 1")
            if self.spin_compress < minimum_radial_rank:
                return
            channels.append(
                MagneticPolynomialChannel(
                    name=name,
                    group=group,
                    start=position,
                    stop=position + width,
                    order=order,
                    geometry_rank=geometry_rank,
                    spin_degree=spin_degree,
                    topology=topology,
                    minimum_radial_rank=minimum_radial_rank,
                )
            )
            position += width

        width = self.spin_compress
        add("local_s2", "order1_local", 1, 1, 0, 2, "center")
        add("edge_l0_dot", "order1_edge", width, 1, 0, 2, "edge_sum")
        add(
            "edge_l0_neighbor_s2",
            "order1_edge",
            width,
            1,
            0,
            2,
            "edge_sum",
        )
        if self.spin_soc and self.spin_l_max >= 2:
            add(
                "edge_l2_pair",
                "order1_joint_soc",
                width,
                1,
                2,
                2,
                "edge_sum",
            )
            add(
                "center_l2_environment",
                "order1_joint_soc",
                width,
                1,
                2,
                2,
                "center_environment",
            )

        if self.spin_order >= 2:
            add("edge_l0_dot2", "order2_edge", width, 2, 0, 4, "edge_sum")
            add(
                "density_l0_self",
                "order2_density",
                width,
                2,
                0,
                2,
                "density_pair",
            )
            if self.spin_l_max >= 1:
                if self.spin_soc:
                    for name, rank in (
                        ("density_l1_longitudinal_self", 0),
                        ("density_l1_axial_self", 1),
                        ("density_l1_stf_self", 2),
                    ):
                        add(
                            name,
                            "order2_density",
                            width,
                            2,
                            rank,
                            2,
                            "density_pair",
                        )
                else:
                    add(
                        "density_l1_product_self",
                        "order2_density",
                        width,
                        2,
                        1,
                        2,
                        "density_pair",
                    )
            if self.spin_l_max >= 2:
                add(
                    "density_l2_product_self",
                    "order2_density",
                    width,
                    2,
                    2,
                    2,
                    "density_pair",
                )
            add(
                "density_l0_dot_response",
                "order2_response",
                width,
                2,
                0,
                4,
                "density_pair",
            )
            add(
                "correlation_same_edge",
                "order2_correlation",
                len(self.radial_pairs),
                2,
                0,
                4,
                "same_edge",
            )
            add(
                "correlation_distinct_neighbor",
                "order2_correlation",
                len(self.radial_pairs),
                2,
                0,
                4,
                "distinct_neighbor",
            )
            if self.spin_soc and self.spin_l_max >= 1:
                add(
                    "coupling_l11_axial",
                    "order2_joint_soc",
                    width,
                    2,
                    1,
                    2,
                    "density_response",
                    minimum_radial_rank=2,
                )
                add(
                    "edge_l11_axial",
                    "order2_joint_soc",
                    width,
                    2,
                    1,
                    2,
                    "edge_anchored_density",
                )
            if self.spin_soc and self.spin_l_max >= 2:
                add(
                    "coupling_l22_axial",
                    "order2_joint_soc",
                    width,
                    2,
                    2,
                    2,
                    "density_response",
                    minimum_radial_rank=2,
                )
                add(
                    "edge_l22_axial",
                    "order2_joint_soc",
                    width,
                    2,
                    2,
                    2,
                    "edge_anchored_density",
                )

        if self.spin_order >= 3:
            add(
                "edge_l0_moment_gate",
                "order3_response",
                width,
                3,
                0,
                4,
                "edge_sum",
            )
            if self.spin_soc and self.spin_l_max >= 1:
                add(
                    "coupling_l11_dot_response",
                    "order3_joint_soc",
                    width,
                    3,
                    1,
                    4,
                    "density_response",
                    minimum_radial_rank=2,
                )
                add(
                    "coupling_l111_p_m_x",
                    "order3_joint_soc",
                    width,
                    3,
                    1,
                    2,
                    "triple_density",
                )
            if self.spin_soc and self.spin_l_max >= 2:
                add(
                    "coupling_l22_dot_response",
                    "order3_joint_soc",
                    width,
                    3,
                    2,
                    4,
                    "density_response",
                    minimum_radial_rank=2,
                )
                add(
                    "coupling_l111_p_qs_x",
                    "order3_joint_soc",
                    width,
                    3,
                    2,
                    2,
                    "triple_density",
                )
                add(
                    "coupling_l112_edge_response",
                    "order3_joint_soc",
                    width,
                    3,
                    2,
                    2,
                    "edge_anchored_triple_density",
                    minimum_radial_rank=2,
                )
            if self.spin_soc and self.spin_l_max >= 1:
                add(
                    "coupling_l111_bulk",
                    "order3_joint_soc",
                    width,
                    3,
                    1,
                    2,
                    "edge_anchored_triple_density",
                    minimum_radial_rank=3,
                )
        return tuple(channels)

    def split_by_token(self, descriptor: torch.Tensor):
        self._validate_dim(descriptor)
        return {
            token.name: descriptor[..., token.start : token.stop]
            for token in self.schema
        }

    def split_by_group(self, descriptor: torch.Tensor):
        self._validate_dim(descriptor)
        grouped = {}
        for token in self.schema:
            grouped.setdefault(token.group, []).append(
                descriptor[..., token.start : token.stop]
            )
        return {name: torch.cat(parts, -1) for name, parts in grouped.items()}

    def _validate_dim(self, descriptor):
        if descriptor.shape[-1] != self.dim:
            raise ValueError(
                f"expected descriptor last dimension {self.dim}, "
                f"got {descriptor.shape[-1]}"
            )

    def _radial_weights(
        self,
        distance,
        pair_i,
        pair_j,
        atom_types,
        coefficient,
    ):
        inside = distance < self.cutoff_radial
        radial = ops.chebyshev_basis(
            distance.clamp(max=self.cutoff_radial),
            self.cutoff_radial,
            self.basis_size_radial,
        )
        radial = radial * inside.to(radial).unsqueeze(-1)
        if coefficient is None:
            return radial[:, : self.spin_compress], radial
        if atom_types is None:
            raise ValueError(
                "atom_types are required when c_spin_2body is supplied"
            )
        if coefficient.ndim == 5:
            if coefficient.shape[0] != 1:
                raise ValueError("c_spin_2body leading dimension must be 1")
            coefficient = coefficient[0]
        expected_tail = (
            self.spin_compress,
            self.basis_size_radial + 1,
        )
        if coefficient.ndim != 4 or coefficient.shape[-2:] != expected_tail:
            raise ValueError(
                "c_spin_2body must have shape [T,T,C,B] or [1,T,T,C,B]"
            )
        pair_coefficient = coefficient[
            atom_types.index_select(0, pair_i),
            atom_types.index_select(0, pair_j),
        ].to(radial)
        weights = torch.bmm(
            pair_coefficient, radial.unsqueeze(-1)
        ).squeeze(-1)
        return weights, radial

    @staticmethod
    def _active_masks(
        atom_types,
        spins,
        dof_type_active,
        env_type_active,
    ):
        dof_atoms = env_atoms = None
        if dof_type_active is not None:
            if atom_types is None:
                raise ValueError("atom_types are required for spin_dof_type")
            dof_atoms = dof_type_active.to(
                device=spins.device, dtype=torch.bool
            )[atom_types]
        if env_type_active is not None:
            if atom_types is None:
                raise ValueError("atom_types are required for spin_env_type")
            env_atoms = env_type_active.to(
                device=spins.device, dtype=torch.bool
            )[atom_types]
        elif dof_atoms is not None:
            env_atoms = dof_atoms
        return dof_atoms, env_atoms

    def _project(self, value, leg):
        return torch.einsum(
            "rc,ncd->nrd", self.radial_leg_mix[leg].to(value), value
        )

    def _project_weights(self, weights, leg):
        return weights @ self.radial_leg_mix[leg].to(weights).T

    @staticmethod
    def _inner(left, right):
        return (left * right).sum(-1)

    def _density_bank(self, features, weights, pair_i, n_atoms):
        names = {}
        parts = []
        start = 0
        for name, value in features:
            stop = start + value.shape[-1]
            names[name] = slice(start, stop)
            parts.append(value)
            start = stop
        edge_bank = torch.cat(parts, -1)
        weighted = (
            weights[:, :, None] * edge_bank[:, None, :]
        ).reshape(weights.shape[0], -1)
        density = _index_sum(
            edge_bank, pair_i, weighted, n_atoms
        ).reshape(n_atoms, self.spin_compress, -1)
        return {name: density[..., span] for name, span in names.items()}

    @staticmethod
    def _commutator_axial(left5, right5):
        left = _stf5_to_matrix(left5)
        right = _stf5_to_matrix(right5)
        commutator = left @ right - right @ left
        return torch.stack(
            (
                0.5 * (commutator[..., 2, 1] - commutator[..., 1, 2]),
                0.5 * (commutator[..., 0, 2] - commutator[..., 2, 0]),
                0.5 * (commutator[..., 1, 0] - commutator[..., 0, 1]),
            ),
            -1,
        )

    @staticmethod
    def _l2_product_on_spin(product):
        """Contract a summed ``q(rhat) outer spin`` bank as ``Q @ spin``."""
        product = product.reshape(product.shape[:-1] + (5, 3))
        result = product.new_zeros(product.shape[:-2] + (3,))
        for component in range(3):
            matrix = _stf5_to_matrix(product[..., :, component])
            result = result + matrix[..., :, component]
        return result

    def _forward_fused_center(
        self, primitive, spins, local_block, center, edge_anchor
    ):
        """Assemble the exact 49D schema from analytic CUDA sector families."""
        n_atoms = primitive.shape[0]
        channel = primitive[:, :76].reshape(n_atoms, 2, 38)

        result = torch.cat(
            (
                local_block,
                self._inner(channel[..., 0:3], spins[:, None, :]),
                primitive[:, 79:81],
                self._inner(
                    self._l2_product_on_spin(channel[..., 23:38]),
                    spins[:, None, :],
                ),
                center[:, 0:2],
                primitive[:, 81:83],
                center[:, 2:14],
                primitive[:, 76:79],
                center[:, 14:17],
                center[:, 17:19],
                edge_anchor[:, 0:2],
                center[:, 19:21],
                edge_anchor[:, 2:4],
                primitive[:, 83:85],
                center[:, 21:29],
                edge_anchor[:, 4:6],
            ),
            -1,
        )
        if result.shape[-1] != self.dim:
            raise RuntimeError(
                "center CUDA sector schema mismatch: "
                f"{result.shape[-1]} != {self.dim}"
            )
        return result

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
        candidate = _oc_cuda_full_descriptor(
            self,
            rij,
            pair_i,
            pair_j,
            spins,
            atom_types,
            c_spin_2body,
            spin_dof_type_active,
            spin_env_type_active,
        )
        if candidate is not None:
            return candidate
        return self._forward_reference(
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

    def _forward_reference(
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
        del rij_coupled, pair_i_coupled, pair_j_coupled
        if c_spin_3body is not None:
            raise ValueError(
                "MagneticPolynomialDescriptor has one generated radial seam; "
                "a separate c_spin_3body tensor is not accepted"
            )
        dof_atoms, env_atoms = self._active_masks(
            atom_types,
            spins,
            spin_dof_type_active,
            spin_env_type_active,
        )
        spin2 = spins.square().sum(-1)
        active = (
            torch.ones(N, dtype=spins.dtype, device=spins.device)
            if dof_atoms is None
            else dof_atoms.to(spins)
        )
        blocks = [spin2.mul(active).unsqueeze(-1)]
        if pair_i.numel() == 0:
            return torch.cat(
                (blocks[0], spins.new_zeros((N, self.dim - 1))), -1
            )

        edge_anchor = None
        center = None
        fused = _oc_cuda_density_bank(
            self, rij, pair_i, pair_j, spins, atom_types, c_spin_2body,
            spin_dof_type_active, spin_env_type_active,
        )
        if fused is not None:
            primitive, auxiliary = fused[:2]
            edge_anchor = fused[2] if len(fused) > 2 else None
            center = _oc_cuda_center_sector(self, primitive, spins)
            if center is not None and primitive.shape[1] == 85:
                return self._forward_fused_center(
                    primitive, spins, blocks[0], center, edge_anchor
                )
            channel_stride = 38 if primitive.shape[1] == 85 else 47
            pair_width = self.spin_compress * (self.spin_compress + 1) // 2
            same_offset = channel_stride * self.spin_compress
            edge_sj2_offset = same_offset + pair_width
            edge_dot2_offset = edge_sj2_offset + self.spin_compress
            edge_gate_offset = edge_dot2_offset + self.spin_compress
            center_pair_offset = 7 * self.spin_compress
            center_projected_offset = center_pair_offset + pair_width
            channel = primitive[:, :same_offset].reshape(
                N, self.spin_compress, channel_stride
            )
            weights = auxiliary[:, :self.spin_compress]
            rhat = auxiliary[
                :, self.spin_compress:self.spin_compress + 3
            ]
            omega = auxiliary[
                :, self.spin_compress + 3:self.spin_compress + 6
            ]
            density = {
                "spin": channel[..., 0:3],
                "polar": channel[..., 3:6],
                "l1_axial": channel[..., 6:9],
                "dot_spin": channel[..., 9:12],
                "l1_long": channel[..., 12:13],
                "l1_stf": channel[..., 13:18],
                "geometry_l2": channel[..., 18:23],
                "l2_product": channel[..., 23:38],
            }
            if channel_stride == 47:
                density["l1_product"] = channel[..., 38:47]
            density["omega"] = torch.cross(
                spins[:, None, :], density["spin"], dim=-1
            )
            density["dot_omega"] = torch.cross(
                spins[:, None, :], density["dot_spin"], dim=-1
            )
            density["l2_on_spin"] = self._l2_product_on_spin(
                density["l2_product"]
            )
            blocks.extend((
                (density["spin"] * spins[:, None, :]).sum(-1),
                primitive[
                    :, edge_sj2_offset:
                    edge_sj2_offset + self.spin_compress
                ],
            ))
            if self.spin_soc and self.spin_l_max >= 2:
                blocks.append(
                    (density["l2_on_spin"] * spins[:, None, :]).sum(-1)
                )
        else:
            distance = torch.linalg.norm(rij, dim=-1).clamp_min(self.eps)
            rhat = rij / distance.unsqueeze(-1)
            weights, _ = self._radial_weights(
                distance, pair_i, pair_j, atom_types, c_spin_2body,
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
            sj2 = spin2.index_select(0, pair_j)
            dot = (si * sj).sum(-1)
            omega = torch.cross(si, sj, dim=-1)
            qrr = _stf5(rhat, rhat)

            edge_terms = [dot, sj2]
            if self.spin_soc and self.spin_l_max >= 2:
                edge_terms.append((_stf5(si, sj) * qrr).sum(-1))
            edge_order1 = torch.stack(edge_terms, -1)
            edge_order1 = _index_sum(
                spins, pair_i,
                (edge_order1[:, :, None] * weights[:, None, :]).reshape(
                    pair_i.shape[0], -1
                ),
                N,
            ).reshape(N, len(edge_terms), self.spin_compress)
            blocks.extend(edge_order1.unbind(1))

            features = [("spin", sj)]
            r_dot_s = (rhat * sj).sum(-1, keepdim=True)
            if self.spin_l_max >= 1:
                if self.spin_soc:
                    features.extend(
                        (
                            ("polar", rhat),
                            ("l1_long", r_dot_s),
                            ("l1_axial", torch.cross(rhat, sj, dim=-1)),
                            ("l1_stf", _stf5(rhat, sj)),
                        )
                    )
                else:
                    features.append(
                        (
                            "l1_product",
                            (rhat[..., :, None] * sj[..., None, :]).reshape(
                                -1, 9
                            ),
                        )
                    )
            if self.spin_l_max >= 2:
                q_matrix = _stf5_to_matrix(qrr)
                features.extend(
                    (
                        ("geometry_l2", qrr),
                        (
                            "l2_product",
                            (qrr[..., :, None] * sj[..., None, :]).reshape(
                                -1, 15
                            ),
                        ),
                    )
                )
                if self.spin_soc:
                    features.append(
                        (
                            "l2_on_spin",
                            torch.matmul(q_matrix, sj.unsqueeze(-1)).squeeze(-1),
                        )
                    )
            if self.spin_order >= 2:
                features.append(("dot_spin", dot.unsqueeze(-1) * sj))
                if self.spin_soc:
                    features.append(("omega", omega))
            if self.spin_order >= 3 and self.spin_soc:
                features.append(("dot_omega", dot.unsqueeze(-1) * omega))
            density = self._density_bank(features, weights, pair_i, N)
        if self.spin_soc and self.spin_l_max >= 2:
            local_q = _stf5(spins, spins)
            blocks.append(
                center[:, :self.spin_compress] if center is not None else
                (density["geometry_l2"] * local_q[:, None]).sum(-1)
            )

        if self.spin_order >= 2:
            blocks.append(
                primitive[
                    :, edge_dot2_offset:
                    edge_dot2_offset + self.spin_compress
                ] if fused is not None else _index_sum(
                    spins,
                    pair_i,
                    weights * dot.square().unsqueeze(-1),
                    N,
                )
            )
            blocks.append(
                center[:, self.spin_compress:2 * self.spin_compress]
                if center is not None else
                self._inner(density["spin"], density["spin"])
            )
            if self.spin_l_max >= 1:
                if self.spin_soc:
                    blocks.extend((
                        center[:, 2 * self.spin_compress:3 * self.spin_compress]
                        if center is not None else
                        density["l1_long"].square().squeeze(-1),
                        center[:, 3 * self.spin_compress:4 * self.spin_compress]
                        if center is not None else
                        self._inner(density["l1_axial"], density["l1_axial"]),
                        center[:, 4 * self.spin_compress:5 * self.spin_compress]
                        if center is not None else
                        self._inner(density["l1_stf"], density["l1_stf"]),
                    ))
                else:
                    blocks.append(
                        self._inner(
                            density["l1_product"], density["l1_product"]
                        )
                    )
            if self.spin_l_max >= 2:
                blocks.append(
                    center[:, 5 * self.spin_compress:6 * self.spin_compress]
                    if center is not None else self._inner(
                        density["l2_product"], density["l2_product"]
                    )
                )
            blocks.append(
                center[:, 6 * self.spin_compress:7 * self.spin_compress]
                if center is not None else
                self._inner(density["spin"], density["dot_spin"])
            )

            first_dot = (
                (density["spin"] * spins[:, None, :]).sum(-1)
                if fused is not None
                else _index_sum(
                    spins, pair_i, weights * dot.unsqueeze(-1), N
                )
            )
            pair_left = self.radial_pair_left
            pair_right = self.radial_pair_right
            same = (
                primitive[:, same_offset:same_offset + pair_width]
                if fused is not None
                else _index_sum(
                    spins,
                    pair_i,
                    weights.index_select(1, pair_left)
                    * weights.index_select(1, pair_right)
                    * dot.square().unsqueeze(-1),
                    N,
                )
            )
            distinct = (
                center[:, center_pair_offset:center_projected_offset]
                if center is not None else
                first_dot.index_select(1, pair_left)
                * first_dot.index_select(1, pair_right) - same
            )
            blocks.extend((same, distinct))

            if self.spin_soc and self.spin_l_max >= 1:
                polar_left = self._project(density["polar"], 0)
                polar_right = self._project(density["polar"], 1)
                omega_response = self._project(density["omega"], 0)
                if "coupling_l11_axial" in self._channel_names:
                    blocks.append(
                        center[
                            :, center_projected_offset:
                            center_projected_offset + self.spin_compress
                        ] if center is not None else self._inner(
                            torch.cross(polar_left, polar_right, dim=-1),
                            omega_response,
                        )
                    )
                polar_i = polar_left.index_select(0, pair_i)
                edge_axis = torch.cross(
                    polar_i, rhat.unsqueeze(1), dim=-1
                )
                edge_weight = self._project_weights(weights, 0)
                blocks.append(
                    edge_anchor[:, :self.spin_compress]
                    if edge_anchor is not None else _index_sum(
                        spins,
                        pair_i,
                        edge_weight
                        * self._inner(edge_axis, omega.unsqueeze(1)),
                        N,
                    )
                )
            if self.spin_soc and self.spin_l_max >= 2:
                q_left = self._project(density["geometry_l2"], 0)
                q_right = self._project(density["geometry_l2"], 1)
                omega_response = self._project(density["omega"], 0)
                if "coupling_l22_axial" in self._channel_names:
                    q_axis = self._commutator_axial(q_left, q_right)
                    blocks.append(
                        center[
                            :, center_projected_offset + self.spin_compress:
                            center_projected_offset + 2 * self.spin_compress
                        ] if center is not None else
                        self._inner(q_axis, omega_response)
                    )
                q_i = _stf5_to_matrix(q_left).index_select(0, pair_i)
                q_on_r = torch.matmul(
                    q_i, rhat[:, None, :, None]
                ).squeeze(-1)
                edge_axis = torch.cross(
                    rhat.unsqueeze(1).expand_as(q_on_r), q_on_r, dim=-1
                )
                edge_weight = self._project_weights(weights, 0)
                blocks.append(
                    edge_anchor[
                        :, self.spin_compress:2 * self.spin_compress
                    ] if edge_anchor is not None else _index_sum(
                        spins,
                        pair_i,
                        edge_weight
                        * self._inner(edge_axis, omega.unsqueeze(1)),
                        N,
                    )
                )

        if self.spin_order >= 3:
            blocks.append(
                primitive[
                    :, edge_gate_offset:
                    edge_gate_offset + self.spin_compress
                ] if fused is not None else
                _oc_cuda_moment_gate(weights, pair_i, pair_j, spins)
            )
            if self.spin_soc and self.spin_l_max >= 1:
                polar_left = self._project(density["polar"], 0)
                polar_right = self._project(density["polar"], 1)
                dot_response = self._project(density["dot_omega"], 0)
                if "coupling_l11_dot_response" in self._channel_names:
                    blocks.append(
                        center[
                            :, center_projected_offset + 2 * self.spin_compress:
                            center_projected_offset + 3 * self.spin_compress
                        ] if center is not None else self._inner(
                            torch.cross(polar_left, polar_right, dim=-1),
                            dot_response,
                        )
                    )
                p0 = polar_left
                m1 = self._project(density["spin"], 1)
                x2 = self._project(density["l1_axial"], 2)
                blocks.append(
                    center[
                        :, center_projected_offset + 3 * self.spin_compress:
                        center_projected_offset + 4 * self.spin_compress
                    ] if center is not None else
                    self._inner(p0, torch.cross(m1, x2, dim=-1))
                )
            if self.spin_soc and self.spin_l_max >= 2:
                q_left = self._project(density["geometry_l2"], 0)
                q_right = self._project(density["geometry_l2"], 1)
                dot_response = self._project(density["dot_omega"], 0)
                if "coupling_l22_dot_response" in self._channel_names:
                    q_axis = self._commutator_axial(q_left, q_right)
                    blocks.append(
                        center[
                            :, center_projected_offset + 4 * self.spin_compress:
                            center_projected_offset + 5 * self.spin_compress
                        ] if center is not None else
                        self._inner(q_axis, dot_response)
                    )
                p0 = self._project(density["polar"], 0)
                qs1 = self._project(density["l2_on_spin"], 1)
                x2 = self._project(density["l1_axial"], 2)
                blocks.append(
                    center[
                        :, center_projected_offset + 5 * self.spin_compress:
                            center_projected_offset + 6 * self.spin_compress
                    ] if center is not None else
                    self._inner(p0, torch.cross(qs1, x2, dim=-1))
                )
                p1 = self._project(density["polar"], 1)
                q0 = _stf5_to_matrix(
                    self._project(density["geometry_l2"], 0)
                )
                q_on_p1 = torch.matmul(q0, p1.unsqueeze(-1)).squeeze(-1)
                mixed_chirality = self._inner(
                    p0, torch.cross(p1, q_on_p1, dim=-1)
                )
                if "coupling_l112_edge_response" in self._channel_names:
                    edge_weight = self._project_weights(weights, 0)
                    edge_response = (rhat * omega).sum(-1, keepdim=True)
                    blocks.append(
                        edge_anchor[:, 2 * self.spin_compress:]
                        if edge_anchor is not None else _index_sum(
                            spins,
                            pair_i,
                            edge_weight
                            * mixed_chirality.index_select(0, pair_i)
                            * edge_response,
                            N,
                        )
                    )
            if "coupling_l111_bulk" in self._channel_names:
                p0 = self._project(density["polar"], 0)
                p1 = self._project(density["polar"], 1)
                p2 = self._project(density["polar"], 2)
                chirality = self._inner(
                    p0, torch.cross(p1, p2, dim=-1)
                )
                edge_weight = self._project_weights(weights, 3)
                edge_response = (rhat * omega).sum(-1, keepdim=True)
                blocks.append(
                    _index_sum(
                        spins,
                        pair_i,
                        edge_weight
                        * chirality.index_select(0, pair_i)
                        * edge_response,
                        N,
                    )
                )

        result = torch.cat(blocks, -1)
        if result.shape[-1] != self.dim:
            raise RuntimeError(
                "magnetic polynomial schema mismatch: "
                f"{result.shape[-1]} != {self.dim}"
            )
        return result
