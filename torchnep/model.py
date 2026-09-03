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

"""
Trainable NEP4 model as a PyTorch nn.Module.

Supports per-type NN architecture with ZBL (including typewise cutoffs).
Uses ops module for core computations (pure PyTorch or CUDA).
"""

import os
import warnings

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

from .constants import (
    ELEMENTS, C3B, C4B, C5B, C4B2, COVALENT_RADIUS,
)
from . import ops
from .spin_schema import (
    make_spin_descriptor,
    normalize_spin_cutoff,
    spin_projection_parameters,
    spin_projection_size,
    spin_checkpoint_lines,
)


class FittingNet(nn.Module):
    """Single-hidden-layer network: descriptor -> atomic energy.

    GPUMD convention: tanh(x @ W - b).
    """

    def __init__(self, input_dim: int, num_neurons: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.empty(input_dim, num_neurons))
        self.b0 = nn.Parameter(torch.zeros(num_neurons))
        self.w1 = nn.Parameter(torch.empty(num_neurons))
        nn.init.normal_(self.w0, std=1.0 / np.sqrt(input_dim + num_neurons))
        nn.init.normal_(self.w1, std=1.0 / np.sqrt(num_neurons + 1))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """q: (N, dim) -> (N,) per-atom energy (no bias; bias is shared in NEPModel)."""
        return torch.tanh(q @ self.w0 - self.b0) @ self.w1


class NEPModel(nn.Module):
    """Trainable NEP4 model.

    Parameters
    ----------
    config : dict
        From parse_nep_in(). Keys: cutoff_radial/angular, n_max_radial/angular,
        basis_size_radial/angular, l_max, neuron, num_types, type_names.
    energy_shift : array-like (num_types,)
        Per-type energy shift from training data.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.num_types = config["num_types"]
        self.type_names = config["type_names"]
        self.rc_radial = config["cutoff_radial"]
        self.rc_angular = config["cutoff_angular"]
        self.n_max_radial = config["n_max_radial"]
        self.n_max_angular = config["n_max_angular"]
        self.basis_size_radial = config["basis_size_radial"]
        self.basis_size_angular = config["basis_size_angular"]
        lm = list(config["l_max"])
        self.l_max_3b = lm[0]
        self.has_q_222  = 1 if (len(lm) > 1 and lm[1] > 0) else 0
        self.has_q_1111 = 1 if (len(lm) > 2 and lm[2] > 0) else 0
        self.has_q_112  = 1 if (len(lm) > 3 and lm[3] > 0) else 0
        self.has_q_123  = 1 if (len(lm) > 4 and lm[4] > 0) else 0
        self.has_q_233  = 1 if (len(lm) > 5 and lm[5] > 0) else 0
        self.has_q_134  = 1 if (len(lm) > 6 and lm[6] > 0) else 0
        self.num_neurons = config["neuron"]

        # Versioned O/C spin descriptor protocol.
        if "spin_descriptor" in config:
            raise ValueError(
                "spin_descriptor is not supported; use spin_mode 2 or 3")
        self.spin_mode = int(config.get("spin_mode", 0))
        if self.spin_mode not in (0, 2, 3):
            raise ValueError("spin_mode must be 0, 2, or 3")
        spin_only_keys = {
            "spin_compress", "spin_basis_size", "spin_l_max",
            "spin_order", "spin_soc", "spin_cutoff", "spin_dof_type",
            "spin_cutoff_by_type", "spin_env_type", "spin_curriculum",
        }
        explicit_keys = config.get("_explicit")
        supplied_spin_keys = (
            set(config) if explicit_keys is None else set(explicit_keys)
        ) & spin_only_keys
        if self.spin_mode not in (2, 3) and supplied_spin_keys:
            raise ValueError(
                "O/C spin parameters require spin_mode 2 or 3; otherwise they "
                "would be silently ignored: "
                + ", ".join(sorted(supplied_spin_keys))
            )
        self.spin_order = int(config.get(
            "spin_order", 3 if self.spin_mode in (2, 3) else 1
        ))
        self.spin_soc = int(config.get(
            "spin_soc", 1 if self.spin_mode in (2, 3) else 0
        ))
        if self.spin_mode in (2, 3) and self.spin_order not in (1, 2, 3):
            raise ValueError("spin_order must be 1, 2, or 3")
        if self.spin_mode in (2, 3) and self.spin_soc not in (0, 1):
            raise ValueError("spin_soc must be 0 or 1")
        self.spin_curriculum = bool(config.get(
            "spin_curriculum",
            self.spin_mode in (2, 3) and self.spin_order == 3,
        ))
        if self.spin_curriculum and not (
            self.spin_mode in (2, 3) and self.spin_order == 3
        ):
            raise ValueError(
                "spin_curriculum requires spin_mode 2 or 3, spin_order 3")
        self.spin_descriptor_kind = (
            "magnetic_polynomial"
            if self.spin_mode == 2
            else "magnetic_polynomial_v3"
            if self.spin_mode == 3
            else None
        )
        self.spin_compress = int(config.get("spin_compress", 2))
        if self.spin_compress < 1:
            raise ValueError("spin_compress must be >= 1")

        def _spin_int_list(key, default, size):
            values = list(config.get(key, default))
            if len(values) != size:
                raise ValueError(f"{key} requires {size} values")
            return [int(value) for value in values]

        self.spin_basis_size = _spin_int_list(
            "spin_basis_size", [8, 0], 2)
        max_spin_compress = self.spin_basis_size[0] + 1
        if self.spin_compress > max_spin_compress:
            raise ValueError(
                "spin_compress must be in [1, "
                f"{max_spin_compress}] for the fixed magnetic radial "
                f"basis B={self.spin_basis_size[0]}"
            )
        self.spin_l_max = _spin_int_list("spin_l_max", [2, 0, 0], 3)
        legacy_spin_cutoff = list(config.get(
            "spin_cutoff",
            ([6.0, 6.0] if self.spin_mode in (2, 3)
             else [self.rc_radial, self.rc_angular]),
        ))
        if self.spin_mode in (2, 3):
            raw_spin_cutoff = config.get(
                "spin_cutoff_by_type", [legacy_spin_cutoff[0]]
            )
            self.spin_cutoff_by_type = normalize_spin_cutoff(
                raw_spin_cutoff, self.num_types
            )
            spin_cutoff_max = max(self.spin_cutoff_by_type)
            self.spin_cutoff = [spin_cutoff_max, spin_cutoff_max]
        else:
            if len(legacy_spin_cutoff) != 2:
                raise ValueError("spin_cutoff requires two values")
            self.spin_cutoff = [float(value) for value in legacy_spin_cutoff]
            if min(self.spin_cutoff) <= 0.0:
                raise ValueError("spin_cutoff values must be > 0")
            self.spin_cutoff_by_type = [
                self.spin_cutoff[0]
            ] * self.num_types
        spin_cutoff_type = torch.tensor(
            self.spin_cutoff_by_type, dtype=torch.float64
        )
        self.register_buffer(
            "spin_cutoff_pair",
            0.5 * (
                spin_cutoff_type[:, None] + spin_cutoff_type[None, :]
            ),
            persistent=False,
        )
        self.spin_has_typewise_cutoff = any(
            value != self.spin_cutoff_by_type[0]
            for value in self.spin_cutoff_by_type[1:]
        )

        def _active_types(key, default=None):
            names = config.get(key)
            if names is None:
                names = default
            active = torch.ones(self.num_types, dtype=torch.bool)
            is_set = config.get(key) is not None
            if names is not None:
                if not names:
                    raise ValueError(f"{key} must enable at least one type")
                unknown = [name for name in names if name not in self.type_names]
                if unknown:
                    raise ValueError(f"{key} contains unknown type(s): {unknown}")
                active.zero_()
                for name in names:
                    active[self.type_names.index(name)] = True
            return active, is_set

        self.is_spin_dof_type_set = False
        self.is_spin_env_type_set = False
        if self.spin_mode:
            spin_dof, self.is_spin_dof_type_set = _active_types(
                "spin_dof_type")
            spin_env, self.is_spin_env_type_set = _active_types(
                "spin_env_type",
                config.get("spin_dof_type", None),
            )
            if bool((spin_dof & ~spin_env).any().item()):
                raise ValueError(
                    "spin_dof_type must be a subset of spin_env_type")
            self.register_buffer("spin_dof_type_active", spin_dof)
            self.register_buffer("spin_env_type_active", spin_env)
        else:
            self.spin_dof_type_active = None
            self.spin_env_type_active = None

        # q_123 / q_233 (extra 4-body bispectrum) need L=3 moments.
        if (self.has_q_123 or self.has_q_233) and self.l_max_3b < 3:
            raise ValueError("q_123 / q_233 require l_max_3b >= 3")
        # q_134 couples L=1, L=3, L=4 moments, so it needs L=4 moments.
        if self.has_q_134 and self.l_max_3b < 4:
            raise ValueError("q_134 requires l_max_3b >= 4")
        # q_1111 is redundant: it equals const * (3-body L=1 descriptor)^2,
        # so it adds no information. Kept for backward compatibility, but warn.
        if self.has_q_1111:
            warnings.warn(
                "has_q_1111 (l_max field 3) is set but is redundant — it "
                "equals a constant times the squared 3-body L=1 descriptor "
                "and adds no information. You can safely set it to 0.",
                stacklevel=2)

        # ZBL
        self.zbl = config.get("zbl", None)
        if self.zbl is not None:
            # Real atomic numbers (H=1). ZBL needs physical Z in Z*Z', Z^0.23.
            atomic_numbers = [ELEMENTS.index(n) + 1 for n in self.type_names]
            self.register_buffer("atomic_numbers",
                                 torch.tensor(atomic_numbers, dtype=torch.long))
            tw = config.get("typewise_cutoff_zbl_factor", None)
            if tw is not None:
                # COVALENT_RADIUS is 0-indexed, atomic_numbers is real Z -> z-1.
                rc_i = [tw * COVALENT_RADIUS[z - 1] for z in atomic_numbers]
                self.register_buffer("zbl_rc_inner_per_type", torch.tensor(rc_i))
                self.register_buffer("zbl_rc_outer_per_type",
                                     torch.tensor([2.0 * r for r in rc_i]))
                self.zbl_rc_inner = min(rc_i)
                self.zbl_rc_outer = max(2.0 * r for r in rc_i)
                self.zbl_typewise_factor = tw
            else:
                self.zbl_rc_inner = self.zbl / 2.0
                self.zbl_rc_outer = self.zbl
                self.zbl_typewise_factor = None

            # (T, T) per-type-pair tables for the compiled ZBL term
            # (ops.compute_zbl_pair): gathering from these keeps the whole
            # ZBL evaluation branch-free and traceable. Built in float64;
            # the module-level .to(dtype) casts them with the other buffers.
            an_f = torch.tensor(atomic_numbers, dtype=torch.float64)
            zi, zj = an_f.view(-1, 1), an_f.view(1, -1)
            self.register_buffer("zbl_zizj_pair", ops.K_C_SP * zi * zj,
                                 persistent=False)
            self.register_buffer("zbl_a_inv_pair",
                                 (zi ** 0.23 + zj ** 0.23) * 2.134563,
                                 persistent=False)
            if tw is not None:
                # NEP_CPU typewise convention: rc_outer per pair is
                # min((cov_i + cov_j) * factor, global rc_outer), rc_inner 0.
                # Built FROM the registered per-type buffer (float32-rounded)
                # so the table matches the eager compute_zbl path bit-for-bit.
                rt = self.zbl_rc_outer_per_type.to(torch.float64)
                rc_o_pair = torch.clamp(0.5 * (rt.view(-1, 1) + rt.view(1, -1)),
                                        max=self.zbl_rc_outer)
                rc_i_pair = torch.zeros_like(rc_o_pair)
            else:
                shape = (len(atomic_numbers), len(atomic_numbers))
                rc_o_pair = torch.full(shape, self.zbl_rc_outer,
                                       dtype=torch.float64)
                rc_i_pair = torch.full(shape, self.zbl_rc_inner,
                                       dtype=torch.float64)
            self.register_buffer("zbl_rc_inner_pair", rc_i_pair,
                                 persistent=False)
            self.register_buffer("zbl_rc_outer_pair", rc_o_pair,
                                 persistent=False)

        n_ap1 = self.n_max_angular + 1
        self.dim_radial = self.n_max_radial + 1
        self.dim_angular_3b = n_ap1 * self.l_max_3b
        self.dim_angular_4b   = n_ap1 if self.has_q_222  else 0
        self.dim_angular_5b   = n_ap1 if self.has_q_1111 else 0
        self.dim_angular_112  = n_ap1 if self.has_q_112  else 0
        self.dim_angular_123  = n_ap1 if self.has_q_123  else 0
        self.dim_angular_233  = n_ap1 if self.has_q_233  else 0
        self.dim_angular_134  = n_ap1 if self.has_q_134  else 0
        self.struct_dim = (self.dim_radial + self.dim_angular_3b +
                           self.dim_angular_4b + self.dim_angular_5b +
                           self.dim_angular_112 +
                           self.dim_angular_123 + self.dim_angular_233 +
                           self.dim_angular_134)
        self.spin_descriptor = None
        self.spin_dim = 0
        if self.spin_mode:
            self.spin_descriptor = make_spin_descriptor(
                self.spin_mode,
                spin_compress=self.spin_compress,
                spin_l_max=self.spin_l_max[0],
                spin_order=self.spin_order,
                spin_soc=bool(self.spin_soc),
                basis_size_radial=self.spin_basis_size[0],
                cutoff_radial=self.spin_cutoff[0],
            )
            self.spin_dim = self.spin_descriptor.dim
        self.dim = self.struct_dim + self.spin_dim
        self.num_lm = sum(2 * ll + 1 for ll in range(1, self.l_max_3b + 1))

        nt = self.num_types
        self.c_param_2 = nn.Parameter(torch.empty(
            nt, nt, self.n_max_radial + 1, self.basis_size_radial + 1))
        self.c_param_3 = nn.Parameter(torch.empty(
            nt, nt, self.n_max_angular + 1, self.basis_size_angular + 1)
        ) if self.l_max_3b > 0 else None
        self.c_param_spin = None
        if self.spin_mode:
            # Structural NEP uses c_param_{2,3}[center, neighbor, n, k].
            # Spin uses the same ordered-pair contraction before constructing
            # any tensor density.  The leading singleton is the established
            # serialized family axis; descriptor width remains spin_compress.
            self.c_param_spin = nn.Parameter(torch.empty(
                1, nt, nt, self.spin_compress,
                self.spin_basis_size[0] + 1))
        nn.init.normal_(self.c_param_2, std=0.1)
        if self.c_param_3 is not None:
            nn.init.normal_(self.c_param_3, std=0.1)
        if self.c_param_spin is not None:
            self._initialize_magnetic_polynomial_radial()

        # Per-type fitting networks (no per-type bias — shared b1 below)
        self.fitting_nets = nn.ModuleList([
            FittingNet(self.dim, self.num_neurons)
            for _ in range(self.num_types)
        ])
        self._spin_order3_gradient_scale = 1.0
        self.spin_order3_input_start = self.dim
        if self.spin_curriculum:
            self.spin_order3_input_start = (
                self.struct_dim + self.spin_descriptor.order_dims[2]
            )
            with torch.no_grad():
                for net in self.fitting_nets:
                    net.w0[self.spin_order3_input_start :].zero_()
        # One shared output bias (GPUMD convention: single common b1)
        self.b1 = nn.Parameter(torch.tensor(0.0))

        # q_scaler (computed from data, not learned)
        self.register_buffer("q_scaler", torch.ones(self.dim))
        if self.spin_mode:
            self.register_buffer(
                "energy_baseline", torch.zeros(self.num_types))
        else:
            self.energy_baseline = None
        self.register_buffer("_c3b", torch.tensor(C3B[:self.num_lm]))
        self.register_buffer("_c4b", torch.tensor(C4B))
        self.register_buffer("_c5b", torch.tensor(C5B))
        self.register_buffer("_c4b2", torch.tensor(C4B2))

    def set_spin_curriculum_scale(self, scale: float):
        """Set the train-only O3 opening scale without changing inference."""
        if not 0.0 <= scale <= 1.0:
            raise ValueError("spin curriculum scale must lie in [0, 1]")
        self._spin_order3_gradient_scale = float(scale)

    def architecture_signature(self):
        """Return the semantic architecture contract for binary checkpoints."""
        signature = {
            "format": "torchnep-nep4-v1",
            "type_names": list(self.type_names),
            "cutoff": [self.rc_radial, self.rc_angular],
            "n_max": [self.n_max_radial, self.n_max_angular],
            "basis_size": [
                self.basis_size_radial, self.basis_size_angular],
            "l_max": [
                self.l_max_3b, self.has_q_222, self.has_q_1111,
                self.has_q_112, self.has_q_123, self.has_q_233,
                self.has_q_134,
            ],
            "neuron": self.num_neurons,
            "spin_mode": self.spin_mode,
            "zbl": (
                None if self.zbl is None else {
                    "cutoff": float(self.zbl),
                    "typewise_factor": self.zbl_typewise_factor,
                }
            ),
        }
        if self.spin_mode in (2, 3):
            signature["spin"] = {
                "order": self.spin_order,
                "compress": self.spin_compress,
                "l_max": self.spin_l_max[0],
                "basis_size": self.spin_basis_size[0],
                "cutoff": (
                    list(self.spin_cutoff_by_type)
                    if self.spin_has_typewise_cutoff
                    else self.spin_cutoff[0]
                ),
                "soc": self.spin_soc,
                "dof_type": [
                    name for index, name in enumerate(self.type_names)
                    if bool(self.spin_dof_type_active[index].item())
                ],
                "env_type": [
                    name for index, name in enumerate(self.type_names)
                    if bool(self.spin_env_type_active[index].item())
                ],
                "channels": [
                    token.name for token in self.spin_descriptor.schema
                ],
            }
        return signature

    @torch.no_grad()
    def apply_spin_curriculum_gradients(self):
        """Apply the historical spin2 gradient-side curriculum.

        spin3 gates the actual optimizer displacement only.  Scaling both the
        gradient and the displacement is a double curriculum for SGD, while
        Adam mostly cancels the first factor but still lets it distort global
        gradient clipping and moment initialization.
        """
        if not self.spin_curriculum or self.spin_mode == 3:
            return
        for net in self.fitting_nets:
            if net.w0.grad is not None:
                net.w0.grad[self.spin_order3_input_start :].mul_(
                    self._spin_order3_gradient_scale
                )

    @torch.no_grad()
    def capture_spin_curriculum_rows(self):
        """Snapshot gated O3 rows before an optimizer step."""
        if (not self.spin_curriculum
                or self._spin_order3_gradient_scale >= 1.0):
            return None
        return tuple(
            net.w0[self.spin_order3_input_start :].clone()
            for net in self.fitting_nets
        )

    @torch.no_grad()
    def apply_spin_curriculum_parameter_step(self, before):
        """Apply the curriculum to the actual Adam parameter displacement.

        Scaling only Adam's input gradient does not scale its normalized first
        update.  Interpolating the post-step rows makes the configured ramp a
        real ramp in function parameters while retaining Adam's moment state.
        """
        if before is None:
            return
        scale = self._spin_order3_gradient_scale
        for net, previous in zip(self.fitting_nets, before):
            current = net.w0[self.spin_order3_input_start :]
            current.copy_(previous + scale * (current - previous))

    @torch.no_grad()
    def _initialize_magnetic_polynomial_radial(self):
        """Initialize every element pair with the same stable radial frame.

        spin3 uses one continuous family, ``x**c * fc``, for every contracted
        rank. spin2 retains its historical third direction, ``(x * fc)**2``,
        and its existing higher-rank continuation. This is a deterministic
        conditioning choice, not a user-facing physics switch;
        element-conditioned coefficients remain trainable afterwards.
        """
        distance = torch.linspace(
            0.0,
            self.spin_cutoff[0] * (1.0 - 1.0e-7),
            4096,
            dtype=torch.float64,
            device=self.c_param_spin.device,
        )
        basis = ops.chebyshev_basis(
            distance, self.spin_cutoff[0], self.spin_basis_size[0]
        )
        x = distance / self.spin_cutoff[0]
        envelope = 0.5 * (torch.cos(torch.pi * x) + 1.0)
        if self.spin_mode == 3:
            # spin3 defines one continuous family for every supported C:
            # w_c(r) = (r/rc)^c fc(r).  spin2 keeps its historical C >= 3
            # initialization byte-for-byte; production C=2 is identical.
            targets = [
                x.pow(degree) * envelope
                for degree in range(self.spin_compress)
            ]
        else:
            targets = [envelope, x * envelope, (x * envelope).square()]
            targets.extend(
                x.pow(degree) * envelope
                for degree in range(3, self.spin_compress)
            )
        target = torch.stack(targets[:self.spin_compress], dim=-1)
        projection = torch.linalg.lstsq(basis, target).solution.T
        self.c_param_spin.copy_(
            projection.to(self.c_param_spin).reshape(
                1, 1, 1, self.spin_compress, self.spin_basis_size[0] + 1
            ).expand_as(self.c_param_spin)
        )

    @torch.no_grad()
    def set_q_scaler(
        self,
        q_min: torch.Tensor,
        q_max: torch.Tensor,
        q_rms: Optional[torch.Tensor] = None,
    ):
        diff = q_max - q_min
        if self.spin_mode == 2:
            magnitude = torch.maximum(
                torch.maximum(q_min.abs(), q_max.abs()),
                torch.ones_like(diff),
            )
            resolved = diff > (
                64.0 * torch.finfo(diff.dtype).eps * magnitude)
            scaler = torch.where(
                resolved, diff.reciprocal(), torch.ones_like(diff))
        elif self.spin_mode == 3:
            if q_rms is None or q_rms.shape != diff.shape:
                raise ValueError(
                    "spin_mode 3 q_scaler requires one RMS value per "
                    "descriptor channel"
                )
            # Keep the structural block on the established range convention,
            # including its relative-resolution guard for constant columns.
            magnitude = torch.maximum(
                torch.maximum(q_min.abs(), q_max.abs()),
                torch.ones_like(diff),
            )
            resolved = diff > (
                64.0 * torch.finfo(diff.dtype).eps * magnitude
            )
            scaler = torch.where(
                resolved, diff.clamp_min(torch.finfo(diff.dtype).tiny).reciprocal(),
                torch.ones_like(diff),
            )

            # The magnetic block cannot be centered: q_spin=0 must remain the
            # exact zero-spin limit.  Normalize by the largest of RMS, range,
            # and a unit floor.  RMS normalizes a large fixed-moment offset;
            # range protects against sparse tails; the floor prevents nearly
            # null symmetry channels from being amplified into tanh.  This
            # form is deliberately one fused max/max/reciprocal chain and
            # needs neither a mean nor a branch per channel.
            spin_slice = slice(self.struct_dim, None)
            spin_rms = q_rms[spin_slice]
            spin_span = torch.maximum(
                torch.maximum(spin_rms, diff[spin_slice]),
                torch.ones_like(spin_rms),
            )
            scaler[spin_slice] = spin_span.reciprocal()
        else:
            scaler = torch.clamp(diff, min=1e-10).reciprocal()
        self.q_scaler.copy_(scaler)

    @torch.no_grad()
    def set_energy_baseline(self, baseline: torch.Tensor):
        if self.energy_baseline is None:
            raise ValueError(
                "energy baseline is only defined for spin_mode 2 or 3")
        self.energy_baseline.copy_(baseline.to(
            dtype=self.energy_baseline.dtype,
            device=self.energy_baseline.device,
        ))

    def _spin_descriptor_coordinates(
        self, rij, pair_i, pair_j, atom_types
    ):
        """Map pairwise cutoffs onto the existing scalar-cutoff kernel.

        Structural NEP evaluates each edge with
        ``rc_ij = (rc_i + rc_j) / 2``.  Scaling an edge by
        ``rc_max / rc_ij`` makes the scalar-cutoff radial basis exactly the
        corresponding pair-cutoff basis while preserving its unit direction.
        Autograd applies the same constant scale to the force pullback, so the
        optimized CUDA descriptor kernels need no duplicate typewise path.
        """
        if not self.spin_has_typewise_cutoff or pair_i.numel() == 0:
            return rij
        center_type = atom_types.index_select(0, pair_i)
        neighbor_type = atom_types.index_select(0, pair_j)
        pair_cutoff = self.spin_cutoff_pair[
            center_type, neighbor_type
        ].to(rij)
        scale = self.spin_cutoff[0] / pair_cutoff
        return rij * scale.unsqueeze(-1)

    def compute_descriptors(self, rij_rad, rij_ang, pi_rad, pj_rad,
                            pi_ang, pj_ang, atom_types, N,
                            backend: str = "loop", spins=None,
                            rij_spin=None, pi_spin=None, pj_spin=None):
        """Compute descriptors. Returns (N, dim). See ops.resolve_backend."""
        q_struct = ops.compute_descriptors(
            rij_rad, rij_ang, pi_rad, pj_rad, pi_ang, pj_ang,
            atom_types, N, self.c_param_2, self.c_param_3,
            self.rc_radial, self.rc_angular,
            self.basis_size_radial, self.basis_size_angular,
            self.n_max_radial, self.n_max_angular,
            self.l_max_3b,
            self.has_q_222, self.has_q_1111, self.has_q_112,
            self.num_lm, self._c3b, self._c4b, self._c5b,
            self._c4b2,
            rij_rad.dtype, rij_rad.device,
            backend=backend,
            has_q_123=self.has_q_123, has_q_233=self.has_q_233,
            has_q_134=self.has_q_134,
        )
        if self.spin_descriptor is None:
            return q_struct
        if spins is None:
            raise ValueError("spin_mode requires per-atom spins")
        if rij_spin is None:
            rij_spin, pi_spin, pj_spin = rij_rad, pi_rad, pj_rad
        descriptor_rij_spin = self._spin_descriptor_coordinates(
            rij_spin, pi_spin, pj_spin, atom_types
        )
        q_spin = self.spin_descriptor(
            descriptor_rij_spin, pi_spin, pj_spin, spins, N, atom_types,
            self.c_param_spin, None,
            self.spin_dof_type_active, self.spin_env_type_active,
        )
        return torch.cat((q_struct, q_spin), dim=-1)

    def forward(self, rij_rad, rij_ang, pi_rad, pj_rad,
                pi_ang, pj_ang, atom_types, N,
                backend: str = "loop", spins=None,
                rij_spin=None, pi_spin=None, pj_spin=None,
                include_baseline: bool = True):
        """Forward pass: returns per-atom energy Ei (N,)."""
        q = self.compute_descriptors(
            rij_rad, rij_ang, pi_rad, pj_rad,
            pi_ang, pj_ang, atom_types, N,
            backend=backend, spins=spins,
            rij_spin=rij_spin, pi_spin=pi_spin, pj_spin=pj_spin)
        q_scaled = q * self.q_scaler

        Ei = torch.zeros(N, dtype=q.dtype, device=q.device)
        dummy_accum = torch.zeros((), dtype=q.dtype, device=q.device)
        dummy_q = q_scaled[:1] if q_scaled.shape[0] > 0 else torch.zeros(
            1, self.dim, dtype=q.dtype, device=q.device)
        if os.environ.get("TORCHNEP_OC_CUDA_GRAPH", "0") == "1":
            # Branch-free, capture-safe multi-element ANN. Boolean indexing
            # lowers to a dynamic nonzero allocation, which CUDA Graphs reject.
            # Gathering the per-type weights keeps the ragged atom dimension
            # explicit and also handles types absent from this batch.
            w0 = torch.stack([net.w0 for net in self.fitting_nets])
            b0 = torch.stack([net.b0 for net in self.fitting_nets])
            w1 = torch.stack([net.w1 for net in self.fitting_nets])
            atom_w0 = w0[atom_types]
            hidden = torch.tanh(
                torch.bmm(q_scaled.unsqueeze(1), atom_w0).squeeze(1)
                - b0[atom_types]
            )
            Ei = (hidden * w1[atom_types]).sum(dim=1)
        else:
            for t in range(self.num_types):
                mask = atom_types == t
                net = self.fitting_nets[t]
                if mask.any():
                    Ei[mask] = net(q_scaled[mask])
                else:
                    dummy_accum = dummy_accum + net(dummy_q).sum()
        Ei = Ei + dummy_accum * 0.0
        Ei = Ei - self.b1
        if include_baseline and self.energy_baseline is not None:
            Ei = Ei + self.energy_baseline[atom_types]
        return Ei

    def compute_properties(self, rij_rad, rij_ang, pi_rad, pj_rad,
                           pi_ang, pj_ang, atom_types, N,
                           struct_idx, num_structures,
                           need_forces=True, need_virial=False,
                           backend: str = "loop", spins=None,
                           need_mforce=False, need_tau=False,
                           rij_spin=None, pi_spin=None, pj_spin=None,
                           create_graph: Optional[bool] = None):
        """Compute energy, forces, virial (autograd-through-rij path)."""
        dtype = rij_rad.dtype
        device = rij_rad.device
        derivative_graph = (
            bool(self.training) if create_graph is None else bool(create_graph))

        if need_forces or need_virial:
            rij_rad = rij_rad.detach().requires_grad_(True)
            rij_ang = rij_ang.detach().requires_grad_(True)
            if rij_spin is not None:
                rij_spin = rij_spin.detach().requires_grad_(True)
        if self.spin_descriptor is not None:
            if spins is None:
                raise ValueError("spin_mode requires per-atom spins")
            if need_mforce or need_tau:
                spins = spins.detach().requires_grad_(True)

        Ei = self.forward(rij_rad, rij_ang, pi_rad, pj_rad,
                          pi_ang, pj_ang, atom_types, N,
                          backend=backend, spins=spins,
                          rij_spin=rij_spin, pi_spin=pi_spin, pj_spin=pj_spin)

        if self.zbl is not None:
            Ei = Ei + ops.compute_zbl(
                atom_types, pi_ang, pj_ang, rij_ang, N,
                self.atomic_numbers.tolist(),
                self.zbl_rc_inner, self.zbl_rc_outer,
                self.zbl_typewise_factor,
                getattr(self, "zbl_rc_inner_per_type", None),
                getattr(self, "zbl_rc_outer_per_type", None),
                dtype, device)

        Etot = torch.zeros(num_structures, dtype=dtype, device=device)
        Etot.scatter_add_(0, struct_idx, Ei)

        result = {"Ei": Ei, "Etot": Etot}

        if need_forces or need_virial:
            grad_inputs = [rij_rad, rij_ang]
            has_separate_spin_pairs = (
                self.spin_descriptor is not None and rij_spin is not None
                and rij_spin is not rij_rad)
            if has_separate_spin_pairs:
                grad_inputs.append(rij_spin)
            grads = torch.autograd.grad(
                Ei.sum(), grad_inputs,
                create_graph=derivative_graph,
                retain_graph=derivative_graph or need_mforce or need_tau,
                allow_unused=True)
            g_rad = grads[0] if grads[0] is not None else torch.zeros_like(rij_rad)
            g_ang = grads[1] if grads[1] is not None else torch.zeros_like(rij_ang)

            rr = rij_rad if derivative_graph else rij_rad.detach()
            ra = rij_ang if derivative_graph else rij_ang.detach()
            gr = g_rad
            ga = g_ang

            forces, virial = ops.accumulate_forces_virial(
                N, pi_rad, pj_rad, rr, gr,
                pi_ang, pj_ang, ra, ga, dtype, device)
            if has_separate_spin_pairs:
                g_spin = (grads[2] if grads[2] is not None
                          else torch.zeros_like(rij_spin))
                empty_i = torch.zeros(0, dtype=torch.long, device=device)
                empty_r = torch.zeros(0, 3, dtype=dtype, device=device)
                f_spin, v_spin = ops.accumulate_forces_virial(
                    N, pi_spin, pj_spin,
                    rij_spin if derivative_graph else rij_spin.detach(), g_spin,
                    empty_i, empty_i, empty_r, empty_r,
                    dtype, device,
                )
                forces = forces + f_spin
                virial = virial + v_spin
            if need_forces:
                result["forces"] = forces
            if need_virial:
                result["virial"] = virial

        if self.spin_descriptor is not None and (need_mforce or need_tau):
            grad_spin = torch.autograd.grad(
                Ei.sum(), spins,
                create_graph=derivative_graph,
                retain_graph=derivative_graph,
                allow_unused=True,
            )[0]
            if grad_spin is None:
                grad_spin = torch.zeros_like(spins)
            mforces = -grad_spin
            active = self.spin_dof_type_active[atom_types].unsqueeze(-1)
            mforces = mforces * active.to(mforces.dtype)
            if need_mforce:
                result["mforces"] = mforces
            if need_tau:
                result["tau"] = torch.cross(spins, mforces, dim=-1)

        return result

    def _cached_core(self, batch, need_forces=True, need_virial=False,
                     backend: str = "loop"):
        """Descriptor + NN + analytical-force part of the cached compute.

        Deliberately free of data-dependent Python control flow — the NN
        dispatch is a branchless weight gather, and ZBL is the branch-free
        table variant (``ops.compute_zbl_pair``) with analytic pair
        gradients — so ``torch.compile`` captures this whole function as
        ONE graph with no breaks (the eager reference path's ``mask.any()``
        branches and inner ``autograd.grad`` each split the graph). Only
        result assembly lives in the ``compute_properties_cached`` wrapper.

        Returns ``(Ei, forces, virial)`` — forces/virial are None when not
        requested.
        """
        dtype = self.q_scaler.dtype
        device = self.q_scaler.device
        N = batch["N"]

        # Descriptors from cached basis.
        # When forces needed: return intermediates (s, gn_ang) for analytical force.
        if need_forces or need_virial:
            q, s, gn_ang = ops.compute_descriptors_cached(
                batch["fk_rad"], batch["fk_ang"], batch["blm"],
                batch["pair_i_rad"], batch["pair_j_rad"],
                batch["pair_i_ang"], batch["pair_j_ang"],
                batch["atom_types"], N, self.c_param_2, self.c_param_3,
                self.n_max_radial, self.n_max_angular,
                self.l_max_3b,
                self.has_q_222, self.has_q_1111, self.has_q_112,
                self.num_lm, self._c3b, self._c4b, self._c5b,
                self._c4b2,
                dtype, device,
                return_intermediates=True,
                backend=backend,
                has_q_123=self.has_q_123, has_q_233=self.has_q_233,
                has_q_134=self.has_q_134,
            )
        else:
            q = ops.compute_descriptors_cached(
                batch["fk_rad"], batch["fk_ang"], batch["blm"],
                batch["pair_i_rad"], batch["pair_j_rad"],
                batch["pair_i_ang"], batch["pair_j_ang"],
                batch["atom_types"], N, self.c_param_2, self.c_param_3,
                self.n_max_radial, self.n_max_angular,
                self.l_max_3b,
                self.has_q_222, self.has_q_1111, self.has_q_112,
                self.num_lm, self._c3b, self._c4b, self._c5b,
                self._c4b2,
                dtype, device,
                backend=backend,
                has_q_123=self.has_q_123, has_q_233=self.has_q_233,
                has_q_134=self.has_q_134,
            )
            s = gn_ang = None

        q_scaled = q * self.q_scaler

        # NN forward + Fp via per-atom GATHERED weights — one batched matmul
        # for all atoms instead of the old per-type loop (which ran every
        # type's net on all atoms and torch.where-selected: T x the flops
        # and ~3T tiny GEMM launches; profiled at 118 aten::mm per training
        # step for 16 types — the single largest GPU cost).
        #
        # Still branch-free (gather / one-hot matmul, fully traceable) and
        # still DDP-safe: the weight stacks are built from EVERY type's
        # parameters, so all parameters join the autograd graph on every
        # forward (absent types just receive zero gradient slices).
        # With gradients enabled the per-atom weights come from a one-hot
        # matmul so the backward into the stacks is one clean GEMM instead
        # of index_put atomics (same trick and thresholds as the mulsum
        # contraction — see ops._gather_c_onehot).
        w0s = torch.stack([n.w0 for n in self.fitting_nets])   # (T, dim, H)
        b0s = torch.stack([n.b0 for n in self.fitting_nets])   # (T, H)
        w1s = torch.stack([n.w1 for n in self.fitting_nets])   # (T, H)
        T = self.num_types
        at = batch["atom_types"]
        need_grad = torch.is_grad_enabled() and w0s.requires_grad
        if need_grad and (T <= 32 or torch.version.hip is not None):
            oh = torch.nn.functional.one_hot(at, T).to(dtype)  # (N, T)
            W0 = (oh @ w0s.reshape(T, -1)).view(N, self.dim, -1)
            B0 = oh @ b0s
            W1 = oh @ w1s
        else:
            W0 = w0s[at]
            B0 = b0s[at]
            W1 = w1s[at]
        if ops.NN_MULSUM:
            # ROCm: rocBLAS is poor on these skinny batched shapes (45% of
            # the MI250X step went to Cijk_* GEMMs); the explicit
            # multiply+reduce lowers to fused Triton kernels instead —
            # measured 25% faster full step on MI250X, identical math.
            h = torch.tanh((q_scaled.unsqueeze(-1) * W0).sum(1) - B0)
            Ei = (h * W1).sum(-1)
            Fp = (W0 * (W1 * (1.0 - h * h)).unsqueeze(1)).sum(-1)
        else:
            h = torch.tanh(torch.bmm(q_scaled.unsqueeze(1), W0).squeeze(1)
                           - B0)
            Ei = (h * W1).sum(-1)
            Fp = torch.bmm(W0, (W1 * (1.0 - h * h)).unsqueeze(-1)).squeeze(-1)

        Fp = Fp * self.q_scaler  # absorb q_scaler into Fp
        Ei = Ei - self.b1  # subtract shared output bias
        if self.energy_baseline is not None:
            Ei = Ei + self.energy_baseline[batch["atom_types"]]

        # ZBL — branch-free table variant, fully inside the compiled graph
        # (energy over all angular pairs; fc is exactly zero beyond the
        # per-pair cutoff, so no boolean compaction is needed). The pair
        # gradient g_zbl rides into compute_analytical_forces where it is
        # folded into the angular scatter for free.
        g_zbl = None
        if self.zbl is not None:
            e_zbl, g_zbl = ops.compute_zbl_pair(
                batch["atom_types"], batch["pair_i_ang"],
                batch["pair_j_ang"], batch["rij_ang"],
                self.zbl_zizj_pair, self.zbl_a_inv_pair,
                self.zbl_rc_inner_pair, self.zbl_rc_outer_pair,
                need_grad=need_forces)
            Ei = Ei.scatter_add(0, batch["pair_i_ang"], e_zbl)

        forces = None
        virial = None
        if need_forces:
            # Analytical forces: fully differentiable through c2/c3 and NN
            # weights (Fp). No create_graph=True needed — chain rule is
            # computed explicitly.
            forces, virial = ops.compute_analytical_forces(
                Fp, batch["atom_types"], N,
                self.c_param_2, self.c_param_3,
                batch["fkp_rad"], batch["fkp_ang"], batch["blm"],
                batch["pair_i_rad"], batch["pair_j_rad"],
                batch["rij_rad"], batch["d12inv_rad"],
                batch["pair_i_ang"], batch["pair_j_ang"],
                batch["rij_ang"], batch["d12inv_ang"],
                s, gn_ang,
                self.n_max_radial, self.n_max_angular,
                self.l_max_3b,
                self.has_q_222, self.has_q_1111, self.has_q_112,
                self.num_lm, self._c3b, self._c4b, self._c5b,
                self._c4b2,
                dtype, device,
                compute_virial=need_virial,
                backend=backend,
                has_q_123=self.has_q_123, has_q_233=self.has_q_233,
                has_q_134=self.has_q_134,
                g_extra_ang=g_zbl,
            )
        return Ei, forces, virial

    def compute_properties_cached(self, batch, need_forces=True, need_virial=False,
                                   backend: str = "loop", core_fn=None,
                                   create_graph: Optional[bool] = None):
        """Compute energy, forces, virial using precomputed basis.

        Uses fully analytical force computation — no create_graph=True needed.
        Forces are differentiable through c2, c3 (via Fp->NN weights and via s->c3).

        ``backend`` in {"loop", "bmm"} — see torchnep.ops.resolve_backend.
        ``core_fn`` optionally substitutes a ``torch.compile``d version of
        ``_cached_core`` (the trainer passes one); the core includes ZBL
        (branch-free table variant), only the per-structure energy
        reduction below stays eager.
        """
        if self.spin_mode in (2, 3):
            # O/C responses are derivatives of the same atomic-energy model.
            # Keep this full autograd route for E/F/V/MF/Tau training; custom
            # CUDA sector Functions replace its hot contraction subgraphs.
            caller_grad_enabled = torch.is_grad_enabled()
            with torch.enable_grad():
                result = self.compute_properties(
                    batch["rij_rad"], batch["rij_ang"],
                    batch["pair_i_rad"], batch["pair_j_rad"],
                    batch["pair_i_ang"], batch["pair_j_ang"],
                    batch["atom_types"], batch["N"], batch["struct_idx"],
                    batch["num_structures"], need_forces=need_forces,
                    need_virial=need_virial, backend=backend,
                    spins=batch.get("spins"),
                    need_mforce=batch.get("need_mforce", False),
                    need_tau=batch.get("need_tau", False),
                    rij_spin=batch.get("rij_spin"),
                    pi_spin=batch.get("pair_i_spin"),
                    pj_spin=batch.get("pair_j_spin"),
                    create_graph=create_graph,
                )
            if not caller_grad_enabled:
                result = {
                    name: value.detach() if torch.is_tensor(value) else value
                    for name, value in result.items()
                }
            return result

        dtype = self.q_scaler.dtype
        device = self.q_scaler.device
        N = batch["N"]

        core = core_fn if core_fn is not None else self._cached_core
        Ei, forces, virial = core(batch, need_forces=need_forces,
                                  need_virial=need_virial, backend=backend)

        Etot = torch.zeros(batch["num_structures"], dtype=dtype, device=device)
        Etot.scatter_add_(0, batch["struct_idx"], Ei)

        result = {"Ei": Ei, "Etot": Etot}
        if need_forces:
            result["forces"] = forces
            if need_virial and virial is not None:
                result["virial"] = virial
        return result

    def load_weights_from_nep_txt(self, path: str):
        """Load trainable weights from a GPUMD nep.txt file into this model.

        The model architecture (num_types, neuron, n_max, basis_size, l_max)
        must match the nep.txt file exactly.  The q_scaler is loaded too and
        kept — it is part of the potential the weights were trained as.

        Typical usage (fine-tuning):

            model = NEPModel(config)          # build from nep.in (same arch)
            model.load_weights_from_nep_txt("pretrained/nep.txt")
            # then call train_nep(..., finetune_from="pretrained/nep.txt")
        """
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        spin_baseline_values = None
        declared_projection_size = None
        first_header = lines[0].split()
        model_tag = first_header[0]
        file_spin_mode = (
            3 if "spin3" in model_tag
            else 2 if "spin2" in model_tag
            else 0
        )
        if file_spin_mode != self.spin_mode:
            raise ValueError(
                f"model format has spin_mode {file_spin_mode}, but the "
                f"target model was built with spin_mode {self.spin_mode}"
            )
        file_has_zbl = "zbl" in model_tag
        if file_has_zbl != (self.zbl is not None):
            raise ValueError(
                "model ZBL setting does not match target config")
        if (len(first_header) < 2
                or int(first_header[1]) != self.num_types
                or first_header[2:] != self.type_names):
            raise ValueError(
                "model element count/order does not match target config: "
                f"{first_header[2:]} != {self.type_names}"
            )
        spin_metadata = {}
        structural_metadata = {}
        allowed_spin_metadata = {
            "spin_mode", "spin_baseline", "spin_basis_size", "spin_l_max",
            "spin_compress", "spin_cutoff", "spin_order", "spin_soc",
            "spin_projection_size", "spin_scaler", "spin_dof_type",
            "spin_env_type",
        }
        numeric_start = len(lines)
        for line_index, line in enumerate(lines[1:], start=1):
            try:
                float(line)
                numeric_start = line_index
                break
            except ValueError:
                pass
        if self.spin_mode:
            if len(lines) < 2 or lines[1].split()[0] != "spin_mode":
                raise ValueError(
                    f"nep4_spin{self.spin_mode} requires spin_mode as the "
                    "first metadata line")
            mode_parts = lines[1].split()
            if len(mode_parts) != 3:
                raise ValueError(
                    "spin_mode metadata must declare mode and line count")
            declared_block_size = int(mode_parts[2])
            block_end = 2 + declared_block_size
            if block_end > numeric_start:
                raise ValueError("truncated counted spin_mode block")
            allowed_block_keys = allowed_spin_metadata - {"spin_mode"}
            for block_line in lines[2:block_end]:
                block_key = block_line.split()[0]
                if block_key not in allowed_block_keys:
                    raise ValueError(
                        "non-spin metadata found inside counted spin_mode "
                        f"block: {block_key}")
            misplaced = [
                line.split()[0] for line in lines[block_end:numeric_start]
                if line.split()[0].startswith("spin_")
            ]
            if misplaced:
                raise ValueError(
                    "spin_mode metadata count does not match its block; "
                    "spin metadata found outside the counted block: "
                    + ", ".join(misplaced))
        else:
            misplaced = [
                line.split()[0] for line in lines[1:numeric_start]
                if line.split()[0].startswith("spin_")
            ]
            if misplaced:
                raise ValueError(
                    "non-spin model contains spin metadata: "
                    + ", ".join(misplaced))
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            try:
                float(parts[0])
                break
            except ValueError:
                pass
            if parts[0] == "spin_descriptor":
                raise ValueError(
                    "spin_descriptor is not supported by the versioned "
                    "O/C spin protocol")
            key = parts[0]
            key_lower = key.lower()
            if key_lower in {"cutoff", "n_max", "basis_size", "l_max", "ann"}:
                if key_lower in structural_metadata:
                    raise ValueError(
                        f"duplicate model metadata: {key_lower}")
                structural_metadata[key_lower] = parts[1:]
            if parts[0].startswith("spin_"):
                if parts[0] not in allowed_spin_metadata:
                    raise ValueError(
                        f"unknown O/C model metadata: {parts[0]}")
                if parts[0] in spin_metadata:
                    raise ValueError(
                        f"duplicate O/C model metadata: {parts[0]}")
                spin_metadata[parts[0]] = parts[1:]
            if parts[0] == "spin_baseline":
                if len(parts) != self.num_types + 1:
                    raise ValueError(
                        "spin_baseline must have one value per type")
                spin_baseline_values = [
                    float(value) for value in parts[1:]
                ]
            elif parts[0] == "spin_projection_size":
                declared_projection_size = int(parts[1])
        if self.spin_mode and spin_baseline_values is None:
            raise ValueError("spin model is missing spin_baseline")
        if self.spin_mode:
            spin_mode_values = spin_metadata.get("spin_mode", [])
            if len(spin_mode_values) != 2:
                raise ValueError(
                    "spin_mode metadata must declare mode and line count")
            declared_spin_lines = int(spin_mode_values[1])
            actual_spin_lines = len(spin_metadata) - 1
            if declared_spin_lines != actual_spin_lines:
                raise ValueError(
                    "spin_mode metadata count does not match its block: "
                    f"{declared_spin_lines} != {actual_spin_lines}")

        required_structural = {
            "cutoff", "n_max", "basis_size", "l_max", "ann"
        }
        missing_structural = sorted(
            required_structural - set(structural_metadata)
        )
        if missing_structural:
            raise ValueError(
                "model is missing structural metadata: "
                + ", ".join(missing_structural)
            )
        cutoff_values = structural_metadata["cutoff"]
        if len(cutoff_values) < 2 or not np.allclose(
                [float(value) for value in cutoff_values[:2]],
                [self.rc_radial, self.rc_angular],
                rtol=0.0, atol=1.0e-12):
            raise ValueError("cutoff does not match target config")
        n_max_values = [
            int(value) for value in structural_metadata["n_max"][:2]
        ]
        if n_max_values != [self.n_max_radial, self.n_max_angular]:
            raise ValueError("n_max does not match target config")
        basis_values = [
            int(value) for value in structural_metadata["basis_size"][:2]
        ]
        if basis_values != [
                self.basis_size_radial, self.basis_size_angular]:
            raise ValueError("basis_size does not match target config")
        file_lmax = [
            int(value) for value in structural_metadata["l_max"]
        ]
        file_lmax.extend([0] * (7 - len(file_lmax)))
        file_lmax = [file_lmax[0], int(file_lmax[1] > 0)] + file_lmax[2:7]
        expected_lmax = [
            self.l_max_3b, self.has_q_222, self.has_q_1111,
            self.has_q_112, self.has_q_123, self.has_q_233,
            self.has_q_134,
        ]
        if file_lmax != expected_lmax:
            raise ValueError("l_max does not match target config")
        ann_values = structural_metadata["ann"]
        if not ann_values or int(ann_values[0]) != self.num_neurons:
            raise ValueError("ANN does not match target config")
        if file_has_zbl:
            zbl_lines = [
                line.split() for line in lines
                if line.split() and line.split()[0] == "zbl"
            ]
            if len(zbl_lines) != 1:
                raise ValueError("ZBL model must contain exactly one zbl line")
            zbl_values = zbl_lines[0][1:]
            expected_zbl = [self.zbl / 2.0, self.zbl]
            if len(zbl_values) not in (2, 3) or not np.allclose(
                    [float(value) for value in zbl_values[:2]],
                    expected_zbl, rtol=0.0, atol=1.0e-12):
                raise ValueError("ZBL cutoff does not match target config")
            file_factor = (
                float(zbl_values[2]) if len(zbl_values) == 3 else None)
            if file_factor != self.zbl_typewise_factor:
                raise ValueError(
                    "ZBL typewise factor does not match target config")
        if self.spin_mode:
            required = {
                "spin_mode": self.spin_mode,
                "spin_basis_size": self.spin_basis_size[0],
                "spin_l_max": self.spin_l_max[0],
                "spin_compress": self.spin_compress,
                "spin_order": self.spin_order,
                "spin_soc": self.spin_soc,
                "spin_scaler": 1,
            }
            for key, expected in required.items():
                values = spin_metadata.get(key)
                if not values or int(values[0]) != expected:
                    actual = None if not values else values[0]
                    raise ValueError(
                        f"{key} does not match target config: "
                        f"{actual} != {expected}"
                    )
            cutoff = spin_metadata.get("spin_cutoff")
            file_spin_cutoff = (
                None if not cutoff
                else normalize_spin_cutoff(cutoff, self.num_types)
            )
            if (file_spin_cutoff is None or not np.allclose(
                    file_spin_cutoff, self.spin_cutoff_by_type,
                    rtol=0.0, atol=1.0e-12)):
                actual = file_spin_cutoff
                raise ValueError(
                    "spin_cutoff does not match target config: "
                    f"{actual} != {self.spin_cutoff_by_type}"
                )
            expected_dof = [
                name for index, name in enumerate(self.type_names)
                if bool(self.spin_dof_type_active[index].item())
            ]
            expected_env = [
                name for index, name in enumerate(self.type_names)
                if bool(self.spin_env_type_active[index].item())
            ]
            file_dof = spin_metadata.get(
                "spin_dof_type", list(self.type_names))
            file_env = spin_metadata.get("spin_env_type", file_dof)
            if file_dof != expected_dof:
                raise ValueError(
                    "spin_dof_type does not match target config: "
                    f"{file_dof} != {expected_dof}"
                )
            if file_env != expected_env:
                raise ValueError(
                    "spin_env_type does not match target config: "
                    f"{file_env} != {expected_env}"
                )
            self.energy_baseline.copy_(torch.tensor(
                spin_baseline_values,
                dtype=self.energy_baseline.dtype,
                device=self.energy_baseline.device,
            ))

        # Count header lines (non-numeric lines at the top)
        header_lines = 0
        for ln in lines:
            try:
                float(ln)
                break
            except ValueError:
                header_lines += 1

        # Use float64 as the numpy transport dtype so we preserve every digit
        # written to nep.txt; the target .copy_() downcasts to the parameter's
        # actual dtype (float32 or float64) without going through float32 first.
        vals = np.asarray([float(ln) for ln in lines[header_lines:]],
                          dtype=np.float64)
        idx = 0
        dim = self.dim
        neurons = self.num_neurons

        for t in range(self.num_types):
            # w0 stored as (neurons, dim) row-major; model keeps (dim, neurons)
            n_w0 = neurons * dim
            w0 = vals[idx:idx + n_w0].reshape(neurons, dim).T
            self.fitting_nets[t].w0.data.copy_(torch.from_numpy(w0.copy()))
            idx += n_w0

            b0 = vals[idx:idx + neurons]
            self.fitting_nets[t].b0.data.copy_(torch.from_numpy(b0.copy()))
            idx += neurons

            w1 = vals[idx:idx + neurons]
            self.fitting_nets[t].w1.data.copy_(torch.from_numpy(w1.copy()))
            idx += neurons

        # Shared output bias
        self.b1.data.fill_(float(vals[idx]))
        idx += 1

        # c_param_2: saved as (n_rad+1, bs_rad+1, nt, nt), model: (nt, nt, n+1, b+1)
        nt = self.num_types
        n_c2 = (self.n_max_radial + 1) * (self.basis_size_radial + 1) * nt * nt
        c2 = vals[idx:idx + n_c2].reshape(
            self.n_max_radial + 1, self.basis_size_radial + 1, nt, nt)
        self.c_param_2.data.copy_(
            torch.from_numpy(np.ascontiguousarray(np.transpose(c2, (2, 3, 0, 1)))))
        idx += n_c2

        if self.c_param_3 is not None:
            n_c3 = (self.n_max_angular + 1) * (self.basis_size_angular + 1) * nt * nt
            c3 = vals[idx:idx + n_c3].reshape(
                self.n_max_angular + 1, self.basis_size_angular + 1, nt, nt)
            self.c_param_3.data.copy_(
                torch.from_numpy(np.ascontiguousarray(np.transpose(c3, (2, 3, 0, 1)))))
            idx += n_c3

        if self.c_param_spin is not None:
            n_cs = self.c_param_spin.numel()
            cs = vals[idx:idx + n_cs].reshape(
                1, self.spin_compress, self.spin_basis_size[0] + 1,
                nt, nt,
            )
            self.c_param_spin.data.copy_(torch.from_numpy(
                np.ascontiguousarray(np.transpose(cs, (0, 3, 4, 1, 2)))))
            idx += n_cs

        projection_parameters = spin_projection_parameters(self)
        expected_projection_size = spin_projection_size(self)
        if self.spin_mode in (2, 3):
            if declared_projection_size is None:
                raise ValueError(
                    f"nep4_spin{self.spin_mode} model is missing "
                    "spin_projection_size"
                )
            if declared_projection_size != expected_projection_size:
                raise ValueError(
                    "spin_projection_size does not match the descriptor: "
                    f"{declared_projection_size} != "
                    f"{expected_projection_size}"
                )
        for parameter in projection_parameters:
            size = parameter.numel()
            values = vals[idx:idx + size]
            if values.size != size:
                raise ValueError("truncated O/C projection weights")
            parameter.data.copy_(
                torch.from_numpy(values.reshape(parameter.shape).copy())
            )
            idx += size

        # q_scaler (buffer — kept; the weights were trained against it)
        q_scaler = vals[idx:idx + dim]
        if q_scaler.size != dim:
            raise ValueError("truncated q_scaler")
        self.q_scaler.copy_(torch.from_numpy(q_scaler.copy()))
        idx += dim
        if idx != vals.size:
            raise ValueError(
                f"unexpected trailing numeric values: {vals.size - idx}"
            )

    def save_nep_txt(self, path: str, max_NN_radial: int,
                     max_NN_angular: int):
        """Save model to GPUMD nep4 nep.txt format.

        ``max_NN_radial`` / ``max_NN_angular`` are mandatory: GPUMD's nep.txt
        parser requires the cutoff line to always carry 4 numbers
        ``cutoff <rc_R> <rc_A> <max_NN_R> <max_NN_A>``. Both must be > 0.
        Compute them from the training set via
        :func:`torchnep.train.compute_max_neighbors`.
        """
        if max_NN_radial <= 0 or max_NN_angular <= 0:
            raise ValueError(
                "save_nep_txt requires max_NN_radial > 0 and "
                "max_NN_angular > 0 (GPUMD's nep.txt format mandates the "
                f"cutoff line to carry both); got "
                f"max_NN_radial={max_NN_radial}, "
                f"max_NN_angular={max_NN_angular}. "
                "Use torchnep.train.compute_max_neighbors(structures) to "
                "obtain them.")
        lines = []
        zbl_suffix = "_zbl" if self.zbl is not None else ""
        model_tag = (
            f"nep4_spin{self.spin_mode}"
            if self.spin_mode
            else "nep4"
        )
        lines.append(f"{model_tag}{zbl_suffix} {self.num_types} "
                     + " ".join(self.type_names))
        if self.spin_mode:
            spin_lines = spin_checkpoint_lines(self)
            lines.append(
                f"spin_mode {self.spin_mode} {len(spin_lines)}"
            )
            lines.extend(spin_lines)

        if self.zbl is not None:
            tw = self.zbl_typewise_factor
            rc_inner_out = self.zbl / 2.0
            rc_outer_out = self.zbl
            if tw is not None:
                lines.append(f"zbl {rc_inner_out} {rc_outer_out} {tw}")
            else:
                lines.append(f"zbl {rc_inner_out} {rc_outer_out}")

        # Format cutoff: integer if whole number (matches GPUMD style).
        # Always emit 4 fields — GPUMD's nep.txt parser requires both
        # max_NN_radial and max_NN_angular on the cutoff line.
        def _fmt(v):
            return str(int(v)) if v == int(v) else str(v)
        lines.append(
            f"cutoff {_fmt(self.rc_radial)} {_fmt(self.rc_angular)} "
            f"{max_NN_radial} {max_NN_angular}")
        lines.append(f"n_max {self.n_max_radial} {self.n_max_angular}")
        lines.append(f"basis_size {self.basis_size_radial} "
                     f"{self.basis_size_angular}")
        # l_max line: L_max + GPUMD-core flags + q_123 / q_233 / q_134.
        #
        # Field 2 (has_q_222) is written with the legacy encoding "2 if on
        # else 0" — this matches GPUMD's fitness.cu (``has_q_222 ? 2 : 0``)
        # so that older GPUMD builds (which read this field as ``L_max_4body``
        # and treated >=2 as "enable q_222") still load the model correctly.
        #
        # Trailing zero flags are trimmed to >=3 fields so a baseline model
        # with no extras prints exactly the classic ``l_max L_3b 2 0``
        # three-token line that older GPUMD builds produced and parsed.
        lmax_flags = [self.l_max_3b,
                      2 if self.has_q_222 else 0,
                      self.has_q_1111,
                      self.has_q_112,
                      self.has_q_123, self.has_q_233, self.has_q_134]
        while len(lmax_flags) > 3 and lmax_flags[-1] == 0:
            lmax_flags.pop()
        lines.append("l_max " + " ".join(str(x) for x in lmax_flags))
        lines.append(f"ANN {self.num_neurons} 0")

        # Per-type NN weights
        for t in range(self.num_types):
            net = self.fitting_nets[t]
            w0 = net.w0.detach().cpu().numpy().T  # (neurons, dim)
            for v in w0.flat:
                lines.append(f"  {v:.10e}")
            for v in net.b0.detach().cpu().numpy():
                lines.append(f"  {v:.10e}")
            for v in net.w1.detach().cpu().numpy():
                lines.append(f"  {v:.10e}")

        # Common output bias (shared across all types — GPUMD convention)
        lines.append(f"  {self.b1.item():.10e}")

        # c2: stored as (n_max+1, basis+1, nt, nt)
        c2 = self.c_param_2.detach().cpu().numpy().transpose(2, 3, 0, 1)
        for v in c2.flat:
            lines.append(f"  {v:.10e}")
        if self.c_param_3 is not None:
            c3 = self.c_param_3.detach().cpu().numpy().transpose(2, 3, 0, 1)
            for v in c3.flat:
                lines.append(f"  {v:.10e}")
        if self.c_param_spin is not None:
            c_spin = self.c_param_spin.detach().cpu().numpy().transpose(
                0, 3, 4, 1, 2)
            for v in c_spin.flat:
                lines.append(f"  {v:.10e}")

        for parameter in spin_projection_parameters(self):
            for value in parameter.detach().cpu().numpy().flat:
                lines.append(f"  {value:.10e}")

        for v in self.q_scaler.detach().cpu().numpy():
            lines.append(f"  {v:.10e}")

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")


@torch.no_grad()
def gpumd_init_parameters(model: NEPModel) -> None:
    """Re-initialise GPUMD-compatible parameters uniform(-1, 1) in place.

    Both the descriptor coefficients (``c_param_2`` / ``c_param_3``) and the
    fitting-network weights (``w0`` / ``b0`` / ``w1``) are covered, so a model
    trained on top of GPUMD's ``c=1`` q_scaler starts from the same
    large-amplitude landscape GPUMD's SNES ``mu`` initialization defines.
    Descriptor architecture parameters without a GPUMD counterpart keep their
    designed initialization. Used for fresh training only.
    """
    torch.nn.init.uniform_(model.c_param_2, -1.0, 1.0)
    if model.c_param_3 is not None:
        torch.nn.init.uniform_(model.c_param_3, -1.0, 1.0)
    if model.c_param_spin is not None:
        torch.nn.init.uniform_(model.c_param_spin, -1.0, 1.0)
    for net in model.fitting_nets:
        torch.nn.init.uniform_(net.w0, -1.0, 1.0)
        torch.nn.init.uniform_(net.b0, -1.0, 1.0)
        torch.nn.init.uniform_(net.w1, -1.0, 1.0)


def slim_model(model: NEPModel, keep_type_names: List[str]) -> NEPModel:
    """Return a new NEPModel containing only the specified element types.

    Mathematically equivalent to the original for structures that contain
    only the kept element types.  Parameters for removed types are discarded:

    - ``fitting_nets``: only the nets for kept types are copied
    - ``c_param_2 / c_param_3``: rows and columns for removed types are dropped
    - ``b1``, ``q_scaler``: copied unchanged (independent of num_types)

    Parameters
    ----------
    model : NEPModel
        Source model (any number of types).
    keep_type_names : list[str]
        Subset of ``model.type_names`` to retain, in any order.
        The output model uses this order.

    Returns
    -------
    NEPModel
        Smaller model on the same device/dtype as the source.
    """
    unknown = [t for t in keep_type_names if t not in model.type_names]
    if unknown:
        raise ValueError(f"Types not in source model: {unknown}")

    keep_idx = [model.type_names.index(t) for t in keep_type_names]

    new_config = {
        "num_types":          len(keep_type_names),
        "type_names":         list(keep_type_names),
        "cutoff_radial":      model.rc_radial,
        "cutoff_angular":     model.rc_angular,
        "n_max_radial":       model.n_max_radial,
        "n_max_angular":      model.n_max_angular,
        "basis_size_radial":  model.basis_size_radial,
        "basis_size_angular": model.basis_size_angular,
        "l_max":              [model.l_max_3b, model.has_q_222, model.has_q_1111,
                               model.has_q_112,
                               model.has_q_123, model.has_q_233, model.has_q_134],
        "neuron":             model.num_neurons,
    }
    if model.zbl is not None:
        new_config["zbl"] = model.zbl
        if getattr(model, "zbl_typewise_factor", None) is not None:
            new_config["typewise_cutoff_zbl_factor"] = model.zbl_typewise_factor
    if model.spin_mode:
        new_config.update({
            "spin_mode": model.spin_mode,
            "spin_compress": model.spin_compress,
            "spin_basis_size": list(model.spin_basis_size),
            "spin_l_max": list(model.spin_l_max),
            "spin_cutoff": list(model.spin_cutoff),
            "spin_cutoff_by_type": [
                model.spin_cutoff_by_type[index] for index in keep_idx
            ],
            "spin_order": model.spin_order,
            "spin_soc": model.spin_soc,
            "spin_curriculum": model.spin_curriculum,
            "spin_dof_type": [
                model.type_names[index]
                for index in keep_idx
                if bool(model.spin_dof_type_active[index].item())
            ],
            "spin_env_type": [
                model.type_names[index]
                for index in keep_idx
                if bool(model.spin_env_type_active[index].item())
            ],
        })

    dev = model.b1.device
    dtype = model.b1.dtype
    new_model = NEPModel(new_config).to(dtype).to(dev)

    # Fitting networks
    for new_i, old_i in enumerate(keep_idx):
        src = model.fitting_nets[old_i]
        dst = new_model.fitting_nets[new_i]
        dst.w0.data.copy_(src.w0.data)
        dst.b0.data.copy_(src.b0.data)
        dst.w1.data.copy_(src.w1.data)

    # Shared bias and q_scaler (independent of num_types)
    new_model.b1.data.copy_(model.b1.data)
    new_model.q_scaler.copy_(model.q_scaler)
    if new_model.energy_baseline is not None:
        new_model.energy_baseline.copy_(model.energy_baseline[keep_idx])

    # c_param: slice rows and columns for kept types
    with torch.no_grad():
        ix = torch.tensor(keep_idx, device=dev)
        new_model.c_param_2.data.copy_(model.c_param_2.data[ix][:, ix])
        if new_model.c_param_3 is not None:
            new_model.c_param_3.data.copy_(model.c_param_3.data[ix][:, ix])
        if new_model.c_param_spin is not None:
            new_model.c_param_spin.data.copy_(
                model.c_param_spin.data[:, ix][:, :, ix])
            for target, source in zip(
                spin_projection_parameters(new_model),
                spin_projection_parameters(model),
            ):
                target.copy_(source)

    return new_model
