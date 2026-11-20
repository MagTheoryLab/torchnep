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

import warnings

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

from .constants import (
    ELEMENTS, C3B, C4B, C5B, C4B2, COVALENT_RADIUS,
)
from . import ops
from .spin_schema import (normalize_spin_config, spin_projection_parameters,
                          read_nep_model, load_nep_weights, save_nep_model,
                          compute_spin_wall)
from .magnetic_polynomial import MagneticPolynomialDescriptor


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
        config = normalize_spin_config(config)
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

        self.spin_mode = config.get("spin_mode", 0)
        self.spin_version = 1 if self.spin_mode else None
        self.spin_order = config.get("spin_order", 2)
        self.spin_soc = config.get("spin_soc", 1)
        self.spin_compress = config.get("spin_compress", 3)
        self.spin_basis_size = config.get("spin_basis_size", [self.basis_size_angular, 0])
        self.spin_l_max = config.get("spin_l_max", [2, 0, 0])
        self.spin_cutoff_by_type = config.get("spin_cutoff_by_type", [self.rc_radial]*self.num_types)
        self.spin_cutoff = config.get("spin_cutoff", [self.rc_radial, self.rc_angular])
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

        active = torch.tensor([name in config.get("spin_type", self.type_names)
                               for name in self.type_names], dtype=torch.bool)
        self.register_buffer("spin_dof_type_active", active if self.spin_mode else None)

        self._set_spin_walls(config)

        # q_123 / q_233 (extra 4-body bispectrum) need L=3 moments.
        if (self.has_q_123 or self.has_q_233) and self.l_max_3b < 3:
            raise ValueError("q_123 / q_233 require l_max_3b >= 3")
        # q_134 couples L=1, L=3, L=4 moments, so it needs L=4 moments.
        if self.has_q_134 and self.l_max_3b < 4:
            raise ValueError("q_134 requires l_max_3b >= 4")
        # q_1111 is redundant: it equals const * (3-body L=1 descriptor)^2,
        # so it adds no information. It remains a supported c516 descriptor channel.
        if self.has_q_1111:
            warnings.warn(
                "has_q_1111 (l_max field 3) is set but is redundant — it "
                "equals a constant times the squared 3-body L=1 descriptor "
                "and adds no information. You can safely set it to 0.",
                stacklevel=2)

        # ZBL
        self.zbl_typewise_factor = None
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
            self.spin_descriptor = MagneticPolynomialDescriptor(
                spin_compress=self.spin_compress,
                spin_l_max=self.spin_l_max[0],
                spin_order=self.spin_order,
                spin_soc=bool(self.spin_soc),
                basis_size_radial=self.spin_basis_size[0],
                cutoff_radial=self.spin_cutoff[0],
                n_max_angular=self.n_max_angular,
            )
            self.spin_dim = self.spin_descriptor.dim
        self.dim = self.struct_dim + self.spin_dim
        if self.spin_mode and self.dim > 339:
            raise ValueError("GPUMD supports at most 339 total descriptors")
        self.num_lm = sum(2 * ll + 1 for ll in range(1, self.l_max_3b + 1))

        nt = self.num_types
        self.c_param_2 = nn.Parameter(torch.empty(
            nt, nt, self.n_max_radial + 1, self.basis_size_radial + 1))
        self.c_param_3 = nn.Parameter(torch.empty(
            nt, nt, self.n_max_angular + 1, self.basis_size_angular + 1)
        ) if self.l_max_3b > 0 else None
        nn.init.normal_(self.c_param_2, std=0.1)
        if self.c_param_3 is not None:
            nn.init.normal_(self.c_param_3, std=0.1)

        # Per-type fitting networks (no per-type bias — shared b1 below)
        self.fitting_nets = nn.ModuleList([
            FittingNet(self.dim, self.num_neurons)
            for _ in range(self.num_types)
        ])
        # One shared output bias (GPUMD convention: single common b1)
        self.b1 = nn.Parameter(torch.tensor(0.0))

        # q_scaler (computed from data, not learned)
        self.register_buffer("q_scaler", torch.ones(self.dim))
        self.register_buffer("energy_baseline", torch.zeros(self.num_types), persistent=True)
        self.register_buffer("_c3b", torch.tensor(C3B[:self.num_lm]))
        self.register_buffer("_c4b", torch.tensor(C4B))
        self.register_buffer("_c5b", torch.tensor(C5B))
        self.register_buffer("_c4b2", torch.tensor(C4B2))

    def _set_spin_walls(self, config, *, dtype=torch.float64, device=None):
        for key in ("spin_wall_high", "spin_wall_low"):
            if config.get(f"{key}_auto") is not None:
                raise ValueError(f"{key} auto requires training data; resolve it before loading a model")
            for suffix in ("s", "k"):
                name = f"{key}_{suffix}"
                values = list(config[name])
                setattr(self, name, values)
                self.register_buffer(f"_{name}", torch.tensor(values, dtype=dtype, device=device),
                                     persistent=False)
            setattr(self, f"has_{key}", any(k > 0.0 for k in config[f"{key}_k"]))
        self.has_spin_wall = self.has_spin_wall_high or self.has_spin_wall_low

    def architecture_signature(self):
        """Return the semantic architecture contract for binary checkpoints."""
        signature = {
            "format": "torchnep-c516-v1",
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
        if self.spin_mode:
            signature["spin"] = {
                "version": self.spin_version,
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
                "type": [
                    name for index, name in enumerate(self.type_names)
                    if bool(self.spin_dof_type_active[index].item())
                ],
            }
            if self.has_spin_wall_high:
                signature["spin"]["wall_high"] = [
                    v for pair in zip(self.spin_wall_high_s, self.spin_wall_high_k)
                    for v in pair
                ]
            if self.has_spin_wall_low:
                signature["spin"]["wall_low"] = [
                    v for pair in zip(self.spin_wall_low_s, self.spin_wall_low_k)
                    for v in pair
                ]
        return signature

    @torch.no_grad()
    def set_q_scaler(
        self,
        q_min: torch.Tensor,
        q_max: torch.Tensor,
        q_rms: Optional[torch.Tensor] = None,
    ):
        diff = q_max - q_min
        scaler = torch.clamp(diff, min=1e-10).reciprocal()
        if self.spin_mode:
            if q_rms is None or q_rms.shape != diff.shape:
                raise ValueError("nep_spin q_scaler requires one RMS value per channel")
            spin = slice(self.struct_dim, None)
            scaler[spin] = torch.maximum(q_rms[spin], diff[spin]).clamp_min(1).reciprocal()
        self.q_scaler.copy_(scaler)

    @torch.no_grad()
    def set_energy_baseline(self, baseline: torch.Tensor):
        if self.energy_baseline is None:
            raise ValueError(
                "model has no energy reference buffer")
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
            self.c_param_3, self.spin_dof_type_active,
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

        if self.has_spin_wall:
            e_wall, _ = compute_spin_wall(
                spins, atom_types,
                self._spin_wall_high_s, self._spin_wall_high_k,
                self._spin_wall_low_s, self._spin_wall_low_k,
            )
            Ei = Ei + e_wall

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

    def _structural_cached(self, batch, derivatives, backend):
        dtype, device, N = self.q_scaler.dtype, self.q_scaler.device, batch['N']
        # Descriptors from cached basis.
        # When forces needed: return intermediates (s, gn_ang) for analytical force.
        if derivatives:
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

        return q, s, gn_ang

    def _fit_cached(self, q, batch):
        dtype, N = q.dtype, q.shape[0]
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

        return Ei, Fp

    def _cached_core(self, batch, need_forces=True, need_virial=False,
                     backend: str = "loop", need_mforce=False, need_tau=False):
        """Descriptor + NN + analytical-force part of the cached compute.

        Deliberately free of data-dependent Python control flow — the NN
        dispatch is a branchless weight gather, and ZBL is the branch-free
        table variant (``ops.compute_zbl_pair``) with analytic pair
        gradients — so ``torch.compile`` captures this whole function as
        ONE graph with no breaks (the eager reference path's ``mask.any()``
        branches and inner ``autograd.grad`` each split the graph). Only
        result assembly lives in the ``compute_properties_cached`` wrapper.

        Returns ``(Ei, forces, virial)`` for ordinary NEP, with a fourth
        magnetic-force output for spin models. Unrequested outputs are None.
        """
        if self.spin_mode:
            return self._cached_spin_core(batch,need_forces,need_virial,backend,need_mforce or need_tau)
        dtype = self.q_scaler.dtype
        device = self.q_scaler.device
        N = batch["N"]

        q, s, gn_ang = self._structural_cached(batch, need_forces or need_virial, backend)
        Ei, Fp = self._fit_cached(q, batch)

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

    @torch.no_grad()
    def _prepare_spin_basis(self, batch):
        from .cuda_harmonics import harmonic_basis_and_derivatives
        tensors = tuple(batch[k] for k in ('rij_rad','rij_ang','rij_spin','spins','atom_types',
                                           'pair_i_spin','pair_j_spin'))
        tensors += (self.spin_dof_type_active,self.spin_cutoff_pair)
        previous = batch.get('_spin_basis_inputs')
        inference = torch.is_inference_mode_enabled()
        if (not inference and previous is not None and
                all(t is old and t._version == version for t,(old,version) in zip(tensors,previous))):
            return
        if previous is not None or inference:
            (batch['fk_rad'],batch['fkp_rad'],batch['d12inv_rad'],
             batch['fk_ang'],batch['fkp_ang'],batch['d12inv_ang'],batch['blm']) = ops.structural_basis(
                 batch['rij_rad'],batch['rij_ang'],self.rc_radial,self.rc_angular,
                 self.basis_size_radial,self.basis_size_angular,self.l_max_3b)
        pi,pj,types = batch['pair_i_spin'],batch['pair_j_spin'],batch['atom_types']
        scale = batch['rij_spin'].new_ones(pi.shape)
        if self.spin_has_typewise_cutoff:
            scale = self.spin_cutoff[0]/self.spin_cutoff_pair[types[pi],types[pj]]
        active = self.spin_dof_type_active[types].to(batch['spins'])
        batch['spin_basis'] = self.spin_descriptor.prepare_basis(batch['rij_spin'],batch['spins'],scale,active)
        hat = batch['rij_ang']*batch['d12inv_ang'][:,None]
        if 0 < self.l_max_3b <= 4:
            _, batch['dblm_ang'] = harmonic_basis_and_derivatives(hat,self.l_max_3b,True)
        else:
            batch['dblm_ang'] = ops._compute_dblm_dhat(*hat.unbind(-1),self.l_max_3b)
        batch['_spin_basis_inputs'] = None if inference else tuple((t,t._version) for t in tensors)

    def _cached_spin_core(self, batch, need_forces, need_virial, backend, need_mforce):
        n, types = batch['N'], batch['atom_types']
        derivative = need_forces or need_virial
        q, s, gn = self._structural_cached(batch,derivative,backend)
        basis = batch['spin_basis']
        pi,pj = batch['pair_i_spin'],batch['pair_j_spin']
        qs, intermediates = self.spin_descriptor._contract(basis['fk'],basis['geometry'],basis['phi'],
                                                          pi,pj,types,self.c_param_3)
        ei, fp = self._fit_cached(torch.cat((q,qs),-1),batch)
        gz = None
        if self.zbl is not None:
            ez, gz = ops.compute_zbl_pair(types,batch['pair_i_ang'],batch['pair_j_ang'],batch['rij_ang'],
                self.zbl_zizj_pair,self.zbl_a_inv_pair,self.zbl_rc_inner_pair,self.zbl_rc_outer_pair,
                need_grad=derivative)
            ei = ei.index_add(0,batch['pair_i_ang'],ez)
        forces = virial = mforces = None
        if derivative:
            forces,virial = ops.compute_analytical_forces(fp[:,:self.struct_dim],types,n,
                self.c_param_2,self.c_param_3,batch['fkp_rad'],batch['fkp_ang'],batch['blm'],
                batch['pair_i_rad'],batch['pair_j_rad'],batch['rij_rad'],batch['d12inv_rad'],
                batch['pair_i_ang'],batch['pair_j_ang'],batch['rij_ang'],batch['d12inv_ang'],s,gn,
                self.n_max_radial,self.n_max_angular,self.l_max_3b,self.has_q_222,self.has_q_1111,self.has_q_112,
                self.num_lm,self._c3b,self._c4b,self._c5b,self._c4b2,ei.dtype,ei.device,
                compute_virial=need_virial,backend=backend,has_q_123=self.has_q_123,
                has_q_233=self.has_q_233,has_q_134=self.has_q_134,g_extra_ang=gz,dblm_dhat=batch['dblm_ang'])
        if derivative or need_mforce:
            gr,gs = self.spin_descriptor.pullback(fp[:,self.struct_dim:],basis,intermediates,pi,pj,types,self.c_param_3)
            if derivative:
                empty_i = pi[:0]
                empty_r = batch['rij_spin'][:0]
                f,v = ops.accumulate_forces_virial(n,pi,pj,batch['rij_spin'],gr,
                    empty_i,empty_i,empty_r,empty_r,ei.dtype,ei.device)
                forces = forces+f
                if need_virial:
                    virial = virial+v
            if need_mforce:
                mforces = -gs
        if self.has_spin_wall:
            e_wall, mf_wall = compute_spin_wall(
                batch['spins'], types,
                self._spin_wall_high_s, self._spin_wall_high_k,
                self._spin_wall_low_s, self._spin_wall_low_k,
            )
            ei = ei + e_wall
            if need_mforce:
                mforces = mforces + mf_wall
        return ei,forces,virial,mforces

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
        if self.spin_mode:
            if torch.is_inference_mode_enabled():
                batch = dict(batch)
            self._prepare_spin_basis(batch)
            core = core_fn if core_fn is not None else self._cached_core
            need_mforce, need_tau = batch.get('need_mforce',False), batch.get('need_tau',False)
            ei, forces, virial, mforces = core(batch,need_forces=need_forces,
                need_virial=need_virial,backend=backend,need_mforce=need_mforce,need_tau=need_tau)
            etot = ei.new_zeros(batch['num_structures']).index_add(0,batch['struct_idx'],ei)
            result = dict(Ei=ei,Etot=etot)
            derivative_graph = self.training if create_graph is None else create_graph
            if need_forces:
                result['forces'] = forces if derivative_graph else forces.detach()
            if need_virial:
                result['virial'] = virial if derivative_graph else virial.detach()
            if need_mforce:
                result['mforces'] = mforces if derivative_graph else mforces.detach()
            if need_tau:
                tau = torch.cross(batch['spins'],mforces,dim=-1)
                result['tau'] = tau if derivative_graph else tau.detach()
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
        load_nep_weights(self, path)

    def save_nep_txt(self, path: str, max_NN_radial: int,
                     max_NN_angular: int):
        """Save model to the GPUMD c516 format_version 1 named-block format.

        ``max_NN_radial`` / ``max_NN_angular`` are mandatory: GPUMD's nep.txt
        parser requires the cutoff line to always carry 4 numbers
        ``cutoff <rc_R> <rc_A> <max_NN_R> <max_NN_A>``. Both must be > 0.
        Compute them from the training set via
        :func:`torchnep.train.compute_max_neighbors`.
        """
        save_nep_model(self, path, max_NN_radial, max_NN_angular)



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
    for parameter in spin_projection_parameters(model):
        torch.nn.init.uniform_(parameter, -1.0, 1.0)
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
            "model": "nep_spin",
            "spin_version": 1,
            "spin_compress": model.spin_compress,
            "spin_basis_size": list(model.spin_basis_size),
            "spin_l_max": list(model.spin_l_max),
            "spin_cutoff": list(model.spin_cutoff),
            "spin_cutoff_by_type": [
                model.spin_cutoff_by_type[index] for index in keep_idx
            ],
            "spin_order": model.spin_order,
            "spin_soc": model.spin_soc,
            "spin_type": [model.type_names[index] for index in keep_idx
                          if bool(model.spin_dof_type_active[index].item())],
        })
        if model.has_spin_wall_high:
            new_config["spin_wall_high"] = [
                v for index in keep_idx
                for v in (model.spin_wall_high_s[index], model.spin_wall_high_k[index])
            ]
        if model.has_spin_wall_low:
            new_config["spin_wall_low"] = [
                v for index in keep_idx
                for v in (model.spin_wall_low_s[index], model.spin_wall_low_k[index])
            ]

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
        for target, source in zip(spin_projection_parameters(new_model), spin_projection_parameters(model)):
            target.copy_(source)

    return new_model
