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
from typing import List

from .constants import (
    ELEMENTS, C3B, C4B, C5B, C4B2, COVALENT_RADIUS,
)
from . import ops
from .spin import SpinNEPLiteChiralDescriptor, SpinNEPLiteDescriptor
from .spin_schema import spin_checkpoint_lines


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

        # Spin NEP Lite protocol. The descriptor family is deliberately not a
        # configurable string: spin_mode=1 selects Lite and spin_chiral=1 adds
        # the chiral extension.
        if "spin_descriptor" in config:
            raise ValueError(
                "spin_descriptor is not supported; use spin_mode 1 and "
                "optional spin_chiral 1")
        self.spin_mode = int(config.get("spin_mode", 0))
        self.spin_chiral = int(config.get("spin_chiral", 0))
        if self.spin_mode not in (0, 1):
            raise ValueError("spin_mode must be 0 or 1")
        if self.spin_chiral not in (0, 1):
            raise ValueError("spin_chiral must be 0 or 1")
        if self.spin_chiral and not self.spin_mode:
            raise ValueError("spin_chiral requires spin_mode 1")
        self.spin_descriptor_kind = (
            "spin_nep_lite_chiral" if self.spin_chiral
            else "spin_nep_lite")
        self.spin_compress = int(config.get("spin_compress", 4))
        if self.spin_compress < 1:
            raise ValueError("spin_compress must be >= 1")

        def _spin_int_list(key, default, size):
            values = list(config.get(key, default))
            if len(values) != size:
                raise ValueError(f"{key} requires {size} values")
            return [int(value) for value in values]

        self.spin_n_max = _spin_int_list("spin_n_max", [3, 0], 2)
        self.spin_basis_size = _spin_int_list(
            "spin_basis_size", [3, 3], 2)
        self.spin_l_max = _spin_int_list("spin_l_max", [4, 0, 0], 3)
        spin_cutoff = list(config.get(
            "spin_cutoff", [self.rc_radial, self.rc_angular]))
        if len(spin_cutoff) != 2:
            raise ValueError("spin_cutoff requires two values")
        self.spin_cutoff = [float(value) for value in spin_cutoff]
        if min(self.spin_cutoff) <= 0:
            raise ValueError("spin_cutoff values must be > 0")

        def _active_types(key, default=None):
            names = config.get(key, default)
            active = torch.ones(self.num_types, dtype=torch.bool)
            is_set = names is not None
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

        spin_dof, self.is_spin_dof_type_set = _active_types("spin_dof_type")
        spin_env, self.is_spin_env_type_set = _active_types(
            "spin_env_type",
            config.get("spin_dof_type", None),
        )
        if bool((spin_dof & ~spin_env).any().item()):
            raise ValueError("spin_dof_type must be a subset of spin_env_type")
        self.register_buffer("spin_dof_type_active", spin_dof)
        self.register_buffer("spin_env_type_active", spin_env)

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
            descriptor_cls = (
                SpinNEPLiteChiralDescriptor if self.spin_chiral
                else SpinNEPLiteDescriptor)
            self.spin_descriptor = descriptor_cls(
                spin_compress=self.spin_compress,
                spin_l_max=self.spin_l_max[0],
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
            self.c_param_spin = nn.Parameter(torch.empty(
                1, nt, nt, self.spin_compress,
                self.spin_basis_size[0] + 1))
        nn.init.normal_(self.c_param_2, std=0.1)
        if self.c_param_3 is not None:
            nn.init.normal_(self.c_param_3, std=0.1)
        if self.c_param_spin is not None:
            self.c_param_spin.data.zero_()
            for index in range(min(
                    self.spin_compress, self.spin_basis_size[0] + 1)):
                self.c_param_spin.data[..., index, index] = 1.0
            self.c_param_spin.data.add_(
                torch.randn_like(self.c_param_spin) * 0.01)

        # Per-type fitting networks (no per-type bias — shared b1 below)
        self.fitting_nets = nn.ModuleList([
            FittingNet(self.dim, self.num_neurons)
            for _ in range(self.num_types)
        ])
        # One shared output bias (GPUMD convention: single common b1)
        self.b1 = nn.Parameter(torch.tensor(0.0))

        # q_scaler (computed from data, not learned)
        self.register_buffer("q_scaler", torch.ones(self.dim))
        self.register_buffer("energy_baseline", torch.zeros(self.num_types))
        self.register_buffer("_c3b", torch.tensor(C3B[:self.num_lm]))
        self.register_buffer("_c4b", torch.tensor(C4B))
        self.register_buffer("_c5b", torch.tensor(C5B))
        self.register_buffer("_c4b2", torch.tensor(C4B2))

    @torch.no_grad()
    def set_q_scaler(self, q_min: torch.Tensor, q_max: torch.Tensor):
        diff = torch.clamp(q_max - q_min, min=1e-10)
        self.q_scaler.copy_(1.0 / diff)

    @torch.no_grad()
    def set_energy_baseline(self, baseline: torch.Tensor):
        self.energy_baseline.copy_(baseline.to(
            dtype=self.energy_baseline.dtype,
            device=self.energy_baseline.device,
        ))

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
        q_spin = self.spin_descriptor(
            rij_spin, pi_spin, pj_spin, spins, N, atom_types,
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
        for t in range(self.num_types):
            mask = atom_types == t
            net = self.fitting_nets[t]
            if mask.any():
                Ei[mask] = net(q_scaled[mask])
            else:
                dummy_accum = dummy_accum + net(dummy_q).sum()
        Ei = Ei + dummy_accum * 0.0
        Ei = Ei - self.b1
        if include_baseline:
            Ei = Ei + self.energy_baseline[atom_types]
        return Ei

    def compute_properties(self, rij_rad, rij_ang, pi_rad, pj_rad,
                           pi_ang, pj_ang, atom_types, N,
                           struct_idx, num_structures,
                           need_forces=True, need_virial=False,
                           backend: str = "loop", spins=None,
                           need_mforce=False, need_tau=False,
                           rij_spin=None, pi_spin=None, pj_spin=None,
                           create_graph: bool | None = None):
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

    def compute_properties_cached(self, batch, need_forces=True, need_virial=False,
                                   backend: str = "loop",
                                   create_graph: bool | None = None):
        """Compute energy, forces, virial using precomputed basis.

        Uses fully analytical force computation — no create_graph=True needed.
        Forces are differentiable through c2, c3 (via Fp->NN weights and via s->c3).

        ``backend`` in {"loop", "bmm"} — see torchnep.ops.resolve_backend.
        """
        dtype = self.q_scaler.dtype
        device = self.q_scaler.device
        N = batch["N"]

        # Spin training uses the autograd path so forces and magnetic forces
        # remain differentiable through the descriptor. The structural-only
        # model keeps the fast analytical path below unchanged.
        if self.spin_descriptor is not None:
            with torch.enable_grad():
                return self.compute_properties(
                    batch["rij_rad"], batch["rij_ang"],
                    batch["pair_i_rad"], batch["pair_j_rad"],
                    batch["pair_i_ang"], batch["pair_j_ang"],
                    batch["atom_types"], N,
                    batch["struct_idx"], batch["num_structures"],
                    need_forces=need_forces,
                    need_virial=need_virial,
                    backend=backend,
                    spins=batch.get("spins"),
                    need_mforce=batch.get("need_mforce", False),
                    need_tau=batch.get("need_tau", False),
                    rij_spin=batch.get("rij_spin"),
                    pi_spin=batch.get("pair_i_spin"),
                    pj_spin=batch.get("pair_j_spin"),
                    create_graph=create_graph,
                )

        # Descriptors from cached basis.
        # When forces needed: return intermediates (s, gn_ang) for analytical force.
        if need_forces:
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

        # NN forward + Fp computation (differentiable through NN weights).
        #
        # Every per-type fitting net is touched in the graph on every forward
        # — even types with no atoms in this batch get a zeroed-out dummy pass.
        # This keeps DDP gradient bookkeeping consistent (no need for
        # find_unused_parameters=True) and, critically, avoids the implicit
        # /world_size gradient dilution that DDP applies to unused parameters
        # (which was biasing rare-type NNs toward lower effective LR).
        Ei = torch.zeros(N, dtype=dtype, device=device)
        Fp = torch.zeros(N, self.dim, dtype=dtype, device=device)
        dummy_accum = torch.zeros((), dtype=dtype, device=device)
        dummy_q = q_scaled[:1] if q_scaled.shape[0] > 0 else torch.zeros(
            1, self.dim, dtype=dtype, device=device)

        for t in range(self.num_types):
            mask = batch["atom_types"] == t
            net = self.fitting_nets[t]
            if mask.any():
                qt = q_scaled[mask]
                z = qt @ net.w0 - net.b0
                h = torch.tanh(z)
                Ei[mask] = h @ net.w1
                tanh_der = 1.0 - h * h
                Fp[mask] = (net.w1 * tanh_der) @ net.w0.T
            else:
                # Dummy forward (the * 0 below nulls the contribution but
                # keeps the net's parameters in the autograd graph).
                z_d = dummy_q @ net.w0 - net.b0
                h_d = torch.tanh(z_d)
                dummy_accum = dummy_accum + (h_d @ net.w1).sum()

        Fp = Fp * self.q_scaler  # absorb q_scaler into Fp
        # Nail the unused-type gradient path into Ei without changing its value.
        Ei = Ei + dummy_accum * 0.0
        Ei = Ei - self.b1  # subtract shared output bias

        # ZBL energy + forces (no trainable params; local autograd on rij_ang).
        # enable_grad: end-of-training predict_from_store wraps this call in
        # torch.no_grad(), under which Ei_zbl.requires_grad would be False
        # and the ZBL force contribution would be silently dropped.
        zbl_forces = None
        zbl_virial = None
        if self.zbl is not None:
            with torch.enable_grad():
                rij_zbl = batch["rij_ang"].detach().requires_grad_(True)
                Ei_zbl = ops.compute_zbl(
                    batch["atom_types"], batch["pair_i_ang"], batch["pair_j_ang"],
                    rij_zbl, N, self.atomic_numbers.tolist(),
                    self.zbl_rc_inner, self.zbl_rc_outer, self.zbl_typewise_factor,
                    getattr(self, "zbl_rc_inner_per_type", None),
                    getattr(self, "zbl_rc_outer_per_type", None), dtype, device)
                if need_forces and Ei_zbl.requires_grad:
                    g_zbl = torch.autograd.grad(Ei_zbl.sum(), rij_zbl,
                                                allow_unused=True)[0]
                else:
                    g_zbl = None
            Ei = Ei + Ei_zbl.detach()
            if g_zbl is not None:
                empty_i = torch.zeros(0, dtype=torch.long, device=device)
                empty_r = torch.zeros(0, 3, dtype=dtype, device=device)
                zbl_forces, zbl_virial = ops.accumulate_forces_virial(
                    N, empty_i, empty_i, empty_r, empty_r,
                    batch["pair_i_ang"], batch["pair_j_ang"],
                    batch["rij_ang"].detach(), g_zbl.detach(),
                    dtype, device,
                )

        Etot = torch.zeros(batch["num_structures"], dtype=dtype, device=device)
        Etot.scatter_add_(0, batch["struct_idx"], Ei)

        result = {"Ei": Ei, "Etot": Etot}

        if need_forces:
            # Analytical forces: fully differentiable through c2/c3 and NN weights (Fp).
            # No create_graph=True needed — chain rule is computed explicitly.
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
            )
            if zbl_forces is not None:
                forces = forces + zbl_forces
                if need_virial:
                    virial = virial + zbl_virial
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

        has_spin_baseline = False
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
                    "spin_descriptor is not supported by the standalone "
                    "Spin NEP Lite protocol")
            if parts[0] == "spin_baseline":
                if len(parts) != self.num_types + 1:
                    raise ValueError(
                        "spin_baseline must have one value per type")
                self.energy_baseline.copy_(torch.tensor(
                    [float(value) for value in parts[1:]],
                    dtype=self.energy_baseline.dtype,
                    device=self.energy_baseline.device,
                ))
                has_spin_baseline = True
        if self.spin_mode and not has_spin_baseline:
            raise ValueError("spin model is missing spin_baseline")

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

        # q_scaler (buffer — kept; the weights were trained against it)
        q_scaler = vals[idx:idx + dim]
        self.q_scaler.copy_(torch.from_numpy(q_scaler.copy()))

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
        model_tag = "nep4_spin1" if self.spin_mode else "nep4"
        lines.append(f"{model_tag}{zbl_suffix} {self.num_types} "
                     + " ".join(self.type_names))
        if self.spin_mode:
            spin_lines = spin_checkpoint_lines(self)
            lines.append(f"spin_mode 1 {len(spin_lines)}")
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

        for v in self.q_scaler.detach().cpu().numpy():
            lines.append(f"  {v:.10e}")

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")


@torch.no_grad()
def gpumd_init_parameters(model: NEPModel) -> None:
    """Re-initialise every trainable parameter uniform(-1, 1) in place — the
    SNES ``mu`` init GPUMD starts from (``snes.cu`` initialize_mu_and_sigma).

    Both the descriptor coefficients (``c_param_2`` / ``c_param_3``) and the
    fitting-network weights (``w0`` / ``b0`` / ``w1``) are covered, so a model
    trained on top of GPUMD's ``c=1`` q_scaler starts from the same
    large-amplitude landscape GPUMD's models inherit. Used for fresh training
    only (never on a fine-tuned / loaded model, whose weights must be kept).
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
            "spin_mode": 1,
            "spin_chiral": model.spin_chiral,
            "spin_compress": model.spin_compress,
            "spin_n_max": list(model.spin_n_max),
            "spin_basis_size": list(model.spin_basis_size),
            "spin_l_max": list(model.spin_l_max),
            "spin_cutoff": list(model.spin_cutoff),
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

    return new_model
