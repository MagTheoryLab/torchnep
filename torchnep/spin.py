"""Spin NEP Lite descriptors with an optional chiral extension."""

from dataclasses import dataclass
import math
from typing import List

import torch

from . import ops

@dataclass(frozen=True)
class SpinToken:
    name: str
    group: str
    start: int
    stop: int

def _sorted_key(indices):
    return tuple(sorted(indices))

def _degree_combos(degree):
    out = []
    for a in range(3):
        for b in range(a, 3):
            if degree == 2:
                out.append((a, b))
                continue
            for c in range(b, 3):
                if degree == 3:
                    out.append((a, b, c))
                    continue
                for d in range(c, 3):
                    out.append((a, b, c, d))
    return out

def _add_unit_sphere_poly(coeffs, indices, degree, scale):
    indices = tuple(indices)
    if len(indices) == degree:
        coeffs[_sorted_key(indices)] += scale
    elif degree == 3 and len(indices) == 1:
        for d in range(3):
            coeffs[_sorted_key(indices + (d, d))] += scale
    elif degree == 4 and len(indices) == 2:
        for d in range(3):
            coeffs[_sorted_key(indices + (d, d))] += scale
    elif degree == 4 and len(indices) == 0:
        for d in range(3):
            for e in range(3):
                coeffs[_sorted_key((d, d, e, e))] += scale
    else:
        raise ValueError("unsupported packed STF term")

def _spin_lite_chiral_packed_constants():
    combos3 = _degree_combos(3)
    combos4 = _degree_combos(4)
    index3 = {key: i for i, key in enumerate(combos3)}
    index4 = {key: i for i, key in enumerate(combos4)}
    m4_from_m3 = []
    m4_coord = []
    m3 = [[0.0 for _ in range(27)] for _ in combos3]
    m4 = [[0.0 for _ in range(81)] for _ in combos4]
    for key in combos4:
        m4_from_m3.append(index3[key[:3]])
        m4_coord.append(key[3])

    for a in range(3):
        for b in range(3):
            for c in range(3):
                flat = (a * 3 + b) * 3 + c
                coeffs = {key: 0.0 for key in combos3}
                _add_unit_sphere_poly(coeffs, (a, b, c), 3, 1.0)
                if a == b:
                    _add_unit_sphere_poly(coeffs, (c,), 3, -0.2)
                if a == c:
                    _add_unit_sphere_poly(coeffs, (b,), 3, -0.2)
                if b == c:
                    _add_unit_sphere_poly(coeffs, (a,), 3, -0.2)
                for key, value in coeffs.items():
                    m3[index3[key]][flat] = value

    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    flat = ((a * 3 + b) * 3 + c) * 3 + d
                    coeffs = {key: 0.0 for key in combos4}
                    _add_unit_sphere_poly(coeffs, (a, b, c, d), 4, 1.0)
                    for p, q, r, s in (
                            (a, b, c, d), (a, c, b, d), (a, d, b, c),
                            (b, c, a, d), (b, d, a, c), (c, d, a, b)):
                        if p == q:
                            _add_unit_sphere_poly(coeffs, (r, s), 4, -1.0 / 7.0)
                    for p, q, r, s in ((a, b, c, d), (a, c, b, d),
                                        (a, d, b, c)):
                        if p == q and r == s:
                            _add_unit_sphere_poly(coeffs, (), 4, 1.0 / 35.0)
                    for key, value in coeffs.items():
                        m4[index4[key]][flat] = value

    eps = [[[0.0 for _ in range(3)] for _ in range(3)] for _ in range(3)]
    eps[0][1][2] = eps[1][2][0] = eps[2][0][1] = 1.0
    eps[0][2][1] = eps[2][1][0] = eps[1][0][2] = -1.0
    return combos3, combos4, m4_from_m3, m4_coord, m3, m4, eps

def build_spin_nep_lite_schema(spin_compress: int = 4,
                               spin_l_max: int = 4,
                               split_l1_dot: bool = False
                               ) -> List[SpinToken]:
    schema = []
    pos = 0

    def add(name, group, width):
        nonlocal pos
        schema.append(SpinToken(name, group, pos, pos + width))
        pos += width

    add("moment_squared", "body1_moment", 1)
    add("moment_quartic", "body1_moment", 1)

    for name in (
            "edge_dot_sum", "edge_dot2_sum", "edge_neighbor_moment2",
            "edge_bond_axis_exchange"):
        add(name, "edge_scalar", spin_compress)

    if spin_l_max >= 0:
        add("spin_angular_l0", "spin_angular", spin_compress)
    if spin_l_max >= 1:
        for name in (
                "spin_angular_l1_longitudinal",
                "spin_angular_l1_cross",
                "spin_angular_l1_stf"):
            add(name, "spin_angular_l1_split", spin_compress)
    for ell in range(2, spin_l_max + 1):
        add(f"spin_angular_l{ell}", "spin_angular", spin_compress)

    add("single_ion_rank2", "single_ion", spin_compress)
    add("spin_angular_l0_dot_sibling", "dot_sibling", spin_compress)
    if spin_l_max >= 1:
        if split_l1_dot:
            for name in (
                    "spin_angular_l1_longitudinal_dot_sibling",
                    "spin_angular_l1_cross_dot_sibling",
                    "spin_angular_l1_stf_dot_sibling"):
                add(name, "dot_sibling", spin_compress)
        else:
            add("spin_angular_l1_dot_sibling", "dot_sibling", spin_compress)

    return schema

def build_spin_nep_lite_chiral_schema(spin_compress: int = 4,
                                      spin_l_max: int = 4,
                                      split_l1_dot: bool = False
                                      ) -> List[SpinToken]:
    schema = build_spin_nep_lite_schema(
        spin_compress, spin_l_max, split_l1_dot)
    pos = schema[-1].stop if schema else 0

    def add(name, group, width):
        nonlocal pos
        schema.append(SpinToken(name, group, pos, pos + width))
        pos += width

    chi_compress = min(2, spin_compress)
    add("lite_chiral_bulk", "lite_chiral", chi_compress)
    add("lite_chiral_polar", "lite_chiral", spin_compress)
    add("lite_chiral_pseudodeviator", "lite_chiral", spin_compress)
    return schema

class SpinNEPLiteDescriptor(torch.nn.Module):
    """NEP-style spin descriptor with fixed Legendre self-contractions."""

    def __init__(self, spin_compress: int = 4,
                 spin_l_max: int = 4,
                 basis_size_radial: int = None,
                 cutoff_radial: float = 5.0,
                 split_l1_dot: bool = False,
                 eps: float = 1.0e-12):
        super().__init__()
        # ponytail: l<=4 is the planned lite family; add a recurrence if higher l matters.
        if spin_l_max < 0 or spin_l_max > 4:
            raise ValueError("spin_nep_lite supports 0 <= spin_l_max[0] <= 4")
        if spin_compress < 1:
            raise ValueError("spin_compress must be >= 1")
        if basis_size_radial is None:
            basis_size_radial = spin_compress - 1
        if basis_size_radial + 1 < spin_compress:
            raise ValueError("basis_size_radial must cover spin_compress")
        if cutoff_radial <= 0.0:
            raise ValueError("spin cutoff must be > 0")

        self.spin_compress = int(spin_compress)
        self.basis_size_radial = int(basis_size_radial)
        self.cutoff_radial = float(cutoff_radial)
        self.eps = float(eps)
        self.spin_l_max = int(spin_l_max)
        self.split_l1_dot = bool(split_l1_dot)
        self.schema = build_spin_nep_lite_schema(
            self.spin_compress, self.spin_l_max, self.split_l1_dot)
        self.dim = self.schema[-1].stop if self.schema else 0

    def split_by_token(self, q: torch.Tensor) -> dict:
        self._validate_descriptor_dim(q)
        return {
            token.name: q[..., token.start:token.stop]
            for token in self.schema
        }

    def split_by_group(self, q: torch.Tensor) -> dict:
        self._validate_descriptor_dim(q)
        groups = {}
        for token in self.schema:
            groups.setdefault(token.group, []).append(
                q[..., token.start:token.stop])
        return {
            name: torch.cat(parts, dim=-1)
            for name, parts in groups.items()
        }

    def _validate_descriptor_dim(self, q: torch.Tensor):
        if q.shape[-1] != self.dim:
            raise ValueError(
                f"expected descriptor last dimension {self.dim}, "
                f"got {q.shape[-1]}")

    def radial_basis(self, distances: torch.Tensor, cutoff: float,
                     basis_size: int) -> torch.Tensor:
        if distances.numel() == 0:
            return distances.new_zeros((0, basis_size + 1))
        inside = distances < cutoff
        radial = ops.chebyshev_basis(
            distances.clamp(max=cutoff), cutoff, basis_size)
        radial = radial[..., :basis_size + 1]
        return radial * inside.to(dtype=distances.dtype).unsqueeze(-1)

    def _active_masks(self, atom_types, spins, spin_dof_type_active,
                      spin_env_type_active):
        dof_atoms = env_atoms = None
        if spin_dof_type_active is not None:
            if atom_types is None:
                raise ValueError(
                    "atom_types are required when spin_dof_type_active is set")
            dof_atoms = spin_dof_type_active.to(
                device=spins.device, dtype=torch.bool)[atom_types]
        if spin_env_type_active is not None:
            if atom_types is None:
                raise ValueError(
                    "atom_types are required when spin_env_type_active is set")
            env_atoms = spin_env_type_active.to(
                device=spins.device, dtype=torch.bool)[atom_types]
        elif dof_atoms is not None:
            env_atoms = dof_atoms
        return dof_atoms, env_atoms

    def _pair_active_scale(self, pair_i: torch.Tensor, pair_j: torch.Tensor,
                           dof_atoms: torch.Tensor, env_atoms: torch.Tensor,
                           dtype: torch.dtype):
        if dof_atoms is None and env_atoms is None:
            return None
        center_active = (torch.ones_like(pair_i, dtype=torch.bool)
                         if dof_atoms is None else dof_atoms[pair_i])
        neighbor_active = (torch.ones_like(pair_j, dtype=torch.bool)
                           if env_atoms is None else env_atoms[pair_j])
        return (center_active & neighbor_active).to(dtype=dtype)

    def _mixed_radial(self, rb: torch.Tensor, pair_i: torch.Tensor,
                      pair_j: torch.Tensor, atom_types: torch.Tensor,
                      coeff: torch.Tensor) -> torch.Tensor:
        if coeff is None:
            return rb[:, :self.spin_compress]
        if atom_types is None:
            raise ValueError("atom_types are required when spin coefficients are set")
        return _type_pair_radial_mixing(rb, pair_i, pair_j, atom_types, coeff)

    def _edge_direction(self, rij: torch.Tensor, dist: torch.Tensor):
        return rij / dist.unsqueeze(-1)

    def _edge_skeletons(self, rhat: torch.Tensor, sj: torch.Tensor,
                        include_rr_on_s: bool = True):
        cross = torch.cross(rhat, sj, dim=-1)
        stf_rs = _stf_outer(rhat, sj).reshape(rhat.shape[0], 9)
        rr_on_s = None
        if include_rr_on_s:
            rr = _stf_outer(rhat, rhat)
            rr_on_s = torch.matmul(rr, sj.unsqueeze(-1)).squeeze(-1)
        r_dot_s = (rhat * sj).sum(dim=-1, keepdim=True)
        return (cross, stf_rs, rr_on_s, r_dot_s)

    @staticmethod
    def _contract(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left * right).sum(dim=-1)

    def forward(self, rij: torch.Tensor, pair_i: torch.Tensor,
                pair_j: torch.Tensor, spins: torch.Tensor, N: int,
                atom_types: torch.Tensor = None,
                c_spin_2body: torch.Tensor = None,
                c_spin_3body: torch.Tensor = None,
                spin_dof_type_active: torch.Tensor = None,
                spin_env_type_active: torch.Tensor = None,
                rij_coupled: torch.Tensor = None,
                pair_i_coupled: torch.Tensor = None,
                pair_j_coupled: torch.Tensor = None) -> torch.Tensor:
        dof_atoms, env_atoms = self._active_masks(
            atom_types, spins, spin_dof_type_active, spin_env_type_active)
        spins_unit = spins
        spin2 = (spins_unit * spins_unit).sum(dim=-1)

        active = (torch.ones(N, dtype=spins.dtype, device=spins.device)
                  if dof_atoms is None else dof_atoms.to(dtype=spins.dtype))
        blocks = [torch.stack((spin2 * active, spin2 * spin2 * active), dim=-1)]

        if pair_i.numel() == 0:
            if self.dim > 2:
                blocks.append(spins.new_zeros((N, self.dim - 2)))
            return torch.cat(blocks, dim=-1)

        dist = torch.linalg.norm(rij, dim=-1).clamp_min(self.eps)
        rhat = self._edge_direction(rij, dist)
        rb = self.radial_basis(
            dist, self.cutoff_radial, self.basis_size_radial)
        pair_scale = self._pair_active_scale(
            pair_i, pair_j, dof_atoms, env_atoms, rb.dtype)
        if pair_scale is not None:
            rb = rb * pair_scale.unsqueeze(-1)
        coeff = None if c_spin_2body is None else c_spin_2body[0]
        weights = self._mixed_radial(rb, pair_i, pair_j, atom_types, coeff)

        si = spins_unit.index_select(0, pair_i)
        sj = spins_unit.index_select(0, pair_j)
        sj2 = spin2.index_select(0, pair_j)
        dot = (si * sj).sum(dim=-1)
        ri_dot_si = (rhat * si).sum(dim=-1)
        ri_dot_sj = (rhat * sj).sum(dim=-1)
        bond_axis = ri_dot_si * ri_dot_sj

        scalar_terms = torch.stack((dot, dot * dot, sj2, bond_axis), dim=-1)
        scalar_values = (
            scalar_terms[:, :, None] * weights[:, None, :]).reshape(
                pair_i.shape[0], 4 * self.spin_compress)
        blocks.append(_index_sum(spins, pair_i, scalar_values, N))

        rho0 = self._edge_density(sj, weights, pair_i, N)
        raw1 = None
        if self.spin_l_max >= 1 and not self.split_l1_dot:
            raw1 = self._edge_density(
                (rhat.unsqueeze(-1) * sj.unsqueeze(-2)).reshape(
                    rhat.shape[0], 9),
                weights, pair_i, N)
        angular_edges = []
        angular_widths = {}
        for ell in range(2, self.spin_l_max + 1):
            ylm = self._real_spherical_harmonics(rhat, ell)
            edge_values = (ylm.unsqueeze(-1) * sj.unsqueeze(1)).reshape(
                rhat.shape[0], -1)
            angular_edges.append((ell, edge_values))
            angular_widths[ell] = edge_values.shape[-1]
        angular_rho = {}
        if angular_edges:
            angular_all = self._edge_density(
                torch.cat([edge for _, edge in angular_edges], dim=-1),
                weights, pair_i, N)
            start = 0
            for ell, _ in angular_edges:
                width = angular_widths[ell]
                angular_rho[ell] = angular_all[:, :, start:start + width]
                start += width

        blocks.append(self._contract(rho0, rho0))

        if self.spin_l_max >= 1:
            cross, stf_rs, _, r_dot_s = self._edge_skeletons(
                rhat, sj, include_rr_on_s=False)
            l1_edge_values = (r_dot_s, cross, stf_rs)
            l1_bases = []
            l1_all = self._edge_density(
                torch.cat(l1_edge_values, dim=-1), weights, pair_i, N)
            start = 0
            for edge_values in l1_edge_values:
                width = edge_values.shape[-1]
                base = l1_all[:, :, start:start + width]
                start += width
                l1_bases.append(base)
                blocks.append(self._contract(base, base))

        for ell in range(2, self.spin_l_max + 1):
            block = angular_rho[ell]
            blocks.append(self._contract(block, block))

        geom_rank2 = self._edge_density(
            _stf_outer(rhat, rhat).reshape(rhat.shape[0], 9),
            weights, pair_i, N).reshape(N, self.spin_compress, 3, 3)
        single_ion = (
            geom_rank2
            * spins_unit[:, None, :, None]
            * spins_unit[:, None, None, :]).sum(dim=(-1, -2))
        blocks.append(single_ion)

        rho0_dot = self._edge_density(
            sj, weights * dot.unsqueeze(-1), pair_i, N)
        blocks.append(self._contract(rho0, rho0_dot))

        if self.spin_l_max >= 1:
            if self.split_l1_dot:
                for base, edge_values in zip(l1_bases, l1_edge_values):
                    dot_base = self._edge_density(
                        edge_values, weights * dot.unsqueeze(-1), pair_i, N)
                    blocks.append(self._contract(base, dot_base))
            else:
                raw1_dot = self._edge_density(
                    (rhat.unsqueeze(-1) * sj.unsqueeze(-2)).reshape(
                        rhat.shape[0], 9),
                    weights * dot.unsqueeze(-1), pair_i, N)
                blocks.append(self._contract(raw1, raw1_dot))

        blocks.extend(self._extra_blocks(
            rhat, weights, pair_i, pair_j, spins_unit, N, geom_rank2, si, sj))
        q = torch.cat(blocks, dim=-1)
        if q.shape[-1] != self.dim:
            raise RuntimeError("spin_nep_lite descriptor schema size mismatch")
        return q

    def forward_with_cache(self, rij: torch.Tensor, pair_i: torch.Tensor,
                           pair_j: torch.Tensor, spins: torch.Tensor, N: int,
                           atom_types: torch.Tensor = None,
                           c_spin_2body: torch.Tensor = None,
                           c_spin_3body: torch.Tensor = None,
                           spin_dof_type_active: torch.Tensor = None,
                           spin_env_type_active: torch.Tensor = None,
                           rij_coupled: torch.Tensor = None,
                           pair_i_coupled: torch.Tensor = None,
                           pair_j_coupled: torch.Tensor = None):
        del pair_i_coupled, pair_j_coupled
        return self._forward_with_manual_cache(
            rij, pair_i, pair_j, spins, N, atom_types,
            c_spin_2body, c_spin_3body, spin_dof_type_active,
            spin_env_type_active, rij_coupled)

    def spin_vjp_from_cache(self, Fp_spin: torch.Tensor, cache) -> torch.Tensor:
        return self._cached_manual_vjps(Fp_spin, cache, need_rij=False)[1]

    def rij_vjps_from_cache(self, Fp_spin: torch.Tensor, cache):
        grad_rij = self._cached_manual_vjps(
            Fp_spin, cache, need_rij=True)[0]
        rij_coupled = cache.get("rij_coupled")
        grad_coupled = (
            grad_rij.new_zeros((0, 3))
            if rij_coupled is None else torch.zeros_like(rij_coupled)
        )
        return grad_rij, grad_coupled

    def radial_basis_and_deriv(self, distances: torch.Tensor):
        if distances.numel() == 0:
            shape = (0, self.basis_size_radial + 1)
            return distances.new_zeros(shape), distances.new_zeros(shape)
        inside = distances < self.cutoff_radial
        radial, radial_deriv = ops.chebyshev_basis_and_deriv(
            distances.clamp(max=self.cutoff_radial),
            self.cutoff_radial, self.basis_size_radial)
        mask = inside.to(dtype=distances.dtype).unsqueeze(-1)
        return radial * mask, radial_deriv * mask

    def _forward_with_manual_cache(
            self, rij: torch.Tensor, pair_i: torch.Tensor,
            pair_j: torch.Tensor, spins: torch.Tensor, N: int,
            atom_types: torch.Tensor = None,
            c_spin_2body: torch.Tensor = None,
            c_spin_3body: torch.Tensor = None,
            spin_dof_type_active: torch.Tensor = None,
            spin_env_type_active: torch.Tensor = None,
            rij_coupled: torch.Tensor = None):
        del c_spin_3body
        dof_atoms, env_atoms = self._active_masks(
            atom_types, spins, spin_dof_type_active, spin_env_type_active)
        spins_unit = spins
        spin2 = (spins_unit * spins_unit).sum(dim=-1)

        active = (torch.ones(N, dtype=spins.dtype, device=spins.device)
                  if dof_atoms is None else dof_atoms.to(dtype=spins.dtype))
        blocks = [torch.stack((spin2 * active, spin2 * spin2 * active), dim=-1)]

        cache = {
            "N": N,
            "rij": rij,
            "rij_coupled": rij_coupled,
            "pair_i": pair_i,
            "pair_j": pair_j,
            "spins": spins,
            "spins_unit": spins_unit,
            "spin2": spin2,
            "active": active,
            "atom_types": atom_types,
            "dof_atoms": dof_atoms,
            "env_atoms": env_atoms,
            "c_spin_2body": c_spin_2body,
            "offset_after_local": 2,
        }
        if pair_i.numel() == 0:
            cache["empty"] = True
            if self.dim > 2:
                blocks.append(spins.new_zeros((N, self.dim - 2)))
            q = torch.cat(blocks, dim=-1)
            return q, cache

        dist = torch.linalg.norm(rij, dim=-1).clamp_min(self.eps)
        rhat = self._edge_direction(rij, dist)
        rb, rbp = self.radial_basis_and_deriv(dist)
        pair_scale = self._pair_active_scale(
            pair_i, pair_j, dof_atoms, env_atoms, rb.dtype)
        rb_scaled = rb if pair_scale is None else rb * pair_scale.unsqueeze(-1)
        coeff = None if c_spin_2body is None else c_spin_2body[0]
        weights = self._mixed_radial(rb_scaled, pair_i, pair_j,
                                     atom_types, coeff)

        si = spins_unit[pair_i]
        sj = spins_unit[pair_j]
        sj2 = spin2[pair_j]
        dot = (si * sj).sum(dim=-1)
        ri_dot_si = (rhat * si).sum(dim=-1)
        ri_dot_sj = (rhat * sj).sum(dim=-1)
        bond_axis = ri_dot_si * ri_dot_sj

        scalar_terms = torch.stack((dot, dot * dot, sj2, bond_axis), dim=-1)
        scalar_values = (
            scalar_terms[:, :, None] * weights[:, None, :]).reshape(
                pair_i.shape[0], 4 * self.spin_compress)
        blocks.append(_index_sum(spins, pair_i, scalar_values, N))

        rho0 = self._edge_density(sj, weights, pair_i, N)
        raw1 = None
        if self.spin_l_max >= 1 and not self.split_l1_dot:
            raw1_edge = (rhat.unsqueeze(-1) * sj.unsqueeze(-2)).reshape(
                rhat.shape[0], 9)
            raw1 = self._edge_density(raw1_edge, weights, pair_i, N)
        else:
            raw1_edge = None

        angular_ylm = {}
        angular_edges = []
        angular_widths = {}
        for ell in range(2, self.spin_l_max + 1):
            ylm = self._real_spherical_harmonics(rhat, ell)
            edge_values = (ylm.unsqueeze(-1) * sj.unsqueeze(1)).reshape(
                rhat.shape[0], -1)
            angular_ylm[ell] = ylm
            angular_edges.append((ell, edge_values))
            angular_widths[ell] = edge_values.shape[-1]
        angular_rho = {}
        if angular_edges:
            angular_all = self._edge_density(
                torch.cat([edge for _, edge in angular_edges], dim=-1),
                weights, pair_i, N)
            start = 0
            for ell, _ in angular_edges:
                width = angular_widths[ell]
                angular_rho[ell] = angular_all[:, :, start:start + width]
                start += width

        blocks.append(self._contract(rho0, rho0))

        l1_edge_values = ()
        l1_bases = []
        if self.spin_l_max >= 1:
            cross, stf_rs, _, r_dot_s = self._edge_skeletons(
                rhat, sj, include_rr_on_s=False)
            l1_edge_values = (r_dot_s, cross, stf_rs)
            l1_all = self._edge_density(
                torch.cat(l1_edge_values, dim=-1), weights, pair_i, N)
            start = 0
            for edge_values in l1_edge_values:
                width = edge_values.shape[-1]
                base = l1_all[:, :, start:start + width]
                start += width
                l1_bases.append(base)
                blocks.append(self._contract(base, base))

        for ell in range(2, self.spin_l_max + 1):
            block = angular_rho[ell]
            blocks.append(self._contract(block, block))

        stf_rr_edge = _stf_outer(rhat, rhat).reshape(rhat.shape[0], 9)
        geom_rank2 = self._edge_density(
            stf_rr_edge, weights, pair_i, N).reshape(
                N, self.spin_compress, 3, 3)
        single_ion = (
            geom_rank2
            * spins_unit[:, None, :, None]
            * spins_unit[:, None, None, :]).sum(dim=(-1, -2))
        blocks.append(single_ion)

        rho0_dot = self._edge_density(
            sj, weights * dot.unsqueeze(-1), pair_i, N)
        blocks.append(self._contract(rho0, rho0_dot))

        raw1_dot = None
        l1_dot_bases = []
        if self.spin_l_max >= 1:
            if self.split_l1_dot:
                for base, edge_values in zip(l1_bases, l1_edge_values):
                    dot_base = self._edge_density(
                        edge_values, weights * dot.unsqueeze(-1), pair_i, N)
                    l1_dot_bases.append(dot_base)
                    blocks.append(self._contract(base, dot_base))
            else:
                raw1_dot = self._edge_density(
                    raw1_edge, weights * dot.unsqueeze(-1), pair_i, N)
                blocks.append(self._contract(raw1, raw1_dot))

        cache.update({
            "empty": False,
            "dist": dist,
            "rhat": rhat,
            "rb": rb,
            "rbp": rbp,
            "pair_scale": pair_scale,
            "coeff": coeff,
            "weights": weights,
            "si": si,
            "sj": sj,
            "sj2": sj2,
            "dot": dot,
            "ri_dot_si": ri_dot_si,
            "ri_dot_sj": ri_dot_sj,
            "bond_axis": bond_axis,
            "rho0": rho0,
            "raw1": raw1,
            "raw1_edge": raw1_edge,
            "angular_ylm": angular_ylm,
            "angular_rho": angular_rho,
            "l1_edge_values": l1_edge_values,
            "l1_bases": l1_bases,
            "stf_rr_edge": stf_rr_edge,
            "geom_rank2": geom_rank2,
            "rho0_dot": rho0_dot,
            "raw1_dot": raw1_dot,
            "l1_dot_bases": l1_dot_bases,
        })
        blocks.extend(self._extra_blocks_cached(
            rhat, weights, pair_i, pair_j, spins_unit, N, cache, geom_rank2,
            si, sj))
        q = torch.cat(blocks, dim=-1)
        cache["dim_without_extra"] = q.shape[-1]
        if q.shape[-1] != self.dim:
            raise RuntimeError("spin_nep_lite descriptor schema size mismatch")
        return q, cache

    def _cached_manual_vjps(self, Fp_spin: torch.Tensor, cache,
                            need_rij: bool):
        cached = cache.get("_manual_vjps")
        fp_key = self._vjp_tensor_key(Fp_spin)
        if cached is not None and cached["fp_key"] == fp_key:
            if cached["has_rij"] or not need_rij:
                return cached["grad_rij"], cached["grad_spins"]
        grad_rij, grad_spins = self._manual_vjps(
            Fp_spin, cache, need_rij=need_rij)
        cache["_manual_vjps"] = {
            "fp_key": fp_key,
            "has_rij": need_rij,
            "grad_rij": grad_rij,
            "grad_spins": grad_spins,
        }
        return grad_rij, grad_spins

    @staticmethod
    def _vjp_tensor_key(tensor: torch.Tensor):
        return (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )

    def _manual_vjps(self, Fp_spin: torch.Tensor, cache, need_rij: bool):
        spins = cache["spins"]
        spins_unit = cache["spins_unit"]
        N = cache["N"]
        grad_spin_unit = torch.zeros_like(spins_unit)
        grad_rij = torch.zeros_like(cache["rij"])

        offset = 0
        spin2 = cache["spin2"]
        active = cache["active"]
        fp0 = Fp_spin[:, offset]
        fp1 = Fp_spin[:, offset + 1]
        scale = active * (2.0 * fp0 + 4.0 * fp1 * spin2)
        grad_spin_unit = grad_spin_unit + scale.unsqueeze(-1) * spins_unit
        offset += 2

        if cache.get("empty", False):
            return grad_rij, self._unit_spin_vjp(grad_spin_unit, cache)

        pair_i = cache["pair_i"]
        pair_j = cache["pair_j"]
        weights = cache["weights"]
        rhat = cache["rhat"]
        si = cache["si"]
        sj = cache["sj"]
        dot = cache["dot"]
        C = self.spin_compress

        grad_weight = torch.zeros_like(weights)
        grad_rhat = torch.zeros_like(rhat)
        grad_si = torch.zeros_like(si)
        grad_sj = torch.zeros_like(sj)
        grad_dot = torch.zeros_like(dot)

        def add_scalar_term(term):
            nonlocal offset, grad_weight
            alpha = Fp_spin[pair_i, offset:offset + C]
            grad_weight = grad_weight + alpha * term.unsqueeze(-1)
            offset += C
            return (alpha * weights).sum(dim=-1)

        grad_dot = grad_dot + add_scalar_term(dot)
        grad_dot = grad_dot + 2.0 * dot * add_scalar_term(dot * dot)
        grad_sj = grad_sj + 2.0 * add_scalar_term(
            cache["sj2"]).unsqueeze(-1) * sj
        g_axis = add_scalar_term(cache["bond_axis"])
        grad_si = grad_si + (g_axis * cache["ri_dot_sj"]).unsqueeze(-1) * rhat
        grad_sj = grad_sj + (g_axis * cache["ri_dot_si"]).unsqueeze(-1) * rhat
        grad_rhat = grad_rhat + (
            (g_axis * cache["ri_dot_sj"]).unsqueeze(-1) * si
            + (g_axis * cache["ri_dot_si"]).unsqueeze(-1) * sj)

        def density_vjp(edge_values, grad_density, mod=None):
            g = grad_density.index_select(0, pair_i)
            if mod is None:
                w_eff = weights
                grad_mod = None
                grad_weight_local = (
                    g * edge_values.unsqueeze(1)).sum(dim=-1)
            else:
                w_eff = weights * mod.unsqueeze(-1)
                grad_mod = (
                    g * weights.unsqueeze(-1)
                    * edge_values.unsqueeze(1)).sum(dim=(1, 2))
                grad_weight_local = (
                    g * edge_values.unsqueeze(1)).sum(dim=-1) * mod.unsqueeze(-1)
            grad_edge_values = (g * w_eff.unsqueeze(-1)).sum(dim=1)
            return grad_weight_local, grad_edge_values, grad_mod

        def apply_density(edge_values, grad_density, mod=None):
            nonlocal grad_weight
            gw, ge, gm = density_vjp(edge_values, grad_density, mod)
            grad_weight = grad_weight + gw
            return ge, gm

        rho0 = cache["rho0"]
        fp = Fp_spin[:, offset:offset + C]
        ge, _ = apply_density(sj, 2.0 * fp.unsqueeze(-1) * rho0)
        grad_sj = grad_sj + ge
        offset += C

        if self.spin_l_max >= 1:
            for block, edge_values, kind in zip(
                    cache["l1_bases"], cache["l1_edge_values"],
                    ("rdot", "cross", "stf_rs")):
                fp = Fp_spin[:, offset:offset + C]
                ge, _ = apply_density(
                    edge_values, 2.0 * fp.unsqueeze(-1) * block)
                grad_rhat, grad_sj = self._edge_value_vjp(
                    kind, ge, rhat, sj, grad_rhat, grad_sj)
                offset += C

        for ell in range(2, self.spin_l_max + 1):
            block = cache["angular_rho"][ell]
            fp = Fp_spin[:, offset:offset + C]
            ylm = cache["angular_ylm"][ell]
            edge_values = (ylm.unsqueeze(-1) * sj.unsqueeze(1)).reshape(
                rhat.shape[0], -1)
            ge, _ = apply_density(edge_values, 2.0 * fp.unsqueeze(-1) * block)
            grad_ys = ge.reshape(rhat.shape[0], ylm.shape[-1], 3)
            grad_sj = grad_sj + (grad_ys * ylm.unsqueeze(-1)).sum(dim=1)
            grad_ylm = (grad_ys * sj.unsqueeze(1)).sum(dim=-1)
            grad_rhat = grad_rhat + self._real_spherical_harmonics_vjp(
                rhat, ell, grad_ylm)
            offset += C

        fp = Fp_spin[:, offset:offset + C]
        geom_rank2 = cache["geom_rank2"]
        s_col = spins_unit.unsqueeze(1).unsqueeze(-1)
        grad_geom = fp[:, :, None, None] * (
            spins_unit[:, None, :, None] * spins_unit[:, None, None, :])
        grad_center = 2.0 * (
            fp[:, :, None] * torch.matmul(
                geom_rank2, s_col).squeeze(-1)).sum(dim=1)
        grad_spin_unit = grad_spin_unit + grad_center
        ge, _ = apply_density(cache["stf_rr_edge"], grad_geom.reshape(N, C, 9))
        grad_rhat = grad_rhat + _stf_outer_vjp(ge.reshape(-1, 3, 3), rhat)
        offset += C

        fp = Fp_spin[:, offset:offset + C]
        grad_rho0 = fp.unsqueeze(-1) * cache["rho0_dot"]
        grad_rho0_dot = fp.unsqueeze(-1) * rho0
        ge, _ = apply_density(sj, grad_rho0)
        grad_sj = grad_sj + ge
        ge, gm = apply_density(sj, grad_rho0_dot, dot)
        grad_sj = grad_sj + ge
        grad_dot = grad_dot + gm
        offset += C

        if self.spin_l_max >= 1:
            if self.split_l1_dot:
                for base, dot_base, edge_values, kind in zip(
                        cache["l1_bases"], cache["l1_dot_bases"],
                        cache["l1_edge_values"], ("rdot", "cross", "stf_rs")):
                    fp = Fp_spin[:, offset:offset + C]
                    grad_base = fp.unsqueeze(-1) * dot_base
                    grad_dot_base = fp.unsqueeze(-1) * base
                    ge, _ = apply_density(edge_values, grad_base)
                    grad_rhat, grad_sj = self._edge_value_vjp(
                        kind, ge, rhat, sj, grad_rhat, grad_sj)
                    ge, gm = apply_density(edge_values, grad_dot_base, dot)
                    grad_rhat, grad_sj = self._edge_value_vjp(
                        kind, ge, rhat, sj, grad_rhat, grad_sj)
                    grad_dot = grad_dot + gm
                    offset += C
            else:
                fp = Fp_spin[:, offset:offset + C]
                grad_raw1 = fp.unsqueeze(-1) * cache["raw1_dot"]
                grad_raw1_dot = fp.unsqueeze(-1) * cache["raw1"]
                ge, _ = apply_density(cache["raw1_edge"], grad_raw1)
                grad_rhat, grad_sj = self._edge_value_vjp(
                    "raw1", ge, rhat, sj, grad_rhat, grad_sj)
                ge, gm = apply_density(cache["raw1_edge"], grad_raw1_dot, dot)
                grad_rhat, grad_sj = self._edge_value_vjp(
                    "raw1", ge, rhat, sj, grad_rhat, grad_sj)
                grad_dot = grad_dot + gm
                offset += C

        grad_si = grad_si + grad_dot.unsqueeze(-1) * sj
        grad_sj = grad_sj + grad_dot.unsqueeze(-1) * si

        grad_state = {
            "offset": offset,
            "grad_weight": grad_weight,
            "grad_rhat": grad_rhat,
            "grad_si": grad_si,
            "grad_sj": grad_sj,
            "grad_spin_unit": grad_spin_unit,
        }
        grad_state = self._extra_vjp(Fp_spin, cache, grad_state, need_rij)
        offset = grad_state["offset"]
        grad_weight = grad_state["grad_weight"]
        grad_rhat = grad_state["grad_rhat"]
        grad_si = grad_state["grad_si"]
        grad_sj = grad_state["grad_sj"]
        grad_spin_unit = grad_state["grad_spin_unit"]
        if offset != self.dim:
            raise RuntimeError("spin_nep_lite VJP schema size mismatch")

        grad_spin_unit = _scatter_spin_add(grad_spin_unit, pair_i, grad_si)
        grad_spin_unit = _scatter_spin_add(grad_spin_unit, pair_j, grad_sj)
        if need_rij:
            grad_dist = self._radial_dist_vjp(cache, grad_weight)
            grad_rij = self._rhat_dist_to_rij_vjp(cache, grad_rhat, grad_dist)
        grad_spins = self._unit_spin_vjp(grad_spin_unit, cache)
        return grad_rij, grad_spins

    def _unit_spin_vjp(self, grad_spin_unit: torch.Tensor, cache):
        return grad_spin_unit

    def _radial_dist_vjp(self, cache, grad_weight):
        rb = cache["rb"]
        grad_rb = torch.zeros_like(rb)
        coeff = cache["coeff"]
        if coeff is None:
            grad_rb[:, :self.spin_compress] = grad_weight
        else:
            atom_types = cache["atom_types"]
            ti = atom_types[cache["pair_i"]]
            tj = atom_types[cache["pair_j"]]
            grad_rb = torch.einsum("es,esk->ek", grad_weight, coeff[ti, tj])
        pair_scale = cache["pair_scale"]
        if pair_scale is not None:
            grad_rb = grad_rb * pair_scale.unsqueeze(-1)
        return (grad_rb * cache["rbp"]).sum(dim=-1)

    def _rhat_dist_to_rij_vjp(self, cache, grad_rhat, grad_dist):
        rij = cache["rij"]
        dist = cache["dist"]
        rhat = cache["rhat"]
        tangential = grad_rhat - (
            grad_rhat * rhat).sum(dim=-1, keepdim=True) * rhat
        return (
            grad_dist.unsqueeze(-1) * rhat
            + tangential / dist.unsqueeze(-1)
        )

    def _edge_value_vjp(self, kind, grad_edge, rhat, sj, grad_rhat, grad_sj):
        if kind == "rdot":
            g = grad_edge[:, :1]
            grad_rhat = grad_rhat + g * sj
            grad_sj = grad_sj + g * rhat
        elif kind == "cross":
            grad_rhat = grad_rhat + torch.cross(sj, grad_edge, dim=-1)
            grad_sj = grad_sj + torch.cross(grad_edge, rhat, dim=-1)
        elif kind == "stf_rs":
            g = grad_edge.reshape(-1, 3, 3)
            grad_rhat = grad_rhat + _stf_outer_pair_vjp(g, sj)
            grad_sj = grad_sj + _stf_outer_pair_vjp(g, rhat)
        elif kind == "raw1":
            g = grad_edge.reshape(-1, 3, 3)
            grad_rhat = grad_rhat + torch.einsum("eab,eb->ea", g, sj)
            grad_sj = grad_sj + torch.einsum("eab,ea->eb", g, rhat)
        else:
            raise ValueError(f"unknown edge value kind {kind}")
        return grad_rhat, grad_sj

    def _extra_blocks_cached(
            self, rhat: torch.Tensor,
            weights: torch.Tensor, pair_i: torch.Tensor, pair_j: torch.Tensor,
            spins_unit: torch.Tensor, N: int, cache,
            geom_rank2: torch.Tensor = None,
            si: torch.Tensor = None, sj: torch.Tensor = None):
        del si, sj
        return self._extra_blocks(
            rhat, weights, pair_i, pair_j, spins_unit, N, geom_rank2)

    def _extra_vjp(self, Fp_spin: torch.Tensor, cache, grad_state,
                   need_rij: bool):
        return grad_state

    def _edge_chunks(self, num_edges: int):
        chunk = min(int(num_edges), 200_000)
        chunk = max(1, chunk)
        for start in range(0, int(num_edges), chunk):
            yield start, min(start + chunk, int(num_edges))

    def _polar_density_vjp(self, cache, grad_polar, grad_weight, grad_rhat):
        pair_i = cache["pair_i"]
        weights = cache["weights"]
        rhat = cache["rhat"]
        g = grad_polar[pair_i]
        grad_weight = grad_weight + (g * rhat.unsqueeze(1)).sum(dim=-1)
        grad_rhat = grad_rhat + (g * weights.unsqueeze(-1)).sum(dim=1)
        return grad_weight, grad_rhat

    def _spin_chiral_vjp(self, cache, grad_chiral, grad_si, grad_sj):
        si = cache["si"]
        sj = cache["sj"]
        grad_si = grad_si + torch.cross(sj, grad_chiral, dim=-1)
        grad_sj = grad_sj + torch.cross(grad_chiral, si, dim=-1)
        return grad_si, grad_sj

    @staticmethod
    def _odd_gate_vjp(r, pi, pj, grad_odd, grad_rhat, grad_pi, grad_pj):
        cross_pi_pj = torch.cross(pi, pj, dim=-1)
        grad_rhat = grad_rhat + (grad_odd.unsqueeze(-1) * cross_pi_pj).sum(dim=1)
        grad_pi = grad_pi + grad_odd.unsqueeze(-1) * torch.cross(
            pj, r, dim=-1)
        grad_pj = grad_pj + grad_odd.unsqueeze(-1) * torch.cross(
            r, pi, dim=-1)
        return grad_rhat, grad_pi, grad_pj

    @staticmethod
    def _bond_chiral_vjp(rhat, spin_chiral, grad_bond,
                         grad_rhat, grad_chiral):
        grad_rhat = grad_rhat + grad_bond.unsqueeze(-1) * spin_chiral
        grad_chiral = grad_chiral + grad_bond.unsqueeze(-1) * rhat
        return grad_rhat, grad_chiral

    def _real_spherical_harmonics_vjp(self, rhat: torch.Tensor,
                                      ell: int,
                                      grad_ylm: torch.Tensor) -> torch.Tensor:
        x, y, z = rhat.unbind(dim=-1)
        pi = math.pi
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(y)
        gz = torch.zeros_like(z)
        if ell == 2:
            c0 = math.sqrt(15.0 / (4.0 * pi))
            c2 = math.sqrt(5.0 / (16.0 * pi))
            c4 = math.sqrt(15.0 / (16.0 * pi))
            g = grad_ylm
            gx.add_(g[:, 0] * c0 * y)
            gy.add_(g[:, 0] * c0 * x)
            gy.add_(g[:, 1] * c0 * z)
            gz.add_(g[:, 1] * c0 * y)
            gx.sub_(g[:, 2] * 2.0 * c2 * x)
            gy.sub_(g[:, 2] * 2.0 * c2 * y)
            gz.add_(g[:, 2] * 4.0 * c2 * z)
            gx.add_(g[:, 3] * c0 * z)
            gz.add_(g[:, 3] * c0 * x)
            gx.add_(g[:, 4] * 2.0 * c4 * x)
            gy.sub_(g[:, 4] * 2.0 * c4 * y)
            return torch.stack((gx, gy, gz), dim=-1)
        if ell == 3:
            x2, y2, z2 = x * x, y * y, z * z
            rho2 = x2 + y2
            a = math.sqrt(35.0 / (32.0 * pi))
            b = math.sqrt(105.0 / (4.0 * pi))
            c = math.sqrt(21.0 / (32.0 * pi))
            d = math.sqrt(7.0 / (16.0 * pi))
            e = math.sqrt(105.0 / (16.0 * pi))
            g = grad_ylm
            gx.add_(g[:, 0] * 6.0 * a * x * y)
            gy.add_(g[:, 0] * 3.0 * a * (x2 - y2))
            gx.add_(g[:, 1] * b * y * z)
            gy.add_(g[:, 1] * b * x * z)
            gz.add_(g[:, 1] * b * x * y)
            gx.sub_(g[:, 2] * 2.0 * c * x * y)
            gy.add_(g[:, 2] * c * (4.0 * z2 - x2 - 3.0 * y2))
            gz.add_(g[:, 2] * 8.0 * c * y * z)
            gx.sub_(g[:, 3] * 6.0 * d * x * z)
            gy.sub_(g[:, 3] * 6.0 * d * y * z)
            gz.add_(g[:, 3] * d * (6.0 * z2 - 3.0 * rho2))
            gx.add_(g[:, 4] * c * (4.0 * z2 - 3.0 * x2 - y2))
            gy.sub_(g[:, 4] * 2.0 * c * x * y)
            gz.add_(g[:, 4] * 8.0 * c * x * z)
            gx.add_(g[:, 5] * 2.0 * e * x * z)
            gy.sub_(g[:, 5] * 2.0 * e * y * z)
            gz.add_(g[:, 5] * e * (x2 - y2))
            gx.add_(g[:, 6] * 3.0 * a * (x2 - y2))
            gy.sub_(g[:, 6] * 6.0 * a * x * y)
            return torch.stack((gx, gy, gz), dim=-1)
        if ell == 4:
            x2, y2, z2 = x * x, y * y, z * z
            a = 0.75 * math.sqrt(35.0 / pi)
            b = 0.75 * math.sqrt(35.0 / (2.0 * pi))
            c = 0.75 * math.sqrt(5.0 / pi)
            d = 0.75 * math.sqrt(5.0 / (2.0 * pi))
            e = (3.0 / 16.0) * math.sqrt(1.0 / pi)
            f = 0.375 * math.sqrt(5.0 / pi)
            h = (3.0 / 16.0) * math.sqrt(35.0 / pi)
            g = grad_ylm
            gx.add_(g[:, 0] * a * y * (3.0 * x2 - y2))
            gy.add_(g[:, 0] * a * x * (x2 - 3.0 * y2))
            gx.add_(g[:, 1] * 6.0 * b * x * y * z)
            gy.add_(g[:, 1] * 3.0 * b * z * (x2 - y2))
            gz.add_(g[:, 1] * b * y * (3.0 * x2 - y2))
            A = 7.0 * z2 - 1.0
            gx.add_(g[:, 2] * c * y * A)
            gy.add_(g[:, 2] * c * x * A)
            gz.add_(g[:, 2] * 14.0 * c * x * y * z)
            B = 7.0 * z2 - 3.0
            gy.add_(g[:, 3] * d * z * B)
            gz.add_(g[:, 3] * d * y * (21.0 * z2 - 3.0))
            gz.add_(g[:, 4] * e * (140.0 * z2 * z - 60.0 * z))
            gx.add_(g[:, 5] * d * z * B)
            gz.add_(g[:, 5] * d * x * (21.0 * z2 - 3.0))
            gx.add_(g[:, 6] * 2.0 * f * x * A)
            gy.sub_(g[:, 6] * 2.0 * f * y * A)
            gz.add_(g[:, 6] * 14.0 * f * z * (x2 - y2))
            gx.add_(g[:, 7] * 3.0 * b * z * (x2 - y2))
            gy.sub_(g[:, 7] * 6.0 * b * x * y * z)
            gz.add_(g[:, 7] * b * x * (x2 - 3.0 * y2))
            gx.add_(g[:, 8] * h * (4.0 * x * x2 - 12.0 * x * y2))
            gy.add_(g[:, 8] * h * (-12.0 * x2 * y + 4.0 * y * y2))
            return torch.stack((gx, gy, gz), dim=-1)
        raise ValueError(f"unsupported spherical harmonic order {ell}")

    def _extra_blocks(self, rhat: torch.Tensor, weights: torch.Tensor,
                      pair_i: torch.Tensor, pair_j: torch.Tensor,
                      spins_unit: torch.Tensor, N: int,
                      geom_rank2: torch.Tensor = None,
                      si: torch.Tensor = None, sj: torch.Tensor = None):
        del si, sj
        return ()

    def _real_spherical_harmonics(self, rhat: torch.Tensor,
                                  ell: int) -> torch.Tensor:
        x, y, z = rhat.unbind(dim=-1)
        pi = math.pi
        if ell == 2:
            x2, y2, z2 = x * x, y * y, z * z
            c0 = math.sqrt(15.0 / (4.0 * pi))
            return torch.stack([
                c0 * x * y,
                c0 * y * z,
                math.sqrt(5.0 / (16.0 * pi)) * (2.0 * z2 - x2 - y2),
                c0 * x * z,
                math.sqrt(15.0 / (16.0 * pi)) * (x2 - y2),
            ], dim=-1)
        if ell == 3:
            x2, y2, z2 = x * x, y * y, z * z
            rho2 = x2 + y2
            z_gate = 4.0 * z2 - rho2
            xy_diff = x2 - y2
            return torch.stack([
                math.sqrt(35.0 / (32.0 * pi)) * y * (3.0 * x2 - y2),
                math.sqrt(105.0 / (4.0 * pi)) * x * y * z,
                math.sqrt(21.0 / (32.0 * pi)) * y * z_gate,
                math.sqrt(7.0 / (16.0 * pi)) * z * (2.0 * z2 - 3.0 * rho2),
                math.sqrt(21.0 / (32.0 * pi)) * x * z_gate,
                math.sqrt(105.0 / (16.0 * pi)) * z * xy_diff,
                math.sqrt(35.0 / (32.0 * pi)) * x * (x2 - 3.0 * y2),
            ], dim=-1)
        if ell == 4:
            x2, y2, z2 = x * x, y * y, z * z
            z_gate1 = 7.0 * z2 - 1.0
            z_gate3 = 7.0 * z2 - 3.0
            xy_diff = x2 - y2
            x2_3y2 = x2 - 3.0 * y2
            x2_3 = 3.0 * x2 - y2
            return torch.stack([
                0.75 * math.sqrt(35.0 / pi) * x * y * xy_diff,
                0.75 * math.sqrt(35.0 / (2.0 * pi)) * y * z * x2_3,
                0.75 * math.sqrt(5.0 / pi) * x * y * z_gate1,
                0.75 * math.sqrt(5.0 / (2.0 * pi)) * y * z * z_gate3,
                (3.0 / 16.0) * math.sqrt(1.0 / pi) * (
                    35.0 * z2 * z2 - 30.0 * z2 + 3.0),
                0.75 * math.sqrt(5.0 / (2.0 * pi)) * x * z * z_gate3,
                0.375 * math.sqrt(5.0 / pi) * xy_diff * z_gate1,
                0.75 * math.sqrt(35.0 / (2.0 * pi)) * x * z * x2_3y2,
                (3.0 / 16.0) * math.sqrt(35.0 / pi) * (
                    x2 * x2 - 6.0 * x2 * y2 + y2 * y2),
            ], dim=-1)
        raise ValueError(f"unsupported spherical harmonic order {ell}")

    def _edge_density(self, edge_values: torch.Tensor, weights: torch.Tensor,
                      pair_i: torch.Tensor, N: int) -> torch.Tensor:
        if edge_values.device.type != "mps":
            width = edge_values.shape[-1]
            C = weights.shape[-1]
            out = edge_values.new_zeros((N * C, width))
            channel = torch.arange(C, device=pair_i.device)
            chunk_budget = (
                64_000_000 if edge_values.device.type == "cuda"
                else 8_000_000)
            chunk = max(1, min(
                edge_values.shape[0],
                chunk_budget // max(width * C, 1)))
            for start in range(0, edge_values.shape[0], chunk):
                end = min(start + chunk, edge_values.shape[0])
                idx = (pair_i[start:end, None] * C + channel).reshape(-1)
                values = (
                    edge_values[start:end, None, :]
                    * weights[start:end, :, None]
                ).reshape(-1, width)
                out.scatter_add_(0, idx[:, None].expand_as(values), values)
            return out.reshape(N, C, width)

        out = edge_values.new_zeros(
            (N, self.spin_compress, edge_values.shape[-1]))
        width = edge_values.shape[-1]
        chunk = max(1, min(edge_values.shape[0], 8_000_000 // max(width, 1)))
        for n in range(self.spin_compress):
            for start in range(0, edge_values.shape[0], chunk):
                end = min(start + chunk, edge_values.shape[0])
                values = edge_values[start:end] * weights[start:end, n:n + 1]
                out[:, n, :] = out[:, n, :] + _index_sum(
                    out[:, n, :], pair_i[start:end], values, N)
        return out

class SpinNEPLiteChiralDescriptor(SpinNEPLiteDescriptor):
    """Lite descriptor plus parity-correct chi/P/T DMI-like tokens."""

    def __init__(self, spin_compress: int = 4,
                 spin_l_max: int = 4,
                 basis_size_radial: int = None,
                 cutoff_radial: float = 5.0,
                 split_l1_dot: bool = False,
                 eps: float = 1.0e-12):
        super().__init__(
            spin_compress=spin_compress,
            spin_l_max=spin_l_max,
            basis_size_radial=basis_size_radial,
            cutoff_radial=cutoff_radial,
            split_l1_dot=split_l1_dot,
            eps=eps)
        self.schema = build_spin_nep_lite_chiral_schema(
            self.spin_compress, self.spin_l_max, self.split_l1_dot)
        self.dim = self.schema[-1].stop if self.schema else 0
        self.chi_compress = min(2, self.spin_compress)
        combos3, _, m4_from_m3, m4_coord, m3, m4, eps_tensor = (
            _spin_lite_chiral_packed_constants())
        self.register_buffer(
            "_chiral_m3_index", torch.tensor(combos3, dtype=torch.long),
            persistent=False)
        self.register_buffer(
            "_chiral_m4_from_m3", torch.tensor(m4_from_m3, dtype=torch.long),
            persistent=False)
        self.register_buffer(
            "_chiral_m4_coord", torch.tensor(m4_coord, dtype=torch.long),
            persistent=False)
        self.register_buffer(
            "_chiral_m3_unpack", torch.tensor(m3, dtype=torch.float64),
            persistent=False)
        self.register_buffer(
            "_chiral_m4_unpack", torch.tensor(m4, dtype=torch.float64),
            persistent=False)
        self.register_buffer(
            "_chiral_eps", torch.tensor(eps_tensor, dtype=torch.float64),
            persistent=False)

    def forward_with_cache(self, rij: torch.Tensor, pair_i: torch.Tensor,
                           pair_j: torch.Tensor, spins: torch.Tensor, N: int,
                           atom_types: torch.Tensor = None,
                           c_spin_2body: torch.Tensor = None,
                           c_spin_3body: torch.Tensor = None,
                           spin_dof_type_active: torch.Tensor = None,
                           spin_env_type_active: torch.Tensor = None,
                           rij_coupled: torch.Tensor = None,
                           pair_i_coupled: torch.Tensor = None,
                           pair_j_coupled: torch.Tensor = None):
        del pair_i_coupled, pair_j_coupled
        return self._forward_with_manual_cache(
            rij, pair_i, pair_j, spins, N, atom_types,
            c_spin_2body, c_spin_3body, spin_dof_type_active,
            spin_env_type_active, rij_coupled)

    def _extra_blocks_cached(
            self, rhat: torch.Tensor,
            weights: torch.Tensor, pair_i: torch.Tensor, pair_j: torch.Tensor,
            spins_unit: torch.Tensor, N: int, cache,
            geom_rank2: torch.Tensor = None,
            si: torch.Tensor = None, sj: torch.Tensor = None):
        fields = self._chiral_fields(weights, rhat, pair_i, N, geom_rank2)
        cache["lite_chiral_fields"] = fields
        return self._chiral_blocks(
            rhat, weights, pair_i, pair_j, spins_unit, fields, si, sj)

    def _extra_blocks(self, rhat: torch.Tensor, weights: torch.Tensor,
                      pair_i: torch.Tensor, pair_j: torch.Tensor,
                      spins_unit: torch.Tensor, N: int,
                      geom_rank2: torch.Tensor = None,
                      si: torch.Tensor = None, sj: torch.Tensor = None):
        fields = self._chiral_fields(weights, rhat, pair_i, N, geom_rank2)
        return self._chiral_blocks(
            rhat, weights, pair_i, pair_j, spins_unit, fields, si, sj)

    def _chiral_blocks(self, rhat, weights, pair_i, pair_j, spins_unit,
                       fields, si=None, sj=None):
        if si is None:
            si = spins_unit[pair_i]
        if sj is None:
            sj = spins_unit[pair_j]
        cross = torch.cross(si, sj, dim=-1)

        chi_i = fields["chirals"].index_select(0, pair_i)
        chi_weights = weights[:, :self.chi_compress]
        chi_axis = rhat[:, None, :] * chi_i[:, :, None]

        polar_axis = torch.cross(
            fields["polars"].index_select(0, pair_i),
            rhat.unsqueeze(1), dim=-1)

        pseudo_axis = (
            fields["pseudodevs"].index_select(0, pair_i)
            * rhat[:, None, None, :]
        ).sum(dim=-1)

        axes = torch.cat((chi_axis, polar_axis, pseudo_axis), dim=1)
        token_weights = torch.cat((chi_weights, weights, weights), dim=-1)
        edge_tokens = (cross[:, None, :] * axes).sum(dim=-1) * token_weights

        tokens = _index_sum(
            rhat, pair_i, edge_tokens, spins_unit.shape[0])
        return (
            tokens[:, :self.chi_compress],
            tokens[:, self.chi_compress:self.chi_compress + self.spin_compress],
            tokens[:, self.chi_compress + self.spin_compress:],
        )

    def _chiral_fields(self, weights: torch.Tensor, rhat: torch.Tensor,
                       pair_i: torch.Tensor, N: int,
                       geom_rank2: torch.Tensor = None):
        if geom_rank2 is None:
            r_stf = _stf_outer(rhat, rhat)
            struct_polar_edge = torch.cat(
                (r_stf.reshape(rhat.shape[0], 9), rhat), dim=-1)
            struct_polar = _index_sum(
                rhat, pair_i,
                (struct_polar_edge[:, None] * weights[:, :, None]).reshape(
                    pair_i.shape[0], self.spin_compress * 12),
                N,
            ).reshape(N, self.spin_compress, 12)
            structs = struct_polar[:, :, :9].reshape(
                N, self.spin_compress, 3, 3)
            polars = struct_polar[:, :, 9:]
        else:
            structs = geom_rank2
            polars = self._edge_density(rhat, weights, pair_i, N)

        chirals, bulk_cache = self._bulk_chirality(
            weights[:, :self.chi_compress], rhat, pair_i, N,
            structs[:, :self.chi_compress], return_cache=True)
        struct_r = (
            structs[pair_i] * rhat[:, None, None, :]
        ).sum(dim=-1)
        pseudo_edge = _stf_outer(
            torch.cross(rhat.unsqueeze(1), struct_r, dim=-1),
            rhat.unsqueeze(1).expand_as(struct_r))
        pseudodevs = _index_sum(
            rhat, pair_i,
            (pseudo_edge * weights[:, :, None, None]).reshape(
                pair_i.shape[0], self.spin_compress * 9),
            N,
        ).reshape(N, self.spin_compress, 3, 3)
        return {
            "structs": structs,
            "polars": polars,
            "chirals": chirals,
            "pseudodevs": pseudodevs,
            "bulk": bulk_cache,
            "struct_r": struct_r,
            "pseudo_edge": pseudo_edge,
        }

    def _bulk_chirality(self, weights: torch.Tensor, rhat: torch.Tensor,
                        pair_i: torch.Tensor, N: int,
                        Q: torch.Tensor = None, return_cache: bool = False):
        num_channels = weights.shape[1]
        u = rhat

        if Q is None:
            uu = torch.einsum("ea,eb->eab", u, u)
            I = torch.eye(3, dtype=rhat.dtype, device=rhat.device)
            Q_edge = uu - I[None] / 3.0
            Q = _index_sum(
                rhat, pair_i,
                (Q_edge[:, None] * weights[:, :, None, None]).reshape(
                    pair_i.shape[0], num_channels * 9),
                N,
            ).reshape(N, num_channels, 3, 3)

        m3 = u[:, self._chiral_m3_index].prod(dim=-1)
        m4 = m3[:, self._chiral_m4_from_m3] * u[:, self._chiral_m4_coord]
        packed = self._edge_density(
            torch.cat((m3, m4), dim=-1), weights, pair_i, N)
        p3 = packed[:, :, :m3.shape[-1]].reshape(-1, m3.shape[-1])
        p4 = packed[:, :, m3.shape[-1]:].reshape(-1, m4.shape[-1])
        O = torch.matmul(p3, self._chiral_m3_unpack).reshape(
            N, num_channels, 3, 3, 3)
        H = torch.matmul(p4, self._chiral_m4_unpack).reshape(
            N, num_channels, 3, 3, 3, 3)
        q_vec = Q.transpose(-1, -2)
        o_vec = O.permute(0, 1, 3, 4, 2)
        h_vec = H.permute(0, 1, 3, 4, 5, 2)
        chirals = (torch.cross(
            q_vec[:, :, :, None, None, :],
            o_vec[:, :, None, :, :, :],
            dim=-1) * h_vec).sum(dim=(-1, -2, -3, -4))
        if not return_cache:
            return chirals
        return chirals, {
            "m3": m3,
            "m4": m4,
            "O": O,
            "H": H,
        }

    def _extra_vjp(self, Fp_spin: torch.Tensor, cache, grad_state,
                   need_rij: bool):
        del need_rij
        offset = grad_state["offset"]
        grad_weight = grad_state["grad_weight"]
        grad_rhat = grad_state["grad_rhat"]
        grad_si = grad_state["grad_si"]
        grad_sj = grad_state["grad_sj"]
        pair_i = cache["pair_i"]
        weights = cache["weights"]
        rhat = cache["rhat"]
        si = cache["si"]
        sj = cache["sj"]
        fields = cache["lite_chiral_fields"]
        C = self.spin_compress
        chiC = self.chi_compress

        spin_chiral = torch.cross(si, sj, dim=-1)
        grad_chiral = torch.zeros_like(spin_chiral)
        grad_chi = fields["chirals"].new_zeros(fields["chirals"].shape)
        grad_polar = fields["polars"].new_zeros(fields["polars"].shape)
        grad_pseudodev = fields["pseudodevs"].new_zeros(
            fields["pseudodevs"].shape)

        alpha = Fp_spin[pair_i, offset:offset + chiC]
        chi_i = fields["chirals"][pair_i]
        xu = (spin_chiral * rhat).sum(dim=-1)
        grad_weight_chi = alpha * xu.unsqueeze(-1) * chi_i
        grad_weight = torch.cat((
            grad_weight[:, :chiC] + grad_weight_chi,
            grad_weight[:, chiC:],
        ), dim=-1)
        aw = alpha * weights[:, :chiC]
        grad_chi = grad_chi.index_add_(
            0, pair_i, aw * xu.unsqueeze(-1))
        grad_chiral = grad_chiral + (
            aw * chi_i).sum(dim=-1, keepdim=True) * rhat
        grad_rhat = grad_rhat + (
            aw * chi_i).sum(dim=-1, keepdim=True) * spin_chiral
        offset += chiC

        alpha = Fp_spin[pair_i, offset:offset + C]
        polar_i = fields["polars"][pair_i]
        polar_axis = torch.cross(polar_i, rhat.unsqueeze(1), dim=-1)
        xa = (spin_chiral.unsqueeze(1) * polar_axis).sum(dim=-1)
        grad_weight = grad_weight + alpha * xa
        aw = alpha * weights
        grad_chiral = grad_chiral + (
            aw.unsqueeze(-1) * polar_axis).sum(dim=1)
        grad_axis = aw.unsqueeze(-1) * spin_chiral.unsqueeze(1)
        grad_polar = grad_polar.index_add_(
            0, pair_i,
            torch.cross(
                rhat.unsqueeze(1).expand_as(grad_axis),
                grad_axis, dim=-1))
        grad_rhat = grad_rhat + torch.cross(
            grad_axis, polar_i, dim=-1).sum(dim=1)
        offset += C

        alpha = Fp_spin[pair_i, offset:offset + C]
        pseudodev_i = fields["pseudodevs"][pair_i]
        pseudo_axis = torch.matmul(
            pseudodev_i, rhat[:, None, :, None]).squeeze(-1)
        xa = (spin_chiral.unsqueeze(1) * pseudo_axis).sum(dim=-1)
        grad_weight = grad_weight + alpha * xa
        aw = alpha * weights
        grad_chiral = grad_chiral + (
            aw.unsqueeze(-1) * pseudo_axis).sum(dim=1)
        grad_axis = aw.unsqueeze(-1) * spin_chiral.unsqueeze(1)
        grad_pseudodev = grad_pseudodev.index_add_(
            0, pair_i,
            grad_axis.unsqueeze(-1) * rhat[:, None, None, :])
        grad_rhat = grad_rhat + torch.matmul(
            pseudodev_i.transpose(-1, -2),
            grad_axis.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        offset += C

        grad_si = grad_si + torch.cross(sj, grad_chiral, dim=-1)
        grad_sj = grad_sj + torch.cross(grad_chiral, si, dim=-1)

        grad_Q = torch.zeros_like(fields["structs"])
        grad_O, grad_H, grad_Q_chi = self._bulk_chirality_vjp(
            grad_chi, fields)
        grad_Q = torch.cat((
            grad_Q[:, :chiC] + grad_Q_chi,
            grad_Q[:, chiC:],
        ), dim=1)

        grad_weight, grad_rhat = self._polar_density_vjp(
            cache, grad_polar, grad_weight, grad_rhat)
        grad_weight, grad_rhat, grad_Q_pseudo = self._pseudodev_vjp(
            cache, fields, grad_pseudodev, grad_weight, grad_rhat)
        grad_Q = grad_Q + grad_Q_pseudo
        grad_weight, grad_rhat = self._qoh_density_vjp(
            cache, fields, grad_Q, grad_O, grad_H,
            grad_weight, grad_rhat)

        grad_state.update({
            "offset": offset,
            "grad_weight": grad_weight,
            "grad_rhat": grad_rhat,
            "grad_si": grad_si,
            "grad_sj": grad_sj,
        })
        return grad_state

    def _bulk_chirality_vjp(self, grad_chi, fields):
        bulk = fields["bulk"]
        Q = fields["structs"][:, :self.chi_compress]
        O = bulk["O"]
        H = bulk["H"]
        eps = self._chiral_eps.to(dtype=Q.dtype, device=Q.device)
        grad_Q = torch.einsum(
            "xab,nkbef,nkxdef,nk->nkad", eps, O, H, grad_chi)
        grad_O = torch.einsum(
            "xab,nkad,nkxdef,nk->nkbef", eps, Q, H, grad_chi)
        grad_H = torch.einsum(
            "xab,nkad,nkbef,nk->nkxdef", eps, Q, O, grad_chi)
        return grad_O, grad_H, grad_Q

    def _pseudodev_vjp(self, cache, fields, grad_pseudodev,
                       grad_weight, grad_rhat):
        pair_i = cache["pair_i"]
        weights = cache["weights"]
        rhat = cache["rhat"]
        structs_i = fields["structs"][pair_i]
        struct_r = fields["struct_r"]
        pseudo_edge = fields["pseudo_edge"]
        grad_center = grad_pseudodev[pair_i]

        grad_weight = grad_weight + (
            grad_center * pseudo_edge).sum(dim=(-1, -2))
        grad_stf = grad_center * weights[:, :, None, None]
        pseudo_axis = torch.cross(
            rhat.unsqueeze(1), struct_r, dim=-1)
        grad_axis = _stf_outer_pair_vjp(
            grad_stf.reshape(-1, 3, 3),
            rhat[:, None, :].expand_as(struct_r).reshape(-1, 3),
        ).reshape_as(struct_r)
        grad_u = _stf_outer_pair_vjp(
            grad_stf.reshape(-1, 3, 3),
            pseudo_axis.reshape(-1, 3),
        ).reshape_as(struct_r)

        grad_u = grad_u + torch.cross(struct_r, grad_axis, dim=-1)
        grad_struct_r = torch.cross(
            grad_axis,
            rhat.unsqueeze(1).expand_as(struct_r), dim=-1)
        grad_Q_edge = (
            grad_struct_r.unsqueeze(-1)
            * rhat[:, None, None, :])
        grad_Q = fields["structs"].new_zeros(fields["structs"].shape)
        grad_Q = grad_Q.index_add_(0, pair_i, grad_Q_edge)
        grad_u = grad_u + torch.matmul(
            structs_i.transpose(-1, -2),
            grad_struct_r.unsqueeze(-1)).squeeze(-1)
        grad_rhat = grad_rhat + grad_u.sum(dim=1)
        return grad_weight, grad_rhat, grad_Q

    def _qoh_density_vjp(self, cache, fields, grad_Q, grad_O, grad_H,
                         grad_weight, grad_rhat):
        pair_i = cache["pair_i"]
        weights = cache["weights"]
        rhat = cache["rhat"]
        chiC = self.chi_compress

        stf_rr = cache["stf_rr_edge"].reshape(-1, 3, 3)
        grad_Q_edge = grad_Q[pair_i]
        grad_weight = grad_weight + (
            grad_Q_edge * stf_rr[:, None]).sum(dim=(-1, -2))
        grad_rhat = grad_rhat + _stf_outer_vjp(
            (grad_Q_edge * weights[:, :, None, None]).sum(dim=1),
            rhat)

        unpack3 = self._chiral_m3_unpack.to(
            dtype=grad_O.dtype, device=grad_O.device)
        unpack4 = self._chiral_m4_unpack.to(
            dtype=grad_H.dtype, device=grad_H.device)
        grad_p3 = torch.matmul(
            grad_O.reshape(-1, 27), unpack3.transpose(0, 1)).reshape(
                grad_O.shape[0], chiC, -1)
        grad_p4 = torch.matmul(
            grad_H.reshape(-1, 81), unpack4.transpose(0, 1)).reshape(
                grad_H.shape[0], chiC, -1)
        g3 = grad_p3[pair_i]
        g4 = grad_p4[pair_i]
        m3 = fields["bulk"]["m3"]
        m4 = fields["bulk"]["m4"]
        grad_weight_qoh = (
            (g3 * m3[:, None, :]).sum(dim=-1)
            + (g4 * m4[:, None, :]).sum(dim=-1))
        grad_weight = torch.cat((
            grad_weight[:, :chiC] + grad_weight_qoh,
            grad_weight[:, chiC:],
        ), dim=-1)
        grad_m3 = (g3 * weights[:, :chiC, None]).sum(dim=1)
        grad_m4 = (g4 * weights[:, :chiC, None]).sum(dim=1)
        grad_rhat = grad_rhat + self._monomial34_vjp(
            rhat, m3, grad_m3, grad_m4)
        return grad_weight, grad_rhat

    def _monomial34_vjp(self, rhat, m3, grad_m3, grad_m4):
        E = rhat.shape[0]
        from_m3 = self._chiral_m4_from_m3
        coord4 = self._chiral_m4_coord
        grad_m3 = grad_m3.scatter_add(
            1, from_m3.unsqueeze(0).expand(E, -1),
            grad_m4 * rhat[:, coord4])
        grad_rhat = torch.zeros_like(rhat)
        grad_rhat = grad_rhat.scatter_add(
            1, coord4.unsqueeze(0).expand(E, -1),
            grad_m4 * m3[:, from_m3])

        index3 = self._chiral_m3_index
        for slot in range(3):
            other0 = index3[:, (slot + 1) % 3]
            other1 = index3[:, (slot + 2) % 3]
            contribution = (
                grad_m3 * rhat[:, other0] * rhat[:, other1])
            grad_rhat = grad_rhat.scatter_add(
                1, index3[:, slot].unsqueeze(0).expand(E, -1),
                contribution)
        return grad_rhat

    def _bulk_chirality_loop(self, weights: torch.Tensor, rhat: torch.Tensor,
                             pair_i: torch.Tensor, N: int):
        out = weights.new_zeros((N, weights.shape[1]))
        for center in range(N):
            edge_ids = torch.nonzero(pair_i == center, as_tuple=False).flatten()
            if edge_ids.numel() < 3:
                continue
            triples = torch.combinations(
                torch.arange(edge_ids.numel(), device=pair_i.device), r=3)
            ids = edge_ids[triples]
            r0, r1, r2 = rhat[ids[:, 0]], rhat[ids[:, 1]], rhat[ids[:, 2]]
            a = (r0 * r1).sum(dim=-1)
            b = (r1 * r2).sum(dim=-1)
            c = (r2 * r0).sum(dim=-1)
            volume = (r0 * torch.cross(r1, r2, dim=-1)).sum(dim=-1)
            angle_gate = (a - b) * (b - c) * (c - a)
            radial = (
                weights[ids[:, 0]]
                * weights[ids[:, 1]]
                * weights[ids[:, 2]])
            out[center] = (radial * (volume * angle_gate)[:, None]).sum(dim=0)
        return out

def _index_sum(like: torch.Tensor, index: torch.Tensor,
               values: torch.Tensor, N: int) -> torch.Tensor:
    if values.device.type == "mps":
        zero = values.new_zeros(values.shape[-1])
        rows = []
        for atom_index in range(N):
            mask = index == atom_index
            rows.append(values[mask].sum(dim=0) if mask.any() else zero)
        return torch.stack(rows, dim=0)
    out = like.new_zeros((N, values.shape[-1]))
    return out.index_add_(0, index, values)

def _scatter_spin_add(base: torch.Tensor, index: torch.Tensor,
                      values: torch.Tensor) -> torch.Tensor:
    return base.scatter_add(0, index.unsqueeze(-1).expand_as(values), values)

def _stf_outer_vjp(grad_stf: torch.Tensor,
                   vector: torch.Tensor) -> torch.Tensor:
    trace = grad_stf.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    sym_action = ((grad_stf + grad_stf.transpose(-1, -2))
                  @ vector.unsqueeze(-1)).squeeze(-1)
    return sym_action - (2.0 / 3.0) * trace.unsqueeze(-1) * vector

def _stf_outer_pair_vjp(grad_stf: torch.Tensor,
                        other: torch.Tensor) -> torch.Tensor:
    trace = grad_stf.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    sym_action = 0.5 * ((grad_stf + grad_stf.transpose(-1, -2))
                        @ other.unsqueeze(-1)).squeeze(-1)
    return sym_action - (1.0 / 3.0) * trace.unsqueeze(-1) * other

def _type_pair_radial_mixing(rb: torch.Tensor, pair_i: torch.Tensor,
                             pair_j: torch.Tensor, atom_types: torch.Tensor,
                             coeff: torch.Tensor) -> torch.Tensor:
    ti = atom_types[pair_i]
    tj = atom_types[pair_j]
    c_pair = coeff[ti, tj]
    return torch.bmm(c_pair, rb.unsqueeze(-1)).squeeze(-1)

def _stf_outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    outer = a.unsqueeze(-1) * b.unsqueeze(-2)
    sym = 0.5 * (outer + outer.transpose(-1, -2))
    trace = sym.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
    eye = torch.eye(3, dtype=a.dtype, device=a.device)
    return sym - trace[..., None, None] * eye
