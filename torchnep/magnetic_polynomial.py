"""GPUMD c516 spin_version 1 tensor descriptor, with PyTorch derivatives.

Layout and parameter order follow force/nep_spin/spin_layout.cuh at fd358023.
Geometry-only degree-zero paths are part of this descriptor, even at zero spin.
"""

import torch

from . import ops
from .spin_tables import COUPLINGS, HARMONICS


class MagneticPolynomialDescriptor(torch.nn.Module):
    """One learned radial mix, coupling-path weights and on-site primitive mix."""

    def __init__(self, spin_compress=3, spin_l_max=2, spin_order=2,
                 spin_soc=True, basis_size_radial=8, cutoff_radial=6.0,
                 n_max_angular=4):
        super().__init__()
        if not 1 <= spin_compress <= 9:
            raise ValueError("spin_compress must be in [1, 9]")
        if not 0 <= spin_l_max <= 4:
            raise ValueError("spin_l_max must be in [0, 4]")
        if not 1 <= spin_order <= 4:
            raise ValueError("spin_order must be in [1, 4]")
        if spin_soc not in (0, 1):
            raise ValueError("spin_soc must be 0 or 1")
        self.spin_compress = spin_compress
        self.spin_l_max = spin_l_max
        self.spin_order = spin_order
        self.spin_soc = bool(spin_soc)
        self.basis_size_radial = basis_size_radial
        self.cutoff_radial = cutoff_radial
        self.primitives = []
        spin_components = 0
        for degree in range(spin_order + 1):
            for rank in range(degree % 2, min(degree, spin_l_max) + 1, 2):
                self.primitives.append((degree, rank, spin_components))
                spin_components += 2 * rank + 1

        # (rank, parity, time) for SOC; (geometry rank, spin rank) otherwise.
        if self.spin_soc:
            keys = sorted({(o, (-1)**g, (-1)**d)
                           for g in range(spin_l_max + 1)
                           for d, j, _ in self.primitives
                           for o in range(abs(g-j), min(g+j, spin_l_max) + 1)})
        else:
            keys = [(g, j) for g in range(spin_l_max + 1)
                    for j in range(min(spin_l_max, spin_order) + 1)]
        self.local_primitives = [offset for d, j, offset in self.primitives if d and j == 0]
        self.dim = len(self.local_primitives)
        self.sectors = []
        components = 0
        for key in keys:
            scalar = key == ((0, 1, 1) if self.spin_soc else (0, 0))
            scalar_column = self.dim if scalar else None
            self.dim += spin_compress if scalar else 0
            width = 2*key[0]+1 if self.spin_soc else (2*key[0]+1)*(2*key[1]+1)
            self.sectors.append((key, components, width, self.dim, scalar_column))
            self.dim += spin_compress
            components += width
        if self.dim > 96:
            raise ValueError("GPUMD supports at most 96 spin descriptors")
        self.components = components
        self.spin_components = spin_components
        terms, centers = [], []
        path = 0
        if self.spin_soc:
            sectors = {key: offset for key, offset, *_ in self.sectors}
            for g in range(spin_l_max + 1):
                for degree, j, offset in self.primitives:
                    for site in range(2 if degree else 1):
                        for o in range(abs(g-j), min(g+j, spin_l_max) + 1):
                            target = sectors[(o, (-1)**g, (-1)**degree)]
                            for l1, l2, lo, a, b, m, value in COUPLINGS:
                                if (l1, l2, lo) == (g, j, o):
                                    terms.append((g*g+a, offset+b, target+m, path, site, value))
                            path += 1
            for (j, parity, time), target, *_ in self.sectors:
                for p, (degree, rank, offset) in enumerate(self.primitives[1:]):
                    if rank == j and parity == 1 and time == (-1)**degree:
                        centers.extend((offset+m, target+m, p) for m in range(2*j+1))
        else:
            for (g, j), target, *_ in self.sectors:
                for p, (degree, rank, offset) in enumerate(self.primitives):
                    if rank != j:
                        continue
                    for site in range(2 if degree else 1):
                        terms.extend((g*g+m, offset+a, target+m*(2*j+1)+a, path, site, 1.)
                                     for m in range(2*g+1) for a in range(2*j+1))
                        path += 1
                    if g == 0 and p > 0:
                        centers.extend((offset+a, target+a, p-1) for a in range(2*j+1))
        self.path_count = path
        # These immutable tables move with the model but do not enter checkpoints.
        for i, name in enumerate(("geometry", "spin", "output", "path", "neighbor")):
            self.register_buffer(name, torch.tensor([t[i] for t in terms], dtype=torch.long), persistent=False)
        self.register_buffer("coupling", torch.tensor([t[5] for t in terms], dtype=torch.float64), persistent=False)
        for i, name in enumerate(("center_spin", "center_output", "center_weight")):
            self.register_buffer(name, torch.tensor([t[i] for t in centers], dtype=torch.long), persistent=False)
        self.radial_mix = torch.nn.Parameter(torch.empty(n_max_angular+1, spin_compress))
        self.weights = torch.nn.Parameter(torch.empty(path, spin_compress))
        self.center_mix = torch.nn.Parameter(torch.empty(len(self.primitives)-1, spin_compress))
        for parameter in self.parameters():
            torch.nn.init.uniform_(parameter, -1., 1.)

    def _harmonics(self, xyz):
        columns = [[] for _ in range((self.spin_l_max+1)**2)]
        for component, x, y, z, coefficient in HARMONICS:
            if component < len(columns):
                columns[component].append(coefficient * xyz[..., 0]**x * xyz[..., 1]**y * xyz[..., 2]**z)
        return torch.stack([sum(column) for column in columns], -1)

    def _primitives(self, spins):
        harmonics = self._harmonics(spins)
        s2 = spins.square().sum(-1, keepdim=True)
        return torch.cat([s2**((degree-rank)//2) * harmonics[..., rank*rank:(rank+1)**2]
                          for degree, rank, _ in self.primitives], -1)

    def forward(self, rij, pair_i, pair_j, spins, n_atoms, atom_types,
                c_angular, spin_type_active=None):
        if spin_type_active is not None:
            spins = spins * spin_type_active[atom_types].to(spins).unsqueeze(-1)
        phi = self._primitives(spins)
        distance = torch.linalg.vector_norm(rij, dim=-1).clamp_min(1e-12)
        geometry = self._harmonics(rij / distance.unsqueeze(-1))
        basis = ops.chebyshev_basis(distance, self.cutoff_radial, self.basis_size_radial)
        angular = ops._type_contraction_bmm(basis, pair_i, pair_j, atom_types, c_angular)
        radial = angular @ self.radial_mix
        edge = torch.einsum('ec,eg,ep->ecgp', radial, geometry, phi[pair_j])
        density = spins.new_zeros(n_atoms, self.spin_compress,
                                 geometry.shape[-1], self.spin_components).index_add(0, pair_i, edge)
        neighbor = density[:, :, self.geometry, self.spin]
        center = density[:, :, self.geometry, 0] * phi[:, self.spin].unsqueeze(1)
        values = torch.where(self.neighbor.bool()[None, None, :], neighbor, center)
        values = values * self.coupling * self.weights[self.path].T.unsqueeze(0)
        tensors = spins.new_zeros(n_atoms, self.spin_compress, self.components).index_add(2, self.output, values)
        local = phi[:, self.center_spin].unsqueeze(1) * self.center_mix[self.center_weight].T.unsqueeze(0)
        tensors = tensors.index_add(2, self.center_output, local)
        columns = [phi[:, index:index+1] for index in self.local_primitives]
        for _, offset, width, _, scalar_column in self.sectors:
            sector = tensors[:, :, offset:offset+width]
            if scalar_column is not None:
                columns.append(sector[:, :, 0])
            columns.append(sector.square().sum(-1) / width)
        return torch.cat(columns, -1)
