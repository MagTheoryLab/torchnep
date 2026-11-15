"""GPUMD c516 spin_version 1 tensor descriptor, with PyTorch derivatives.

Layout and parameter order follow force/nep_spin/spin_layout.cuh at fd358023.
Geometry-only degree-zero paths are part of this descriptor, even at zero spin.
"""

import torch

from . import ops
from .spin_tables import COUPLINGS, HARMONICS
from .cuda_harmonics import fused_harmonics, harmonic_basis_and_derivatives


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
        self.register_buffer("local_spin", torch.tensor(self.local_primitives,dtype=torch.long),persistent=False)
        self.register_buffer("density_index", self.geometry*spin_components+self.spin,persistent=False)
        self.register_buffer("center_density_index", self.geometry*spin_components,persistent=False)
        self.radial_mix = torch.nn.Parameter(torch.empty(n_max_angular+1, spin_compress))
        self.weights = torch.nn.Parameter(torch.empty(path, spin_compress))
        self.center_mix = torch.nn.Parameter(torch.empty(len(self.primitives)-1, spin_compress))
        for parameter in self.parameters():
            torch.nn.init.uniform_(parameter, -1., 1.)

    def _harmonics(self, xyz):
        fused = fused_harmonics(xyz, self.spin_l_max)
        if fused is not None:
            return fused
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
        if n_atoms != spins.shape[0]:
            raise ValueError('n_atoms must match the number of spin rows')
        if spin_type_active is not None:
            spins = spins * spin_type_active[atom_types].to(spins).unsqueeze(-1)
        phi = self._primitives(spins)
        distance = torch.linalg.vector_norm(rij, dim=-1).clamp_min(1e-12)
        geometry = self._harmonics(rij / distance.unsqueeze(-1))
        basis = ops.chebyshev_basis(distance, self.cutoff_radial, self.basis_size_radial)
        return self._contract(basis, geometry, phi, pair_i, pair_j, atom_types, c_angular)[0]

    def _contract(self, basis, geometry, phi, pair_i, pair_j, atom_types, c_angular):
        n_atoms = phi.shape[0]
        angular = ops._type_contraction_mulsum(basis, pair_i, pair_j, atom_types, c_angular)
        radial = angular @ self.radial_mix
        edge = torch.einsum('ec,eg,ep->ecgp', radial, geometry, phi[pair_j])
        density = phi.new_zeros(n_atoms, self.spin_compress,
                                 geometry.shape[-1], self.spin_components).index_add(0, pair_i, edge)
        neighbor = density[:, :, self.geometry, self.spin]
        center = density[:, :, self.geometry, 0] * phi[:, self.spin].unsqueeze(1)
        values = torch.where(self.neighbor.bool()[None, None, :], neighbor, center)
        values = values * self.coupling * self.weights[self.path].T.unsqueeze(0)
        tensors = phi.new_zeros(n_atoms, self.spin_compress, self.components).index_add(2, self.output, values)
        local = phi[:, self.center_spin].unsqueeze(1) * self.center_mix[self.center_weight].T.unsqueeze(0)
        tensors = tensors.index_add(2, self.center_output, local)
        columns = [phi[:, index:index+1] for index in self.local_primitives]
        for _, offset, width, _, scalar_column in self.sectors:
            sector = tensors[:, :, offset:offset+width]
            if scalar_column is not None:
                columns.append(sector[:, :, 0])
            columns.append(sector.square().sum(-1) / width)
        return torch.cat(columns, -1), (density, tensors, radial)

    @torch.no_grad()
    def prepare_basis(self, rij, spins, scale, active):
        """Cache input-only primitives/Jacobians; learned projections stay live."""
        effective = spins * active[:,None]
        harmonics, dh = harmonic_basis_and_derivatives(effective,self.spin_l_max)
        s2 = effective.square().sum(-1,keepdim=True)
        values, derivatives = [], []
        for degree, rank, _ in self.primitives:
            power = (degree-rank)//2
            h = harmonics[:,rank*rank:(rank+1)**2]
            derivative = dh[:,rank*rank:(rank+1)**2,:] * (s2**power).unsqueeze(-1)
            if power:
                derivative = derivative + h.unsqueeze(-1) * (2*power*s2**(power-1)*effective).unsqueeze(1)
            values.append(s2**power*h)
            derivatives.append(derivative*active[:,None,None])
        phi = torch.cat(values,-1)
        dphi = torch.cat(derivatives,1)
        scaled = rij*scale[:,None]
        norm = scaled.norm(dim=-1)
        distance = norm.clamp_min(1e-12)
        hat = scaled/distance[:,None]
        geometry, dy = harmonic_basis_and_derivatives(hat,self.spin_l_max)
        live_norm = (norm>=1e-12).to(rij.dtype)
        dy = (dy - (dy*hat[:,None,:]).sum(-1,keepdim=True)*hat[:,None,:]*live_norm[:,None,None])
        dy = dy*(scale/distance)[:,None,None]
        fk, fkp = ops.chebyshev_basis_and_deriv(distance,self.cutoff_radial,self.basis_size_radial)
        fkp = fkp*(scale*live_norm)[:,None]
        return dict(phi=phi,dphi=dphi,geometry=geometry,dgeometry=dy,
                    fk=fk,fkp=fkp,hat=hat)

    def pullback(self, fp, basis, intermediates, pair_i, pair_j, atom_types, c_angular):
        """Exact chain rule to pair coordinates and spins, differentiable in weights."""
        density, tensors, radial = intermediates
        phi, geometry = basis['phi'], basis['geometry']
        n, c = phi.shape[0], self.spin_compress
        gt = []
        for _, offset, width, column, scalar in self.sectors:
            grad = tensors[:,:,offset:offset+width] * (2./width) * fp[:,column:column+c,None]
            if scalar is not None:
                direct = torch.cat((fp[:,scalar:scalar+c,None],grad.new_zeros(n,c,width-1)),-1)
                grad = grad+direct
            gt.append(grad)
        gt = torch.cat(gt,-1)
        neighbor = self.neighbor.bool()[None,None,:]
        gv = gt[:,:,self.output]*self.coupling*self.weights[self.path].T.unsqueeze(0)
        gd = phi.new_zeros(n,c,geometry.shape[-1]*self.spin_components)
        gd = gd.index_add(2,self.density_index,gv*neighbor)
        gd = gd.index_add(2,self.center_density_index,gv*(~neighbor)*phi[:,self.spin].unsqueeze(1))
        gp = torch.zeros_like(phi).index_add(1,self.local_spin,fp[:,:len(self.local_primitives)])
        gp = gp.index_add(1,self.spin,(gv*(~neighbor)*density[:,:,self.geometry,0]).sum(1))
        gp = gp.index_add(1,self.center_spin,(gt[:,:,self.center_output]*self.center_mix[self.center_weight].T.unsqueeze(0)).sum(1))
        edge_pull = gd.reshape(n,c,geometry.shape[-1],self.spin_components)[pair_i]
        gr = torch.einsum('ecgp,eg,ep->ec',edge_pull,geometry,phi[pair_j])
        gy = torch.einsum('ecgp,ec,ep->eg',edge_pull,radial,phi[pair_j])
        gp = gp.index_add(0,pair_j,torch.einsum('ecgp,ec,eg->ep',edge_pull,radial,geometry))
        radial_prime = ops._type_contraction_mulsum(basis['fkp'],pair_i,pair_j,atom_types,c_angular) @ self.radial_mix
        dr = (gr*radial_prime).sum(-1,keepdim=True)*basis['hat']
        dr = dr + (gy.unsqueeze(-1)*basis['dgeometry']).sum(1)
        ds = (gp.unsqueeze(-1)*basis['dphi']).sum(1)
        return dr, ds
