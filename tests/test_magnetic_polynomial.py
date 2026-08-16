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

import ast
from pathlib import Path

import pytest
import torch

from torchnep.magnetic_polynomial import MagneticPolynomialDescriptor


DTYPE = torch.float64


def _environment(n_atoms=7, seed=20260809):
    generator = torch.Generator().manual_seed(seed)
    pair_i = []
    pair_j = []
    rij = []
    positions = torch.randn(n_atoms, 3, generator=generator, dtype=DTYPE)
    for center in range(n_atoms):
        for neighbor in range(n_atoms):
            if center == neighbor:
                continue
            pair_i.append(center)
            pair_j.append(neighbor)
            rij.append(positions[neighbor] - positions[center])
    return {
        "rij": torch.stack(rij),
        "pair_i": torch.tensor(pair_i, dtype=torch.long),
        "pair_j": torch.tensor(pair_j, dtype=torch.long),
        "spins": torch.randn(n_atoms, 3, generator=generator, dtype=DTYPE),
        "N": n_atoms,
    }


def _evaluate(descriptor, environment, **overrides):
    values = dict(environment)
    values.update(overrides)
    return descriptor(**values)


def _rotation(seed=17, reflection=False):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(3, 3, generator=generator, dtype=DTYPE)
    q, _ = torch.linalg.qr(matrix)
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    if reflection:
        q = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=DTYPE)) @ q
    return q


def test_descriptor_has_only_the_ops_internal_dependency():
    source = Path(__file__).parents[1] / "torchnep" / "magnetic_polynomial.py"
    tree = ast.parse(source.read_text())
    relative = [
        (node.module, tuple(alias.name for alias in node.names))
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level
    ]
    assert relative == [(None, ("ops",))]


def test_orders_are_literal_prefixes():
    environment = _environment()
    descriptors = [
        MagneticPolynomialDescriptor(
            spin_compress=3,
            spin_l_max=2,
            spin_order=order,
            spin_soc=True,
        ).to(dtype=DTYPE)
        for order in (1, 2, 3)
    ]
    outputs = [_evaluate(descriptor, environment) for descriptor in descriptors]
    assert descriptors[0].dim < descriptors[1].dim < descriptors[2].dim
    torch.testing.assert_close(outputs[1][..., : descriptors[0].dim], outputs[0])
    torch.testing.assert_close(outputs[2][..., : descriptors[1].dim], outputs[1])
    assert all(
        token.order <= descriptors[-1].spin_order
        for token in descriptors[-1].channel_grammar
    )


@pytest.mark.parametrize("reflection", (False, True))
def test_soc_descriptor_is_o3_and_time_reversal_invariant(reflection):
    environment = _environment(seed=7)
    descriptor = MagneticPolynomialDescriptor(
        spin_compress=3,
        spin_l_max=2,
        spin_order=3,
        spin_soc=True,
    ).to(dtype=DTYPE)
    reference = _evaluate(descriptor, environment)
    rotation = _rotation(reflection=reflection)
    determinant = torch.linalg.det(rotation)
    transformed = _evaluate(
        descriptor,
        environment,
        rij=environment["rij"] @ rotation.T,
        spins=determinant * environment["spins"] @ rotation.T,
    )
    reversed_spin = _evaluate(
        descriptor, environment, spins=-environment["spins"]
    )
    torch.testing.assert_close(transformed, reference, rtol=3.0e-11, atol=3.0e-11)
    torch.testing.assert_close(reversed_spin, reference, rtol=3.0e-11, atol=3.0e-11)


def test_non_soc_allows_independent_space_and_spin_rotations():
    environment = _environment(seed=19)
    descriptor = MagneticPolynomialDescriptor(
        spin_compress=3,
        spin_l_max=2,
        spin_order=3,
        spin_soc=False,
    ).to(dtype=DTYPE)
    reference = _evaluate(descriptor, environment)
    space = _rotation(seed=3, reflection=True)
    spin = _rotation(seed=5, reflection=False)
    transformed = _evaluate(
        descriptor,
        environment,
        rij=environment["rij"] @ space.T,
        spins=environment["spins"] @ spin.T,
    )
    torch.testing.assert_close(transformed, reference, rtol=3.0e-11, atol=3.0e-11)


def test_zero_spin_has_no_magnetic_energy_leakage():
    environment = _environment(seed=23)
    descriptor = MagneticPolynomialDescriptor(
        spin_compress=3,
        spin_l_max=2,
        spin_order=3,
        spin_soc=True,
    ).to(dtype=DTYPE)
    result = _evaluate(
        descriptor, environment, spins=torch.zeros_like(environment["spins"])
    )
    torch.testing.assert_close(result, torch.zeros_like(result))


def test_order3_contains_nonzero_triple_density_coordinates():
    environment = _environment(n_atoms=8, seed=31)
    descriptor = MagneticPolynomialDescriptor(
        spin_compress=3,
        spin_l_max=2,
        spin_order=3,
        spin_soc=True,
    ).to(dtype=DTYPE)
    result = _evaluate(descriptor, environment)
    tokens = descriptor.split_by_token(result)
    assert tokens["coupling_l111_bulk"].abs().max() > 1.0e-10
    assert tokens["coupling_l111_p_m_x"].abs().max() > 1.0e-10


def test_order3_rank_two_keeps_supported_channels_and_omits_bulk_chirality():
    environment = _environment(n_atoms=8, seed=32)
    order2 = MagneticPolynomialDescriptor(
        spin_compress=2,
        spin_l_max=2,
        spin_order=2,
        spin_soc=True,
    ).to(dtype=DTYPE)
    order3 = MagneticPolynomialDescriptor(
        spin_compress=2,
        spin_l_max=2,
        spin_order=3,
        spin_soc=True,
    ).to(dtype=DTYPE)
    result2 = _evaluate(order2, environment)
    result3 = _evaluate(order3, environment)
    torch.testing.assert_close(result3[..., : order2.dim], result2)
    tokens = order3.split_by_token(result3)
    assert order3.dim == 51
    assert "coupling_l111_bulk" not in tokens
    assert tokens["coupling_l11_dot_response"].abs().max() > 1.0e-10
    assert tokens["coupling_l111_p_m_x"].abs().max() > 1.0e-10


def test_element_conditioned_radial_mixing_changes_only_active_types():
    environment = _environment(n_atoms=6, seed=37)
    descriptor = MagneticPolynomialDescriptor(
        spin_compress=3,
        spin_l_max=2,
        spin_order=2,
        spin_soc=True,
    ).to(dtype=DTYPE)
    atom_types = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    coefficient = torch.zeros(1, 2, 2, 3, 9, dtype=DTYPE)
    coefficient[..., 0, 0] = 1.0
    coefficient[..., 1, 1] = 1.0
    coefficient[..., 2, 2] = 1.0
    reference = _evaluate(
        descriptor,
        environment,
        atom_types=atom_types,
        c_spin_2body=coefficient,
    )
    modified = coefficient.clone()
    modified[:, 0, 1] *= 1.7
    changed = _evaluate(
        descriptor,
        environment,
        atom_types=atom_types,
        c_spin_2body=modified,
    )
    center_type_zero = atom_types == 0
    center_type_one = atom_types == 1
    active_delta = (
        changed[center_type_zero] - reference[center_type_zero]
    )
    assert active_delta.abs().max() > 1.0e-10
    torch.testing.assert_close(
        changed[center_type_one], reference[center_type_one]
    )


def test_autograd_matches_finite_difference_for_geometry_and_spin():
    environment = _environment(n_atoms=5, seed=41)
    descriptor = MagneticPolynomialDescriptor(
        spin_compress=3,
        spin_l_max=2,
        spin_order=3,
        spin_soc=True,
    ).to(dtype=DTYPE)
    rij = environment["rij"].clone().requires_grad_(True)
    spins = environment["spins"].clone().requires_grad_(True)
    head = torch.linspace(-0.4, 0.6, descriptor.dim, dtype=DTYPE)

    def energy(rij_value, spin_value):
        atomic = _evaluate(
            descriptor, environment, rij=rij_value, spins=spin_value
        )
        return (atomic * head).sum()

    value = energy(rij, spins)
    grad_rij, grad_spin = torch.autograd.grad(value, (rij, spins))
    step = 2.0e-6

    rij_plus = rij.detach().clone()
    rij_minus = rij.detach().clone()
    rij_plus[2, 1] += step
    rij_minus[2, 1] -= step
    fd_rij = (
        energy(rij_plus, spins.detach()) - energy(rij_minus, spins.detach())
    ) / (2.0 * step)

    spin_plus = spins.detach().clone()
    spin_minus = spins.detach().clone()
    spin_plus[1, 2] += step
    spin_minus[1, 2] -= step
    fd_spin = (
        energy(rij.detach(), spin_plus) - energy(rij.detach(), spin_minus)
    ) / (2.0 * step)

    torch.testing.assert_close(grad_rij[2, 1], fd_rij, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(
        grad_spin[1, 2], fd_spin, rtol=2.0e-5, atol=2.0e-6
    )


def test_separate_three_body_coefficient_fails_closed():
    environment = _environment(n_atoms=4)
    descriptor = MagneticPolynomialDescriptor(spin_order=2).to(dtype=DTYPE)
    with pytest.raises(ValueError, match="one generated radial seam"):
        _evaluate(
            descriptor,
            environment,
            c_spin_3body=torch.ones(1, dtype=DTYPE),
        )
