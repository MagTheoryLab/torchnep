import torch

from example.oc_spin.benchmark_spin_type_resolution_fege import (
    _spin_pair_resolution,
    _tie_spin_pair_table,
)
from torchnep.magnetic_polynomial_v3 import MagneticPolynomialDescriptorV3


DTYPE = torch.float64


def _case():
    descriptor = MagneticPolynomialDescriptorV3(
        spin_compress=2,
        spin_l_max=1,
        spin_order=2,
        spin_soc=False,
    ).to(dtype=DTYPE)
    rij = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.2, 0.0]], dtype=DTYPE
    )
    pair_i = torch.tensor([0, 0], dtype=torch.long)
    pair_j = torch.tensor([1, 2], dtype=torch.long)
    spins = torch.tensor(
        [[1.0, 0.2, 0.0], [0.3, 0.7, 0.0], [-0.2, 0.1, 0.9]],
        dtype=DTYPE,
    )
    atom_types = torch.tensor([0, 0, 1], dtype=torch.long)
    coefficient = torch.randn(1, 2, 2, 2, 9, dtype=DTYPE)
    return descriptor, rij, pair_i, pair_j, spins, atom_types, coefficient


def _evaluate(case, coefficient):
    descriptor, rij, pair_i, pair_j, spins, atom_types, _ = case
    return descriptor(
        rij,
        pair_i,
        pair_j,
        spins,
        3,
        atom_types,
        coefficient,
    )


def test_pair_tied_adapter_matches_an_explicit_shared_table():
    case = _case()
    coefficient = case[-1]
    expected = _evaluate(case, _tie_spin_pair_table(coefficient))
    with _spin_pair_resolution(resolved=False):
        actual = _evaluate(case, coefficient)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_pair_tied_adapter_keeps_all_pair_gradients_equal():
    case = _case()
    coefficient = case[-1].requires_grad_(True)
    with _spin_pair_resolution(resolved=False):
        _evaluate(case, coefficient).square().sum().backward()

    reference = coefficient.grad[:, :1, :1]
    torch.testing.assert_close(
        coefficient.grad,
        reference.expand_as(coefficient.grad),
        rtol=0.0,
        atol=0.0,
    )
