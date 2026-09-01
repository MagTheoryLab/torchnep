import pytest
import torch

from torchnep.oc_cuda_train import (
    _moment_gate_reference,
    moment_gate,
)


def _fixture(n_atoms, neighbors, dtype=torch.float64):
    generator = torch.Generator(device="cpu").manual_seed(20260901 + n_atoms)
    pair_i = torch.arange(n_atoms).repeat_interleave(neighbors)
    offsets = torch.arange(1, neighbors + 1).repeat(n_atoms)
    pair_j = (pair_i + offsets) % n_atoms
    weights = torch.randn(pair_i.numel(), 2, generator=generator, dtype=dtype)
    spins = torch.randn(n_atoms, 3, generator=generator, dtype=dtype)
    return weights, pair_i, pair_j, spins


def test_moment_gate_cpu_fallback_matches_reference_and_double_backward():
    weights, pair_i, pair_j, spins = _fixture(5, 3)
    weights.requires_grad_(True)
    spins.requires_grad_(True)
    actual = moment_gate(weights, pair_i, pair_j, spins)
    expected = _moment_gate_reference(weights, pair_i, pair_j, spins)
    torch.testing.assert_close(actual, expected)

    pull = torch.randn_like(actual)
    first = torch.autograd.grad(
        (actual * pull).sum(), (weights, spins), create_graph=True
    )
    loss = first[0].square().mean() + first[1].square().mean()
    second = torch.autograd.grad(loss, (weights, spins))
    assert all(torch.isfinite(value).all() for value in second)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("n_atoms,neighbors", ((7, 3), (11, 5)))
def test_cuda_moment_gate_forward_vjp_hvp_parity(
    monkeypatch, n_atoms, neighbors
):
    monkeypatch.setenv("TORCHNEP_OC_CUDA_TRAIN", "1")
    values = _fixture(n_atoms, neighbors)
    weights_a = values[0].cuda().requires_grad_(True)
    weights_b = values[0].cuda().requires_grad_(True)
    pair_i = values[1].cuda()
    pair_j = values[2].cuda()
    spins_a = values[3].cuda().requires_grad_(True)
    spins_b = values[3].cuda().requires_grad_(True)
    pull_a = torch.randn(
        n_atoms, 2, dtype=torch.float64, device="cuda", requires_grad=True
    )
    pull_b = pull_a.detach().clone().requires_grad_(True)

    actual = moment_gate(weights_a, pair_i, pair_j, spins_a)
    expected = _moment_gate_reference(weights_b, pair_i, pair_j, spins_b)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)

    first_a = torch.autograd.grad(
        (actual * pull_a).sum(), (weights_a, spins_a), create_graph=True
    )
    first_b = torch.autograd.grad(
        (expected * pull_b).sum(), (weights_b, spins_b), create_graph=True
    )
    for candidate, reference in zip(first_a, first_b):
        torch.testing.assert_close(candidate, reference, rtol=3e-12, atol=3e-12)

    tangent_w = torch.randn_like(weights_a)
    tangent_s = torch.randn_like(spins_a)
    tangent_p = torch.randn_like(pull_a)
    scalar_a = (
        (first_a[0] * tangent_w).sum()
        + (first_a[1] * tangent_s).sum()
    )
    scalar_b = (
        (first_b[0] * tangent_w).sum()
        + (first_b[1] * tangent_s).sum()
    )
    second_a = torch.autograd.grad(
        scalar_a, (weights_a, spins_a, pull_a)
    )
    second_b = torch.autograd.grad(
        scalar_b, (weights_b, spins_b, pull_b)
    )
    for candidate, reference in zip(second_a, second_b):
        torch.testing.assert_close(candidate, reference, rtol=2e-11, atol=2e-11)

    # HVP symmetry is the JVP/VJP consistency gate for this scalar sector.
    lhs = sum((value * tangent).sum() for value, tangent in zip(
        second_a, (tangent_w, tangent_s, tangent_p)
    ))
    rhs = sum((value * tangent).sum() for value, tangent in zip(
        second_b, (tangent_w, tangent_s, tangent_p)
    ))
    torch.testing.assert_close(lhs, rhs, rtol=2e-11, atol=2e-11)
