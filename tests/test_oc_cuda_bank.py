import copy
import os

import pytest
import torch

from test_oc_cuda_full import _config, _inputs
from torchnep.model import NEPModel


def _descriptor(model, values, enabled):
    os.environ["TORCHNEP_OC_CUDA_FULL"] = "0"
    os.environ["TORCHNEP_OC_CUDA_TRAIN"] = "0"
    os.environ["TORCHNEP_OC_CUDA_BANK"] = "1" if enabled else "0"
    os.environ["TORCHNEP_OC_CUDA_CENTER"] = "0"
    rij, pair_i, pair_j, spins, atom_types = values
    return model.spin_descriptor(
        rij, pair_i, pair_j, spins, spins.shape[0], atom_types,
        model.c_param_spin, None, model.spin_dof_type_active,
        model.spin_env_type_active,
    )


def _stats(actual, expected):
    error = (actual - expected).abs()
    relative = error / expected.abs().clamp_min(1.0e-12)
    return error.max().item(), relative.max().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_o3c2_density_bank_forward_vjp_hvp(monkeypatch):
    monkeypatch.setenv("TORCHNEP_OC_CUDA_PREBUILT", "1")
    source = NEPModel(_config()).cuda().double()
    reference = copy.deepcopy(source)
    candidate = copy.deepcopy(source)
    raw = _inputs()
    ref_values = tuple(value.detach().clone() for value in raw)
    cuda_values = tuple(value.detach().clone() for value in raw)
    for values in (ref_values, cuda_values):
        values[0].requires_grad_(True)
        values[3].requires_grad_(True)

    expected = _descriptor(reference, ref_values, False)
    actual = _descriptor(candidate, cuda_values, True)
    forward = _stats(actual, expected)

    generator = torch.Generator(device="cuda").manual_seed(20260901)
    pull = torch.randn(
        expected.shape, generator=generator, dtype=expected.dtype,
        device=expected.device,
    )
    ref_inputs = (
        ref_values[0], ref_values[3], reference.c_param_spin,
        reference.spin_descriptor.radial_leg_mix,
    )
    cuda_inputs = (
        cuda_values[0], cuda_values[3], candidate.c_param_spin,
        candidate.spin_descriptor.radial_leg_mix,
    )
    first_ref = torch.autograd.grad(
        (expected * pull).sum(), ref_inputs, create_graph=True
    )
    first_cuda = torch.autograd.grad(
        (actual * pull).sum(), cuda_inputs, create_graph=True
    )
    first = [_stats(value, oracle) for value, oracle in zip(first_cuda, first_ref)]

    tangents = tuple(
        torch.randn(
            value.shape, generator=generator, dtype=value.dtype,
            device=value.device,
        )
        for value in first_ref
    )
    scalar_ref = sum(
        (value * tangent).sum() for value, tangent in zip(first_ref, tangents)
    )
    scalar_cuda = sum(
        (value * tangent).sum() for value, tangent in zip(first_cuda, tangents)
    )
    second_ref = torch.autograd.grad(scalar_ref, ref_inputs)
    second_cuda = torch.autograd.grad(scalar_cuda, cuda_inputs)
    second = [
        _stats(value, oracle) for value, oracle in zip(second_cuda, second_ref)
    ]
    consistency_values = tuple(value.detach().clone() for value in raw)
    consistency_values[0].requires_grad_(True)
    consistency_values[3].requires_grad_(True)
    tangent_rij = torch.randn_like(consistency_values[0])
    tangent_spins = torch.randn_like(consistency_values[3])

    def candidate_with_inputs(rij, spins):
        values = (
            rij, consistency_values[1], consistency_values[2], spins,
            consistency_values[4],
        )
        return _descriptor(candidate, values, True)

    value, jvp = torch.autograd.functional.jvp(
        candidate_with_inputs,
        (consistency_values[0], consistency_values[3]),
        (tangent_rij, tangent_spins),
    )
    cotangent = torch.randn_like(value)
    vjp = torch.autograd.grad(
        (candidate_with_inputs(
            consistency_values[0], consistency_values[3]
        ) * cotangent).sum(),
        (consistency_values[0], consistency_values[3]),
    )
    lhs = (jvp * cotangent).sum()
    rhs = (vjp[0] * tangent_rij).sum() + (vjp[1] * tangent_spins).sum()
    adjoint = _stats(lhs.reshape(1), rhs.reshape(1))
    print(
        f"density_bank forward={forward} vjp={first} hvp={second} "
        f"jvp_vjp={adjoint}"
    )
    torch.testing.assert_close(actual, expected, rtol=3.0e-11, atol=3.0e-11)
    for value, oracle in zip(first_cuda, first_ref):
        torch.testing.assert_close(value, oracle, rtol=2.0e-9, atol=2.0e-9)
    for value, oracle in zip(second_cuda, second_ref):
        torch.testing.assert_close(value, oracle, rtol=2.0e-8, atol=3.0e-9)
    torch.testing.assert_close(lhs, rhs, rtol=2.0e-8, atol=3.0e-9)
