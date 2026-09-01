import copy

import pytest
import torch

from torchnep.model import NEPModel


def _config():
    return {
        "num_types": 3,
        "type_names": ["Fe", "Ge", "Ni"],
        "cutoff_radial": 6.0,
        "cutoff_angular": 5.0,
        "n_max_radial": 4,
        "n_max_angular": 4,
        "basis_size_radial": 8,
        "basis_size_angular": 8,
        "l_max": [4, 2, 0],
        "neuron": 8,
        "spin_mode": 2,
        "spin_cutoff": [6.0, 6.0],
        "spin_l_max": [2, 0, 0],
        "spin_compress": 2,
        "spin_basis_size": [8, 0],
        "spin_order": 3,
        "spin_soc": 1,
        "spin_curriculum": False,
        "spin_dof_type": ["Fe", "Ge", "Ni"],
        "spin_env_type": ["Fe", "Ge", "Ni"],
    }


def _inputs(dtype=torch.float64):
    generator = torch.Generator(device="cpu").manual_seed(20260901)
    n_atoms = 7
    neighbors = 3
    pair_i = torch.arange(n_atoms).repeat_interleave(neighbors)
    shift = torch.arange(1, neighbors + 1).repeat(n_atoms)
    pair_j = (pair_i + shift) % n_atoms
    rij = torch.randn(pair_i.numel(), 3, generator=generator, dtype=dtype)
    spins = torch.randn(n_atoms, 3, generator=generator, dtype=dtype)
    atom_types = torch.arange(n_atoms, dtype=torch.long) % 3
    return rij.cuda(), pair_i.cuda(), pair_j.cuda(), spins.cuda(), atom_types.cuda()


def _descriptor(model, values, enabled):
    import os

    os.environ["TORCHNEP_OC_CUDA_FULL"] = "1" if enabled else "0"
    rij, pair_i, pair_j, spins, atom_types = values
    return model.spin_descriptor(
        rij,
        pair_i,
        pair_j,
        spins,
        spins.shape[0],
        atom_types,
        model.c_param_spin,
        None,
        model.spin_dof_type_active,
        model.spin_env_type_active,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_full_o3c2_generated_cuda_forward_vjp_hvp(monkeypatch):
    monkeypatch.setenv("TORCHNEP_OC_CUDA_TRAIN", "0")
    source = NEPModel(_config()).cuda().double()
    reference = copy.deepcopy(source)
    candidate = copy.deepcopy(source)
    raw = _inputs()
    ref_values = tuple(value.detach().clone() for value in raw)
    cuda_values = tuple(value.detach().clone() for value in raw)
    ref_values[0].requires_grad_(True)
    ref_values[3].requires_grad_(True)
    cuda_values[0].requires_grad_(True)
    cuda_values[3].requires_grad_(True)

    expected = _descriptor(reference, ref_values, False)
    actual = _descriptor(candidate, cuda_values, True)
    torch.testing.assert_close(actual, expected, rtol=2.0e-11, atol=2.0e-11)

    pull = torch.randn_like(expected)
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
    for value, oracle in zip(first_cuda, first_ref):
        torch.testing.assert_close(value, oracle, rtol=5.0e-10, atol=5.0e-10)

    tangents = tuple(torch.randn_like(value) for value in first_ref)
    scalar_ref = sum((value * tangent).sum()
                     for value, tangent in zip(first_ref, tangents))
    scalar_cuda = sum((value * tangent).sum()
                      for value, tangent in zip(first_cuda, tangents))
    second_ref = torch.autograd.grad(scalar_ref, ref_inputs)
    second_cuda = torch.autograd.grad(scalar_cuda, cuda_inputs)
    for value, oracle in zip(second_cuda, second_ref):
        torch.testing.assert_close(value, oracle, rtol=2.0e-8, atol=2.0e-9)

    # An independent reverse-over-reverse JVP must obey the adjoint identity
    # <J v, u> = <v, J^T u> for the position and spin inputs.
    consistency_values = tuple(value.detach().clone() for value in raw)
    consistency_values[0].requires_grad_(True)
    consistency_values[3].requires_grad_(True)
    tangent_rij = torch.randn_like(consistency_values[0])
    tangent_spins = torch.randn_like(consistency_values[3])

    def candidate_with_inputs(rij, spins):
        values = (
            rij,
            consistency_values[1],
            consistency_values[2],
            spins,
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
    absolute = (lhs - rhs).abs()
    relative = absolute / rhs.abs().clamp_min(1.0e-12)
    print(
        f"jvp_vjp_abs={absolute.item():.17e} "
        f"jvp_vjp_rel={relative.item():.17e}"
    )
    torch.testing.assert_close(lhs, rhs, rtol=2.0e-8, atol=2.0e-9)
