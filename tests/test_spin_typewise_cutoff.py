# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.

import copy

import numpy as np
import pytest
import torch

from torchnep.data import parse_nep_in
from torchnep.magnetic_polynomial_v3 import MagneticPolynomialDescriptorV3
from torchnep.model import NEPModel, slim_model
from torchnep.nep import NEPCalculator
from torchnep.train import compute_q_scaler, preprocess_structures


DTYPE = torch.float64


def _write_input(path, cutoffs="4 6"):
    path.write_text(
        "type 2 Fe Ge\n"
        "cutoff 6 5\n"
        "n_max 2 2\n"
        "basis_size 4 4\n"
        "l_max 2 0 0\n"
        "neuron 6\n"
        "spin_mode 3\n"
        "spin_order 2\n"
        "spin_compress 2\n"
        "spin_l_max 2\n"
        "spin_soc 1\n"
        f"spin_cutoff {cutoffs}\n"
    )


def test_spin_typewise_cutoff_parser_matches_structural_pair_rule(tmp_path):
    path = tmp_path / "nep.in"
    _write_input(path)
    config = parse_nep_in(path)
    model = NEPModel(config).double()

    assert config["spin_cutoff_by_type"] == [4.0, 6.0]
    assert config["spin_cutoff"] == [6.0, 6.0]
    assert model.spin_cutoff_by_type == [4.0, 6.0]
    torch.testing.assert_close(
        model.spin_cutoff_pair,
        torch.tensor([[4.0, 5.0], [5.0, 6.0]], dtype=DTYPE),
    )

    legacy = tmp_path / "legacy.in"
    _write_input(legacy, "5.5")
    legacy_config = parse_nep_in(legacy)
    assert legacy_config["spin_cutoff_by_type"] == [5.5, 5.5]
    assert legacy_config["spin_cutoff"] == [5.5, 5.5]


def test_typewise_spin_cutoff_does_not_change_nonspin_cutoff_shape(tmp_path):
    path = tmp_path / "nep.in"
    path.write_text("type 2 Fe Ge\ncutoff 6 5\n")
    config = parse_nep_in(path)
    model = NEPModel(config)
    assert config["spin_cutoff"] == [6.0, 5.0]
    assert config["spin_cutoff_by_type"] == [6.0, 6.0]
    assert model.spin_cutoff == [6.0, 5.0]


def test_spin_typewise_cutoff_rejects_partial_type_list(tmp_path):
    path = tmp_path / "bad.in"
    path.write_text(
        "type 3 Fe Ge Ni\n"
        "spin_mode 3\n"
        "spin_cutoff 4 6\n"
    )
    with pytest.raises(ValueError, match=r"one value per element type \(3\)"):
        parse_nep_in(path)


def test_spin_neighbor_filter_uses_mean_pair_cutoff(tmp_path):
    path = tmp_path / "nep.in"
    _write_input(path)
    config = parse_nep_in(path)
    frame = {
        "natoms": 3,
        "species": ["Fe", "Ge", "Ge"],
        "positions": np.array(
            [[0.0, 0.0, 0.0], [5.2, 0.0, 0.0], [10.7, 0.0, 0.0]]
        ),
        "cell": np.diag([30.0, 30.0, 30.0]),
        "spins": np.ones((3, 3)),
    }
    structure = preprocess_structures(
        [frame], config, dtype=np.float64, n_workers=1
    )[0]
    edge_types = {
        (
            int(structure["atom_types"][center]),
            int(structure["atom_types"][neighbor]),
        )
        for center, neighbor in zip(
            structure["pair_i_spin"], structure["pair_j_spin"]
        )
    }
    # Fe--Ge at 5.2 A is outside mean(4, 6)=5 A.  Ge--Ge at
    # 5.5 A remains inside 6 A, in both directed storage orientations.
    assert (0, 1) not in edge_types
    assert (1, 0) not in edge_types
    assert edge_types == {(1, 1)}


def test_spin_pair_coordinate_map_is_exact_and_differentiable(tmp_path):
    path = tmp_path / "nep.in"
    _write_input(path)
    model = NEPModel(parse_nep_in(path)).double()
    pair_i = torch.tensor([0], dtype=torch.long)
    pair_j = torch.tensor([1], dtype=torch.long)
    atom_types = torch.tensor([0, 1], dtype=torch.long)
    spins = torch.tensor(
        [[0.2, -0.3, 1.1], [0.4, 0.7, -0.2]], dtype=DTYPE
    )
    rij = torch.tensor([[2.5, 0.4, -0.2]], dtype=DTYPE)
    mapped = model._spin_descriptor_coordinates(
        rij, pair_i, pair_j, atom_types
    )
    torch.testing.assert_close(mapped, rij * (6.0 / 5.0))

    reference = MagneticPolynomialDescriptorV3(
        spin_compress=model.spin_compress,
        spin_l_max=model.spin_l_max[0],
        spin_order=model.spin_order,
        spin_soc=bool(model.spin_soc),
        basis_size_radial=model.spin_basis_size[0],
        cutoff_radial=5.0,
    ).double()
    with torch.no_grad():
        reference.radial_leg_mix.copy_(model.spin_descriptor.radial_leg_mix)
    expected = reference(
        rij, pair_i, pair_j, spins, 2, atom_types, model.c_param_spin
    )
    actual = model.spin_descriptor(
        mapped, pair_i, pair_j, spins, 2, atom_types, model.c_param_spin
    )
    torch.testing.assert_close(actual, expected, rtol=2.0e-12, atol=2.0e-12)

    direction = torch.tensor([[0.3, -0.2, 0.1]], dtype=DTYPE)
    head = torch.linspace(
        -0.2, 0.3, model.spin_dim, dtype=DTYPE
    )

    def energy(coordinates):
        effective = model._spin_descriptor_coordinates(
            coordinates, pair_i, pair_j, atom_types
        )
        descriptor = model.spin_descriptor(
            effective, pair_i, pair_j, spins, 2,
            atom_types, model.c_param_spin,
        )
        return (descriptor * head).sum()

    variable = rij.clone().requires_grad_(True)
    analytic = (torch.autograd.grad(energy(variable), variable)[0]
                * direction).sum()
    step = 2.0e-6
    finite = (
        energy(rij + step * direction) - energy(rij - step * direction)
    ) / (2.0 * step)
    torch.testing.assert_close(analytic, finite, rtol=2.0e-6, atol=2.0e-7)


def test_spin_q_scaler_uses_typewise_coordinates(tmp_path, monkeypatch):
    path = tmp_path / "nep.in"
    _write_input(path)
    model = NEPModel(parse_nep_in(path)).double()
    pair_i = torch.tensor([0], dtype=torch.long)
    pair_j = torch.tensor([1], dtype=torch.long)
    atom_types = torch.tensor([0, 1], dtype=torch.long)
    rij = torch.tensor([[2.5, 0.4, -0.2]], dtype=DTYPE)
    spins = torch.tensor(
        [[0.2, -0.3, 1.1], [0.4, 0.7, -0.2]], dtype=DTYPE
    )

    class Store:
        n = 1

        def collate(self, _):
            return {
                "fk_rad": None,
                "fk_ang": None,
                "blm": None,
                "pair_i_rad": pair_i,
                "pair_j_rad": pair_j,
                "pair_i_ang": pair_i,
                "pair_j_ang": pair_j,
                "pair_i_spin": pair_i,
                "pair_j_spin": pair_j,
                "rij_spin": rij,
                "spins": spins,
                "atom_types": atom_types,
                "N": 2,
            }

    monkeypatch.setattr(
        "torchnep.train.ops.compute_descriptors_cached",
        lambda *args, **kwargs: torch.zeros(
            2, model.struct_dim, dtype=DTYPE
        ),
    )
    q_min, q_max = compute_q_scaler(model, Store())
    mapped = model._spin_descriptor_coordinates(
        rij, pair_i, pair_j, atom_types
    )
    spin_q = model.spin_descriptor(
        mapped, pair_i, pair_j, spins, 2,
        atom_types, model.c_param_spin,
    )
    expected = torch.cat(
        (torch.zeros(2, model.struct_dim, dtype=DTYPE), spin_q), dim=1
    )
    torch.testing.assert_close(q_min, expected.min(0).values)
    torch.testing.assert_close(q_max, expected.max(0).values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_spin_typewise_cutoff_cuda_forward_vjp_hvp(tmp_path, monkeypatch):
    """The pair-coordinate pullback composes with the production CUDA bank."""
    path = tmp_path / "nep.in"
    _write_input(path)
    source = NEPModel(parse_nep_in(path)).cuda().double()
    reference = copy.deepcopy(source)
    candidate = copy.deepcopy(source)
    pair_i = torch.tensor([0, 0, 1, 1], dtype=torch.long, device="cuda")
    pair_j = torch.tensor([1, 1, 0, 0], dtype=torch.long, device="cuda")
    atom_types = torch.tensor([0, 1], dtype=torch.long, device="cuda")
    raw_rij = torch.tensor(
        [[1.2, 0.3, -0.1], [2.1, -0.2, 0.4],
         [-1.2, -0.3, 0.1], [-2.1, 0.2, -0.4]],
        dtype=DTYPE, device="cuda",
    )
    raw_spins = torch.tensor(
        [[0.2, -0.3, 1.1], [0.4, 0.7, -0.2]],
        dtype=DTYPE, device="cuda",
    )
    ref_rij = raw_rij.clone().requires_grad_(True)
    ref_spins = raw_spins.clone().requires_grad_(True)
    cuda_rij = raw_rij.clone().requires_grad_(True)
    cuda_spins = raw_spins.clone().requires_grad_(True)

    def evaluate(model, rij, spins, *, cuda_bank):
        monkeypatch.setenv("TORCHNEP_OC_CUDA_PREBUILT", "1")
        monkeypatch.setenv("TORCHNEP_OC_CUDA_FULL", "0")
        monkeypatch.setenv("TORCHNEP_OC_CUDA_TRAIN", "0")
        monkeypatch.setenv(
            "TORCHNEP_OC_CUDA_BANK", "1" if cuda_bank else "0"
        )
        monkeypatch.setenv("TORCHNEP_OC_CUDA_CENTER", "0")
        mapped = model._spin_descriptor_coordinates(
            rij, pair_i, pair_j, atom_types
        )
        return model.spin_descriptor(
            mapped, pair_i, pair_j, spins, 2, atom_types,
            model.c_param_spin, None,
            model.spin_dof_type_active, model.spin_env_type_active,
        )

    expected = evaluate(reference, ref_rij, ref_spins, cuda_bank=False)
    actual = evaluate(candidate, cuda_rij, cuda_spins, cuda_bank=True)
    torch.testing.assert_close(actual, expected, rtol=3.0e-11, atol=3.0e-11)

    generator = torch.Generator(device="cuda").manual_seed(20260903)
    pull = torch.randn(
        expected.shape, dtype=DTYPE, device="cuda", generator=generator
    )
    ref_inputs = (ref_rij, ref_spins, reference.c_param_spin)
    cuda_inputs = (cuda_rij, cuda_spins, candidate.c_param_spin)
    first_ref = torch.autograd.grad(
        (expected * pull).sum(), ref_inputs, create_graph=True
    )
    first_cuda = torch.autograd.grad(
        (actual * pull).sum(), cuda_inputs, create_graph=True
    )
    for value, oracle in zip(first_cuda, first_ref):
        torch.testing.assert_close(value, oracle, rtol=3.0e-9, atol=3.0e-9)

    tangents = tuple(
        torch.randn(
            value.shape, dtype=DTYPE, device="cuda", generator=generator
        )
        for value in first_ref
    )
    scalar_ref = sum(
        (value * tangent).sum()
        for value, tangent in zip(first_ref, tangents)
    )
    scalar_cuda = sum(
        (value * tangent).sum()
        for value, tangent in zip(first_cuda, tangents)
    )
    second_ref = torch.autograd.grad(scalar_ref, ref_inputs)
    second_cuda = torch.autograd.grad(scalar_cuda, cuda_inputs)
    for value, oracle in zip(second_cuda, second_ref):
        torch.testing.assert_close(value, oracle, rtol=3.0e-8, atol=5.0e-9)


def test_spin_typewise_cutoff_round_trip(tmp_path):
    input_path = tmp_path / "nep.in"
    _write_input(input_path)
    config = parse_nep_in(input_path)
    source = NEPModel(config).double()
    model_path = tmp_path / "nep4_spin3.txt"
    source.save_nep_txt(
        model_path, max_NN_radial=32, max_NN_angular=32
    )
    assert "spin_cutoff 4.0 6.0\n" in model_path.read_text()

    restored = NEPModel(config).double()
    restored.load_weights_from_nep_txt(model_path)
    calculator = NEPCalculator(model_path, dtype=DTYPE)
    assert restored.spin_cutoff_by_type == [4.0, 6.0]
    assert calculator.spin_cutoff_by_type == [4.0, 6.0]
    torch.testing.assert_close(
        calculator.spin_cutoff_pair,
        torch.tensor([[4.0, 5.0], [5.0, 6.0]], dtype=DTYPE),
    )

    mismatch_path = tmp_path / "mismatch.in"
    _write_input(mismatch_path, "5 6")
    mismatch = NEPModel(parse_nep_in(mismatch_path)).double()
    assert mismatch.architecture_signature() != source.architecture_signature()
    with pytest.raises(ValueError, match="spin_cutoff does not match"):
        mismatch.load_weights_from_nep_txt(model_path)

    slimmed = slim_model(source, ["Ge"])
    assert slimmed.spin_cutoff_by_type == [6.0]
    torch.testing.assert_close(
        slimmed.spin_cutoff_pair,
        torch.tensor([[6.0]], dtype=DTYPE),
    )
