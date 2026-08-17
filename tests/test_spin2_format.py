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

from pathlib import Path

import pytest
import torch

from torchnep.data import build_neighbor_list_np, parse_nep_in
from torchnep.model import NEPModel, slim_model
from torchnep.nep import NEPCalculator
from torchnep.spin_schema import resolve_gpumd_qscaler


DTYPE = torch.float64


def _config(order=3, compress=3):
    return {
        "num_types": 2,
        "type_names": ["Fe", "Ge"],
        "cutoff_radial": 6.0,
        "cutoff_angular": 5.0,
        "n_max_radial": 4,
        "n_max_angular": 4,
        "basis_size_radial": 8,
        "basis_size_angular": 8,
        "l_max": [4, 2, 0],
        "neuron": 8,
        "spin_mode": 2,
        "spin_compress": compress,
        "spin_basis_size": [8, 0],
        "spin_l_max": [2, 0, 0],
        "spin_cutoff": [6.0, 6.0],
        "spin_order": order,
        "spin_soc": 1,
        "spin_dof_type": ["Fe", "Ge"],
        "spin_env_type": ["Fe", "Ge"],
    }


def _write_input(path: Path, order=3, compress=3):
    path.write_text(
        "type 2 Fe Ge\n"
        "cutoff 6 5\n"
        "n_max 4 4\n"
        "basis_size 8 8\n"
        "l_max 4 2 0\n"
        "neuron 8\n"
        "spin_mode 2\n"
        "spin_cutoff 6\n"
        f"spin_compress {compress}\n"
        "spin_l_max 2\n"
        f"spin_order {order}\n"
        "spin_soc 1\n"
    )


def test_spin2_parser_exposes_only_nested_truncations(tmp_path):
    path = tmp_path / "nep.in"
    _write_input(path)
    config = parse_nep_in(path)
    assert config["spin_mode"] == 2
    assert config["spin_compress"] == 3
    assert config["spin_order"] == 3
    assert config["spin_l_max"] == [2, 0, 0]
    assert config["spin_basis_size"] == [8, 0]
    assert config["spin_cutoff"] == [6.0, 6.0]


def test_direct_spin2_defaults_match_nep_input_defaults():
    config = _config(order=3, compress=2)
    for key in (
        "spin_compress", "spin_basis_size", "spin_l_max", "spin_cutoff",
        "spin_order", "spin_soc",
    ):
        config.pop(key)
    model = NEPModel(config)
    assert model.spin_compress == 2
    assert model.spin_basis_size == [8, 0]
    assert model.spin_l_max == [2, 0, 0]
    assert model.spin_cutoff == [6.0, 6.0]
    assert model.spin_order == 3
    assert model.spin_soc == 1


def test_spin2_parser_rejects_unversioned_spin_options(tmp_path):
    path = tmp_path / "nep.in"
    _write_input(path)
    with path.open("a") as stream:
        stream.write("spin_experimental_block 1\n")
    with pytest.raises(ValueError, match="unsupported O/C spin parameter"):
        parse_nep_in(path)


def test_spin_options_require_explicit_spin_mode_two(tmp_path):
    path = tmp_path / "nep.in"
    path.write_text(
        "type 1 Fe\n"
        "spin_order 3\n"
    )
    with pytest.raises(ValueError, match="require 'spin_mode 2'"):
        parse_nep_in(path)

    config = _config(order=3, compress=2)
    config["spin_mode"] = 0
    with pytest.raises(ValueError, match="require spin_mode 2"):
        NEPModel(config)


def test_retired_unpublished_spin_mode_three_is_rejected(tmp_path):
    path = tmp_path / "nep.in"
    path.write_text("type 1 Fe\nspin_mode 3\n")
    with pytest.raises(ValueError, match="spin_mode must be 0 or 2"):
        parse_nep_in(path)

    config = _config(order=3, compress=2)
    config["spin_mode"] = 3
    with pytest.raises(ValueError, match="spin_mode must be 0 or 2"):
        NEPModel(config)


def test_spin2_dimensions_and_qscaler_policy():
    assert NEPModel(_config(order=2, compress=2)).spin_dim == 37
    assert NEPModel(_config(order=2, compress=3)).spin_dim == 58
    assert NEPModel(_config(order=3, compress=2)).spin_dim == 49
    assert NEPModel(_config(order=3, compress=3)).spin_dim == 79
    assert resolve_gpumd_qscaler(2, None) is False
    assert resolve_gpumd_qscaler(2, False) is False
    with pytest.raises(ValueError, match="spin_mode 2"):
        resolve_gpumd_qscaler(2, True)


def test_spin2_qscaler_does_not_amplify_constant_channels():
    model = NEPModel(_config(order=3, compress=2)).double()
    q_min = torch.zeros(model.dim, dtype=DTYPE)
    q_max = torch.ones(model.dim, dtype=DTYPE)
    q_max[model.struct_dim] = 0.0
    model.set_q_scaler(q_min, q_max)
    assert model.q_scaler[model.struct_dim].item() == 1.0
    assert torch.isfinite(model.q_scaler).all()


def test_spin2_accepts_rank_two_order_three(tmp_path):
    path = tmp_path / "nep.in"
    _write_input(path, order=3, compress=2)
    config = parse_nep_in(path)
    assert config["spin_order"] == 3
    assert config["spin_compress"] == 2


def test_spin2_rejects_compression_above_fixed_radial_basis(tmp_path):
    path = tmp_path / "nep.in"
    _write_input(path, order=3, compress=10)
    with pytest.raises(ValueError, match=r"spin_compress must be in \[1, 9\].*B=8"):
        parse_nep_in(path)

    config = _config(order=3, compress=10)
    with pytest.raises(ValueError, match=r"spin_compress must be in \[1, 9\].*B=8"):
        NEPModel(config)


def test_o3c2_has_o2c2_prefix_and_omits_rank3_bulk():
    order2 = NEPModel(_config(order=2, compress=2)).spin_descriptor
    order3 = NEPModel(_config(order=3, compress=2)).spin_descriptor
    assert order2.dim == 37
    assert order3.dim == 49
    assert tuple(order3.schema[: len(order2.schema)]) == tuple(order2.schema)
    assert "coupling_l111_bulk" not in {
        token.name for token in order3.schema
    }
    assert "coupling_l111_p_l_x" not in {
        token.name for token in order3.schema
    }


def test_slim_model_preserves_oc_descriptor_for_kept_elements():
    torch.manual_seed(17)
    source = NEPModel(_config(order=3, compress=2)).double()
    with torch.no_grad():
        source.c_param_spin.normal_()
        source.spin_descriptor.radial_leg_mix.normal_()
    slimmed = slim_model(source, ["Fe"])

    pair_i = torch.tensor([0, 1], dtype=torch.long)
    pair_j = torch.tensor([1, 0], dtype=torch.long)
    rij = torch.tensor(
        [[1.2, 0.1, -0.2], [-1.2, -0.1, 0.2]], dtype=DTYPE)
    spins = torch.tensor(
        [[0.8, 0.1, 0.2], [-0.2, 0.7, 0.3]], dtype=DTYPE)
    atom_types = torch.zeros(2, dtype=torch.long)

    expected = source.spin_descriptor(
        rij, pair_i, pair_j, spins, 2, atom_types,
        source.c_param_spin, None,
        source.spin_dof_type_active, source.spin_env_type_active,
    )
    actual = slimmed.spin_descriptor(
        rij, pair_i, pair_j, spins, 2, atom_types,
        slimmed.c_param_spin, None,
        slimmed.spin_dof_type_active, slimmed.spin_env_type_active,
    )
    torch.testing.assert_close(actual, expected)


def test_spin2_force_and_mforce_loss_backpropagates_to_descriptor_parameters():
    model = NEPModel(_config(order=3, compress=3)).double()
    descriptor = model.spin_descriptor
    pair_i = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    pair_j = torch.tensor([1, 2, 0, 2, 0, 1], dtype=torch.long)
    rij = torch.tensor(
        [
            [1.2, 0.1, 0.2],
            [-0.2, 1.3, 0.3],
            [-1.2, -0.1, -0.2],
            [-1.4, 1.2, 0.1],
            [0.2, -1.3, -0.3],
            [1.4, -1.2, -0.1],
        ],
        dtype=DTYPE,
        requires_grad=True,
    )
    spins = torch.tensor(
        [[0.8, 0.1, 0.2], [0.2, 0.9, -0.1], [-0.3, 0.4, 0.7]],
        dtype=DTYPE,
        requires_grad=True,
    )
    atom_types = torch.tensor([0, 1, 0], dtype=torch.long)
    atomic = descriptor(
        rij,
        pair_i,
        pair_j,
        spins,
        3,
        atom_types,
        model.c_param_spin,
        None,
        model.spin_dof_type_active,
        model.spin_env_type_active,
    )
    head = torch.linspace(-0.2, 0.3, descriptor.dim, dtype=DTYPE)
    energy = (atomic * head).sum()
    grad_rij, grad_spin = torch.autograd.grad(
        energy, (rij, spins), create_graph=True
    )
    loss = grad_rij.square().mean() + grad_spin.square().mean()
    loss.backward()
    assert torch.isfinite(model.c_param_spin.grad).all()
    assert torch.isfinite(descriptor.radial_leg_mix.grad).all()


@pytest.mark.parametrize(
    "order,compress,dimension",
    ((2, 2, 37), (3, 2, 49), (3, 3, 79)),
)
def test_nep4_spin2_round_trip(order, compress, dimension, tmp_path):
    torch.manual_seed(20260810 + order)
    source = NEPModel(_config(order=order, compress=compress)).double()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)
        source.q_scaler.copy_(
            torch.linspace(0.4, 1.4, source.dim, dtype=DTYPE)
        )
        source.energy_baseline.copy_(
            torch.tensor([-0.2, 0.3], dtype=DTYPE)
        )

    first = tmp_path / f"nep_spin2_o{order}.txt"
    source.save_nep_txt(first, max_NN_radial=32, max_NN_angular=32)
    text = first.read_text()
    assert text.startswith("nep4_spin2 2 Fe Ge\n")
    assert f"spin_order {order}\n" in text
    assert f"spin_compress {compress}\n" in text
    assert f"spin_projection_size {4 * compress * compress}\n" in text

    restored = NEPModel(_config(order=order, compress=compress)).double()
    restored.load_weights_from_nep_txt(first)
    for name, expected in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected)

    calculator = NEPCalculator(first, dtype=DTYPE)
    assert calculator.spin_mode == 2
    assert calculator.spin_descriptor.dim == dimension
    assert calculator.spin_descriptor_kind == "magnetic_polynomial"


def test_spin2_load_rejects_active_type_mismatch(tmp_path):
    source_config = _config(order=3, compress=2)
    source_config["spin_dof_type"] = ["Fe"]
    source = NEPModel(source_config).double()
    path = tmp_path / "restricted.txt"
    source.save_nep_txt(path, max_NN_radial=16, max_NN_angular=16)

    target = NEPModel(_config(order=3, compress=2)).double()
    with pytest.raises(ValueError, match="spin_dof_type"):
        target.load_weights_from_nep_txt(path)


def test_spin_env_defaults_to_spin_dof_across_all_readers(tmp_path):
    input_path = tmp_path / "nep.in"
    _write_input(input_path, order=3, compress=2)
    with input_path.open("a") as stream:
        stream.write("spin_dof_type Fe\n")
    model = NEPModel(parse_nep_in(input_path)).double()
    expected = torch.tensor([True, False])
    torch.testing.assert_close(model.spin_dof_type_active, expected)
    torch.testing.assert_close(model.spin_env_type_active, expected)

    model_path = tmp_path / "dof_default_env.txt"
    model.save_nep_txt(
        model_path, max_NN_radial=16, max_NN_angular=16)
    calculator = NEPCalculator(model_path, dtype=DTYPE)
    torch.testing.assert_close(calculator.spin_dof_type_active, expected)
    torch.testing.assert_close(calculator.spin_env_type_active, expected)


def test_spin2_text_load_rejects_structural_cutoff_mismatch(tmp_path):
    source = NEPModel(_config(order=3, compress=2)).double()
    path = tmp_path / "wrong_cutoff.txt"
    source.save_nep_txt(path, max_NN_radial=16, max_NN_angular=16)
    path.write_text(path.read_text().replace(
        "cutoff 6 5 16 16", "cutoff 7 5 16 16"))
    target = NEPModel(_config(order=3, compress=2)).double()
    with pytest.raises(ValueError, match="cutoff does not match"):
        target.load_weights_from_nep_txt(path)


def test_spin2_calculator_rejects_unknown_counted_metadata(tmp_path):
    model = NEPModel(_config(order=3, compress=2)).double()
    path = tmp_path / "bad_metadata.txt"
    model.save_nep_txt(path, max_NN_radial=16, max_NN_angular=16)
    path.write_text(path.read_text().replace(
        "spin_scaler 1", "spin_scalre 1"))
    with pytest.raises(ValueError, match="unknown O/C model metadata"):
        NEPCalculator(path, dtype=DTYPE)


def test_spin2_text_load_rejects_wrong_metadata_count(tmp_path):
    source = NEPModel(_config(order=3, compress=2)).double()
    path = tmp_path / "wrong_count.txt"
    source.save_nep_txt(path, max_NN_radial=16, max_NN_angular=16)
    lines = path.read_text().splitlines()
    parts = lines[1].split()
    parts[2] = str(int(parts[2]) - 1)
    lines[1] = " ".join(parts)
    path.write_text("\n".join(lines) + "\n")
    target = NEPModel(_config(order=3, compress=2)).double()
    with pytest.raises(ValueError, match="count does not match"):
        target.load_weights_from_nep_txt(path)


def test_o3c2_model_and_text_calculator_derivative_parity(tmp_path):
    torch.manual_seed(29)
    model = NEPModel(_config(order=3, compress=2)).double().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.08)
        model.q_scaler.copy_(
            torch.linspace(0.3, 1.1, model.dim, dtype=DTYPE))
        model.energy_baseline.copy_(
            torch.tensor([-0.2, 0.1], dtype=DTYPE))
    path = tmp_path / "o3c2.txt"
    model.save_nep_txt(path, max_NN_radial=32, max_NN_angular=32)
    calculator = NEPCalculator(path, dtype=DTYPE)

    species = ["Fe", "Ge", "Fe"]
    positions = torch.tensor(
        [[1.0, 1.1, 1.2], [2.4, 1.0, 1.3], [1.3, 2.7, 1.8]],
        dtype=DTYPE,
    )
    cell = torch.eye(3, dtype=DTYPE) * 20.0
    spins = torch.tensor(
        [[0.8, 0.1, 0.2], [-0.1, 0.7, 0.3], [0.2, -0.3, 0.9]],
        dtype=DTYPE,
    )
    pair_i_np, pair_j_np, rij_np = build_neighbor_list_np(
        positions.numpy(), cell.numpy(), 6.0)
    pair_i = torch.from_numpy(pair_i_np)
    pair_j = torch.from_numpy(pair_j_np)
    rij = torch.from_numpy(rij_np)
    distance = torch.linalg.norm(rij, dim=-1)
    radial = distance < 6.0
    angular = distance < 5.0
    atom_types = torch.tensor([0, 1, 0], dtype=torch.long)
    reference = model.compute_properties(
        rij[radial], rij[angular],
        pair_i[radial], pair_j[radial],
        pair_i[angular], pair_j[angular],
        atom_types, 3, torch.zeros(3, dtype=torch.long), 1,
        need_forces=True, need_virial=True, spins=spins,
        need_mforce=True, need_tau=True,
        rij_spin=rij[radial], pi_spin=pair_i[radial],
        pj_spin=pair_j[radial], create_graph=False,
    )
    actual = calculator.compute(
        species, positions.numpy(), cell.numpy(), spins=spins.numpy())
    for model_key, calculator_key in (
        ("Ei", "energy"), ("forces", "forces"),
        ("virial", "virial"), ("mforces", "mforces"),
        ("tau", "tau"),
    ):
        torch.testing.assert_close(
            reference[model_key].detach(), actual[calculator_key],
            rtol=2.0e-9, atol=2.0e-9,
        )
    descriptor = calculator.get_descriptor(
        species, positions.numpy(), cell.numpy(), spins=spins.numpy())
    assert descriptor.shape == (3, model.dim)


def test_o3c2_periodic_full_model_matches_independent_finite_differences(
    tmp_path,
):
    """All reported responses must derive from the same scalar energy."""
    torch.manual_seed(43)
    model = NEPModel(_config(order=3, compress=2)).double().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.05)
        model.q_scaler.copy_(
            torch.linspace(0.4, 1.0, model.dim, dtype=DTYPE))
        model.energy_baseline.copy_(
            torch.tensor([-0.1, 0.2], dtype=DTYPE))
    path = tmp_path / "o3c2_fd.txt"
    model.save_nep_txt(path, max_NN_radial=64, max_NN_angular=64)
    calculator = NEPCalculator(path, dtype=DTYPE)

    species = ["Fe", "Ge", "Fe"]
    positions = torch.tensor(
        [[0.9, 1.1, 1.3], [2.5, 1.4, 1.0], [1.4, 2.9, 2.2]],
        dtype=DTYPE,
    )
    cell = torch.diag(torch.tensor([7.0, 7.2, 7.4], dtype=DTYPE))
    spins = torch.tensor(
        [[0.8, 0.1, 0.2], [-0.2, 0.7, 0.3], [0.2, -0.4, 0.9]],
        dtype=DTYPE,
    )

    def total_energy(pos, box, spin):
        return calculator.compute(
            species, pos.numpy(), box.numpy(), spins=spin.numpy()
        )["energy"].sum().item()

    analytic = calculator.compute(
        species, positions.numpy(), cell.numpy(), spins=spins.numpy())
    step = 1.0e-5

    atom = 1
    force_fd = torch.empty(3, dtype=DTYPE)
    for axis in range(3):
        plus = positions.clone()
        minus = positions.clone()
        plus[atom, axis] += step
        minus[atom, axis] -= step
        force_fd[axis] = -(
            total_energy(plus, cell, spins)
            - total_energy(minus, cell, spins)
        ) / (2.0 * step)
    assert torch.linalg.vector_norm(force_fd) > 1.0e-6
    torch.testing.assert_close(
        analytic["forces"][atom], force_fd, rtol=2.0e-4, atol=2.0e-6)

    strain_step = 1.0e-5
    energy_plus = total_energy(
        positions * (1.0 + strain_step),
        cell * (1.0 + strain_step),
        spins,
    )
    energy_minus = total_energy(
        positions * (1.0 - strain_step),
        cell * (1.0 - strain_step),
        spins,
    )
    strain_derivative_fd = (energy_plus - energy_minus) / (
        2.0 * strain_step
    )
    assert abs(strain_derivative_fd) > 1.0e-6
    total_virial = analytic["virial"].sum(dim=0).reshape(3, 3)
    torch.testing.assert_close(
        -torch.trace(total_virial),
        torch.tensor(strain_derivative_fd, dtype=DTYPE),
        rtol=2.0e-4,
        atol=2.0e-6,
    )

    spin_atom = 2
    mforce_fd = torch.empty(3, dtype=DTYPE)
    for axis in range(3):
        plus = spins.clone()
        minus = spins.clone()
        plus[spin_atom, axis] += step
        minus[spin_atom, axis] -= step
        mforce_fd[axis] = -(
            total_energy(positions, cell, plus)
            - total_energy(positions, cell, minus)
        ) / (2.0 * step)
    assert torch.linalg.vector_norm(mforce_fd) > 1.0e-6
    torch.testing.assert_close(
        analytic["mforces"][spin_atom], mforce_fd,
        rtol=2.0e-4, atol=2.0e-6,
    )
    tau_fd = torch.cross(spins[spin_atom], mforce_fd, dim=0)
    assert torch.linalg.vector_norm(tau_fd) > 1.0e-6
    torch.testing.assert_close(
        analytic["tau"][spin_atom], tau_fd,
        rtol=2.0e-4, atol=2.0e-6,
    )
