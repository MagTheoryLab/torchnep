import json
import numpy as np
import pytest
import os
from pathlib import Path
import socket
import subprocess
import sys
import torch

from torchnep.data import build_neighbor_list_np, parse_nep_in, read_xyz
from torchnep.model import NEPModel
from torchnep.nep import NEPCalculator
from torchnep.train import (
    GPUDataStore,
    compute_max_neighbors,
    preprocess_structures,
    train_nep,
)
from torchnep.spin import (
    SpinNEPLiteChiralDescriptor,
    SpinNEPLiteDescriptor,
)

DATA_DIR = Path(__file__).resolve().parent / "data"


def _config(chiral=0):
    return {
        "num_types": 2,
        "type_names": ["Fe", "Ge"],
        "cutoff_radial": 4.5,
        "cutoff_angular": 4.0,
        "n_max_radial": 2,
        "n_max_angular": 2,
        "basis_size_radial": 2,
        "basis_size_angular": 2,
        "l_max": [2, 0, 0],
        "neuron": 6,
        "spin_mode": 1,
        "spin_chiral": chiral,
        "spin_compress": 2,
        "spin_n_max": [1, 0],
        "spin_basis_size": [2, 2],
        "spin_l_max": [2, 0, 0],
        "spin_cutoff": [4.5, 4.0],
        "spin_dof_type": ["Fe"],
        "spin_env_type": ["Fe", "Ge"],
    }


def _geometry(dtype=torch.float64):
    positions = np.array([
        [1.0, 1.0, 1.0],
        [2.2, 1.1, 1.0],
        [1.3, 2.0, 1.4],
        [0.8, 1.5, 2.1],
    ])
    cell = np.eye(3) * 8.0
    pi, pj, rij = build_neighbor_list_np(positions, cell, 4.5)
    return (
        torch.tensor(rij, dtype=dtype),
        torch.tensor(pi, dtype=torch.long),
        torch.tensor(pj, dtype=torch.long),
        positions,
        cell,
    )


def test_nep_in_spin_protocol_selects_lite_and_rejects_descriptor(tmp_path):
    path = tmp_path / "nep.in"
    path.write_text(
        "type 2 Fe Ge\n"
        "spin_mode 1\n"
        "spin_chiral 1\n"
        "spin_compress 2\n"
        "spin_basis_size 2 2\n"
        "spin_l_max 2 0 0\n"
        "spin_cutoff 4.5 4.0\n"
        "spin_dof_type Fe\n"
        "spin_env_type Fe Ge\n"
        "lambda_m 0.7\n"
        "lambda_tau 0.2\n"
        "eval_batch 3\n"
        "stage2 1\n"
        "stage2_lambda_m 0.4\n"
        "stage2_lambda_tau 0.1\n"
    )
    config = parse_nep_in(path)
    assert config["spin_mode"] == 1
    assert config["spin_chiral"] == 1
    assert config["lambda_m"] == pytest.approx(0.7)
    assert config["lambda_tau"] == pytest.approx(0.2)
    assert config["eval_batch_size"] == 3
    assert config["stage2_pref_m"] == pytest.approx(0.4)
    assert config["stage2_pref_tau"] == pytest.approx(0.1)
    assert isinstance(
        NEPModel(config).spin_descriptor, SpinNEPLiteChiralDescriptor)

    path.write_text("type 1 Fe\nspin_mode 1\nspin_descriptor spin_nep_lite\n")
    with pytest.raises(ValueError, match="spin_descriptor"):
        parse_nep_in(path)


def test_lite_and_chiral_descriptors_are_rotation_invariant():
    rij, pi, pj, _, _ = _geometry()
    spins = torch.tensor([
        [0.8, 0.1, 0.2],
        [0.2, 0.9, -0.1],
        [-0.3, 0.4, 0.7],
        [0.5, -0.2, 0.6],
    ], dtype=torch.float64)
    axis = torch.tensor([0.3, -0.4, 0.8], dtype=torch.float64)
    axis = axis / axis.norm()
    angle = 0.71
    cross = torch.tensor([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ], dtype=torch.float64)
    rotation = (
        torch.eye(3, dtype=torch.float64) * torch.cos(torch.tensor(angle))
        + (1.0 - torch.cos(torch.tensor(angle))) * axis[:, None] * axis[None, :]
        + torch.sin(torch.tensor(angle)) * cross
    )
    coeff = torch.eye(3, dtype=torch.float64).reshape(1, 1, 1, 3, 3)
    types = torch.zeros(4, dtype=torch.long)
    for cls in (SpinNEPLiteDescriptor, SpinNEPLiteChiralDescriptor):
        descriptor = cls(
            spin_compress=3, spin_l_max=3,
            basis_size_radial=2, cutoff_radial=4.5).double()
        q = descriptor(rij, pi, pj, spins, 4, types, coeff)
        q_rot = descriptor(
            rij @ rotation.T, pi, pj, spins @ rotation.T,
            4, types, coeff)
        torch.testing.assert_close(q, q_rot, atol=2e-9, rtol=2e-9)


def test_spin_model_force_and_mforce_support_double_backward():
    model = NEPModel(_config(chiral=1)).double()
    model.train()
    rij, pi, pj, _, _ = _geometry()
    atom_types = torch.tensor([0, 1, 0, 1])
    spins = torch.tensor([
        [0.8, 0.1, 0.2],
        [0.2, 0.9, -0.1],
        [-0.3, 0.4, 0.7],
        [0.5, -0.2, 0.6],
    ], dtype=torch.float64)
    result = model.compute_properties(
        rij, rij, pi, pj, pi, pj, atom_types, 4,
        torch.zeros(4, dtype=torch.long), 1,
        need_forces=True, need_virial=True,
        spins=spins, need_mforce=True,
    )
    loss = result["forces"].square().mean() + result["mforces"].square().mean()
    loss.backward()
    assert model.c_param_spin.grad is not None
    assert torch.isfinite(model.c_param_spin.grad).all()

    model.zero_grad(set_to_none=True)
    model.eval()
    result = model.compute_properties(
        rij, rij, pi, pj, pi, pj, atom_types, 4,
        torch.zeros(4, dtype=torch.long), 1,
        need_forces=True, spins=spins, need_mforce=True,
        create_graph=True,
    )
    (result["forces"].square().mean()
     + result["mforces"].square().mean()).backward()
    assert model.c_param_spin.grad is not None
    assert torch.isfinite(model.c_param_spin.grad).all()


def _cached_spin_batch(chiral, spin_compress=2, spin_l_max=2):
    config = dict(_config(chiral=chiral))
    config["spin_compress"] = spin_compress
    config["spin_n_max"] = [spin_compress - 1, 0]
    config["spin_basis_size"] = [spin_compress - 1, spin_compress - 1]
    config["spin_l_max"] = [spin_l_max, 0, 0]
    config["spin_cutoff"] = [5.2, 4.0]
    frame = {
        "natoms": 4,
        "species": ["Fe", "Ge", "Fe", "Ge"],
        "cell": np.eye(3) * 12.0,
        "positions": np.array([
            [1.0, 1.0, 1.0],
            [2.2, 1.1, 1.0],
            [1.3, 2.0, 1.4],
            [5.9, 1.0, 1.0],
        ], dtype=np.float64),
        "energy": 0.0,
        "spins": np.array([
            [0.8, 0.1, 0.2],
            [0.2, 0.9, -0.1],
            [-0.3, 0.4, 0.7],
            [0.5, -0.2, 0.6],
        ], dtype=np.float64),
    }
    structures = preprocess_structures(
        [frame], config, dtype=np.float64, n_workers=1)
    store = GPUDataStore(
        structures, torch.device("cpu"), torch.float64, config=config)
    batch = store.collate([0])
    batch["need_mforce"] = True
    batch["need_tau"] = True
    return config, frame, batch


def test_export_neighbor_capacity_covers_spin_cutoff_extension():
    config = dict(_config(chiral=1))
    config["cutoff_radial"] = 1.5
    config["cutoff_angular"] = 1.5
    config["spin_cutoff"] = [5.2, 1.5]
    frame = {
        "natoms": 4,
        "species": ["Fe", "Ge", "Fe", "Ge"],
        "cell": np.eye(3) * 20.0,
        "positions": np.array([
            [1.0, 1.0, 1.0],
            [4.0, 1.0, 1.0],
            [7.0, 1.0, 1.0],
            [10.0, 1.0, 1.0],
        ], dtype=np.float64),
        "energy": 0.0,
        "spins": np.ones((4, 3), dtype=np.float64),
    }
    structure = preprocess_structures(
        [frame], config, dtype=np.float64, n_workers=1)[0]
    radial_max = int(np.bincount(
        structure["pair_i_rad"], minlength=frame["natoms"]).max())
    spin_max = int(np.bincount(
        structure["pair_i_spin"], minlength=frame["natoms"]).max())

    assert spin_max > radial_max
    max_radial, _ = compute_max_neighbors([structure])
    assert max_radial == spin_max


def _autograd_spin_properties(model, batch):
    return model.compute_properties(
        batch["rij_rad"], batch["rij_ang"],
        batch["pair_i_rad"], batch["pair_j_rad"],
        batch["pair_i_ang"], batch["pair_j_ang"],
        batch["atom_types"], batch["N"],
        batch["struct_idx"], batch["num_structures"],
        need_forces=True, need_virial=True,
        spins=batch["spins"], need_mforce=True, need_tau=True,
        rij_spin=batch["rij_spin"],
        pi_spin=batch["pair_i_spin"],
        pj_spin=batch["pair_j_spin"],
    )


def _spin_training_loss(result):
    return sum(
        result[name].square().mean()
        for name in ("Etot", "forces", "virial", "mforces", "tau")
    )


@pytest.mark.parametrize("chiral", [0, 1])
def test_cached_spin_chain_matches_autograd_outputs_and_training_grads(chiral):
    torch.manual_seed(11)
    config, _, batch = _cached_spin_batch(chiral)
    model = NEPModel(config).double().train()
    model.set_energy_baseline(torch.tensor([-1.25, 0.4]))

    assert batch["rij_spin"].shape[0] > batch["rij_rad"].shape[0]

    auto = _autograd_spin_properties(model, batch)
    _spin_training_loss(auto).backward()
    auto_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    model.zero_grad(set_to_none=True)
    cached = model.compute_properties_cached(
        batch, need_forces=True, need_virial=True)
    _spin_training_loss(cached).backward()
    cached_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    for name in ("Ei", "Etot", "forces", "virial", "mforces", "tau"):
        torch.testing.assert_close(
            cached[name], auto[name], atol=2e-9, rtol=2e-8)
    assert cached_grads.keys() == auto_grads.keys()
    for name in cached_grads:
        torch.testing.assert_close(
            cached_grads[name], auto_grads[name], atol=2e-8, rtol=2e-7,
            msg=lambda message, name=name: f"{name}: {message}")


@pytest.mark.parametrize(
    ("spin_compress", "spin_l_max"), [(1, 0), (3, 3), (4, 4)])
def test_chiral_analytical_vjp_matches_autograd_across_protocol_shapes(
        spin_compress, spin_l_max):
    torch.manual_seed(19)
    config, _, batch = _cached_spin_batch(
        1, spin_compress=spin_compress, spin_l_max=spin_l_max)
    model = NEPModel(config).double().train()

    auto = _autograd_spin_properties(model, batch)
    _spin_training_loss(auto).backward()
    auto_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    model.zero_grad(set_to_none=True)
    cached = model.compute_properties_cached(
        batch, need_forces=True, need_virial=True)
    _spin_training_loss(cached).backward()
    cached_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    for name in ("Ei", "Etot", "forces", "virial", "mforces", "tau"):
        torch.testing.assert_close(
            cached[name], auto[name], atol=3e-9, rtol=3e-8)
    assert cached_grads.keys() == auto_grads.keys()
    for name in cached_grads:
        torch.testing.assert_close(
            cached_grads[name], auto_grads[name], atol=3e-8, rtol=3e-7,
            msg=lambda message, name=name: f"{name}: {message}")


@pytest.mark.parametrize("chiral", [0, 1])
def test_cached_spin_chain_does_not_call_whole_model_autograd(
        monkeypatch, chiral):
    config, _, batch = _cached_spin_batch(chiral)
    model = NEPModel(config).double().train()

    def fail(*args, **kwargs):
        raise AssertionError("cached spin training used whole-model autograd")

    monkeypatch.setattr(model, "compute_properties", fail)
    with monkeypatch.context() as local_patch:
        local_patch.setattr(torch.autograd, "grad", fail)
        result = model.compute_properties_cached(
            batch, need_forces=True, need_virial=True)
    _spin_training_loss(result).backward()
    assert torch.isfinite(result["forces"]).all()
    assert torch.isfinite(result["mforces"]).all()


@pytest.mark.parametrize("chiral", [0, 1])
def test_cached_spin_chain_supports_virial_without_force_output(chiral):
    config, _, batch = _cached_spin_batch(chiral)
    batch["need_mforce"] = False
    batch["need_tau"] = False
    model = NEPModel(config).double().eval()

    cached = model.compute_properties_cached(
        batch, need_forces=False, need_virial=True,
        create_graph=False)
    auto = model.compute_properties(
        batch["rij_rad"], batch["rij_ang"],
        batch["pair_i_rad"], batch["pair_j_rad"],
        batch["pair_i_ang"], batch["pair_j_ang"],
        batch["atom_types"], batch["N"],
        batch["struct_idx"], batch["num_structures"],
        need_forces=False, need_virial=True,
        spins=batch["spins"], rij_spin=batch["rij_spin"],
        pi_spin=batch["pair_i_spin"], pj_spin=batch["pair_j_spin"],
        create_graph=False,
    )

    assert "forces" not in cached
    torch.testing.assert_close(
        cached["virial"], auto["virial"], atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize("chiral", [0, 1])
def test_cached_spin_force_and_mforce_match_finite_differences(chiral):
    config, frame, batch = _cached_spin_batch(chiral)
    model = NEPModel(config).double().eval()
    result = model.compute_properties_cached(
        batch, need_forces=True, need_virial=True,
        create_graph=False)
    eps = 1.0e-6

    def energy_at_positions(positions):
        displaced = dict(frame)
        displaced["positions"] = positions
        structures = preprocess_structures(
            [displaced], config, dtype=np.float64, n_workers=1)
        store = GPUDataStore(
            structures, torch.device("cpu"), torch.float64, config=config)
        local_batch = store.collate([0])
        return model.compute_properties_cached(
            local_batch, need_forces=False)["Etot"][0]

    positions_plus = frame["positions"].copy()
    positions_minus = frame["positions"].copy()
    positions_plus[1, 0] += eps
    positions_minus[1, 0] -= eps
    force_fd = -(
        energy_at_positions(positions_plus)
        - energy_at_positions(positions_minus)
    ) / (2.0 * eps)
    torch.testing.assert_close(
        result["forces"][1, 0], force_fd, atol=2e-8, rtol=2e-7)

    def energy_at_spins(spins):
        local_batch = dict(batch)
        local_batch["spins"] = spins
        return model.compute_properties_cached(
            local_batch, need_forces=False)["Etot"][0]

    spins_plus = batch["spins"].clone()
    spins_minus = batch["spins"].clone()
    spins_plus[0, 1] += eps
    spins_minus[0, 1] -= eps
    mforce_fd = -(
        energy_at_spins(spins_plus) - energy_at_spins(spins_minus)
    ) / (2.0 * eps)
    torch.testing.assert_close(
        result["mforces"][0, 1], mforce_fd, atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize("chiral", [0, 1])
def test_spin_nep_txt_roundtrip_uses_counted_protocol(tmp_path, chiral):
    torch.manual_seed(7)
    model = NEPModel(_config(chiral=chiral)).double()
    model.set_energy_baseline(torch.tensor([-2.0, 0.4]))
    model.q_scaler.copy_(torch.linspace(0.3, 1.2, model.dim))
    path = tmp_path / "nep.txt"
    model.save_nep_txt(path, 16, 16)

    text = path.read_text()
    assert text.startswith("nep4_spin1 2 Fe Ge\nspin_mode 1 ")
    assert f"spin_chiral {chiral}\n" in text
    assert "spin_descriptor" not in text

    calculator = NEPCalculator(path)
    assert calculator.spin_mode == 1
    assert calculator.spin_chiral == chiral
    torch.testing.assert_close(
        calculator.energy_baseline,
        model.energy_baseline,
        atol=1e-12, rtol=1e-12)

    _, _, _, positions, cell = _geometry()
    spins = np.array([
        [0.8, 0.1, 0.2],
        [0.2, 0.9, -0.1],
        [-0.3, 0.4, 0.7],
        [0.5, -0.2, 0.6],
    ])
    result = calculator.compute(
        ["Fe", "Ge", "Fe", "Ge"], positions, cell, spins=spins)
    assert result["energy"].shape == (4,)
    assert result["forces"].shape == (4, 3)
    assert result["mforces"].shape == (4, 3)
    assert result["tau"].shape == (4, 3)
    assert torch.isfinite(result["mforces"]).all()
    torch.testing.assert_close(
        result["tau"],
        torch.cross(torch.as_tensor(spins, dtype=torch.float64),
                    result["mforces"], dim=-1),
    )
    assert torch.count_nonzero(result["mforces"][1::2]) == 0


def test_spin_lite_chiral_matches_pre_migration_numeric_oracle():
    """Pin the standalone migration to the old Lite/chiral implementation."""
    oracle = json.loads(
        (DATA_DIR / "spin_lite_chiral_oracle.json").read_text())
    assert oracle["provenance"]["commit"] == (
        "4cf46779a66294e0ed3bd54f60ae88b3b4e708ae")

    calculator = NEPCalculator(
        DATA_DIR / "nep_spin_lite_chiral_oracle.txt",
        dtype=torch.float64,
        device="cpu",
    )
    result = calculator.compute(
        oracle["species"],
        np.asarray(oracle["positions"]),
        np.asarray(oracle["cell"]),
        spins=np.asarray(oracle["spins"]),
        compute_descriptor=True,
    )

    for name in (
            "energy", "forces", "virial", "mforces", "tau", "descriptor"):
        actual = result[name].detach().cpu().numpy()
        expected = np.asarray(oracle[name])
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_xyz_parser_reads_spin_and_mforce_aliases(tmp_path):
    path = tmp_path / "train.xyz"
    path.write_text(
        "1\n"
        'Lattice="5 0 0 0 5 0 0 0 5" energy=0 '
        "Properties=species:S:1:pos:R:3:moment:R:3:magnetic_force:R:3\n"
        "Fe 0 0 0 1 2 3 4 5 6\n"
    )
    frame = read_xyz(path)[0]
    np.testing.assert_allclose(frame["spins"], [[1, 2, 3]])
    np.testing.assert_allclose(frame["mforces"], [[4, 5, 6]])
    np.testing.assert_allclose(frame["tau"], [[-3, 6, -3]])


def _write_spin_training_case(tmp_path):
    config = tmp_path / "nep.in"
    config.write_text(
        "type 1 Fe\n"
        "cutoff 3.5 3.5\n"
        "n_max 1 1\n"
        "basis_size 1 1\n"
        "l_max 1 0 0\n"
        "neuron 4\n"
        "spin_mode 1\n"
        "spin_chiral 0\n"
        "spin_compress 1\n"
        "spin_n_max 0 0\n"
        "spin_basis_size 0 0\n"
        "spin_l_max 1 0 0\n"
        "spin_cutoff 3.5 3.5\n"
        "lambda_e 0.1\n"
        "lambda_f 0\n"
        "lambda_v 0\n"
        "lambda_m 1\n"
        "lambda_tau 0.2\n"
        "batch 1\n"
        "eval_batch 1\n"
        "epoch 2\n"
        "lr 0.001\n"
        "stage2 1\n"
        "start_stage2 2\n"
        "stage2_lambda_m 0.5\n"
        "stage2_lambda_tau 0.1\n"
    )
    xyz = tmp_path / "train.xyz"
    xyz.write_text(
        "3\n"
        'Lattice="6 0 0 0 6 0 0 0 6" energy=-3 '
        "Properties=species:S:1:pos:R:3:spin:R:3:mforce:R:3\n"
        "Fe 1 1 1 1 0 0 0.1 0 0\n"
        "Fe 2.2 1 1 0 1 0 0 0.1 0\n"
        "Fe 1.4 2 1.2 0 0 1 0 0 0.1\n"
        "3\n"
        'Lattice="6 0 0 0 6 0 0 0 6" energy=-2.7 '
        "Properties=species:S:1:pos:R:3:spin:R:3:mforce:R:3\n"
        "Fe 1 1 1 0.9 0.1 0 0.05 0.02 0\n"
        "Fe 2.1 1.2 1 0.1 0.8 0.2 0 0.06 0.01\n"
        "Fe 1.2 2.1 1.3 0.2 0 0.8 0.01 0 0.07\n"
    )
    return config, xyz


def test_single_device_training_smoke_with_mforce(tmp_path):
    config, xyz = _write_spin_training_case(tmp_path)
    output = tmp_path / "output"
    train_nep(
        str(config), str(xyz), output_dir=str(output),
        device="cpu", precision="float64", restart=False,
        prediction_interval=0, use_gpumd_qscaler=False, run_seed=1,
        valid_file=str(xyz),
    )
    assert (output / "nep_final.txt").exists()
    assert (output / "mforce_train.out").exists()
    assert (output / "tau_train.out").exists()
    assert (output / "mforce_test.out").exists()
    assert (output / "tau_test.out").exists()
    header = (output / "loss.out").read_text().splitlines()[0]
    assert "rmse_tau(eV)" in header
    assert "rmse_m_test(eV/muB)" in header
    assert "rmse_tau_test(eV)" in header
    tau = np.loadtxt(output / "tau_train.out")
    assert tau.shape == (6, 6)
    assert np.isfinite(tau[:, :3]).all()
    assert "spin_descriptor" not in (output / "nep_final.txt").read_text()


def test_two_rank_sharded_training_smoke_with_mforce(tmp_path):
    config, xyz = _write_spin_training_case(tmp_path)
    output = tmp_path / "output_ddp"
    driver = tmp_path / "run_ddp.py"
    driver.write_text(
        "import sys\n"
        "from torchnep import train_nep_sharded\n"
        "train_nep_sharded(\n"
        "    sys.argv[1], sys.argv[2], output_dir=sys.argv[3],\n"
        "    precision='float64', restart=False, prediction_interval=0,\n"
        "    use_gpumd_qscaler=False, run_seed=1,\n"
        "    valid_file=sys.argv[2],\n"
        ")\n"
    )
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = (
        repo_root + os.pathsep + env["PYTHONPATH"]
        if env.get("PYTHONPATH") else repo_root)
    env["OMP_NUM_THREADS"] = "1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        master_port = str(sock.getsockname()[1])
    completed = subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run",
            "--nnodes=1", "--node_rank=0", "--nproc_per_node=2",
            "--master_addr=127.0.0.1", f"--master_port={master_port}",
            str(driver), str(config), str(xyz), str(output),
        ],
        cwd=tmp_path, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout
    for name in (
        "nep_best.txt", "nep_final.txt", "loss.out",
        "mforce_train.out", "tau_train.out",
        "mforce_test.out", "tau_test.out",
    ):
        assert (output / name).stat().st_size > 0, completed.stdout
    header = (output / "loss.out").read_text().splitlines()[0]
    assert "rmse_m(eV/muB)" in header
    assert "rmse_tau_test(eV)" in header
    for name in (
        "mforce_train.out", "tau_train.out",
        "mforce_test.out", "tau_test.out",
    ):
        values = np.loadtxt(output / name)
        assert values.shape == (6, 6)
        assert np.isfinite(values[:, :3]).all()
