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

import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np

from torchnep import train_nep


def _write_case(tmp_path):
    config = tmp_path / "nep.in"
    config.write_text(
        "type 1 Fe\n"
        "cutoff 3.5 3.5\n"
        "n_max 1 1\n"
        "basis_size 1 1\n"
        "l_max 1 0 0\n"
        "neuron 4\n"
        "spin_mode 3\n"
        "spin_compress 2\n"
        "spin_l_max 2\n"
        "spin_order 3\n"
        "spin_soc 1\n"
        "spin_cutoff 3.5\n"
        "lambda_e 0.1\n"
        "lambda_f 0.1\n"
        "lambda_v 0.1\n"
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
        'virial="0 0 0 0 0 0 0 0 0" '
        "Properties=species:S:1:pos:R:3:force:R:3:spin:R:3:mforce:R:3\n"
        "Fe 1 1 1 0 0 0 1 0 0 0.1 0 0\n"
        "Fe 2.2 1 1 0 0 0 0 1 0 0 0.1 0\n"
        "Fe 1.4 2 1.2 0 0 0 0 0 1 0 0 0.1\n"
        "3\n"
        'Lattice="6 0 0 0 6 0 0 0 6" energy=-2.7 '
        'virial="0.1 0 0 0 0.1 0 0 0 0.1" '
        "Properties=species:S:1:pos:R:3:force:R:3:spin:R:3:mforce:R:3\n"
        "Fe 1 1 1 0 0 0 0.9 0.1 0 0.05 0.02 0\n"
        "Fe 2.1 1.2 1 0 0 0 0.1 0.8 0.2 0 0.06 0.01\n"
        "Fe 1.2 2.1 1.3 0 0 0 0.2 0 0.8 0.01 0 0.07\n"
        "3\n"
        'Lattice="6 0 0 0 6 0 0 0 6" energy=-2.9 '
        'virial="-0.1 0 0 0 -0.1 0 0 0 -0.1" '
        "Properties=species:S:1:pos:R:3:force:R:3:spin:R:3:mforce:R:3\n"
        "Fe 1.2 1 1 0 0 0 0.8 0.2 0.1 0.07 0.01 0\n"
        "Fe 2.4 1.1 1 0 0 0 0.2 0.7 0.3 0 0.08 0.02\n"
        "Fe 1.3 2.3 1.5 0 0 0 0.1 0.2 0.8 0.02 0 0.09\n"
    )
    valid = tmp_path / "valid.xyz"
    valid.write_text(
        "3\n"
        'Lattice="6 0 0 0 6 0 0 0 6" energy=-2.85 '
        'virial="0.05 0 0 0 0.05 0 0 0 0.05" '
        "Properties=species:S:1:pos:R:3:force:R:3:spin:R:3:mforce:R:3\n"
        "Fe 1.1 1 1 0 0 0 0.7 0.2 0.1 0.04 0.01 0\n"
        "Fe 2.3 1.1 1 0 0 0 0.2 0.7 0.2 0 0.05 0.02\n"
        "Fe 1.3 2.2 1.4 0 0 0 0.1 0.1 0.9 0.02 0 0.06\n"
    )
    _mark_rotation_scan(xyz, "train-rotation")
    _mark_rotation_scan(valid, "valid-rotation", repeat_single=True)
    return config, xyz, valid


def _mark_rotation_scan(path, group, repeat_single=False):
    """Add a minimal three-point controlled rotation scan to an XYZ."""
    lines = path.read_text().splitlines()
    blocks = []
    start = 0
    while start < len(lines):
        natoms = int(lines[start])
        stop = start + natoms + 2
        blocks.append(lines[start:stop])
        start = stop
    if repeat_single and len(blocks) == 1:
        blocks = [list(blocks[0]) for _ in range(3)]
    coordinates = (-0.1, 0.0, 0.1)
    assert len(blocks) == len(coordinates)
    rendered = []
    for block, coordinate in zip(blocks, coordinates):
        header = block[1].replace(
            "Properties=",
            f"response_group={group} response_probe=rotation "
            f"response_coordinate={coordinate} Properties=",
        )
        header = header.replace(
            "mforce:R:3", "mforce:R:3:spin_tangent:R:3")
        rendered.extend([block[0], header])
        rendered.extend(line + " 0 1 0" for line in block[2:])
    path.write_text("\n".join(rendered) + "\n")


def _assert_outputs(output):
    for name in (
        "nep_final.txt",
        "nep_average.txt",
        "mforce_train.out",
        "tau_train.out",
        "mforce_test.out",
        "tau_test.out",
    ):
        assert (output / name).stat().st_size > 0
    potential = (output / "nep_final.txt").read_text()
    assert potential.startswith("nep4_spin3 1 Fe\n")
    baseline_line = next(
        line for line in potential.splitlines()
        if line.startswith("spin_baseline ")
    )
    assert float(baseline_line.split()[1]) != 0.0
    values = np.loadtxt(output / "mforce_train.out")
    assert values.shape == (9, 6)
    assert np.isfinite(values).all()
    # The single-frame validation set is padded to two DDP ranks. Padding
    # must not overwrite the real frame's reference columns with NaN.
    energy_test = np.loadtxt(output / "energy_test.out").reshape(-1, 2)
    force_test = np.loadtxt(output / "force_test.out").reshape(-1, 6)
    virial_test = np.loadtxt(output / "virial_test.out").reshape(-1, 12)
    mforce_test = np.loadtxt(output / "mforce_test.out").reshape(-1, 6)
    assert np.isfinite(energy_test[:, 1]).all()
    assert np.isfinite(force_test[:, 3:]).all()
    assert np.isfinite(virial_test[:, 6:]).all()
    assert np.isfinite(mforce_test[:, 3:]).all()
    header = (output / "loss.out").read_text().splitlines()[0]
    assert "generator_curve" in header


def test_single_device_oc_training_smoke(tmp_path):
    config, xyz, valid = _write_case(tmp_path)
    output = tmp_path / "single"
    train_nep(
        str(config),
        str(xyz),
        output_dir=str(output),
        device="cpu",
        precision="float64",
        restart=False,
        use_swa=True,
        prediction_interval=0,
        run_seed=1,
        valid_file=str(valid),
    )
    _assert_outputs(output)
    log = (output / "output.log").read_text()
    assert "O3 curriculum epoch 1: gradient_scale=0.000000" in log
    assert "O3 curriculum epoch 2: gradient_scale=1.000000" in log
    assert "forces: autograd" in log
    assert "generator curves: 3 training frames, 3 validation frames" in log
    assert "use_gpumd_qscaler: descriptor coeffs" not in log


def test_two_rank_oc_training_smoke(tmp_path):
    config, xyz, valid = _write_case(tmp_path)
    output = tmp_path / "sharded"
    driver = tmp_path / "run_ddp.py"
    driver.write_text(
        "import sys\n"
        "from torchnep import train_nep_sharded\n"
            "train_nep_sharded(\n"
            "    sys.argv[1], sys.argv[2], output_dir=sys.argv[3],\n"
            "    precision='float64', restart=False, prediction_interval=0,\n"
            "    use_swa=True,\n"
        "    run_seed=1, valid_file=sys.argv[4],\n"
        ")\n"
    )
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = (
        repo_root + os.pathsep + env["PYTHONPATH"]
        if env.get("PYTHONPATH") else repo_root
    )
    env["OMP_NUM_THREADS"] = "1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        master_port = str(sock.getsockname()[1])
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--node_rank=0",
            "--nproc_per_node=2",
            "--master_addr=127.0.0.1",
            f"--master_port={master_port}",
            str(driver),
            str(config),
            str(xyz),
            str(output),
            str(valid),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout
    _assert_outputs(output)
