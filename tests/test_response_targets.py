"""Energy/mforce response contracts; trainer smoke runs on the target GPU."""
import os
import importlib
import subprocess
import sys

import pytest
import torch

from torchnep.response_training import generator_curve_loss, response_residual
from torchnep.joint_update import physical_residual
from torchnep import train_nep
from test_oc_spin_training_smoke import _write_case
from torchnep.data import parse_nep_in, read_xyz
from torchnep.train import load_response_dataset


def test_energy_offset_and_huber_identity():
    torch.manual_seed(5)
    e = torch.randn(6, dtype=torch.float64, requires_grad=True)
    m = torch.randn(12, 3, dtype=torch.float64, requires_grad=True)
    batch = dict(energy=torch.randn(6, dtype=torch.float64),
                 natoms=torch.full((6,), 2), mforces=torch.randn_like(m),
                 spin_tangent=torch.randn_like(m),
                 struct_idx=torch.arange(6).repeat_interleave(2),
                 response_group_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
                 response_coordinates=torch.tensor([0., 1., 2., 0., 1., 2.]),
                 num_structures=6)
    result = dict(Etot=e, mforces=m)
    r = response_residual(result, batch, "mforce")
    legacy = generator_curve_loss(m, batch["mforces"], batch["spin_tangent"],
                                  batch["struct_idx"], batch["response_group_ids"],
                                  batch["response_coordinates"], 6)
    torch.testing.assert_close(r.square().sum(), legacy)
    energy = response_residual(result, batch, "energy", energy_scale=0.1)
    result["Etot"] = e + torch.tensor([3., 3., 3., -9., -9., -9.])
    torch.testing.assert_close(energy, response_residual(result, batch, "energy", energy_scale=0.1))
    r = response_residual(dict(Etot=batch["energy"], mforces=batch["mforces"]),
                          batch, "both")
    assert torch.isfinite(r).all() and r.square().sum() == 0
    assert torch.isfinite(torch.autograd.grad(energy.square().sum(), e)[0]).all()


@pytest.mark.parametrize("target", ["energy", "mforce", "both"])
def test_joint_response_trainer(tmp_path, target, monkeypatch):
    config, xyz, valid = _write_case(tmp_path)
    config.write_text(config.read_text().replace("spin_mode 2", "spin_mode 3")
                      + f"\njoint_rank4 1\nspin_curriculum 0\n"
                      f"spin_response_target {target}\nspin_response_energy_scale 0.01\n")
    mod = importlib.import_module("torchnep.train")
    original = mod.optimizer_step
    checked = []
    def step(optimizer, model, batch, weights, backend, schedule, **kw):
        from torchnep.response_training import response_from_store
        import inspect
        expected = inspect.currentframe().f_back.f_locals["loss"].detach()
        actual = physical_residual(model, batch, weights, backend).square().sum()
        for rb, store, w in kw.get("responses", ()):
            actual = actual + w * response_from_store(model, rb, store, backend).square().sum()
        torch.testing.assert_close(actual.detach(), expected, rtol=1e-8, atol=1e-10)
        if kw.get("responses"):
            checked.append(True)
        return original(optimizer, model, batch, weights, backend, schedule, **kw)
    monkeypatch.setattr(mod, "optimizer_step", step)
    out = tmp_path / "out"
    train_nep(str(config), str(xyz), output_dir=str(out),
              device=os.environ.get("JOINT_TEST_DEVICE", "cuda"), precision="float64",
              use_compile=False, restart=False, run_seed=7,
              checkpoint_interval=1, prediction_interval=0)
    assert checked
    checkpoint = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["joint_rank4"]["refreshes"] >= 2


def test_energy_groups_and_input_contract(tmp_path):
    config, xyz, _ = _write_case(tmp_path)
    config.write_text(config.read_text() + "spin_response_target energy\n")
    cfg = parse_nep_in(str(config))
    frames, _ = load_response_dataset(cfg, str(xyz), None)
    assert len(frames) == 3
    assert not any(frame["spin_tangent"].any() for frame in frames)
    for invalid in ("spin_response_target junk", "spin_response_energy_scale 0",
                    "spin_response_energy_scale nan"):
        config.write_text(invalid+"\n")
        with pytest.raises(ValueError):
            parse_nep_in(str(config))


@pytest.mark.parametrize("target", ["energy", "both"])
def test_ddp_response(tmp_path, target):
    if os.environ.get("TORCHNEP_TEST_DDP") != "1":
        pytest.skip("requires two CUDA devices")
    config, xyz, _ = _write_case(tmp_path)
    config.write_text(config.read_text().replace("spin_mode 2", "spin_mode 3") +
                      f"joint_rank4 1\nspin_curriculum 0\nspin_response_target {target}\n"
                      "spin_response_energy_scale 0.01\n")
    driver = tmp_path/"ddp.py"
    driver.write_text("""import sys
from torchnep import train_nep_sharded
train_nep_sharded(sys.argv[1], sys.argv[2], output_dir=sys.argv[3],
precision="float64", use_compile=False, restart=False, run_seed=7,
checkpoint_interval=1, prediction_interval=0)
""")
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run",
                             "--standalone", "--nproc_per_node=2", str(driver),
                             str(config), str(xyz), str(tmp_path/"out")],
                            env=dict(os.environ, OMP_NUM_THREADS="1"),
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout+result.stderr
    saved = torch.load(tmp_path/"out/checkpoint.pt", map_location="cpu", weights_only=False)
    assert saved["loss_weights"]["response_target"] == target
    assert saved["joint_rank4"]["refreshes"] >= 2
