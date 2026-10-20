"""Production trainer smoke: Stage 2, curriculum, virial and DDP padding."""
import os
import importlib
import inspect
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_oc_spin_training_smoke import _write_case
from torchnep import train_nep
from torchnep.joint_update import physical_residual


def _case(tmp_path, curriculum):
    config, xyz, valid = _write_case(tmp_path)
    config.write_text(config.read_text().replace('spin_mode 2', 'spin_mode 3')
                      .replace('lambda_spin_response 0.1', 'lambda_spin_response 0')
                      .replace('epoch 2', 'epoch 4').replace('start_stage2 2', 'start_stage2 3')
                      + f'joint_rank4 1\nspin_curriculum {curriculum}\n')
    return config, xyz, valid


def _check(output):
    ckpt = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert ckpt['epoch'] == 4
    state = ckpt['joint_rank4']
    assert state['steps'] > 0 and state['refreshes'] >= 2
    assert ckpt['in_stage2']
    assert np.isfinite(np.loadtxt(output / 'loss.out')).all()
    for name in ['mforce_train.out', 'tau_train.out', 'virial_train.out']:
        assert np.isfinite(np.loadtxt(output / name)).all()


@pytest.mark.parametrize('curriculum', [0, 1])
@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_single_joint_stage2(tmp_path, curriculum, precision, monkeypatch):
    config, xyz, valid = _case(tmp_path, curriculum)
    out = tmp_path / 'output'
    train_module = importlib.import_module('torchnep.train')
    original_step = train_module.optimizer_step
    seen = set()
    def checked_step(optimizer, model, batch, weights, backend, schedule):
        if weights not in seen:
            expected = inspect.currentframe().f_back.f_locals['loss'].detach()
            residual = physical_residual(model, batch, weights, backend)
            torch.testing.assert_close(residual.square().sum().detach(), expected,
                                       rtol=5e-5 if precision == 'float32' else 1e-9,
                                       atol=1e-8)
            seen.add(weights)
        return original_step(optimizer, model, batch, weights, backend, schedule)
    monkeypatch.setattr(train_module, 'optimizer_step', checked_step)
    train_nep(str(config), str(xyz), output_dir=str(out),
              device=os.environ.get('JOINT_TEST_DEVICE', 'cpu'), precision=precision,
              use_compile=False, restart=False, run_seed=7,
              checkpoint_interval=1, prediction_interval=0)
    _check(out)
    assert len(seen) == 2


@pytest.mark.parametrize('curriculum', [0, 1])
def test_ddp_joint_stage2_padding(tmp_path, curriculum):
    if os.environ.get('TORCHNEP_TEST_DDP') != '1':
        pytest.skip('set TORCHNEP_TEST_DDP=1 to run distributed smoke')
    config, xyz, valid = _case(tmp_path, curriculum)
    driver = tmp_path / 'ddp.py'
    driver.write_text('''import sys
from torchnep import train_nep_sharded
train_nep_sharded(sys.argv[1], sys.argv[2], output_dir=sys.argv[3],
    precision="float64", use_compile=False, restart=False,
    run_seed=7, checkpoint_interval=1, prediction_interval=0)
''')
    out = tmp_path / 'ddp-output'
    env = dict(os.environ, OMP_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run',
                             '--standalone', '--nproc_per_node=2', str(driver),
                             str(config), str(xyz), str(out)], env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    _check(out)
