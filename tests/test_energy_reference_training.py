"""GPUMD energy-reference conventions through the training entry points."""
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
import pytest
import torch

from torchnep import train_nep
from torchnep.data import parse_nep_in, read_xyz
from torchnep.model import NEPModel
from torchnep.nep import NEPCalculator
from torchnep.spin_schema import read_nep_model
from torchnep.train import fit_energy_baseline_frames


REFERENCE = np.array([-2.72, -7.44])


def _frame(species, energy_tags):
    rows = [f'{len(species)}',
            'Lattice="10 0 0 0 10 0 0 0 10" '
            f'{energy_tags} Properties=species:S:1:pos:R:3:spin:R:3']
    positions = ('1 1 1 1 0 0', '2.2 1 1 0 1 0', '1 2.4 1 0 0 1')
    rows.extend(f'{name} {position}' for name, position in zip(species, positions))
    return '\n'.join(rows) + '\n'


def _case(tmp_path, spin=False):
    config, data, valid = (tmp_path / name for name in ('nep.in', 'train.xyz', 'test.xyz'))
    config.write_text('type 2 Fe Ge\ncutoff 3.5 3.5\nn_max 1 1\nbasis_size 1 1\n'
                      'l_max 2 0 0\nneuron 4\nepoch 1\nbatch 5\n'
                      'lambda_e 1\nlambda_f 0\nlambda_v 0\n'
                      + ('model nep_spin\nspin_version 1\n' if spin else ''))
    data.write_text(_frame(['Fe', 'Fe'], 'energy=-4 weight=2')
                    + _frame(['Fe', 'Ge'], 'energy=-14 energy_weight=3')
                    + _frame(['Ge'] * 3, 'energy=-18 energy_weight=2')
                    + _frame(['Fe', 'Ge'], 'energy=600 energy_weight=0')
                    + _frame(['Fe', 'Fe'], ''))
    valid.write_text(_frame(['Ge', 'Ge'], 'energy=100'))
    return config, data, valid


def _foundation(config, path):
    model = NEPModel(parse_nep_in(config)).double()
    with torch.no_grad():
        model.b1.fill_(1.25)
        model.energy_baseline.copy_(torch.tensor([23., -11.], dtype=torch.float64))
        for net in model.fitting_nets:
            net.w1.zero_()
    model.save_nep_txt(path, 4, 4)


@pytest.mark.parametrize('spin', [False, True])
def test_energy_reference_training_and_reparameterization(tmp_path, spin):
    config, data, valid = _case(tmp_path, spin)
    output = tmp_path / 'output'
    train_nep(str(config), str(data), output_dir=str(output), device='cpu',
              precision='float64', restart=False, run_seed=3, valid_file=str(valid))
    potential = output / 'nep_final.txt'
    np.testing.assert_allclose(read_nep_model(potential)[2], REFERENCE, atol=1e-12, rtol=0)

    calc = NEPCalculator(potential)
    frame = read_xyz(str(data))[1]
    args = (frame['species'], frame['positions'], frame['cell'], frame['spins'])
    before = calc.compute(*args)
    energy_table = np.loadtxt(output / 'energy_train.out')
    assert float(before['energy'].mean()) == pytest.approx(energy_table[1, 0], abs=1e-7)
    calc.energy_baseline.add_(2.5)
    calc.b1.add_(2.5)
    same = calc.compute(*args)
    torch.testing.assert_close(same['energy'], before['energy'], rtol=0, atol=1e-12)
    calc.energy_baseline.add_(torch.tensor([.25, -.5], dtype=calc.dtype))
    shifted = calc.compute(*args)
    torch.testing.assert_close(shifted['energy'] - before['energy'],
                               torch.tensor([.25, -.5], dtype=calc.dtype), rtol=0, atol=1e-12)
    for key in ('forces', 'virial', 'mforces', 'tau'):
        if key in before:
            torch.testing.assert_close(shifted[key], before[key], rtol=0, atol=1e-12)


def test_rank_deficient_reference_uses_minimum_norm():
    frames = [{'species': ['Fe', 'Fe', 'Ge'], 'energy': -15.0}]
    with pytest.warns(RuntimeWarning, match='rank 1 of 3'):
        fit = fit_energy_baseline_frames(frames, ['Fe', 'Ge', 'Ni'])
    np.testing.assert_allclose(fit, [-6., -3., 0.], atol=1e-12, rtol=0)


@pytest.mark.parametrize('mode', ['single', 'sharded', 'streamed'])
def test_ordinary_finetune_and_resume_energy_reference(tmp_path, mode):
    config, data, _ = _case(tmp_path)
    foundation = tmp_path / 'foundation.txt'
    _foundation(config, foundation)
    output = tmp_path / 'fit'
    kwargs = dict(output_dir=str(output), precision='float64', restart=False, run_seed=3)
    if mode == 'single':
        train_nep(str(config), str(data), device='cpu', finetune_from=str(foundation), **kwargs)
        config.write_text(config.read_text().replace('epoch 1', 'epoch 2'))
        train_nep(str(config), str(data), device='cpu',
                  resume_from=str(output / 'checkpoint.pt'), **kwargs)
    else:
        runner = tmp_path / 'train_ddp.py'
        runner.write_text('''import sys
from pathlib import Path
import torch.distributed as dist
from torchnep import train_nep_sharded
config, data, output, foundation = sys.argv[1:]
kwargs = dict(output_dir=output, precision='float64', restart=False, run_seed=3)
train_nep_sharded(config, data, finetune_from=foundation, **kwargs)
if dist.get_rank() == 0:
    path = Path(config)
    path.write_text(path.read_text().replace('epoch 1', 'epoch 2'))
dist.barrier()
train_nep_sharded(config, data, resume_from=str(Path(output) / 'checkpoint.pt'), **kwargs)
''')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                   PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                   TORCHNEP_STREAM_THRESHOLD='0' if mode == 'streamed' else str(2**40))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        run = subprocess.run([sys.executable, '-m', 'torch.distributed.run',
                              '--nproc_per_node=2', '--master_addr=127.0.0.1',
                              f'--master_port={port}', str(runner), str(config), str(data),
                              str(output), str(foundation)], capture_output=True,
                             text=True, env=env, timeout=120)
        assert run.returncode == 0, run.stdout + run.stderr

    # GPUMD starts from the target fit and subtracts the per-type weighted
    # network mean. The foundation has zero w1, hence residual Ei = -b1.
    np.testing.assert_allclose(read_nep_model(output / 'nep_final.txt')[2],
                               REFERENCE + 1.25, atol=1e-12, rtol=0)
    energy = np.loadtxt(output / 'energy_train.out')[:3]
    assert np.dot(energy[:, 0] - energy[:, 1], [1., 3., 2.]) == pytest.approx(0., abs=1e-10)
