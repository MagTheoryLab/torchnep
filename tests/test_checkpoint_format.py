"""Current signed checkpoints only: no missing metadata, defaults or migrations."""
import copy

import pytest
import torch

from torchnep.model import NEPModel
from torchnep.train import (
    _load_checkpoint, _load_finetune_weights, _restore_scheduler_state,
    _save_checkpoint, validate_resume_data_lineage,
)


def _case(tmp_path, spin):
    config = dict(num_types=1, type_names=['Fe'], cutoff_radial=6., cutoff_angular=5.,
                  n_max_radial=1, n_max_angular=1, basis_size_radial=2,
                  basis_size_angular=2, l_max=[2, 0, 0], neuron=4)
    if spin:
        config['model'] = 'nep_spin'
    model = NEPModel(config).double()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    path = tmp_path / 'checkpoint.pt'
    _save_checkpoint(
        path, model, optimizer, scheduler, epoch=2, best_loss=float('inf'),
        run_seed=31, loss_weights={'lambda_e': 1.}, best_true_loss=float('inf'),
        best_valid_loss=float('inf'), valid_info={'data': {'sha256': 'fixture'}},
        schedule_state={'stage2_start': None, 'use_swa': False},
        early_stop_state={'best': float('inf'), 'wait': 0, 'prev_in_stage2': False},
    )
    return path, model, optimizer, scheduler


@pytest.mark.parametrize('spin', [False, True])
def test_current_checkpoint_resume_and_finetune(tmp_path, spin):
    path, model, optimizer, scheduler = _case(tmp_path, spin)
    before = copy.deepcopy(model.state_dict())
    info = _load_checkpoint(path, model, optimizer, None, torch.device('cpu'))
    _restore_scheduler_state(info, scheduler, None)
    assert info['epoch'] == 2 and info['run_seed'] == 31
    _load_finetune_weights(model, str(path), torch.device('cpu'))
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize('spin', [False, True])
@pytest.mark.parametrize('field', [
    'model_signature', 'run_seed', 'scheduler_state', 'valid_info', 'schedule_state',
])
def test_resume_rejects_missing_metadata(tmp_path, spin, field):
    path, model, optimizer, _ = _case(tmp_path, spin)
    payload = torch.load(path, weights_only=False)
    payload.pop(field)
    torch.save(payload, path)
    with pytest.raises(ValueError, match=field):
        _load_checkpoint(path, model, optimizer, None, torch.device('cpu'))


@pytest.mark.parametrize('spin', [False, True])
@pytest.mark.parametrize('mutation', ['raw', 'unsigned', 'prefix', 'old_signature', 'missing_baseline'])
def test_finetune_rejects_old_checkpoint_forms(tmp_path, spin, mutation):
    path, model, _, _ = _case(tmp_path, spin)
    payload = torch.load(path, weights_only=False)
    if mutation == 'raw':
        payload = payload['model_state']
    elif mutation == 'unsigned':
        payload.pop('model_signature')
    elif mutation == 'prefix':
        payload['model_state'] = {'model.'+key: val for key, val in payload['model_state'].items()}
    elif mutation == 'old_signature':
        payload['model_signature']['format'] = 'torchnep-nep4-v1'
    else:
        payload['model_state'].pop('energy_baseline')
    torch.save(payload, path)
    with pytest.raises((ValueError, RuntimeError)):
        _load_finetune_weights(model, str(path), torch.device('cpu'))


@pytest.mark.parametrize('saved', [None, {}, {'valid_file': None, 'valid_ratio': 0.1}])
def test_resume_requires_content_addressed_lineage(saved):
    with pytest.raises(ValueError, match='content-addressed data lineage'):
        validate_resume_data_lineage(saved, {'data': {}})


def test_resume_rejects_curriculum_schedule_including_disabled(tmp_path):
    path, model, optimizer, _ = _case(tmp_path, False)
    payload = torch.load(path, weights_only=False)
    payload['schedule_state']['curriculum_enabled'] = False
    torch.save(payload, path)
    with pytest.raises(ValueError, match='schedule_state'):
        _load_checkpoint(path, model, optimizer, None, torch.device('cpu'))


def test_scheduler_state_is_required(tmp_path):
    _, _, _, scheduler = _case(tmp_path, False)
    with pytest.raises(ValueError, match='scheduler_state'):
        _restore_scheduler_state({'in_stage2': False}, scheduler, None)
