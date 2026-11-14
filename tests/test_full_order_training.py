"""All c516 channels train immediately; removed schedule inputs fail closed."""
import pytest
import torch

from torchnep.data import parse_nep_in
from torchnep.model import NEPModel


def test_full_order_from_initialization(tmp_path):
    path = tmp_path / 'nep.in'
    path.write_text('type 1 Fe\nmodel nep_spin\nspin_version 1\nspin_order 4\n')
    config = parse_nep_in(path)
    assert 'spin_curriculum' not in config
    model = NEPModel(config).double()
    start = model.struct_dim
    assert start < model.dim
    assert not hasattr(model, 'spin_curriculum')
    for net in model.fitting_nets:
        assert torch.count_nonzero(net.w0[start:]) == net.w0[start:].numel()
    weight = model.fitting_nets[0].w0
    before = weight.detach().clone()
    weight.sum().backward()
    torch.optim.Adam([weight], lr=1e-3).step()
    assert torch.count_nonzero(weight[start:] - before[start:]) == weight[start:].numel()


@pytest.mark.parametrize('value', ['0', '1', '2', 'bad', ''])
def test_curriculum_input_is_rejected(tmp_path, value):
    path = tmp_path / 'nep.in'
    path.write_text(f'type 1 Fe\nmodel nep_spin\nspin_curriculum {value}\n')
    with pytest.raises(ValueError, match='spin_curriculum'):
        parse_nep_in(path)


@pytest.mark.parametrize('enabled', [False, True])
def test_curriculum_config_is_rejected(tmp_path, enabled):
    path = tmp_path / 'nep.in'
    path.write_text('type 1 Fe\nmodel nep_spin\n')
    config = parse_nep_in(path)
    with pytest.raises(ValueError, match='unsupported spin configuration'):
        NEPModel(dict(config, spin_curriculum=enabled))
