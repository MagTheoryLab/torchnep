"""Full-order initialization and fail-closed migration of the retired schedule."""
import copy
import pytest
import torch
from torchnep.data import parse_nep_in
from torchnep.model import NEPModel
from torchnep.train import validate_retired_curriculum_resume
from torchnep.joint_update import JointRank4


@pytest.mark.parametrize('mode', [2, 3])
def test_full_order_from_initialization(tmp_path, mode):
    path = tmp_path / 'nep.in'
    path.write_text(f'type 1 Fe\nspin_mode {mode}\n')
    cfg = parse_nep_in(path)
    assert 'spin_curriculum' not in cfg
    torch.manual_seed(17)
    model = NEPModel(cfg).double()
    start = model.struct_dim + model.spin_descriptor.order_dims[2]
    assert start < model.dim
    assert not hasattr(model, 'spin_curriculum')
    for net in model.fitting_nets:
        assert torch.count_nonzero(net.w0[start:]) == net.w0[start:].numel()
    # Old disabled dictionaries have identical parameter/scaler initialization.
    torch.manual_seed(17)
    old_disabled = NEPModel(dict(cfg, spin_curriculum=False)).double()
    for key, value in model.state_dict().items():
        # LAPACK least-squares radial initialization can differ at ~1e-15.
        atol = 1e-12 if key == 'c_param_spin' else 0
        torch.testing.assert_close(value, old_disabled.state_dict()[key], rtol=0, atol=atol)
    weight = model.fitting_nets[0].w0
    before = weight.detach().clone()
    weight.sum().backward()
    torch.optim.Adam([weight], lr=1e-3).step()
    assert torch.count_nonzero(weight[start:] - before[start:]) == weight[start:].numel()


def test_retired_input(tmp_path):
    path = tmp_path / 'nep.in'
    path.write_text('type 1 Fe\nspin_mode 3\nspin_curriculum 0\n')
    with pytest.warns(FutureWarning, match='obsolete and ignored'):
        cfg = parse_nep_in(path)
    assert 'spin_curriculum' not in cfg
    for value in ['1', '2', 'bad', '']:
        path.write_text(f'type 1 Fe\nspin_mode 3\nspin_curriculum {value}\n')
        with pytest.raises(ValueError, match='has been removed'):
            parse_nep_in(path)
    with pytest.raises(ValueError, match='has been removed'):
        NEPModel(dict(cfg, spin_curriculum=True))


def test_old_resume_boundary():
    validate_retired_curriculum_resume({}, 1)
    validate_retired_curriculum_resume({'curriculum_enabled':False}, 1)
    validate_retired_curriculum_resume({'curriculum_end':20}, 20)
    validate_retired_curriculum_resume({'curriculum_enabled':True, 'curriculum_end':20}, 21)
    for schedule in [{'curriculum_end':20}, {'curriculum_enabled':True}]:
        with pytest.raises(ValueError, match='unfinished spin_curriculum'):
            validate_retired_curriculum_resume(schedule, 10)


def test_legacy_joint_cache_token_migration():
    p = torch.nn.Parameter(torch.ones(2))
    update = JointRank4(torch.optim.Adam([p]))
    state = dict(version=1, steps=4, refreshes=1,
                 cache={'token':((1.,1.,0.,1.,0.),1.,False,())})
    update.load_state_dict(copy.deepcopy(state))
    assert update.cache['token'] == ((1.,1.,0.,1.,0.),())
    state['cache']['token'] = ((1.,1.,0.,1.,0.),0.5,False,())
    update.load_state_dict(state)
    assert update.cache is None
