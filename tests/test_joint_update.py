"""Joint update algebra, checkpoint and objective contracts (CPU or CUDA)."""
import copy
import os

import pytest
import torch

from torchnep.data import parse_nep_in
from torchnep.joint_update import JointRank4, physical_residual


def setup_problem(dtype=torch.float64):
    device = os.environ.get('JOINT_TEST_DEVICE', 'cpu')
    p = torch.nn.Parameter(torch.tensor([.2, -.3, .4, .7], device=device, dtype=dtype))
    a = p.new_tensor([[1., 2, 3, 0], [2, -1, 0, 1], [0, 2, 1, -2]])
    optimizer = torch.optim.AdamW([p], lr=.01, amsgrad=True)
    return p, a, optimizer, JointRank4(optimizer)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_woodbury_dense_equivalence_grad_and_rng_preservation(dtype):
    p, a, opt, joint = setup_problem(dtype)
    loss = (a @ p).square().sum()
    loss.backward()
    grad = p.grad.clone()
    before = p.detach().clone()
    ref = torch.nn.Parameter(before.clone())
    ref.grad = grad.clone()
    refopt = torch.optim.AdamW([ref], lr=.01, amsgrad=True)
    refopt.step()
    rng = torch.random.get_rng_state().clone()
    joint.step(lambda: a @ p, (1.,))
    torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    u, root = joint.cache['u'].double(), joint.cache['root'].double()
    gram = u.T @ u
    lam = torch.linalg.eigvalsh(gram)[-1]
    dense = torch.eye(p.numel(), device=p.device, dtype=torch.float64) + u @ u.T / lam
    expected = torch.linalg.solve(dense, root * (ref.detach()-before).double()) / root
    torch.testing.assert_close(p.double(), before.double()+expected, rtol=2e-6, atol=1e-8)
    assert .49999 <= joint.last_norm_ratio <= 1.00001


def test_cache_refresh_and_exact_resume():
    p, a, opt, joint = setup_problem()
    def advance(p, opt, joint, token=0):
        opt.zero_grad(); (a @ p).square().sum().backward()
        joint.step(lambda: a @ p, token)
    for _ in range(7):
        advance(p, opt, joint)
    assert joint.refreshes == 1
    q, _, qopt, qjoint = setup_problem()
    with torch.no_grad():
        q.copy_(p)
    qopt.load_state_dict(copy.deepcopy(opt.state_dict()))
    qjoint.load_state_dict(copy.deepcopy(joint.state_dict()))
    for _ in range(18):
        advance(p, opt, joint); advance(q, qopt, qjoint)
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert joint.refreshes == 2
    advance(p, opt, joint, token=1)
    assert joint.refreshes == 3


def test_parser_default_and_rejections(tmp_path):
    path = tmp_path / 'nep.in'
    path.write_text('type 1 Fe\nspin_mode 3\nspin_curriculum 0\n')
    assert parse_nep_in(str(path))['joint_rank4'] is False
    path.write_text(path.read_text() + 'joint_rank4 1\n')
    assert parse_nep_in(str(path))['joint_rank4'] is True
    for extra in ['joint_rank4 2', 'spin_mode 2']:
        original = path.read_text()
        path.write_text(original + extra + '\n')
        with pytest.raises(ValueError):
            parse_nep_in(str(path))
        path.write_text(original)


def test_residual_includes_symmetric_virial_active_spins_and_padding():
    device = os.environ.get('JOINT_TEST_DEVICE', 'cpu')
    x = torch.tensor(2., dtype=torch.float64, device=device, requires_grad=True)
    class Model:
        spin_mode = 3
        spin_dof_type_active = torch.tensor([True, False], device=device)
        def compute_properties_cached(self, batch, **kwargs):
            return dict(Etot=x.expand(2), forces=x.expand(3, 3),
                        virial=x.expand(3, 9), mforces=x.expand(3, 3), tau=x.expand(3, 3))
    batch = dict(natoms=x.new_tensor([2, 1]), atom_types=torch.tensor([0, 1, 0], device=device),
                 struct_idx=torch.tensor([0, 0, 1], device=device), num_structures=2)
    for key, shape in [('energy', (2,)), ('forces', (3, 3)), ('virial', (2, 9)),
                       ('mforces', (3, 3)), ('tau', (3, 3))]:
        batch[key] = x.new_zeros(shape)
    for key, values in [('energy_mask', [True, False]), ('force_mask', [True, True, False]),
                        ('virial_mask', [True, False]), ('mforce_mask', [True, True, False]),
                        ('tau_mask', [True, True, False])]:
        batch[key] = torch.tensor(values, device=device)
    r = physical_residual(Model(), batch, (1, 2, 3, 4, 5), 'mulsum')
    # E: x²/4; F,V,MF: x²; tau: 3x²/2. Inactive atom and padding excluded.
    expected = x*x*(.25 + 2 + 3 + 4 + 5*1.5)
    torch.testing.assert_close(r.square().sum(), expected)
    torch.testing.assert_close(torch.autograd.grad(r.square().sum(), x)[0], 2*expected/x)
