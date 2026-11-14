"""Independent polynomial reference for fused harmonic VJP/double backward."""
import pytest
import torch

from torchnep.cuda_harmonics import fused_harmonics
from torchnep.spin_tables import HARMONICS
from torchnep import ops


def _reference(xyz, lmax):
    columns = [[] for _ in range((lmax + 1)**2)]
    for component, x, y, z, c in HARMONICS:
        if component < len(columns):
            columns[component].append(c * xyz[...,0]**x * xyz[...,1]**y * xyz[...,2]**z)
    return torch.stack([sum(column) for column in columns], -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('lmax', range(5))
@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
@pytest.mark.parametrize('structural', [False,True])
def test_harmonic_forward_vjp_double_backward(monkeypatch,lmax,dtype,structural):
    if structural and lmax == 0:
        pytest.skip('structural basis starts at L1')
    torch.manual_seed(11)
    # Noncontiguous inputs, a launch tail, zero spins and unnormalized vectors.
    x = torch.randn(3,129,device='cuda',dtype=dtype).T.requires_grad_()
    with torch.no_grad():
        x[0] = 0
    width = (lmax+1)**2-int(structural)
    g = torch.randn(129,width,device='cuda',dtype=dtype,requires_grad=True)
    h = torch.randn_like(x)

    def evaluate(fn):
        q = fn(x,lmax)
        dx, = torch.autograd.grad((q*g).sum(),x,create_graph=True)
        second = (torch.autograd.grad((dx*h).sum(),(g,x),allow_unused=True)
                  if dx.requires_grad else (None,None))
        dg, dxx = (torch.zeros_like(ref) if value is None else value
                   for value, ref in zip(second,(g,x)))
        return q, dx, dg, dxx

    actual = evaluate(lambda xyz, order: fused_harmonics(xyz,order,structural))
    with monkeypatch.context() as patch:
        patch.setenv('TORCHNEP_CUDA_HARMONICS','0')
        expected = evaluate(lambda xyz, order: ops.angular_basis(*xyz.unbind(-1),order)
                            if structural else _reference(xyz,order))
    for name, value, ref in zip(('q','vjp','jvp','hvp'),actual,expected):
        error = (value-ref).abs()
        print(f'structural={structural} L={lmax} {dtype} {name}: max_abs={error.max().item():.3e} '
              f'rms={error.square().mean().sqrt().item():.3e}')
        torch.testing.assert_close(value,ref,rtol=2e-12 if dtype==torch.float64 else 2e-5,
                                   atol=2e-11 if dtype==torch.float64 else 2e-4)
    empty = fused_harmonics(x[:0],lmax,structural)
    assert empty.shape == (0,width)
    assert torch.autograd.grad(empty.sum(),x)[0].count_nonzero() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('structural',[False,True])
def test_harmonic_gradcheck_and_gradgradcheck(structural):
    x = torch.randn(4,3,device='cuda',dtype=torch.float64,requires_grad=True)
    fn = lambda xyz: fused_harmonics(xyz,4,structural)
    assert torch.autograd.gradcheck(fn,(x,),fast_mode=True)
    assert torch.autograd.gradgradcheck(fn,(x,),fast_mode=True)


@pytest.mark.parametrize('lmax',range(1,5))
def test_structural_monomial_tables_against_tensor_reference(lmax):
    from torchnep.cuda_harmonics import _tables
    xyz = torch.randn(19,3,dtype=torch.float64)
    c, powers, offsets = _tables(xyz,lmax,True)
    terms = c * xyz[:,None,:].pow(powers[None,:,:]).prod(-1)
    actual = torch.stack([terms[:,a:b].sum(-1) for a,b in zip(offsets[:-1],offsets[1:])],-1)
    expected = ops.angular_basis(*xyz.unbind(-1),lmax)
    torch.testing.assert_close(actual,expected,rtol=2e-12,atol=2e-11)
