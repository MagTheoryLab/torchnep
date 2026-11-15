"""GPUMD spin_version 1 layout, symmetry and differentiable energy contract."""
import copy

import pytest
import torch

from torchnep.magnetic_polynomial import MagneticPolynomialDescriptor


def _case(soc=1, order=2, lmax=2, compress=3):
    torch.manual_seed(23)
    descriptor = MagneticPolynomialDescriptor(compress, lmax, order, soc,
                                               basis_size_radial=3, n_max_angular=2).double()
    positions = torch.randn(4, 3, dtype=torch.float64)
    i, j = torch.where(~torch.eye(4, dtype=torch.bool))
    rij = positions[j] - positions[i]
    spins = torch.randn(4, 3, dtype=torch.float64)
    types = torch.tensor([0, 1, 0, 1])
    angular = torch.randn(2, 2, 3, 4, dtype=torch.float64)
    return descriptor, (rij, i, j, spins, 4, types, angular)


@pytest.mark.parametrize('soc,dimension,paths,components', [(1,34,33,34),(0,31,21,81)])
def test_c516_default_layout(soc, dimension, paths, components):
    descriptor = MagneticPolynomialDescriptor(spin_soc=soc)
    assert descriptor.dim == dimension
    assert descriptor.path_count == paths
    assert descriptor.components == components
    assert descriptor.radial_mix.shape == (5,3)
    assert descriptor.center_mix.shape == (3,3)


@torch.random.fork_rng(devices=[])
def test_all_declared_c516_truncations():
    # SOC widths per radial channel, frozen from c516's layout (72dc784).
    soc_widths = [(2,2,2,2), (6,6,6,6), (9,11,11,11),
                  (12,15,15,15), (15,19,19,19)]
    dimensions = []
    for compress in range(1,10):
        for lmax in range(5):
            for order in range(1,5):
                for soc in (0,1):
                    descriptor = MagneticPolynomialDescriptor(compress,lmax,order,soc)
                    width = soc_widths[lmax][order-1] if soc else (lmax+1)*(min(lmax,order)+1)+1
                    assert descriptor.dim == order//2 + compress*width
                    dimensions.append(descriptor.dim)
    assert len(dimensions) == 360
    assert max(dimensions) == 236


@pytest.mark.parametrize('soc', [0,1])
@pytest.mark.parametrize('reflection', [False,True])
def test_rotation_inversion_and_time_reversal(soc, reflection):
    descriptor, args = _case(soc=soc, order=4, lmax=4, compress=1)
    rij, i, j, spins, n, types, angular = args
    rotation, _ = torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
    rotation[:,0] *= torch.linalg.det(rotation)
    if reflection:
        rotation[:,0] *= -1
    expected = descriptor(*args)
    # Spins are axial vectors under reflection.
    actual = descriptor(rij@rotation, i,j, torch.linalg.det(rotation)*spins@rotation, n,types,angular)
    torch.testing.assert_close(actual, expected, rtol=2e-12,atol=2e-11)
    torch.testing.assert_close(descriptor(rij,i,j,-spins,n,types,angular),expected,rtol=0,atol=0)
    if not soc:
        torch.testing.assert_close(descriptor(rij@rotation,i,j,spins,n,types,angular),expected,rtol=2e-12,atol=2e-11)


@pytest.mark.parametrize('soc',[0,1])
def test_gradcheck_and_gradgradcheck_including_projection_weights(soc):
    descriptor,args = _case(soc=soc,order=2,lmax=1,compress=1)
    rij,i,j,spins,n,types,angular=args
    params = dict(descriptor.named_parameters())
    def forward(r,s,c,*weights):
        from torch.func import functional_call
        return functional_call(descriptor,dict(zip(params,weights)),(r,i,j,s,n,types,c))
    inputs=(rij.requires_grad_(),spins.requires_grad_(),angular.requires_grad_(),*params.values())
    assert torch.autograd.gradcheck(forward,inputs,fast_mode=True)
    assert torch.autograd.gradgradcheck(forward,inputs,fast_mode=True)


def test_zero_spin_geometry_and_inactive_type_semantics():
    descriptor,args=_case()
    rij,i,j,spins,n,types,angular=args
    zeros=torch.zeros_like(spins,requires_grad=True)
    q=descriptor(rij,i,j,zeros,n,types,angular)
    assert torch.count_nonzero(q) > 0  # Degree-zero geometry paths are intentional in c516.
    derivative=torch.autograd.grad(q.sum(),zeros,create_graph=True)[0]
    assert torch.isfinite(derivative).all()
    torch.testing.assert_close(derivative,torch.zeros_like(zeros),rtol=0,atol=0)
    active=torch.tensor([True,False])
    expected=descriptor(rij,i,j,spins*active[types,None],n,types,angular)
    actual=descriptor(*args,spin_type_active=active)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    perm=torch.randperm(i.numel())
    torch.testing.assert_close(descriptor(rij[perm],i[perm],j[perm],spins,n,types,angular),descriptor(*args))


def test_shared_structural_angular_weights_receive_magnetic_gradients():
    descriptor,args=_case()
    *rest,angular=args
    angular.requires_grad_()
    q=descriptor(*rest,angular)
    gradient=torch.autograd.grad(q.sum(),angular)[0]
    assert torch.count_nonzero(gradient)>0
    assert not hasattr(descriptor,'radial_leg_mix')


@pytest.mark.parametrize('soc,order,lmax,compress', [(1,2,2,3),(1,4,4,1),(0,4,4,1)])
@pytest.mark.parametrize('device,dtype', [('cpu',torch.float64), *[
    pytest.param('cuda', dtype, marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='CUDA required'))
    for dtype in (torch.float64,torch.float32)]])
def test_radial_contraction_matches_reference_through_second_derivatives(
        monkeypatch, soc, order, lmax, compress, device, dtype):
    from torchnep import ops

    descriptor, args = _case(soc, order, lmax, compress)
    descriptor = descriptor.to(device=device, dtype=dtype)
    args = tuple(x.to(device=device, dtype=dtype if x.is_floating_point() else x.dtype)
                 if isinstance(x, torch.Tensor) else x for x in args)
    for index in (0,3,6):
        args[index].requires_grad_()
    inputs = (args[0],args[3],args[6],*descriptor.parameters())

    def evaluate():
        q = descriptor(*args)
        gradients = torch.autograd.grad(q.square().mean(), inputs, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in gradients), inputs)
        return (q,*gradients,*second)

    actual = evaluate()
    with monkeypatch.context() as patch:
        patch.setattr(ops, '_type_contraction_mulsum', ops._type_contraction_bmm)
        patch.setattr('torchnep.magnetic_polynomial.fused_harmonics', lambda *args: None)
        reference = evaluate()
    for value, expected in zip(actual, reference):
        torch.testing.assert_close(value, expected,
                                   rtol=2e-11 if dtype == torch.float64 else 2e-4,
                                   atol=2e-10 if dtype == torch.float64 else 2e-4)
