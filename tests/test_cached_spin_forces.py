"""Live-weight cached force/MF projection against independent raw-coordinate AD."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from torchnep.model import NEPModel, gpumd_init_parameters
from torchnep.spin_schema import read_nep_model
from torchnep.train import StreamDataStore, preprocess_structures

DATA = Path(__file__).parent/'data/spin_c516'


def _case(name='c3l2s2',case='triclinic',device='cpu',dtype=torch.float64):
    config,_,_ = read_nep_model(DATA/(name+'.nep'))
    model = NEPModel(config).to(device=device,dtype=dtype)
    model.load_weights_from_nep_txt(DATA/(name+'.nep'))
    frame = json.loads((DATA/(name+'.json')).read_text())['cases'][case]
    structure = dict(natoms=len(frame['types']),species=frame['types'],
                     positions=np.asarray(frame['positions']),cell=np.asarray(frame['cell']).reshape(3,3),
                     spins=np.asarray(frame['spins']))
    store = StreamDataStore(preprocess_structures([structure],config,
                            np.float64 if dtype==torch.float64 else np.float32),
                            torch.device(device),dtype,config=config)
    batch = store.collate([0])
    batch.update(need_mforce=True,need_tau=True)
    return model,batch


def _raw(model,batch,forces=True,virial=True,backend="loop"):
    return model.compute_properties(batch['rij_rad'],batch['rij_ang'],
        batch['pair_i_rad'],batch['pair_j_rad'],batch['pair_i_ang'],batch['pair_j_ang'],
        batch['atom_types'],batch['N'],batch['struct_idx'],batch['num_structures'],
        spins=batch['spins'],rij_spin=batch['rij_spin'],pi_spin=batch['pair_i_spin'],pj_spin=batch['pair_j_spin'],
        need_forces=forces,need_virial=virial,need_mforce=True,need_tau=True,create_graph=True,backend=backend)


def _loss(out):
    return sum((value.square().sum()+value.sum()*.03) for value in out.values())


@pytest.mark.parametrize('name',['c3l2s2','c2l4s4','c1l0s1'])
@pytest.mark.parametrize('case',['large','triclinic','zero','isolated'])
@pytest.mark.parametrize('device,dtype',[('cpu',torch.float64), *[
    pytest.param('cuda',dtype,marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required'))
    for dtype in (torch.float32,torch.float64)]])
def test_cached_outputs_and_all_parameter_gradients(name,case,device,dtype):
    model,batch = _case(name,case,device,dtype)
    actual = model.compute_properties_cached(batch,need_virial=True,backend='mulsum')
    ref = _raw(model,batch)
    tolerance = dict(rtol=3e-9,atol=3e-10) if dtype==torch.float64 else dict(rtol=5e-4,atol=3e-5)
    assert actual.keys()==ref.keys()
    for key in actual:
        torch.testing.assert_close(actual[key],ref[key],**tolerance,msg=lambda s:f'{key}: {s}')
    params = tuple(model.parameters())
    got = torch.autograd.grad(_loss(actual),params,allow_unused=True)
    expected = torch.autograd.grad(_loss(ref),params,allow_unused=True)
    for (name,_),a,b in zip(model.named_parameters(),got,expected):
        assert (a is None)==(b is None),name
        if a is not None:
            torch.testing.assert_close(a,b,**tolerance,msg=lambda s:f'{name}: {s}')


def test_cached_basis_reuse_parameter_updates_and_input_invalidation():
    model,batch = _case()
    model.compute_properties_cached(batch,need_virial=True)
    cached = batch['spin_basis']
    with torch.no_grad():
        for p in model.parameters():p.add_(.001)
    out = model.compute_properties_cached(batch,need_virial=True)
    assert batch['spin_basis'] is cached
    for key,value in _raw(model,batch).items():
        torch.testing.assert_close(out[key],value,rtol=2e-9,atol=2e-10)
    for change in ['spin_inplace','spin_replace','strain']:
        if change=='spin_inplace':batch['spins'].mul_(.9)
        elif change=='spin_replace':batch['spins']=batch['spins']+.1
        else:
            for key in ('rij_rad','rij_ang','rij_spin'):batch[key]=batch[key]*1.002
        cached=batch['spin_basis']
        out=model.compute_properties_cached(batch,need_virial=True)
        assert batch['spin_basis'] is not cached
        for key,value in _raw(model,batch).items():
            torch.testing.assert_close(out[key],value,rtol=2e-9,atol=2e-10)


def test_virial_only_and_nograd_inference():
    model,batch = _case()
    out=model.compute_properties_cached(batch,need_forces=False,need_virial=True)
    ref=_raw(model,batch,False,True)
    for key in ref:torch.testing.assert_close(out[key],ref[key],rtol=2e-9,atol=2e-10)
    with torch.no_grad():
        out=model.compute_properties_cached(batch,need_virial=True)
    assert all(not value.requires_grad for value in out.values())
    basis=batch['spin_basis']
    with torch.inference_mode():
        out=model.compute_properties_cached(batch,need_virial=True)
    assert batch['spin_basis'] is basis
    out=model.compute_properties_cached(batch,need_virial=True)
    assert out['mforces'].requires_grad


@pytest.mark.parametrize('soc,order,lmax',[(0,2,2),(0,4,4),(1,3,2)])
@pytest.mark.parametrize('cutoffs',[[4.,5.,6.],[3.,4.,6.]])
def test_general_projection_and_typewise_cutoff(soc,order,lmax,cutoffs):
    config,_,_=read_nep_model(DATA/'c3l2s2.nep')
    config.update(spin_soc=soc,spin_order=order,spin_l_max=[lmax,0,0],spin_compress=1,
                  spin_cutoff_by_type=cutoffs,spin_cutoff=[6.,6.])
    model=NEPModel(config).double()
    _,batch=_case()
    batch['spins']=batch['spins']*.1
    actual=model.compute_properties_cached(batch,need_virial=True,backend='mulsum')
    ref=_raw(model,batch)
    for key in ref:torch.testing.assert_close(actual[key],ref[key],rtol=3e-9,atol=3e-9)
    ga=torch.autograd.grad(_loss(actual),tuple(model.parameters()),allow_unused=True)
    gb=torch.autograd.grad(_loss(ref),tuple(model.parameters()),allow_unused=True)
    for a,b in zip(ga,gb):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=3e-8,atol=3e-8)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_compiled_cached_core_outputs_gradients_and_changing_shapes():
    from torchnep.compiled_autograd import apply_compile_patches
    apply_compile_patches()
    model,batch = _case(device='cuda',dtype=torch.float32)
    compiled=torch.compile(model._cached_core,dynamic=True,fullgraph=True)
    for case in ('triclinic','large','isolated'):
        _,batch=_case(case=case,device='cuda',dtype=torch.float32)
        actual=model.compute_properties_cached(batch,need_virial=True,backend='mulsum',core_fn=compiled)
        expected=_raw(model,batch)
        for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=8e-4,atol=5e-5)
        ga=torch.autograd.grad(_loss(actual),tuple(model.parameters()),allow_unused=True)
        gb=torch.autograd.grad(_loss(expected),tuple(model.parameters()),allow_unused=True)
        for a,b in zip(ga,gb):
            assert (a is None)==(b is None)
            if a is not None:torch.testing.assert_close(a,b,rtol=1e-3,atol=5e-5)


@pytest.mark.parametrize('epsilon',[1e-4,1e-5])
def test_force_mforce_and_raw9_strain_are_energy_derivatives(epsilon):
    model,batch=_case()
    result=model.compute_properties_cached(batch,need_virial=True)
    atom=int(model.spin_dof_type_active[batch['atom_types']].nonzero()[0])
    def energy(kind,delta):
        b=dict(batch,need_mforce=False,need_tau=False)
        if kind=='spin':
            b['spins']=batch['spins'].clone();b['spins'][atom,0]+=delta
        else:
            for field,prefix in [('rij_rad','rad'),('rij_ang','ang'),('rij_spin','spin')]:
                r=batch[field].clone()
                if kind=='position':
                    r[:,0]+=delta*((batch['pair_j_'+prefix]==atom).to(r)-(batch['pair_i_'+prefix]==atom).to(r))
                else:r[:,1]+=delta*r[:,0]
                b[field]=r
        return model.compute_properties_cached(b,need_forces=False)['Etot'].sum()
    for kind,expected in [('spin',result['mforces'][atom,0]),('position',result['forces'][atom,0]),
                           ('strain',result['virial'][:,1].sum())]:
        derivative=-(energy(kind,epsilon)-energy(kind,-epsilon))/(2*epsilon)
        torch.testing.assert_close(derivative,expected,rtol=3e-5,atol=2e-7)


def _matrix_case(compress,lmax,order,soc,dtype,maximal_structural=False,cutoffs=None,device='cuda'):
    config,_,_=read_nep_model(DATA/'c3l2s2.nep')
    config.update(spin_compress=compress,spin_l_max=[lmax,0,0],spin_order=order,spin_soc=soc,
                  n_max_radial=1,n_max_angular=1,basis_size_radial=2,basis_size_angular=2,
                  spin_basis_size=[2,0],l_max=[2,0,0],neuron=4)
    if maximal_structural:
        config.update(n_max_radial=12,n_max_angular=8,basis_size_radial=0,
                      basis_size_angular=8,spin_basis_size=[8,0],l_max=[8,2,1])
    if cutoffs is not None:
        config['spin_cutoff_by_type']=cutoffs
    torch.manual_seed(17+compress+10*lmax+100*order+1000*soc)
    model=NEPModel(config).to(device=device,dtype=dtype)
    gpumd_init_parameters(model)
    _,batch=_case(device=device,dtype=dtype)
    batch['spins']=batch['spins']*.1
    from torchnep import ops
    (batch['fk_rad'],batch['fkp_rad'],batch['d12inv_rad'],batch['fk_ang'],
     batch['fkp_ang'],batch['d12inv_ang'],batch['blm'])=ops.structural_basis(
         batch['rij_rad'],batch['rij_ang'],6.,5.,config['basis_size_radial'],
         config['basis_size_angular'],config['l_max'][0])
    with torch.no_grad():
        q=model.compute_descriptors(batch['rij_rad'],batch['rij_ang'],batch['pair_i_rad'],batch['pair_j_rad'],
            batch['pair_i_ang'],batch['pair_j_ang'],batch['atom_types'],batch['N'],backend='mulsum',
            spins=batch['spins'],rij_spin=batch['rij_spin'],pi_spin=batch['pair_i_spin'],pj_spin=batch['pair_j_spin'])
        model.q_scaler.copy_(torch.maximum(q.square().mean(0).sqrt(),q.max(0).values-q.min(0).values).clamp_min(1).reciprocal())
    return model,batch


_MATRIX_SKIP = not torch.cuda.is_available() or os.environ.get('TORCHNEP_TEST_SPIN_MATRIX')!='1'


@pytest.mark.skipif(_MATRIX_SKIP,reason='set TORCHNEP_TEST_SPIN_MATRIX=1 on CUDA')
def test_all_legal_magnetic_truncations_against_raw_ad():
    from torchnep.compiled_autograd import apply_compile_patches
    apply_compile_patches()
    passed=0
    for compress in range(1,10):
        for lmax in range(5):
            for order in range(1,5):
                for soc in (0,1):
                    model,batch=_matrix_case(compress,lmax,order,soc,torch.float64)
                    actual=model.compute_properties_cached(batch,need_virial=True,backend='mulsum')
                    expected=_raw(model,batch,backend='mulsum')
                    worst=0.
                    for key in expected:
                        worst=max(worst,(actual[key]-expected[key]).abs().max().item())
                        torch.testing.assert_close(actual[key],expected[key],rtol=2e-9,atol=2e-11,
                                                   msg=lambda s:f'C{compress} L{lmax} O{order} SOC{soc} {key}: {s}')
                    ga=torch.autograd.grad(_loss(actual),tuple(model.parameters()),allow_unused=True)
                    gb=torch.autograd.grad(_loss(expected),tuple(model.parameters()),allow_unused=True)
                    for (name,_),a,b in zip(model.named_parameters(),ga,gb):
                        assert (a is None)==(b is None),name
                        if a is not None:
                            torch.testing.assert_close(a,b,rtol=2e-9,atol=2e-11,
                                                       msg=lambda s:f'C{compress} L{lmax} O{order} SOC{soc} {name}: {s}')
                    passed+=1
                    print(f'SPIN_MATRIX C{compress} L{lmax} O{order} SOC{soc} output_max_abs={worst:.3e}',flush=True)
    assert passed==360
    print(f'SPIN_MATRIX_SUMMARY passed={passed}',flush=True)


@pytest.mark.skipif(_MATRIX_SKIP,reason='set TORCHNEP_TEST_SPIN_MATRIX=1 on CUDA')
@pytest.mark.parametrize('compress,lmax,order,soc,maximal_structural,dtype',[
    (9,0,4,1,False,torch.float32),(1,4,4,0,False,torch.float32),(2,4,4,1,False,torch.float32),
    (3,1,3,1,False,torch.float32),(3,3,2,0,False,torch.float32),(1,0,1,0,False,torch.float32),
    *[(9,l,o,s,True,d) for l,o,s in ((2,2,1),(4,4,0),(4,4,1))
      for d in (torch.float32,torch.float64)]])
def test_compiled_magnetic_boundary_configurations(compress,lmax,order,soc,maximal_structural,dtype):
    from torchnep.compiled_autograd import apply_compile_patches
    apply_compile_patches()
    model,batch=_matrix_case(compress,lmax,order,soc,dtype,maximal_structural)
    compiled=torch.compile(model._cached_core,dynamic=True,fullgraph=True)
    actual=model.compute_properties_cached(batch,need_virial=True,backend='mulsum',core_fn=compiled)
    expected=_raw(model,batch,backend='mulsum')
    tolerance=dict(rtol=2e-9,atol=2e-11) if dtype==torch.float64 else dict(rtol=5e-4,atol=5e-6)
    for key in expected:torch.testing.assert_close(actual[key],expected[key],**tolerance)
    ga=torch.autograd.grad(_loss(actual),tuple(model.parameters()),allow_unused=True)
    gb=torch.autograd.grad(_loss(expected),tuple(model.parameters()),allow_unused=True)
    if maximal_structural and dtype==torch.float32:
        # L8 force-loss gradients can cancel to near zero in FP32. Qualify both
        # executions against matched-input FP64, including the full gradient norm.
        reference=copy.deepcopy(model).double()
        batch64={k:v.double() if isinstance(v,torch.Tensor) and v.is_floating_point() else v
                 for k,v in batch.items()}
        oracle=_raw(reference,batch64,backend='mulsum')
        gradients=torch.autograd.grad(_loss(oracle),tuple(reference.parameters()),allow_unused=True)
        for (name,_),a,b,g in zip(model.named_parameters(),ga,gb,gradients):
            assert (a is None)==(b is None)==(g is None),name
            if g is not None:
                for candidate in (a,b):
                    error=(candidate.double()-g).abs()
                    assert error.norm() <= 5e-5*g.norm().clamp_min(1),name
                    assert error.max() <= 5e-5*g.abs().max().clamp_min(1),name
    else:
        tolerance=dict(rtol=2e-9,atol=2e-11) if dtype==torch.float64 else dict(rtol=2e-3,atol=2e-5)
        for a,b in zip(ga,gb):
            assert (a is None)==(b is None)
            if a is not None:torch.testing.assert_close(a,b,**tolerance)


@pytest.mark.skipif(os.environ.get('TORCHNEP_TEST_SPIN_MATRIX')!='1',
                    reason='set TORCHNEP_TEST_SPIN_MATRIX=1')
@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='CUDA required'))])
@torch.no_grad()
def test_all_magnetic_truncations_symmetry_and_derivative_covariance(device):
    from torchnep.compiled_autograd import apply_compile_patches
    apply_compile_patches()
    rotation,_=torch.linalg.qr(torch.tensor([[.3,-.7,.2],[.8,.4,-.5],[-.6,.1,.9]],
                                         dtype=torch.float64,device=device))
    rotation[:,0]*=torch.linalg.det(rotation)
    identity=torch.eye(3,dtype=torch.float64,device=device)
    reflection=rotation@torch.diag(identity.new_tensor([-1.,1.,1.]))
    passed=0
    for compress in range(1,10):
        for lmax in range(5):
            for order in range(1,5):
                for soc in (0,1):
                    model,batch=_matrix_case(compress,lmax,order,soc,torch.float64,
                                            cutoffs=[3.,4.,6.],device=device)
                    model.eval()
                    # Order-four primitives must be appreciable in this check.
                    batch['spins']=batch['spins']*10
                    def descriptors(b):
                        return model.compute_descriptors(b['rij_rad'],b['rij_ang'],
                            b['pair_i_rad'],b['pair_j_rad'],b['pair_i_ang'],b['pair_j_ang'],
                            b['atom_types'],b['N'],backend='mulsum',spins=b['spins'],
                            rij_spin=b['rij_spin'],pi_spin=b['pair_i_spin'],pj_spin=b['pair_j_spin'])
                    q=descriptors(batch)
                    model.q_scaler.copy_(torch.maximum(q.square().mean(0).sqrt(),
                        q.max(0).values-q.min(0).values).clamp_min(1).reciprocal())
                    expected=model.compute_properties_cached(batch,need_virial=True,backend='mulsum')
                    transforms=[('joint rotation',rotation,rotation,1.),
                        ('joint reflection',reflection,-reflection,1.),
                        ('inversion',-identity,identity,1.),
                        ('time reversal',identity,identity,-1.)]
                    if not soc:
                        transforms.extend([('lattice rotation',rotation,identity,1.),
                            ('lattice reflection',reflection,identity,1.),
                            ('spin rotation',identity,rotation.T,1.)])
                    for label,space,spin,sign in transforms:
                        changed=dict(batch,spins=sign*batch['spins']@spin)
                        for key in ('rij_rad','rij_ang','rij_spin'):
                            changed[key]=batch[key]@space
                        actual_q=descriptors(changed)
                        message=f'C{compress} L{lmax} O{order} SOC{soc} {label}'
                        # Magnetic harmonics use FP64 tables. Structural constants
                        # retain their production FP32 rounding after model.double().
                        torch.testing.assert_close(actual_q[:,model.struct_dim:],q[:,model.struct_dim:],
                            rtol=2e-10,atol=2e-11,msg=message+' magnetic descriptors')
                        torch.testing.assert_close(actual_q[:,:model.struct_dim],q[:,:model.struct_dim],
                            rtol=3e-7,atol=3e-9,msg=message+' structural descriptors')
                        actual=model.compute_properties_cached(changed,need_virial=True,backend='mulsum')
                        targets=dict(Ei=expected['Ei'],Etot=expected['Etot'],
                            forces=expected['forces']@space,mforces=sign*expected['mforces']@spin,
                            virial=(space.T@expected['virial'].reshape(-1,3,3)@space).reshape(-1,9),
                            tau=torch.linalg.det(spin)*expected['tau']@spin)
                        for key,value in targets.items():
                            torch.testing.assert_close(actual[key],value,rtol=8e-7,atol=2e-9,
                                                       msg=message+' '+key)
                    passed+=1
                    print(f'SPIN_SYMMETRY C{compress} L{lmax} O{order} SOC{soc} passed',flush=True)
    assert passed==360
    print(f'SPIN_SYMMETRY_SUMMARY passed={passed}',flush=True)
