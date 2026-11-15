"""Named block format and independent frozen GPUMD spin oracles."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from torchnep.data import parse_nep_in
from torchnep.model import NEPModel
from torchnep.nep import NEPCalculator
from torchnep.spin_schema import read_nep_model

DATA=Path(__file__).parent/'data'


@pytest.mark.parametrize('name',['c3l2s2','c2l4s4','c1l0s1'])
@pytest.mark.parametrize('device,dtype',[('cpu',torch.float64), *[
    pytest.param('cuda',dtype,marks=pytest.mark.skipif(
        not torch.cuda.is_available(),reason='CUDA required'))
    for dtype in (torch.float64,torch.float32)]])
def test_independent_gpumd_spin_oracle(name,device,dtype):
    folder=DATA/'spin_c516'
    oracle=json.loads((folder/(name+'.json')).read_text())
    calc=NEPCalculator(folder/(name+'.nep'),dtype=dtype,device=device)
    assert calc.dim==oracle['dimension']
    for case_name,case in oracle['cases'].items():
        out=calc.compute(case['types'],np.asarray(case['positions']),np.asarray(case['cell']).reshape(3,3),spins=np.asarray(case['spins']))
        targets={key:np.asarray(case[target]) for key,target in
                 [('energy','potential'),('forces','force'),('mforces','mforce'),('virial','virial')]}
        targets['tau']=np.cross(np.asarray(case['spins']),targets['mforces'])
        rtol,atol=(2e-6,2e-7) if dtype==torch.float64 else (2e-5,2e-6)
        for key,expected in targets.items():
            value=out[key].sum(0) if key=='virial' else out[key]
            actual=value.cpu().numpy().astype(np.float64)
            error=np.abs(actual-expected)
            index=np.unravel_index(error.argmax(),error.shape)
            print('ORACLE_ERROR',json.dumps(dict(model=name,case=case_name,device=device,dtype=str(dtype),
                output=key,max_abs=float(error.max()),rms=float(np.sqrt(np.mean(error**2))),
                max_rel=float((error/np.maximum(np.abs(expected),1e-7)).max()),
                worst_index=[int(i) for i in index],reference=float(expected[index]),candidate=float(actual[index]),
                above_gate=int(np.count_nonzero(error>atol+rtol*np.abs(expected))))))
            np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol)


@pytest.mark.parametrize('spin',[False,True])
def test_named_format_roundtrip_and_parameter_order(tmp_path,spin):
    if spin:
        config,_,_=read_nep_model(DATA/'spin_c516/c3l2s2.nep')
        model=NEPModel(config).double()
        model.load_weights_from_nep_txt(DATA/'spin_c516/c3l2s2.nep')
    else:
        config,_,_=read_nep_model(DATA/'nep_CrCoNi.txt')
        model=NEPModel(config).double()
        model.load_weights_from_nep_txt(DATA/'nep_CrCoNi.txt')
    model.energy_baseline.copy_(torch.arange(model.num_types,dtype=torch.float64)*0.25)
    path=tmp_path/'nep.txt'
    model.save_nep_txt(path,128,128)
    text=path.read_text()
    assert text.startswith(f'nep {model.num_types} ')
    assert f'\nformat_version 1\nmodel {"nep_spin" if spin else "nep"}\n' in text
    parsed,blocks,_=read_nep_model(path)
    restored=NEPModel(parsed).double()
    restored.load_weights_from_nep_txt(path)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)
    torch.testing.assert_close(restored.energy_baseline,model.energy_baseline,rtol=0,atol=0)
    calc=NEPCalculator(path)
    torch.testing.assert_close(calc.c2,model.c_param_2,rtol=0,atol=0)
    torch.testing.assert_close(calc.c3,model.c_param_3,rtol=0,atol=0)
    expected=np.concatenate([t.detach().reshape(-1).numpy() for t in [model.c_param_2,model.c_param_3]])
    start=sum((model.dim+2)*model.num_neurons for _ in model.type_names)+1
    np.testing.assert_array_equal(blocks['parameters'][start:start+expected.size],expected)


@pytest.mark.parametrize('lmax,order,soc,magnetic',[(2,2,1,100),(4,4,0,236),(4,4,1,173)])
@torch.random.fork_rng(devices=[])
def test_extended_spin_shape_roundtrip(tmp_path,lmax,order,soc,magnetic):
    config,_,_=read_nep_model(DATA/'spin_c516/c3l2s2.nep')
    config.update(spin_compress=9,spin_l_max=[lmax,0,0],spin_order=order,spin_soc=soc,
                  n_max_radial=12,n_max_angular=8,basis_size_radial=0,
                  l_max=[8,2,1],neuron=4)
    model=NEPModel(config).double()
    assert model.dim == 103+magnetic
    path=tmp_path/'extended.nep'
    model.save_nep_txt(path,128,128)
    parsed,blocks,_=read_nep_model(path)
    assert len(blocks['q_scaler']) == model.dim
    restored=NEPModel(parsed).double()
    restored.load_weights_from_nep_txt(path)
    for name,value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name],value,rtol=0,atol=0)
    assert NEPCalculator(path).dim == model.dim


@pytest.mark.parametrize('cutoffs',[[3.],[3.,4.,6.]])
@torch.random.fork_rng(devices=[])
def test_spin_cutoff_lower_boundary_roundtrip(tmp_path,cutoffs):
    path=tmp_path/'nep.in'
    path.write_text('type 3 Fe Bi O\nmodel nep_spin\ncutoff 6 5\nspin_cutoff '+
                    ' '.join(map(str,cutoffs))+'\n')
    config=parse_nep_in(path)
    expected=cutoffs*3 if len(cutoffs)==1 else cutoffs
    assert config['spin_cutoff_by_type'] == expected
    model=NEPModel(config)
    path=tmp_path/'cutoff.nep'
    model.save_nep_txt(path,128,128)
    assert read_nep_model(path)[0]['spin_cutoff_by_type'] == expected
    assert NEPCalculator(path).spin_cutoff_by_type == expected


@pytest.mark.parametrize('cutoff',[0.,2.999999,1e-100,float('nan'),float('inf'),6.001])
@pytest.mark.parametrize('position',['uniform','first','last'])
@torch.random.fork_rng(devices=[])
def test_spin_cutoff_contract_rejected_at_all_entry_points(tmp_path,cutoff,position):
    cutoffs={'uniform':[cutoff],'first':[cutoff,6.,6.],'last':[6.,6.,cutoff]}[position]
    values=' '.join(map(str,cutoffs))
    path=tmp_path/'nep.in'
    path.write_text('type 3 Fe Bi O\nmodel nep_spin\ncutoff 6 5\nspin_cutoff '+values+'\n')
    with pytest.raises(ValueError,match='spin_cutoff'):
        parse_nep_in(path)
    config,_,_=read_nep_model(DATA/'spin_c516/c3l2s2.nep')
    config['spin_cutoff_by_type']=cutoffs
    with pytest.raises(ValueError,match='spin_cutoff'):
        NEPModel(config)
    path=tmp_path/'invalid.nep'
    path.write_text((DATA/'spin_c516/c3l2s2.nep').read_text().replace('spin_cutoff 6\n','spin_cutoff '+values+'\n'))
    with pytest.raises(ValueError,match='spin_cutoff'):
        NEPCalculator(path)


@pytest.mark.parametrize('field,value',[('spin_order','1'),('spin_soc','0'),('spin_cutoff','5'),('spin_type','Bi'),('spin_version','2')])
def test_semantic_mismatch_fails_before_mutation(tmp_path,field,value):
    source=DATA/'spin_c516/c3l2s2.nep'
    config,_,_=read_nep_model(source)
    model=NEPModel(config).double()
    before={k:v.clone() for k,v in model.state_dict().items()}
    text=source.read_text()
    lines=text.splitlines()
    lines=[f'{field} {value}' if line.startswith(field+' ') else line for line in lines]
    path=tmp_path/'bad.nep';path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError):model.load_weights_from_nep_txt(path)
    for key,val in before.items():torch.testing.assert_close(model.state_dict()[key],val,rtol=0,atol=0)


@pytest.mark.parametrize('extra',['spin_mode 2','spin_mode 3','spin_env_type Fe','spin_dof_type Fe','spin_version 2','spin_basis_size 9','spin_cutoff 9','spin_order 5','spin_l_max 5','spin_compress 10','spin_cutoff nan'])
def test_removed_or_invalid_input_rejected(tmp_path,extra):
    path=tmp_path/'nep.in'
    path.write_text('type 1 Fe\nmodel nep_spin\n'+extra+'\n')
    with pytest.raises(ValueError):parse_nep_in(path)


def test_ordinary_energy_reference_survives_all_prediction_paths_and_checkpoint(tmp_path):
    from torchnep.train import StreamDataStore, preprocess_structures
    config=dict(num_types=2,type_names=['Fe','Ge'],cutoff_radial=6.,cutoff_angular=5.,
                n_max_radial=2,n_max_angular=2,basis_size_radial=3,basis_size_angular=3,
                l_max=[2,0,0],neuron=4)
    # Match the calculator's FP64 constants, rather than upcasting rounded FP32.
    previous_dtype=torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model=NEPModel(config)
        restored=NEPModel(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.energy_baseline.copy_(torch.tensor([.25,-.5],dtype=torch.float64))
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.energy_baseline,model.energy_baseline,rtol=0,atol=0)
    incomplete=dict(model.state_dict());incomplete.pop('energy_baseline')
    with pytest.raises(RuntimeError, match='energy_baseline'):
        restored.load_state_dict(incomplete)
    path=tmp_path/'ordinary.txt';model.save_nep_txt(path,128,128)
    calc=NEPCalculator(path)
    frame=dict(natoms=3,species=['Fe','Ge','Fe'],positions=np.array([[1.,1.,1.],[2.5,1.,1.],[1.,3.,1.]]),cell=np.eye(3)*20)
    store=StreamDataStore(preprocess_structures([frame],config,np.float64),torch.device('cpu'),torch.float64,config=config)
    batch=store.collate([0])
    direct=calc.compute(frame['species'],frame['positions'],frame['cell'])
    cached=calc.compute_batch(batch)
    tiled=calc.compute_tiled(frame['species'],frame['positions'],frame['cell'],block_size=2)
    trained=model.compute_properties_cached(batch,need_forces=True,need_virial=True)
    # Exercise the pure energy/force function consumed by make_fx, on CPU too.
    from torchnep.compiled_autograd import CompiledAutogradForce
    traced = CompiledAutogradForce(model)
    energy, forces, virial = traced._raw(
        traced._pvals(), batch['rij_rad'], batch['rij_ang'],
        batch['pair_i_rad'], batch['pair_j_rad'],
        batch['pair_i_ang'], batch['pair_j_ang'], batch['atom_types'],
    )
    trace_input = dict(energy=energy, forces=forces, virial=virial)
    for output in (cached,tiled,trained,trace_input):
        energy=output['energy'] if 'energy' in output else output['Ei']
        torch.testing.assert_close(energy,direct['energy'],rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(output['forces'],direct['forces'],rtol=1e-9,atol=1e-9)


@pytest.mark.parametrize('mutation', ['duplicate','missing','unknown','count','nan','negative_scaler'])
def test_malformed_named_model_rejected(tmp_path,mutation):
    source=DATA/'spin_c516/c3l2s2.nep'
    text=source.read_text()
    if mutation=='duplicate': text=text.replace('spin_soc 1','spin_soc 1\nspin_soc 1')
    elif mutation=='missing': text=text.replace('spin_order 2\n','')
    elif mutation=='unknown': text=text.replace('spin_order 2','spin_block 1\nspin_order 2')
    elif mutation=='count': text=text.replace('@ parameters 7270','@ parameters 7269')
    elif mutation=='nan':
        lines=text.splitlines();index=next(i for i,line in enumerate(lines) if line.startswith('@ parameters'))+1
        lines[index]='nan';text='\n'.join(lines)
    else:
        lines=text.splitlines();index=next(i for i,line in enumerate(lines) if line.startswith('@ q_scaler'))+1
        lines[index]='-1';text='\n'.join(lines)
    path=tmp_path/'bad.nep';path.write_text(text)
    with pytest.raises(ValueError):NEPCalculator(path)


@pytest.mark.parametrize('header', ['nep4', 'nep4_zbl', 'nep4_spin2', 'nep4_spin3'])
def test_text_readers_reject_removed_headers(tmp_path, header):
    path = tmp_path / 'old.txt'
    path.write_text(f'{header} 1 Fe\n')
    config, _, _ = read_nep_model(DATA/'nep_CrCoNi.txt')
    model = NEPModel(config).double()
    for read in (NEPCalculator, model.load_weights_from_nep_txt):
        with pytest.raises(ValueError, match='expected nep'):
            read(path)


@pytest.mark.parametrize('key', ['lambda_1', 'lambda_2', 'pos_noise'])
def test_removed_noop_options_are_rejected(tmp_path, key):
    path = tmp_path / 'nep.in'
    path.write_text(f'type 1 Fe\n{key} 0\n')
    with pytest.raises(ValueError, match=key):
        parse_nep_in(path)
