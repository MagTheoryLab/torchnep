import os
import numpy as np
import pytest
import torch

from torchnep import train_nep
from torchnep.data import parse_nep_in
from torchnep.model import NEPModel
from torchnep.response_training import _spin_path_tangents, response_residual
from test_oc_spin_training_smoke import _write_case


def test_sphere_log_exact_and_antipodal():
    x = np.array([-.4, 0., .7])
    for varying in (False, True):
        m = 4.+.1*x+.02*x*x if varying else np.full(3,4.)
        dm = .1+.04*x if varying else np.zeros(3)
        s = np.stack([m*np.cos(x), m*np.sin(x), x*0],axis=1)[:,None,:]
        expected = np.stack([dm*np.cos(x)-m*np.sin(x),dm*np.sin(x)+m*np.cos(x),x*0],axis=1)
        np.testing.assert_allclose(_spin_path_tangents(s,x)[:,0], expected, atol=1e-12)
    with pytest.raises(ValueError, match="Antipodal"):
        _spin_path_tangents(np.array([[[1.,0,0]],[[-1.,0,0]],[[0.,1.,0]]]),x)
    zero = np.zeros((3,1,3))
    np.testing.assert_array_equal(_spin_path_tangents(zero,x),zero)


def test_group_scale_invariance():
    ref = torch.tensor([[1.,0,0], [2.,0,0], [3.,0,0]]*2, dtype=torch.float64)
    batch = dict(mforces=ref, spin_tangent=torch.ones_like(ref),
                 struct_idx=torch.arange(6), num_structures=6,
                 response_group_ids=torch.tensor([0,0,0,1,1,1]))
    first = response_residual(dict(mforces=1.1*ref),batch)
    scaled = ref.clone()
    scaled[3:] *= 1e4
    batch["mforces"] = scaled
    second = response_residual(dict(mforces=1.1*scaled),batch)
    torch.testing.assert_close(first,second,rtol=1e-12,atol=1e-12)


def test_refinement_invalid_inputs(tmp_path):
    path = tmp_path/"nep.in"
    for entry in ("response_refine 2", "response_refine_steps 0",
                  "response_preserve_scale 1 2", "response_preserve_scale 1 nan 2"):
        path.write_text("type 1 Fe\nspin_mode 3\n"+entry+"\n")
        with pytest.raises(ValueError):
            parse_nep_in(str(path))
    path.write_text("type 1 Fe\nspin_mode 3\nresponse_refine 1\n")
    with pytest.raises(ValueError, match="finetune_from"):
        train_nep(str(path),"unused.xyz",output_dir=str(tmp_path/"out"))


def test_prediction_precision_contract():
    # All streaming/store/sharded text writers use the same sufficient precision.
    from pathlib import Path
    import torchnep.predict as predict
    source = Path(predict.__file__).read_text()
    assert 'fmt="%.10g"' not in source
    assert 'fmt="%.16g"' in source
    from io import StringIO
    values = np.array([[-12345.123456789123, -12345.123456789123+1e-7]])
    text = StringIO()
    np.savetxt(text, values, fmt="%.16g")
    text.seek(0)
    np.testing.assert_allclose(np.loadtxt(text),values.ravel(),rtol=0,atol=1e-11)


def test_cache_inactive_species_magnetic_dof(tmp_path):
    from torchnep.data import read_xyz
    from torchnep.train import StreamDataStore, preprocess_structures
    from torchnep.response_refine import FrozenCache
    config, xyz, _ = _write_case(tmp_path)
    config.write_text(config.read_text().replace("type 1 Fe", "type 2 Fe O")+
                      "\nspin_dof_type Fe\nspin_env_type Fe O\n")
    xyz.write_text(xyz.read_text().replace("Fe 1.4", "O 1.4"))
    cfg = parse_nep_in(str(config))
    dev = torch.device(os.environ.get("JOINT_TEST_DEVICE", "cuda"))
    store = StreamDataStore(preprocess_structures(read_xyz(str(xyz)),cfg,np.float64),
                            dev,torch.float64,config=cfg)
    model = NEPModel(cfg).to(device=dev,dtype=torch.float64)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.fitting_nets[0].parameters():
        p.requires_grad_(True)
    cache = FrozenCache(model, store, True)
    with torch.no_grad():
        for p in model.fitting_nets[0].parameters():
            p.add_(1e-3*torch.sin(torch.arange(p.numel(),device=dev,dtype=p.dtype)).reshape_as(p))
    cache.check()
    assert torch.count_nonzero(cache.predict()["mforces"][2]) == 0


@pytest.mark.parametrize("target", ["energy", "mforce", "both"])
def test_production_refinement_entry(tmp_path, target):
    config, xyz, _ = _write_case(tmp_path)
    if target == "energy":
        lines = xyz.read_text().splitlines()
        # A second frame shape exercises cache grouping rather than BFO's 40 atoms.
        xyz.write_text(xyz.read_text()+"\n".join(["2",*lines[1:4]])+"\n")
    config.write_text(config.read_text()+f"\nresponse_refine 1\nresponse_refine_steps 3\n"
                      f"joint_rank4 0\nspin_response_target {target}\n"
                      "spin_response_energy_scale .01\n")
    cfg = parse_nep_in(str(config))
    model = NEPModel(cfg).double()
    model.save_nep_txt(str(tmp_path/"initial.txt"),100,100)
    model.load_weights_from_nep_txt(str(tmp_path/"initial.txt"))
    result = train_nep(str(config),str(xyz),output_dir=str(tmp_path/"out"),
                       finetune_from=str(tmp_path/"initial.txt"),precision="float64",
                       device=os.environ.get("JOINT_TEST_DEVICE","cuda"),use_compile=False)
    assert np.isfinite(result["final_response"])
    assert (tmp_path/"out/nep_refined_candidate.txt").exists()
    assert (tmp_path/"out/nep_final.txt").exists() == result["accepted"]
    saved = NEPModel(cfg).double()
    saved.load_weights_from_nep_txt(str(tmp_path/"out/nep_refined_candidate.txt"))
    for name, tensor in model.state_dict().items():
        if not name.startswith("fitting_nets."):
            torch.testing.assert_close(saved.state_dict()[name],tensor,rtol=1e-12,atol=1e-12)
    with pytest.raises(ValueError, match="empty output"):
        train_nep(str(config),str(xyz),output_dir=str(tmp_path/"out"),
                  finetune_from=str(tmp_path/"initial.txt"),precision="float64")
