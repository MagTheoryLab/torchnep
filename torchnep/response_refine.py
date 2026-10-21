"""Frozen-descriptor response refinement through the production training entry.

One scalar ANN is retained. Caches are rebuilt from the loaded model and
checked against its live derivatives; no material-specific dimensions or
external derivative caches are accepted. This is not an exact-resume format.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from contextlib import contextmanager

import numpy as np
import torch

from .data import parse_nep_in, read_xyz
from .model import NEPModel
from .response_training import (GENERATOR_CURVE_SCHEMA, configure_response_store,
                                response_residual)


@contextmanager
def _reference_jacobian():
    # Batched VJPs need tensor-only ops; custom CUDA pullbacks have no vmap rule.
    # Restore all execution choices before live-autograd parity and evaluation.
    keys = ("TORCHNEP_OC_CUDA_CENTER", "TORCHNEP_OC_CUDA_BANK",
            "TORCHNEP_OC_CUDA_FULL", "TORCHNEP_OC_CUDA_STRUCT_CENTER")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        os.environ.update({k: "0" for k in keys})
        yield
    finally:
        for k, value in saved.items():
            if value is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = value


def _descriptors(model, b, dr=None, ds=None):
    rij = {}
    for name in ("rad", "ang", "spin"):
        r = b["rij_"+name]
        if dr is not None:
            r = r + dr[b["pair_j_"+name]]-dr[b["pair_i_"+name]]
        rij[name] = r
    return model.compute_descriptors(
        rij["rad"], rij["ang"], b["pair_i_rad"], b["pair_j_rad"],
        b["pair_i_ang"], b["pair_j_ang"], b["atom_types"], b["N"],
        spins=b["spins"] if ds is None else b["spins"]+ds,
        rij_spin=rij["spin"], pi_spin=b["pair_i_spin"],
        pj_spin=b["pair_j_spin"]) * model.q_scaler


def _ann(model, q, types, derivative=False):
    result = torch.zeros_like(q) if derivative else q.new_zeros(len(q))
    for t, net in enumerate(model.fitting_nets):
        mask = types == t
        h = torch.tanh(q[mask] @ net.w0-net.b0)
        result[mask] = ((1-h.square())*net.w1) @ net.w0.T if derivative else h @ net.w1
    return result


def _live(model, batch, derivatives=True):
    return model.compute_properties_cached(
        dict(batch, need_mforce=derivatives, need_tau=derivatives),
        need_forces=derivatives, need_virial=False, backend="loop", create_graph=False)


class FrozenCache:
    """Fixed-feature energy and derivative changes, grouped by frame shape."""
    def __init__(self, model, store, derivatives=False):
        self.model, self.store = model, store
        self.buckets = []
        qs, ts, fs, bases, force, mf = [], [], [], [], [], []
        grouped, used_bytes = {}, 0
        stride = 1 if derivatives else 64
        for i in range(0, store.n, stride):
            b = store.collate(list(range(i, min(i+stride, store.n))))
            mask = model.spin_dof_type_active[b["atom_types"]]
            with torch.no_grad():
                q = _descriptors(model, b)[mask].detach()
            qs.append(q)
            ts.append(b["atom_types"][mask])
            fs.append(b["struct_idx"][mask]+i)
            live = _live(model, b, derivatives)
            bases.append(live["Etot"].detach())
            if derivatives:
                force.append(live["forces"].detach())
                mf.append(live["mforces"].detach())
                if len(q) == 0:
                    continue
                n, d = b["N"], q.shape[1]
                used_bytes += 2*q.numel()*n*3*q.element_size()
                if used_bytes > 2*1024**3:
                    raise ValueError("Derivative cache exceeds 2 GiB; supply a smaller retention.xyz")
                dr = torch.zeros((n, 3), device=q.device, dtype=q.dtype, requires_grad=True)
                ds = torch.zeros_like(dr, requires_grad=True)
                with _reference_jacobian():
                    qq = _descriptors(model, b, dr, ds)[mask].flatten()
                torch.testing.assert_close(qq.detach(), q.flatten(), rtol=1e-10, atol=1e-10)
                jr, js = [], []
                # Bounded batched VJP avoids a full output identity allocation.
                for start in range(0, qq.numel(), 16):
                    stop = min(start+16, qq.numel())
                    eye = qq.new_zeros((stop-start, qq.numel()))
                    eye[torch.arange(stop-start), torch.arange(start, stop)] = 1
                    gr, gs = torch.autograd.grad(qq, (dr, ds), eye,
                                                is_grads_batched=True, retain_graph=True,
                                                allow_unused=True)
                    if gr is None:
                        gr = dr.new_zeros((stop-start, n, 3))
                    if gs is None:
                        gs = ds.new_zeros((stop-start, n, 3))
                    jr.append(gr.detach())
                    # Match compute_properties_cached: inactive species have no
                    # magnetic DOF, even if formal spin-input derivatives exist.
                    js.append(gs.detach()*mask[None, :, None])
                shape = (len(q), d, n, 3)
                item = (i, torch.cat(jr).reshape(shape), torch.cat(js).reshape(shape))
                grouped.setdefault((n, len(q)), []).append(item)
        self.q, self.types, self.frames = torch.cat(qs), torch.cat(ts), torch.cat(fs)
        self.base_e = torch.cat(bases)
        self.old_e = _ann(model, self.q, self.types).detach()
        self.old_dq = _ann(model, self.q, self.types, True).detach()
        self.base_f = torch.cat(force) if derivatives else None
        self.base_m = torch.cat(mf) if derivatives else None
        qoffset = np.cumsum([0]+[len(q) for q in qs])
        aoffset = np.cumsum([0]+store.natoms)
        for (n, na), items in grouped.items():
            qids = torch.tensor([[*range(qoffset[i], qoffset[i+1])] for i,_,_ in items],
                                device=self.q.device)
            aids = torch.tensor([[*range(aoffset[i], aoffset[i+1])] for i,_,_ in items],
                                device=self.q.device)
            self.buckets.append((qids, aids, torch.stack([r for _,r,_ in items]),
                                 torch.stack([s for _,_,s in items])))
        self.bytes = used_bytes

    def predict(self):
        delta = _ann(self.model, self.q, self.types)-self.old_e
        e = self.base_e.index_add(0, self.frames, delta)
        result = {"Etot": e}
        if self.base_f is not None:
            dq = _ann(self.model, self.q, self.types, True)-self.old_dq
            f, m = self.base_f.clone(), self.base_m.clone()
            for qi, ai, jr, js in self.buckets:
                f = f.index_add(0, ai.flatten(), -torch.einsum("bad,badnk->bnk", dq[qi], jr).flatten(0,1))
                m = m.index_add(0, ai.flatten(), -torch.einsum("bad,badnk->bnk", dq[qi], js).flatten(0,1))
            result.update(forces=f, mforces=m)
        return result

    def check(self):
        """Compare changed-ANN cached predictions against independent live AD."""
        got = self.predict()
        offset = 0
        for i, n in enumerate(self.store.natoms):
            live = _live(self.model, self.store.collate([i]), self.base_f is not None)
            torch.testing.assert_close(got["Etot"][i:i+1], live["Etot"], rtol=0, atol=1e-9)
            if self.base_f is not None:
                for key in ("forces", "mforces"):
                    torch.testing.assert_close(got[key][offset:offset+n], live[key], rtol=1e-8, atol=1e-9,
                                               msg=lambda msg: f"frame {i}, {key}: {msg}")
            offset += n


def _metrics(model, store):
    sums, counts = np.zeros(4), np.zeros(4)
    for start in range(0, store.n, 64):
        b = store.collate(list(range(start, min(start+64, store.n))))
        out = _live(model, b)
        mask = model.spin_dof_type_active[b["atom_types"]]
        errors = [(out["Etot"]-b["energy"])/b["natoms"], out["forces"]-b["forces"],
                  out["mforces"][mask]-b["mforces"][mask], out["tau"][mask]-b["tau"][mask]]
        for k, err in enumerate(errors):
            sums[k] += float(err.detach().square().sum())
            counts[k] += 2*err.shape[0] if k == 3 else err.numel()
    return dict(zip(("E", "F", "MF", "torque"), np.sqrt(sums/counts).tolist()))


def refine_response(config_file, data_file, model_file, *, output_dir,
                    device=None, valid_file=None, energy_key="energy"):
    """Opt-in frozen response L-BFGS. Read retention.xyz beside train.xyz.

    If absent, select at most 256 evenly spaced training frames, plus response
    frames. E retention always covers all train frames. Only spin-active ANNs
    change. All caches are reconstructed; scaler/baseline/descriptors stay fixed.
    Publish nep_final.txt only after full E/F/MF/torque and response gates pass.
    """
    from .train import (StreamDataStore, preprocess_structures, load_response_dataset,
                        compute_max_neighbors)
    started = time.perf_counter()
    cfg = parse_nep_in(config_file)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("response_refine supports one process/device, not DDP")
    if cfg.get("joint_rank4", False):
        raise ValueError("response_refine uses L-BFGS: set joint_rank4 0")
    if cfg.get("lambda_spin_response", 0) <= 0 or cfg.get("spin_mode", 0) not in (2,3):
        raise ValueError("response_refine requires O/C spin and lambda_spin_response > 0")
    if not str(model_file).endswith(".txt"):
        raise ValueError("response_refine currently loads a nep.txt, not an optimizer checkpoint")
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("response_refine requires an empty output_dir; never overwrites a prior run")
    valid_frames = read_xyz(valid_file, energy_key=energy_key) if valid_file else None
    response, valid_response = load_response_dataset(cfg, str(data_file), valid_frames,
                                                      energy_key=energy_key)
    frames = read_xyz(str(data_file), energy_key=energy_key)
    if not frames:
        raise ValueError("Refinement requires nonempty train.xyz")
    retention_path = Path(data_file).parent/"retention.xyz"
    indices = np.linspace(0, len(frames)-1, min(256,len(frames)), dtype=int).tolist()
    retention = read_xyz(str(retention_path), energy_key=energy_key) if retention_path.exists() else [frames[i] for i in indices]+response
    if valid_frames:
        def identity(f):
            return (tuple(f["species"]), *(np.round(np.asarray(f[k], dtype=np.float64),8).tobytes()
                                          for k in ("positions","cell","spins")))
        heldout = {identity(f) for f in valid_frames}
        if any(identity(f) in heldout for f in frames+retention):
            raise ValueError("train/retention shares a structure/spin frame with validation")
    for f in frames+retention+(valid_frames or []):
        if not all(k in f for k in ("energy", "forces", "spins", "mforces")):
            raise ValueError("Refinement requires complete E/F/spin/MF labels; virial is optional")
        if not all(np.isfinite(np.asarray(f[k])).all() for k in ("energy", "forces", "spins", "mforces")):
            raise ValueError("Refinement requires finite E/F/spin/MF labels")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    def make_store(data):
        structures = preprocess_structures(data, cfg, np.float64)
        return StreamDataStore(structures, dev, torch.float64, config=cfg), structures
    store, structures = make_store(frames)
    rs, _ = make_store(response)
    keep, _ = make_store(retention)
    vs = make_store(valid_frames)[0] if valid_frames else None
    configure_response_store(rs, cfg)
    rb = rs.collate(list(range(rs.n)))
    model = NEPModel(cfg).to(device=dev, dtype=torch.float64)
    model.load_weights_from_nep_txt(str(model_file))
    for p in model.parameters():
        p.requires_grad_(False)
    for t, net in enumerate(model.fitting_nets):
        if model.spin_dof_type_active[t]:
            for p in net.parameters():
                p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No spin-active ANN parameters")
    model.eval()
    target = cfg.get("spin_response_target", "mforce")
    scale = cfg.get("spin_response_energy_scale", 1e-6)
    preserve = cfg.get("response_preserve_scale", [5e-5,.003,.003])
    out.mkdir(parents=True, exist_ok=True)
    print(f"response refinement: {store.n} train, {rs.n} response, {keep.n} retention frames", flush=True)
    baseline = _metrics(model, store)
    valid_base = _metrics(model, vs) if vs else None
    ec = FrozenCache(model, store)
    print("response refinement: full energy cache ready", flush=True)
    rc = FrozenCache(model, rs, derivatives=target != "energy")
    kc = FrozenCache(model, keep, derivatives=True)
    print(f"response refinement: derivative cache ready ({kc.bytes+rc.bytes} bytes)", flush=True)
    old = [p.detach().clone() for p in params]
    # Test cache away from its construction point, not the tautological delta=0.
    with torch.no_grad():
        for p in params:
            p.add_(1e-5*torch.sin(torch.arange(p.numel(), device=dev, dtype=p.dtype)).reshape_as(p))
    rc.check()
    kc.check()
    with torch.no_grad():
        for p, reference in zip(params, old):
            p.copy_(reference)
    cache_seconds = time.perf_counter()-started
    kb = keep.collate(list(range(keep.n)))
    active = model.spin_dof_type_active[kb["atom_types"]]
    ns = torch.tensor(store.natoms, device=dev, dtype=torch.float64)
    def response_loss():
        return response_residual(rc.predict(), rb, target, energy_scale=scale).square().sum()
    initial_response = float(response_loss().detach())
    steps = cfg.get("response_refine_steps", 2000)
    opt = torch.optim.LBFGS(params, lr=1, max_iter=steps, max_eval=steps*7//4,
                           history_size=50, line_search_fn="strong_wolfe",
                           tolerance_grad=1e-9, tolerance_change=1e-12)
    history = []
    def closure():
        opt.zero_grad()
        kr = kc.predict()
        losses = [cfg["lambda_spin_response"]*response_loss(),
                  (((ec.predict()["Etot"]-ec.base_e)/ns)/preserve[0]).square().mean(),
                  ((kr["forces"]-kc.base_f)/preserve[1]).square().mean(),
                  ((kr["mforces"][active]-kc.base_m[active])/preserve[2]).square().mean(),
                  sum(((p-o)/.1).square().mean() for p,o in zip(params,old))]
        loss = sum(losses)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite refinement objective")
        loss.backward()
        history.append([float(x.detach()) for x in losses])
        if len(history) % 100 == 0:
            print(f"response refinement closure={len(history)} loss={float(loss):.8g}", flush=True)
        return loss
    tick = time.perf_counter()
    opt.step(closure)
    optimize_seconds = time.perf_counter()-tick
    rc.check()
    # Acceptance applies to the serialized potential users will run, not only
    # the higher-precision in-memory ANN. Do not reuse fixed-q caches afterward.
    maxima = compute_max_neighbors(structures)
    model.save_nep_txt(str(out/"nep_refined_candidate.txt"), *maxima)
    model.load_weights_from_nep_txt(str(out/"nep_refined_candidate.txt"))
    final = _metrics(model, store)
    valid_final = _metrics(model, vs) if vs else None
    final_prediction = _live(model, rb, derivatives=target != "energy")
    final_response = float(response_residual(final_prediction, rb, target,
                                            energy_scale=scale).square().sum().detach())
    retention_ok = all(final[k] <= baseline[k]*1.01+1e-12 for k in baseline)
    if vs:
        retention_ok &= all(valid_final[k] <= valid_base[k]*1.01+1e-12 for k in valid_base)
    accepted = retention_ok and final_response <= min(initial_response, 1.)
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result = dict(schema=GENERATOR_CURVE_SCHEMA, baseline=baseline, final=final,
                  valid_baseline=valid_base, valid_final=valid_final,
                  initial_response=initial_response, final_response=final_response,
                  accepted=bool(accepted), retention_pass=bool(retention_ok),
                  response_target=target, response_energy_scale=scale,
                  preserve_scale=preserve, closures=len(history), history=history,
                  preparation_seconds=cache_seconds, optimize_seconds=optimize_seconds,
                  total_seconds=time.perf_counter()-started, derivative_cache_bytes=kc.bytes+rc.bytes,
                  model_sha256=digest(model_file), train_sha256=digest(data_file),
                  validation_sha256=digest(valid_file) if valid_file else None,
                  response_sha256=digest(Path(data_file).parent/"response.xyz"),
                  config_sha256=digest(config_file), source_sha256=digest(__file__),
                  source_hashes={name:digest(Path(__file__).with_name(name)) for name in
                                 ("response_refine.py","response_training.py","model.py","data.py","train.py")},
                  torch_version=torch.__version__, device=str(dev),
                  gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
                  retention_sha256=digest(retention_path) if retention_path.exists() else None,
                  retention_indices=None if retention_path.exists() else indices,
                  scope="Frozen spin-active ANN refinement; full train regression is not a holdout",
                  response_pred_energy=final_prediction["Etot"].detach().cpu().tolist(),
                  response_ref_energy=rb["energy"].cpu().tolist(),
                  response_natoms=rs.natoms)
    (out/"refinement.json").write_text(json.dumps(result, indent=2))
    if accepted:
        model.save_nep_txt(str(out/"nep_final.txt"), *maxima)
    torch.save({"model_state": model.state_dict(), "config": cfg,
                "refinement": result, "exact_resume_supported": False}, out/"refinement.pt")
    print(f"response refinement accepted={accepted}; report={out/'refinement.json'}", flush=True)
    return result
