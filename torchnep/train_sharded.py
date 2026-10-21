# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.
# TorchNEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# TorchNEP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with TorchNEP.  If not, see <http://www.gnu.org/licenses/>.

"""
Data-sharded distributed NEP training.

Each rank keeps only 1/N of the preprocessed structures in its device data
store, so device data memory scales as 1/N. The current reader still parses
the complete XYZ on every rank before discarding non-local host frames.
Gradients are all-reduced by DDP; q_scaler statistics and per-epoch metrics
are all-reduced explicitly.

Usage (typical multi-GPU launch via torchrun):

    torchrun --nproc_per_node=N run_train.py

where run_train.py calls ``train_nep_sharded(...)`` instead of ``train_nep``.

Backend is chosen automatically: NCCL when every rank has its own CUDA
device, otherwise gloo (covers CPU-only testing and the GPU-sharing case).
Single-rank (N=1) runs but offers nothing over ``train_nep`` in that case.
"""

import os
import atexit
import time
import warnings
import torch
import torch.distributed as dist
import numpy as np
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel

import torch.nn as nn
from .joint_update import configure_joint_update, optimizer_step
from .model import NEPModel, gpumd_init_parameters
from .data import (read_xyz, parse_nep_in, valid_split_indices,
                   stratified_split_indices)
from . import ops
from . import __version__
from .predict import predict_from_store_sharded
from .model import slim_model
from .spin_schema import resolve_gpumd_qscaler
from .response_training import (
    GENERATOR_CURVE_HUBER_DELTA,
    GENERATOR_CURVE_MEAN_WEIGHT,
    GENERATOR_CURVE_SCALE_FLOOR,
    GENERATOR_CURVE_SCHEMA,
    balanced_group_step_batches,
    complete_group_batches,
    generator_curve_loss,
    configure_response_store,
    response_from_store,
)


_PG_ATEXIT_DONE = False


def _register_pg_atexit():
    """Register a one-time atexit hook to destroy the process group on exit.

    We deliberately do NOT destroy the group at the end of train_nep_sharded so
    the same process can run several trainings back to back; cleanup happens
    once, at interpreter shutdown, which also avoids the "process group has NOT
    been destroyed" warning a bare leak would print.
    """
    global _PG_ATEXIT_DONE
    if _PG_ATEXIT_DONE:
        return
    _PG_ATEXIT_DONE = True

    def _cleanup():
        if dist.is_available() and dist.is_initialized():
            # Rendezvous before tearing down: rank 0 hosts the TCPStore, and
            # if it exits first the other ranks' NCCL heartbeat monitors spam
            # "failed to recv" warnings while polling the dead store.
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()

    atexit.register(_cleanup)


class _NEPDDPShim(nn.Module):
    """Thin wrapper whose forward calls compute_properties{_cached}.

    Why this exists: DDP's gradient all-reduce is armed inside
    ``DistributedDataParallel.forward`` (it calls
    ``reducer.prepare_for_backward`` on the output). Calling
    ``self.module.compute_properties_cached(...)`` directly — even though the
    parameters live in the module — bypasses DDP's forward and therefore the
    reducer. Each rank then keeps its local gradient; weights drift per-rank
    and only rank 0's overfit copy is saved. Putting the compute call inside
    this shim's ``forward`` puts it on the DDP path.
    """

    def __init__(self, model: NEPModel, use_compile: bool = False,
                 use_autograd_forces: bool = False):
        super().__init__()
        self.model = model
        # torch.compile is applied to the bound analytical compute method, not
        # to ``model`` — the heavy compute lives in compute_properties_cached,
        # NOT in NEPModel.forward(), so compiling the module would be a no-op.
        # The compiled call stays INSIDE this shim's forward, so it remains on
        # DDP's forward path and the reducer still arms backward all-reduce; the
        # compiled region does not span the DDP boundary.
        self._ag_compute = None
        if use_compile and hasattr(torch, "compile"):
            if use_autograd_forces:
                # make_fx-traced autograd forces (see compiled_autograd):
                # parameters enter the traced graph as function inputs, so
                # loss.backward() still fills the same nn.Parameter .grads
                # that DDP's reducer hooks watch — every parameter appears
                # in the graph (weight stacks touch all types), so no
                # find_unused_parameters needed.
                from .compiled_autograd import CompiledAutogradForce
                self._ag_compute = CompiledAutogradForce(model)
            else:
                # Compile the branch-free core; result assembly stays eager
                # in the wrapper (see NEPModel._cached_core).
                import functools
                self._compute_cached = functools.partial(
                    model.compute_properties_cached,
                    core_fn=torch.compile(model._cached_core, dynamic=True))
        if not hasattr(self, "_compute_cached"):
            self._compute_cached = model.compute_properties_cached

    def forward(self, batch, use_autograd_forces: bool,
                need_forces: bool, need_virial: bool,
                need_mforce: bool, need_tau: bool, backend: str):
        if use_autograd_forces:
            fn = (self._ag_compute.compute_properties
                  if self._ag_compute is not None
                  else self.model.compute_properties)
            return fn(
                batch["rij_rad"], batch["rij_ang"],
                batch["pair_i_rad"], batch["pair_j_rad"],
                batch["pair_i_ang"], batch["pair_j_ang"],
                batch["atom_types"], batch["N"],
                batch["struct_idx"], batch["num_structures"],
                need_forces=need_forces, need_virial=need_virial,
                backend=backend, spins=batch.get("spins"),
                need_mforce=need_mforce, need_tau=need_tau,
                rij_spin=batch.get("rij_spin"),
                pi_spin=batch.get("pair_i_spin"),
                pj_spin=batch.get("pair_j_spin"))
        batch["need_mforce"] = need_mforce
        batch["need_tau"] = need_tau
        return self._compute_cached(
            batch, need_forces=need_forces, need_virial=need_virial,
            backend=backend)

from .train import (
    _BANNER, _AUTHOR,
    _backend_info, StreamDataStore, iter_collated,
    format_config_summary,
    preprocess_structures, compute_max_neighbors,
    fit_energy_baseline_frames, spin_curriculum_scale,
    spin_curriculum_end,
    prepare_spin3_curriculum_clip, restore_spin3_curriculum_clip,
    refresh_order3_q_scaler, reset_order3_optimizer_state,
    virial6_for_loss,
    grouped_validation_indices, validate_spin_frames,
    validate_response_group_disjoint, build_data_lineage,
    load_response_dataset,
    validate_resume_data_lineage,
    _save_checkpoint, _load_checkpoint,
    _restore_scheduler_state,
    _trim_loss_log, _accumulate_true_loss_sums,
    _load_finetune_weights,
    evaluate_generator_curves,
    _make_optimizer, _make_lr_scheduler, _scheduler_step,
    _compile_check, _quiet_compile_logs, _maybe_enable_tf32,
    _clean_warning_format, _default_alloc_conf,
)


# ---------------------------------------------------------------------------
# Sharded q_scaler
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_q_scaler_sharded(model, data_store, batch_size=1000,
                               backend="loop", gpumd_init=False,
                               return_rms=False, local_real_flags=None):
    """Compute descriptor min/max over the local shard, then all-reduce.

    Uses cached basis from data_store (no Chebyshev recompute). batch_size
    here is q-scaler-only (independent from training batch), default 1000.

    ``gpumd_init`` mirrors train.compute_q_scaler: False (default) uses the
    model's actual init coefficients (self-consistent); True forces all
    coefficients to 1.0, matching GPUMD's generation-0 q_scaler.

    ``local_real_flags`` excludes the duplicate frames used only to equalize
    DDP step counts.  Min/max are duplicate-invariant, but the spin3 RMS is
    not, so padded atoms must never enter any scaler statistic.
    """
    model.eval()
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    q_min = torch.full((model.dim,), float("inf"), dtype=dtype, device=dev)
    q_max = torch.full((model.dim,), float("-inf"), dtype=dtype, device=dev)
    q_square_sum = (
        torch.zeros(model.dim, dtype=dtype, device=dev)
        if return_rms else None
    )
    q_count = torch.zeros((), dtype=torch.int64, device=dev)
    if (local_real_flags is not None
            and len(local_real_flags) != data_store.n):
        raise ValueError(
            "local_real_flags must contain one value per local frame"
        )

    if gpumd_init:
        c2 = torch.ones_like(model.c_param_2)
        c3 = (torch.ones_like(model.c_param_3)
              if model.c_param_3 is not None else None)
        c_spin = (torch.ones_like(model.c_param_spin)
                  if model.c_param_spin is not None else None)
    else:
        c2, c3 = model.c_param_2, model.c_param_3
        c_spin = model.c_param_spin

    for start in range(0, data_store.n, batch_size):
        end = min(start + batch_size, data_store.n)
        frame_indices = list(range(start, end))
        batch = data_store.collate(frame_indices)
        q = ops.compute_descriptors_cached(
            batch["fk_rad"], batch["fk_ang"], batch["blm"],
            batch["pair_i_rad"], batch["pair_j_rad"],
            batch["pair_i_ang"], batch["pair_j_ang"],
            batch["atom_types"], batch["N"],
            c2, c3,
            model.n_max_radial, model.n_max_angular,
            model.l_max_3b,
            model.has_q_222, model.has_q_1111, model.has_q_112,
            model.num_lm, model._c3b, model._c4b, model._c5b,
            model._c4b2,
            dtype, dev,
            backend=backend,
            has_q_123=model.has_q_123, has_q_233=model.has_q_233,
            has_q_134=model.has_q_134,
        )
        if model.spin_descriptor is not None:
            q_spin = model.spin_descriptor(
                batch["rij_spin"], batch["pair_i_spin"],
                batch["pair_j_spin"], batch["spins"], batch["N"],
                batch["atom_types"], c_spin, None,
                model.spin_dof_type_active,
                model.spin_env_type_active,
            )
            q = torch.cat((q, q_spin), dim=-1)
        if local_real_flags is not None:
            frame_is_real = torch.as_tensor(
                local_real_flags[start:end], dtype=torch.bool, device=dev
            )
            atom_is_real = frame_is_real.index_select(0, batch["struct_idx"])
            q = q[atom_is_real]
        if q.shape[0] == 0:
            continue
        q_min = torch.min(q_min, q.min(0).values)
        q_max = torch.max(q_max, q.max(0).values)
        if return_rms:
            q_square_sum.add_(q.square().sum(0))
            q_count.add_(q.shape[0])

    dist.all_reduce(q_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(q_max, op=dist.ReduceOp.MAX)
    if return_rms:
        dist.all_reduce(q_square_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(q_count, op=dist.ReduceOp.SUM)

    model.train()
    if return_rms:
        if int(q_count.item()) == 0:
            raise ValueError("cannot compute q RMS from an empty dataset")
        return q_min, q_max, (q_square_sum / q_count).sqrt()
    return q_min, q_max


# ---------------------------------------------------------------------------
# Sharded training entry point
# ---------------------------------------------------------------------------

def train_nep_sharded(
    config_file: str,
    data_file: str,
    output_dir: str = ".",
    precision: str = "float32",
    use_autograd_forces: bool = False,
    use_swa: bool = False,
    swa_start: int = None,
    use_compile: bool = None,
    print_interval: int = 1,
    restart: bool = True,
    checkpoint_interval: int = 100,
    prediction_interval: int = 100,
    finetune_from: str = None,
    resume_from: str = None,
    recompute_q_scaler: bool = False,
    slim_types: bool = False,
    energy_key: str = "energy",
    use_gpumd_qscaler: bool = False,
    run_seed: int = None,
    valid_file: str = None,
    valid_ratio: float = None,
    valid_strategy: str = "stratified",
):
    """Data-sharded NEP training.  Launch via torchrun (or any launcher that
    sets RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT).

    All hyperparameters (epoch / batch / eval_batch / lr /
    lambda_e,f,v,m,tau / stage2* / ...) come from ``config_file`` — see
    README for the nep.in reference.

    Each rank's preprocessed device data store contains only its shard, so
    device data memory scales as 1/world_size. The input XYZ is currently
    parsed on every rank before non-local host frames are released. Gradients
    are all-reduced by DDP; q_scaler and epoch metrics are all-reduced
    explicitly.

        torchrun --nproc_per_node=N run_train.py

    Parameters mirror ``train_nep`` (same restart semantics: resume is an
    exact continuation — lr/optimizer/scheduler/SWA/best gates from the
    checkpoint, nep.in's lr never re-applied on resume; ``resume_from``
    continues from a specific checkpoint such as ``checkpoint_stage1.pt``;
    ``finetune_from`` keeps the source model's q_scaler unless
    ``recompute_q_scaler=True``). Runtime-exclusive differences are the DDP
    launch, CUDA/gloo backend auto-select, and distributed aggregation of
    q_scaler / epoch metrics / the frozen-weight best evaluation.

    ``run_seed`` mirrors ``train_nep``: None -> a fresh random seed each run
    (rank 0 draws it and broadcasts, so all replicas agree); pass an int for a
    fully reproducible run. It seeds the weight init and the per-epoch shuffle,
    and is saved into / restored from the checkpoint for exact resumption.

    ``valid_file`` / ``valid_ratio`` mirror ``train_nep``: a validation set
    (own .xyz, or a run_seed-drawn holdout fraction of data_file) evaluated
    every epoch — nep_best and the plateau LR schedule then follow the
    validation loss. The validation frames are sharded across ranks like the
    training frames; the error sums are all-reduced, so every rank sees the
    identical validation loss (schedulers stay in lock-step).

    Each rank keeps its shard in host memory and streams only the current
    batch to its GPU (basis computed on the fly, CPU assembly prefetched
    one batch ahead) — per-GPU memory scales with ``batch``, not shard
    size. DDP collectives are untouched: same step count per rank,
    gradients all-reduced as usual.
    """
    _clean_warning_format()
    _default_alloc_conf()
    # Quiet CPU-thread oversubscription for multi-process runs (torchrun
    # sets this itself and warns when unset; other launchers may not).
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # ---- Distributed init ------------------------------------------------
    # Wrap local_rank around the number of visible GPUs — lets several
    # processes share one GPU (useful for locally simulating multi-rank DDP).
    # NCCL refuses to share a GPU across ranks, so fall back to gloo (slower
    # but correct) when the ranks on this node outnumber its GPUs.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cuda_available = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if cuda_available else 0
    if cuda_available:
        gpu_id = local_rank % max(1, n_gpus)
        torch.cuda.set_device(gpu_id)
        dev = torch.device(f"cuda:{gpu_id}")
    else:
        gpu_id = None
        dev = torch.device("cpu")
    if not dist.is_initialized():
        world_size_env = int(os.environ.get("WORLD_SIZE", 1))
        # GPU sharing is a per-NODE question: on a multi-node run each node
        # hosts LOCAL_WORLD_SIZE ranks (torchrun sets it; srun-style
        # launchers expose SLURM_NTASKS_PER_NODE), and NCCL is fine as long
        # as no two ranks on one node share a GPU. Comparing the GLOBAL
        # world size against the local GPU count (the old heuristic) forced
        # every multi-node run onto gloo.
        local_world_env = int(
            os.environ.get("LOCAL_WORLD_SIZE",
                           os.environ.get("SLURM_NTASKS_PER_NODE",
                                          world_size_env)))
        ddp_backend = ("nccl" if cuda_available and local_world_env <= n_gpus
                       else "gloo")
        # Pass device_id so NCCL can bind the rank to its CUDA device
        # deterministically (silences "Guessing device ID based on global
        # rank" and the collective-context warnings, and prevents hangs on
        # heterogeneous rank->GPU mappings). gloo ignores device_id.
        dist.init_process_group(backend=ddp_backend,
                                device_id=dev if cuda_available else None)
        # Tear the group down once, at process exit — NOT at the end of this
        # function. That lets a single script call train_nep_sharded() more than
        # once (e.g. compile vs no-compile back to back): re-initialising a
        # torchrun process group after destroying it desynchronises the ranks
        # and fails NCCL eager-connect ("remote process exited / Connection
        # refused"). See the end-of-function barrier.
        _register_pg_atexit()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    is_main = rank == 0
    dtype = torch.float32 if precision == "float32" else torch.float64

    # ---- Resume target + master RNG seed -----------------------------------
    # Resolved BEFORE data loading: the validation split (valid_ratio) draws
    # from the seed, so on resume the checkpoint's saved seed must win here —
    # otherwise the resumed run would carve out a DIFFERENT validation subset
    # and leak old validation frames into training. Only the seed is peeked;
    # the full checkpoint restore happens after model/optimizer construction.
    ckpt_path = os.path.join(output_dir, "checkpoint.pt")
    if resume_from is not None and finetune_from is not None:
        raise ValueError("resume_from and finetune_from are mutually "
                         "exclusive: resume_from continues a run, "
                         "finetune_from starts a new one from weights")
    resume_ckpt = None
    if resume_from is not None:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"resume_from: {resume_from} not found")
        resume_ckpt = resume_from
    elif restart and os.path.exists(ckpt_path) and finetune_from is None:
        resume_ckpt = ckpt_path
    if resume_ckpt is not None:
        _saved_seed = torch.load(resume_ckpt, map_location="cpu",
                                 weights_only=False).get("run_seed")
        if _saved_seed is not None:
            run_seed = _saved_seed  # continue the original streams + split

    # rank 0 owns the seed (picks a random one when run_seed is None) and
    # broadcasts it so every replica seeds identically — the DDP wrapper later
    # broadcasts rank-0 weights anyway, but a shared seed also makes the
    # per-epoch shuffle streams reproducible for a given run_seed. Seeded here,
    # before model construction, so the NN weight init draws from it.
    if run_seed is None and is_main:
        import random
        run_seed = random.randrange(1, 2**31 - 1)
    seed_t = torch.tensor([run_seed if run_seed is not None else 0],
                          dtype=torch.long, device=dev)
    dist.broadcast(seed_t, src=0)
    run_seed = int(seed_t.item())
    torch.manual_seed(run_seed)

    # Only rank 0 logs. All app output already routes through _log (is_main-
    # guarded), but Python warnings (e.g. NEPModel's q_1111 redundancy notice)
    # fire on every rank — silence them on non-main ranks so the console shows
    # one copy, not world_size copies.
    if not is_main:
        warnings.filterwarnings("ignore")

    # ---- Logging (rank 0 only) -------------------------------------------
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
    dist.barrier()

    _out_log_file = (open(os.path.join(output_dir, "output.log"),
                          "a" if restart else "w")
                     if is_main else None)

    def _log(msg=""):
        if not is_main:
            return
        # flush=True so sbatch / piped stdout doesn't block-buffer log lines
        # (default block buffering hides progress between long-running steps)
        print(msg, flush=True)
        _out_log_file.write(msg + "\n")
        _out_log_file.flush()

    # ---- Banner ----------------------------------------------------------
    total_t0 = time.time()
    _log(_BANNER.rstrip())
    _log(f"   torchnep  v{__version__}   author: {_AUTHOR}")
    _log(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log("")
    for line in _backend_info(dev, world_size):
        _log(line)
    _log(f"Precision: {precision}")
    _maybe_enable_tf32(dev, dtype, _log)   # sets the flag on every rank
    _log(f"Mode     : data-sharded DDP ({world_size} ranks, "
         f"each holds 1/{world_size} of structures)")
    _log("")

    # ---- Config (all hyperparameters from nep.in) -----------------------
    orig_config = parse_nep_in(config_file)
    if orig_config.get("response_refine", False):
        raise ValueError("response_refine is single-device L-BFGS; use train_nep, not DDP")
    use_gpumd_qscaler = resolve_gpumd_qscaler(
        orig_config.get("spin_mode", 0), use_gpumd_qscaler
    )
    if is_main:
        for line in format_config_summary(orig_config):
            _log(line)
        _log("")
    config = orig_config
    num_epochs         = config["num_epochs"]
    batch_size         = config["batch_size"]
    eval_batch_size    = config["eval_batch_size"]
    lr                 = config["lr"]
    stop_lr            = config["stop_lr"]
    scheduler_patience = config["scheduler_patience"]
    early_stop         = int(config.get("early_stop", 0))  # 0 = off
    scheduler_factor   = config["scheduler_factor"]
    lr_scheduler_mode  = config["lr_scheduler"]     # "plateau" | "step"
    max_grad_norm      = config["max_grad_norm"]
    pref_e             = config["lambda_e"]
    pref_f             = config["lambda_f"]
    pref_v             = config["lambda_v"]
    pref_m             = config["lambda_m"]
    pref_tau           = config["lambda_tau"]
    stage2             = config["stage2"]
    start_stage2       = config.get("start_stage2")
    stage2_lr          = config["stage2_lr"]
    stage2_pref_e      = config["stage2_pref_e"]
    stage2_pref_f      = config["stage2_pref_f"]
    stage2_pref_v      = config["stage2_pref_v"]
    stage2_pref_m      = config["stage2_pref_m"]
    stage2_pref_tau    = config["stage2_pref_tau"]

    # ---- Data: parse globally, retain 1/world_size in each device store ---
    _log("Data")
    _log("----")
    lineage_holder = [
        build_data_lineage(data_file, valid_file, valid_ratio, energy_key,
                           config.get("lambda_spin_response", 0.0))
        if is_main else None
    ]
    dist.broadcast_object_list(lineage_holder, src=0)
    initial_data_lineage = lineage_holder[0]

    # Streamed shard loading for large files: rank 0 makes one index pass
    # (byte offset + natoms per frame, no parsing), broadcasts the offsets,
    # and every rank then seek-reads ONLY its own frames. This replaces the
    # old "every rank parses the whole file" path, whose readlines() line
    # list alone costs tens of GB per rank on a multi-10-GB file.
    _STREAM_THRESHOLD = int(os.environ.get("TORCHNEP_STREAM_THRESHOLD",
                                           2 * 1024 ** 3))
    streamed = os.path.getsize(data_file) >= _STREAM_THRESHOLD
    _stream_valid_local = None
    _slim_keep = None
    if streamed:
        from .data import index_xyz, read_xyz_at
        bcast_dev = dev if dist.get_backend() == "nccl" else torch.device("cpu")
        if is_main:
            t0 = time.time()
            offs_np, _nat_np = index_xyz(data_file)
            _log(f"  indexed {len(offs_np)} structures from {data_file} in "
                 f"{time.time() - t0:.1f}s (streamed shard loading, "
                 f"{os.path.getsize(data_file) / 2**30:.1f} GiB; "
                 f"energy label: {energy_key})")
            n_all_t = torch.tensor([len(offs_np)], dtype=torch.long,
                                   device=bcast_dev)
        else:
            n_all_t = torch.zeros(1, dtype=torch.long, device=bcast_dev)
        dist.broadcast(n_all_t, 0)
        n_all = int(n_all_t.item())
        offs_t = torch.empty(n_all, dtype=torch.long, device=bcast_dev)
        if is_main:
            offs_t.copy_(torch.from_numpy(offs_np).to(bcast_dev))
        dist.broadcast(offs_t, 0)
        offs_all = offs_t.cpu().numpy()
        del offs_t

        if valid_file is not None and valid_ratio is not None:
            raise ValueError(
                "valid_file and valid_ratio are mutually exclusive")
        val_idx_s = None
        if valid_ratio is not None:
            if valid_strategy == "stratified":
                _log("  streamed loader: the stratified valid split needs "
                     "full-file metadata — using the random strategy "
                     "(same run_seed) for this large file")
            train_idx_s, val_idx_s = valid_split_indices(
                n_all, valid_ratio, run_seed)
            _log(f"  valid_ratio={valid_ratio}: held out {len(val_idx_s)} "
                 f"frames for validation, {len(train_idx_s)} remain for "
                 f"training (split drawn from run_seed)")
        else:
            train_idx_s = np.arange(n_all)
        train_offs = offs_all[np.asarray(train_idx_s, dtype=np.int64)]
        n_total = len(train_offs)

        shuffle_g = torch.Generator()
        shuffle_g.manual_seed(0)
        global_perm = torch.randperm(n_total, generator=shuffle_g).tolist()
        n_local = (n_total + world_size - 1) // world_size  # ceil
        pad = n_local * world_size - n_total
        real_flags = [True] * n_total
        if pad:
            global_perm = global_perm + global_perm[:pad]
            real_flags.extend([False] * pad)
        local_global_idx = global_perm[rank * n_local : (rank + 1) * n_local]
        local_real_flags = real_flags[rank * n_local : (rank + 1) * n_local]
        t0 = time.time()
        frames = None
        local_frames = read_xyz_at(
            data_file, train_offs[np.asarray(local_global_idx)],
            energy_key=energy_key)
        _log(f"  parsed local shard ({len(local_frames)} frames) in "
             f"{time.time() - t0:.1f}s")

        valid_frames = None
        n_valid_total = 0
        valid_local_global_idx = None
        if val_idx_s is not None and len(val_idx_s):
            n_valid_total = len(val_idx_s)
            voffs = offs_all[np.asarray(val_idx_s, dtype=np.int64)]
            vsh_g = torch.Generator()
            vsh_g.manual_seed(0)
            vperm_sh = torch.randperm(n_valid_total,
                                      generator=vsh_g).tolist()
            n_vlocal = (n_valid_total + world_size - 1) // world_size
            vpad = n_vlocal * world_size - n_valid_total
            vreal_flags = [True] * n_valid_total
            if vpad:
                vperm_sh = vperm_sh + vperm_sh[:vpad]
                vreal_flags.extend([False] * vpad)
            valid_local_global_idx = \
                vperm_sh[rank * n_vlocal : (rank + 1) * n_vlocal]
            valid_local_real_flags = \
                vreal_flags[rank * n_vlocal : (rank + 1) * n_vlocal]
            _stream_valid_local = read_xyz_at(
                data_file, voffs[np.asarray(valid_local_global_idx)],
                energy_key=energy_key)

        if slim_types:
            # Global species union from the local shards (bitmask MAX).
            present = torch.zeros(len(orig_config["type_names"]),
                                  dtype=torch.long, device=dev)
            name_pos = {t: i for i, t in
                        enumerate(orig_config["type_names"])}
            for f_ in local_frames + (_stream_valid_local or []):
                for s in set(f_["species"]):
                    present[name_pos[s]] = 1
            dist.all_reduce(present, op=dist.ReduceOp.MAX)
            keep = [t for i, t in enumerate(orig_config["type_names"])
                    if int(present[i])]
            removed = [t for t in orig_config["type_names"]
                       if t not in keep]
            if removed:
                _slim_keep = keep
                config = dict(orig_config)
                config["type_names"] = keep
                config["num_types"] = len(keep)
                for key in ("spin_dof_type", "spin_env_type"):
                    if config.get(key) is not None:
                        config[key] = [
                            name for name in config[key] if name in keep
                        ]
                if (config.get("spin_mode", 0)
                        and config.get("spin_dof_type") == []):
                    raise ValueError(
                        "slim_types removed every active spin_dof_type")
                _log(f"  slim_types: {orig_config['type_names']} -> {keep} "
                     f"(removing: {removed})")
            else:
                _log("  slim_types: all types present in data, "
                     "nothing to remove")

    if not streamed:
        frames = read_xyz(data_file, energy_key=energy_key)
        _log(f"  read {len(frames)} structures from {data_file} "
             f"(energy label: {energy_key})")

    # ---- Validation set (same semantics as train_nep) ---------------------
    # The ratio split draws from run_seed via a dedicated generator —
    # identical on every rank (seed was broadcast above), so all ranks agree
    # on the partition. Sorted indices keep *_test.out rows in input order.
    if valid_file is not None and valid_ratio is not None:
        raise ValueError("valid_file and valid_ratio are mutually exclusive")
    if not streamed:
        valid_frames = None
    if valid_file is not None:
        if os.path.samefile(data_file, valid_file):
            raise ValueError("valid_file must not be the training data file")
        valid_frames = read_xyz(valid_file, energy_key=energy_key)
        _log(f"  read {len(valid_frames)} validation structures "
             f"from {valid_file}")
    elif not streamed and valid_ratio is not None:
        if not 0.0 < valid_ratio < 1.0:
            raise ValueError(f"valid_ratio must be in (0, 1), "
                             f"got {valid_ratio}")
        has_response_groups = any("response_group" in f for f in frames)
        if config.get("spin_mode", 0) in (2, 3) and has_response_groups:
            val_idx = grouped_validation_indices(frames, valid_ratio, run_seed)
            val_set = set(val_idx)
            train_idx = [i for i in range(len(frames)) if i not in val_set]
            _log(f"  valid_ratio={valid_ratio}: held out {len(val_idx)} "
                 f"frames, {len(train_idx)} remain for training "
                 f"(whole response groups, split drawn from run_seed)")
        elif valid_strategy == "stratified":
            metas = [(f["natoms"], f["species"]) for f in frames]
            train_idx, val_idx, st = stratified_split_indices(
                metas, valid_ratio, run_seed)
            _log(f"  valid_ratio={valid_ratio} (stratified by composition x "
                 f"cell size): held out {len(val_idx)} frames, "
                 f"{len(train_idx)} remain (split drawn from run_seed)")
        elif valid_strategy == "random":
            train_idx, val_idx = valid_split_indices(
                len(frames), valid_ratio, run_seed)
            _log(f"  valid_ratio={valid_ratio}: held out {len(val_idx)} "
                 f"frames for validation, {len(train_idx)} remain for "
                 f"training (split drawn from run_seed)")
        else:
            raise ValueError(f"unknown valid_strategy: {valid_strategy!r}")
        valid_frames = [frames[i] for i in val_idx]
        frames = [frames[i] for i in train_idx]
    contract_train_frames = local_frames if streamed else frames
    contract_valid_frames = (
        _stream_valid_local if streamed and _stream_valid_local is not None
        else valid_frames)
    if config.get("spin_mode", 0):
        validate_spin_frames(contract_train_frames, "training")
        if contract_valid_frames is not None:
            validate_spin_frames(contract_valid_frames, "validation")
    if valid_file is not None and not streamed:
        validate_response_group_disjoint(frames, valid_frames)
    response_loss_weight = float(config.get("lambda_spin_response", 0.0))
    # The small response set is replicated; ordinary data remains sharded.
    # Check against full validation on every rank, including streamed loading.
    response_validation = (
        read_xyz(valid_file, energy_key=energy_key)
        if response_loss_weight > 0 and streamed and valid_file is not None
        else valid_frames)
    response_frames, valid_response_frames = load_response_dataset(
        config, data_file, response_validation, energy_key)
    response_enabled = bool(response_frames)
    if valid_response_frames and not response_enabled:
        raise ValueError(
            "validation contains rotation response groups but training has "
            "none; keep at least one complete response scan in each split")
    if response_enabled and not config["stage2"]:
        raise ValueError(
            "rotation response groups require stage2=1; the generator-curve "
            "objective is intentionally activated only in Stage 2")
    if response_enabled and response_validation is not None \
            and not valid_response_frames:
        raise ValueError(
            "training contains rotation response groups but validation has "
            "none; provide a group-disjoint validation response scan so "
            "nep_best is selected on the response objective")
    if response_enabled:
        _log(f"  generator curves: {len(response_frames)} training frames"
             + (f", {len(valid_response_frames)} validation frames"
                if valid_frames is not None else ""))
        _log(f"  generator objective: {GENERATOR_CURVE_SCHEMA}; "
             "one replicated complete group per ordinary optimizer step")
    if not streamed:
        n_total = len(frames)

    # slim_types: all ranks agree on which types to keep (deterministic scan)
    if not streamed and slim_types:
        seen_species = set(s for f in frames for s in f["species"])
        if valid_frames is not None:
            seen_species |= set(s for f in valid_frames
                                for s in f["species"])
        keep = [t for t in orig_config["type_names"] if t in seen_species]
        removed = [t for t in orig_config["type_names"] if t not in keep]
        if removed:
            _slim_keep = keep
            config = dict(orig_config)
            config["type_names"] = keep
            config["num_types"] = len(keep)
            for key in ("spin_dof_type", "spin_env_type"):
                if config.get(key) is not None:
                    config[key] = [
                        name for name in config[key] if name in keep
                    ]
            if (config.get("spin_mode", 0)
                    and config.get("spin_dof_type") == []):
                raise ValueError(
                    "slim_types removed every active spin_dof_type")
            _log(f"  slim_types: {orig_config['type_names']} -> {keep} "
                 f"(removing: {removed})")
        else:
            _log("  slim_types: all types present in data, nothing to remove")

    energy_baseline_values = None
    if config.get("spin_mode", 0):
        if not streamed:
            energy_baseline_values = fit_energy_baseline_frames(
                frames, config["type_names"])
        else:
            type_index = {
                name: index for index, name in enumerate(config["type_names"])}
            ntype = len(type_index)
            ata = np.zeros((ntype, ntype), dtype=np.float64)
            aty = np.zeros(ntype, dtype=np.float64)
            for frame, is_real in zip(local_frames, local_real_flags):
                if not is_real or "energy" not in frame:
                    continue
                x = np.zeros(ntype, dtype=np.float64)
                for species in frame["species"]:
                    x[type_index[species]] += 1.0
                x /= len(frame["species"])
                y = float(frame["energy"]) / len(frame["species"])
                ata += np.outer(x, x)
                aty += x * y
            stats = torch.from_numpy(
                np.concatenate([ata.ravel(), aty])).to(dev)
            dist.all_reduce(stats)
            stats_np = stats.cpu().numpy()
            ata = stats_np[:ntype * ntype].reshape(ntype, ntype)
            aty = stats_np[ntype * ntype:]
            energy_baseline_values = np.linalg.lstsq(
                ata, aty, rcond=None)[0]

    # Random sharding: shuffle globally with a fixed seed (identical on
    # every rank -> all ranks agree on the partition), then give rank r the
    # r-th equal slice. To keep step counts identical across ranks (DDP
    # collectives require lock-step iteration) AND keep every frame in the
    # training/predict set, we pad the perm with duplicates from the head
    # until it divides evenly. Padding copies keep ranks in lock-step, but
    # their label masks are disabled below so the scientific objective sees
    # every original frame exactly once.
    if n_total == 0:
        raise ValueError("training data contains no frames")
    if not streamed:
        shuffle_g = torch.Generator()
        shuffle_g.manual_seed(0)
        global_perm = torch.randperm(n_total, generator=shuffle_g).tolist()
        n_local = (n_total + world_size - 1) // world_size  # ceil
        pad = n_local * world_size - n_total
        real_flags = [True] * n_total
        if pad:
            global_perm = global_perm + global_perm[:pad]
            real_flags.extend([False] * pad)
        local_global_idx = global_perm[rank * n_local : (rank + 1) * n_local]
        local_real_flags = real_flags[rank * n_local : (rank + 1) * n_local]
        local_frames = [frames[i] for i in local_global_idx]
    pad_note = (f", {pad} frame(s) duplicated for even split"
                if pad else "")
    _log(f"  sharded across {world_size} ranks: "
         f"{len(local_frames)} frames per rank{pad_note}")

    t0 = time.time()
    np_dtype = np.float64 if precision == "float64" else np.float32
    structures = preprocess_structures(local_frames, config, np_dtype)
    response_structures = (
        preprocess_structures(response_frames, config, np_dtype)
        if response_enabled else None
    )
    for structure, is_real in zip(structures, local_real_flags):
        if not is_real:
            for label in ("energy", "forces", "virial", "mforces", "tau"):
                structure.pop(label, None)
    _log(f"  built neighbor lists (local shard) in {time.time() - t0:.1f}s")

    # Validation frames: sharded across ranks with the same padded-perm
    # scheme as the training frames. Padding labels are disabled, so frozen
    # validation metrics also contain every original frame exactly once.
    structures_v = None
    response_structures_v = None
    if not streamed or valid_frames is not None:
        valid_local_global_idx = None
        n_valid_total = 0
    if _stream_valid_local is not None:
        structures_v = preprocess_structures(_stream_valid_local, config,
                                             np_dtype)
        for structure, is_real in zip(
                structures_v, valid_local_real_flags):
            if not is_real:
                for label in (
                    "energy", "forces", "virial", "mforces", "tau"
                ):
                    structure.pop(label, None)
        del _stream_valid_local
    if valid_frames is not None:
        n_valid_total = len(valid_frames)
        vsh_g = torch.Generator()
        vsh_g.manual_seed(0)
        vperm_sh = torch.randperm(n_valid_total, generator=vsh_g).tolist()
        n_vlocal = (n_valid_total + world_size - 1) // world_size  # ceil
        vpad = n_vlocal * world_size - n_valid_total
        vreal_flags = [True] * n_valid_total
        if vpad:
            vperm_sh.extend(
                vperm_sh[index % n_valid_total]
                for index in range(vpad))
            vreal_flags.extend([False] * vpad)
        valid_local_global_idx = \
            vperm_sh[rank * n_vlocal : (rank + 1) * n_vlocal]
        valid_local_real_flags = \
            vreal_flags[rank * n_vlocal : (rank + 1) * n_vlocal]
        structures_v = preprocess_structures(
            [valid_frames[i] for i in valid_local_global_idx],
            config, np_dtype)
        for structure, is_real in zip(
                structures_v, valid_local_real_flags):
            if not is_real:
                for label in (
                    "energy", "forces", "virial", "mforces", "tau"
                ):
                    structure.pop(label, None)
        del valid_frames

    if valid_response_frames:
        response_structures_v = preprocess_structures(
            valid_response_frames, config, np_dtype)

    # Every rank has to parse the full XYZ to agree on the deterministic
    # split, but only the preprocessed local shard is needed from here on.
    # Drop the full-frame and local-frame dictionaries before allocating the
    # device store so steady-state host memory does not retain both copies.
    del frames, local_frames

    local_max_rad, local_max_ang = compute_max_neighbors(structures)
    if structures_v is not None:
        # nep.txt records max neighbor counts for MD buffer sizing — cover
        # the validation frames too (folded into the same all-reduce).
        v_max_rad, v_max_ang = compute_max_neighbors(structures_v)
        local_max_rad = max(local_max_rad, v_max_rad)
        local_max_ang = max(local_max_ang, v_max_ang)
    nn_t = torch.tensor([local_max_rad, local_max_ang], dtype=torch.long,
                        device=dev)
    dist.all_reduce(nn_t, op=dist.ReduceOp.MAX)
    max_NN_rad, max_NN_ang = int(nn_t[0].item()), int(nn_t[1].item())

    t0 = time.time()
    data_store = StreamDataStore(structures, dev, dtype, config=config)
    del structures
    response_store = (
        StreamDataStore(response_structures, dev, dtype, config=config)
        if response_structures is not None else None
    )
    del response_structures
    configure_response_store(response_store, config)
    valid_store = None
    valid_response_store = None
    if structures_v is not None:
        valid_store = StreamDataStore(structures_v, dev, dtype, config=config)
        del structures_v
    if response_structures_v is not None:
        valid_response_store = StreamDataStore(
            response_structures_v, dev, dtype, config=config)
        del response_structures_v
        configure_response_store(valid_response_store, config)
    if cuda_available:
        torch.cuda.synchronize()
    _log(f"  data store ready: shard in host memory, batches streamed "
         f"to {dev} ({time.time() - t0:.1f}s)")

    # Aggregate data counts across all ranks for the banner
    counts_t = torch.tensor(
        [data_store.n, data_store.n_energy,
         data_store.n_forces, data_store.n_virial,
         data_store.n_mforces, data_store.n_tau],
        dtype=torch.long, device=dev)
    dist.all_reduce(counts_t)
    g_n, g_ne, g_nf, g_nv, g_nm, g_nt = counts_t.tolist()
    coverage = f"  coverage (global): {g_ne} E / {g_nf} F / {g_nv} V"
    if orig_config.get("spin_mode", 0):
        coverage += f" / {g_nm} M / {g_nt} T"
    _log(coverage)
    if valid_store is not None:
        vcounts_t = torch.tensor(
            [valid_store.n_energy, valid_store.n_forces,
             valid_store.n_virial, valid_store.n_mforces,
             valid_store.n_tau, int(valid_store.has_forces),
             int(valid_store.has_virial), int(valid_store.has_mforces),
             int(valid_store.has_tau)],
            dtype=torch.long, device=dev)
        dist.all_reduce(vcounts_t)
        gv_ne, gv_nf, gv_nv, gv_nm, gv_nt = vcounts_t[:5].tolist()
        # Global channel flags for the per-epoch validation eval — per-rank
        # shards may lack a channel that other shards have.
        valid_has_forces = bool(vcounts_t[5].item() > 0)
        valid_has_virial = bool(vcounts_t[6].item() > 0)
        valid_has_mforces = bool(vcounts_t[7].item() > 0)
        valid_has_tau = bool(vcounts_t[8].item() > 0)
        valid_coverage = (
            f"  validation coverage (global, incl. padding): "
            f"{gv_ne} E / {gv_nf} F / {gv_nv} V")
        if orig_config.get("spin_mode", 0):
            valid_coverage += f" / {gv_nm} M / {gv_nt} T"
        _log(valid_coverage)
    _log("")

    # ---- Model -----------------------------------------------------------
    _log("Model")
    _log("-----")
    _log(f"  run_seed: {run_seed}")
    model = NEPModel(config).to(dtype).to(dev)
    if model.spin_mode in (2, 3) and not use_autograd_forces:
        use_autograd_forces = True
        _log(f"  spin_mode {model.spin_mode}: using the full autograd "
             "E/F/V/MF/Tau path")
    if energy_baseline_values is not None:
        baseline = torch.as_tensor(
            energy_baseline_values, dtype=dtype, device=dev
        )
        dist.broadcast(baseline, src=0)
        model.set_energy_baseline(baseline)

    # use_gpumd_qscaler: reproduce GPUMD-compatible SNES initialization
    # (descriptor coeffs and NN weights) plus the c=1 q_scaler.
    # Re-init on rank 0 then broadcast so every replica starts identical;
    # skipped under finetune_from (keep the loaded parameters).
    if use_gpumd_qscaler and finetune_from is None:
        with torch.no_grad():
            if rank == 0:
                gpumd_init_parameters(model)
            dist.broadcast(model.c_param_2.data, src=0)
            if model.c_param_3 is not None:
                dist.broadcast(model.c_param_3.data, src=0)
            for net in model.fitting_nets:
                dist.broadcast(net.w0.data, src=0)
                dist.broadcast(net.b0.data, src=0)
                dist.broadcast(net.w1.data, src=0)
        _log("  use_gpumd_qscaler: descriptor coeffs + NN weights re-init "
             "uniform(-1,1), q_scaler will use c=1 (GPUMD-consistent)")

    # b1 (global energy offset) is determined analytically (folded into the
    # training pass + the best-model eval), not by gradient descent — keep it
    # out of the optimizer (and out of DDP's gradient sync, since it gets no
    # grad).
    model.b1.requires_grad_(False)

    if finetune_from is not None:
        if _slim_keep is not None:
            full_model = NEPModel(orig_config).to(dtype).to(dev)
            _load_finetune_weights(full_model, finetune_from, dev)
            slimmed = slim_model(full_model, config["type_names"])
            model.load_state_dict(slimmed.state_dict())
            del full_model, slimmed
            _log(f"  fine-tuning from {finetune_from}  "
                 f"[{orig_config['num_types']} -> {config['num_types']} types]")
        else:
            _load_finetune_weights(model, finetune_from, dev)
            _log(f"  fine-tuning from {finetune_from}")
        _log(f"  {sum(p.numel() for p in model.parameters())} parameters, "
             f"dim={model.dim}, b1={model.b1.item():.4f}")
    else:
        # mean_epa: weighted average across all ranks
        baseline_np = (
            model.energy_baseline.detach().cpu().numpy()
            if model.energy_baseline is not None
            else np.zeros(model.num_types, dtype=np.float64)
        )
        local_epa_sum = sum(
            (
                data_store.energy[i]
                - float(data_store.type_counts[i] @ baseline_np)
            ) / data_store.natoms[i]
            for i in range(data_store.n) if data_store.has_energy_flag[i]
        )
        local_n_e = float(data_store.n_energy)
        epa_t = torch.tensor([local_epa_sum, local_n_e], device=dev, dtype=torch.float64)
        dist.all_reduce(epa_t)
        mean_epa = float(epa_t[0] / epa_t[1]) if epa_t[1] > 0 else 0.0
        with torch.no_grad():
            model.b1.fill_(-mean_epa)
        _log(f"  {sum(p.numel() for p in model.parameters())} parameters, "
             f"dim={model.dim}, b1 init={model.b1.item():.4f}")

    # Channel availability: OR across ranks.
    flags_t = torch.tensor(
        [int(data_store.has_forces), int(data_store.has_virial),
         int(data_store.has_mforces), int(data_store.has_tau)],
        dtype=torch.long, device=dev)
    dist.all_reduce(flags_t, op=dist.ReduceOp.MAX)
    global_has_forces = bool(flags_t[0].item())
    global_has_virial = bool(flags_t[1].item())
    global_has_mforces = bool(flags_t[2].item())
    global_has_tau = bool(flags_t[3].item())

    # Decide whether torch.compile will actually run BEFORE resolving the
    # backend — the backend choice depends on it. Only the analytical path is
    # compiled; the autograd path's create_graph=True double backward is
    # incompatible with compile, so it stays eager.
    # use_compile=None means AUTO: compile on GPU (where it is 3-5x),
    # stay eager on CPU (C++ Inductor warmup outweighs the gain for
    # typical CPU-sized runs). Explicit True/False always wins; every
    # failed capability check degrades to eager with a log line instead
    # of raising.
    if use_compile is None:
        use_compile = dev.type in ("cuda", "xpu")
    compile_on = False
    compile_msg = None
    if use_compile:
        ok, msg = _compile_check(dev)
        if not ok:
            compile_msg = f"  torch.compile: disabled — {msg}"
        elif use_autograd_forces and model.spin_descriptor is not None:
            compile_msg = ("  torch.compile: disabled for O/C custom "
                           "autograd sectors")
        elif use_autograd_forces:
            compile_on = True
            compile_msg = ("  torch.compile: enabled (make_fx autograd "
                           "force graph)")
        else:
            compile_on = True
            method = "analytical compute method"
            compile_msg = f"  torch.compile: enabled ({method})"

    # Backends are chosen automatically (see ops.resolve_backend):
    # ``backend`` is the eager choice (q_scaler / eval), ``train_backend``
    # adds the use_compile rule.
    from .ops import resolve_backend as _resolve_backend
    backend = _resolve_backend("auto", num_types=model.num_types,
                               device_type=dev.type)
    train_backend = _resolve_backend("auto", num_types=model.num_types,
                                     use_compile=compile_on,
                                     device_type=dev.type)
    force_str = "autograd" if use_autograd_forces else "analytical"
    if train_backend != backend:
        _log(f"  backend: {backend} (q_scaler) / {train_backend} (training), "
             f"forces: {force_str}")
    else:
        _log(f"  backend: {backend}, forces: {force_str}")

    if finetune_from is not None and not recompute_q_scaler:
        # The q_scaler is part of the potential definition: the loaded NN
        # weights were trained against it. Keep it (same as train_nep).
        _log("  q_scaler: kept from finetune source (not recomputed)")
    else:
        if finetune_from is not None:
            _log("  q_scaler: RECOMPUTED from the new dataset "
                 "(recompute_q_scaler=True) — the loaded weights will "
                 "see rescaled descriptors and must re-adapt")
        # q_scaler: local shard -> all_reduce. The pass uses the training
        # batch size so its transient GPU footprint stays batch-bound
        # (min/max is chunking-invariant — result unchanged).
        t_qs = time.time()
        q_stats = _compute_q_scaler_sharded(
            model, data_store, backend=backend,
            batch_size=batch_size,
            gpumd_init=use_gpumd_qscaler,
            return_rms=model.spin_mode == 3,
            local_real_flags=(
                local_real_flags if model.spin_mode == 3 else None
            ))
        model.set_q_scaler(*q_stats)
        if cuda_available:
            torch.cuda.synchronize()
        _log(f"  q_scaler in {time.time() - t_qs:.1f}s "
             f"(all-reduce across shards)")

    # Wrap in a shim whose forward calls compute_properties{_cached} — this
    # keeps the force/virial compute on DDP's forward path so the reducer can
    # arm backward all-reduce. See _NEPDDPShim docstring. torch.compile is
    # applied INSIDE the shim (to the bound analytical method), not to the
    # module — compiling the module only wraps NEPModel.forward(), which the
    # training loop never calls, so it would be a no-op (the bug this fixes).
    if compile_msg is not None:
        _log(compile_msg)
    if compile_on:
        _quiet_compile_logs()
        # Fuse the per-batch basis kernels too — same numerical status
        # as the compiled compute (Inductor-level ~1e-7 deviations).
        data_store.compile_basis()
        if valid_store is not None:
            valid_store.compile_basis()
    shim = _NEPDDPShim(model, use_compile=compile_on,
                       use_autograd_forces=use_autograd_forces)
    # All per-type nets are always touched in compute_properties_cached (dummy
    # pass for types absent in a given batch) so DDP sees every parameter in
    # every step — no need for find_unused_parameters, and no implicit grad
    # dilution for rare types.
    model = DDP(shim,
                device_ids=[gpu_id] if cuda_available else None,
                find_unused_parameters=False)
    # raw_model: unwrap DDP -> shim -> NEPModel (compile now lives on a bound
    # method inside the shim, so model.model is the plain NEPModel).
    _shim = model.module
    raw_model = _shim.model

    # b1 is analytically determined (not gradient-trained) — exclude it from
    # the optimizer, matching the single-GPU path.
    trainable_named = [(n, p) for n, p in raw_model.named_parameters()
                       if n != "b1" and p.requires_grad]
    # weight_decay > 0 switches to AdamW (decoupled decay) — see train_nep.
    weight_decay = config["weight_decay"]
    optimizer = _make_optimizer(trainable_named, lr, weight_decay)
    configure_joint_update(optimizer, config)
    if config["joint_rank4"]:
        _log("  joint_rank4: joint Adam displacement, rank 4, refresh <=24 steps")

    if stage2 and start_stage2 is None:
        start_stage2 = max(1, int(num_epochs * 0.5))
    if stage2 and not 1 <= start_stage2 <= num_epochs:
        raise ValueError(
            "start_stage2 must lie within the training run: "
            f"1 <= {start_stage2} <= {num_epochs}")
    curriculum_enabled = (
        raw_model.spin_curriculum and finetune_from is None
    )
    curriculum_end = spin_curriculum_end(
        raw_model.spin_mode, num_epochs, stage2, start_stage2
    )
    if curriculum_enabled and not 2 <= curriculum_end <= num_epochs:
        raise ValueError(
            "spin curriculum requires its full-O3 epoch to lie in the "
            f"training run: 2 <= {curriculum_end} <= {num_epochs}"
        )
    schedule_state = {
        "stage2_start": start_stage2 if stage2 else None,
        "curriculum_end": curriculum_end if curriculum_enabled else None,
        "curriculum_enabled": curriculum_enabled,
        "use_swa": bool(stage2 and use_swa),
    }
    if raw_model.spin_mode == 3:
        schedule_state["order3_scaler_refreshed"] = False

    # SWA window: average only the tail of the run (default: the last 100
    # epochs). Averaging the whole of stage 2 drags the energy back toward
    # mid-descent weights (E converges late); the tail is a converged
    # cloud, so averaging there is pure noise reduction.
    if swa_start is None:
        swa_start = max(1, num_epochs - 99)

    lr_scheduler = _make_lr_scheduler(
        optimizer, lr_scheduler_mode, scheduler_factor,
        scheduler_patience, stop_lr)

    def _loss_fn(pred, ref):
        return torch.mean((pred - ref) ** 2)

    def _evaluate_sharded(store, eval_bs, need_forces, need_virial,
                          need_mforce, need_tau, loss_weights,
                          fit_energy_offset=False):
        """Frozen-weight evaluation of one sharded store on all ranks."""
        sums = _accumulate_true_loss_sums(
            store, eval_bs, raw_model,
            raw_model.compute_properties, _shim._compute_cached,
            use_autograd_forces, train_backend,
            need_forces, need_virial, need_mforce, need_tau, dtype, dev)
        sums_t = torch.tensor(sums, device=dev, dtype=torch.float64)
        dist.all_reduce(sums_t)
        (s_le, s_lf, s_lv, s_ls, s_lm, s_lt,
         n_e, n_f, n_v, n_m, n_t, s_e_resid) = sums_t.tolist()
        delta = (
            s_e_resid / n_e if fit_energy_offset and n_e > 0 else 0.0)
        if fit_energy_offset and n_e > 0:
            with torch.no_grad():
                raw_model.b1.add_(delta)
        mse_e = max(0.0, s_le / max(n_e, 1.0) - delta * delta)
        mse_f = s_lf / max(n_f, 1.0)
        mse_v = s_lv / max(n_v, 1.0)
        mse_s = s_ls / max(n_v, 1.0)
        mse_m = s_lm / max(n_m, 1.0)
        mse_t = s_lt / max(n_t, 1.0)
        pe, pf, pv, pm, pt = loss_weights
        total = pe * mse_e + pf * mse_f + pv * mse_v + pm * mse_m + pt * mse_t
        return (total, np.sqrt(mse_e), np.sqrt(mse_f), np.sqrt(mse_v),
                np.sqrt(mse_s), np.sqrt(mse_m), np.sqrt(mse_t))

    swa_model = None
    stage2_scheduler = None
    if stage2:
        # Stage 2 may have its own decay schedule (same fallback as train.py)
        stage2_scheduler = _make_lr_scheduler(
            optimizer, lr_scheduler_mode,
            config.get("stage2_scheduler_factor", scheduler_factor),
            config.get("stage2_scheduler_patience", scheduler_patience),
            stop_lr)
        if use_swa:
            # EVERY rank maintains the average (DDP keeps raw_model
            # identical across ranks, so the averages are identical too).
            # Keeping it main-only deadlocked the tail: rank 0 entered the
            # SWA-b1 all_reduce while the other ranks skipped the block and
            # sat in the final predict's all_gather — collective sequences
            # diverged and both sides hit the NCCL watchdog timeout.
            swa_model = AveragedModel(raw_model)

    # Snapshot of current loss weights — saved in checkpoint so a restart can
    # detect that the user edited nep.in and reset best_loss (otherwise the
    # old-scale best_loss would keep the new run from ever saving a new best).
    cur_loss_weights = {
        "lambda_e": pref_e, "lambda_f": pref_f, "lambda_v": pref_v,
        "lambda_m": pref_m, "lambda_tau": pref_tau,
        "stage2_pref_e": stage2_pref_e, "stage2_pref_f": stage2_pref_f,
        "stage2_pref_v": stage2_pref_v, "stage2_pref_m": stage2_pref_m,
        "stage2_pref_tau": stage2_pref_tau,
        "lr_scheduler_mode": lr_scheduler_mode,
        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience,
        "stage2_scheduler_factor": config.get(
            "stage2_scheduler_factor", scheduler_factor),
        "stage2_scheduler_patience": config.get(
            "stage2_scheduler_patience", scheduler_patience),
        "stop_lr": stop_lr,
        "generator_curve_schema": (
            GENERATOR_CURVE_SCHEMA if response_enabled else None),
        "generator_curve_weight": (
            response_loss_weight if response_enabled else 0.0),
        "generator_curve_huber_delta": (
            GENERATOR_CURVE_HUBER_DELTA if response_enabled else 0.0),
        "generator_curve_mean_weight": (
            GENERATOR_CURVE_MEAN_WEIGHT if response_enabled else 0.0),
        "generator_curve_scale_floor": (
            GENERATOR_CURVE_SCALE_FLOOR if response_enabled else 0.0),
        "generator_curve_normalization_policy": "group_target_rms" if response_enabled else None,
    }
    if response_enabled and config.get("spin_response_target", "mforce") != "mforce":
        cur_loss_weights["response_target"] = config["spin_response_target"]
        cur_loss_weights["response_energy_scale"] = config["spin_response_energy_scale"]
    if raw_model.spin_mode == 3:
        cur_loss_weights["generator_curve_step_balance"] = (
            "exact-per-epoch" if response_enabled else None
        )

    # resume_ckpt / ckpt_path were resolved before data loading (the seed
    # peek); here the full training state is actually restored.
    start_epoch = 1
    best_loss = float("inf")
    best_true_loss = float("inf")
    best_valid_loss = float("inf")
    cur_valid_info = initial_data_lineage
    if valid_strategy != "random":
        cur_valid_info["valid_strategy"] = valid_strategy
    stage2_lr_applied = False  # tracks whether stage2 lr/reset has fired yet
    resume_early_stop_state = None
    if resume_ckpt is not None:
        # Every rank loads the same file; model weights go into raw_model
        # (the plain NEPModel — keeps checkpoints interchangeable with
        # single-GPU runs). Every rank also maintains and restores the same
        # SWA state so final collective evaluation stays in lockstep.
        info = _load_checkpoint(
            resume_ckpt, raw_model, optimizer, swa_model, dev)
        if weight_decay > 0:  # see train_nep: reapply after state load
            for _g in optimizer.param_groups:
                _g["weight_decay"] = weight_decay
        start_epoch = info["epoch"] + 1
        best_loss = info["best_loss"]
        best_true_loss = info["best_true_loss"]
        best_valid_loss = info["best_valid_loss"]
        # A checkpoint already in stage 2 pins the EFFECTIVE stage-2 start
        # (may be earlier than nep.in's after a stage-1 early-stop jump);
        # stage-1 checkpoints defer to nep.in — see train_nep.
        if stage2 and info["in_stage2"] and info.get("start_stage2") is not None:
            start_stage2 = info["start_stage2"]
        saved_valid_info = info.get("valid_info")
        validate_resume_data_lineage(
            saved_valid_info, cur_valid_info,
            require_versioned=raw_model.spin_descriptor is not None,
        )
        saved_schedule = info.get("schedule_state")
        if saved_schedule is not None:
            if raw_model.spin_mode == 3:
                schedule_state["order3_scaler_refreshed"] = bool(
                    saved_schedule.get("order3_scaler_refreshed", False)
                )
            saved_stage2_start = saved_schedule.get("stage2_start")
            saved_use_swa = bool(saved_schedule.get("use_swa", False))
            if info["in_stage2"]:
                if (not stage2
                        or saved_stage2_start != schedule_state["stage2_start"]):
                    raise ValueError(
                        "a Stage-2 checkpoint must resume with the same "
                        "stage2/start_stage2 schedule")
                if saved_use_swa != schedule_state["use_swa"]:
                    raise ValueError(
                        "a Stage-2 checkpoint must resume with the same "
                        "use_swa setting")
                if saved_use_swa and not info["has_swa_state"]:
                    raise ValueError(
                        "Stage-2 checkpoint declares SWA but has no swa_state")
            elif stage2 and start_stage2 < start_epoch:
                raise ValueError(
                    "start_stage2 is earlier than the resumed epoch; resume "
                    "from checkpoint_stage1.pt or choose a future boundary")
            saved_curriculum_end = saved_schedule.get("curriculum_end")
            saved_curriculum_enabled = bool(saved_schedule.get(
                "curriculum_enabled", saved_curriculum_end is not None))
            if not saved_curriculum_enabled:
                curriculum_enabled = False
                schedule_state["curriculum_enabled"] = False
                schedule_state["curriculum_end"] = None
            elif curriculum_enabled and saved_curriculum_end is not None:
                if info["epoch"] >= saved_curriculum_end:
                    curriculum_enabled = False
                    curriculum_end = saved_curriculum_end
                    schedule_state["curriculum_end"] = saved_curriculum_end
                elif curriculum_end != saved_curriculum_end:
                    raise ValueError(
                        "resume would change an unfinished O2-to-O3 "
                        "curriculum boundary; keep the original schedule or "
                        "start a new run")
        elif raw_model.spin_descriptor is not None:
            raise ValueError(
                "spin checkpoint is missing schedule_state and cannot be "
                "resumed with production curriculum semantics")
        if is_main and info["best_model_text"] is not None:
            with open(os.path.join(output_dir, "nep_best.txt"), "w") as stream:
                stream.write(info["best_model_text"])
        elif (raw_model.spin_descriptor is not None
              and np.isfinite(best_loss)
              and info["best_model_text"] is None):
            raise ValueError(
                "spin checkpoint is missing its historical best model")
        resume_early_stop_state = info.get("early_stop_state")
        _log(f"Resumed from {resume_ckpt}: epoch {start_epoch - 1}, "
             f"best_loss={best_loss:.4e}, run_seed={run_seed}")
        # Resume = exact continuation: lr (and the whole optimizer state)
        # comes from the checkpoint moment. nep.in's lr only applies to
        # fresh runs / finetunes — and stage2_lr at a fresh stage-2 entry.
        _log(f"  lr taken from checkpoint: "
             f"{optimizer.param_groups[0]['lr']:.2e} "
             f"(nep.in lr is not applied on resume)")
        saved_loss_weights = info["loss_weights"]
        training_config_changed = (
            saved_loss_weights is not None
            and saved_loss_weights != cur_loss_weights
        )
        if training_config_changed:
            _log("Loss weights changed since checkpoint was saved — "
                 "resetting best losses so the new scale can establish "
                 "a new best.")
            _log(f"  saved:   {saved_loss_weights}")
            _log(f"  current: {cur_loss_weights}")
            best_loss = float("inf")
            best_true_loss = float("inf")
            best_valid_loss = float("inf")
            resume_early_stop_state = None
            lr_scheduler = _make_lr_scheduler(
                optimizer, lr_scheduler_mode, scheduler_factor,
                scheduler_patience, stop_lr)
            if stage2:
                stage2_scheduler = _make_lr_scheduler(
                    optimizer, lr_scheduler_mode,
                    config.get("stage2_scheduler_factor", scheduler_factor),
                    config.get(
                        "stage2_scheduler_patience", scheduler_patience),
                    stop_lr)
            if info["in_stage2"] and swa_model is not None:
                swa_model = AveragedModel(raw_model)
                if is_main:
                    _log("  SWA accumulator reset for the new "
                         "objective/config")
            _log("  scheduler state reset for the new objective/config")
        else:
            _restore_scheduler_state(
                info, lr_scheduler, stage2_scheduler,
                require_versioned=raw_model.spin_descriptor is not None,
            )

    n_local = data_store.n
    # has_forces / has_virial are recomputed per-epoch inside the loop using
    # the CURRENT stage's weights — so a stage-1 weight of 0 can still enable
    # computation in stage 2 (and vice versa). Do NOT latch them here.

    loss_log = None
    if is_main:
        loss_log_path = os.path.join(output_dir, "loss.out")
        if start_epoch > 1:
            # Keep loss.out in 1:1 correspondence with the actual history.
            _trim_loss_log(loss_log_path, start_epoch - 1)
        write_header = (start_epoch == 1
                        or not os.path.exists(loss_log_path)
                        or os.path.getsize(loss_log_path) == 0)
        loss_log = open(loss_log_path, "w" if start_epoch == 1 else "a")
        if write_header:
            hdr = ("# epoch  loss  rmse_e(eV/atom)  rmse_f(eV/A)  "
                   "rmse_v(eV/atom)  rmse_stress(GPa)")
            if raw_model.spin_descriptor is not None:
                hdr += "  rmse_m(eV/muB)  rmse_tau(eV)"
            if response_enabled:
                hdr += "  generator_curve"
            if valid_store is not None:
                # GPUMD loss.out convention: train RMSEs, then test RMSEs.
                # (The loss column is then the VALIDATION loss — it is what
                # drives the scheduler and the best-model choice.)
                hdr += ("  rmse_e_test(eV/atom)  rmse_f_test(eV/A)  "
                        "rmse_v_test(eV/atom)  rmse_stress_test(GPa)")
                if raw_model.spin_descriptor is not None:
                    hdr += "  rmse_m_test(eV/muB)  rmse_tau_test(eV)"
                if response_enabled:
                    hdr += "  generator_curve_test"
            loss_log.write(hdr + "\n")

    # All training hyperparameters (lr/scheduler/loss weights/stage2 ...)
    # already printed by format_config_summary above; here we just announce
    # the runtime epoch range — different from `epoch` in nep.in when
    # resuming from a checkpoint.
    stage2_tag = (f", Stage 2 from epoch {start_stage2} "
                  f"(SWA={'on' if use_swa else 'off'})") if stage2 else ""
    curriculum_tag = (
        f", O2->O3 curriculum fully open at epoch {curriculum_end}"
        if curriculum_enabled else ""
    )
    _log("")
    if early_stop:
        monitored = "validation loss" if valid_store is not None else "train loss"
        if early_stop <= scheduler_patience:
            _log(f"WARNING: early_stop ({early_stop}) <= scheduler_patience "
                 f"({scheduler_patience}); use early_stop > scheduler_patience.")
        _log(f"Early stopping: stop if {monitored} does not improve for "
             f"{early_stop} epochs")
    _log(
        f"Training: epochs {start_epoch}..{num_epochs}"
        f"{stage2_tag}{curriculum_tag}"
    )
    _log("=" * 72)

    # In the last third of the run a candidate best is verified by a
    # frozen-weight evaluation (each rank evaluates its shard, sums are
    # all-reduced) before nep_best is written — same criterion as train_nep.
    true_eval_start = (2 * num_epochs) // 3 + 1

    def _save_best():
        raw_model.save_nep_txt(
            os.path.join(output_dir, "nep_best.txt"),
            max_NN_rad, max_NN_ang)

    # Early-stopping tracker (see train_nep). sched_loss is all-reduced, so
    # every rank sees the same value and stops on the same epoch.
    es_best = (
        float(resume_early_stop_state["best"])
        if resume_early_stop_state is not None else float("inf")
    )
    es_wait = (
        int(resume_early_stop_state["wait"])
        if resume_early_stop_state is not None else 0
    )
    prev_in_stage2 = (
        bool(resume_early_stop_state["prev_in_stage2"])
        if resume_early_stop_state is not None else False
    )

    train_t0 = time.time()

    # GPU-side non-finite-gradient guard when the fused optimizer supports
    # the AMP found_inf hook — see train_nep.
    async_guard = (not config["joint_rank4"] and dev.type == "cuda"
                   and optimizer.param_groups[0].get("fused", False)
                   and getattr(optimizer, "_step_supports_amp_scaling",
                               False))

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            t_epoch = time.time()
            model.train()
            if raw_model.spin_curriculum:
                scale = (
                    spin_curriculum_scale(epoch, curriculum_end)
                    if curriculum_enabled else 1.0
                )
                raw_model.set_spin_curriculum_scale(scale)
                warmup_end = max(1, curriculum_end // 2)
                if epoch in (start_epoch, warmup_end, curriculum_end):
                    scale_name = (
                        "step_scale" if raw_model.spin_mode == 3
                        else "gradient_scale"
                    )
                    _log(
                        f"  O3 curriculum epoch {epoch}: "
                        f"{scale_name}={scale:.6f}"
                    )
                if (raw_model.spin_mode == 3 and curriculum_enabled
                        and epoch == warmup_end + 1
                        and not schedule_state["order3_scaler_refreshed"]):
                    fresh_stats = _compute_q_scaler_sharded(
                        raw_model, data_store, backend=backend,
                        batch_size=batch_size, gpumd_init=False,
                        return_rms=True,
                        local_real_flags=local_real_flags,
                    )
                    if not refresh_order3_q_scaler(
                            raw_model, *fresh_stats):
                        raise RuntimeError(
                            "spin3 O3 q_scaler refresh lost its zero-row "
                            "function-preserving precondition"
                        )
                    reset_order3_optimizer_state(raw_model, optimizer)
                    schedule_state["order3_scaler_refreshed"] = True
                    _log("  O3 curriculum: refreshed order-3 q_scaler and "
                         "reset its Adam moments")

            # Per-epoch local frame shuffle. Each rank independently
            # permutes its own n_local frames (different seeds across ranks
            # so each rank sees a fresh order). Step count = ceil(n_local /
            # batch_size) is identical across ranks because n_local is —
            # so DDP collectives stay in lock-step.
            g = torch.Generator()
            g.manual_seed(run_seed + epoch * world_size + rank)
            perm = torch.randperm(n_local, generator=g).tolist()

            # Device-side accumulators, fetched (and all-reduced) once per
            # epoch — see train_nep. Layout: E/F/V/stress/M/Tau sums,
            # E/F/V/M/Tau counts, and the signed E residual.
            acc = torch.zeros(12, dtype=torch.float64, device=dev)
            max_gn_t = torch.zeros((), dtype=dtype, device=dev)
            n_bad_t = torch.zeros((), dtype=dtype, device=dev)

            in_stage2 = stage2 and epoch >= start_stage2
            if in_stage2:
                (cur_pref_e, cur_pref_f, cur_pref_v,
                 cur_pref_m, cur_pref_tau) = (
                    stage2_pref_e, stage2_pref_f, stage2_pref_v,
                    stage2_pref_m, stage2_pref_tau)
                # Apply stage 2 lr + reset the first time we hit stage 2 in
                # THIS run — covers both the natural transition and resuming
                # from a stage-1 checkpoint whose start_epoch has already
                # crossed into stage 2 (``epoch == start_stage2`` would miss
                # the second case, leaving the optimizer on stage-1 lr).
                if not stage2_lr_applied:
                    stage2_lr_applied = True
                    # Distinguish a NATURAL stage-1 -> stage-2 crossing in this
                    # run from RESUMING a checkpoint that was already in stage 2.
                    fresh_transition = start_epoch <= start_stage2
                    _log(f"\n{'='*72}")
                    if fresh_transition:
                        # Crossing into stage 2 now: rank 0 writes a FULL
                        # end-of-stage-1 checkpoint (model + optimizer +
                        # scheduler through epoch-1) so stage 2 can be
                        # redone from this exact point via resume_from.
                        # Weights/optimizer are identical across ranks
                        # (DDP), so rank 0's copy is THE state.
                        if is_main:
                            _save_checkpoint(
                                os.path.join(output_dir,
                                             "checkpoint_stage1.pt"),
                                raw_model, optimizer, lr_scheduler,
                                epoch - 1, best_loss,
                                loss_weights=cur_loss_weights,
                                in_stage2=False, swa_model=None,
                                best_true_loss=best_true_loss,
                                run_seed=run_seed,
                                best_valid_loss=best_valid_loss,
                                valid_info=cur_valid_info,
                                schedule_state=schedule_state,
                                early_stop_state={
                                    "best": es_best, "wait": es_wait,
                                    "prev_in_stage2": prev_in_stage2,
                                })
                            _log("Saved end-of-stage-1 checkpoint: "
                                 "checkpoint_stage1.pt (redo stage 2 from "
                                 "it via resume_from)")
                        for pg in optimizer.param_groups:
                            pg['lr'] = stage2_lr
                        best_loss = float("inf")
                        best_true_loss = float("inf")
                        best_valid_loss = float("inf")
                        _log(f"Stage 2 started at epoch {epoch}: "
                             f"E_w={cur_pref_e}, F_w={cur_pref_f}, "
                             f"V_w={cur_pref_v}, M_w={cur_pref_m}, "
                             f"Tau_w={cur_pref_tau}, lr={stage2_lr:.2e}")
                    else:
                        # Resumed mid-stage-2: keep the checkpoint's restored
                        # (possibly decayed) lr / optimizer / scheduler /
                        # best losses — do NOT re-apply nep.in's stage2_lr
                        # (resume = exact continuation).
                        cur_lr = optimizer.param_groups[0]['lr']
                        _log(f"Stage 2 resumed at epoch {epoch}: "
                             f"E_w={cur_pref_e}, F_w={cur_pref_f}, "
                             f"V_w={cur_pref_v}, M_w={cur_pref_m}, "
                             f"Tau_w={cur_pref_tau}, lr={cur_lr:.2e} "
                             f"(kept from checkpoint)")
                    _log(f"{'='*72}")
            else:
                cur_pref_e, cur_pref_f, cur_pref_v = pref_e, pref_f, pref_v
                cur_pref_m, cur_pref_tau = pref_m, pref_tau

            # Per-epoch compute eligibility: a weight of 0 means "don't compute
            # this channel". Recomputed every epoch so a stage-1 zero weight
            # doesn't block stage-2 computation — and so pref_v=0 really skips
            # virial compute/backward.
            has_forces = global_has_forces and cur_pref_f > 0
            has_virial = global_has_virial and cur_pref_v > 0
            has_mforces = (
                raw_model.spin_descriptor is not None
                and global_has_mforces and cur_pref_m > 0)
            has_tau = (
                raw_model.spin_descriptor is not None
                and global_has_tau and cur_pref_tau > 0)

            batch_indices = [perm[start:start + batch_size]
                             for start in range(0, n_local, batch_size)]
            # The response subset is deliberately replicated. Every rank uses
            # the same group schedule, so DDP averages an identical response
            # gradient while reducing the sharded ordinary gradient. spin3
            # balances every group's total coefficient within the epoch.
            if in_stage2 and response_store is not None:
                if raw_model.spin_mode == 3:
                    response_batches = balanced_group_step_batches(
                        response_store.response_group_ids,
                        len(batch_indices), seed=run_seed + epoch,
                    )
                else:
                    legacy = complete_group_batches(
                        response_store.response_group_ids, 1,
                        seed=run_seed + epoch,
                    )
                    response_batches = [
                        [(legacy[index % len(legacy)], 1.0)]
                        for index in range(len(batch_indices))
                    ]
            else:
                response_batches = []
            for batch_index, batch in enumerate(
                    iter_collated(data_store, batch_indices)):

                # Go through DDP wrapper (not raw_model.compute_*) so the
                # reducer arms backward all-reduce for this step.
                result = model(batch, use_autograd_forces,
                               has_forces, has_virial,
                               has_mforces, has_tau, train_backend)

                e_pa_pred = result["Etot"] / batch["natoms"]
                e_pa_ref = batch["energy"] / batch["natoms"]
                active = (
                    raw_model.spin_dof_type_active[batch["atom_types"]]
                    if has_mforces or has_tau else None)
                if has_mforces:
                    m_diff = result["mforces"] - batch["mforces"]
                    m_mask = batch["mforce_mask"] & active
                else:
                    m_diff = None
                    m_mask = None
                t_mask = batch["tau_mask"] & active if has_tau else None
                emf = batch["energy_mask"].to(dtype)
                fmf = (batch["force_mask"].to(dtype).unsqueeze(-1)
                       if has_forces else None)
                vmf = (batch["virial_mask"].to(dtype).unsqueeze(-1)
                       if has_virial else None)
                mmf = (m_mask.to(dtype).unsqueeze(-1)
                       if m_mask is not None else None)
                tmf = (t_mask.to(dtype).unsqueeze(-1)
                       if t_mask is not None else None)

                # --- DDP-correct normalisation --------------------------
                # DDP averages gradients by world_size. If each rank used a
                # plain local .mean() loss, the implicit per-rank n_local in
                # the denominator leaks into the global gradient (only
                # equivalent to single-card global mean when all n_local are
                # equal — which is NOT the case here because frames have
                # different atom counts).
                # Fix: each rank computes SUM-of-squared-errors, and we divide
                # by the GLOBAL count (all-reduced per batch). The * world_size
                # factor cancels DDP's /world_size averaging — giving a true
                # global-mean loss regardless of how atoms are sharded.
                # Counts stay DEVICE tensors end to end (the all_reduce and
                # the divisions are stream-ordered) — fetching them per batch
                # would drain the compute stream every step. Loss and metric
                # terms are likewise branchless masked sums; the scalars are
                # read back together after the grad-norm guard's sync, where
                # they cost nothing.
                counts = torch.stack([
                    emf.sum(),
                    fmf.sum() if fmf is not None else emf.new_zeros(()),
                    vmf.sum() if vmf is not None else emf.new_zeros(()),
                    mmf.sum() if mmf is not None else emf.new_zeros(()),
                    tmf.sum() if tmf is not None else emf.new_zeros(()),
                ]).to(torch.float64)
                dist.all_reduce(counts)
                cg = counts.clamp(min=1.0).to(dtype)
                ws = float(world_size)

                # A padding-only local batch can have every supervision mask
                # cleared.  Keep a zero-valued dependency on the DDP forward
                # so backward still arms and completes the same reducer hooks
                # on every rank; an independent scalar would have no grad_fn
                # and leave peers blocked in their gradient all-reduce.
                loss = result["Etot"].sum() * 0.0
                m_le = m_resid = m_lf = m_lv = m_ls = None
                m_lm = m_lt = None

                if batch["has_e"]:
                    de = (e_pa_pred - e_pa_ref) * emf
                    loss = loss + cur_pref_e * (de ** 2).sum() * ws / cg[0]
                    m_le = (de ** 2).sum()           # global SSE
                    # Signed residual for the analytical b1 update (folded into
                    # this pass; all-reduced below with the other metrics).
                    m_resid = de.sum()

                if fmf is not None and batch["has_f"]:
                    df = (result["forces"] - batch["forces"]) * fmf
                    # 3 components per atom -> divide by (3 * n_f_g)
                    loss = loss + cur_pref_f * (df ** 2).sum() * ws / (3.0 * cg[1])
                    m_lf = (df ** 2).sum() / 3.0

                if (vmf is not None and batch["has_v"] and "virial" in result
                        and batch["virial"].shape[1] == 9):
                    na = batch["natoms"].unsqueeze(-1)

                    def _v_pred_pa(res):
                        v_sys = torch.zeros(batch["num_structures"], 9,
                                            dtype=dtype, device=dev)
                        si = batch["struct_idx"].unsqueeze(-1).expand_as(
                            res["virial"])
                        v_sys.scatter_add_(0, si, res["virial"])
                        return virial6_for_loss(
                            v_sys, raw_model.spin_mode
                        ) / na
                    v_pred_pa = _v_pred_pa(result)
                    v_ref_pa = virial6_for_loss(
                        batch["virial"], raw_model.spin_mode
                    ) / na
                    dv = (v_pred_pa - v_ref_pa) * vmf
                    # 6 components per frame -> divide by (6 * n_v_g)
                    loss = loss + cur_pref_v * (dv ** 2).sum() * ws / (6.0 * cg[2])
                    m_lv = (dv ** 2).sum() / 6.0
                    # Stress (eV/A**3) = virial_total / V. Sign cancels in MSE.
                    # clamp: masked-out frames may carry volume 0 — their
                    # contribution is already zeroed by the mask factor.
                    scale = (batch["natoms"]
                             / batch["volumes"].clamp(min=1e-9)).unsqueeze(-1)
                    m_ls = ((dv * scale) ** 2).sum() / 6.0

                if mmf is not None and batch["has_m"]:
                    masked_m_diff = m_diff * mmf
                    sum_sq_m = masked_m_diff.square().sum()
                    loss = loss + (
                        cur_pref_m * sum_sq_m * ws / (3 * cg[3])
                    )
                    m_lm = sum_sq_m / 3

                if tmf is not None and batch["has_tau"]:
                    t_diff = (result["tau"] - batch["tau"]) * tmf
                    sum_sq_t = t_diff.square().sum()
                    tau_dof = 2 if raw_model.spin_mode == 3 else 3
                    loss = loss + (
                        cur_pref_tau * sum_sq_t * ws / (tau_dof * cg[4])
                    )
                    m_lt = sum_sq_t / tau_dof

                joint_response = []
                if response_batches:
                    for response_idx, response_weight in response_batches[
                            batch_index]:
                        response_batch = response_store.collate(response_idx)
                        # Use raw_model for the replicated branch. The ordinary
                        # DDP forward already armed the reducer, whose hooks
                        # all-reduce the summed ordinary+response gradient.
                        curve_loss = response_from_store(
                            raw_model, response_batch, response_store,
                            train_backend).square().sum()
                        joint_response.append((response_batch, response_store,
                                               response_loss_weight * response_weight))
                        loss = loss + (
                            response_loss_weight * response_weight
                            * curve_loss
                        )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # spin2 applies its historical gradient gate after DDP
                # reduction; spin3 gates only the realized displacement.
                raw_model.apply_spin_curriculum_gradients()
                curriculum_rows = raw_model.capture_spin_curriculum_rows()
                if max_grad_norm > 0:
                    curriculum_clip_rows = prepare_spin3_curriculum_clip(
                        raw_model
                    )
                    gn_t = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm)
                    restore_spin3_curriculum_clip(
                        raw_model, curriculum_clip_rows, gn_t, max_grad_norm
                    )
                else:
                    gn_t = torch.sqrt(sum(
                        p.grad.norm() ** 2 for p in raw_model.parameters()
                        if p.grad is not None))

                if async_guard:
                    # GPU-side non-finite skip via the fused-Adam AMP hook —
                    # see train_nep. NOTE: with the sync guard each rank
                    # could in principle skip independently (a latent
                    # lock-step hazard); the async flag keeps every rank
                    # stepping, so collectives always stay aligned.
                    bad = (~torch.isfinite(gn_t)).to(dtype)
                    optimizer.found_inf = bad
                    optimizer.step()
                    n_bad_t += bad
                    ok_f = (1.0 - bad).to(torch.float64)
                else:
                    if config["joint_rank4"]:
                        # Every rank must enter (or skip) the sketch collectives.
                        bad = (~torch.isfinite(gn_t)).to(dtype)
                        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
                        if bool(bad):
                            optimizer.zero_grad(set_to_none=True)
                            continue
                    gn = float(gn_t)
                    if not np.isfinite(gn):
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    optimizer_step(
                        optimizer, raw_model, batch,
                        (cur_pref_e, cur_pref_f, cur_pref_v, cur_pref_m, cur_pref_tau),
                        train_backend, schedule_state, cg,
                        **({"responses": joint_response} if joint_response else {}))
                    ok_f = 1.0

                raw_model.apply_spin_curriculum_parameter_step(curriculum_rows)

                if (in_stage2 and swa_model is not None
                        and epoch >= swa_start):
                    swa_model.update_parameters(raw_model)

                zero64 = acc.new_zeros(())
                acc += ok_f * torch.stack([
                    m_le.double() if m_le is not None else zero64,
                    m_lf.double() if m_lf is not None else zero64,
                    m_lv.double() if m_lv is not None else zero64,
                    m_ls.double() if m_ls is not None else zero64,
                    m_lm.double() if m_lm is not None else zero64,
                    m_lt.double() if m_lt is not None else zero64,
                    (batch["energy_mask"].sum().double()
                     if m_le is not None else zero64),
                    (batch["force_mask"].sum().double()
                     if m_lf is not None else zero64),
                    (batch["virial_mask"].sum().double()
                     if m_lv is not None else zero64),
                    (m_mask.sum().double()
                     if m_lm is not None else zero64),
                    (t_mask.sum().double()
                     if m_lt is not None else zero64),
                    m_resid.double() if m_resid is not None else zero64,
                ])
                max_gn_t = torch.maximum(
                    max_gn_t, torch.nan_to_num(gn_t, 0.0, 0.0, 0.0))

            # One epoch-level all-reduce + fetch of the device accumulators
            # (the only metric sync of the epoch).
            dist.all_reduce(acc)
            dist.all_reduce(max_gn_t, op=dist.ReduceOp.MAX)
            dist.all_reduce(n_bad_t, op=dist.ReduceOp.MAX)
            (sum_le, sum_lf, sum_lv, sum_ls, sum_lm, sum_lt,
             sum_e_structs, sum_f_atoms, sum_v_structs,
             sum_m_atoms, sum_t_atoms, sum_e_resid) = acc.tolist()
            max_gn = float(max_gn_t)
            if is_main and int(n_bad_t):
                _log(f"  warning: {int(n_bad_t)} step(s) skipped this epoch "
                     f"(non-finite gradient norm)")

            # Analytical b1 (GPUMD-style), folded into the training pass and
            # all-reduced above — identical on every rank. Updated before the
            # best-model eval / nep_best save so weights and offset stay
            # consistent. b1 is not gradient-trained (optimizer-excluded).
            if sum_e_structs > 0:
                with torch.no_grad():
                    raw_model.b1.add_(sum_e_resid / sum_e_structs)

            # Per-sample (not per-batch) averaging so avg_loss is self-
            # consistent with rmse_{e,f,v}: avg_loss == \Sigma pref_X * MSE_X
            # where each MSE_X aggregates over all samples in the epoch.
            from .constants import EV_PER_A3_TO_GPa
            mse_e = sum_le / max(sum_e_structs, 1)
            mse_f = sum_lf / max(sum_f_atoms, 1) if sum_lf > 0 else 0.0
            mse_v = sum_lv / max(sum_v_structs, 1) if sum_lv > 0 else 0.0
            mse_s = sum_ls / max(sum_v_structs, 1) if sum_ls > 0 else 0.0
            mse_m = sum_lm / max(sum_m_atoms, 1) if sum_lm > 0 else 0.0
            mse_t = sum_lt / max(sum_t_atoms, 1) if sum_lt > 0 else 0.0
            avg_loss = (cur_pref_e * mse_e + cur_pref_f * mse_f
                        + cur_pref_v * mse_v + cur_pref_m * mse_m
                        + cur_pref_tau * mse_t)
            rmse_e = np.sqrt(mse_e)                           # eV/atom
            rmse_f = np.sqrt(mse_f)                           # eV/A
            rmse_v = np.sqrt(mse_v)                           # eV/atom
            rmse_s_gpa = np.sqrt(mse_s) * EV_PER_A3_TO_GPa    # GPa
            rmse_m = np.sqrt(mse_m)                           # eV/muB
            rmse_t = np.sqrt(mse_t)                           # eV

            # Magnetic-force and torque losses are derivatives with respect
            # to spin coordinates. As in train_nep, use one frozen-weight
            # pass for the epoch metric/scheduler/best gate instead of mixing
            # predictions from weights that changed between minibatches.
            if raw_model.spin_descriptor is not None:
                (avg_loss, rmse_e, rmse_f, rmse_v, rmse_s,
                 rmse_m, rmse_t) = _evaluate_sharded(
                    data_store, eval_batch_size,
                    has_forces, has_virial, has_mforces, has_tau,
                    (cur_pref_e, cur_pref_f, cur_pref_v,
                     cur_pref_m, cur_pref_tau),
                    fit_energy_offset=True)
                rmse_s_gpa = rmse_s * EV_PER_A3_TO_GPa

            response_metric = None
            if in_stage2 and response_store is not None:
                response_metric = evaluate_generator_curves(
                    response_store, eval_batch_size, raw_model,
                    train_backend)
                avg_loss += response_loss_weight * response_metric

            # Validation (COLLECTIVE): each rank evaluates its valid shard
            # with frozen weights, the error sums are all-reduced, so every
            # rank sees the bit-identical validation loss — required because
            # this loss drives the scheduler and best-save branch on all
            # ranks. b1 stays train-fitted (never re-fitted on validation —
            # that would leak validation info into the saved model).
            valid_loss = None
            valid_response_metric = None
            if valid_store is not None:
                (valid_loss, v_rmse_e, v_rmse_f, v_rmse_v,
                 v_rmse_s, v_rmse_m, v_rmse_t) = _evaluate_sharded(
                    valid_store, eval_batch_size,
                    valid_has_forces and cur_pref_f > 0,
                    valid_has_virial and cur_pref_v > 0,
                    (raw_model.spin_descriptor is not None
                     and valid_has_mforces and cur_pref_m > 0),
                    (raw_model.spin_descriptor is not None
                     and valid_has_tau and cur_pref_tau > 0),
                    (cur_pref_e, cur_pref_f, cur_pref_v,
                     cur_pref_m, cur_pref_tau),
                    fit_energy_offset=False)
                v_rmse_s *= EV_PER_A3_TO_GPa
                if in_stage2 and valid_response_store is not None:
                    valid_response_metric = evaluate_generator_curves(
                        valid_response_store, eval_batch_size, raw_model,
                        train_backend,
                        normalization_scale=(
                            response_store.generator_curve_scale))
                    valid_loss += (
                        response_loss_weight * valid_response_metric)
            dt = time.time() - t_epoch

            # With a validation set, the validation loss IS the run's loss:
            # it drives the plateau scheduler, the best-model choice, and the
            # displayed/logged "loss" value (the train RMSE columns remain).
            sched_loss = valid_loss if valid_loss is not None else avg_loss
            if in_stage2 and stage2_scheduler is not None:
                _scheduler_step(stage2_scheduler, sched_loss,
                                lr_scheduler_mode, optimizer, stop_lr)
            elif not in_stage2:
                _scheduler_step(lr_scheduler, sched_loss,
                                lr_scheduler_mode, optimizer, stop_lr)

            # Early stopping (see train_nep). Every rank sees the identical
            # all-reduced sched_loss, so all ranks stop on the same epoch —
            # and, per-stage (MACE-style), all ranks jump into stage 2
            # together when stage 1 plateaus with a stage 2 still pending.
            stop_now = False
            curriculum_ready = (
                not curriculum_enabled or epoch >= curriculum_end)
            if early_stop and curriculum_ready:
                if in_stage2 and not prev_in_stage2:
                    es_best = float("inf")
                    es_wait = 0
                if sched_loss < es_best * (1.0 - 1e-4):
                    es_best = sched_loss
                    es_wait = 0
                else:
                    es_wait += 1
                    if es_wait >= early_stop:
                        if stage2 and not in_stage2:
                            _log(f"Early stop (stage 1): {monitored} did "
                                 f"not improve for {early_stop} epochs — "
                                 f"starting stage 2 at epoch {epoch + 1} "
                                 f"(was scheduled for epoch {start_stage2}).")
                            start_stage2 = epoch + 1
                            es_best = float("inf")
                            es_wait = 0
                        else:
                            stop_now = True
            prev_in_stage2 = in_stage2

            if is_main:
                row = (f"{epoch} {sched_loss:.6e} {rmse_e:.6f} "
                       f"{rmse_f:.6f} {rmse_v:.6f} {rmse_s_gpa:.4f}")
                if raw_model.spin_descriptor is not None:
                    row += f" {rmse_m:.6f} {rmse_t:.6f}"
                if response_enabled:
                    row += (f" {response_metric:.6e}"
                            if response_metric is not None else " nan")
                if valid_loss is not None:
                    row += (f" {v_rmse_e:.6f} {v_rmse_f:.6f} {v_rmse_v:.6f} "
                            f"{v_rmse_s:.4f}")
                    if raw_model.spin_descriptor is not None:
                        row += f" {v_rmse_m:.6f} {v_rmse_t:.6f}"
                    if response_enabled:
                        row += (f" {valid_response_metric:.6e}"
                                if valid_response_metric is not None
                                else " nan")
                loss_log.write(row + "\n")
                loss_log.flush()

                stage_str = "[S2] " if in_stage2 else ""
                cur_lr = optimizer.param_groups[0]['lr']
                v_str = (f" | V {rmse_v:.5f} eV/atom | S {rmse_s_gpa:.3f} GPa"
                         if has_virial else "")
                spin_str = ""
                if has_mforces:
                    spin_str += f" | M {rmse_m:.5f} eV/muB"
                if has_tau:
                    spin_str += f" | T {rmse_t:.5f} eV"
                if response_metric is not None:
                    spin_str += f" | Gcurve {response_metric:.4f}"
                valid_str = ""
                if valid_loss is not None:
                    valid_str = (f" | test E {v_rmse_e:.5f} F {v_rmse_f:.5f}"
                                 + (f" V {v_rmse_v:.5f} S {v_rmse_s:.3f}"
                                    if has_virial else ""))
                    if (raw_model.spin_descriptor is not None
                            and valid_has_mforces and cur_pref_m > 0):
                        valid_str += f" M {v_rmse_m:.5f}"
                    if (raw_model.spin_descriptor is not None
                            and valid_has_tau and cur_pref_tau > 0):
                        valid_str += f" T {v_rmse_t:.5f}"
                    if valid_response_metric is not None:
                        valid_str += (
                            f" Gcurve {valid_response_metric:.4f}")
                line = (f"{stage_str}Epoch {epoch:4d} | "
                        f"loss {sched_loss:.4e} | "
                        f"E {rmse_e:.5f} eV/atom | F {rmse_f:.5f} eV/A"
                        f"{v_str}{spin_str}{valid_str} | gnorm {max_gn:.1f} | "
                        f"lr {cur_lr:.2e} | {dt:.1f}s")
                if epoch % print_interval == 0 or epoch == 1:
                    _log(line)
                else:
                    _out_log_file.write(line + "\n")
                    _out_log_file.flush()

            # ---- best-model bookkeeping (COLLECTIVE — outside is_main) ----
            # avg_loss comes from the all-reduced metrics, so every rank
            # sees the same value and takes the same branch — required
            # because the frozen evaluation is itself a collective (each
            # rank evaluates its local shard, the six error sums are
            # all-reduced, and the aggregated loss is bit-identical across
            # ranks). Only rank 0 touches files. Early in the run avg_loss
            # decides directly; in the last third a new minimum only
            # nominates the weights, and the final epoch is always
            # evaluated, so nep_best can never end up worse than nep_final.
            new_min = avg_loss < best_loss
            if new_min:
                best_loss = avg_loss
            if valid_store is not None:
                # Valid-based selection: valid_loss is already a frozen-
                # weight true evaluation (all-reduced, identical on every
                # rank), so the noisy-proxy / true-eval two-step below is
                # unnecessary. Only rank 0 writes the file.
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    if is_main:
                        _save_best()
            elif (epoch < true_eval_start
                  or raw_model.spin_descriptor is not None):
                if new_min:
                    if raw_model.spin_descriptor is not None:
                        best_true_loss = avg_loss
                    if is_main:
                        _save_best()
            elif new_min or epoch == num_epochs or stop_now:
                t_loss, *_ = _evaluate_sharded(
                    data_store, eval_batch_size,
                    has_forces, has_virial, has_mforces, has_tau,
                    (cur_pref_e, cur_pref_f, cur_pref_v,
                     cur_pref_m, cur_pref_tau),
                    fit_energy_offset=True)
                if t_loss < best_true_loss:
                    best_true_loss = t_loss
                    if is_main:
                        _save_best()

            if is_main and (epoch % checkpoint_interval == 0
                            or epoch == num_epochs or stop_now):
                _save_checkpoint(
                    ckpt_path, raw_model, optimizer,
                    stage2_scheduler if in_stage2 else lr_scheduler,
                    epoch, best_loss, loss_weights=cur_loss_weights,
                    in_stage2=in_stage2,
                    swa_model=swa_model if in_stage2 else None,
                    best_true_loss=best_true_loss, run_seed=run_seed,
                    best_valid_loss=best_valid_loss,
                    valid_info=cur_valid_info,
                    schedule_state=schedule_state,
                    early_stop_state={
                        "best": es_best, "wait": es_wait,
                        "prev_in_stage2": prev_in_stage2,
                    }, start_stage2=start_stage2)

            # Interim predict — uses the CURRENT-epoch weights (not nep_best)
            # so the predict loss matches the line just logged for this epoch:
            # it should fall between this epoch's and the next epoch's
            # displayed loss. Each rank predicts its own data_store shard;
            # arrays are gathered onto rank 0, which writes the output files
            # in input-xyz order. No xyz re-read, no temp model file, no
            # neighbor-list rebuild.
            # Skip on the final epoch — the end-of-training predict runs
            # right after and would overwrite this output anyway.
            if (prediction_interval > 0
                    and epoch % prediction_interval == 0
                    and epoch != num_epochs):
                predict_from_store_sharded(
                    raw_model, data_store, local_global_idx,
                    n_total_frames=n_total,
                    output_dir=output_dir,
                    batch_size=eval_batch_size, verbose=False,
                    local_real_flags=local_real_flags)
                if valid_store is not None:
                    predict_from_store_sharded(
                        raw_model, valid_store, valid_local_global_idx,
                        n_total_frames=n_valid_total,
                        output_dir=output_dir,
                        batch_size=eval_batch_size,
                        verbose=False, suffix="test",
                        local_real_flags=valid_local_real_flags)

            if stop_now:
                _log(f"Early stop: {monitored} did not improve for "
                     f"{early_stop} epochs (stopped at epoch {epoch}/"
                     f"{num_epochs}).")
                break
    finally:
        if is_main and loss_log is not None:
            loss_log.close()

    # b1 is already exact for the final weights: the final epoch always runs
    # the best-model eval, which solves and sets the optimal b1 (all ranks).
    # Recomputing here would desync nep_final's offset from the value the
    # best comparison used and could make nep_final beat nep_best.
    if is_main:
        raw_model.save_nep_txt(os.path.join(output_dir, "nep_final.txt"),
                               max_NN_rad, max_NN_ang)
    if swa_model is not None and int(swa_model.n_averaged) == 0:
        if is_main:
            _log("SWA window never reached (run ended before swa_start) — "
                 "nep_average.txt not written")
        swa_model = None
    if swa_model is not None:
        # All ranks: load the averaged weights and re-solve b1 from the
        # GLOBAL energy residual (b1's per-epoch analytic values were
        # averaged along the trajectory — a stale offset for the averaged
        # weights; see train_nep). Energy-only pass over each shard,
        # all-reduced.
        swa_state = swa_model.module.state_dict()
        final_state = {k: v.clone() for k, v in raw_model.state_dict().items()}
        raw_model.load_state_dict(swa_state)
        _evaluate_sharded(
            data_store, eval_batch_size,
            False, False, False, False,
            (1.0, 0.0, 0.0, 0.0, 0.0),
            fit_energy_offset=True,
        )
        if is_main:
            raw_model.save_nep_txt(os.path.join(output_dir, "nep_average.txt"),
                                   max_NN_rad, max_NN_ang)
            _log("SWA model saved to nep_average.txt (b1 re-solved for the "
                 "averaged weights)")
        raw_model.load_state_dict(final_state)

    if is_main:
        train_time = time.time() - train_t0
        h, rem = divmod(train_time, 3600)
        m_, s = divmod(rem, 60)
        if valid_store is not None:
            _log(f"\nDone. Best validation loss (nep_best): "
                 f"{best_valid_loss:.6e}")
        else:
            _log(f"\nDone. Best loss: {best_loss:.6e}")
        _log(f"Training time: {int(h):02d}:{int(m_):02d}:{s:04.1f}")

        _log("\nRunning prediction on training set (final-epoch model)...")

    # End-of-training predict: every rank still holds its data_store shard,
    # so we reuse it — no xyz re-read, no neighbor-list rebuild, no model-
    # file round-trip. Each rank predicts its own shard; rank 0 gathers the
    # per-frame arrays via all_gather_object and writes the output files.
    pred_t0 = time.time()
    predict_from_store_sharded(
        raw_model, data_store, local_global_idx,
        n_total_frames=n_total,
        output_dir=output_dir,
        batch_size=eval_batch_size, verbose=is_main,
        local_real_flags=local_real_flags)
    if valid_store is not None:
        predict_from_store_sharded(
            raw_model, valid_store, valid_local_global_idx,
            n_total_frames=n_valid_total,
            output_dir=output_dir,
            batch_size=eval_batch_size, verbose=False,
            suffix="test", local_real_flags=valid_local_real_flags)
    if is_main:
        _log(f"  Prediction time: {time.time() - pred_t0:.1f}s")

    # data_store is no longer needed — free it now that predict is done.
    del data_store, valid_store
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    if is_main:
        total_time = time.time() - total_t0
        h, rem = divmod(total_time, 3600)
        m_, s = divmod(rem, 60)
        _log(f"\nTotal time (data + train + predict): "
             f"{int(h):02d}:{int(m_):02d}:{s:04.1f}")
        _log(f"Output: {output_dir}/")
        _out_log_file.close()

    # Resync all ranks before returning. The end-of-training prediction writes
    # are rank-0 only, so non-main ranks finish well ahead; without this barrier
    # a *second* train_nep_sharded() call in the same script would start its
    # collectives while rank 0 is still writing, desynchronising the group.
    # The group itself is kept alive (destroyed once at process exit, see
    # _register_pg_atexit) so the re-init that previously crashed never happens.
    if dist.is_initialized():
        dist.barrier()
