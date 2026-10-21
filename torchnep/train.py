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
NEP training with PyTorch — single-GPU / CPU entry point.

Use ``train_nep`` launched with plain ``python`` for one-GPU, CPU, or Mac
workloads. For multi-GPU training, use ``train_nep_sharded`` launched with
``torchrun`` — it shards the dataset across ranks so each GPU only holds
1/world_size of the structures, enabling datasets much larger than any one
card's memory.

    # single GPU / CPU / Mac
    python run_train.py                        # calls train_nep(...)

    # multi-GPU, single node
    torchrun --standalone --nproc_per_node=N run_train.py   # train_nep_sharded

    # multi-node (via SLURM)
    see example/run_multi_node.sbatch
"""

import contextlib
import hashlib
import math
import os
import platform
import time
import warnings
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from torch.optim.swa_utils import AveragedModel

from .model import NEPModel, slim_model, gpumd_init_parameters
from .virial import virial6_for_labels as virial6_for_loss
from .data import (read_xyz, parse_nep_in, valid_split_indices,
                   stratified_split_indices, build_neighbor_list_np)
from . import ops
from . import __version__
from .predict import predict_from_store
from .spin_schema import resolve_gpumd_qscaler
from .oc_cuda_graph import (
    OCCudaGraphTrainCache,
    configure_graph_optimizer,
    graph_cache_requested,
    graph_support_reason,
)
from .response_training import (
    GENERATOR_CURVE_HUBER_DELTA,
    GENERATOR_CURVE_MEAN_WEIGHT,
    GENERATOR_CURVE_SCALE_FLOOR,
    GENERATOR_CURVE_SCHEMA,
    balanced_group_step_batches,
    complete_group_batches,
    generator_curve_loss,
    group_balanced_generator_rms,
    normalized_group_batches,
    select_generator_curve_frames,
    select_response_frames,
    configure_response_store,
    response_from_store,
)


# ---------------------------------------------------------------------------
# Banner & environment info
# ---------------------------------------------------------------------------

_BANNER = r"""
████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗███╗   ██╗███████╗██████╗
╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║████╗  ██║██╔════╝██╔══██╗
   ██║   ██║   ██║██████╔╝██║     ███████║██╔██╗ ██║█████╗  ██████╔╝
   ██║   ██║   ██║██╔══██╗██║     ██╔══██║██║╚██╗██║██╔══╝  ██╔═══╝
   ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║██║ ╚████║███████╗██║
   ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝
"""

_AUTHOR = "Yongchao Wu, yongchao.wu@aalto.fi"


def _backend_info(dev: torch.device, world_size: int = 1) -> List[str]:
    """Describe compute backend for the startup banner.

    ``world_size`` is used by ``train_nep_sharded`` to surface DDP info;
    single-GPU ``train_nep`` keeps the default of 1.

    PyTorch-ROCm exposes AMD GPUs through the ``torch.cuda`` namespace, so
    the cuda branch also handles ROCm. Intel GPUs use ``torch.xpu``.
    """
    lines = []
    if dev.type == "cuda":
        n_visible = torch.cuda.device_count()
        is_rocm = getattr(torch.version, "hip", None) is not None
        tag = "ROCm" if is_rocm else "CUDA"
        if world_size > 1:
            lines.append(f"Backend  : {tag} (DDP, {world_size} processes)")
        else:
            lines.append(f"Backend  : {tag}")
        names = {torch.cuda.get_device_name(i) for i in range(n_visible)}
        name_str = ", ".join(sorted(names))
        total_gb = sum(torch.cuda.get_device_properties(i).total_memory
                       for i in range(n_visible)) / 1e9
        lines.append(f"Devices  : {n_visible} x {name_str}  ({total_gb:.1f} GB total)")
    elif dev.type == "xpu":
        n_visible = torch.xpu.device_count()
        lines.append(f"Backend  : XPU (Intel){' (DDP, ' + str(world_size) + ' processes)' if world_size > 1 else ''}")
        names = {torch.xpu.get_device_name(i) for i in range(n_visible)}
        lines.append(f"Devices  : {n_visible} x {', '.join(sorted(names))}")
    elif dev.type == "mps":
        lines.append("Backend  : MPS (Apple Silicon)")
    else:
        lines.append(f"Backend  : CPU ({platform.processor() or platform.machine()})")
    lines.append(f"PyTorch  : {torch.__version__}")
    return lines


def _default_device() -> str:
    """Select best available device: CUDA/ROCm -> XPU -> MPS -> CPU.

    PyTorch-ROCm routes AMD GPUs through the cuda namespace, so the cuda
    probe catches both. Other torch backends (e.g. XPU for Intel) are
    detected if their namespace is present and reports an available device.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_config_summary(config: dict) -> List[str]:
    """Render the parsed nep.in config as printable lines.

    Each value is prefixed with ``(input)`` if it appeared in nep.in or
    ``(default)`` if ``parse_nep_in`` filled it in. Helps users verify the
    parameters their run is actually using.

    Reads ``config["_explicit"]`` (the set of keys present in nep.in before
    defaults were applied); falls back to "(input)" for everything if that
    marker is absent.
    """
    explicit = config.get("_explicit", None)
    def tag(*keys):
        # If any of the canonical keys for a grouped option is explicit
        # (e.g. user wrote `cutoff 6 4`, which sets cutoff_radial AND
        # cutoff_angular), mark the whole row as (input).
        if explicit is None:
            return "(input)  "
        return "(input)  " if any(k in explicit for k in keys) else "(default)"

    lines = []
    lines.append("Model architecture")
    lines.append("------------------")
    lines.append(f"  {tag('type_names', 'num_types'):10}  types        "
                 f"{config['num_types']}  {' '.join(config['type_names'])}")
    lines.append(f"  {tag('cutoff_radial', 'cutoff_angular'):10}  cutoff       "
                 f"{config['cutoff_radial']} {config['cutoff_angular']}")
    lines.append(f"  {tag('n_max_radial', 'n_max_angular'):10}  n_max        "
                 f"{config['n_max_radial']} {config['n_max_angular']}")
    lines.append(f"  {tag('basis_size_radial', 'basis_size_angular'):10}  "
                 f"basis_size   "
                 f"{config['basis_size_radial']} {config['basis_size_angular']}")
    lm_pad = (config['l_max'] + [0] * 7)[:7]
    lines.append(f"  {tag('l_max'):10}  l_max        "
                 f"{' '.join(str(x) for x in lm_pad)}\n"
                 f"  {'':10}  (L_3b, q_222, q_1111, q_112, q_123, q_233, q_134)")
    lines.append(f"  {tag('neuron'):10}  neuron       {config['neuron']}")
    if config.get("zbl") is not None:
        zbl_extra = ""
        if config.get("typewise_cutoff_zbl_factor") is not None:
            zbl_extra = f"  typewise factor {config['typewise_cutoff_zbl_factor']}"
        lines.append(f"  {tag('zbl'):10}  zbl          {config['zbl']}{zbl_extra}")
    if config.get("spin_mode", 0):
        mode = config["spin_mode"]
        family = "O/C magnetic polynomial"
        lines.append(
            f"  {tag('spin_mode'):10}  spin_mode    {mode} ({family})"
        )
        lines.append(f"  {tag('spin_compress'):10}  spin_compress "
                     f"{config['spin_compress']}")
        lines.append(f"  {tag('spin_basis_size'):10}  spin_basis_size "
                     f"{config['spin_basis_size'][0]}")
        lines.append(f"  {tag('spin_l_max'):10}  spin_l_max  "
                     f"{config['spin_l_max'][0]}")
        lines.append(f"  {tag('spin_order'):10}  spin_order  "
                     f"{config['spin_order']}")
        lines.append(f"  {tag('spin_soc'):10}  spin_soc     "
                     f"{config['spin_soc']}")
        spin_cutoffs = config.get(
            "spin_cutoff_by_type", [config["spin_cutoff"][0]]
        )
        if all(value == spin_cutoffs[0] for value in spin_cutoffs[1:]):
            spin_cutoffs = spin_cutoffs[:1]
        spin_cutoff_text = " ".join(str(value) for value in spin_cutoffs)
        lines.append(f"  {tag('spin_cutoff'):10}  spin_cutoff "
                     f"{spin_cutoff_text}")

    lines.append("")
    lines.append("Training schedule (Stage 1)")
    lines.append("---------------------------")
    lines.append(f"  {tag('num_epochs'):10}  epoch        {config['num_epochs']}")
    lines.append(f"  {tag('batch_size'):10}  batch        {config['batch_size']}")
    lines.append(f"  {tag('eval_batch_size'):10}  eval_batch   "
                 f"{config['eval_batch_size']}")
    lines.append(f"  {tag('lr'):10}  lr           {config['lr']}")
    lines.append(f"  {tag('stop_lr'):10}  stop_lr      {config['stop_lr']}")
    lines.append(f"  {tag('lr_scheduler'):10}  lr_scheduler {config['lr_scheduler']}")
    # `scheduler_patience` serves both modes: for plateau it is the number
    # of non-improving epochs before LR reduction; for step it is the LR
    # step interval (StepLR's `step_size`).
    patience_label = ("step_size" if config['lr_scheduler'] == 'step'
                      else "patience")
    lines.append(f"  {tag('scheduler_patience'):10}  {patience_label:11}  "
                 f"{config['scheduler_patience']}")
    lines.append(f"  {tag('scheduler_factor'):10}  factor       "
                 f"{config['scheduler_factor']}")
    lines.append(f"  {tag('max_grad_norm'):10}  max_grad     {config['max_grad_norm']}")
    lines.append(f"  {tag('lambda_e'):10}  lambda_e     {config['lambda_e']}")
    lines.append(f"  {tag('lambda_f'):10}  lambda_f     {config['lambda_f']}")
    lines.append(f"  {tag('lambda_v'):10}  lambda_v     {config['lambda_v']}")
    if config.get("spin_mode", 0):
        lines.append(f"  {tag('lambda_m'):10}  lambda_m     "
                     f"{config['lambda_m']}")
        lines.append(f"  {tag('lambda_tau'):10}  lambda_tau   "
                     f"{config['lambda_tau']}")
    if config.get("weight_decay", 0.0):
        lines.append(f"  {tag('weight_decay'):10}  weight_decay "
                     f"{config['weight_decay']}")
    if config.get("stage2"):
        lines.append("")
        lines.append("Training schedule (Stage 2)")
        lines.append("---------------------------")
        ss = config.get("start_stage2")
        ss_str = f"{ss}" if ss is not None else f"auto (0.5 * {config['num_epochs']})"
        lines.append(f"  {tag('start_stage2'):10}  start_stage2 {ss_str}")
        lines.append(f"  {tag('stage2_lr'):10}  stage2_lr    {config['stage2_lr']}")
        # The stage 2 scheduler can override stage 1's patience / factor.
        # Fall back to the stage 1 values when not explicitly set.
        s2p = config.get('stage2_scheduler_patience', config['scheduler_patience'])
        s2f = config.get('stage2_scheduler_factor',   config['scheduler_factor'])
        lines.append(f"  {tag('stage2_scheduler_patience'):10}  "
                     f"stage2_scheduler_patience {s2p}")
        lines.append(f"  {tag('stage2_scheduler_factor'):10}  "
                     f"stage2_scheduler_factor   {s2f}")
        lines.append(f"  {tag('stage2_pref_e'):10}  stage2_lambda_e {config['stage2_pref_e']}")
        lines.append(f"  {tag('stage2_pref_f'):10}  stage2_lambda_f {config['stage2_pref_f']}")
        lines.append(f"  {tag('stage2_pref_v'):10}  stage2_lambda_v {config['stage2_pref_v']}")
        if config.get("spin_mode", 0):
            lines.append(f"  {tag('stage2_pref_m'):10}  stage2_lambda_m "
                         f"{config['stage2_pref_m']}")
            lines.append(f"  {tag('stage2_pref_tau'):10}  "
                         f"stage2_lambda_tau {config['stage2_pref_tau']}")

    return lines




# ---------------------------------------------------------------------------
# Data store — host-resident, batches streamed to the device
# ---------------------------------------------------------------------------

class StreamDataStore:
    """The training data store: host-resident, GPU memory scales with batch
    size only.

    Every per-structure array stays in host memory. ``collate(indices)``
    assembles the requested frames on the CPU, copies just that batch to the
    device (via a pinned staging copy so the H2D transfer is async), and
    computes the Chebyshev/angular basis for the batch on the fly. The
    background-prefetch iteration in ``iter_collated`` plus the pinned async
    copies hide the streaming work behind the device compute — benchmarks
    across 1-16 element types showed speed parity with a fully preloaded
    GPU store (which this class replaced) at ~10-15x less GPU memory, so
    streaming is the only data path.

    The per-batch basis is chunking-invariant per-pair elementwise math, so
    every batch is bit-identical no matter how the dataset is batched.
    """

    def __init__(self, structures: List[Dict], device: torch.device,
                 dtype: torch.dtype, config: dict = None):
        if config is None:
            raise ValueError("StreamDataStore requires config (it computes "
                             "the cached basis per batch)")
        self.device = device
        self.dtype = dtype
        self.n = len(structures)
        self.has_cached_basis = True
        self.has_spin = bool(config.get("spin_mode", 0))
        if self.has_spin and any("spins" not in s for s in structures):
            raise ValueError(
                "spin_mode 2 or 3 requires a spin/moment vector for every atom")
        self._pin = device.type == "cuda"

        # Basis parameters (needed at collate time).
        self._rc_r = config["cutoff_radial"]
        self._rc_a = config["cutoff_angular"]
        self._bs_r = config["basis_size_radial"]
        self._bs_a = config["basis_size_angular"]
        self._l3 = config["l_max"][0]
        self._num_lm = (sum(2 * ll + 1 for ll in range(1, self._l3 + 1))
                        if self._l3 >= 1 else 0)
        self._basis_fn = self._basis_impl  # swapped by compile_basis()

        self.natoms = [int(s["natoms"]) for s in structures]
        self._nat = np.asarray(self.natoms, dtype=np.int64)
        self._nrad = np.asarray([len(s["pair_i_rad"]) for s in structures],
                                dtype=np.int64)
        self._nang = np.asarray([len(s["pair_i_ang"]) for s in structures],
                                dtype=np.int64)
        self._nspin = (np.asarray(
            [len(s["pair_i_spin"]) for s in structures], dtype=np.int64)
            if self.has_spin else None)
        self._nat_cum = np.concatenate([[0], np.cumsum(self._nat)])
        self._nrad_cum = np.concatenate([[0], np.cumsum(self._nrad)])
        self._nang_cum = np.concatenate([[0], np.cumsum(self._nang)])
        self._nspin_cum = (np.concatenate([[0], np.cumsum(self._nspin)])
                           if self.has_spin else None)
        self.type_counts = [
            np.bincount(s["atom_types"], minlength=len(config["type_names"]))
            for s in structures
        ]

        # One concatenated CPU tensor per field (frame order); collate
        # slices them via the cumulative offsets above.
        self._at_all = torch.from_numpy(
            np.concatenate([s["atom_types"] for s in structures]))
        self._pi_r_all = torch.from_numpy(
            np.concatenate([s["pair_i_rad"] for s in structures]))
        self._pj_r_all = torch.from_numpy(
            np.concatenate([s["pair_j_rad"] for s in structures]))
        self._rij_r_all = torch.from_numpy(
            np.concatenate([s["rij_rad"] for s in structures])).to(dtype)
        self._pi_a_all = torch.from_numpy(
            np.concatenate([s["pair_i_ang"] for s in structures]))
        self._pj_a_all = torch.from_numpy(
            np.concatenate([s["pair_j_ang"] for s in structures]))
        self._rij_a_all = torch.from_numpy(
            np.concatenate([s["rij_ang"] for s in structures])).to(dtype)
        if self.has_spin:
            self._pi_s_all = torch.from_numpy(
                np.concatenate([s["pair_i_spin"] for s in structures]))
            self._pj_s_all = torch.from_numpy(
                np.concatenate([s["pair_j_spin"] for s in structures]))
            self._rij_s_all = torch.from_numpy(
                np.concatenate([s["rij_spin"] for s in structures])).to(dtype)

        self.energy = [float(s["energy"]) if "energy" in s else 0.0
                       for s in structures]
        self.has_energy_flag = ["energy" in s for s in structures]
        self.has_forces_flag = ["forces" in s for s in structures]
        self.has_virial_flag = ["virial" in s for s in structures]
        self.has_spins_flag = [
            self.has_spin and "spins" in s for s in structures]
        self.has_mforces_flag = [
            self.has_spin and "mforces" in s for s in structures]
        self.has_tau_flag = [
            self.has_spin and "tau" in s for s in structures]
        response_names = sorted({
            str(s["response_group"])
            for s in structures if "response_group" in s
        })
        response_index = {
            name: index for index, name in enumerate(response_names)
        }
        self.response_group_ids = [
            response_index.get(str(s.get("response_group")), -1)
            for s in structures
        ]
        response_coordinates = np.asarray([
            float(s.get("response_coordinate", 0.0)) for s in structures
        ], dtype=np.float32 if dtype == torch.float32 else np.float64)
        self.response_coordinates = torch.from_numpy(
            response_coordinates
        ).to(dtype=dtype)
        self.has_spin_tangent_flag = [
            self.has_spin and "spin_tangent" in s for s in structures
        ]

        np_dtype = np.float32 if dtype == torch.float32 else np.float64
        f_parts = []
        for s in structures:
            if "forces" in s:
                f_parts.append(np.asarray(s["forces"]).reshape(-1, 3))
            else:
                f_parts.append(np.zeros((s["natoms"], 3), dtype=np_dtype))
        self._f_all = torch.from_numpy(
            np.concatenate(f_parts).astype(np_dtype, copy=False))

        if self.has_spin:
            spin_parts = []
            mforce_parts = []
            tau_parts = []
            tangent_parts = []
            for s in structures:
                natoms = s["natoms"]
                spin_parts.append(np.asarray(
                    s.get("spins", np.zeros((natoms, 3)))).reshape(-1, 3))
                mforce_parts.append(np.asarray(
                    s.get("mforces", np.zeros((natoms, 3)))).reshape(-1, 3))
                tau_parts.append(np.asarray(
                    s.get("tau", np.zeros((natoms, 3)))).reshape(-1, 3))
                tangent_parts.append(np.asarray(
                    s.get("spin_tangent", np.zeros((natoms, 3)))
                ).reshape(-1, 3))
            np_dtype = np.float32 if dtype == torch.float32 else np.float64
            self._spins_all = torch.from_numpy(
                np.concatenate(spin_parts).astype(np_dtype, copy=False))
            self._mforces_all = torch.from_numpy(
                np.concatenate(mforce_parts).astype(np_dtype, copy=False))
            self._tau_all = torch.from_numpy(
                np.concatenate(tau_parts).astype(np_dtype, copy=False))
            self._tangent_all = torch.from_numpy(
                np.concatenate(tangent_parts).astype(np_dtype, copy=False))

        v_parts = []
        for s in structures:
            if "virial" in s:
                v = np.asarray(s["virial"]).reshape(-1)
                if v.shape[0] == 6:
                    v_parts.append(np.array([v[0], v[3], v[5],
                                             v[3], v[1], v[4],
                                             v[5], v[4], v[2]]))
                else:
                    v_parts.append(v[:9])
            else:
                v_parts.append(np.zeros(9))
        self._v_all = torch.from_numpy(
            np.stack(v_parts).astype(np_dtype, copy=False))

        # Per-frame CPU views — predict_from_store reads
        # ``data_store.forces[i].cpu()`` / ``.virial[i].cpu()`` (no-ops on
        # these CPU views).
        self.forces = list(torch.split(self._f_all, self.natoms))
        self.virial = list(torch.unbind(self._v_all, dim=0))

        # Device copy is kept for predict_from_store (which calls .cpu() on
        # it); collate gathers from the CPU copy and ships the gathered
        # slice through the same pinned/async channel as every other field —
        # indexing the device copy with a pageable index tensor would force
        # a full compute-stream drain per batch (a hidden ~5-10 ms/step
        # sync, the single largest pipeline stall found by profiling).
        vol_cat = np.asarray([s.get("volume", 0.0) for s in structures],
                             dtype=np_dtype)
        self._vol_cpu = torch.from_numpy(vol_cat).to(dtype)
        self.volumes = self._vol_cpu.to(device=device)

        self._e_flag_t = torch.tensor(self.has_energy_flag, dtype=torch.bool)
        self._f_flag_t = torch.tensor(self.has_forces_flag, dtype=torch.bool)
        self._v_flag_t = torch.tensor(self.has_virial_flag, dtype=torch.bool)
        self._energy_t = torch.tensor(self.energy, dtype=dtype)
        self._nat_t = torch.from_numpy(self._nat)
        if self.has_spin:
            self._m_flag_t = torch.tensor(
                self.has_mforces_flag, dtype=torch.bool)
            self._tau_flag_t = torch.tensor(self.has_tau_flag, dtype=torch.bool)
            self._response_group_t = torch.tensor(
                self.response_group_ids, dtype=torch.long)
            self.spins = list(torch.split(self._spins_all, self.natoms))
            self.mforces = list(torch.split(self._mforces_all, self.natoms))
            self.tau = list(torch.split(self._tau_all, self.natoms))
            self.spin_tangent = list(
                torch.split(self._tangent_all, self.natoms))

        self.n_energy = sum(self.has_energy_flag)
        self.n_forces = sum(self.has_forces_flag)
        self.n_virial = sum(self.has_virial_flag)
        self.n_mforces = sum(self.has_mforces_flag)
        self.n_tau = sum(self.has_tau_flag)
        self.has_forces = self.n_forces > 0
        self.has_virial = self.n_virial > 0
        self.has_mforces = self.n_mforces > 0
        self.has_tau = self.n_tau > 0
        self.has_generator_curves = bool(
            self.has_spin
            and self.n > 0
            and all(group >= 0 for group in self.response_group_ids)
            and all(self.has_spin_tangent_flag)
            and all(self.has_mforces_flag)
        )
        self.generator_curve_scale = None
        if self.has_generator_curves:
            target_generators = torch.stack([
                (mforce * tangent).sum()
                for mforce, tangent in zip(
                    self.mforces, self.spin_tangent)
            ])
            response_group_ids = torch.as_tensor(
                self.response_group_ids, dtype=torch.long)
            self.generator_curve_scale = (
                group_balanced_generator_rms(
                    target_generators.detach().to(torch.float64),
                    response_group_ids,
                ).clamp_min(GENERATOR_CURVE_SCALE_FLOOR)
                .to(device=device, dtype=dtype)
            ).detach()

    def _cat(self, t, indices, cum):
        """Concatenate the per-frame slices of ``t`` for ``indices`` (CPU)."""
        return torch.cat([t[cum[i]:cum[i + 1]] for i in indices])

    def _to_dev(self, t):
        """Host -> device via pinned staging (async under the CUDA caching
        host allocator, which defers reuse of the staging block until the
        copy's recorded event completes)."""
        if self._pin:
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device)

    def _assemble_cpu(self, indices: List[int]) -> Dict:
        """CPU half of collate: gather the frames' arrays into contiguous
        (pinned) host tensors. Runs entirely on the CPU, so a background
        thread can execute it while the device chews the previous batch
        (see ``iter_collated``)."""
        idx = np.asarray(indices, dtype=np.int64)
        nat = self._nat[idx]
        nr = self._nrad[idx]
        na = self._nang[idx]
        offsets = np.concatenate([[0], np.cumsum(nat)])

        idx_t = torch.from_numpy(idx)
        nat_t = torch.from_numpy(nat)
        nr_t = torch.from_numpy(nr)
        na_t = torch.from_numpy(na)
        off_t = torch.from_numpy(offsets[:-1])

        # Pair indices are frame-local in storage; shift each frame's pairs
        # by its atom offset within the batch.
        off_rep_r = torch.repeat_interleave(off_t, nr_t)
        off_rep_a = torch.repeat_interleave(off_t, na_t)
        if self.has_spin:
            ns = self._nspin[idx]
            ns_t = torch.from_numpy(ns)
            off_rep_s = torch.repeat_interleave(off_t, ns_t)

        pin = self._pin
        def _stage(t):
            return t.pin_memory() if pin else t

        out = {
            "N": int(offsets[-1]), "num_structures": len(indices),
            "idx_t": idx_t,
            "atom_types": _stage(self._cat(self._at_all, idx, self._nat_cum)),
            "struct_idx": _stage(torch.repeat_interleave(
                torch.arange(len(indices), dtype=torch.long), nat_t)),
            "pair_i_rad": _stage(
                self._cat(self._pi_r_all, idx, self._nrad_cum) + off_rep_r),
            "pair_j_rad": _stage(
                self._cat(self._pj_r_all, idx, self._nrad_cum) + off_rep_r),
            "rij_rad": _stage(self._cat(self._rij_r_all, idx, self._nrad_cum)),
            "pair_i_ang": _stage(
                self._cat(self._pi_a_all, idx, self._nang_cum) + off_rep_a),
            "pair_j_ang": _stage(
                self._cat(self._pj_a_all, idx, self._nang_cum) + off_rep_a),
            "rij_ang": _stage(self._cat(self._rij_a_all, idx, self._nang_cum)),
            "energy": _stage(self._energy_t[idx_t]),
            "natoms": _stage(self._nat_t[idx_t].to(self.dtype)),
            "energy_mask": _stage(self._e_flag_t[idx_t]),
            "forces": _stage(self._cat(self._f_all, idx, self._nat_cum)),
            "force_mask": _stage(torch.repeat_interleave(
                self._f_flag_t[idx_t], nat_t)),
            "virial": _stage(self._v_all[idx_t].to(self.dtype)),
            "virial_mask": _stage(self._v_flag_t[idx_t]),
            "volumes": _stage(self._vol_cpu[idx_t]),
            # CPU-side mask summaries: the training loop branches on
            # "does this batch have any energy/force/virial labels" — as
            # host bools they cost nothing, while `mask.any()` on the
            # device tensor would drain the compute stream every step.
            "has_e": bool(self._e_flag_t[idx_t].any()),
            "has_f": bool(self._f_flag_t[idx_t].any()),
            "has_v": bool(self._v_flag_t[idx_t].any()),
        }
        if self.has_spin:
            out.update({
                "pair_i_spin": _stage(
                    self._cat(self._pi_s_all, idx, self._nspin_cum)
                    + off_rep_s),
                "pair_j_spin": _stage(
                    self._cat(self._pj_s_all, idx, self._nspin_cum)
                    + off_rep_s),
                "rij_spin": _stage(
                    self._cat(self._rij_s_all, idx, self._nspin_cum)),
                "spins": _stage(self._cat(
                    self._spins_all, idx, self._nat_cum)),
                "spin_tangent": _stage(self._cat(
                    self._tangent_all, idx, self._nat_cum)),
                "mforces": _stage(self._cat(
                    self._mforces_all, idx, self._nat_cum)),
                "mforce_mask": _stage(torch.repeat_interleave(
                    self._m_flag_t[idx_t], nat_t)),
                "tau": _stage(self._cat(
                    self._tau_all, idx, self._nat_cum)),
                "tau_mask": _stage(torch.repeat_interleave(
                    self._tau_flag_t[idx_t], nat_t)),
                "response_group_ids": _stage(
                    self._response_group_t[idx_t]),
                "response_coordinates": _stage(
                    self.response_coordinates[idx_t]),
                "has_m": bool(self._m_flag_t[idx_t].any()),
                "has_tau": bool(self._tau_flag_t[idx_t].any()),
            })
        return out

    _DEVICE_KEYS = ("atom_types", "struct_idx", "pair_i_rad", "pair_j_rad",
                    "rij_rad", "pair_i_ang", "pair_j_ang", "rij_ang",
                    "energy", "natoms", "energy_mask", "forces",
                    "force_mask", "virial", "virial_mask", "volumes")
    _SPIN_DEVICE_KEYS = (
        "pair_i_spin", "pair_j_spin", "rij_spin", "spins",
        "spin_tangent", "mforces", "mforce_mask", "tau", "tau_mask",
        "response_group_ids", "response_coordinates",
    )

    def _basis_impl(self, rij_r, rij_a):
        """Per-batch Chebyshev/angular basis.

        Per-pair elementwise math — chunking-invariant, so results do not
        depend on how the dataset is batched. ``compile_basis`` may swap in
        a torch.compile'd version of this same method.
        """
        dr = torch.norm(rij_r, dim=-1)
        fk_r, fkp_r = ops.chebyshev_basis_and_deriv(dr, self._rc_r, self._bs_r)
        da = torch.norm(rij_a, dim=-1)
        fk_a, fkp_a = ops.chebyshev_basis_and_deriv(da, self._rc_a, self._bs_a)
        dinv_a = 1.0 / da
        if self._num_lm > 0:
            blm = ops.angular_basis(
                rij_a[:, 0] * dinv_a, rij_a[:, 1] * dinv_a,
                rij_a[:, 2] * dinv_a, self._l3)
        else:
            blm = torch.zeros(rij_a.shape[0], 0, dtype=self.dtype,
                              device=self.device)
        return fk_r, fkp_r, 1.0 / dr, fk_a, fkp_a, dinv_a, blm

    def compile_basis(self):
        """torch.compile the per-batch basis compute (~4x fewer kernel
        launches per batch). Only used when the training compute itself is
        compiled: Inductor's fusion reassociates float ops, so the compiled
        basis deviates from the eager one at the same ~1e-7 level the
        compiled training compute already introduces. Never enabled in
        eager runs, which keeps eager training exactly chunking-invariant.
        """
        self._basis_fn = torch.compile(self._basis_impl, dynamic=True)

    def _finalize(self, staged: Dict) -> Dict:
        """Device half of collate: ship the staged host tensors to the
        device (async — they are pinned) and compute the batch's basis."""
        dev = self.device
        batch = {"N": staged["N"], "num_structures": staged["num_structures"],
                 "has_e": staged["has_e"], "has_f": staged["has_f"],
                 "has_v": staged["has_v"]}
        if self.has_spin:
            batch["has_m"] = staged["has_m"]
            batch["has_tau"] = staged["has_tau"]
        keys = self._DEVICE_KEYS + (
            self._SPIN_DEVICE_KEYS if self.has_spin else ())
        for key in keys:
            t = staged[key]
            batch[key] = (t.to(dev, non_blocking=True) if self._pin
                          else t.to(dev))
        (batch["fk_rad"], batch["fkp_rad"], batch["d12inv_rad"],
         batch["fk_ang"], batch["fkp_ang"], batch["d12inv_ang"],
         batch["blm"]) = self._basis_fn(batch["rij_rad"], batch["rij_ang"])

        return batch

    def collate(self, indices: List[int]) -> Dict:
        """Assemble one batch on the CPU, ship it to the device, and compute
        the batch's basis there. Returns the collated batch dict."""
        return self._finalize(self._assemble_cpu(indices))


_PROF = int(os.environ.get("TORCHNEP_PROFILE", "0") or "0")
_SENTINEL = object()


def _timed_iter(gen, sink, sync_cuda):
    """Wrap a batch iterator, splitting wall time between "waiting for /
    finalizing data" (sink[0]) and "the consumer's loop body" (sink[1]).

    With ``sync_cuda`` (TORCHNEP_PROFILE=2) a device synchronize runs at
    both boundaries, converting the CPU walls into true GPU-inclusive
    phase times (at the cost of disabling pipeline overlap — profiling
    mode only).
    """
    gen = iter(gen)
    while True:
        t0 = time.perf_counter()
        item = next(gen, _SENTINEL)
        if sync_cuda:
            torch.cuda.synchronize()
        sink[0] += time.perf_counter() - t0
        if item is _SENTINEL:
            return
        t0 = time.perf_counter()
        yield item
        if sync_cuda:
            torch.cuda.synchronize()
        sink[1] += time.perf_counter() - t0


def iter_collated(data_store, index_lists):
    """Yield collated device batches for ``index_lists``, in order.

    For a ``StreamDataStore`` the CPU half of each collate (gather + pinning)
    runs in a background thread one batch ahead, overlapping with the
    device compute of the current batch; only the cheap device half
    (async H2D + basis kernels) stays on the caller's thread. Batches are
    identical to calling ``store.collate`` directly — this only changes
    WHEN the CPU work happens, never what it produces.

    For any other store this is a plain sequential collate loop.
    """
    if not isinstance(data_store, StreamDataStore) or len(index_lists) <= 1:
        for idx in index_lists:
            yield data_store.collate(idx)
        return

    import queue
    import threading

    q = queue.Queue(maxsize=2)
    stop = threading.Event()

    def _worker():
        try:
            for idx in index_lists:
                if stop.is_set():
                    return
                q.put(("ok", data_store._assemble_cpu(idx)))
        except BaseException as e:          # surface in the consumer
            q.put(("err", e))
            return
        q.put(("done", None))

    t = threading.Thread(target=_worker, daemon=True,
                         name="torchnep-stream-prefetch")
    t.start()
    try:
        while True:
            tag, payload = q.get()
            if tag == "done":
                break
            if tag == "err":
                raise payload
            yield data_store._finalize(payload)
    finally:
        stop.set()
        # Unblock the worker if it is waiting on a full queue.
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
        t.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess_one_frame(args):
    """Worker: build neighbor lists for a single frame. Picklable for mp.Pool."""
    (frame, rc_rad, rc_ang, rc_spin_by_type, max_rc,
     type_names, dtype) = args
    positions = frame["positions"].astype(dtype)
    cell = frame["cell"].astype(dtype)
    atom_types = np.array([type_names.index(s) for s in frame["species"]],
                          dtype=np.int64)
    pair_i, pair_j, rij = build_neighbor_list_np(positions, cell, max_rc)
    dij = np.linalg.norm(rij, axis=1)
    rad_mask = dij < rc_rad
    ang_mask = dij < rc_ang
    s = {
        "natoms": frame["natoms"],
        "atom_types": atom_types,
        "volume": float(abs(np.linalg.det(cell))),   # A**3, used for stress RMSE
        "pair_i_rad": pair_i[rad_mask], "pair_j_rad": pair_j[rad_mask],
        "rij_rad": rij[rad_mask].astype(dtype),
        "pair_i_ang": pair_i[ang_mask], "pair_j_ang": pair_j[ang_mask],
        "rij_ang": rij[ang_mask].astype(dtype),
    }
    if rc_spin_by_type is not None:
        spin_cutoff_type = np.asarray(rc_spin_by_type, dtype=dtype)
        spin_cutoff_pair = 0.5 * (
            spin_cutoff_type[atom_types[pair_i]]
            + spin_cutoff_type[atom_types[pair_j]]
        )
        spin_mask = dij < spin_cutoff_pair
        s.update({
            "pair_i_spin": pair_i[spin_mask],
            "pair_j_spin": pair_j[spin_mask],
            "rij_spin": rij[spin_mask].astype(dtype),
        })
    if "energy" in frame:
        s["energy"] = frame["energy"]
    if "forces" in frame:
        s["forces"] = frame["forces"].astype(dtype)
    if "virial" in frame:
        s["virial"] = frame["virial"].astype(dtype)
    if "spins" in frame:
        s["spins"] = frame["spins"].astype(dtype)
    if "mforces" in frame:
        s["mforces"] = frame["mforces"].astype(dtype)
    if "tau" in frame:
        s["tau"] = frame["tau"].astype(dtype)
    if "spin_tangent" in frame:
        s["spin_tangent"] = frame["spin_tangent"].astype(dtype)
    for key in (
        "response_group", "response_probe", "response_coordinate",
        "response_branch",
    ):
        if key in frame:
            s[key] = frame[key]
    return s


def _preproc_workers(n_workers=None):
    """Worker count for neighbor-list pools (see preprocess_structures)."""
    if n_workers is not None:
        return n_workers
    # sched_getaffinity respects cgroup/slurm CPU limits;
    # os.cpu_count() reports the whole node and oversubscribes the
    # allocation (e.g. 80 workers fighting over a 24-core cgroup).
    try:
        cpu_total = len(os.sched_getaffinity(0))
    except AttributeError:          # non-Linux
        cpu_total = os.cpu_count() or 1
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    n_workers = max(1, cpu_total // local_world)
    return int(os.environ.get("TORCHNEP_PREPROC_WORKERS", n_workers))


def make_preproc_pool(n_workers=None):
    """A reusable neighbor-list worker pool (fork by default, see
    preprocess_structures), or None when pooling is disabled. Callers that
    preprocess many small chunks (streamed prediction) create it once instead
    of paying the pool start-up per chunk."""
    n_workers = _preproc_workers(n_workers)
    if n_workers <= 1:
        return None
    import multiprocessing as mp
    method = os.environ.get("TORCHNEP_MP_START_METHOD", "fork")
    try:
        ctx = mp.get_context(method)
    except ValueError:
        ctx = mp.get_context("spawn")
    return ctx.Pool(n_workers)


def preprocess_structures(frames, config, dtype=np.float32, n_workers=None,
                          pool=None):
    """Build neighbor lists for all frames, parallelized across CPU cores.

    Per-frame work is embarrassingly parallel. Worker behavior:

    - **Worker count** defaults to ``cpu_count() // LOCAL_WORLD_SIZE`` so
      DDP ranks on the same node don't oversubscribe CPU cores. Override
      with ``TORCHNEP_PREPROC_WORKERS`` env var.

    - **Start method** defaults to ``fork`` (fastest). Workers do pure numpy
      (neighbor list + type lookup) and never touch CUDA, so fork is safe
      even after the parent has initialized CUDA — same pattern used by
      PyTorch's ``DataLoader(num_workers>0)``. Override with
      ``TORCHNEP_MP_START_METHOD=spawn`` on systems where fork-after-CUDA
      behaves pathologically (rare).

    Disable pooling entirely with ``n_workers=1`` (useful for debugging).
    ``pool``: an existing pool from :func:`make_preproc_pool` to reuse (the
    caller owns it); otherwise a pool is created and closed per call.
    """
    rc_rad = config["cutoff_radial"]
    rc_ang = config["cutoff_angular"]
    rc_spin_by_type = (
        list(config.get(
            "spin_cutoff_by_type",
            [float(config.get("spin_cutoff", [rc_rad, rc_ang])[0])]
            * len(config["type_names"]),
        ))
        if config.get("spin_mode", 0) else None
    )
    type_names = config["type_names"]
    max_rc = max(
        rc_rad, rc_ang,
        max(rc_spin_by_type) if rc_spin_by_type is not None else 0.0,
    )
    args = [
        (f, rc_rad, rc_ang, rc_spin_by_type, max_rc, type_names, dtype)
        for f in frames
    ]

    if pool is not None:
        if len(frames) < 64:
            return [_preprocess_one_frame(a) for a in args]
        n_workers = getattr(pool, "_processes", 1) or 1
        return pool.map(_preprocess_one_frame, args,
                        chunksize=max(1, len(frames) // (n_workers * 4)))

    n_workers = _preproc_workers(n_workers)
    if n_workers <= 1 or len(frames) < 64:
        return [_preprocess_one_frame(a) for a in args]

    import multiprocessing as mp
    method = os.environ.get("TORCHNEP_MP_START_METHOD", "fork")
    try:
        ctx = mp.get_context(method)
    except ValueError:
        ctx = mp.get_context("spawn")
    chunksize = max(1, len(frames) // (n_workers * 4))
    with ctx.Pool(n_workers) as pool:
        return pool.map(_preprocess_one_frame, args, chunksize=chunksize)


def compute_max_neighbors(structures):
    """Return (max_NN_radial, max_NN_angular) over all structures.

    ``max_NN_radial`` also covers the spin neighbor list because GPUMD uses
    the first exported capacity for radial/spin storage.
    """
    max_rad = max_ang = 0
    for s in structures:
        n = s["natoms"]
        if len(s["pair_i_rad"]) > 0:
            counts = np.bincount(s["pair_i_rad"], minlength=n)
            max_rad = max(max_rad, int(counts.max()))
        if len(s.get("pair_i_spin", ())) > 0:
            counts = np.bincount(s["pair_i_spin"], minlength=n)
            max_rad = max(max_rad, int(counts.max()))
        if len(s["pair_i_ang"]) > 0:
            counts = np.bincount(s["pair_i_ang"], minlength=n)
            max_ang = max(max_ang, int(counts.max()))
    return max_rad, max_ang


def fit_energy_baseline(structures, num_types):
    """Per-type baseline for the same per-structure E/N metric as training."""
    labelled = [structure for structure in structures if "energy" in structure]
    if not labelled:
        return np.zeros(num_types, dtype=np.float64)
    counts = np.stack([
        np.bincount(structure["atom_types"], minlength=num_types)
        for structure in labelled
    ]).astype(np.float64)
    natoms = np.asarray(
        [structure["natoms"] for structure in labelled], dtype=np.float64)
    totals = np.asarray(
        [structure["energy"] for structure in labelled], dtype=np.float64)
    baseline, _, rank, _ = np.linalg.lstsq(
        counts / natoms[:, None], totals / natoms, rcond=None)
    if rank < num_types and int(os.environ.get("RANK", "0")) == 0:
        warnings.warn(
            "per-element spin_baseline is non-unique for this composition "
            "matrix; it is a conditioning offset, not a physical atomic "
            "energy",
            RuntimeWarning,
        )
    return baseline


def fit_energy_baseline_frames(frames, type_names):
    """Per-type E/N baseline without building neighbor lists."""
    labelled = [frame for frame in frames if "energy" in frame]
    if not labelled:
        return np.zeros(len(type_names), dtype=np.float64)
    type_index = {name: index for index, name in enumerate(type_names)}
    counts = np.zeros((len(labelled), len(type_names)), dtype=np.float64)
    for row, frame in enumerate(labelled):
        for species in frame["species"]:
            counts[row, type_index[species]] += 1.0
    totals = np.asarray(
        [frame["energy"] for frame in labelled], dtype=np.float64
    )
    natoms = np.asarray(
        [len(frame["species"]) for frame in labelled], dtype=np.float64)
    baseline, _, rank, _ = np.linalg.lstsq(
        counts / natoms[:, None], totals / natoms, rcond=None)
    if rank < len(type_names) and int(os.environ.get("RANK", "0")) == 0:
        warnings.warn(
            "per-element spin_baseline is non-unique for this composition "
            "matrix; it is a conditioning offset, not a physical atomic "
            "energy",
            RuntimeWarning,
        )
    return baseline


def grouped_validation_indices(frames, valid_ratio, run_seed):
    """Hold out whole response groups; ungrouped frames remain independent."""
    groups = {}
    for index, frame in enumerate(frames):
        key = frame.get("response_group", ("frame", index))
        groups.setdefault(key, []).append(index)
    if len(groups) < 2:
        raise ValueError(
            "valid_ratio requires at least two independent response groups; "
            "use valid_file when one group contains the whole dataset"
        )
    keys = list(groups)
    generator = torch.Generator()
    generator.manual_seed(run_seed)
    order = torch.randperm(len(keys), generator=generator).tolist()
    target = max(1, int(round(valid_ratio * len(frames))))
    selected = []
    count = 0
    for position in order:
        candidate = groups[keys[position]]
        if count and count >= target:
            break
        if len(selected) + len(candidate) >= len(frames):
            continue
        selected.extend(candidate)
        count += len(candidate)
    if not selected or len(selected) >= len(frames):
        raise ValueError(
            "response-group validation split could not leave both partitions "
            "non-empty; provide a separate valid_file"
        )
    return sorted(selected)


def validate_spin_frames(frames, source_name):
    """Fail closed when an O/C dataset contains frames without spin vectors."""
    missing = [
        index for index, frame in enumerate(frames) if "spins" not in frame
    ]
    if missing:
        preview = ", ".join(str(index) for index in missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(
            f"spin_mode 2 or 3 requires spins in every {source_name} frame; "
            f"missing at indices {preview}{suffix}"
        )


def validate_response_group_disjoint(train_frames, valid_frames):
    """Reject response trajectories split across explicit train/valid files."""
    train_groups = {
        frame["response_group"]
        for frame in train_frames if "response_group" in frame
    }
    valid_groups = {
        frame["response_group"]
        for frame in valid_frames if "response_group" in frame
    }
    overlap = sorted(train_groups & valid_groups)
    if overlap:
        preview = ", ".join(map(str, overlap[:5]))
        suffix = "..." if len(overlap) > 5 else ""
        raise ValueError(
            "train and validation files share response_group values: "
            f"{preview}{suffix}"
        )


def build_data_lineage(data_file, valid_file, valid_ratio,
                       energy_key="energy", response_weight=0.0):
    """Content-address the data contract saved in a training checkpoint."""
    def _identity(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return {
            "path": os.path.realpath(path),
            "sha256": digest.hexdigest(),
        }

    result = {
        "schema": "torchnep-data-lineage-v1",
        "data": _identity(data_file),
        "validation": _identity(valid_file) if valid_file is not None else None,
        "valid_ratio": valid_ratio,
        "energy_key": energy_key,
    }
    if response_weight > 0:
        if not os.path.isfile(response_data_path(data_file)):
            raise ValueError("lambda_spin_response > 0 requires response.xyz beside train.xyz")
        result["response"] = _identity(response_data_path(data_file))
    return result


def response_data_path(data_file):
    """The response dataset lives beside train.xyz, never in the output dir."""
    return os.path.join(os.path.dirname(os.path.abspath(data_file)), "response.xyz")


def load_response_dataset(config, data_file, valid_frames, energy_key="energy"):
    """Load complete response groups without adding ordinary observations."""
    weight = float(config.get("lambda_spin_response", 0.0))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("lambda_spin_response must be finite and non-negative")
    if weight == 0:
        return [], []
    if config.get("spin_mode", 0) not in (2, 3):
        raise ValueError("lambda_spin_response requires spin_mode 2 or 3")
    path = response_data_path(data_file)
    if not os.path.isfile(path):
        raise ValueError("lambda_spin_response > 0 requires response.xyz beside train.xyz")
    frames = read_xyz(path, energy_key=energy_key)
    for index, frame in enumerate(frames):
        required = ("energy", "forces", "spins", "mforces")
        missing = [key for key in required if key not in frame]
        if missing:
            raise ValueError(
                f"response.xyz frame {index} is missing required labels: "
                + ", ".join(missing))
        for key in required:
            value = np.asarray(frame[key])
            expected = () if key == "energy" else (frame["natoms"], 3)
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError(
                    f"response.xyz frame {index}: {key} must be finite "
                    f"with shape {expected}")
    validate_spin_frames(frames, "response")
    active = config.get("spin_dof_type")
    target = config.get("spin_response_target", "mforce")
    selected = select_response_frames(frames, active, target)
    if not selected or len(selected) != len(frames):
        kind = "energy" if target == "energy" else "rotation"
        raise ValueError(f"response.xyz must contain only complete {kind} response groups")
    if valid_frames is not None:
        validate_response_group_disjoint(selected, valid_frames)
        # Catch identical observations with missing/renamed group metadata.
        def geometry(frame):
            return (tuple(frame["species"]),
                    np.round(frame["positions"], 8).tobytes(),
                    np.round(frame["cell"], 8).tobytes(),
                    np.round(frame.get("spins", []), 8).tobytes())
        heldout = {geometry(frame) for frame in valid_frames}
        if any(geometry(frame) in heldout for frame in selected):
            raise ValueError("response.xyz shares a structure/spin frame with validation")
    validation = (select_response_frames(valid_frames, active, target)
                  if valid_frames is not None else [])
    return selected, validation


def validate_resume_data_lineage(saved, current, require_versioned=False):
    """Fail closed when resume would silently change a scientific split."""
    if saved is None:
        if require_versioned:
            raise ValueError(
                "spin checkpoint is missing content-addressed data lineage")
        return
    if "data" not in saved:
        # Backward compatibility for old nonmagnetic checkpoints, which only
        # recorded the validation path/ratio. Spin2 has no legacy production
        # checkpoint contract and therefore takes the fail-closed branch.
        if require_versioned:
            raise ValueError(
                "spin checkpoint is missing content-addressed data lineage")
        legacy_current = {
            "valid_file": (
                None if current["validation"] is None
                else current["validation"]["path"]
            ),
            "valid_ratio": current["valid_ratio"],
        }
        legacy_saved = {
            "valid_file": (
                None if saved.get("valid_file") is None
                else os.path.realpath(saved["valid_file"])
            ),
            "valid_ratio": saved.get("valid_ratio"),
        }
        if legacy_saved != legacy_current:
            raise ValueError(
                "validation settings changed since the checkpoint was saved")
        warnings.warn(
            "legacy checkpoint has no data-content fingerprint; resume can "
            "only verify its validation path/ratio",
            RuntimeWarning,
        )
        return
    if saved != current:
        raise ValueError(
            "training/validation data content or split changed since the "
            "checkpoint was saved; start a new run or fine-tune instead. "
            f"saved={saved}, current={current}")


@torch.no_grad()
def compute_q_scaler(model, data_store, batch_size=1000, backend="loop",
                     gpumd_init=False, return_rms=False):
    """Compute descriptor min/max across training set.

    Uses the cached-basis path (reusing data_store's precomputed Chebyshev +
    angular basis) — orders of magnitude faster than recomputing from rij.

    ``batch_size`` is a q-scaler-only knob; independent from the training
    batch size because q-scaler has no backward and can tolerate much bigger
    batches (default 1000). Set smaller if GPU memory is tight.
    ``backend`` should match the training backend so the type-pair contraction
    order of operations (and hence the floating-point accumulation) is the
    same as what training will see.

    ``gpumd_init`` selects how the descriptor coefficients are set for this
    pass. False (default): the model's actual (init) coefficients — q_scaler is
    self-consistent with the init. True: all coefficients forced to 1.0,
    exactly matching GPUMD's generation-0 q_scaler (``initial_para``).
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
    q_count = 0

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
        batch = data_store.collate(list(range(start, end)))
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
            has_q_123=model.has_q_123,
            has_q_233=model.has_q_233,
            has_q_134=model.has_q_134,
        )
        if model.spin_descriptor is not None:
            rij_spin = model._spin_descriptor_coordinates(
                batch["rij_spin"], batch["pair_i_spin"],
                batch["pair_j_spin"], batch["atom_types"],
            )
            q_spin = model.spin_descriptor(
                rij_spin, batch["pair_i_spin"],
                batch["pair_j_spin"], batch["spins"], batch["N"],
                batch["atom_types"], c_spin, None,
                model.spin_dof_type_active,
                model.spin_env_type_active,
            )
            q = torch.cat((q, q_spin), dim=-1)
        q_min = torch.min(q_min, q.min(0).values)
        q_max = torch.max(q_max, q.max(0).values)
        if return_rms:
            q_square_sum.add_(q.square().sum(0))
            q_count += q.shape[0]

    model.train()
    if return_rms:
        if q_count == 0:
            raise ValueError("cannot compute q RMS from an empty dataset")
        return q_min, q_max, (q_square_sum / q_count).sqrt()
    return q_min, q_max


@torch.no_grad()
def recompute_b1_shift(raw_model, data_store, batch_size, backend):
    """Set the energy offset ``b1`` to its analytical optimum (GPUMD-style).

    ``b1`` is not a gradient-trained parameter — its gradient is tiny (the
    mean per-atom energy error) so Adam moves it slowly, which on some datasets
    leaves the predicted energies globally shifted. Instead we solve the 1-D
    least-squares offset in closed form each epoch: the optimal shift is the
    (current) mean per-atom residual, and since the predicted energy already
    contains ``b1`` it is an additive correction

        b1 <- b1 + mean_over_energy_structs(E_pred/Na - E_ref/Na).

    This mirrors GPUMD, which recomputes the energy shift every generation so
    the energy loss is offset-free and the overall level cannot drift.
    Returns the new b1 value (float).
    """
    was_training = raw_model.training
    raw_model.eval()
    dev = raw_model.b1.device
    dtype = raw_model.b1.dtype
    num = torch.zeros((), dtype=dtype, device=dev)
    den = 0
    for start in range(0, data_store.n, batch_size):
        end = min(start + batch_size, data_store.n)
        batch = data_store.collate(list(range(start, end)))
        e_mask = batch["energy_mask"]
        if not bool(e_mask.any()):
            continue
        res = raw_model.compute_properties_cached(
            batch, need_forces=False, need_virial=False, backend=backend)
        e_pa_pred = res["Etot"] / batch["natoms"]
        e_pa_ref = batch["energy"] / batch["natoms"]
        diff = (e_pa_pred - e_pa_ref)[e_mask]
        num = num + diff.sum()
        den += int(e_mask.sum().item())
    if den > 0:
        raw_model.b1.add_(num / den)
    if was_training:
        raw_model.train()
    return float(raw_model.b1.item())


# ---------------------------------------------------------------------------
# LR scheduler helpers
# ---------------------------------------------------------------------------

def _default_alloc_conf():
    """Default the CUDA caching allocator to expandable_segments.

    Batches have a different shape every step, so the block-pool
    allocator fragments badly — measured on unep16: reserved memory 26
    GiB for 5.7 GiB actually allocated at batch 512; expandable segments
    collapse that to ~6.3 GiB at identical speed. Applied only when the
    user has not set PYTORCH_ALLOC_CONF / PYTORCH_CUDA_ALLOC_CONF
    themselves, on CUDA (not ROCm), and only before the CUDA context
    exists (the allocator reads the env once at first use).
    """
    if ("PYTORCH_ALLOC_CONF" in os.environ
            or "PYTORCH_CUDA_ALLOC_CONF" in os.environ):
        return
    if torch.version.hip is not None or not torch.cuda.is_available():
        return
    if torch.cuda.is_initialized():
        return
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


from .joint_update import configure_joint_update, optimizer_step


def _make_optimizer(named_params, lr, weight_decay):
    """Adam (weight_decay=0) or AdamW (>0), fused for CUDA float32.

    weight_decay > 0 switches to AdamW (decoupled decay, the MACE-style
    regularizer): plain-Adam L2-in-the-gradient gets rescaled by the
    second-moment normalization, so decay strength would vary per
    parameter — AdamW applies it directly to the weights. With
    weight_decay = 0 both optimizers are identical.

    ALL trainable parameters decay, including the b0 biases (b1 is not
    in the optimizer — it is solved analytically). A bias-exempt group
    was tried and rejected: it fit the training set identically but lost
    ~30-40% on validation/independent-test energies (unep16, 2 seeds) —
    for this architecture the thresholds b0 need the same shrinkage
    pressure as the weights, matching GPUMD's all-parameter
    regularization.

    ``fused=True`` runs the whole update as one kernel per device/dtype
    group instead of several ``foreach`` launches per state tensor — same
    algorithm, only the arithmetic grouping differs. PyTorch's fused Adam
    path is not reliable for this model's float64 training state, so float64
    deliberately uses the regular optimizer. The try/except keeps older
    builds or exotic device combos on the default path.
    """
    params = [p for _, p in named_params]
    cls = torch.optim.AdamW if weight_decay > 0 else torch.optim.Adam
    kwargs = dict(lr=lr, weight_decay=weight_decay, amsgrad=True)
    if (params and params[0].device.type == "cuda"
            and all(p.dtype == torch.float32 for p in params)):
        try:
            return cls(params, fused=True, **kwargs)
        except (RuntimeError, TypeError, ValueError):
            pass
    return cls(params, **kwargs)


def _make_lr_scheduler(optimizer, mode, factor, patience, min_lr):
    """Build the LR scheduler — "plateau" (default) or "step".

    "plateau" -> ReduceLROnPlateau(factor, patience, min_lr=min_lr).
    "step"    -> StepLR(step_size=patience, gamma=factor); min_lr enforced
                manually after each step() via _scheduler_step (StepLR has
                no min_lr).

    ``patience`` is shared between the two modes: for plateau it is the
    number of non-improving epochs before LR is reduced; for step it is the
    interval between scheduled reductions.
    """
    if mode == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=patience, gamma=factor)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor,
        patience=patience, min_lr=min_lr)


def _scheduler_step(scheduler, avg_loss, mode, optimizer, min_lr):
    """Advance the scheduler; for 'step' mode, clamp LR at min_lr manually."""
    if mode == "step":
        scheduler.step()
        for pg in optimizer.param_groups:
            if pg["lr"] < min_lr:
                pg["lr"] = min_lr
    else:
        scheduler.step(avg_loss)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(path, model, optimizer, scheduler, epoch, best_loss,
                     loss_weights=None, in_stage2=False, swa_model=None,
                     best_true_loss=None, run_seed=None,
                     best_valid_loss=None, valid_info=None,
                     schedule_state=None, early_stop_state=None,
                     start_stage2=None):
    """Write a full training checkpoint.

    ``scheduler`` must be the ACTIVE one (stage2_scheduler while in stage 2,
    lr_scheduler otherwise); ``in_stage2`` records which it was so the load
    side can restore the state into the right object.

    ``start_stage2`` records the EFFECTIVE stage-2 start epoch — it can be
    earlier than nep.in's value when a stage-1 early stop jumped into stage 2
    ahead of schedule. The load side restores it only for checkpoints already
    in stage 2, so a resumed run stays there (stage-1 checkpoints defer to
    nep.in, keeping the documented redo-stage-2-with-edited-settings flow).
    """
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    m = m.module if hasattr(m, "module") else m
    state = {
        "epoch": epoch,
        "best_loss": best_loss,
        "model_state": m.state_dict(),
        "model_signature": m.architecture_signature(),
        "optimizer_state": optimizer.state_dict(),
        "in_stage2": in_stage2,
        "run_seed": run_seed,
        "start_stage2": start_stage2,
    }
    joint = getattr(optimizer, "joint_rank4", None)
    if joint is not None:
        state["joint_rank4"] = joint.state_dict()
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    if loss_weights is not None:
        state["loss_weights"] = loss_weights
    if swa_model is not None:
        state["swa_state"] = swa_model.state_dict()
    if best_true_loss is not None:
        state["best_true_loss"] = best_true_loss
    if best_valid_loss is not None:
        state["best_valid_loss"] = best_valid_loss
    if valid_info is not None:
        state["valid_info"] = valid_info
    if schedule_state is not None:
        state["schedule_state"] = schedule_state
    if early_stop_state is not None:
        state["early_stop_state"] = early_stop_state
    best_model_path = os.path.join(
        os.path.dirname(os.path.abspath(path)), "nep_best.txt")
    if os.path.exists(best_model_path):
        with open(best_model_path) as stream:
            state["best_model_text"] = stream.read()
    torch.save(state, path)


def _load_checkpoint(path, model, optimizer, swa_model, device):
    """Load a checkpoint so the run continues exactly where it stopped.

    The optimizer state carries the lr of the checkpoint moment — nep.in's
    lr is never re-applied on resume. The raw scheduler state is returned so
    the caller can first decide whether the scheduler configuration changed:
    identical semantics restore strictly; an intentional config change starts
    a fresh scheduler. SWA state is restored when both sides have it.

    Returns a dict with epoch / best_loss / best_true_loss / loss_weights /
    in_stage2.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    m = m.module if hasattr(m, "module") else m
    model_state = ckpt["model_state"]
    # Checkpoints written by older DDP runs saved the shim's state_dict,
    # whose keys carry a uniform "model." prefix — strip it so they load
    # into a plain NEPModel (new checkpoints always store plain keys).
    if model_state and all(k.startswith("model.") for k in model_state):
        model_state = {k[len("model."):]: v for k, v in model_state.items()}
    saved_signature = ckpt.get("model_signature")
    expected_signature = m.architecture_signature()
    if saved_signature is None:
        if m.spin_mode:
            raise ValueError(
                "spin checkpoint is missing model_signature; export a "
                "versioned nep4_spin2/nep4_spin3 model before fine-tuning")
        warnings.warn(
            "legacy checkpoint has no architecture signature; only tensor "
            "shapes can be checked",
            RuntimeWarning,
        )
    elif saved_signature != expected_signature:
        raise ValueError(
            "checkpoint architecture does not match the current config: "
            f"{saved_signature} != {expected_signature}")
    joint = getattr(optimizer, "joint_rank4", None)
    if (joint is not None) != (ckpt.get("joint_rank4") is not None):
        raise ValueError("resume requires the same joint_rank4 setting; use finetune_from to change optimizer")
    m.load_state_dict(model_state)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if joint is not None:
        joint.load_state_dict(ckpt["joint_rank4"])
    in_stage2 = ckpt.get("in_stage2", False)
    if swa_model is not None and ckpt.get("swa_state") is not None:
        try:
            swa_model.load_state_dict(ckpt["swa_state"])
        except (KeyError, ValueError, RuntimeError) as error:
            raise ValueError(
                "checkpoint SWA state does not match the current model"
            ) from error
    return {
        "epoch": ckpt["epoch"],
        "best_loss": ckpt["best_loss"],
        "best_true_loss": ckpt.get("best_true_loss", float("inf")),
        "loss_weights": ckpt.get("loss_weights"),
        "in_stage2": in_stage2,
        "run_seed": ckpt.get("run_seed"),
        "best_valid_loss": ckpt.get("best_valid_loss", float("inf")),
        "valid_info": ckpt.get("valid_info"),
        "schedule_state": ckpt.get("schedule_state"),
        "early_stop_state": ckpt.get("early_stop_state"),
        "has_swa_state": ckpt.get("swa_state") is not None,
        "best_model_text": ckpt.get("best_model_text"),
        "scheduler_state": ckpt.get("scheduler_state"),
        "start_stage2": ckpt.get("start_stage2"),
    }


def _restore_scheduler_state(info, lr_scheduler, stage2_scheduler,
                             require_versioned=False):
    """Strictly restore the active scheduler for an unchanged objective."""
    target = (
        stage2_scheduler
        if info["in_stage2"] and stage2_scheduler is not None
        else lr_scheduler
    )
    state = info.get("scheduler_state")
    if target is None:
        return
    if state is None:
        if require_versioned:
            raise ValueError(
                "spin checkpoint is missing scheduler_state")
        warnings.warn(
            "legacy checkpoint has no scheduler_state; scheduler restarts",
            RuntimeWarning,
        )
        return
    try:
        target.load_state_dict(state)
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError(
            "checkpoint scheduler_state is incompatible or corrupted"
        ) from error


def _load_finetune_weights(target, path, device):
    """Load fine-tune weights with the same semantic gate as resume."""
    if not path.endswith(".pt"):
        target.load_weights_from_nep_txt(path)
        return
    payload = torch.load(path, map_location=device, weights_only=False)
    if "model_state" in payload:
        state = payload["model_state"]
        signature = payload.get("model_signature")
    else:
        state = payload
        signature = None
    if state and all(key.startswith("model.") for key in state):
        state = {key[len("model."):]: value for key, value in state.items()}
    expected = target.architecture_signature()
    if signature is None:
        if target.spin_mode:
            raise ValueError(
                "spin .pt weights are missing model_signature; use a "
                "versioned checkpoint or nep4_spin2/nep4_spin3 text model")
        warnings.warn(
            "legacy .pt weights have no architecture signature; only tensor "
            "shapes can be checked",
            RuntimeWarning,
        )
    elif signature != expected:
        raise ValueError(
            "fine-tune architecture does not match the current config: "
            f"{signature} != {expected}")
    target.load_state_dict(state, strict=True)


def _trim_loss_log(path, last_epoch):
    """Drop loss.out rows past ``last_epoch``.

    A run killed between a checkpoint and the next one has logged epochs the
    resumed run will retrain; without trimming, those epochs would appear
    twice in the file.
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        lines = f.readlines()
    kept = []
    for ln in lines:
        try:
            ep = int(ln.split()[0])
        except (ValueError, IndexError):
            kept.append(ln)  # header
            continue
        if ep <= last_epoch:
            kept.append(ln)
    if len(kept) != len(lines):
        with open(path, "w") as f:
            f.writelines(kept)


def _accumulate_true_loss_sums(data_store, batch_size, raw_model,
                               compute_props, compute_props_cached,
                               use_autograd_forces, backend,
                               has_forces, has_virial, has_mforce, has_tau,
                               dtype, dev):
    """Frozen-weight squared-error sums over the LOCAL data_store.

    Mirrors the training-epoch accumulation exactly (same masks, same
    per-sample units), but forward-only on one fixed set of weights.
    Returns structural/magnetic squared-error sums, sample counts, and the
    signed energy residual so
    callers can finish the per-sample averaging themselves — the DDP path
    all-reduces these numbers across ranks first, which makes the aggregated
    loss EXACTLY the full-dataset value (a sum of per-shard sums), identical
    to the single-GPU result. ``sum_ls`` is the stress squared-error sum in
    (eV/Å³)² (same normalisation as the train-loop accumulator).
    ``sum_e_resid`` (Σ signed per-atom energy residual) lets the caller solve
    the exact optimal energy offset b1 for these frozen weights in this same
    pass: δ = sum_e_resid / n_e, and the offset-corrected energy MSE is
    sum_le/n_e − δ².
    """
    sum_le = sum_lf = sum_lv = sum_ls = 0.0
    sum_lm = sum_lt = 0.0
    sum_e_resid = 0.0      # Σ signed per-atom energy residual (for exact b1)
    n_e = n_f = n_v = n_m = n_t = 0

    was_training = raw_model.training
    raw_model.eval()  # autograd path: .training drives create_graph
    # The analytical path needs no graph at all; the autograd path still
    # needs grad-enabled rij for the force evaluation, so no_grad only
    # wraps the cached (analytical) compute.
    spin_grad = (raw_model.spin_descriptor is not None
                 and (has_forces or has_virial or has_mforce or has_tau))
    ctx = (contextlib.nullcontext()
           if (use_autograd_forces or spin_grad) else torch.no_grad())
    try:
        with ctx:
            eval_indices = [list(range(s, min(s + batch_size, data_store.n)))
                            for s in range(0, data_store.n, batch_size)]
            for batch in iter_collated(data_store, eval_indices):
                if use_autograd_forces:
                    result = compute_props(
                        batch["rij_rad"], batch["rij_ang"],
                        batch["pair_i_rad"], batch["pair_j_rad"],
                        batch["pair_i_ang"], batch["pair_j_ang"],
                        batch["atom_types"], batch["N"],
                        batch["struct_idx"], batch["num_structures"],
                        need_forces=has_forces, need_virial=has_virial,
                        backend=backend, spins=batch.get("spins"),
                        need_mforce=has_mforce, need_tau=has_tau,
                        rij_spin=batch.get("rij_spin"),
                        pi_spin=batch.get("pair_i_spin"),
                        pj_spin=batch.get("pair_j_spin"))
                else:
                    batch["need_mforce"] = has_mforce
                    batch["need_tau"] = has_tau
                    result = compute_props_cached(
                        batch, need_forces=has_forces,
                        need_virial=has_virial, backend=backend)

                e_mask = batch["energy_mask"]
                if e_mask.any():
                    e_pa_pred = result["Etot"] / batch["natoms"]
                    e_pa_ref = batch["energy"] / batch["natoms"]
                    diff_e = e_pa_pred[e_mask] - e_pa_ref[e_mask]
                    sum_le += (diff_e ** 2).sum().item()
                    sum_e_resid += diff_e.sum().item()
                    n_e += int(e_mask.sum().item())

                if has_forces:
                    f_mask = batch["force_mask"]
                    if f_mask.any():
                        f_diff = result["forces"][f_mask] - batch["forces"][f_mask]
                        sum_lf += (f_diff ** 2).mean(dim=1).sum().item()
                        n_f += int(f_mask.sum().item())

                if (has_virial and "virial" in result
                        and batch["virial"].shape[1] == 9):
                    v_mask = batch["virial_mask"]
                    if v_mask.any():
                        v_atom = result["virial"]
                        v_sys = torch.zeros(batch["num_structures"], 9,
                                            dtype=dtype, device=dev)
                        si = batch["struct_idx"].unsqueeze(-1).expand_as(v_atom)
                        v_sys.scatter_add_(0, si, v_atom)
                        na = batch["natoms"][v_mask].unsqueeze(-1)
                        v_pred6 = virial6_for_loss(
                            v_sys, raw_model.spin_mode
                        )
                        v_ref6 = virial6_for_loss(
                            batch["virial"], raw_model.spin_mode
                        )
                        v_diff = (v_pred6[v_mask] - v_ref6[v_mask]) / na
                        sum_lv += (v_diff ** 2).mean(dim=1).sum().item()
                        # Stress (eV/A**3) = virial_per_atom * natoms / V.
                        s_scale = (batch["natoms"][v_mask]
                                   / batch["volumes"][v_mask]).unsqueeze(-1)
                        sum_ls += ((v_diff * s_scale) ** 2) \
                            .mean(dim=1).sum().item()
                        n_v += int(v_mask.sum().item())

                if has_mforce and "mforces" in result:
                    active = raw_model.spin_dof_type_active[
                        batch["atom_types"]]
                    m_diff = result["mforces"] - batch["mforces"]
                    m_mask = batch["mforce_mask"] & active
                    if m_mask.any():
                        sum_lm += (
                            m_diff[m_mask].square().sum(dim=1) / 3
                        ).sum().item()
                        n_m += int(m_mask.sum().item())

                if has_tau and "tau" in result:
                    active = raw_model.spin_dof_type_active[
                        batch["atom_types"]]
                    t_mask = batch["tau_mask"] & active
                    if t_mask.any():
                        t_diff = result["tau"][t_mask] - batch["tau"][t_mask]
                        tau_dof = 2 if raw_model.spin_mode == 3 else 3
                        sum_lt += (
                            t_diff.square().sum(dim=1) / tau_dof
                        ).sum().item()
                        n_t += int(t_mask.sum().item())
    finally:
        if was_training:
            raw_model.train()

    return (sum_le, sum_lf, sum_lv, sum_ls, sum_lm, sum_lt,
            n_e, n_f, n_v, n_m, n_t, sum_e_resid)


def _evaluate_true_loss(data_store, batch_size, raw_model,
                        compute_props, compute_props_cached,
                        use_autograd_forces, backend,
                        pref_e, pref_f, pref_v, pref_m, pref_tau, dtype, dev):
    """Weighted loss + RMSEs of the CURRENT (frozen) weights, full dataset.

    Single-GPU wrapper around ``_accumulate_true_loss_sums`` — used to
    decide whether a candidate epoch really is the best model (see the
    best-save block in the epoch loop).

    Side effect: sets the exact optimal energy offset ``b1`` for these frozen
    weights (δ = sum_e_resid / n_e) so the evaluated loss and the saved model
    agree. This is what keeps nep_best ≤ nep_final: every candidate (the final
    epoch always among them) is judged AND saved with its own exact b1.

    Returns loss and E/F/V/stress/M/tau RMSEs; energy is offset-corrected.
    """
    has_forces = data_store.has_forces and pref_f > 0
    has_virial = data_store.has_virial and pref_v > 0
    has_mforce = (raw_model.spin_descriptor is not None
                  and data_store.has_mforces and pref_m > 0)
    has_tau = (raw_model.spin_descriptor is not None
               and data_store.has_tau and pref_tau > 0)
    sums = _accumulate_true_loss_sums(
        data_store, batch_size, raw_model,
        compute_props, compute_props_cached,
        use_autograd_forces, backend, has_forces, has_virial,
        has_mforce, has_tau, dtype, dev)
    (sum_le, sum_lf, sum_lv, sum_ls, sum_lm, sum_lt,
     n_e, n_f, n_v, n_m, n_t, sum_e_resid) = sums
    # Exact optimal b1 for these frozen weights, solved from this same pass.
    delta = sum_e_resid / n_e if n_e > 0 else 0.0
    if n_e > 0:
        with torch.no_grad():
            raw_model.b1.add_(delta)
    # Offset-corrected energy MSE = Var(residual) = E[r²] − δ². Clamp tiny
    # negatives from float round-off.
    mse_e = max(0.0, sum_le / max(n_e, 1) - delta * delta)
    mse_f = sum_lf / max(n_f, 1)
    mse_v = sum_lv / max(n_v, 1)
    mse_s = sum_ls / max(n_v, 1)
    mse_m = sum_lm / max(n_m, 1)
    mse_t = sum_lt / max(n_t, 1)
    true_loss = (pref_e * mse_e + pref_f * mse_f + pref_v * mse_v
                 + pref_m * mse_m + pref_tau * mse_t)
    return (true_loss, np.sqrt(mse_e), np.sqrt(mse_f), np.sqrt(mse_v),
            np.sqrt(mse_s), np.sqrt(mse_m), np.sqrt(mse_t))


def _evaluate_valid_loss(valid_store, batch_size, raw_model,
                         compute_props, compute_props_cached,
                         use_autograd_forces, backend,
                         pref_e, pref_f, pref_v, pref_m, pref_tau, dtype, dev):
    """Weighted loss + RMSEs of the frozen weights on the VALIDATION set.

    Unlike ``_evaluate_true_loss`` this never touches ``b1``: the energy
    offset is fitted on training data only (the analytical b1 update in the
    epoch loop) — re-fitting it on validation energies would leak validation
    information into the saved model. Energy MSE is therefore the plain
    residual mean square with the current (train-fitted) b1.

    Returns validation loss and E/F/V/stress/M/tau RMSEs.
    """
    from .constants import EV_PER_A3_TO_GPa
    has_forces = valid_store.has_forces and pref_f > 0
    has_virial = valid_store.has_virial and pref_v > 0
    has_mforce = (raw_model.spin_descriptor is not None
                  and valid_store.has_mforces and pref_m > 0)
    has_tau = (raw_model.spin_descriptor is not None
               and valid_store.has_tau and pref_tau > 0)
    sums = _accumulate_true_loss_sums(
        valid_store, batch_size, raw_model,
        compute_props, compute_props_cached,
        use_autograd_forces, backend, has_forces, has_virial,
        has_mforce, has_tau, dtype, dev)
    (sum_le, sum_lf, sum_lv, sum_ls, sum_lm, sum_lt,
     n_e, n_f, n_v, n_m, n_t, _) = sums
    mse_e = sum_le / max(n_e, 1)
    mse_f = sum_lf / max(n_f, 1)
    mse_v = sum_lv / max(n_v, 1)
    mse_s = sum_ls / max(n_v, 1)
    mse_m = sum_lm / max(n_m, 1)
    mse_t = sum_lt / max(n_t, 1)
    valid_loss = (pref_e * mse_e + pref_f * mse_f + pref_v * mse_v
                  + pref_m * mse_m + pref_tau * mse_t)
    return (valid_loss, np.sqrt(mse_e), np.sqrt(mse_f), np.sqrt(mse_v),
            np.sqrt(mse_s) * EV_PER_A3_TO_GPa,
            np.sqrt(mse_m), np.sqrt(mse_t))


def evaluate_generator_curves(response_store, batch_size, raw_model,
                              backend, normalization_scale=None):
    """Evaluate the selected energy/mforce response objective at fixed weights."""
    if response_store is None:
        return None
    was_training = raw_model.training
    raw_model.eval()
    weighted_sum = 0.0
    if normalization_scale is None:
        normalization_scale = response_store.generator_curve_scale
    try:
        with torch.enable_grad():
            batches = normalized_group_batches(
                response_store.response_group_ids, batch_size, seed=0)
            for indices, batch_weight in batches:
                batch = response_store.collate(indices)
                value = response_from_store(
                    raw_model, batch, response_store, backend, False,
                    normalization_scale).square().sum()
                weighted_sum += value.item() * batch_weight
    finally:
        if was_training:
            raw_model.train()
    return weighted_sum


def _quiet_compile_logs():
    """Silence torch.compile's graph-break WARNING spam.

    The analytical compute has intentional data-dependent control flow (per-type
    loops with boolean masks -> aten.nonzero, ZBL .item()) that Dynamo cannot
    trace, so it logs a graph break for each — harmless (it just falls back to
    eager for those ops), but very noisy, especially under DDP (every rank,
    every recompile). Raise the dynamo/inductor loggers to ERROR so genuine
    compile failures still surface. To see the graph breaks again, set
    logging.getLogger("torch._dynamo").setLevel(logging.WARNING).
    """
    import logging
    for name in ("torch._dynamo", "torch._inductor",
                 "torch.fx.experimental.symbolic_shapes"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _clean_warning_format():
    """Show warnings as a one-line ``Category: message`` — drop the
    ``file:line:`` prefix and the source-line echo. The default formatter prints
    the triggering line of code under the message (e.g. ``model =
    NEPModel(...)`` below the redundant-q_1111 notice), which is just noise.
    """
    import warnings
    warnings.formatwarning = (
        lambda message, category, filename, lineno, line=None:
        f"{category.__name__}: {message}\n")


def _maybe_enable_tf32(dev, dtype, log):
    """TF32 tensor-core matmul — OFF by default.

    NEP's matmuls are small (the NN layer and the descriptor contraction) and
    the workload is dominated by gather/scatter/elementwise ops, so TF32 gives
    no measurable speedup here while slightly reducing float32 matmul precision.
    Opt in with ``TORCHNEP_TF32=1`` (may help on A100/H100). Either way we
    silence Inductor's "consider set_float32_matmul_precision('high')" nag. Only
    applies to float32 CUDA; float64 is never touched.
    """
    if dev.type == "cuda" and dtype == torch.float32:
        import warnings
        warnings.filterwarnings("ignore", message=".*TensorFloat32.*")
        if os.environ.get("TORCHNEP_TF32", "0") == "1":
            torch.set_float32_matmul_precision("high")
            log("  TF32 matmul: enabled (TORCHNEP_TF32=1)")


def _compile_check(dev):
    """Decide whether torch.compile can be used; never raises.

    Returns (ok: bool, msg: str). The CUDA backend (TorchInductor) lowers to
    Triton kernels, so Triton must be importable — this is the usual failure
    mode on a cluster compute node. On CPU, Inductor uses the C++ backend and
    needs no Triton. ``msg`` is a short reason when ``ok`` is False, so callers
    can log a warning and fall back to eager instead of crashing mid-training.
    """
    if not hasattr(torch, "compile"):
        return False, "torch.compile is unavailable in this PyTorch build"
    if dev.type == "cuda":
        import importlib.util
        if importlib.util.find_spec("triton") is None:
            return False, ("Triton not found (the CUDA TorchInductor backend "
                           "needs it) — install triton, or set "
                           "use_compile=False to silence this")
    else:
        # CPU Inductor generates and builds C++ — needs a compiler.
        import shutil
        if not any(shutil.which(c) for c in ("g++", "clang++", "c++")):
            return False, ("no C++ compiler found (the CPU TorchInductor "
                           "backend needs one) — falling back to eager")
    return True, ""


# ---------------------------------------------------------------------------
# Unified training entry point (single GPU or torchrun DDP)
# ---------------------------------------------------------------------------


def validate_retired_curriculum_resume(schedule, epoch):
    """Reject a legacy checkpoint whose removed update schedule is unfinished."""
    end = schedule.get("curriculum_end")
    enabled = bool(schedule.get("curriculum_enabled", end is not None))
    if enabled and (end is None or epoch < end):
        raise ValueError(
            "unfinished spin_curriculum checkpoint cannot be resumed after removal; "
            "use weight-only fine-tuning from its exported nep.txt instead")


def train_nep(
    config_file: str,
    data_file: str,
    output_dir: str = ".",
    device: str = None,
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
    """Train a NEP model on a single device (GPU / CPU / MPS).

    Hyperparameters (epoch / batch / lr / lambda_e,f,v / stage2* / …) come
    from ``config_file`` only. See README for the full nep.in reference.

    Launch:  python run_train.py

    Parameters
    ----------
    config_file, data_file, output_dir : paths.
    device : "cuda" | "cpu" | "mps" — auto-detected if omitted.
    precision : "float32" (default) or "float64".
    use_autograd_forces : True -> autograd-through-rij forces (slower, gold
        standard); False (default) -> analytical chain rule. O/C spin models
        automatically select the full autograd E/F/V/MF/Tau path.
    use_swa : True -> maintain an averaged model during stage 2 and save it
        as ``nep_average.txt`` at the end.
    swa_start : first epoch included in the SWA average (default: the
        last 100 epochs). Averaging all of stage 2 degrades energies.
    use_compile : None (default) = auto — compile on GPU, eager on
        CPU; True/False force it. Missing Triton (GPU) or C++
        toolchain (CPU) degrades to eager with a log note.
        epoch after a one-time first-epoch compilation cost; needs Triton, which
        ships with the CUDA PyTorch build). With autograd forces the
        first-order gradient is materialized via make_fx and compiled too.
    print_interval : log a line to screen every N epochs (all epochs still
        land in output.log).
    restart : on fresh output_dir, write new log; otherwise resume from
        checkpoint.pt if present.
    checkpoint_interval : save checkpoint.pt every N epochs.
    prediction_interval : every N epochs, run predict_from_store with the
        current-epoch weights and overwrite {energy,force,virial}_train.out
        in output_dir — lets you watch the parity-plot converge live.
        Set to 0 or a negative value to disable.
    finetune_from : path to an existing .pt or nep.txt to load weights from
        (weights only — a NEW training starts from them: epoch 1, nep.in lr,
        fresh optimizer). The q_scaler stored with the weights is kept.
    resume_from : path to a checkpoint (.pt) to CONTINUE from — restores
        model, optimizer (incl. lr), scheduler, epoch and best losses, so
        the run picks up exactly where that checkpoint left off. Use
        ``checkpoint_stage1.pt`` to redo stage 2 with edited stage2_*
        settings. Mutually exclusive with finetune_from; takes precedence
        over the automatic checkpoint.pt pickup.
    recompute_q_scaler : only with finetune_from. Default False — the
        q_scaler stored with the loaded weights is kept (it is part of the
        potential they were trained as). True recomputes it from the new
        dataset, which rescales the NN inputs: the loaded model starts off
        wrong and must re-adapt (see bench_test/restart_rework/t8).
    slim_types : drop element types not present in data_file before training.
    energy_key : name of the comment-line tag read as the reference energy
        (default ``"energy"``). Set to ``"atomization_energy"`` to train
        against atomization energies instead of totals.
    use_gpumd_qscaler : Default False — torch's default init with the
        self-consistent q_scaler (computed from the model's actual init
        coefficients). On a 600-epoch 4-seed PdCuNiP benchmark this
        converges to clearly better minima than the GPUMD-style start
        (~12% lower E/V RMSE, ~3% lower F, train and validation alike).
        True reproduces GPUMD's initialization instead: every parameter
        re-initialised uniform(-1, 1) (SNES mu init) and the q_scaler
        computed with all coefficients = 1.0 (GPUMD's generation-0
        ``initial_para``) — useful for GPUMD-comparison runs. Either way
        the saved nep.txt is fully GPUMD-compatible (the scaler is stored
        in the file). Only applies to fresh training (ignored under
        finetune_from).
    run_seed : master RNG seed for this run. None (default) -> a fresh random
        seed each run, so repeated runs differ (independent weight init AND
        per-epoch batch shuffle) — the stochastic-testing behaviour. Pass an
        int to make a run fully reproducible: the same seed reproduces both
        the initial weights and the batch ordering bit-for-bit. The seed is
        saved into checkpoint.pt and restored on resume, so a resumed run
        continues along the very same shuffle stream regardless of what seed
        (or None) is passed at resume time.
    valid_file : path to a validation .xyz. Evaluated (frozen weights, full
        set) every epoch: nep_best is the epoch with the lowest validation
        loss and the plateau LR scheduler steps on the validation loss —
        both anti-overfitting. Writes GPUMD-style *_test.out files.
    valid_ratio : alternative to valid_file — hold out this fraction of
        data_file (e.g. 0.1) as the validation set. The split is drawn from
        run_seed, so it is reproducible for a given seed and is preserved
        exactly on resume (the checkpoint's seed wins). Mutually exclusive
        with valid_file.
    """
    _clean_warning_format()
    _default_alloc_conf()

    if parse_nep_in(config_file).get("response_refine", False):
        from .response_refine import refine_response
        if (resume_from or recompute_q_scaler or slim_types or valid_ratio
                or precision != "float64" or not finetune_from):
            raise ValueError("response_refine requires finetune_from nep.txt and float64; "
                             "no resume, scaler recomputation, type slimming or random split")
        return refine_response(config_file, data_file, finetune_from,
                               output_dir=output_dir, device=device,
                               valid_file=valid_file, energy_key=energy_key)

    # ---- Device ----------------------------------------------------------
    if device is None:
        device = _default_device()
    dev = torch.device(device)
    dtype = torch.float32 if precision == "float32" else torch.float64

    # ---- Logging ---------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    _out_log_file = open(os.path.join(output_dir, "output.log"),
                         "a" if restart else "w")

    def _log(msg=""):
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
    for line in _backend_info(dev):
        _log(line)
    _log(f"Precision: {precision}")
    _maybe_enable_tf32(dev, dtype, _log)
    _log("")

    # ---- Config (all hyperparameters from nep.in) -----------------------
    orig_config = parse_nep_in(config_file)
    use_gpumd_qscaler = resolve_gpumd_qscaler(
        orig_config.get("spin_mode", 0), use_gpumd_qscaler
    )
    for line in format_config_summary(orig_config):
        _log(line)
    _log("")
    config = orig_config
    # Training schedule + loss weights
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
    # Optional stage-2 block
    stage2             = config["stage2"]
    start_stage2       = config.get("start_stage2")  # may be None -> auto 0.5*num_epochs
    stage2_lr          = config["stage2_lr"]
    stage2_pref_e      = config["stage2_pref_e"]
    stage2_pref_f      = config["stage2_pref_f"]
    stage2_pref_v      = config["stage2_pref_v"]
    stage2_pref_m      = config["stage2_pref_m"]
    stage2_pref_tau    = config["stage2_pref_tau"]
    # Stage 2 can have its own decay schedule (default: same as stage 1).
    # Use this when the two stages span different LR ranges — e.g. stage 1
    # 1e-2 -> 1e-3 (factor 0.794, 10 decays) vs stage 2 1e-3 -> 1e-5
    # (factor 0.631, 10 decays).
    stage2_scheduler_patience = config.get("stage2_scheduler_patience",
                                           scheduler_patience)
    stage2_scheduler_factor   = config.get("stage2_scheduler_factor",
                                           scheduler_factor)

    # ---- Resume target + master RNG seed (BEFORE data loading) ------------
    # The seed must be settled before the data section because the validation
    # split (valid_ratio) is drawn from it — and on resume the checkpoint's
    # saved seed must win, or the resumed run would carve out a DIFFERENT
    # validation subset and leak old validation frames into training. Only
    # the seed is peeked here; the full checkpoint load happens later, after
    # model/optimizer construction.
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
    if run_seed is None:
        import random
        run_seed = random.randrange(1, 2**31 - 1)

    # ---- Data ------------------------------------------------------------
    _log("Data")
    _log("----")
    # Hash the exact input bytes before parsing them. If the file changes
    # after this point, the saved lineage remains tied to the pre-read
    # snapshot and a later resume will fail rather than bless new contents.
    initial_data_lineage = build_data_lineage(
        data_file, valid_file, valid_ratio, energy_key,
        config.get("lambda_spin_response", 0.0))
    frames = read_xyz(data_file, energy_key=energy_key)
    _log(f"  read {len(frames)} structures from {data_file} "
         f"(energy label: {energy_key})")

    # ---- Validation set ----------------------------------------------------
    # valid_file: separate .xyz. valid_ratio: hold out a random fraction of
    # data_file, drawn from run_seed via a dedicated generator (the global
    # torch RNG is untouched, so weight init matches a no-valid run). Sorted
    # indices keep the *_test.out rows in input-xyz order.
    if valid_file is not None and valid_ratio is not None:
        raise ValueError("valid_file and valid_ratio are mutually exclusive")
    valid_frames = None
    if valid_file is not None:
        if os.path.samefile(data_file, valid_file):
            raise ValueError("valid_file must not be the training data file")
        valid_frames = read_xyz(valid_file, energy_key=energy_key)
        _log(f"  read {len(valid_frames)} validation structures "
             f"from {valid_file}")
    elif valid_ratio is not None:
        if not 0.0 < valid_ratio < 1.0:
            raise ValueError(f"valid_ratio must be in (0, 1), "
                             f"got {valid_ratio}")
        # The same draw export_valid_split reproduces (see data.py) — the
        # split can be exported as GPUMD-ready train.xyz/test.xyz files.
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
                 f"cell size): {st['n_strata']} strata; "
                 f"{st['n_tiny_frames']} tiny-cell (<=4 atoms) frames and "
                 f"{st['n_rare_frames']} rare-stratum frames kept fully in "
                 f"training; held out {len(val_idx)} frames, "
                 f"{len(train_idx)} remain (split drawn from run_seed)")
            if st.get("fallback") == "random":
                _log("  stratified split starved the validation set "
                     "(dataset is dominated by tiny/rare strata) — fell "
                     "back to a random split with the same seed")
        elif valid_strategy == "random":
            train_idx, val_idx = valid_split_indices(len(frames), valid_ratio,
                                                     run_seed)
            _log(f"  valid_ratio={valid_ratio}: held out {len(val_idx)} "
                 f"frames for validation, {len(train_idx)} remain for "
                 f"training (split drawn from run_seed)")
        else:
            raise ValueError(f"unknown valid_strategy: {valid_strategy!r}")
        valid_frames = [frames[i] for i in val_idx]
        frames = [frames[i] for i in train_idx]

    if config.get("spin_mode", 0):
        validate_spin_frames(frames, "training")
        if valid_frames is not None:
            validate_spin_frames(valid_frames, "validation")
    if valid_file is not None:
        validate_response_group_disjoint(frames, valid_frames)

    response_loss_weight = float(config.get("lambda_spin_response", 0.0))
    response_frames, valid_response_frames = load_response_dataset(
        config, data_file, valid_frames, energy_key)
    response_enabled = bool(response_frames)
    if valid_response_frames and not response_enabled:
        raise ValueError(
            "validation contains rotation response groups but training has "
            "none; keep at least one complete response scan in each split")
    if response_enabled and not config["stage2"]:
        raise ValueError(
            "rotation response groups require stage2=1; the generator-curve "
            "objective is intentionally activated only in Stage 2")
    if response_enabled and valid_frames is not None \
            and not valid_response_frames:
        raise ValueError(
            "training contains rotation response groups but validation has "
            "none; provide a group-disjoint validation response scan so "
            "nep_best is selected on the response objective")
    if response_enabled:
        _log(f"  response target: {config['spin_response_target']}; "
             f"energy scale: {config['spin_response_energy_scale']:g} eV/atom")
        _log(f"  generator curves: {len(response_frames)} training frames"
             + (f", {len(valid_response_frames)} validation frames"
                if valid_frames is not None else ""))
        _log(f"  generator objective: {GENERATOR_CURVE_SCHEMA}; "
             "one complete group per ordinary optimizer step")

    # Single-GPU: per-epoch shuffle is done at iteration time via
    # torch.randperm(n_structs). We deliberately do NOT pre-sort by natoms
    # — natoms-homogeneous minibatches are biased gradient estimates and
    # converge significantly worse than i.i.d. shuffled minibatches (and
    # torch.compile dislikes the wider spread of input shapes that sorting
    # produces).
    n_structs = len(frames)

    # slim_types: detect which element types actually appear in the data and
    # narrow config before building neighbor lists / data store / model.
    # This makes the entire training run faster, not just the output file.
    _slim_keep = None  # None = no slimming; list = types to keep
    if slim_types:
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

    t0 = time.time()
    np_dtype = np.float64 if precision == "float64" else np.float32
    structures = preprocess_structures(frames, config, np_dtype)
    response_structures = (
        preprocess_structures(response_frames, config, np_dtype)
        if response_enabled else None
    )
    _log(f"  built neighbor lists in {time.time() - t0:.1f}s")

    max_NN_rad, max_NN_ang = compute_max_neighbors(structures)
    energy_baseline_values = (
        fit_energy_baseline(structures, config["num_types"])
        if config.get("spin_mode", 0) else None)

    t0 = time.time()
    data_store = StreamDataStore(structures, dev, dtype, config=config)
    del structures
    response_store = (
        StreamDataStore(response_structures, dev, dtype, config=config)
        if response_structures is not None else None
    )
    del response_structures
    configure_response_store(response_store, config)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    _log(f"  data store ready: dataset in host memory, batches streamed "
         f"to {dev} ({time.time() - t0:.1f}s)")
    _log(f"  coverage: {data_store.n_energy} E / "
         f"{data_store.n_forces} F / {data_store.n_virial} V"
         + (f" / {data_store.n_mforces} M / {data_store.n_tau} T"
            if config.get("spin_mode", 0) else ""))

    valid_store = None
    valid_response_store = None
    if valid_frames is not None:
        t0 = time.time()
        structures_v = preprocess_structures(valid_frames, config, np_dtype)
        # nep.txt records max neighbor counts for MD buffer sizing — cover
        # the validation frames too (in ratio mode they came from data_file,
        # so this matches what a no-valid run would have written).
        vNN_rad, vNN_ang = compute_max_neighbors(structures_v)
        max_NN_rad = max(max_NN_rad, vNN_rad)
        max_NN_ang = max(max_NN_ang, vNN_ang)
        valid_store = StreamDataStore(structures_v, dev, dtype, config=config)
        if valid_response_frames:
            response_structures_v = preprocess_structures(
                valid_response_frames, config, np_dtype)
            valid_response_store = StreamDataStore(
                response_structures_v, dev, dtype, config=config)
            del response_structures_v
            configure_response_store(valid_response_store, config)
        del structures_v, valid_frames
        _log(f"  validation set ready in {time.time() - t0:.1f}s — "
             f"coverage: {valid_store.n_energy} E / "
             f"{valid_store.n_forces} F / {valid_store.n_virial} V"
             + (f" / {valid_store.n_mforces} M / {valid_store.n_tau} T"
                if config.get("spin_mode", 0) else ""))
    _log("")

    # ---- Master RNG seed -------------------------------------------------
    # Seed the global torch RNG BEFORE model construction so both the NN weight
    # init (model.py nn.init.*) and the descriptor-coeff re-init below draw
    # deterministically from it. A given run_seed therefore fixes the initial
    # weights; the same seed also drives the per-epoch shuffle (see the loop).
    # The seed itself was settled before data loading (checkpoint's seed on
    # resume, user int, or a fresh random draw) — see the resume/seed block.
    torch.manual_seed(run_seed)


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
        model.set_energy_baseline(torch.as_tensor(
            energy_baseline_values, dtype=dtype, device=dev))

    # use_gpumd_qscaler: reproduce GPUMD's init for fresh training. GPUMD (SNES
    # mu init, snes.cu initialize_mu_and_sigma) draws every GPUMD parameter
    # uniform(-1, 1) — both the descriptor coefficients and the NN weights — so
    # to stay consistent we re-init both here, paired with the c=1 q_scaler
    # computed below. Re-initialising the descriptor coeffs alone would leave the
    # NN at torch's small-variance init, whose shallow binding landscape does not
    # match the one GPUMD's models inherit from their large init. Skipped under
    # finetune_from, where the loaded trained parameters must be kept.
    if use_gpumd_qscaler and finetune_from is None:
        gpumd_init_parameters(model)
        _log("  use_gpumd_qscaler: descriptor coeffs + NN weights re-init "
             "uniform(-1,1), q_scaler will use c=1 (GPUMD-consistent)")

    # b1 (global energy offset) is determined analytically each epoch, not by
    # gradient descent (see recompute_b1_shift) — exclude it from the optimizer.
    model.b1.requires_grad_(False)
    trainable_named = [(n, p) for n, p in model.named_parameters()
                       if n != "b1" and p.requires_grad]
    trainable_params = [p for _, p in trainable_named]
    # Number of trainable variables — the divisor GPUMD uses to turn the raw
    # L1/L2 sums into a mean(|w|) and RMS(w) (see the reg block in the epoch
    # loop). GPUMD's number_of_variables also counts the single global energy
    # shift (our b1), which we train analytically; the 1-param difference is
    # negligible against the thousands of weights here.
    if finetune_from is not None:
        # Load pre-trained weights; skip random b1 init from mean_epa.
        # The stored q_scaler is part of the loaded model and is KEPT (see
        # the q_scaler block below) — recomputing it on the new dataset
        # would rescale the NN inputs the weights were trained against.
        ft_path = finetune_from

        if _slim_keep is not None:
            # Load full pre-trained model (orig arch), then slim to current config
            full_model = NEPModel(orig_config).to(dtype).to(dev)
            _load_finetune_weights(full_model, ft_path, dev)
            slimmed = slim_model(full_model, config["type_names"])
            model.load_state_dict(slimmed.state_dict())
            del full_model, slimmed
            _log(f"  fine-tuning from {ft_path}  "
                 f"[{orig_config['num_types']} -> {config['num_types']} types]")
        else:
            _load_finetune_weights(model, ft_path, dev)
            _log(f"  fine-tuning from {ft_path}")
        _log(f"  {sum(p.numel() for p in model.parameters())} parameters, "
             f"dim={model.dim}, b1={model.b1.item():.4f}")
    else:
        baseline_np = (
            model.energy_baseline.detach().cpu().numpy()
            if model.energy_baseline is not None
            else np.zeros(model.num_types, dtype=np.float64)
        )
        energy_per_atom = [
            (data_store.energy[i] - float(
                data_store.type_counts[i] @ baseline_np))
            / data_store.natoms[i]
            for i in range(data_store.n)
            if data_store.has_energy_flag[i]
        ]
        mean_epa = float(np.mean(energy_per_atom)) if energy_per_atom else 0.0
        with torch.no_grad():
            model.b1.fill_(-mean_epa)
        _log(f"  {sum(p.numel() for p in model.parameters())} parameters, "
             f"dim={model.dim}, b1 init={model.b1.item():.4f}")

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
            # O/C uses custom analytic CUDA Functions with explicit mixed
            # pullbacks.  The generic make_fx compiler does not accept the
            # spin/ragged inputs and must not replace that production path.
            compile_msg = ("  torch.compile: disabled for O/C custom "
                           "autograd sectors")
        elif use_autograd_forces:
            # The nested create_graph=True double backward cannot be lowered
            # directly, but the make_fx route (materialize the first-order
            # gradient as ordinary FX ops, then compile) can — see
            # torchnep/compiled_autograd.py.
            compile_on = True
            compile_msg = ("  torch.compile: enabled (autograd forces via "
                           "make_fx-materialized gradient)")
        else:
            compile_on = True
            method = "analytical compute method"
            compile_msg = f"  torch.compile: enabled ({method})"

    # Backends are chosen automatically: ``backend`` is the eager choice
    # (q_scaler / eval passes), ``train_backend`` adds the use_compile rule
    # (mulsum under compile). See ops.resolve_backend for the policy and the
    # MI250X/V100/A2000 benchmarks behind it.
    from .ops import resolve_backend as _resolve_backend
    backend = _resolve_backend("auto", num_types=model.num_types,
                               device_type=dev.type)
    train_backend = _resolve_backend("auto", num_types=model.num_types,
                                     use_compile=compile_on,
                                     device_type=dev.type)
    force_str = "autograd" if use_autograd_forces else "analytical"
    _log(f"  backend: {train_backend}, forces: {force_str}")

    if finetune_from is not None and not recompute_q_scaler:
        # The q_scaler is part of the potential definition: the loaded NN
        # weights were trained against it. Keep it; recomputing from the
        # new data (with the loaded, trained c-params) would rescale every
        # descriptor component and wreck the loaded model.
        _log("  q_scaler: kept from finetune source (not recomputed)")
    else:
        if finetune_from is not None:
            _log("  q_scaler: RECOMPUTED from the new dataset "
                 "(recompute_q_scaler=True) — the loaded weights will "
                 "see rescaled descriptors and must re-adapt")
        t0 = time.time()
        # The q-scaler pass uses the training batch size so its transient
        # GPU footprint stays batch-bound (min/max over the dataset is
        # chunking-invariant, so the result is unchanged).
        q_stats = compute_q_scaler(
            model, data_store, backend=backend, batch_size=batch_size,
            gpumd_init=use_gpumd_qscaler,
            return_rms=model.spin_mode == 3,
        )
        model.set_q_scaler(*q_stats)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        _log(f"  q_scaler in {time.time() - t0:.1f}s")

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # torch.compile: compile the bound compute METHOD, not the module — the
    # training loop calls compute_properties_cached directly (not forward()), so
    # compiling the module would be a no-op. Compiling the bound method shares
    # parameters with raw_model. dynamic=True avoids a recompile per batch shape.
    compute_props = raw_model.compute_properties
    compute_props_cached = raw_model.compute_properties_cached
    if compile_on:
        _quiet_compile_logs()
        if use_autograd_forces:
            from .compiled_autograd import CompiledAutogradForce
            compute_props = CompiledAutogradForce(raw_model).compute_properties
        else:
            # Compile the branch-free core only — ZBL and result assembly
            # stay eager in the wrapper. Compiling the whole method instead
            # leaves 3 graph breaks (per-type masks + ZBL autograd.grad) and
            # ~2.7x the kernel launches.
            import functools
            _compiled_core = torch.compile(raw_model._cached_core,
                                           dynamic=True)
            compute_props_cached = functools.partial(
                raw_model.compute_properties_cached, core_fn=_compiled_core)
            # Fuse the per-batch basis kernels too — same numerical status
            # as the compiled compute (~1e-7 Inductor deviations).
            data_store.compile_basis()
            if valid_store is not None:
                valid_store.compile_basis()
    if compile_msg is not None:
        _log(compile_msg)

    oc_graph_reason = graph_support_reason(raw_model, dev)
    if graph_cache_requested() and use_autograd_forces and not oc_graph_reason:
        oc_graph_enabled = True
        # Keep capture on the production Torch oracle.  The native moment-gate
        # HVP and the shape-specialized Inductor experiment are separate,
        # opt-in research paths and must not leak into this cache.
        os.environ["TORCHNEP_OC_CUDA_FULL"] = "0"
        os.environ["TORCHNEP_OC_CUDA_TRAIN"] = "0"
        os.environ["TORCHNEP_OC_CUDA_GRAPH"] = "1"
        _log("  O/C CUDA train graph: enabled "
             "(Torch reference, bounded exact-ragged-shape LRU)")
    else:
        oc_graph_enabled = False
        if graph_cache_requested():
            reason = (
                oc_graph_reason or
                "the graph cache requires autograd response training"
            )
            _log(f"  O/C CUDA train graph: disabled — {reason}")

    weight_decay = config["weight_decay"]
    if oc_graph_enabled and config["joint_rank4"]:
        raise ValueError("joint_rank4 requires the eager optimizer path; disable TORCHNEP_OC_CUDA_GRAPH")
    if oc_graph_enabled:
        optimizer_cls = (torch.optim.AdamW if weight_decay > 0
                         else torch.optim.Adam)
        optimizer = optimizer_cls(
            trainable_params, lr=lr, weight_decay=weight_decay,
            amsgrad=True, capturable=True, fused=True)
    else:
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
    schedule_state = {
        "stage2_start": start_stage2 if stage2 else None,
        "use_swa": bool(stage2 and use_swa),
    }

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

    swa_model = None
    stage2_scheduler = None
    if stage2:
        stage2_scheduler = _make_lr_scheduler(
            optimizer, lr_scheduler_mode, stage2_scheduler_factor,
            stage2_scheduler_patience, stop_lr)
        if use_swa:
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
        "stage2_scheduler_factor": stage2_scheduler_factor,
        "stage2_scheduler_patience": stage2_scheduler_patience,
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
        info = _load_checkpoint(
            resume_ckpt, model, optimizer, swa_model, dev)
        # load_state_dict restores the checkpoint's param_groups, which
        # would silently zero a newly requested weight_decay — reapply it.
        if weight_decay > 0:
            for _g in optimizer.param_groups:
                _g["weight_decay"] = weight_decay
        start_epoch = info["epoch"] + 1
        best_loss = info["best_loss"]
        best_true_loss = info["best_true_loss"]
        best_valid_loss = info["best_valid_loss"]
        # A checkpoint already in stage 2 pins the EFFECTIVE stage-2 start —
        # possibly earlier than nep.in's, when a stage-1 early stop jumped
        # ahead of schedule. Stage-1 checkpoints defer to nep.in (so the
        # redo-stage-2-with-edited-settings flow keeps working).
        if stage2 and info["in_stage2"] and info.get("start_stage2") is not None:
            start_stage2 = info["start_stage2"]
        saved_valid_info = info.get("valid_info")
        validate_resume_data_lineage(
            saved_valid_info, cur_valid_info,
            require_versioned=raw_model.spin_descriptor is not None,
        )
        saved_schedule = info.get("schedule_state")
        if saved_schedule is not None:
            validate_retired_curriculum_resume(saved_schedule, info["epoch"])
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
        elif raw_model.spin_descriptor is not None:
            raise ValueError(
                "spin checkpoint is missing schedule_state and cannot be "
                "resumed with verified training schedule semantics")
        if info["best_model_text"] is not None:
            with open(os.path.join(output_dir, "nep_best.txt"), "w") as stream:
                stream.write(info["best_model_text"])
        elif (raw_model.spin_descriptor is not None
              and np.isfinite(best_loss)):
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
                    optimizer, lr_scheduler_mode, stage2_scheduler_factor,
                    stage2_scheduler_patience, stop_lr)
            if info["in_stage2"] and swa_model is not None:
                swa_model = AveragedModel(raw_model)
                _log("  SWA accumulator reset for the new objective/config")
            _log("  scheduler state reset for the new objective/config")
        else:
            _restore_scheduler_state(
                info, lr_scheduler, stage2_scheduler,
                require_versioned=raw_model.spin_descriptor is not None,
            )

    oc_graph_cache = None
    if oc_graph_enabled:
        configure_graph_optimizer(optimizer)
        oc_graph_cache = OCCudaGraphTrainCache(
            # The loop backend contains a data-dependent ``Tensor.any()``
            # host branch and cannot be captured.  bmm is numerically
            # identical and has a fixed launch topology for each ragged key.
            raw_model, optimizer, backend="bmm",
        )

    n_structs = data_store.n
    # has_forces / has_virial are recomputed per-epoch inside the loop using
    # the CURRENT stage's weights — so a stage-1 weight of 0 can still enable
    # computation in stage 2 (and vice versa). Do NOT latch them here.

    loss_log_path = os.path.join(output_dir, "loss.out")
    if start_epoch > 1:
        # Keep loss.out in 1:1 correspondence with the actual history: rows
        # past the checkpoint epoch belong to epochs the resumed run is
        # about to retrain.
        _trim_loss_log(loss_log_path, start_epoch - 1)
    write_header = (start_epoch == 1 or not os.path.exists(loss_log_path)
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
    # In the last third of the run a candidate best is verified by a
    # frozen-weight evaluation over the full dataset before nep_best is
    # written (see the best-save block in the loop).
    true_eval_start = (2 * num_epochs) // 3 + 1
    _log("")
    if early_stop:
        monitored = "validation loss" if valid_store is not None else "train loss"
        if early_stop <= scheduler_patience:
            _log(f"WARNING: early_stop ({early_stop}) <= scheduler_patience "
                 f"({scheduler_patience}); the LR may never decay before the "
                 f"run stops. Use early_stop > scheduler_patience.")
        _log(f"Early stopping: stop if {monitored} does not improve for "
             f"{early_stop} epochs")
    _log(
        f"Training: epochs {start_epoch}..{num_epochs}"
        f"{stage2_tag}"
    )
    _log("=" * 72)

    def _save_best():
        raw_model.save_nep_txt(
            os.path.join(output_dir, "nep_best.txt"),
            max_NN_rad, max_NN_ang)

    # Early-stopping tracker on the monitored metric (sched_loss: validation
    # loss when a valid set is present, else the epoch train loss). Reset when
    # stage 2 starts, since the loss scale changes with the stage-2 weights.
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

    # Non-finite-gradient guard flavour: with a fused CUDA optimizer the
    # skip runs GPU-side via the AMP found_inf hook (no per-step host
    # sync — the sync otherwise costs a full launch-pipeline stall every
    # step); anywhere else the classic synchronous check runs.
    async_guard = (not config["joint_rank4"] and dev.type == "cuda"
                   and optimizer.param_groups[0].get("fused", False)
                   and getattr(optimizer, "_step_supports_amp_scaling",
                               False))

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            t_epoch = time.time()
            model.train()

            # Per-epoch frame-level shuffle (i.i.d. minibatches). Seeded by
            # run_seed + epoch: distinct order every epoch, reproducible for a
            # given run_seed, and resumed runs continue along the same stream
            # (run_seed is restored from the checkpoint).
            g = torch.Generator()
            g.manual_seed(run_seed + epoch)
            perm = torch.randperm(n_structs, generator=g).tolist()

            # Metric accumulators live on the DEVICE and are fetched once
            # per epoch — per-step .item() reads would stall the launch
            # pipeline (the CPU must run a full step ahead of the GPU for
            # the streamed collate + kernel launches to hide).
            # Layout: E/F/V/stress/M/Tau sums, E/F/V/M/Tau counts, E residual.
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
                        # Crossing into stage 2 now: save a FULL end-of-
                        # stage-1 checkpoint (model + optimizer + scheduler
                        # state through epoch-1), so stage 2 can be redone
                        # from this exact point via resume_from after
                        # editing the stage2_* settings in nep.in. Then
                        # apply the stage-2 lr and reset the best losses
                        # (the stage-2 loss weights put them on a
                        # different scale).
                        _save_checkpoint(
                            os.path.join(output_dir, "checkpoint_stage1.pt"),
                            model, optimizer, lr_scheduler, epoch - 1,
                            best_loss, loss_weights=cur_loss_weights,
                            in_stage2=False, swa_model=None,
                            best_true_loss=best_true_loss, run_seed=run_seed,
                            best_valid_loss=best_valid_loss,
                            valid_info=cur_valid_info,
                            schedule_state=schedule_state,
                            early_stop_state={
                                "best": es_best, "wait": es_wait,
                                "prev_in_stage2": prev_in_stage2,
                            })
                        _log("Saved end-of-stage-1 checkpoint: "
                             "checkpoint_stage1.pt (redo stage 2 from it "
                             "via resume_from)")
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
                        # (resume = exact continuation). best losses are
                        # reset only if the loss weights changed (handled
                        # at checkpoint load).
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
            # this channel". This is recomputed every epoch so a stage-1 zero
            # weight doesn't block stage-2 computation (see stage transition
            # above) — and so pref_v=0 really skips virial compute/backward.
            has_forces = data_store.has_forces and cur_pref_f > 0
            has_virial = data_store.has_virial and cur_pref_v > 0
            has_mforces = (
                raw_model.spin_descriptor is not None
                and data_store.has_mforces and cur_pref_m > 0)
            has_tau = (
                raw_model.spin_descriptor is not None
                and data_store.has_tau and cur_pref_tau > 0)

            batch_indices = [perm[start:start + batch_size]
                             for start in range(0, n_structs, batch_size)]
            # Pair response groups with the existing ordinary optimizer steps.
            # spin3 makes the per-group total coefficient exactly equal within
            # every epoch; spin2 retains its historical modulo cycle.
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
            _tp = [0.0, 0.0]
            if _PROF and dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            _t_loop0 = time.perf_counter()
            for batch_index, batch in enumerate(_timed_iter(
                    iter_collated(data_store, batch_indices),
                    _tp, _PROF >= 2 and dev.type == "cuda")):
                batch["need_mforce"] = has_mforces
                batch["need_tau"] = has_tau

                # A generator-response group changes the objective and shape
                # independently of the ordinary ragged batch, so keep that
                # less common stage-2 path eager.  Every ordinary O3-C2 batch
                # replays descriptor -> E/F/V/MF/Tau -> backward -> Adam.
                if oc_graph_cache is not None and not response_batches:
                    graph_step = oc_graph_cache.step(
                        batch,
                        weights=(
                            cur_pref_e, cur_pref_f, cur_pref_v,
                            cur_pref_m, cur_pref_tau,
                        ),
                        l1_coeff=0.0,
                        l2_coeff=0.0,
                        max_grad_norm=max_grad_norm,
                        need_forces=has_forces,
                        need_virial=has_virial,
                        need_mforce=has_mforces,
                        need_tau=has_tau,
                    )
                    graph_metrics = graph_step.metrics
                    graph_ok = (1.0 - graph_step.found_inf).to(torch.float64)
                    acc += graph_ok * torch.stack([
                        graph_metrics["sum_le"].double(),
                        graph_metrics["sum_lf"].double(),
                        graph_metrics["sum_lv"].double(),
                        graph_metrics["sum_ls"].double(),
                        graph_metrics["sum_lm"].double(),
                        graph_metrics["sum_lt"].double(),
                        graph_metrics["count_e"].double(),
                        graph_metrics["count_f"].double(),
                        graph_metrics["count_v"].double(),
                        graph_metrics["count_m"].double(),
                        graph_metrics["count_t"].double(),
                        graph_metrics["sum_e_resid"].double(),
                    ])
                    max_gn_t = torch.maximum(
                        max_gn_t,
                        torch.nan_to_num(graph_step.grad_norm, 0.0, 0.0, 0.0))
                    n_bad_t += graph_step.found_inf.to(dtype)
                    if (in_stage2 and swa_model is not None
                            and epoch >= swa_start):
                        swa_model.update_parameters(raw_model)
                    continue

                if use_autograd_forces:
                    result = compute_props(
                        batch["rij_rad"], batch["rij_ang"],
                        batch["pair_i_rad"], batch["pair_j_rad"],
                        batch["pair_i_ang"], batch["pair_j_ang"],
                        batch["atom_types"], batch["N"],
                        batch["struct_idx"], batch["num_structures"],
                        need_forces=has_forces, need_virial=has_virial,
                        backend=backend, spins=batch.get("spins"),
                        need_mforce=has_mforces,
                        need_tau=has_tau,
                        rij_spin=batch.get("rij_spin"),
                        pi_spin=batch.get("pair_i_spin"),
                        pj_spin=batch.get("pair_j_spin"))
                else:
                    result = compute_props_cached(
                        batch, need_forces=has_forces, need_virial=has_virial,
                        backend=train_backend)

                e_pa_pred = result["Etot"] / batch["natoms"]
                e_pa_ref = batch["energy"] / batch["natoms"]
                loss = torch.tensor(0.0, dtype=dtype, device=dev)
                # Loss + metrics are built BRANCHLESSLY (masked weighted
                # sums, host-side has_* flags) and kept as device scalars:
                # boolean-mask indexing or `.any()` here would drain the
                # compute stream mid-step. The scalars are fetched together
                # AFTER the one unavoidable sync (the grad-norm guard), where
                # the reads are free. Same math as the old masked-select
                # form: sum(d^2 * mask) / sum(mask) == mean over selected.
                m_le = m_resid = m_lf = m_lv = m_ls = None
                m_lm = m_lt = None
                if batch["has_e"]:
                    emf = batch["energy_mask"].to(dtype)
                    ne_b = emf.sum()
                    de = (e_pa_pred - e_pa_ref) * emf
                    loss = loss + cur_pref_e * ((de ** 2).sum() / ne_b)
                    m_le = (de ** 2).sum()
                    # Signed residual for the analytical b1 update (folded
                    # into this pass — no extra forward).
                    m_resid = de.sum()

                if has_forces and batch["has_f"]:
                    fmf = batch["force_mask"].to(dtype).unsqueeze(-1)
                    nf_b = fmf.sum()
                    df = (result["forces"] - batch["forces"]) * fmf
                    loss = loss + cur_pref_f * ((df ** 2).sum() / (3.0 * nf_b))
                    m_lf = (df ** 2).sum() / 3.0

                if (has_virial and "virial" in result and batch["has_v"]
                        and batch["virial"].shape[1] == 9):
                    vmf = batch["virial_mask"].to(dtype).unsqueeze(-1)
                    nv_b = vmf.sum()
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
                    v_ref_pa = virial6_for_loss(
                        batch["virial"], raw_model.spin_mode
                    ) / na
                    v_pred_pa = _v_pred_pa(result)
                    dv = (v_pred_pa - v_ref_pa) * vmf
                    loss = loss + cur_pref_v * ((dv ** 2).sum() / (6.0 * nv_b))
                    m_lv = (dv ** 2).sum() / 6.0
                    # Stress RMSE (eV/A**3): convert the same diff using
                    # per-frame (natoms/volume). Sign cancels under MSE.
                    # clamp: masked-out frames may carry volume 0 — their
                    # contribution is already zeroed by the mask factor.
                    scale = (batch["natoms"]
                             / batch["volumes"].clamp(min=1e-9)).unsqueeze(-1)
                    m_ls = ((dv * scale) ** 2).sum() / 6.0

                if has_mforces and batch["has_m"]:
                    dm = result["mforces"] - batch["mforces"]
                    m_mask = (
                        batch["mforce_mask"]
                        & raw_model.spin_dof_type_active[
                            batch["atom_types"]])
                    mmf = m_mask.to(dtype).unsqueeze(-1)
                    nm_b = mmf.sum()
                    dm = dm * mmf
                    loss = loss + cur_pref_m * (
                        dm.square().sum()
                        / (3 * nm_b.clamp_min(1.0)))
                    m_lm = dm.square().sum() / 3

                if has_tau and batch["has_tau"]:
                    t_mask = (
                        batch["tau_mask"]
                        & raw_model.spin_dof_type_active[
                            batch["atom_types"]])
                    tmf = t_mask.to(dtype).unsqueeze(-1)
                    nt_b = tmf.sum()
                    dtau = (result["tau"] - batch["tau"]) * tmf
                    tau_dof = 2 if raw_model.spin_mode == 3 else 3
                    loss = loss + cur_pref_tau * (
                        dtau.square().sum()
                        / (tau_dof * nt_b.clamp_min(1.0)))
                    m_lt = dtau.square().sum() / tau_dof

                joint_response = []
                if response_batches:
                    for response_idx, response_weight in response_batches[
                            batch_index]:
                        response_batch = response_store.collate(response_idx)
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

                if max_grad_norm > 0:
                    gn_t = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm)
                else:
                    gn_t = torch.sqrt(sum(
                        p.grad.norm()**2 for p in raw_model.parameters()
                        if p.grad is not None))

                if async_guard:
                    # Non-finite-gradient protection WITHOUT a host sync:
                    # the fused Adam kernel skips the whole update GPU-side
                    # when ``found_inf`` is nonzero (the AMP mechanism), so
                    # the parameters are protected exactly as with the
                    # synchronous skip. Divergence from the sync path only
                    # on the (pathological, ~never) bad step itself: SWA
                    # still averages the (unchanged) weights, and that
                    # step's metrics are excluded via the same flag.
                    bad = (~torch.isfinite(gn_t)).to(dtype)
                    optimizer.found_inf = bad
                    optimizer.step()
                    n_bad_t += bad
                    ok_f = (1.0 - bad).to(torch.float64)
                else:
                    gn = float(gn_t)
                    if not np.isfinite(gn):
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    optimizer_step(
                        optimizer, raw_model, batch,
                        (cur_pref_e, cur_pref_f, cur_pref_v, cur_pref_m, cur_pref_tau),
                        train_backend, schedule_state,
                        **({"responses": joint_response} if joint_response else {}))
                    ok_f = 1.0

                if (in_stage2 and swa_model is not None
                        and epoch >= swa_start):
                    swa_model.update_parameters(raw_model)

                # Device-side metric accumulation — one fused add per step,
                # fetched once per epoch. Bad steps contribute zero via ok_f
                # (consistently for sums AND counts).
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

            if _PROF:
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                _t_train_wall = time.perf_counter() - _t_loop0

            # One epoch-level fetch of every device accumulator (the only
            # metric sync of the epoch).
            (sum_le, sum_lf, sum_lv, sum_ls, sum_lm, sum_lt,
             n_e_f, n_f_f, n_v_f, n_m_f, n_t_f, sum_e_resid) = acc.tolist()
            sum_e_structs = int(round(n_e_f))
            sum_f_atoms = int(round(n_f_f))
            sum_v_structs = int(round(n_v_f))
            sum_m_atoms = int(round(n_m_f))
            sum_t_atoms = int(round(n_t_f))
            max_gn = float(max_gn_t)
            n_bad = int(n_bad_t)
            if n_bad:
                _log(f"  warning: {n_bad} step(s) skipped this epoch "
                     f"(non-finite gradient norm)")


            # Analytical b1 (GPUMD-style), folded into the training pass: b1
            # absorbs this epoch's mean per-atom energy residual. Updated AFTER
            # the batch loop but BEFORE the best-model eval / nep_best save, so
            # the saved weights and offset stay consistent. ``b1`` is not a
            # gradient parameter (see the optimizer exclusion above).
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
            # Output units: eV/atom (E, V), eV/A (F), GPa (stress).
            rmse_e = np.sqrt(mse_e)
            rmse_f = np.sqrt(mse_f)
            rmse_v = np.sqrt(mse_v)
            rmse_s_gpa = np.sqrt(mse_s) * EV_PER_A3_TO_GPa
            rmse_m = np.sqrt(mse_m)
            rmse_t = np.sqrt(mse_t)

            # Spin losses contain derivatives with respect to the magnetic
            # coordinates. Evaluate the frozen end-of-epoch weights on the
            # complete training set so logging, scheduling and nep_best all
            # use the same model state rather than moving minibatch averages.
            if raw_model.spin_descriptor is not None:
                (avg_loss, rmse_e, rmse_f, rmse_v, rmse_s,
                 rmse_m, rmse_t) = _evaluate_true_loss(
                    data_store, eval_batch_size, raw_model,
                    compute_props, compute_props_cached,
                    use_autograd_forces, train_backend,
                    cur_pref_e, cur_pref_f, cur_pref_v,
                    cur_pref_m, cur_pref_tau, dtype, dev)
                rmse_s_gpa = rmse_s * EV_PER_A3_TO_GPa

            response_metric = None
            if in_stage2 and response_store is not None:
                response_metric = evaluate_generator_curves(
                    response_store, eval_batch_size, raw_model,
                    train_backend)
                avg_loss += response_loss_weight * response_metric

            # Validation: frozen-weight full pass every epoch (b1 already
            # train-fitted by the analytical update above; never re-fitted on
            # validation data). This loss drives the plateau scheduler and
            # the best-model choice below — the anti-overfitting signal.
            _t_v0 = time.perf_counter()
            valid_loss = None
            valid_response_metric = None
            if valid_store is not None:
                (valid_loss, v_rmse_e, v_rmse_f, v_rmse_v, v_rmse_s,
                 v_rmse_m, v_rmse_t) = \
                    _evaluate_valid_loss(
                        valid_store, eval_batch_size, raw_model,
                        compute_props, compute_props_cached,
                        use_autograd_forces, train_backend,
                        cur_pref_e, cur_pref_f, cur_pref_v,
                        cur_pref_m, cur_pref_tau, dtype, dev)
                if in_stage2 and valid_response_store is not None:
                    valid_response_metric = evaluate_generator_curves(
                        valid_response_store, eval_batch_size, raw_model,
                        train_backend,
                        normalization_scale=(
                            response_store.generator_curve_scale))
                    valid_loss += (
                        response_loss_weight * valid_response_metric)
            if _PROF and dev.type == "cuda":
                torch.cuda.synchronize()
            _t_valid = time.perf_counter() - _t_v0
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

            # Early stopping on the monitored metric. Improvement uses the same
            # 1e-4 relative threshold as ReduceLROnPlateau, so "no improvement"
            # means the same thing to both. The counter resets when stage 2
            # begins (the loss scale shifts with the stage-2 weights). In DDP
            # every rank sees the identical (all-reduced) sched_loss, so all
            # ranks reach the same decision and stop together.
            #
            # Early stop is per-STAGE (MACE-style): if stage 1 plateaus while
            # a stage 2 is configured but not yet started, the run jumps into
            # stage 2 at the next epoch instead of terminating — only a
            # plateau in the final stage ends the run. The advanced
            # start_stage2 is saved in the checkpoint so a resumed run stays
            # in stage 2 (nep.in's start_stage2 would say otherwise).
            stop_now = False
            if early_stop:
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
                            if valid_response_metric is not None else " nan")
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
                if raw_model.spin_descriptor is not None:
                    if valid_store.has_mforces and cur_pref_m > 0:
                        valid_str += f" M {v_rmse_m:.5f}"
                    if valid_store.has_tau and cur_pref_tau > 0:
                        valid_str += f" T {v_rmse_t:.5f}"
                if valid_response_metric is not None:
                    valid_str += f" Gcurve {valid_response_metric:.4f}"
            line = (f"{stage_str}Epoch {epoch:4d} | loss {sched_loss:.4e} | "
                    f"E {rmse_e:.5f} eV/atom | F {rmse_f:.5f} eV/A"
                    f"{v_str}{spin_str}{valid_str} | gnorm {max_gn:.1f} | "
                    f"lr {cur_lr:.2e} | {dt:.1f}s")
            if oc_graph_cache is not None:
                graph_stats = oc_graph_cache.stats()
                line += (
                    f" | graph {graph_stats['hit_rate'] * 100:.1f}% hit "
                    f"({graph_stats['entries']}/{graph_stats['max_entries']}, "
                    f"capture {graph_stats['total_capture_ms']:.1f} ms)"
                )
            if epoch % print_interval == 0 or epoch == 1:
                _log(line)
            else:
                _out_log_file.write(line + "\n")
                _out_log_file.flush()

            # Best-model bookkeeping.
            # With a validation set, nep_best is simply the epoch with the
            # lowest validation loss — valid_loss is already a frozen-weight
            # true evaluation (computed every epoch), so the noisy-proxy /
            # true-eval two-step below is unnecessary. best_loss still tracks
            # the train loss for the log / checkpoint.
            #
            # Without validation: avg_loss averages over weights that keep
            # moving within the epoch, so it is a noisy proxy for the
            # end-of-epoch weights actually saved. Early in the run that is
            # fine (the model improves much faster than the noise). In the
            # last third — where best really gets decided — a new avg_loss
            # minimum only NOMINATES the weights: they become nep_best only
            # if a frozen-weight pass over the full dataset (same masks and
            # averaging as the screen numbers) beats the best true loss so
            # far. The final epoch is always evaluated, so nep_best can
            # never end up worse than nep_final.
            new_min = avg_loss < best_loss
            if new_min:
                best_loss = avg_loss
            if valid_store is not None:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    _save_best()
            elif (epoch < true_eval_start
                  or raw_model.spin_descriptor is not None):
                if new_min:
                    if raw_model.spin_descriptor is not None:
                        best_true_loss = avg_loss
                    _save_best()
            elif new_min or epoch == num_epochs or stop_now:
                t_loss, _te, _tf, _tv, _ts, _tm, _tt = _evaluate_true_loss(
                    data_store, eval_batch_size, raw_model,
                    compute_props, compute_props_cached,
                    use_autograd_forces, train_backend,
                    cur_pref_e, cur_pref_f, cur_pref_v,
                    cur_pref_m, cur_pref_tau, dtype, dev)
                if t_loss < best_true_loss:
                    best_true_loss = t_loss
                    _save_best()

            if epoch % checkpoint_interval == 0 or epoch == num_epochs or stop_now:
                _save_checkpoint(
                    ckpt_path, model, optimizer,
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
                    },
                    start_stage2=start_stage2)

            # Interim predict — overwrites the same output files, so users can
            # refresh the parity plot live. Runs on the CURRENT-epoch weights
            # (not nep_best) so the predict loss matches what was just logged
            # for this epoch: it should fall between this epoch's and the next
            # epoch's displayed loss (current weights = end-of-epoch, whereas
            # the screen average covers weights that were still improving
            # throughout the epoch).
            # Skip on the final epoch — the end-of-training predict (below)
            # immediately overwrites these files with the final-epoch result.
            if (prediction_interval > 0
                    and epoch % prediction_interval == 0
                    and epoch != num_epochs):
                # Silent interim predict — reuses data_store's preprocessed
                # neighbor lists + basis (no xyz re-read, no recompute).
                predict_from_store(raw_model, data_store, output_dir,
                                   batch_size=eval_batch_size, backend=backend,
                                   verbose=False)
                if valid_store is not None:
                    predict_from_store(raw_model, valid_store, output_dir,
                                       batch_size=eval_batch_size,
                                       backend=backend, verbose=False,
                                       suffix="test")

            if _PROF:
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                    _pa = torch.cuda.max_memory_allocated() / 2**30
                    _pr = torch.cuda.max_memory_reserved() / 2**30
                else:
                    _pa = _pr = 0.0
                _t_tail = (time.perf_counter() - _t_v0) - _t_valid
                _log(f"  [prof] train {_t_train_wall:6.2f}s "
                     f"(data-wait {_tp[0]:6.2f} step-cpu {_tp[1]:6.2f}) | "
                     f"valid {_t_valid:5.2f}s tail {_t_tail:5.2f}s | "
                     f"peak alloc {_pa:.2f} res {_pr:.2f} GiB")

            if stop_now:
                _log(f"Early stop: {monitored} did not improve for "
                     f"{early_stop} epochs (stopped at epoch {epoch}/"
                     f"{num_epochs}).")
                break
    finally:
        if loss_log is not None:
            loss_log.close()

    # Final-epoch model (what the current weights actually are). b1 is already
    # exact for these weights: the final epoch (epoch == num_epochs) always
    # runs the best-model eval, which solves and sets the optimal b1. Do NOT
    # recompute it here — that would give nep_final a different (lower-loss) b1
    # than the value the best-model comparison used, which could make nep_final
    # beat the saved nep_best.
    raw_model.save_nep_txt(os.path.join(output_dir, "nep_final.txt"),
                           max_NN_rad, max_NN_ang)
    # SWA-averaged model (only when user opted in and stage 2 ran).
    if swa_model is not None and int(swa_model.n_averaged.item()) == 0:
        _log("SWA window never reached (run ended before swa_start) — "
             "nep_average.txt not written")
        swa_model = None
    if swa_model is not None:
        swa_state = swa_model.module.state_dict()
        # Keep a copy of the final-epoch weights so we can restore them
        # after saving SWA — the end-of-training predict below must see
        # final weights, not SWA-averaged ones.
        final_state = {k: v.clone() for k, v in raw_model.state_dict().items()}
        raw_model.load_state_dict(swa_state)
        recompute_b1_shift(
            raw_model, data_store, eval_batch_size, backend
        )
        raw_model.save_nep_txt(os.path.join(output_dir, "nep_average.txt"),
                               max_NN_rad, max_NN_ang)
        raw_model.load_state_dict(final_state)
        _log("SWA model saved to nep_average.txt (b1 re-solved for the "
             "averaged weights)")

    train_time = time.time() - train_t0
    h, rem = divmod(train_time, 3600)
    m_, s = divmod(rem, 60)
    if valid_store is not None:
        _log(f"\nDone. Best validation loss (nep_best): "
             f"{best_valid_loss:.6e}")
    else:
        _log(f"\nDone. Best loss: {best_loss:.6e}")
    _log(f"Training time: {int(h):02d}:{int(m_):02d}:{s:04.1f}")

    # End-of-training predict reuses the in-memory data_store (no xyz re-read)
    # and the final-epoch weights in raw_model (no model-file round-trip).
    _log("\nRunning prediction on training set (final-epoch model)...")
    pred_t0 = time.time()
    predict_from_store(raw_model, data_store, output_dir,
                       batch_size=eval_batch_size, backend=backend,
                       verbose=False)
    if valid_store is not None:
        predict_from_store(raw_model, valid_store, output_dir,
                           batch_size=eval_batch_size, backend=backend,
                           verbose=False, suffix="test")
    _log(f"  Prediction time: {time.time() - pred_t0:.1f}s")

    total_time = time.time() - total_t0
    h, rem = divmod(total_time, 3600)
    m_, s = divmod(rem, 60)
    _log(f"\nTotal time (data + train + predict): "
         f"{int(h):02d}:{int(m_):02d}:{s:04.1f}")
    _log(f"Output: {output_dir}/")
    _out_log_file.close()
