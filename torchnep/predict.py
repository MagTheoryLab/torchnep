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
Full-dataset prediction for NEP models.

Pipeline:
  1. read_xyz
  2. preprocess_structures(...)  — multi-process neighbor lists (CPU)
  3. concatenate all structures into flat tensors uploaded to the GPU once
  4. batched compute_batch loop (basis recomputed per batch, no host transfer)
  5. vectorised numpy.savetxt for outputs (per-atom virial matches GPUMD)
"""

import os
import time
import torch
import numpy as np

from .nep import NEPCalculator
from .data import read_xyz
from . import ops


# GPUMD writes this sentinel into the reference column of virial_train.out /
# stress_train.out for structures that carry no reference virial (see GPUMD
# src/main_nep/structure.cu: ``structure.virial[m] = -1e6``). For stress it
# also skips the unit conversion when the reference is below -1e5 (fitness.cu,
# ``if (ref_value > -1e5)``). We mirror both so a missing virial reads as
# -1e6 (not NaN) and the files stay drop-in comparable for third-party tools.
_MISSING_VIRIAL = -1e6
_MISSING_VIRIAL_CUTOFF = -1e5


def _stress_from_virial(virial_pa, nat_col, vol_col, keep_missing):
    """Per-atom virial (eV/atom) -> stress (GPa), matching GPUMD's sign:
    ``stress = +virial_total / V`` (so stress shares virial's sign, unlike a
    physical Cauchy stress).

    ``virial_pa``   : (n, 6) per-atom virial.
    ``nat_col``     : (n, 1) atom counts.
    ``vol_col``     : (n, 1) cell volumes (<=0 treated as 1 to avoid div-by-0).
    ``keep_missing``: if True, rows holding the -1e6 missing-virial sentinel are
                      passed through unscaled (GPUMD's reference-column rule).
    """
    from .constants import EV_PER_A3_TO_GPa
    vol_safe = np.where(vol_col > 0, vol_col, 1.0)
    stress = virial_pa * (nat_col / vol_safe * EV_PER_A3_TO_GPa)
    if keep_missing:
        stress = np.where(virial_pa > _MISSING_VIRIAL_CUTOFF, stress, virial_pa)
    return stress


def _virial9_to_6(v9):
    r"""Re-order length-9 row-major virial (xx,xy,xz,yx,yy,yz,zx,zy,zz) into
    the GPUMD 6-vector (xx,yy,zz,xy,yz,zx).

    Picks single-triangular components to match GPUMD's per-atom virial
    convention (src/main_nep/nep.cu: s_virial_xy = -r[0]*f[1], s_virial_yz
    = -r[1]*f[2], s_virial_zx = -r[2]*f[0]). GPUMD does not average with
    the symmetric partner — it stores only one entry of each off-diagonal
    pair during force accumulation. Per-FRAME totals are still symmetric
    because the sum \Sigma -r_\alpha f_\beta over directed pairs equals
    \Sigma -r_\beta f_\alpha by Newton's 3rd law, so this choice is
    numerically equivalent to averaging at the per-frame output level but
    gives cheaper bit-identical match with GPUMD outputs."""
    out = np.empty((v9.shape[0], 6), dtype=v9.dtype)
    out[:, 0] = v9[:, 0]   # xx = -r_x f_x
    out[:, 1] = v9[:, 4]   # yy = -r_y f_y
    out[:, 2] = v9[:, 8]   # zz = -r_z f_z
    out[:, 3] = v9[:, 1]   # xy = -r_x f_y
    out[:, 4] = v9[:, 5]   # yz = -r_y f_z
    out[:, 5] = v9[:, 6]   # zx = -r_z f_x
    return out


def predict_dataset(
    model_file: str,
    xyz_file: str,
    output_dir: str = ".",
    dtype: str = "float64",
    device: str = None,
    batch_size: int = 1000,
    verbose: bool = True,
    backend: str = "auto",
    energy_key: str = "energy",
    output_descriptor: int = 0,
):
    """Run batched prediction on a full dataset and save GPUMD-format outputs.

    Outputs (per-atom for energy and virial; per-atom raw for forces):
      - energy_train.out:  e_pred  e_target              (eV/atom, per frame)
      - force_train.out:   fx fy fz  fx_t fy_t fz_t      (eV/A, per atom)
      - virial_train.out:  xx yy zz xy yz zx (pred, ref) (eV/atom, per frame)
      - descriptor.out (only when ``output_descriptor != 0``):
          mode 1 — per-frame averaged scaled descriptor, one row per frame
          mode 2 — per-atom scaled descriptor, one row per atom
        Matches GPUMD's ``output_descriptor`` / ``descriptor.out`` schema.

    The format mirrors GPUMD's *_train.out files, so the two can be diffed
    column by column.

    Parameters
    ----------
    output_descriptor : int
        0 — disabled (default).
        1 — write per-frame averaged ``q * q_scaler`` to descriptor.out.
        2 — write per-atom ``q * q_scaler`` to descriptor.out.

    ``backend`` in {"auto", "loop", "bmm"} — see
    ``torchnep.ops.resolve_backend``.
    """
    if device is None:
        # cuda probe also catches ROCm (PyTorch-HIP uses the cuda namespace).
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dt = torch.float64 if dtype == "float64" else torch.float32
    np_dtype = np.float64 if dtype == "float64" else np.float32

    def _log(msg):
        if verbose:
            print(msg, flush=True)

    t_total = time.time()
    calc = NEPCalculator(model_file, dtype=dt, device=device)
    rc_rad, rc_ang = calc.rc_radial, calc.rc_angular
    basis_r, basis_a = calc.basis_size_radial, calc.basis_size_angular
    l_max_3b = calc.l_max_3b
    num_lm = calc.num_lm

    # Resolve "auto" now that num_types is known (>=8 -> "bmm", else "loop").
    backend = ops.resolve_backend(backend, num_types=calc.num_types)
    _log(f"  backend: {backend}")

    # 1) Read xyz
    t0 = time.time()
    frames = read_xyz(xyz_file, energy_key=energy_key)
    if calc.spin_mode and any("spins" not in frame for frame in frames):
        raise ValueError(
            "spin model requires a spin/moment vector for every atom")
    n_struct = len(frames)
    _log(f"  read_xyz:    {time.time() - t0:5.1f}s   ({n_struct} frames, "
         f"energy label: {energy_key})")

    # 2) Multi-process neighbor-list construction
    from .train import preprocess_structures  # local import: avoid cycle

    t0 = time.time()
    pp_config = {
        "cutoff_radial": rc_rad,
        "cutoff_angular": rc_ang,
        "type_names": calc.type_names,
        "spin_mode": calc.spin_mode,
        "spin_cutoff": calc.spin_cutoff,
    }
    structures = preprocess_structures(frames, pp_config, np_dtype)
    del frames
    _log(f"  neighbors:   {time.time() - t0:5.1f}s")

    # 3) Concatenate everything once and move raw data to the GPU.
    t0 = time.time()
    natoms_arr = np.asarray([s["natoms"] for s in structures], dtype=np.int64)
    volumes_arr = np.asarray([s.get("volume", 0.0) for s in structures],
                             dtype=np.float64)
    nrad_arr = np.asarray([len(s["pair_i_rad"]) for s in structures],
                          dtype=np.int64)
    nang_arr = np.asarray([len(s["pair_i_ang"]) for s in structures],
                          dtype=np.int64)
    nspin_arr = (
        np.asarray([len(s["pair_i_spin"]) for s in structures],
                   dtype=np.int64)
        if calc.spin_mode else None
    )
    nat_cum = np.concatenate([[0], np.cumsum(natoms_arr)])
    nrad_cum = np.concatenate([[0], np.cumsum(nrad_arr)])
    nang_cum = np.concatenate([[0], np.cumsum(nang_arr)])
    nspin_cum = (
        np.concatenate([[0], np.cumsum(nspin_arr)])
        if calc.spin_mode else None
    )
    N_atoms_total = int(nat_cum[-1])

    # Pair indices are global (atom positions in the concatenation), so a
    # per-batch slice only needs to subtract the batch's first-atom offset.
    all_at = np.concatenate([s["atom_types"] for s in structures])
    all_pi_r = np.concatenate(
        [s["pair_i_rad"] + nat_cum[i] for i, s in enumerate(structures)])
    all_pj_r = np.concatenate(
        [s["pair_j_rad"] + nat_cum[i] for i, s in enumerate(structures)])
    all_rij_r = np.concatenate([s["rij_rad"] for s in structures])
    all_pi_a = np.concatenate(
        [s["pair_i_ang"] + nat_cum[i] for i, s in enumerate(structures)])
    all_pj_a = np.concatenate(
        [s["pair_j_ang"] + nat_cum[i] for i, s in enumerate(structures)])
    all_rij_a = np.concatenate([s["rij_ang"] for s in structures])
    if calc.spin_mode:
        all_pi_s = np.concatenate(
            [s["pair_i_spin"] + nat_cum[i]
             for i, s in enumerate(structures)])
        all_pj_s = np.concatenate(
            [s["pair_j_spin"] + nat_cum[i]
             for i, s in enumerate(structures)])
        all_rij_s = np.concatenate([s["rij_spin"] for s in structures])
        all_spins = np.concatenate([s["spins"] for s in structures])

    energy_ref = np.array(
        [s["energy"] if s.get("energy") is not None else np.nan
         for s in structures], dtype=np.float64)

    has_forces_global = any(s.get("forces") is not None for s in structures)
    forces_ref = None
    if has_forces_global:
        forces_ref = np.full((N_atoms_total, 3), np.nan, dtype=np.float64)
        for i, s in enumerate(structures):
            f = s.get("forces")
            if f is not None:
                forces_ref[nat_cum[i]:nat_cum[i + 1]] = \
                    np.asarray(f).reshape(-1, 3)

    has_mforces_global = any(
        s.get("mforces") is not None for s in structures)
    mforces_ref = None
    tau_ref = None
    if calc.spin_mode:
        mforces_ref = np.full(
            (N_atoms_total, 3), np.nan, dtype=np.float64)
        tau_ref = np.full(
            (N_atoms_total, 3), np.nan, dtype=np.float64)
        for i, s in enumerate(structures):
            magnetic_force = s.get("mforces")
            if magnetic_force is not None:
                a_lo, a_hi = nat_cum[i], nat_cum[i + 1]
                mforces_ref[a_lo:a_hi] = np.asarray(
                    magnetic_force).reshape(-1, 3)
                tau_ref[a_lo:a_hi] = np.asarray(s["tau"]).reshape(-1, 3)

    has_virial_global = any(s.get("virial") is not None for s in structures)
    virial_ref = np.full((n_struct, 6), _MISSING_VIRIAL, dtype=np.float64)
    if has_virial_global:
        # Per-atom virial, to match GPUMD's *_train.out columns. Off-diagonal
        # pairs are picked the same way as the prediction, so the two columns
        # are computed on the exact same definition.
        for i, s in enumerate(structures):
            v = s.get("virial")
            if v is None:
                continue
            v = np.asarray(v).flatten()
            inv_n = 1.0 / float(s["natoms"])
            if v.size == 9:
                # Single-triangular pick to match GPUMD (see _virial9_to_6).
                # Input XYZ virial is symmetric (DFT), so xy=yx etc., making
                # this choice numerically equal to averaging.
                virial_ref[i] = [
                    v[0] * inv_n,   # xx
                    v[4] * inv_n,   # yy
                    v[8] * inv_n,   # zz
                    v[1] * inv_n,   # xy
                    v[5] * inv_n,   # yz
                    v[6] * inv_n,   # zx
                ]
            elif v.size >= 6:
                virial_ref[i] = v[:6] * inv_n

    del structures  # free CPU memory

    at_gpu    = torch.from_numpy(all_at).to(device=device)
    pi_r_gpu  = torch.from_numpy(all_pi_r).to(device=device)
    pj_r_gpu  = torch.from_numpy(all_pj_r).to(device=device)
    rij_r_gpu = torch.from_numpy(all_rij_r).to(device=device, dtype=dt)
    pi_a_gpu  = torch.from_numpy(all_pi_a).to(device=device)
    pj_a_gpu  = torch.from_numpy(all_pj_a).to(device=device)
    rij_a_gpu = torch.from_numpy(all_rij_a).to(device=device, dtype=dt)
    if calc.spin_mode:
        pi_s_gpu = torch.from_numpy(all_pi_s).to(device=device)
        pj_s_gpu = torch.from_numpy(all_pj_s).to(device=device)
        rij_s_gpu = torch.from_numpy(all_rij_s).to(device=device, dtype=dt)
        spins_gpu = torch.from_numpy(all_spins).to(device=device, dtype=dt)
    natoms_gpu = torch.from_numpy(natoms_arr).to(device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    _log(f"  upload:      {time.time() - t0:5.1f}s")
    del (all_at, all_pi_r, all_pj_r, all_rij_r,
         all_pi_a, all_pj_a, all_rij_a)
    if calc.spin_mode:
        del all_pi_s, all_pj_s, all_rij_s, all_spins

    # 4) Batched compute loop ------------------------------------------------
    e_pred_arr = np.empty(n_struct, dtype=np.float64)
    # Predictions are independent of whether the input carries reference
    # labels.  In particular, an unlabeled inference XYZ must still produce
    # the model force rather than a zero-filled placeholder.
    f_pred_arr = np.empty((N_atoms_total, 3), dtype=np.float64)
    v_pred_arr = np.empty((n_struct, 6), dtype=np.float64)
    m_pred_arr = (
        np.empty((N_atoms_total, 3), dtype=np.float64)
        if calc.spin_mode else None)
    t_pred_arr = (
        np.empty((N_atoms_total, 3), dtype=np.float64)
        if calc.spin_mode else None)

    # Descriptor buffer (only allocated when requested).
    #   mode 1 -> (n_struct, dim)         per-frame averaged scaled descriptor
    #   mode 2 -> (N_atoms_total, dim)    per-atom scaled descriptor
    d_pred_arr = None
    if output_descriptor == 1:
        d_pred_arr = np.empty((n_struct, calc.dim), dtype=np.float64)
    elif output_descriptor == 2:
        d_pred_arr = np.empty((N_atoms_total, calc.dim), dtype=np.float64)

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n_struct, batch_size):
            end = min(start + batch_size, n_struct)
            B = end - start
            a_lo, a_hi = int(nat_cum[start]), int(nat_cum[end])
            r_lo, r_hi = int(nrad_cum[start]), int(nrad_cum[end])
            g_lo, g_hi = int(nang_cum[start]), int(nang_cum[end])
            N = a_hi - a_lo

            atom_types = at_gpu[a_lo:a_hi]
            pi_r = pi_r_gpu[r_lo:r_hi] - a_lo
            pj_r = pj_r_gpu[r_lo:r_hi] - a_lo
            rij_r = rij_r_gpu[r_lo:r_hi]
            pi_a = pi_a_gpu[g_lo:g_hi] - a_lo
            pj_a = pj_a_gpu[g_lo:g_hi] - a_lo
            rij_a = rij_a_gpu[g_lo:g_hi]
            struct_idx = torch.repeat_interleave(
                torch.arange(B, device=device, dtype=torch.long),
                natoms_gpu[start:end])

            dr = torch.norm(rij_r, dim=-1)
            fk_r, fkp_r = ops.chebyshev_basis_and_deriv(dr, rc_rad, basis_r)
            d12inv_r = 1.0 / dr.clamp(min=1e-10)

            if rij_a.shape[0] > 0:
                da = torch.norm(rij_a, dim=-1)
                fk_a, fkp_a = ops.chebyshev_basis_and_deriv(
                    da, rc_ang, basis_a)
                d12inv_a = 1.0 / da.clamp(min=1e-10)
                blm = ops.angular_basis(
                    rij_a[:, 0] * d12inv_a,
                    rij_a[:, 1] * d12inv_a,
                    rij_a[:, 2] * d12inv_a,
                    l_max_3b)
            else:
                fk_a = torch.zeros(0, basis_a + 1, dtype=dt, device=device)
                fkp_a = torch.zeros(0, basis_a + 1, dtype=dt, device=device)
                d12inv_a = torch.zeros(0, dtype=dt, device=device)
                blm = torch.zeros(0, num_lm, dtype=dt, device=device)

            batch = {
                "N": N, "num_structures": B,
                "atom_types": atom_types, "struct_idx": struct_idx,
                "pair_i_rad": pi_r, "pair_j_rad": pj_r, "rij_rad": rij_r,
                "fk_rad": fk_r, "fkp_rad": fkp_r, "d12inv_rad": d12inv_r,
                "pair_i_ang": pi_a, "pair_j_ang": pj_a, "rij_ang": rij_a,
                "fk_ang": fk_a, "fkp_ang": fkp_a, "d12inv_ang": d12inv_a,
                "blm": blm,
            }
            if calc.spin_mode:
                s_lo = int(nspin_cum[start])
                s_hi = int(nspin_cum[end])
                batch.update({
                    "pair_i_spin": pi_s_gpu[s_lo:s_hi] - a_lo,
                    "pair_j_spin": pj_s_gpu[s_lo:s_hi] - a_lo,
                    "rij_spin": rij_s_gpu[s_lo:s_hi],
                    "spins": spins_gpu[a_lo:a_hi],
                })
            result = calc.compute_batch(batch, backend=backend)

            Etot = result["Etot"]
            v_per_frame = torch.zeros(B, 9, dtype=dt, device=device)
            v_per_frame.scatter_add_(
                0, struct_idx.unsqueeze(-1).expand(-1, 9), result["virial"])

            # Single H2D copy per batch for the per-frame outputs
            Etot_np = Etot.cpu().numpy()
            v_np = v_per_frame.cpu().numpy()
            nat_slice = natoms_arr[start:end].astype(np.float64)
            e_pred_arr[start:end] = Etot_np / nat_slice
            v_pred = _virial9_to_6(v_np) / nat_slice[:, None]
            v_pred_arr[start:end] = v_pred

            f_pred_arr[a_lo:a_hi] = result["forces"].cpu().numpy()
            if m_pred_arr is not None:
                m_pred_arr[a_lo:a_hi] = result["mforces"].cpu().numpy()
                t_pred_arr[a_lo:a_hi] = result["tau"].cpu().numpy()

            if output_descriptor:
                # ``compute_batch`` already returns ``q * q_scaler``.
                desc_np = result["descriptor"].cpu().numpy()
                if output_descriptor == 2:
                    d_pred_arr[a_lo:a_hi] = desc_np
                else:
                    # Per-frame mean: scatter-sum onto frame index then /Na.
                    sums = np.zeros((B, calc.dim), dtype=np.float64)
                    si_np = struct_idx.cpu().numpy()
                    np.add.at(sums, si_np, desc_np)
                    d_pred_arr[start:end] = sums / natoms_arr[start:end][:, None]

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    _log(f"  compute:     {time.time() - t0:5.1f}s")

    # 5) Vectorised text output --------------------------------------------
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)

    e_ref_pa = energy_ref / natoms_arr.astype(np.float64)
    np.savetxt(os.path.join(output_dir, "energy_train.out"),
               np.column_stack([e_pred_arr, e_ref_pa]), fmt="%.10g")

    if forces_ref is None:
        forces_ref = np.full((N_atoms_total, 3), np.nan, dtype=np.float64)
    np.savetxt(os.path.join(output_dir, "force_train.out"),
               np.column_stack([f_pred_arr, forces_ref]), fmt="%.10g")
    if m_pred_arr is not None:
        np.savetxt(
            os.path.join(output_dir, "mforce_train.out"),
            np.column_stack([m_pred_arr, mforces_ref]),
            fmt="%.10g",
        )
        np.savetxt(
            os.path.join(output_dir, "tau_train.out"),
            np.column_stack([t_pred_arr, tau_ref]),
            fmt="%.10g",
        )

    np.savetxt(os.path.join(output_dir, "virial_train.out"),
               np.column_stack([v_pred_arr, virial_ref]), fmt="%.10g")

    # Stress (GPa) = +virial_total / V * EV_PER_A3_TO_GPa, matching GPUMD's
    # convention so stress and virial carry the same sign (see
    # _stress_from_virial). Missing references keep the -1e6 sentinel unscaled.
    nat_col = natoms_arr.astype(np.float64)[:, None]
    vol_col = volumes_arr[:, None]
    stress_pred = _stress_from_virial(v_pred_arr, nat_col, vol_col,
                                      keep_missing=False)
    stress_ref = _stress_from_virial(virial_ref, nat_col, vol_col,
                                     keep_missing=True)
    np.savetxt(os.path.join(output_dir, "stress_train.out"),
               np.column_stack([stress_pred, stress_ref]), fmt="%.10g")

    if d_pred_arr is not None:
        np.savetxt(os.path.join(output_dir, "descriptor.out"),
                   d_pred_arr, fmt="%.10g")
    _log(f"  write:       {time.time() - t0:5.1f}s")

    _log(f"  TOTAL:       {time.time() - t_total:5.1f}s   "
         f"-> {output_dir}/(energy|force|virial|stress)_train.out")


# ---------------------------------------------------------------------------
# End-of-training prediction that reuses the in-memory model + GPUDataStore
# (no xyz re-read, no neighbor-list rebuild, no second GPU upload).
# ---------------------------------------------------------------------------

def predict_from_store(model, data_store, output_dir: str,
                       batch_size: int = 1000,
                       backend: str = "auto",
                       verbose: bool = True,
                       suffix: str = "train"):
    """Run prediction using an already-loaded NEPModel + GPUDataStore.

    Designed for the end of training: reuses the preprocessed data_store so
    there is no xyz re-read / neighbor-list rebuild / GPU upload. The
    prediction dtype matches the training dtype (= data_store dtype).

    Writes GPUMD-format outputs in ``output_dir`` (same columns and format as
    ``predict_dataset``):
      energy_train.out  — per-frame (pred, ref) in eV/atom
      force_train.out   — per-atom (fx,fy,fz pred, ref) in eV/A
      virial_train.out  — per-frame (xx,yy,zz,xy,yz,zx pred, ref) in eV/atom
      mforce_train.out  — spin models: per-atom magnetic force (pred, ref)
      tau_train.out     — spin models: per-atom spin x mforce (pred, ref)

    ``suffix`` picks the file-name tail: "train" (default) for the training
    set, "test" for the validation set (GPUMD's *_test.out naming).
    """
    def _log(msg):
        if verbose:
            print(msg, flush=True)

    dev   = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_struct = data_store.n
    backend = ops.resolve_backend(backend, num_types=model.num_types)

    nat_arr = np.asarray(data_store.natoms, dtype=np.int64)
    nat_cum = np.concatenate([[0], np.cumsum(nat_arr)])
    N_atoms_total = int(nat_cum[-1])

    e_pred = np.empty(n_struct, dtype=np.float64)
    f_pred = np.full((N_atoms_total, 3), np.nan, dtype=np.float64)
    m_pred = (
        np.full((N_atoms_total, 3), np.nan, dtype=np.float64)
        if model.spin_descriptor is not None else None)
    t_pred = (
        np.full((N_atoms_total, 3), np.nan, dtype=np.float64)
        if model.spin_descriptor is not None else None)
    v_pred = np.empty((n_struct, 6), dtype=np.float64)

    was_training = model.training
    model.eval()
    t_compute = time.time()
    with torch.no_grad():
        for start in range(0, n_struct, batch_size):
            end = min(start + batch_size, n_struct)
            idx = list(range(start, end))
            B = end - start
            batch = data_store.collate(idx)
            batch["need_mforce"] = model.spin_descriptor is not None
            batch["need_tau"] = model.spin_descriptor is not None
            if model.spin_mode == 2:
                with torch.enable_grad():
                    r = model.compute_properties(
                        batch["rij_rad"],
                        batch["rij_ang"],
                        batch["pair_i_rad"],
                        batch["pair_j_rad"],
                        batch["pair_i_ang"],
                        batch["pair_j_ang"],
                        batch["atom_types"],
                        batch["N"],
                        batch["struct_idx"],
                        batch["num_structures"],
                        need_forces=True,
                        need_virial=True,
                        backend=backend,
                        spins=batch["spins"],
                        need_mforce=True,
                        need_tau=True,
                        rij_spin=batch.get("rij_spin"),
                        pi_spin=batch.get("pair_i_spin"),
                        pj_spin=batch.get("pair_j_spin"),
                        create_graph=False,
                    )
                r = {
                    key: value.detach()
                    for key, value in r.items()
                }
            else:
                r = model.compute_properties_cached(
                    batch,
                    need_forces=True,
                    need_virial=True,
                    backend=backend,
                )

            nat_slice = nat_arr[start:end].astype(np.float64)
            e_pred[start:end] = r["Etot"].cpu().numpy() / nat_slice

            # Sum per-atom (N,9) virial into per-frame (B,9), then reorder.
            v_per = torch.zeros(B, 9, dtype=dtype, device=dev)
            v_per.scatter_add_(0,
                batch["struct_idx"].unsqueeze(-1).expand(-1, 9), r["virial"])
            v9 = v_per.cpu().numpy()
            v_pred[start:end] = _virial9_to_6(v9) / nat_slice[:, None]

            a_lo = int(nat_cum[start]); a_hi = int(nat_cum[end])
            f_pred[a_lo:a_hi] = r["forces"].cpu().numpy()
            if m_pred is not None:
                m_pred[a_lo:a_hi] = r["mforces"].cpu().numpy()
                t_pred[a_lo:a_hi] = r["tau"].cpu().numpy()
    if was_training:
        model.train()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    _log(f"  compute:  {time.time() - t_compute:5.1f}s")

    # Reference values (from data_store; only fill where the flag is set).
    energy_ref = np.array(
        [data_store.energy[i] if data_store.has_energy_flag[i] else np.nan
         for i in range(n_struct)], dtype=np.float64)
    e_ref_pa = energy_ref / nat_arr.astype(np.float64)

    forces_ref = np.full((N_atoms_total, 3), np.nan, dtype=np.float64)
    for i in range(n_struct):
        if data_store.has_forces_flag[i]:
            a_lo, a_hi = int(nat_cum[i]), int(nat_cum[i + 1])
            forces_ref[a_lo:a_hi] = data_store.forces[i].cpu().numpy()

    virial_ref = np.full((n_struct, 6), _MISSING_VIRIAL, dtype=np.float64)
    for i in range(n_struct):
        if data_store.has_virial_flag[i]:
            v9 = data_store.virial[i].cpu().numpy().flatten()  # length-9
            n = float(nat_arr[i])
            # Single-triangular pick to match GPUMD (see _virial9_to_6).
            virial_ref[i] = [
                v9[0] / n,   # xx
                v9[4] / n,   # yy
                v9[8] / n,   # zz
                v9[1] / n,   # xy
                v9[5] / n,   # yz
                v9[6] / n,   # zx
            ]

    os.makedirs(output_dir, exist_ok=True)
    t_write = time.time()
    np.savetxt(os.path.join(output_dir, f"energy_{suffix}.out"),
               np.column_stack([e_pred, e_ref_pa]), fmt="%.10g")
    np.savetxt(os.path.join(output_dir, f"force_{suffix}.out"),
               np.column_stack([f_pred, forces_ref]), fmt="%.10g")
    if m_pred is not None:
        mforces_ref = np.full(
            (N_atoms_total, 3), np.nan, dtype=np.float64)
        tau_ref = np.full(
            (N_atoms_total, 3), np.nan, dtype=np.float64)
        for i in range(n_struct):
            if data_store.has_mforces_flag[i]:
                a_lo, a_hi = int(nat_cum[i]), int(nat_cum[i + 1])
                mforces_ref[a_lo:a_hi] = (
                    data_store.mforces[i].cpu().numpy())
                tau_ref[a_lo:a_hi] = data_store.tau[i].cpu().numpy()
        np.savetxt(
            os.path.join(output_dir, f"mforce_{suffix}.out"),
            np.column_stack([m_pred, mforces_ref]), fmt="%.10g")
        np.savetxt(
            os.path.join(output_dir, f"tau_{suffix}.out"),
            np.column_stack([t_pred, tau_ref]), fmt="%.10g")
    np.savetxt(os.path.join(output_dir, f"virial_{suffix}.out"),
               np.column_stack([v_pred, virial_ref]), fmt="%.10g")

    # Stress (GPa) = +virial_total / V * conversion (GPUMD sign; see
    # predict_dataset / _stress_from_virial). Missing refs keep -1e6 unscaled.
    vol_arr = data_store.volumes.detach().cpu().numpy().astype(np.float64)
    nat_col = nat_arr.astype(np.float64)[:, None]
    vol_col = vol_arr[:, None]
    stress_pred = _stress_from_virial(v_pred, nat_col, vol_col,
                                      keep_missing=False)
    stress_ref = _stress_from_virial(virial_ref, nat_col, vol_col,
                                     keep_missing=True)
    np.savetxt(os.path.join(output_dir, f"stress_{suffix}.out"),
               np.column_stack([stress_pred, stress_ref]), fmt="%.10g")

    extra = "|mforce|tau" if model.spin_descriptor is not None else ""
    _log(f"  write:    {time.time() - t_write:5.1f}s   "
         f"-> {output_dir}/(energy|force|virial|stress{extra})_{suffix}.out")


# ---------------------------------------------------------------------------
# Sharded variant: each DDP rank predicts its own data_store shard, then all
# per-frame arrays are gathered to rank 0 and written out in input-xyz order.
# No xyz re-read, no temp nep.txt, no second neighbor-list build.
# ---------------------------------------------------------------------------

def _compute_local_predictions(model, data_store, batch_size, backend):
    """Run prediction on one rank's local data_store -> numpy arrays.

    Returns a dict of per-frame / per-atom arrays (pred and ref) plus the
    natoms and volume metadata needed to merge + write on rank 0.
    """
    dev   = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_struct = data_store.n
    backend = ops.resolve_backend(backend, num_types=model.num_types)

    nat_arr = np.asarray(data_store.natoms, dtype=np.int64)
    nat_cum = np.concatenate([[0], np.cumsum(nat_arr)])
    N_atoms_local = int(nat_cum[-1])

    e_pred = np.zeros(n_struct, dtype=np.float64)
    f_pred = np.zeros((N_atoms_local, 3), dtype=np.float64)
    v_pred = np.zeros((n_struct, 6), dtype=np.float64)
    m_pred = (
        np.zeros((N_atoms_local, 3), dtype=np.float64)
        if model.spin_descriptor is not None else None)
    t_pred = (
        np.zeros((N_atoms_local, 3), dtype=np.float64)
        if model.spin_descriptor is not None else None)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, n_struct, batch_size):
            end = min(start + batch_size, n_struct)
            B = end - start
            batch = data_store.collate(list(range(start, end)))
            batch["need_mforce"] = model.spin_descriptor is not None
            batch["need_tau"] = model.spin_descriptor is not None
            if model.spin_mode == 2:
                # The systematic magnetic-polynomial path currently obtains
                # geometry and spin derivatives from autograd.  End-of-run
                # prediction is normally under no_grad, so re-enable it only
                # for this descriptor and detach the completed observables.
                with torch.enable_grad():
                    r = model.compute_properties(
                        batch["rij_rad"],
                        batch["rij_ang"],
                        batch["pair_i_rad"],
                        batch["pair_j_rad"],
                        batch["pair_i_ang"],
                        batch["pair_j_ang"],
                        batch["atom_types"],
                        batch["N"],
                        batch["struct_idx"],
                        batch["num_structures"],
                        need_forces=True,
                        need_virial=True,
                        backend=backend,
                        spins=batch["spins"],
                        need_mforce=True,
                        need_tau=True,
                        rij_spin=batch.get("rij_spin"),
                        pi_spin=batch.get("pair_i_spin"),
                        pj_spin=batch.get("pair_j_spin"),
                        create_graph=False,
                    )
                r = {key: value.detach() for key, value in r.items()}
            else:
                r = model.compute_properties_cached(
                    batch, need_forces=True, need_virial=True, backend=backend)

            nat_slice = nat_arr[start:end].astype(np.float64)
            e_pred[start:end] = r["Etot"].cpu().numpy() / nat_slice

            v_per = torch.zeros(B, 9, dtype=dtype, device=dev)
            v_per.scatter_add_(
                0, batch["struct_idx"].unsqueeze(-1).expand(-1, 9), r["virial"])
            v_pred[start:end] = (_virial9_to_6(v_per.cpu().numpy())
                                 / nat_slice[:, None])

            a_lo, a_hi = int(nat_cum[start]), int(nat_cum[end])
            f_pred[a_lo:a_hi] = r["forces"].cpu().numpy()
            if m_pred is not None:
                m_pred[a_lo:a_hi] = r["mforces"].cpu().numpy()
                t_pred[a_lo:a_hi] = r["tau"].cpu().numpy()
    if was_training:
        model.train()
    if dev.type == "cuda":
        torch.cuda.synchronize()

    # Reference values (only fill where the flag is set)
    energy_ref = np.array(
        [data_store.energy[i] if data_store.has_energy_flag[i] else np.nan
         for i in range(n_struct)], dtype=np.float64)
    e_ref_pa = energy_ref / nat_arr.astype(np.float64)

    forces_ref = np.full((N_atoms_local, 3), np.nan, dtype=np.float64)
    mforces_ref = (
        np.full((N_atoms_local, 3), np.nan, dtype=np.float64)
        if model.spin_descriptor is not None else None)
    tau_ref = (
        np.full((N_atoms_local, 3), np.nan, dtype=np.float64)
        if model.spin_descriptor is not None else None)
    for i in range(n_struct):
        a_lo, a_hi = int(nat_cum[i]), int(nat_cum[i + 1])
        if data_store.has_forces_flag[i]:
            forces_ref[a_lo:a_hi] = data_store.forces[i].cpu().numpy()
        if mforces_ref is not None and data_store.has_mforces_flag[i]:
            mforces_ref[a_lo:a_hi] = data_store.mforces[i].cpu().numpy()
            tau_ref[a_lo:a_hi] = data_store.tau[i].cpu().numpy()

    virial_ref = np.full((n_struct, 6), _MISSING_VIRIAL, dtype=np.float64)
    for i in range(n_struct):
        if data_store.has_virial_flag[i]:
            v9 = data_store.virial[i].cpu().numpy().flatten()
            n = float(nat_arr[i])
            virial_ref[i] = [v9[0] / n, v9[4] / n, v9[8] / n,
                             v9[1] / n, v9[5] / n, v9[6] / n]

    vol_arr = data_store.volumes.detach().cpu().numpy().astype(np.float64)
    return {
        "natoms":    nat_arr,
        "volumes":   vol_arr,
        "e_pred":    e_pred,    "e_ref":    e_ref_pa,
        "f_pred":    f_pred,    "f_ref":    forces_ref,
        "v_pred":    v_pred,    "v_ref":    virial_ref,
        "m_pred":    m_pred,    "m_ref":    mforces_ref,
        "t_pred":    t_pred,    "t_ref":    tau_ref,
    }


def _write_predictions(output_dir: str, n_total_frames: int,
                       natoms, volumes,
                       e_pred, e_ref, f_pred, f_ref, v_pred, v_ref,
                       m_pred=None, m_ref=None, t_pred=None, t_ref=None,
                       suffix: str = "train"):
    """Write the four *_{suffix}.out files in frame-order. All arrays are
    already in global input-xyz order (frame 0 first)."""
    os.makedirs(output_dir, exist_ok=True)

    np.savetxt(os.path.join(output_dir, f"energy_{suffix}.out"),
               np.column_stack([e_pred, e_ref]), fmt="%.10g")
    np.savetxt(os.path.join(output_dir, f"force_{suffix}.out"),
               np.column_stack([f_pred, f_ref]), fmt="%.10g")
    np.savetxt(os.path.join(output_dir, f"virial_{suffix}.out"),
               np.column_stack([v_pred, v_ref]), fmt="%.10g")
    if m_pred is not None:
        np.savetxt(os.path.join(output_dir, f"mforce_{suffix}.out"),
                   np.column_stack([m_pred, m_ref]), fmt="%.10g")
        np.savetxt(os.path.join(output_dir, f"tau_{suffix}.out"),
                   np.column_stack([t_pred, t_ref]), fmt="%.10g")

    # Stress (GPa), GPUMD sign (+virial/V); missing refs keep -1e6 unscaled.
    nat_col = natoms.astype(np.float64)[:, None]
    vol_col = volumes[:, None]
    stress_pred = _stress_from_virial(v_pred, nat_col, vol_col,
                                      keep_missing=False)
    stress_ref = _stress_from_virial(v_ref, nat_col, vol_col,
                                     keep_missing=True)
    np.savetxt(os.path.join(output_dir, f"stress_{suffix}.out"),
               np.column_stack([stress_pred, stress_ref]), fmt="%.10g")


def predict_from_store_sharded(model, data_store, local_global_idx,
                                n_total_frames: int, output_dir: str,
                                batch_size: int = 1000,
                                backend: str = "auto",
                                verbose: bool = True,
                                suffix: str = "train",
                                local_real_flags=None):
    """DDP equivalent of ``predict_from_store``: each rank predicts its local
    data-store shard, arrays are gathered to rank 0, and rank 0 writes the
    structural files plus magnetic-force/torque files for spin models in
    input-xyz order ("train" or "test").

    No xyz re-read, no neighbor-list rebuild, no temp nep.txt model file.

    Parameters
    ----------
    model : NEPModel  (DDP replica — parameters are in sync across ranks).
    data_store : GPUDataStore  (this rank's local shard).
    local_global_idx : list[int]  original xyz-frame index for each local
        frame (length == ``data_store.n``). Supplied by the random shard
        assignment in ``train_nep_sharded``.
    n_total_frames : int  total frames across all ranks (pre-drop).
    """
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_main = rank == 0

    def _log(msg):
        if verbose and is_main:
            print(msg, flush=True)

    t_compute = time.time()
    local = _compute_local_predictions(model, data_store, batch_size, backend)
    local["global_idx"] = np.asarray(local_global_idx, dtype=np.int64)
    local["is_real"] = np.asarray(
        ([True] * len(local_global_idx)
         if local_real_flags is None else local_real_flags),
        dtype=bool,
    )
    if local["is_real"].shape != local["global_idx"].shape:
        raise ValueError("local_real_flags must match local_global_idx")
    _log(f"  compute:  {time.time() - t_compute:5.1f}s")

    # Gather per-rank arrays onto rank 0. all_gather_object stays on the
    # object-pickle path (cheap for our sizes: the per-rank arrays together
    # are the same size as the full dataset).
    t_gather = time.time()
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    if is_main:
        _log(f"  gather:   {time.time() - t_gather:5.1f}s")

    if not is_main:
        return

    # Allocate global arrays and scatter each rank's REAL contribution into
    # the slots given by its global_idx. Padding copies have their labels
    # removed for loss fairness; they must be skipped here too, otherwise
    # their NaN references overwrite the real frame's reference columns.
    natoms_g = np.zeros(n_total_frames, dtype=np.int64)
    volumes_g = np.zeros(n_total_frames, dtype=np.float64)
    e_pred_g = np.full(n_total_frames, np.nan)
    e_ref_g  = np.full(n_total_frames, np.nan)
    v_pred_g = np.full((n_total_frames, 6), np.nan)
    v_ref_g  = np.full((n_total_frames, 6), _MISSING_VIRIAL)

    # First pass: frame-level arrays + natoms (to size the atom-level arrays).
    seen_real = np.zeros(n_total_frames, dtype=bool)
    for part in gathered:
        real = part["is_real"]
        gi = part["global_idx"][real]
        if seen_real[gi].any():
            raise ValueError("duplicate real frame in sharded prediction")
        seen_real[gi] = True
        natoms_g[gi]  = part["natoms"][real]
        volumes_g[gi] = part["volumes"][real]
        e_pred_g[gi]  = part["e_pred"][real]
        e_ref_g[gi]   = part["e_ref"][real]
        v_pred_g[gi]  = part["v_pred"][real]
        v_ref_g[gi]   = part["v_ref"][real]
    if not seen_real.all():
        missing = np.flatnonzero(~seen_real)
        raise ValueError(
            f"sharded prediction is missing real frames: {missing[:5]}")

    # Global per-atom offsets are determined by input-xyz order so they match
    # what predict_dataset / predict_from_store would emit.
    nat_cum_g = np.concatenate([[0], np.cumsum(natoms_g)])
    N_atoms_global = int(nat_cum_g[-1])
    f_pred_g = np.full((N_atoms_global, 3), np.nan)
    f_ref_g  = np.full((N_atoms_global, 3), np.nan)
    has_spin = gathered[0]["m_pred"] is not None
    m_pred_g = (
        np.full((N_atoms_global, 3), np.nan) if has_spin else None)
    m_ref_g = (
        np.full((N_atoms_global, 3), np.nan) if has_spin else None)
    t_pred_g = (
        np.full((N_atoms_global, 3), np.nan) if has_spin else None)
    t_ref_g = (
        np.full((N_atoms_global, 3), np.nan) if has_spin else None)

    for part in gathered:
        gi = part["global_idx"]
        # Local atom offsets in the rank's concatenated arrays:
        nat_part = part["natoms"]
        part_cum = np.concatenate([[0], np.cumsum(nat_part)])
        for li, gidx in enumerate(gi):
            if not part["is_real"][li]:
                continue
            la, lb = int(part_cum[li]),  int(part_cum[li + 1])
            ga, gb = int(nat_cum_g[gidx]), int(nat_cum_g[gidx + 1])
            f_pred_g[ga:gb] = part["f_pred"][la:lb]
            f_ref_g[ga:gb]  = part["f_ref"][la:lb]
            if has_spin:
                m_pred_g[ga:gb] = part["m_pred"][la:lb]
                m_ref_g[ga:gb] = part["m_ref"][la:lb]
                t_pred_g[ga:gb] = part["t_pred"][la:lb]
                t_ref_g[ga:gb] = part["t_ref"][la:lb]

    t_write = time.time()
    _write_predictions(output_dir, n_total_frames,
                       natoms_g, volumes_g,
                       e_pred_g, e_ref_g,
                       f_pred_g, f_ref_g,
                       v_pred_g, v_ref_g,
                       m_pred=m_pred_g, m_ref=m_ref_g,
                       t_pred=t_pred_g, t_ref=t_ref_g,
                       suffix=suffix)
    extra = "|mforce|tau" if has_spin else ""
    _log(f"  write:    {time.time() - t_write:5.1f}s   "
         f"-> {output_dir}/(energy|force|virial|stress{extra})_{suffix}.out")
