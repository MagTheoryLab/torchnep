# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.
# TorchNEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Matched FeGe training and conditioning gate for ``nep4_spin3``.

This is deliberately a dataset-path benchmark rather than a bundled fixture:
the audited FeGe teacher set is too large and too provenance-sensitive to copy
into the source tree.  Modes 2 and 3 use the same split, seed, architecture,
optimizer and labels.  The JSON result records the input hash, held-out metrics,
per-epoch training time and initial descriptor conditioning.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

from torchnep.data import parse_nep_in, read_xyz, valid_split_indices
from torchnep.model import NEPModel
from torchnep.train import (
    StreamDataStore,
    compute_q_scaler,
    grouped_validation_indices,
    preprocess_structures,
    train_nep,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_text(
    mode: int,
    epochs: int,
    batch_size: int,
    lr: float,
    spin_l_max: int,
    mforce_mode: str,
    lambda_e: float,
    lambda_f: float,
    lambda_m: float,
) -> str:
    mforce_line = (
        f"spin_mforce_mode {mforce_mode}\n" if mode == 3 else ""
    )
    return f"""type 2 Fe Ge
cutoff 6 5
n_max 4 4
basis_size 8 8
l_max 4 2 0
neuron 30

spin_mode {mode}
spin_cutoff 6
spin_l_max {spin_l_max}
spin_compress 2
spin_order 3
spin_soc 1
spin_curriculum 0
{mforce_line}spin_dof_type Fe Ge
spin_env_type Fe Ge

epoch {epochs}
batch {batch_size}
eval_batch {batch_size}
lr {lr:.16g}
optimizer adamw
weight_decay 0.0001
max_grad_norm 5
lambda_e {lambda_e:.16g}
lambda_f {lambda_f:.16g}
lambda_v 0
lambda_m {lambda_m:.16g}
lambda_tau 0
q_scaler_init model
"""


def _initial_statistics(
    config_path: Path,
    data_path: Path,
    seed: int,
    batch_size: int,
    valid_ratio: float,
    dtype: torch.dtype,
):
    config = parse_nep_in(config_path)
    frames = read_xyz(data_path)
    if config["spin_mode"] in (2, 3) and any(
            "response_group" in frame for frame in frames):
        valid_indices = grouped_validation_indices(frames, valid_ratio, seed)
        held_out = set(valid_indices)
        train_indices = [
            index for index in range(len(frames)) if index not in held_out
        ]
    else:
        train_indices, _ = valid_split_indices(
            len(frames), valid_ratio, seed
        )
    structures = preprocess_structures(
        [frames[index] for index in train_indices],
        config,
        np.float64 if dtype == torch.float64 else np.float32,
    )
    store = StreamDataStore(
        structures, torch.device("cuda"), dtype, config=config
    )
    torch.manual_seed(seed)
    model = NEPModel(config).to(device="cuda", dtype=dtype)
    q_min, q_max, q_rms = compute_q_scaler(
        model,
        store,
        batch_size=batch_size,
        backend="loop",
        return_rms=True,
    )
    if model.spin_mode == 3:
        model.set_q_scaler(q_min, q_max, q_rms)
    else:
        model.set_q_scaler(q_min, q_max)
    start = model.struct_dim
    span = q_max[start:] - q_min[start:]
    peak = torch.maximum(q_min[start:].abs(), q_max[start:].abs())
    scaled_peak = peak * model.q_scaler[start:]
    rows = []
    for token in model.spin_descriptor.schema:
        local = slice(token.start, token.stop)
        rows.append({
            "name": token.name,
            "start": token.start,
            "stop": token.stop,
            "span_min": float(span[local].min()),
            "span_max": float(span[local].max()),
            "rms_min": float(q_rms[start:][local].min()),
            "rms_max": float(q_rms[start:][local].max()),
            "scaled_peak_max": float(scaled_peak[local].max()),
        })
    summary = {
        "spin_dim": model.spin_dim,
        "scaler_min": float(model.q_scaler[start:].min()),
        "scaler_max": float(model.q_scaler[start:].max()),
        "scaled_peak_max": float(scaled_peak.max()),
        "scaled_peak_l2": float(torch.linalg.vector_norm(scaled_peak)),
        "channels": rows,
    }
    del model, store, structures, frames
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _read_loss(path: Path):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    header = lines[0].lstrip("# ").split()
    records = []
    for line in lines[1:]:
        values = line.split()
        records.append({
            name: int(value) if name == "epoch" else float(value)
            for name, value in zip(header, values)
        })
    return records


def _summarize_run(output_dir: Path, wall_seconds: float):
    records = _read_loss(output_dir / "loss.out")
    best = min(records, key=lambda row: row["loss"])
    profile_seconds = [
        float(match.group(1))
        for match in re.finditer(
            r"\[prof\] train\s+([0-9.]+)s",
            (output_dir / "output.log").read_text(),
        )
    ]
    timed = profile_seconds[1:] if len(profile_seconds) > 1 else profile_seconds
    return {
        "wall_seconds": wall_seconds,
        "epoch_train_seconds_median_excluding_first": (
            float(np.median(timed)) if timed else None
        ),
        "epoch_train_seconds": profile_seconds,
        "best_validation": best,
        "final": records[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11])
    parser.add_argument("--modes", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--spin-l-max", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--mforce-mode", choices=("full", "transverse"),
                        default="full")
    parser.add_argument("--lambda-e", type=float, default=1.0)
    parser.add_argument("--lambda-f", type=float, default=1.0)
    parser.add_argument("--lambda-m", type=float, default=1.0)
    parser.add_argument("--precision", choices=("float32", "float64"),
                        default="float32")
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the FeGe gate requires a CUDA device")
    if any(mode not in (2, 3) for mode in args.modes):
        raise ValueError("--modes accepts only 2 and 3")
    if args.mforce_mode != "full" and 2 in args.modes:
        raise ValueError(
            "transverse magnetic-force training is a spin_mode 3 contract"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64 if args.precision == "float64" else torch.float32
    report = {
        "schema": "nep4-spin3-fege-ab-v1",
        "data": str(args.data.resolve()),
        "data_sha256": _sha256(args.data),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "precision": args.precision,
        "valid_ratio": args.valid_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "spin_l_max": args.spin_l_max,
        "mforce_mode": args.mforce_mode,
        "loss_weights": {
            "energy": args.lambda_e,
            "force": args.lambda_f,
            "mforce": args.lambda_m,
        },
        "runs": [],
    }
    for seed in args.seeds:
        for mode in args.modes:
            run_dir = (
                args.output / f"spin{mode}-l{args.spin_l_max}-seed{seed}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / "nep.in"
            config_path.write_text(_config_text(
                mode, args.epochs, args.batch_size, args.lr, args.spin_l_max,
                args.mforce_mode, args.lambda_e, args.lambda_f, args.lambda_m,
            ))
            row = {
                "mode": mode,
                "seed": seed,
                "conditioning": _initial_statistics(
                    config_path,
                    args.data,
                    seed,
                    args.batch_size,
                    args.valid_ratio,
                    dtype,
                ),
            }
            if not args.stats_only:
                started = time.perf_counter()
                train_nep(
                    config_file=str(config_path),
                    data_file=str(args.data),
                    output_dir=str(run_dir),
                    device="cuda",
                    precision=args.precision,
                    use_compile=False,
                    print_interval=max(1, args.epochs // 10),
                    restart=False,
                    checkpoint_interval=max(args.epochs + 1, 100),
                    prediction_interval=0,
                    run_seed=seed,
                    valid_ratio=args.valid_ratio,
                    valid_strategy="random",
                )
                torch.cuda.synchronize()
                row["training"] = _summarize_run(
                    run_dir, time.perf_counter() - started
                )
            report["runs"].append(row)
            (args.output / "result.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
