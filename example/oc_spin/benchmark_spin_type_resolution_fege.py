# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.

"""Matched FeGe ablation of spin element resolution only.

Both variants retain the ordinary structural NEP type-pair coefficients and
the established per-type atomic-energy networks.  They also have identical
descriptor width, parameter count, initialization, split and optimizer.  The
only controlled difference is whether the spin radial table

    c_spin[center_type, neighbor_type, radial_channel, basis]

may specialize by ordered element pair.  ``pair_tied`` replaces the table by
its pair-axis mean at the descriptor seam, so every stored entry receives the
same gradient and remains tied.  ``pair_resolved`` uses the production model
unchanged.  No research switch is added to ``nep.in`` or the model format.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import time
from unittest import mock

import numpy as np
import torch

from example.oc_spin.benchmark_spin3_fege import (
    _config_text,
    _initial_statistics,
    _sha256,
    _summarize_run,
)
from torchnep.data import parse_nep_in
from torchnep.magnetic_polynomial_v3 import MagneticPolynomialDescriptorV3
from torchnep.model import NEPModel
from torchnep.train import train_nep


VARIANTS = {
    "pair_tied": False,
    "pair_resolved": True,
}

METRIC_KEYS = {
    "energy": "rmse_e_test(eV/atom)",
    "force": "rmse_f_test(eV/A)",
    "mforce": "rmse_m_test(eV/muB)",
}


def _tie_spin_pair_table(coefficient):
    """Return one learned radial table shared by all ordered type pairs."""
    if coefficient is None:
        return None
    if coefficient.ndim == 5:
        pair_axes = (1, 2)
    elif coefficient.ndim == 4:
        pair_axes = (0, 1)
    else:
        raise ValueError("spin pair table must have rank 4 or 5")
    shared = coefficient.mean(dim=pair_axes, keepdim=True)
    return shared.expand_as(coefficient)


@contextmanager
def _spin_pair_resolution(*, resolved: bool):
    """Select the experiment-only pair-tied adapter at the descriptor seam."""
    if resolved:
        yield
        return

    original = MagneticPolynomialDescriptorV3.forward

    def tied_forward(descriptor, *args, **kwargs):
        positional = list(args)
        if len(positional) > 6:
            positional[6] = _tie_spin_pair_table(positional[6])
        elif "c_spin_2body" in kwargs:
            kwargs = dict(kwargs)
            kwargs["c_spin_2body"] = _tie_spin_pair_table(
                kwargs["c_spin_2body"]
            )
        return original(descriptor, *positional, **kwargs)

    with mock.patch.object(
        MagneticPolynomialDescriptorV3, "forward", tied_forward
    ):
        yield


def _pair_spread(config_path: Path, potential_path: Path) -> float:
    """Maximum deviation from a completely pair-tied saved coefficient."""
    model = NEPModel(parse_nep_in(config_path)).double()
    model.load_weights_from_nep_txt(str(potential_path))
    coefficient = model.c_param_spin
    shared = coefficient.mean(dim=(1, 2), keepdim=True)
    return float((coefficient - shared).abs().max())


def _percent_improvement(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def _summary(report):
    by_variant = {variant: {} for variant in VARIANTS}
    for row in report["runs"]:
        best = row["training"]["best_validation"]
        by_variant[row["variant"]][row["seed"]] = {
            name: float(best[key]) for name, key in METRIC_KEYS.items()
        } | {
            "pair_spread": float(row["pair_spread"]),
        }

    expected = set(report["seeds"])
    for variant, rows in by_variant.items():
        if set(rows) != expected:
            raise RuntimeError(f"missing completed {variant} seeds")

    metrics = {}
    for variant, rows in by_variant.items():
        metrics[variant] = {}
        for metric in (*METRIC_KEYS, "pair_spread"):
            values = [rows[seed][metric] for seed in report["seeds"]]
            metrics[variant][metric] = {
                "values": values,
                "mean": float(np.mean(values)),
                "sample_std": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ),
            }

    return {
        "metrics": metrics,
        "pair_resolved_improvement_percent": {
            metric: _percent_improvement(
                metrics["pair_tied"][metric]["mean"],
                metrics["pair_resolved"][metric]["mean"],
            )
            for metric in METRIC_KEYS
        },
        "paired_seed_wins": {
            metric: sum(
                by_variant["pair_resolved"][seed][metric]
                < by_variant["pair_tied"][seed][metric]
                for seed in report["seeds"]
            )
            for metric in METRIC_KEYS
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the FeGe type-resolution ablation requires CUDA")

    args.output.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64 if args.precision == "float64" else torch.float32
    report = {
        "schema": "nep4-spin3-type-resolution-ab-v1",
        "data": str(args.data.resolve()),
        "data_sha256": _sha256(args.data),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "precision": args.precision,
        "valid_ratio": args.valid_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seeds": list(args.seeds),
        "loss_weights": {"energy": 1.0, "force": 1.0, "mforce": 1.0},
        "controlled_difference": (
            "independent versus pair-tied c_spin; structural type resolution "
            "and per-type ANN are unchanged"
        ),
        "variants": VARIANTS,
        "runs": [],
    }

    for seed in args.seeds:
        for variant, resolved in VARIANTS.items():
            run_dir = args.output / f"{variant}-seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / "nep.in"
            config_path.write_text(
                _config_text(3, args.epochs, args.batch_size, args.lr, 2,
                             "full", 1.0, 1.0, 1.0)
            )
            with _spin_pair_resolution(resolved=resolved):
                row = {
                    "variant": variant,
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
                    row["pair_spread"] = _pair_spread(
                        config_path, run_dir / "nep_final.txt"
                    )
            report["runs"].append(row)
            if not args.stats_only:
                report["summary"] = (
                    _summary(report)
                    if len(report["runs"]) == len(VARIANTS) * len(args.seeds)
                    else None
                )
            (args.output / "result.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            gc.collect()
            torch.cuda.empty_cache()

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
