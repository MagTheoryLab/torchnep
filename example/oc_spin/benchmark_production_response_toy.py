# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.
# TorchNEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Sealed production-trainer A/B for grouped magnetic response.

The two variants use identical noisy E/MF labels, O3-C2 descriptor, scalar
ANN, split, seed, and epochs.  The only difference is whether controlled
rotation frames declare ``spin_tangent`` and therefore activate the internal
Stage-2 centered generator-curve objective.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from torchnep import train_nep
from torchnep.nep import NEPCalculator


SPECIES = ["Fe"] * 4
CELL = np.diag([7.2, 7.2, 7.2])
REFERENCE_POSITIONS = np.array([
    [1.4, 1.5, 1.6],
    [3.1, 1.5, 1.6],
    [1.5, 3.0, 1.8],
    [1.7, 1.8, 3.2],
])
SCAN_COORDINATES = np.linspace(-0.30, 0.30, 7)
BLIND_COORDINATES = np.linspace(-0.38, 0.38, 25)


def _base(rng):
    positions = REFERENCE_POSITIONS + rng.normal(0.0, 0.10, (4, 3))
    m0, m1 = rng.uniform(0.75, 1.35, 2)
    return positions, float(m0), float(m1)


def _coefficients(positions):
    rij = positions[1] - positions[0]
    distance = np.linalg.norm(rij)
    environment = np.sin(positions[2:].sum())
    return {
        "j": 0.075 + 0.040 * np.exp(-((distance - 1.75) / 0.35) ** 2)
             + 0.006 * environment,
        "d": 0.014 + 0.010 * (distance - 1.70),
        "b": 0.012 + 0.003 * np.cos(positions[3].sum()),
        "k": 0.008 + 0.002 * np.sin(positions[2].sum()),
    }


def _teacher(positions, spins):
    coefficients = _coefficients(positions)
    s0, s1 = spins[0], spins[1]
    dot = float(np.dot(s0, s1))
    cross_y = float(np.cross(s0, s1)[1])
    structural = (
        20.0
        + 0.8 * float(((positions - REFERENCE_POSITIONS) ** 2).sum())
        + 0.25 * np.sin(float(positions.sum()))
    )
    energy = (
        structural
        + coefficients["j"] * dot
        + coefficients["d"] * cross_y
        + coefficients["b"] * dot * dot
        + coefficients["k"] * (s0[2] ** 2 + s1[2] ** 2)
    )
    gradient0 = (
        coefficients["j"] * s1
        + coefficients["d"] * np.cross(s1, np.array([0.0, 1.0, 0.0]))
        + 2.0 * coefficients["b"] * dot * s1
        + np.array([0.0, 0.0, 2.0 * coefficients["k"] * s0[2]])
    )
    gradient1 = (
        coefficients["j"] * s0
        + coefficients["d"] * np.cross(np.array([0.0, 1.0, 0.0]), s0)
        + 2.0 * coefficients["b"] * dot * s0
        + np.array([0.0, 0.0, 2.0 * coefficients["k"] * s1[2]])
    )
    mforces = np.zeros((4, 3))
    mforces[0] = -gradient0
    mforces[1] = -gradient1
    return float(energy), mforces, coefficients


def _scan_frame(base, theta, group, noise_rng=None):
    positions, m0, m1 = base
    spins = np.zeros((4, 3))
    spins[0] = [0.0, 0.0, m0]
    spins[1] = [m1 * np.sin(theta), 0.0, m1 * np.cos(theta)]
    tangent = np.zeros((4, 3))
    tangent[1] = [m1 * np.cos(theta), 0.0, -m1 * np.sin(theta)]
    energy, mforces, coefficients = _teacher(positions, spins)
    exact_energy = energy
    exact_mforces = mforces.copy()
    if noise_rng is not None:
        energy += float(noise_rng.normal(0.0, 0.005))
        mforces = mforces + noise_rng.normal(0.0, 0.004, mforces.shape)
    return {
        "positions": positions.copy(), "spins": spins,
        "tangent": tangent, "energy": energy, "mforces": mforces,
        "exact_energy": exact_energy, "exact_mforces": exact_mforces,
        "coordinate": float(theta), "group": group,
        "coefficients": coefficients,
    }


def _ordinary_frame(base, rng, noise_rng):
    positions, m0, m1 = base
    spins = np.zeros((4, 3))
    for atom, magnitude in ((0, m0), (1, m1)):
        direction = rng.normal(size=3)
        spins[atom] = magnitude * direction / np.linalg.norm(direction)
    energy, mforces, _ = _teacher(positions, spins)
    return {
        "positions": positions.copy(), "spins": spins,
        "energy": energy + float(noise_rng.normal(0.0, 0.005)),
        "mforces": mforces + noise_rng.normal(0.0, 0.004, mforces.shape),
    }


def _dataset(seed=20260816):
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 1)
    train = [_ordinary_frame(_base(rng), rng, noise_rng) for _ in range(48)]
    valid = [_ordinary_frame(_base(rng), rng, noise_rng) for _ in range(12)]
    for index in range(12):
        base = _base(rng)
        train.extend(_scan_frame(
            base, theta, f"train-{index:02d}", noise_rng)
            for theta in SCAN_COORDINATES)
    for index in range(4):
        base = _base(rng)
        valid.extend(_scan_frame(
            base, theta, f"valid-{index:02d}", noise_rng)
            for theta in SCAN_COORDINATES)
    blind = []
    for index in range(8):
        base = _base(rng)
        blind.append([
            _scan_frame(base, theta, f"blind-{index:02d}")
            for theta in BLIND_COORDINATES
        ])
    return train, valid, blind


def _write_xyz(path, frames, with_tangent):
    lines = []
    for frame in frames:
        response = ""
        tangent_schema = ""
        if "group" in frame:
            response = (
                f" response_group={frame['group']} response_probe=rotation"
                f" response_coordinate={frame['coordinate']:.12g}")
            if with_tangent:
                tangent_schema = ":spin_tangent:R:3"
        lines.extend([
            "4",
            'Lattice="7.2 0 0 0 7.2 0 0 0 7.2"'
            f" energy={frame['energy']:.16e}{response} "
            "Properties=species:S:1:pos:R:3:spin:R:3:mforce:R:3"
            f"{tangent_schema}",
        ])
        for atom in range(4):
            values = [
                *frame["positions"][atom], *frame["spins"][atom],
                *frame["mforces"][atom],
            ]
            if tangent_schema:
                values.extend(frame["tangent"][atom])
            lines.append("Fe " + " ".join(f"{value:.16e}" for value in values))
    path.write_text("\n".join(lines) + "\n")


def _write_config(path, epochs):
    path.write_text(
        "type 1 Fe\n"
        "cutoff 4.0 4.0\n"
        "n_max 4 4\n"
        "basis_size 8 8\n"
        "l_max 4 2 0\n"
        "neuron 16\n"
        "spin_mode 3\n"
        "spin_compress 2\n"
        "spin_l_max 2\n"
        "spin_order 3\n"
        "spin_soc 1\n"
        "spin_cutoff 4.0\n"
        "lambda_e 1.0\n"
        "lambda_f 0.0\n"
        "lambda_v 0.0\n"
        "lambda_m 0.10\n"
        "lambda_tau 0.0\n"
        "stage2 1\n"
        f"start_stage2 {max(2, epochs // 2)}\n"
        "stage2_lambda_e 0.20\n"
        "stage2_lambda_f 0.0\n"
        "stage2_lambda_v 0.0\n"
        "stage2_lambda_m 1.0\n"
        "stage2_lambda_tau 0.0\n"
        "stage2_lr 0.0005\n"
        "batch 16\n"
        "eval_batch 64\n"
        f"epoch {epochs}\n"
        "lr 0.001\n"
        "scheduler_patience 20\n"
        "stage2_scheduler_patience 20\n"
    )


def _fit_curve(theta, values):
    design = np.column_stack((np.sin(theta), np.cos(theta), np.sin(2 * theta)))
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _nrmse(prediction, target):
    scale = np.sqrt(np.mean(np.square(target)))
    return float(np.sqrt(np.mean(np.square(prediction - target))) / max(scale, 1e-14))


def _evaluate(model_file, blind, device):
    calculator = NEPCalculator(model_file, dtype=torch.float32, device=device)
    all_target = []
    all_prediction = []
    j_target, j_prediction = [], []
    d_target, d_prediction = [], []
    curves = []
    for group in blind:
        theta = np.array([frame["coordinate"] for frame in group])
        target = []
        prediction = []
        for frame in group:
            result = calculator.compute(
                SPECIES, frame["positions"], CELL, spins=frame["spins"])
            predicted_mforce = result["mforces"].detach().cpu().numpy()
            target.append(float(np.sum(
                frame["exact_mforces"] * frame["tangent"])))
            prediction.append(float(np.sum(
                predicted_mforce * frame["tangent"])))
        target = np.asarray(target)
        prediction = np.asarray(prediction)
        target_fit = _fit_curve(theta, target)
        prediction_fit = _fit_curve(theta, prediction)
        all_target.extend(target)
        all_prediction.extend(prediction)
        j_target.append(target_fit[0])
        j_prediction.append(prediction_fit[0])
        d_target.append(-target_fit[1])
        d_prediction.append(-prediction_fit[1])
        curves.append({
            "theta": theta.tolist(), "target": target.tolist(),
            "prediction": prediction.tolist(),
        })
    return {
        "generator_nrmse": _nrmse(np.asarray(all_prediction), np.asarray(all_target)),
        "j_nrmse": _nrmse(np.asarray(j_prediction), np.asarray(j_target)),
        "d_nrmse": _nrmse(np.asarray(d_prediction), np.asarray(d_target)),
        "j_target": j_target, "j_prediction": j_prediction,
        "d_target": d_target, "d_prediction": d_prediction,
        "curves": curves,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=("absolute", "curve", "joint"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train, valid, blind = _dataset()
    train_file = args.output / "train.xyz"
    valid_file = args.output / "valid.xyz"
    config_file = args.output / "nep.in"
    with_tangent = args.variant != "absolute"
    _write_xyz(train_file, train, with_tangent)
    _write_xyz(valid_file, valid, with_tangent)
    _write_config(config_file, args.epochs)
    model_dir = args.output / "model"
    train_nep(
        str(config_file), str(train_file), output_dir=str(model_dir),
        device=args.device, precision="float32", restart=False,
        prediction_interval=0, run_seed=args.seed,
        valid_file=str(valid_file), use_swa=False,
    )
    result = {
        "variant": args.variant, "seed": args.seed, "epochs": args.epochs,
        "metrics": _evaluate(model_dir / "nep_best.txt", blind, args.device),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"variant": args.variant, "seed": args.seed,
                      **{k: v for k, v in result["metrics"].items()
                         if k.endswith("nrmse")}}, indent=2))


if __name__ == "__main__":
    main()
