# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.
# TorchNEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Sealed production-trainer A/B for grouped magnetic response.

Every run uses identical noisy E/MF labels, O3-C2 descriptor, scalar ANN,
split, seed, and epochs. ``standard`` runs supervise the controlled scans only
through their ordinary E/MF labels. ``generator`` runs additionally declare
``spin_tangent`` and activate the Stage-2 generator-curve objective. Explicit
magnetic-force weights make a global ``lambda_m`` sweep an independent control
for the targeted generator reweighting.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from torchnep import train_nep
from torchnep.nep import NEPCalculator
from torchnep.response_training import (
    GENERATOR_CURVE_MEAN_WEIGHT,
    GENERATOR_CURVE_SCALE_FLOOR,
    GENERATOR_CURVE_WEIGHT,
)
from torchnep.spin_diagnostics import (
    generator_reweighting_health,
    relative_vector_error,
    svd_health,
)


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


def _write_config(path, epochs, lambda_m, stage2_lambda_m, spin_mode=2):
    if lambda_m < 0.0 or stage2_lambda_m < 0.0:
        raise ValueError("magnetic-force weights must be nonnegative")
    path.write_text(
        "type 1 Fe\n"
        "cutoff 4.0 4.0\n"
        "n_max 4 4\n"
        "basis_size 8 8\n"
        "l_max 4 2 0\n"
        "neuron 16\n"
        f"spin_mode {spin_mode}\n"
        "spin_compress 2\n"
        "spin_l_max 2\n"
        "spin_order 3\n"
        "spin_soc 1\n"
        "spin_cutoff 4.0\n"
        "lambda_e 1.0\n"
        "lambda_f 0.0\n"
        "lambda_v 0.0\n"
        f"lambda_m {lambda_m:.16g}\n"
        "lambda_tau 0.0\n"
        "stage2 1\n"
        f"start_stage2 {max(2, epochs // 2)}\n"
        "stage2_lambda_e 0.20\n"
        "stage2_lambda_f 0.0\n"
        "stage2_lambda_v 0.0\n"
        f"stage2_lambda_m {stage2_lambda_m:.16g}\n"
        "stage2_lambda_tau 0.0\n"
        "stage2_lr 0.0005\n"
        "batch 16\n"
        "eval_batch 64\n"
        f"epoch {epochs}\n"
        "lr 0.001\n"
        "scheduler_patience 20\n"
        "stage2_scheduler_patience 20\n"
    )


def _recipe(
    variant, lambda_m, stage2_lambda_m,
    generator_weight=GENERATOR_CURVE_WEIGHT,
):
    if variant not in ("standard", "generator"):
        raise ValueError(f"unknown response benchmark variant: {variant}")
    if generator_weight < 0.0:
        raise ValueError("generator weight must be nonnegative")
    active_generator_weight = (
        float(generator_weight) if variant == "generator" else 0.0
    )
    return {
        "variant": variant,
        "lambda_m": float(lambda_m),
        "stage2_lambda_m": float(stage2_lambda_m),
        "generator_enabled": variant == "generator",
        "generator_weight": active_generator_weight,
        "label": (
            f"{variant} | lambda_m={float(lambda_m):g} -> "
            f"{float(stage2_lambda_m):g}"
            + (
                f" | lambda_R={active_generator_weight:g}"
                if variant == "generator" else ""
            )
        ),
    }


def _train_with_internal_generator_weight(recipe, **kwargs):
    """Use the same explicit response weight as production inputs."""
    config = Path(kwargs["config_file"])
    config.write_text(config.read_text() +
                      f"\nlambda_spin_response {recipe['generator_weight']}\n")
    train_nep(**kwargs)


def _fit_curve(theta, values):
    design = np.column_stack((np.sin(theta), np.cos(theta), np.sin(2 * theta)))
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _nrmse(prediction, target):
    scale = np.sqrt(np.mean(np.square(target)))
    return float(np.sqrt(np.mean(np.square(prediction - target))) / max(scale, 1e-14))


def _evaluate_generalization(calculator, frames):
    def evaluate_slice(selected):
        energy_target = []
        energy_prediction = []
        mforce_target = []
        mforce_prediction = []
        for frame in selected:
            exact_energy, exact_mforces, _ = _teacher(
                frame["positions"], frame["spins"]
            )
            result = calculator.compute(
                SPECIES, frame["positions"], CELL, spins=frame["spins"]
            )
            energy_target.append(exact_energy / len(SPECIES))
            energy_prediction.append(
                float(result["energy"].sum().item()) / len(SPECIES)
            )
            mforce_target.append(exact_mforces)
            mforce_prediction.append(
                result["mforces"].detach().cpu().numpy()
            )
        if not selected:
            return None
        energy_target = np.asarray(energy_target)
        energy_prediction = np.asarray(energy_prediction)
        mforce_target = np.concatenate(mforce_target, axis=0)
        mforce_prediction = np.concatenate(mforce_prediction, axis=0)
        return {
            "structures": len(selected),
            "energy_rmse_ev_atom": float(np.sqrt(np.mean(np.square(
                energy_prediction - energy_target
            )))),
            "mforce_rmse_ev_mub": float(np.sqrt(np.mean(np.square(
                mforce_prediction - mforce_target
            )))),
            "mforce_nrmse": _nrmse(mforce_prediction, mforce_target),
        }

    ordinary = [frame for frame in frames if "group" not in frame]
    response = [frame for frame in frames if "group" in frame]
    return {
        "all": evaluate_slice(frames),
        "ordinary": evaluate_slice(ordinary),
        "response": evaluate_slice(response),
    }


def _evaluate(model_file, blind, general_frames, device):
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
        "generalization": _evaluate_generalization(
            calculator, general_frames
        ),
    }


def _descriptor_health(calculator, frames):
    scaled = []
    for frame in frames:
        scaled.append(calculator.get_descriptor(
            SPECIES, frame["positions"], CELL, spins=frame["spins"]
        ))
    scaled = np.concatenate(scaled, axis=0)[:, calculator.struct_dim:]
    scaler = calculator.q_scaler.detach().cpu().numpy()[
        calculator.struct_dim:
    ]
    raw = scaled / scaler[None, :]
    descriptor = calculator.spin_descriptor
    labels = []
    for token in descriptor.schema:
        labels.extend(
            f"{token.name}[{offset}]"
            for offset in range(token.stop - token.start)
        )
    sectors = {}
    previous = 0
    for order in range(1, descriptor.spin_order + 1):
        stop = descriptor.order_dims[order]
        sectors[f"O{order}_increment"] = {
            "raw": svd_health(
                raw[:, previous:stop], labels=labels[previous:stop]
            ),
            "scaled": svd_health(
                scaled[:, previous:stop], labels=labels[previous:stop]
            ),
            "correlation": svd_health(
                raw[:, previous:stop], standardize=True,
                labels=labels[previous:stop],
            ),
        }
        previous = stop
    return {
        "raw": svd_health(raw, labels=labels),
        "scaled": svd_health(scaled, labels=labels),
        "correlation": svd_health(
            raw, standardize=True, labels=labels
        ),
        "sectors": sectors,
    }


def _finite_difference_health(calculator, frame, step=1.0e-5):
    positions = np.asarray(frame["positions"], dtype=np.float64)
    spins = np.asarray(frame["spins"], dtype=np.float64)
    cell = np.asarray(CELL, dtype=np.float64)
    n_atoms = positions.shape[0]
    spin_dim = calculator.spin_descriptor.dim

    def compute(pos, box, spin, descriptor=False):
        result = calculator.compute(
            SPECIES, pos, box, spins=spin,
            compute_descriptor=descriptor,
        )
        if descriptor:
            return result["descriptor"].detach().cpu().numpy()[
                :, calculator.struct_dim:
            ]
        return result

    def total_energy(pos, box, spin):
        return float(compute(pos, box, spin)["energy"].sum().item())

    analytic = compute(positions, cell, spins)
    force_fd = np.empty_like(positions)
    mforce_fd = np.empty_like(spins)
    position_jacobian = np.empty((n_atoms * spin_dim, n_atoms * 3))
    spin_jacobian = np.empty_like(position_jacobian)
    for atom in range(n_atoms):
        for axis in range(3):
            column = 3 * atom + axis
            plus = positions.copy()
            minus = positions.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            force_fd[atom, axis] = -(
                total_energy(plus, cell, spins)
                - total_energy(minus, cell, spins)
            ) / (2.0 * step)
            position_jacobian[:, column] = (
                compute(plus, cell, spins, descriptor=True)
                - compute(minus, cell, spins, descriptor=True)
            ).reshape(-1) / (2.0 * step)

            plus_spin = spins.copy()
            minus_spin = spins.copy()
            plus_spin[atom, axis] += step
            minus_spin[atom, axis] -= step
            mforce_fd[atom, axis] = -(
                total_energy(positions, cell, plus_spin)
                - total_energy(positions, cell, minus_spin)
            ) / (2.0 * step)
            spin_jacobian[:, column] = (
                compute(positions, cell, plus_spin, descriptor=True)
                - compute(positions, cell, minus_spin, descriptor=True)
            ).reshape(-1) / (2.0 * step)

    plus_positions = positions * (1.0 + step)
    minus_positions = positions * (1.0 - step)
    plus_cell = cell * (1.0 + step)
    minus_cell = cell * (1.0 - step)
    strain_energy_fd = (
        total_energy(plus_positions, plus_cell, spins)
        - total_energy(minus_positions, minus_cell, spins)
    ) / (2.0 * step)
    strain_descriptor = (
        compute(plus_positions, plus_cell, spins, descriptor=True)
        - compute(minus_positions, minus_cell, spins, descriptor=True)
    ) / (2.0 * step)
    analytic_virial = analytic["virial"].sum(dim=0).reshape(3, 3)
    strain_energy_analytic = -float(torch.trace(analytic_virial).item())

    sector_norms = {}
    previous = 0
    for order in range(1, calculator.spin_descriptor.spin_order + 1):
        stop = calculator.spin_descriptor.order_dims[order]
        position_sector = position_jacobian.reshape(
            n_atoms, spin_dim, -1
        )[:, previous:stop]
        spin_sector = spin_jacobian.reshape(
            n_atoms, spin_dim, -1
        )[:, previous:stop]
        strain_sector = strain_descriptor[:, previous:stop]
        sector_norms[f"O{order}_increment"] = {
            "position_rms": float(np.sqrt(np.mean(position_sector ** 2))),
            "spin_rms": float(np.sqrt(np.mean(spin_sector ** 2))),
            "isotropic_strain_rms": float(
                np.sqrt(np.mean(strain_sector ** 2))
            ),
        }
        previous = stop

    spin_limits = {}
    for label, scale in (("normal", 1.0), ("near_zero", 1.0e-8), ("zero", 0.0)):
        values = compute(
            positions, cell, spins * scale, descriptor=True
        )
        spin_limits[label] = {
            "finite": bool(np.isfinite(values).all()),
            "max_abs": float(np.max(np.abs(values))),
            "exact_zero": bool(np.count_nonzero(values) == 0),
        }

    force = analytic["forces"].detach().cpu().numpy()
    mforce = analytic["mforces"].detach().cpu().numpy()
    translation = np.zeros((n_atoms * 3, 3))
    for axis in range(3):
        translation[axis::3, axis] = 1.0 / np.sqrt(n_atoms)
    complete_basis, _ = np.linalg.qr(translation, mode="complete")
    translation_free = complete_basis[:, 3:]
    quotient_position_jacobian = position_jacobian @ translation_free
    return {
        "energy_derivative_closure": {
            "force": relative_vector_error(force, force_fd),
            "mforce": relative_vector_error(mforce, mforce_fd),
            "isotropic_strain": relative_vector_error(
                np.array([strain_energy_analytic]),
                np.array([strain_energy_fd]),
            ),
        },
        "descriptor_jacobian": {
            "position_with_translation_modes": svd_health(
                position_jacobian, center=False
            ),
            "position": svd_health(
                quotient_position_jacobian, center=False
            ),
            "spin": svd_health(spin_jacobian, center=False),
            "sector_rms": sector_norms,
        },
        "spin_limits": spin_limits,
    }


def _training_log_health(path):
    values = np.loadtxt(path, comments="#", ndmin=2)
    finite = np.isfinite(values)
    active_columns = finite.any(axis=0)
    invalid_columns = 0
    leading_nan_columns = 0
    for column in range(values.shape[1]):
        indices = np.flatnonzero(finite[:, column])
        if indices.size == 0:
            continue
        first = int(indices[0])
        leading_nan_columns += int(first > 0)
        if not finite[first:, column].all():
            invalid_columns += 1
    observed_finite = bool(
        invalid_columns == 0 and not np.isinf(values).any()
    )
    finite_values = values[finite]
    return {
        "epochs": int(values.shape[0]),
        "columns": int(values.shape[1]),
        "active_columns": int(active_columns.sum()),
        "inactive_nan_columns": int((~active_columns).sum()),
        "leading_nan_columns": leading_nan_columns,
        "invalid_nonfinite_columns": invalid_columns,
        "all_observed_finite": observed_finite,
        "max_abs_observed": float(np.max(np.abs(finite_values))),
    }


def _response_reweighting_health(frames, recipe):
    response_frames = [frame for frame in frames if "group" in frame]
    tangents = np.concatenate(
        [frame["tangent"] for frame in response_frames], axis=0
    )
    targets = np.concatenate(
        [frame["mforces"] for frame in response_frames], axis=0
    )
    struct_idx = np.repeat(
        np.arange(len(response_frames), dtype=np.int64), len(SPECIES)
    )
    group_lookup = {}
    group_ids = []
    for frame in response_frames:
        group_ids.append(group_lookup.setdefault(
            frame["group"], len(group_lookup)
        ))
    return generator_reweighting_health(
        tangents,
        targets,
        struct_idx,
        group_ids,
        mforce_weight=recipe["stage2_lambda_m"],
        generator_weight=recipe["generator_weight"],
        ordinary_mforce_components=sum(
            np.asarray(frame["mforces"]).size for frame in frames
        ),
        mean_weight=GENERATOR_CURVE_MEAN_WEIGHT,
        scale_floor=GENERATOR_CURVE_SCALE_FLOOR,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=("standard", "generator"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lambda-m", type=float, default=0.10)
    parser.add_argument("--stage2-lambda-m", type=float, default=1.0)
    parser.add_argument(
        "--generator-weight", type=float, default=GENERATOR_CURVE_WEIGHT,
        help="Internal benchmark seam; this is not a nep.in parameter.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--spin-mode", type=int, choices=(2, 3), default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Reuse an existing model/nep_best.txt and refresh result.json.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train, valid, blind = _dataset()
    train_file = args.output / "train.xyz"
    valid_file = args.output / "valid.xyz"
    config_file = args.output / "nep.in"
    recipe = _recipe(
        args.variant, args.lambda_m, args.stage2_lambda_m,
        args.generator_weight,
    )
    with_tangent = recipe["generator_enabled"]
    model_dir = args.output / "model"
    if args.evaluate_only:
        result_path = args.output / "result.json"
        if result_path.is_file():
            stored = json.loads(result_path.read_text())
            if stored.get("variant") != args.variant \
                    or int(stored.get("seed", -1)) != args.seed \
                    or int(stored.get("spin_mode", 2)) != args.spin_mode:
                raise ValueError(
                    "evaluate-only variant/seed/spin-mode does not match "
                    "result.json"
                )
            recipe = stored.get("recipe", recipe)
        if not (model_dir / "nep_best.txt").is_file():
            raise FileNotFoundError(
                f"missing existing model: {model_dir / 'nep_best.txt'}"
            )
    else:
        _write_xyz(train_file, train, with_tangent)
        _write_xyz(valid_file, valid, with_tangent)
        if with_tangent:
            _write_xyz(train_file.parent / "response.xyz",
                       [frame for frame in train if "group" in frame], True)
        _write_config(
            config_file, args.epochs, args.lambda_m, args.stage2_lambda_m,
            spin_mode=args.spin_mode,
        )
        _train_with_internal_generator_weight(
            recipe,
            config_file=str(config_file), data_file=str(train_file),
            output_dir=str(model_dir),
            device=args.device, precision="float32", restart=False,
            prediction_interval=0, run_seed=args.seed,
            valid_file=str(valid_file), use_swa=False,
        )
    result = {
        "variant": args.variant, "seed": args.seed, "epochs": args.epochs,
        "spin_mode": args.spin_mode,
        "recipe": recipe,
        "metrics": _evaluate(
            model_dir / "nep_best.txt", blind, valid, args.device
        ),
    }
    health_calculator = NEPCalculator(
        model_dir / "nep_best.txt", dtype=torch.float64, device=args.device
    )
    result["health"] = {
        "descriptor": {
            "training": _descriptor_health(health_calculator, train),
            "blind": _descriptor_health(
                health_calculator,
                [frame for group in blind for frame in group],
            ),
        },
        "derivatives": _finite_difference_health(
            health_calculator, blind[0][len(blind[0]) // 2]
        ),
        "response_reweighting": _response_reweighting_health(train, recipe),
        "training_log": _training_log_health(model_dir / "loss.out"),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"variant": args.variant, "seed": args.seed,
                      **{k: v for k, v in result["metrics"].items()
                         if k.endswith("nrmse")}}, indent=2))


if __name__ == "__main__":
    main()
