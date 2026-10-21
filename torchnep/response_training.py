# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.
# TorchNEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Internal grouped magnetic-response supervision.

The production model remains one conservative scalar-energy ANN.  This
module supervises group-centered energy differences and/or magnetic-force
generator curves, without constructing energy second differences or exposing
J/D model outputs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import torch


GENERATOR_CURVE_WEIGHT = 0.1
GENERATOR_CURVE_HUBER_DELTA = 1.0
GENERATOR_CURVE_MEAN_WEIGHT = 0.25
GENERATOR_CURVE_SCALE_FLOOR = 1.0e-6
GENERATOR_CURVE_SCHEMA = \
    "local-sphere-log-group-rms-mforce-generator-v5"


def _spin_path_tangents(spins, coordinates):
    """Differentiate magnitude and direction separately along a sampled path.

    Use sphere-log angular displacements and separate magnitude differences.
    Constant-rate great-circle rotation is exact on resolved stencils; other
    paths retain discretization error. Zero-moment pairs use Cartesian
    differences and antipodal nonzero pairs are rejected as ambiguous.
    """
    tangents = np.zeros_like(spins, dtype=np.float64)
    for k, x in enumerate(coordinates):
        nodes = [0, 1, 2] if k == 0 else (
            [len(coordinates)-3, len(coordinates)-2, len(coordinates)-1]
            if k == len(coordinates)-1 else [k-1, k, k+1])
        center = spins[k]
        r = np.linalg.norm(center, axis=-1, keepdims=True)
        # A resolved zero crossing has no sphere direction. Use Cartesian
        # differences for the whole local stencil, not only its zero pairs.
        resolved = np.all(np.linalg.norm(spins[nodes], axis=-1) > 1e-12, axis=0)
        for j, node in enumerate(nodes):
            a, b = [coordinates[nodes[i]] for i in range(3) if i != j]
            weight = (2*x-a-b)/((coordinates[node]-a)*(coordinates[node]-b))
            neighbour = spins[node]
            q = np.linalg.norm(neighbour, axis=-1, keepdims=True)
            regular = ((r > 1e-12) & (q > 1e-12)).ravel() & resolved
            displacement = neighbour-center
            n, u = center[regular]/r[regular], neighbour[regular]/q[regular]
            cosine = np.clip(np.sum(n*u, axis=-1, keepdims=True), -1, 1)
            tangent = u-cosine*n
            sine = np.linalg.norm(tangent, axis=-1, keepdims=True)
            if np.any((cosine < 0) & (sine < 1e-10)):
                raise ValueError("Antipodal spins in response rotation stencil; add intermediate frames")
            factor = np.ones_like(sine)
            np.divide(np.arctan2(sine, cosine), sine, out=factor, where=sine > 1e-12)
            displacement[regular] = r[regular]*factor*tangent+(q[regular]-r[regular])*n
            tangents[k] += weight*displacement
    return tangents


def select_generator_curve_frames(
    frames: Sequence[Dict], active_species=None
) -> List[Dict]:
    """Return validated rotation-scan frames in deterministic group order.

    A participating frame declares ``response_probe=rotation`` and must have
    a response group, a numeric angular coordinate, converged DFT spins, and
    magnetic-force labels.  ``dS/dtheta`` is derived from the complete group;
    it is not a user-provided label.  Groups need at least three distinct
    coordinates.  Non-rotation frames are ignored.
    """
    active = None if active_species is None else set(active_species)
    candidates = defaultdict(list)
    for index, frame in enumerate(frames):
        if frame.get("response_probe") != "rotation":
            continue
        if "response_group" not in frame:
            raise ValueError(
                f"rotation response frame {index} is missing: "
                "response_group")
        candidates[str(frame["response_group"])].append((index, frame))

    groups = {}
    for group, indexed_members in candidates.items():
        groups[group] = []
        for index, frame in indexed_members:
            if "spin_tangent" in frame:
                raise ValueError(
                    "spin_tangent is not an input label; the trainer derives "
                    "dS/dtheta from each final DFT spin path")
            missing = [
                key for key in (
                    "response_coordinate", "mforces", "spins", "species",
                    "positions", "cell",
                ) if key not in frame
            ]
            if missing:
                raise ValueError(
                    f"rotation response frame {index} is missing: "
                    + ", ".join(missing)
                )
            coordinate = frame["response_coordinate"]
            if not isinstance(
                    coordinate,
                    (int, float, np.integer, np.floating)):
                raise ValueError(
                    "rotation response_coordinate must be numeric, got "
                    f"{coordinate!r} at frame {index}")
            if not np.isfinite(float(coordinate)):
                raise ValueError(
                    "rotation response_coordinate must be finite, got "
                    f"{coordinate!r} at frame {index}")
            mforces = np.asarray(frame["mforces"])
            if mforces.shape != (frame["natoms"], 3):
                raise ValueError(
                    f"mforces at frame {index} must have shape "
                    f"({frame['natoms']}, 3), got {mforces.shape}")
            if not np.isfinite(mforces).all():
                raise ValueError(
                    f"mforces at frame {index} contains NaN or Inf")
            spins = np.asarray(frame["spins"])
            positions = np.asarray(frame["positions"])
            cell = np.asarray(frame["cell"])
            if spins.shape != (frame["natoms"], 3):
                raise ValueError(
                    f"spins at frame {index} must have shape "
                    f"({frame['natoms']}, 3), got {spins.shape}")
            if positions.shape != (frame["natoms"], 3):
                raise ValueError(
                    f"positions at frame {index} must have shape "
                    f"({frame['natoms']}, 3), got {positions.shape}")
            if cell.shape != (3, 3):
                raise ValueError(
                    f"cell at frame {index} must have shape (3, 3), "
                    f"got {cell.shape}")
            if len(frame["species"]) != frame["natoms"]:
                raise ValueError(
                    f"species at frame {index} must contain natoms entries")
            if not np.isfinite(spins).all() or \
                    not np.isfinite(positions).all() or \
                    not np.isfinite(cell).all():
                raise ValueError(
                    f"rotation response frame {index} has nonfinite "
                    "spin or geometry values")
            groups[group].append(frame)

    selected = []
    for group in sorted(groups):
        members = sorted(
            groups[group], key=lambda frame: float(frame["response_coordinate"])
        )
        coordinates = {
            float(frame["response_coordinate"]) for frame in members
        }
        if len(coordinates) < 3:
            raise ValueError(
                f"rotation response_group {group!r} needs at least three "
                "distinct response_coordinate values"
            )
        reference = members[0]
        reference_positions = np.asarray(reference["positions"])
        reference_cell = np.asarray(reference["cell"])
        for frame in members[1:]:
            if frame["natoms"] != reference["natoms"] or \
                    tuple(frame["species"]) != tuple(reference["species"]):
                raise ValueError(
                    f"rotation response_group {group!r} changes atom count, "
                    "species, or atom order")
            if not np.allclose(
                    frame["positions"], reference_positions,
                    rtol=0.0, atol=1.0e-10) or not np.allclose(
                    frame["cell"], reference_cell,
                    rtol=0.0, atol=1.0e-10):
                raise ValueError(
                    f"rotation response_group {group!r} changes geometry; "
                    "put each strained geometry in a separate group")
        if all(np.allclose(
                frame["spins"], reference["spins"],
                rtol=0.0, atol=1.0e-12) for frame in members[1:]):
            raise ValueError(
                f"rotation response_group {group!r} contains no spin "
                "rotation across coordinates")
        coordinate_array = np.asarray([
            float(frame["response_coordinate"]) for frame in members
        ], dtype=np.float64)
        if np.any(np.diff(coordinate_array) <= 0.0):
            raise ValueError(
                f"rotation response_group {group!r} has duplicate "
                "response_coordinate values")
        spin_path = np.asarray([
            np.asarray(frame["spins"], dtype=np.float64) for frame in members
        ])
        if active is not None:
            spin_path[:, [s not in active for s in reference["species"]], :] = 0.0
        tangents = _spin_path_tangents(spin_path, coordinate_array)
        tangent_power = float(np.square(tangents).sum())
        if tangent_power == 0.0:
            raise ValueError(
                f"rotation response_group {group!r} has an all-zero "
                "spin-path generator")
        for frame, tangent in zip(members, tangents):
            derived = dict(frame)
            if active is not None:
                tangent = tangent.copy()
                inactive = [
                    species not in active for species in frame["species"]
                ]
                tangent[inactive] = 0.0
            derived["spin_tangent"] = tangent
            selected.append(derived)
    return selected


def select_response_frames(frames, active_species=None, target="mforce"):
    """Energy groups need no inferred path derivative or fixed geometry.

    Energy differences may include the small relaxations in actual DFT frames;
    they must not be relabelled as a pure angular derivative. Complete labels
    are validated by load_response_dataset even for energy-only supervision.
    """
    if target not in ("energy", "mforce", "both"):
        raise ValueError("unknown spin_response_target")
    if target != "energy":
        return select_generator_curve_frames(frames, active_species)
    groups = defaultdict(list)
    for frame in frames:
        if frame.get("response_probe") not in ("energy", "rotation"):
            continue
        if "response_group" not in frame:
            raise ValueError("energy response frame requires response_group")
        if "spin_tangent" in frame:
            raise ValueError("spin_tangent is not an input label")
        groups[str(frame["response_group"])].append(frame)
    selected = []
    for name in sorted(groups):
        members = groups[name]
        if len(members) < 2:
            raise ValueError("energy response_group requires at least two frames")
        species = tuple(members[0]["species"])
        for index, frame in enumerate(members):
            if tuple(frame["species"]) != species:
                raise ValueError("energy response_group changes composition or atom order")
            derived = dict(frame)
            # This coordinate only orders/identifies energy observations.
            derived["response_coordinate"] = float(frame.get("response_coordinate", index))
            if not np.isfinite(derived["response_coordinate"]):
                raise ValueError("response_coordinate must be finite")
            derived["spin_tangent"] = np.zeros_like(frame["spins"])
            selected.append(derived)
    return selected


def configure_response_store(store, config):
    if store is not None:
        store.response_target = config.get("spin_response_target", "mforce")
        store.response_energy_scale = config.get("spin_response_energy_scale", 1e-6)


def _huber_residual(x):
    """Signed square root of Huber loss, finite derivative at zero."""
    a = x.abs()
    factor = torch.where(a <= 1, torch.full_like(a, 0.5),
                         (a - 0.5) / a.square().clamp_min(1e-30))
    return x * factor.sqrt()


def response_residual(result, batch, target="mforce",
                      generator_scale=None, energy_scale=1e-6):
    """Residual with squared norm exactly equal to the response objective.

    Energy uses centered per-atom errors divided by an explicit eV/atom scale.
    No target-derived division by a vanishing MAE amplitude; no fabricated MF.
    The mforce branch preserves the existing mean-plus-shape Huber objective.
    """
    if target not in ("energy", "mforce", "both"):
        raise ValueError("unknown spin_response_target")
    if not np.isfinite(energy_scale) or energy_scale <= 0:
        raise ValueError("response energy scale must be positive")
    groups = batch["response_group_ids"]
    unique = torch.unique(groups)
    parts = []
    if target in ("energy", "both"):
        error = (result["Etot"] - batch["energy"]) / batch["natoms"]
        for g in unique:
            values = error[groups == g]
            parts.append((values - values.mean()) /
                         (energy_scale * (len(unique) * values.numel()) ** 0.5))
    if target in ("mforce", "both"):
        args = (batch["spin_tangent"], batch["struct_idx"], batch["num_structures"])
        prediction = _frame_generator(result["mforces"], *args)
        reference = _frame_generator(batch["mforces"], *args)
        for g in unique:
            x, y = prediction[groups == g], reference[groups == g]
            scale = y.detach().square().mean().sqrt().clamp_min(GENERATOR_CURVE_SCALE_FLOOR)
            parts.append(_huber_residual(((x-x.mean())-(y-y.mean())) / scale)
                         / (len(unique)*x.numel()) ** 0.5)
            parts.append(_huber_residual((x.mean()-y.mean()).reshape(1) / scale)
                         * (GENERATOR_CURVE_MEAN_WEIGHT/len(unique)) ** 0.5)
    return torch.cat(parts)


def response_from_store(model, batch, store, backend, create_graph=True,
                        generator_scale=None):
    batch = dict(batch)
    batch["need_mforce"] = getattr(store, "response_target", "mforce") != "energy"
    batch["need_tau"] = False
    result = model.compute_properties_cached(
        batch, need_forces=False, need_virial=False, backend=backend,
        create_graph=create_graph)
    return response_residual(
        result, batch, getattr(store, "response_target", "mforce"),
        store.generator_curve_scale if generator_scale is None else generator_scale,
        getattr(store, "response_energy_scale", 1e-6))


def complete_group_batches(
    group_ids: Sequence[int], batch_size: int, seed: int = 0
) -> List[List[int]]:
    """Pack complete response groups without splitting a curve across batches."""
    groups = defaultdict(list)
    for index, group in enumerate(group_ids):
        groups[int(group)].append(index)
    keys = sorted(groups)
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(keys), generator=generator).tolist()
    batches: List[List[int]] = []
    current: List[int] = []
    for position in order:
        unit = groups[keys[position]]
        if current and len(current) + len(unit) > batch_size:
            batches.append(current)
            current = []
        current.extend(unit)
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def normalized_group_batches(
    group_ids: Sequence[int], batch_size: int, seed: int = 0
):
    """Return complete-group batches and their share of one response update.

    Each response group has weight ``1 / num_groups`` irrespective of the
    number of scan points it contains.  The returned batch weights sum to one,
    so accumulating all weighted batch gradients before one optimizer step is
    invariant to batch packing and to replication of the full group set.
    """
    batches = complete_group_batches(group_ids, batch_size, seed)
    groups = {int(group) for group in group_ids}
    if not groups:
        return []
    normalizer = float(len(groups))
    weighted = []
    for indices in batches:
        batch_groups = {int(group_ids[index]) for index in indices}
        weighted.append((indices, len(batch_groups) / normalizer))
    return weighted


def balanced_group_step_batches(
    group_ids: Sequence[int], num_steps: int, seed: int = 0
):
    """Assign complete response groups to optimizer steps with exact balance.

    Returns one ``[(indices, weight), ...]`` list per optimizer step.  Across
    the epoch every group has total coefficient ``num_steps / num_groups`` and
    all returned weights sum to ``num_steps``.  When there are fewer ordinary
    steps than groups, several complete groups share a step; otherwise groups
    cycle and repeated groups receive the inverse of their occurrence count.
    This keeps the mean response strength unchanged while removing the
    first-groups-win bias of a truncated modulo cycle.
    """
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    if num_steps == 0:
        return []
    units = complete_group_batches(group_ids, batch_size=1, seed=seed)
    num_groups = len(units)
    if num_groups == 0:
        return []

    if num_steps < num_groups:
        # Partition one full shuffled group pass over the available steps.
        # generator_curve_loss averages the groups within a combined batch,
        # hence h * num_steps / num_groups gives every group coefficient
        # num_steps / num_groups exactly.
        quotient, remainder = divmod(num_groups, num_steps)
        result = []
        cursor = 0
        for step in range(num_steps):
            width = quotient + (1 if step < remainder else 0)
            selected = units[cursor:cursor + width]
            cursor += width
            indices = [index for unit in selected for index in unit]
            result.append([(
                indices, width * num_steps / float(num_groups)
            )])
        return result

    selected_units = [step % num_groups for step in range(num_steps)]
    counts = [0] * num_groups
    for unit in selected_units:
        counts[unit] += 1
    return [[(
        units[unit], num_steps / float(num_groups * counts[unit])
    )] for unit in selected_units]


def _frame_generator(
    mforces: torch.Tensor,
    tangent: torch.Tensor,
    struct_idx: torch.Tensor,
    num_structures: int,
) -> torch.Tensor:
    atomic = (mforces * tangent).sum(dim=-1)
    value = atomic.new_zeros(num_structures)
    value.scatter_add_(0, struct_idx, atomic)
    return value


def group_balanced_generator_rms(
    values: torch.Tensor, group_ids
) -> torch.Tensor:
    """RMS with one equal contribution from every complete response group."""
    groups = torch.as_tensor(
        group_ids, dtype=torch.long, device=values.device
    ).reshape(-1)
    if groups.numel() != values.numel():
        raise ValueError("group_ids must contain one value per generator value")
    powers = [
        values[groups == group].square().mean()
        for group in torch.unique(groups, sorted=True)
    ]
    if not powers:
        return values.new_zeros(())
    return torch.stack(powers).mean().sqrt()


def generator_curve_loss(
    predicted_mforces: torch.Tensor,
    target_mforces: torch.Tensor,
    spin_tangent: torch.Tensor,
    struct_idx: torch.Tensor,
    group_ids: torch.Tensor,
    coordinates: torch.Tensor,
    num_structures: int,
    huber_delta: float = GENERATOR_CURVE_HUBER_DELTA,
    mean_weight: float = GENERATOR_CURVE_MEAN_WEIGHT,
    normalization_scale=None,
) -> torch.Tensor:
    """Robust, group-balanced loss on magnetic-force generator curves.

    The target and prediction are centered independently within each group,
    isolating the angular shape. A separate group-mean term preserves the
    constant response component without naming or exposing a D-like output.
    Every group contributes equally, regardless of scan length or whether its
    physical shape is sine-like, cosine-like, or contains higher harmonics.
    Each complete group uses its own uncentered target RMS for both terms.
    ``normalization_scale`` is retained for call compatibility but no longer
    overrides group-local scaling (v5); complete groups must stay together.
    ``coordinates`` remains part of the stable call seam, but target values are
    never regressed against it to manufacture training weights.
    """
    prediction = _frame_generator(
        predicted_mforces, spin_tangent, struct_idx, num_structures
    )
    target = _frame_generator(
        target_mforces, spin_tangent, struct_idx, num_structures
    )
    if group_ids.numel() != prediction.numel():
        raise ValueError(
            "response_group_ids must contain one value per structure")
    if coordinates.numel() != prediction.numel():
        raise ValueError(
            "response_coordinates must contain one value per structure")

    shape_losses = []
    mean_losses = []
    delta = target.new_tensor(float(huber_delta))
    absolute_floor = target.new_tensor(GENERATOR_CURVE_SCALE_FLOOR)
    for group in torch.unique(group_ids, sorted=True):
        mask = group_ids == group
        y = target[mask]
        global_scale = y.detach().square().mean().sqrt().clamp_min(absolute_floor)
        yp = prediction[mask]
        y_mean = y.mean()
        yp_mean = yp.mean()
        yc = y - y_mean
        ypc = yp - yp_mean

        # The group-local total-generator RMS is neutral to phase/harmonic content
        # and, unlike a group-mean scale, stays finite for physically important
        # zero-mean scans.  The absolute floor defines the all-zero-label case.
        shape_residual = (ypc - yc) / global_scale
        shape_absolute = shape_residual.abs()
        shape_losses.append(torch.where(
            shape_absolute <= delta,
            0.5 * shape_residual.square(),
            delta * (shape_absolute - 0.5 * delta),
        ).mean())

        if mean_weight > 0:
            mean_residual = (yp_mean - y_mean) / global_scale
            mean_absolute = mean_residual.abs()
            mean_losses.append(torch.where(
                mean_absolute <= delta,
                0.5 * mean_residual.square(),
                delta * (mean_absolute - 0.5 * delta),
            ))

    if not shape_losses:
        return prediction.sum() * 0.0
    shape_loss = torch.stack(shape_losses).mean()
    if mean_weight <= 0:
        return shape_loss
    return shape_loss + float(mean_weight) * torch.stack(mean_losses).mean()
