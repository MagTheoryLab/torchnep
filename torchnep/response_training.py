# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.
# TorchNEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Internal grouped magnetic-response supervision.

The production model remains one conservative scalar-energy ANN.  This
module only changes how controlled rotation scans supervise that energy:
it matches the centered magnetic-force generator curve directly, without
constructing an energy second difference or exposing J/D model outputs.
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
    "neutral-group-balanced-mean-plus-centered-mforce-generator-v3"


def select_generator_curve_frames(frames: Sequence[Dict]) -> List[Dict]:
    """Return validated rotation-scan frames in deterministic group order.

    A participating frame declares ``response_probe=rotation`` and must have
    a response group, a numeric angular coordinate, magnetic-force labels,
    and one per-atom ``spin_tangent = dS/dtheta`` vector.  Groups need at
    least three distinct coordinates.  Non-rotation frames are ignored.
    """
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
        tangent_flags = [
            "spin_tangent" in frame for _, frame in indexed_members
        ]
        # Backward-compatible metadata-only scans remain available for
        # group-disjoint splitting and external acceptance analysis.
        if not any(tangent_flags):
            continue
        if not all(tangent_flags):
            raise ValueError(
                f"rotation response_group {group!r} has spin_tangent on "
                "only part of the scan")
        groups[group] = []
        for index, frame in indexed_members:
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
            tangent = np.asarray(frame["spin_tangent"])
            if tangent.shape != (frame["natoms"], 3):
                raise ValueError(
                    f"spin_tangent at frame {index} must have shape "
                    f"({frame['natoms']}, 3), got {tangent.shape}")
            mforces = np.asarray(frame["mforces"])
            if mforces.shape != (frame["natoms"], 3):
                raise ValueError(
                    f"mforces at frame {index} must have shape "
                    f"({frame['natoms']}, 3), got {mforces.shape}")
            if not np.isfinite(tangent).all():
                raise ValueError(
                    f"spin_tangent at frame {index} contains NaN or Inf")
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
            spin_norm = np.linalg.norm(spins, axis=1)
            tangent_norm = np.linalg.norm(tangent, axis=1)
            dot = np.einsum("ij,ij->i", spins, tangent)
            tolerance = 1.0e-7 * np.maximum(
                spin_norm * tangent_norm, 1.0)
            if np.any(np.abs(dot) > tolerance):
                raise ValueError(
                    f"spin_tangent at frame {index} is not orthogonal to "
                    "the fixed-magnitude rotation spin")
            if np.any((spin_norm < 1.0e-12) & (tangent_norm > 1.0e-12)):
                raise ValueError(
                    f"spin_tangent at frame {index} rotates a zero spin")
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
        reference_magnitudes = np.linalg.norm(
            np.asarray(reference["spins"]), axis=1)
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
            if not np.allclose(
                    np.linalg.norm(frame["spins"], axis=1),
                    reference_magnitudes, rtol=1.0e-8, atol=1.0e-10):
                raise ValueError(
                    f"rotation response_group {group!r} changes spin "
                    "magnitudes")
        if all(np.allclose(
                frame["spins"], reference["spins"],
                rtol=0.0, atol=1.0e-12) for frame in members[1:]):
            raise ValueError(
                f"rotation response_group {group!r} contains no spin "
                "rotation across coordinates")
        tangent_power = sum(
            float(np.square(np.asarray(frame["spin_tangent"])).sum())
            for frame in members
        )
        if tangent_power == 0.0:
            raise ValueError(
                f"rotation response_group {group!r} has an all-zero "
                "spin_tangent generator")
        selected.extend(members)
    return selected


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
    physical shape is sine-like, cosine-like, or contains higher harmonics. A
    single RMS target-generator scale is shared by the centered and mean terms;
    production training passes the scale of the complete response subset so it
    is independent of batch packing.
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
    if normalization_scale is None:
        global_scale = group_balanced_generator_rms(
            target.detach(), group_ids
        )
    else:
        global_scale = torch.as_tensor(
            normalization_scale, dtype=target.dtype, device=target.device
        ).detach()
    global_scale = global_scale.clamp_min(absolute_floor)
    for group in torch.unique(group_ids, sorted=True):
        mask = group_ids == group
        y = target[mask]
        yp = prediction[mask]
        y_mean = y.mean()
        yp_mean = yp.mean()
        yc = y - y_mean
        ypc = yp - yp_mean

        # The shared total-generator RMS is neutral to phase/harmonic content
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
