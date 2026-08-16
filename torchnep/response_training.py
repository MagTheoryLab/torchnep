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


GENERATOR_CURVE_WEIGHT = 0.3
GENERATOR_CURVE_HUBER_DELTA = 1.0
GENERATOR_CURVE_MEAN_WEIGHT = 0.25
GENERATOR_CURVE_SCHEMA = "mean-plus-centered-mforce-generator-v2"


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
                key for key in ("response_coordinate", "mforces")
                if key not in frame
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
            tangent = np.asarray(frame["spin_tangent"])
            if tangent.shape != (frame["natoms"], 3):
                raise ValueError(
                    f"spin_tangent at frame {index} must have shape "
                    f"({frame['natoms']}, 3), got {tangent.shape}")
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
) -> torch.Tensor:
    """Robust loss on mean plus centered magnetic-force generator curves.

    The target and prediction are centered independently within each group,
    isolating the angular shape. A separate group-mean term preserves the
    constant response component without naming or exposing a D-like output.
    A detached linear-signal/noise estimate downweights inconsistent groups.
    Normalisation is global across the batch, so a nearly-zero group cannot
    acquire an arbitrarily large weight.
    """
    prediction = _frame_generator(
        predicted_mforces, spin_tangent, struct_idx, num_structures
    )
    target = _frame_generator(
        target_mforces, spin_tangent, struct_idx, num_structures
    )
    centered_prediction = torch.empty_like(prediction)
    centered_target = torch.empty_like(target)
    reliability = torch.empty_like(target)
    mean_prediction = []
    mean_target = []
    for group in torch.unique(group_ids, sorted=True):
        mask = group_ids == group
        x = coordinates[mask]
        x = x - x.mean()
        y = target[mask]
        yp = prediction[mask]
        yc = y - y.mean()
        mean_target.append(y.mean())
        mean_prediction.append(yp.mean())
        centered_target[mask] = yc
        centered_prediction[mask] = yp - yp.mean()
        denominator = x.square().sum().clamp_min(torch.finfo(x.dtype).eps)
        slope = (x * yc).sum() / denominator
        signal = slope * x
        noise = yc - signal
        signal_power = signal.square().mean()
        noise_power = noise.square().mean()
        score = signal_power / (
            signal_power + noise_power + torch.finfo(x.dtype).eps
        )
        reliability[mask] = score.detach()

    scale = centered_target.detach().square().mean().sqrt().clamp_min(
        64.0 * torch.finfo(target.dtype).eps
    )
    residual = (centered_prediction - centered_target) / scale
    absolute = residual.abs()
    delta = residual.new_tensor(float(huber_delta))
    huber = torch.where(
        absolute <= delta,
        0.5 * residual.square(),
        delta * (absolute - 0.5 * delta),
    )
    weight = reliability.clamp_min(0.05)
    shape_loss = (weight * huber).sum() / weight.sum().clamp_min(1.0)
    if mean_weight <= 0:
        return shape_loss

    mean_prediction = torch.stack(mean_prediction)
    mean_target = torch.stack(mean_target)
    mean_scale = mean_target.detach().square().mean().sqrt().clamp_min(
        64.0 * torch.finfo(target.dtype).eps
    )
    mean_residual = (mean_prediction - mean_target) / mean_scale
    mean_absolute = mean_residual.abs()
    mean_huber = torch.where(
        mean_absolute <= delta,
        0.5 * mean_residual.square(),
        delta * (mean_absolute - 0.5 * delta),
    ).mean()
    return shape_loss + float(mean_weight) * mean_huber
