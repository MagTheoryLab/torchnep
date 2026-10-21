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

"""Numerical health summaries for O/C descriptor acceptance tests.

These helpers diagnose a sampled matrix; they do not claim that a descriptor
is well conditioned for data outside that sample.  Constant columns are kept
separate from the active-column condition number so symmetry-forced zeros do
not get disguised as a very large finite value.
"""

from typing import Dict, Optional, Sequence

import numpy as np


def svd_health(
    values: np.ndarray,
    *,
    center: bool = True,
    standardize: bool = False,
    labels: Optional[Sequence[str]] = None,
) -> Dict:
    """Return a JSON-safe rank/conditioning certificate for a 2-D matrix.

    Descriptor sample matrices should use the default centering.  Jacobians
    are linear maps rather than samples and must pass ``center=False``.
    """
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("SVD health input must be a 2-D matrix")
    if matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("SVD health requires at least two rows and one column")
    if not np.isfinite(matrix).all():
        raise ValueError("SVD health input contains NaN or Inf")
    if labels is not None and len(labels) != matrix.shape[1]:
        raise ValueError("SVD health labels must match the feature count")

    magnitude = np.maximum(np.max(np.abs(matrix), axis=0), 1.0)
    activity = (
        np.ptp(matrix, axis=0)
        if center else np.max(np.abs(matrix), axis=0)
    )
    active = activity > 64.0 * np.finfo(np.float64).eps * magnitude
    active_matrix = matrix[:, active]
    result = {
        "samples": int(matrix.shape[0]),
        "features": int(matrix.shape[1]),
        "active_features": int(active.sum()),
        "constant_features": int((~active).sum()),
    }
    if active_matrix.shape[1] == 0:
        result.update({
            "effective_rank": 0,
            "full_column_rank": False,
            "condition_number": None,
            "nonzero_condition_number": None,
            "stable_rank": 0.0,
            "smallest_nonzero_singular_value": None,
            "singular_values": [],
        })
        return result

    centered = (
        active_matrix - active_matrix.mean(axis=0, keepdims=True)
        if center else active_matrix
    )
    if standardize:
        rms = np.sqrt(np.mean(np.square(centered), axis=0))
        centered = centered / rms[None, :]
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    leading = float(singular[0]) if singular.size else 0.0
    tolerance = (
        max(centered.shape) * np.finfo(np.float64).eps * leading
    )
    rank = int(np.count_nonzero(singular > tolerance))
    full_column_rank = rank == active_matrix.shape[1]
    smallest = float(singular[rank - 1]) if rank else None
    nonzero_condition = (
        leading / smallest if smallest is not None and smallest > 0.0
        else None
    )
    condition = nonzero_condition if full_column_rank else None
    stable_rank = (
        float(np.square(singular).sum() / (leading * leading))
        if leading > 0.0 else 0.0
    )
    result.update({
        "effective_rank": rank,
        "full_column_rank": bool(full_column_rank),
        "condition_number": condition,
        "nonzero_condition_number": nonzero_condition,
        "stable_rank": stable_rank,
        "smallest_nonzero_singular_value": smallest,
        "singular_values": [float(value) for value in singular],
    })
    if labels is not None:
        active_labels = np.asarray(labels, dtype=object)[active]
        weak = []
        for index in range(max(0, singular.size - 3), singular.size):
            vector = right[index]
            strongest = np.argsort(np.abs(vector))[::-1][:8]
            weak.append({
                "singular_value": float(singular[index]),
                "loadings": [
                    {
                        "feature": str(active_labels[column]),
                        "coefficient": float(vector[column]),
                    }
                    for column in strongest
                ],
            })
        result["weakest_directions"] = weak
    return result


def relative_vector_error(prediction: np.ndarray, reference: np.ndarray) -> Dict:
    """Return scale-aware RMS and maximum absolute errors."""
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference shapes must match")
    residual = prediction - reference
    reference_rms = float(np.sqrt(np.mean(np.square(reference))))
    residual_rms = float(np.sqrt(np.mean(np.square(residual))))
    scale = max(reference_rms, 64.0 * np.finfo(np.float64).eps)
    return {
        "reference_rms": reference_rms,
        "residual_rms": residual_rms,
        "relative_rms": residual_rms / scale,
        "max_abs": float(np.max(np.abs(residual))),
        "finite": bool(
            np.isfinite(prediction).all() and np.isfinite(reference).all()
        ),
    }


def generator_reweighting_health(
    spin_tangent: np.ndarray,
    target_mforces: np.ndarray,
    struct_idx: np.ndarray,
    group_ids: Sequence,
    *,
    mforce_weight: float,
    generator_weight: float,
    ordinary_mforce_components: Optional[int] = None,
    mean_weight: float = 0.25,
    scale_floor: float = 1.0e-6,
) -> Dict:
    """Describe the local quadratic metric added by generator supervision.

    A rotation generator is a linear projection of already supervised magnetic
    forces.  With complete magnetic-force labels it therefore cannot add a new
    observation row space.  It can only reweight the response-aligned subspace.
    This certificate makes that distinction explicit by constructing the exact
    mean-plus-centered group operator used by the quadratic branch of the Huber
    objective.

    The reported curvature gain is local to zero residual, where the Huber loss
    is quadratic.  It uses the full-dataset MSE reference
    ``2 * mforce_weight / ordinary_mforce_components``; individual minibatches
    can have a different instantaneous normalization.
    """
    tangent = np.asarray(spin_tangent, dtype=np.float64)
    target = np.asarray(target_mforces, dtype=np.float64)
    atom_to_structure = np.asarray(struct_idx, dtype=np.int64).reshape(-1)
    groups = np.asarray(group_ids)
    if tangent.ndim != 2 or tangent.shape[1] != 3:
        raise ValueError("spin_tangent must have shape (num_atoms, 3)")
    if target.shape != tangent.shape:
        raise ValueError("target_mforces must match spin_tangent")
    if atom_to_structure.shape[0] != tangent.shape[0]:
        raise ValueError("struct_idx must contain one entry per atom")
    if not np.isfinite(tangent).all() or not np.isfinite(target).all():
        raise ValueError("generator metric inputs contain NaN or Inf")
    if mforce_weight < 0.0 or generator_weight < 0.0:
        raise ValueError("loss weights must be nonnegative")
    if mean_weight < 0.0:
        raise ValueError("mean_weight must be nonnegative")
    if scale_floor <= 0.0:
        raise ValueError("scale_floor must be positive")
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("group_ids must be a nonempty one-dimensional array")
    num_structures = int(groups.size)
    if atom_to_structure.size == 0:
        raise ValueError("generator metric requires at least one atom")
    if atom_to_structure.min() < 0 or atom_to_structure.max() >= num_structures:
        raise ValueError("struct_idx is outside the group_ids range")
    present = np.unique(atom_to_structure)
    if not np.array_equal(present, np.arange(num_structures)):
        raise ValueError("every response structure must contain at least one atom")

    num_components = int(tangent.size)
    ordinary_components = (
        num_components if ordinary_mforce_components is None
        else int(ordinary_mforce_components)
    )
    if ordinary_components < num_components:
        raise ValueError(
            "ordinary_mforce_components cannot be smaller than the response "
            "component count"
        )
    frame_operator = np.zeros((num_structures, num_components), dtype=np.float64)
    flat_tangent = tangent.reshape(-1, 3)
    for atom, structure in enumerate(atom_to_structure):
        start = 3 * atom
        frame_operator[structure, start:start + 3] = flat_tangent[atom]

    target_generator = frame_operator @ target.reshape(-1)
    unique_groups = list(dict.fromkeys(groups.tolist()))
    group_powers = []
    for group in unique_groups:
        mask = groups == group
        if not np.any(mask):
            continue
        group_powers.append(float(np.mean(target_generator[mask] ** 2)))
    normalization_scale = max(
        float(np.sqrt(np.mean(group_powers))), float(scale_floor)
    )

    rows = []
    num_groups = len(unique_groups)
    for group in unique_groups:
        members = np.flatnonzero(groups == group)
        count = int(members.size)
        if count < 1:
            continue
        local_scale = max(float(np.sqrt(np.mean(target_generator[members]**2))), float(scale_floor))
        local = frame_operator[members]
        centering = np.eye(count) - np.ones((count, count)) / count
        rows.append(
            (centering @ local)
            / (local_scale * np.sqrt(num_groups * count))
        )
        if mean_weight > 0.0:
            rows.append(
                local.mean(axis=0, keepdims=True)
                * np.sqrt(mean_weight / num_groups)
                / local_scale
            )
    operator = np.concatenate(rows, axis=0)
    operator_health = svd_health(operator, center=False)
    singular = np.linalg.svd(operator, compute_uv=False)
    tolerance = (
        max(operator.shape) * np.finfo(np.float64).eps * singular[0]
        if singular.size else 0.0
    )
    nonzero = singular[singular > tolerance]

    # The ordinary MF loss observes every component, so the strict rank is
    # already the ambient dimension whenever its weight is positive.  The
    # generator rows are a linear map of those observations and cannot add
    # rank.  They selectively amplify curvature along their own row space.
    ordinary_curvature = (
        2.0 * float(mforce_weight) / ordinary_components
        if mforce_weight > 0.0 else 0.0
    )
    if ordinary_curvature > 0.0 and nonzero.size:
        gains = 1.0 + float(generator_weight) * nonzero ** 2 / ordinary_curvature
        gain_min = float(gains.min())
        gain_max = float(gains.max())
    elif generator_weight == 0.0:
        gain_min = gain_max = 1.0
    else:
        gain_min = gain_max = None

    complete_mforce_supervision = mforce_weight > 0.0
    strict_rank_gain = (
        0 if complete_mforce_supervision or generator_weight == 0.0
        else int(operator_health["effective_rank"])
    )
    return {
        "mechanism": (
            "targeted_reweighting_of_existing_mforce_rows"
            if complete_mforce_supervision
            else "generator_only_observation_rows"
        ),
        "local_quadratic_certificate": True,
        "ambient_mforce_components": num_components,
        "ordinary_mforce_components": ordinary_components,
        "response_structures": num_structures,
        "response_groups": num_groups,
        "normalization_scale": normalization_scale,
        "normalization_policy": "group_target_rms",
        "group_normalization_scales": [max(float(np.sqrt(p)), float(scale_floor)) for p in group_powers],
        "generator_operator": operator_health,
        "complete_mforce_supervision": complete_mforce_supervision,
        "strict_observation_rank_gain": strict_rank_gain,
        "ordinary_mforce_weight": float(mforce_weight),
        "generator_weight": float(generator_weight),
        "orthogonal_curvature_gain": 1.0,
        "aligned_curvature_gain_min": gain_min,
        "aligned_curvature_gain_max": gain_max,
    }
