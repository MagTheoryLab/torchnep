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
