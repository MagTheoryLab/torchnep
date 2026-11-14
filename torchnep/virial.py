# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.
# TorchNEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Virial projections shared by training and prediction boundaries."""

import numpy as np
import torch


# Row-major raw9 -> GPUMD six-vector order: xx, yy, zz, xy, yz, zx.
VIRIAL6_INDICES = (0, 4, 8, 1, 5, 6)


def virial6_for_labels(virial: torch.Tensor, spin_mode: int) -> torch.Tensor:
    """Project a raw9 virial onto the six components compared with labels.

    Spin models compare the symmetric strain derivative supplied by DFT
    labels while leaving the antisymmetric raw virial available internally.
    Ordinary models retain their established six-component projection.
    """
    if virial.shape[-1] != 9:
        raise ValueError("virial projection expects a row-major raw9 tensor")
    if spin_mode != 1:
        return virial[..., VIRIAL6_INDICES]
    half = virial.new_tensor(0.5)
    return torch.stack((
        virial[..., 0],
        virial[..., 4],
        virial[..., 8],
        half * (virial[..., 1] + virial[..., 3]),
        half * (virial[..., 5] + virial[..., 7]),
        half * (virial[..., 6] + virial[..., 2]),
    ), dim=-1)


def virial9_to_6_numpy(v9: np.ndarray, spin_mode: int) -> np.ndarray:
    """NumPy equivalent of :func:`virial6_for_labels` for output paths."""
    v9 = np.asarray(v9)
    if v9.shape[-1] != 9:
        raise ValueError("virial projection expects a row-major raw9 array")
    if spin_mode != 1:
        return v9[..., VIRIAL6_INDICES]
    out = np.empty(v9.shape[:-1] + (6,), dtype=v9.dtype)
    out[..., 0] = v9[..., 0]
    out[..., 1] = v9[..., 4]
    out[..., 2] = v9[..., 8]
    out[..., 3] = 0.5 * (v9[..., 1] + v9[..., 3])
    out[..., 4] = 0.5 * (v9[..., 5] + v9[..., 7])
    out[..., 5] = 0.5 * (v9[..., 6] + v9[..., 2])
    return out
