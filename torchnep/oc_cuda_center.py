"""Analytic center-owned O2/O3 contractions for the narrow O3-C2 trainer."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import threading

import torch


_EXTENSION = None
_EXTENSION_LOCK = threading.Lock()
CENTER_WIDTH = 29


def _enabled() -> bool:
    return os.environ.get("TORCHNEP_OC_CUDA_CENTER", "0") == "1"


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is None:
            if os.environ.get("TORCHNEP_OC_CUDA_PREBUILT", "0") == "1":
                _EXTENSION = importlib.import_module(
                    "torchnep_oc_center_sector_v1"
                )
                return _EXTENSION
            from torch.utils.cpp_extension import load

            source = Path(__file__).with_name("csrc") / "oc_center_sector.cu"
            _EXTENSION = load(
                name="torchnep_oc_center_sector_v1",
                sources=[str(source)],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "-lineinfo"],
                verbose=False,
            )
    return _EXTENSION


class _CenterSectorVjp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, primitive, spins, radial_leg_mix):
        ctx.save_for_backward(grad_output, primitive, spins, radial_leg_mix)
        return tuple(_load_extension().backward(
            grad_output.contiguous(), primitive.contiguous(),
            spins.contiguous(), radial_leg_mix.contiguous(),
        ))

    @staticmethod
    def backward(
        ctx, grad_grad_primitive, grad_grad_spins, grad_grad_radial_leg_mix,
    ):
        grad_output, primitive, spins, radial_leg_mix = ctx.saved_tensors
        if grad_grad_primitive is None:
            grad_grad_primitive = torch.zeros_like(primitive)
        if grad_grad_spins is None:
            grad_grad_spins = torch.zeros_like(spins)
        if grad_grad_radial_leg_mix is None:
            grad_grad_radial_leg_mix = torch.zeros_like(radial_leg_mix)
        pulled = _load_extension().double_backward(
            grad_output.contiguous(), primitive.contiguous(),
            spins.contiguous(), radial_leg_mix.contiguous(),
            grad_grad_primitive.contiguous(), grad_grad_spins.contiguous(),
            grad_grad_radial_leg_mix.contiguous(),
        )
        need = ctx.needs_input_grad
        return (
            pulled[0] if need[0] else None,
            pulled[1] if need[1] else None,
            pulled[2] if need[2] else None,
            pulled[3] if need[3] else None,
        )


class _CenterSector(torch.autograd.Function):
    @staticmethod
    def forward(ctx, primitive, spins, radial_leg_mix):
        ctx.save_for_backward(primitive, spins, radial_leg_mix)
        return _load_extension().forward(
            primitive.contiguous(), spins.contiguous(),
            radial_leg_mix.contiguous(),
        )

    @staticmethod
    def backward(ctx, grad_output):
        primitive, spins, radial_leg_mix = ctx.saved_tensors
        if grad_output is None:
            grad_output = primitive.new_zeros(
                (primitive.shape[0], CENTER_WIDTH)
            )
        pulled = _CenterSectorVjp.apply(
            grad_output, primitive, spins, radial_leg_mix
        )
        need = ctx.needs_input_grad
        return tuple(
            value if required else None
            for value, required in zip(pulled, need)
        )


def center_sector(descriptor, primitive, spins):
    """Return ``None`` outside the exact internal O3-C2 CUDA surface."""
    if not _enabled() or not primitive.is_cuda:
        return None
    if not (
        descriptor.spin_compress == 2
        and descriptor.spin_l_max == 2
        and descriptor.spin_order == 3
        and descriptor.spin_soc
        and primitive.ndim == 2
        and primitive.shape[1] == 85
        and spins.ndim == 2
        and spins.shape[1] == 3
    ):
        return None
    return _CenterSector.apply(
        primitive, spins, descriptor.radial_leg_mix
    )
