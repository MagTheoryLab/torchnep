"""Runtime-C center-owned analytic sector family and mixed pullback."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import threading

import torch


_EXTENSION = None
_EXTENSION_LOCK = threading.Lock()


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is None:
            if os.environ.get("TORCHNEP_OC_CUDA_PREBUILT", "0") == "1":
                _EXTENSION = importlib.import_module(
                    "torchnep_oc_center_sector_generic_v1"
                )
            else:
                from torch.utils.cpp_extension import load

                source = (
                    Path(__file__).with_name("csrc")
                    / "oc_center_sector_generic.cu"
                )
                _EXTENSION = load(
                    name="torchnep_oc_center_sector_generic_v1",
                    sources=[str(source)],
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=["-O3", "-lineinfo"],
                    verbose=False,
                )
    return _EXTENSION


class _CenterVjp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, bank, spins, radial_leg_mix):
        ctx.save_for_backward(grad_output, bank, spins, radial_leg_mix)
        return tuple(
            _load_extension().backward(
                grad_output.contiguous(),
                bank.contiguous(),
                spins.contiguous(),
                radial_leg_mix.contiguous(),
            )
        )

    @staticmethod
    def backward(ctx, h_bank, h_spins, h_radial_leg_mix):
        grad_output, bank, spins, radial_leg_mix = ctx.saved_tensors
        if h_bank is None:
            h_bank = torch.zeros_like(bank)
        if h_spins is None:
            h_spins = torch.zeros_like(spins)
        if h_radial_leg_mix is None:
            h_radial_leg_mix = torch.zeros_like(radial_leg_mix)
        pulled = _load_extension().double_backward(
            grad_output.contiguous(),
            bank.contiguous(),
            spins.contiguous(),
            radial_leg_mix.contiguous(),
            h_bank.contiguous(),
            h_spins.contiguous(),
            h_radial_leg_mix.contiguous(),
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
    def forward(ctx, bank, spins, radial_leg_mix):
        ctx.save_for_backward(bank, spins, radial_leg_mix)
        return _load_extension().forward(
            bank.contiguous(),
            spins.contiguous(),
            radial_leg_mix.contiguous(),
        )

    @staticmethod
    def backward(ctx, grad_output):
        bank, spins, radial_leg_mix = ctx.saved_tensors
        channels = radial_leg_mix.shape[1]
        if grad_output is None:
            width = 13 * channels + channels * (channels + 1) // 2
            grad_output = bank.new_zeros((bank.shape[0], width))
        pulled = _CenterVjp.apply(
            grad_output, bank, spins, radial_leg_mix
        )
        need = ctx.needs_input_grad
        return (
            pulled[0] if need[0] else None,
            pulled[1] if need[1] else None,
            pulled[2] if need[2] else None,
        )


def center_sector(bank, spins, radial_leg_mix):
    """Return runtime-C center contractions, excluding fixed O3-C2."""
    if not bank.is_cuda or bank.shape[1] == 85:
        return None
    return _CenterSector.apply(bank, spins, radial_leg_mix)
