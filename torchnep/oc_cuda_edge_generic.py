"""Runtime-C analytic edge-anchored sector family and mixed pullback."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import threading

import torch

_EXTENSION = None
_LOCK = threading.Lock()


def _extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _LOCK:
        if _EXTENSION is None:
            if os.environ.get("TORCHNEP_OC_CUDA_PREBUILT", "0") == "1":
                _EXTENSION = importlib.import_module(
                    "torchnep_oc_edge_sector_generic_v1"
                )
            else:
                from torch.utils.cpp_extension import load
                source = Path(__file__).with_name("csrc") / "oc_edge_sector_generic.cu"
                _EXTENSION = load(
                    name="torchnep_oc_edge_sector_generic_v1",
                    sources=[str(source)], extra_cflags=["-O3"],
                    extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False,
                )
    return _EXTENSION


class _EdgeVjp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, bank, auxiliary, pair_i, mix):
        ctx.save_for_backward(grad_output, bank, auxiliary, pair_i, mix)
        return tuple(_extension().backward(
            grad_output.contiguous(), bank.contiguous(),
            auxiliary.contiguous(), pair_i.contiguous(), mix.contiguous(),
        ))

    @staticmethod
    def backward(ctx, h_bank, h_auxiliary, h_mix):
        grad_output, bank, auxiliary, pair_i, mix = ctx.saved_tensors
        if h_bank is None:
            h_bank = torch.zeros_like(bank)
        if h_auxiliary is None:
            h_auxiliary = torch.zeros_like(auxiliary)
        if h_mix is None:
            h_mix = torch.zeros_like(mix)
        pulled = _extension().double_backward(
            grad_output.contiguous(), bank.contiguous(),
            auxiliary.contiguous(), pair_i.contiguous(), mix.contiguous(),
            h_bank.contiguous(), h_auxiliary.contiguous(), h_mix.contiguous(),
        )
        need = ctx.needs_input_grad
        return (
            pulled[0] if need[0] else None,
            pulled[1] if need[1] else None,
            pulled[2] if need[2] else None,
            None,
            pulled[3] if need[4] else None,
        )


class _Edge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bank, auxiliary, pair_i, mix):
        ctx.save_for_backward(bank, auxiliary, pair_i, mix)
        return _extension().forward(
            bank.contiguous(), auxiliary.contiguous(),
            pair_i.contiguous(), mix.contiguous(),
        )

    @staticmethod
    def backward(ctx, grad_output):
        bank, auxiliary, pair_i, mix = ctx.saved_tensors
        if grad_output is None:
            grad_output = bank.new_zeros((bank.shape[0], 3 * mix.shape[1]))
        pulled = _EdgeVjp.apply(grad_output, bank, auxiliary, pair_i, mix)
        need = ctx.needs_input_grad
        return (
            pulled[0] if need[0] else None,
            pulled[1] if need[1] else None,
            None,
            pulled[2] if need[3] else None,
        )


def edge_sector(bank, auxiliary, pair_i, radial_leg_mix):
    if not bank.is_cuda or bank.shape[1] == 85:
        return None
    return _Edge.apply(bank, auxiliary, pair_i, radial_leg_mix)
