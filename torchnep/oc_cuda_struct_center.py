"""Fused structural angular center contractions for the O3-C2 trainer."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import threading

import torch


_EXTENSION = None
_EXTENSION_LOCK = threading.Lock()
STRUCT_CENTER_WIDTH = 25


def _enabled() -> bool:
    return os.environ.get("TORCHNEP_OC_CUDA_STRUCT_CENTER", "0") == "1"


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is None:
            if os.environ.get("TORCHNEP_OC_CUDA_PREBUILT", "0") == "1":
                _EXTENSION = importlib.import_module(
                    "torchnep_oc_struct_center_v1"
                )
                return _EXTENSION
            from torch.utils.cpp_extension import load

            source = Path(__file__).with_name("csrc") / "oc_struct_center.cu"
            _EXTENSION = load(
                name="torchnep_oc_struct_center_v1",
                sources=[str(source)],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "-lineinfo"],
                verbose=False,
            )
    return _EXTENSION


class _StructCenterVjp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, moments, c3b, c4b):
        ctx.save_for_backward(grad_output, moments, c3b, c4b)
        return _load_extension().backward(
            grad_output.contiguous(), moments.contiguous(),
            c3b.contiguous(), c4b.contiguous(),
        )

    @staticmethod
    def backward(ctx, grad_grad_moments):
        grad_output, moments, c3b, c4b = ctx.saved_tensors
        if grad_grad_moments is None:
            grad_grad_moments = torch.zeros_like(moments)
        grad2_output, grad2_moments = _load_extension().double_backward(
            grad_output.contiguous(), moments.contiguous(),
            c3b.contiguous(), c4b.contiguous(),
            grad_grad_moments.contiguous(),
        )
        need = ctx.needs_input_grad
        return (
            grad2_output if need[0] else None,
            grad2_moments if need[1] else None,
            None,
            None,
        )


class _StructCenter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, moments, c3b, c4b):
        ctx.save_for_backward(moments, c3b, c4b)
        return _load_extension().forward(
            moments.contiguous(), c3b.contiguous(), c4b.contiguous()
        )

    @staticmethod
    def backward(ctx, grad_output):
        moments, c3b, c4b = ctx.saved_tensors
        if grad_output is None:
            grad_output = moments.new_zeros(
                (moments.shape[0], STRUCT_CENTER_WIDTH)
            )
        grad_moments = _StructCenterVjp.apply(
            grad_output, moments, c3b, c4b
        )
        return grad_moments if ctx.needs_input_grad[0] else None, None, None


def structural_center_sector(
    moments,
    c3b,
    c4b,
    *,
    n_max_angular,
    l_max_3b,
    has_q_222,
    has_q_1111,
    has_q_112,
    has_q_123,
    has_q_233,
    has_q_134,
):
    """Return ``None`` outside the fixed 25D structural angular surface."""
    if not _enabled() or not moments.is_cuda:
        return None
    if not (
        moments.ndim == 3
        and moments.shape[1] == 5
        and moments.shape[2] == 24
        and int(n_max_angular) == 4
        and int(l_max_3b) == 4
        and int(has_q_222) == 1
        and not any((has_q_1111, has_q_112, has_q_123,
                     has_q_233, has_q_134))
        and c3b.numel() == 24
        and c4b.numel() >= 5
    ):
        return None
    return _StructCenter.apply(moments, c3b, c4b)
