"""Generic analytic structural angular-center contractions on CUDA."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import threading

import torch

from .constants import Q123_TERMS, Q233_TERMS, Q134_TERMS


_EXTENSION = None
_EXTENSION_LOCK = threading.Lock()
_TERM_CACHE = {}


def _enabled() -> bool:
    return os.environ.get("TORCHNEP_OC_CUDA_STRUCT_CENTER", "1") != "0"


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


def _packed_terms(reference):
    key = (reference.device, reference.dtype)
    cached = _TERM_CACHE.get(key)
    if cached is not None:
        return cached
    packed = []
    for terms in (Q123_TERMS, Q233_TERMS, Q134_TERMS):
        packed.extend((
            torch.tensor(
                [coefficient for coefficient, _ in terms],
                dtype=reference.dtype, device=reference.device,
            ),
            torch.tensor(
                [indices for _, indices in terms],
                dtype=torch.long, device=reference.device,
            ),
        ))
    cached = tuple(packed)
    _TERM_CACHE[key] = cached
    return cached


class _StructCenterVjp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, moments, c3b, c4b, c5b, c4b2,
                q123c, q123i, q233c, q233i, q134c, q134i,
                l_max, flags):
        ctx.save_for_backward(
            grad_output, moments, c3b, c4b, c5b, c4b2,
            q123c, q123i, q233c, q233i, q134c, q134i,
        )
        ctx.l_max = l_max
        ctx.flags = flags
        return _load_extension().backward(
            grad_output.contiguous(), moments.contiguous(),
            c3b.contiguous(), c4b.contiguous(), c5b.contiguous(),
            c4b2.contiguous(), q123c, q123i, q233c, q233i, q134c, q134i,
            l_max, *flags,
        )

    @staticmethod
    def backward(ctx, grad_grad_moments):
        saved = ctx.saved_tensors
        grad_output, moments = saved[:2]
        if grad_grad_moments is None:
            grad_grad_moments = torch.zeros_like(moments)
        grad2_output, grad2_moments = _load_extension().double_backward(
            *(tensor.contiguous() for tensor in saved),
            grad_grad_moments.contiguous(), ctx.l_max, *ctx.flags,
        )
        result = [None] * 14
        result[0] = grad2_output if ctx.needs_input_grad[0] else None
        result[1] = grad2_moments if ctx.needs_input_grad[1] else None
        return tuple(result)


class _StructCenter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, moments, c3b, c4b, c5b, c4b2,
                q123c, q123i, q233c, q233i, q134c, q134i,
                l_max, flags):
        ctx.save_for_backward(
            moments, c3b, c4b, c5b, c4b2,
            q123c, q123i, q233c, q233i, q134c, q134i,
        )
        ctx.l_max = l_max
        ctx.flags = flags
        return _load_extension().forward(
            moments.contiguous(), c3b.contiguous(), c4b.contiguous(),
            c5b.contiguous(), c4b2.contiguous(),
            q123c.contiguous(), q123i.contiguous(),
            q233c.contiguous(), q233i.contiguous(),
            q134c.contiguous(), q134i.contiguous(),
            l_max, *flags,
        )

    @staticmethod
    def backward(ctx, grad_output):
        moments = ctx.saved_tensors[0]
        if grad_output is None:
            width = moments.shape[1] * (ctx.l_max + sum(ctx.flags))
            grad_output = moments.new_zeros((moments.shape[0], width))
        grad_moments = _StructCenterVjp.apply(
            grad_output, *ctx.saved_tensors, ctx.l_max, ctx.flags
        )
        result = [None] * 13
        result[0] = grad_moments if ctx.needs_input_grad[0] else None
        return tuple(result)


def structural_center_sector(
    moments, c3b, c4b, c5b, c4b2, *, n_max_angular, l_max_3b,
    has_q_222, has_q_1111, has_q_112, has_q_123, has_q_233, has_q_134,
):
    """Fuse every legal structural angular sector with runtime radial width."""
    if not _enabled() or not moments.is_cuda:
        return None
    radial_channels = int(n_max_angular) + 1
    num_lm = int(l_max_3b) * (int(l_max_3b) + 2)
    flags = tuple(int(bool(value)) for value in (
        has_q_222, has_q_1111, has_q_112,
        has_q_123, has_q_233, has_q_134,
    ))
    if not (
        moments.ndim == 3
        and moments.shape[1] == radial_channels
        and moments.shape[2] == num_lm
        and 1 <= int(l_max_3b) <= 4
        and (not flags[0] or int(l_max_3b) >= 2)
        and (not flags[1] or int(l_max_3b) >= 1)
        and (not flags[2] or int(l_max_3b) >= 2)
        and (not flags[3] or int(l_max_3b) >= 3)
        and (not flags[4] or int(l_max_3b) >= 3)
        and (not flags[5] or int(l_max_3b) >= 4)
        and c3b.numel() >= num_lm
        and c4b.numel() >= 5
        and c5b.numel() >= 3
        and c4b2.numel() >= 5
    ):
        return None
    return _StructCenter.apply(
        moments, c3b, c4b, c5b, c4b2, *_packed_terms(moments),
        int(l_max_3b), flags,
    )
