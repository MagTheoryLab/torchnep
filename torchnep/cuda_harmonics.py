"""Fused c516 solid harmonics and their first/second CUDA pullbacks."""

import os
from math import comb
from pathlib import Path
import threading

import torch

from .spin_tables import HARMONICS
from .constants import Z_COEFFICIENT


_EXTENSION = None
_LOCK = threading.Lock()
_TABLES = {}


def _extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _LOCK:
        if _EXTENSION is None:
            from torch.utils.cpp_extension import load
            _EXTENSION = load(
                name="torchnep_harmonics_v1",
                sources=[str(Path(__file__).with_name("csrc") / name)
                         for name in ("harmonics.cpp", "harmonics.cu")],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "-lineinfo", "--fmad=false"],
                verbose=False,
            )
    return _EXTENSION


def _tables(xyz, lmax, structural):
    key = (xyz.device, xyz.dtype, lmax, structural)
    if key not in _TABLES:
        source = HARMONICS
        if structural:
            source, component = [], 0
            for order in range(1,lmax+1):
                for m in range(order+1):
                    for part in range(1 if m == 0 else 2):
                        for z in range((order+m)%2,order-m+1,2):
                            c = Z_COEFFICIENT[order][m][z]
                            for y in range(part,m+1,2):
                                value = c * comb(m,y) * (-1)**(y//2)
                                if value:
                                    source.append((component,m-y,y,z,value))
                        component += 1
        terms, offsets = [], [0]
        for component in range((lmax+1)**2 - int(structural)):
            terms.extend(t for t in source if t[0] == component)
            offsets.append(len(terms))
        _TABLES[key] = (
            xyz.new_tensor([t[4] for t in terms]),
            torch.tensor([t[1:4] for t in terms], dtype=torch.int64, device=xyz.device),
            torch.tensor(offsets, dtype=torch.int64, device=xyz.device),
        )
    return _TABLES[key]


class _HarmonicsVjp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pull, xyz, coefficients, powers, offsets):
        ctx.save_for_backward(pull, xyz, coefficients, powers, offsets)
        return _extension().backward(pull.contiguous(), xyz, coefficients, powers, offsets)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, tangent):
        pull, xyz, coefficients, powers, offsets = ctx.saved_tensors
        dpull, dxyz = _extension().double_backward(
            pull.contiguous(), xyz, coefficients, powers, offsets, tangent.contiguous())
        return dpull, dxyz, None, None, None


class _Harmonics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, coefficients, powers, offsets):
        ctx.save_for_backward(xyz, coefficients, powers, offsets)
        return _extension().forward(xyz, coefficients, powers, offsets)

    @staticmethod
    def backward(ctx, pull):
        return _HarmonicsVjp.apply(pull, *ctx.saved_tensors), None, None, None


def fused_harmonics(xyz, lmax, structural=False):
    """Return None for the independent tensor-only reference execution path."""
    if (not xyz.is_cuda or xyz.dtype not in (torch.float32, torch.float64)
            or not 0 <= lmax <= 4 or (structural and lmax == 0)
            or os.environ.get("TORCHNEP_CUDA_HARMONICS", "1") == "0"
            or torch._dynamo.is_compiling()):
        return None
    flat = xyz.reshape(-1, 3).contiguous()
    return _Harmonics.apply(flat, *_tables(flat, lmax, structural)).reshape(
        *xyz.shape[:-1], (lmax+1)**2 - int(structural))


@torch.no_grad()
def harmonic_basis_and_derivatives(xyz, lmax, structural=False):
    """Fixed-input values and Cartesian Jacobian for the training batch cache."""
    flat = xyz.reshape(-1,3).contiguous()
    c, powers, offsets = _tables(flat,lmax,structural)
    if xyz.is_cuda and lmax <= 4 and os.environ.get('TORCHNEP_CUDA_HARMONICS','1') != '0':
        return _extension().jacobian(flat,c,powers,offsets)
    columns = torch.repeat_interleave(torch.arange(len(offsets)-1,device=xyz.device),
                                      offsets[1:]-offsets[:-1])
    monomials = flat[:,None,:].pow(powers[None,:,:]).prod(-1) * c
    value = flat.new_zeros(len(flat),len(offsets)-1).index_add(1,columns,monomials)
    derivatives = []
    for axis in range(3):
        exponents = powers.clone()
        exponents[:,axis] = (exponents[:,axis]-1).clamp_min(0)
        terms = flat[:,None,:].pow(exponents[None,:,:]).prod(-1) * (c*powers[:,axis])
        derivatives.append(torch.zeros_like(value).index_add(1,columns,terms))
    return value, torch.stack(derivatives,-1)
