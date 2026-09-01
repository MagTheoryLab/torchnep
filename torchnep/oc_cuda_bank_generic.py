"""Runtime-C/B analytic CUDA density bank with an exact mixed pullback."""

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
                    "torchnep_oc_density_bank_generic_v1"
                )
            else:
                from torch.utils.cpp_extension import load

                source = (
                    Path(__file__).with_name("csrc")
                    / "oc_density_bank_generic.cu"
                )
                _EXTENSION = load(
                    name="torchnep_oc_density_bank_generic_v1",
                    sources=[str(source)],
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=["-O3", "-lineinfo"],
                    verbose=False,
                )
    return _EXTENSION


def _widths(coefficient):
    channels = coefficient.shape[3]
    return 47 * channels + channels * (channels + 1) // 2 + 3 * channels, channels + 6


class _GenericDensityBankVjp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, grad_bank, grad_auxiliary, rij, pair_i, pair_j, spins,
        atom_types, coefficient, dof_active, env_active, cutoff, eps,
    ):
        ctx.cutoff = float(cutoff)
        ctx.eps = float(eps)
        ctx.save_for_backward(
            grad_bank, grad_auxiliary, rij, pair_i, pair_j, spins,
            atom_types, coefficient, dof_active, env_active,
        )
        return tuple(_load_extension().backward(
            grad_bank.contiguous(), grad_auxiliary.contiguous(),
            rij.contiguous(), pair_i.contiguous(), pair_j.contiguous(),
            spins.contiguous(), atom_types.contiguous(),
            coefficient.contiguous(), dof_active.contiguous(),
            env_active.contiguous(), ctx.cutoff, ctx.eps,
        ))

    @staticmethod
    def backward(ctx, h_rij, h_spins, h_coefficient):
        (grad_bank, grad_auxiliary, rij, pair_i, pair_j, spins,
         atom_types, coefficient, dof_active, env_active) = ctx.saved_tensors
        if h_rij is None:
            h_rij = torch.zeros_like(rij)
        if h_spins is None:
            h_spins = torch.zeros_like(spins)
        if h_coefficient is None:
            h_coefficient = torch.zeros_like(coefficient)
        pulled = _load_extension().double_backward(
            grad_bank.contiguous(), grad_auxiliary.contiguous(),
            rij.contiguous(), pair_i.contiguous(), pair_j.contiguous(),
            spins.contiguous(), atom_types.contiguous(),
            coefficient.contiguous(), dof_active.contiguous(),
            env_active.contiguous(), h_rij.contiguous(), h_spins.contiguous(),
            h_coefficient.contiguous(), ctx.cutoff, ctx.eps,
        )
        need = ctx.needs_input_grad
        return (
            pulled[0] if need[0] else None,
            pulled[1] if need[1] else None,
            pulled[2] if need[2] else None,
            None,
            None,
            pulled[3] if need[5] else None,
            None,
            pulled[4] if need[7] else None,
            None,
            None,
            None,
            None,
        )


class _GenericDensityBank(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, rij, pair_i, pair_j, spins, atom_types, coefficient,
        dof_active, env_active, cutoff, eps,
    ):
        ctx.cutoff = float(cutoff)
        ctx.eps = float(eps)
        ctx.save_for_backward(
            rij, pair_i, pair_j, spins, atom_types, coefficient,
            dof_active, env_active,
        )
        return tuple(_load_extension().forward(
            rij.contiguous(), pair_i.contiguous(), pair_j.contiguous(),
            spins.contiguous(), atom_types.contiguous(),
            coefficient.contiguous(), dof_active.contiguous(),
            env_active.contiguous(), ctx.cutoff, ctx.eps,
        ))

    @staticmethod
    def backward(ctx, grad_bank, grad_auxiliary):
        (rij, pair_i, pair_j, spins, atom_types, coefficient,
         dof_active, env_active) = ctx.saved_tensors
        bank_width, aux_width = _widths(coefficient)
        if grad_bank is None:
            grad_bank = rij.new_zeros((spins.shape[0], bank_width))
        if grad_auxiliary is None:
            grad_auxiliary = rij.new_zeros((rij.shape[0], aux_width))
        pulled = _GenericDensityBankVjp.apply(
            grad_bank, grad_auxiliary, rij, pair_i, pair_j, spins,
            atom_types, coefficient, dof_active, env_active,
            ctx.cutoff, ctx.eps,
        )
        need = ctx.needs_input_grad
        return (
            pulled[0] if need[0] else None,
            None,
            None,
            pulled[1] if need[3] else None,
            None,
            pulled[2] if need[5] else None,
            None,
            None,
            None,
            None,
        )


def generic_density_bank(
    descriptor, rij, pair_i, pair_j, spins, atom_types, coefficient,
    dof_active, env_active,
):
    """Return the canonical analytic bank for any legal runtime C and B."""
    if not rij.is_cuda:
        return None
    required = (atom_types, coefficient, dof_active, env_active)
    if any(value is None for value in required):
        return None
    return _GenericDensityBank.apply(
        rij, pair_i, pair_j, spins, atom_types, coefficient,
        dof_active, env_active, descriptor.cutoff_radial, descriptor.eps,
    )
