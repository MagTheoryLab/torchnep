"""Analytic CUDA primitive-bank sector for the narrow O3-C2 trainer.

The custom op fuses the shared radial basis/projection, normalized geometry,
rank-2 STF harmonics and every edge-local O1/O2/O3 primitive into two outputs:
an atom-owned moment bank and a small edge auxiliary bank.  Its native VJP and
directional analytic HVP keep force/mforce/virial training differentiable
without materializing a Jacobian or specializing on a ragged shape.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import threading

import torch


_EXTENSION = None
_EXTENSION_LOCK = threading.Lock()
BANK_WIDTH = 85
AUX_WIDTH = 8


def _enabled() -> bool:
    return os.environ.get("TORCHNEP_OC_CUDA_BANK", "0") == "1"


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is None:
            if os.environ.get("TORCHNEP_OC_CUDA_PREBUILT", "0") == "1":
                _EXTENSION = importlib.import_module(
                    "torchnep_oc_density_bank_v1"
                )
                return _EXTENSION
            from torch.utils.cpp_extension import load

            source = Path(__file__).with_name("csrc") / "oc_density_bank.cu"
            _EXTENSION = load(
                name="torchnep_oc_density_bank_v1",
                sources=[str(source)],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "-lineinfo"],
                verbose=False,
            )
    return _EXTENSION


class _DensityBankVjp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grad_bank,
        grad_aux,
        rij,
        pair_i,
        pair_j,
        spins,
        atom_types,
        coefficient,
        dof_active,
        env_active,
        cutoff,
        eps,
    ):
        ctx.cutoff = float(cutoff)
        ctx.eps = float(eps)
        ctx.save_for_backward(
            grad_bank, grad_aux, rij, pair_i, pair_j, spins, atom_types,
            coefficient, dof_active, env_active,
        )
        return tuple(_load_extension().backward(
            grad_bank.contiguous(), grad_aux.contiguous(), rij.contiguous(),
            pair_i.contiguous(), pair_j.contiguous(), spins.contiguous(),
            atom_types.contiguous(), coefficient.contiguous(),
            dof_active.contiguous(), env_active.contiguous(),
            ctx.cutoff, ctx.eps,
        ))

    @staticmethod
    def backward(ctx, grad_grad_rij, grad_grad_spins, grad_grad_coefficient):
        (grad_bank, grad_aux, rij, pair_i, pair_j, spins, atom_types,
         coefficient, dof_active, env_active) = ctx.saved_tensors
        if grad_grad_rij is None:
            grad_grad_rij = torch.zeros_like(rij)
        if grad_grad_spins is None:
            grad_grad_spins = torch.zeros_like(spins)
        if grad_grad_coefficient is None:
            grad_grad_coefficient = torch.zeros_like(coefficient)
        pulled = _load_extension().double_backward(
            grad_bank.contiguous(), grad_aux.contiguous(), rij.contiguous(),
            pair_i.contiguous(), pair_j.contiguous(), spins.contiguous(),
            atom_types.contiguous(), coefficient.contiguous(),
            dof_active.contiguous(), env_active.contiguous(),
            grad_grad_rij.contiguous(), grad_grad_spins.contiguous(),
            grad_grad_coefficient.contiguous(), ctx.cutoff, ctx.eps,
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


class _DensityBank(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rij,
        pair_i,
        pair_j,
        spins,
        atom_types,
        coefficient,
        dof_active,
        env_active,
        cutoff,
        eps,
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
    def backward(ctx, grad_bank, grad_aux):
        (rij, pair_i, pair_j, spins, atom_types, coefficient,
         dof_active, env_active) = ctx.saved_tensors
        if grad_bank is None:
            grad_bank = rij.new_zeros((spins.shape[0], BANK_WIDTH))
        if grad_aux is None:
            grad_aux = rij.new_zeros((rij.shape[0], AUX_WIDTH))
        pulled = _DensityBankVjp.apply(
            grad_bank, grad_aux, rij, pair_i, pair_j, spins, atom_types,
            coefficient, dof_active, env_active, ctx.cutoff, ctx.eps,
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


def density_bank(
    descriptor,
    rij,
    pair_i,
    pair_j,
    spins,
    atom_types,
    coefficient,
    dof_active,
    env_active,
):
    """Return ``None`` outside the exact internal O3-C2 CUDA surface."""
    if not _enabled() or not rij.is_cuda:
        return None
    if not (
        descriptor.spin_compress == 2
        and descriptor.spin_l_max == 2
        and descriptor.spin_order == 3
        and descriptor.spin_soc
        and descriptor.basis_size_radial == 8
        and atom_types is not None
        and coefficient is not None
        and dof_active is not None
        and env_active is not None
    ):
        return None
    return _DensityBank.apply(
        rij, pair_i, pair_j, spins, atom_types, coefficient,
        dof_active, env_active, descriptor.cutoff_radial, descriptor.eps,
    )
