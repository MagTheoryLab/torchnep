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

from .oc_cuda_bank_generic import generic_density_bank
from .oc_cuda_edge_generic import edge_sector as generic_edge_sector


_EXTENSION = None
_EXTENSION_LOCK = threading.Lock()
BANK_WIDTH = 85
AUX_WIDTH = 8
EDGE_ANCHOR_WIDTH = 6


def _enabled() -> bool:
    if os.environ.get("TORCHNEP_OC_CUDA_FULL", "0") == "1":
        return False
    return os.environ.get("TORCHNEP_OC_CUDA_BANK", "1") != "0"


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


class _EdgeAnchoredVjp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, bank, auxiliary, pair_i, radial_leg_mix):
        ctx.save_for_backward(
            grad_output, bank, auxiliary, pair_i, radial_leg_mix
        )
        return tuple(_load_extension().edge_anchor_backward(
            grad_output.contiguous(), bank.contiguous(),
            auxiliary.contiguous(), pair_i.contiguous(),
            radial_leg_mix.contiguous(),
        ))

    @staticmethod
    def backward(ctx, grad_grad_bank, grad_grad_auxiliary, grad_grad_mix):
        grad_output, bank, auxiliary, pair_i, radial_leg_mix = ctx.saved_tensors
        if grad_grad_bank is None:
            grad_grad_bank = torch.zeros_like(bank)
        if grad_grad_auxiliary is None:
            grad_grad_auxiliary = torch.zeros_like(auxiliary)
        if grad_grad_mix is None:
            grad_grad_mix = torch.zeros_like(radial_leg_mix)
        pulled = _load_extension().edge_anchor_double_backward(
            grad_output.contiguous(), bank.contiguous(),
            auxiliary.contiguous(), pair_i.contiguous(),
            radial_leg_mix.contiguous(), grad_grad_bank.contiguous(),
            grad_grad_auxiliary.contiguous(), grad_grad_mix.contiguous(),
        )
        need = ctx.needs_input_grad
        return (
            pulled[0] if need[0] else None,
            pulled[1] if need[1] else None,
            pulled[2] if need[2] else None,
            None,
            pulled[3] if need[4] else None,
        )


class _EdgeAnchoredSector(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bank, auxiliary, pair_i, radial_leg_mix):
        ctx.save_for_backward(bank, auxiliary, pair_i, radial_leg_mix)
        return _load_extension().edge_anchor_forward(
            bank.contiguous(), auxiliary.contiguous(), pair_i.contiguous(),
            radial_leg_mix.contiguous(),
        )

    @staticmethod
    def backward(ctx, grad_output):
        bank, auxiliary, pair_i, radial_leg_mix = ctx.saved_tensors
        if grad_output is None:
            grad_output = bank.new_zeros((bank.shape[0], EDGE_ANCHOR_WIDTH))
        pulled = _EdgeAnchoredVjp.apply(
            grad_output, bank, auxiliary, pair_i, radial_leg_mix
        )
        need = ctx.needs_input_grad
        return (
            pulled[0] if need[0] else None,
            pulled[1] if need[1] else None,
            None,
            pulled[2] if need[3] else None,
        )


def edge_anchored_sector(descriptor, bank, auxiliary, pair_i):
    """Fuse the remaining l11/l22/l112 edge-anchored O3-C2 families."""
    if not bank.is_cuda:
        return None
    generic = generic_edge_sector(
        bank, auxiliary, pair_i, descriptor.radial_leg_mix
    )
    if generic is not None:
        return generic
    if os.environ.get("TORCHNEP_OC_CUDA_CENTER", "1") == "0":
        return None
    return _EdgeAnchoredSector.apply(
        bank, auxiliary, pair_i, descriptor.radial_leg_mix
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
    """Use the specialized O3-C2 bank, otherwise the runtime-C/B bank."""
    if not _enabled() or not rij.is_cuda:
        return None
    exact_o3c2 = (
        descriptor.spin_compress == 2
        and descriptor.spin_l_max == 2
        and descriptor.spin_order == 3
        and descriptor.spin_soc
        and descriptor.basis_size_radial == 8
        and atom_types is not None
        and coefficient is not None
        and dof_active is not None
        and env_active is not None
    )
    if not exact_o3c2:
        generic = generic_density_bank(
            descriptor, rij, pair_i, pair_j, spins, atom_types, coefficient,
            dof_active, env_active,
        )
        if generic is None:
            return None
        bank, auxiliary = generic
        edge_anchor = edge_anchored_sector(
            descriptor, bank, auxiliary, pair_i
        )
        return (bank, auxiliary) if edge_anchor is None else (
            bank, auxiliary, edge_anchor
        )
    bank, auxiliary = _DensityBank.apply(
        rij, pair_i, pair_j, spins, atom_types, coefficient,
        dof_active, env_active, descriptor.cutoff_radial, descriptor.eps,
    )
    edge_anchor = edge_anchored_sector(
        descriptor, bank, auxiliary, pair_i
    )
    if edge_anchor is None:
        return bank, auxiliary
    return bank, auxiliary, edge_anchor
