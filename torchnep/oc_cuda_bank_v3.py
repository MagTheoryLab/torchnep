"""Shared spin2/spin3 analytic CUDA primitive bank for O3-C2.

The extension entry points added for spin3 traverse every directed edge once
and emit both the frozen 85-value spin2 bank and the 16 dot-weighted Cartesian
moments needed by the new angular exchange.  Existing spin2 entry points and
tensor layouts are unchanged.
"""

from __future__ import annotations

import os

import torch

from .oc_cuda_bank import _load_extension


BANK_WIDTH = 85
EXTRA_WIDTH = 16
AUX_WIDTH = 8
ANGULAR_WIDTH = 6


class _AngularCenterV3Vjp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, moment, same):
        ctx.save_for_backward(grad_output, moment, same)
        return tuple(_load_extension().angular_center_v3_backward(
            grad_output.contiguous(), moment.contiguous(), same.contiguous()
        ))

    @staticmethod
    def backward(ctx, h_moment, h_same):
        grad_output, moment, same = ctx.saved_tensors
        if h_moment is None:
            h_moment = torch.zeros_like(moment)
        if h_same is None:
            h_same = torch.zeros_like(same)
        pulled = _load_extension().angular_center_v3_double_backward(
            grad_output.contiguous(),
            moment.contiguous(),
            same.contiguous(),
            h_moment.contiguous(),
            h_same.contiguous(),
        )
        need = ctx.needs_input_grad
        return (
            pulled[0] if need[0] else None,
            pulled[1] if need[1] else None,
            None,
        )


class _AngularCenterV3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, moment, same):
        ctx.save_for_backward(moment, same)
        return _load_extension().angular_center_v3_forward(
            moment.contiguous(), same.contiguous()
        )

    @staticmethod
    def backward(ctx, grad_output):
        moment, same = ctx.saved_tensors
        if grad_output is None:
            grad_output = moment.new_zeros((moment.shape[0], ANGULAR_WIDTH))
        pulled = _AngularCenterV3Vjp.apply(grad_output, moment, same)
        need = ctx.needs_input_grad
        return (
            pulled[0] if need[0] else None,
            pulled[1] if need[1] else None,
        )


class _DensityBankV3Vjp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grad_bank,
        grad_extra,
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
            grad_bank,
            grad_extra,
            grad_aux,
            rij,
            pair_i,
            pair_j,
            spins,
            atom_types,
            coefficient,
            dof_active,
            env_active,
        )
        return tuple(_load_extension().backward_v3(
            grad_bank.contiguous(),
            grad_extra.contiguous(),
            grad_aux.contiguous(),
            rij.contiguous(),
            pair_i.contiguous(),
            pair_j.contiguous(),
            spins.contiguous(),
            atom_types.contiguous(),
            coefficient.contiguous(),
            dof_active.contiguous(),
            env_active.contiguous(),
            ctx.cutoff,
            ctx.eps,
        ))

    @staticmethod
    def backward(ctx, h_rij, h_spins, h_coefficient):
        (
            grad_bank,
            grad_extra,
            grad_aux,
            rij,
            pair_i,
            pair_j,
            spins,
            atom_types,
            coefficient,
            dof_active,
            env_active,
        ) = ctx.saved_tensors
        if h_rij is None:
            h_rij = torch.zeros_like(rij)
        if h_spins is None:
            h_spins = torch.zeros_like(spins)
        if h_coefficient is None:
            h_coefficient = torch.zeros_like(coefficient)
        pulled = _load_extension().double_backward_v3(
            grad_bank.contiguous(),
            grad_extra.contiguous(),
            grad_aux.contiguous(),
            rij.contiguous(),
            pair_i.contiguous(),
            pair_j.contiguous(),
            spins.contiguous(),
            atom_types.contiguous(),
            coefficient.contiguous(),
            dof_active.contiguous(),
            env_active.contiguous(),
            h_rij.contiguous(),
            h_spins.contiguous(),
            h_coefficient.contiguous(),
            ctx.cutoff,
            ctx.eps,
        )
        need = ctx.needs_input_grad
        return (
            pulled[0] if need[0] else None,
            pulled[1] if need[1] else None,
            pulled[2] if need[2] else None,
            pulled[3] if need[3] else None,
            None,
            None,
            pulled[4] if need[6] else None,
            None,
            pulled[5] if need[8] else None,
            None,
            None,
            None,
            None,
        )


class _DensityBankV3(torch.autograd.Function):
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
            rij,
            pair_i,
            pair_j,
            spins,
            atom_types,
            coefficient,
            dof_active,
            env_active,
        )
        return tuple(_load_extension().forward_v3(
            rij.contiguous(),
            pair_i.contiguous(),
            pair_j.contiguous(),
            spins.contiguous(),
            atom_types.contiguous(),
            coefficient.contiguous(),
            dof_active.contiguous(),
            env_active.contiguous(),
            ctx.cutoff,
            ctx.eps,
        ))

    @staticmethod
    def backward(ctx, grad_bank, grad_extra, grad_aux):
        (
            rij,
            pair_i,
            pair_j,
            spins,
            atom_types,
            coefficient,
            dof_active,
            env_active,
        ) = ctx.saved_tensors
        if grad_bank is None:
            grad_bank = rij.new_zeros((spins.shape[0], BANK_WIDTH))
        if grad_extra is None:
            grad_extra = rij.new_zeros((spins.shape[0], EXTRA_WIDTH))
        if grad_aux is None:
            grad_aux = rij.new_zeros((rij.shape[0], AUX_WIDTH))
        pulled = _DensityBankV3Vjp.apply(
            grad_bank,
            grad_extra,
            grad_aux,
            rij,
            pair_i,
            pair_j,
            spins,
            atom_types,
            coefficient,
            dof_active,
            env_active,
            ctx.cutoff,
            ctx.eps,
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


def density_bank_v3(
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
    """Return the one-pass O3-C2 bank, or ``None`` off its exact seam."""
    if (
        os.environ.get("TORCHNEP_SPIN3_SHARED_BANK", "1") == "0"
        or os.environ.get("TORCHNEP_OC_CUDA_BANK", "1") == "0"
        or os.environ.get("TORCHNEP_OC_CUDA_FULL", "0") == "1"
        or not rij.is_cuda
    ):
        return None
    exact = (
        descriptor.spin_compress == 2
        and descriptor.spin_l_max == 2
        and descriptor.spin_order == 3
        and descriptor.spin_soc
        and descriptor.basis_size_radial == 8
        and atom_types is not None
        and coefficient is not None
        and coefficient.ndim == 5
        and coefficient.shape[-2:] == (2, 9)
        and dof_active is not None
        and env_active is not None
    )
    if not exact:
        return None
    extension = _load_extension()
    if not all(hasattr(extension, name) for name in (
        "forward_v3", "backward_v3", "double_backward_v3"
    )):
        return None
    return _DensityBankV3.apply(
        rij,
        pair_i,
        pair_j,
        spins,
        atom_types,
        coefficient,
        dof_active,
        env_active,
        descriptor.cutoff_radial,
        descriptor.eps,
    )


def angular_center_v3(moment, same):
    """Contract the l=1/l=2 Cartesian moments in one analytic CUDA op."""
    if not moment.is_cuda:
        return None
    extension = _load_extension()
    if not all(hasattr(extension, name) for name in (
        "angular_center_v3_forward",
        "angular_center_v3_backward",
        "angular_center_v3_double_backward",
    )):
        return None
    return _AngularCenterV3.apply(moment, same)
