"""Internal full-49D O3-C2 generated CUDA pullback experiment.

The mathematical implementation remains
``MagneticPolynomialDescriptor._forward_reference``.  This module makes the
projection parameter explicit with ``torch.func.functional_call`` and wraps
the complete descriptor in a custom autograd boundary.  Forward, VJP and HVP
are then compiled as shape-specialized CUDA kernels without copying the
descriptor algebra into a second implementation.
"""

from __future__ import annotations

import os

import torch


_TEMPLATE = None
_TEMPLATE_SIGNATURE = None
_COMPILED_FORWARD = None
_COMPILED_VJP = None
_COMPILED_HVP = None


class _ReferenceModule(torch.nn.Module):
    def __init__(self, descriptor):
        super().__init__()
        self.descriptor = descriptor

    def forward(
        self,
        rij,
        pair_i,
        pair_j,
        spins,
        atom_types,
        coefficient,
        dof_active,
        env_active,
    ):
        return self.descriptor._forward_reference(
            rij,
            pair_i,
            pair_j,
            spins,
            spins.shape[0],
            atom_types,
            coefficient,
            None,
            dof_active,
            env_active,
        )


def _template(descriptor=None):
    global _TEMPLATE, _TEMPLATE_SIGNATURE
    if _TEMPLATE is None:
        if descriptor is None:
            raise RuntimeError("full O3-C2 template has not been initialized")
        _TEMPLATE = _ReferenceModule(descriptor)
        _TEMPLATE_SIGNATURE = (
            descriptor.spin_compress,
            descriptor.spin_l_max,
            descriptor.spin_order,
            descriptor.spin_soc,
            descriptor.basis_size_radial,
            descriptor.cutoff_radial,
            descriptor.eps,
            descriptor.dim,
        )
    elif descriptor is not None:
        signature = (
            descriptor.spin_compress,
            descriptor.spin_l_max,
            descriptor.spin_order,
            descriptor.spin_soc,
            descriptor.basis_size_radial,
            descriptor.cutoff_radial,
            descriptor.eps,
            descriptor.dim,
        )
        if signature != _TEMPLATE_SIGNATURE:
            raise RuntimeError(
                "full O3-C2 compiled template cannot mix descriptor configs "
                "in one process"
            )
    return _TEMPLATE


def _reference(
    rij,
    pair_i,
    pair_j,
    spins,
    atom_types,
    coefficient,
    dof_active,
    env_active,
    leg_mix,
):
    module = _template()
    return torch.func.functional_call(
        module,
        {"descriptor.radial_leg_mix": leg_mix},
        (
            rij,
            pair_i,
            pair_j,
            spins,
            atom_types,
            coefficient,
            dof_active,
            env_active,
        ),
        # Only radial_leg_mix is made explicit for differentiation. The pair
        # index tensors remain immutable descriptor buffers, so requiring a
        # complete parameter/buffer mapping would reject this partial override.
        strict=False,
    )


def _vjp_reference(
    grad_output,
    rij,
    pair_i,
    pair_j,
    spins,
    atom_types,
    coefficient,
    dof_active,
    env_active,
    leg_mix,
):
    def scalar(rr, ss, cc, ll):
        value = _reference(
            rr,
            pair_i,
            pair_j,
            ss,
            atom_types,
            cc,
            dof_active,
            env_active,
            ll,
        )
        return (value * grad_output).sum()

    return torch.func.grad(scalar, argnums=(0, 1, 2, 3))(
        rij, spins, coefficient, leg_mix
    )


def _hvp_reference(
    grad_output,
    rij,
    pair_i,
    pair_j,
    spins,
    atom_types,
    coefficient,
    dof_active,
    env_active,
    leg_mix,
    grad_grad_rij,
    grad_grad_spins,
    grad_grad_coefficient,
    grad_grad_leg_mix,
):
    def local(go, rr, ss, cc, ll):
        return _vjp_reference(
            go,
            rr,
            pair_i,
            pair_j,
            ss,
            atom_types,
            cc,
            dof_active,
            env_active,
            ll,
        )

    _, pullback = torch.func.vjp(
        local, grad_output, rij, spins, coefficient, leg_mix
    )
    return pullback(
        (
            grad_grad_rij,
            grad_grad_spins,
            grad_grad_coefficient,
            grad_grad_leg_mix,
        )
    )


def _compile(function):
    from torch._functorch import config as functorch_config

    functorch_config.donated_buffer = False
    # The validation matrix intentionally exercises several ragged edge counts,
    # dtypes and grad modes. Keep their specializations cached instead of
    # falling through after Dynamo's small general-purpose default limit.
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.cache_size_limit = 64
    # PyTorch on Sai hits an internal IntList/GenericList assertion when
    # tracing torch.func.grad with symbolic ragged dimensions. Static
    # specialization still accepts arbitrary atom/edge counts through
    # Dynamo's guarded recompilation cache, while keeping each CUDA graph
    # fully fused for its observed shape.
    return torch.compile(function, dynamic=False, fullgraph=True)


def _forward_function():
    global _COMPILED_FORWARD
    if _COMPILED_FORWARD is None:
        _COMPILED_FORWARD = _compile(_reference)
    return _COMPILED_FORWARD


def _vjp_function():
    global _COMPILED_VJP
    if _COMPILED_VJP is None:
        _COMPILED_VJP = _compile(_vjp_reference)
    return _COMPILED_VJP


def _hvp_function():
    global _COMPILED_HVP
    if _COMPILED_HVP is None:
        _COMPILED_HVP = _compile(_hvp_reference)
    return _COMPILED_HVP


class _FullDescriptorVjp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grad_output,
        rij,
        pair_i,
        pair_j,
        spins,
        atom_types,
        coefficient,
        dof_active,
        env_active,
        leg_mix,
    ):
        pulled = _vjp_function()(
            grad_output,
            rij,
            pair_i,
            pair_j,
            spins,
            atom_types,
            coefficient,
            dof_active,
            env_active,
            leg_mix,
        )
        ctx.save_for_backward(
            grad_output,
            rij,
            pair_i,
            pair_j,
            spins,
            atom_types,
            coefficient,
            dof_active,
            env_active,
            leg_mix,
        )
        return pulled

    @staticmethod
    def backward(
        ctx,
        grad_grad_rij,
        grad_grad_spins,
        grad_grad_coefficient,
        grad_grad_leg_mix,
    ):
        saved = ctx.saved_tensors
        differentiable = (saved[1], saved[4], saved[6], saved[9])
        cotangents = (
            grad_grad_rij,
            grad_grad_spins,
            grad_grad_coefficient,
            grad_grad_leg_mix,
        )
        cotangents = tuple(
            torch.zeros_like(value) if gradient is None else gradient
            for gradient, value in zip(cotangents, differentiable)
        )
        pulled = _hvp_function()(*saved, *cotangents)
        need = ctx.needs_input_grad
        return (
            pulled[0] if need[0] else None,
            pulled[1] if need[1] else None,
            None,
            None,
            pulled[2] if need[4] else None,
            None,
            pulled[3] if need[6] else None,
            None,
            None,
            pulled[4] if need[9] else None,
        )


class _FullDescriptor(torch.autograd.Function):
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
        leg_mix,
    ):
        result = _forward_function()(
            rij,
            pair_i,
            pair_j,
            spins,
            atom_types,
            coefficient,
            dof_active,
            env_active,
            leg_mix,
        )
        ctx.save_for_backward(
            rij,
            pair_i,
            pair_j,
            spins,
            atom_types,
            coefficient,
            dof_active,
            env_active,
            leg_mix,
        )
        return result

    @staticmethod
    def backward(ctx, grad_output):
        pulled = _FullDescriptorVjp.apply(grad_output, *ctx.saved_tensors)
        return pulled[0], None, None, pulled[1], None, pulled[2], None, None, pulled[3]


def full_descriptor(
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
    """Return the compiled 49D path, or ``None`` for the reference path."""
    compatible = (
        os.environ.get("TORCHNEP_OC_CUDA_FULL", "0") == "1"
        and rij.is_cuda
        and spins.is_cuda
        and descriptor.spin_order == 3
        and descriptor.spin_compress == 2
        and descriptor.spin_l_max == 2
        and descriptor.spin_soc
        and descriptor.dim == 49
        and atom_types is not None
        and coefficient is not None
        and dof_active is not None
        and env_active is not None
    )
    if not compatible:
        return None
    _template(descriptor)
    return _FullDescriptor.apply(
        rij,
        pair_i,
        pair_j,
        spins,
        atom_types,
        coefficient,
        dof_active,
        env_active,
        descriptor.radial_leg_mix,
    )
