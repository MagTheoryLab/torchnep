"""Internal O3-C2 CUDA training-pullback prototype.

The production reference remains :mod:`torchnep.magnetic_polynomial`.  This
module replaces one complete O3 sector only when explicitly enabled through
``TORCHNEP_OC_CUDA_TRAIN=1`` and the tensors are CUDA C=2 tensors.  It is kept
internal until the full 49-dimensional descriptor passes the production gate.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
import importlib

import torch


_EXTENSION = None
_EXTENSION_LOCK = threading.Lock()
_COMPILED_VJP = None


def _enabled() -> bool:
    return os.environ.get("TORCHNEP_OC_CUDA_TRAIN", "0") == "1"


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is None:
            if os.environ.get("TORCHNEP_OC_CUDA_PREBUILT", "0") == "1":
                _EXTENSION = importlib.import_module("torchnep_oc_moment_gate_v1")
                return _EXTENSION
            from torch.utils.cpp_extension import load

            source = Path(__file__).with_name("csrc") / "oc_moment_gate.cu"
            _EXTENSION = load(
                name="torchnep_oc_moment_gate_v1",
                sources=[str(source)],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "-lineinfo"],
                verbose=os.environ.get("TORCHNEP_OC_CUDA_VERBOSE", "0") == "1",
            )
    return _EXTENSION


def _moment_gate_reference(
    weights: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    spins: torch.Tensor,
) -> torch.Tensor:
    """Pure-Torch oracle for the O3 radial moment-gate sector."""
    si = spins.index_select(0, pair_i)
    sj = spins.index_select(0, pair_j)
    dot = (si * sj).sum(-1)
    moment = dot * (si.square().sum(-1) + sj.square().sum(-1))
    result = spins.new_zeros((spins.shape[0], weights.shape[1]))
    return result.index_add(0, pair_i, weights * moment.unsqueeze(-1))


def _moment_gate_vjp_reference(
    grad_output: torch.Tensor,
    weights: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    spins: torch.Tensor,
):
    """Analytic VJP oracle, expressed with differentiable Torch primitives."""
    si = spins.index_select(0, pair_i)
    sj = spins.index_select(0, pair_j)
    si2 = si.square().sum(-1)
    sj2 = sj.square().sum(-1)
    dot = (si * sj).sum(-1)
    moment = dot * (si2 + sj2)
    edge_pull = grad_output.index_select(0, pair_i)
    grad_weights = edge_pull * moment.unsqueeze(-1)
    scale = (edge_pull * weights).sum(-1)
    pull_i = scale.unsqueeze(-1) * (
        sj * (si2 + sj2).unsqueeze(-1) + 2.0 * dot.unsqueeze(-1) * si
    )
    pull_j = scale.unsqueeze(-1) * (
        si * (si2 + sj2).unsqueeze(-1) + 2.0 * dot.unsqueeze(-1) * sj
    )
    grad_spins = spins.new_zeros(spins.shape)
    grad_spins = grad_spins.index_add(0, pair_i, pull_i)
    grad_spins = grad_spins.index_add(0, pair_j, pull_j)
    return grad_weights, grad_spins


def _compiled_vjp_reference():
    global _COMPILED_VJP
    if _COMPILED_VJP is None:
        if os.environ.get("TORCHNEP_OC_CUDA_COMPILE_HVP", "1") == "1":
            # AOTAutograd otherwise donates saved buffers and rejects the
            # retained graph needed when a response loss differentiates this
            # VJP.  This prototype requires the double-backward contract.
            from torch._functorch import config as functorch_config

            functorch_config.donated_buffer = False
            _COMPILED_VJP = torch.compile(
                _moment_gate_vjp_reference,
                dynamic=True,
                fullgraph=True,
            )
        else:
            _COMPILED_VJP = _moment_gate_vjp_reference
    return _COMPILED_VJP


class _MomentGateVjp(torch.autograd.Function):
    """Native first VJP with a differentiable, compiled HVP definition."""

    @staticmethod
    def forward(ctx, grad_output, weights, pair_i, pair_j, spins):
        grad_weights, grad_spins = _load_extension().backward(
            grad_output.contiguous(),
            weights.contiguous(),
            pair_i.contiguous(),
            pair_j.contiguous(),
            spins.contiguous(),
        )
        ctx.save_for_backward(grad_output, weights, pair_i, pair_j, spins)
        return grad_weights, grad_spins

    @staticmethod
    def backward(ctx, grad_grad_weights, grad_grad_spins):
        grad_output, weights, pair_i, pair_j, spins = ctx.saved_tensors
        need = ctx.needs_input_grad
        with torch.enable_grad():
            # autograd.grad requires every listed input to require gradients.
            # Compute the narrow VJP pull once, then mask unused custom-op
            # inputs on return.  PyTorch may pass None for either cotangent.
            go = grad_output.detach().requires_grad_(True)
            ww = weights.detach().requires_grad_(True)
            ss = spins.detach().requires_grad_(True)
            grad_weights, grad_spins = _compiled_vjp_reference()(
                go, ww, pair_i, pair_j, ss
            )
            if grad_grad_weights is None:
                grad_grad_weights = torch.zeros_like(grad_weights)
            if grad_grad_spins is None:
                grad_grad_spins = torch.zeros_like(grad_spins)
            pulled = torch.autograd.grad(
                (grad_weights, grad_spins),
                (go, ww, ss),
                grad_outputs=(grad_grad_weights, grad_grad_spins),
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            )
        return (
            pulled[0] if need[0] else None,
            pulled[1] if need[1] else None,
            None,
            None,
            pulled[2] if need[4] else None,
        )


class _MomentGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, pair_i, pair_j, spins):
        result = _load_extension().forward(
            weights.contiguous(),
            pair_i.contiguous(),
            pair_j.contiguous(),
            spins.contiguous(),
        )
        ctx.save_for_backward(weights, pair_i, pair_j, spins)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        weights, pair_i, pair_j, spins = ctx.saved_tensors
        grad_weights, grad_spins = _MomentGateVjp.apply(
            grad_output, weights, pair_i, pair_j, spins
        )
        return grad_weights, None, None, grad_spins


def moment_gate(
    weights: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    spins: torch.Tensor,
) -> torch.Tensor:
    """Return the O3 moment-gate sector, selecting CUDA only when enabled."""
    compatible = (
        _enabled()
        and os.environ.get("TORCHNEP_OC_CUDA_FULL", "0") != "1"
        and weights.is_cuda
        and spins.is_cuda
        and weights.ndim == 2
        and weights.shape[1] == 2
        and weights.dtype in (torch.float32, torch.float64)
        and spins.dtype == weights.dtype
        and pair_i.dtype == torch.long
        and pair_j.dtype == torch.long
    )
    if compatible:
        return _MomentGate.apply(weights, pair_i, pair_j, spins)
    return _moment_gate_reference(weights, pair_i, pair_j, spins)
