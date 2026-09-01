# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.
# TorchNEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Bounded CUDA-Graph cache for spin-mode-2 O/C training.

Each entry owns exact-shape ragged buffers and a complete
descriptor -> E/F/V/MF/Tau -> loss -> backward -> Adam replay.  The
descriptor remains the pure Torch oracle during capture: this module must not
activate the shape-specialized Inductor full-pullback experiment.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
import time
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch


_GRAPH_ENV = "TORCHNEP_OC_CUDA_TRAIN_GRAPH_CACHE"
_GRAPH_CACHE_SIZE_ENV = "TORCHNEP_OC_CUDA_GRAPH_CACHE_SIZE"
_COMPILE_TAIL_ENV = "TORCHNEP_OC_CUDA_COMPILE_TAIL"
_VIRIAL_6 = [0, 1, 2, 4, 5, 8]
_INPUT_KEYS = (
    "rij_rad", "rij_ang", "pair_i_rad", "pair_j_rad",
    "pair_i_ang", "pair_j_ang", "atom_types", "struct_idx",
    "spins", "rij_spin", "pair_i_spin", "pair_j_spin",
    "energy", "natoms", "volumes", "energy_mask", "forces",
    "force_mask", "virial", "virial_mask", "mforces",
    "mforce_mask", "tau", "tau_mask",
)


def graph_cache_requested() -> bool:
    return os.environ.get(_GRAPH_ENV, "0") == "1"


def graph_cache_size() -> int:
    raw = os.environ.get(_GRAPH_CACHE_SIZE_ENV, "8")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_GRAPH_CACHE_SIZE_ENV} must be an integer"
        ) from exc
    if value < 1:
        raise ValueError(f"{_GRAPH_CACHE_SIZE_ENV} must be at least 1")
    return value


def graph_support_reason(model, device: torch.device) -> str:
    """Return empty for a reference-supported spin-mode-2 CUDA trainer."""
    if not graph_cache_requested():
        return f"{_GRAPH_ENV}=1 was not requested"
    if device.type != "cuda" or not torch.cuda.is_available():
        return "the O/C graph cache requires CUDA"
    if model.spin_mode != 2:
        return "the graph cache is restricted to spin_mode 2"
    if model.zbl is not None:
        return "ZBL has not passed CUDA-Graph joint-training validation"
    return ""


def configure_graph_optimizer(optimizer: torch.optim.Optimizer) -> None:
    """Make one Adam parameter group capture-safe without replacing state."""
    if not isinstance(optimizer, torch.optim.Adam):
        raise TypeError("the O/C CUDA graph cache currently requires Adam")
    if len(optimizer.param_groups) != 1:
        raise ValueError("the O/C CUDA graph cache requires one parameter group")
    group = optimizer.param_groups[0]
    group["capturable"] = True
    group["fused"] = True
    group["foreach"] = None
    for parameter, state in optimizer.state.items():
        step = state.get("step")
        if torch.is_tensor(step) and step.device != parameter.device:
            state["step"] = step.to(device=parameter.device)


def _optimizer_fingerprint(optimizer: torch.optim.Optimizer) -> Tuple:
    group = optimizer.param_groups[0]
    lr = group["lr"]
    if torch.is_tensor(lr):
        lr = float(lr.detach().item())
    return (
        float(lr), tuple(float(value) for value in group["betas"]),
        float(group["eps"]), float(group["weight_decay"]),
        bool(group["amsgrad"]), bool(group["maximize"]),
    )


def _batch_key(
    batch: Mapping,
    optimizer: torch.optim.Optimizer,
    need_forces: bool,
    need_virial: bool,
    need_mforce: bool,
    need_tau: bool,
) -> Tuple:
    tensors = []
    for name in _INPUT_KEYS:
        value = batch.get(name)
        if torch.is_tensor(value):
            tensors.append((name, tuple(value.shape), value.dtype, value.device))
    return (
        int(batch["N"]), int(batch["num_structures"]), tuple(tensors),
        bool(need_forces), bool(need_virial), bool(need_mforce),
        bool(need_tau), _optimizer_fingerprint(optimizer),
    )


def _masked_sumsq(
    delta: torch.Tensor,
    mask: torch.Tensor,
    components: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    weight = mask.to(delta.dtype)
    while weight.ndim < delta.ndim:
        weight = weight.unsqueeze(-1)
    count = mask.to(delta.dtype).sum()
    metric_sum = (delta.square() * weight).sum() / float(components)
    return metric_sum / count.clamp_min(1.0), metric_sum


def _joint_loss_metrics(
    model,
    batch: Mapping[str, torch.Tensor],
    weights: torch.Tensor,
    need_forces: bool,
    need_virial: bool,
    need_mforce: bool,
    need_tau: bool,
    backend: str,
    loss_tail=None,
):
    result = model.compute_properties(
        batch["rij_rad"], batch["rij_ang"],
        batch["pair_i_rad"], batch["pair_j_rad"],
        batch["pair_i_ang"], batch["pair_j_ang"],
        batch["atom_types"], batch["N"], batch["struct_idx"],
        batch["num_structures"], need_forces=need_forces,
        need_virial=need_virial, backend=backend, spins=batch["spins"],
        need_mforce=need_mforce, need_tau=need_tau,
        rij_spin=batch["rij_spin"], pi_spin=batch["pair_i_spin"],
        pj_spin=batch["pair_j_spin"], create_graph=True,
    )
    if loss_tail is not None:
        return loss_tail(result, batch, weights)
    return _joint_loss_from_result(
        model, result, batch, weights, need_forces, need_virial,
        need_mforce, need_tau,
    )


def _joint_loss_from_result(
    model,
    result: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    weights: torch.Tensor,
    need_forces: bool,
    need_virial: bool,
    need_mforce: bool,
    need_tau: bool,
):
    """Pure tensor tail from E/F/V/MF/Tau responses to loss and metrics."""
    zero = result["Etot"].sum() * 0.0
    metrics = {
        name: zero.clone() for name in (
            "sum_le", "sum_lf", "sum_lv", "sum_ls", "sum_lm",
            "sum_lt", "sum_e_resid", "count_e", "count_f", "count_v",
            "count_m", "count_t",
        )
    }

    e_delta = (
        result["Etot"] / batch["natoms"]
        - batch["energy"] / batch["natoms"]
    )
    e_loss, metrics["sum_le"] = _masked_sumsq(
        e_delta, batch["energy_mask"], 1
    )
    e_weight = batch["energy_mask"].to(e_delta.dtype)
    metrics["sum_e_resid"] = (e_delta * e_weight).sum()
    metrics["count_e"] = e_weight.sum()
    loss = weights[0] * e_loss

    if need_forces:
        f_loss, metrics["sum_lf"] = _masked_sumsq(
            result["forces"] - batch["forces"], batch["force_mask"], 3
        )
        metrics["count_f"] = batch["force_mask"].to(e_delta.dtype).sum()
        loss = loss + weights[1] * f_loss

    if need_virial:
        v_system = torch.zeros(
            batch["num_structures"], 9, dtype=e_delta.dtype,
            device=e_delta.device,
        )
        index = batch["struct_idx"].unsqueeze(-1).expand_as(result["virial"])
        v_system.scatter_add_(0, index, result["virial"])
        virial_components = batch.get("virial_components")
        if virial_components is None:
            virial_components = torch.tensor(
                _VIRIAL_6, dtype=torch.long, device=e_delta.device
            )
        v_delta = (
            v_system.index_select(1, virial_components)
            / batch["natoms"].unsqueeze(-1)
            - batch["virial"].index_select(1, virial_components)
            / batch["natoms"].unsqueeze(-1)
        )
        v_loss, metrics["sum_lv"] = _masked_sumsq(
            v_delta, batch["virial_mask"], 6
        )
        stress_delta = v_delta * (
            batch["natoms"] / batch["volumes"]
        ).unsqueeze(-1)
        _, metrics["sum_ls"] = _masked_sumsq(
            stress_delta, batch["virial_mask"], 6
        )
        metrics["count_v"] = batch["virial_mask"].to(e_delta.dtype).sum()
        loss = loss + weights[2] * v_loss

    active = model.spin_dof_type_active[batch["atom_types"]]
    if need_mforce:
        m_mask = batch["mforce_mask"] & active
        m_loss, metrics["sum_lm"] = _masked_sumsq(
            result["mforces"] - batch["mforces"], m_mask, 3
        )
        metrics["count_m"] = m_mask.to(e_delta.dtype).sum()
        loss = loss + weights[3] * m_loss
    if need_tau:
        t_mask = batch["tau_mask"] & active
        t_loss, metrics["sum_lt"] = _masked_sumsq(
            result["tau"] - batch["tau"], t_mask, 3
        )
        metrics["count_t"] = t_mask.to(e_delta.dtype).sum()
        loss = loss + weights[4] * t_loss
    return loss, metrics


def _gradient_norm(grads: Iterable[torch.Tensor]) -> torch.Tensor:
    norms = torch._foreach_norm(list(grads), 2.0)
    return torch.linalg.vector_norm(torch.stack(norms), 2.0)


@dataclass
class GraphStepResult:
    loss: torch.Tensor
    grad_norm: torch.Tensor
    found_inf: torch.Tensor
    metrics: Dict[str, torch.Tensor]
    cache_hit: bool
    capture_ms: float


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    static_batch: Dict
    controls: torch.Tensor
    force_skip: torch.Tensor
    loss: torch.Tensor
    grad_norm: torch.Tensor
    found_inf: torch.Tensor
    metrics: Dict[str, torch.Tensor]
    capture_ms: float


class OCCudaGraphTrainCache:
    """Bounded LRU of exact-ragged-shape complete training graphs."""

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        *,
        backend: str = "bmm",
        max_entries: Optional[int] = None,
        warmup: int = 1,
    ):
        configure_graph_optimizer(optimizer)
        self.model = model
        self.optimizer = optimizer
        self.backend = backend
        self.max_entries = (
            graph_cache_size() if max_entries is None else int(max_entries)
        )
        if self.max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.warmup = max(1, int(warmup))
        self._entries: OrderedDict[Tuple, _GraphEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_capture_ms = 0.0
        self._compiled_loss_tails = {}

    def _loss_tail(
        self, need_forces: bool, need_virial: bool,
        need_mforce: bool, need_tau: bool,
    ):
        if os.environ.get(_COMPILE_TAIL_ENV, "0") != "1":
            return None
        key = (need_forces, need_virial, need_mforce, need_tau)
        compiled = self._compiled_loss_tails.get(key)
        if compiled is None:
            model = self.model

            def tail(result, batch, weights):
                return _joint_loss_from_result(
                    model, result, batch, weights, *key
                )

            compiled = torch.compile(tail, dynamic=True, fullgraph=True)
            self._compiled_loss_tails[key] = compiled
        return compiled

    def __len__(self):
        return len(self._entries)

    def stats(self) -> Dict[str, float]:
        requests = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hits / requests if requests else 0.0,
            "total_capture_ms": self.total_capture_ms,
        }

    def _make_static_batch(
        self, batch: Mapping, need_mforce: bool, need_tau: bool
    ) -> Dict:
        static = {
            name: batch[name].detach().clone()
            for name in _INPUT_KEYS if torch.is_tensor(batch.get(name))
        }
        static.update({
            "N": int(batch["N"]),
            "num_structures": int(batch["num_structures"]),
            "need_mforce": bool(need_mforce),
            "need_tau": bool(need_tau),
            "virial_components": torch.tensor(
                _VIRIAL_6, dtype=torch.long,
                device=batch["rij_rad"].device,
            ),
        })
        return static

    def _capture(
        self,
        batch: Mapping,
        *,
        need_forces: bool,
        need_virial: bool,
        need_mforce: bool,
        need_tau: bool,
        curriculum_enabled: bool,
    ) -> _GraphEntry:
        static_batch = self._make_static_batch(batch, need_mforce, need_tau)
        dtype = static_batch["rij_rad"].dtype
        device = static_batch["rij_rad"].device
        controls = torch.zeros(9, dtype=dtype, device=device)
        force_skip = torch.ones((), dtype=torch.float32, device=device)
        parameters = [
            parameter for group in self.optimizer.param_groups
            for parameter in group["params"] if parameter.requires_grad
        ]
        loss_tail = self._loss_tail(
            need_forces, need_virial, need_mforce, need_tau
        )

        def train_step():
            self.optimizer.zero_grad(set_to_none=False)
            loss, metrics = _joint_loss_metrics(
                self.model, static_batch, controls[:5], need_forces,
                need_virial, need_mforce, need_tau, self.backend, loss_tail,
            )
            loss.backward()
            grads = [parameter.grad for parameter in parameters]
            torch._foreach_add_(
                grads, torch._foreach_mul(parameters, controls[6])
            )
            torch._foreach_add_(
                grads,
                torch._foreach_mul(torch._foreach_sign(parameters), controls[5]),
            )

            before = []
            if curriculum_enabled:
                for net in self.model.fitting_nets:
                    grad_rows = net.w0.grad[
                        self.model.spin_order3_input_start:
                    ]
                    grad_rows.mul_(controls[8])
                    before.append(net.w0[
                        self.model.spin_order3_input_start:
                    ].clone())

            grad_norm = _gradient_norm(grads)
            clip = (controls[7] / (grad_norm + 1.0e-6)).clamp(max=1.0)
            torch._foreach_mul_(grads, clip)
            found_inf = (~torch.isfinite(grad_norm)).to(torch.float32)
            self.optimizer.found_inf = torch.maximum(found_inf, force_skip)
            self.optimizer.step()

            if curriculum_enabled:
                for net, previous in zip(self.model.fitting_nets, before):
                    current = net.w0[self.model.spin_order3_input_start:]
                    current.copy_(
                        previous + controls[8] * (current - previous)
                    )
            return loss, grad_norm, found_inf, metrics

        start = time.perf_counter()
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(self.warmup):
                train_step()
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        self.optimizer.zero_grad(set_to_none=False)
        with torch.cuda.graph(graph):
            loss, grad_norm, found_inf, metrics = train_step()
        torch.cuda.synchronize()
        capture_ms = (time.perf_counter() - start) * 1.0e3
        force_skip.zero_()
        torch.cuda.synchronize()
        return _GraphEntry(
            graph, static_batch, controls, force_skip, loss, grad_norm,
            found_inf, metrics, capture_ms,
        )

    def step(
        self,
        batch: Mapping,
        *,
        weights: Tuple[float, float, float, float, float],
        l1_coeff: float,
        l2_coeff: float,
        max_grad_norm: float,
        curriculum_scale: float,
        need_forces: bool,
        need_virial: bool,
        need_mforce: bool,
        need_tau: bool,
    ) -> GraphStepResult:
        key = _batch_key(
            batch, self.optimizer, need_forces, need_virial,
            need_mforce, need_tau,
        )
        entry = self._entries.pop(key, None)
        cache_hit = entry is not None
        if cache_hit:
            self.hits += 1
        else:
            self.misses += 1
            entry = self._capture(
                batch, need_forces=need_forces, need_virial=need_virial,
                need_mforce=need_mforce, need_tau=need_tau,
                curriculum_enabled=bool(self.model.spin_curriculum),
            )
            self.total_capture_ms += entry.capture_ms
            if len(self._entries) >= self.max_entries:
                self._entries.popitem(last=False)
                self.evictions += 1
        self._entries[key] = entry

        for name, target in entry.static_batch.items():
            source = batch.get(name)
            if torch.is_tensor(target) and torch.is_tensor(source):
                target.copy_(source)
        grad_limit = (
            float(max_grad_norm) if max_grad_norm > 0 else float("inf")
        )
        control_values = (
            *weights, float(l1_coeff), float(l2_coeff), grad_limit,
            float(curriculum_scale),
        )
        entry.controls.copy_(entry.controls.new_tensor(control_values))
        entry.graph.replay()
        return GraphStepResult(
            entry.loss, entry.grad_norm, entry.found_inf, entry.metrics,
            cache_hit, 0.0 if cache_hit else entry.capture_ms,
        )
