# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.
# TorchNEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Correctness and cost gate for the ``nep4_spin3`` descriptor.

The benchmark compares spin3 with the frozen spin2 implementation on the
same ragged local environments and magnetic radial coefficients.  It reports
descriptor forward, first pullback, and force-training pullback timings.  The
last case differentiates a loss built from position and spin derivatives, so
it exercises the second derivatives required by E/F/MF training.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import time

import torch

from torchnep.model import NEPModel


def _config(spin_mode: int):
    return {
        "num_types": 2,
        "type_names": ["Fe", "Ge"],
        "cutoff_radial": 6.0,
        "cutoff_angular": 5.0,
        "n_max_radial": 4,
        "n_max_angular": 4,
        "basis_size_radial": 8,
        "basis_size_angular": 8,
        "l_max": [4, 2, 0],
        "neuron": 30,
        "spin_mode": spin_mode,
        "spin_cutoff": [6.0, 6.0],
        "spin_l_max": [2, 0, 0],
        "spin_compress": 2,
        "spin_basis_size": [8, 0],
        "spin_order": 3,
        "spin_soc": 1,
        "spin_curriculum": False,
        "spin_dof_type": ["Fe", "Ge"],
        "spin_env_type": ["Fe", "Ge"],
    }


def _ragged_case(atoms: int, neighbors: int, seed: int, dtype):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pair_i = torch.arange(atoms).repeat_interleave(neighbors)
    shift = torch.arange(1, neighbors + 1).repeat(atoms)
    pair_j = (pair_i + shift) % atoms
    direction = torch.randn(
        pair_i.numel(), 3, generator=generator, dtype=dtype
    )
    direction /= torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    radius = 1.2 + 4.4 * torch.rand(
        pair_i.numel(), 1, generator=generator, dtype=dtype
    )
    spins = torch.randn(atoms, 3, generator=generator, dtype=dtype)
    magnitude = 0.4 + 2.2 * torch.rand(
        atoms, 1, generator=generator, dtype=dtype
    )
    spins = magnitude * spins / torch.linalg.vector_norm(
        spins, dim=-1, keepdim=True
    )
    return {
        "rij": (radius * direction).cuda(),
        "pair_i": pair_i.cuda(),
        "pair_j": pair_j.cuda(),
        "spins": spins.cuda(),
        "atom_types": (torch.arange(atoms) % 2).cuda(),
        "atoms": atoms,
        "edges": pair_i.numel(),
    }


def _descriptor(descriptor, coefficient, batch):
    active = torch.ones(2, dtype=torch.bool, device="cuda")
    return descriptor(
        batch["rij"],
        batch["pair_i"],
        batch["pair_j"],
        batch["spins"],
        batch["atoms"],
        batch["atom_types"],
        coefficient,
        None,
        active,
        active,
    )


def _error(candidate, reference):
    delta = (candidate - reference).abs()
    scale = reference.abs().clamp_min(1.0e-12)
    return {
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "max_rel": float((delta / scale).max()) if delta.numel() else 0.0,
    }


def _legacy_projection(spin2, spin3, q3):
    pieces = []
    by_name = spin3.split_by_token(q3)
    for token in spin2.schema:
        name = (
            "correlation_distinct_l0"
            if token.name == "correlation_distinct_neighbor"
            else token.name
        )
        pieces.append(by_name[name])
    return torch.cat(pieces, dim=-1)


def _forward_and_training_pullback(descriptor, coefficient, batch, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    head = torch.randn(
        descriptor.dim,
        generator=generator,
        dtype=batch["rij"].dtype,
        device="cuda",
    )
    q = _descriptor(descriptor, coefficient, batch)
    energy = (q * head).sum()
    first = torch.autograd.grad(
        energy,
        (batch["rij"], batch["spins"]),
        create_graph=True,
    )
    response_loss = (
        1.0e-3 * energy.square()
        + first[0].square().mean()
        + first[1].square().mean()
    )
    second = torch.autograd.grad(
        response_loss,
        (coefficient, descriptor.radial_leg_mix),
    )
    return q, first, second, response_loss


def _correctness(dtype):
    batch0 = _ragged_case(24, 7, 20260903, dtype)
    models = {
        mode: NEPModel(_config(mode)).cuda().to(dtype)
        for mode in (2, 3)
    }
    with torch.no_grad():
        models[3].c_param_spin.copy_(models[2].c_param_spin)
        models[3].spin_descriptor.radial_leg_mix.copy_(
            models[2].spin_descriptor.radial_leg_mix
        )

    # First prove that spin3 is an exact semantic superset of spin2 when both
    # use the same pure-Torch reference contractions.
    os.environ["TORCHNEP_OC_CUDA_BANK"] = "0"
    q2 = _descriptor(
        models[2].spin_descriptor, models[2].c_param_spin, batch0
    )
    q3 = _descriptor(
        models[3].spin_descriptor, models[3].c_param_spin, batch0
    )
    superset = _error(
        _legacy_projection(models[2].spin_descriptor,
                           models[3].spin_descriptor, q3),
        q2,
    )

    # Then compare the production analytic CUDA bank with that reference,
    # including the double backward required by force/mforce supervision.
    def run(enabled):
        descriptor = copy.deepcopy(models[3].spin_descriptor)
        coefficient = models[3].c_param_spin.detach().clone().requires_grad_()
        batch = {
            key: (
                value.detach().clone().requires_grad_(
                    key in ("rij", "spins")
                )
                if torch.is_tensor(value) and value.is_floating_point()
                else value
            )
            for key, value in batch0.items()
        }
        os.environ["TORCHNEP_OC_CUDA_BANK"] = "1" if enabled else "0"
        return _forward_and_training_pullback(
            descriptor, coefficient, batch, seed=20260904
        )

    reference = run(False)
    candidate = run(True)
    cuda = {
        "descriptor": _error(candidate[0], reference[0]),
        "position_pullback": _error(candidate[1][0], reference[1][0]),
        "spin_pullback": _error(candidate[1][1], reference[1][1]),
        "coefficient_training_pullback": _error(
            candidate[2][0], reference[2][0]
        ),
        "leg_mix_training_pullback": _error(
            candidate[2][1], reference[2][1]
        ),
        "response_loss_abs": abs(float(candidate[3] - reference[3])),
    }
    return {"spin3_superset_spin2": superset, "cuda_vs_reference": cuda}


def _median_mad(values):
    median = statistics.median(values)
    return {
        "median_ms": median,
        "mad_ms": statistics.median(abs(value - median) for value in values),
        "samples_ms": values,
    }


def _timed(call, warmup, repeats):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1.0e3)
    return _median_mad(values)


def _performance(mode, batch0, warmup, repeats):
    model = NEPModel(_config(mode)).cuda().float().train()
    coefficient = model.c_param_spin
    descriptor = model.spin_descriptor
    generator = torch.Generator(device="cuda").manual_seed(20260905)
    head = torch.randn(
        descriptor.dim, generator=generator, device="cuda"
    )

    def fresh_batch():
        return {
            key: (
                value.detach().requires_grad_(key in ("rij", "spins"))
                if torch.is_tensor(value) and value.is_floating_point()
                else value
            )
            for key, value in batch0.items()
        }

    def forward():
        with torch.no_grad():
            _descriptor(descriptor, coefficient, batch0)

    def first_pullback():
        batch = fresh_batch()
        q = _descriptor(descriptor, coefficient, batch)
        torch.autograd.grad(
            (q * head).sum(),
            (batch["rij"], batch["spins"], coefficient,
             descriptor.radial_leg_mix),
        )

    def training_pullback():
        batch = fresh_batch()
        q = _descriptor(descriptor, coefficient, batch)
        energy = (q * head).sum()
        position, spin = torch.autograd.grad(
            energy, (batch["rij"], batch["spins"]), create_graph=True
        )
        loss = (
            1.0e-3 * energy.square()
            + position.square().mean()
            + spin.square().mean()
        )
        torch.autograd.grad(
            loss, tuple(descriptor.parameters()) + (coefficient,)
        )

    # Materialize the extension and allocator state before resetting memory.
    training_pullback()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    result = {
        "dimension": descriptor.dim,
        "forward": _timed(forward, warmup, repeats),
        "first_pullback": _timed(first_pullback, warmup, repeats),
        "training_pullback": _timed(
            training_pullback, warmup, max(3, repeats // 2)
        ),
    }
    result["peak_memory_mib"] = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", type=int, default=512)
    parser.add_argument("--neighbors", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_spin3_descriptor requires CUDA")
    os.environ["TORCHNEP_OC_CUDA_FULL"] = "0"
    os.environ["TORCHNEP_OC_CUDA_BANK"] = "1"
    torch.manual_seed(20260903)
    torch.cuda.manual_seed_all(20260903)
    batch = _ragged_case(
        args.atoms, args.neighbors, 20260906, torch.float32
    )
    performance = {
        f"spin{mode}": _performance(
            mode, batch, args.warmup, args.repeats
        )
        for mode in (2, 3)
    }
    overhead = {}
    for phase in ("forward", "first_pullback", "training_pullback"):
        old = performance["spin2"][phase]["median_ms"]
        new = performance["spin3"][phase]["median_ms"]
        overhead[phase] = {
            "ratio": new / old,
            "overhead_percent": 100.0 * (new / old - 1.0),
        }
    payload = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "case": {
            "atoms": args.atoms,
            "neighbors": args.neighbors,
            "edges": batch["edges"],
            "dtype": "float32",
        },
        "correctness_float64": _correctness(torch.float64),
        "performance": performance,
        "spin3_over_spin2": overhead,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
