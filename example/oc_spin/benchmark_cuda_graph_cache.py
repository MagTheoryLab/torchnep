"""Matched V100 A/B for the bounded generic O/C CUDA training graph cache."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import time

import torch

from benchmark_cuda_train_pullback import config, ragged_case
from torchnep.model import NEPModel
from torchnep.oc_cuda_graph import OCCudaGraphTrainCache, _joint_loss_metrics


WEIGHTS = (1.0, 0.7, 0.2, 0.5, 0.3)


def median_mad(samples):
    median = statistics.median(samples)
    return {
        "median_ms": median,
        "mad_ms": statistics.median(abs(value - median) for value in samples),
        "samples_ms": samples,
    }


def synchronize_timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1.0e3


def training_batch(n_atoms, neighbors, seed):
    raw = ragged_case(n_atoms, neighbors, seed, "noncollinear", torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(seed + 91)
    structures = raw["num_structures"]
    counts = torch.bincount(raw["struct_idx"].cpu(), minlength=structures)
    return {
        "rij_rad": raw["rij_rad"], "rij_ang": raw["rij_ang"],
        "pair_i_rad": raw["pi_rad"], "pair_j_rad": raw["pj_rad"],
        "pair_i_ang": raw["pi_ang"], "pair_j_ang": raw["pj_ang"],
        "atom_types": raw["atom_types"], "struct_idx": raw["struct_idx"],
        "spins": raw["spins"], "rij_spin": raw["rij_spin"],
        "pair_i_spin": raw["pi_spin"], "pair_j_spin": raw["pj_spin"],
        "energy": torch.randn(structures, generator=generator).cuda(),
        "natoms": counts.to(torch.float32).cuda(),
        "volumes": (counts.to(torch.float32) * 13.0).cuda(),
        "energy_mask": torch.ones(structures, dtype=torch.bool, device="cuda"),
        "forces": torch.randn(n_atoms, 3, generator=generator).cuda(),
        "force_mask": torch.ones(n_atoms, dtype=torch.bool, device="cuda"),
        "virial": torch.randn(structures, 9, generator=generator).cuda(),
        "virial_mask": torch.ones(structures, dtype=torch.bool, device="cuda"),
        "mforces": torch.randn(n_atoms, 3, generator=generator).cuda(),
        "mforce_mask": torch.ones(n_atoms, dtype=torch.bool, device="cuda"),
        "tau": torch.randn(n_atoms, 3, generator=generator).cuda(),
        "tau_mask": torch.ones(n_atoms, dtype=torch.bool, device="cuda"),
        "N": n_atoms, "num_structures": structures, "edges": raw["edges"],
    }


def optimizer(model):
    model.b1.requires_grad_(False)
    return torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=2.0e-4, weight_decay=0.0, amsgrad=True, fused=True, capturable=True,
    )


def eager_step(model, opt, batch, samples=None):
    torch.cuda.synchronize()
    full_start = time.perf_counter()
    response_start = full_start
    loss, _ = _joint_loss_metrics(
        model, batch, torch.tensor(WEIGHTS, device="cuda"),
        True, True, True, True, "bmm",
    )
    torch.cuda.synchronize()
    response_ms = (time.perf_counter() - response_start) * 1.0e3
    opt.zero_grad(set_to_none=False)
    backward_start = time.perf_counter()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    torch.cuda.synchronize()
    backward_ms = (time.perf_counter() - backward_start) * 1.0e3
    step_start = time.perf_counter()
    opt.step()
    torch.cuda.synchronize()
    optimizer_ms = (time.perf_counter() - step_start) * 1.0e3
    row = {
        "response_ms": response_ms, "outer_backward_ms": backward_ms,
        "optimizer_ms": optimizer_ms,
        "train_step_ms": (time.perf_counter() - full_start) * 1.0e3,
    }
    if samples is not None:
        for name, value in row.items():
            samples[name].append(value)
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--cache-size", type=int, default=3)
    parser.add_argument("--bank", action="store_true")
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--struct-center", action="store_true")
    parser.add_argument("--compile-tail", action="store_true")
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--compress", type=int, default=2)
    parser.add_argument("--spin-l", type=int, default=2)
    parser.add_argument("--soc", type=int, choices=(0, 1), default=1)
    parser.add_argument("--spin-basis", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ["TORCHNEP_OC_CUDA_FULL"] = "0"
    os.environ["TORCHNEP_OC_CUDA_TRAIN"] = "0"
    os.environ["TORCHNEP_OC_CUDA_BANK"] = (
        "1" if args.bank or args.center else "0"
    )
    os.environ["TORCHNEP_OC_CUDA_CENTER"] = "1" if args.center else "0"
    os.environ["TORCHNEP_OC_CUDA_STRUCT_CENTER"] = (
        "1" if args.struct_center else "0"
    )
    os.environ["TORCHNEP_OC_CUDA_COMPILE_TAIL"] = (
        "1" if args.compile_tail else "0"
    )
    os.environ["TORCHNEP_OC_CUDA_GRAPH"] = "1"
    torch.manual_seed(19)
    model_config = config()
    model_config.update({
        "spin_order": args.order,
        "spin_compress": args.compress,
        "spin_l_max": [args.spin_l, 0, 0],
        "spin_soc": args.soc,
        "spin_basis_size": [args.spin_basis, 0],
        "spin_curriculum": False,
    })
    base = NEPModel(model_config).cuda().train()
    batches = {
        n_atoms: training_batch(n_atoms, 16, 1000 + n_atoms)
        for n_atoms in (32, 108, 256)
    }
    results = []
    cold_sum = 0.0
    hot_sum = 0.0
    eager_sum = 0.0
    for n_atoms, batch in batches.items():
        eager_model = copy.deepcopy(base)
        eager_optimizer = optimizer(eager_model)
        eager_step(eager_model, eager_optimizer, batch)
        phase_samples = {
            "response_ms": [], "outer_backward_ms": [],
            "optimizer_ms": [], "train_step_ms": [],
        }
        eager_memory_base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.repeats):
            eager_step(eager_model, eager_optimizer, batch, phase_samples)
        eager_peak = torch.cuda.max_memory_allocated() - eager_memory_base
        eager_final = [
            parameter.detach().clone() for parameter in eager_model.parameters()
        ]
        del eager_model, eager_optimizer
        torch.cuda.empty_cache()

        graph_model = copy.deepcopy(base)
        graph_optimizer = optimizer(graph_model)
        cache = OCCudaGraphTrainCache(
            graph_model, graph_optimizer, backend="bmm",
            max_entries=args.cache_size, warmup=1,
        )
        kwargs = dict(
            weights=WEIGHTS, l1_coeff=0.0, l2_coeff=0.0,
            max_grad_norm=10.0, curriculum_scale=1.0,
            need_forces=True, need_virial=True, need_mforce=True, need_tau=True,
        )
        graph_memory_base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        _, cold_ms = synchronize_timed(lambda: cache.step(batch, **kwargs))
        hot_samples = []
        for _ in range(args.repeats):
            _, elapsed = synchronize_timed(lambda: cache.step(batch, **kwargs))
            hot_samples.append(elapsed)
        graph_peak = torch.cuda.max_memory_allocated() - graph_memory_base
        parameter_max_abs = max(
            (candidate - reference).abs().max().item()
            for candidate, reference in zip(graph_model.parameters(), eager_final)
        )
        eager_stats = {
            name: median_mad(values) for name, values in phase_samples.items()
        }
        hot_stats = median_mad(hot_samples)
        eager_train = eager_stats["train_step_ms"]["median_ms"]
        hot_train = hot_stats["median_ms"]
        results.append({
            "atoms": n_atoms, "edges": batch["edges"],
            "eager": eager_stats, "graph_hot": hot_stats,
            "graph_cold_ms": cold_ms,
            "hot_speedup": eager_train / hot_train,
            "ten_step_parameter_max_abs": parameter_max_abs,
            "eager_peak_bytes": eager_peak,
            "graph_peak_bytes": graph_peak,
            "cache": cache.stats(),
        })
        eager_sum += eager_train
        hot_sum += hot_train
        cold_sum += cold_ms

    eager_trace = args.cycles * eager_sum
    graph_trace = cold_sum + (args.cycles - 1) * hot_sum
    payload = {
        "device": torch.cuda.get_device_name(), "dtype": "float32",
        "repeats": args.repeats, "cycles": args.cycles,
        "analytic_bank": args.bank or args.center,
        "analytic_center": args.center,
        "analytic_struct_center": args.struct_center,
        "compiled_loss_tail": args.compile_tail,
        "spin_config": {
            "order": args.order, "compress": args.compress,
            "l_max": args.spin_l, "soc": args.soc,
            "basis_size": args.spin_basis,
            "descriptor_dim": base.spin_dim,
        },
        "results": results,
        "trace": {
            "eager_ms": eager_trace, "graph_cold_amortized_ms": graph_trace,
            "speedup": eager_trace / graph_trace,
            "hot_cycle_speedup": eager_sum / hot_sum,
            "cold_sum_ms": cold_sum,
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
