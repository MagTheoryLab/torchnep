"""Close cold-shape and large-system O3-C2 CUDA training evidence gaps."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import time

import torch

import benchmark_cuda_train_pullback as pullback
from benchmark_cuda_graph_cache import WEIGHTS, eager_step, optimizer, training_batch
from torchnep.model import NEPModel
from torchnep.oc_cuda_graph import OCCudaGraphTrainCache


GRAPH_KWARGS = dict(
    weights=WEIGHTS,
    l1_coeff=0.0,
    l2_coeff=0.0,
    max_grad_norm=10.0,
    curriculum_scale=1.0,
    need_forces=True,
    need_virial=True,
    need_mforce=True,
    need_tau=True,
)


def median_mad(values):
    median = statistics.median(values)
    return {
        "median_ms": median,
        "mad_ms": statistics.median(abs(value - median) for value in values),
        "samples_ms": values,
    }


def configure_candidate():
    os.environ["TORCHNEP_OC_CUDA_FULL"] = "0"
    os.environ["TORCHNEP_OC_CUDA_TRAIN"] = "0"
    os.environ["TORCHNEP_OC_CUDA_BANK"] = "1"
    os.environ["TORCHNEP_OC_CUDA_CENTER"] = "1"
    os.environ["TORCHNEP_OC_CUDA_STRUCT_CENTER"] = "1"
    os.environ["TORCHNEP_OC_CUDA_GRAPH"] = "1"
    pullback.CANDIDATE_KIND = "center"


def timed_step(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1.0e3


def trace_scenario(base, batches, order, cache_size):
    eager_model = copy.deepcopy(base)
    eager_optimizer = optimizer(eager_model)
    eager_times = []
    for key in order:
        _, elapsed = timed_step(
            lambda key=key: eager_step(
                eager_model, eager_optimizer, batches[key]
            )
        )
        eager_times.append(elapsed)
    eager_parameters = [
        parameter.detach().clone() for parameter in eager_model.parameters()
    ]
    del eager_model, eager_optimizer
    torch.cuda.empty_cache()

    graph_model = copy.deepcopy(base)
    graph_optimizer = optimizer(graph_model)
    cache = OCCudaGraphTrainCache(
        graph_model, graph_optimizer, backend="bmm",
        max_entries=cache_size, warmup=1,
    )
    graph_times = []
    capture_times = []
    hits = []
    for key in order:
        result, elapsed = timed_step(
            lambda key=key: cache.step(batches[key], **GRAPH_KWARGS)
        )
        graph_times.append(elapsed)
        capture_times.append(result.capture_ms)
        hits.append(result.cache_hit)
    parameter_max_abs = max(
        (candidate - reference).abs().max().item()
        for candidate, reference in zip(graph_model.parameters(), eager_parameters)
    )
    eager_total = sum(eager_times)
    graph_total = sum(graph_times)
    cold = [value for value in capture_times if value > 0.0]
    hot = [value for value, hit in zip(graph_times, hits) if hit]
    return {
        "steps": len(order),
        "unique_shapes": len(set(order)),
        "cache_size": cache_size,
        "eager_total_ms": eager_total,
        "graph_total_ms": graph_total,
        "end_to_end_speedup": eager_total / graph_total,
        "hit_rate": sum(hits) / len(hits),
        "captures": len(cold),
        "evictions": cache.stats()["evictions"],
        "capture": median_mad(cold) if cold else None,
        "hot_step": median_mad(hot) if hot else None,
        "eager_step": median_mad(eager_times),
        "parameter_max_abs": parameter_max_abs,
        "cache": cache.stats(),
    }


def cold_shape_matrix(base):
    shapes = [
        (32, 11), (40, 13), (48, 15), (56, 17),
        (64, 19), (72, 21), (80, 12), (88, 14),
        (96, 16), (108, 18), (128, 20), (160, 22),
    ]
    batches = {
        index: training_batch(atoms, neighbors, 4100 + index)
        for index, (atoms, neighbors) in enumerate(shapes)
    }
    # Exclude process-wide extension/library initialization from the exact-shape
    # cold ledger. Each scenario below still owns a fresh optimizer and cache,
    # so optimizer materialization, Graph warmup, capture, and replay are counted.
    prime_model = copy.deepcopy(base)
    prime_optimizer = optimizer(prime_model)
    eager_step(prime_model, prime_optimizer, batches[0])
    prime_cache = OCCudaGraphTrainCache(
        prime_model, prime_optimizer, backend="bmm", max_entries=1, warmup=1
    )
    prime_cache.step(batches[0], **GRAPH_KWARGS)
    torch.cuda.synchronize()
    del prime_model, prime_optimizer, prime_cache
    torch.cuda.empty_cache()
    stable = [0, 5, 11]
    scenarios = {
        "stable3_10cycles": (stable * 10, 3),
        "stable3_30cycles": (stable * 30, 3),
        "ragged12_roundrobin_3cycles": (list(range(12)) * 3, 8),
        "ragged12_grouped_3repeats": (
            [index for index in range(12) for _ in range(3)], 8
        ),
    }
    return {
        "shape_table": [
            {"id": index, "atoms": atoms, "neighbors": neighbors,
             "edges": batches[index]["edges"]}
            for index, (atoms, neighbors) in enumerate(shapes)
        ],
        "scenarios": {
            name: trace_scenario(base, batches, order, cache_size)
            for name, (order, cache_size) in scenarios.items()
        },
    }


def large_system_matrix(base, warmup, repeats):
    rows = []
    for atoms, neighbors in ((864, 16), (2048, 16)):
        batch64 = pullback.ragged_case(
            atoms, neighbors, 6000 + atoms, "noncollinear", torch.float64
        )
        batch = {
            key: (value.float() if torch.is_tensor(value)
                  and value.is_floating_point() else value)
            for key, value in batch64.items()
        }
        reference = pullback.performance_case(
            base, batch, False, warmup, repeats
        )
        candidate = pullback.performance_case(
            base, batch, True, warmup, repeats
        )
        graph = pullback.cuda_graph_train_case(
            base, batch, warmup, repeats
        )
        rows.append({
            "atoms": atoms,
            "edges": batch["edges"],
            "reference": reference,
            "candidate": candidate,
            "graph": graph,
            "candidate_eager_speedup": (
                reference["train_step"]["median_ms"]
                / candidate["train_step"]["median_ms"]
            ),
            "graph_hot_speedup": (
                reference["train_step"]["median_ms"]
                / graph["train_step"]["median_ms"]
            ),
        })
        del batch64, batch
        torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cold", "large", "all"), default="all")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_candidate()
    torch.manual_seed(20260901)
    torch.cuda.manual_seed_all(20260901)
    base = NEPModel(pullback.config()).cuda().float().train()
    with torch.no_grad():
        for parameter in base.parameters():
            parameter.normal_(std=0.035)
    payload = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": "float32",
        "candidate": "analytic-bank-center-edge-anchor",
        "cold_shape": cold_shape_matrix(base) if args.mode in ("cold", "all") else None,
        "large_system": large_system_matrix(base, args.warmup, args.repeats)
        if args.mode in ("large", "all") else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
