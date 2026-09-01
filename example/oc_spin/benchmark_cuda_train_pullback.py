"""Matched O3-C2 CUDA training-pullback correctness and timing gate.

This is intentionally a narrow research harness.  It compares the internal
CUDA moment-gate vertical slice with the unchanged Torch implementation on
identical ragged graphs, model parameters, losses, dtype and optimizer.
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


CANDIDATE_KIND = "narrow"
GRAPH_CORRECTNESS = False


def config():
    return {
        "num_types": 3,
        "type_names": ["Fe", "Ge", "Ni"],
        "cutoff_radial": 6.0,
        "cutoff_angular": 5.0,
        "n_max_radial": 4,
        "n_max_angular": 4,
        "basis_size_radial": 8,
        "basis_size_angular": 8,
        "l_max": [4, 2, 0],
        "neuron": 30,
        "spin_mode": 2,
        "spin_cutoff": [6.0, 6.0],
        "spin_l_max": [2, 0, 0],
        "spin_compress": 2,
        "spin_basis_size": [8, 0],
        "spin_order": 3,
        "spin_soc": 1,
        "spin_curriculum": False,
        "spin_dof_type": ["Fe", "Ge", "Ni"],
        "spin_env_type": ["Fe", "Ge", "Ni"],
    }


def ragged_case(n_atoms, neighbors, seed, spin_kind, dtype):
    """Flattened atoms/edges plus structure offsets; no padding or fixed N."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    counts = [n_atoms // 3, n_atoms // 3, n_atoms - 2 * (n_atoms // 3)]
    offsets = [0, counts[0], counts[0] + counts[1], n_atoms]
    pair_i_parts = []
    pair_j_parts = []
    for start, stop in zip(offsets[:-1], offsets[1:]):
        count = stop - start
        local_i = torch.arange(count).repeat_interleave(neighbors)
        shift = torch.arange(1, neighbors + 1).repeat(count)
        pair_i_parts.append(local_i + start)
        pair_j_parts.append((local_i + shift) % count + start)
    pair_i = torch.cat(pair_i_parts)
    pair_j = torch.cat(pair_j_parts)
    directions = torch.randn(pair_i.numel(), 3, generator=generator, dtype=dtype)
    directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    radii = 1.1 + 3.7 * torch.rand(
        pair_i.numel(), 1, generator=generator, dtype=dtype
    )
    rij = directions * radii
    atom_types = torch.arange(n_atoms, dtype=torch.long) % 3
    base = torch.randn(n_atoms, 3, generator=generator, dtype=dtype)
    if spin_kind == "zero":
        spins = torch.zeros_like(base)
    elif spin_kind == "collinear":
        spins = torch.zeros_like(base)
        spins[:, 2] = torch.linspace(0.2, 1.4, n_atoms, dtype=dtype)
    else:
        magnitude = torch.linspace(0.15, 1.55, n_atoms, dtype=dtype).unsqueeze(1)
        spins = base / torch.linalg.vector_norm(base, dim=1, keepdim=True) * magnitude
    struct_idx = torch.repeat_interleave(
        torch.arange(3, dtype=torch.long), torch.tensor(counts)
    )
    angular = torch.linalg.vector_norm(rij, dim=1) < 5.0
    return {
        "rij_rad": rij.cuda(),
        "rij_ang": rij[angular].cuda(),
        "pi_rad": pair_i.cuda(),
        "pj_rad": pair_j.cuda(),
        "pi_ang": pair_i[angular].cuda(),
        "pj_ang": pair_j[angular].cuda(),
        "atom_types": atom_types.cuda(),
        "N": n_atoms,
        "struct_idx": struct_idx.cuda(),
        "num_structures": 3,
        "spins": spins.cuda(),
        "rij_spin": rij.cuda(),
        "pi_spin": pair_i.cuda(),
        "pj_spin": pair_j.cuda(),
        "edges": pair_i.numel(),
    }


def set_cuda_slice(enabled):
    os.environ["TORCHNEP_OC_CUDA_TRAIN"] = (
        "1" if enabled and CANDIDATE_KIND == "narrow" else "0"
    )
    os.environ["TORCHNEP_OC_CUDA_FULL"] = (
        "1" if enabled and CANDIDATE_KIND == "full" else "0"
    )
    os.environ["TORCHNEP_OC_CUDA_BANK"] = (
        "1" if enabled and CANDIDATE_KIND in ("bank", "center") else "0"
    )
    os.environ["TORCHNEP_OC_CUDA_CENTER"] = (
        "1" if enabled and CANDIDATE_KIND == "center" else "0"
    )


def descriptors(model, batch, enabled):
    set_cuda_slice(enabled)
    return model.compute_descriptors(
        batch["rij_rad"], batch["rij_ang"], batch["pi_rad"], batch["pj_rad"],
        batch["pi_ang"], batch["pj_ang"], batch["atom_types"], batch["N"],
        spins=batch["spins"], rij_spin=batch["rij_spin"],
        pi_spin=batch["pi_spin"], pj_spin=batch["pj_spin"],
    )


def response(model, batch, enabled, create_graph=True):
    set_cuda_slice(enabled)
    backend = (
        "bmm" if os.environ.get("TORCHNEP_OC_CUDA_GRAPH", "0") == "1"
        else "loop"
    )
    return model.compute_properties(
        batch["rij_rad"], batch["rij_ang"], batch["pi_rad"], batch["pj_rad"],
        batch["pi_ang"], batch["pj_ang"], batch["atom_types"], batch["N"],
        batch["struct_idx"], batch["num_structures"], need_forces=True,
        need_virial=True, spins=batch["spins"], need_mforce=True,
        need_tau=True, rij_spin=batch["rij_spin"], pi_spin=batch["pi_spin"],
        pj_spin=batch["pj_spin"], create_graph=create_graph, backend=backend,
    )


def joint_loss(result, batch):
    virial = torch.zeros(
        batch["num_structures"], 9, dtype=result["virial"].dtype, device="cuda"
    )
    virial.index_add_(0, batch["struct_idx"], result["virial"])
    return (
        result["Etot"].square().mean()
        + 0.7 * result["forces"].square().mean()
        + 0.2 * virial.square().mean() / (batch["N"] ** 2)
        + 0.5 * result["mforces"].square().mean()
        + 0.3 * result["tau"].square().mean()
    )


def error_stats(candidate, reference):
    delta = (candidate - reference).abs()
    scale = reference.abs().clamp_min(1.0e-12)
    return {
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "max_rel": float((delta / scale).max().item()) if delta.numel() else 0.0,
    }


def named_gradients(model, loss, create_graph=False):
    named = [(name, value) for name, value in model.named_parameters()
             if value.requires_grad]
    values = torch.autograd.grad(
        loss, [value for _, value in named], create_graph=create_graph,
        allow_unused=True,
    )
    return {
        name: (torch.zeros_like(parameter) if value is None else value)
        for (name, parameter), value in zip(named, values)
    }


def compare_gradients(candidate, reference):
    rows = {name: error_stats(candidate[name], reference[name]) for name in candidate}
    worst_abs = max(rows, key=lambda name: rows[name]["max_abs"])
    worst_rel = max(rows, key=lambda name: rows[name]["max_rel"])
    return {
        "max_abs": rows[worst_abs]["max_abs"],
        "max_rel": rows[worst_rel]["max_rel"],
        "worst_abs_parameter": worst_abs,
        "worst_rel_parameter": worst_rel,
        "parameters": rows,
    }


def correctness_case(base, batch):
    os.environ["TORCHNEP_OC_CUDA_GRAPH"] = "0"
    reference = copy.deepcopy(base)
    candidate = copy.deepcopy(base)
    q_ref = descriptors(reference, batch, False)
    q_cuda = descriptors(candidate, batch, True)

    out_ref = response(reference, batch, False, create_graph=True)
    os.environ["TORCHNEP_OC_CUDA_GRAPH"] = "1" if GRAPH_CORRECTNESS else "0"
    out_cuda = response(candidate, batch, True, create_graph=True)
    outputs = {"descriptor": error_stats(q_cuda, q_ref)}
    for name in ("Etot", "forces", "virial", "mforces", "tau"):
        outputs[name] = error_stats(out_cuda[name], out_ref[name])

    energy_ref = named_gradients(reference, out_ref["Etot"].square().mean())
    energy_cuda = named_gradients(candidate, out_cuda["Etot"].square().mean())

    reference = copy.deepcopy(base)
    candidate = copy.deepcopy(base)
    os.environ["TORCHNEP_OC_CUDA_GRAPH"] = "0"
    loss_ref = joint_loss(response(reference, batch, False, True), batch)
    os.environ["TORCHNEP_OC_CUDA_GRAPH"] = "1" if GRAPH_CORRECTNESS else "0"
    loss_cuda = joint_loss(response(candidate, batch, True, True), batch)
    joint_ref = named_gradients(reference, loss_ref)
    joint_cuda = named_gradients(candidate, loss_cuda)
    result = {
        "atoms": batch["N"], "edges": batch["edges"], "outputs": outputs,
        "energy_gradient": compare_gradients(energy_cuda, energy_ref),
        "joint_gradient": compare_gradients(joint_cuda, joint_ref),
        "loss_abs": abs(float(loss_cuda.item() - loss_ref.item())),
    }
    os.environ["TORCHNEP_OC_CUDA_GRAPH"] = "0"
    return result


def directional_fd(base, batch, step=1.0e-5):
    model = copy.deepcopy(base)
    loss = joint_loss(response(model, batch, True, True), batch)
    gradients = named_gradients(model, loss)
    generator = torch.Generator(device="cuda").manual_seed(917)
    direction = {}
    norm2 = torch.zeros((), dtype=loss.dtype, device="cuda")
    for name, parameter in model.named_parameters():
        value = torch.randn(parameter.shape, generator=generator,
                            dtype=parameter.dtype, device=parameter.device)
        direction[name] = value
        norm2 += value.square().sum()
    norm = norm2.sqrt()
    for name in direction:
        direction[name] /= norm
    analytic = sum((gradients[name] * direction[name]).sum()
                   for name in gradients)

    values = []
    for sign in (1.0, -1.0):
        probe = copy.deepcopy(base)
        with torch.no_grad():
            for name, parameter in probe.named_parameters():
                parameter.add_(direction[name], alpha=sign * step)
        values.append(joint_loss(response(probe, batch, True, False), batch))
    finite = (values[0] - values[1]) / (2.0 * step)
    return {
        "analytic": float(analytic.item()), "finite_difference": float(finite.item()),
        **error_stats(analytic.reshape(1), finite.reshape(1)), "step": step,
    }


def median_mad(values):
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return {"median_ms": median, "mad_ms": mad}


def timed(callable_, warmup, repeats):
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        callable_()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1.0e3)
    return {**median_mad(values), "samples_ms": values}


def performance_case(base, batch, enabled, warmup, repeats):
    model = copy.deepcopy(base).float().train()
    batch = {key: (value.float() if torch.is_tensor(value) and value.is_floating_point()
                   else value) for key, value in batch.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)

    def descriptor_step():
        with torch.no_grad():
            descriptors(model, batch, enabled)

    def response_step():
        response(model, batch, enabled, True)

    def outer_step():
        optimizer.zero_grad(set_to_none=True)
        result = response(model, batch, enabled, True)
        loss = joint_loss(result, batch)
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1.0e3

    def optimizer_step_only():
        optimizer.step()

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        loss = joint_loss(response(model, batch, enabled, True), batch)
        loss.backward()
        optimizer.step()

    # Compile and allocate optimizer state before measured samples.
    for _ in range(warmup):
        train_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    descriptor_time = timed(descriptor_step, warmup, repeats)
    response_time = timed(response_step, warmup, repeats)
    outer_values = [outer_step() for _ in range(repeats)]
    optimizer_time = timed(optimizer_step_only, warmup, repeats)
    full_time = timed(train_step, warmup, repeats)
    return {
        "descriptor_forward": descriptor_time,
        "response": response_time,
        "outer_backward": median_mad(outer_values),
        "optimizer_step": optimizer_time,
        "train_step": full_time,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }


def cuda_graph_train_case(base, batch, warmup, repeats):
    """Capture one complete candidate train step for this ragged shape."""
    os.environ["TORCHNEP_OC_CUDA_GRAPH"] = "1"
    model = copy.deepcopy(base).float().train()
    source_batch = {
        key: (value.float() if torch.is_tensor(value) and value.is_floating_point()
              else value)
        for key, value in batch.items()
    }
    static_batch = {
        key: (value.detach().clone() if torch.is_tensor(value) else value)
        for key, value in source_batch.items()
    }
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0, capturable=True
    )

    def train_step():
        optimizer.zero_grad(set_to_none=False)
        loss = joint_loss(response(model, static_batch, True, True), static_batch)
        loss.backward()
        optimizer.step()

    # Materialize autograd metadata, gradients and Adam state before CUDA
    # stream capture. The narrow native-HVP candidate must not invoke Inductor;
    # this timer therefore exposes any remaining shape-dependent cold cost.
    cold_start = time.perf_counter()
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(max(warmup, 3)):
            train_step()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=False)
    with torch.cuda.graph(graph):
        optimizer.zero_grad(set_to_none=False)
        loss = joint_loss(response(model, static_batch, True, True), static_batch)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    cold_prepare_capture_ms = (time.perf_counter() - cold_start) * 1.0e3
    torch.cuda.reset_peak_memory_stats()
    copy_keys = [
        key for key, value in static_batch.items() if torch.is_tensor(value)
    ]

    def replay_with_input_copy():
        for key in copy_keys:
            static_batch[key].copy_(source_batch[key])
        graph.replay()

    replay = timed(replay_with_input_copy, warmup, repeats)
    result = {
        "train_step": replay,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "shape_cache_key": [batch["N"], batch["edges"]],
        "cold_prepare_capture_ms": cold_prepare_capture_ms,
    }
    os.environ["TORCHNEP_OC_CUDA_GRAPH"] = "0"
    return result


def cuda_graph_optimizer_parity(base, batch, replay_steps=3):
    """Compare nonzero-lr Adam updates between eager and captured steps."""
    os.environ["TORCHNEP_OC_CUDA_GRAPH"] = "1"
    captured = copy.deepcopy(base).float().train()
    captured_opt = torch.optim.Adam(
        captured.parameters(), lr=2.0e-4, capturable=True
    )
    values = {
        key: (value.float() if torch.is_tensor(value) and value.is_floating_point()
              else value)
        for key, value in batch.items()
    }
    static_values = {
        key: (value.detach().clone() if torch.is_tensor(value) else value)
        for key, value in values.items()
    }

    def step(model, optimizer, step_batch):
        optimizer.zero_grad(set_to_none=False)
        loss = joint_loss(response(model, step_batch, True, True), step_batch)
        loss.backward()
        optimizer.step()
        return loss

    # Materialize all state once, then fork model and optimizer from the exact
    # same checkpoint. Independent float32 scatter warmups are not bitwise
    # deterministic and would contaminate the graph-vs-eager comparison.
    for _ in range(3):
        step(captured, captured_opt, static_values)
    torch.cuda.synchronize()
    eager = copy.deepcopy(captured)
    eager_opt = torch.optim.Adam(
        eager.parameters(), lr=2.0e-4, capturable=True
    )
    eager_opt.load_state_dict(copy.deepcopy(captured_opt.state_dict()))

    graph = torch.cuda.CUDAGraph()
    before_capture = {
        name: parameter.detach().clone()
        for name, parameter in captured.named_parameters()
    }
    captured_opt.zero_grad(set_to_none=False)
    with torch.cuda.graph(graph):
        captured_opt.zero_grad(set_to_none=False)
        captured_loss = joint_loss(
            response(captured, static_values, True, True), static_values
        )
        captured_loss.backward()
        captured_opt.step()
    torch.cuda.synchronize()
    capture_delta = max(
        (parameter - before_capture[name]).abs().max().item()
        for name, parameter in captured.named_parameters()
    )
    capture_executed_update = capture_delta > 0.0
    eager_loss = None
    if capture_executed_update:
        eager_loss = step(eager, eager_opt, values)
    for _ in range(replay_steps):
        for key, value in static_values.items():
            if torch.is_tensor(value):
                value.copy_(values[key])
        graph.replay()
        eager_loss = step(eager, eager_opt, values)
    torch.cuda.synchronize()

    rows = {}
    for (name, expected), (_, actual) in zip(
        eager.named_parameters(), captured.named_parameters()
    ):
        rows[name] = error_stats(actual, expected)
    worst_abs = max(rows, key=lambda name: rows[name]["max_abs"])
    worst_rel = max(rows, key=lambda name: rows[name]["max_rel"])
    result = {
        "steps_after_warmup": replay_steps + int(capture_executed_update),
        "capture_parameter_delta": capture_delta,
        "capture_executed_update": capture_executed_update,
        "parameter_max_abs": rows[worst_abs]["max_abs"],
        "parameter_max_rel": rows[worst_rel]["max_rel"],
        "worst_abs_parameter": worst_abs,
        "worst_rel_parameter": worst_rel,
        "loss": error_stats(captured_loss, eager_loss),
    }
    os.environ["TORCHNEP_OC_CUDA_GRAPH"] = "0"
    return result


def main():
    global CANDIDATE_KIND, GRAPH_CORRECTNESS
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument(
        "--candidate",
        choices=("reference", "narrow", "full", "bank", "center"),
        default="narrow",
    )
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--performance-only", action="store_true")
    parser.add_argument("--graph-parity-only", action="store_true")
    parser.add_argument("--graph-correctness", action="store_true")
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    CANDIDATE_KIND = args.candidate
    GRAPH_CORRECTNESS = args.graph_correctness
    torch.manual_seed(20260901)
    torch.cuda.manual_seed_all(20260901)
    base64 = NEPModel(config()).cuda().double().train()
    assert (base64.struct_dim, base64.spin_dim, base64.dim) == (30, 49, 79)
    with torch.no_grad():
        for parameter in base64.parameters():
            parameter.normal_(std=0.035)

    correctness = []
    specifications = [
        (32, 11, 101, "zero", 1.0),
        (32, 13, 103, "collinear", 1.0),
        (32, 17, 107, "noncollinear", 0.92),
        (32, 17, 107, "noncollinear", 1.0),
        (108, 19, 109, "noncollinear", 1.08),
        (108, 19, 109, "noncollinear", 1.0),
        (256, 23, 113, "noncollinear", 1.0),
    ]
    batches64 = {}
    for atoms, neighbors, seed, spin_kind, geometry_scale in specifications:
        batch = ragged_case(atoms, neighbors, seed, spin_kind, torch.float64)
        if geometry_scale != 1.0:
            for key in ("rij_rad", "rij_ang", "rij_spin"):
                batch[key] = batch[key] * geometry_scale
        if not args.performance_only:
            correctness.append({
                "seed": seed, "spin_kind": spin_kind,
                "geometry_scale": geometry_scale,
                **correctness_case(base64, batch),
            })
        if spin_kind == "noncollinear" and geometry_scale == 1.0:
            batches64[atoms] = batch

    if args.performance_only:
        fd = None
    else:
        os.environ["TORCHNEP_OC_CUDA_GRAPH"] = (
            "1" if GRAPH_CORRECTNESS else "0"
        )
        fd = directional_fd(base64, batches64[32])
        os.environ["TORCHNEP_OC_CUDA_GRAPH"] = "0"
    base32 = copy.deepcopy(base64).float()
    performance = []
    performance_atoms = (
        () if args.graph_parity_only or args.correctness_only
        else (32, 108, 256)
    )
    for atoms in performance_atoms:
        batch = {key: (value.float() if torch.is_tensor(value)
                       and value.is_floating_point() else value)
                 for key, value in batches64[atoms].items()}
        ref = performance_case(base32, batch, False, args.warmup, args.repeats)
        cuda = performance_case(base32, batch, True, args.warmup, args.repeats)
        row = {
            "atoms": atoms, "edges": batch["edges"], "reference": ref,
            "cuda_slice": cuda,
            "train_speedup": ref["train_step"]["median_ms"]
                             / cuda["train_step"]["median_ms"],
        }
        if args.cuda_graph:
            graph = cuda_graph_train_case(
                base32, batch, args.warmup, args.repeats
            )
            row["cuda_graph"] = graph
            row["cuda_graph_speedup"] = (
                ref["train_step"]["median_ms"]
                / graph["train_step"]["median_ms"]
            )
        performance.append(row)

    graph_optimizer = (
        cuda_graph_optimizer_parity(base32, batches64[32])
        if args.cuda_graph else None
    )
    result = {
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "cuda": torch.version.cuda, "dtype_correctness": "float64",
        "candidate": CANDIDATE_KIND,
        "dtype_performance": "float32", "dimensions": [30, 49, 79],
        "correctness": correctness, "directional_fd": fd,
        "cuda_graph_optimizer_parity": graph_optimizer,
        "performance": performance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
