"""PyTorch CUPTI fallback for O3-C2 response/backward/Graph attribution."""

from __future__ import annotations

from collections import defaultdict
import copy
import json
import os
from pathlib import Path

import torch

from benchmark_cuda_graph_cache import WEIGHTS, optimizer, training_batch
from benchmark_cuda_train_pullback import config
from torchnep.model import NEPModel
from torchnep.oc_cuda_graph import OCCudaGraphTrainCache, _joint_loss_metrics


os.environ["TORCHNEP_OC_CUDA_FULL"] = "0"
os.environ["TORCHNEP_OC_CUDA_TRAIN"] = "0"
os.environ["TORCHNEP_OC_CUDA_BANK"] = "1"
os.environ["TORCHNEP_OC_CUDA_CENTER"] = "1"
os.environ["TORCHNEP_OC_CUDA_GRAPH"] = "1"
torch.manual_seed(19)
batch = training_batch(256, 16, 1256)
weights = torch.tensor(WEIGHTS, device="cuda")


def loss_value(model):
    return _joint_loss_metrics(
        model, batch, weights, True, True, True, True, "bmm"
    )[0]


def event_device_us(event):
    for name in ("device_time_total", "cuda_time_total"):
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def summarize(profile):
    raw = defaultdict(lambda: [0, 0.0, 0.0])
    for event in profile.events():
        device_type = str(getattr(event, "device_type", ""))
        if "CUDA" not in device_type:
            continue
        duration = event_device_us(event)
        raw[event.name][0] += 1
        raw[event.name][1] += duration
        raw[event.name][2] = max(raw[event.name][2], duration)
    raw_rows = [
        {"name": name, "count": values[0],
         "total_us": values[1], "max_us": values[2]}
        for name, values in raw.items()
    ]
    raw_rows.sort(key=lambda row: row["total_us"], reverse=True)
    averages = []
    for event in profile.key_averages():
        device_total = float(getattr(
            event, "self_device_time_total",
            getattr(event, "self_cuda_time_total", 0.0),
        ))
        if device_total <= 0.0:
            continue
        averages.append({
            "name": event.key, "count": event.count,
            "self_device_total_us": device_total,
            "cpu_total_us": float(event.cpu_time_total),
        })
    averages.sort(
        key=lambda row: row["self_device_total_us"], reverse=True
    )
    return {"device_events": raw_rows, "operator_averages": averages}


def profile_call(call):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profile:
        call()
        torch.cuda.synchronize()
    return summarize(profile)


def eager_model():
    model = copy.deepcopy(NEPModel(config()).cuda().train())
    opt = optimizer(model)
    for _ in range(3):
        opt.zero_grad(set_to_none=False)
        loss = loss_value(model)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    return model, opt


def main():
    model, opt = eager_model()
    # Warm the clipping-only kernels too; the ordinary train-step warmup above
    # does not call clip_grad_norm_, and profiling their first lazy load would
    # contaminate the outer-backward CPU ledger.
    opt.zero_grad(set_to_none=False)
    warm_loss = loss_value(model)
    warm_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    torch.cuda.synchronize()

    def response_calls():
        for _ in range(5):
            loss_value(model)

    response = profile_call(response_calls)
    opt.zero_grad(set_to_none=False)
    loss = loss_value(model)
    torch.cuda.synchronize()
    backward = profile_call(
        lambda: (loss.backward(), torch.nn.utils.clip_grad_norm_(
            model.parameters(), 10.0
        ))
    )

    graph_model = copy.deepcopy(NEPModel(config()).cuda().train())
    graph_optimizer = optimizer(graph_model)
    cache = OCCudaGraphTrainCache(
        graph_model, graph_optimizer, backend="bmm", max_entries=1, warmup=1
    )
    kwargs = dict(
        weights=WEIGHTS, l1_coeff=0.0, l2_coeff=0.0,
        max_grad_norm=10.0, curriculum_scale=1.0,
        need_forces=True, need_virial=True, need_mforce=True, need_tau=True,
    )
    cache.step(batch, **kwargs)
    torch.cuda.synchronize()
    graph = profile_call(lambda: [cache.step(batch, **kwargs) for _ in range(10)])
    payload = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "shape": {"atoms": batch["N"], "edges": batch["edges"]},
        "response_5": response,
        "outer_backward_1": backward,
        "hot_graph_10": graph,
    }
    output = Path(os.environ["TORCHNEP_PROFILE_OUTPUT"])
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
