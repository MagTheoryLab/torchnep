"""NSYS entrypoint for the hot O3-C2 full-train CUDA Graph replay."""

import copy
import os

import torch

from benchmark_cuda_graph_cache import (
    WEIGHTS,
    config,
    optimizer,
    training_batch,
)
from torchnep.model import NEPModel
from torchnep.oc_cuda_graph import OCCudaGraphTrainCache


os.environ["TORCHNEP_OC_CUDA_FULL"] = "0"
os.environ["TORCHNEP_OC_CUDA_TRAIN"] = "0"
os.environ["TORCHNEP_OC_CUDA_GRAPH"] = "1"
torch.manual_seed(19)
model = copy.deepcopy(NEPModel(config()).cuda().train())
opt = optimizer(model)
batch = training_batch(256, 16, 1256)
cache = OCCudaGraphTrainCache(model, opt, backend="bmm", max_entries=1, warmup=1)
kwargs = dict(
    weights=WEIGHTS, l1_coeff=0.0, l2_coeff=0.0,
    max_grad_norm=10.0, curriculum_scale=1.0,
    need_forces=True, need_virial=True, need_mforce=True, need_tau=True,
)
cache.step(batch, **kwargs)
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("hot_graph_256x4096")
for _ in range(20):
    cache.step(batch, **kwargs)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
