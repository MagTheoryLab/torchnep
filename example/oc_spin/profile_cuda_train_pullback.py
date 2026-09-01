"""Emit one NVTX-annotated matched joint-training step for NSYS."""

from __future__ import annotations

import argparse

import torch

from benchmark_cuda_train_pullback import (
    config,
    joint_loss,
    ragged_case,
    response,
)
from torchnep.model import NEPModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-slice", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(20260901)
    model = NEPModel(config()).cuda().float().train()
    batch = ragged_case(256, 23, 113, "noncollinear", torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)

    def step(mark=False):
        optimizer.zero_grad(set_to_none=True)
        if mark:
            torch.cuda.nvtx.range_push("response")
        result = response(model, batch, args.cuda_slice, True)
        if mark:
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("outer_backward")
        joint_loss(result, batch).backward()
        if mark:
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("optimizer_step")
        optimizer.step()
        if mark:
            torch.cuda.nvtx.range_pop()

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("full_train_step")
    step(mark=True)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
