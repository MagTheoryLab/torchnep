# Joint rank-4 training update

Enable in `nep.in` (TorchNEP only):

```text
model nep_spin
spin_version 1
joint_rank4 1
lambda_spin_response 0
```

`joint_rank4` defaults to `0`; older inputs retain Adam/AdamW unchanged.
It changes the optimizer, not the descriptor, ANN, model format or inference.
All configured orders train from initialization, without curriculum gates.
No fixed structural learning-rate multiplier is used.

Four random projections of the weighted E/F/virial/MF/torque residual Jacobian
span all trainable parameter blocks jointly. Their low-rank curvature damps
the proposed Adam displacement through a Woodbury solve. The damping is the
largest eigenvalue of the whitened sketch Gram matrix, bounding its inverse's
eigenvalues between 1/2 and 1. The Adam moments keep their ordinary updates.
The sketch and diagonal metric are reused for at most 24 accepted steps.
Changes to loss weights or the response contract invalidate reuse.
Independent per-rank Rademacher probes are SUM-reduced using global label
counts; padding contributes no observations. Sketch draws do not consume the
training RNG. Both single-device and sharded trainers use the same helper.

Current version-2 joint-update checkpoints include the sketch, its age, counters
and objective token. Earlier cache versions are rejected without migration. Exact
resume requires the same setting; use `finetune_from` when changing optimizers.
The extra VJPs use eager autograd even when the ordinary compute is compiled.
CUDA-graph optimizer capture is currently rejected rather than silently
approximated. Nonzero `lambda_spin_response` is supported for energy, mforce
and combined targets; see [response targets](RESPONSE_TARGETS.md).
The sketch includes the weighted response residuals. E/F/V/MF/torque masks and the
spin symmetric six-component virial convention are preserved.

Historical experiments used the removed spin3 descriptor and do not validate
the current c516 descriptor. Revalidate convergence and physical responses for
the current model before relying on an optimizer comparison.
