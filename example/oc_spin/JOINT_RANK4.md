# Joint rank-4 training update

Enable in `nep.in` (TorchNEP only):

```text
spin_mode 3
spin_curriculum 0
joint_rank4 1
lambda_spin_response 0
```

`joint_rank4` defaults to `0`; older inputs retain Adam/AdamW unchanged.
It changes the optimizer, not the descriptor, ANN, model format or inference.
`spin_curriculum` remains independent. Both settings are accepted; curriculum
is applied to the final corrected displacement, and a scale change invalidates
the cache. No fixed structural learning-rate multiplier is used.

Four random projections of the weighted E/F/virial/MF/torque residual Jacobian
span all trainable parameter blocks jointly. Their low-rank curvature damps
the proposed Adam displacement through a Woodbury solve. The damping is the
largest eigenvalue of the whitened sketch Gram matrix, bounding its inverse's
eigenvalues between 1/2 and 1. The Adam moments keep their ordinary updates.
The sketch and diagonal metric are reused for at most 24 accepted steps.
Stage-2 weights, curriculum scale and the O3-scaler phase invalidate reuse.
Independent per-rank Rademacher probes are SUM-reduced using global label
counts; padding contributes no observations. Sketch draws do not consume the
training RNG. Both single-device and sharded trainers use the same helper.

Checkpoints include the sketch, its age, counters and objective token. Exact
resume requires the same setting; use `finetune_from` when changing optimizers.
The extra VJPs use eager autograd even when the ordinary compute is compiled.
CUDA-graph optimizer capture and nonzero `lambda_spin_response` are currently
rejected rather than silently approximated. E/F/V/MF/torque masks and the
spin3 symmetric six-component virial convention are preserved.

The motivating matched 600-epoch BFO experiment reduced full-training MF RMSE
from 33.224 to 25.370 meV/μB and torque RMSE from 9.749 to 7.188 meV, with
observed wall time 51.15 versus 54.16 min on separate four-V100 allocations.
These are training-set results, not a generalization certificate or matched-node
speed benchmark. Easy-plane and short-wavelength spiral failures remained;
rank-4 training must not be presented as resolving those physical deficiencies.
