# Grouped response supervision (TorchNEP)

Put `response.xyz` beside `train.xyz`. Keep complete E/F/spin/MF labels;
virial is optional. Response observations are separate from the ordinary
training sampler. Put a frame in `train.xyz` too if its absolute labels
should contribute to the ordinary loss.

```text
lambda_spin_response 1
spin_response_target energy
spin_response_energy_scale 1e-6
joint_rank4 1
stage2 1
start_stage2 1
```

These are TorchNEP inputs, not new GPUMD/SNES keywords. The scale is eV/atom;
`1e-6` is 0.001 meV/atom, not a universal estimate of DFT noise. Set it from
the desired resolution and a defensible noise floor, not from an arbitrarily
small measured signal. Use float64 for very small energy differences.

`spin_response_target` selects:

* `mforce` (default): the existing mean-plus-centered rotation-generator
  Huber objective, normalized by each group's own uncentered target RMS
  (floor 1e-6). Each group needs at least three distinct coordinates,
  the same geometry and atom order, and varying spins.
* `energy`: the mean squared centered per-atom energy error divided by
  `spin_response_energy_scale**2`. Each group needs at least two frames,
  the same composition and atom order. Geometry can change: in that case
  it supervises the actual total energy difference, not a pure spin derivative.
* `both`: sum of the two group-balanced objectives. Rotation-group validity
  rules apply. The user must check E/MF consistency before choosing this mode;
  the code does not automatically declare DFT labels trustworthy.

Energy metadata example:

```text
response_probe=energy response_group=mae-parent-01 response_coordinate=0
```

For energy-only groups, the coordinate is an optional ordering label; it is
not used to generate derivative targets. Rotation uses the existing
`response_probe=rotation` and numeric path coordinate in radians.
No `spin_tangent` input is accepted.

Tangents use local sphere-log angular displacements and separate moment
magnitude differences. Resolved constant-rate great-circle rotations are
exact up to rounding; arbitrary paths still have finite-stencil error.
Antipodal pairs in nonzero stencils are ambiguous and rejected. A local
stencil touching zero uses Cartesian differences throughout, retaining
resolved longitudinal zero crossings (a refinement of GPUMD's pairwise
zero fallback). This v5 objective changes old training weights and tangents:
use weight-only fine tuning, not exact resume of a v4 response run. Group-local
normalization is not a noise filter; audit weak/noisy labels before use.

The energy residual is
`[(E_pred-E_ref)/N - mean_group((E_pred-E_ref)/N)] / scale`.
Groups, not individual frames or all pair combinations, have equal weight.
This removes a group-constant offset, not structural contributions that
vary within a group. It does not create missing short-wave or long-wave labels.

The total weight remains `lambda_spin_response`. Response supervision is
active in Stage 2, included in epoch evaluation and best-model selection.
Validation groups must be disjoint from training response groups and contain
compatible response scans. Target, scale and file content enter the exact
resume contract; use fine-tuning rather than exact resume to change them.

`joint_rank4` includes both ordinary and response residual Jacobians. Its
sketch uses signed square-root Huber residuals for the mforce branch, so its
squared norm matches the actual objective, including group/step weights.
DDP normalizes replicated response sketches to avoid multiplying curvature
by the number of ranks. It does not freeze descriptors or change scaler policy.

Response supervision alone is not a promise of reproducing L-BFGS refinement.
Ordinary rank4 training and frozen-descriptor L-BFGS use different update and
retention mechanisms. Compare their physical response scans and ordinary
quantity regression, not only their response loss.

## Production L-BFGS refinement

Use the usual `train_nep` entry with an existing matching **nep.txt**, float64,
and an empty output directory:

```python
from torchnep import train_nep
train_nep("nep.in", "train.xyz", finetune_from="initial_nep.txt",
          output_dir="refined", precision="float64", device="cuda",
          use_compile=False, restart=False)
```

Append to the matching architecture's `nep.in`:

```text
response_refine 1
response_refine_steps 2000
lambda_spin_response 1
spin_response_target energy
spin_response_energy_scale 1e-7
response_preserve_scale 5e-5 0.003 0.003
joint_rank4 0
```

The last scales are E in eV/atom, F in eV/Angstrom, and MF in eV/moment-unit.
They penalize drift from the starting model, not error against new labels.
These defaults reproduce the BFO reference protocol, not universal noise
tolerances. The parameter-change prior uses scale 0.1, one mean square per ANN
parameter tensor. `lambda_spin_response` weights only the response objective.

This explicitly selected final refinement replaces the epoch/Stage 2 Adam
schedule; it does not run automatically after ordinary training. Descriptors,
scaler, baseline and nonmagnetic element ANNs are fixed; only spin-active
element ANNs change. The model/head and inference format remain unchanged.
`joint_rank4 1` is rejected in this mode, not silently ignored. Ordinary
Adam/rank4 training continues to support all three response targets.

`train.xyz` supplies full energy retention and full E/F/MF/torque regression.
Optional `retention.xyz` beside it chooses the F/MF preservation subset with
the same full-label contract. Without it, at most 256 evenly spaced train
frames plus all response frames are used; record and inspect the selection.
Caches are rebuilt from the loaded model, including general variable atom
counts and element types. A perturbed-ANN live-autograd parity check is
mandatory. The derivative-cache budget is 2 GiB; choose a smaller explicit
retention set if it is exceeded. No old material-specific cache is loaded.

`refinement.json` records hashes, objective terms, timing, selected frames and
full before/after metrics. `nep_refined_candidate.txt` is always a candidate;
`nep_final.txt` is written only if each E/F/MF/torque RMSE increases by no more
than 1% and unweighted response loss is <=1 and no worse than initially.
An optional validation file is kept out of fitting and must also pass the
ordinary-quantity gate. The numerical gate is not a proof of the correct easy
axis, spiral ground state or extrapolation: inspect physical scans separately.

The initial implementation is single-process/device, float64, and fresh
refinement only. It rejects exact resume, random holdout splitting, scaler
recalibration, and nonempty output folders. `refinement.pt` records final
weights and provenance, not an exact-resume optimizer checkpoint. This is a
TorchNEP interface, not a new GPUMD/SNES input contract.
