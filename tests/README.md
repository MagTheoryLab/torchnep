# TorchNep tests

Pure pytest suite (`numpy` + `torch`; `ase` only for the ASE test).

```bash
pytest tests/                      # full suite
pytest tests/ -k float32           # one dtype
TEST_DEVICE=cpu pytest tests/      # restrict device (default: cpu + cuda if present)
```

| file | covers |
| --- | --- |
| `test_c516_format.py` | Ordinary/spin named blocks, parameter order, malformed metadata, energy-reference persistence, extended shapes up to 339 total coordinates, the 3 Å scalar/per-type spin cutoff contract, and 12 independent frozen spin oracle cases. |
| `test_checkpoint_format.py` | Current signed checkpoint resume/fine-tuning; rejection of unsigned/raw/prefixed state, missing metadata, and retired schedules. |
| `test_full_order_training.py` | All magnetic channels update from initialization; every curriculum spelling is rejected. |
| `test_magnetic_polynomial.py` | All 360 declared c516 C/L/order/SOC layouts, SOC/non-SOC symmetry, zero-spin/inactive-type behavior and first/second derivatives. |
| `test_cuda_harmonics.py` | L0–L4 spin and L1–L4 lattice polynomial values, VJP/JVP/HVP, zero/empty/noncontiguous inputs, CPU table expansion, and CUDA FP32/FP64 gradcheck/gradgradcheck. |
| `test_cached_spin_forces.py` | Cached E/F/raw9 V/MF/torque and every parameter gradient vs raw AD; all orders/SOC/typewise cutoff, cache invalidation, live parameter updates, inference, compiled changing shapes, energy finite differences, and all 360 truncations' symmetry and derivative covariance. |
| `test_gpumd_parity.py` | E / F / V / descriptor vs the GPUMD reference (incl. compressed CrCoNi frames where ZBL forces reach ~120 eV/Å); analytical vs autograd; train path vs predict path. |
| `test_descriptors.py` | Angular basis L=1..8; gradient checks; the six higher-body channels (q_222, q_1111, q_112, q_123, q_233, q_134) — GPUMD-polynomial match and rotational invariance; backend auto-resolution and loop/bmm/mulsum numerical equivalence. |
| `test_neighbor.py` | Cell-list vs brute-force neighbor search; tiled / auto-block paths. |
| `test_parsing.py` | Current c516 `l_max` input spellings and named-block model parsing. |
| `test_ase_calculator.py` | Optional ASE calculator (energy/forces/stress, ZBL split). |
| `test_b1_and_gpumd_qscaler.py` | Analytical `b1` offset (residual → 0, `nep_best` ≤ `nep_final`); `use_gpumd_qscaler` reproduces GPUMD's `c=1` q_scaler; `gpumd_init_parameters` re-inits coeffs **and** NN weights uniform(−1,1); weight_decay (AdamW) shrinks the weights. |
| `test_run_seed_and_valid.py` | `run_seed` reproducibility; `valid_file` / `valid_ratio` deterministic split, best-model selection on validation loss, `*_test.out`, split preserved across resume; `early_stop` fires on a plateau (validation-loss branch), is off by default, is per-stage (a stage-1 plateau jumps into Stage 2, surviving resume). `export_valid_split` reproduces the internal split (verbatim frames, matches `energy_test.out`). |
| `test_stream_mode.py` | `StreamDataStore` (the training data store): `collate` is bit-exact vs an independently assembled reference (concatenation + offsets + basis straight from the ops functions); metadata/mask consistency incl. missing channels; prefetched `iter_collated` matches direct collate; 2-rank DDP same-seed reproducibility. The DDP case is local-only: set `TORCHNEP_TEST_DDP=1` (skipped in CI — multi-process rendezvous is unreliable on shared runners). |
| `test_compiled_autograd.py` | `CompiledAutogradForce` (make_fx-materialized autograd forces): outputs and parameter gradients (second-order path through the force loss) match eager autograd across batch shapes on one dynamic graph; energy-only calls fall back to eager. CUDA-only — auto-skipped on CPU hosts/CI. |

Run the complete magnetic CUDA truncation matrix, including raw-coordinate AD
output/parameter-gradient comparisons, compiled boundary configurations and
symmetry checks. SOC requires joint lattice/axial-spin rotations; non-SOC also
requires independent lattice and spin rotations. Both check inversion and time
reversal, with forces, magnetic forces, virial and torque transformed accordingly:

```bash
TORCHNEP_TEST_SPIN_MATRIX=1 pytest -q -s tests/test_cached_spin_forces.py -k 'all_legal or boundary or symmetry'
```

**Tolerance vs GPUMD:** `rtol=1e-5, atol=2e-4`.

The ordinary model fixture uses c516 named blocks; its parameters/scalers and
existing numerical reference arrays are unchanged. See [fixture provenance](data/README.md).
Old text headers are tested only as rejected inputs.

**Re-baking the reference** (only if numerical model values or `CrCoNi.xyz` change):

```bash
GPUMD_NEP=/path/to/GPUMD/src/nep python tests/bake_fixtures.py
```
