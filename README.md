# TorchNEP

[![PyPI](https://img.shields.io/pypi/v/torchnep?logo=pypi&logoColor=white)](https://pypi.org/project/torchnep/)
[![Python](https://img.shields.io/pypi/pyversions/torchnep?logo=python&logoColor=white)](https://pypi.org/project/torchnep/)
[![Tests](https://github.com/mushroomfire/torchnep/actions/workflows/test.yml/badge.svg)](https://github.com/mushroomfire/torchnep/actions/workflows/test.yml)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue)](https://www.gnu.org/licenses/gpl-3.0)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Downloads](https://img.shields.io/pypi/dm/torchnep)](https://pypi.org/project/torchnep/)

A pure PyTorch implementation of the [NEP4](https://gpumd.org/theory/nep.html) (Neuroevolution Potential) training framework.

## ✨ Features

- 🔄 **GPUMD-compatible** — standard c516 exports load directly into GPUMD for MD simulation
- 🎯 **Two-stage training** — Stage 1: force-focused; Stage 2: energy-focused
- 🖥️ **Multi-GPU training** — distributed data parallel (DDP) on one node or multiple nodes
- ⚡ **Fast on NVIDIA and AMD** — `torch.compile` support with automatic backend selection, tuned on both CUDA and ROCm GPUs
- 💾 **Memory-friendly** — the dataset stays in host memory and batches are streamed to the GPU, so memory scales with batch size, not dataset size
- 🔧 **Fine-tuning** — load any `nep.txt` or `checkpoint.pt` to fine-tune; optionally slim the model to only the element types present in the new dataset
- 🛡️ **ZBL** — universal ZBL repulsive potential with optional typewise cutoffs
- 🧲 **Magnetic potentials** — the GPUMD c516 `model nep_spin / spin_version 1`
  tensor descriptor, with forces, virials, magnetic forces and torques derived
  from one scalar energy


---

## 📦 Installation

TorchNEP needs only `torch >= 2.0` and `numpy`, but neither is installed automatically — install the PyTorch build that matches your CUDA/CPU setup first (see the [official guide](https://pytorch.org/get-started/locally/); numpy comes with it).

Then install TorchNEP with:

```bash
pip install torchnep -U
```

or install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/mushroomfire/torchnep.git
```

or install from source code:

```bash
git clone https://github.com/mushroomfire/torchnep.git
cd torchnep
pip install .
```

---

## 📄 Training data (extended-XYZ)

TorchNEP reads extended-XYZ files. The parser is strict — the rules below are
enforced, and violations raise on load.

### Comment line tags

- `Lattice="ax ay az bx by bz cx cy cz"` — **mandatory**. Nine floats in Å
  giving the three lattice vectors as rows. Every frame is treated as fully
  periodic, so `pbc=...` is ignored. For isolated clusters/molecules or a
  non-periodic direction, use a vacuum box wider than the NEP cutoff.
- `energy=<value>`— optional, eV. System energy.
- `virial="vxx vxy vxz vyx vyy vyz vzx vzy vzz"` — optional, eV. Must
  have exactly 9 components. Positive values denote compressed states,
  negative denote stretched states (GPUMD convention).
- `stress="sxx sxy sxz syx syy syz szx szy szz"` — optional, eV/Å³.
  Must have exactly 9 components. Positive = stretched, negative =
  compressed — opposite sign to virial. If both `virial` and `stress` are
  present, `virial` wins.
- `response_group`, `response_probe`, `response_coordinate`, and
  `response_branch` — optional controlled-response metadata. Validation
  splitting keeps every `response_group` intact. For
  `response_probe=rotation`, three or more distinct coordinates, converged DFT
  spins, and magnetic-force labels activate the internal Stage-2
  generator-curve objective. The trainer derives the path tangent internally.

### Per-atom columns

The `Properties=...` schema declares column layout. TorchNEP recognizes:

- `species:S:1` — chemical symbol (case-sensitive; must match the
  `type` list in `nep.in`).
- `pos:R:3` — Cartesian position in Å.
- `force:R:3` or `forces:R:3` — reference force in eV/Å (optional).
- `spin:R:3`, `spins:R:3`, `moment:R:3`, or `moments:R:3` — atomic
  magnetic moment vector.
- `mforce:R:3`, `mforces:R:3`, or `magnetic_force:R:3` — magnetic
  force reference, using the convention \(m=-\partial E/\partial S\).
  Magnetic-force loss fits all three components, including the longitudinal
  component. For fixed-magnitude constrained-DFT fields, set `lambda_m 0`
  and `lambda_tau` to a positive value to train only on the derived torque
  \(\tau=S\times m\). For Stage 2, use `stage2_lambda_m 0` and a positive
  `stage2_lambda_tau`. Torque is derived, not predicted by a separate head.
- `spin_tangent` is not an input label. For `response_probe=rotation`, the
  trainer derives `dS/dtheta` from the converged DFT spins in each complete
  group. Together with magnetic
  forces, this supervises both the group mean and centered shape of
  `sum_i mforce_i dot dS_i/dtheta`; no explicit J or D label is required.

---

## ⚙️ Training Parameters

### Model architecture (GPUMD-compatible)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `type` | required | `N name1 name2 ...` — number and names of element types |
| `cutoff` | `8 4` | Radial and angular cutoff (Å) |
| `n_max` | `6 6` | Radial and angular expansion orders |
| `basis_size` | `6 6` | Chebyshev basis size per channel (radial / angular)|
| `l_max` | `4 1 0` | `L_3b q_222 q_1111 q_112 q_123 q_233 q_134` — max L of 3-body terms (1–8) plus up to six boolean flags (matching GPUMD) enabling each higher-body invariant|
| `neuron` | `30` | Neurons in the (single) hidden layer |
| `zbl` | — | ZBL outer cutoff (Å); enables short-range repulsion |
| `use_typewise_cutoff_zbl` | — | Scale ZBL cutoffs by covalent radii |
| `model` | `nep` | `nep` or `nep_spin` |
| `spin_version` | `1` | The sole supported magnetic descriptor/layout |
| `spin_order` | `2` | Maximum primitive spin polynomial degree (`1`–`4`) |
| `spin_compress` | `3` | Learned radial channels C (`1`–`9`) |
| `spin_l_max` | `2` | Maximum tensor rank (`0`–`4`) |
| `spin_cutoff` | radial cutoff | One value or one per element, each between 3 Å and the structural radial cutoff; pair cutoffs are arithmetic means |
| `spin_soc` | `1` | Joint lattice/spin rotations (`1`) or independent rotations (`0`) |
| `spin_type` | all types | Elements with independent spins; excluded types retain geometric environments |
| `spin_wall_high` | off | Longitudinal upper wall. Threshold in μB, stiffness in eV/μB³ |
| `spin_wall_low` | off | Longitudinal lower wall. Threshold in μB, stiffness in eV/μB |

Each wall accepts one `threshold stiffness` pair for all spin-active elements
or one pair per element in the `type` order. Zero stiffness disables that wall.
The training input also accepts `spin_wall_high auto 0.5 2.0`. At the start of a
run, the trainer sets each spin-active element's threshold to its maximum
training moment plus 0.5 μB. A low wall with `auto` subtracts the margin from
the training minimum and is disabled when the result is nonpositive.
The calculation uses the training split, including all training shards in DDP.
Exported thresholds are numeric.
Exports with an enabled wall carry TorchNEP metadata that current GPUMD c516
does not read. Existing c516 files without wall metadata load with both walls off.

Select `model nep_spin` and `spin_version 1`. The C3/L2/O2 default has 34
magnetic coordinates with SOC, 31 without SOC.
All 360 combinations of C=1–9, L=0–4, order=1–4 and SOC=0/1 are supported,
with at most 236 magnetic coordinates (C9/L4/O4 without SOC). Current c516
supports up to 339 total coordinates, including its maximum 103 structural
coordinates. Training inputs and model files both enforce the spin cutoff
range above; models with magnetic cutoffs below 3 Å are rejected.

Angular structural coefficients are reused at the spin cutoff; `spin_basis_size` follows the angular basis and
may only repeat that value. All configured primitives train from the first
epoch. Zero spins retain degree-zero geometric channels. Magnetic scaling is
`1/max(RMS, range, 1)`, without centering.

Truncation keeps complete tensor representations. SOC preserves joint lattice
and axial-spin rotations; non-SOC also preserves independent lattice and spin
rotations. Both preserve spatial inversion and global spin time reversal.
At L=0, spin dependence reduces to magnitudes; with order=1 the ANN has no spin
dependence.

Ordinary and spin exports use the GPUMD c516 named-block format:

```text
nep <number of types> <elements...>
format_version 1
model nep                 # or nep_spin, with spin_version 1
...
@ parameters <count>
...
@ q_scaler <dimension>
...
```

Parameters are per-type ANN weights, shared bias, pair-first structural radial
and angular coefficients, then (spin only) radial mix, path weights and center
mix. Only the c516 named-block format is accepted. Older `nep4[_zbl]` and
`nep4_spin2/3` text files are rejected. Experimental
`spin_mode 2/3`, split DOF/environment masks, their weights and specialized CUDA
experiments have been removed; those spin models require retraining.
See [the magnetic example and response-data contract](example/oc_spin/README.md)
and [the complete input](example/oc_spin/nep.in). TorchNEP keeps its PyTorch
training optimizer; matching the descriptor does not reproduce GPUMD SNES runs.

Spin training caches input-only basis values and Cartesian derivatives in each
batch, then projects the scalar ANN's derivatives analytically to forces,
virial, magnetic forces and torque. All descriptor projections and ANN weights
remain trainable. Input changes invalidate the batch cache; caches are never
serialized. GPU training compiles the complete cached core by default, just as
ordinary NEP does. Compilation adds startup time, so short runs may be slower;
`use_compile=False` selects the eager cached path, while
`use_autograd_forces=True` retains raw-coordinate autograd as a reference.

The shared angular-coefficient contraction uses one-hot matmul to avoid
contended gradient atomics at low element counts. On CUDA, lattice and spin
harmonics through L4 use fused polynomial evaluation and analytic derivatives.
CPU, higher lattice orders and compiled force tracing retain the tensor reference.

### Training hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epoch` | `600` | Total training epochs |
| `batch` | `32` | Structures per gradient step |
| `lr` | `0.01` | Initial learning rate |
| `stop_lr` | `1e-6` | Minimum learning rate (scheduler floor) |
| `lambda_e` | `0.01` | Energy loss weight |
| `lambda_f` | `1.0` | Force loss weight |
| `lambda_v` | `0.01` | Virial loss weight |
| `lambda_m` | `1.0` | Magnetic-force loss weight |
| `lambda_tau` | `0.0` | Extra \(|S|^2\)-weighted transverse magnetic-force loss; spin torque RMSE uses its two intrinsic components |
| `max_grad_norm` | `10.0` | Gradient clipping threshold |
| `lr_scheduler` | `plateau` | LR schedule — `plateau` (ReduceLROnPlateau) or `step` (StepLR). Stage 1 and Stage 2 share this mode |
| `scheduler_patience` | `15` | For `plateau`: epochs without improvement before LR reduction. For `step`: epoch interval between LR reductions |
| `early_stop` | `0` | Stop if the monitored loss (validation loss when a validation set is used, else training loss) hasn't improved for N epochs (`0` = off). Per-stage: a stage-1 plateau jumps into Stage 2 instead of ending the run. Use a value larger than `scheduler_patience` |
| `scheduler_factor` | `0.7` | LR reduction factor — multiplied on each decay in both modes |
| `stage2` | `0` | Enable Stage 2 (`1` = on) |
| `start_stage2` | 50 % of epochs | Epoch to switch to Stage 2 |
| `stage2_lr` | `1e-3` | Stage 2 learning rate |
| `stage2_scheduler_patience` | `scheduler_patience` | Stage 2 scheduler patience (overrides Stage 1's; same semantics) |
| `stage2_scheduler_factor` | `scheduler_factor` | Stage 2 LR decay factor (overrides Stage 1's)|
| `stage2_lambda_e` | `1.0` | Stage 2 energy weight |
| `stage2_lambda_f` | `0.05` | Stage 2 force weight |
| `stage2_lambda_v` | `0.1` | Stage 2 virial weight |
| `stage2_lambda_m` | `lambda_m` | Stage 2 magnetic-force weight |
| `stage2_lambda_tau` | `lambda_tau` | Stage 2 torque weight |
| `weight_decay` | `1e-4` | AdamW decoupled weight decay on all trainable parameters (`b1` is solved analytically and never decays). `0` falls back to plain Adam. Unsupported keys (`lambda_1`, `lambda_2`, `pos_noise`) are rejected |
| `joint_rank4` | `0` | Optional joint rank-4 preconditioning of the Adam displacement for `model nep_spin`; refresh at most every 24 steps. Supports E/F/V/MF/torque and grouped response losses |
| `response_refine` | `0` | Opt-in frozen-descriptor L-BFGS final refinement through `train_nep(..., finetune_from=...)`; float64, single device, `joint_rank4 0`. See [response refinement](example/oc_spin/RESPONSE_TARGETS.md). |
| `response_refine_steps` | `2000` | Maximum L-BFGS iterations (line searches can evaluate the objective multiple times). |
| `response_preserve_scale` | `5e-5 0.003 0.003` | Prediction-drift scales for E/atom, F, MF in response refinement; not DFT error estimates. |
| `spin_response_target` | `mforce` | Response objective: `energy` (centered per-atom energy differences), `mforce` (rotation generator), or `both`. Enabled by `lambda_spin_response`; see [input and units](example/oc_spin/RESPONSE_TARGETS.md). |
| `spin_response_energy_scale` | `1e-6` | Positive energy-response residual scale in eV/atom; choose from target resolution and label noise. Applies to `energy`/`both`. |

### Runtime arguments

Everything that is not about hyperparameter *values* lives on the Python
function (`train_nep` / `train_nep_sharded`):

| Argument | Default | What it controls |
|---|---|---|
| `device` | auto | `"cuda"` / `"xpu"` / `"mps"` / `"cpu"`; any other stream-based PyTorch accelerator should also work if passed explicitly |
| `precision` | `"float32"` | dtype for training + store, `"float32"` or `"float64"` |
| `backend` | `"auto"` | `"loop"`, `"bmm"`, or `"auto"` |
| `use_autograd_forces` | `False` | autograd-through-rij; O/C spin models select the full autograd E/F/V/MF/Tau path automatically |
| `use_swa` | `False` | maintain SWA-averaged model and save `nep_average.txt` |
| `swa_start` | last 100 epochs | first epoch included in the SWA average |
| `use_compile` | `None` | auto: compile on GPU, eager on CPU; `True`/`False` force it. Missing Triton/C++ toolchain degrades to eager with a log note |
| `print_interval` | `1` | log to screen every N epochs |
| `checkpoint_interval` | `100` | save `checkpoint.pt` every N epochs |
| `prediction_interval` | `100` | every N epochs run predict with the current-epoch weights and overwrite `{energy,force,virial}_train.out` |
| `restart` | `True` | resume from `checkpoint.pt` if present |
| `finetune_from` | `None` | load weights from a `.pt` or `nep.txt` and start a NEW training from them |
| `resume_from` | `None` | path to a checkpoint to CONTINUE from (e.g. `checkpoint_stage1.pt` to redo Stage 2); takes precedence over the automatic `checkpoint.pt` pickup |
| `recompute_q_scaler` | `False` | only with `finetune_from`: recompute the descriptor scaler on the new data instead of keeping the source model's |
| `slim_types` | `False` | drop element types absent from the dataset |
| `energy_key` | `"energy"` | comment-line tag read as reference energy (e.g. `"atomization_energy"`) |
| `use_gpumd_qscaler` | `None` | Descriptor-aware default. O/C models use self-consistent scaling (`False`); non-spin GPUMD-compatible models retain the generation-0 convention |
| `run_seed` | `None` | master RNG seed. `None` = random each run; an int makes the run reproducible (weight init + batch shuffle). Saved in `checkpoint.pt`, restored on resume |
| `valid_file` | `None` | validation `.xyz`, `nep_best` and the plateau LR schedule follow the validation loss; writes GPUMD-style `*_test.out` |
| `valid_ratio` | `None` | hold out this fraction (e.g. `0.1`) of `data_file` as the validation set; the split is drawn from `run_seed` and preserved on resume. Mutually exclusive with `valid_file` |
| `valid_strategy` | `"stratified"` | `"random"` or `"stratified"` (split within (element combination × cell-size) groups; tiny cells ≤4 atoms and groups < 20 frames stay in training; auto-falls back to `"random"` if the validation set would be starved) |

---

## 📊 Output Files

| File | Contents |
|------|----------|
| `nep_best.txt` | Best model |
| `nep_final.txt`    | Model at the last epoch (used for the end-of-training predict) |
| `nep_average.txt` | SWA-averaged model (only with `use_swa=True`) |
| `checkpoint.pt`    | Full training state |
| `checkpoint_stage1.pt` | Full end-of-Stage-1 checkpoint |
| `output.log`       | Full console log |
| `loss.out`         | Per-epoch: epoch, loss, RMSE_E/F/V/stress and, for spin models, RMSE_MF/Tau; a validation set appends the matching test columns and makes validation loss the monitored loss |
| `energy_train.out` | Per-frame predicted vs reference E/atom (eV/atom) |
| `force_train.out` | Per-atom predicted vs reference Fx Fy Fz (eV/Å) |
| `virial_train.out` | Per-frame predicted vs reference virial xx yy zz xy yz zx (eV/atom) |
| `stress_train.out` | Per-frame predicted vs reference stress (GPa) |
| `mforce_train.out` | Spin models: per-atom predicted vs reference magnetic force |
| `tau_train.out` | Spin models: per-atom predicted vs reference spin torque |
| `*_test.out` | Matching E/F/V/stress and, for spin models, MF/Tau files for the validation set; only written with `valid_file`/`valid_ratio` |

---

## 🚀 Launch training

### Single GPU / CPU / MPS — `train_nep`

```python
# run_train.py
from torchnep import train_nep
train_nep("nep.in", "train.xyz", output_dir="output")

# with a validation set (either form):
train_nep("nep.in", "train.xyz", output_dir="output", valid_file="valid.xyz")
train_nep("nep.in", "train.xyz", output_dir="output", valid_ratio=0.1)

# export the exact valid_ratio split as GPUMD-ready files — train the same
# partition in GPUMD (or anything else) and compare loss curves directly:
from torchnep import export_valid_split
export_valid_split("train.xyz", valid_ratio=0.1, run_seed=42,
                   output_dir="split")   # writes split/train.xyz + split/test.xyz
# coverage-aware split (same strategy as valid_strategy="stratified"):
export_valid_split("train.xyz", valid_ratio=0.1, run_seed=42,
                   output_dir="split", strategy="stratified")
```

```bash
python run_train.py
```

Ordinary NEP and NEP-Spin fit `energy_reference` from the training-set
compositions and energies per atom. Optional extended XYZ tags `weight` and
`energy_weight` multiply the energy fitting objective by
`weight² × energy_weight`. The shared `b1` is fitted with `energy_weight`.
Training uses residual energies, and predictions restore the per-element
reference. Resume inherits the checkpoint reference. Ordinary fine-tuning
fits the target reference and removes the source network's weighted mean
energy per element from it. Spin fine-tuning inherits the source reference.

### Multi-GPU, single node — `train_nep_sharded`

Each rank keeps only `1/N` of the preprocessed structures in its GPU data
store, so device data memory scales as `1/N`. The current implementation still
parses the complete XYZ on every rank before releasing non-local host frames;
peak host memory is therefore not yet `1/N`.

```python
# run_train.py
from torchnep import train_nep_sharded
train_nep_sharded("nep.in", "train.xyz", output_dir="output")
```

```bash
torchrun --standalone --nproc_per_node=4 run_train.py    # 4 GPUs on this node
```

### Multi-GPU, multi-node (SLURM) — `train_nep_sharded`

For M nodes × N GPUs each, the key SLURM directives are:

```bash
#SBATCH --nodes=2                  # M nodes
#SBATCH --ntasks-per-node=1        # 1 srun task per node; torchrun fans out to all GPUs
#SBATCH --gpus-per-node=4          # N GPUs per node
#SBATCH --cpus-per-task=16         # CPU cores per node

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

srun --nodes=$SLURM_NNODES --ntasks-per-node=1 bash -c "
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=\$SLURM_GPUS_ON_NODE \
    --node_rank=\$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    run_train.py
"
```

---

## 🔁 Restart and Resume

Two ways to resume:

```python
# 1) automatic: looks for checkpoint.pt in output_dir (restart=True default)
train_nep("nep.in", "train.xyz", output_dir="output")

# 2) explicit: continue from a specific checkpoint
train_nep("nep.in", "train.xyz", output_dir="output",
          resume_from="output/checkpoint_stage1.pt")
```

Only current TorchNEP checkpoints can be resumed. Architecture signature,
energy references, saved seed, scheduler state, content-addressed data lineage
and training schedule are required for ordinary and spin models alike.
Unsigned/raw state dictionaries, old DDP-prefixed state, missing-reference
checkpoints and curriculum-era schedules/caches are rejected. No default
values are injected or checkpoint versions migrated. These `.pt` files store
TorchNEP training state; they are not GPUMD SNES restart files.
`spin_curriculum` is rejected for every value, including `0`.

### What you can safely change on restart

| Parameter | Safe to change? | Notes |
|-----------|----------------|-------|
| `epoch` | Conditional | Extending a run is supported. Specify `start_stage2` explicitly when extending a two-stage run so its automatic boundary does not move. |
| `lambda_e` / `lambda_f` / `lambda_v` | Yes | New weights take effect next epoch. |
| `stage2_lambda_e` / `stage2_lambda_f` / `stage2_lambda_v` | Yes | Same auto-reset rule. |
| `batch` | Yes | — |
| `stage2`, `start_stage2` | Conditional | Before Stage 2, a future boundary may be added. Once a checkpoint is in Stage 2, both settings must remain unchanged. To redo Stage 2, resume from `checkpoint_stage1.pt` |
| `stage2_lr` | Only at the transition | Applied **once**, when training first crosses Stage 1 → Stage 2. If you resume from a checkpoint that was *already* in Stage 2, the checkpoint's current (possibly-decayed) LR is kept — editing `stage2_lr` then has no effect. To re-enter Stage 2 with a new LR, `resume_from=".../checkpoint_stage1.pt"`. |
| `lr_scheduler` (`plateau` ↔ `step`) | Yes | Scheduler state from the old mode is incompatible and silently discarded; the new scheduler starts fresh from the current LR |
| `scheduler_patience` / `scheduler_factor` | Yes | Applied immediately |
| `stage2_scheduler_patience` / `stage2_scheduler_factor` | Yes | Applied immediately to the Stage 2 scheduler |
| `lr` (Stage 1) | **No** | Resume keeps the checkpoint's LR |
| `run_seed` | **No** (ignored) | The checkpoint's saved seed wins on resume — keeps the shuffle stream and the `valid_ratio` split unchanged |
| `valid_file` / `valid_ratio` / data contents | **No** | Checkpoints store SHA-256 data lineage; resume fails closed if training/validation content or split settings change. Start a new run or fine-tune instead |
| Architecture (`neuron`, `cutoff`, `n_max`, `basis_size`, `l_max`, `type`) | **No** | Dimensions are fixed in the saved weights |

For a bitwise-comparable DDP continuation, also keep world size, per-rank
batch size, precision, and software/runtime versions unchanged. The checkpoint
guards scientific model/data/schedule semantics; it cannot make a changed
distributed reduction order numerically identical.

---

## 🔧 Fine-Tuning

Fine-tuning starts from a pre-trained model's weights instead of random initialisation. The architecture (`nep.in` parameters) must match the source model, but the new dataset's element types may be a subset of the original.

### Basic fine-tuning

```python
train_nep(
    "nep.in",
    "new_data.xyz",
    output_dir="finetune_output",
    finetune_from="pretrained/nep.txt",   # or a "pretrained/checkpoint.pt"
    slim_types=True,
)
```

`finetune_from` accepts:
- `nep.txt` — GPUMD c516 `format_version 1`, `model nep` / `model nep_spin`
  files, with semantic architecture validation
- `checkpoint.pt` — current TorchNEP checkpoint with `model_state` and an exact
  `torchnep-c516-v1` architecture signature; weights are extracted automatically

If the new dataset contains fewer element types than the original model, `slim_types=True` removes the unused types **before training begins**, shrinking the model and speeding up training.

### Standalone model slimming

```python
from torchnep.model import NEPModel, slim_model
from torchnep.data import parse_nep_in

config = parse_nep_in("nep.in")
model = NEPModel(config)
model.load_weights_from_nep_txt("nep.txt")

slimmed = slim_model(model, ["Cr", "Ni"])
slimmed.save_nep_txt("nep_slim.txt", max_NN_radial, max_NN_angular)
```

---

## 🔮 Prediction

### Single-structure prediction

```python
from torchnep.nep import NEPCalculator
import numpy as np

calc = NEPCalculator("nep.txt")
result = calc.compute(
    species=["Cr", "Cr", "Ni"],
    positions=np.array([[0,0,0],[1.5,0,0],[3,0,0]]),
    cell=np.eye(3) * 6.0,
)
print(result["energy"])         # (N,) per-atom energy (eV); sum for total
print(result["forces"])         # (N, 3) forces (eV/Å)
print(result["virial"])         # (N, 9) per-atom virial (eV)

# Split the NEP (neural-network) part from the ZBL repulsive part:
result = calc.compute(..., return_components=True)
print(result["energy_nep"], result["energy_zbl"])   # sum == result["energy"]
```

For an O/C model, pass the per-atom moment vectors and read the conservative
magnetic derivatives from the same scalar energy:

```python
spins = np.array([[0, 0, 2.2], [0.3, 0, 2.1], [0, 0.2, 0.1]])
result = calc.compute(
    species=["Cr", "Cr", "Ni"],
    positions=np.array([[0,0,0],[1.5,0,0],[3,0,0]]),
    cell=np.eye(3) * 6.0,
    spins=spins,
)
print(result["mforces"])  # -dE/dS
print(result["tau"])      # S x mforce
```

To add a wall at inference to an existing c516 spin potential, use
`NEPCalculator("nep.txt", spin_wall_high=(6.12, 0.45))`. The ASE `NEP`
calculator accepts the same keywords. An omitted keyword uses the value in
the file, and `(0, 0)` disables that wall. The override applies to that
calculator instance without rewriting the model file.

### ASE calculator

If ASE is installed, any ASE workflow (relaxation, MD, EOS, …) can drive a trained model:

```python
from ase.io import read
from torchnep.ase_calculator import NEP

atoms = read("POSCAR")
atoms.calc = NEP("nep.txt", dtype='float32', device='cuda')
print(atoms.get_potential_energy())   # eV
print(atoms.get_forces())             # (N, 3) eV/Å
print(atoms.get_stress())             # Voigt 6-vector eV/Å³ (periodic cells)

# NEP / ZBL / total breakdown of energy, forces, and stress:
parts = atoms.calc.get_components()
print(parts["nep"]["energy"], parts["zbl"]["energy"], parts["total"]["energy"])
```

For `model nep_spin`, attach an `(N,3)` `spins` array before evaluation;
`atoms.calc.results["mforces"]` and `atoms.calc.results["tau"]` are then
available. Optional ZBL contributes no magnetic derivative. Export uses the
same `model nep_spin` block format with an optional `zbl` field.

### Full-dataset prediction

Runs streamed, batched inference on an entire `.xyz` file and writes GPUMD-compatible output files.
The file is indexed once, then processed chunk by chunk (~`chunk_atoms` atoms each: read → neighbor
lists → device-sized batches → rows appended to the outputs), so host memory is bounded by the chunk and
device memory by the batch — a dataset of any size finishes on any machine, only the wall time differs.
A progress bar (tqdm if installed) shows frames done.

```python
from torchnep import predict_dataset

predict_dataset(
    "nep.txt",
    "test.xyz",
    output_dir="results",
    output_descriptor=0,   # 0=off, 1=per-frame mean, 2=per-atom (matches GPUMD)
    batch_size=None,       # auto from free GPU memory (OOM-halving retry); or an int
    chunk_atoms=None,      # atoms per streamed chunk, default 200000 (env TORCHNEP_PREDICT_CHUNK_ATOMS)
)
# writes energy/force/virial/stress_train.out; spin models also write
# mforce/tau_train.out; descriptor.out is optional
```

### Multi-GPU / multi-node prediction

`predict_dataset_sharded` has the same arguments and writes the same files, but is launched with one
process per GPU (`torchrun --nproc_per_node=8 script.py`, or `srun` under Slurm).


```python
from torchnep import predict_dataset_sharded
predict_dataset_sharded("nep.txt", "huge.xyz", output_dir="results")
```

---

## 🗂️ Source layout

The `torchnep/` package is organised as follows:

| File | Role |
|------|------|
| `__init__.py` | Public API — re-exports the entry points `train_nep`, `train_nep_sharded`, `predict_dataset`, `predict_dataset_sharded` |
| `data.py` | I/O and parsing — reads extended-XYZ frames and `nep.in`, plus the NumPy brute-force neighbor builder used for training |
| `neighbor.py` | PyTorch linked-cell (cell-list) neighbor search, O(N) for the large structures of an ASE-driven MD run |
| `model.py` | Trainable NEP4 model (`NEPModel`) as an `nn.Module`, per-type fitting nets, ZBL, and `slim_model` |
| `ops.py` | Core differentiable kernels — Chebyshev/angular basis, descriptors, ANN evaluation, ZBL; pure-PyTorch `loop`/`bmm`/`mulsum` backends |
| `nep.py` | `NEPCalculator` — loads a `nep.txt` and computes energy/forces/virial/descriptors for single structures |
| `predict.py` | Streamed, batched full-dataset inference (`predict_dataset`), writing GPUMD-compatible `*_train.out` files |
| `train.py` | Single-GPU/CPU training (`train_nep`): host-resident streaming data store (`StreamDataStore` + prefetching `iter_collated`), two-stage loop, schedulers, checkpoint/restart, periodic predict |
| `train_sharded.py` | Data-sharded multi-GPU/multi-node training (`train_nep_sharded`) via DDP |
| `compiled_autograd.py` | `torch.compile` for the autograd force path: the first-order dE/drij gradient is materialized into the graph with `make_fx`, so `use_autograd_forces=True` + `use_compile=True` runs one fused dynamic-shape graph instead of an uncompilable double backward |
| `ase_calculator.py` | ASE `Calculator` wrapper (`NEP`) for relaxation, MD, EOS, phonons, … |
| `constants.py` | Shared constants — element table, covalent radii, NEP polynomial coefficients |

---

## 📚 Citation

If you use TorchNEP in your research, please cite the following paper:


```bibtex
@misc{wu2026torchne,
      title={TorchNEP: Ultra-Efficient and Accurate Training of Neuroevolution Potentials}, 
      author={Yong-Chao Wu and Xiaoya Chang and Tero Mäkinen and Amin Esfandiarpour and Jian-Li Shao and Tapio Ala-Nissila and Zheyong Fan and Mikko Alava},
      year={2026},
      eprint={2606.19557},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2606.19557}, 
}
```
