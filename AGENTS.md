# O/C spin production invariants

These rules apply to changes that touch the versioned `nep4_spin2` path.

- Keep `MagneticPolynomialDescriptor` independent. Do not make it inherit an
  older magnetic descriptor or reintroduce experimental descriptor registries.
- Preserve order nesting: O2-C2 is the exact first 37 magnetic channels of
  O3-C2; the O3-C2 production width is 49 magnetic channels.
- Keep one scalar, element-conditioned atomic-energy ANN. Forces, virial,
  magnetic forces, and torques must remain derivatives of that same energy.
- A zero-spin input must make every magnetic descriptor channel exactly zero.
- Keep the public truncation surface narrow: `spin_compress`, `spin_l_max`,
  `spin_order`, `spin_soc`, and `spin_cutoff`. Research block switches do not
  belong in `nep.in` or the serialized model format.
- Train the complete configured O/C representation from the first epoch.
  Do not reintroduce staged order gates, zeroed O3 ANN initialization, or
  curriculum-triggered scaler refresh. The retired `spin_curriculum 0`
  spelling is accepted only as a deprecated no-op for old inputs.
- `spin_mode 2` uses self-consistent descriptor scaling; do not enable the
  generation-0 GPUMD q-scaler for it.
- Stage 2, per-element LS conditioning, and grouped
  validation are training semantics, not inference/model-format switches.
- Controlled J-like, D-like, anisotropy, strain, and volume scans are a data
  and acceptance protocol. Do not hard-code their Hamiltonian names as model
  outputs or descriptor channels.
- Rotation-response training may supervise the mean plus centered shape of
  the magnetic-force generator curve only through generic
  group/coordinate/tangent metadata.
  Keep complete groups together, avoid energy second differences, and keep
  the response objective inside Stage 2 and best-model selection.
- Keep train/validation response groups disjoint and checkpoint data lineage
  content-addressed. A resumed run must not silently change its holdout.
- Any text or binary weight load must validate the complete semantic
  architecture, not only tensor shapes.
- Before proposing the branch, run the O/C unit tests, the nonmagnetic full
  regression suite, and the two-rank padding smoke. Do not commit generated
  models, logs, predictions, caches, or campaign outputs.
