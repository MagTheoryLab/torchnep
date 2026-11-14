# GPUMD c516 spin production invariants

These rules apply to `model nep_spin / spin_version 1`, synchronized with GPUMD c516.

- Keep `MagneticPolynomialDescriptor` independent. Do not make it inherit an
  older magnetic descriptor or reintroduce experimental descriptor registries.
- Match `spin_layout.cuh` primitive, sector and parameter ordering. Default
  C3/L2/O2 has 34 SOC or 31 non-SOC magnetic coordinates; orders are not prefixes.
- Keep one scalar, element-conditioned atomic-energy ANN. Forces, virial,
  magnetic forces, and torques must remain derivatives of that same energy.
- Match GPUMD at zero spin: degree-zero geometric channels remain; all spin
  derivatives must be finite. Excluded `spin_type` inputs have zero effective spin.
- Keep the public truncation surface narrow: `spin_compress`, `spin_l_max`,
  `spin_order`, `spin_soc`, and `spin_cutoff`. Research block switches do not
  belong in `nep.in` or the serialized model format.
- Train the complete configured O/C representation from the first epoch.
  Do not reintroduce staged order gates, zeroed O3 ANN initialization, or
  curriculum-triggered scaler refresh. Reject every `spin_curriculum` input,
  including disabled spellings and disabled config dictionaries.
- Spin scaling is `1/max(RMS, range, 1)`, without centering. Reuse structural
  angular coefficients; do not add independent magnetic radial coefficients.
- Read and write ordinary and magnetic models only in the c516 named-block
  format. Do not add old-text readers, unsigned checkpoint fallbacks, default
  state injection, DDP-prefix rewriting, or retired schedule/cache migrations.
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
