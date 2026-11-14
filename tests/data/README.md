# Ordinary NEP regression fixture

`nep_CrCoNi.txt` uses the GPUMD c516 `nep / format_version 1 / model nep`
named-block format, including typewise ZBL. The fixture's numeric model values
are unchanged; structural coefficients are serialized pair-first, as c516 requires.

The format-only migration was checked before deleting the old reader: energy,
force, virial and scaled descriptors on the first CrCoNi frame were bit-identical
in FP64. The existing `CrCoNi.gpumd.npz` and `virial_mix.gpumd.npz` arrays were
not regenerated or changed. The full parity suite continues to use these
independently baked references.

Source model SHA256 before the container change:
`98e4e98e84007b794506c9e0c0ef5993beec19aff27bb667850fbbec35e142b7`.

Current named-block model SHA256:
`eb0c31c9c458ef22ff1c391f20fbc5298d12d33d64fbeee1a0492f7c9c5de2ff`.

Only the current file is shipped. No old reader or format converter is retained.
The spin fixtures and their independent provenance are in [spin_c516](spin_c516/README.md).
