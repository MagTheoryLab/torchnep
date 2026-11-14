# Independent c516 spin oracle fixtures

Copied from GPUMD-MagTheoryLab `c516` at
`fd358023c51239d7290d048928b3a0b298451993`,
`tests_pytest/fixtures/nep_spin/{c3l2s2,c2l4s4,c1l0s1}.{nep,json}`.
These are deliberate regression fixtures, not training outputs.

Upstream `tests_pytest/generate_spin_oracles.py` evaluates the independent
FP64 Torch/e3nn prototype without using the native coefficient tables or
kernels. JSON retains source hashes and model hash. Coverage includes large
orthogonal and triclinic periodic cells, zero spins, isolated centers and
nonzero input moments on inactive Bi/O atoms. Energies and derivatives come
from the same scalar energy.

Upstream serializes the original parameter values but rounds parameters and
scalers to FP32 before evaluating its JSON reference. We preserve both files
verbatim and use `rtol=2e-6, atol=2e-7`; observed FP64 differences are below
1.4e-7 across the twelve cases. These fixtures test inference, not the
quality of trained magnetic extrapolation or GPU performance.
