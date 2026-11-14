"""Build the structural center contraction without requiring Ninja.

Sai's production PyTorch environment does not ship Ninja.  This setuptools
entry point deliberately uses the standard compiler backend; the resulting
module is loaded only when TORCHNEP_OC_CUDA_PREBUILT=1.
"""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


HERE = Path(__file__).resolve().parent

setup(
    name="torchnep-oc-cuda-training-sectors",
    ext_modules=[
        CUDAExtension(
            name="torchnep_oc_struct_center_v1",
            sources=[str(HERE / "oc_struct_center.cu")],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-lineinfo"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
