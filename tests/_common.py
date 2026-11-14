# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.
# TorchNEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# TorchNEP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with TorchNEP.  If not, see <http://www.gnu.org/licenses/>.

"""Shared helpers for the torchnep test suite.

Only depends on the standard library + numpy + torch + torchnep itself.
No third-party packages (mdapy / polars / etc.) — predictions are checked
against frozen GPUMD-baked fixtures in ``data/*.gpumd.npz``.
"""
import os
from pathlib import Path
from typing import List

import numpy as np
import torch


DATA_DIR = Path(__file__).resolve().parent / "data"

DTYPE_MAP = {"float32": torch.float32, "float64": torch.float64}
NP_DTYPE_MAP = {"float32": np.float32, "float64": np.float64}


# --- environment-driven selection (run a subset for CI / debugging) ----------

def devices() -> List[str]:
    """Devices to test on. Honors $TEST_DEVICE (e.g. ``cpu``, ``cuda``)."""
    env = os.environ.get("TEST_DEVICE")
    if env:
        return [env]
    ds = ["cpu"]
    if torch.cuda.is_available():
        ds.append("cuda")
    return ds


def dtypes() -> List[str]:
    """Dtypes to test on. Honors $TEST_DTYPE (``float32`` / ``float64``)."""
    env = os.environ.get("TEST_DTYPE")
    if env:
        return [env]
    return ["float32", "float64"]


# --- fixture catalogue --------------------------------------------------------

# Each fixture: (nep.txt, xyz, npz) and the architectural feature it exercises.
# When adding a fixture, also rebake the .gpumd.npz via ``bake_fixtures.py``.
FIXTURES = [
    {
        "name":  "CrCoNi",
        "nep":   DATA_DIR / "nep_CrCoNi.txt",
        "xyz":   DATA_DIR / "CrCoNi.xyz",
        "ref":   DATA_DIR / "CrCoNi.gpumd.npz",
        "note":  "format_version 1, model nep, typewise ZBL, l_max 4 2 1. "
                 "Multi-frame: original (NN ~1.5 A) plus compressed/rattled "
                 "frames down to NN ~1.18 A that drive the ZBL repulsion "
                 "to/below its inner cutoff (1.25 A).",
    },
]


def load_reference(npz_path: Path) -> dict:
    """Load a frozen GPUMD reference fixture.

    Returns a dict with keys:
      ``E_per_atom``  : (Nframes,)            eV/atom
      ``F``           : (Natoms, 3)           eV/A
      ``V_per_atom``  : (Nframes, 6)          eV, GPUMD-ordered (xx,yy,zz,xy,yz,zx)
      ``D_per_atom``  : (Natoms, dim) scaled descriptor (q * q_scaler).
    """
    z = np.load(npz_path)
    return {
        "E_per_atom": z["E_per_atom"],
        "F":          z["F"],
        "V_per_atom": z["V_per_atom"],
        "D_per_atom": z["D_per_atom"],
    }


def write_gpumd_xyz(frames, dst_path: Path) -> None:
    """Write a list of read_xyz()-style frames to a GPUMD-compatible xyz.

    GPUMD prediction-mode requires ``energy=`` and a ``force:R:3`` column;
    we inject zero-filled defaults when the source frame doesn't carry them.
    """
    lines = []
    for fr in frames:
        natoms = fr["natoms"]
        cell = np.asarray(fr["cell"]).reshape(-1)
        cell_s = " ".join(f"{x:.10f}" for x in cell)
        energy = fr.get("energy", 0.0)
        if energy is None:
            energy = 0.0
        forces = fr.get("forces")
        if forces is None:
            forces = np.zeros((natoms, 3), dtype=float)
        lines.append(f"{natoms}")
        lines.append(
            f'pbc="T T T" Lattice="{cell_s}" energy={energy:.10f} '
            f'Properties=species:S:1:pos:R:3:force:R:3'
        )
        positions = np.asarray(fr["positions"])
        species = list(fr["species"])
        for i in range(natoms):
            x, y, z = positions[i]
            fx, fy, fz = forces[i]
            lines.append(
                f"{species[i]} {x:.10f} {y:.10f} {z:.10f} "
                f"{fx:.10f} {fy:.10f} {fz:.10f}")
    dst_path.write_text("\n".join(lines) + "\n")


# Mixed-virial fixture: first three CrCoNi frames, with an explicit virial tag
# on frames 0 and 2 and NONE on frame 1. Exercises predict_dataset's output
# parity with GPUMD for both the present-virial path (reference scaled to
# stress) and the missing-virial path (reference written as the -1e6 sentinel
# in both virial_train.out and stress_train.out). The baker and the test both
# build the input from this single definition so they stay in lock-step.
VIRIAL_MIX_NAME = "virial_mix"
VIRIAL_MIX_NFRAMES = 3
VIRIAL_MIX = {            # frame index -> 9-component (row-major) virial string
    0: "12 1 2 1 25 3 2 3 31",
    2: "-8 0.5 0 0.5 -6 1 0 1 -7",
}


def write_virial_mix_xyz(frames, dst_path: Path) -> None:
    """Write the mixed-virial fixture (GPUMD-compatible: energy + force column,
    plus a virial tag on the frames named in ``VIRIAL_MIX``). Forces are zero —
    only the virial/stress columns are under test."""
    lines = []
    for k, fr in enumerate(frames[:VIRIAL_MIX_NFRAMES]):
        na = fr["natoms"]
        cell = " ".join(f"{x:.10f}" for x in np.asarray(fr["cell"]).reshape(-1))
        tag = f"energy={-3.0 * na:.6f} "
        if k in VIRIAL_MIX:
            tag += f'virial="{VIRIAL_MIX[k]}" '
        tag += (f'pbc="T T T" Lattice="{cell}" '
                f'Properties=species:S:1:pos:R:3:force:R:3')
        lines.append(str(na))
        lines.append(tag)
        pos = np.asarray(fr["positions"])
        sp = list(fr["species"])
        for i in range(na):
            x, y, z = pos[i]
            lines.append(f"{sp[i]} {x:.10f} {y:.10f} {z:.10f} 0.0 0.0 0.0")
    dst_path.write_text("\n".join(lines) + "\n")


def parse_nep_header(nep_path: Path) -> dict:
    """Extract just the architecture fields we need to reconstruct nep.in for
    GPUMD prediction mode (cutoff / n_max / basis_size / l_max / neuron /
    type / zbl)."""
    from torchnep.spin_schema import read_nep_model
    config, _, _ = read_nep_model(nep_path)
    out = {
        "model": config["model"], "num_types": config["num_types"],
        "type_names": config["type_names"], "rc_radial": config["cutoff_radial"],
        "rc_angular": config["cutoff_angular"], "n_max_radial": config["n_max_radial"],
        "n_max_angular": config["n_max_angular"], "basis_size_radial": config["basis_size_radial"],
        "basis_size_angular": config["basis_size_angular"], "l_max": config["l_max"],
        "neuron": config["neuron"],
    }
    if config.get("zbl") is not None:
        out["zbl_outer"] = config["zbl"]
        if "typewise_cutoff_zbl_factor" in config:
            out["zbl_factor"] = config["typewise_cutoff_zbl_factor"]
    return out


def write_nep_in(hdr: dict, dst: Path, output_descriptor: int = 0) -> None:
    """Write a nep.in that mirrors the architecture of a nep.txt for GPUMD
    prediction mode.

    ``output_descriptor`` in {0, 1, 2}: 0 disables descriptor.out; 1 writes
    one row per frame (averaged); 2 writes one row per atom.
    """
    lines = [
        f"type {hdr['num_types']} {' '.join(hdr['type_names'])}",
        f"model {hdr['model']}",
        "prediction 1",
        "batch 1",
        f"cutoff {int(hdr['rc_radial'])} {int(hdr['rc_angular'])}",
        f"n_max {hdr['n_max_radial']} {hdr['n_max_angular']}",
        f"basis_size {hdr['basis_size_radial']} {hdr['basis_size_angular']}",
        f"l_max {' '.join(str(x) for x in hdr['l_max'])}",
        f"neuron {hdr['neuron']}",
    ]
    if "zbl_outer" in hdr:
        lines.append(f"zbl {hdr['zbl_outer']}")
        if "zbl_factor" in hdr:
            lines.append(f"use_typewise_cutoff_zbl {hdr['zbl_factor']}")
    if output_descriptor:
        lines.append(f"output_descriptor {output_descriptor}")
    dst.write_text("\n".join(lines) + "\n")
