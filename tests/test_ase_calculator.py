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

"""Tests for the optional ASE calculator (torchnep.ase_calculator.NEP).

ase is an optional dependency, so the whole module is skipped when it is not
installed.  Coverage:
  1. energy / forces / stress match the standalone NEPCalculator.compute and
     the frozen GPUMD reference for a ZBL fixture.
  2. forces and stress are consistent with finite differences of the energy.
  3. the NEP / ZBL / total component split sums back to the total exactly, and
     ZBL is non-trivial for a ZBL model.
  4. a non-periodic system yields energy/forces but no stress.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("ase")
from ase import Atoms                                   # noqa: E402

from torchnep.ase_calculator import NEP                 # noqa: E402
from torchnep.model import NEPModel                     # noqa: E402
from torchnep.nep import NEPCalculator                  # noqa: E402
from torchnep.data import read_xyz                      # noqa: E402
from _common import FIXTURES, load_reference            # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
# CrCoNi: nep4_zbl, typewise ZBL — exercises the NEP+ZBL split.
ZBL_FIX = next(f for f in FIXTURES if f["name"] == "CrCoNi")


def _atoms_from_frame(fr):
    return Atoms(symbols=list(fr["species"]),
                 positions=np.asarray(fr["positions"]),
                 cell=np.asarray(fr["cell"]), pbc=True)


@pytest.fixture
def spin_model(tmp_path):
    config = {
        "num_types": 2,
        "type_names": ["Fe", "Ge"],
        "cutoff_radial": 4.5,
        "cutoff_angular": 4.0,
        "n_max_radial": 2,
        "n_max_angular": 2,
        "basis_size_radial": 2,
        "basis_size_angular": 2,
        "l_max": [2, 0, 0],
        "neuron": 6,
        "spin_mode": 1,
        "spin_chiral": 1,
        "spin_compress": 2,
        "spin_n_max": [1, 0],
        "spin_basis_size": [2, 2],
        "spin_l_max": [2, 0, 0],
        "spin_cutoff": [4.5, 4.0],
        "spin_dof_type": ["Fe"],
        "spin_env_type": ["Fe", "Ge"],
    }
    path = tmp_path / "spin-nep.txt"
    torch.manual_seed(7)
    NEPModel(config).double().save_nep_txt(path, 16, 16)
    return path


def _spin_atoms():
    atoms = Atoms(
        symbols=["Fe", "Ge", "Fe", "Ge"],
        positions=[
            [1.0, 1.0, 1.0],
            [2.2, 1.1, 1.0],
            [1.3, 2.0, 1.4],
            [0.8, 1.5, 2.1],
        ],
        cell=np.eye(3) * 8.0,
        pbc=True,
    )
    atoms.set_array("spins", np.array([
        [0.8, 0.1, 0.2],
        [0.2, 0.9, -0.1],
        [-0.3, 0.4, 0.7],
        [0.5, -0.2, 0.6],
    ]))
    return atoms


def test_matches_core_and_gpumd_reference():
    fr = read_xyz(str(ZBL_FIX["xyz"]))[0]
    atoms = _atoms_from_frame(fr)
    atoms.calc = NEP(str(ZBL_FIX["nep"]), dtype="float64")

    e = atoms.get_potential_energy()
    f = atoms.get_forces()

    # vs standalone core
    core = NEPCalculator(str(ZBL_FIX["nep"]), dtype=torch.float64)
    cres = core.compute(list(fr["species"]), np.asarray(fr["positions"]),
                        np.asarray(fr["cell"]))
    assert abs(e - float(cres["energy"].sum())) < 1e-9
    assert np.abs(f - cres["forces"].numpy()).max() < 1e-9

    # vs frozen GPUMD reference (eV/atom)
    ref = load_reference(ZBL_FIX["ref"])
    assert abs(e / len(atoms) - ref["E_per_atom"][0]) < 1e-4
    assert np.abs(f - ref["F"][:len(atoms)]).max() < 1e-4


def test_forces_finite_difference():
    fr = read_xyz(str(ZBL_FIX["xyz"]))[0]
    atoms = _atoms_from_frame(fr)
    atoms.calc = NEP(str(ZBL_FIX["nep"]), dtype="float64")
    f = atoms.get_forces()

    rng = np.random.default_rng(0)
    h = 1e-4
    for k in rng.choice(len(atoms), size=4, replace=False):
        for c in range(3):
            a = atoms.copy(); a.calc = NEP(str(ZBL_FIX["nep"]), dtype="float64")
            p = a.get_positions(); p[k, c] += h; a.set_positions(p)
            ep = a.get_potential_energy()
            p[k, c] -= 2 * h; a.set_positions(p)
            em = a.get_potential_energy()
            fd = -(ep - em) / (2 * h)
            assert abs(fd - f[k, c]) < 1e-5


def test_stress_finite_difference():
    fr = read_xyz(str(ZBL_FIX["xyz"]))[0]
    atoms = _atoms_from_frame(fr)
    atoms.calc = NEP(str(ZBL_FIX["nep"]), dtype="float64")
    stress = atoms.get_stress()  # Voigt (xx, yy, zz, yz, xz, xy)

    V = atoms.get_volume()
    h = 1e-5
    # Diagonal strain components via finite difference of energy.
    voigt = [(0, 0), (1, 1), (2, 2)]
    for vi, (a, b) in enumerate(voigt):
        eps = np.eye(3)
        cp = atoms.copy(); cp.calc = NEP(str(ZBL_FIX["nep"]), dtype="float64")
        defo = eps.copy(); defo[a, b] += h
        cp.set_cell(atoms.get_cell() @ defo.T, scale_atoms=True)
        ep = cp.get_potential_energy()
        defo = eps.copy(); defo[a, b] -= h
        cp.set_cell(atoms.get_cell() @ defo.T, scale_atoms=True)
        em = cp.get_potential_energy()
        sigma_fd = (ep - em) / (2 * h) / V
        assert abs(sigma_fd - stress[vi]) < 1e-4


def test_component_split_sums_to_total():
    fr = read_xyz(str(ZBL_FIX["xyz"]))[0]
    atoms = _atoms_from_frame(fr)
    atoms.calc = NEP(str(ZBL_FIX["nep"]), dtype="float64")

    comp = atoms.calc.get_components(atoms)
    # energy: nep + zbl == total
    assert abs(comp["nep"]["energy"] + comp["zbl"]["energy"]
               - comp["total"]["energy"]) < 1e-9
    # total matches the plain ASE energy
    assert abs(comp["total"]["energy"] - atoms.get_potential_energy()) < 1e-9
    # forces and stress also split additively
    assert np.abs(comp["nep"]["forces"] + comp["zbl"]["forces"]
                  - comp["total"]["forces"]).max() < 1e-9
    assert np.abs(comp["nep"]["stress"] + comp["zbl"]["stress"]
                  - comp["total"]["stress"]).max() < 1e-12
    # ZBL is a real, non-zero repulsive contribution for this fixture
    assert abs(comp["zbl"]["energy"]) > 1e-6
    assert np.abs(comp["zbl"]["forces"]).max() > 1e-6

    # get_energy_components is the energy-only view of the same split
    ec = atoms.calc.get_energy_components(atoms)
    assert ec == {k: comp[k]["energy"] for k in comp}


def test_non_periodic_has_no_stress():
    # A small isolated cluster (no cell / no pbc): energy & forces, no stress.
    fr = read_xyz(str(ZBL_FIX["xyz"]))[0]
    pos = np.asarray(fr["positions"])[:6]
    atoms = Atoms(symbols=list(fr["species"])[:6], positions=pos, pbc=False)
    atoms.calc = NEP(str(ZBL_FIX["nep"]), dtype="float64")
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    assert np.isfinite(e)
    assert np.isfinite(f).all()
    assert "stress" not in atoms.calc.results


def test_spin_results_match_core(spin_model):
    atoms = _spin_atoms()
    atoms.calc = NEP(str(spin_model), dtype="float64")

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    mforces = atoms.calc.get_magnetic_forces(atoms)
    tau = atoms.calc.get_spin_torques(atoms)

    core = NEPCalculator(str(spin_model), dtype=torch.float64)
    expected = core.compute(
        atoms.get_chemical_symbols(), atoms.get_positions(),
        np.asarray(atoms.cell), spins=atoms.arrays["spins"])
    assert energy == pytest.approx(float(expected["energy"].sum()), abs=1e-10)
    assert np.allclose(forces, expected["forces"].numpy(), atol=1e-10)
    assert np.allclose(mforces, expected["mforces"].numpy(), atol=1e-10)
    assert np.allclose(tau, expected["tau"].numpy(), atol=1e-10)
    assert np.allclose(tau, np.cross(atoms.arrays["spins"], mforces))
    assert np.allclose(mforces[[1, 3]], 0.0)


def test_spin_array_change_invalidates_ase_cache(spin_model):
    atoms = _spin_atoms()
    atoms.calc = NEP(str(spin_model), dtype="float64")
    first = atoms.calc.get_magnetic_forces(atoms)

    atoms.arrays["spins"][0, 0] += 0.25
    second = atoms.calc.get_magnetic_forces(atoms)

    assert not np.allclose(first, second)
    assert np.array_equal(
        atoms.calc.atoms.arrays["spins"], atoms.arrays["spins"])


def test_spin_input_conventions_and_tiled_boundary(spin_model):
    atoms = _spin_atoms()
    spins = atoms.arrays.pop("spins")
    atoms.set_initial_magnetic_moments(spins[:, 2])
    atoms.calc = NEP(str(spin_model), dtype="float64", tiled="auto")
    atoms.calc.tiled_threshold = 0
    assert np.isfinite(atoms.get_potential_energy())
    expected_spins = np.column_stack((
        np.zeros(len(atoms)), np.zeros(len(atoms)), spins[:, 2]))
    assert np.array_equal(atoms.calc._spin_vectors(atoms), expected_spins)

    atoms.calc = NEP(str(spin_model), dtype="float64", tiled=True)
    with pytest.raises(NotImplementedError, match="tiled=True"):
        atoms.get_potential_energy()

    atoms.set_initial_magnetic_moments(None)
    atoms.calc = NEP(str(spin_model), dtype="float64")
    with pytest.raises(ValueError, match="requires per-atom moments"):
        atoms.get_potential_energy()
