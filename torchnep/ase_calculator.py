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

"""ASE interface for torchnep NEP4 models.

This module provides :class:`NEP`, a standard ``ase.calculators.calculator.
Calculator`` so that any ASE workflow (relaxation, MD, EOS, phonons, ...) can
drive a trained NEP4 potential::

    from ase.io import read
    from torchnep.ase_calculator import NEP

    atoms = read("POSCAR")
    atoms.calc = NEP("nep.txt")
    print(atoms.get_potential_energy())
    print(atoms.get_forces())
    print(atoms.get_stress())          # only for periodic cells

``ase`` is an OPTIONAL dependency of torchnep — the core package (training and
prediction) never imports it.  Importing *this* module requires ase to be
installed; if it is not, a clear ImportError is raised here rather than deep
inside ASE.  This is also why ``NEP`` is intentionally not re-exported from
``torchnep/__init__.py``.
"""

import numpy as np

try:
    from ase.calculators.calculator import (
        Calculator,
        PropertyNotImplementedError,
        all_changes,
    )
    from ase.stress import full_3x3_to_voigt_6_stress
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only sans ase
    raise ModuleNotFoundError(
        "torchnep.ase_calculator requires the optional dependency 'ase'. "
        "Install it with `pip install ase`."
    ) from exc

from .nep import NEPCalculator as _NEPCore


def _resolve_dtype(dtype):
    import torch
    if isinstance(dtype, torch.dtype):
        return dtype
    return {"float64": torch.float64, "float32": torch.float32,
            "double": torch.float64, "single": torch.float32}[str(dtype)]


class NEP(Calculator):
    """ASE calculator backed by a torchnep NEP4 model (``nep.txt``).

    Parameters
    ----------
    model_file : str
        Path to a GPUMD-format ``nep.txt`` (NEP4).
    dtype : str or torch.dtype
        Compute precision; ``"float64"`` (default) reproduces GPUMD to
        round-off, ``"float32"`` is faster.
    device : str or torch.device
        Torch device, e.g. ``"cpu"`` (default) or ``"cuda"``.
    **kwargs
        Forwarded to ``ase.calculators.calculator.Calculator`` (e.g. ``label``).

    Notes
    -----
    ``stress`` is only reported for fully periodic cells (a finite volume is
    required); for molecules / clusters it is omitted.  Non-periodic systems
    are handled by embedding the atoms in a vacuum box large enough that no
    atom sees a periodic image within the model cutoff.
    """

    implemented_properties = [
        "energy", "energies", "free_energy", "forces", "stress",
        "mforces", "tau",
    ]

    def __init__(self, model_file, dtype="float64", device="cpu",
                 tiled="auto", block_size="auto", compile=False, **kwargs):
        super().__init__(**kwargs)
        self.nep = _NEPCore(model_file, dtype=_resolve_dtype(dtype),
                            device=device)
        # tiled: memory-bounded analytical inference for large cells.
        #   True   -> always tile;  False -> never (autograd path);
        #   "auto" -> tile once the system exceeds ``tiled_threshold`` atoms.
        self.tiled = tiled
        # block_size: "auto" sizes each tile from free memory; or an int override.
        self.block_size = block_size
        self.tiled_threshold = 50000
        # compile: torch.compile the tiled kernels (CPU/CUDA only; one-time
        # warm-up, then a dynamic graph that survives MD pair-count changes).
        self.compile = bool(compile)

    # -- helpers --------------------------------------------------------------
    def _cell_for_neighbors(self, atoms):
        """Return (cell_3x3, periodic) for the neighbor-list builder.

        Fully periodic cells are passed through. Otherwise we wrap the atoms in
        an orthorhombic vacuum box padded by ``2*rc`` so the builder's PBC
        replicas never fall within the cutoff (emulating an isolated system).
        """
        if atoms.cell.rank == 3 and atoms.pbc.all():
            return np.asarray(atoms.get_cell()[:], dtype=float), True
        pos = atoms.get_positions()
        rc = max(self.nep.rc_radial, self.nep.rc_angular)
        if self.nep.spin_mode:
            rc = max(rc, self.nep.spin_cutoff[0])
        span = pos.max(axis=0) - pos.min(axis=0) if len(pos) else np.zeros(3)
        box = np.diag(span + 4.0 * rc + 1.0)
        return box, False

    def _spin_vectors(self, atoms):
        """Return ASE magnetic moments as an ``(N, 3)`` spin array.

        Vector arrays named ``spins``, ``spin``, ``moments``, or ``moment``
        take precedence. ASE's standard ``initial_magmoms`` array is also
        accepted; scalar collinear moments are interpreted as z-axis vectors.
        """
        spins = None
        for name in ("spins", "spin", "moments", "moment"):
            if atoms.has(name):
                spins = np.asarray(atoms.get_array(name), dtype=float)
                break
        if spins is None and atoms.has("initial_magmoms"):
            spins = np.asarray(
                atoms.get_initial_magnetic_moments(), dtype=float)
            if spins.ndim == 1:
                spins = np.column_stack((
                    np.zeros_like(spins), np.zeros_like(spins), spins))
        if spins is None:
            raise ValueError(
                "Spin NEP requires per-atom moments. Set an (N, 3) ASE array "
                "named 'spins', 'spin', 'moments', or 'moment', or call "
                "atoms.set_initial_magnetic_moments(...).")
        expected = (len(atoms), 3)
        if spins.shape != expected:
            raise ValueError(
                f"ASE spin moments must have shape {expected}, got "
                f"{spins.shape}")
        if not np.isfinite(spins).all():
            raise ValueError("ASE spin moments must contain only finite values")
        return spins

    def check_state(self, atoms, tol=1e-15):
        """Include custom ASE spin arrays in calculator cache invalidation."""
        changes = super().check_state(atoms, tol=tol)
        if not self.nep.spin_mode or self.atoms is None:
            return changes
        try:
            previous = self._spin_vectors(self.atoms)
            current = self._spin_vectors(atoms)
            spin_changed = (
                previous.shape != current.shape
                or not np.allclose(previous, current, rtol=0.0, atol=tol))
        except ValueError:
            spin_changed = True
        if spin_changed and "spins" not in changes:
            changes.append("spins")
        return changes

    # -- ASE entry point ------------------------------------------------------
    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms  # set by super().calculate

        cell, periodic = self._cell_for_neighbors(atoms)
        species = atoms.get_chemical_symbols()
        spins = self._spin_vectors(atoms) if self.nep.spin_mode else None
        use_tiled = (self.tiled is True or
                     (self.tiled == "auto" and len(species) >= self.tiled_threshold))
        if self.nep.spin_mode and self.tiled is True:
            raise NotImplementedError(
                "tiled=True is not supported for O/C spin models; use "
                "tiled=False or tiled='auto'")
        if self.nep.spin_mode:
            # The analytical tiled kernels do not yet implement spin.
            use_tiled = False
        if use_tiled:
            res = self.nep.compute_tiled(
                species=species, positions=atoms.get_positions(),
                cell=cell, block_size=self.block_size, compile=self.compile)
        else:
            res = self.nep.compute(
                species=species, positions=atoms.get_positions(), cell=cell,
                spins=spins)
        energies = res["energy"].detach().cpu().numpy()
        forces = res["forces"].detach().cpu().numpy()

        self.results["energies"] = energies
        self.results["energy"] = float(energies.sum())
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = forces
        if self.nep.spin_mode:
            self.results["mforces"] = (
                res["mforces"].detach().cpu().numpy())
            self.results["tau"] = res["tau"].detach().cpu().numpy()

        if periodic:
            self.results["stress"] = self._stress_from_virial(
                res["virial"], atoms.get_volume())

    def get_magnetic_forces(self, atoms=None):
        """Return magnetic forces ``-dE/dspin`` as an ``(N, 3)`` array."""
        if not self.nep.spin_mode:
            raise PropertyNotImplementedError(
                "magnetic forces are only available for spin models")
        return self.get_property("mforces", atoms)

    def get_spin_torques(self, atoms=None):
        """Return per-atom ``tau = spin x mforce`` as an ``(N, 3)`` array."""
        if not self.nep.spin_mode:
            raise PropertyNotImplementedError(
                "spin torques are only available for spin models")
        return self.get_property("tau", atoms)

    @staticmethod
    def _stress_from_virial(virial, volume):
        # Per-atom virial is (N, 9) row-major; W_ab = Σ rij_a * F_b.
        # ASE stress σ = -W / V (Voigt order xx, yy, zz, yz, xz, xy).
        w = virial.detach().cpu().numpy().sum(0).reshape(3, 3)
        return full_3x3_to_voigt_6_stress(-w / volume)

    def get_energy_components(self, atoms=None):
        """Return the NEP / ZBL / total potential-energy split (eV).

        Useful to inspect how much of the energy comes from the ZBL repulsive
        baseline versus the neural-network NEP part::

            {'nep': ..., 'zbl': ..., 'total': ...}

        For a model trained without ZBL, ``'zbl'`` is 0 and ``'nep'`` equals
        ``'total'``.  See :meth:`get_components` for forces and stress too.
        """
        return {k: v["energy"] for k, v in self.get_components(atoms).items()}

    def get_components(self, atoms=None):
        """Full NEP / ZBL / total breakdown of energy, forces, and stress.

        Returns ``{'nep': {...}, 'zbl': {...}, 'total': {...}}`` where each
        inner dict has ``'energy'`` (float, eV), ``'forces'`` ((N, 3) eV/Å) and,
        for periodic cells, ``'stress'`` (6-vector Voigt, eV/Å³).  The ``total``
        block equals what ``get_potential_energy`` / ``get_forces`` /
        ``get_stress`` return.
        """
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError("No atoms supplied and none attached to the calculator.")
        if self.nep.spin_mode:
            raise NotImplementedError(
                "NEP/ZBL component splitting is not implemented for "
                "O/C spin models")

        cell, periodic = self._cell_for_neighbors(atoms)
        res = self.nep.compute(
            species=atoms.get_chemical_symbols(),
            positions=atoms.get_positions(),
            cell=cell,
            return_components=True,
        )
        vol = atoms.get_volume() if periodic else None
        out = {}
        for name, esuffix in (("nep", "_nep"), ("zbl", "_zbl"), ("total", "")):
            e = res["energy" + esuffix].detach().cpu().numpy()
            block = {"energy": float(e.sum()),
                     "forces": res["forces" + esuffix].detach().cpu().numpy()}
            if periodic:
                block["stress"] = self._stress_from_virial(
                    res["virial" + esuffix], vol)
            out[name] = block
        return out
