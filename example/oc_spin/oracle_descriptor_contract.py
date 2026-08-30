# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.
# TorchNEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Sealed descriptor-level oracle for the O/C magnetic construction.

This program answers three deliberately narrow questions without training an
ANN:

1. Which independently evaluated scalar teachers lie in the *linear* span of
   the current 37/49/58/79-dimensional magnetic descriptors?
2. Does increasing C from two to three strictly retain an existing descriptor
   and arbitrary one-hidden-layer atomic-energy head?
3. Are nominal channels active and numerically resolved on the sampled local
   environments, including their position, spin, strain and volume gradients?

The positive teachers use the same declared Chebyshev radial basis but an
independent implementation of every contraction.  They do not read a column
from :class:`MagneticPolynomialDescriptor`.  The three negative controls are
outside the declared finite cutoff/angular/order linear span.  Consequently a
negative-control miss is a boundary certificate, not a claim about the full
nonlinear ANN function class.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch

from torchnep.magnetic_polynomial import MagneticPolynomialDescriptor
from torchnep.spin_diagnostics import relative_vector_error, svd_health


DTYPE = torch.float64
CUTOFF = 6.0
BASIS_SIZE = 8
N_TYPES = 2

DESCRIPTORS = {
    "o2c2": {"order": 2, "compress": 2, "dimension": 37},
    "o3c2": {"order": 3, "compress": 2, "dimension": 49},
    "o2c3": {"order": 2, "compress": 3, "dimension": 58},
    "o3c3": {"order": 3, "compress": 3, "dimension": 79},
}

PROFILES = {
    "smoke": {"train": 96, "blind": 32, "derivative": 2},
    "full": {"train": 384, "blind": 128, "derivative": 8},
}

POSITIVE_TEACHERS = (
    "J",
    "D",
    "biquadratic",
    "anisotropy_l2",
    "environment_axial",
    "rank2_commutator",
    "l112_edge_response",
    "rank3_bulk",
    "variable_magnitude",
    "strain_volume_composite",
)
NEGATIVE_TEACHERS = ("angular_l_gt_2", "order_gt_3", "long_range")
TEACHERS = POSITIVE_TEACHERS + NEGATIVE_TEACHERS

# Expected membership in the *linear descriptor span*.  This matrix is part of
# the certificate and is kept separate from the observed numerical verdict.
EXPECTED_REACHABILITY = {
    "J": set(DESCRIPTORS),
    "D": set(DESCRIPTORS),
    "biquadratic": set(DESCRIPTORS),
    "anisotropy_l2": set(DESCRIPTORS),
    "environment_axial": set(DESCRIPTORS),
    "rank2_commutator": set(DESCRIPTORS),
    "l112_edge_response": {"o3c2", "o3c3"},
    "rank3_bulk": {"o3c3"},
    "variable_magnitude": {"o3c2", "o3c3"},
    "strain_volume_composite": set(DESCRIPTORS),
    "angular_l_gt_2": set(),
    "order_gt_3": set(),
    "long_range": set(),
}

RAW_TOKEN_FOR_TEACHER = {
    "J": ("edge_l0_dot", 0),
    "D": ("edge_l11_axial", 0),
    "biquadratic": ("edge_l0_dot2", 1),
    "anisotropy_l2": ("edge_l2_pair", 0),
    "environment_axial": ("coupling_l11_axial", 0),
    "rank2_commutator": ("coupling_l22_axial", 0),
    "l112_edge_response": ("coupling_l112_edge_response", 0),
    "rank3_bulk": ("coupling_l111_bulk", 2),
    "variable_magnitude": ("edge_l0_moment_gate", 0),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_files(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path.resolve() for path in paths if path.is_file()):
        try:
            label = str(path.relative_to(root.resolve()))
        except ValueError:
            label = str(path)
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _run_git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=repo, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout.decode(errors="replace")


def _git_provenance(repo: Path) -> Dict[str, Any]:
    try:
        git_root = Path(
            _run_git(repo, "rev-parse", "--show-toplevel").strip()
        ).resolve()
        if git_root != repo.resolve():
            raise RuntimeError(
                f"{repo} is nested below unrelated Git root {git_root}"
            )
        commit = _run_git(repo, "rev-parse", "HEAD").strip()
    except (subprocess.CalledProcessError, RuntimeError):
        required = (
            "CLOSURE_GIT_COMMIT", "CLOSURE_GIT_DIRTY",
            "CLOSURE_DIRTY_DIFF_SHA256",
        )
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                "the synced source tree has no Git metadata; export "
                "CLOSURE_GIT_COMMIT, CLOSURE_GIT_DIRTY and "
                "CLOSURE_DIRTY_DIFF_SHA256 from the local frozen source "
                "provenance before submitting"
            )
        dirty_value = os.environ["CLOSURE_GIT_DIRTY"]
        if dirty_value not in ("0", "1"):
            raise RuntimeError("CLOSURE_GIT_DIRTY must be exactly 0 or 1")
        return {
            "commit": os.environ["CLOSURE_GIT_COMMIT"],
            "dirty": dirty_value == "1",
            "dirty_state_sha256": os.environ[
                "CLOSURE_DIRTY_DIFF_SHA256"],
            "status_porcelain": [],
            "provenance_source": "submission-environment",
        }
    status = _run_git(
        repo, "status", "--porcelain=v1", "--untracked-files=all"
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"], cwd=repo, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    digest = hashlib.sha256(status.encode() + b"\0" + diff)
    for relative in sorted(
        _run_git(repo, "ls-files", "--others", "--exclude-standard")
        .splitlines()
    ):
        path = repo / relative
        if path.is_file():
            digest.update(relative.encode() + b"\0" + path.read_bytes())
    return {
        "commit": commit,
        "dirty": bool(status),
        "dirty_state_sha256": digest.hexdigest(),
        "status_porcelain": status.splitlines(),
        "provenance_source": "live-git-worktree",
    }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_provenance(repo: Path) -> Dict[str, Any]:
    sources = [
        Path(__file__).resolve(),
        repo / "torchnep" / "magnetic_polynomial.py",
        repo / "torchnep" / "ops.py",
        repo / "torchnep" / "spin_diagnostics.py",
    ]
    return {
        "source_sha256": _hash_files(sources, repo),
        "files": [str(path.relative_to(repo)) for path in sources],
    }


def _local_import_provenance(repo: Path) -> Dict[str, str]:
    """Fail closed if a source-only run imports an installed old package."""
    expected = (repo / "torchnep").resolve()
    imported = {
        "MagneticPolynomialDescriptor": Path(
            inspect.getsourcefile(MagneticPolynomialDescriptor)).resolve(),
        "svd_health": Path(inspect.getsourcefile(svd_health)).resolve(),
    }
    wrong = {
        name: str(path) for name, path in imported.items()
        if expected not in path.parents
    }
    if wrong:
        raise RuntimeError(
            "descriptor oracle imported TorchNEP outside the synchronized "
            f"source tree {expected}: {wrong}"
        )
    return {name: str(path) for name, path in imported.items()}


def _index_sum(
    reference: torch.Tensor,
    index: torch.Tensor,
    values: torch.Tensor,
    size: int,
) -> torch.Tensor:
    result = reference.new_zeros((size, values.shape[-1]))
    if index.numel():
        result.index_add_(0, index, values)
    return result


def _stf5(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Independent orthonormal STF rank-two implementation."""
    xx = left[..., 0] * right[..., 0]
    yy = left[..., 1] * right[..., 1]
    zz = left[..., 2] * right[..., 2]
    xy = 0.5 * (left[..., 0] * right[..., 1]
                + left[..., 1] * right[..., 0])
    yz = 0.5 * (left[..., 1] * right[..., 2]
                + left[..., 2] * right[..., 1])
    zx = 0.5 * (left[..., 2] * right[..., 0]
                + left[..., 0] * right[..., 2])
    return torch.stack((
        (2.0 * zz - xx - yy) / math.sqrt(6.0),
        (xx - yy) / math.sqrt(2.0),
        math.sqrt(2.0) * xy,
        math.sqrt(2.0) * yz,
        math.sqrt(2.0) * zx,
    ), dim=-1)


def _stf5_to_matrix(value: torch.Tensor) -> torch.Tensor:
    q0, q1, q2, q3, q4 = value.unbind(-1)
    xx = -q0 / math.sqrt(6.0) + q1 / math.sqrt(2.0)
    yy = -q0 / math.sqrt(6.0) - q1 / math.sqrt(2.0)
    zz = 2.0 * q0 / math.sqrt(6.0)
    xy = q2 / math.sqrt(2.0)
    yz = q3 / math.sqrt(2.0)
    zx = q4 / math.sqrt(2.0)
    return torch.stack(
        (xx, xy, zx, xy, yy, yz, zx, yz, zz), dim=-1
    ).reshape(value.shape[:-1] + (3, 3))


def _commutator_axial(left5: torch.Tensor, right5: torch.Tensor) -> torch.Tensor:
    left = _stf5_to_matrix(left5)
    right = _stf5_to_matrix(right5)
    commutator = left @ right - right @ left
    return torch.stack((
        0.5 * (commutator[..., 2, 1] - commutator[..., 1, 2]),
        0.5 * (commutator[..., 0, 2] - commutator[..., 2, 0]),
        0.5 * (commutator[..., 1, 0] - commutator[..., 0, 1]),
    ), dim=-1)


def _raw_chebyshev_basis(distance: torch.Tensor) -> torch.Tensor:
    """Declared radial basis, implemented independently from torchnep.ops."""
    clipped = distance.clamp(max=CUTOFF)
    fc = 0.5 * torch.cos(torch.pi * clipped / CUTOFF) + 0.5
    x = 2.0 * (clipped / CUTOFF - 1.0).square() - 1.0
    polynomials = [torch.ones_like(x), x]
    for _ in range(2, BASIS_SIZE + 1):
        polynomials.append(
            2.0 * x * polynomials[-1] - polynomials[-2]
        )
    basis = 0.5 * (torch.stack(polynomials, -1) + 1.0)
    basis = basis * fc.unsqueeze(-1)
    return basis * (distance < CUTOFF).to(distance).unsqueeze(-1)


def _leg_mix(compress: int, *, device: torch.device) -> torch.Tensor:
    """Common C-nested CP legs used by both raw and production descriptors."""
    result = torch.zeros(4, compress, compress, dtype=DTYPE, device=device)
    result[0].copy_(torch.eye(compress, dtype=DTYPE, device=device))
    result[3].copy_(torch.eye(compress, dtype=DTYPE, device=device))
    if compress == 1:
        result[1, 0, 0] = 1.0
        result[2, 0, 0] = 1.0
    else:
        result[1, 0, 1] = 1.0
        result[1, 1, 0] = 1.0
        result[2, 0, 0] = 1.0
        result[2, 1, 1] = 1.0
        if compress >= 3:
            # The first two rows are the exact C2 embedding.  The third rows
            # expose three independent density legs for the rank-three bulk
            # teacher without feeding the new radial coordinate back into C2.
            result[1, 2, 0] = 1.0
            result[2, 2, 1] = 1.0
        for channel in range(3, compress):
            result[1, channel, channel] = 1.0
            result[2, channel, channel] = 1.0
    return result


def _radial_coefficients(
    compress: int, *, device: torch.device,
) -> torch.Tensor:
    coefficients = torch.zeros(
        1, N_TYPES, N_TYPES, compress, BASIS_SIZE + 1,
        dtype=DTYPE, device=device,
    )
    for channel in range(compress):
        coefficients[..., channel, channel] = 1.0
    return coefficients


def build_descriptor(
    order: int, compress: int, device: torch.device,
) -> Tuple[MagneticPolynomialDescriptor, torch.Tensor]:
    descriptor = MagneticPolynomialDescriptor(
        spin_compress=compress,
        spin_l_max=2,
        spin_order=order,
        spin_soc=True,
        basis_size_radial=BASIS_SIZE,
        cutoff_radial=CUTOFF,
    ).to(device=device, dtype=DTYPE)
    with torch.no_grad():
        descriptor.radial_leg_mix.copy_(_leg_mix(compress, device=device))
    return descriptor, _radial_coefficients(compress, device=device)


def _directed_pairs(
    n_atoms: int, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pair_i = []
    pair_j = []
    for center in range(n_atoms):
        for neighbor in range(n_atoms):
            if center != neighbor:
                pair_i.append(center)
                pair_j.append(neighbor)
    return (
        torch.tensor(pair_i, dtype=torch.long, device=device),
        torch.tensor(pair_j, dtype=torch.long, device=device),
    )


def generate_environments(
    count: int, seed: int, device: torch.device,
) -> Sequence[Dict[str, torch.Tensor]]:
    """Generate nonperiodic core-plus-satellite local environments."""
    rng = np.random.default_rng(seed)
    environments = []
    template = np.array([
        [-1.00, -0.70, -0.45],
        [0.85, -0.62, 0.35],
        [-0.35, 1.05, -0.20],
        [0.30, 0.25, 1.10],
        [1.10, 0.90, -0.75],
        [-0.90, 0.50, 0.95],
    ])
    for _ in range(count):
        deformation = np.eye(3) + rng.normal(0.0, 0.07, (3, 3))
        core = (template + rng.normal(0.0, 0.12, template.shape)) @ deformation
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        satellite = direction * rng.uniform(7.1, 8.0)
        positions = np.vstack((core, satellite))
        directions = rng.normal(size=(len(positions), 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        magnitudes = rng.uniform(0.25, 2.10, (len(positions), 1))
        spins = directions * magnitudes
        types = rng.integers(0, N_TYPES, len(positions), dtype=np.int64)
        pair_i, pair_j = _directed_pairs(len(positions), device)
        environments.append({
            "positions": torch.tensor(positions, dtype=DTYPE, device=device),
            "spins": torch.tensor(spins, dtype=DTYPE, device=device),
            "atom_types": torch.tensor(types, dtype=torch.long, device=device),
            "pair_i": pair_i,
            "pair_j": pair_j,
        })
    return environments


def _deformed_inputs(
    environment: Mapping[str, torch.Tensor],
    *,
    require_grad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = environment["positions"].detach().clone()
    spins = environment["spins"].detach().clone()
    strain = positions.new_zeros(())
    volume = positions.new_zeros(())
    if require_grad:
        positions.requires_grad_(True)
        spins.requires_grad_(True)
        strain.requires_grad_(True)
        volume.requires_grad_(True)
    scale = torch.stack((
        torch.exp(volume / 3.0 + strain),
        torch.exp(volume / 3.0 - strain),
        torch.exp(volume / 3.0),
    ))
    deformed = positions * scale.unsqueeze(0)
    pair_i = environment["pair_i"]
    pair_j = environment["pair_j"]
    rij = deformed.index_select(0, pair_j) - deformed.index_select(0, pair_i)
    return positions, spins, strain, volume, rij


def _density(
    values: torch.Tensor,
    weights: torch.Tensor,
    pair_i: torch.Tensor,
    n_atoms: int,
) -> torch.Tensor:
    weighted = (
        weights[:, :, None] * values[:, None, :]
    ).reshape(weights.shape[0], -1)
    return _index_sum(values, pair_i, weighted, n_atoms).reshape(
        n_atoms, weights.shape[1], values.shape[-1]
    )


def _project(value: torch.Tensor, mix: torch.Tensor, leg: int) -> torch.Tensor:
    return torch.einsum("rc,ncd->nrd", mix[leg], value)


def _raw_atomic_tokens(
    rij: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    spins: torch.Tensor,
    *,
    compress: int = 3,
) -> Dict[str, torch.Tensor]:
    """Independent raw contraction evaluator for publication teachers."""
    n_atoms = spins.shape[0]
    distance = torch.linalg.vector_norm(rij, dim=-1).clamp_min(1.0e-12)
    rhat = rij / distance.unsqueeze(-1)
    weights = _raw_chebyshev_basis(distance)[:, :compress]
    mix = _leg_mix(compress, device=spins.device)
    si = spins.index_select(0, pair_i)
    sj = spins.index_select(0, pair_j)
    spin2 = spins.square().sum(-1)
    si2 = spin2.index_select(0, pair_i)
    sj2 = spin2.index_select(0, pair_j)
    dot = (si * sj).sum(-1)
    omega = torch.cross(si, sj, dim=-1)
    qrr = _stf5(rhat, rhat)

    polar = _density(rhat, weights, pair_i, n_atoms)
    spin_density = _density(sj, weights, pair_i, n_atoms)
    axial = _density(torch.cross(rhat, sj, dim=-1), weights, pair_i, n_atoms)
    geometry_l2 = _density(qrr, weights, pair_i, n_atoms)
    l2_on_spin = _density(
        torch.matmul(_stf5_to_matrix(qrr), sj.unsqueeze(-1)).squeeze(-1),
        weights, pair_i, n_atoms,
    )
    omega_density = _density(omega, weights, pair_i, n_atoms)
    dot_omega = _density(dot.unsqueeze(-1) * omega, weights, pair_i, n_atoms)

    def edge_sum(value: torch.Tensor, edge_weights: torch.Tensor = weights):
        return _index_sum(
            spins, pair_i,
            edge_weights * value.unsqueeze(-1), n_atoms,
        )

    tokens: Dict[str, torch.Tensor] = {
        "edge_l0_dot": edge_sum(dot),
        "edge_l0_dot2": edge_sum(dot.square()),
        "edge_l2_pair": edge_sum((_stf5(si, sj) * qrr).sum(-1)),
        "edge_l0_moment_gate": edge_sum(dot * (si2 + sj2)),
    }

    polar0 = _project(polar, mix, 0)
    polar1 = _project(polar, mix, 1)
    omega0 = _project(omega_density, mix, 0)
    tokens["coupling_l11_axial"] = (
        torch.cross(polar0, polar1, dim=-1) * omega0
    ).sum(-1)

    edge_weight0 = weights @ mix[0].T
    polar_i = polar0.index_select(0, pair_i)
    edge_axis = torch.cross(polar_i, rhat.unsqueeze(1), dim=-1)
    edge_response = (edge_axis * omega.unsqueeze(1)).sum(-1)
    tokens["edge_l11_axial"] = _index_sum(
        spins, pair_i, edge_weight0 * edge_response, n_atoms
    )

    q0 = _project(geometry_l2, mix, 0)
    q1 = _project(geometry_l2, mix, 1)
    q_axis = _commutator_axial(q0, q1)
    tokens["coupling_l22_axial"] = (q_axis * omega0).sum(-1)

    q0_matrix = _stf5_to_matrix(q0)
    q_on_p1 = torch.matmul(q0_matrix, polar1.unsqueeze(-1)).squeeze(-1)
    mixed_chirality = (
        polar0 * torch.cross(polar1, q_on_p1, dim=-1)
    ).sum(-1)
    scalar_edge_response = (rhat * omega).sum(-1, keepdim=True)
    tokens["coupling_l112_edge_response"] = _index_sum(
        spins, pair_i,
        edge_weight0
        * mixed_chirality.index_select(0, pair_i)
        * scalar_edge_response,
        n_atoms,
    )

    polar2 = _project(polar, mix, 2)
    bulk_chirality = (
        polar0 * torch.cross(polar1, polar2, dim=-1)
    ).sum(-1)
    edge_weight3 = weights @ mix[3].T
    tokens["coupling_l111_bulk"] = _index_sum(
        spins, pair_i,
        edge_weight3
        * bulk_chirality.index_select(0, pair_i)
        * scalar_edge_response,
        n_atoms,
    )
    return tokens


def raw_teacher_values(
    rij: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    spins: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    tokens = _raw_atomic_tokens(
        rij, pair_i, pair_j, spins, compress=3
    )
    result = {
        name: tokens[token].sum(0)[index]
        for name, (token, index) in RAW_TOKEN_FOR_TEACHER.items()
    }
    result["strain_volume_composite"] = (
        0.31 * result["J"]
        - 0.23 * result["D"]
        + 0.17 * result["anisotropy_l2"]
        + 0.11 * result["environment_axial"]
    )

    distance = torch.linalg.vector_norm(rij, dim=-1).clamp_min(1.0e-12)
    rhat = rij / distance.unsqueeze(-1)
    si = spins.index_select(0, pair_i)
    sj = spins.index_select(0, pair_j)
    basis = _raw_chebyshev_basis(distance)
    # Higher angular content: a scalar under joint O(3), even under time
    # reversal, but containing angular ranks above the declared l_max=2.
    result["angular_l_gt_2"] = (
        basis[:, 0]
        * (rhat * si).sum(-1).pow(3)
        * (rhat * sj).sum(-1).pow(3)
    ).sum()
    # Fourth-order product of independently accumulated edge-response
    # densities.  This is outside the declared linear O<=3 grammar.
    dot = (si * sj).sum(-1)
    first0 = _index_sum(
        spins, pair_i, (basis[:, 0] * dot).unsqueeze(-1), spins.shape[0]
    ).squeeze(-1)
    first1 = _index_sum(
        spins, pair_i, (basis[:, 1] * dot).unsqueeze(-1), spins.shape[0]
    ).squeeze(-1)
    result["order_gt_3"] = (first0.square() * first1.square()).sum()
    far = distance > (CUTOFF + 0.25)
    result["long_range"] = (
        far.to(distance)
        * torch.exp(-(distance - CUTOFF) / 2.0)
        * dot
    ).sum()
    return result


def _descriptor_atomic(
    descriptor: MagneticPolynomialDescriptor,
    coefficients: torch.Tensor,
    environment: Mapping[str, torch.Tensor],
    rij: torch.Tensor,
    spins: torch.Tensor,
) -> torch.Tensor:
    return descriptor(
        rij,
        environment["pair_i"],
        environment["pair_j"],
        spins,
        spins.shape[0],
        environment["atom_types"],
        coefficients,
        None,
    )


def _column_labels(descriptor: MagneticPolynomialDescriptor) -> Sequence[str]:
    labels = []
    for token in descriptor.schema:
        labels.extend(
            f"{token.name}[{index}]" for index in range(token.width)
        )
    return labels


def collect_matrices(
    descriptor: MagneticPolynomialDescriptor,
    coefficients: torch.Tensor,
    environments: Sequence[Mapping[str, torch.Tensor]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    atomic_rows = []
    global_rows = []
    teacher_rows = {name: [] for name in TEACHERS}
    for environment in environments:
        _, spins, _, _, rij = _deformed_inputs(environment)
        atomic = _descriptor_atomic(
            descriptor, coefficients, environment, rij, spins
        )
        atomic_rows.append(atomic.detach().cpu().numpy())
        global_rows.append(atomic.sum(0).detach().cpu().numpy())
        raw = raw_teacher_values(
            rij, environment["pair_i"], environment["pair_j"], spins
        )
        for name in TEACHERS:
            teacher_rows[name].append(float(raw[name].detach().cpu()))
    return (
        np.concatenate(atomic_rows, axis=0),
        np.stack(global_rows),
        {name: np.asarray(values) for name, values in teacher_rows.items()},
    )


def _linear_fit(matrix: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    scale = np.sqrt(np.mean(np.square(matrix), axis=0))
    magnitude = np.maximum(np.max(np.abs(matrix), axis=0), 1.0)
    active = scale > 64.0 * np.finfo(np.float64).eps * magnitude
    normalized = matrix[:, active] / scale[active]
    solution, _, rank, singular = np.linalg.lstsq(
        normalized, target, rcond=1.0e-13
    )
    head = np.zeros(matrix.shape[1], dtype=np.float64)
    head[active] = solution / scale[active]
    return {
        "head": head,
        "rank": int(rank),
        "singular_values": [float(value) for value in singular],
        "active_columns": int(active.sum()),
    }


def _scalar_error(prediction: np.ndarray, reference: np.ndarray) -> Dict[str, Any]:
    return relative_vector_error(prediction.reshape(-1), reference.reshape(-1))


def _gradient_tuple(
    scalar: torch.Tensor,
    inputs: Sequence[torch.Tensor],
    *,
    retain_graph: bool,
) -> Tuple[torch.Tensor, ...]:
    gradients = torch.autograd.grad(
        scalar, tuple(inputs), retain_graph=retain_graph, allow_unused=True
    )
    return tuple(
        torch.zeros_like(value) if gradient is None else gradient
        for value, gradient in zip(inputs, gradients)
    )


def derivative_certificate(
    descriptor: MagneticPolynomialDescriptor,
    coefficients: torch.Tensor,
    head: np.ndarray,
    teacher: str,
    environments: Sequence[Mapping[str, torch.Tensor]],
) -> Dict[str, Any]:
    predicted = {name: [] for name in ("energy", "force", "mforce", "strain")}
    reference = {name: [] for name in predicted}
    head_tensor = torch.tensor(head, dtype=DTYPE, device=coefficients.device)
    for environment in environments:
        positions, spins, strain, volume, rij = _deformed_inputs(
            environment, require_grad=True
        )
        atomic = _descriptor_atomic(
            descriptor, coefficients, environment, rij, spins
        )
        prediction = (atomic.sum(0) * head_tensor).sum()
        target = raw_teacher_values(
            rij, environment["pair_i"], environment["pair_j"], spins
        )[teacher]
        inputs = (positions, spins, strain, volume)
        target_grad = _gradient_tuple(target, inputs, retain_graph=True)
        prediction_grad = _gradient_tuple(
            prediction, inputs, retain_graph=False
        )
        predicted["energy"].append([float(prediction.detach().cpu())])
        reference["energy"].append([float(target.detach().cpu())])
        predicted["force"].append((-prediction_grad[0]).detach().cpu().numpy())
        reference["force"].append((-target_grad[0]).detach().cpu().numpy())
        predicted["mforce"].append((-prediction_grad[1]).detach().cpu().numpy())
        reference["mforce"].append((-target_grad[1]).detach().cpu().numpy())
        predicted["strain"].append(np.asarray([
            float(prediction_grad[2].detach().cpu()),
            float(prediction_grad[3].detach().cpu()),
        ]))
        reference["strain"].append(np.asarray([
            float(target_grad[2].detach().cpu()),
            float(target_grad[3].detach().cpu()),
        ]))
    return {
        name: relative_vector_error(
            np.concatenate([
                np.asarray(value).reshape(-1) for value in predicted[name]
            ]),
            np.concatenate([
                np.asarray(value).reshape(-1) for value in reference[name]
            ]),
        )
        for name in predicted
    }


def _token_index_map(
    source: MagneticPolynomialDescriptor,
    target: MagneticPolynomialDescriptor,
) -> Sequence[int]:
    target_tokens = {token.name: token for token in target.schema}
    mapping = []
    for source_token in source.schema:
        target_token = target_tokens[source_token.name]
        if source_token.name in {
            "correlation_same_edge", "correlation_distinct_neighbor"
        }:
            source_pairs = source.radial_pairs
            target_pairs = {
                pair: index for index, pair in enumerate(target.radial_pairs)
            }
            mapping.extend(
                target_token.start + target_pairs[pair] for pair in source_pairs
            )
        else:
            mapping.extend(
                target_token.start + index for index in range(source_token.width)
            )
    return mapping


def embedding_certificate(
    order: int,
    environments: Sequence[Mapping[str, torch.Tensor]],
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    source, source_coefficients = build_descriptor(order, 2, device)
    target, target_coefficients = build_descriptor(order, 3, device)
    mapping = _token_index_map(source, target)
    generator = torch.Generator(device=device).manual_seed(seed + order)
    hidden = 7
    source_w0 = torch.randn(
        source.dim, hidden, dtype=DTYPE, device=device, generator=generator
    ) * 0.17
    bias = torch.randn(
        hidden, dtype=DTYPE, device=device, generator=generator
    ) * 0.11
    output = torch.randn(
        hidden, dtype=DTYPE, device=device, generator=generator
    ) * 0.19
    target_w0 = torch.zeros(
        target.dim, hidden, dtype=DTYPE, device=device
    )
    target_w0[torch.tensor(mapping, dtype=torch.long, device=device)] = source_w0

    observed = {
        name: []
        for name in ("descriptor", "energy", "force", "mforce", "strain")
    }
    reference = {name: [] for name in observed}
    for environment in environments:
        positions, spins, strain, volume, rij = _deformed_inputs(
            environment, require_grad=True
        )
        source_q = _descriptor_atomic(
            source, source_coefficients, environment, rij, spins
        )
        target_q = _descriptor_atomic(
            target, target_coefficients, environment, rij, spins
        )
        mapped = target_q.index_select(
            1, torch.tensor(mapping, dtype=torch.long, device=device)
        )
        source_energy = (torch.tanh(source_q @ source_w0 - bias) @ output).sum()
        target_energy = (torch.tanh(target_q @ target_w0 - bias) @ output).sum()
        inputs = (positions, spins, strain, volume)
        source_grad = _gradient_tuple(source_energy, inputs, retain_graph=True)
        target_grad = _gradient_tuple(target_energy, inputs, retain_graph=False)
        observed["descriptor"].append(mapped.detach().cpu().numpy())
        reference["descriptor"].append(source_q.detach().cpu().numpy())
        observed["energy"].append(np.asarray([float(target_energy.detach().cpu())]))
        reference["energy"].append(np.asarray([float(source_energy.detach().cpu())]))
        observed["force"].append((-target_grad[0]).detach().cpu().numpy())
        reference["force"].append((-source_grad[0]).detach().cpu().numpy())
        observed["mforce"].append((-target_grad[1]).detach().cpu().numpy())
        reference["mforce"].append((-source_grad[1]).detach().cpu().numpy())
        observed["strain"].append(np.asarray([
            float(target_grad[2].detach().cpu()),
            float(target_grad[3].detach().cpu()),
        ]))
        reference["strain"].append(np.asarray([
            float(source_grad[2].detach().cpu()),
            float(source_grad[3].detach().cpu()),
        ]))

    errors = {
        name: relative_vector_error(
            np.concatenate([np.asarray(value).reshape(-1) for value in observed[name]]),
            np.concatenate([
                np.asarray(value).reshape(-1) for value in reference[name]
            ]),
        )
        for name in observed
    }
    return {
        "order": order,
        "source_dimension": source.dim,
        "target_dimension": target.dim,
        "source_to_target_columns": [int(index) for index in mapping],
        "radial_parameter_embedding": {
            "source_shape": list(source_coefficients.shape),
            "target_shape": list(target_coefficients.shape),
            "retained_rows": [0, 1],
            "max_abs_error": float(
                (target_coefficients[..., :2, :] - source_coefficients)
                .abs().max().detach().cpu()
            ),
        },
        "leg_parameter_embedding": {
            "retained_block": [[0, 0], [0, 1], [1, 0], [1, 1]],
            "max_abs_error": float(
                (target.radial_leg_mix[:, :2, :2]
                 - source.radial_leg_mix).abs().max().detach().cpu()
            ),
            "cross_block_max_abs": float(
                target.radial_leg_mix[:, :2, 2:].abs().max().detach().cpu()
            ),
        },
        "ann_head_embedding": {
            "hidden_width": hidden,
            "mapped_rows": len(mapping),
            "new_rows_zero": bool(
                torch.count_nonzero(
                    target_w0.index_select(
                        0,
                        torch.tensor(
                            sorted(set(range(target.dim)) - set(mapping)),
                            dtype=torch.long,
                            device=device,
                        ),
                    )
                ).item() == 0
            ),
        },
        "errors": errors,
        "passed": all(
            metric["relative_rms"] < 2.0e-10
            or metric["residual_rms"] < 2.0e-11
            for metric in errors.values()
        ),
    }


def raw_formula_certificate(
    descriptor: MagneticPolynomialDescriptor,
    coefficients: torch.Tensor,
    environments: Sequence[Mapping[str, torch.Tensor]],
) -> Dict[str, Any]:
    """Compare independent raw formulae with named production coordinates."""
    errors: Dict[str, Any] = {}
    for teacher, (token_name, coordinate) in RAW_TOKEN_FOR_TEACHER.items():
        if token_name not in {token.name for token in descriptor.schema}:
            continue
        prediction = []
        reference = []
        for environment in environments:
            _, spins, _, _, rij = _deformed_inputs(environment)
            q = _descriptor_atomic(
                descriptor, coefficients, environment, rij, spins
            )
            token = next(
                item for item in descriptor.schema if item.name == token_name
            )
            prediction.append(float(
                q[:, token.start + coordinate].sum().detach().cpu()
            ))
            raw = raw_teacher_values(
                rij, environment["pair_i"], environment["pair_j"], spins
            )[teacher]
            reference.append(float(raw.detach().cpu()))
        errors[teacher] = relative_vector_error(
            np.asarray(prediction), np.asarray(reference)
        )
    return errors


def token_health(
    descriptor: MagneticPolynomialDescriptor,
    atomic_matrix: np.ndarray,
    environment: Mapping[str, torch.Tensor],
    coefficients: torch.Tensor,
) -> Dict[str, Any]:
    positions, spins, strain, volume, rij = _deformed_inputs(
        environment, require_grad=True
    )
    q = _descriptor_atomic(descriptor, coefficients, environment, rij, spins)
    result = {}
    for token_index, token in enumerate(descriptor.schema):
        block = atomic_matrix[:, token.start:token.stop]
        weights = torch.linspace(
            0.7, 1.3, token.width, dtype=DTYPE, device=coefficients.device
        )
        scalar = (q[:, token.start:token.stop].sum(0) * weights).sum()
        gradients = _gradient_tuple(
            scalar, (positions, spins, strain, volume),
            retain_graph=token_index + 1 < len(descriptor.schema),
        )
        result[token.name] = {
            "metadata": {
                "start": token.start,
                "stop": token.stop,
                "order": token.order,
                "minimum_radial_rank": token.minimum_radial_rank,
                "spin_degree": token.spin_degree,
                "geometry_rank": token.geometry_rank,
                "topology": token.topology,
            },
            "value_rms": float(np.sqrt(np.mean(np.square(block)))),
            "value_max_abs": float(np.max(np.abs(block))),
            "span": svd_health(block, center=True, standardize=False),
            "gradient_rms": {
                "position": float(gradients[0].square().mean().sqrt().detach().cpu()),
                "spin": float(gradients[1].square().mean().sqrt().detach().cpu()),
                "strain": float(gradients[2].abs().detach().cpu()),
                "volume": float(gradients[3].abs().detach().cpu()),
            },
        }
    return result


def _dataset_sha256(
    train: Sequence[Mapping[str, torch.Tensor]],
    blind: Sequence[Mapping[str, torch.Tensor]],
) -> str:
    digest = hashlib.sha256()
    for split, environments in (("train", train), ("blind", blind)):
        digest.update(split.encode() + b"\0")
        for environment in environments:
            for name in ("positions", "spins", "atom_types", "pair_i", "pair_j"):
                value = environment[name].detach().cpu().contiguous().numpy()
                digest.update(name.encode() + b"\0")
                digest.update(value.tobytes())
    return digest.hexdigest()


def run_certificate(
    *,
    profile: str,
    seed: int,
    device: str,
) -> Dict[str, Any]:
    started = time.perf_counter()
    settings = PROFILES[profile]
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device requests CUDA but torch.cuda.is_available() is false"
        )
    train = generate_environments(settings["train"], seed, torch_device)
    blind = generate_environments(settings["blind"], seed + 1000003, torch_device)
    result: Dict[str, Any] = {
        "schema_version": 1,
        "experiment": "oc_descriptor_contract",
        "profile": profile,
        "seed": seed,
        "device": str(torch_device),
        "dtype": str(DTYPE),
        "sample_counts": dict(settings),
        "dataset_sha256": _dataset_sha256(train, blind),
        "expected_reachability": {
            teacher: sorted(models)
            for teacher, models in EXPECTED_REACHABILITY.items()
        },
        "descriptors": {},
        "embedding": {},
    }

    derivative_subset = blind[:settings["derivative"]]
    for model_name, model_settings in DESCRIPTORS.items():
        descriptor, coefficients = build_descriptor(
            model_settings["order"], model_settings["compress"], torch_device
        )
        if descriptor.dim != model_settings["dimension"]:
            raise RuntimeError(
                f"{model_name} dimension drift: {descriptor.dim} != "
                f"{model_settings['dimension']}"
            )
        train_atomic, train_global, train_targets = collect_matrices(
            descriptor, coefficients, train
        )
        blind_atomic, blind_global, blind_targets = collect_matrices(
            descriptor, coefficients, blind
        )
        labels = _column_labels(descriptor)
        fits = {}
        for teacher in TEACHERS:
            fitted = _linear_fit(train_global, train_targets[teacher])
            train_prediction = train_global @ fitted["head"]
            blind_prediction = blind_global @ fitted["head"]
            expected = model_name in EXPECTED_REACHABILITY[teacher]
            threshold = 1.0e-8
            train_error = _scalar_error(train_prediction, train_targets[teacher])
            blind_error = _scalar_error(blind_prediction, blind_targets[teacher])
            derivatives = derivative_certificate(
                descriptor, coefficients, fitted["head"], teacher,
                derivative_subset,
            )
            reached = (
                blind_error["relative_rms"] < threshold
                or blind_error["residual_rms"] < 1.0e-10
            )
            derivative_reached = all(
                metric["relative_rms"] < 2.0e-7
                or metric["residual_rms"] < 2.0e-9
                for metric in derivatives.values()
            )
            fits[teacher] = {
                "expected_reachable": expected,
                "observed_reachable": bool(reached and derivative_reached),
                "matches_expectation": bool(
                    expected == (reached and derivative_reached)
                ),
                "train": train_error,
                "blind": blind_error,
                "blind_derivatives": derivatives,
                "linear_solve_rank": fitted["rank"],
                "linear_solve_active_columns": fitted["active_columns"],
                "head": fitted["head"].tolist(),
            }
        result["descriptors"][model_name] = {
            "order": model_settings["order"],
            "compress": model_settings["compress"],
            "dimension": descriptor.dim,
            "schema": [
                {
                    "name": token.name,
                    "start": token.start,
                    "stop": token.stop,
                    "order": token.order,
                    "minimum_radial_rank": token.minimum_radial_rank,
                }
                for token in descriptor.schema
            ],
            "covariance": {
                "atomic_raw": svd_health(
                    train_atomic, labels=labels, center=True, standardize=False
                ),
                "atomic_standardized": svd_health(
                    train_atomic, labels=labels, center=True, standardize=True
                ),
                "global_raw": svd_health(
                    train_global, labels=labels, center=True, standardize=False
                ),
                "global_standardized": svd_health(
                    train_global, labels=labels, center=True, standardize=True
                ),
                "blind_atomic_raw": svd_health(
                    blind_atomic, labels=labels, center=True, standardize=False
                ),
            },
            "raw_formula_oracle": raw_formula_certificate(
                descriptor, coefficients, blind[: min(8, len(blind))]
            ),
            "token_health": token_health(
                descriptor, train_atomic, blind[0], coefficients
            ),
            "linear_function_span": fits,
        }

    embedding_frames = blind[: min(6, len(blind))]
    for order in (2, 3):
        result["embedding"][f"o{order}_c2_to_c3"] = embedding_certificate(
            order, embedding_frames, torch_device, seed
        )

    all_span_matches = all(
        fit["matches_expectation"]
        for descriptor in result["descriptors"].values()
        for fit in descriptor["linear_function_span"].values()
    )
    all_raw_match = all(
        metric["relative_rms"] < 2.0e-10
        or metric["residual_rms"] < 2.0e-11
        for descriptor in result["descriptors"].values()
        for metric in descriptor["raw_formula_oracle"].values()
    )
    all_embedding = all(
        item["passed"] for item in result["embedding"].values()
    )
    repo = _repository_root()
    result["acceptance"] = {
        "raw_formulae_match_production_coordinates": all_raw_match,
        "reachability_matches_declared_hierarchy": all_span_matches,
        "c2_to_c3_embedding_preserves_ann_and_derivatives": all_embedding,
        "passed": bool(all_raw_match and all_span_matches and all_embedding),
    }
    result["provenance"] = {
        "git": _git_provenance(repo),
        "source": _source_provenance(repo),
        "imports": _local_import_provenance(repo),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(torch_device)
            if torch_device.type == "cuda" else None
        ),
        "command": sys.argv,
    }
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    result = run_certificate(
        profile=arguments.profile,
        seed=arguments.seed,
        device=arguments.device,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "output": str(arguments.output),
        "acceptance": result["acceptance"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, sort_keys=True))
    return 0 if result["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
