# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.
# TorchNEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import math

import numpy as np
import pytest
import torch

from torchnep import ops
from torchnep.data import parse_nep_in
from torchnep.magnetic_polynomial import (
    MagneticPolynomialDescriptor,
    _stf5,
)
from torchnep.magnetic_polynomial_v3 import MagneticPolynomialDescriptorV3
from torchnep.model import NEPModel
from torchnep.nep import NEPCalculator
from torchnep.response_training import balanced_group_step_batches
from torchnep.spin_schema import resolve_gpumd_qscaler
from torchnep.train import (
    magnetic_force_residual,
    prepare_spin3_curriculum_clip,
    refresh_order3_q_scaler,
    reset_order3_optimizer_state,
    restore_spin3_curriculum_clip,
    spin_curriculum_end,
    virial6_for_loss,
)
from torchnep.virial import virial9_to_6_numpy


DTYPE = torch.float64


def _config(order=3, compress=2, soc=1):
    return {
        "num_types": 2,
        "type_names": ["Fe", "Ge"],
        "cutoff_radial": 6.0,
        "cutoff_angular": 5.0,
        "n_max_radial": 3,
        "n_max_angular": 3,
        "basis_size_radial": 6,
        "basis_size_angular": 6,
        "l_max": [3, 2, 0],
        "neuron": 8,
        "spin_mode": 3,
        "spin_compress": compress,
        "spin_basis_size": [8, 0],
        "spin_l_max": [2, 0, 0],
        "spin_cutoff": [6.0, 6.0],
        "spin_order": order,
        "spin_soc": soc,
        "spin_curriculum": False,
        "spin_dof_type": ["Fe", "Ge"],
        "spin_env_type": ["Fe", "Ge"],
    }


def _environment(n_atoms=6, seed=20260903):
    generator = torch.Generator().manual_seed(seed)
    positions = torch.randn(n_atoms, 3, generator=generator, dtype=DTYPE)
    pair_i = []
    pair_j = []
    rij = []
    for center in range(n_atoms):
        for neighbor in range(n_atoms):
            if center != neighbor:
                pair_i.append(center)
                pair_j.append(neighbor)
                rij.append(positions[neighbor] - positions[center])
    return {
        "rij": torch.stack(rij),
        "pair_i": torch.tensor(pair_i, dtype=torch.long),
        "pair_j": torch.tensor(pair_j, dtype=torch.long),
        "spins": torch.randn(n_atoms, 3, generator=generator, dtype=DTYPE),
        "N": n_atoms,
    }


def _rotation(seed=41, reflection=False):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(3, 3, generator=generator, dtype=DTYPE)
    rotation, _ = torch.linalg.qr(matrix)
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    if reflection:
        rotation = torch.diag(torch.tensor(
            [-1.0, 1.0, 1.0], dtype=DTYPE
        )) @ rotation
    return rotation


def test_spin3_parser_and_fixed_production_dimensions(tmp_path):
    path = tmp_path / "nep.in"
    path.write_text(
        "type 2 Fe Ge\n"
        "spin_mode 3\n"
        "spin_order 3\n"
        "spin_compress 2\n"
        "spin_l_max 2\n"
        "spin_soc 1\n"
        "spin_mforce_mode 1\n"
    )
    config = parse_nep_in(path)
    model = NEPModel(config)
    assert config["spin_mode"] == 3
    assert model.spin_descriptor_kind == "magnetic_polynomial_v3"
    assert model.spin_descriptor.order_dims == {1: 9, 2: 43, 3: 55}
    assert model.spin_dim == 55
    assert config["spin_mforce_mode"] == "transverse"
    assert resolve_gpumd_qscaler(3, None) is False
    with pytest.raises(ValueError, match="spin_mode 3"):
        resolve_gpumd_qscaler(3, True)


def test_spin3_descriptor_width_is_independent_of_element_count():
    dimensions = []
    type_names = ["Fe", "Ge", "Ni", "Co"]
    for num_types in (1, 2, 4):
        config = _config()
        config.update({
            "num_types": num_types,
            "type_names": type_names[:num_types],
            "spin_dof_type": type_names[:num_types],
            "spin_env_type": type_names[:num_types],
        })
        model = NEPModel(config)
        dimensions.append((model.spin_dim, model.dim))
        assert model.c_param_spin.shape == (
            1,
            num_types,
            num_types,
            model.spin_compress,
            model.spin_basis_size[0] + 1,
        )
        assert len(model.fitting_nets) == num_types

    assert dimensions == [dimensions[0]] * 3


def test_spin3_ordered_pair_mixing_breaks_a_type_blind_collision():
    descriptor = MagneticPolynomialDescriptorV3(
        spin_compress=2,
        spin_l_max=1,
        spin_order=2,
        spin_soc=False,
    ).to(dtype=DTYPE)
    rij = torch.tensor(
        [[1.25, 0.0, 0.0], [1.25, 0.0, 0.0]], dtype=DTYPE
    )
    pair_i = torch.tensor([0, 0], dtype=torch.long)
    pair_j = torch.tensor([1, 2], dtype=torch.long)
    atom_types = torch.tensor([0, 0, 1], dtype=torch.long)
    spins_a = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=DTYPE,
    )
    spins_b = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=DTYPE,
    )
    tied = torch.zeros(1, 2, 2, 2, 9, dtype=DTYPE)
    tied[..., 0, 0] = 1.0
    tied[..., 1, 1] = 1.0

    def evaluate(spins, coefficient):
        return descriptor(
            rij,
            pair_i,
            pair_j,
            spins,
            3,
            atom_types,
            coefficient,
        )[0]

    # With one shared pair table, exchanging the two equal-geometry neighbor
    # spins is an exact descriptor collision.
    torch.testing.assert_close(
        evaluate(spins_a, tied),
        evaluate(spins_b, tied),
        rtol=0.0,
        atol=0.0,
    )

    resolved = tied.clone()
    resolved[:, 0, 1] *= 1.7
    assert not torch.allclose(
        evaluate(spins_a, resolved),
        evaluate(spins_b, resolved),
    )


def test_spin3_mforce_contract_is_versioned_and_validated(tmp_path):
    missing = tmp_path / "missing.nep.in"
    missing.write_text("type 1 Fe\nspin_mode 3\n")
    assert parse_nep_in(missing)["spin_mforce_mode"] == "full"

    invalid = tmp_path / "invalid.nep.in"
    invalid.write_text("type 1 Fe\nspin_mode 3\nspin_mforce_mode tangent\n")
    with pytest.raises(ValueError, match="full.*transverse"):
        parse_nep_in(invalid)

    spin2 = tmp_path / "spin2.nep.in"
    spin2.write_text("type 1 Fe\nspin_mode 2\nspin_mforce_mode 1\n")
    with pytest.raises(ValueError, match="spin_mode 3 training contract"):
        parse_nep_in(spin2)


@pytest.mark.parametrize("value, expected", [("0", "full"), ("1", "transverse")])
def test_spin3_mforce_numeric_input_and_summary(tmp_path, value, expected):
    from torchnep.train import format_config_summary

    path = tmp_path / "nep.in"
    path.write_text(f"type 1 Fe\nspin_mode 3\nspin_mforce_mode {value}\n")
    config = parse_nep_in(path)
    assert config["spin_mforce_mode"] == expected
    assert any(f"spin_mforce_mode {value}" in line
               for line in format_config_summary(config))


@pytest.mark.parametrize("value", ["2", "-1", "full", "transverse", "", "0 1"])
def test_spin3_mforce_rejects_invalid_input(tmp_path, value):
    path = tmp_path / "nep.in"
    path.write_text(f"type 1 Fe\nspin_mode 3\nspin_mforce_mode {value}\n")
    with pytest.raises(ValueError, match="0 .*full.*1 .*transverse"):
        parse_nep_in(path)


def test_spin3_o3c2_schema_is_frozen():
    descriptor = MagneticPolynomialDescriptorV3().to(dtype=DTYPE)
    expected = [
        ("local_s2", 0, 1),
        ("edge_l0_dot", 1, 3),
        ("edge_l0_neighbor_s2", 3, 5),
        ("edge_l2_pair", 5, 7),
        ("center_l2_environment", 7, 9),
        ("edge_l0_dot2", 9, 11),
        ("density_l0_self", 11, 13),
        ("density_l1_longitudinal_self", 13, 15),
        ("density_l1_axial_self", 15, 17),
        ("density_l1_stf_self", 17, 19),
        ("density_l2_product_self", 19, 21),
        ("density_l0_dot_response", 21, 23),
        ("correlation_same_edge", 23, 26),
        ("correlation_distinct_l0", 26, 29),
        ("correlation_distinct_l1", 29, 32),
        ("correlation_distinct_l2", 32, 35),
        ("coupling_l11_axial", 35, 37),
        ("edge_l11_axial", 37, 39),
        ("coupling_l22_axial", 39, 41),
        ("edge_l22_axial", 41, 43),
        ("edge_l0_moment_gate", 43, 45),
        ("coupling_l11_dot_response", 45, 47),
        ("coupling_l111_p_m_x", 47, 49),
        ("coupling_l22_dot_response", 49, 51),
        ("coupling_l111_p_qs_x", 51, 53),
        ("coupling_l112_edge_response", 53, 55),
    ]
    assert [(x.name, x.start, x.stop) for x in descriptor.schema] == expected


@pytest.mark.parametrize("compress", (1, 2, 3))
def test_spin3_orders_are_literal_prefixes(compress):
    environment = _environment()
    descriptors = [
        MagneticPolynomialDescriptorV3(
            spin_compress=compress,
            spin_order=order,
        ).to(dtype=DTYPE)
        for order in (1, 2, 3)
    ]
    outputs = [descriptor(**environment) for descriptor in descriptors]
    torch.testing.assert_close(
        outputs[1][..., :descriptors[0].dim], outputs[0]
    )
    torch.testing.assert_close(
        outputs[2][..., :descriptors[1].dim], outputs[1]
    )


def test_cartesian_angular_exchange_matches_direct_neighbor_pair_oracle():
    descriptor = MagneticPolynomialDescriptorV3().to(dtype=DTYPE)
    environment = _environment(n_atoms=5, seed=17)
    atom_types = torch.zeros(environment["N"], dtype=torch.long)
    coefficient = torch.randn(1, 1, 1, 2, 9, dtype=DTYPE)
    active = torch.ones(1, dtype=torch.bool)
    result = descriptor(
        **environment,
        atom_types=atom_types,
        c_spin_2body=coefficient,
        spin_dof_type_active=active,
        spin_env_type_active=active,
    )
    blocks = descriptor.split_by_token(result)

    rij = environment["rij"]
    pair_i = environment["pair_i"]
    pair_j = environment["pair_j"]
    spins = environment["spins"]
    distance = torch.linalg.vector_norm(rij, dim=-1)
    rhat = rij / distance.unsqueeze(-1)
    weights, _ = descriptor.core._radial_weights(
        distance, pair_i, pair_j, atom_types, coefficient
    )
    dot = (spins[pair_i] * spins[pair_j]).sum(-1)
    radial_pairs = ((0, 0), (0, 1), (1, 1))
    for ell in range(3):
        expected = torch.zeros(environment["N"], 3, dtype=DTYPE)
        for edge in range(pair_i.numel()):
            for other in range(pair_i.numel()):
                if pair_i[edge] != pair_i[other] or edge == other:
                    continue
                cosine = (rhat[edge] * rhat[other]).sum()
                legendre = (
                    torch.ones_like(cosine)
                    if ell == 0
                    else cosine
                    if ell == 1
                    else 0.5 * (3.0 * cosine.square() - 1.0)
                )
                for channel, (left, right) in enumerate(radial_pairs):
                    expected[pair_i[edge], channel] += (
                        weights[edge, left]
                        * weights[other, right]
                        * dot[edge]
                        * dot[other]
                        * legendre
                    )
        torch.testing.assert_close(
            blocks[f"correlation_distinct_l{ell}"], expected,
            rtol=2.0e-11, atol=2.0e-11,
        )


def test_low_cost_cartesian_contractions_equal_legendre_kernels():
    generator = torch.Generator().manual_seed(19)
    left = torch.randn(31, 3, generator=generator, dtype=DTYPE)
    right = torch.randn(31, 3, generator=generator, dtype=DTYPE)
    left /= torch.linalg.vector_norm(left, dim=-1, keepdim=True)
    right /= torch.linalg.vector_norm(right, dim=-1, keepdim=True)
    cosine = (left * right).sum(-1)
    rank1 = (left * right).sum(-1)
    rank2 = 1.5 * (
        _stf5(left, left) * _stf5(right, right)
    ).sum(-1)
    torch.testing.assert_close(rank1, cosine)
    torch.testing.assert_close(
        rank2, 0.5 * (3.0 * cosine.square() - 1.0),
        rtol=2.0e-13, atol=2.0e-13,
    )


@pytest.mark.parametrize(
    "compress,lmax,order,soc",
    ((1, 0, 2, 0), (2, 1, 2, 0), (2, 2, 3, 1), (3, 2, 3, 1)),
)
def test_spin3_legacy_projection_is_exact_spin2(compress, lmax, order, soc):
    environment = _environment(n_atoms=5, seed=43 + compress + lmax)
    spin2 = MagneticPolynomialDescriptor(
        spin_compress=compress,
        spin_l_max=lmax,
        spin_order=order,
        spin_soc=bool(soc),
    ).to(dtype=DTYPE)
    spin3 = MagneticPolynomialDescriptorV3(
        spin_compress=compress,
        spin_l_max=lmax,
        spin_order=order,
        spin_soc=bool(soc),
    ).to(dtype=DTYPE)
    with torch.no_grad():
        spin3.radial_leg_mix.copy_(spin2.radial_leg_mix)
    q2 = spin2(**environment)
    q3 = spin3(**environment)
    parts = spin3.split_by_token(q3)
    projection = torch.cat([
        parts[
            "correlation_distinct_l0"
            if token.name == "correlation_distinct_neighbor"
            else token.name
        ]
        for token in spin2.schema
    ], dim=-1)
    torch.testing.assert_close(projection, q2, rtol=2.0e-12, atol=2.0e-12)


def test_spin3_is_invariant_to_directed_neighbor_storage_order():
    descriptor = MagneticPolynomialDescriptorV3().to(dtype=DTYPE)
    environment = _environment(n_atoms=6, seed=67)
    reference = descriptor(**environment)
    generator = torch.Generator().manual_seed(68)
    order = torch.randperm(environment["pair_i"].numel(), generator=generator)
    candidate = descriptor(
        environment["rij"].index_select(0, order),
        environment["pair_i"].index_select(0, order),
        environment["pair_j"].index_select(0, order),
        environment["spins"],
        environment["N"],
    )
    torch.testing.assert_close(candidate, reference, rtol=2.0e-12, atol=2.0e-12)


def test_spin3_closes_the_collinear_s4_angle_blind_spot():
    descriptor = MagneticPolynomialDescriptorV3().to(dtype=DTYPE)
    pair_i = torch.tensor([0, 0], dtype=torch.long)
    pair_j = torch.tensor([1, 2], dtype=torch.long)
    spins = torch.tensor(
        [[0.0, 0.0, 2.0], [0.0, 0.0, 1.3], [0.0, 0.0, -1.7]],
        dtype=DTYPE,
    )

    def at(angle):
        rij = torch.tensor([
            [2.5, 0.0, 0.0],
            [2.5 * math.cos(angle), 2.5 * math.sin(angle), 0.0],
        ], dtype=DTYPE)
        return descriptor.split_by_token(
            descriptor(rij, pair_i, pair_j, spins, 3)
        )

    angle60 = at(math.pi / 3.0)
    angle90 = at(math.pi / 2.0)
    angle120 = at(2.0 * math.pi / 3.0)
    assert (angle60["correlation_distinct_l1"]
            - angle120["correlation_distinct_l1"]).abs().max() > 1.0e-3
    assert (angle60["correlation_distinct_l2"]
            - angle90["correlation_distinct_l2"]).abs().max() > 1.0e-3


@pytest.mark.parametrize("reflection", (False, True))
def test_spin3_joint_o3_and_time_reversal_invariance(reflection):
    descriptor = MagneticPolynomialDescriptorV3().to(dtype=DTYPE)
    environment = _environment(seed=29)
    reference = descriptor(**environment)
    rotation = _rotation(reflection=reflection)
    determinant = torch.linalg.det(rotation)
    transformed = descriptor(
        **{
            **environment,
            "rij": environment["rij"] @ rotation.T,
            "spins": determinant * environment["spins"] @ rotation.T,
        }
    )
    reversed_spin = descriptor(
        **{**environment, "spins": -environment["spins"]}
    )
    torch.testing.assert_close(
        transformed, reference, rtol=4.0e-11, atol=4.0e-11
    )
    torch.testing.assert_close(
        reversed_spin, reference, rtol=4.0e-11, atol=4.0e-11
    )


def test_spin3_declared_parities_hold_across_schema_families():
    environment = _environment(n_atoms=4, seed=37)
    for compress in (1, 2, 3):
        for lmax in (0, 1, 2):
            for order in (1, 2, 3):
                for soc in (False, True):
                    descriptor = MagneticPolynomialDescriptorV3(
                        spin_compress=compress,
                        spin_l_max=lmax,
                        spin_order=order,
                        spin_soc=soc,
                    ).to(dtype=DTYPE)
                    reference = descriptor(**environment)
                    inverted = descriptor(
                        **{**environment, "rij": -environment["rij"]}
                    )
                    reversed_spin = descriptor(
                        **{**environment, "spins": -environment["spins"]}
                    )
                    scale = float(reference.detach().abs().max()) + 1.0
                    for token in descriptor.schema:
                        span = slice(token.start, token.stop)
                        torch.testing.assert_close(
                            inverted[..., span],
                            token.inversion_parity * reference[..., span],
                            rtol=0.0,
                            atol=2.0e-11 * scale,
                            msg=f"{token.name} inversion parity",
                        )
                        torch.testing.assert_close(
                            reversed_spin[..., span],
                            token.time_reversal_parity
                            * reference[..., span],
                            rtol=0.0,
                            atol=2.0e-11 * scale,
                            msg=f"{token.name} time-reversal parity",
                        )


def test_spin3_new_sector_has_spin_degree_four_and_zero_spin_limit():
    descriptor = MagneticPolynomialDescriptorV3().to(dtype=DTYPE)
    environment = _environment(seed=53)
    reference = descriptor(**environment)
    scaled = descriptor(
        **{**environment, "spins": 1.7 * environment["spins"]}
    )
    parts = descriptor.split_by_token(reference)
    scaled_parts = descriptor.split_by_token(scaled)
    for ell in range(3):
        name = f"correlation_distinct_l{ell}"
        torch.testing.assert_close(
            scaled_parts[name], 1.7 ** 4 * parts[name],
            rtol=3.0e-11, atol=3.0e-11,
        )
    zero = descriptor(
        **{**environment, "spins": torch.zeros_like(environment["spins"])}
    )
    torch.testing.assert_close(zero, torch.zeros_like(zero))


def test_spin3_full_gradient_and_directional_finite_difference():
    descriptor = MagneticPolynomialDescriptorV3().to(dtype=DTYPE)
    environment = _environment(n_atoms=4, seed=71)
    rij = environment["rij"].clone().requires_grad_(True)
    spins = environment["spins"].clone().requires_grad_(True)
    coefficient = torch.randn(
        1, 1, 1, 2, 9, dtype=DTYPE, requires_grad=True
    )
    atom_types = torch.zeros(environment["N"], dtype=torch.long)
    head = torch.linspace(-0.2, 0.3, descriptor.dim, dtype=DTYPE)

    def energy(rij_value, spin_value, coefficient_value):
        q = descriptor(
            rij_value, environment["pair_i"], environment["pair_j"],
            spin_value, environment["N"], atom_types, coefficient_value,
        )
        return (q * head).sum()

    value = energy(rij, spins, coefficient)
    gradients = torch.autograd.grad(value, (rij, spins, coefficient))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    generator = torch.Generator().manual_seed(83)
    directions = tuple(
        torch.randn(x.shape, generator=generator, dtype=DTYPE)
        for x in (rij, spins, coefficient)
    )
    analytic = sum((g * d).sum() for g, d in zip(gradients, directions))
    step = 2.0e-6
    plus = energy(*(x + step * d for x, d in zip(
        (rij.detach(), spins.detach(), coefficient.detach()), directions
    )))
    minus = energy(*(x - step * d for x, d in zip(
        (rij.detach(), spins.detach(), coefficient.detach()), directions
    )))
    finite_difference = (plus - minus) / (2.0 * step)
    torch.testing.assert_close(
        analytic, finite_difference, rtol=2.0e-6, atol=2.0e-6
    )


def test_spin3_radial_initialization_is_one_continuous_moment_family():
    config3 = _config(compress=4)
    model3 = NEPModel(config3).double()
    distance = torch.linspace(
        0.0, model3.spin_cutoff[0] * (1.0 - 1.0e-7), 257, dtype=DTYPE
    )
    basis = ops.chebyshev_basis(
        distance, model3.spin_cutoff[0], model3.spin_basis_size[0]
    )
    coefficient = model3.c_param_spin[0, 0, 0]
    radial = basis @ coefficient.T
    x = distance / model3.spin_cutoff[0]
    envelope = 0.5 * (torch.cos(torch.pi * x) + 1.0)
    expected = torch.stack(
        [x.pow(degree) * envelope for degree in range(4)], dim=-1
    )
    # The fixed B8 basis is a least-squares representation of these target
    # moments rather than a monomial basis; lock the intended approximation
    # error and prove the c=2 row follows x**2*fc, not the old (x*fc)**2 row.
    torch.testing.assert_close(radial, expected, rtol=0.0, atol=1.5e-3)
    target_error = (radial[:, 2] - x.square() * envelope).square().mean()
    legacy_error = (
        radial[:, 2] - (x * envelope).square()
    ).square().mean()
    assert target_error * 1000.0 < legacy_error

    config2 = {**_config(compress=2), "spin_mode": 2}
    model2 = NEPModel(config2).double()
    model3_c2 = NEPModel(_config(compress=2)).double()
    torch.testing.assert_close(
        model3_c2.c_param_spin, model2.c_param_spin,
        # Two independent LAPACK least-squares calls may differ by a few
        # binary64 ulps on GPU-node CPU libraries even for identical targets.
        rtol=0.0, atol=1.0e-14,
    )


def test_spin3_rms_range_scaler_is_zero_preserving_and_bounded():
    model = NEPModel(_config()).double()
    q_min = torch.zeros(model.dim, dtype=DTYPE)
    q_max = torch.full((model.dim,), 2.0, dtype=DTYPE)
    q_rms = torch.ones(model.dim, dtype=DTYPE)
    start = model.struct_dim
    q_min[start] = q_max[start] = 20.0
    q_rms[start] = 20.0
    q_min[start + 1] = -2.0
    q_max[start + 1] = 2.0
    q_rms[start + 1] = 10.0
    q_min[start + 2] = 0.0
    q_max[start + 2] = 1.0e-12
    q_rms[start + 2] = 1.0e-12
    q_min[start + 3] = 0.0
    q_max[start + 3] = 20.0
    q_rms[start + 3] = 1.0e-4
    q_min[start + 4] = q_max[start + 4] = 0.0
    q_rms[start + 4] = 0.0
    model.set_q_scaler(q_min, q_max, q_rms)
    assert model.q_scaler[0].item() == pytest.approx(0.5)
    assert model.q_scaler[start].item() == pytest.approx(0.05)
    assert model.q_scaler[start + 1].item() == pytest.approx(0.1)
    assert model.q_scaler[start + 2].item() == pytest.approx(1.0e12)
    assert model.q_scaler[start + 3].item() == pytest.approx(0.05)
    assert model.q_scaler[start + 4].item() == pytest.approx(1.0)
    assert (q_max[start + 3] * model.q_scaler[start + 3]).item() \
        == pytest.approx(1.0)
    assert (q_max[start + 2] * model.q_scaler[start + 2]).item() \
        == pytest.approx(1.0)
    zero_spin = torch.zeros(3, model.spin_dim, dtype=DTYPE)
    torch.testing.assert_close(
        zero_spin * model.q_scaler[start:], zero_spin
    )


def test_spin3_curriculum_uses_one_gate_and_refreshes_safely():
    config = _config()
    config["spin_curriculum"] = True
    model = NEPModel(config).double()
    start = model.spin_order3_input_start

    weight = model.fitting_nets[0].w0
    model.set_spin_curriculum_scale(0.25)
    sum(net.w0.sum() for net in model.fitting_nets).backward()
    model.apply_spin_curriculum_gradients()
    torch.testing.assert_close(
        weight.grad[start:], torch.ones_like(weight.grad[start:])
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3, amsgrad=True)
    optimizer.step()
    with torch.no_grad():
        for net in model.fitting_nets:
            net.w0[start:].zero_()
    old_scaler = model.q_scaler.clone()
    q_min = torch.zeros(model.dim, dtype=DTYPE)
    q_max = torch.full((model.dim,), 2.0, dtype=DTYPE)
    q_rms = torch.ones(model.dim, dtype=DTYPE)
    q_rms[start:] = 4.0
    q_max[start:] = 8.0
    assert refresh_order3_q_scaler(model, q_min, q_max, q_rms)
    torch.testing.assert_close(model.q_scaler[:start], old_scaler[:start])
    torch.testing.assert_close(
        model.q_scaler[start:], torch.full_like(model.q_scaler[start:], 0.125)
    )

    reset_order3_optimizer_state(model, optimizer)
    for net in model.fitting_nets:
        state = optimizer.state[net.w0]
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            torch.testing.assert_close(
                state[name][start:], torch.zeros_like(state[name][start:])
            )


def test_spin3_frozen_rows_do_not_clip_active_curriculum_gradients():
    config = _config()
    config["spin_curriculum"] = True
    model = NEPModel(config).double()
    start = model.spin_order3_input_start
    model.set_spin_curriculum_scale(0.0)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    weight = model.fitting_nets[0].w0
    weight.grad[0, 0] = 3.0
    weight.grad[start:, 0] = 100.0

    saved = prepare_spin3_curriculum_clip(model)
    effective_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    restore_spin3_curriculum_clip(model, saved, effective_norm, 10.0)

    assert effective_norm.item() == pytest.approx(3.0)
    assert weight.grad[0, 0].item() == pytest.approx(3.0)
    torch.testing.assert_close(
        weight.grad[start:, 0],
        torch.full_like(weight.grad[start:, 0], 100.0),
    )


def test_spin3_curriculum_precedes_stage2_without_changing_spin2():
    assert spin_curriculum_end(2, 100, True, 50) == 50
    assert spin_curriculum_end(3, 100, True, 50) == 40
    assert spin_curriculum_end(3, 100, False, None) == 66


def test_spin3_virial_loss_uses_low_cost_symmetric_contraction():
    virial = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    ], dtype=DTYPE)
    expected = torch.tensor(
        [[1.0, 5.0, 9.0, 3.0, 7.0, 5.0]], dtype=DTYPE
    )
    torch.testing.assert_close(virial6_for_loss(virial, 3), expected)
    changed = virial.clone()
    changed[:, 1] += 11.0
    changed[:, 3] -= 11.0
    torch.testing.assert_close(
        virial6_for_loss(changed, 3), virial6_for_loss(virial, 3)
    )
    assert not torch.equal(
        virial6_for_loss(changed, 2), virial6_for_loss(virial, 2)
    )


def test_spin3_prediction_virial_matches_label_projection():
    virial = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    ])
    np.testing.assert_array_equal(
        virial9_to_6_numpy(virial, 2),
        np.array([[1.0, 5.0, 9.0, 2.0, 6.0, 7.0]]),
    )
    np.testing.assert_array_equal(
        virial9_to_6_numpy(virial, 3),
        np.array([[1.0, 5.0, 9.0, 3.0, 7.0, 5.0]]),
    )


def test_spin3_transverse_mforce_contract_and_intrinsic_dof():
    prediction = torch.tensor(
        [[3.0, 4.0, 5.0], [1.0, 2.0, 3.0]], dtype=DTYPE
    )
    reference = torch.zeros_like(prediction)
    spins = torch.tensor(
        [[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]], dtype=DTYPE
    )
    full, full_dof, full_defined = magnetic_force_residual(
        prediction, reference, spins, "full"
    )
    torch.testing.assert_close(full, prediction)
    assert full_dof == 3
    assert full_defined.tolist() == [True, True]

    transverse, transverse_dof, transverse_defined = magnetic_force_residual(
        prediction, reference, spins, "transverse"
    )
    torch.testing.assert_close(
        transverse[0], torch.tensor([3.0, 4.0, 0.0], dtype=DTYPE)
    )
    torch.testing.assert_close(transverse[1], torch.zeros(3, dtype=DTYPE))
    assert transverse_dof == 2
    assert transverse_defined.tolist() == [True, False]


@pytest.mark.parametrize("num_steps", (2, 5))
def test_spin3_response_groups_are_exactly_balanced_per_epoch(num_steps):
    group_ids = [0, 0, 1, 1, 2, 2]
    schedule = balanced_group_step_batches(
        group_ids, num_steps=num_steps, seed=7
    )
    assert len(schedule) == num_steps
    totals = {group: 0.0 for group in set(group_ids)}
    total_weight = 0.0
    for step in schedule:
        for indices, weight in step:
            groups = {group_ids[index] for index in indices}
            contribution = weight / len(groups)
            for group in groups:
                totals[group] += contribution
            total_weight += weight
    for value in totals.values():
        assert value == pytest.approx(num_steps / 3.0)
    assert total_weight == pytest.approx(num_steps)


def test_nep4_spin3_round_trip(tmp_path):
    torch.manual_seed(101)
    source = NEPModel(_config()).double()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.05)
        source.q_scaler.copy_(
            torch.linspace(0.3, 1.3, source.dim, dtype=DTYPE)
        )
        source.energy_baseline.copy_(
            torch.tensor([-0.4, 0.2], dtype=DTYPE)
        )
    path = tmp_path / "nep4_spin3.txt"
    source.save_nep_txt(path, max_NN_radial=32, max_NN_angular=32)
    assert path.read_text().startswith("nep4_spin3 2 Fe Ge\n")

    restored = NEPModel(_config()).double()
    restored.load_weights_from_nep_txt(path)
    for name, expected in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected)

    calculator = NEPCalculator(path, dtype=DTYPE)
    assert calculator.spin_mode == 3
    assert calculator.spin_dim == 55
    assert calculator.spin_descriptor_kind == "magnetic_polynomial_v3"
