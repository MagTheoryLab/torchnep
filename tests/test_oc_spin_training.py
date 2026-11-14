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

import copy

import numpy as np
import pytest
import torch

from torchnep.data import parse_nep_in, read_xyz
from torchnep.model import NEPModel
from torchnep.response_training import (
    GENERATOR_CURVE_SCHEMA,
    complete_group_batches,
    generator_curve_loss,
    group_balanced_generator_rms,
    normalized_group_batches,
    select_generator_curve_frames,
)
from torchnep.spin_diagnostics import (
    generator_reweighting_health,
    relative_vector_error,
    svd_health,
)
from torchnep.train import (
    _load_finetune_weights,
    build_data_lineage,
    fit_energy_baseline_frames,
    grouped_validation_indices,
    validate_resume_data_lineage,
)


def _config():
    return {
        "num_types": 1,
        "type_names": ["Fe"],
        "cutoff_radial": 6.0,
        "cutoff_angular": 5.0,
        "n_max_radial": 4,
        "n_max_angular": 4,
        "basis_size_radial": 8,
        "basis_size_angular": 8,
        "l_max": [4, 2, 0],
        "neuron": 8,
        "model": "nep_spin",
        "spin_compress": 2,
        "spin_basis_size": [8, 0],
        "spin_l_max": [2, 0, 0],
        "spin_cutoff": [6.0],
        "spin_order": 3,
        "spin_soc": 1,
    }


def _rotation_frame(coordinate, group="g"):
    spin = np.array([[np.cos(coordinate), np.sin(coordinate), 0.0]])
    return {
        "natoms": 1,
        "species": ["Fe"],
        "positions": np.array([[1.0, 1.0, 1.0]]),
        "cell": np.diag([6.0, 6.0, 6.0]),
        "spins": spin,
        "response_group": group,
        "response_probe": "rotation",
        "response_coordinate": coordinate,
        "mforces": np.zeros((1, 3)),
    }


def test_svd_health_separates_constant_and_rank_deficient_columns():
    x = np.linspace(-1.0, 1.0, 41)
    full = np.column_stack((x, x * x, np.sin(2.3 * x), np.ones_like(x)))
    healthy = svd_health(full)
    assert healthy["active_features"] == 3
    assert healthy["constant_features"] == 1
    assert healthy["effective_rank"] == 3
    assert healthy["full_column_rank"] is True
    assert np.isfinite(healthy["condition_number"])

    duplicate = np.column_stack((x, x * x, x))
    deficient = svd_health(duplicate)
    assert deficient["effective_rank"] == 2
    assert deficient["full_column_rank"] is False
    assert deficient["condition_number"] is None
    assert np.isfinite(deficient["nonzero_condition_number"])


def test_jacobian_health_does_not_center_linear_map_columns():
    jacobian = np.column_stack((np.ones(7), np.arange(7, dtype=float)))
    health = svd_health(jacobian, center=False)
    assert health["active_features"] == 2
    assert health["effective_rank"] == 2
    assert health["full_column_rank"] is True

    error = relative_vector_error(
        np.array([1.0, 2.0]), np.array([1.0, 2.1]))
    assert error["finite"] is True
    assert error["relative_rms"] > 0.0


def test_generator_reweighting_is_rank_neutral_and_directional():
    coordinates = np.array([-0.2, 0.0, 0.2, -0.2, 0.0, 0.2])
    tangent = np.column_stack((
        -np.sin(coordinates), np.cos(coordinates), np.zeros(6)
    ))
    target = np.column_stack((
        np.cos(coordinates), np.sin(coordinates), np.zeros(6)
    ))
    struct_idx = np.arange(6, dtype=np.int64)
    groups = np.array([0, 0, 0, 1, 1, 1])

    standard = generator_reweighting_health(
        tangent, target, struct_idx, groups,
        mforce_weight=1.0, generator_weight=0.0,
    )
    generator = generator_reweighting_health(
        tangent, target, struct_idx, groups,
        mforce_weight=1.0, generator_weight=0.3,
    )
    generator_only = generator_reweighting_health(
        tangent, target, struct_idx, groups,
        mforce_weight=0.0, generator_weight=0.3,
    )

    assert standard["strict_observation_rank_gain"] == 0
    assert generator["strict_observation_rank_gain"] == 0
    assert generator_only["strict_observation_rank_gain"] == 6
    assert generator_only["complete_mforce_supervision"] is False
    assert generator["generator_operator"]["effective_rank"] == 6
    assert generator["ordinary_mforce_components"] == 18
    assert standard["aligned_curvature_gain_min"] == 1.0
    assert standard["aligned_curvature_gain_max"] == 1.0
    assert generator["orthogonal_curvature_gain"] == 1.0
    assert generator["aligned_curvature_gain_min"] > 1.0
    assert generator["aligned_curvature_gain_max"] >= \
        generator["aligned_curvature_gain_min"]


def test_generator_reweighting_rejects_incomplete_structure_map():
    tangent = np.ones((2, 3))
    target = np.ones((2, 3))
    with pytest.raises(ValueError, match="every response structure"):
        generator_reweighting_health(
            tangent, target, np.array([0, 0]), np.array([0, 1]),
            mforce_weight=1.0, generator_weight=0.3,
        )




def test_spin_defaults_match_gpumd_c516(tmp_path):
    path = tmp_path / "nep.in"
    path.write_text(
        "type 1 Fe\n"
        "cutoff 6 5\n"
        "n_max 4 4\n"
        "basis_size 8 8\n"
        "l_max 4 2 0\n"
        "model nep_spin\nspin_version 1\n"
    )
    config = parse_nep_in(path)
    assert config["spin_order"] == 2
    assert config["spin_compress"] == 3
    assert config["spin_l_max"] == [2, 0, 0]
    assert config["spin_soc"] == 1
    assert "spin_curriculum" not in config
    assert NEPModel(config).spin_dim == 34










def test_energy_baseline_is_per_element_least_squares():
    frames = [
        {"species": ["Fe", "Fe"], "energy": -4.0},
        {"species": ["Ge"], "energy": -1.0},
        {"species": ["Fe", "Ge"], "energy": -3.0},
    ]
    baseline = fit_energy_baseline_frames(frames, ["Fe", "Ge"])
    np.testing.assert_allclose(baseline, [-2.0, -1.0], atol=1.0e-12)


def test_response_groups_are_never_split_between_train_and_validation():
    frames = [
        {"response_group": "a"},
        {"response_group": "a"},
        {"response_group": "b"},
        {"response_group": "b"},
        {},
        {},
    ]
    validation = set(grouped_validation_indices(frames, 0.34, 17))
    for group in ("a", "b"):
        members = {
            index
            for index, frame in enumerate(frames)
            if frame.get("response_group") == group
        }
        assert members <= validation or members.isdisjoint(validation)


def test_response_metadata_and_spin_targets_round_trip_from_extxyz(tmp_path):
    path = tmp_path / "response.xyz"
    path.write_text(
        "1\n"
        'Lattice="6 0 0 0 6 0 0 0 6" energy=-2 '
        "response_group=cell0 response_probe=rotation "
        "response_coordinate=0.01 response_branch=strain "
        "response_family=parent0 response_path_coordinate=-0.02 "
        "Properties=species:S:1:pos:R:3:spin:R:3:mforce:R:3\n"
        "Fe 0 0 0 0 0 2 0.1 0.2 0.3\n"
    )
    frame = read_xyz(path)[0]
    assert frame["response_group"] == "cell0"
    assert frame["response_probe"] == "rotation"
    assert frame["response_coordinate"] == 0.01
    assert frame["response_branch"] == "strain"
    assert frame["response_family"] == "parent0"
    assert frame["response_stat_family"] == "parent0"
    assert frame["response_path_coordinate"] == -0.02
    np.testing.assert_allclose(frame["spins"], [[0.0, 0.0, 2.0]])
    np.testing.assert_allclose(frame["mforces"], [[0.1, 0.2, 0.3]])
    np.testing.assert_allclose(frame["tau"], [[-0.4, 0.2, 0.0]])


def test_extxyz_rejects_spin_tangent_as_an_input_label(tmp_path):
    path = tmp_path / "response-with-tangent.xyz"
    path.write_text(
        "1\n"
        'Lattice="6 0 0 0 6 0 0 0 6" energy=-2 '
        "Properties=species:S:1:pos:R:3:spin:R:3:mforce:R:3:"
        "spin_tangent:R:3\n"
        "Fe 0 0 0 0 0 2 0.1 0.2 0.3 0 1 0\n"
    )
    with pytest.raises(ValueError, match="not an input label"):
        read_xyz(path)


def test_generator_curve_selection_and_complete_group_batches():
    frames = []
    for group in ("b", "a"):
        for coordinate in (0.1, -0.1, 0.0):
            frames.append(_rotation_frame(coordinate, group))
    selected = select_generator_curve_frames(frames)
    assert [frame["response_group"] for frame in selected] == \
        ["a", "a", "a", "b", "b", "b"]
    assert [frame["response_coordinate"] for frame in selected[:3]] == \
        [-0.1, 0.0, 0.1]
    batches = complete_group_batches([0, 0, 0, 1, 1, 1], 4, seed=7)
    assert sorted(index for batch in batches for index in batch) == list(range(6))
    assert all(not ({0, 1, 2} & set(batch)
                    and {3, 4, 5} & set(batch)) for batch in batches)


def test_generator_curve_loss_removes_group_intercept_but_keeps_shape():
    tangent = torch.tensor([[1.0, 0.0, 0.0]] * 3)
    struct_idx = torch.arange(3)
    groups = torch.zeros(3, dtype=torch.long)
    coordinates = torch.tensor([-1.0, 0.0, 1.0])
    target = torch.tensor([[1.0, 0.0, 0.0],
                           [2.0, 0.0, 0.0],
                           [3.0, 0.0, 0.0]])
    shifted = target + torch.tensor([5.0, 0.0, 0.0])
    exact = generator_curve_loss(
        shifted, target, tangent, struct_idx, groups, coordinates, 3,
        mean_weight=0.0)
    assert exact.item() < 1.0e-12
    wrong_shape = target[[0, 2, 1]]
    wrong = generator_curve_loss(
        wrong_shape, target, tangent, struct_idx, groups, coordinates, 3)
    assert wrong.item() > 1.0e-3


def test_generator_curve_mean_term_preserves_constant_response():
    tangent = torch.tensor([[1.0, 0.0, 0.0]] * 3)
    struct_idx = torch.arange(3)
    groups = torch.zeros(3, dtype=torch.long)
    coordinates = torch.tensor([-1.0, 0.0, 1.0])
    target = torch.tensor([[1.0, 0.0, 0.0],
                           [2.0, 0.0, 0.0],
                           [3.0, 0.0, 0.0]])
    shifted = target + torch.tensor([1.0, 0.0, 0.0])
    joint = generator_curve_loss(
        shifted, target, tangent, struct_idx, groups, coordinates, 3)
    centered_only = generator_curve_loss(
        shifted, target, tangent, struct_idx, groups, coordinates, 3,
        mean_weight=0.0)
    assert joint.item() > 1.0e-3
    assert centered_only.item() < 1.0e-12


def test_rotation_curve_requires_three_distinct_coordinates():
    frames = [_rotation_frame(coordinate)
              for coordinate in (0.0, 0.0, 0.1)]
    with pytest.raises(ValueError, match="three distinct"):
        select_generator_curve_frames(frames)


def test_rotation_metadata_requires_converged_spins_and_mforces():
    frames = [_rotation_frame(coordinate) for coordinate in (-0.1, 0.0, 0.1)]
    frames[1].pop("mforces")
    with pytest.raises(ValueError, match="mforces"):
        select_generator_curve_frames(frames)


def test_rotation_curve_rejects_explicit_tangent_input():
    frames = [_rotation_frame(coordinate)
              for coordinate in (-0.1, 0.0, 0.1)]
    frames[0]["spin_tangent"] = np.full((1, 3), np.nan)
    with pytest.raises(ValueError, match="not an input label"):
        select_generator_curve_frames(frames)


@pytest.mark.parametrize("field", ("response_coordinate", "spins", "mforces"))
def test_rotation_curve_rejects_nonfinite_generator_inputs(field):
    frames = [_rotation_frame(coordinate)
              for coordinate in (-0.1, 0.0, 0.1)]
    if field == "response_coordinate":
        frames[1][field] = np.nan
    else:
        frames[1][field][0, 0] = np.inf
    with pytest.raises(ValueError, match="finite|NaN or Inf"):
        select_generator_curve_frames(frames)


def test_rotation_curve_rejects_an_all_zero_generator():
    frames = [_rotation_frame(coordinate)
              for coordinate in (-0.1, 0.0, 0.1)]
    for frame in frames:
        frame["spins"][:] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="no spin rotation|all-zero"):
        select_generator_curve_frames(frames)


def test_rotation_curve_rejects_mixed_parent_geometry():
    frames = [_rotation_frame(coordinate)
              for coordinate in (-0.1, 0.0, 0.1)]
    frames[1]["positions"][0, 0] += 0.01
    with pytest.raises(ValueError, match="changes geometry"):
        select_generator_curve_frames(frames)


@pytest.mark.parametrize("spacing", (1.e-2, 1.e-3, 1.e-4))
def test_rotation_curve_derives_nonuniform_final_dft_spin_path(spacing):
    # Magnitude/direction differentiation is consistent, not polynomial-
    # exact in Cartesian coordinates; check its second-order error bound.
    coordinates = np.array([-2., 0., 3.]) * spacing
    frames = [_rotation_frame(coordinate) for coordinate in coordinates]
    for frame, coordinate in zip(frames, coordinates):
        # Retain longitudinal relaxation in the converged DFT moment path.
        frame["spins"] = np.array([[
            1.0 + 0.5 * coordinate,
            coordinate * coordinate,
            0.0,
        ]])
    selected = select_generator_curve_frames(frames)
    for frame, coordinate in zip(selected, coordinates):
        assert frame["spin_tangent"][0] == pytest.approx(
            [0.5, 2.0 * coordinate, 0.0], abs=10. * spacing**2)


def test_rotation_coarse_endpoints_preserve_physical_magnitude_derivative():
    coordinates = np.linspace(0., np.pi, 7)
    for slope in (0., 0.2):
        frames = [_rotation_frame(t) for t in coordinates]
        for frame, t in zip(frames, coordinates):
            frame['spins'] = np.array([[(2. + slope*t)*np.cos(t),
                                       (2. + slope*t)*np.sin(t), 0.]])
        selected = select_generator_curve_frames(frames)
        for frame, t in zip(selected, coordinates):
            direction = np.array([np.cos(t), np.sin(t), 0.])
            assert np.dot(frame['spin_tangent'][0], direction) == \
                pytest.approx(slope, abs=2.e-14)


def test_rotation_zero_crossing_keeps_cartesian_derivative():
    frames = [_rotation_frame(t) for t in (-0.2, 0., 0.3)]
    for frame in frames:
        frame['spins'] = np.array([[frame['response_coordinate'], 0., 0.]])
    selected = select_generator_curve_frames(frames)
    for frame in selected:
        np.testing.assert_allclose(frame['spin_tangent'], [[1., 0., 0.]])


def test_rotation_duplicate_coordinate_rejected_before_differentiation():
    frames = [_rotation_frame(t) for t in (0., .1, .1, .2)]
    with pytest.raises(ValueError, match='duplicate'):
        select_generator_curve_frames(frames)


def test_rotation_tangent_covariance_and_coordinate_units():
    frames = [_rotation_frame(t) for t in (-0.4, 0., 0.7)]
    reference = select_generator_curve_frames(frames)
    rotation, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))
    transformed = copy.deepcopy(frames)
    for frame in transformed:
        frame['spins'] = frame['spins'] @ rotation.T
        frame['response_coordinate'] *= 180. / np.pi
    result = select_generator_curve_frames(transformed)
    for a, b in zip(reference, result):
        np.testing.assert_allclose(b['spin_tangent'],
            a['spin_tangent'] @ rotation.T * np.pi / 180., atol=1.e-14)


def test_rotation_curve_masks_inactive_spin_types_after_derivation():
    frames = [_rotation_frame(coordinate)
              for coordinate in (-0.1, 0.0, 0.1)]
    for frame in frames:
        frame["natoms"] = 2
        frame["species"] = ["Fe", "O"]
        frame["positions"] = np.zeros((2, 3))
        frame["spins"] = np.repeat(frame["spins"], 2, axis=0)
        frame["mforces"] = np.zeros((2, 3))
    selected = select_generator_curve_frames(frames, active_species=["Fe"])
    assert all(np.linalg.norm(frame["spin_tangent"][0]) > 0.0
               for frame in selected)
    assert all(np.array_equal(frame["spin_tangent"][1], np.zeros(3))
               for frame in selected)


def test_spin_resume_data_lineage_detects_in_place_edits(tmp_path):
    data = tmp_path / "train.xyz"
    data.write_text("original\n")
    saved = build_data_lineage(data, None, 0.1)
    data.write_text("modified\n")
    current = build_data_lineage(data, None, 0.1)
    with pytest.raises(ValueError, match="data content or split"):
        validate_resume_data_lineage(
            saved, current)


def test_spin_pt_finetune_requires_matching_architecture_signature(tmp_path):
    source = NEPModel(_config())
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({
        "model_state": source.state_dict(),
        "model_signature": source.architecture_signature(),
    }, checkpoint)
    changed = _config()
    changed["spin_cutoff"] = [5.5]
    target = NEPModel(changed)
    with pytest.raises(ValueError, match="architecture"):
        _load_finetune_weights(target, str(checkpoint), torch.device("cpu"))


def test_generator_curve_is_neutral_to_sin_cos_and_sin2_targets():
    """Target phase or harmonic content must not manufacture group weights."""
    dtype = torch.float64
    n_points = 16
    theta = (
        2.0 * torch.pi * torch.arange(n_points, dtype=dtype) / n_points
        - torch.pi
    )
    waves = (torch.sin(theta), torch.cos(theta), torch.sin(2.0 * theta))
    target_values = torch.cat(waves)
    common_error = 0.1 * torch.cos(3.0 * theta)
    predicted_values = target_values + common_error.repeat(len(waves))
    predicted = torch.stack((
        predicted_values,
        torch.zeros_like(predicted_values),
        torch.zeros_like(predicted_values),
    ), dim=1).requires_grad_()
    target = torch.stack((
        target_values,
        torch.zeros_like(target_values),
        torch.zeros_like(target_values),
    ), dim=1)
    tangent = torch.zeros_like(target)
    tangent[:, 0] = 1.0
    struct_idx = torch.arange(target_values.numel())
    groups = torch.arange(len(waves)).repeat_interleave(n_points)
    coordinates = theta.repeat(len(waves))

    loss = generator_curve_loss(
        predicted, target, tangent, struct_idx, groups, coordinates,
        target_values.numel(), mean_weight=0.0)
    loss.backward()
    gradient_norms = predicted.grad[:, 0].reshape(len(waves), n_points).norm(
        dim=1)
    torch.testing.assert_close(
        gradient_norms,
        gradient_norms[0].expand_as(gradient_norms),
        rtol=1.0e-12, atol=1.0e-12)
    assert "group-rms" in GENERATOR_CURVE_SCHEMA


def test_generator_curve_zero_mean_and_zero_target_scales_stay_finite():
    dtype = torch.float64
    n_points = 16
    theta = 2.0 * torch.pi * torch.arange(n_points, dtype=dtype) / n_points
    cosine = torch.cos(theta)
    tangent = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype).repeat(
        n_points, 1)
    struct_idx = torch.arange(n_points)
    groups = torch.zeros(n_points, dtype=torch.long)

    target = torch.zeros(n_points, 3, dtype=dtype)
    target[:, 0] = cosine
    shifted = target.clone()
    shifted[:, 0] += 0.1
    zero_mean_loss = generator_curve_loss(
        shifted, target, tangent, struct_idx, groups, theta, n_points)
    torch.testing.assert_close(
        zero_mean_loss,
        torch.tensor(0.0025, dtype=dtype),
        rtol=1.0e-12, atol=1.0e-12)

    zero_target = torch.zeros_like(target)
    small_prediction = torch.zeros_like(target)
    small_prediction[:, 0] = 1.0e-7 * cosine
    zero_target_loss = generator_curve_loss(
        small_prediction, zero_target, tangent, struct_idx, groups, theta,
        n_points)
    assert torch.isfinite(zero_target_loss)
    assert 0.0 < zero_target_loss.item() < 1.0


def test_generator_curve_group_replication_and_batch_packing_are_invariant():
    dtype = torch.float64
    n_points = 8
    theta = 2.0 * torch.pi * torch.arange(n_points, dtype=dtype) / n_points

    def make_case(repeats):
        waves = (torch.sin(theta), torch.cos(theta), torch.sin(2.0 * theta))
        values = []
        errors = []
        group_ids = []
        for repeat in range(repeats):
            for index, wave in enumerate(waves):
                values.append(wave)
                errors.append(0.05 * torch.cos((index + 3.0) * theta))
                group_ids.extend([repeat * len(waves) + index] * n_points)
        values = torch.cat(values)
        errors = torch.cat(errors)
        target = torch.zeros(values.numel(), 3, dtype=dtype)
        target[:, 0] = values
        tangent = torch.zeros_like(target)
        tangent[:, 0] = 1.0
        return target, tangent, errors, torch.tensor(group_ids), \
            theta.repeat(repeats * len(waves))

    def full_loss_and_gradient(repeats):
        target, tangent, errors, groups, coordinates = make_case(repeats)
        amplitude = torch.tensor(0.4, dtype=dtype, requires_grad=True)
        predicted = target.clone()
        predicted[:, 0] += amplitude * errors
        loss = generator_curve_loss(
            predicted, target, tangent, torch.arange(target.shape[0]),
            groups, coordinates, target.shape[0], mean_weight=0.0)
        loss.backward()
        return loss.detach(), amplitude.grad.detach()

    once = full_loss_and_gradient(1)
    twice = full_loss_and_gradient(2)
    torch.testing.assert_close(once[0], twice[0], rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(once[1], twice[1], rtol=1.0e-12, atol=1.0e-12)

    target, tangent, errors, groups, coordinates = make_case(1)
    predicted = target.clone()
    predicted[:, 0] += 0.4 * errors
    full = generator_curve_loss(
        predicted, target, tangent, torch.arange(target.shape[0]), groups,
        coordinates, target.shape[0], mean_weight=0.0)
    for packed_size in (n_points, 2 * n_points + 1):
        packed = full.new_zeros(())
        for indices, weight in normalized_group_batches(
                groups.tolist(), packed_size, seed=9):
            index = torch.tensor(indices, dtype=torch.long)
            packed = packed + weight * generator_curve_loss(
                predicted[index], target[index], tangent[index],
                torch.arange(index.numel()), groups[index],
                coordinates[index], index.numel(), mean_weight=0.0,
                normalization_scale=target[:, 0].square().mean().sqrt())
        torch.testing.assert_close(
            packed, full, rtol=1.0e-12, atol=1.0e-12)


def test_generator_scale_is_group_balanced_for_unequal_scan_lengths():
    short = torch.tensor([1.0, -1.0], dtype=torch.float64)
    long = torch.tensor([4.0, -4.0], dtype=torch.float64)
    expected = torch.tensor(8.5, dtype=torch.float64).sqrt()

    values = torch.cat((short, long))
    groups = torch.tensor([0, 0, 1, 1])
    torch.testing.assert_close(
        group_balanced_generator_rms(values, groups), expected)

    repeated_values = torch.cat((short, long.repeat(7)))
    repeated_groups = torch.tensor([0, 0] + [1] * 14)
    torch.testing.assert_close(
        group_balanced_generator_rms(repeated_values, repeated_groups),
        expected,
    )
