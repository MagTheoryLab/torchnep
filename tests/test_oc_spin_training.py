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
from torchnep.spin_diagnostics import relative_vector_error, svd_health
from torchnep.train import (
    _load_finetune_weights,
    build_data_lineage,
    fit_energy_baseline_frames,
    grouped_validation_indices,
    spin_curriculum_scale,
    validate_resume_data_lineage,
)


def _config(curriculum=True):
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
        "spin_mode": 2,
        "spin_compress": 2,
        "spin_basis_size": [8, 0],
        "spin_l_max": [2, 0, 0],
        "spin_cutoff": [6.0, 6.0],
        "spin_order": 3,
        "spin_soc": 1,
        "spin_curriculum": curriculum,
    }


def _rotation_frame(coordinate, group="g"):
    spin = np.array([[np.cos(coordinate), np.sin(coordinate), 0.0]])
    tangent = np.array([[-np.sin(coordinate), np.cos(coordinate), 0.0]])
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
        "spin_tangent": tangent,
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


def test_oc_defaults_are_o3c2_with_curriculum(tmp_path):
    path = tmp_path / "nep.in"
    path.write_text(
        "type 1 Fe\n"
        "cutoff 6 5\n"
        "n_max 4 4\n"
        "basis_size 8 8\n"
        "l_max 4 2 0\n"
        "spin_mode 2\n"
    )
    config = parse_nep_in(path)
    assert config["spin_order"] == 3
    assert config["spin_compress"] == 2
    assert config["spin_l_max"] == [2, 0, 0]
    assert config["spin_soc"] == 1
    assert config["spin_curriculum"] is True
    assert NEPModel(config).spin_dim == 49


def test_oc_curriculum_can_be_disabled_in_nep_input(tmp_path):
    path = tmp_path / "nep.in"
    path.write_text(
        "type 1 Fe\n"
        "spin_mode 2\n"
        "spin_order 3\n"
        "spin_curriculum 0\n"
    )
    config = parse_nep_in(path)
    assert config["spin_curriculum"] is False
    assert NEPModel(config).spin_curriculum is False


def test_oc_curriculum_parser_rejects_non_boolean_value(tmp_path):
    path = tmp_path / "nep.in"
    path.write_text(
        "type 1 Fe\n"
        "spin_mode 2\n"
        "spin_curriculum 2\n"
    )
    with pytest.raises(ValueError, match="spin_curriculum must be 0 or 1"):
        parse_nep_in(path)


def test_curriculum_starts_at_exact_o2_prefix_and_ramps_gradients():
    model = NEPModel(_config()).double()
    start = model.struct_dim + model.spin_descriptor.order_dims[2]
    assert model.spin_order3_input_start == start
    assert model.dim - start == 12
    for net in model.fitting_nets:
        torch.testing.assert_close(
            net.w0[start:], torch.zeros_like(net.w0[start:])
        )

    weight = model.fitting_nets[0].w0
    model.set_spin_curriculum_scale(0.0)
    weight.sum().backward()
    model.apply_spin_curriculum_gradients()
    assert torch.count_nonzero(weight.grad[:start]) > 0
    assert torch.count_nonzero(weight.grad[start:]) == 0

    weight.grad = None
    model.set_spin_curriculum_scale(0.25)
    weight.sum().backward()
    model.apply_spin_curriculum_gradients()
    torch.testing.assert_close(
        weight.grad[start:], torch.full_like(weight.grad[start:], 0.25)
    )

    # Adam normalizes its first moment, so gradient scaling alone would make
    # the first parameter update almost independent of 0.25. The production
    # seam also scales the actual post-Adam row displacement.
    full = copy.deepcopy(model)
    for candidate, scale in ((model, 0.25), (full, 1.0)):
        candidate.zero_grad(set_to_none=True)
        candidate.set_spin_curriculum_scale(scale)
        candidate.fitting_nets[0].w0.sum().backward()
        candidate.apply_spin_curriculum_gradients()
        optimizer = torch.optim.Adam(candidate.parameters(), lr=1.0e-3)
        before = candidate.capture_spin_curriculum_rows()
        optimizer.step()
        candidate.apply_spin_curriculum_parameter_step(before)
    delta_small = model.fitting_nets[0].w0[start:].abs().mean()
    delta_full = full.fitting_nets[0].w0[start:].abs().mean()
    torch.testing.assert_close(
        delta_small, 0.25 * delta_full, rtol=2.0e-5, atol=1.0e-10)


def test_curriculum_schedule_is_fully_open_at_stage2_entry():
    assert spin_curriculum_scale(1, 400) == 0.0
    assert spin_curriculum_scale(200, 400) == 0.0
    assert spin_curriculum_scale(300, 400) == 0.5
    assert spin_curriculum_scale(400, 400) == 1.0
    assert spin_curriculum_scale(600, 400) == 1.0


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
        "Properties=species:S:1:pos:R:3:spin:R:3:mforce:R:3:"
        "spin_tangent:R:3\n"
        "Fe 0 0 0 0 0 2 0.1 0.2 0.3 0 1 0\n"
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
    np.testing.assert_allclose(frame["spin_tangent"], [[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(frame["tau"], [[-0.4, 0.2, 0.0]])


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


def test_rotation_metadata_without_tangent_remains_evaluation_only():
    frames = [{
        "natoms": 1,
        "response_group": "g",
        "response_probe": "rotation",
        "response_coordinate": coordinate,
    } for coordinate in (-0.1, 0.0, 0.1)]
    assert select_generator_curve_frames(frames) == []


def test_rotation_curve_rejects_partially_declared_tangents():
    frames = [_rotation_frame(coordinate)
              for coordinate in (-0.1, 0.0, 0.1)]
    frames[0].pop("spin_tangent")
    with pytest.raises(ValueError, match="only part"):
        select_generator_curve_frames(frames)


@pytest.mark.parametrize("field", ("response_coordinate", "spin_tangent", "mforces"))
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
        frame["spin_tangent"].fill(0.0)
    with pytest.raises(ValueError, match="all-zero"):
        select_generator_curve_frames(frames)


def test_rotation_curve_rejects_mixed_parent_geometry():
    frames = [_rotation_frame(coordinate)
              for coordinate in (-0.1, 0.0, 0.1)]
    frames[1]["positions"][0, 0] += 0.01
    with pytest.raises(ValueError, match="changes geometry"):
        select_generator_curve_frames(frames)


def test_rotation_curve_rejects_longitudinal_tangent():
    frames = [_rotation_frame(coordinate)
              for coordinate in (-0.1, 0.0, 0.1)]
    frames[1]["spin_tangent"] = frames[1]["spins"].copy()
    with pytest.raises(ValueError, match="not orthogonal"):
        select_generator_curve_frames(frames)


def test_spin_resume_data_lineage_detects_in_place_edits(tmp_path):
    data = tmp_path / "train.xyz"
    data.write_text("original\n")
    saved = build_data_lineage(data, None, 0.1)
    data.write_text("modified\n")
    current = build_data_lineage(data, None, 0.1)
    with pytest.raises(ValueError, match="data content or split"):
        validate_resume_data_lineage(
            saved, current, require_versioned=True)


def test_spin_pt_finetune_requires_matching_architecture_signature(tmp_path):
    source = NEPModel(_config())
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({
        "model_state": source.state_dict(),
        "model_signature": source.architecture_signature(),
    }, checkpoint)
    changed = _config()
    changed["spin_cutoff"] = [5.5, 5.5]
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
    assert "neutral" in GENERATOR_CURVE_SCHEMA


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
