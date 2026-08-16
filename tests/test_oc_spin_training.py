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
    complete_group_batches,
    generator_curve_loss,
    select_generator_curve_frames,
)
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
        "spin_mode": 3,
        "spin_compress": 2,
        "spin_basis_size": [8, 0],
        "spin_l_max": [2, 0, 0],
        "spin_cutoff": [6.0, 6.0],
        "spin_order": 3,
        "spin_soc": 1,
        "spin_curriculum": curriculum,
    }


def test_oc_defaults_are_o3c2_with_curriculum(tmp_path):
    path = tmp_path / "nep.in"
    path.write_text(
        "type 1 Fe\n"
        "cutoff 6 5\n"
        "n_max 4 4\n"
        "basis_size 8 8\n"
        "l_max 4 2 0\n"
        "spin_mode 3\n"
    )
    config = parse_nep_in(path)
    assert config["spin_order"] == 3
    assert config["spin_compress"] == 2
    assert config["spin_l_max"] == [2, 0, 0]
    assert config["spin_soc"] == 1
    assert config["spin_curriculum"] is True
    assert NEPModel(config).spin_dim == 51


def test_curriculum_starts_at_exact_o2_prefix_and_ramps_gradients():
    model = NEPModel(_config()).double()
    start = model.struct_dim + model.spin_descriptor.order_dims[2]
    assert model.spin_order3_input_start == start
    assert model.dim - start == 14
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
        "response_coordinate=0.01 response_branch=plus "
        "Properties=species:S:1:pos:R:3:spin:R:3:mforce:R:3:"
        "spin_tangent:R:3\n"
        "Fe 0 0 0 0 0 2 0.1 0.2 0.3 0 1 0\n"
    )
    frame = read_xyz(path)[0]
    assert frame["response_group"] == "cell0"
    assert frame["response_probe"] == "rotation"
    assert frame["response_coordinate"] == 0.01
    assert frame["response_branch"] == "plus"
    np.testing.assert_allclose(frame["spins"], [[0.0, 0.0, 2.0]])
    np.testing.assert_allclose(frame["mforces"], [[0.1, 0.2, 0.3]])
    np.testing.assert_allclose(frame["spin_tangent"], [[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(frame["tau"], [[-0.4, 0.2, 0.0]])


def test_generator_curve_selection_and_complete_group_batches():
    frames = []
    for group in ("b", "a"):
        for coordinate in (0.1, -0.1, 0.0):
            frames.append({
                "natoms": 1,
                "response_group": group,
                "response_probe": "rotation",
                "response_coordinate": coordinate,
                "mforces": np.zeros((1, 3)),
                "spin_tangent": np.ones((1, 3)),
            })
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
    frames = [{
        "natoms": 1,
        "response_group": "g",
        "response_probe": "rotation",
        "response_coordinate": coordinate,
        "mforces": np.zeros((1, 3)),
        "spin_tangent": np.zeros((1, 3)),
    } for coordinate in (0.0, 0.0, 0.1)]
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
    frames = [{
        "natoms": 1,
        "response_group": "g",
        "response_probe": "rotation",
        "response_coordinate": coordinate,
        "mforces": np.zeros((1, 3)),
        **({"spin_tangent": np.zeros((1, 3))} if index else {}),
    } for index, coordinate in enumerate((-0.1, 0.0, 0.1))]
    with pytest.raises(ValueError, match="only part"):
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
