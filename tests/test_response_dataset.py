import copy
import numpy as np

import pytest

from test_oc_spin_training_smoke import _write_case
from torchnep.data import parse_nep_in, read_xyz
from torchnep.response_training import select_generator_curve_frames
from torchnep.train import (
    build_data_lineage, load_response_dataset, validate_resume_data_lineage,
)


def test_response_default_off_and_invalid_weights(tmp_path):
    config, xyz, valid = _write_case(tmp_path)
    cfg = parse_nep_in(str(config))
    cfg["lambda_spin_response"] = 0
    (tmp_path / "response.xyz").write_text("broken file")
    assert load_response_dataset(cfg, xyz, None) == ([], [])
    for value in (-1, float("nan"), float("inf")):
        cfg["lambda_spin_response"] = value
        with pytest.raises(ValueError, match="finite and non-negative"):
            load_response_dataset(cfg, xyz, None)


def test_response_required_and_group_validation(tmp_path):
    config, xyz, valid = _write_case(tmp_path)
    cfg = parse_nep_in(str(config))
    path = tmp_path / "response.xyz"
    contents = path.read_text()
    path.unlink()
    with pytest.raises(ValueError, match="requires response.xyz"):
        load_response_dataset(cfg, xyz, None)
    path.write_text(contents.replace("response_probe=rotation", "response_probe=volume"))
    with pytest.raises(ValueError, match="complete rotation"):
        load_response_dataset(cfg, xyz, None)
    path.write_text(contents)
    train, validation = load_response_dataset(cfg, xyz, read_xyz(str(valid)))
    assert len(train) == len(validation) == 3
    assert all("spin_tangent" in f for f in train)
    original = select_generator_curve_frames(read_xyz(str(xyz)))
    for actual, expected in zip(train, original):
        np.testing.assert_array_equal(actual["spin_tangent"], expected["spin_tangent"])


@pytest.mark.parametrize("missing", [None, "energy", "forces", "spins", "mforces"])
def test_response_requires_full_labels_but_not_virial(tmp_path, missing):
    config, xyz, _ = _write_case(tmp_path)
    path = tmp_path / "response.xyz"
    lines = []
    for theta in (-0.1, 0.0, 0.1):
        schema = 'Properties=species:S:1:pos:R:3'
        values = 'Fe 1 1 1'
        for key, field, vector in [
            ("forces", "force", "0 0 0"),
            ("spins", "spin", f"{np.cos(theta)} {np.sin(theta)} 0"),
            ("mforces", "mforce", "0 0.1 0"),
        ]:
            if missing != key:
                schema += f':{field}:R:3'
                values += ' ' + vector
        energy = '' if missing == 'energy' else ' energy=-1'
        lines.extend([
            "1",
            'Lattice="6 0 0 0 6 0 0 0 6" response_group=r '
            f'response_probe=rotation response_coordinate={theta} '
            + schema + energy,
            values,
        ])
    path.write_text("\n".join(lines) + "\n")
    if missing:
        with pytest.raises(ValueError, match=missing):
            load_response_dataset(parse_nep_in(str(config)), xyz, None)
        return
    frames, _ = load_response_dataset(parse_nep_in(str(config)), xyz, None)
    assert len(frames) == 3
    assert all("energy" in f and "forces" in f and "virial" not in f for f in frames)


def test_response_duplicate_validation_and_lineage(tmp_path):
    config, xyz, valid = _write_case(tmp_path)
    cfg = parse_nep_in(str(config))
    duplicate = copy.deepcopy(read_xyz(str(xyz)))
    for frame in duplicate:
        frame.pop("response_group")
        frame.pop("response_probe")
    with pytest.raises(ValueError, match="structure/spin"):
        load_response_dataset(cfg, xyz, duplicate)
    saved = build_data_lineage(xyz, valid, None, response_weight=0.1)
    path = tmp_path / "response.xyz"
    path.write_text(path.read_text() + "\n")
    current = build_data_lineage(xyz, valid, None, response_weight=0.1)
    with pytest.raises(ValueError, match="data content"):
        validate_resume_data_lineage(saved, current)
    assert "response" not in build_data_lineage(xyz, valid, None)
