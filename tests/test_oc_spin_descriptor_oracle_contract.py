# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.
# TorchNEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from pathlib import Path

import pytest
import torch

from example.oc_spin import oracle_descriptor_contract as oracle


DTYPE = torch.float64


def test_publication_oracle_covers_current_four_dimensions_and_boundaries():
    observed = {}
    for name, settings in oracle.DESCRIPTORS.items():
        descriptor, _ = oracle.build_descriptor(
            settings["order"], settings["compress"], torch.device("cpu")
        )
        observed[name] = descriptor.dim
    assert observed == {
        "o2c2": 37,
        "o3c2": 49,
        "o2c3": 58,
        "o3c3": 79,
    }
    assert set(oracle.EXPECTED_REACHABILITY) == set(oracle.TEACHERS)
    assert oracle.EXPECTED_REACHABILITY["l112_edge_response"] == {
        "o3c2", "o3c3"
    }
    assert oracle.EXPECTED_REACHABILITY["rank3_bulk"] == {"o3c3"}
    for name in oracle.NEGATIVE_TEACHERS:
        assert oracle.EXPECTED_REACHABILITY[name] == set()


def test_independent_raw_formulae_match_named_production_coordinates():
    device = torch.device("cpu")
    descriptor, coefficients = oracle.build_descriptor(3, 3, device)
    environments = oracle.generate_environments(3, 20260829, device)
    errors = oracle.raw_formula_certificate(
        descriptor, coefficients, environments
    )
    assert set(errors) == set(oracle.RAW_TOKEN_FOR_TEACHER)
    for name, error in errors.items():
        assert error["residual_rms"] < 2.0e-11, (name, error)


@pytest.mark.parametrize("order,source_dim,target_dim", ((2, 37, 58), (3, 49, 79)))
def test_c2_to_c3_embedding_preserves_ann_energy_and_derivatives(
    order, source_dim, target_dim,
):
    device = torch.device("cpu")
    environments = oracle.generate_environments(2, 20260831 + order, device)
    certificate = oracle.embedding_certificate(
        order, environments, device, seed=20260829
    )
    assert certificate["source_dimension"] == source_dim
    assert certificate["target_dimension"] == target_dim
    assert len(certificate["source_to_target_columns"]) == source_dim
    assert len(set(certificate["source_to_target_columns"])) == source_dim
    assert certificate["ann_head_embedding"]["new_rows_zero"] is True
    assert certificate["passed"] is True
    for error in certificate["errors"].values():
        assert error["residual_rms"] < 2.0e-11


def test_oracle_cli_has_sealed_smoke_and_full_profiles():
    assert oracle.PROFILES["smoke"]["train"] > max(
        settings["dimension"] for settings in oracle.DESCRIPTORS.values()
    )
    assert oracle.PROFILES["full"]["train"] > oracle.PROFILES["smoke"]["train"]
    source = Path(oracle.__file__).read_text()
    for flag in ("--profile", "--seed", "--device", "--output"):
        assert flag in source
    assert "dataset_sha256" in source
    assert "dirty_state_sha256" in source
