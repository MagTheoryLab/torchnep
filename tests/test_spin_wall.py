# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.

import math

import numpy as np
import pytest
import torch

from torchnep.data import parse_nep_in
from torchnep.model import NEPModel, slim_model
from torchnep.nep import NEPCalculator
from torchnep.spin_schema import compute_spin_wall, read_nep_model
from torchnep.train import StreamDataStore, preprocess_structures


DTYPE = torch.float64


def _write_input(path, extra="", types="3 Bi Fe O", spin_type="Fe"):
    lines = [
        f"type {types}",
        "cutoff 6 5",
        "n_max 2 2",
        "basis_size 4 4",
        "l_max 2 0 0",
        "neuron 6",
        "model nep_spin",
        "spin_version 1",
        "spin_order 2",
        "spin_compress 2",
        "spin_l_max 2",
        "spin_soc 1",
        "spin_cutoff 6",
    ]
    if spin_type:
        lines.append(f"spin_type {spin_type}")
    if extra:
        lines.append(extra.strip())
    path.write_text("\n".join(lines) + "\n")


def test_spin_wall_parsing_shorthand_and_per_type(tmp_path):
    p1 = tmp_path / "shorthand.in"
    _write_input(p1, "spin_wall_high 5.5 2.0\nspin_wall_low 2.5 1.5")
    cfg1 = parse_nep_in(p1)
    assert cfg1["spin_wall_high_s"] == [0.0, 5.5, 0.0]
    assert cfg1["spin_wall_high_k"] == [0.0, 2.0, 0.0]
    assert cfg1["spin_wall_low_s"] == [0.0, 2.5, 0.0]
    assert cfg1["spin_wall_low_k"] == [0.0, 1.5, 0.0]

    p2 = tmp_path / "per_type.in"
    _write_input(
        p2,
        "spin_wall_high 4.5 2.0 3.0 1.5\nspin_wall_low 1.5 1.0 0.8 0.5",
        types="2 Fe Co",
        spin_type="Fe Co",
    )
    cfg2 = parse_nep_in(p2)
    assert cfg2["spin_wall_high_s"] == [4.5, 3.0]
    assert cfg2["spin_wall_high_k"] == [2.0, 1.5]
    assert cfg2["spin_wall_low_s"] == [1.5, 0.8]
    assert cfg2["spin_wall_low_k"] == [1.0, 0.5]


def test_spin_wall_validation_errors(tmp_path):
    # Non-spin model rejects spin_wall_*
    non_spin = tmp_path / "non_spin.in"
    non_spin.write_text("type 1 Fe\ncutoff 6 5\nspin_wall_high 5.5 2.0\n")
    with pytest.raises(ValueError, match="require model nep_spin"):
        parse_nep_in(non_spin)

    # Active stiffness on excluded spin_type
    bad_inactive = tmp_path / "bad_inactive.in"
    _write_input(bad_inactive, "spin_wall_high 5.5 1.0 5.5 2.0 0 0")
    with pytest.raises(ValueError, match="inactive spin type"):
        parse_nep_in(bad_inactive)

    # s_lo >= s_hi
    bad_order = tmp_path / "bad_order.in"
    _write_input(bad_order, "spin_wall_high 4.0 2.0\nspin_wall_low 4.0 1.0")
    with pytest.raises(ValueError, match="strictly less than"):
        parse_nep_in(bad_order)

    # Zero threshold with positive stiffness
    bad_zero = tmp_path / "bad_zero.in"
    _write_input(bad_zero, "spin_wall_high 0.0 2.0")
    with pytest.raises(ValueError, match="positive threshold"):
        parse_nep_in(bad_zero)


@pytest.mark.parametrize("key", ["spin_wall_high", "spin_wall_low"])
def test_disabled_shorthand_wall_roundtrips_as_no_wall(tmp_path, key):
    input_file = tmp_path / "nep.in"
    _write_input(input_file, f"{key} 5.5 0")
    config = parse_nep_in(input_file)
    assert config[f"{key}_s"] == [0.0, 0.0, 0.0]
    model = NEPModel(config).double()
    model_file = tmp_path / "nep.txt"
    model.save_nep_txt(model_file, 32, 32)
    assert key not in model_file.read_text()
    model.load_weights_from_nep_txt(model_file)


def test_spin_wall_small_thresholds_match_autograd():
    spins = torch.tensor([[0.0, 0.0, 0.0], [5e-13, 0.0, 0.0]],
                         dtype=torch.float64, requires_grad=True)
    zero = spins.new_tensor([0.0])
    e, f = compute_spin_wall(spins, torch.zeros(2, dtype=torch.long),
                             spins.new_tensor([1e-13]), spins.new_tensor([1e24]), zero, zero)
    assert e[0].item() == 0.0
    torch.testing.assert_close(f[:, 0], spins.new_tensor([0.0, -0.16]), rtol=1e-12, atol=0.0)
    torch.testing.assert_close(f, -torch.autograd.grad(e.sum(), spins)[0], rtol=1e-12, atol=0.0)

    spins = torch.tensor([[0.0, 0.0, 0.0], [5e-24, 0.0, 0.0]],
                         dtype=torch.float32, requires_grad=True)
    zero = spins.new_tensor([0.0])
    e, f = compute_spin_wall(spins, torch.zeros(2, dtype=torch.long),
                             zero, zero, spins.new_tensor([1e-23]), spins.new_tensor([1.0]))
    assert torch.isfinite(e).all()
    torch.testing.assert_close(f[:, 0], spins.new_tensor([0.0, 0.5625]))
    torch.testing.assert_close(f, -torch.autograd.grad(e.sum(), spins)[0])


def test_compute_spin_wall_math_derivatives_symmetry_and_zero_spin():
    hi_s = torch.tensor([0.0, 5.5, 0.0], dtype=DTYPE)
    hi_k = torch.tensor([0.0, 2.0, 0.0], dtype=DTYPE)
    lo_s = torch.tensor([0.0, 2.5, 0.0], dtype=DTYPE)
    lo_k = torch.tensor([0.0, 1.5, 0.0], dtype=DTYPE)
    atom_types = torch.tensor([0, 1, 1, 1, 1, 2], dtype=torch.long)

    # Fe atoms at: |s|=0, |s|=s_lo/sqrt(5) (peak low force), |s|=4.0 (inside), |s|=6.5 (above)
    peak_m = 2.5 / math.sqrt(5.0)
    spins = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [peak_m * 0.6, -peak_m * 0.8, 0.0],
            [2.4, 3.2, 0.0],
            [0.0, 3.9, 5.2],
            [1.0, 2.0, 3.0],
        ],
        dtype=DTYPE,
        requires_grad=True,
    )

    e_wall, mf_wall = compute_spin_wall(spins, atom_types, hi_s, hi_k, lo_s, lo_k)
    grad_auto = torch.autograd.grad(e_wall.sum(), spins)[0]

    assert torch.isfinite(e_wall).all()
    assert torch.isfinite(mf_wall).all()
    assert torch.isfinite(grad_auto).all()
    torch.testing.assert_close(mf_wall, -grad_auto, rtol=1e-12, atol=1e-12)

    # Zero torque everywhere: s x mf_wall == 0
    torque = torch.cross(spins.detach(), mf_wall.detach(), dim=-1)
    torch.testing.assert_close(torque, torch.zeros_like(torque), rtol=0.0, atol=1e-12)

    # Inside [s_lo, s_hi] (atom 3, |s|=4.0) and inactive species (atoms 0, 5) are identically zero
    assert e_wall[0].item() == 0.0
    assert e_wall[3].item() == 0.0
    assert e_wall[5].item() == 0.0
    torch.testing.assert_close(mf_wall[3], torch.zeros(3, dtype=DTYPE), rtol=0.0, atol=0.0)

    # At |s|=0 (atom 1): smooth peak energy = k_lo * s_lo / 3, zero force
    assert math.isclose(e_wall[1].item(), 1.5 * 2.5 / 3.0, rel_tol=1e-12)
    torch.testing.assert_close(mf_wall[1], torch.zeros(3, dtype=DTYPE), rtol=0.0, atol=1e-12)

    # Peak low-wall force at s_lo / sqrt(5) (atom 2) equals (32 / (25 * sqrt(5))) * k_lo
    expected_peak_f = (32.0 / (25.0 * math.sqrt(5.0))) * 1.5
    assert math.isclose(torch.linalg.vector_norm(mf_wall[2]).item(), expected_peak_f, rel_tol=1e-12)

    # High-wall energy & force at |s|=6.5 (atom 4): excess = 1.0
    assert math.isclose(e_wall[4].item(), 2.0 / 3.0, rel_tol=1e-12)
    assert math.isclose(torch.linalg.vector_norm(mf_wall[4]).item(), 2.0, rel_tol=1e-12)
    # Restoring force points inward (opposite to spin)
    assert torch.dot(mf_wall[4], spins[4]).item() < 0.0

    # Time-reversal symmetry: E(-s) == E(s), M(-s) == -M(s)
    e_rev, mf_rev = compute_spin_wall(-spins.detach(), atom_types, hi_s, hi_k, lo_s, lo_k)
    torch.testing.assert_close(e_rev, e_wall.detach(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(mf_rev, -mf_wall.detach(), rtol=1e-12, atol=1e-12)


def test_spin_wall_model_cached_autograd_calculator_parity_and_roundtrip(tmp_path):
    torch.manual_seed(7)
    base_in = tmp_path / "base.in"
    wall_in = tmp_path / "wall.in"
    _write_input(base_in)
    _write_input(wall_in, "spin_wall_high 5.0 2.5\nspin_wall_low 2.0 1.2")

    base_cfg = parse_nep_in(base_in)
    wall_cfg = parse_nep_in(wall_in)

    base_model = NEPModel(base_cfg).double()
    wall_model = NEPModel(wall_cfg).double()
    wall_model.load_state_dict(base_model.state_dict())

    # 3 Fe atoms: one below s_lo (|s|=1.0), one inside (|s|=3.5), one above s_hi (|s|=6.2)
    frame = {
        "natoms": 5,
        "species": ["Bi", "Fe", "Fe", "Fe", "O"],
        "positions": np.array(
            [
                [0.0, 0.0, 0.0],
                [2.1, 0.2, 0.1],
                [4.2, 0.3, -0.2],
                [1.1, 2.4, 0.5],
                [3.1, 1.9, 1.1],
            ],
            dtype=np.float64,
        ),
        "cell": np.diag([12.0, 12.0, 12.0]),
        "spins": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.6, 0.8, 0.0],       # |s| = 1.0 < 2.0 (low wall active)
                [0.0, 2.1, 2.8],       # |s| = 3.5 in [2.0, 5.0] (wall == 0)
                [3.72, -4.96, 0.0],    # |s| = 6.2 > 5.0 (high wall active)
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    }

    structs = preprocess_structures([frame], wall_cfg, dtype=np.float64, n_workers=1)
    store = StreamDataStore(structs, torch.device("cpu"), DTYPE, config=wall_cfg)
    batch = store.collate([0])
    batch["need_mforce"] = True
    batch["need_tau"] = True

    res_base = base_model.compute_properties_cached(batch, need_forces=True, need_virial=True)
    res_cached = wall_model.compute_properties_cached(batch, need_forces=True, need_virial=True)
    res_auto = wall_model.compute_properties(
        batch["rij_rad"], batch["rij_ang"],
        batch["pair_i_rad"], batch["pair_j_rad"],
        batch["pair_i_ang"], batch["pair_j_ang"],
        batch["atom_types"], batch["N"],
        batch["struct_idx"], batch["num_structures"],
        need_forces=True, need_virial=True,
        spins=batch["spins"], need_mforce=True, need_tau=True,
        rij_spin=batch["rij_spin"], pi_spin=batch["pair_i_spin"], pj_spin=batch["pair_j_spin"],
    )

    # Cached analytical path matches autograd path exactly
    for key in ("Ei", "Etot", "forces", "virial", "mforces", "tau"):
        torch.testing.assert_close(res_cached[key], res_auto[key], rtol=1e-11, atol=1e-11)

    # Atomic forces, virial, and spin torque are completely unaffected by the radial spin wall
    torch.testing.assert_close(res_cached["forces"], res_base["forces"], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(res_cached["virial"], res_base["virial"], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(res_cached["tau"], res_base["tau"], rtol=1e-11, atol=1e-11)

    # In-envelope Fe atom (index 2) is unaffected; out-of-envelope atoms (1 and 3) receive wall terms
    torch.testing.assert_close(res_cached["Ei"][2], res_base["Ei"][2], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(res_cached["mforces"][2], res_base["mforces"][2], rtol=0.0, atol=1e-12)
    assert res_cached["Ei"][1].item() > res_base["Ei"][1].item()
    assert res_cached["Ei"][3].item() > res_base["Ei"][3].item()

    # Round-trip save/load nep.txt & NEPCalculator parity
    base_txt = tmp_path / "base.txt"
    wall_txt = tmp_path / "wall.txt"
    base_model.save_nep_txt(base_txt, max_NN_radial=32, max_NN_angular=32)
    wall_model.save_nep_txt(wall_txt, max_NN_radial=32, max_NN_angular=32)

    assert "spin_wall_high" not in base_txt.read_text()
    assert "spin_wall_low" not in base_txt.read_text()
    assert "spin_wall_high 0 0 5 2.5 0 0\n" in wall_txt.read_text()
    assert "spin_wall_low 0 0 2 1.2 0 0\n" in wall_txt.read_text()

    restored = NEPModel(wall_cfg).double()
    restored.load_weights_from_nep_txt(wall_txt)
    assert restored.architecture_signature() == wall_model.architecture_signature()
    assert restored.architecture_signature() != base_model.architecture_signature()

    with pytest.raises(ValueError, match="spin_wall_high"):
        base_model.load_weights_from_nep_txt(wall_txt)
    with pytest.raises(ValueError, match="spin_wall_high"):
        wall_model.load_weights_from_nep_txt(base_txt)

    args = frame["species"], frame["positions"], frame["cell"]
    bare = NEPCalculator(base_txt, dtype=DTYPE)
    bare_res = bare.compute(*args, spins=frame["spins"])
    torch.testing.assert_close(bare_res["energy"], res_base["Ei"])
    torch.testing.assert_close(bare_res["mforces"], res_base["mforces"])
    overlay = NEPCalculator(base_txt, dtype=DTYPE, spin_wall_high=(5.0, 2.5), spin_wall_low=(2.0, 1.2))
    calc = NEPCalculator(wall_txt, dtype=DTYPE)
    with pytest.raises(ValueError, match="spin_wall_high auto requires training data"):
        NEPCalculator(wall_txt, spin_wall_high=("auto", 0.5, 2.0))
    calc_res = calc.compute(*args, spins=frame["spins"], return_components=True)
    calc_batch = calc.compute_batch(batch)
    torch.testing.assert_close(calc_res["energy"], res_cached["Ei"], rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(calc_res["mforces"], res_cached["mforces"], rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(calc_batch["Ei"], res_cached["Ei"], rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(calc_batch["mforces"], res_cached["mforces"], rtol=1e-11, atol=1e-11)
    overlay_res = overlay.compute(*args, spins=frame["spins"])
    overlay_batch = overlay.compute_batch(batch)
    for name in ("energy", "mforces", "forces", "virial"):
        torch.testing.assert_close(overlay_res[name], calc_res[name], rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(overlay_batch["Ei"], calc_batch["Ei"], rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(overlay_batch["mforces"], calc_batch["mforces"], rtol=1e-11, atol=1e-11)

    # slim_model preserves per-type walls for kept elements
    slimmed = slim_model(wall_model, ["Fe", "O"])
    assert slimmed.spin_wall_high_s == [5.0, 0.0]
    assert slimmed.spin_wall_high_k == [2.5, 0.0]
    assert slimmed.spin_wall_low_s == [2.0, 0.0]
    assert slimmed.spin_wall_low_k == [1.2, 0.0]


def test_spin_wall_auto_resolution(tmp_path):
    from torchnep.spin_schema import resolve_auto_spin_walls, slim_type_config

    p = tmp_path / "auto.in"
    _write_input(
        p,
        "spin_wall_high auto 0.5 2.0\nspin_wall_low auto 0.6 1.5",
        types="3 Bi Fe Ni",
        spin_type="Fe Ni",
    )
    cfg = parse_nep_in(p)
    assert cfg["spin_wall_high_auto"] == (0.5, 2.0)
    assert cfg["spin_wall_low_auto"] == (0.6, 1.5)
    with pytest.raises(ValueError, match="spin_wall_high auto requires training data"):
        NEPModel(cfg)

    # slim_type_config preserves auto specs prior to frame resolution
    slim_cfg = slim_type_config(cfg, ["Fe", "Ni"])
    assert slim_cfg["spin_wall_high_auto"] == (0.5, 2.0)
    assert slim_cfg["spin_wall_low_auto"] == (0.6, 1.5)

    frames = [
        {
            "species": ["Bi", "Fe", "Ni"],
            "spins": np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.4, 0.0]]),
        },
        {
            "species": ["Bi", "Fe", "Ni"],
            "spins": np.array([[0.0, 0.0, 0.0], [0.0, 4.8, 0.0], [0.0, 0.0, 0.9]]),
        },
    ]
    resolved = resolve_auto_spin_walls(cfg, frames)
    assert resolved["spin_wall_high_auto"] is None
    assert resolved["spin_wall_low_auto"] is None
    # Fe: max=4.8 -> 5.3, min=3.0 -> 2.4; Ni: max=0.9 -> 1.4, min=0.4 - 0.6 <= 0 -> low disabled
    assert resolved["spin_wall_high_s"] == pytest.approx([0.0, 5.3, 1.4])
    assert resolved["spin_wall_high_k"] == [0.0, 2.0, 2.0]
    assert resolved["spin_wall_low_s"] == pytest.approx([0.0, 2.4, 0.0])
    assert resolved["spin_wall_low_k"] == [0.0, 1.5, 0.0]

    # Exported nep.txt writes concrete resolved numbers (never 'auto')
    model = NEPModel(resolved).double()
    out_txt = tmp_path / "auto.txt"
    model.save_nep_txt(out_txt, max_NN_radial=32, max_NN_angular=32)
    reread, _, _ = read_nep_model(out_txt)
    for key in ("spin_wall_high", "spin_wall_low"):
        assert reread[f"{key}_auto"] is None
        assert reread[f"{key}_s"] == resolved[f"{key}_s"]
        assert reread[f"{key}_k"] == resolved[f"{key}_k"]
    model.load_weights_from_nep_txt(out_txt)
