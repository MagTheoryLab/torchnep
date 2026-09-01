import copy
import itertools
import os

import pytest
import torch

from torchnep import train_nep
from torchnep.model import NEPModel
from torchnep.oc_cuda_graph import OCCudaGraphTrainCache, _joint_loss_metrics


def _config(*, order=3, compress=2, spin_l=2, soc=1, spin_basis=8,
            l_max=None, n_max=(4, 4), basis_size=(8, 8), neuron=30,
            cutoffs=(6.0, 5.0), spin_cutoff=6.0,
            dof_types=None, env_types=None):
    if l_max is None:
        l_max = [4, 2, 0]
    return {
        "num_types": 3,
        "type_names": ["Fe", "Ge", "Ni"],
        "cutoff_radial": cutoffs[0],
        "cutoff_angular": cutoffs[1],
        "n_max_radial": n_max[0],
        "n_max_angular": n_max[1],
        "basis_size_radial": basis_size[0],
        "basis_size_angular": basis_size[1],
        "l_max": l_max,
        "neuron": neuron,
        "spin_mode": 2,
        "spin_cutoff": [spin_cutoff, spin_cutoff],
        "spin_l_max": [spin_l, 0, 0],
        "spin_compress": compress,
        "spin_basis_size": [spin_basis, 0],
        "spin_order": order,
        "spin_soc": soc,
        "spin_curriculum": False,
        "spin_dof_type": (
            ["Fe", "Ge", "Ni"] if dof_types is None else dof_types
        ),
        "spin_env_type": (
            ["Fe", "Ge", "Ni"] if env_types is None else env_types
        ),
    }


def _batch(n_atoms, neighbors=6, seed=20260901):
    generator = torch.Generator(device="cpu").manual_seed(seed + n_atoms)
    counts = [n_atoms // 2, n_atoms - n_atoms // 2]
    offsets = [0, counts[0], n_atoms]
    pair_i_parts = []
    pair_j_parts = []
    for start, stop in zip(offsets[:-1], offsets[1:]):
        count = stop - start
        local_i = torch.arange(count).repeat_interleave(neighbors)
        shifts = torch.arange(1, neighbors + 1).repeat(count)
        pair_i_parts.append(local_i + start)
        pair_j_parts.append((local_i + shifts) % count + start)
    pair_i = torch.cat(pair_i_parts)
    pair_j = torch.cat(pair_j_parts)
    rij = torch.randn(pair_i.numel(), 3, generator=generator)
    rij /= torch.linalg.vector_norm(rij, dim=1, keepdim=True)
    rij *= 1.0 + 3.5 * torch.rand(
        pair_i.numel(), 1, generator=generator
    )
    angular = torch.linalg.vector_norm(rij, dim=1) < 5.0
    spins = torch.randn(n_atoms, 3, generator=generator)
    struct_idx = torch.repeat_interleave(
        torch.arange(2, dtype=torch.long), torch.tensor(counts)
    )

    def cuda(value):
        return value.cuda()

    return {
        "rij_rad": cuda(rij),
        "rij_ang": cuda(rij[angular]),
        "pair_i_rad": cuda(pair_i),
        "pair_j_rad": cuda(pair_j),
        "pair_i_ang": cuda(pair_i[angular]),
        "pair_j_ang": cuda(pair_j[angular]),
        "atom_types": cuda(torch.arange(n_atoms, dtype=torch.long) % 3),
        "struct_idx": cuda(struct_idx),
        "spins": cuda(spins),
        "rij_spin": cuda(rij),
        "pair_i_spin": cuda(pair_i),
        "pair_j_spin": cuda(pair_j),
        "energy": cuda(torch.randn(2, generator=generator)),
        "natoms": cuda(torch.tensor(counts, dtype=torch.float32)),
        "volumes": cuda(torch.tensor(counts, dtype=torch.float32) * 13.0),
        "energy_mask": cuda(torch.ones(2, dtype=torch.bool)),
        "forces": cuda(torch.randn(n_atoms, 3, generator=generator)),
        "force_mask": cuda(torch.ones(n_atoms, dtype=torch.bool)),
        "virial": cuda(torch.randn(2, 9, generator=generator)),
        "virial_mask": cuda(torch.ones(2, dtype=torch.bool)),
        "mforces": cuda(torch.randn(n_atoms, 3, generator=generator)),
        "mforce_mask": cuda(torch.ones(n_atoms, dtype=torch.bool)),
        "tau": cuda(torch.randn(n_atoms, 3, generator=generator)),
        "tau_mask": cuda(torch.ones(n_atoms, dtype=torch.bool)),
        "N": n_atoms,
        "num_structures": 2,
    }


def _optimizer(model, lr=2.0e-4):
    model.b1.requires_grad_(False)
    return torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.0, amsgrad=True, fused=True, capturable=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("compile_tail", [False, True])
def test_o3c2_graph_step_matches_fused_eager_adam(monkeypatch, compile_tail):
    monkeypatch.setenv("TORCHNEP_OC_CUDA_FULL", "0")
    monkeypatch.setenv("TORCHNEP_OC_CUDA_TRAIN", "0")
    monkeypatch.setenv("TORCHNEP_OC_CUDA_GRAPH", "1")
    monkeypatch.setenv(
        "TORCHNEP_OC_CUDA_COMPILE_TAIL", "1" if compile_tail else "0"
    )
    torch.manual_seed(11)
    base = NEPModel(_config()).cuda().train()
    eager = copy.deepcopy(base)
    captured = copy.deepcopy(base)
    batch = _batch(16)
    weights = (1.0, 0.7, 0.2, 0.5, 0.3)
    eager_optimizer = _optimizer(eager)
    graph_optimizer = _optimizer(captured)

    eager_optimizer.zero_grad(set_to_none=False)
    eager_loss, _ = _joint_loss_metrics(
        eager, batch, torch.tensor(weights, device="cuda"),
        True, True, True, True, "bmm",
    )
    eager_loss.backward()
    eager_optimizer.step()

    cache = OCCudaGraphTrainCache(
        captured, graph_optimizer, backend="bmm", warmup=1
    )
    actual = cache.step(
        batch, weights=weights, l1_coeff=0.0, l2_coeff=0.0,
        max_grad_norm=0.0, curriculum_scale=1.0,
        need_forces=True, need_virial=True, need_mforce=True, need_tau=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual.loss, eager_loss, rtol=2e-5, atol=2e-6)
    worst = max(
        (left - right).abs().max().item()
        for left, right in zip(captured.parameters(), eager.parameters())
    )
    assert worst < 3.0e-6
    steps = [
        float(state["step"].item())
        for state in graph_optimizer.state.values() if "step" in state
    ]
    assert steps and set(steps) == {1.0}
    assert actual.found_inf.item() == 0.0


_OC_MATRIX = list(itertools.product((1, 2, 3), (1, 2, 3, 5, 9), (0, 1, 2),
                                    (0, 1)))


def _assert_general_graph_joint_step(
    monkeypatch, config, seed, *, struct_center=False
):
    monkeypatch.setenv("TORCHNEP_OC_CUDA_TRAIN_GRAPH_CACHE", "1")
    monkeypatch.setenv("TORCHNEP_OC_CUDA_FULL", "0")
    monkeypatch.setenv("TORCHNEP_OC_CUDA_TRAIN", "0")
    monkeypatch.setenv("TORCHNEP_OC_CUDA_BANK", "0")
    monkeypatch.setenv("TORCHNEP_OC_CUDA_CENTER", "0")
    monkeypatch.setenv("TORCHNEP_OC_CUDA_STRUCT_CENTER", "0")
    monkeypatch.setenv("TORCHNEP_OC_CUDA_COMPILE_TAIL", "0")
    monkeypatch.setenv("TORCHNEP_OC_CUDA_GRAPH", "1")
    torch.manual_seed(seed)
    reference = NEPModel(config).cuda().train()
    captured = copy.deepcopy(reference)
    reference_optimizer = _optimizer(reference)
    graph_optimizer = _optimizer(captured)
    batch = _batch(6, neighbors=2, seed=3100)
    weights = (1.0, 0.7, 0.2, 0.5, 0.3)

    reference_optimizer.zero_grad(set_to_none=False)
    reference_loss, _ = _joint_loss_metrics(
        reference, batch, torch.tensor(weights, device="cuda"),
        True, True, True, True, "bmm",
    )
    reference_loss.backward()
    reference_optimizer.step()

    monkeypatch.setenv(
        "TORCHNEP_OC_CUDA_STRUCT_CENTER", "1" if struct_center else "0"
    )
    cache = OCCudaGraphTrainCache(
        captured, graph_optimizer, backend="bmm", warmup=1
    )
    actual = cache.step(
        batch, weights=weights, l1_coeff=0.0, l2_coeff=0.0,
        max_grad_norm=0.0, curriculum_scale=1.0,
        need_forces=True, need_virial=True, need_mforce=True, need_tau=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual.loss, reference_loss, rtol=3e-5, atol=3e-6)
    worst_abs = (0.0, "")
    worst_rel = (0.0, "")
    for (name, left), right in zip(
        captured.named_parameters(), reference.parameters()
    ):
        delta = (left - right).abs()
        max_abs = delta.max().item()
        max_rel = (delta / right.abs().clamp_min(1.0e-12)).max().item()
        if max_abs > worst_abs[0]:
            worst_abs = (max_abs, name)
        if max_rel > worst_rel[0]:
            worst_rel = (max_rel, name)
        torch.testing.assert_close(left, right, rtol=2.0e-3, atol=1.0e-5)
    return worst_abs, worst_rel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "order,compress,spin_l,soc", _OC_MATRIX,
    ids=lambda value: str(value),
)
def test_graph_joint_step_all_discrete_o_l_soc_and_runtime_c_samples(
    monkeypatch, order, compress, spin_l, soc
):
    """Every discrete O/L/SOC combination and C from scalar to B=9 bound."""
    config = _config(
        order=order, compress=compress, spin_l=spin_l, soc=soc,
        spin_basis=max(4, compress - 1), neuron=8,
        n_max=(1, 1), basis_size=(2, 2), l_max=[2, 1, 0],
    )
    _assert_general_graph_joint_step(
        monkeypatch, config,
        1000 + 100 * order + 10 * compress + 2 * spin_l + soc,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("case", (
    dict(order=1, compress=1, spin_l=0, soc=0, spin_basis=0,
         n_max=(0, 0), basis_size=(0, 0), neuron=1,
         l_max=[0, 0, 0], cutoffs=(3.7, 2.8), spin_cutoff=3.2),
    dict(order=2, compress=4, spin_l=1, soc=0, spin_basis=11,
         n_max=(1, 3), basis_size=(2, 5), neuron=7,
         l_max=[1, 0, 0], cutoffs=(5.1, 4.2), spin_cutoff=4.8),
    dict(order=3, compress=3, spin_l=2, soc=1, spin_basis=4,
         n_max=(3, 2), basis_size=(6, 7), neuron=16,
         l_max=[3, 1, 1, 1, 1, 1, 0],
         dof_types=["Fe"], env_types=["Fe", "Ge"], spin_cutoff=5.4),
    dict(order=3, compress=9, spin_l=2, soc=1, spin_basis=8,
         n_max=(7, 6), basis_size=(12, 10), neuron=64,
         l_max=[4, 1, 1, 1, 1, 1, 1], cutoffs=(6.7, 5.8),
         spin_cutoff=6.3),
))
def test_graph_joint_step_parameter_boundaries(monkeypatch, case):
    """Non-default radial widths, cutoffs, ANN widths and active-type masks."""
    _assert_general_graph_joint_step(
        monkeypatch, _config(**case), 7000 + case["compress"],
        struct_center=case["l_max"][0] > 0,
    )


_STRUCT_SECTOR_MATRIX = [
    [4, *flags] for flags in itertools.product((0, 1), repeat=6)
] + [[l_max, 0, 0, 0, 0, 0, 0] for l_max in range(4)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("l_max", _STRUCT_SECTOR_MATRIX)
def test_graph_joint_step_all_structural_sector_combinations(
    monkeypatch, l_max
):
    """All optional structural sectors plus every lower angular prefix."""
    flags = l_max[1:]
    config = _config(
        order=1, compress=1, spin_l=0, soc=0, spin_basis=4,
        neuron=8, n_max=(2, 2), basis_size=(3, 3), l_max=l_max,
    )
    _assert_general_graph_joint_step(
        monkeypatch, config, 5000 + l_max[0] * 100 + sum(
            bit << index for index, bit in enumerate(flags)
        ), struct_center=l_max[0] > 0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_graph_cache_is_exact_shape_keyed_and_bounded(monkeypatch):
    monkeypatch.setenv("TORCHNEP_OC_CUDA_FULL", "0")
    monkeypatch.setenv("TORCHNEP_OC_CUDA_TRAIN", "0")
    monkeypatch.setenv("TORCHNEP_OC_CUDA_GRAPH", "1")
    model = NEPModel(_config()).cuda().train()
    optimizer = _optimizer(model, lr=0.0)
    cache = OCCudaGraphTrainCache(
        model, optimizer, backend="bmm", max_entries=2, warmup=1
    )
    kwargs = dict(
        weights=(1.0, 0.7, 0.2, 0.5, 0.3), l1_coeff=0.0,
        l2_coeff=0.0, max_grad_norm=0.0, curriculum_scale=1.0,
        need_forces=True, need_virial=True, need_mforce=True, need_tau=True,
    )
    first = _batch(12)
    assert not cache.step(first, **kwargs).cache_hit
    assert cache.step(first, **kwargs).cache_hit
    assert not cache.step(_batch(14), **kwargs).cache_hit
    assert not cache.step(_batch(16), **kwargs).cache_hit
    stats = cache.stats()
    assert stats["entries"] == 2
    assert stats["hits"] == 1
    assert stats["misses"] == 3
    assert stats["evictions"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_train_nep_uses_bounded_graph_cache_for_changing_shapes(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("TORCHNEP_OC_CUDA_TRAIN_GRAPH_CACHE", "1")
    monkeypatch.setenv("TORCHNEP_OC_CUDA_GRAPH_CACHE_SIZE", "2")
    config = tmp_path / "nep.in"
    config.write_text(
        "type 1 Fe\n"
        "cutoff 6 5\n"
        "n_max 4 4\n"
        "basis_size 8 8\n"
        "l_max 4 2 0\n"
        "neuron 8\n"
        "spin_mode 2\n"
        "spin_compress 2\n"
        "spin_l_max 2\n"
        "spin_order 3\n"
        "spin_soc 1\n"
        "spin_cutoff 6\n"
        "spin_curriculum 0\n"
        "lambda_e 1\n"
        "lambda_f 0.7\n"
        "lambda_v 0.2\n"
        "lambda_m 0.5\n"
        "lambda_tau 0.3\n"
        "batch 1\n"
        "eval_batch 1\n"
        "epoch 2\n"
        "lr 0.0002\n"
    )
    xyz = tmp_path / "train.xyz"
    frames = []
    for n_atoms, energy in ((3, -3.0), (4, -3.8)):
        frames.extend([
            str(n_atoms),
            'Lattice="8 0 0 0 8 0 0 0 8" '
            f'energy={energy} virial="0.1 0 0 0 0.2 0 0 0 0.3" '
            "Properties=species:S:1:pos:R:3:force:R:3:spin:R:3:mforce:R:3",
        ])
        for atom in range(n_atoms):
            frames.append(
                f"Fe {1.0 + atom * 1.1} {1.0 + atom * 0.2} "
                f"{1.0 + atom * 0.1} 0 0 0 "
                f"{0.2 + atom * 0.1} {0.3 + atom * 0.05} "
                f"{0.4 - atom * 0.03} 0.01 0.02 0.03"
            )
    xyz.write_text("\n".join(frames) + "\n")
    output = tmp_path / "output"
    train_nep(
        str(config), str(xyz), output_dir=str(output), device="cuda",
        precision="float32", restart=False, prediction_interval=0,
        run_seed=3,
    )
    log = (output / "output.log").read_text()
    assert "O/C CUDA train graph: enabled" in log
    assert "graph 50.0% hit (2/2" in log
    assert (output / "nep_final.txt").stat().st_size > 0
