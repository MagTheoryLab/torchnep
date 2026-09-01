import copy
import os

import pytest
import torch

from torchnep import train_nep
from torchnep.model import NEPModel
from torchnep.oc_cuda_graph import OCCudaGraphTrainCache, _joint_loss_metrics


def _config():
    return {
        "num_types": 3,
        "type_names": ["Fe", "Ge", "Ni"],
        "cutoff_radial": 6.0,
        "cutoff_angular": 5.0,
        "n_max_radial": 4,
        "n_max_angular": 4,
        "basis_size_radial": 8,
        "basis_size_angular": 8,
        "l_max": [4, 2, 0],
        "neuron": 30,
        "spin_mode": 2,
        "spin_cutoff": [6.0, 6.0],
        "spin_l_max": [2, 0, 0],
        "spin_compress": 2,
        "spin_basis_size": [8, 0],
        "spin_order": 3,
        "spin_soc": 1,
        "spin_curriculum": False,
        "spin_dof_type": ["Fe", "Ge", "Ni"],
        "spin_env_type": ["Fe", "Ge", "Ni"],
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
    assert "O3-C2 CUDA train graph: enabled" in log
    assert "graph 50.0% hit (2/2" in log
    assert (output / "nep_final.txt").stat().st_size > 0
