import torch
import pytest

from torchnep.constants import C3B, C4B
from torchnep.oc_cuda_struct_center import structural_center_sector


def _reference(moments, c3b, c4b):
    blocks = []
    for li in range(4):
        order = li + 1
        width = 2 * order + 1
        start = order * order - 1
        values = moments[:, :, start:start + width].square()
        block = c3b[start] * values[:, :, 0]
        block = block + 2.0 * (
            c3b[start + 1:start + width] * values[:, :, 1:]
        ).sum(-1)
        blocks.append(block)
    s20, s21r, s21i = moments[:, :, 3:6].unbind(-1)
    s22r, s22i = moments[:, :, 6:8].unbind(-1)
    q4 = (
        c4b[0] * s20**3
        + c4b[1] * s20 * (s21r**2 + s21i**2)
        + c4b[2] * s20 * (s22r**2 + s22i**2)
        + c4b[3] * s22r * (s21i**2 - s21r**2)
        + c4b[4] * s21r * s21i * s22i
    )
    return torch.cat((*blocks, q4), dim=1)


def _candidate(moments, c3b, c4b):
    return structural_center_sector(
        moments, c3b, c4b,
        n_max_angular=4, l_max_3b=4, has_q_222=1,
        has_q_1111=0, has_q_112=0, has_q_123=0,
        has_q_233=0, has_q_134=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_structural_center_forward_vjp_hvp(monkeypatch):
    monkeypatch.setenv("TORCHNEP_OC_CUDA_PREBUILT", "1")
    monkeypatch.setenv("TORCHNEP_OC_CUDA_STRUCT_CENTER", "1")
    generator = torch.Generator(device="cuda").manual_seed(20260901)
    ref_moments = torch.randn(
        37, 5, 24, generator=generator, device="cuda", dtype=torch.float64,
        requires_grad=True,
    )
    cuda_moments = ref_moments.detach().clone().requires_grad_(True)
    c3b = torch.tensor(C3B[:24], device="cuda", dtype=torch.float64)
    c4b = torch.tensor(C4B, device="cuda", dtype=torch.float64)
    expected = _reference(ref_moments, c3b, c4b)
    actual = _candidate(cuda_moments, c3b, c4b)
    pull = torch.randn(
        actual.shape, generator=generator, device="cuda", dtype=torch.float64
    )
    first_ref, = torch.autograd.grad(
        (expected * pull).sum(), (ref_moments,), create_graph=True
    )
    first_cuda, = torch.autograd.grad(
        (actual * pull).sum(), (cuda_moments,), create_graph=True
    )
    tangent = torch.randn(
        first_ref.shape, generator=generator, device="cuda", dtype=torch.float64
    )
    second_ref, = torch.autograd.grad(
        (first_ref * tangent).sum(), (ref_moments,)
    )
    second_cuda, = torch.autograd.grad(
        (first_cuda * tangent).sum(), (cuda_moments,)
    )
    _, jvp = torch.autograd.functional.jvp(
        lambda value: _candidate(value, c3b, c4b),
        (cuda_moments,), (tangent,),
    )
    lhs = (jvp * pull).sum()
    vjp, = torch.autograd.grad(
        (_candidate(cuda_moments, c3b, c4b) * pull).sum(),
        (cuda_moments,),
    )
    rhs = (vjp * tangent).sum()
    forward_abs = (actual - expected).abs().max().item()
    vjp_abs = (first_cuda - first_ref).abs().max().item()
    hvp_abs = (second_cuda - second_ref).abs().max().item()
    adjoint_abs = (lhs - rhs).abs().item()
    print(
        f"struct_center forward={forward_abs:.3e} vjp={vjp_abs:.3e} "
        f"hvp={hvp_abs:.3e} adjoint={adjoint_abs:.3e}"
    )
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(first_cuda, first_ref, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(second_cuda, second_ref, rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(lhs, rhs, rtol=3e-12, atol=3e-12)
