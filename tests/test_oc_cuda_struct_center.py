import pytest
import torch

from torchnep.constants import (
    C3B, C4B, C5B, C4B2, Q123_TERMS, Q233_TERMS, Q134_TERMS,
)
from torchnep.oc_cuda_struct_center import structural_center_sector
from torchnep.ops import _eval_extra


def _reference(moments, c3b, c4b, c5b, c4b2, l_max, flags):
    blocks = []
    for order in range(1, l_max + 1):
        width = 2 * order + 1
        start = order * order - 1
        values = moments[:, :, start:start + width].square()
        block = c3b[start] * values[:, :, 0]
        block = block + 2.0 * (
            c3b[start + 1:start + width] * values[:, :, 1:]
        ).sum(-1)
        blocks.append(block)
    q222, q1111, q112, q123, q233, q134 = flags
    if q222:
        a, b, c, d, e = moments[:, :, 3:8].unbind(-1)
        blocks.append(
            c4b[0] * a**3 + c4b[1] * a * (b**2 + c**2)
            + c4b[2] * a * (d**2 + e**2)
            + c4b[3] * d * (c**2 - b**2) + c4b[4] * b * c * e
        )
    if q1111:
        a, b, c = moments[:, :, :3].unbind(-1)
        aa, ss = a.square(), b.square() + c.square()
        blocks.append(c5b[0] * aa.square() + c5b[1] * aa * ss
                      + c5b[2] * ss.square())
    if q112:
        a, b, c, d, e, f, g, h = moments[:, :, :8].unbind(-1)
        blocks.append(
            c4b2[0] * a * a * d + c4b2[1] * a * (b * e + c * f)
            + c4b2[2] * d * (b * b + c * c)
            + c4b2[3] * g * (b * b - c * c) + c4b2[4] * b * c * h
        )
    if q123:
        blocks.append(_eval_extra(moments, Q123_TERMS))
    if q233:
        blocks.append(_eval_extra(moments, Q233_TERMS))
    if q134:
        blocks.append(_eval_extra(moments, Q134_TERMS))
    return torch.cat(blocks, dim=1)


def _candidate(moments, coefficients, l_max, flags):
    c3b, c4b, c5b, c4b2 = coefficients
    return structural_center_sector(
        moments, c3b, c4b, c5b, c4b2,
        n_max_angular=moments.shape[1] - 1, l_max_3b=l_max,
        has_q_222=flags[0], has_q_1111=flags[1], has_q_112=flags[2],
        has_q_123=flags[3], has_q_233=flags[4], has_q_134=flags[5],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("radial,l_max,flags", [
    (1, 1, (0, 0, 0, 0, 0, 0)),
    (3, 2, (1, 1, 1, 0, 0, 0)),
    (5, 3, (1, 1, 1, 1, 1, 0)),
    (2, 4, (1, 1, 1, 1, 1, 1)),
])
def test_structural_center_forward_vjp_hvp(
    monkeypatch, radial, l_max, flags
):
    monkeypatch.setenv("TORCHNEP_OC_CUDA_PREBUILT", "1")
    monkeypatch.setenv("TORCHNEP_OC_CUDA_STRUCT_CENTER", "1")
    generator = torch.Generator(device="cuda").manual_seed(
        20260901 + 10 * radial + l_max
    )
    num_lm = l_max * (l_max + 2)
    ref_moments = torch.randn(
        37, radial, num_lm, generator=generator, device="cuda",
        dtype=torch.float64, requires_grad=True,
    )
    cuda_moments = ref_moments.detach().clone().requires_grad_(True)
    coefficients = tuple(torch.tensor(value, device="cuda", dtype=torch.float64)
                         for value in (C3B[:num_lm], C4B, C5B, C4B2))
    expected = _reference(ref_moments, *coefficients, l_max, flags)
    actual = _candidate(cuda_moments, coefficients, l_max, flags)
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
        lambda value: _candidate(value, coefficients, l_max, flags),
        (cuda_moments,), (tangent,),
    )
    lhs = (jvp * pull).sum()
    vjp, = torch.autograd.grad(
        (_candidate(cuda_moments, coefficients, l_max, flags) * pull).sum(),
        (cuda_moments,),
    )
    rhs = (vjp * tangent).sum()
    print(
        f"struct_center R={radial} L={l_max} flags={flags} "
        f"forward={(actual - expected).abs().max().item():.3e} "
        f"vjp={(first_cuda - first_ref).abs().max().item():.3e} "
        f"hvp={(second_cuda - second_ref).abs().max().item():.3e} "
        f"adjoint={(lhs - rhs).abs().item():.3e}"
    )
    torch.testing.assert_close(actual, expected, rtol=3e-13, atol=3e-13)
    torch.testing.assert_close(first_cuda, first_ref, rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(second_cuda, second_ref, rtol=5e-12, atol=5e-12)
    torch.testing.assert_close(lhs, rhs, rtol=5e-12, atol=5e-12)
