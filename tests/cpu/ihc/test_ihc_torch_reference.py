"""CPU tests for the pure-torch iHC reference.

The reference is pinned against an independent per-element float64 loop
formulation, so the golden model itself cannot drift silently (the failure mode
that kept the split-half RoPE bug alive: two implementations sharing one
convention). Gate semantics are pinned explicitly because the GPU mutant gates
(pre/post swapped, post magnitude 1.0) rely on them.
"""

import math

import pytest
import torch

from b12x.norm.mhc import _ihc_torch_ref as ihc_ref

pytestmark = pytest.mark.cpu_only

HC = ihc_ref.IHC_MULT  # 4
MIXES = ihc_ref.IHC_MIXES  # 8
MAGNITUDE = 2.0
HC_EPS = 1.0e-6
NORM_EPS = 1.0e-5


def _make_inputs(*, tokens: int, hidden: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    streams = (
        torch.randn((tokens, HC, hidden), generator=gen, dtype=torch.float32) / 3
    ).to(torch.bfloat16)
    fn = (
        torch.randn((MIXES, HC * hidden), generator=gen, dtype=torch.float32) / 64
    )
    hc_scale = torch.randn((2,), generator=gen, dtype=torch.float32) / 3
    hc_base = torch.randn((MIXES,), generator=gen, dtype=torch.float32) / 5
    return streams, fn, hc_scale, hc_base


def _naive_pre(streams, fn, hc_scale, hc_base, *, magnitude, hc_eps, norm_eps):
    """Per-element float64 loop formulation of the iHC pre contract."""
    t_n, _, hidden = streams.shape
    post = torch.zeros((t_n, HC), dtype=torch.float64)
    y = torch.zeros((t_n, hidden), dtype=torch.float64)
    for t in range(t_n):
        x = streams[t].double()  # [4, H]
        flat = x.reshape(-1)  # [4H]
        msq = (flat * flat).mean().item() + norm_eps
        rsqrt = 1.0 / math.sqrt(msq)
        for s in range(HC):
            pre_logit = hc_base[s].item()
            post_logit = hc_base[HC + s].item()
            for k in range(HC * hidden):
                v = flat[k].item() * fn[s, k].item() * rsqrt
                pre_logit += v * hc_scale[0].item()
                post_logit += (
                    flat[k].item() * fn[HC + s, k].item() * rsqrt * hc_scale[1].item()
                )
            post[t, s] = magnitude / (1.0 + math.exp(-post_logit)) + hc_eps
            pre = 1.0 / (1.0 + math.exp(-pre_logit)) + hc_eps
            y[t] += pre * x[s]
    return post, y


def _naive_post(x, residual, post):
    t_n, _, hidden = residual.shape
    out = torch.zeros_like(residual, dtype=torch.float64)
    for t in range(t_n):
        for s in range(HC):
            for d in range(hidden):
                out[t, s, d] = (
                    post[t, s].item() * x[t, d].item() + residual[t, s, d].item()
                )
    return out


def _naive_head(streams, fn, hc_head_scale, hc_head_base, *, hc_eps, norm_eps):
    t_n, _, hidden = streams.shape
    y = torch.zeros((t_n, hidden), dtype=torch.float64)
    for t in range(t_n):
        x = streams[t].double()
        flat = x.reshape(-1)
        rsqrt = 1.0 / math.sqrt((flat * flat).mean().item() + norm_eps)
        for s in range(HC):
            logit = hc_head_base[s].item()
            for k in range(HC * hidden):
                logit += flat[k].item() * fn[s, k].item() * rsqrt * hc_head_scale[0].item()
            pre = 1.0 / (1.0 + math.exp(-logit)) + hc_eps
            y[t] += pre * x[s]
    return y


def _assert_close_fp64_anchor(got: torch.Tensor, anchor: torch.Tensor) -> None:
    # The reference runs FP32; the anchor is float64 loops. Only FP32 rounding
    # separates them over short rows.
    torch.testing.assert_close(
        got.double(), anchor, atol=5e-6, rtol=5e-6, msg=lambda m: f"vs anchor: {m}"
    )


@pytest.mark.parametrize(("tokens", "hidden"), [(1, 6), (3, 12), (2, 5)])
def test_pre_matches_naive_float64(tokens: int, hidden: int) -> None:
    streams, fn, hc_scale, hc_base = _make_inputs(tokens=tokens, hidden=hidden, seed=1)
    post, y = ihc_ref.ihc_pre_reference(
        streams,
        fn,
        hc_scale,
        hc_base,
        magnitude=MAGNITUDE,
        hc_eps=HC_EPS,
        norm_eps=NORM_EPS,
        y_dtype=torch.float32,
    )
    n_post, n_y = _naive_pre(
        streams,
        fn,
        hc_scale,
        hc_base,
        magnitude=MAGNITUDE,
        hc_eps=HC_EPS,
        norm_eps=NORM_EPS,
    )
    assert post.dtype == torch.float32 and y.dtype == torch.float32
    assert post.shape == (tokens, HC) and y.shape == (tokens, hidden)
    _assert_close_fp64_anchor(post, n_post)
    _assert_close_fp64_anchor(y, n_y)


@pytest.mark.parametrize(("tokens", "hidden"), [(1, 6), (3, 12)])
def test_post_matches_naive_float64(tokens: int, hidden: int) -> None:
    gen = torch.Generator().manual_seed(2)
    x = (torch.randn((tokens, hidden), generator=gen) / 4).to(torch.bfloat16)
    residual = (torch.randn((tokens, HC, hidden), generator=gen) / 3).to(torch.bfloat16)
    post = torch.randn((tokens, HC), generator=gen, dtype=torch.float32)
    got = ihc_ref.ihc_post_reference(x, residual, post)
    anchor = _naive_post(x, residual, post)
    assert got.dtype == torch.bfloat16 and got.shape == residual.shape
    # BF16 rounding of the FP32 accumulation dominates here (~1e-2 absolute at
    # these magnitudes); the FP32 math itself matches the float64 anchor.
    torch.testing.assert_close(got.float().double(), anchor, atol=1e-2, rtol=1e-2)
    err = (got.float().double() - anchor).norm() / anchor.norm()
    assert err < 5e-3, f"rel-norm diff {err:.4g} exceeds 5e-3"


@pytest.mark.parametrize(("tokens", "hidden"), [(1, 6), (2, 11)])
def test_head_matches_naive_float64(tokens: int, hidden: int) -> None:
    streams, _, _, _ = _make_inputs(tokens=tokens, hidden=hidden, seed=3)
    gen = torch.Generator().manual_seed(4)
    fn = torch.randn((HC, HC * hidden), generator=gen, dtype=torch.float32) / 64
    scale = torch.randn((1,), generator=gen, dtype=torch.float32) / 3
    base = torch.randn((HC,), generator=gen, dtype=torch.float32) / 5
    y = ihc_ref.ihc_head_reference(
        streams, fn, scale, base, hc_eps=HC_EPS, norm_eps=NORM_EPS, y_dtype=torch.float32
    )
    anchor = _naive_head(streams, fn, scale, base, hc_eps=HC_EPS, norm_eps=NORM_EPS)
    _assert_close_fp64_anchor(y, anchor)


def test_gate_ranges_pin_magnitude_two_and_eps() -> None:
    """Pin the exact gate formulas the GPU mutants must not be able to fake.

    pre  in (hc_eps, 1 + hc_eps); post in (hc_eps, magnitude + hc_eps) with the
    default magnitude 2.0. A post-magnitude-1.0 mutant therefore cannot land in
    the reference range, and swapped pre/post gates cannot either (their ranges
    differ by the factor of magnitude).
    """
    streams, fn, hc_scale, hc_base = _make_inputs(tokens=4, hidden=8, seed=5)
    post, _ = ihc_ref.ihc_pre_reference(
        streams, fn, hc_scale, hc_base, hc_eps=HC_EPS, norm_eps=NORM_EPS
    )
    assert ihc_ref.ihc_pre_reference.__kwdefaults__["magnitude"] == 2.0
    assert torch.all(post > HC_EPS) and torch.all(post < MAGNITUDE + HC_EPS)

    post1, _ = ihc_ref.ihc_pre_reference(
        streams,
        fn,
        hc_scale,
        hc_base,
        magnitude=1.0,
        hc_eps=HC_EPS,
        norm_eps=NORM_EPS,
    )
    # magnitude 1.0 caps the gate at 1 + eps; with magnitude 2.0 roughly half of
    # the gates sit above 1 + eps for this input.
    assert torch.all(post1 < 1.0 + HC_EPS)
    assert (post > 1.0 + HC_EPS).float().mean() > 0.2


def test_hc_eps_is_an_additive_shift() -> None:
    streams, fn, hc_scale, hc_base = _make_inputs(tokens=3, hidden=8, seed=6)
    shift = 1.0e-3
    post_a, y_a = ihc_ref.ihc_pre_reference(
        streams, fn, hc_scale, hc_base, hc_eps=0.0, norm_eps=NORM_EPS
    )
    post_b, y_b = ihc_ref.ihc_pre_reference(
        streams, fn, hc_scale, hc_base, hc_eps=shift, norm_eps=NORM_EPS
    )
    torch.testing.assert_close(post_b - post_a, torch.full_like(post_a, shift))
    # y changes only through the shifted pre gates; it must change at all (so a
    # dropped-eps mutant is observable) but stays close.
    assert not torch.equal(y_a, y_b)
    assert (y_b - y_a).abs().max() < 10 * shift * streams.abs().amax()


def test_fn_layout_is_8_rows_no_comb() -> None:
    streams, fn, hc_scale, hc_base = _make_inputs(tokens=2, hidden=8, seed=7)
    assert fn.shape == (8, 4 * 8)
    with pytest.raises(ValueError, match="8-row"):
        ihc_ref.ihc_pre_reference(
            streams, fn[:6], hc_scale, hc_base  # pre-only rows: not the iHC layout
        )
    with pytest.raises(ValueError, match="8-row"):
        # mHC's 24-row layout must be rejected: iHC has no comb rows.
        big = torch.zeros((24, 4 * 8))
        ihc_ref.ihc_pre_reference(streams, big, hc_scale, hc_base)


def test_pipeline_composition_contract() -> None:
    """One full sublayer: pre -> branch -> post update, per the plan's formula.

    update = post_i * branch + stream_i  (no comb term), then the head merges.
    """
    tokens, hidden = 3, 10
    streams, fn, hc_scale, hc_base = _make_inputs(tokens=tokens, hidden=hidden, seed=8)
    gen = torch.Generator().manual_seed(9)
    branch_fn = torch.randn((hidden, hidden), generator=gen, dtype=torch.float32)
    branch = torch.nn.functional.linear(
        streams.sum(dim=1).float(), branch_fn
    ).to(torch.bfloat16)
    post, y = ihc_ref.ihc_pre_reference(
        streams, fn, hc_scale, hc_base, hc_eps=HC_EPS, norm_eps=NORM_EPS
    )
    out = ihc_ref.ihc_post_reference(branch, streams, post)
    # Contract checks against the plan statement, computed independently.
    # The contract accumulates in FP32 and rounds to the branch dtype once.
    expected = (
        post.unsqueeze(-1).float() * branch.float().unsqueeze(-2) + streams.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    assert not torch.equal(out, streams)  # the update actually moves the streams
    head_fn = torch.randn((HC, HC * hidden), generator=gen, dtype=torch.float32) / 64
    merged = ihc_ref.ihc_head_reference(
        out,
        head_fn,
        torch.tensor([0.01]),
        torch.zeros((HC,)),
        hc_eps=HC_EPS,
        norm_eps=NORM_EPS,
    )
    assert merged.shape == (tokens, hidden) and merged.dtype == torch.bfloat16
