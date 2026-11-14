"""GPU parity tests: the b12x iHC operator versus the pure-torch reference.

Primary geometry is Hy4's: hidden 6144, ``fn`` FP32 ``[8, 4*6144]`` (the iHC
8-row layout), 4 streams, ``magnitude=2.0``, ``hc_eps=1e-6``, ``norm_eps=1e-5``.
The other whitelisted hidden sizes (4096/5120/7168, the mHC family) are covered
as spot checks. Tolerances are stated in ``ihc_tolerances``; the mutant-kill
tests use the same gates.
"""

import pytest
import torch

from b12x.norm.mhc import _ihc_torch_ref as ref
from b12x.norm.mhc import ihc

from ihc_tolerances import assert_close_bf16, assert_close_pair, assert_close_post

HC = ihc.IHC_MULT  # 4
MIXES = ihc.IHC_MIXES  # 8
MAGNITUDE = 2.0
HC_EPS = 1.0e-6
NORM_EPS = 1.0e-5

if not torch.cuda.is_available():
    pytest.skip("iHC operator tests require CUDA", allow_module_level=True)


def _make_inputs(*, tokens: int, hidden: int, seed: int = 0, outer=()):
    gen = torch.Generator().manual_seed(seed)
    shape = tuple(outer) + (tokens, HC, hidden)
    streams = (
        torch.randn(shape, generator=gen, dtype=torch.float32).cuda() / 3
    ).to(torch.bfloat16)
    fn = (
        torch.randn((MIXES, HC * hidden), generator=gen, dtype=torch.float32).cuda() / 64
    )
    hc_scale = torch.randn((2,), generator=gen, dtype=torch.float32).cuda() / 3
    hc_base = torch.randn((MIXES,), generator=gen, dtype=torch.float32).cuda() / 5
    return streams.contiguous(), fn.contiguous(), hc_scale.contiguous(), hc_base.contiguous()


def _make_branch(*, tokens: int, hidden: int, seed: int, outer=()):
    gen = torch.Generator().manual_seed(seed)
    shape = tuple(outer) + (tokens, hidden)
    return (
        torch.randn(shape, generator=gen, dtype=torch.float32).cuda() / 4
    ).to(torch.bfloat16).contiguous()


# (tokens, hidden): hidden 6144 at decode-ish, chunked and prefill-ish token
# counts is the Hy4 shape; the other hidden sizes are whitelist spot checks.
_PARITY_CASES = [
    (1, 6144),
    (2, 6144),
    (17, 6144),
    (384, 6144),
    (2048, 6144),
    (1, 4096),
    (384, 5120),
    (33, 7168),
]


@pytest.mark.parametrize(("tokens", "hidden"), _PARITY_CASES)
def test_pre_matches_reference(tokens: int, hidden: int) -> None:
    streams, fn, hc_scale, hc_base = _make_inputs(
        tokens=tokens, hidden=hidden, seed=tokens * 7 + hidden
    )
    got = ihc.run_pre(streams, fn, hc_scale, hc_base)
    want = ref.ihc_pre_reference(
        streams,
        fn,
        hc_scale,
        hc_base,
        magnitude=MAGNITUDE,
        hc_eps=HC_EPS,
        norm_eps=NORM_EPS,
    )
    assert got[0].shape == want[0].shape == (tokens, HC)
    assert got[1].shape == want[1].shape == (tokens, hidden)
    assert got[0].dtype == torch.float32 and got[1].dtype == torch.bfloat16
    assert_close_pair(got, want)


@pytest.mark.parametrize(("tokens", "hidden"), _PARITY_CASES)
def test_post_matches_reference(tokens: int, hidden: int) -> None:
    streams, fn, hc_scale, hc_base = _make_inputs(
        tokens=tokens, hidden=hidden, seed=tokens * 11 + hidden
    )
    branch = _make_branch(tokens=tokens, hidden=hidden, seed=tokens + hidden)
    post, _ = ihc.run_pre(streams, fn, hc_scale, hc_base)
    got = ihc.run_post(branch, streams, post)
    want = ref.ihc_post_reference(branch, streams, post)
    assert got.shape == streams.shape and got.dtype == torch.bfloat16
    assert_close_bf16(got, want)


@pytest.mark.parametrize(("tokens", "hidden"), _PARITY_CASES)
def test_head_matches_reference(tokens: int, hidden: int) -> None:
    streams, _, _, _ = _make_inputs(tokens=tokens, hidden=hidden, seed=tokens * 13)
    gen = torch.Generator().manual_seed(hidden + 1)
    fn = (torch.randn((HC, HC * hidden), generator=gen, dtype=torch.float32).cuda() / 64)
    scale = (torch.randn((1,), generator=gen, dtype=torch.float32).cuda() / 3)
    base = (torch.randn((HC,), generator=gen, dtype=torch.float32).cuda() / 5)
    got = ihc.run_head(streams, fn.contiguous(), scale.contiguous(), base.contiguous())
    want = ref.ihc_head_reference(streams, fn, scale, base, hc_eps=HC_EPS, norm_eps=NORM_EPS)
    assert got.shape == (tokens, hidden) and got.dtype == torch.bfloat16
    assert_close_bf16(got, want)


@pytest.mark.parametrize(("tokens", "hidden"), [(1, 6144), (384, 6144)])
def test_torch_and_triton_backends_agree(tokens: int, hidden: int, monkeypatch) -> None:
    streams, fn, hc_scale, hc_base = _make_inputs(
        tokens=tokens, hidden=hidden, seed=99 + tokens
    )
    branch = _make_branch(tokens=tokens, hidden=hidden, seed=tokens)
    monkeypatch.setenv("B12X_IHC_BACKEND", "triton")
    post_t, y_t = ihc.run_pre(streams, fn, hc_scale, hc_base)
    out_t = ihc.run_post(branch, streams, post_t)
    monkeypatch.setenv("B12X_IHC_BACKEND", "torch")
    post_e, y_e = ihc.run_pre(streams, fn, hc_scale, hc_base)
    out_e = ihc.run_post(branch, streams, post_e)
    assert_close_post(post_t, post_e)
    assert_close_bf16(y_t, y_e)
    assert_close_bf16(out_t, out_e)


@pytest.mark.parametrize("hidden", [6144, 4096, 5120, 7168])
def test_tuning_table_covers_whitelist(hidden: int) -> None:
    """Every whitelisted hidden size — including Hy4's 6144 — has a tuned row
    that satisfies the mHC kernel tile divisibility."""
    config = ihc.tuning_for(hidden)
    assert config.hidden_size == hidden
    assert hidden in ihc.IHC_TUNING
    assert hidden % ihc.IHC_SOURCE_TILE_H == 0
    assert hidden % ihc.IHC_GRAM_BLOCK_H == 0
    assert (HC * hidden) % config.block_k == 0
    assert hidden % config.block_d == 0


def test_whitelist_accepts_6144_and_rejects_others() -> None:
    # 6144 is the point of the exercise: the mHC whitelist is
    # 4096/5120/7168 and iHC extends it.
    assert 6144 in ihc.IHC_SUPPORTED_HIDDEN_SIZES
    for bad in (3072, 5000, 8192):
        streams, fn, hc_scale, hc_base = _make_inputs(tokens=1, hidden=bad, seed=1)
        with pytest.raises(ValueError, match="6144"):
            ihc.run_pre(streams, fn, hc_scale, hc_base)
        with pytest.raises(ValueError, match="6144"):
            ihc.tuning_for(bad)


def test_rejects_mhc_24_row_fn_layout() -> None:
    """iHC needs the 8-row layout; mHC's 24-row weight (comb rows included)
    must not be silently accepted."""
    streams, _, hc_scale, hc_base = _make_inputs(tokens=2, hidden=6144, seed=2)
    for rows in (24, 6, 4):
        fn = torch.zeros((rows, 4 * 6144), dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError, match="8-row"):
            ihc.run_pre(streams, fn, hc_scale, hc_base)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s, f, sc, b: (s.half(), f, sc, b),
        lambda s, f, sc, b: (s, f.to(torch.bfloat16), sc, b),
        lambda s, f, sc, b: (s, f, sc[:1], b),
        lambda s, f, sc, b: (s, f, sc, b[:4]),
        lambda s, f, sc, b: (s.transpose(1, 2).contiguous().transpose(1, 2), f, sc, b),
    ],
    ids=["streams_fp16", "fn_bf16", "scale_rows", "base_rows", "noncontiguous"],
)
def test_rejects_bad_contracts(mutate) -> None:
    streams, fn, hc_scale, hc_base = _make_inputs(tokens=2, hidden=6144, seed=3)
    streams, fn, hc_scale, hc_base = mutate(streams, fn, hc_scale, hc_base)
    with pytest.raises(ValueError):
        ihc.run_pre(streams, fn, hc_scale, hc_base)


def test_out_buffers_match_functional() -> None:
    tokens, hidden = 33, 6144
    streams, fn, hc_scale, hc_base = _make_inputs(tokens=tokens, hidden=hidden, seed=4)
    branch = _make_branch(tokens=tokens, hidden=hidden, seed=5)
    out_post = torch.empty((tokens, HC), dtype=torch.float32, device="cuda")
    out_y = torch.empty((tokens, hidden), dtype=torch.bfloat16, device="cuda")
    post, y = ihc.run_pre(
        streams, fn, hc_scale, hc_base, out_post=out_post, out_y=out_y
    )
    assert post is out_post and y is out_y
    f_post, f_y = ihc.run_pre(streams, fn, hc_scale, hc_base)
    assert_close_post(post, f_post)
    assert_close_bf16(y, f_y)

    out = torch.empty_like(streams)
    got = ihc.run_post(branch, streams, post, out=out)
    assert got is out
    assert_close_bf16(out, ihc.run_post(branch, streams, f_post))

    out_h = torch.empty((tokens, hidden), dtype=torch.bfloat16, device="cuda")
    gen = torch.Generator().manual_seed(6)
    hfn = (
        torch.randn((HC, HC * hidden), generator=gen, dtype=torch.float32).cuda() / 64
    ).contiguous()
    hs = (
        torch.randn((1,), generator=gen, dtype=torch.float32).cuda() / 3
    ).contiguous()
    hb = (
        torch.randn((HC,), generator=gen, dtype=torch.float32).cuda() / 5
    ).contiguous()
    got_h = ihc.run_head(streams, hfn, hs, hb, out=out_h)
    assert got_h is out_h
    assert_close_bf16(out_h, ihc.run_head(streams, hfn, hs, hb))


def test_leading_dims_and_post_4x1() -> None:
    """Rank-4 [B, T, 4, H] streams (the HF layout) and [..., 4, 1] post gates."""
    outer, tokens, hidden = (2,), 7, 6144
    streams, fn, hc_scale, hc_base = _make_inputs(
        tokens=tokens, hidden=hidden, seed=7, outer=outer
    )
    branch = _make_branch(tokens=tokens, hidden=hidden, seed=8, outer=outer)
    post, y = ihc.run_pre(streams, fn, hc_scale, hc_base)
    assert post.shape == outer + (tokens, HC) and y.shape == outer + (tokens, hidden)
    want = ref.ihc_pre_reference(streams, fn, hc_scale, hc_base)
    assert_close_pair((post, y), want)
    out = ihc.run_post(branch, streams, post.unsqueeze(-1))
    assert_close_bf16(out, ref.ihc_post_reference(branch, streams, post))


def test_zero_tokens_are_served() -> None:
    streams, fn, hc_scale, hc_base = _make_inputs(tokens=0, hidden=6144, seed=9)
    post, y = ihc.run_pre(streams, fn, hc_scale, hc_base)
    assert post.shape == (0, HC) and y.shape == (0, 6144)
    branch = _make_branch(tokens=0, hidden=6144, seed=10)
    out = ihc.run_post(branch, streams, post)
    assert out.shape == streams.shape
