"""Hy4 indexer keys: interleaved RoPE at write time (plan item 4).

128-dim indexer keys, rotary pe in the LAST ``qk_rope_head_dim`` = 64 dims,
``is_neox_style=False`` adjacent-pair interleaved rotation, ``rope_theta=1e7``,
``k_norm`` (LayerNorm over all 128 dims) applied BEFORE the split, keys
rotated at cache-write time (``pack_index_k_cache_reference`` writing the
paged index cache).

Witness: vLLM ``ref/vllm_hy4_attention.py`` ``Indexer.prepare_inputs``.

Mutant that must die here: the split-half (NeoX) rotation — pairs ``(i,
i+32)`` instead of the checkpoint's interleaved ``(2i, 2i+1)``.  It rotates
the wrong dims, breaks the trained relative-position geometry and corrupts
the DSA top-k selection.
"""

from __future__ import annotations

import math

import pytest
import torch

from b12x.attention.dsa_indexer.reference import (
    pack_index_k_cache_reference,
    paged_decode_logits_reference,
    unpack_index_k_cache_reference,
)
from b12x.attention.dsa_indexer.rope import (
    INDEX_HEAD_DIM,
    INDEX_ROPE_DIM,
    INDEX_ROPE_THETA,
    apply_indexer_rope,
    indexer_k_norm,
    indexer_rope_inv_freq,
    prepare_indexer_keys,
)

_NOPE_DIM = INDEX_HEAD_DIM - INDEX_ROPE_DIM  # 64


# ──────────────────────── independent witness math ─────────────────────────


def _witness_prepare_indexer_keys(
    k: torch.Tensor,
    positions: torch.Tensor,
    *,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scalar transcription of vLLM's Indexer.prepare_inputs (fp64)."""
    tokens = int(k.shape[0])
    out = torch.empty_like(k, dtype=torch.float64)
    for t in range(tokens):
        row = k[t].to(torch.float64)
        # nn.LayerNorm(128, eps=1e-6) over the FULL key, before the split.
        mean = float(row.mean())
        var = float(((row - mean) ** 2).mean())
        row = (row - mean) / math.sqrt(var + eps)
        if weight is not None:
            row = row * weight.to(torch.float64)
        if bias is not None:
            row = row + bias.to(torch.float64)
        out[t, :_NOPE_DIM] = row[:_NOPE_DIM]
        for i in range(INDEX_ROPE_DIM // 2):
            angle = float(positions[t]) * INDEX_ROPE_THETA ** (
                -2.0 * i / INDEX_ROPE_DIM
            )
            c, s = math.cos(angle), math.sin(angle)
            x0 = float(row[_NOPE_DIM + 2 * i])
            x1 = float(row[_NOPE_DIM + 2 * i + 1])
            out[t, _NOPE_DIM + 2 * i] = x0 * c - x1 * s
            out[t, _NOPE_DIM + 2 * i + 1] = x0 * s + x1 * c
    return out


def _mutant_split_half_keys(
    k: torch.Tensor,
    positions: torch.Tensor,
    *,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """The checkpoint bug: NeoX/split-half rotation of the pe block.

    Pairs ``(i, i + 32)`` within the last 64 dims instead of the interleaved
    ``(2i, 2i + 1)``; same inv_freq, so only the dim pairing is wrong.
    """
    k_normed = indexer_k_norm(k, weight=weight, bias=bias, eps=eps).to(torch.float64)
    half = INDEX_ROPE_DIM // 2
    out = k_normed.clone()
    pe = k_normed[:, _NOPE_DIM:]
    inv = indexer_rope_inv_freq(
        INDEX_ROPE_DIM, INDEX_ROPE_THETA, dtype=torch.float64
    )
    ang = positions.to(torch.float64).unsqueeze(1) * inv.unsqueeze(0)  # [t, 32]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = pe[:, :half], pe[:, half:]
    out[:, _NOPE_DIM : _NOPE_DIM + half] = x1 * cos - x2 * sin
    out[:, _NOPE_DIM + half :] = x1 * sin + x2 * cos
    return out


def _corpus(seed: int, tokens: int = 97):
    torch.manual_seed(seed)
    k = torch.randn(tokens, INDEX_HEAD_DIM, dtype=torch.float32)
    weight = torch.randn(INDEX_HEAD_DIM, dtype=torch.float32) * 0.5 + 1.0
    bias = torch.randn(INDEX_HEAD_DIM, dtype=torch.float32) * 0.1
    positions = torch.arange(tokens, dtype=torch.int64) * 3 + 7
    return k, weight, bias, positions


# ─────────────────────────── witness equivalence ───────────────────────────


@pytest.mark.cpu_only
def test_key_rope_matches_vllm_witness() -> None:
    k, weight, bias, positions = _corpus(seed=20261114)
    golden = _witness_prepare_indexer_keys(
        k, positions, weight=weight, bias=bias
    )
    actual = prepare_indexer_keys(
        k, positions, k_norm_weight=weight, k_norm_bias=bias
    )
    torch.testing.assert_close(
        actual.double(), golden, rtol=1e-5, atol=1e-6
    )
    # FP64 input tightens to machine agreement with the witness loops.
    actual64 = prepare_indexer_keys(
        k.double(), positions, k_norm_weight=weight, k_norm_bias=bias
    )
    torch.testing.assert_close(actual64, golden, rtol=1e-12, atol=1e-12)


@pytest.mark.cpu_only
def test_pe_block_lives_in_the_last_64_dims() -> None:
    """Rotation touches ONLY the trailing qk_rope_head_dim block (PTM layout)."""
    k, weight, bias, positions = _corpus(seed=11)
    normed = indexer_k_norm(k, weight=weight, bias=bias)
    out = prepare_indexer_keys(k, positions, k_norm_weight=weight, k_norm_bias=bias)
    torch.testing.assert_close(out[:, :_NOPE_DIM], normed[:, :_NOPE_DIM])
    assert not torch.allclose(out[:, _NOPE_DIM:], normed[:, _NOPE_DIM:])
    # A pe-first layout mutant would rotate the first 64 dims: it must not.
    pe_first = apply_indexer_rope(
        torch.cat([normed[:, _NOPE_DIM:], normed[:, :_NOPE_DIM]], dim=-1),
        positions,
    )
    assert not torch.allclose(
        out, torch.cat([pe_first[:, _NOPE_DIM:], pe_first[:, :_NOPE_DIM]], dim=-1)
    )


@pytest.mark.cpu_only
def test_k_norm_runs_before_the_split() -> None:
    """Norm spans all 128 dims before the split; per-half norms are wrong."""
    k, weight, bias, positions = _corpus(seed=23)
    out = prepare_indexer_keys(k, positions, k_norm_weight=weight, k_norm_bias=bias)

    # Mutant: split first, LayerNorm each 64-dim half separately, then rotate.
    half_mutant = torch.empty_like(k, dtype=torch.float32)
    for lo, hi in ((0, _NOPE_DIM), (_NOPE_DIM, INDEX_HEAD_DIM)):
        chunk = k[:, lo:hi]
        mean = chunk.mean(dim=-1, keepdim=True)
        var = chunk.var(dim=-1, keepdim=True, unbiased=False)
        half_mutant[:, lo:hi] = (chunk - mean) * torch.rsqrt(var + 1e-6)
    half_mutant[:, _NOPE_DIM:] = apply_indexer_rope(
        half_mutant, positions
    )[:, _NOPE_DIM:]
    assert not torch.allclose(out, half_mutant, rtol=1e-3, atol=1e-4), (
        "per-half k_norm matches the witness: norm order is not pinned"
    )


@pytest.mark.cpu_only
def test_relative_position_scores_depend_only_on_offset() -> None:
    """Interleaved RoPE gives q(p1)·k(p2) = f(p1 - p2) on the pe block."""
    torch.manual_seed(5)
    q = torch.randn(4, INDEX_HEAD_DIM, dtype=torch.float64)
    k = torch.randn(6, INDEX_HEAD_DIM, dtype=torch.float64)
    p_q = torch.tensor([10, 40, 77, 128], dtype=torch.int64)
    p_k = torch.tensor([1, 9, 40, 71, 99, 130], dtype=torch.int64)
    q_rot = apply_indexer_rope(q, p_q)
    k_rot = apply_indexer_rope(k, p_k)
    base = q_rot @ k_rot.T
    for shift in (1, 7, 53):
        shifted = apply_indexer_rope(q, p_q + shift) @ apply_indexer_rope(
            k, p_k + shift
        ).T
        torch.testing.assert_close(shifted, base, rtol=1e-9, atol=1e-9)


# ────────────────────────── mutant: split-half/NeoX ────────────────────────


@pytest.mark.cpu_only
def test_mutant_split_half_neox_dies_on_keys() -> None:
    k, weight, bias, positions = _corpus(seed=31)
    golden = prepare_indexer_keys(
        k, positions, k_norm_weight=weight, k_norm_bias=bias
    ).double()
    mutant = _mutant_split_half_keys(k, positions, weight=weight, bias=bias)
    delta = (golden - mutant).abs()
    assert float(delta.max()) > 0.1, "split-half mutant is numerically indistinguishable"
    assert float(delta.mean()) > 0.01


@pytest.mark.cpu_only
def test_mutant_split_half_neox_corrupts_topk_selection() -> None:
    """The DSA top-k selection digest must change under the mutant."""
    torch.manual_seed(20261114)
    tokens, rows, heads = 512, 24, 4
    topk = 16
    k_raw = torch.randn(tokens, heads, INDEX_HEAD_DIM, dtype=torch.float32)
    q_raw = torch.randn(rows, heads, INDEX_HEAD_DIM, dtype=torch.float32)
    weight = torch.randn(INDEX_HEAD_DIM, dtype=torch.float32) * 0.5 + 1.0
    bias = torch.randn(INDEX_HEAD_DIM, dtype=torch.float32) * 0.05
    pos_k = torch.arange(tokens, dtype=torch.int64)
    pos_q = (torch.arange(rows, dtype=torch.int64) * 17 + 50) % tokens

    q_rot = apply_indexer_rope(q_raw, pos_q)
    k_good = torch.stack(
        [
            prepare_indexer_keys(
                k_raw[:, h], pos_k, k_norm_weight=weight, k_norm_bias=bias
            )
            for h in range(heads)
        ],
        dim=1,
    )
    k_bad = torch.stack(
        [
            _mutant_split_half_keys(
                k_raw[:, h], pos_k, weight=weight, bias=bias
            ).float()
            for h in range(heads)
        ],
        dim=1,
    )

    def selection_digest(k_all: torch.Tensor) -> torch.Tensor:
        # DSA-style scores: relu(q·k) weighted, summed over indexer heads.
        scores = torch.einsum("rhd,thd->rht", q_rot, k_all).relu()
        logits = scores.sum(dim=1)  # [rows, tokens]
        return logits.topk(topk, dim=-1).indices.sort(dim=-1).values

    good = selection_digest(k_good)
    bad = selection_digest(k_bad)
    differing_rows = int((good != bad).any(dim=-1).sum())
    assert differing_rows >= rows // 2, (
        f"only {differing_rows}/{rows} rows changed selection under the "
        "split-half mutant: the digest cannot kill it"
    )


# ─────────────────────── rotated at write time (FP8) ───────────────────────


@pytest.mark.cpu_only
def test_pack_rotates_keys_at_write_time() -> None:
    k, weight, bias, positions = _corpus(seed=47, tokens=130)
    page_size = 64
    cache = pack_index_k_cache_reference(
        k,
        page_size=page_size,
        positions=positions,
        k_norm_weight=weight,
        k_norm_bias=bias,
    )
    prepared = prepare_indexer_keys(
        k, positions, k_norm_weight=weight, k_norm_bias=bias
    )
    # Byte-identical to packing pre-prepared rows: the write applies the rope.
    verbatim = pack_index_k_cache_reference(prepared, page_size=page_size)
    assert torch.equal(cache.view(torch.uint8), verbatim.view(torch.uint8))
    # The paged cache stores post-rope rows (up to FP8 quantization).
    unpacked = unpack_index_k_cache_reference(
        cache, num_tokens=k.shape[0], page_size=page_size
    )
    torch.testing.assert_close(
        unpacked, prepared, rtol=0.03, atol=0.02 * float(prepared.abs().max())
    )
    # And the stored rows are NOT the raw keys: the rotation really happened.
    raw_packed = pack_index_k_cache_reference(k, page_size=page_size)
    assert not torch.equal(cache.view(torch.uint8), raw_packed.view(torch.uint8))


@pytest.mark.cpu_only
def test_pack_without_positions_stays_verbatim() -> None:
    k, _, _, _ = _corpus(seed=53, tokens=70)
    cache = pack_index_k_cache_reference(k)
    unpacked = unpack_index_k_cache_reference(cache, num_tokens=k.shape[0])
    torch.testing.assert_close(
        unpacked, k, rtol=0.03, atol=0.02 * float(k.abs().max())
    )
    with pytest.raises(ValueError, match="require positions"):
        pack_index_k_cache_reference(k, k_norm_weight=torch.ones(INDEX_HEAD_DIM))


# ─────────────── GPU: rotated cache through the paged index cache ──────────


def _make_real_page_table(*, page_starts, seqlens, width_blocks, device):
    real_page_table = torch.full(
        (len(seqlens), width_blocks), -1, dtype=torch.int32, device=device
    )
    for row_idx, (page_start, seq_len) in enumerate(
        zip(page_starts, seqlens, strict=True)
    ):
        block_count = (int(seq_len) + 63) // 64
        if block_count:
            real_page_table[row_idx, :block_count] = torch.arange(
                page_start,
                page_start + block_count,
                dtype=torch.int32,
                device=device,
            )
    return real_page_table


def test_paged_index_cache_consumes_write_time_rope() -> None:
    """The rotated-at-write cache feeds the real paged decode kernel."""
    from ..conftest import require_b12x
    from b12x.attention.dsa_indexer._impl import (
        build_paged_mqa_schedule_metadata,
        paged_decode_logits,
    )
    from b12x.attention.dsa_indexer.scratch import (
        B12XIndexerPagedScratchCaps,
        INDEXER_PAGED_ROUTE_TILED,
        plan_indexer_paged_scratch,
    )

    require_b12x()
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu").manual_seed(72_104)

    q_rows, num_heads = 4, 8
    page_starts = [2, 8, 12, 16]
    seqlens = [65, 96, 128, 191]
    num_tokens = (max(page_starts) + 3) * 64
    weight = torch.randn(INDEX_HEAD_DIM, generator=gen) * 0.5 + 1.0
    bias = torch.randn(INDEX_HEAD_DIM, generator=gen) * 0.05
    k_raw = torch.randn((num_tokens, INDEX_HEAD_DIM), generator=gen) / 3
    positions = torch.arange(num_tokens, dtype=torch.int64)

    # FP8 key path: rotate at write into the paged index cache.
    index_k_cache = pack_index_k_cache_reference(
        k_raw,
        positions=positions,
        k_norm_weight=weight,
        k_norm_bias=bias,
    ).to(device)

    real_page_table = _make_real_page_table(
        page_starts=page_starts,
        seqlens=seqlens,
        width_blocks=3,
        device=device,
    )
    seqlens_t = torch.tensor(seqlens, dtype=torch.int32, device=device)
    q_fp8 = (
        torch.randn((q_rows, num_heads, INDEX_HEAD_DIM), generator=gen) / 2
    ).to(device).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen).to(device)

    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device=device,
            num_q_heads=num_heads,
            max_q_rows=q_rows,
            max_page_table_width=3,
            topk=1,
            reserve_paged_logits=False,
            route=INDEXER_PAGED_ROUTE_TILED,
        )
    )
    (spec,) = plan.scratch_specs()
    binding = plan.bind(
        scratch=torch.empty(spec.shape, dtype=spec.dtype, device=spec.device),
        real_page_table=real_page_table,
        cache_seqlens_int32=seqlens_t,
        active_width=None,
        schedule_metadata=build_paged_mqa_schedule_metadata(seqlens_t, 64, 8),
        expected_num_q_heads=num_heads,
    )
    actual = paged_decode_logits(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        binding=binding,
    )
    expected = paged_decode_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=seqlens_t,
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    # Mutant visibility on the GPU path: a NeoX-written cache must score
    # differently through the same kernel (kills the write-form silently
    # succeeding with wrong geometry).
    mutant_rows = torch.stack(
        [
            _mutant_split_half_keys(k_raw, positions, weight=weight, bias=bias)
        ]
    ).reshape(num_tokens, INDEX_HEAD_DIM)
    mutant_cache = pack_index_k_cache_reference(mutant_rows).to(device)
    mutant_logits = paged_decode_logits(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=mutant_cache,
        binding=binding,
    )
    torch.cuda.synchronize(device)
    assert not torch.allclose(
        mutant_logits.float(), expected.float(), rtol=1e-3, atol=1e-3
    ), "split-half cache scores match the interleaved witness through the kernel"
