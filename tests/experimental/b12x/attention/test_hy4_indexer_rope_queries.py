"""Hy4 indexer QUERY RoPE (defect 15a).

The indexer rotates its queries too: 128-dim queries (``index_head_dim``)
with the rotary block in the LAST ``qk_rope_head_dim`` = 64 dims,
``is_neox_style=False`` adjacent-pair interleaved rotation,
``rope_theta = 1e7``, and NO ``k_norm`` — the witness normalizes keys only
(``k = self.k_norm(k)``; ``q`` goes straight from ``wq_b`` to the split).
Keys are rotated once at cache-write time (``prepare_indexer_keys``); the
queries are rotated per call with the SAME machinery (``apply_indexer_rope``),
so query and key rotations cannot drift apart.

Witness: vLLM ``ref/vllm_hy4_attention.py`` ``Indexer.prepare_inputs``.

Mutants that must die here:
  (a) the split-half (NeoX) query rotation — pairs ``(i, i + 32)`` instead of
      the checkpoint's interleaved ``(2i, 2i + 1)`` — killed at the
      key-selection-digest level AND through the real paged decode-logits
      kernel;
  (b) ``k_norm`` on the query: the witness never normalizes q, and a normed
      query moves off the trained score geometry.
"""

from __future__ import annotations

import math

import pytest
import torch

from b12x.attention.dsa_indexer.reference import (
    pack_index_k_cache_reference,
    paged_decode_logits_reference,
)
from b12x.attention.dsa_indexer.rope import (
    INDEX_HEAD_DIM,
    INDEX_ROPE_DIM,
    INDEX_ROPE_THETA,
    apply_indexer_rope,
    indexer_k_norm,
    indexer_rope_inv_freq,
    prepare_indexer_keys,
    prepare_indexer_queries,
)

_NOPE_DIM = INDEX_HEAD_DIM - INDEX_ROPE_DIM  # 64

# ──────────────────────── independent witness math ─────────────────────────


def _witness_prepare_indexer_queries(
    q: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Scalar transcription of vLLM's query path (fp64): split, rotate, NO norm."""
    tokens = int(q.shape[0])
    out = torch.empty_like(q, dtype=torch.float64)
    for t in range(tokens):
        row = q[t].to(torch.float64)
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


def _mutant_split_half_queries(
    q: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """The NeoX/split-half query rotation: pairs ``(i, i + 32)`` in the pe block.

    Same inv_freq as the interleaved form, so only the dim pairing is wrong.
    """
    qf = q.to(torch.float64)
    half = INDEX_ROPE_DIM // 2
    out = qf.clone()
    pe = qf[..., _NOPE_DIM:]
    inv = indexer_rope_inv_freq(
        INDEX_ROPE_DIM, INDEX_ROPE_THETA, dtype=torch.float64
    )
    ang = positions.to(torch.float64).unsqueeze(1) * inv.unsqueeze(0)  # [t, 32]
    cos, sin = ang.cos(), ang.sin()
    lead = (positions.shape[0],) + (1,) * (qf.ndim - 2) + (half,)
    cos, sin = cos.reshape(lead), sin.reshape(lead)
    x1, x2 = pe[..., :half], pe[..., half:]
    out[..., _NOPE_DIM : _NOPE_DIM + half] = x1 * cos - x2 * sin
    out[..., _NOPE_DIM + half :] = x1 * sin + x2 * cos
    return out


def _mutant_normed_queries(
    q: torch.Tensor,
    positions: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """The norm-on-query mutant: keys' ``k_norm`` wrongly applied to q."""
    return apply_indexer_rope(
        indexer_k_norm(q, weight=weight, bias=bias), positions
    )


def _corpus(seed: int, tokens: int = 97):
    torch.manual_seed(seed)
    q = torch.randn(tokens, INDEX_HEAD_DIM, dtype=torch.float32)
    weight = torch.randn(INDEX_HEAD_DIM, dtype=torch.float32) * 0.5 + 1.0
    bias = torch.randn(INDEX_HEAD_DIM, dtype=torch.float32) * 0.1
    positions = torch.arange(tokens, dtype=torch.int64) * 3 + 7
    return q, weight, bias, positions


# ─────────────────────────── witness equivalence ───────────────────────────


@pytest.mark.cpu_only
def test_query_rope_matches_vllm_witness() -> None:
    q, _, _, positions = _corpus(seed=20261115)
    golden = _witness_prepare_indexer_queries(q, positions)
    actual = prepare_indexer_queries(q, positions)
    torch.testing.assert_close(actual.double(), golden, rtol=1e-5, atol=1e-6)
    # FP64 input tightens to machine agreement with the witness loops.
    actual64 = prepare_indexer_queries(q.double(), positions)
    torch.testing.assert_close(actual64, golden, rtol=1e-12, atol=1e-12)
    # The head-batched witness shape rotates every head identically.
    q_heads = q.reshape(-1, 1, INDEX_HEAD_DIM).repeat(1, 3, 1)
    batched = prepare_indexer_queries(q_heads, positions)
    for h in range(3):
        torch.testing.assert_close(
            batched[:, h].double(), golden, rtol=1e-5, atol=1e-6
        )


@pytest.mark.cpu_only
def test_query_pe_block_lives_in_the_last_64_dims() -> None:
    """Rotation touches ONLY the trailing qk_rope_head_dim block (PTM layout)."""
    q, _, _, positions = _corpus(seed=11)
    out = prepare_indexer_queries(q, positions)
    assert torch.equal(out[:, :_NOPE_DIM], q[:, :_NOPE_DIM])
    assert not torch.allclose(out[:, _NOPE_DIM:], q[:, _NOPE_DIM:])


@pytest.mark.cpu_only
def test_queries_get_no_k_norm() -> None:
    """The query is rotated RAW: no mean/variance step anywhere in the path."""
    q, weight, bias, positions = _corpus(seed=23)
    out = prepare_indexer_queries(q, positions)
    # The nope block must pass through bit-identically (a norm would move it).
    assert torch.equal(out[:, :_NOPE_DIM], q[:, :_NOPE_DIM].double())
    # The norm-on-query mutant differs everywhere and is killed here.
    mutant = _mutant_normed_queries(q, positions, weight, bias)
    delta = (out - mutant).abs()
    assert float(delta.max()) > 0.1, "norm-on-query mutant is indistinguishable"
    # And a norm whose weight/bias are identity still differs (the variance step).
    identity_mutant = apply_indexer_rope(indexer_k_norm(q), positions)
    assert not torch.allclose(out, identity_mutant, rtol=1e-3, atol=1e-3), (
        "identity k_norm on q matches the witness: the norm step is not pinned"
    )


@pytest.mark.cpu_only
def test_query_and_key_rotations_share_one_geometry() -> None:
    """q(p1)·k(p2) shifts invariantly: one rotation machinery on both sides."""
    torch.manual_seed(5)
    q = torch.randn(4, INDEX_HEAD_DIM, dtype=torch.float64)
    k = torch.randn(6, INDEX_HEAD_DIM, dtype=torch.float64)
    weight = torch.randn(INDEX_HEAD_DIM, dtype=torch.float64) * 0.5 + 1.0
    bias = torch.randn(INDEX_HEAD_DIM, dtype=torch.float64) * 0.05
    p_q = torch.tensor([10, 40, 77, 128], dtype=torch.int64)
    p_k = torch.tensor([1, 9, 40, 71, 99, 130], dtype=torch.int64)
    q_rot = prepare_indexer_queries(q, p_q)
    k_rot = prepare_indexer_keys(k, p_k, k_norm_weight=weight, k_norm_bias=bias)
    base = q_rot @ k_rot.T
    for shift in (1, 7, 53):
        shifted = prepare_indexer_queries(q, p_q + shift) @ prepare_indexer_keys(
            k, p_k + shift, k_norm_weight=weight, k_norm_bias=bias
        ).T
        torch.testing.assert_close(shifted, base, rtol=1e-9, atol=1e-9)


# ──────────────── mutants at the key-selection-digest level ────────────────


def _selection_digest(q_rot: torch.Tensor, k_all: torch.Tensor, topk: int):
    # DSA-style scores: relu(q·k) weighted, summed over indexer heads.
    scores = torch.einsum("rhd,thd->rht", q_rot, k_all).relu()
    logits = scores.sum(dim=1)  # [rows, tokens]
    return logits.topk(topk, dim=-1).indices.sort(dim=-1).values


def _digest_corpus(seed: int = 20261115):
    torch.manual_seed(seed)
    tokens, rows, heads, topk = 512, 24, 4, 16
    k_raw = torch.randn(tokens, heads, INDEX_HEAD_DIM, dtype=torch.float32)
    q_raw = torch.randn(rows, heads, INDEX_HEAD_DIM, dtype=torch.float32)
    weight = torch.randn(INDEX_HEAD_DIM, dtype=torch.float32) * 0.5 + 1.0
    bias = torch.randn(INDEX_HEAD_DIM, dtype=torch.float32) * 0.05
    pos_k = torch.arange(tokens, dtype=torch.int64)
    pos_q = (torch.arange(rows, dtype=torch.int64) * 17 + 50) % tokens
    k_good = torch.stack(
        [
            prepare_indexer_keys(
                k_raw[:, h], pos_k, k_norm_weight=weight, k_norm_bias=bias
            )
            for h in range(heads)
        ],
        dim=1,
    )
    return q_raw, k_good, weight, bias, pos_q, topk


@pytest.mark.cpu_only
def test_mutant_split_half_neox_dies_on_queries() -> None:
    q, _, _, positions = _corpus(seed=31)
    golden = prepare_indexer_queries(q, positions).double()
    mutant = _mutant_split_half_queries(q, positions)
    delta = (golden - mutant).abs()
    assert float(delta.max()) > 0.1, "split-half mutant is numerically indistinguishable"
    assert float(delta.mean()) > 0.01


@pytest.mark.cpu_only
def test_mutant_split_half_neox_corrupts_query_selection_digest() -> None:
    """The DSA top-k selection digest must change under the query-side mutant."""
    q_raw, k_good, _, _, pos_q, topk = _digest_corpus()
    rows = q_raw.shape[0]
    good = _selection_digest(
        prepare_indexer_queries(q_raw, pos_q), k_good, topk
    )
    bad = _selection_digest(
        _mutant_split_half_queries(q_raw, pos_q).float(), k_good, topk
    )
    differing_rows = int((good != bad).any(dim=-1).sum())
    assert differing_rows >= rows // 2, (
        f"only {differing_rows}/{rows} rows changed selection under the "
        "query split-half mutant: the digest cannot kill it"
    )


@pytest.mark.cpu_only
def test_mutant_k_norm_on_query_corrupts_query_selection_digest() -> None:
    """The norm-on-query mutant must also change the selection digest."""
    q_raw, k_good, weight, bias, pos_q, topk = _digest_corpus(seed=20261116)
    rows = q_raw.shape[0]
    good = _selection_digest(
        prepare_indexer_queries(q_raw, pos_q), k_good, topk
    )
    bad = _selection_digest(
        _mutant_normed_queries(q_raw, pos_q, weight, bias).float(), k_good, topk
    )
    differing_rows = int((good != bad).any(dim=-1).sum())
    assert differing_rows >= rows // 2, (
        f"only {differing_rows}/{rows} rows changed selection under the "
        "norm-on-query mutant: the digest cannot kill it"
    )


# ─────────────────── GPU: the real paged decode-logits kernel ──────────────


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


def test_kernel_selection_digest_kills_query_neox_mutant() -> None:
    """Write-time-rotated keys, per-call rotated queries, real kernel.

    The good queries must match the kernel's own reference, and the NeoX
    query rotation must change the kernel's selection digest — the mutant
    dies at kernel level, not just in torch math.
    """
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
    gen = torch.Generator(device="cpu").manual_seed(72_105)

    q_rows, num_heads = 6, 8
    page_starts = [2, 8, 12, 16, 20, 24]
    seqlens = [65, 96, 128, 191, 200, 231]
    num_tokens = (max(page_starts) + 5) * 64
    weight = torch.randn(INDEX_HEAD_DIM, generator=gen) * 0.5 + 1.0
    bias = torch.randn(INDEX_HEAD_DIM, generator=gen) * 0.05
    k_raw = torch.randn((num_tokens, INDEX_HEAD_DIM), generator=gen) / 3
    key_positions = torch.arange(num_tokens, dtype=torch.int64)
    q_raw = torch.randn((q_rows, num_heads, INDEX_HEAD_DIM), generator=gen) / 2
    query_positions = torch.arange(q_rows, dtype=torch.int64) * 29 + 130
    weights = torch.randn((q_rows, num_heads), generator=gen)

    # FP8 key path: rotate at write into the paged index cache.
    index_k_cache = pack_index_k_cache_reference(
        k_raw,
        positions=key_positions,
        k_norm_weight=weight,
        k_norm_bias=bias,
    ).to(device)
    real_page_table = _make_real_page_table(
        page_starts=page_starts,
        seqlens=seqlens,
        width_blocks=4,
        device=device,
    )
    seqlens_t = torch.tensor(seqlens, dtype=torch.int32, device=device)
    weights = weights.to(device)

    def run(q_prepared: torch.Tensor) -> torch.Tensor:
        q_fp8 = q_prepared.to(device).to(torch.float8_e4m3fn)
        plan = plan_indexer_paged_scratch(
            B12XIndexerPagedScratchCaps(
                device=device,
                num_q_heads=num_heads,
                max_q_rows=q_rows,
                max_page_table_width=4,
                topk=1,
                reserve_paged_logits=False,
                route=INDEXER_PAGED_ROUTE_TILED,
            )
        )
        (spec,) = plan.scratch_specs()
        binding = plan.bind(
            scratch=torch.empty(
                spec.shape, dtype=spec.dtype, device=spec.device
            ),
            real_page_table=real_page_table,
            cache_seqlens_int32=seqlens_t,
            active_width=None,
            schedule_metadata=build_paged_mqa_schedule_metadata(seqlens_t, 64, 8),
            expected_num_q_heads=num_heads,
        )
        return paged_decode_logits(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            binding=binding,
        )

    q_good = prepare_indexer_queries(q_raw, query_positions).float()
    good_logits = run(q_good)
    q_fp8 = q_good.to(device).to(torch.float8_e4m3fn)
    expected = paged_decode_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=seqlens_t,
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(good_logits, expected, atol=1e-4, rtol=1e-4)

    def digest(logits: torch.Tensor) -> torch.Tensor:
        topk = 8
        return logits.topk(topk, dim=-1).indices.sort(dim=-1).values

    good_digest = digest(good_logits)
    q_bad = _mutant_split_half_queries(q_raw, query_positions).float()
    bad_logits = run(q_bad)
    torch.cuda.synchronize(device)
    bad_digest = digest(bad_logits)
    differing_rows = int((good_digest != bad_digest).any(dim=-1).sum())
    assert differing_rows >= q_rows // 2, (
        f"only {differing_rows}/{q_rows} kernel selection rows changed under "
        "the query split-half mutant: the kernel-level digest cannot kill it"
    )
    assert not torch.allclose(
        bad_logits.float(), good_logits.float(), rtol=1e-3, atol=1e-3
    ), "split-half query logits match the interleaved witness through the kernel"
