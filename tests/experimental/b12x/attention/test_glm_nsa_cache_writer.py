# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM_NSA 656-byte fused-record cache writer (``concat_and_cache_glm_next_mla``).

The Hy4 engine's cache contract is the fused 656 B/token GLM_NSA record the
sparse-MLA readers consume (``traits.kv_gmem_stride == 656``):

    [  0, 512)  512 x E4M3 latent nope bytes (four 128-dim groups)
    [512, 528)  4 x inline fp32 group scales (ARBITRARY: amax / 448)
    [528, 656)  64 x BF16 RoPE (raw bytes, no quantization)

The writer consumes the decoded latent row ``kv_c`` (num_tokens, 576) and must
produce exactly the layout ``reference.pack_mla_kv_cache_reference`` defines
(the authoritative encoder). These tests pin the record layout to that
reference and prove the writer-produced records drive the real sparse-MLA
attention kernels (decode + extend) against the numeric oracle.
"""

from __future__ import annotations

import pytest
import torch

from b12x.attention import sparse_mla
from b12x.attention._shared.mla.reference import (
    _sparse_attention_reference,
    pack_mla_kv_cache_reference,
    unpack_mla_kv_cache_reference,
)
from b12x.attention._shared.mla.traits import ModelType
from b12x.preparation import PreparedCall, PreparationSession

from ..conftest import require_b12x as require_sm120


_GLM_NSA_RECORD_BYTES = 656
_GLM_NSA_LATENT_DIM = 576
_GLM_NOPE_DIM = 512
_GLM_ROPE_DIM = 64
_PAGE_SIZE = 64
_SM_SCALE = 576**-0.5


def _reference_pack_rows(latent: torch.Tensor) -> torch.Tensor:
    """``pack_mla_kv_cache_reference`` over (num_records, 576) rows -> (N, 656)."""
    packed = pack_mla_kv_cache_reference(latent[:, :_GLM_NOPE_DIM], latent[:, _GLM_NOPE_DIM:])
    return packed.reshape(packed.shape[0], _GLM_NSA_RECORD_BYTES)


def _writer_session_write(latent, cache, slots, session):
    writer = sparse_mla.plan_cache_writer(latent, cache, slots)
    session.prepare(
        (
            writer.request(
                name="glm-nsa-fused-record-writer",
                prepare_call=lambda state: PreparedCall(
                    run=lambda: state.run(latent, cache, slots),
                    owners=(latent, cache, slots),
                ),
            ),
        )
    )
    sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots, plan=writer)
    return writer


@pytest.mark.parametrize("page_size", (1, 64), ids=("page1", "page64"))
@torch.inference_mode()
def test_glm_nsa_writer_record_layout_matches_reference(page_size: int) -> None:
    """The fused record is byte-faithful to ``pack_mla_kv_cache_reference``.

    RoPE bytes must be exact (raw copy); the nope lane is the same ARBITRARY
    FP8 recipe the reference encodes (scale = group amax / 448) with only the
    writer's rcp.approx-scale rounding allowed to differ, so dequantized
    values must agree to well under one E4M3 LSB and scale bytes to ~1 ulp.
    """
    device = require_sm120()
    generator = torch.Generator(device=device).manual_seed(20261115)
    num_pages, num_records = 3, 3 * page_size
    cache = torch.zeros(
        (num_pages, page_size, _GLM_NSA_RECORD_BYTES), dtype=torch.uint8, device=device
    )
    latent = (
        torch.randn(
            (num_records, _GLM_NSA_LATENT_DIM), generator=generator, device=device
        )
        / 8
    ).to(torch.bfloat16)
    slots = torch.arange(num_records, dtype=torch.int64, device=device)
    # Padded (negative) slots must leave their records untouched.
    slots[1] = -1
    expected = _reference_pack_rows(latent)

    with PreparationSession(device=device, compile_workers=0) as session:
        _writer_session_write(latent, cache, slots, session)
    torch.cuda.synchronize(device)

    got = cache.reshape(num_records, _GLM_NSA_RECORD_BYTES)
    untouched = got[1]
    assert not bool(untouched.any()), "a skipped (-1) slot record was written"

    # RoPE tail: raw BF16 bytes, exact (the skipped row is excluded — its
    # record is deliberately left untouched).
    live = torch.arange(num_records, device=device) != 1
    assert torch.equal(got[live, 528:], expected[live, 528:]), "RoPE tail is not a raw copy"
    # Scales: the same ARBITRARY recipe (amax / 448), rounding aside.
    got_scales = got[:, 512:528].contiguous().view(torch.float32)
    exp_scales = expected[:, 512:528].contiguous().view(torch.float32)
    torch.testing.assert_close(
        got_scales[live], exp_scales[live], rtol=2e-6, atol=1e-9
    )
    # Nope lane: identical quantization recipe; dequantized records agree to
    # far under one E4M3 LSB of the group scale (only scale rounding and the
    # writer's rcp.approx may differ from the reference's division).
    got_deq = unpack_mla_kv_cache_reference(got[live].unsqueeze(1)).squeeze(1)
    exp_deq = unpack_mla_kv_cache_reference(expected[live].unsqueeze(1)).squeeze(1)
    diff = (got_deq.float() - exp_deq.float()).abs()
    group_amax = exp_deq[:, :_GLM_NOPE_DIM].float().abs().amax(dim=1, keepdim=True)
    assert bool((diff[:, :_GLM_NOPE_DIM] <= 0.02 * group_amax + 1e-5).all()), (
        f"nope dequant drift {diff[:, :_GLM_NOPE_DIM].max().item():.6f} exceeds "
        "2% of the group scale"
    )
    exact_bytes = (got[live, :512] == expected[live, :512]).float().mean().item()
    assert exact_bytes > 0.98, (
        f"only {exact_bytes:.4f} of nope bytes matched the reference encoder exactly"
    )


@pytest.mark.parametrize("mode", ("decode", "extend"))
@torch.inference_mode()
def test_glm_nsa_writer_feeds_attention_kernel(mode: str) -> None:
    """Records written by the fused writer drive the real GLM_NSA kernels.

    End-to-end contract: writer -> 656 B record cache -> sparse-MLA attention
    (the exact path the Hy4 recipe's cache write + attend takes) numerically
    matches the oracle over the reference-packed records.
    """
    device = require_sm120()
    generator = torch.Generator(device=device).manual_seed(20261116)
    rows, heads, num_pages = 2, 8, 4
    num_records = num_pages * _PAGE_SIZE
    # Extend (prefill MG) supports GLM_NSA topk in {512, 1024, 2048}.
    width = 129 if mode == "decode" else 512
    latent = (
        torch.randn(
            (num_records, _GLM_NSA_LATENT_DIM), generator=generator, device=device
        )
        / 4
    ).to(torch.bfloat16)
    cache = torch.zeros(
        (num_pages, _PAGE_SIZE, _GLM_NSA_RECORD_BYTES), dtype=torch.uint8, device=device
    )
    slots = torch.arange(num_records, dtype=torch.int64, device=device)
    q = (
        torch.randn((rows, heads, _GLM_NSA_LATENT_DIM), generator=generator, device=device)
        / 4
    ).to(torch.bfloat16)
    physical_ids = (
        torch.randperm(num_records - _PAGE_SIZE, generator=generator, device=device)
        + _PAGE_SIZE
    ).to(torch.int32)
    selected = torch.full((rows, width), -1, dtype=torch.int32, device=device)
    selected[0, : min(width, 129)].copy_(physical_ids[: min(width, 129)])
    selected[1, :65].copy_(physical_ids[127:192])
    active = torch.tensor(
        [min(width, 129), 65], dtype=torch.int32, device=device
    )
    cache_seqlens = torch.full((rows,), num_records, dtype=torch.int32, device=device)

    plan = sparse_mla.plan(
        sparse_mla.Caps(
            device=device,
            num_q_heads=heads,
            max_q_rows=rows,
            max_batch=rows,
            max_width=width,
            softmax_scale=_SM_SCALE,
            max_kv_rows=int(active.max()),
            kv_dtype=torch.uint8,
            head_dim=_GLM_NSA_LATENT_DIM,
            v_head_dim=_GLM_NOPE_DIM,
            page_size=_PAGE_SIZE,
            model_type=ModelType.GLM_NSA,
            cache_record_bytes=_GLM_NSA_RECORD_BYTES,
            mode=mode,
            return_lse=True,
            lse_scale="base2",
        )
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)

    def prepare_attention(state):
        runtime = state.bind(
            scratch=scratch,
            q=q,
            selected_indices=selected,
            cache_seqlens_int32=cache_seqlens,
            nsa_cache_seqlens_int32=active,
            kv_cache=cache,
        )
        return PreparedCall(
            run=lambda: state.run(runtime, kv_cache=cache), owners=(runtime,)
        )

    with PreparationSession(device=device, compile_workers=0) as session:
        _writer_session_write(latent, cache, slots, session)
        session.prepare(
            (
                plan.request(
                    name="glm-nsa-attention-over-writer-records",
                    prepare_call=prepare_attention,
                ),
            )
        )
        binding = sparse_mla.bind(
            plan,
            scratch=scratch,
            q=q,
            kv_cache=cache,
            selected_indices=selected,
            cache_lengths=cache_seqlens,
            selected_lengths=active,
        )
        actual, actual_lse = sparse_mla.run(binding)
        torch.cuda.synchronize(device)

    # Oracle over the reference-packed records of the SAME latent (V is the
    # nope-only first 512 dims — V_HAS_ROPE=false for GLM_NSA).
    expected_cache = _reference_pack_rows(latent)
    dequantized = unpack_mla_kv_cache_reference(
        expected_cache.reshape(num_records, 1, _GLM_NSA_RECORD_BYTES)
    )[:, 0]
    expected, expected_lse = _sparse_attention_reference(
        q_all=q,
        k_all=dequantized,
        v_all=dequantized[:, :_GLM_NOPE_DIM],
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=_SM_SCALE,
        return_lse=True,
    )
    actual_f, expected_f = actual.float(), expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.reshape(-1), expected_f.reshape(-1), dim=0
    ).item()
    max_abs = (actual_f - expected_f).abs().max().item()
    assert cosine > 0.995, f"cosine similarity {cosine:.8f} did not exceed 0.995"
    assert max_abs < 0.03, f"max absolute error {max_abs:.8f} exceeded 0.03"
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0.0, atol=0.05)
