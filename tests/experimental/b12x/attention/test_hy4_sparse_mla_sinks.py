"""Hy4 gated-attention sinks on the GLM_NSA record at top-k 2048.

The Hy4 sink is a GPT-OSS-style learnable per-head natural-log logit. b12x
folds it into the sparse-MLA softmax exactly like the decode epilogue
(``_shared/mla/decode_math.py``)::

    out  *= sigmoid(lse_e - sink)
    lse_e = logaddexp(lse_e, sink)          (empty rows -> sink)

These tests pin that fold on the GLM_NSA record (576-dim QK, 656-byte
ARBITRARY_FP32 records, top-k 2048) for decode, extend/prefill and verify,
through the public ``sparse_mla.plan -> bind -> run`` lifecycle.

Mutants that must die here:
  (a) sink disabled: a non-zero sink must change the output and the LSE, and
      the change must equal the decode-path fold exactly.
"""

from __future__ import annotations

import math

import pytest
import torch

from b12x.preparation import PreparedCall, PreparationSession

from b12x.attention import sparse_mla
from b12x.attention._shared.mla.reference import (
    pack_mla_kv_cache_reference,
    sparse_mla_reference,
)

from ..conftest import require_b12x

LN2 = math.log(2.0)

NOPE_DIM = 512
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM  # GLM_NSA qk_head_dim = 576
V_HEAD_DIM = NOPE_DIM
TOPK = 2048
CACHE_TOKENS = 2304  # > topk so selection is genuinely sparse
PAGE_SIZE = 64


def _make_case(*, rows: int, heads: int, width: int = TOPK, seed: int = 20261114):
    torch.manual_seed(seed)
    device = "cuda"
    k_nope = (
        torch.randn(CACHE_TOKENS, NOPE_DIM, device=device, dtype=torch.bfloat16) / 4
    )
    k_rope = (
        torch.randn(CACHE_TOKENS, ROPE_DIM, device=device, dtype=torch.bfloat16) / 4
    )
    kv_cache = pack_mla_kv_cache_reference(k_nope, k_rope)
    q_all = torch.randn(rows, heads, HEAD_DIM, device=device, dtype=torch.bfloat16) / 4
    selected = torch.stack(
        [
            torch.randperm(CACHE_TOKENS, device=device)[:width].sort().values
            for _ in range(rows)
        ]
    ).to(torch.int32)
    cache_seqlens = torch.full(
        (rows,), CACHE_TOKENS, dtype=torch.int32, device=device
    )
    active = torch.full((rows,), width, dtype=torch.int32, device=device)
    return q_all, kv_cache, selected, cache_seqlens, active


def _sink_logits(heads: int, seed: int) -> torch.Tensor:
    """Learnable per-head sink logits bracketing the row LSE.

    At top-k 2048 the row LSE sits near log(2048) ~= 7.6 natural; sinks far
    below it are numerically invisible (the fold factor rounds to 1) and the
    fold tests become unfailable. Hy4's ``learnable_sink`` logits are trained
    and unbounded, so test them at magnitudes that actually gate the
    normalizer: every head's factor must land in (0.05, 0.95).
    """
    torch.manual_seed(seed)
    return torch.linspace(6.0, 10.0, heads, device="cuda") + torch.randn(
        heads, device="cuda"
    ) * 0.3


def _fold_sink(
    out: torch.Tensor, lse_base2: torch.Tensor, sink: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The decode-epilogue fold, in torch (decode_math.py:3420-3440, 3589-3606)."""
    sink_e = sink.to(torch.float64)
    lse_e = lse_base2.to(torch.float64) * LN2
    factor = torch.sigmoid(lse_e - sink_e.unsqueeze(0))
    folded_out = out.to(torch.float32) * factor.unsqueeze(-1).float()
    folded_lse2 = torch.logaddexp(lse_e, sink_e.unsqueeze(0)) / LN2
    return folded_out, folded_lse2


def _run_public(
    q_all,
    kv_cache,
    selected,
    cache_seqlens,
    active,
    *,
    mode: str,
    sink: torch.Tensor | None,
    width: int = TOPK,
    lse_scale: str = "base2",
    want_binding: bool = False,
):
    rows, heads, _ = q_all.shape
    sm_scale = HEAD_DIM**-0.5
    caps_kwargs = dict(
        device=q_all.device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_batch=rows,
        max_width=width,
        softmax_scale=sm_scale,
        kv_dtype=torch.uint8,
        head_dim=HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        page_size=PAGE_SIZE,
        mode=mode,
        return_lse=True,
        lse_scale=lse_scale,
    )
    declaration = sparse_mla.plan(
        sparse_mla.Caps(
            **caps_kwargs, has_attention_sink=sink is not None
        )
    )
    owned = {}

    def prepare_call(state):
        spec = state.scratch_specs()[0]
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=q_all.device)
        runtime = state.bind(
            scratch=scratch,
            q=q_all,
            kv_cache=kv_cache,
            selected_indices=selected,
            cache_seqlens_int32=cache_seqlens,
            nsa_cache_seqlens_int32=active,
        )
        owned["scratch"] = scratch
        state.prime(runtime, kv_cache=kv_cache, attention_sink=sink)
        return PreparedCall(
            run=lambda: state.run(runtime, kv_cache=kv_cache, attention_sink=sink)
        )

    session = PreparationSession(device=q_all.device, compile_workers=0)
    with session:
        result = session.prepare(
            (
                declaration.request(
                    name="hy4-sink", prepare_call=prepare_call,
                ),
            )
        )
        plan = result.plans["hy4-sink"]
        binding = sparse_mla.bind(
            plan,
            scratch=owned["scratch"],
            q=q_all,
            kv_cache=kv_cache,
            selected_indices=selected,
            cache_lengths=cache_seqlens,
            selected_lengths=active,
            attention_sink=sink,
        )
        out, lse = sparse_mla.run(binding)
    if want_binding:
        return out, lse, binding
    return out, lse


def _reference(q_all, kv_cache, selected, active, sink):
    ref_out, ref_lse2 = sparse_mla_reference(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=HEAD_DIM**-0.5,
        v_head_dim=V_HEAD_DIM,
        return_lse=True,
    )
    if sink is not None:
        ref_out, ref_lse2 = _fold_sink(ref_out, ref_lse2, sink)
    return ref_out, ref_lse2


def _assert_close(actual, expected, *, what: str):
    torch.cuda.synchronize()
    assert bool(torch.isfinite(actual).all().item()), f"{what}: non-finite output"
    actual_f = actual.float()
    expected_f = expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.reshape(-1), expected_f.reshape(-1), dim=0
    )
    assert float(cosine.item()) > 0.999, f"{what}: cosine {float(cosine.item()):.6f}"
    torch.testing.assert_close(
        actual_f, expected_f, rtol=5e-2, atol=5e-2, msg=lambda m: f"{what}: {m}"
    )


def _assert_lse_close(actual_lse2, expected_lse2, *, what: str):
    # Base-2 LSE; only compare rows with live selection.
    live = expected_lse2.isfinite()
    assert bool(live.any().item()), f"{what}: no live rows"
    torch.testing.assert_close(
        actual_lse2[live].float(),
        expected_lse2[live].float(),
        rtol=0.0,
        atol=0.05,
        msg=lambda m: f"{what}: {m}",
    )


def _assert_sink_visible(off_lse2: torch.Tensor, sink: torch.Tensor, *, what: str):
    """Precondition: the sink must gate the normalizer, not round to 1."""
    live = off_lse2.isfinite()
    factor = torch.sigmoid(
        off_lse2.to(torch.float64) * LN2 - sink.double().unsqueeze(0)
    )[live]
    assert bool(((factor > 0.05) & (factor < 0.95)).all().item()), (
        f"{what}: sink is numerically invisible (factors {factor.float()})"
    )


@pytest.mark.parametrize("mode", ["extend", "verify"])
def test_glm_nsa_topk2048_sink_folds_like_decode(mode: str) -> None:
    require_b12x()
    rows, heads = 2, 8
    q, kv, sel, lens, active = _make_case(rows=rows, heads=heads)
    sink = _sink_logits(heads, seed=7)

    out, lse = _run_public(q, kv, sel, lens, active, mode=mode, sink=sink)
    ref_out, ref_lse = _reference(q, kv, sel, active, sink)
    _, off_lse = _run_public(q, kv, sel, lens, active, mode=mode, sink=None)
    _assert_sink_visible(off_lse, sink, what=mode)
    _assert_close(out, ref_out, what=f"{mode}-with-sink")
    _assert_lse_close(lse, ref_lse, what=f"{mode}-with-sink-lse")


def test_decode_topk2048_sink_folds_like_decode() -> None:
    require_b12x()
    rows, heads = 2, 8
    q, kv, sel, lens, active = _make_case(rows=rows, heads=heads)
    sink = _sink_logits(heads, seed=7)

    out, lse = _run_public(q, kv, sel, lens, active, mode="decode", sink=sink)
    ref_out, ref_lse = _reference(q, kv, sel, active, sink)
    _, off_lse = _run_public(q, kv, sel, lens, active, mode="decode", sink=None)
    _assert_sink_visible(off_lse, sink, what="decode")
    _assert_close(out, ref_out, what="decode-with-sink")
    _assert_lse_close(lse, ref_lse, what="decode-with-sink-lse")


@pytest.mark.parametrize("mode", ["decode", "extend"])
def test_sink_on_matches_decode_fold_of_sink_off(mode: str) -> None:
    """sink-on == sink-off * sigmoid(lse_off - sink): the exact decode fold.

    Asserted numerically between two real kernel runs, not against the torch
    reference, so a plumbing bug that applies the sink twice or drops it from
    one of the paths cannot pass.
    """
    require_b12x()
    rows, heads = 2, 8
    q, kv, sel, lens, active = _make_case(rows=rows, heads=heads)
    sink = _sink_logits(heads, seed=11)

    off_out, off_lse = _run_public(q, kv, sel, lens, active, mode=mode, sink=None)
    on_out, on_lse = _run_public(q, kv, sel, lens, active, mode=mode, sink=sink)
    _assert_sink_visible(off_lse, sink, what=mode)
    folded_out, folded_lse = _fold_sink(off_out, off_lse, sink)

    _assert_close(on_out, folded_out.to(torch.bfloat16), what=f"{mode}-fold-out")
    _assert_lse_close(on_lse, folded_lse, what=f"{mode}-fold-lse")


def test_mutant_sink_disabled_dies() -> None:
    """(a) sink disabled: a large sink must visibly change both out and LSE."""
    require_b12x()
    rows, heads = 2, 8
    q, kv, sel, lens, active = _make_case(rows=rows, heads=heads)
    # The row LSE over 2048 keys sits near log(2048) ~= 7.6 natural; a sink
    # logit of 12 dominates it and must attenuate every live row hard.
    sink = torch.full((heads,), 12.0, device="cuda")

    off_out, off_lse = _run_public(q, kv, sel, lens, active, mode="extend", sink=None)
    on_out, on_lse = _run_public(q, kv, sel, lens, active, mode="extend", sink=sink)

    scale = torch.sigmoid(off_lse.to(torch.float64) * LN2 - 12.0).float()
    # Every live row must be attenuated by the sink factor ...
    live = off_lse.isfinite()
    assert bool((scale[live] < 0.5).all().item()), f"weak sink attenuation: {scale[live]}"
    assert not torch.allclose(
        on_out.float(), off_out.float(), rtol=1e-3, atol=1e-3
    ), "sink-on equals sink-off: the sink is not reaching the kernel"
    # ... and the LSE must be pulled toward the sink value.
    assert torch.allclose(
        on_lse[live],
        torch.full_like(on_lse, 12.0 / LN2)[live],
        atol=0.5,
        rtol=0.0,
    )
    assert not torch.allclose(on_lse[live], off_lse[live], atol=1e-2, rtol=0.0)


def test_sink_presence_contract_enforced() -> None:
    """The plan-level caps contract still governs sink presence in every mode."""
    require_b12x()
    rows, heads = 1, 8
    q, kv, sel, lens, active = _make_case(rows=rows, heads=heads)
    sink = torch.zeros(heads, device="cuda")
    for mode in ("extend", "verify", "decode"):
        declaration = sparse_mla.plan(
            sparse_mla.Caps(
                device=q.device,
                num_q_heads=heads,
                max_q_rows=rows,
                max_batch=rows,
                max_width=TOPK,
                softmax_scale=HEAD_DIM**-0.5,
                kv_dtype=torch.uint8,
                head_dim=HEAD_DIM,
                v_head_dim=V_HEAD_DIM,
                page_size=PAGE_SIZE,
                mode=mode,
                return_lse=True,
                has_attention_sink=True,
            )
        )
        owned = {}

        def prepare_call(state):
            spec = state.scratch_specs()[0]
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=q.device)
            runtime = state.bind(
                scratch=scratch,
                q=q,
                kv_cache=kv,
                selected_indices=sel,
                cache_seqlens_int32=lens,
                nsa_cache_seqlens_int32=active,
            )
            owned["scratch"] = scratch
            state.prime(runtime, kv_cache=kv, attention_sink=sink)
            return PreparedCall(run=lambda: None)

        session = PreparationSession(device=q.device, compile_workers=0)
        with session:
            result = session.prepare(
                (declaration.request(name="presence", prepare_call=prepare_call),)
            )
            plan = result.plans["presence"]
            with pytest.raises(ValueError, match="attention_sink presence"):
                sparse_mla.bind(
                    plan,
                    scratch=owned["scratch"],
                    q=q,
                    kv_cache=kv,
                    selected_indices=sel,
                    cache_lengths=lens,
                    selected_lengths=active,
                    attention_sink=None,
                )


# ───────────────── defect 12: LSE base enforced at the boundary ─────────────


@pytest.mark.parametrize("mode", ["decode", "extend", "verify"])
def test_lse_base_contract_enforced_at_boundary(mode: str) -> None:
    """The returned LSE is in the DECLARED base for every mode (defect 12).

    Prefill/extend/verify kernels natively emit base-2; the bug (the mutant)
    returned base-2 whatever ``lse_scale`` said while the CP merge assumed
    natural logs.  A natural declaration must now return natural logs and a
    base2 declaration base-2 logs — never the other one.
    """
    require_b12x()
    rows, heads = 2, 8
    q, kv, sel, lens, active = _make_case(rows=rows, heads=heads)
    ref_out, ref_lse2 = sparse_mla_reference(
        q_all=q,
        kv_cache=kv,
        page_table_1=sel,
        active_token_counts=active,
        sm_scale=HEAD_DIM**-0.5,
        v_head_dim=V_HEAD_DIM,
        return_lse=True,
    )
    live = ref_lse2.isfinite()

    # base2 declaration: exactly the kernel's base-2 LSE.
    out2, lse2, b2 = _run_public(
        q, kv, sel, lens, active,
        mode=mode, sink=None, lse_scale="base2", want_binding=True,
    )
    assert b2.lse_base == "base2"
    _assert_lse_close(lse2, ref_lse2, what=f"{mode}-declared-base2")
    assert not torch.allclose(
        lse2[live].float(), (ref_lse2[live] * LN2).float(), atol=0.5, rtol=0.0
    ), f"{mode}: base2 declaration returned natural logs"

    # natural declaration: the boundary must convert (the mutant returns base-2).
    outn, lsen, bn = _run_public(
        q, kv, sel, lens, active,
        mode=mode, sink=None, lse_scale="natural", want_binding=True,
    )
    assert bn.lse_base == "natural"
    torch.testing.assert_close(
        lsen[live].float(),
        (ref_lse2[live] * LN2).float(),
        rtol=0.0,
        atol=0.08,
        msg=lambda m: f"{mode}-declared-natural: {m}",
    )
    assert not torch.allclose(
        lsen[live].float(), ref_lse2[live].float(), atol=0.5, rtol=0.0
    ), f"{mode}: lse_scale='natural' returned base-2 LSE (defect-12 mutant)"


def test_cp_provenance_rejects_plan_sink_plus_merge_sink() -> None:
    """A plan with a sink plus a merge with a sink must fail loudly (defect 11).

    The binding carries the fold provenance (``sink_folded``) and the log base
    (``lse_base``); a CP merge composed from them must reject the double fold
    and accept the supported single-fold shape.
    """
    require_b12x()
    from b12x.comm.pcie.pcie_dcp_a2a import lse_reduce_scatter_reference

    rows, heads = 1, 8
    q, kv, sel, lens, active = _make_case(rows=rows, heads=heads)
    sink = _sink_logits(heads, seed=3)

    # Plan WITH a sink: its LSE already carries the fold.
    _, _, bound = _run_public(
        q, kv, sel, lens, active,
        mode="decode", sink=sink, want_binding=True,
    )
    assert bound.sink_folded is True
    assert bound.lse_base == "base2"

    world_out = torch.zeros(2, 2, heads, 8)
    world_lse = torch.zeros(2, 2, heads)
    with pytest.raises(ValueError, match="double sink fold"):
        lse_reduce_scatter_reference(
            world_out,
            world_lse,
            0,
            lse_base=bound.lse_base,
            attention_sink=sink.cpu(),
            partials_sink_folded=bound.sink_folded,
            return_lse=True,
        )

    # Plan WITHOUT a sink: the merge is the single fold site and must work.
    _, _, plain = _run_public(
        q, kv, sel, lens, active,
        mode="decode", sink=None, want_binding=True,
    )
    assert plain.sink_folded is False
    merged, merged_lse = lse_reduce_scatter_reference(
        world_out,
        world_lse,
        0,
        lse_base=plain.lse_base,
        attention_sink=sink.cpu(),
        partials_sink_folded=plain.sink_folded,
        return_lse=True,
    )
    assert merged.shape == (2, heads // 2, 8)
    assert merged_lse.shape == (2, heads // 2)
