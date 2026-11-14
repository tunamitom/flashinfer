"""Shared-selection IndexCache cell (plan item 5).

21 ``full`` layers (0, 1, 5, … 77) run the DSA indexer; the 57 ``shared``
layers bind the full layer's ``selected`` buffer through the same
``data_ptr`` and allocate no index-key pages.  ``Caps(output_index_space=
"physical")`` makes the indexer emit flat physical cache slots that feed
``sparse_mla.bind(selected_indices=...)`` directly.

The tested cell is the 4-layer group (layers 1-4): layer 1 runs the indexer,
layers 2-4 consume layer 1's ``selected`` buffer.

Mutants that must die here (hy4-plan §1.5): stale or recomputed selection in
a shared layer.
"""

from __future__ import annotations

import pytest
import torch

from b12x.attention.dsa_indexer.index_cache import (
    HY4_FULL_LAYER_IDS,
    HY4_NUM_LAYERS,
    IndexCache,
    IndexerLayer,
    hy4_group_cell_layers,
    hy4_indexer_layers,
)
from b12x.attention.dsa_indexer.paged import (
    INDEX_HEAD_DIM,
    PAGED_INDEX_PAGE_SIZE,
)

_SCALE_BYTES = 4
_PAGE_BYTES = PAGED_INDEX_PAGE_SIZE * (INDEX_HEAD_DIM + _SCALE_BYTES)


def _group_cell(**kwargs) -> IndexCache:
    defaults = dict(device="cpu", topk=32, max_q_rows=4, index_pages_per_layer=2)
    defaults.update(kwargs)
    return IndexCache(hy4_group_cell_layers(1), **defaults)


# ──────────────────────────────── layout ──────────────────────────────────


def test_hy4_layout_matches_checkpoint_config() -> None:
    layers = hy4_indexer_layers()
    assert len(layers) == HY4_NUM_LAYERS == 78
    assert HY4_FULL_LAYER_IDS == (0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41,
                                  45, 49, 53, 57, 61, 65, 69, 73, 77)
    full = [layer.layer_id for layer in layers if layer.kind == "full"]
    shared = [layer.layer_id for layer in layers if layer.kind == "shared"]
    assert tuple(full) == HY4_FULL_LAYER_IDS
    assert len(full) == 21 and len(shared) == 57
    # Shared layers bind the previous full layer's buffer.
    by_id = {layer.layer_id: layer for layer in layers}
    assert by_id[2].owner_id == 1
    assert by_id[3].owner_id == 1
    assert by_id[4].owner_id == 1
    assert by_id[6].owner_id == 5
    assert by_id[76].owner_id == 73
    assert by_id[0].owner_id == 0 and by_id[77].owner_id == 77


def test_group_cell_shared_layers_bind_layer1_buffer() -> None:
    """Layers 2-4 consume layer 1's selected buffer: same data_ptr, no copy."""
    cell = _group_cell()
    owner = cell.selected(1)
    for shared_id in (2, 3, 4):
        buf = cell.selected(shared_id)
        assert buf.data_ptr() == owner.data_ptr()
        assert cell.shares_selection(shared_id, 1)
        assert cell.layer(shared_id).owner_id == 1
    # Writing through layer 1's handle is immediately visible to shared layers
    # (live aliasing, not a snapshot).
    cell.indexer_output(1).fill_(7)
    assert bool((cell.selected(2) == 7).all())
    assert cell.shares_selection(2, 3) and cell.shares_selection(3, 4)


def test_shared_layers_allocate_no_index_key_pages() -> None:
    cell = _group_cell(index_pages_per_layer=3)
    assert cell.index_key_pages(1) is not None
    assert cell.index_key_pages(1).shape == (3, _PAGE_BYTES)
    for shared_id in (2, 3, 4):
        assert cell.index_key_pages(shared_id) is None
    assert cell.num_index_key_pages == 3
    assert cell.index_key_bytes == 3 * _PAGE_BYTES

    model = IndexCache(
        hy4_indexer_layers(),
        device="cpu",
        topk=32,
        max_q_rows=2,
        index_pages_per_layer=4,
    )
    assert model.num_index_key_pages == 21 * 4
    assert model.index_key_bytes == 21 * 4 * _PAGE_BYTES
    assert model.selection_exchange_count == 21
    assert all(model.index_key_pages(i) is None for i in model.shared_layer_ids())


def test_only_full_layers_run_the_indexer() -> None:
    cell = _group_cell()
    assert cell.indexer_output(1) is cell.selected(1)
    for shared_id in (2, 3, 4):
        with pytest.raises(ValueError, match="only full layers"):
            cell.indexer_output(shared_id)
        with pytest.raises(ValueError, match="only full layers"):
            cell.physical_caps(shared_id, num_q_heads=4, max_page_table_width=2)


def test_caps_pin_physical_index_space() -> None:
    cell = _group_cell()
    caps = cell.physical_caps(1, num_q_heads=4, max_page_table_width=2)
    assert caps.output_index_space == "physical"
    assert caps.topk == cell.topk
    assert caps.max_q_rows == cell.max_q_rows
    assert caps.device == torch.device("cpu")
    with pytest.raises(ValueError, match="owned by the IndexCache"):
        cell.physical_caps(
            1,
            num_q_heads=4,
            max_page_table_width=2,
            output_index_space="logical",
        )


def test_cell_rejects_dangling_or_shared_owners() -> None:
    with pytest.raises(ValueError, match="unknown owner"):
        IndexCache(
            (IndexerLayer(3, "shared", 1),),
            device="cpu",
            topk=8,
            max_q_rows=1,
        )
    with pytest.raises(ValueError, match="must bind a full owner"):
        IndexCache(
            (IndexerLayer(1, "shared", 2), IndexerLayer(2, "shared", 1)),
            device="cpu",
            topk=8,
            max_q_rows=1,
        )
    with pytest.raises(ValueError, match="owns its own"):
        IndexerLayer(4, "full", 1)


# ──────────────────── selection semantics + mutant kills ──────────────────


def _mini_case(seed: int = 20261114):
    torch.manual_seed(seed)
    tokens, rows, d_k, d_v, topk = 384, 8, 48, 32, 48
    k_pool = torch.randn(tokens, d_k) * 0.5
    v_pool = torch.randn(tokens, d_v)
    q_full = torch.randn(rows, d_k)          # layer 1's indexer query
    q_shared = {
        layer: torch.randn(rows, d_k) for layer in (2, 3, 4)  # per-layer queries
    }
    return k_pool, v_pool, q_full, q_shared, topk


def _select(q: torch.Tensor, k_pool: torch.Tensor, topk: int) -> torch.Tensor:
    scores = q @ k_pool.T * 3.0
    return scores.topk(topk, dim=-1).indices.sort(dim=-1).values


def _attend(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    selection: torch.Tensor,
) -> torch.Tensor:
    scores = (q @ k_pool.T * 3.0).gather(1, selection)
    probs = scores.softmax(dim=-1)
    values = v_pool[selection]  # [rows, topk, d_v]
    return torch.einsum("rk,rkd->rd", probs, values)


def test_shared_layers_consume_the_live_full_layer_selection() -> None:
    k_pool, v_pool, q_full, q_shared, topk = _mini_case()
    cell = _group_cell(topk=topk, max_q_rows=q_full.shape[0], index_pages_per_layer=0)
    selection = _select(q_full, k_pool, topk)
    cell.indexer_output(1).copy_(selection.to(torch.int32))

    for layer_id in (2, 3, 4):
        shared_sel = cell.selected(layer_id).long()
        assert torch.equal(shared_sel, selection)
        out = _attend(q_shared[layer_id], k_pool, v_pool, shared_sel)
        expected = _attend(q_shared[layer_id], k_pool, v_pool, selection)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_mutant_stale_selection_in_shared_layer_dies() -> None:
    """A snapshot from an earlier step must NOT reproduce the ground truth."""
    k_pool, v_pool, q_full, q_shared, topk = _mini_case()
    cell = _group_cell(topk=topk, max_q_rows=q_full.shape[0], index_pages_per_layer=0)

    # The buffer state BEFORE the full layer wrote: no selection (-1).
    before_write = cell.selected(2).clone()
    assert bool((before_write == -1).all())

    selection = _select(q_full, k_pool, topk)
    cell.indexer_output(1).copy_(selection.to(torch.int32))
    expected = _attend(q_shared[2], k_pool, v_pool, cell.selected(2).long())

    # Stale: the previous step's selection snapshot, kept instead of reading
    # the live buffer.
    stale = _select(q_full.flip(0), k_pool, topk).to(torch.int32).clone()
    assert stale.data_ptr() != cell.selected(2).data_ptr(), (
        "the cell handed a shared layer a private buffer"
    )
    mutant = _attend(q_shared[2], k_pool, v_pool, stale.long())
    assert not torch.allclose(mutant, expected, rtol=0.02, atol=0.02), (
        "stale selection matches the live-selection ground truth"
    )
    assert not torch.allclose(mutant, expected, rtol=1e-3, atol=1e-3)
    # The pre-write (-1) snapshot is likewise not a valid selection.
    with pytest.raises((IndexError, RuntimeError), match="out of bounds"):
        _attend(q_shared[2], k_pool, v_pool, before_write.long())


def test_mutant_recomputed_selection_in_shared_layer_dies() -> None:
    """A shared layer that runs its own top-k must NOT match the ground truth.

    Recomputing is also what a shared layer with its own index-key pages
    would do; the cell allocates none and the digest must differ.
    """
    k_pool, v_pool, q_full, q_shared, topk = _mini_case()
    cell = _group_cell(topk=topk, max_q_rows=q_full.shape[0], index_pages_per_layer=0)
    selection = _select(q_full, k_pool, topk)
    cell.indexer_output(1).copy_(selection.to(torch.int32))
    expected = _attend(q_shared[3], k_pool, v_pool, cell.selected(3).long())

    recomputed = _select(q_shared[3], k_pool, topk)  # its own query, its own top-k
    assert not torch.equal(recomputed, selection), (
        "seed collapsed: recomputed selection equals the shared one"
    )
    mutant = _attend(q_shared[3], k_pool, v_pool, recomputed)
    assert not torch.allclose(mutant, expected, rtol=0.02, atol=0.02), (
        "recomputed selection matches the shared ground truth"
    )
    assert not torch.allclose(mutant, expected, rtol=1e-3, atol=1e-3)
    # And the shared handle must still alias the owner's buffer.
    assert cell.selected(3).data_ptr() == cell.selected(1).data_ptr()
    assert cell.index_key_pages(3) is None


# ───────────── GPU: physical slots feed sparse_mla.bind directly ───────────


def _run_sparse_mla_decode(q_all, kv_cache, selected, cache_seqlens, active, *, width, heads):
    from b12x.preparation import PreparedCall, PreparationSession
    from b12x.attention import sparse_mla

    rows = q_all.shape[0]
    sm_scale = 576**-0.5
    declaration = sparse_mla.plan(
        sparse_mla.Caps(
            device=q_all.device,
            num_q_heads=heads,
            max_q_rows=rows,
            max_width=width,
            softmax_scale=sm_scale,
            kv_dtype=torch.uint8,
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
        state.prime(runtime, kv_cache=kv_cache)
        return PreparedCall(run=lambda: state.run(runtime, kv_cache=kv_cache))

    session = PreparationSession(device=q_all.device)
    result = session.prepare(
        (declaration.request(name="cell-sparse-mla", prepare_call=prepare_call),)
    )
    plan = result.plans["cell-sparse-mla"]
    binding = sparse_mla.bind(
        plan,
        scratch=owned["scratch"],
        q=q_all,
        kv_cache=kv_cache,
        selected_indices=selected,
        cache_lengths=cache_seqlens,
        selected_lengths=active,
    )
    return sparse_mla.run(binding)


def test_group_cell_physical_slots_feed_sparse_mla() -> None:
    """End-to-end on switch-D scratch cards: dsa_indexer physical-slot output
    feeds ``sparse_mla.bind(selected_indices=...)`` for every shared layer of
    the 4-layer cell, and both stale/recomputed mutants die on GPU too.
    """
    from ..conftest import require_b12x
    from b12x.preparation import PreparedCall, PreparationSession
    from b12x.attention import dsa_indexer as dsa
    from b12x.attention._shared.mla.reference import (
        pack_mla_kv_cache_reference,
        sparse_mla_reference,
    )
    from b12x.attention.dsa_indexer.reference import (
        pack_index_k_cache_reference,
        unpack_index_k_cache_reference,
    )

    device = require_b12x()
    torch.manual_seed(20261114)
    rows, heads, topk = 2, 16, 512  # topk must be 512/1024/2048 (tiled topk)
    mla_heads = 8
    max_pages = 32
    tokens = max_pages * PAGED_INDEX_PAGE_SIZE  # 2048 physical slots

    # ── full layer 1: rotated-at-write keys into the cell's index pages ──
    k_raw = torch.randn(tokens, INDEX_HEAD_DIM, device=device) / 2
    positions = torch.arange(tokens, device=device)
    k_norm_weight = torch.randn(INDEX_HEAD_DIM, device=device) * 0.5 + 1.0
    k_norm_bias = torch.randn(INDEX_HEAD_DIM, device=device) * 0.05
    packed = pack_index_k_cache_reference(
        k_raw,
        positions=positions,
        k_norm_weight=k_norm_weight,
        k_norm_bias=k_norm_bias,
    )
    decoded = unpack_index_k_cache_reference(packed, num_tokens=tokens).float()

    cell = IndexCache(
        hy4_group_cell_layers(1),
        device=device,
        topk=topk,
        max_q_rows=rows,
        index_pages_per_layer=max_pages,
    )
    assert cell.num_index_key_pages == max_pages  # shared layers: none
    cell.index_key_pages(1).copy_(packed)

    pages = torch.arange(max_pages, dtype=torch.int32, device=device)[None].repeat(
        rows, 1
    )
    lengths = torch.full((rows,), tokens, dtype=torch.int32, device=device)
    q_fp8 = torch.randn(rows, heads, INDEX_HEAD_DIM, device=device).to(
        torch.float8_e4m3fn
    )
    weights = torch.rand(rows, heads, device=device)

    caps = cell.physical_caps(
        1,
        num_q_heads=heads,
        max_page_table_width=max_pages,
        mode="decode",
        route="paged_tiled",
        supertile_k=1024,
    )
    assert caps.output_index_space == "physical"
    operands = dict(
        q_fp8=q_fp8,
        query_weights=weights,
        index_k_cache=cell.index_key_pages(1),
        page_table=pages,
        cache_lengths=lengths,
        active_width=None,
        output_indices=cell.indexer_output(1),
        output_scores=None,
    )
    plan = dsa.plan(caps, invocation=dsa.invocation_from_tensors(caps, **operands))
    owned = {}

    def prepare(state):
        (spec,) = state.layout.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        binding = state.bind(
            scratch=scratch,
            real_page_table=pages,
            cache_seqlens_int32=lengths,
            active_width=None,
            expected_num_q_heads=heads,
            shared_page_table=False,
            output_physical_slots=True,
        )
        owned["scratch"] = scratch
        return PreparedCall(
            run=lambda: state.run(
                binding,
                q_fp8=q_fp8,
                query_weights=weights,
                index_k_cache=cell.index_key_pages(1),
                output_indices=cell.indexer_output(1),
                output_scores=None,
            )
        )

    # The recomputed-selection mutant: the shared layer's own indexer run
    # into a private buffer (its own key pages would be the full-mutant; the
    # query flip is enough to give it a different top-k).
    private = torch.full_like(cell.indexer_output(1), -1)
    with PreparationSession(device=device, autotune=False) as session:
        session.prepare((plan.request(name="cell-indexer", prepare_call=prepare),))
        binding = dsa.bind(plan, scratch=owned["scratch"], **operands)
        dsa.run(binding)
        mutant_operands = dict(operands)
        mutant_operands["q_fp8"] = (-q_fp8.float()).to(torch.float8_e4m3fn)
        mutant_operands["output_indices"] = private
        binding2 = dsa.bind(plan, scratch=owned["scratch"], **mutant_operands)
        dsa.run(binding2)
    torch.cuda.synchronize(device)

    # ── the selection is physical slots: page_id * 64 + slot == token id ──
    logits = torch.einsum("rhd,kd->rhk", q_fp8.float(), decoded)
    ref_scores = (logits.relu_() * weights[:, :, None]).sum(dim=1)
    expected_sel = (
        ref_scores.topk(topk, dim=1).indices.sort(dim=1).values.to(torch.int32)
    )
    torch.testing.assert_close(
        cell.indexer_output(1).sort(dim=1).values, expected_sel, rtol=0, atol=0
    )
    # The shared layers read the very buffer the indexer wrote.
    for layer_id in (2, 3, 4):
        assert cell.selected(layer_id).data_ptr() == cell.indexer_output(1).data_ptr()
        assert cell.index_key_pages(layer_id) is None

    # ── shared layers 2-4: the same slots feed sparse_mla.bind directly ──
    k_nope = torch.randn(tokens, 512, device=device, dtype=torch.bfloat16) / 4
    k_rope = torch.randn(tokens, 64, device=device, dtype=torch.bfloat16) / 4
    kv_cache = pack_mla_kv_cache_reference(k_nope, k_rope)
    cache_seqlens = torch.full(
        (rows,), tokens, dtype=torch.int32, device=device
    )
    active = torch.full((rows,), topk, dtype=torch.int32, device=device)

    for layer_id in (2, 3, 4):
        q_layer = (
            torch.randn(rows, mla_heads, 576, device=device, dtype=torch.bfloat16) / 4
        )
        out = _run_sparse_mla_decode(
            q_layer,
            kv_cache,
            cell.selected(layer_id),
            cache_seqlens,
            active,
            width=topk,
            heads=mla_heads,
        )
        ref = sparse_mla_reference(
            q_all=q_layer,
            kv_cache=kv_cache,
            page_table_1=cell.selected(layer_id),
            active_token_counts=active,
            sm_scale=576**-0.5,
            v_head_dim=512,
        )
        torch.cuda.synchronize(device)
        cosine = torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref.float().flatten(), dim=0
        )
        assert float(cosine.item()) > 0.99
        torch.testing.assert_close(
            out.float(), ref.to(out.dtype).float(), rtol=5e-2, atol=5e-2
        )

    # Ground truth for the mutant checks: layer 2's query against the live
    # shared selection, with the same kernel-vs-reference error budget as the
    # positive path (used to keep the mutant margins scale-free).
    q_gt = torch.randn(rows, mla_heads, 576, device=device, dtype=torch.bfloat16) / 4
    gt = _run_sparse_mla_decode(
        q_gt, kv_cache, cell.selected(2), cache_seqlens, active,
        width=topk, heads=mla_heads,
    )
    gt_ref = sparse_mla_reference(
        q_all=q_gt,
        kv_cache=kv_cache,
        page_table_1=cell.selected(2),
        active_token_counts=active,
        sm_scale=576**-0.5,
        v_head_dim=512,
    )
    torch.cuda.synchronize(device)
    positive_err = float((gt.float() - gt_ref.float()).abs().max())

    def _assert_mutant_dies(mutant_out, name):
        torch.cuda.synchronize(device)
        delta = float((mutant_out.float() - gt.float()).abs().max())
        assert delta > 10 * max(positive_err, 1e-6), (
            f"{name}: mutant error {delta:.5f} is within the positive-path "
            f"budget {positive_err:.5f} — the check cannot discriminate"
        )
        cosine = torch.nn.functional.cosine_similarity(
            mutant_out.float().flatten(), gt.float().flatten(), dim=0
        )
        assert float(cosine.item()) < 0.98, (
            f"{name}: mutant output is cosine-similar to the shared ground "
            f"truth ({float(cosine.item()):.5f})"
        )

    # (a) stale: the previous step's selection snapshot.  It must hold
    # genuinely different slots — a reorder of the same set would leave the
    # permutation-invariant attention output unchanged and kill nothing.
    prev_scores = (
        torch.einsum("rhd,kd->rhk", q_fp8.float().flip(0), decoded).relu_()
        * weights.flip(0)[:, :, None]
    ).sum(dim=1)
    stale = (
        prev_scores.topk(topk, dim=1).indices.sort(dim=1).values.to(torch.int32)
    ).clone()
    assert stale.data_ptr() != cell.selected(2).data_ptr()
    overlap = (stale == cell.selected(2)).float().mean()
    assert float(overlap) < 0.75, f"stale snapshot shares {float(overlap):.2f} of slots"
    out_stale = _run_sparse_mla_decode(
        q_gt, kv_cache, stale, cache_seqlens, active, width=topk, heads=mla_heads
    )
    _assert_mutant_dies(out_stale, "stale selection")

    # (b) recomputed: the shared layer's own indexer run (computed above into
    # `private`) must not reproduce the shared-selection ground truth.
    assert private.data_ptr() != cell.selected(2).data_ptr()
    assert not torch.equal(
        private.sort(dim=1).values, cell.selected(2).sort(dim=1).values
    ), "recomputed selection equals the shared selection"
    out_recomputed = _run_sparse_mla_decode(
        q_gt, kv_cache, private, cache_seqlens, active, width=topk, heads=mla_heads
    )
    _assert_mutant_dies(out_recomputed, "recomputed selection")
    # The cell still hands shared layers the owner's buffer, never a private one.
    assert cell.selected(2).data_ptr() == cell.indexer_output(1).data_ptr()
