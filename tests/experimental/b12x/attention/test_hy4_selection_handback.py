"""Cross-rank top-k selection hand-back (defect 15b).

Under context parallelism the CP index merge reduces every rank's local
top-k candidates to ONE global top-k per query row (``merge_tiled_topk_candidates``
over the planes ``comm.pcie.pcie_dcp_topk`` stages into each row owner).  A
rank's sparse MLA attends over the records it OWNS, so the merged selection
must be taken back to per-rank selections: the entries this rank owns, mapped
to its local physical slots, compacted with a per-row live count and ``-1``
padded.  Hy4's 57 shared layers reach the hand-back through their owner full
layer's buffer (``IndexCache``: the group cell binds layer 1's buffer).

Witness: kimi3 ``ignition/recipes/kimi3/dsa.py`` (``PreparedIndexMerge``,
``select()``) + ``kimi3/dcp.py`` (``ContextShards.token_slots``, ``compact``).

Mutants that must die here:
  (a) no hand-back — a rank (or shared layer) consuming the merged global
      selection as if it were local attends the wrong records;
  (b) a stale/privatized hand-back for shared layers — a shared layer must
      see the live owner buffer after every hand-back.
"""

from __future__ import annotations

import pytest
import torch

from b12x.attention import sparse_mla
from b12x.attention._shared.mla.reference import (
    pack_mla_kv_cache_reference,
    sparse_mla_reference,
)
from b12x.attention.dsa_indexer.index_cache import (
    IndexCache,
    hy4_group_cell_layers,
    hy4_indexer_layers,
)
from b12x.attention.dsa_indexer.selection_handback import (
    PreparedSelectionHandback,
    compact_selection,
    prepare_selection_handback,
    rank_local_slots,
    selection_handback_reference,
)
from b12x.attention.dsa_indexer.tiled_topk import merge_tiled_topk_candidates

from b12x.preparation import PreparedCall, PreparationSession

from ..conftest import require_b12x

NOPE_DIM = 512
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM  # 576
V_HEAD_DIM = NOPE_DIM
PAGE_SIZE = 64


# ─────────────────────────── reference semantics ───────────────────────────


@pytest.mark.cpu_only
def test_rank_local_slots_maps_interleaved_ownership() -> None:
    """Global p -> rank p % world at local p // world; -1 elsewhere (kimi3)."""
    positions = torch.tensor(
        [[0, 1, 2, 3, 4, 5, -1, 7], [10, 11, -1, -1, -1, -1, -1, -1]],
        dtype=torch.int32,
    )
    got0 = rank_local_slots(positions, world_size=2, rank=0)
    got1 = rank_local_slots(positions, world_size=2, rank=1)
    assert got0.tolist() == [[0, -1, 1, -1, 2, -1, -1, -1], [5, -1, -1, -1, -1, -1, -1, -1]]
    assert got1.tolist() == [[-1, 0, -1, 1, -1, 2, -1, 3], [-1, 5, -1, -1, -1, -1, -1, -1]]
    # Ownership is exclusive: no slot is live on both ranks.
    assert not bool(((got0 >= 0) & (got1 >= 0)).any())
    # A paged layout maps local indices through a slot table (and can drop).
    slot_map = torch.tensor([9, 8, -1, 7, 6, 5, 4, 3], dtype=torch.int32)
    got = rank_local_slots(
        torch.tensor([[0, 2, 4, 6, 16]], dtype=torch.int32),
        world_size=2,
        rank=0,
        slot_map=slot_map,
    )
    assert got.tolist() == [[9, 8, -1, 7, -1]]  # local 2 dropped, 8 out of range
    with pytest.raises(ValueError, match="rank"):
        rank_local_slots(positions, world_size=2, rank=2)
    with pytest.raises(ValueError, match="world_size"):
        rank_local_slots(positions, world_size=0, rank=0)


@pytest.mark.cpu_only
def test_compact_selection_matches_naive() -> None:
    """Stable compaction to the front, -1 padding, per-row counts."""
    ids = torch.tensor(
        [[5, -1, 3, -1, 7, -1], [-1, -1, -1, -1, -1, -1], [1, 2, 3, 4, 5, 6]],
        dtype=torch.int32,
    )
    live = ids >= 0
    scratch = torch.empty((3, 7), dtype=torch.int32)
    out, counts = compact_selection(ids, live, scratch)
    assert out.tolist() == [[5, 3, 7, -1, -1, -1], [-1] * 6, [1, 2, 3, 4, 5, 6]]
    assert counts.tolist() == [3, 0, 6]
    # Naive per-row agreement on a random corpus.
    torch.manual_seed(3)
    ids = torch.randint(-1, 1000, (17, 31), dtype=torch.int32)
    live = ids >= 0
    scratch = torch.empty((17, 32), dtype=torch.int32)
    out, counts = compact_selection(ids, live, scratch)
    for row in range(17):
        expected = [int(v) for v in ids[row][live[row]].tolist()]
        expected += [-1] * (31 - len(expected))
        assert out[row].tolist() == expected
        assert int(counts[row]) == len([v for v in expected if v >= 0])


@pytest.mark.cpu_only
def test_handback_partitions_the_merged_selection_2_rank() -> None:
    """2-rank merge -> hand-back: the ranks' selections partition the merge.

    The CP index merge (``merge_tiled_topk_candidates`` over the two ranks'
    candidate planes) produces one global top-k per row; each rank's hand-back
    must recover exactly the entries it owns, compacted in merged order, and
    the two hand-backs together must cover every live merged entry exactly
    once.  One row is deliberately rank-0-only so an all-empty rank-1 row
    exercises the -1/count-0 path.
    """
    world, rows, topk = 2, 3, 16
    local_tokens = 64
    torch.manual_seed(20261114)
    scores = torch.rand(world, rows, local_tokens)
    scores[1, 1] = -torch.inf  # row 1: every winner comes from rank 0

    # Each rank offers its local top-k as (score, global position) candidates.
    cand_values = torch.empty((world, rows, topk), dtype=torch.float32)
    cand_indices = torch.empty((world, rows, topk), dtype=torch.int32)
    for rank in range(world):
        values, local = scores[rank].topk(topk, dim=-1)
        cand_values[rank] = values
        cand_indices[rank] = (local * world + rank).to(torch.int32)

    merged_values, merged = merge_tiled_topk_candidates(
        candidate_values=cand_values,
        candidate_indices=cand_indices,
        topk=topk,
    )
    assert merged.shape == (rows, topk)

    recovered = []
    for rank in range(world):
        scratch = torch.empty((rows, topk + 1), dtype=torch.int32)
        local_sel, counts = selection_handback_reference(
            merged, world_size=world, rank=rank, scratch=scratch
        )
        recovered.append((local_sel, counts))
        # Ground truth from the merged row itself: owned entries in merged
        # order, mapped to local slots, compacted, -1 padded.
        for row in range(rows):
            expected = [
                int(p) // world
                for p in merged[row].tolist()
                if p >= 0 and int(p) % world == rank
            ]
            got = [int(v) for v in local_sel[row].tolist() if v >= 0]
            assert got == expected, f"rank {rank} row {row}: {got} != {expected}"
            assert int(counts[row]) == len(expected)

    # Partition property: the two hand-backs cover the merged live set exactly.
    for row in range(rows):
        merged_live = sorted(int(p) for p in merged[row].tolist() if p >= 0)
        union = []
        for rank, (local_sel, _) in enumerate(recovered):
            union += [
                int(v) * world + rank for v in local_sel[row].tolist() if v >= 0
            ]
        assert sorted(union) == merged_live, f"row {row}: hand-backs lost entries"

    # The forced row: rank 1 recovers nothing there.
    assert int(recovered[1][1][1]) == 0
    assert bool((recovered[1][0][1] == -1).all())


@pytest.mark.cpu_only
def test_prepared_handback_matches_reference() -> None:
    """The geometry-pinned hand-back is the reference, and validates once."""
    world, rows, topk = 2, 4, 8
    torch.manual_seed(5)
    merged = torch.randint(-1, 32, (rows, topk), dtype=torch.int32)
    prepared = prepare_selection_handback(merged=merged, world_size=world, rank=1)
    assert isinstance(prepared, PreparedSelectionHandback)
    out, counts = prepared.run(merged)
    scratch = torch.empty((rows, topk + 1), dtype=torch.int32)
    ref_out, ref_counts = selection_handback_reference(
        merged, world_size=world, rank=1, scratch=scratch
    )
    assert torch.equal(out, ref_out)
    assert torch.equal(counts, ref_counts)
    with pytest.raises(ValueError, match="merged"):
        prepared.run(merged[:2])


# ────────────────── GPU: the 2-rank CP decode design end to end ─────────────


def _run_sparse_mla(q, kv, selected, cache_seqlens, selected_lengths, *, rows, heads, width):
    """The public sparse_mla plan -> bind -> run lifecycle (decode, no sink)."""
    sm_scale = HEAD_DIM**-0.5
    declaration = sparse_mla.plan(
        sparse_mla.Caps(
            device=q.device,
            num_q_heads=heads,
            max_q_rows=rows,
            max_batch=rows,
            max_width=width,
            softmax_scale=sm_scale,
            kv_dtype=torch.uint8,
            head_dim=HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            page_size=PAGE_SIZE,
            mode="decode",
            return_lse=True,
            lse_scale="base2",
            has_attention_sink=False,
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
            selected_indices=selected,
            cache_seqlens_int32=cache_seqlens,
            nsa_cache_seqlens_int32=selected_lengths,
        )
        owned["scratch"] = scratch
        state.prime(runtime, kv_cache=kv)
        return PreparedCall(run=lambda: state.run(runtime, kv_cache=kv))

    session = PreparationSession(device=q.device, compile_workers=0)
    with session:
        result = session.prepare(
            (declaration.request(name="handback", prepare_call=prepare_call),)
        )
        plan = result.plans["handback"]
        binding = sparse_mla.bind(
            plan,
            scratch=owned["scratch"],
            q=q,
            kv_cache=kv,
            selected_indices=selected,
            cache_lengths=cache_seqlens,
            selected_lengths=selected_lengths,
            attention_sink=None,
        )
        return sparse_mla.run(binding)


def _two_rank_case(device, *, rows=3, topk=16, local_tokens=256, seed=20261114):
    """Global records, per-rank shards (kimi3 interleave), merge, hand-backs."""
    world = 2
    torch.manual_seed(seed)
    k_nope = (
        torch.randn(world * local_tokens, NOPE_DIM, device=device, dtype=torch.bfloat16)
        / 4
    )
    k_rope = (
        torch.randn(world * local_tokens, ROPE_DIM, device=device, dtype=torch.bfloat16)
        / 4
    )
    q = torch.randn(rows, 8, HEAD_DIM, device=device, dtype=torch.bfloat16) / 4

    # Each rank scores ITS OWN shard's keys and offers its local top-k.
    cand_values = torch.empty((world, rows, topk), dtype=torch.float32, device=device)
    cand_indices = torch.empty((world, rows, topk), dtype=torch.int32, device=device)
    for rank in range(world):
        scores = torch.rand(rows, local_tokens, device=device)
        if rank == 1:
            scores[1] = -torch.inf  # row 1 is served by rank 0 alone
        values, local = scores.topk(topk, dim=-1)
        cand_values[rank] = values
        cand_indices[rank] = (local * world + rank).to(torch.int32)

    _, merged = merge_tiled_topk_candidates(
        candidate_values=cand_values, candidate_indices=cand_indices, topk=topk
    )

    handbacks = []
    for rank in range(world):
        scratch = torch.empty((rows, topk + 1), dtype=torch.int32, device=device)
        handbacks.append(
            selection_handback_reference(
                merged, world_size=world, rank=rank, scratch=scratch
            )
        )

    # Per-rank caches: local record l of rank r is global record l * world + r.
    rank_kv = []
    for rank in range(world):
        rank_kv.append(
            pack_mla_kv_cache_reference(k_nope[rank::world], k_rope[rank::world])
        )
    global_kv = pack_mla_kv_cache_reference(k_nope, k_rope)
    return q, k_nope, k_rope, global_kv, rank_kv, merged, handbacks


def test_2_rank_gpu_handback_feeds_sparse_mla() -> None:
    """Merge -> hand-back -> per-rank sparse MLA -> LSE combine == global truth.

    The whole context-parallel decode design on two simulated ranks and the
    real kernels: each rank attends the handed-back selection over the records
    it owns, and ``lse_reduce_scatter_reference`` combines the partials by
    their LSE.  The result must match single-shot sparse MLA over the merged
    global selection.
    """
    from b12x.comm.pcie.pcie_dcp_a2a import lse_reduce_scatter_reference

    require_b12x()
    device = torch.device("cuda")
    rows, topk, world, heads = 3, 16, 2, 8
    q, _, _, global_kv, rank_kv, merged, handbacks = _two_rank_case(
        device, rows=rows, topk=topk
    )

    merged_counts = (merged >= 0).sum(-1).to(torch.int32)
    # The merged rows are NOT compacted (merge order), so the truth attends
    # every slot and lets the -1 padding mark dead entries.
    full_width = torch.full((rows,), topk, dtype=torch.int32, device=device)
    truth_out, truth_lse2 = sparse_mla_reference(
        q_all=q,
        kv_cache=global_kv,
        page_table_1=merged,
        active_token_counts=full_width,
        sm_scale=HEAD_DIM**-0.5,
        v_head_dim=V_HEAD_DIM,
        return_lse=True,
    )

    partials_out = []
    partials_lse = []
    for rank in range(world):
        local_sel, counts = handbacks[rank]
        out, lse = _run_sparse_mla(
            q,
            rank_kv[rank],
            local_sel,
            torch.full(
                (rows,), int(rank_kv[rank].shape[0]), dtype=torch.int32, device=device
            ),
            counts,
            rows=rows,
            heads=heads,
            width=topk,
        )
        partials_out.append(out)
        partials_lse.append(lse)
    stacked_out = torch.stack(partials_out)  # [world, rows, heads, dim]
    stacked_lse = torch.stack(partials_lse)  # [world, rows, heads]

    for rank in range(world):
        merged_out, merged_lse = lse_reduce_scatter_reference(
            stacked_out,
            stacked_lse,
            rank,
            lse_base="base2",
            partials_sink_folded=False,
            return_lse=True,
        )
        head_slice = slice(rank * (heads // world), (rank + 1) * (heads // world))
        expected = truth_out[:, head_slice]
        expected_lse = truth_lse2[:, head_slice]
        got, got_lse = merged_out.float(), merged_lse.float()
        cosine = torch.nn.functional.cosine_similarity(
            got.reshape(-1), expected.float().reshape(-1), dim=0
        )
        assert float(cosine.item()) > 0.999, f"rank {rank}: cosine {float(cosine.item()):.6f}"
        scale = float(expected.float().abs().max())
        torch.testing.assert_close(
            got,
            expected.float(),
            rtol=2e-2,
            atol=2e-2 * max(scale, 1e-6),
            msg=lambda m: f"rank {rank} output: {m}",
        )
        live = expected_lse.isfinite()
        torch.testing.assert_close(
            got_lse[live],
            expected_lse[live].float(),
            rtol=0.0,
            atol=5e-2,
            msg=lambda m: f"rank {rank} lse: {m}",
        )


def test_mutant_no_handback_dies_2_rank() -> None:
    """(a) consuming the merged global selection as local must NOT pass.

    The no-hand-back mutant reads global positions as if they were this
    rank's local slots — it attends different records and the combined output
    misses the ground truth.
    """
    from b12x.comm.pcie.pcie_dcp_a2a import lse_reduce_scatter_reference

    require_b12x()
    device = torch.device("cuda")
    rows, topk, world, heads = 3, 16, 2, 8
    q, _, _, global_kv, rank_kv, merged, handbacks = _two_rank_case(
        device, rows=rows, topk=topk
    )
    merged_counts = (merged >= 0).sum(-1).to(torch.int32)
    full_width = torch.full((rows,), topk, dtype=torch.int32, device=device)
    truth_out, _ = sparse_mla_reference(
        q_all=q,
        kv_cache=global_kv,
        page_table_1=merged,
        active_token_counts=full_width,
        sm_scale=HEAD_DIM**-0.5,
        v_head_dim=V_HEAD_DIM,
        return_lse=True,
    )

    # Mutant: skip the hand-back entirely; each rank feeds the merged global
    # positions straight into its local cache.
    partials_out, partials_lse = [], []
    for rank in range(world):
        out, lse = _run_sparse_mla(
            q,
            rank_kv[rank],
            merged,
            torch.full(
                (rows,), int(rank_kv[rank].shape[0]), dtype=torch.int32, device=device
            ),
            full_width,
            rows=rows,
            heads=heads,
            width=topk,
        )
        partials_out.append(out)
        partials_lse.append(lse)
    stacked_out = torch.stack(partials_out)
    stacked_lse = torch.stack(partials_lse)
    for rank in range(world):
        mutant_out = lse_reduce_scatter_reference(
            stacked_out,
            stacked_lse,
            rank,
            lse_base="base2",
            partials_sink_folded=False,
        )
        head_slice = slice(rank * (heads // world), (rank + 1) * (heads // world))
        expected = truth_out[:, head_slice].float()
        delta = float((mutant_out.float() - expected).abs().max())
        scale = float(expected.abs().max())
        assert delta > 0.1 * max(scale, 1e-6), (
            f"rank {rank}: the no-hand-back mutant matches the ground truth "
            f"(delta {delta:.5f} vs scale {scale:.5f})"
        )


# ───────────── shared layers: 57 of them bind the owner's buffer ────────────


def _shared_cell_case(device, *, rows=3, topk=16, local_tokens=64, seed=7):
    world = 2
    torch.manual_seed(seed)
    merged = torch.randint(-1, world * local_tokens, (rows, topk), dtype=torch.int32, device=device)
    cache = IndexCache(
        hy4_indexer_layers(),
        device=device,
        topk=topk,
        max_q_rows=rows,
        index_pages_per_layer=0,
    )
    return world, local_tokens, merged, cache


def test_shared_layers_see_the_live_handback() -> None:
    """All 57 shared layers reach their owner's hand-back buffers (defect 15b).

    The hand-back runs once per group (into layer 1's buffers here); every
    shared layer of the map resolves ``selected()``/``counts()`` to its owner's
    buffers by ``data_ptr`` and reads the live content — including when the
    hand-back is requested THROUGH a shared layer id.
    """
    require_b12x()
    device = torch.device("cuda")
    world, _, merged, cache = _shared_cell_case(device)

    layers = hy4_indexer_layers()
    assert len(cache.shared_layer_ids()) == 57  # 21 full + 57 shared = 78

    for rank in range(world):
        selected, counts = cache.handback(1, merged, world_size=world, rank=rank)
        expected_sel, expected_counts = selection_handback_reference(
            merged,
            world_size=world,
            rank=rank,
            scratch=torch.empty(
                (merged.shape[0], merged.shape[1] + 1),
                dtype=torch.int32,
                device=device,
            ),
        )
        assert torch.equal(selected[: merged.shape[0]], expected_sel)
        assert torch.equal(counts[: merged.shape[0]], expected_counts)
        for layer in layers:
            owner_sel = cache.selected(layer.owner_id)
            owner_counts = cache.counts(layer.owner_id)
            assert cache.selected(layer.layer_id).data_ptr() == owner_sel.data_ptr()
            assert cache.counts(layer.layer_id).data_ptr() == owner_counts.data_ptr()
        # Requesting the hand-back through a shared layer id reaches layer 1's
        # buffers: identical content, same data_ptr, no private copy.
        sel2, counts2 = cache.handback(2, merged, world_size=world, rank=rank)
        assert sel2.data_ptr() == cache.selected(1).data_ptr()
        assert counts2.data_ptr() == cache.counts(1).data_ptr()
        assert torch.equal(sel2[: merged.shape[0]], expected_sel)


def test_mutant_stale_shared_handback_dies() -> None:
    """(b) a snapshot taken at the first hand-back is stale by the second."""
    require_b12x()
    device = torch.device("cuda")
    world, _, merged, cache = _shared_cell_case(device, seed=11)

    cache.handback(1, merged, world_size=world, rank=0)
    stale = cache.selected(2)[: merged.shape[0]].clone()  # the mutant's copy
    second = torch.roll(merged, shifts=3, dims=1)  # the next forward's merge
    cache.handback(1, second, world_size=world, rank=0)
    live = cache.selected(3)[: second.shape[0]]  # another shared layer, live
    assert torch.equal(live, cache.selected(1)[: second.shape[0]])
    assert not torch.equal(stale, live), (
        "the stale snapshot matches the live hand-back: staleness cannot die"
    )
    assert not torch.equal(
        cache.selected(2)[: second.shape[0]], stale
    ), "a shared layer still serves the first hand-back's snapshot"


def test_shared_layer_handback_feeds_sparse_mla_like_the_owner() -> None:
    """A shared layer's bound buffers produce the owner's attention exactly."""
    require_b12x()
    device = torch.device("cuda")
    rows, topk, heads = 2, 16, 8
    world = 2
    torch.manual_seed(3)
    k_nope = torch.randn(64, NOPE_DIM, device=device, dtype=torch.bfloat16) / 4
    k_rope = torch.randn(64, ROPE_DIM, device=device, dtype=torch.bfloat16) / 4
    kv = pack_mla_kv_cache_reference(k_nope, k_rope)
    q = torch.randn(rows, heads, HEAD_DIM, device=device, dtype=torch.bfloat16) / 4
    merged = torch.randint(-1, 64, (rows, topk), dtype=torch.int32, device=device)
    cache = IndexCache(
        hy4_group_cell_layers(1),
        device=device,
        topk=topk,
        max_q_rows=rows,
        index_pages_per_layer=0,
    )
    selected, counts = cache.handback(1, merged, world_size=world, rank=0)
    lengths = torch.full((rows,), 64, dtype=torch.int32, device=device)

    owner_out, owner_lse = _run_sparse_mla(
        q, kv, selected, lengths, counts, rows=rows, heads=heads, width=topk
    )
    shared_sel, shared_counts = cache.selected(2), cache.counts(2)
    shared_out, shared_lse = _run_sparse_mla(
        q, kv, shared_sel, lengths, shared_counts, rows=rows, heads=heads, width=topk
    )
    torch.testing.assert_close(shared_out, owner_out, rtol=0.0, atol=0.0)
    torch.testing.assert_close(shared_lse, owner_lse, rtol=0.0, atol=0.0)
