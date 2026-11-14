"""Context-parallel attention-sink: applied EXACTLY ONCE after the merge.

Under context parallelism every rank folds its context shard WITHOUT the
sink; the cross-rank LSE reduction has no sink input, so the sink is applied
once after the merge (``out *= sigmoid(lse - sink); lse = logaddexp(lse,
sink)``). These tests pin the "exactly once" semantics against an absolute
ground truth (a single softmax over the union of all shards PLUS one sink row
per query head — the decode-epilogue semantics).

Mutants that must die here:
  (a) sink disabled: merged output must differ from the sunk ground truth.
  (b) sink applied per-rank: folding each shard's partial before the merge
      (one phantom sink row per rank) must NOT match the ground truth.

CPU tests exercise the reference at 2/8/16 ranks now. GPU tests run the real
``PCIeDCPA2A.lse_reduce_scatter`` at 2 ranks on the switch-D scratch cards;
8/16-rank runs are parameterized but deferred (cards are not free).
"""

from __future__ import annotations

import math
import os

import pytest
import torch

from b12x.comm.pcie.pcie_dcp_a2a import (
    apply_attention_sink_after_merge,
    lse_reduce_scatter_reference,
)

LN2 = math.log(2.0)
_SHARD_SIZES = [17, 23, 5, 31, 9, 13, 41, 3, 7, 19, 29, 11, 37, 2, 43, 6]


def _shard_attention_case(
    *, world: int, batch: int, heads: int, head_dim: int, shard_sizes, seed: int
):
    """Real attention partials over per-rank context shards (fp64 truth)."""
    torch.manual_seed(seed)
    q = torch.randn(batch, heads, head_dim, dtype=torch.float64)
    # Strong sinks on every head so each rank's slice discriminates the
    # once-folded truth from both mutants (weak sinks round the mutant
    # delta below any sane tolerance).
    sink = torch.linspace(3.0, 5.0, heads, dtype=torch.float64)
    partials_out = []
    partials_lse = []
    all_scores = []
    all_values = []
    for shard in shard_sizes:
        k = torch.randn(shard, heads, head_dim, dtype=torch.float64) / math.sqrt(
            head_dim
        )
        v = torch.randn(shard, heads, head_dim, dtype=torch.float64)
        scores = torch.einsum("bhd,shd->bhs", q, k)  # [batch, heads, shard]
        all_scores.append(scores)
        all_values.append(v)
        lse_r = torch.logsumexp(scores, dim=-1)  # [batch, heads]
        probs = torch.exp(scores - lse_r.unsqueeze(-1))
        out_r = torch.einsum("bhs,shd->bhd", probs, v)
        partials_out.append(out_r)
        partials_lse.append(lse_r)
    stacked_out = torch.stack(partials_out)  # [world, batch, heads, head_dim]
    stacked_lse = torch.stack(partials_lse)  # [world, batch, heads]

    # Absolute ground truth: one softmax over the union of every shard's keys
    # PLUS one sink row per head (the decode-epilogue semantics).
    full_scores = torch.cat(all_scores, dim=-1)
    full_values = torch.cat(all_values, dim=0)
    total = torch.cat(
        [full_scores, sink.view(1, heads, 1).expand(batch, heads, 1)], dim=-1
    )
    lse_gt = torch.logsumexp(total, dim=-1)
    probs_gt = torch.exp(full_scores - lse_gt.unsqueeze(-1))
    out_gt = torch.einsum("bhs,shd->bhd", probs_gt, full_values)
    return stacked_out, stacked_lse, sink, out_gt, lse_gt


@pytest.mark.cpu_only
@pytest.mark.parametrize("world", [2, 8, 16])
def test_reference_sink_applied_exactly_once(world: int) -> None:
    heads = 16
    assert heads % world == 0
    stacked_out, stacked_lse, sink, out_gt, lse_gt = _shard_attention_case(
        world=world,
        batch=2,
        heads=heads,
        head_dim=8,
        shard_sizes=_SHARD_SIZES[:world],
        seed=20261114 + world,
    )
    for rank in range(world):
        merged, merged_lse = lse_reduce_scatter_reference(
            stacked_out.float(),
            stacked_lse.float(),
            rank,
            lse_base="natural",
            attention_sink=sink.float(),
            partials_sink_folded=False,
            return_lse=True,
        )
        head_slice = slice(rank * (heads // world), (rank + 1) * (heads // world))
        torch.testing.assert_close(
            merged.double(), out_gt[:, head_slice], rtol=2e-5, atol=1e-5
        )
        torch.testing.assert_close(
            merged_lse.double(),
            lse_gt[:, head_slice],
            rtol=2e-5,
            atol=1e-5,
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("world", [2, 8, 16])
def test_mutant_sink_disabled_dies(world: int) -> None:
    """(a) sink disabled: the un-sunk merge must differ from the ground truth."""
    heads = 16
    stacked_out, stacked_lse, sink, out_gt, _ = _shard_attention_case(
        world=world,
        batch=2,
        heads=heads,
        head_dim=8,
        shard_sizes=_SHARD_SIZES[:world],
        seed=20261114 + world,
    )
    for rank in range(world):
        merged = lse_reduce_scatter_reference(
            stacked_out.float(), stacked_lse.float(), rank, lse_base="natural"
        )
        head_slice = slice(rank * (heads // world), (rank + 1) * (heads // world))
        assert not torch.allclose(
            merged.double(), out_gt[:, head_slice], rtol=0.02, atol=1e-3
        ), f"rank {rank}: sink-off merge matches the sunk ground truth"


@pytest.mark.cpu_only
@pytest.mark.parametrize("world", [2, 8, 16])
def test_mutant_per_rank_fold_dies(world: int) -> None:
    """(b) sink applied per-rank must NOT match the once-folded ground truth.

    Folding the sink into each shard's partial before the merge adds one
    phantom sink row per rank: the output is over-attenuated and the merged
    LSE is wrong. This is the context-parallel double-application bug.
    """
    heads = 16
    stacked_out, stacked_lse, sink, out_gt, lse_gt = _shard_attention_case(
        world=world,
        batch=2,
        heads=heads,
        head_dim=8,
        shard_sizes=_SHARD_SIZES[:world],
        seed=20261114 + world,
    )
    for rank in range(world):
        # Mutant: fold the sink into EVERY rank's partial first ...
        per_rank_out = []
        per_rank_lse = []
        for r in range(world):
            out_r, lse_r = apply_attention_sink_after_merge(
                stacked_out[r].float().clone(),
                stacked_lse[r].float(),
                sink.float(),
                lse_base="natural",
                partials_sink_folded=False,
            )
            per_rank_out.append(out_r)
            per_rank_lse.append(lse_r)
        mutated = torch.stack(per_rank_out)
        mutated_lse = torch.stack(per_rank_lse)
        # ... then merge without any post-merge step.
        merged = lse_reduce_scatter_reference(
            mutated, mutated_lse, rank, lse_base="natural"
        )
        head_slice = slice(rank * (heads // world), (rank + 1) * (heads // world))
        assert not torch.allclose(
            merged.double(), out_gt[:, head_slice], rtol=0.02, atol=1e-3
        ), f"rank {rank}: per-rank fold passes for the once-folded truth"
        assert not torch.allclose(
            merged.double(), out_gt[:, head_slice], rtol=1e-3, atol=1e-4
        ), f"rank {rank}: per-rank fold is numerically indistinguishable"


@pytest.mark.cpu_only
def test_post_merge_fold_matches_decode_epilogue_form() -> None:
    """The post-merge step is algebraically the decode path's fold."""
    torch.manual_seed(4)
    batch, heads, head_dim = 3, 8, 8
    out = torch.randn(batch, heads, head_dim, dtype=torch.float32)
    lse = torch.randn(batch, heads, dtype=torch.float32) + 6.0
    sink = torch.linspace(-1.0, 8.0, heads, dtype=torch.float32)
    folded_out, folded_lse = apply_attention_sink_after_merge(
        out.clone(), lse, sink, lse_base="natural", partials_sink_folded=False
    )
    factor = torch.sigmoid(lse.double() - sink.double().unsqueeze(0))
    expected_out = out.double() * factor.unsqueeze(-1)
    expected_lse = torch.logaddexp(
        lse.double(), sink.double().unsqueeze(0)
    )
    torch.testing.assert_close(folded_out.double(), expected_out, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        folded_lse.double(), expected_lse, rtol=1e-6, atol=1e-7
    )

    # Base-2 LSE variant gives the same output fold and the base-2 LSE fold.
    folded2_out, folded2_lse = apply_attention_sink_after_merge(
        out.clone(),
        lse / LN2,
        sink,
        lse_base="base2",
        partials_sink_folded=False,
    )
    torch.testing.assert_close(folded2_out.double(), expected_out, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        folded2_lse.double(), expected_lse / LN2, rtol=1e-6, atol=1e-6
    )

    # Empty rows fold to the sink and a zero output.
    empty_out = torch.zeros(1, heads, head_dim, dtype=torch.float32)
    empty_lse = torch.full((1, heads), -float("inf"), dtype=torch.float32)
    folded_out, folded_lse = apply_attention_sink_after_merge(
        empty_out.clone(), empty_lse, sink, lse_base="natural", partials_sink_folded=False
    )
    assert bool((folded_out == 0).all().item())
    torch.testing.assert_close(
        folded_lse.squeeze(0), sink, rtol=1e-6, atol=1e-6
    )


@pytest.mark.cpu_only
def test_post_merge_fold_rejects_head_mismatch() -> None:
    out = torch.zeros(2, 4, 8)
    lse = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="attention_sink must cover"):
        apply_attention_sink_after_merge(
            out, lse, torch.zeros(3), lse_base="natural", partials_sink_folded=False
        )
    with pytest.raises(ValueError, match="merged_lse must match"):
        apply_attention_sink_after_merge(
            out,
            torch.zeros(2, 5),
            torch.zeros(4),
            lse_base="natural",
            partials_sink_folded=False,
        )


# ─────────────────────── defect 10: empty rows are -inf ─────────────────────


@pytest.mark.cpu_only
@pytest.mark.parametrize("lse_base", ["natural", "base2"])
def test_empty_row_reference_returns_neg_inf(lse_base: str) -> None:
    """An all-empty row must return -inf, not a clamp floor (defect 10).

    The in-place ``weight_sum.clamp_min_(1e-10)`` bug turned the empty row's
    0 into 1e-10 and reported ``log(1e-10) = -23.03`` (``log2`` = -33.2 in
    base 2).  Exact -inf is the contract: no rank contributed a finite LSE.
    """
    world, batch, heads, head_dim = 2, 3, 4, 8
    out = torch.randn(world, batch, heads, head_dim, dtype=torch.float64)
    lses = torch.randn(world, batch, heads, dtype=torch.float64)
    # Row 1 is empty on EVERY rank (and one head of row 0 too).
    lses[:, 1, :] = -torch.inf
    lses[:, 0, 2] = -torch.inf
    lses_in = lses if lse_base == "natural" else lses / LN2
    for rank in range(world):
        merged, merged_lse = lse_reduce_scatter_reference(
            out.float(),
            lses_in.float(),
            rank,
            lse_base=lse_base,
            return_lse=True,
        )
        head_slice = slice(rank * (heads // world), (rank + 1) * (heads // world))
        empty = merged_lse[1]  # batch row 1 is empty on EVERY rank
        assert bool(torch.isneginf(empty).all().item()), (
            f"rank {rank} empty rows must be exactly -inf, got {empty.tolist()} "
            f"(a clamp floor near -23.03/-33.2 is the defect-10 bug)"
        )
        # The empty ROW's output must be zero, not garbage.
        assert bool((merged[1] == 0).all().item())
        # Batch row 0 is live except for head 2 (empty on every rank): rank 1's
        # shard holds heads 2..3, so its first entry is -inf and its second
        # finite; rank 0's two entries are finite. Batch row 2 is fully live.
        if rank == 0:
            assert bool(torch.isfinite(merged_lse[0]).all().item())
        else:
            assert bool(torch.isneginf(merged_lse[0, 0]).item())
            assert bool(torch.isfinite(merged_lse[0, 1]).item())
        assert bool(torch.isfinite(merged_lse[2]).all().item())
        # Empty rows folded with a sink collapse to the sink exactly (reported
        # in the declared base).
        sink = torch.linspace(-1.0, 2.0, heads, dtype=torch.float32)
        _, folded_lse = lse_reduce_scatter_reference(
            out.float(),
            lses_in.float(),
            rank,
            lse_base=lse_base,
            attention_sink=sink,
            partials_sink_folded=False,
            return_lse=True,
        )
        expected_sink = sink[head_slice] / LN2 if lse_base == "base2" else sink[head_slice]
        torch.testing.assert_close(folded_lse[1], expected_sink)


# ──────────────────── defect 11: double-fold rejected loudly ─────────────────


@pytest.mark.cpu_only
def test_mutant_double_fold_rejected_loudly() -> None:
    """A plan with a sink plus a merge with a sink must FAIL (defect 11).

    The producing ``sparse_mla`` binding exposes ``sink_folded``; feeding it as
    ``partials_sink_folded=`` to a merge that also folds the sink must raise —
    the double-fold mutant (silently accepting both) is killed here.
    """
    world, batch, heads, head_dim = 2, 2, 4, 8
    out = torch.randn(world, batch, heads, head_dim)
    lses = torch.randn(world, batch, heads)
    sink = torch.zeros(heads)

    # The composition: a sparse_mla plan with a sink (binding.sink_folded=True)
    # feeding a merge with a sink.
    binding_sink_folded = True  # plan(has_attention_sink=True) with sink bound
    with pytest.raises(ValueError, match="double sink fold"):
        lse_reduce_scatter_reference(
            out,
            lses,
            0,
            lse_base="natural",
            attention_sink=sink,
            partials_sink_folded=binding_sink_folded,
        )
    with pytest.raises(ValueError, match="double sink fold"):
        apply_attention_sink_after_merge(
            out[0].clone(),
            lses[0],
            sink,
            lse_base="natural",
            partials_sink_folded=True,
        )
    # Provenance must be DECLARED when the merge folds the sink.
    with pytest.raises(ValueError, match="declare partials_sink_folded"):
        lse_reduce_scatter_reference(
            out, lses, 0, lse_base="natural", attention_sink=sink
        )
    with pytest.raises(ValueError, match="declare partials_sink_folded"):
        apply_attention_sink_after_merge(out[0].clone(), lses[0], sink, lse_base="natural")
    # And a plan with a sink handed to a merge WITHOUT a sink fold is the
    # supported CP shape: pass it through unchanged (the flag still guards).
    merged = lse_reduce_scatter_reference(
        out, lses, 0, lse_base="natural", partials_sink_folded=False
    )
    assert merged.shape == (batch, heads // world, head_dim)


# ───────────────── defect 12: the LSE base is explicit, enforced ────────────


@pytest.mark.cpu_only
def test_lse_base_must_be_stated_explicitly() -> None:
    """Nothing defaults: omitting the base raises (defect 12)."""
    out = torch.zeros(2, 1, 2, 4)
    lses = torch.zeros(2, 1, 2)
    with pytest.raises(ValueError, match="must be stated explicitly"):
        lse_reduce_scatter_reference(out, lses, 0)
    with pytest.raises(ValueError, match="must be stated explicitly"):
        apply_attention_sink_after_merge(
            out[0], lses[0], torch.zeros(2), partials_sink_folded=False
        )
    with pytest.raises(ValueError, match="only one of"):
        lse_reduce_scatter_reference(
            out, lses, 0, lse_base="natural", is_lse_base_on_e=True
        )
    with pytest.raises(ValueError, match="lse_base must be one of"):
        lse_reduce_scatter_reference(out, lses, 0, lse_base="e")
    # The legacy boolean alias is explicit input and still resolves.
    merged = lse_reduce_scatter_reference(out, lses, 0, is_lse_base_on_e=True)
    assert merged.shape == (1, 1, 4)


@pytest.mark.cpu_only
def test_mutant_base_mismatch_dies() -> None:
    """The wrong base re-weights the merge: it must NOT match truth (defect 12).

    Base-2 partial LSEs merged as base-2 match the softmax ground truth; the
    same partials merged as natural logs (the historical silent default) must
    not.  Kills the base-mismatch mutant on both the merge and the fold.
    """
    heads = 8
    stacked_out, stacked_lse, sink, out_gt, lse_gt = _shard_attention_case(
        world=2,
        batch=2,
        heads=heads,
        head_dim=8,
        shard_sizes=_SHARD_SIZES[:2],
        seed=77,
    )
    stacked_lse2 = stacked_lse / LN2  # base-2 partials
    for rank in range(2):
        head_slice = slice(rank * (heads // 2), (rank + 1) * (heads // 2))
        good, good_lse = lse_reduce_scatter_reference(
            stacked_out.float(),
            stacked_lse2.float(),
            rank,
            lse_base="base2",
            attention_sink=sink.float(),
            partials_sink_folded=False,
            return_lse=True,
        )
        torch.testing.assert_close(
            good.double(), out_gt[:, head_slice], rtol=2e-5, atol=1e-5
        )
        torch.testing.assert_close(
            good_lse.double(), lse_gt[:, head_slice] / LN2, rtol=2e-5, atol=1e-5
        )
        # Mismatched base: must NOT match the ground truth.
        bad = lse_reduce_scatter_reference(
            stacked_out.float(),
            stacked_lse2.float(),
            rank,
            lse_base="natural",  # the mismatch mutant
        )
        assert not torch.allclose(
            bad.double(), out_gt[:, head_slice], rtol=1e-3, atol=1e-4
        ), f"rank {rank}: a base-mismatched merge matches ground truth"

        # Fold: the base-2 form folded as base-2 must equal the natural-base
        # fold of the same LSE; the mismatched base must not.
        lse_e = torch.linspace(-3.0, 7.0, 2 * heads).reshape(2, heads) + 1.0
        out_e = torch.randn(2, heads, 8)
        ref_out, ref_lse = apply_attention_sink_after_merge(
            out_e.clone(),
            lse_e.float(),
            sink.float(),
            lse_base="natural",
            partials_sink_folded=False,
        )
        good_out, good_lse = apply_attention_sink_after_merge(
            out_e.clone(),
            (lse_e / LN2).float(),
            sink.float(),
            lse_base="base2",
            partials_sink_folded=False,
        )
        torch.testing.assert_close(good_out.double(), ref_out.double())
        torch.testing.assert_close(
            good_lse.double() * LN2, ref_lse.double(), rtol=1e-6, atol=1e-6
        )
        bad_out, bad_lse = apply_attention_sink_after_merge(
            out_e.clone(),
            (lse_e / LN2).float(),
            sink.float(),
            lse_base="natural",  # the mismatch mutant
            partials_sink_folded=False,
        )
        assert not torch.allclose(
            bad_lse.double() * LN2, ref_lse.double(), rtol=1e-3, atol=1e-3
        ), "a base-mismatched fold matches the correct-base fold"
        assert not torch.allclose(
            bad_out.double(), ref_out.double(), rtol=1e-3, atol=1e-3
        ), "a base-mismatched fold output matches the correct-base fold"


# ──────── defect 14: realistic sinks (<= 3.2) vs comparable LSE ─────────────


@pytest.mark.cpu_only
def test_realistic_sink_regime_16_heads_matches_softmax_column() -> None:
    """Sinks <= 3.2 against a comparable row LSE, 16-head groups (defect 14).

    Hy4's trained sink logits are small (<= 3.2) and the top-k row LSE is of
    the same order once the score scale is realistic; the fold must still be
    visible (every factor strictly inside (0.05, 0.95)) and must match a real
    softmax with a sink column — not the sigmoid/logaddexp formula — at
    tolerances scaled to the output magnitude.
    """
    torch.manual_seed(20261114)
    world, batch, heads, head_dim = 2, 3, 16, 8
    assert heads % world == 0
    # Realistic sinks: small positive learned logits, <= 3.2.
    sink = torch.linspace(0.5, 3.2, heads, dtype=torch.float64)
    q = torch.randn(batch, heads, head_dim, dtype=torch.float64)
    all_scores, all_values, partials_out, partials_lse = [], [], [], []
    for shard in (24, 32):
        k = torch.randn(shard, heads, head_dim, dtype=torch.float64) / math.sqrt(
            head_dim
        )
        v = torch.randn(shard, heads, head_dim, dtype=torch.float64)
        # Score scale chosen so the shard LSE lands near the sink band
        # (~2-3.5 nats) — the COMPARABLE regime where the fold is real work.
        scores = torch.einsum("bhd,shd->bhs", q, k) * 0.35 - 1.0
        all_scores.append(scores)
        all_values.append(v)
        lse_r = torch.logsumexp(scores, dim=-1)
        probs = torch.exp(scores - lse_r.unsqueeze(-1))
        partials_out.append(torch.einsum("bhs,shd->bhd", probs, v))
        partials_lse.append(lse_r)
    stacked_out = torch.stack(partials_out)
    stacked_lse = torch.stack(partials_lse)

    # Ground truth: one softmax over the union of the shards' logits PLUS one
    # sink column per head — a real softmax, not the fold formula.
    full_scores = torch.cat(all_scores, dim=-1)
    full_values = torch.cat(all_values, dim=0)
    total = torch.cat(
        [full_scores, sink.view(1, heads, 1).expand(batch, heads, 1)], dim=-1
    )
    lse_gt = torch.logsumexp(total, dim=-1)
    probs_gt = torch.exp(full_scores - lse_gt.unsqueeze(-1))
    out_gt = torch.einsum("bhs,shd->bhd", probs_gt, full_values)

    # Visibility precondition: the fold must be real work in this regime.
    nosink_lse = torch.logsumexp(full_scores, dim=-1)
    factor = torch.sigmoid(nosink_lse - sink.view(1, heads))
    assert bool(((factor > 0.05) & (factor < 0.95)).all().item()), (
        f"realistic-sink regime is degenerate: factors {factor.flatten()[:8]}"
    )
    assert float(sink.max()) <= 3.2

    for rank in range(world):
        merged, merged_lse = lse_reduce_scatter_reference(
            stacked_out.float(),
            stacked_lse.float(),
            rank,
            lse_base="natural",
            attention_sink=sink.float(),
            partials_sink_folded=False,
            return_lse=True,
        )
        head_slice = slice(rank * (heads // world), (rank + 1) * (heads // world))
        # Tolerances scaled to the OUTPUT MAGNITUDE (the old flat tolerances
        # were ten times the output scale and could not fail).
        out_scale = float(out_gt[:, head_slice].abs().max())
        lse_scale = max(float(lse_gt[:, head_slice].abs().max()), 1.0)
        torch.testing.assert_close(
            merged.double(),
            out_gt[:, head_slice],
            rtol=2e-5,
            atol=5e-3 * out_scale,
        )
        torch.testing.assert_close(
            merged_lse.double(),
            lse_gt[:, head_slice],
            rtol=2e-5,
            atol=5e-3 * lse_scale,
        )


# ───────────────────────────── GPU: the real merge ───────────────────────────

_WORLD_SIZES = (2, 8, 16)
TOTAL_HEADS = 16
HEAD_DIM = 64
MAX_BATCH = 8


def _require_dcp_env():
    if os.getenv("B12X_RUN_PCIE_DCP_A2A_TEST") != "1":
        pytest.skip(
            "set B12X_RUN_PCIE_DCP_A2A_TEST=1 to run PCIe DCP A2A GPU tests"
        )
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")


def _rank_partial(step: int, source_rank: int, world: int, batch: int, device):
    """Deterministic per-rank partial (out, lse) WITHOUT any sink."""
    generator = torch.Generator(device="cpu").manual_seed(90000 * step + source_rank)
    output = (
        torch.randn(batch, TOTAL_HEADS, HEAD_DIM, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        / 4
    )
    lse = torch.randn(batch, TOTAL_HEADS, generator=generator).to(device=device) + 6.0
    # Exercise invalid-contribution handling: one shard contributes nothing.
    if source_rank == 0 and batch > 1:
        lse[1, 3] = -torch.inf
    return output, lse


def _sink_worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist
    import torch.multiprocessing as mp  # noqa: F401  (spawn context)

    from b12x.comm.pcie import _dcp_preparation as preparation
    from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2A
    from b12x.preparation import CollectiveRequirement, PreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    channel = PCIeDCPA2A.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_batch_size=MAX_BATCH,
        total_heads=TOTAL_HEADS,
        head_dim=HEAD_DIM,
    )

    def barrier(key, ranks):
        assert tuple(ranks) == tuple(range(world_size))
        dist.barrier(device_ids=[rank])

    call = dict(
        partial_output=torch.zeros(
            MAX_BATCH, TOTAL_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
        ),
        partial_lse=torch.zeros(MAX_BATCH, TOTAL_HEADS, device=device),
        out=torch.empty(
            MAX_BATCH, channel.heads_per_rank, HEAD_DIM, dtype=torch.bfloat16, device=device
        ),
        lse_base="natural",
    )
    session = PreparationSession(
        device=device,
        autotune=False,
        compile_workers=1,
        collective_barrier=barrier,
    )
    try:
        query = preparation.query_from_runtime(
            channel, surface="DcpAllToAll.lse_reduce_scatter", call=call
        )
        plan = preparation.plan(query, runtime=channel)
        session.prepare(
            (
                plan.request(
                    name="lse-sink",
                    collective=CollectiveRequirement(
                        key="lse-sink", ranks=tuple(range(world_size))
                    ),
                    prepare_call=lambda state: preparation.prepare_call(
                        state, **call
                    ),
                ),
            ),
            coordinator=lambda state: (
                state.ready_collectives[0].key if state.ready_collectives else None
            ),
        )

        torch.manual_seed(31337 + rank)
        # A learned per-head sink vector shared by every rank; each rank folds
        # only its local shard's slice after the merge.  Sink band 6..10: at
        # top-k 2048 the row LSE is ~7.6 nats, so O(1) sinks are invisible and
        # the mutant checks below could not discriminate (see the factor-
        # visibility precondition at each step).
        sink_full = torch.linspace(6.0, 10.0, TOTAL_HEADS, device=device)

        for step, batch in enumerate((1, 4, MAX_BATCH), start=1):
            partial_output = torch.empty(
                batch, TOTAL_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
            )
            partial_lse = torch.empty(batch, TOTAL_HEADS, device=device)
            stacked_out = []
            stacked_lse = []
            for source in range(world_size):
                out_r, lse_r = _rank_partial(step, source, world_size, batch, device)
                stacked_out.append(out_r)
                stacked_lse.append(lse_r)
            partial_output.copy_(stacked_out[rank])
            partial_lse.copy_(stacked_lse[rank])

            # Reference: once-folded ground truth for this rank's head shard.
            expected, expected_lse = lse_reduce_scatter_reference(
                torch.stack(stacked_out).float(),
                torch.stack(stacked_lse).float(),
                rank,
                lse_base="natural",
                attention_sink=sink_full.float(),
                partials_sink_folded=False,
                return_lse=True,
            )
            expected = expected.to(torch.bfloat16)

            # Factor-visibility precondition: the sink-on and sink-off merges
            # (and the per-rank-fold mutant) can only be told apart where the
            # fold factor sigmoid(lse - sink) is visibly below 1.  With sinks
            # in 6..10 and the merged shard LSE near 6-7 nats this holds on
            # essentially every entry; fail loudly if a change re-introduces
            # invisible O(1) sinks.
            nosink_lse = lse_reduce_scatter_reference(
                torch.stack(stacked_out).float(),
                torch.stack(stacked_lse).float(),
                rank,
                lse_base="natural",
                return_lse=True,
            )[1]
            head_slice = slice(
                rank * (TOTAL_HEADS // world_size),
                (rank + 1) * (TOTAL_HEADS // world_size),
            )
            factor = torch.sigmoid(
                nosink_lse.double()
                - sink_full.float()[head_slice].double().unsqueeze(0)
            )
            visibility = 1.0 - factor
            assert float(visibility.max().item()) > 0.3, (
                f"step {step}: sink fold factor is everywhere ~1 "
                f"(max invisibility {float(visibility.max().item()):.4f})"
            )
            assert float((visibility > 5e-2).to(torch.float32).mean().item()) > 0.9, (
                f"step {step}: sink fold is invisible on most rows; "
                "sinks must sit near the row LSE"
            )

            actual, actual_lse = channel.lse_reduce_scatter(
                partial_output,
                partial_lse,
                plan=plan,
                lse_base="natural",
                attention_sink=sink_full,
                partials_sink_folded=False,
                return_lse=True,
            )
            torch.cuda.synchronize(device)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(
                actual_lse,
                expected_lse.float(),
                rtol=0.0,
                atol=0.06,
            )

            # (a) sink disabled: must NOT match the sunk ground truth.
            off = channel.lse_reduce_scatter(
                partial_output, partial_lse, plan=plan, lse_base="natural"
            )
            torch.cuda.synchronize(device)
            assert not torch.allclose(
                off.float(), expected.float(), rtol=0.02, atol=0.02
            ), "sink-off merge matches the sunk truth: the sink is disabled"

            # (b) per-rank fold mutant must NOT match the once-folded truth.
            per_rank_out = []
            per_rank_lse = []
            for r in range(world_size):
                out_r, lse_r = apply_attention_sink_after_merge(
                    stacked_out[r].float().clone(),
                    stacked_lse[r].float(),
                    sink_full.float(),
                    lse_base="natural",
                    partials_sink_folded=False,
                )
                per_rank_out.append(out_r)
                per_rank_lse.append(lse_r)
            mutated = lse_reduce_scatter_reference(
                torch.stack(per_rank_out),
                torch.stack(per_rank_lse),
                rank,
                lse_base="natural",
            ).to(torch.bfloat16)
            assert not torch.allclose(
                mutated.float(), expected.float(), rtol=0.02, atol=0.02
            ), "per-rank sink fold matches the once-folded truth"
            dist.barrier(device_ids=[rank])
    finally:
        session.close()
        channel.close()
        dist.destroy_process_group()


def _import_socket_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_device_sink_applied_exactly_once(world_size: int) -> None:
    _require_dcp_env()
    import torch.multiprocessing as mp

    if world_size != 2:
        pytest.skip(
            f"deferred: {world_size}-rank DCP sink cell needs {world_size} free "
            "cards; switch D currently has 4 scratch GPUs (run at 2 ranks now)"
        )
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    mp.spawn(
        _sink_worker,
        args=(world_size, _import_socket_port()),
        nprocs=world_size,
        join=True,
    )
