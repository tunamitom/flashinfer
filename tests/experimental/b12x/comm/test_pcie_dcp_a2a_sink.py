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
            attention_sink=sink.float(),
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
            stacked_out.float(), stacked_lse.float(), rank
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
            )
            per_rank_out.append(out_r)
            per_rank_lse.append(lse_r)
        mutated = torch.stack(per_rank_out)
        mutated_lse = torch.stack(per_rank_lse)
        # ... then merge without any post-merge step.
        merged = lse_reduce_scatter_reference(
            mutated, mutated_lse, rank, is_lse_base_on_e=True
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
        out.clone(), lse, sink
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
        out.clone(), lse / LN2, sink, is_lse_base_on_e=False
    )
    torch.testing.assert_close(folded2_out.double(), expected_out, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        folded2_lse.double(), expected_lse / LN2, rtol=1e-6, atol=1e-6
    )

    # Empty rows fold to the sink and a zero output.
    empty_out = torch.zeros(1, heads, head_dim, dtype=torch.float32)
    empty_lse = torch.full((1, heads), -float("inf"), dtype=torch.float32)
    folded_out, folded_lse = apply_attention_sink_after_merge(
        empty_out.clone(), empty_lse, sink
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
        apply_attention_sink_after_merge(out, lse, torch.zeros(3))
    with pytest.raises(ValueError, match="merged_lse must match"):
        apply_attention_sink_after_merge(out, torch.zeros(2, 5), torch.zeros(4))


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
        # only its local shard's slice after the merge.
        sink_full = torch.linspace(2.0, 8.0, TOTAL_HEADS, device=device)

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
                attention_sink=sink_full.float(),
                return_lse=True,
            )
            expected = expected.to(torch.bfloat16)

            actual, actual_lse = channel.lse_reduce_scatter(
                partial_output,
                partial_lse,
                plan=plan,
                attention_sink=sink_full,
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
            off = channel.lse_reduce_scatter(partial_output, partial_lse, plan=plan)
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
                )
                per_rank_out.append(out_r)
                per_rank_lse.append(lse_r)
            mutated = lse_reduce_scatter_reference(
                torch.stack(per_rank_out),
                torch.stack(per_rank_lse),
                rank,
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
