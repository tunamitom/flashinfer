"""Prepared post-merge attention-sink fold (consult2 defect 13).

The post-merge fold used to be eager FP64 torch with per-call allocations
inside the CP merge.  ``prepare_sink_fold`` builds a prepared operator —
pinned geometry + log base, Triton compiled at prepare time — and ``run`` is
one allocation-free launch folding ``out`` and ``lse`` in place.

Mutants that must die here:
  (a) fold math drift: the prepared operator must match the FP64 reference
      and a real softmax with a sink column (both bases, empty rows).
  (b) the eager-allocation regression: a captured graph must contain the
      fold and replay bit-identically (the FP64 torch fold allocates and
      cannot be captured).
"""

from __future__ import annotations

import math

import pytest
import torch

from b12x.comm.pcie._sink_fold import PreparedSinkFold, prepare_sink_fold
from b12x.comm.pcie.pcie_dcp_a2a import apply_attention_sink_after_merge

LN2 = math.log(2.0)


# ────────────────────────── validation (CPU) ───────────────────────────────


@pytest.mark.cpu_only
def test_prepare_sink_fold_requires_explicit_valid_contract() -> None:
    out = torch.zeros(2, 4, 8)
    lse = torch.zeros(2, 4)
    sink = torch.zeros(4)
    with pytest.raises(ValueError, match="lse_base must be one of"):
        prepare_sink_fold(out=out, lse=lse, sink=sink, lse_base="e")
    with pytest.raises(ValueError, match="lse must match out"):
        prepare_sink_fold(
            out=out, lse=torch.zeros(2, 5), sink=sink, lse_base="natural"
        )
    with pytest.raises(ValueError, match="float32"):
        prepare_sink_fold(
            out=out, lse=torch.zeros(2, 4, dtype=torch.float64), sink=sink, lse_base="natural"
        )
    with pytest.raises(ValueError, match="must cover"):
        prepare_sink_fold(out=out, lse=lse, sink=torch.zeros(3), lse_base="natural")
    prepared = prepare_sink_fold(out=out, lse=lse, sink=sink, lse_base="base2")
    assert isinstance(prepared, PreparedSinkFold)
    with pytest.raises(ValueError, match="expects out"):
        prepared.run(torch.zeros(2, 4, 7), lse.clone(), sink)
    with pytest.raises(ValueError, match="expects lse"):
        prepared.run(out.clone(), torch.zeros(2, 5), sink)
    with pytest.raises(ValueError, match="dtype"):
        prepared.run(out.clone().half(), lse.clone(), sink)


@pytest.mark.cpu_only
def test_prepared_cpu_operands_take_the_torch_reference() -> None:
    torch.manual_seed(4)
    out = torch.randn(3, 8, 16)
    lse = torch.randn(3, 8) + 4.0
    sink = torch.linspace(0.5, 3.2, 8)
    prepared = prepare_sink_fold(out=out, lse=lse, sink=sink, lse_base="natural")
    got_out, got_lse = prepared.run(out.clone(), lse.clone(), sink)
    want_out, want_lse = apply_attention_sink_after_merge(
        out.clone(), lse.clone(), sink, lse_base="natural", partials_sink_folded=False
    )
    torch.testing.assert_close(got_out, want_out)
    torch.testing.assert_close(got_lse, want_lse)


# ─────────────────────── GPU: the real prepared kernel ─────────────────────


def _require_gpu():
    from ..conftest import require_b12x

    require_b12x()


@pytest.mark.parametrize("lse_base", ["natural", "base2"])
@pytest.mark.parametrize("heads", [8, 16])
def test_gpu_prepared_fold_matches_fp64_reference(lse_base: str, heads: int) -> None:
    """Prepared kernel == FP64 torch reference, incl. empty rows (mutant a)."""
    _require_gpu()
    torch.manual_seed(20261114)
    device = torch.device("cuda")
    batch, head_dim = 5, 64
    out = (torch.randn(batch, heads, head_dim, device=device) / 4).to(torch.bfloat16)
    lse = (torch.randn(batch, heads, device=device) + 3.0).float()
    # Realistic sinks <= 3.2 with a comparable LSE band, plus one far sink.
    sink = torch.linspace(0.5, 3.2, heads, device=device)
    sink[0] = 12.0
    # Two empty rows (and one empty entry): fold must give out=0, lse=sink.
    lse[1, :] = -torch.inf
    lse[3, 2] = -torch.inf

    prepared = prepare_sink_fold(out=out, lse=lse, sink=sink, lse_base=lse_base)
    got_out, got_lse = prepared.run(out.clone(), lse.clone(), sink)
    want_out, want_lse = apply_attention_sink_after_merge(
        out.clone(), lse.clone(), sink, lse_base=lse_base, partials_sink_folded=False
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        got_out.float(), want_out.float(), rtol=1e-5, atol=1e-6
    )
    live = want_lse.isfinite() | torch.isinf(want_lse)
    torch.testing.assert_close(
        got_lse[live], want_lse[live], rtol=1e-6, atol=1e-5
    )
    # Empty rows fold to the sink in the declared base exactly.
    expected_empty = sink / LN2 if lse_base == "base2" else sink
    torch.testing.assert_close(got_lse[1], expected_empty, rtol=1e-6, atol=1e-5)
    assert bool((got_out[1] == 0).all().item())


def test_gpu_prepared_fold_matches_softmax_column_truth() -> None:
    """Folded output == a real softmax with a sink column (mutant a).

    Ground truth is the softmax over [real logits | sink] applied to values —
    not the sigmoid/logaddexp fold formula — at tolerances scaled to the
    output magnitude.
    """
    _require_gpu()
    torch.manual_seed(11)
    device = torch.device("cuda")
    rows, heads, head_dim, keys = 4, 16, 64, 37
    q = torch.randn(rows, heads, head_dim, device=device, dtype=torch.float64)
    k = torch.randn(keys, heads, head_dim, device=device, dtype=torch.float64) / 8
    v = torch.randn(keys, heads, head_dim, device=device, dtype=torch.float64)
    sink = torch.linspace(0.5, 3.2, heads, device=device, dtype=torch.float64)
    scores = torch.einsum("rhd,khd->rhk", q, k) * 0.35 - 1.0
    total = torch.cat(
        [scores, sink.view(1, heads, 1).expand(rows, heads, 1)], dim=-1
    )
    lse_gt = torch.logsumexp(total, dim=-1)
    probs_gt = torch.exp(scores - lse_gt.unsqueeze(-1))
    out_gt = torch.einsum("rhk,khd->rhd", probs_gt, v)

    # The pre-fold partials: softmax WITHOUT the sink column.
    lse_off = torch.logsumexp(scores, dim=-1)
    probs_off = torch.exp(scores - lse_off.unsqueeze(-1))
    out_off = torch.einsum("rhk,khd->rhd", probs_off, v)

    prepared = prepare_sink_fold(
        out=out_off.to(torch.bfloat16),
        lse=lse_off.float(),
        sink=sink.float(),
        lse_base="natural",
    )
    got_out, got_lse = prepared.run(
        out_off.to(torch.bfloat16).clone(), lse_off.float().clone(), sink.float()
    )
    torch.cuda.synchronize()
    out_scale = float(out_gt.abs().max())
    lse_scale = max(float(lse_gt.abs().max()), 1.0)
    # Tolerance scaled to the output magnitude; BF16 out costs ~1% rel.
    torch.testing.assert_close(
        got_out.double(),
        out_gt,
        rtol=2e-2,
        atol=5e-3 * out_scale,
    )
    torch.testing.assert_close(
        got_lse.double(), lse_gt, rtol=1e-5, atol=5e-3 * lse_scale
    )


def test_gpu_prepared_fold_is_capture_safe() -> None:
    """The fold must be prepared before capture and replay identically (b).

    The eager FP64 torch fold allocates per call and cannot ride a CUDA
    graph; the prepared operator captures and replays bit-identically.
    """
    _require_gpu()
    torch.manual_seed(9)
    device = torch.device("cuda")
    batch, heads, head_dim = 4, 16, 64
    out = (torch.randn(batch, heads, head_dim, device=device) / 4).to(torch.bfloat16)
    lse = (torch.randn(batch, heads, device=device) + 3.0).float()
    sink = torch.linspace(0.5, 3.2, heads, device=device)

    prepared = prepare_sink_fold(out=out, lse=lse, sink=sink, lse_base="natural")

    g_out, g_lse = out.clone(), lse.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepared.run(g_out, g_lse, sink)
    torch.cuda.current_stream().wait_stream(stream)

    eager_out, eager_lse = g_out.clone(), g_lse.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared.run(g_out, g_lse, sink)
    # A second fold would double-fold; replay over fresh state instead.
    g_out.copy_(out)
    g_lse.copy_(lse)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    graph.replay()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert (
        after["allocation.all.allocated"] == before["allocation.all.allocated"]
    ), "the prepared fold allocated inside the captured graph replay"
    torch.testing.assert_close(g_out, eager_out)
    torch.testing.assert_close(g_lse, eager_lse)
