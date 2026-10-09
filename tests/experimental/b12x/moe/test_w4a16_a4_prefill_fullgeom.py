# SPDX-License-Identifier: Apache-2.0
"""Full-GLM-geometry execution for the A4 prefill admission fixes.

The admission suite binds at capacity 8192 but never executes; the execution
and CSF/graph cases run at the small fixture (E=8, H=512, topk=2, live up to
1537). These tests close that gap at the serving geometry
(E=256, H=6144, I=256, topk=8), through the patched planner with NO sizing
override: full- and near-full-chunk execution for both CSF readers, output
parity between the readers on identical operands, finite/nonempty results, a
replay check with changed inputs and poisoned reusable scratch, and an
explicit A16 execution control at the same geometry.

GPU tests skip without a consumer-Blackwell device.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from .test_w4a16_a4_prefill import (
    _bind,
    _env,
    _inputs,
    _make_case,
    _plan,
    _prepare,
    _scales,
)

# Serving geometry (GLM-5.3, one TP8 rank).
E, H, I, TOPK = 256, 6144, 256, 8
CAPACITY = 8192


def _fixture(monkeypatch, *, csf, seed=31):
    """Calibrated prepared weights + capacity-8192 plan + fresh scratch."""
    _env(monkeypatch, True, 1)
    case = _make_case(seed, I, H, E, TOPK)
    a1g, a2g = _scales(case)
    experts = _prepare(case, a1g, a2g, csf=csf)
    assert experts._impl.a4_prefill_scales
    xp = _plan(experts, CAPACITY, top_k=TOPK)
    dev = torch.device("cuda")
    scratch = tuple(
        torch.empty(s.shape, dtype=s.dtype, device=dev) for s in xp.scratch_specs()
    )
    return case, experts, xp, scratch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("csf", [False, True])
def test_a4_full_geometry_execution_inline_vs_expanded(csf, monkeypatch):
    """Full (8192) and near-full (8167) execution through the patched
    planner, no sizing override. For CSF weights the inline and explicitly
    expanded readers must agree bit-exact on identical operands; the CSF
    reader descriptor must reflect the requested expansion; every output is
    finite and nonempty."""
    require_b12x()
    from b12x.moe import fused_moe as moe

    case, experts, xp, scratch = _fixture(monkeypatch, csf=csf)
    outputs = {}
    for tokens in (CAPACITY, CAPACITY - 25):
        x, ids, wts = _inputs(tokens, tokens, H, E, TOPK)
        expansions = (False, True) if csf else (False,)
        for expanded in expansions:
            if csf and expanded and tokens > 1536:
                # scales_expanded=True is a caller contract: the expansion
                # must ALREADY be ordered before the bound call.
                assert moe.expand_scales(experts)
            out = torch.empty_like(x)
            binding = _bind(
                xp,
                x,
                ids,
                wts,
                out,
                scratch,
                a4_prefill=True,
                scales_expanded=expanded,
            )
            launch = binding.a4_prefill_launches
            assert launch is not None, (csf, tokens, expanded)
            if csf:
                consume_expanded = expanded and tokens > 1536
                # the bound reader descriptor reflects the requested expansion
                assert (launch.csf_inline_words is None) is consume_expanded, (
                    tokens,
                    expanded,
                    launch.csf_inline_words,
                )
            moe.run(binding=binding)
            torch.cuda.synchronize()
            assert torch.isfinite(out.float()).all(), (csf, tokens, expanded)
            assert out.float().abs().max() > 0, (csf, tokens, expanded)
            outputs[(tokens, expanded)] = out
    if csf:
        for tokens in (CAPACITY, CAPACITY - 25):
            torch.testing.assert_close(
                outputs[(tokens, False)],
                outputs[(tokens, True)],
                rtol=0,
                atol=0,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a4_full_geometry_graph_replay_with_poisoned_scratch(monkeypatch):
    """CUDA-graph capture at full capacity through the patched planner:
    changed inputs must change the output through the same graph, and replay
    after poisoning the reusable scratch reproduces the eager result exactly
    with no hidden allocations or pointer moves."""
    require_b12x()
    from b12x.moe import fused_moe as moe
    from b12x._lib.runtime_control import kernel_resolution_guard

    case, experts, xp, scratch = _fixture(monkeypatch, csf=True)
    tokens = CAPACITY
    x, ids, wts = _inputs(tokens, 7, H, E, TOPK)
    out = torch.empty_like(x)
    binding = _bind(xp, x, ids, wts, out, scratch, a4_prefill=True)
    assert binding.a4_prefill_launches is not None

    moe.run(binding=binding)
    torch.cuda.synchronize()
    eager = out.clone()

    changed_x, changed_ids, changed_wts = _inputs(tokens, 900 + tokens, H, E, TOPK)
    graph = torch.cuda.CUDAGraph()
    pointers = tuple(t.data_ptr() for t in (*scratch, out))
    try:
        with kernel_resolution_guard("A4 full-geometry launches"):
            with torch.cuda.graph(graph):
                moe.run(binding=binding)
            x.copy_(changed_x)
            ids.copy_(changed_ids)
            wts.copy_(changed_wts)
            moe.run(binding=binding)
            torch.cuda.synchronize()
            changed_eager = out.clone()
            assert not torch.equal(eager, changed_eager)

            for buffer in scratch:
                buffer.fill_(77)
            out.fill_(float("nan"))
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert torch.cuda.max_memory_allocated() == allocated
            assert pointers == tuple(t.data_ptr() for t in (*scratch, out))
            torch.testing.assert_close(out, changed_eager, rtol=0, atol=0)
    finally:
        graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a4_full_geometry_explicit_a16_control(monkeypatch):
    """Explicit A16 at the same geometry and capacity (patched planner):
    binds the A16 fused launch — never A4 — executes, stays finite and
    nonzero. The A4 reservation changes nothing about A16 behavior."""
    require_b12x()
    from b12x.moe import fused_moe as moe

    case, experts, xp, scratch = _fixture(monkeypatch, csf=True)
    tokens = CAPACITY
    x, ids, wts = _inputs(tokens, 11, H, E, TOPK)
    out = torch.empty_like(x)
    binding = _bind(xp, x, ids, wts, out, scratch, a4_prefill=False)
    assert binding.a4_prefill_launches is None
    assert binding.fused_launch is not None
    moe.run(binding=binding)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    assert out.float().abs().max() > 0
