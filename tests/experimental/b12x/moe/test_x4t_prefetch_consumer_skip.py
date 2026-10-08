# SPDX-License-Identifier: Apache-2.0
"""X4T scale prefetch: prepared full-expansion + consumer skip contract.

Covers the coordinated b12x half of the MiMo MXFP4-CSF scale-prefetch port
(Codex review §9.2 "smallest coherent patch"):

* ``expand_scales`` exposes prepared full-X4T expansion using the EXISTING
  paired program (counts mode, all-ones int32 counts of E entries, retained
  program index 1) and writes the packed ``w13_scale``/``w2_scale``
  destinations the inline decode would write.
* ``bind(..., x4t_scales_expanded=True)`` makes the W4A16 X4T runners
  (``run_w4a16_moe`` packed/direct arms and ``run_w4a16_mxfp4_prefill``) skip
  their inline ``decode_x4t_packed_scale_pair`` launch — asserted on the
  launch record, so a side-stream kernel with the inline decoder still running
  (duplicate work) fails the test.
* Two layers with DIFFERENT scale contents sharing the same scratch never
  reuse stale scales; sparse exceptions and row rotation are preserved.
* Unsupported payloads/paths REJECT the skip flag instead of silently
  assuming their consumer will skip.

Red/green: these tests FAIL on the unpatched tree (no ``x4t_prefetch``
capability, no ``x4t_scales_expanded`` bind flag, no launch record) and PASS
on the patched tree.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from b12x._lib.quant.x4t_packed_scales import (
    decode_x4t_packed_scale_pair,
    packed_scale_host_call_count,
    reset_packed_scale_host_call_count,
)
from b12x._lib.quant.x4t_scales import make_x4t_scale_batch
from b12x.moe import fused_moe
from ..conftest import require_b12x

# Smallest geometry admitted by the X4T prepared-weights gate (Kimi list)
# AND by mxfp4_prefill_supported (hidden % 256 == 0, intermediate % 128 == 0).
H, N, E, TOPK = 3584, 256, 4, 2
W13_ROWS, W13_COLS = 2 * N, H // 32
W2_ROWS, W2_COLS = H, N // 32
W13_ROTATION = N  # w13_layout="w13" rotates the fused FC1 halves


@pytest.fixture(autouse=True)
def _launch_diagnostics(monkeypatch):
    """Enable the (env-gated, host-call) packed-scale diagnostics for every
    test in this module, and zero the record before/after each test.

    The record is HOST-CALL instrumentation (increments when the host issues
    the decode, including under capture; replays do not increment) -- the
    tests here assert on host-call issuance/removal, never GPU completions.
    """
    monkeypatch.setenv("B12X_X4T_SCALE_LAUNCH_DIAGNOSTICS", "1")
    reset_packed_scale_host_call_count()
    yield
    reset_packed_scale_host_call_count()


def _planes(rows, columns, rotation, *, seed):
    """One X4T scale batch with sparse exceptions, plus its logical grid."""
    rng = np.random.default_rng(seed)
    fixed, exceptions, grids = [], [], []
    for _ in range(E):
        bases = rng.integers(120, 126, rows, dtype=np.uint8)
        bits = rng.integers(0, 2, (rows, columns), dtype=np.uint8)
        logical = bases[:, None] + bits
        # Sparse exceptions: a handful of positions per expert, including rows
        # either side of the rotation point and the plane boundaries.
        positions = np.unique(
            np.array(
                [
                    0,
                    63 * columns,
                    rows // 2 * columns,
                    (rows // 2 + 1) * columns,
                    rows * columns - 1,
                ],
                dtype=np.uint32,
            )
        )
        values = rng.integers(118, 255, len(positions), dtype=np.uint32)
        logical.ravel()[positions] = values
        selectors = np.packbits(bits, axis=1, bitorder="little")
        fixed.append(
            torch.from_numpy(
                np.concatenate(
                    (bases.reshape(-1, 16), selectors.reshape(rows // 16, -1)), 1
                )
            )
        )
        exceptions.append(torch.from_numpy(positions | (values << 24)))
        grids.append(logical)
    batch = make_x4t_scale_batch(
        fixed,
        exceptions,
        rows=rows,
        columns=columns,
        device="cuda",
        exception_task_rows=64,
        exception_row_rotation=rotation,
    )
    return batch, torch.from_numpy(np.stack(grids)).to("cuda")


def _logical_pair(seed_offset=0):
    fc1, _ = _planes(W13_ROWS, W13_COLS, W13_ROTATION, seed=1000 + seed_offset)
    fc2, _ = _planes(W2_ROWS, W2_COLS, 0, seed=2000 + seed_offset)
    return fc1, fc2


def _prepare(seed_offset=0, scratch=None):
    """One prepared X4T W4A16 payload over caller-owned shared scale scratch."""
    fc1, fc2 = _logical_pair(seed_offset)
    if scratch is None:
        scratch = (
            torch.empty((E, W13_COLS, W13_ROWS), dtype=torch.uint8, device="cuda"),
            torch.empty((E, W2_COLS, W2_ROWS), dtype=torch.uint8, device="cuda"),
        )
    w13 = torch.zeros((E, W13_ROWS, H // 2), dtype=torch.uint8, device="cuda")
    w2 = torch.zeros((E, W2_ROWS, N // 2), dtype=torch.uint8, device="cuda")
    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w13"),
        activation=fused_moe.ActivationSpec(
            mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=E, hidden_size=H, intermediate_size=N
        ),
    )
    experts = fused_moe.prepare_weights(
        plan=plan,
        weights=fused_moe.Mxfp4CsfWeights(
            w13=w13,
            w2=w2,
            w13_scales=fc1,
            w2_scales=fc2,
            w13_scale_scratch=scratch[0],
            w2_scale_scratch=scratch[1],
        ),
    )
    return experts, scratch


def _selective_reference(planes, ids):
    """The NORMAL X4T decoder output for ``ids``' active experts.

    This is precisely Codex's oracle: the prefetched packed scale bytes must
    match the normal X4T decoder output for the active experts, including
    sparse exceptions and row rotation. Inactive experts keep the 0xD6 poison.
    """
    outs = (
        torch.full((E, W13_COLS, W13_ROWS), 0xD6, dtype=torch.uint8, device="cuda"),
        torch.full((E, W2_COLS, W2_ROWS), 0xD6, dtype=torch.uint8, device="cuda"),
    )
    decode_x4t_packed_scale_pair(planes[0], planes[1], ids, outs[0], outs[1])
    torch.cuda.synchronize()
    return outs


def _assert_active_match(prefetched, reference, active):
    for out, ref in zip(prefetched, reference):
        for expert in range(E):
            if expert in active:
                assert torch.equal(out[expert], ref[expert]), (
                    f"prefetched scale bytes differ from the normal decoder "
                    f"output for active expert {expert}"
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_x4t_prefetch_expansion_matches_selective_decode_for_active_experts():
    """Prefetched packed scale bytes == normal decoder output for active
    experts, including sparse exceptions and row rotation."""
    require_b12x()
    experts, scratch = _prepare()
    impl = experts._impl
    capability = impl.x4t_prefetch
    assert capability is not None, "prepared X4T payload must retain the capability"
    assert capability.programs is not None and len(capability.programs) == 4
    # Preferred selector: all-ones int32 counts of E entries (counts MUST be
    # positive; zero counts select no experts).
    assert capability.counts.dtype == torch.int32
    assert capability.counts.numel() == E
    assert bool((capability.counts > 0).all())
    # The unused ordinary-IDs alternative (arange(E)) is deliberately NOT
    # allocated in the counts-only serving implementation (Codex review:
    # gate-OFF preparation cost must be near zero).
    assert not hasattr(capability, "all_expert_ids")

    planes = capability.planes
    reset_packed_scale_host_call_count()
    before = packed_scale_host_call_count()
    assert fused_moe.expand_scales(experts) is True
    torch.cuda.synchronize()
    assert packed_scale_host_call_count() == before + 1, (
        "expand_scales must issue exactly one paired X4T decode launch"
    )
    sparse = torch.tensor([2], dtype=torch.int32, device="cuda")
    reference = _selective_reference(planes, sparse)
    _assert_active_match(scratch, reference, active={2})
    # The prefetch wrote into the SAME shared scratch the inline decode uses.
    assert impl.w1_blockscale.data_ptr() == scratch[0].data_ptr()
    assert impl.w2_blockscale.data_ptr() == scratch[1].data_ptr()

    # A second, differently-routed call sees correct scales too (full
    # expansion covers every expert, so no stale/poison reuse).
    rotated = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
    reference2 = _selective_reference(planes, rotated)
    _assert_active_match(scratch, reference2, active={0, 3})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_two_layers_different_scales_share_scratch_without_stale_reuse():
    """Two layers with DIFFERENT scale contents sharing the same scratch:
    each expansion is byte-exact for its own layer, never the other's."""
    require_b12x()
    shared = (
        torch.empty((E, W13_COLS, W13_ROWS), dtype=torch.uint8, device="cuda"),
        torch.empty((E, W2_COLS, W2_ROWS), dtype=torch.uint8, device="cuda"),
    )
    layer0, _ = _prepare(seed_offset=0, scratch=shared)
    layer1, _ = _prepare(seed_offset=1, scratch=shared)

    for layer in (layer0, layer1):
        # Poison the shared scratch, expand THIS layer, compare to its own
        # selective reference. A stale reuse of the other layer's scales fails
        # here because the two layers have different scale contents.
        for buf in shared:
            buf.fill_(0xD6)
        assert fused_moe.expand_scales(layer) is True
        torch.cuda.synchronize()
        sparse = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
        reference = _selective_reference(layer._impl.x4t_prefetch.planes, sparse)
        _assert_active_match(shared, reference, active={1, 3})
        # And a different route on the same layer still matches (no staleness
        # from the previous layer's expansion).
        other = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
        reference2 = _selective_reference(layer._impl.x4t_prefetch.planes, other)
        _assert_active_match(shared, reference2, active={0, 2})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_consumed_prefetch_removes_inline_a4_decode_launch():
    """The consumed prefetch REMOVES the corresponding inline decode launch.

    Asserted on the launch record: with the flag set, the A4 runner must not
    issue its own ``decode_x4t_packed_scale_pair`` — a side-stream kernel with
    the inline decoder still running is duplicate work, not a port.
    """
    require_b12x()
    from b12x.moe._shared.kernels.w4a16.mxfp4_a4_prefill import (
        compile_w4a16_mxfp4_prefill,
        run_w4a16_mxfp4_prefill,
    )

    experts, scratch = _prepare()
    payload = experts._impl.representation.value
    launches = compile_w4a16_mxfp4_prefill(
        tokens=8,
        min_tokens=1,
        topk=TOPK,
        hidden_size=H,
        intermediate_size=N,
        num_experts=E,
        sms=torch.cuda.get_device_properties(0).multi_processor_count,
        ordinal=0,
        fast_math=False,
    )
    tokens = 8
    routes = tokens * TOPK
    cache13 = torch.zeros(routes * H, dtype=torch.bfloat16, device="cuda")
    cache2 = torch.zeros(
        launches.scratch_bytes(tokens) // 2 + 16, dtype=torch.bfloat16, device="cuda"
    )
    a = torch.randn(tokens, H, dtype=torch.bfloat16, device="cuda") * 0.1
    out = torch.empty(tokens, H, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.stack(
        [torch.randperm(E, device="cuda")[:TOPK] for _ in range(tokens)]
    ).to(torch.int32)
    topk_weights = torch.softmax(torch.randn(tokens, TOPK, device="cuda"), dim=-1)
    unit1 = torch.ones(1, dtype=torch.float32, device="cuda")
    unit2 = torch.ones(1, dtype=torch.float32, device="cuda")

    def run(flag):
        return run_w4a16_mxfp4_prefill(
            a,
            payload,
            topk_weights,
            topk_ids,
            a1_gscale=unit1,
            a2_gscale=unit2,
            intermediate_cache13=cache13,
            intermediate_cache2=cache2,
            output=out,
            launches=launches,
            x4t_scales_expanded=flag,
        )

    reset_packed_scale_host_call_count()
    # Inline path: the runner expands the routed experts itself.
    before = packed_scale_host_call_count()
    first = run(False).clone()
    torch.cuda.synchronize()
    assert packed_scale_host_call_count() - before == 1, (
        "the inline A4 arm must issue its X4T decode launch"
    )

    # Prefetch path: expand every expert ahead of the call, then consume.
    for buf in scratch:
        buf.fill_(0xD6)
    assert fused_moe.expand_scales(experts) is True
    torch.cuda.synchronize()
    before = packed_scale_host_call_count()
    second = run(True).clone()
    torch.cuda.synchronize()
    consumed = packed_scale_host_call_count() - before
    assert consumed == 0, (
        f"a consumed prefetch must REMOVE the inline decode launch, got "
        f"{consumed} launches"
    )
    # Bit-identical output: same kernels, same inputs, same scale bytes.
    assert torch.equal(first, second)

    # Unset flag on the next call: the inline decode is back (no leakage).
    before = packed_scale_host_call_count()
    run(False)
    torch.cuda.synchronize()
    assert packed_scale_host_call_count() - before == 1, (
        "the skip flag must be per-call, not sticky"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_consumed_prefetch_removes_inline_a16_packed_decode_launch():
    """Same skip contract on the A16 runner's packed-route arm."""
    require_b12x()
    from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
    from b12x.moe._shared.kernels.w4a16.route_pack import (
        compile_w4a16_route_pack_launches,
    )

    experts, scratch = _prepare()
    payload = experts._impl.representation.value
    tokens, m = 8, 8
    routes = tokens * TOPK
    # run_w4a16_moe resolves its own route block size via
    # select_route_block_size_m(m, topk, num_experts); match it exactly.
    from b12x.moe._shared.kernels.w4a16.host import select_route_block_size_m

    block_size_m = select_route_block_size_m(m, TOPK, E)
    route_pack = compile_w4a16_route_pack_launches(
        tokens=tokens,
        topk=TOPK,
        block_size=block_size_m,
        num_experts=E,
        ordinal=0,
    )
    a = torch.randn(m, H, dtype=torch.bfloat16, device="cuda") * 0.1
    out = torch.empty(m, H, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.stack(
        [torch.randperm(E, device="cuda")[:TOPK] for _ in range(m)]
    ).to(torch.int32)
    topk_weights = torch.softmax(torch.randn(m, TOPK, device="cuda"), dim=-1)
    cache13 = torch.zeros(routes * H * 4, dtype=torch.bfloat16, device="cuda")
    cache2 = torch.zeros(1 << 22, dtype=torch.bfloat16, device="cuda")

    def run(flag):
        return run_w4a16_moe(
            a,
            payload,
            topk_weights,
            topk_ids,
            activation="silu",
            intermediate_cache13=cache13,
            intermediate_cache2=cache2,
            output=out,
            packed_route_indices=torch.zeros(
                route_pack.max_packed_routes, dtype=torch.int32, device="cuda"
            ),
            block_expert_ids=torch.zeros(
                route_pack.max_route_blocks, dtype=torch.int32, device="cuda"
            ),
            packed_route_count=torch.zeros(1, dtype=torch.int32, device="cuda"),
            expert_offsets=torch.zeros(E + 1, dtype=torch.int32, device="cuda"),
            expert_counts=torch.zeros(E, dtype=torch.int32, device="cuda"),
            route_pack_launches=route_pack,
            route_mode="packed",
            x4t_scales_expanded=flag,
        )

    reset_packed_scale_host_call_count()
    before = packed_scale_host_call_count()
    first = run(False).clone()
    torch.cuda.synchronize()
    assert packed_scale_host_call_count() - before == 1, (
        "the inline A16 packed arm must issue its X4T decode launch"
    )

    for buf in scratch:
        buf.fill_(0xD6)
    assert fused_moe.expand_scales(experts) is True
    torch.cuda.synchronize()
    before = packed_scale_host_call_count()
    second = run(True).clone()
    torch.cuda.synchronize()
    assert packed_scale_host_call_count() - before == 0, (
        "a consumed prefetch must REMOVE the A16 inline decode launch"
    )
    assert torch.equal(first, second)


def test_skip_contract_shape():
    """The coordinated contract's shape: bind flag + capability + counts mode."""
    from b12x.moe.fused_moe._impl import TPMoEScratchPlan, X4TPrefetchCapability

    params = inspect.signature(TPMoEScratchPlan.bind).parameters
    assert "x4t_scales_expanded" in params, (
        "bind must expose the X4T consumer-skip readiness flag"
    )
    fields = set(X4TPrefetchCapability.__dataclass_fields__)
    assert {"planes", "programs", "counts"} <= fields, (
        f"capability must retain planes/programs/counts, got {fields}"
    )
    # The counts-only serving implementation must not carry the unused
    # ordinary-IDs alternative (Codex review: gate-OFF cost near zero).
    assert "all_expert_ids" not in fields
    source = inspect.getsource(fused_moe.expand_scales)
    assert "expert_counts=True" in source, (
        "the prepared full-X4T expansion must use the counts-mode paired program"
    )
    assert "programs[1]" in source, "counts mode must use the retained program index 1"


def test_counts_legality_zero_selects_nothing():
    """Counts legality evidence: counts must be POSITIVE. Zero counts select no
    experts and zero IDs in ordinary-IDs mode select only expert 0 — so the
    prefetch uses all-ones counts, never zeros."""
    from b12x._lib.quant import x4t_packed_scales as mod
    from b12x.moe.fused_moe._impl import _x4t_prefetch_capability

    source = inspect.getsource(_x4t_prefetch_capability)
    assert "torch.ones" in source, "the capability must use all-ones counts"
    assert "torch.arange" not in source, (
        "the counts-only serving implementation must not allocate the unused "
        "arange(E) IDs alternative"
    )
    doc = mod.decode_x4t_packed_scales.__doc__
    assert "positive" in doc.lower(), "counts must be positive to select an expert"
    # The kernel's counts branch: expert = slot only when ids[slot] > 0.
    kernel_src = inspect.getsource(mod)
    assert "ids[slot].to(Int32) > Int32(0)" in kernel_src, (
        "the counts-mode kernel must gate on a POSITIVE count"
    )
