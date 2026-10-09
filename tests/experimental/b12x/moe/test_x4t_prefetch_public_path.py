# SPDX-License-Identifier: Apache-2.0
"""X4T scale prefetch: real-weight public-path integration (Codex §8.3).

Focused GPU integration cases the review required before any qualification:

* **Real weights**: nonzero finite-range FP4 nibbles, nontrivial (skewed)
  routes and bounded e8m0 scale exponents -- output equality is sensitive to
  ordinary wrong finite scales, unlike the zero-weight launch-removal tests
  in test_x4t_prefetch_consumer_skip.py.
* **Public path**: the prepared ``plan_weights/prepare_weights`` ->
  ``plan_execution`` -> ``PreparationSession``/``request`` -> ``bind`` ->
  ``run`` pipeline with the skip flag OFF and ON (not just runner keywords).
* **Consumer arms**: the A16 route arms (source-native payload, production
  decode layout -- small-M direct arm and packed/direct route arm) and the
  MXFP4 A4 prefill runner (packed payload, production prefill shape), plus a
  back-to-back mixed decode-then-prefill consumption of ONE shared prefetch
  (the §6.3 superset property; the X4T payload has no per-subcall row split
  -- the decode/prefill distinction is a per-call dispatch at bind).
* **Two-stream overlap**: two layers' actual shared scratch expanded and
  consumed on TWO REAL CUDA streams with events and back-to-back forwards,
  no synchronize between expansion and consumption.
* **Capture + replay**: transition through real CUDA-graph capture AND
  replay with a pending prefetch. Empirical contract recorded here: a
  capture CANNOT adopt an in-flight cross-stream dependency (waiting a
  still-pending side-stream event inside capture raises
  cudaErrorStreamCaptureInvalidated), so the lifecycle drains the pending
  prefetch BEFORE capture begins and the captured graph keeps the inline
  decode (the flag is never declared under capture); replays re-expand and
  stay bit-exact over poisoned scratch.
* **Rejection paths**: ``bind`` rejects the flag for a capability-less
  payload and for a non-W4A16 implementation, by calling bind; the
  kernel-level empty/zero route sets are executed (empty ids = no-op, zero
  counts select nothing).

All binding/capture happens INSIDE the PreparationSession (the session owns
the compiled programs; leaving it releases them). Small geometry only -- no
MiMo checkpoint needed. The expansion byte oracle is covered by
test_x4t_prefetch_consumer_skip.py.
"""

from __future__ import annotations

import os
import subprocess
import sys

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
from b12x.preparation import PreparationSession, PreparedCall
from b12x.preparation.types import require_prepared

from ..conftest import require_b12x

# Kimi-list X4T geometry, admissible for BOTH the source-native (w31) A16
# arms and the packed (w13) MXFP4 A4 prefill runner (hidden % 256 == 0,
# intermediate % 128 == 0).
H, N, E, TOPK = 3584, 256, 4, 2

W13_ROWS, W13_COLS = 2 * N, H // 32  # FC1 scale grid (rows, K/32 columns)
W2_ROWS, W2_COLS = H, N // 32  # FC2 scale grid

# Isolated child program for the tightened negative arm (Codex §10.1): a
# real pending side-stream prefetch, then wait_event(pending) INSIDE
# torch.cuda.graph capture. Empirical contract (pinned on this build,
# CUDA 12.x / torch 2.14): the wait raises torch.AcceleratorError with the
# CUDA root error cudaErrorStreamCaptureIsolation ("dependency created on
# uncaptured work in another stream") and the subsequent capture_end fails
# with cudaErrorStreamCaptureInvalidated. The child prints the observed
# class + both CUDA error names and exits 0 so the parent can assert on the
# EXACT error contract; it is a subprocess because the invalidated capture
# poisons this process's capture machinery for later captures.
_NEGATIVE_ARM_SUBPROCESS = """
import sys

sys.path.insert(0, {root!r})

import torch

from tests.experimental.b12x.moe import test_x4t_prefetch_public_path as T

T.require_b12x()

scratch_pair = T._scratch()
experts = T._prepare(scratch_pair, native=True, scale_seed=7000)
tokens = 8
plan = T._public_plan(experts, tokens)

def body(states):
    scratch = T._state_scratch(states["main"])
    a, ids, weights = T._activations(tokens)
    main = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    moe_bad = torch.cuda.Event()
    pending = torch.cuda.Event()
    reader = torch.empty_like(a)
    T._bind_run(plan, scratch, a=a, ids=ids, weights=weights,
                output=reader, expanded=False)
    moe_bad.record(main)
    with torch.cuda.stream(side):
        side.wait_event(moe_bad)
        assert T.fused_moe.expand_scales(experts) is True
        pending.record(side)  # still pending: never waited, never synced
    graph_bad = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph_bad):
            torch.cuda.current_stream().wait_event(pending)
        print("NEGATIVE_ARM_RESULT=no-exception")
    except Exception as exc:
        import traceback

        cls = type(exc).__name__
        chain = [exc]
        while chain[-1].__cause__ is not None:
            chain.append(chain[-1].__cause__)
        text = "\\n".join(str(e) for e in chain)
        text += "\\n" + traceback.format_exc()
        print(f"NEGATIVE_ARM_RESULT={{cls}}")
        print("NEGATIVE_ARM_TEXT_START")
        print(text)
        print("NEGATIVE_ARM_TEXT_END")
    torch.cuda.synchronize()
    del graph_bad

T._in_session({{"main": plan}}, body, tokens=tokens)
""".format(
    root=os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    )
)


@pytest.fixture(autouse=True)
def _launch_diagnostics(monkeypatch):
    """Enable the env-gated HOST-CALL packed-scale diagnostics for this module."""
    monkeypatch.setenv("B12X_X4T_SCALE_LAUNCH_DIAGNOSTICS", "1")
    reset_packed_scale_host_call_count()
    yield
    reset_packed_scale_host_call_count()


def _planes(rows, columns, rotation, *, seed):
    """One X4T scale batch with sparse exceptions, bounded e8m0 exponents.

    Scale bytes stay in a bounded finite range (exponents 2^-9..2^8 for the
    bases and exceptions alike) so real FP4 weights produce finite nontrivial
    outputs -- the review's "bounded scale exponents" requirement. Zero
    weights or extreme exponents could not detect wrong finite scales.
    """
    rng = np.random.default_rng(seed)
    fixed, exceptions = [], []
    for _ in range(E):
        bases = rng.integers(118, 127, rows, dtype=np.uint8)  # 2^-9 .. 2^-1
        bits = rng.integers(0, 2, (rows, columns), dtype=np.uint8)
        logical = bases[:, None] + bits
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
        # Bounded exceptions too: exponents up to 2^8 keep the dot products
        # finite in bf16 while differing from every base value.
        values = rng.integers(128, 136, len(positions), dtype=np.uint32)
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
    return make_x4t_scale_batch(
        fixed,
        exceptions,
        rows=rows,
        columns=columns,
        device="cuda",
        exception_task_rows=64,
        exception_row_rotation=rotation,
    )


def _weights(seed=7):
    """Nonzero FP4 nibbles: every byte carries varied nonzero values."""
    rng = np.random.default_rng(seed)
    w13 = torch.from_numpy(
        rng.integers(1, 256, (E, W13_ROWS, H // 2), dtype=np.uint8)
    ).to("cuda")
    w2 = torch.from_numpy(
        rng.integers(1, 256, (E, W2_ROWS, N // 2), dtype=np.uint8)
    ).to("cuda")
    return w13, w2


def _prepare(scratch, *, native, seed=7, scale_seed=1000):
    """One prepared X4T W4A16 payload over caller-owned shared scale scratch.

    ``native=True`` plans the source-native (w31) payload -- the layout the
    small-M direct A16 arm requires -- through the public planner with an
    explicit source-native packing constraint; ``native=False`` plans the
    packed (w13, rotated) payload the MXFP4 A4 prefill runner admits.
    """
    rotation = 0 if native else N
    fc1 = _planes(W13_ROWS, W13_COLS, rotation, seed=scale_seed)
    fc2 = _planes(W2_ROWS, W2_COLS, 0, seed=scale_seed + 500)
    w13, w2 = _weights(seed)
    constraints = None
    if native:
        constraints = fused_moe.WeightPlanConstraints(
            required_packing=fused_moe.WeightPacking.SOURCE_NATIVE
        )
    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(
            format="fp4_e8m0_k32", w13_layout="w31" if native else "w13"
        ),
        activation=fused_moe.ActivationSpec(
            mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=E, hidden_size=H, intermediate_size=N
        ),
        constraints=constraints,
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
    assert experts._impl.x4t_prefetch is not None
    if native:
        assert experts.plan.prepared_format.packing is (
            fused_moe.WeightPacking.SOURCE_NATIVE
        )
    return experts


def _scratch():
    return (
        torch.empty((E, W13_COLS, W13_ROWS), dtype=torch.uint8, device="cuda"),
        torch.empty((E, W2_COLS, W2_ROWS), dtype=torch.uint8, device="cuda"),
    )


def _activations(tokens, *, seed=5):
    """Modest BF16 activations and skewed nontrivial routes (every expert
    hit, unevenly)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn(tokens, H, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    shares = torch.tensor([1.0, 2.0, 3.0, 8.0], device="cuda")
    ids = torch.multinomial(shares.expand(tokens, -1), TOPK, generator=g)
    ids = ids.to(torch.int32)
    weights = torch.softmax(torch.randn(tokens, TOPK, device="cuda", generator=g), -1)
    return a, ids.contiguous(), weights.contiguous()


def _public_plan(experts, tokens):
    """The PUBLIC prepared plan for these experts (vLLM's B12xExperts drives
    exactly plan_execution + PreparationSession + bind + run)."""
    return fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=tokens, top_k=TOPK),
        routing=fused_moe.RoutingSpec(),
    )


def _bind_run(plan, scratch, *, a, ids, weights, output, expanded):
    """The public bind/run with the consumer-skip flag OFF or ON.

    ``fused_moe.bind`` resolves the plan's prepared state (which carries and
    enforces its prepared experts), mirroring vLLM's B12xExperts.apply which
    always binds the layer's installed prepared experts.
    """
    binding = fused_moe.bind(
        plan,
        scratch=scratch,
        a=a,
        topk_weights=weights,
        topk_ids=ids,
        output=output,
        input_scales_static=True,
        **({"x4t_scales_expanded": True} if expanded else {}),
    )
    return fused_moe.run(binding=binding)


def _poison(scratch_pair):
    for buf in scratch_pair:
        buf.fill_(0xD6)


def _in_session(plans, body, *, tokens):
    """Prepare every (name, plan) through one PreparationSession and run
    ``body(states)`` inside it, where ``states`` maps name -> the materialized
    _FusedMoeState. Binding and capture must stay inside the session: it owns
    the compiled programs, and leaving releases them (a later bind lazily
    re-prepares, which is illegal under capture)."""
    with PreparationSession(
        device=torch.device("cuda"), autotune=False, compile_workers=0
    ) as session:
        requests = tuple(
            plan.request(
                name=name,
                prepare_call=lambda state, t=tokens: _primer(state, t),
            )
            for name, plan in plans.items()
        )
        session.prepare(requests)
        states = {
            name: require_prepared(plan, "moe.decode") for name, plan in plans.items()
        }
        return body(states)


def _primer(state, tokens):
    scratch = tuple(
        torch.empty(s.shape, dtype=s.dtype, device=s.device)
        for s in state.scratch.scratch_specs()
    )
    a, ids, weights = _activations(tokens)
    output = torch.empty(tokens, H, dtype=torch.bfloat16, device="cuda")
    binding = state.bind(
        scratch=scratch,
        a=a,
        topk_weights=weights,
        topk_ids=ids,
        output=output,
        input_scales_static=True,
    )
    return PreparedCall(run=lambda: state.run(binding), owners=scratch)


def _state_scratch(state):
    return tuple(
        torch.empty(s.shape, dtype=s.dtype, device=s.device)
        for s in state.scratch.scratch_specs()
    )


# ---------------------------------------------------------------------------
# 1. Real-weight public-path equivalence + sensitivity (A16 arms)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("tokens", [4, 64])
def test_public_path_native_a16_prefetch_off_on_bit_identical(tokens):
    """Source-native (production decode layout) payload through the public
    plan/bind/run path: OFF (inline decode) and ON (consumed prefetch)
    produce BIT-IDENTICAL finite nontrivial output, and a deliberately WRONG
    layer's finite scales change the result (sensitivity). tokens=4 covers
    the small-M direct arm; tokens=64 the packed/direct route arm."""
    require_b12x()
    scratch_pair = _scratch()
    experts = _prepare(scratch_pair, native=True, scale_seed=1000)
    plan = _public_plan(experts, tokens)

    def body(states):
        scratch = _state_scratch(states["main"])
        a, ids, weights = _activations(tokens)

        # OFF: the runner expands the routed experts inline.
        out_off = torch.empty(tokens, H, dtype=torch.bfloat16, device="cuda")
        reset_packed_scale_host_call_count()
        _bind_run(
            plan, scratch, a=a, ids=ids, weights=weights, output=out_off, expanded=False
        )
        torch.cuda.synchronize()
        assert packed_scale_host_call_count() >= 1, "inline path must issue its decode"
        assert torch.isfinite(out_off).all()
        assert torch.count_nonzero(out_off) > 0, (
            "nonzero weights must give nontrivial output"
        )

        # ON: poison the shared scale scratch, prefetch EVERY expert through
        # the public expand_scales, consume with the flag -- no inline decode.
        _poison(scratch_pair)
        assert fused_moe.expand_scales(experts) is True
        out_on = torch.empty_like(out_off)
        reset_packed_scale_host_call_count()
        _bind_run(
            plan, scratch, a=a, ids=ids, weights=weights, output=out_on, expanded=True
        )
        torch.cuda.synchronize()
        assert packed_scale_host_call_count() == 0, (
            "the consumed prefetch must remove the inline decode host call"
        )
        torch.testing.assert_close(out_on, out_off, rtol=0, atol=0)

        # Sensitivity: a deliberately WRONG layer's finite scales (same
        # geometry, different exponents) expanded into the same scratch MUST
        # change the output when consumed -- proving the equality above
        # actually inspects the scale values (zero-weight fixtures could not
        # detect this).
        wrong = _prepare(scratch_pair, native=True, scale_seed=9000)
        _poison(scratch_pair)
        assert fused_moe.expand_scales(wrong) is True
        out_wrong = torch.empty_like(out_off)
        _bind_run(
            plan,
            scratch,
            a=a,
            ids=ids,
            weights=weights,
            output=out_wrong,
            expanded=True,
        )
        torch.cuda.synchronize()
        assert not torch.equal(out_wrong, out_off), (
            "finite WRONG scales must change the result (output equality is "
            "sensitive to scale values)"
        )
        assert torch.isfinite(out_wrong).all()

    _in_session({"main": plan}, body, tokens=tokens)


# ---------------------------------------------------------------------------
# 2. Mixed decode/prefill consumption of ONE shared prefetch (A4 + A16)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_public_path_packed_mxfp4_a4_and_a16_consume_one_prefetch(monkeypatch):
    """PACKED payload through the public path: the A16 route arm and the
    MXFP4 A4 prefill runner (its own B12X_W4A16_MXFP4_PREFILL_MIN_TOKENS
    admission) both consume ONE shared prefetch bit-exactly, back-to-back in
    a mixed decode-then-prefill sequence -- the §6.3 superset property
    across consumer arms. (The X4T payload has no per-subcall row split:
    the decode/prefill distinction is a per-call dispatch at bind.)"""
    require_b12x()
    # Compile the MXFP4 A4 prefill launches into the plan with the MXFP4 knob
    # alone (the production knob); the NVFP4 A4 knob stays unset.
    monkeypatch.setenv("B12X_W4A16_MXFP4_PREFILL_MIN_TOKENS", "8")
    monkeypatch.delenv("B12X_W4A16_A4_PREFILL_MIN_TOKENS", raising=False)
    scratch_pair = _scratch()
    experts = _prepare(scratch_pair, native=False, scale_seed=2000)
    tokens = 64
    plan = _public_plan(experts, tokens)

    def body(states):
        scratch = _state_scratch(states["main"])
        a, ids, weights = _activations(tokens)
        small_a, small_ids, small_w = _activations(8, seed=6)

        # Inline references for both shapes (flag OFF): the large call takes
        # the MXFP4 A4 prefill runner (tokens >= its min), the small call the
        # A16 route arm.
        out_big_off = torch.empty(tokens, H, dtype=torch.bfloat16, device="cuda")
        out_small_off = torch.empty(8, H, dtype=torch.bfloat16, device="cuda")
        _bind_run(
            plan,
            scratch,
            a=a,
            ids=ids,
            weights=weights,
            output=out_big_off,
            expanded=False,
        )
        _bind_run(
            plan,
            scratch,
            a=small_a,
            ids=small_ids,
            weights=small_w,
            output=out_small_off,
            expanded=False,
        )
        torch.cuda.synchronize()
        assert torch.isfinite(out_big_off).all() and torch.isfinite(out_small_off).all()
        assert torch.count_nonzero(out_big_off) > 0

        # One shared prefetch, then a back-to-back mixed consumption: the
        # small (decode-shaped, A16 arm) call and the large (prefill, MXFP4
        # A4 arm) call BOTH read the same all-expert expansion.
        _poison(scratch_pair)
        assert fused_moe.expand_scales(experts) is True
        out_small_on = torch.empty_like(out_small_off)
        out_big_on = torch.empty_like(out_big_off)
        reset_packed_scale_host_call_count()
        _bind_run(
            plan,
            scratch,
            a=small_a,
            ids=small_ids,
            weights=small_w,
            output=out_small_on,
            expanded=True,
        )
        _bind_run(
            plan,
            scratch,
            a=a,
            ids=ids,
            weights=weights,
            output=out_big_on,
            expanded=True,
        )
        torch.cuda.synchronize()
        assert packed_scale_host_call_count() == 0, (
            "both consumer arms must skip their inline decode for the shared "
            "prefetch (the full expansion is a superset for both)"
        )
        torch.testing.assert_close(out_small_on, out_small_off, rtol=0, atol=0)
        torch.testing.assert_close(out_big_on, out_big_off, rtol=0, atol=0)

    _in_session({"main": plan}, body, tokens=tokens)


# ---------------------------------------------------------------------------
# 3. Two-stream overlap: real CUDA streams, events, no host synchronize
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_two_layers_share_scratch_across_two_real_streams():
    """Two layers with DIFFERENT scale contents over ONE shared scratch pair,
    produced and consumed on TWO REAL CUDA streams with events, back-to-back
    forwards, and NO torch.cuda.synchronize between expansion and
    consumption (the previous tests synchronized between the two; the vLLM
    tests used fake streams). Broken ordering reads another layer's scales
    and fails the bit-exact output check."""
    require_b12x()
    scratch_pair = _scratch()
    layer_a = _prepare(scratch_pair, native=True, scale_seed=3100)
    layer_b = _prepare(scratch_pair, native=True, scale_seed=4400)
    tokens = 16
    plan_a = _public_plan(layer_a, tokens)
    plan_b = _public_plan(layer_b, tokens)

    def body(states):
        scratch_a = _state_scratch(states["a"])
        scratch_b = _state_scratch(states["b"])
        a, ids, weights = _activations(tokens)

        # Inline references first (also proves both layers run correctly).
        ref_a = torch.empty(tokens, H, dtype=torch.bfloat16, device="cuda")
        ref_b = torch.empty(tokens, H, dtype=torch.bfloat16, device="cuda")
        _bind_run(
            plan_a,
            scratch_a,
            a=a,
            ids=ids,
            weights=weights,
            output=ref_a,
            expanded=False,
        )
        _bind_run(
            plan_b,
            scratch_b,
            a=a,
            ids=ids,
            weights=weights,
            output=ref_b,
            expanded=False,
        )
        torch.cuda.synchronize()
        ref_a, ref_b = ref_a.clone(), ref_b.clone()

        main = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        moe_done = torch.cuda.Event()
        ready = torch.cuda.Event()

        # Forward 1: layer A consumes inline; the side stream then expands
        # B's scales (waits A's MoE readers), records readiness. No sync.
        out_a = torch.empty_like(ref_a)
        _bind_run(
            plan_a,
            scratch_a,
            a=a,
            ids=ids,
            weights=weights,
            output=out_a,
            expanded=False,
        )
        moe_done.record(main)
        with torch.cuda.stream(side):
            side.wait_event(moe_done)
            assert fused_moe.expand_scales(layer_b) is True
            ready.record(side)
        # Back-to-back forward 2 ON THE MAIN STREAM: B waits the side-stream
        # readiness event and consumes the prefetch (flag ON).
        out_b = torch.empty_like(ref_b)
        main.wait_event(ready)
        _bind_run(
            plan_b,
            scratch_b,
            a=a,
            ids=ids,
            weights=weights,
            output=out_b,
            expanded=True,
        )
        # Back-to-back forward 3: A again -- the side stream re-expands A
        # over the shared scratch after B's MoE is done; A waits, consumes.
        moe_done2 = torch.cuda.Event()
        ready2 = torch.cuda.Event()
        moe_done2.record(main)
        with torch.cuda.stream(side):
            side.wait_event(moe_done2)
            assert fused_moe.expand_scales(layer_a) is True
            ready2.record(side)
        main.wait_event(ready2)
        out_a2 = torch.empty_like(ref_a)
        _bind_run(
            plan_a,
            scratch_a,
            a=a,
            ids=ids,
            weights=weights,
            output=out_a2,
            expanded=True,
        )
        # Only now -- all async work enqueued and ordered by events alone --
        # do we synchronize to read results.
        torch.cuda.synchronize()
        torch.testing.assert_close(out_a, ref_a, rtol=0, atol=0)
        torch.testing.assert_close(out_b, ref_b, rtol=0, atol=0)
        torch.testing.assert_close(out_a2, ref_a, rtol=0, atol=0)

    _in_session({"a": plan_a, "b": plan_b}, body, tokens=tokens)


# ---------------------------------------------------------------------------
# 4. Capture + replay with a pending prefetch (real CUDA graphs)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_capture_and_replay_transition_with_pending_prefetch():
    """Transition through REAL CUDA-graph capture AND replay with a PENDING
    side-stream prefetch. Empirical contract pinned by this test: a capture
    CANNOT adopt an in-flight cross-stream dependency (waiting the pending
    event inside capture raises cudaErrorStreamCaptureInvalidated), so the
    serving lifecycle drains BEFORE capture begins; the captured graph keeps
    the INLINE decode (the flag is never declared under capture), so replays
    re-expand over poisoned scratch and stay bit-exact. No fake wait_event."""
    require_b12x()
    scratch_pair = _scratch()
    experts = _prepare(scratch_pair, native=True, scale_seed=7000)
    tokens = 8
    plan = _public_plan(experts, tokens)

    def body(states):
        scratch = _state_scratch(states["main"])
        a, ids, weights = _activations(tokens)

        # Eager reference (inline decode).
        out_ref = torch.empty(tokens, H, dtype=torch.bfloat16, device="cuda")
        _bind_run(
            plan, scratch, a=a, ids=ids, weights=weights, output=out_ref, expanded=False
        )
        torch.cuda.synchronize()
        out_ref = out_ref.clone()

        # A real PENDING prefetch: main-stream reader done, side-stream
        # expansion recorded, NOT waited, NOT synchronized.
        main = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        moe_done = torch.cuda.Event()
        ready = torch.cuda.Event()
        reader = torch.empty_like(out_ref)
        _bind_run(
            plan, scratch, a=a, ids=ids, weights=weights, output=reader, expanded=False
        )
        moe_done.record(main)
        with torch.cuda.stream(side):
            side.wait_event(moe_done)
            assert fused_moe.expand_scales(experts) is True
            ready.record(side)

        # The POSITIVE lifecycle block first (this was clobbered by an edit):
        # drain-before-capture -> capture with the flag OFF -> two replays over
        # POISONED scratch, all bit-exact. The pending prefetch above is
        # drained by an explicit device synchronize (the serving lifecycle's
        # graph_capture context does the same) BEFORE capture begins.
        torch.cuda.synchronize()  # drain the pending side-stream prefetch
        _poison(scratch_pair)
        graph = torch.cuda.CUDAGraph()
        out_cap = torch.empty_like(out_ref)
        with torch.cuda.graph(graph):
            # The flag is never declared under capture: the captured graph
            # keeps the INLINE decode and re-expands on every replay.
            _bind_run(
                plan,
                scratch,
                a=a,
                ids=ids,
                weights=weights,
                output=out_cap,
                expanded=False,
            )
        assert packed_scale_host_call_count() >= 1, (
            "capture is a HOST launch of the inline decode (the counter "
            "increments for captures but never for replays)"
        )
        _poison(scratch_pair)
        reset_packed_scale_host_call_count()
        graph.replay()
        torch.cuda.synchronize()
        assert packed_scale_host_call_count() == 0, "replays launch no host calls"
        torch.testing.assert_close(out_cap, out_ref, rtol=0, atol=0)
        _poison(scratch_pair)
        graph.replay()
        torch.cuda.synchronize()
        assert packed_scale_host_call_count() == 0, "replays launch no host calls"
        torch.testing.assert_close(out_cap, out_ref, rtol=0, atol=0)
        del graph

        # Prove the negative LAST (after the lifecycle is verified): adopting
        # the IN-FLIGHT dependency inside capture is illegal and invalidates
        # that capture -- the exact hazard the review flagged; a fake
        # wait_event would hide it. Run after the replay assertions because
        # an invalidated capture leaves the context's capture machinery in a
        # failed state for that graph only; the successful graph above is
        # already instantiated and unaffected.
        # Codex §10.1: tightened from pytest.raises(Exception) to the ACTUAL
        # CUDA capture error class/message (torch.AcceleratorError with
        # cudaErrorStreamCaptureIsolation / cudaErrorStreamCaptureInvalidated
        # -- pinned empirically on this build; an arbitrary exception is not
        # proof of the documented CUDA error class). Because an invalidated
        # capture can poison subsequent captures IN THIS PROCESS, the
        # wait-inside-capture runs in an ISOLATED SUBPROCESS: the parent
        # asserts on the child's exit status and stderr.
        repo_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
            )
        )
        compat_dir = os.path.join(
            repo_root, "flashinfer", "experimental", "b12x", "_compat"
        )
        child_env = {**os.environ, "B12X_X4T_SCALE_LAUNCH_DIAGNOSTICS": "1"}
        child_env["PYTHONPATH"] = os.pathsep.join(
            p for p in (repo_root, compat_dir, os.environ.get("PYTHONPATH")) if p
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                _NEGATIVE_ARM_SUBPROCESS,
            ],
            capture_output=True,
            text=True,
            env=child_env,
            cwd=repo_root,
            timeout=600,
        )
        assert proc.returncode == 0, (
            "the isolated wait-inside-capture child must report the intended "
            f"CUDA capture error; got rc={proc.returncode} "
            f"stdout={proc.stdout[-500:]} stderr={proc.stderr[-2000:]}"
        )
        assert "NEGATIVE_ARM_RESULT=AcceleratorError" in proc.stdout, (
            f"the child must surface torch.AcceleratorError; stdout="
            f"{proc.stdout[-500:]}"
        )
        assert "cudaErrorStreamCaptureIsolation" in proc.stdout, (
            "the root error must be the CUDA capture-isolation error "
            f"(dependency on uncaptured work); stdout={proc.stdout[-2000:]}"
        )
        assert "cudaErrorStreamCaptureInvalidated" in proc.stdout, (
            "capture_end must fail with the CUDA capture-invalidated error; "
            f"stdout={proc.stdout[-2000:]}"
        )

    _in_session({"main": plan}, body, tokens=tokens)


# ---------------------------------------------------------------------------
# 5. Rejection paths and empty/zero route sets (executed, not inspected)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bind_rejects_skip_flag_for_capability_less_payload():
    """bind REJECTS x4t_scales_expanded=True for a payload without the
    retained paired-plane capability (plain packed FP4 weights), by actually
    calling bind through the public plan."""
    require_b12x()
    rng = np.random.default_rng(3)
    w13 = torch.from_numpy(
        rng.integers(1, 256, (E, W13_ROWS, H // 2), dtype=np.uint8)
    ).to("cuda")
    w2 = torch.from_numpy(
        rng.integers(1, 256, (E, W2_ROWS, N // 2), dtype=np.uint8)
    ).to("cuda")
    s13 = torch.ones((E, W13_ROWS, W13_COLS), dtype=torch.uint8, device="cuda")
    s2 = torch.ones((E, W2_ROWS, W2_COLS), dtype=torch.uint8, device="cuda")
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
        weights=fused_moe.PackedWeights(
            w13=w13,
            w2=w2,
            w13_block_scales=s13,
            w2_block_scales=s2,
            w13_global_scales=torch.ones(E, device="cuda"),
            w2_global_scales=torch.ones(E, device="cuda"),
        ),
    )
    assert experts._impl.x4t_prefetch is None
    tokens = 8
    exec_plan = _public_plan(experts, tokens)

    def body(states):
        scratch = _state_scratch(states["main"])
        a, ids, weights = _activations(tokens)
        out = torch.empty(tokens, H, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="x4t_scales_expanded"):
            fused_moe.bind(
                exec_plan,
                scratch=scratch,
                a=a,
                topk_weights=weights,
                topk_ids=ids,
                output=out,
                input_scales_static=True,
                x4t_scales_expanded=True,
            )

    _in_session({"main": exec_plan}, body, tokens=tokens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bind_rejects_skip_flag_for_non_w4a16_implementation():
    """bind REJECTS x4t_scales_expanded=True for a non-W4A16 plan (the A8
    MXFP4 recipe over the same scale planes), by actually calling bind."""
    require_b12x()
    scratch_pair = _scratch()
    fc1 = _planes(W13_ROWS, W13_COLS, N, seed=5500)
    fc2 = _planes(W2_ROWS, W2_COLS, 0, seed=6000)
    w13, w2 = _weights(seed=7)
    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w13"),
        activation=fused_moe.ActivationSpec(
            mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16
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
            w13_scale_scratch=scratch_pair[0],
            w2_scale_scratch=scratch_pair[1],
        ),
    )
    tokens = 8
    exec_plan = _public_plan(experts, tokens)

    def body(states):
        scratch = _state_scratch(states["main"])
        a, ids, weights = _activations(tokens)
        out = torch.empty(tokens, H, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="x4t_scales_expanded"):
            fused_moe.bind(
                exec_plan,
                scratch=scratch,
                a=a,
                topk_weights=weights,
                topk_ids=ids,
                output=out,
                input_scales_static=True,
                x4t_scales_expanded=True,
            )

    _in_session({"main": exec_plan}, body, tokens=tokens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_empty_ids_and_zero_counts_route_sets_execute():
    """Kernel-level route sentinels, EXECUTED on GPU (not source-inspected):
    an empty ids tensor is a no-op (no host decode call, scratch untouched)
    and an all-zero counts buffer selects NOTHING (poison survives)."""
    require_b12x()
    scratch_pair = _scratch()
    experts = _prepare(scratch_pair, native=True, scale_seed=8000)
    planes = experts._impl.x4t_prefetch.planes
    _poison(scratch_pair)
    before = scratch_pair[0].clone()

    reset_packed_scale_host_call_count()
    decode_x4t_packed_scale_pair(
        planes[0],
        planes[1],
        torch.empty(0, dtype=torch.int32, device="cuda"),
        experts._impl.w1_blockscale,
        experts._impl.w2_blockscale,
    )
    torch.cuda.synchronize()
    assert packed_scale_host_call_count() == 0, "empty ids must be a no-op"
    assert torch.equal(scratch_pair[0], before), "empty ids must not write"

    # Zero counts select NO experts (counts must be POSITIVE): executed.
    zeros = torch.zeros(E, dtype=torch.int32, device="cuda")
    decode_x4t_packed_scale_pair(
        planes[0],
        planes[1],
        zeros,
        experts._impl.w1_blockscale,
        experts._impl.w2_blockscale,
        expert_counts=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(scratch_pair[0], before), (
        "zero counts must select nothing (poison survives)"
    )
