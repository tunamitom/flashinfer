"""CPU policy and identity tests for the paired-gather Kimi top-16 select.

The fused select that rides the paired DCP gather (``all_gather_pair_kimi_topk``)
implements the tie / near-tie / non-finite selection contract documented at
``pair_push_enabled`` in ``b12x.comm.pcie._dcp_a2a_cute``. These tests execute
that contract as a pure-Python model of the kernel's packed-key selection (the
model mirrors ``_kimi_pack_key`` and ``float_order_key`` exactly) and pin the
kernel source to the same policy, so neither side can drift silently.

Model notes:

- ``pack_key(score, expert)`` puts the float-order key of the score in the high
  32 bits and the *complemented* expert id in the low 16, so one unsigned max
  both ranks candidates and resolves score ties in favour of the lower expert
  id.
- Exhausted or out-of-range candidates carry the zero key, which sorts below
  every valid packed key (including ``-inf`` scores), so an exhausted queue can
  never re-offer an expert id.
- The fused select scores with ``0.5 * tanh(0.5 * x) + 0.5`` and the standalone
  ``kimi_topk16`` with ``1 / (1 + exp(-x))``; the two forms agree to at most a
  few fp32 ULP, which is the documented near-tie tolerance.
"""

from __future__ import annotations

import inspect
import struct

import pytest

_EXPERTS = 896
_TOPK = 16


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _float_order_key(value: float) -> int:
    """Mirror of ``_cute_intrinsics.float_order_key`` (float order -> u32)."""
    bits = struct.unpack("<I", struct.pack("<f", _f32(value)))[0]
    mask = ((bits >> 31) * 0x7FFFFFFF) | 0x80000000
    return bits ^ mask


def _pack_key(score: float, expert: int) -> int:
    """Mirror of ``_dcp_a2a_cute._kimi_pack_key``."""
    return (_float_order_key(score) << 32) | (0xFFFF - expert)


def _key_expert(key: int) -> int:
    return 0xFFFF - (key & 0xFFFF)


def _selection_scores(logits, bias, *, fused: bool):
    """Selection scores as the kernel computes them (fused or standalone form)."""
    import math

    scores = []
    for logit, correction in zip(logits, bias):
        if fused:
            if not (math.isfinite(logit) and math.isfinite(correction)):
                scores.append(float("-inf"))
                continue
            unbiased = 0.5 * math.tanh(0.5 * logit) + 0.5
            selection = unbiased + correction
        else:
            unbiased = 1.0 / (1.0 + math.exp(-logit)) if math.isfinite(logit) else 0.0
            selection = unbiased + correction
            if not math.isfinite(selection):
                selection = float("inf") if selection > 0 else float("-inf")
        if selection == 0.0:
            selection = 0.0  # canonicalise signed zero
        scores.append(_f32(selection))
    return scores


def _topk_ids(scores, *, topk: int = _TOPK, experts: int = _EXPERTS):
    """Packed-key top-k: unsigned max over (score, complemented expert) keys."""
    keys = []
    for expert in range(experts):
        score = scores[expert] if expert < len(scores) else float("-inf")
        keys.append(_pack_key(score, expert))
    winners = sorted(keys, reverse=True)[:topk]
    return [_key_expert(key) for key in winners], winners


def _renormalized_weights(scores, ids):
    """Renormalized output weights for the documented zero-weight edge case."""
    weights = []
    for expert in ids:
        score = scores[expert]
        unbiased = 0.0 if score == float("-inf") else score
        weights.append(unbiased)
    total = 0.0
    for weight in weights:
        total += weight
    scale = 1.0 / (total + 1.0e-20)
    return [_f32(w * scale) for w in weights]


def _next_after(value: float, steps: int = 1) -> float:
    current = _f32(value)
    for _ in range(steps):
        bits = struct.unpack("<I", struct.pack("<f", current))[0]
        bits = bits + 1 if current >= 0 else bits - 1
        current = struct.unpack("<f", struct.pack("<I", bits))[0]
    return current


def _clear_row(offset: float = 0.0):
    """A row with 16 clearly separated finite candidates and a clean tail."""
    logits = [0.0] * _EXPERTS
    bias = [0.0] * _EXPERTS
    for rank, expert in enumerate(range(32, 48)):
        bias[expert] = offset + 4.0 - rank * 0.25
    return logits, bias


class TestTiePolicy:
    def test_equal_scores_pick_the_lower_expert_id(self):
        logits = [0.0] * _EXPERTS
        bias = [0.0] * _EXPERTS
        for expert in (400, 7, 895, 12, 300):
            bias[expert] = 2.0
        ids, _ = _topk_ids(_selection_scores(logits, bias, fused=True))
        assert ids[:5] == [7, 12, 300, 400, 895]

    def test_tie_break_is_identical_in_the_standalone_form(self):
        logits = [0.0] * _EXPERTS
        bias = [0.0] * _EXPERTS
        for expert in (23, 24, 25):
            bias[expert] = 1.5
        fused_ids, _ = _topk_ids(_selection_scores(logits, bias, fused=True))
        standalone_ids, _ = _topk_ids(_selection_scores(logits, bias, fused=False))
        assert fused_ids[:3] == [23, 24, 25]
        assert standalone_ids[:3] == [23, 24, 25]

    def test_signed_zero_is_canonicalised_before_packing(self):
        # Packed keys distinguish -0.0 from +0.0, which would break the
        # deterministic tie order, so both kernels canonicalize a zero
        # selection score to +0.0 before packing.
        assert _pack_key(0.0, 1) != _pack_key(-0.0, 1)
        from b12x.comm.pcie import _dcp_a2a_cute as kernels

        fused_source = inspect.getsource(kernels._AllGatherPairLaunch.kernel)
        standalone_source = inspect.getsource(kernels._KimiTopK16Launch.kernel)
        for source in (fused_source, standalone_source):
            assert "if selection == Float32(0.0):" in source
            assert source.count("selection = Float32(0.0)") >= 1


class TestNearTiePolicy:
    def _boundary_row(self, flip: bool):
        """16 clear winners plus a 16/17 boundary pair one ULP apart."""
        logits = [0.0] * _EXPERTS
        bias = [-10.0] * _EXPERTS
        base = 3.0
        for rank, expert in enumerate(range(0, 15)):
            bias[expert] = base - rank * 0.25
        boundary = _f32(base - 15 * 0.25)
        bias[15] = boundary
        bias[16] = _next_after(boundary, 2) if flip else _next_after(boundary, -2)
        return logits, bias

    def test_near_tie_boundary_swap_is_one_for_one(self):
        clear = _selection_scores(*self._boundary_row(flip=False), fused=True)
        flipped = _selection_scores(*self._boundary_row(flip=True), fused=True)
        clear_ids, _ = _topk_ids(clear)
        flipped_ids, _ = _topk_ids(flipped)

        assert len(set(clear_ids)) == _TOPK
        assert len(set(flipped_ids)) == _TOPK
        differing = [
            (left, right)
            for left, right in zip(clear_ids, flipped_ids)
            if left != right
        ]
        # The boundary pair swaps in one slot; every other slot is identical.
        assert len(differing) == 1
        slot = clear_ids.index(differing[0][0])
        assert flipped_ids[slot] == differing[0][1]
        assert {differing[0][0], differing[0][1]} == {15, 16}
        # And the swap is a documented near-tie: the two scores sit within a
        # few fp32 ULP of each other.
        left = clear[15]
        right = clear[16]
        gap = abs(_float_order_key(left) - _float_order_key(right))
        assert gap <= 4

    def test_same_set_order_swap_keeps_membership(self):
        # Two candidates with equal selection scores order by expert id in both
        # forms, so an order swap can never change set membership on its own.
        logits = [0.0] * _EXPERTS
        bias = [-10.0] * _EXPERTS
        bias[100] = 2.0
        bias[101] = 2.0
        fused_ids, _ = _topk_ids(_selection_scores(logits, bias, fused=True))
        standalone_ids, _ = _topk_ids(_selection_scores(logits, bias, fused=False))
        assert set(fused_ids) == set(standalone_ids)


class TestNonFinitePolicy:
    def test_non_finite_scores_rank_below_every_finite_score(self):
        assert _pack_key(float("-inf"), 0) < _pack_key(_f32(-3.0e38), 895)
        assert _pack_key(float("-inf"), 0) < _pack_key(_f32(0.0), 895)

    def test_zero_sentinel_sorts_below_negative_infinity(self):
        assert 0 < _pack_key(float("-inf"), 895)

    def test_non_finite_row_returns_lowest_ids_at_zero_weight(self):
        logits = [float("nan")] * _EXPERTS
        bias = [0.0] * _EXPERTS
        scores = _selection_scores(logits, bias, fused=True)
        ids, _ = _topk_ids(scores)
        assert ids == list(range(_TOPK))
        weights = _renormalized_weights(scores, ids)
        assert weights == [0.0] * _TOPK

    def test_one_finite_candidate_is_never_displaced(self):
        logits = [float("-inf")] * _EXPERTS
        bias = [0.0] * _EXPERTS
        logits[77] = 0.5
        scores = _selection_scores(logits, bias, fused=True)
        ids, _ = _topk_ids(scores)
        assert ids[0] == 77
        assert len(set(ids)) == _TOPK


class TestExhaustedQueuePolicy:
    def test_sparse_finite_rows_never_repeat_an_expert_id(self):
        import random

        rng = random.Random(20261111)
        for finite in (0, 1, 3, 15, 16, 17):
            for _ in range(8):
                logits = [float("-inf")] * _EXPERTS
                bias = [0.0] * _EXPERTS
                for expert in rng.sample(range(_EXPERTS), finite):
                    logits[expert] = rng.uniform(-4.0, 4.0)
                scores = _selection_scores(logits, bias, fused=True)
                ids, _ = _topk_ids(scores)
                assert len(ids) == _TOPK
                assert len(set(ids)) == _TOPK, (finite, ids)

    def test_fused_kernel_refills_with_the_zero_sentinel_in_both_levels(self):
        from b12x.comm.pcie import _dcp_a2a_cute as kernels

        source = inspect.getsource(kernels._AllGatherPairLaunch.kernel)
        assert source.count("tail_key = cutlass.Uint64(0)") == 2
        # The pre-fix sentinel re-packed (-inf, expert(key)) and could
        # re-offer an already-selected expert; it must not come back.
        assert "tail_key = _kimi_pack_key" not in source

    def test_out_of_range_expert_lanes_use_the_zero_sentinel(self):
        from b12x.comm.pcie import _dcp_a2a_cute as kernels

        source = inspect.getsource(kernels._AllGatherPairLaunch.kernel)
        assert "keys[item] = cutlass.Uint64(0)" in source
        assert "if expert_id < Int32(896):" in source


class TestSelectionIdentity:
    def test_pack_key_tie_break_matches_the_standalone_kernel(self):
        from b12x.comm.pcie import _dcp_a2a_cute as kernels

        pack_source = inspect.getsource(kernels._kimi_pack_key)
        assert "0xFFFF" in pack_source
        # Both selects share the key encoding, so ties resolve identically.
        fused_source = inspect.getsource(kernels._AllGatherPairLaunch.kernel)
        standalone_source = inspect.getsource(kernels._KimiTopK16Launch.kernel)
        assert "_kimi_pack_key(" in fused_source
        assert "_kimi_pack_key(" in standalone_source

    def test_clear_rows_select_the_same_ids_in_both_forms(self):
        logits, bias = _clear_row()
        fused_ids, _ = _topk_ids(_selection_scores(logits, bias, fused=True))
        standalone_ids, _ = _topk_ids(_selection_scores(logits, bias, fused=False))
        assert fused_ids[:16] == list(range(32, 48))
        assert fused_ids == standalone_ids


class TestPairPushSwitch:
    @pytest.mark.parametrize("enabled", [False, True])
    def test_pair_push_switch_selects_transport_and_compile_identity(
        self, monkeypatch, enabled: bool
    ) -> None:
        from b12x._lib.compile_plan import ProgramKey
        from b12x.comm.pcie import _dcp_a2a_cute as kernels

        if enabled:
            monkeypatch.setenv("B12X_PCIE_DCP_PAIR_PUSH", "1")
        else:
            monkeypatch.delenv("B12X_PCIE_DCP_PAIR_PUSH", raising=False)
        assert kernels.pair_push_enabled() is enabled

        compiled = []

        def fake_compile(launch, *args, compile_spec, **kwargs):
            def raw(*a, **k):
                return None

            raw.__b12x_programs__ = (ProgramKey("cute", "test", "dcp-pair-push"),)
            compiled.append((launch, compile_spec))
            return raw

        monkeypatch.setattr(kernels, "b12x_compile", fake_compile)
        monkeypatch.setattr(
            kernels,
            "raise_if_kernel_resolution_frozen",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(kernels, "_u32_ptr", lambda *args, **kwargs: 0)
        monkeypatch.setattr(kernels, "_f32_ptr", lambda *args, **kwargs: 0)
        monkeypatch.setattr(kernels, "_i32_ptr", lambda *args, **kwargs: 0)
        monkeypatch.setattr(kernels, "_u8_ptr", lambda *args, **kwargs: 0)
        monkeypatch.setattr(kernels, "current_cuda_stream", lambda: 0)

        for kimi_topk in (False, True):
            kernels._get_compiled_all_gather_pair.__wrapped__(
                2, 0, 512, True, kimi_topk, kernels.pair_push_enabled()
            )

        assert [launch._push for launch, _ in compiled] == [enabled, enabled]
        assert [launch._kimi_topk for launch, _ in compiled] == [False, True]
        for launch, spec in compiled:
            assert ("transport" in spec.json_key) is enabled

        # The default pull transport keeps its historical cache key exactly;
        # the push variant is keyed separately.
        pull_key = kernels._pair_launcher_key(2, 0, 512, True, True, False)
        push_key = kernels._pair_launcher_key(2, 0, 512, True, True, True)
        assert pull_key == (2, 0, 512, True, True)
        assert push_key == (2, 0, 512, True, True, "push")

    def test_pair_push_reads_the_switch_like_the_other_transport_switches(self) -> None:
        from b12x.comm.pcie import _dcp_a2a_cute as kernels

        assert kernels.PAIR_PUSH_ENV == "B12X_PCIE_DCP_PAIR_PUSH"
        source = inspect.getsource(kernels.pair_push_enabled)
        assert 'os.environ.get(PAIR_PUSH_ENV, "0")' in source
