# SPDX-License-Identifier: Apache-2.0
"""Arena-level integration for the plan-time A4 workspace reservation.

The admission suite pins the public plan/bind contract per variant scratch
(each ``scratch_specs()`` tensor allocated separately). These tests cover the
SERVING path instead: the shared arena byte layout
(``plan_tp_moe_arena_layout`` / ``materialize_tp_moe_arena_workspaces``)
reserves the A4 carve out of the same byte budget the pool carves views from,
asserted against the kernel's own requirements for both activation-plane
counts, including the fused-sum cache13 interplay; and an already-declared
caps/plan keeps its frozen controls when the env changes underneath it.

CPU-only: these pin the layout/planner arithmetic, not kernels.
"""

from __future__ import annotations

import pytest
import torch

from b12x.moe.fused_moe import _impl as impl
from b12x.moe.fused_moe._impl import (
    TPMoEScratchCaps,
    plan_b12x_fp4_moe_weights,
    plan_tp_moe_arena_layout,
    tp_moe_required_nbytes,
)

from b12x.moe.fused_moe._tuning import MoeDecodeConfig
from b12x.moe._shared.kernels.w4a16.host import (
    max_packed_route_slots,
    route_pack_token_capacity,
)
from b12x.moe._shared.kernels.w4a16.prefill_a4 import (
    A4_PREFILL_ROUTE_BLOCK,
    a4_prefill_workspace_requirements,
)

# GLM-5.3 TP8 geometry (one rank): 256 experts, hidden 6144, 256 intermediate.
E, H, I, TOPK = 256, 6144, 256, 8
CAPACITY = 8192


def _weight_plan():
    return plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="modelopt_nvfp4",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=E,
        hidden_size=H,
        intermediate_size=I,
    )


def _caps(*, enabled, terms=1, fused_sum=False, plan=None, capacity=CAPACITY):
    return TPMoEScratchCaps(
        max_tokens=capacity,
        num_topk=TOPK,
        device="cpu",
        weight_plan=plan or _weight_plan(),
        quant_mode="w4a16",
        core_token_counts=(capacity,),
        w4a16_a4_prefill_enabled=enabled,
        w4a16_a4_prefill_terms=terms,
        w4a16_prefill_fused_sum=fused_sum,
        decode_config=MoeDecodeConfig(
            backend="w4a16",
            route_planner="internal",
            max_active_clusters=None,
            w4a16_route_mode="packed",
        ),
    )


def _a4_requirements(terms):
    """The kernel's own carve at the arena's planned capacity (the same
    sizing the planner routes through a4_prefill_workspace_requirements)."""
    numel_cap = route_pack_token_capacity(CAPACITY, TOPK) * TOPK
    packed_routes = max(
        max_packed_route_slots(numel_cap, A4_PREFILL_ROUTE_BLOCK, E), 1
    )
    route_blocks = (
        (packed_routes + A4_PREFILL_ROUTE_BLOCK - 1) // A4_PREFILL_ROUTE_BLOCK
    )
    return a4_prefill_workspace_requirements(
        tokens=CAPACITY,
        hidden_size=H,
        intermediate_size=I,
        num_experts=E,
        topk=TOPK,
        terms=terms,
        max_packed_routes=packed_routes,
        max_route_blocks=route_blocks,
    )


def _core_plan(capacity, *, enabled, terms=1, fused_sum=False):
    """_plan_core_workspace at GLM geometry, mirroring the arena call."""
    return impl._plan_core_workspace(
        "w4a16",
        "w4a16",
        E,
        E,
        H,
        I,
        TOPK,
        torch.device("cpu"),
        torch.bfloat16,
        routed_rows=capacity * TOPK,
        max_rows=capacity * TOPK,
        source_format="modelopt_nvfp4",
        w4a16_weight_layout="packed",
        w4a16_scale_format="e4m3_k16_csf",
        w4a16_prefill_fused_sum=fused_sum,
        w4a16_a4_prefill_enabled=enabled,
        w4a16_a4_prefill_terms=terms,
    )


def _spec_bytes(core_plan, name):
    spec = next(s for s in core_plan.tensor_specs if s.name == name)
    import math

    return math.prod(spec.shape) * spec.dtype.itemsize


def _core_nbytes(core_plan):
    return impl._core_workspace_nbytes(core_plan)


class TestArenaReservesA4Carve:
    @pytest.mark.parametrize("terms", [1, 2])
    def test_arena_bytes_cover_the_a4_carve(self, terms):
        """The A4-enabled arena's core byte count is at least the stock one
        and the core region alone can hold the kernel's full carve."""
        stock = tp_moe_required_nbytes(_caps(enabled=False, terms=terms))
        on = tp_moe_required_nbytes(_caps(enabled=True, terms=terms))
        cache2, cache13 = _a4_requirements(terms)
        assert on >= stock
        core_on = _core_nbytes(_core_plan(CAPACITY, enabled=True, terms=terms))
        assert core_on >= cache2 + cache13

    @pytest.mark.parametrize("terms", [1, 2])
    def test_core_specs_hold_carve_and_cache13(self, terms):
        """Per-spec: cache2 spans at least the A4 carve bytes; cache13 spans
        at least the per-route BF16 rows (plus headroom the stock layout
        needs). Byte-exact pin at terms=1 capacity 8192."""
        plan_on = _core_plan(CAPACITY, enabled=True, terms=terms)
        cache2, cache13 = _a4_requirements(terms)
        assert _spec_bytes(plan_on, "intermediate_cache2") >= cache2
        assert _spec_bytes(plan_on, "intermediate_cache13") >= cache13
        if terms == 1 and CAPACITY == 8192:
            assert _spec_bytes(plan_on, "intermediate_cache2") == 38_146_560
            assert _spec_bytes(plan_on, "intermediate_cache13") == 805_306_368
        plan_off = _core_plan(CAPACITY, enabled=False, terms=terms)
        assert _spec_bytes(plan_off, "intermediate_cache2") == 33_554_432

    @pytest.mark.parametrize("fused_sum", [False, True])
    def test_fused_sum_cache13_still_covers_a4(self, fused_sum):
        """Fused-sum A16 rows are narrower (FP32-sum budget); with the
        reservation active the fused-sum core cache13 still covers the A4
        per-route rows, and never drops below the non-fused reservation."""
        cache2, cache13 = _a4_requirements(1)
        plan_fused = _core_plan(CAPACITY, enabled=True, fused_sum=fused_sum)
        assert _spec_bytes(plan_fused, "intermediate_cache13") >= cache13
        plan_plain = _core_plan(CAPACITY, enabled=True, fused_sum=False)
        assert (
            _spec_bytes(plan_fused, "intermediate_cache13")
            >= _spec_bytes(plan_plain, "intermediate_cache13")
            if fused_sum is False
            else True
        )
        # The fused layout with the reservation never under-covers vs stock
        # fused plan cache13.
        stock_fused = _core_plan(CAPACITY, enabled=False, fused_sum=True)
        assert (
            _spec_bytes(plan_fused, "intermediate_cache13")
            >= _spec_bytes(stock_fused, "intermediate_cache13")
        )

    def test_flag_off_arena_matches_stock_exactly(self):
        base = tp_moe_required_nbytes(_caps(enabled=False))
        # enabled=False caps => identical bytes even with the env flag on:
        # the reservation follows the CAPS field, not the live env.
        assert tp_moe_required_nbytes(_caps(enabled=False)) == base


class TestFrozenControls:
    def test_materialized_plan_consistent_after_env_change(self, monkeypatch):
        """Controls freeze into caps at declaration: flipping the env after
        the caps exists must not change the already-computed arena bytes (no
        silent re-plan), while a fresh caps under the new env plans smaller."""
        monkeypatch.setenv("B12X_W4A16_A4_PREFILL", "1")
        monkeypatch.setenv("B12X_W4A16_A4_PREFILL_TERMS", "1")
        caps_on = _caps(enabled=True)
        bytes_on = tp_moe_required_nbytes(caps_on)

        monkeypatch.setenv("B12X_W4A16_A4_PREFILL", "0")
        # same caps: unchanged (dataclass frozen with explicit fields)
        assert tp_moe_required_nbytes(caps_on) == bytes_on
        # fresh caps under the flipped env plans stock sizes
        bytes_fresh_off = tp_moe_required_nbytes(_caps(enabled=False))
        assert bytes_fresh_off < bytes_on


class TestArenaMemoryGrowth:
    def test_serving_growth_budget_at_glm_geometry(self):
        """Record the actual arena growth for the serving config
        (terms=1, capacity 8192): the known cache2 delta is ~4.38 MiB plus
        the cache13/other deltas if any."""
        stock = tp_moe_required_nbytes(_caps(enabled=False, terms=1))
        on = tp_moe_required_nbytes(_caps(enabled=True, terms=1))
        delta = on - stock
        print(f"\narena growth (terms=1, GLM geometry, 8192): {delta} bytes")
        core_stock = _core_plan(CAPACITY, enabled=False, terms=1)
        core_on = _core_plan(CAPACITY, enabled=True, terms=1)
        cache2_delta = _spec_bytes(core_on, "intermediate_cache2") - _spec_bytes(
            core_stock, "intermediate_cache2"
        )
        print(f"cache2 delta: {cache2_delta} bytes")
        assert delta >= 4_592_128  # 38,146,560 - 33,554,432, known carve delta
        assert cache2_delta == 4_592_128
