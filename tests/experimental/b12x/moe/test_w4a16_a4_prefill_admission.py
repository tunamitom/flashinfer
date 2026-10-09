# SPDX-License-Identifier: Apache-2.0
"""Admission regressions for the NVFP4 A4 prefill workspace reservation.

The stock planner sized ``intermediate_cache2`` as ``routed_capacity *
local_intermediate_size`` elements while the A4 prefill pipeline
(``prefill_a4._carve_layout``) carves quantized input/intermediate planes,
their scales, and padded route metadata out of that same buffer. At GLM's
geometry (E=256, H=6144, I=256, topk=8) and 8192-token capacity the carve
needs 38,146,560 bytes against a 33,554,432-byte stock allocation, so
``select_a4()``/``fits_buffers()`` silently fell back to W4A16 for near-full
chunks (only 7195 live tokens fit). These tests pin the plan-time reservation
(max of the A16 and A4 requirements, sized by the kernel's own carve) and the
admission outcome at full/near-full capacity.

Host-side tests are CPU-only and exercise plan-time sizing and the selector
contract. GPU tests bind real prepared weights at GLM's geometry and assert
the bound launch selection; they skip without a consumer-Blackwell GPU.
"""

from __future__ import annotations

import math

import pytest
import torch

from b12x.moe._shared.kernels.w4a16.host import (
    max_packed_route_slots,
    route_pack_token_capacity,
)
from b12x.moe._shared.kernels.w4a16.prefill_a4 import (
    A4_PREFILL_ROUTE_BLOCK,
    a4_prefill_sizing_launches,
)

# GLM-5.3 TP8 routed-expert geometry: 256 experts, hidden 6144, 256
# intermediate channels per TP8 rank, top-k 8.
E, H, I, TOPK = 256, 6144, 256, 8
CAPACITIES = (2048, 4096, 8192)

# Ground truth: the compiled launches' own carve at this geometry (verified
# byte-exact against W4A16A4PrefillLaunches.scratch_bytes, 2026-10-09 probe).
CACHE2_TERMS1 = {2048: 9_636_864, 4096: 19_140_096, 8192: 38_146_560}
CACHE2_TERMS2 = {2048: 19_074_048, 4096: 38_014_464, 8192: 75_895_296}
CACHE13 = {2048: 201_326_592, 4096: 402_653_184, 8192: 805_306_368}
STOCK_CACHE2 = {2048: 8_388_608, 4096: 16_777_216, 8192: 33_554_432}


def _route_pack_capacity(capacity):
    """Route-pack slots/blocks the A4 launch set compiles with: the pack
    buckets tokens to the next power of two and pads every expert's route
    block (A4_PREFILL_ROUTE_BLOCK)."""
    numel_cap = route_pack_token_capacity(capacity, TOPK) * TOPK
    packed_routes = max(
        max_packed_route_slots(numel_cap, A4_PREFILL_ROUTE_BLOCK, E), 1
    )
    route_blocks = (packed_routes + A4_PREFILL_ROUTE_BLOCK - 1) // A4_PREFILL_ROUTE_BLOCK
    return packed_routes, route_blocks


def _sizing_launch(capacity, terms=1):
    packed_routes, route_blocks = _route_pack_capacity(capacity)
    return a4_prefill_sizing_launches(
        tokens=capacity,
        hidden_size=H,
        intermediate_size=I,
        num_experts=E,
        topk=TOPK,
        terms=terms,
        max_packed_routes=packed_routes,
        max_route_blocks=route_blocks,
    )


def _plan_sizes(capacity, *, enabled, terms=1, fused_sum=False):
    """Plan the GLM core workspace on CPU and return buffer sizes in bytes."""
    from b12x.moe.fused_moe import _impl as impl

    core = impl._plan_core_workspace(
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
    return {
        spec.name: math.prod(spec.shape) * spec.dtype.itemsize
        for spec in core.tensor_specs
    }


class TestPlannerReservation:
    """The plan-time reservation must cover the kernel's own carve."""

    @pytest.mark.parametrize("capacity", CAPACITIES)
    def test_cache2_reserved_matches_kernel_carve_terms1(self, capacity):
        sizes = _plan_sizes(capacity, enabled=True, terms=1)
        assert sizes["intermediate_cache2"] == CACHE2_TERMS1[capacity]
        # At 8192 the spec pins the requirement to the byte.
        if capacity == 8192:
            assert sizes["intermediate_cache2"] == 38_146_560

    @pytest.mark.parametrize("capacity", CAPACITIES)
    def test_cache2_reserved_matches_kernel_carve_terms2(self, capacity):
        sizes = _plan_sizes(capacity, enabled=True, terms=2)
        assert sizes["intermediate_cache2"] == CACHE2_TERMS2[capacity]

    @pytest.mark.parametrize("capacity", CAPACITIES)
    def test_cache13_reserved_covers_route_rows(self, capacity):
        for terms in (1, 2):
            sizes = _plan_sizes(capacity, enabled=True, terms=terms)
            assert sizes["intermediate_cache13"] == CACHE13[capacity]

    @pytest.mark.parametrize("capacity", CAPACITIES)
    @pytest.mark.parametrize("terms", [1, 2])
    def test_reserved_buffers_admit_full_capacity(self, capacity, terms):
        """fits_buffers() (the compiled launch's own admission check) must
        pass for the A4 pipeline at full prepared capacity."""
        sizes = _plan_sizes(capacity, enabled=True, terms=terms)
        launch = _sizing_launch(capacity, terms)
        assert launch.fits_buffers(
            capacity, sizes["intermediate_cache13"], sizes["intermediate_cache2"]
        )
        assert launch.scratch_bytes(capacity) == sizes["intermediate_cache2"]
        assert launch.cache13_bytes(capacity) <= sizes["intermediate_cache13"]

    @pytest.mark.parametrize("capacity", CAPACITIES)
    @pytest.mark.parametrize("terms", [1, 2])
    def test_reserved_buffers_admit_near_full_capacity(self, capacity, terms):
        sizes = _plan_sizes(capacity, enabled=True, terms=terms)
        launch = _sizing_launch(capacity, terms)
        for live in (1, 512, capacity // 2, capacity - 25, capacity - 1):
            assert launch.fits_buffers(
                live, sizes["intermediate_cache13"], sizes["intermediate_cache2"]
            ), live

    def test_fused_sum_plan_still_reserves_cache13(self):
        """With the prefill fused sum enabled the A16 cache13 shrinks to the
        FP32 accumulator budget; the A4 per-route BF16 rows must still fit."""
        sizes = _plan_sizes(8192, enabled=True, terms=1, fused_sum=True)
        launch = _sizing_launch(8192, 1)
        assert sizes["intermediate_cache13"] >= CACHE13[8192]
        assert launch.fits_buffers(
            8192, sizes["intermediate_cache13"], sizes["intermediate_cache2"]
        )


class TestStockBehaviorUnchanged:
    """Flag-off plans keep the exact stock allocation, and that stock plan is
    the reproduced admission blocker (regression pin)."""

    @pytest.mark.parametrize("capacity", CAPACITIES)
    def test_flag_off_allocation_is_stock(self, capacity):
        sizes = _plan_sizes(capacity, enabled=False)
        assert sizes["intermediate_cache2"] == STOCK_CACHE2[capacity]
        assert sizes["intermediate_cache13"] == CACHE13[capacity]

    @pytest.mark.parametrize("capacity", CAPACITIES)
    def test_stock_cache2_is_routed_capacity_times_intermediate(self, capacity):
        sizes = _plan_sizes(capacity, enabled=False)
        assert sizes["intermediate_cache2"] == capacity * TOPK * I * 2

    def test_stock_plan_reproduces_the_admission_blocker(self):
        """The pre-fix contract: at 8192 stock buffers admit at most 7195 live
        A4 tokens, so full chunks fall back to W4A16."""
        sizes = _plan_sizes(8192, enabled=False)
        launch = _sizing_launch(8192, 1)
        assert not launch.fits_buffers(
            8192, sizes["intermediate_cache13"], sizes["intermediate_cache2"]
        )
        fitting = [
            t
            for t in range(1, 8193)
            if launch.fits_buffers(
                t, sizes["intermediate_cache13"], sizes["intermediate_cache2"]
            )
        ]
        assert max(fitting) == 7195

    def test_reserved_plan_admits_every_live_count(self):
        sizes = _plan_sizes(8192, enabled=True, terms=1)
        launch = _sizing_launch(8192, 1)
        assert all(
            launch.fits_buffers(
                t, sizes["intermediate_cache13"], sizes["intermediate_cache2"]
            )
            for t in range(1, 8193)
        )


class TestSelectorContract:
    """select_a4 preserves explicit precision: force=True is required, and
    short buffers still reject (fits_buffers is never bypassed)."""

    def _selector(self, capacity):
        from b12x.moe.fused_moe._preparation import _W4A16PrimaryLaunches

        return _W4A16PrimaryLaunches(
            tokens=capacity,
            route_mode="packed",
            packed=None,
            packed_mapped=None,
            direct=None,
            direct_mapped=None,
            topk_sum=None,
            mapped_topk_sum=None,
            route_pack=None,
            a4_prefill=_sizing_launch(capacity, 1),
        )

    def test_explicit_a4_admitted_at_full_capacity(self):
        sizes = _plan_sizes(8192, enabled=True, terms=1)
        selector = self._selector(8192)
        args = dict(
            tokens=8192,
            route_ids_dtype=torch.int32,
            has_route_map=False,
            activation_amax=None,
            apply_router_weight_on_input=False,
            cache13_bytes=sizes["intermediate_cache13"],
            cache2_bytes=sizes["intermediate_cache2"],
            force=True,
        )
        assert selector.select_a4(**args) is not None
        # Non-explicit selection keeps W4A16 regardless of buffers.
        assert selector.select_a4(**{**args, "force": None}) is None
        assert selector.select_a4(**{**args, "force": False}) is None

    def test_short_buffers_still_reject(self):
        """A stock-sized cache2 must not be admitted even when A4 is forced:
        the reservation fixes the plan, fits_buffers stays authoritative."""
        sizes = _plan_sizes(8192, enabled=False)
        selector = self._selector(8192)
        assert (
            selector.select_a4(
                tokens=8192,
                route_ids_dtype=torch.int32,
                has_route_map=False,
                activation_amax=None,
                apply_router_weight_on_input=False,
                cache13_bytes=sizes["intermediate_cache13"],
                cache2_bytes=sizes["intermediate_cache2"],
                force=True,
            )
            is None
        )


# ---------------------------------------------------------------------------
# GPU admission: real prepared weights, real binds, GLM geometry.
# ---------------------------------------------------------------------------


def _gpu_case(monkeypatch, *, enabled, terms, capacity, csf=False, seed=11):
    from .test_w4a16_a4_prefill import (
        _bind,
        _env,
        _inputs,
        _make_case,
        _plan,
        _prepare,
    )

    _env(monkeypatch, enabled, terms)
    case = _make_case(seed, I, H, E, TOPK)
    # Calibrated activation globals retain the A4 scales; exact values are an
    # admission matter here, not a numeric one.
    a1g = torch.full((E,), 4.0, device="cuda")
    a2g = torch.full((E,), 4.0, device="cuda")
    experts = _prepare(case, a1g, a2g, csf=csf)
    xp = _plan(experts, capacity, top_k=TOPK)
    dev = torch.device("cuda")
    scratch = tuple(
        torch.empty(s.shape, dtype=s.dtype, device=dev) for s in xp.scratch_specs()
    )
    out = torch.empty(capacity, H, device=dev, dtype=torch.bfloat16)
    return case, experts, xp, scratch, out, _bind, _inputs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "capacity,terms",
    [(2048, 1), (4096, 1), (8192, 1), (8192, 2)],
)
def test_a4_binds_full_capacity_on_gpu(monkeypatch, capacity, terms):
    """Explicit a4_prefill=True must bind an A4 launch at full prepared
    capacity (and near-full live counts) once the plan reserves the workspace.
    """
    from ..conftest import require_b12x

    require_b12x()
    _, experts, xp, scratch, out, _bind, _inputs = _gpu_case(
        monkeypatch, enabled=True, terms=terms, capacity=capacity
    )
    assert experts._impl.a4_prefill_scales
    for live in (capacity - 25, capacity):
        x, ids, wts = _inputs(live, live, H, E, TOPK)
        binding = _bind(xp, x, ids, wts, out[:live], scratch, a4_prefill=True)
        launch = binding.a4_prefill_launches
        assert launch is not None, live
        assert launch.terms == terms
        cache2_bytes = (
            binding.intermediate_cache2.numel()
            * binding.intermediate_cache2.element_size()
        )
        cache13_bytes = (
            binding.intermediate_cache13.numel()
            * binding.intermediate_cache13.element_size()
        )
        assert launch.fits_buffers(live, cache13_bytes, cache2_bytes)
        if live == capacity:
            expected = CACHE2_TERMS1[capacity] if terms == 1 else CACHE2_TERMS2[capacity]
            assert launch.scratch_bytes(live) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flag_off_and_explicit_a16_bind_a16_with_stock_buffers(monkeypatch):
    """Flag-off keeps stock buffers and never binds A4; an A4-flagged plan
    with explicit a4_prefill=False/None still binds the A16 launch."""
    from ..conftest import require_b12x

    require_b12x()
    # Flag off: stock allocation, A4 never selected even when requested.
    case, experts, xp, scratch, out, _bind, _inputs = _gpu_case(
        monkeypatch, enabled=False, terms=1, capacity=8192
    )
    x, ids, wts = _inputs(8192, 5, H, E, TOPK)
    binding = _bind(xp, x, ids, wts, out, scratch, a4_prefill=True)
    assert binding.a4_prefill_launches is None
    cache2_bytes = (
        binding.intermediate_cache2.numel()
        * binding.intermediate_cache2.element_size()
    )
    assert cache2_bytes == STOCK_CACHE2[8192]
    assert binding.fused_launch is not None

    # Flag on, explicit A16: the enlarged plan still binds W4A16.
    _, experts, xp, scratch, out, _bind, _inputs = _gpu_case(
        monkeypatch, enabled=True, terms=1, capacity=8192
    )
    for choice in (False, None):
        binding = _bind(xp, x, ids, wts, out, scratch, a4_prefill=choice)
        assert binding.a4_prefill_launches is None
        assert binding.fused_launch is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_csf_scales_admit_full_capacity_on_gpu(monkeypatch):
    """CSF-compressed NVFP4 weights at GLM geometry: full-capacity A4 binding
    with both the inline and the expanded scale readers (scales_expanded
    selects the expanded launch above the CSF stage limit)."""
    from ..conftest import require_b12x

    require_b12x()
    _, experts, xp, scratch, out, _bind, _inputs = _gpu_case(
        monkeypatch, enabled=True, terms=1, capacity=8192, csf=True, seed=13
    )
    assert experts._impl.a4_prefill_scales
    x, ids, wts = _inputs(8192, 7, H, E, TOPK)
    for expanded in (False, True):
        binding = _bind(
            xp, x, ids, wts, out, scratch, a4_prefill=True, scales_expanded=expanded
        )
        launch = binding.a4_prefill_launches
        assert launch is not None, expanded
        assert launch.fits_buffers(
            8192,
            binding.intermediate_cache13.numel()
            * binding.intermediate_cache13.element_size(),
            binding.intermediate_cache2.numel()
            * binding.intermediate_cache2.element_size(),
        )