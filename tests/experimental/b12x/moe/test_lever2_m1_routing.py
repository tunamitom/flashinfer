"""Lever-2 M1: CSF expansion default selection + fused-route-pack gate, CPU-only.

Covers the pure-Python logic changed for Track A1:
  * ``_default_direct_routes`` / ``_default_config`` (cap 8/16 planner cliff),
  * ``_fused_route_pack_geometry`` (packed-fusion activation gate),
  * ``B12X_CSF_GROUPS_PER_TASK`` / ``B12X_W4A16_CSF_MIN_TOKENS`` probe hooks,
  * design models: CsfRoutePack placement vs the stable Triton route pack
    (plan M1 item 3 order-equivalence claim) and the CsfRoutePack bitonic
    network (which the CPU cannot execute; these models are the oracle the
    later GPU harness compares real kernels against).
"""

import os

import pytest

torch = pytest.importorskip("torch")

from b12x._lib.quant.csf_blocks_tuning import (
    TUNING,
    CsfExpansionConfig,
    CsfExpansionQuery,
    _default_config,
    _default_direct_routes,
    _default_groups_per_task,
    _uses_presence,
)
from b12x.moe.fused_moe._preparation import _fused_route_pack_geometry
from b12x.moe.fused_moe._tuning import _w4a16_csf_min_tokens

pytestmark = pytest.mark.cpu_only

HY4_GEOMETRY = (128, 4, 4, 4, 128, 4, 256, 0, 1)


def _query(routes, experts=256):
    geometry = (128, 4, 4, 4, 128, 4, experts, 0, 1)
    return CsfExpansionQuery(geometry, geometry, routes, "int32")


# ---------------------------------------------------------------------------
# Default selection: caps 1/4/8/16 (routes 8/32/64/128 at top_k 8) and edges.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "routes,experts,direct,bitmap",
    [
        (8, 256, False, False),      # cap 1: below presence range
        (32, 256, False, False),     # cap 4: below presence range
        (64, 256, True, False),      # cap 8: direct default (M1 fix)
        (128, 256, True, False),     # cap 16: direct default (M1 fix)
        (256, 256, False, True),     # routes == 16*E? no: 256 < 4096 -> presence
        (4096, 256, False, False),   # full-expert mode
        (64, 8, False, True),        # routes > E(=8): direct predicate fails
        (64, 64, True, False),       # routes == E: direct allowed
    ],
)
def test_default_direct_routes_selection(routes, experts, direct, bitmap):
    query = _query(routes, experts)
    assert _uses_presence(query) == (64 <= routes < 16 * experts)
    assert _default_direct_routes(query) is direct
    config = _default_config(query, None)
    assert config.direct_routes is direct
    assert config.bitmap_routes is bitmap
    assert not (config.direct_routes and config.bitmap_routes)
    TUNING.validate_config(query, config, None)


def test_default_config_honors_groups_probe(monkeypatch):
    query = _query(64)
    assert _default_config(query, None).groups_per_task == 8
    monkeypatch.setenv("B12X_CSF_GROUPS_PER_TASK", "16")
    assert _default_groups_per_task() == 16
    assert _default_config(query, None).groups_per_task == 16
    monkeypatch.setenv("B12X_CSF_GROUPS_PER_TASK", "3")
    with pytest.raises(ValueError, match="B12X_CSF_GROUPS_PER_TASK"):
        _default_groups_per_task()


def test_default_config_validates_against_contract():
    for routes in (8, 32, 64, 128, 256, 4096):
        query = _query(routes)
        config = _default_config(query, None)
        TUNING.validate_query(query, None)
        TUNING.validate_config(query, config, None)
        # Every default must be representable by the tuning contract knobs.
        encoded = TUNING.encode_config(config)
        assert TUNING.decode_config(encoded) == config


def test_csf_min_tokens_probe(monkeypatch):
    assert _w4a16_csf_min_tokens() == 16
    monkeypatch.setenv("B12X_W4A16_CSF_MIN_TOKENS", "32")
    assert _w4a16_csf_min_tokens() == 32
    monkeypatch.setenv("B12X_W4A16_CSF_MIN_TOKENS", "0")
    with pytest.raises(ValueError, match="B12X_W4A16_CSF_MIN_TOKENS"):
        _w4a16_csf_min_tokens()


# ---------------------------------------------------------------------------
# Fused CsfRoutePack gate (Hy4: E=256, groups_per_task=8 -> threads=256).
# ---------------------------------------------------------------------------

def _gate(routes, slots, blocks, threads=256, route_e=256, first_experts=256):
    rounded = 1 << (routes - 1).bit_length() if routes > 0 else 0
    return _fused_route_pack_geometry(
        routes=routes,
        slots=slots,
        blocks=blocks,
        rounded=rounded,
        threads=threads,
        route_e=route_e,
        first_experts=first_experts,
        block_m=8,
    )


def test_fused_route_pack_gate_activates_caps_8_and_16():
    # Cap 8: 64 routes, 512 slots; cap 16: 128 routes, 2048 slots.
    assert _gate(64, 512, 64) == (64, 8, 512, 64)
    assert _gate(128, 2048, 128) == (128, 8, 2048, 128)
    # Caps 1/4 stay legal (fused pack may serve direct GEMM modes too).
    assert _gate(8, 64, 8) == (8, 8, 64, 8)
    assert _gate(32, 256, 32) == (32, 8, 256, 32)


def test_fused_route_pack_gate_rejections():
    assert _gate(0, 0, 0) is None
    assert _gate(256, 2048, 256) is None            # routes > 128
    assert _gate(64, 8192, 1024) is None            # slots > 4096
    assert _gate(128, 2048, 128, threads=64) is None  # rounded > threads
    assert _gate(64, 512, 64, first_experts=128) is None  # expert mismatch
    assert _gate(4096, 4096, 256, route_e=256) is None    # routes >= 16*E
    # Overflow guard with a huge expert namespace.
    assert _gate(128, 2048, 128, route_e=2**24, first_experts=2**24) is None


# ---------------------------------------------------------------------------
# Design models (plan M1 item 3): CsfRoutePack vs stable Triton placement.
# These are reference/oracle models of DSL kernels the CPU cannot execute.
# ---------------------------------------------------------------------------

def _stable_triton_reference(ids, block_m, num_experts):
    """Placement spec of the stable Triton route pack (counting sort).

    Experts own block_m-padded segments in expert-id order; within a segment
    routes keep ascending route-index order. ``packed_route_count`` is the sum
    of padded segment sizes (parallel path contract).
    """
    live = len(ids)
    order = sorted(
        (i for i, e in enumerate(ids) if 0 <= e < num_experts),
        key=lambda i: (ids[i], i),
    )
    counts = [0] * num_experts
    for i in order:
        counts[ids[i]] += 1
    offsets = []
    base = 0
    for e in range(num_experts):
        offsets.append(base)
        base += ((counts[e] + block_m - 1) // block_m) * block_m
    total = base
    slots = [live] * total
    rank = [0] * num_experts
    for i in order:
        e = ids[i]
        slots[offsets[e] + rank[e]] = i
        rank[e] += 1
    block_experts = []
    for b in range(total // block_m):
        row = b * block_m
        owner = -1
        for e in range(num_experts):
            if offsets[e] <= row:
                owner = e
        block_experts.append(owner if row < total else -1)
    return slots, block_experts, total, offsets, counts


def _csf_route_pack_reference(ids, block_m, num_experts):
    """Placement spec of the fused CsfRoutePack (bitonic keys, aligned groups).

    Sort keys ``expert * rounded + tid`` place each expert's routes in
    route-index order; group g starts where the previous group's live range
    rounds up to a block_m boundary. ``packed_route_count`` ends at the block
    holding the last live route (fused-packer contract).
    """
    live = len(ids)
    keys = sorted(
        ((e if 0 <= e < num_experts else num_experts), i)
        for i, e in enumerate(ids)
    )
    # Group runs in sorted order (invalid routes form the trailing group).
    groups = []
    for expert, i in keys:
        if not groups or groups[-1][0] != expert:
            groups.append((expert, []))
        groups[-1][1].append(i)
    slots = {}
    pad_total = 0
    prev_size = 0
    for gi, (expert, members) in enumerate(groups):
        if gi > 0:
            pad_total += (block_m - prev_size % block_m) % block_m
        start = sum(len(m) for _, m in groups[:gi]) + pad_total
        assert start % block_m == 0
        for k, i in enumerate(members):
            # Invalid routes consume lane slots but never write the array.
            if expert < num_experts:
                slots[start + k] = i
        prev_size = len(members)
    live_slots = sorted(slots)
    count = 0
    if live_slots:
        count = (live_slots[-1] // block_m + 1) * block_m
    # CsfRoutePack writes experts[slot // block_m] at every block_m-th route of
    # each group: exactly the first live route of every block holding routes.
    block_experts = {}
    for slot, i in sorted(slots.items()):
        block_experts.setdefault(slot // block_m, ids[i])
    return slots, block_experts, count


def _bitonic_network_sorts(values):
    """Simulate the CsfRoutePack bitonic exchange network (model).

    Mirrors csf_route_pack.py: levels/steps/distance/take_min over ``rounded``
    lanes; butterfly exchanges for distance < 32, larger distances are exact
    value swaps in this model.
    """
    rounded = len(values)
    assert rounded & (rounded - 1) == 0
    levels = (rounded - 1).bit_length()
    keys = list(values)
    for level in range(1, levels + 1):
        for step in range(level):
            distance = 1 << (level - step - 1)
            for tid in range(rounded):
                other = tid ^ distance
                if other >= rounded or other < tid:
                    continue
                take_min = ((tid & (1 << level)) == 0) == ((tid & distance) == 0)
                pair = (keys[tid], keys[other])
                lo, hi = min(pair), max(pair)
                if take_min:
                    keys[tid], keys[other] = lo, hi
                else:
                    keys[tid], keys[other] = hi, lo
    return keys


@pytest.mark.parametrize("rounded", [8, 16, 32, 64, 128])
def test_csf_route_pack_bitonic_network_sorts(rounded, seed=1234):
    import random

    rng = random.Random(seed + rounded)
    values = [rng.randrange(0, 2**30) for _ in range(rounded)]
    assert _bitonic_network_sorts(values) == sorted(values)


@pytest.mark.parametrize(
    "live,block_m,experts",
    [
        (8, 8, 256),
        (64, 8, 256),
        (128, 16, 256),
        (96, 8, 256),
        (200, 8, 256),
        (65, 8, 8),
    ],
)
def test_csf_route_pack_matches_stable_order(live, block_m, experts, seed=7):
    import random

    rng = random.Random(seed + live)
    ids = [rng.randrange(-1, experts + 1) for _ in range(live)]
    stable_slots, stable_blocks, stable_total, _, _ = _stable_triton_reference(
        ids, block_m, experts
    )
    csf_slots, csf_blocks, csf_count = _csf_route_pack_reference(
        ids, block_m, experts
    )
    # Exact same sorted route order on every slot holding a live route.
    for slot, route in csf_slots.items():
        assert stable_slots[slot] == route
    assert csf_count <= stable_total
    assert csf_count % block_m == 0
    # Every block the fused pack reports with live routes holds the same owner
    # as the stable reference's binary-search block resolution. Padded tail
    # blocks differ by contract (-1 vs owner) and never carry live routes.
    for b, owner in csf_blocks.items():
        if b * block_m < csf_count:
            assert stable_blocks[b] == owner
