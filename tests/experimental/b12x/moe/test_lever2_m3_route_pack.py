"""Lever-2 M3a: O(N) stable route-pack design models, CPU-only.

The Triton kernels cannot execute here; these tests model the exact arithmetic
of ``_pack_topk_routes_block_hist_kernel`` + ``_pack_topk_routes_block_prefix_kernel``
against the legacy O(num_blocks^2) rescan loop inside
``_pack_topk_routes_stable_sort_kernel``, and prove the scatter output is
bit-identical for both ``before`` sources. They are the oracle the later GPU
harness (harness/verify_m3a_route_pack.py) compares real kernels against.
"""

import random

import pytest

torch = pytest.importorskip("torch")

from b12x.moe._shared.kernels.w4a16.route_pack import (
    block_counts_usable,
    route_block_counts_elements,
)

pytestmark = pytest.mark.cpu_only

BLOCK_T = 256


def _resolve_ids(ids, expert_map, num_experts):
    """Mirror the kernels' expert-id resolution (mask + optional expert_map)."""
    resolved = []
    for raw in ids:
        valid = 0 <= raw < num_experts
        e = raw
        if expert_map is not None and valid:
            e = expert_map[raw]
            valid = 0 <= e < num_experts
        resolved.append(e if valid else num_experts)
    return resolved


def _legacy_before(resolved, pid, block_t, num_experts):
    """Model of the stable kernel's legacy rescan loop (O(num_blocks^2))."""
    before = [0] * num_experts
    for block in range(0, pid):
        chunk = resolved[block * block_t : (block + 1) * block_t]
        for e in chunk:
            if e < num_experts:
                before[e] += 1
    return before


def _block_hist_rows(resolved, block_t, num_experts):
    """Model of _pack_topk_routes_block_hist_kernel (one row per block)."""
    rows = []
    for start in range(0, len(resolved), block_t):
        row = [0] * num_experts
        for e in resolved[start : start + block_t]:
            if e < num_experts:
                row[e] += 1
        rows.append(row)
    return rows


def _block_prefix(rows, num_experts):
    """Model of _pack_topk_routes_block_prefix_kernel (in-place exclusive scan)."""
    running = [0] * num_experts
    out = []
    for row in rows:
        out.append(list(running))
        for e in range(num_experts):
            running[e] += row[e]
    return out


@pytest.mark.parametrize("use_map", [False, True])
@pytest.mark.parametrize(
    "live,block_t,experts",
    [
        (8, BLOCK_T, 256),
        (256, BLOCK_T, 256),
        (1024, BLOCK_T, 256),
        (4096, BLOCK_T, 256),
        (300, BLOCK_T, 8),
    ],
)
def test_block_counts_before_matches_legacy_rescan(live, block_t, experts, use_map):
    rng = random.Random(live * 31 + experts + int(use_map))
    expert_map = None
    if use_map:
        expert_map = [
            rng.randrange(-1, experts) if rng.random() < 0.2 else i % experts
            for i in range(experts)
        ]
    ids = [rng.randrange(-1, experts + 2) for _ in range(live)]
    resolved = _resolve_ids(ids, expert_map, experts)
    rows = _block_hist_rows(resolved, block_t, experts)
    prefix = _block_prefix(rows, experts)
    num_blocks = (live + block_t - 1) // block_t
    for pid in range(num_blocks):
        assert prefix[pid] == _legacy_before(resolved, pid, block_t, experts)


def _stable_scatter(ids, block_t, experts, before_source):
    """Shared scatter model: placement depends on `before` only through sums."""
    resolved = _resolve_ids(ids, None, experts)
    live = len(ids)
    num_blocks = (live + block_t - 1) // block_t
    rows = _block_hist_rows(resolved, block_t, experts)
    prefix = _block_prefix(rows, experts)
    legacy = [
        _legacy_before(resolved, pid, block_t, experts) for pid in range(num_blocks)
    ]
    order = sorted(
        (i for i, e in enumerate(resolved) if e < experts),
        key=lambda i: (resolved[i], i),
    )
    counts = [0] * experts
    for i in order:
        counts[resolved[i]] += 1
    offsets = []
    base = 0
    for e in range(experts):
        offsets.append(base)
        base += ((counts[e] + block_t - 1) // block_t) * block_t
    slots = [live] * base
    for pid in range(num_blocks):
        before = legacy[pid] if before_source == "legacy" else prefix[pid]
        # Rank within the block's same-expert runs, in route order.
        within = [0] * experts
        for i in range(pid * block_t, min((pid + 1) * block_t, live)):
            e = resolved[i]
            if e < experts:
                slots[offsets[e] + before[e] + within[e]] = i
                within[e] += 1
    return slots


@pytest.mark.parametrize("live", [512, 1024, 4096])
def test_m3a_scatter_output_is_bit_identical(live):
    rng = random.Random(live)
    ids = [rng.randrange(-1, 256 + 2) for _ in range(live)]
    assert _stable_scatter(ids, BLOCK_T, 256, "legacy") == _stable_scatter(
        ids, BLOCK_T, 256, "prefix"
    )


# ---------------------------------------------------------------------------
# Scratch sizing + gating (the real helper functions).
# ---------------------------------------------------------------------------

def test_route_block_counts_elements_sizing():
    # Hy4 decode caps: tiny (64 routes fit in a single sort block).
    assert route_block_counts_elements(64, 8, 256) == 1 * 256
    # Prefill R=2048: 16384 routes -> 64 blocks x 256 experts = 64 KiB.
    assert route_block_counts_elements(2048 * 8, 8, 256) == 64 * 256
    # Prefill R=4096: 32768 routes -> 128 blocks x 256 experts = 128 KiB.
    assert route_block_counts_elements(4096 * 8, 8, 256) == 128 * 256
    # Row budget is capped at _BLOCK_PREFIX_MAX_BLOCKS (256 rows).
    assert route_block_counts_elements(16384 * 8, 8, 256) == 256 * 256


def test_block_counts_usable_gating():
    class _Fake:
        def __init__(self, n):
            self._n = n

        def numel(self):
            return self._n

    scratch = _Fake(128 * 256)
    assert block_counts_usable(
        scratch, num_blocks=128, num_experts=256, stable=True
    )
    assert not block_counts_usable(
        scratch, num_blocks=128, num_experts=256, stable=False
    )
    assert not block_counts_usable(
        None, num_blocks=128, num_experts=256, stable=True
    )
    assert not block_counts_usable(
        scratch, num_blocks=129, num_experts=256, stable=True
    )
    assert not block_counts_usable(
        scratch, num_blocks=129, num_experts=257, stable=True
    )
    assert not block_counts_usable(
        scratch, num_blocks=0, num_experts=256, stable=True
    )


def test_block_counts_usable_min_sort_blocks_guard():
    """M3a crossover guard (junction item 4b): `num_blocks >= 32` sort blocks.

    Decode/prefill row math at topk=8, `_SORT_BLOCK_T=256`:
    R=256 -> 8 blocks, R=512 -> 16 blocks (legacy 1-launch path is 1-2us
    faster there per the SPARE graph A/B), R=1024 -> 32 blocks (M3a wins).
    """
    class _Fake:
        def __init__(self, n):
            self._n = n

        def numel(self):
            return self._n

    big = _Fake(256 * 256)
    assert not block_counts_usable(
        big, num_blocks=8, num_experts=256, stable=True
    )  # R=256: legacy
    assert not block_counts_usable(
        big, num_blocks=16, num_experts=256, stable=True
    )  # R=512: legacy
    assert not block_counts_usable(
        big, num_blocks=31, num_experts=256, stable=True
    )  # just below the guard
    assert block_counts_usable(
        big, num_blocks=32, num_experts=256, stable=True
    )  # R=1024: M3a
    assert block_counts_usable(
        big, num_blocks=64, num_experts=256, stable=True
    )  # R=2048: M3a
