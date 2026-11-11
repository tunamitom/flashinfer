"""CPU tests for the TP8 two-island all-reduce runtime and its owner surface.

No GPU, no peer processes: construction math, peer maps, the capacity envelope
it shares with PCIeTwoShotBF16, prepared-launch ABI and teardown ordering are
exercised with the same mock pattern as test_pcie_hierarchical's island tests.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from b12x.comm.pcie import pcie_island_ar
from b12x.comm.pcie._island_ar_cute import (
    HEADER_BYTES,
    MAX_BLOCKS,
    island_ar_peers,
)
from b12x.comm.pcie.pcie_island_ar import (
    PCIeIslandARAllReduce,
    SUPPORTED_WORLD_SIZES,
    _pick_blocks,
)


def test_island_ar_peers_map_island_lanes_and_the_uplink_partner() -> None:
    assert SUPPORTED_WORLD_SIZES == (8,)
    assert island_ar_peers(0, 8) == (1, 2, 3, 4)
    assert island_ar_peers(3, 8) == (0, 1, 2, 7)
    assert island_ar_peers(4, 8) == (0, 5, 6, 7)
    assert island_ar_peers(5, 8) == (1, 4, 6, 7)
    for rank in range(8):
        peers = island_ar_peers(rank, 8)
        assert len(peers) == 4 and rank not in peers
        # Three island lanes plus the same lane across the uplink.
        assert sum(peer // 4 == rank // 4 for peer in peers) == 3
        assert [peer for peer in peers if peer // 4 != rank // 4] == [
            rank % 4 + 4 if rank < 4 else rank % 4
        ]
    with pytest.raises(ValueError, match="TP8"):
        island_ar_peers(0, 16)
    with pytest.raises(ValueError, match="invalid rank"):
        island_ar_peers(8, 8)


def test_island_ar_blocks_scales_to_the_contract_shapes() -> None:
    assert _pick_blocks(6) == 8
    assert _pick_blocks(4096) == 8
    assert _pick_blocks(1 << 20) == 16
    assert _pick_blocks(3072 * 7168) == 32
    assert (
        max(_pick_blocks(elements) for elements in (3072 * 7168, 3072 * 3584))
        <= MAX_BLOCKS
    )


def _constructed(
    monkeypatch, *, world=8, rank=0, max_rows=16, row_elems=64, blocks=None, ipc=None
):
    """Build the real runtime object against mocked IPC and process groups."""
    shared = SimpleNamespace(
        local_ptr=303,
        remote_ptrs=[101, 102, 103, 104],
        peer_ptrs=(303, 101, 102, 103, 104),
    )
    monkeypatch.setattr(pcie_island_ar.dist, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(pcie_island_ar.dist, "get_world_size", lambda group=None: world)
    monkeypatch.setattr(
        pcie_island_ar.PCIeOneshotAllReduce,
        "_allocate_shared_buffer",
        staticmethod(lambda *args, **kwargs: shared),
    )
    monkeypatch.setattr(
        pcie_island_ar, "_finish_collective_runtime_setup", lambda **kwargs: None
    )
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    fake = MagicMock()
    return PCIeIslandARAllReduce(
        exchange_group=object(),
        device=torch.device("cuda:0"),
        max_rows=max_rows,
        row_elems=row_elems,
        blocks=blocks,
        ipc=ipc if ipc is not None else fake,
    )


def test_island_ar_construction_slab_layout_and_peer_map(monkeypatch) -> None:
    runtime = _constructed(monkeypatch, max_rows=16, row_elems=64)
    assert runtime.max_elements == 16 * 64
    assert runtime.mapped_peers == island_ar_peers(0, 8)
    assert runtime.mapped_peer_count == 4  # three island lanes plus the uplink partner
    quarter_bytes = runtime.quarter_capacity * 4
    vector_bytes = quarter_bytes * 4
    assert runtime.stage_offset == (HEADER_BYTES + 255) // 256 * 256
    assert runtime.part_offset % 256 == 0 and runtime.final_offset % 256 == 0
    assert runtime.slab_bytes == (
        (runtime.final_offset + 2 * vector_bytes + 255) // 256 * 256
    )
    # Double-buffered regions: a generation stride is one vector for stage/final.
    assert runtime.slab_bytes >= runtime.final_offset + 2 * vector_bytes


def test_island_ar_requires_tp8_and_positive_even_capacity(monkeypatch) -> None:
    with pytest.raises(ValueError, match="TP8"):
        _constructed(monkeypatch, world=16)
    monkeypatch.setattr(pcie_island_ar.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(pcie_island_ar.dist, "get_world_size", lambda group=None: 8)
    with pytest.raises(ValueError, match="positive integers"):
        _constructed(monkeypatch, max_rows=0)
    with pytest.raises(ValueError, match="blocks must be one of"):
        _constructed(monkeypatch, blocks=64)
    with pytest.raises(ValueError, match="CUDA device"):
        _constructed(monkeypatch)
        monkeypatch.setattr(
            pcie_island_ar, "_normalize_device", lambda device: torch.device("cpu")
        )
        _constructed(monkeypatch)


def test_island_ar_accepts_matches_the_twoshot_capacity_envelope(monkeypatch) -> None:
    runtime = _constructed(monkeypatch, max_rows=16, row_elems=64)
    runtime.device = torch.device("cpu")
    good = torch.zeros(16, 64, dtype=torch.bfloat16)
    assert runtime.accepts(good) and runtime.should_allreduce(good)
    assert runtime.accepts(torch.zeros(8, 128, dtype=torch.bfloat16))
    assert not runtime.accepts(torch.zeros(16, 65, dtype=torch.bfloat16))  # odd count
    assert not runtime.accepts(
        torch.zeros(33, 64, dtype=torch.bfloat16)
    )  # beyond the envelope
    assert not runtime.accepts(torch.zeros(16, 64))  # not BF16
    assert not runtime.accepts(torch.zeros(16, 65)[:, :64])  # not contiguous
    assert not runtime.accepts(torch.zeros(0, dtype=torch.bfloat16))


def test_island_ar_all_reduce_validates_and_runs_the_prepared_launcher(
    monkeypatch,
) -> None:
    runtime = _constructed(monkeypatch)
    runtime.device = torch.device("cpu")
    runtime._launcher = MagicMock()
    state = MagicMock()
    state.query.call = {"inp": {"shape": (8, 64)}, "blocks": 8}
    state.launcher.return_value = runtime._launcher
    monkeypatch.setattr(pcie_island_ar, "require_prepared", lambda *_: state)
    inp = torch.zeros(8, 64, dtype=torch.bfloat16)
    out = torch.empty_like(inp)
    result = runtime.all_reduce(inp, plan=object(), out=out)
    assert result is out
    state.require_runtime.assert_called_once_with(runtime)
    runtime._launcher.assert_called_once()
    args = runtime._launcher.call_args[0]
    assert args[1] == inp.data_ptr() and args[2] == out.data_ptr()
    assert (
        args[6] == runtime.quarter_capacity and args[7] == inp.numel() and args[8] == 8
    )
    with pytest.raises(ValueError, match="shape differs from the prepared island plan"):
        runtime.all_reduce(
            torch.zeros(4, 64, dtype=torch.bfloat16), plan=object(), out=out
        )
    with pytest.raises(ValueError, match="island all-reduce requirements"):
        runtime.all_reduce(torch.zeros(65, dtype=torch.bfloat16), plan=object())
    with pytest.raises(ValueError, match="blocks differs"):
        runtime.all_reduce(inp, plan=object(), out=out, blocks=16)


def test_island_ar_capture_allocates_implicit_output(monkeypatch) -> None:
    runtime = _constructed(monkeypatch)
    runtime.device = torch.device("cpu")
    runtime._launcher = MagicMock()
    state = MagicMock()
    state.query.call = {"inp": {"shape": (8, 64)}, "blocks": 8}
    state.launcher.return_value = runtime._launcher
    monkeypatch.setattr(pcie_island_ar, "require_prepared", lambda *_: state)
    inp = torch.zeros(8, 64, dtype=torch.bfloat16)
    with runtime.capture() as captured:
        out = captured.all_reduce(inp, plan=object())
    assert captured is runtime
    assert out.shape == inp.shape and out.dtype == inp.dtype
    runtime._launcher.assert_called_once()


def test_island_ar_close_coordinates_ipc_teardown_and_is_idempotent(
    monkeypatch,
) -> None:
    runtime = _constructed(monkeypatch)
    runtime.device = torch.device("cuda:0")
    runtime.group = object()
    runtime._slab_ptrs = (101, 102, 303)
    runtime._remote_ptrs = [101, 102]
    runtime._local_ptr = 303
    events = []

    def close_remote(ptr):
        assert runtime._slab_ptrs == ()
        events.append(("close", ptr))

    def free_local(ptr):
        assert runtime._remote_ptrs == []
        events.append(("free", ptr))

    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: events.append(("synchronize", device))
    )
    monkeypatch.setattr(
        pcie_island_ar.dist,
        "barrier",
        lambda *, group: events.append(("barrier", group)),
    )
    runtime._ipc.cudaIpcCloseMemHandle.side_effect = close_remote
    runtime._ipc.cudaFree.side_effect = free_local
    runtime.close()
    runtime.close()
    assert events == [
        ("synchronize", runtime.device),
        ("barrier", runtime.group),
        ("close", 101),
        ("close", 102),
        ("barrier", runtime.group),
        ("free", 303),
        ("barrier", runtime.group),
    ]
    assert runtime._closed and runtime._slab_ptrs == ()
    assert runtime._remote_ptrs == [] and runtime._local_ptr == 0


def test_island_ar_owner_surface_declares_native_metadata(monkeypatch) -> None:
    from b12x.comm.pcie import _owner_preparation, _preparation, _tuning
    from b12x.comm.pcie.pcie_island_ar import _pick_blocks as pick

    assert _tuning.SURFACES["PCIeIslandARAllReduce.all_reduce"] == (8,)
    assert "PCIeIslandARAllReduce.all_reduce" in _preparation._OWNER_SURFACES
    runtime = _constructed(monkeypatch, max_rows=16, row_elems=64)
    source = torch.zeros(8, 64, dtype=torch.bfloat16)
    output = torch.empty_like(source)
    query = _owner_preparation.query_from_runtime(
        runtime,
        surface="PCIeIslandARAllReduce.all_reduce",
        call=dict(inp=source, out=output),
    )
    assert query.surface == "PCIeIslandARAllReduce.all_reduce"
    assert query.topology == "pcie_ipc" and query.world_size == 8
    assert query.call["blocks"] == pick(source.numel())
    assert query.call["inp"]["shape"] == (8, 64)
    assert query.call["output"]["shape"] == (8, 64)
    assert query.setup["max_elements"] == runtime.max_elements
    assert query.setup["slab_bytes"] == runtime.slab_bytes
    assert query.setup["peer_ranks"] == runtime.mapped_peers
    plan = _owner_preparation.plan(query, runtime=runtime)
    prepared = _owner_preparation.prepared_call(
        SimpleNamespace(
            launcher=lambda: runtime._launcher, query=query, runtime=runtime
        ),
        inp=source,
        out=output,
    )
    assert prepared.output is output and prepared.capture_safe is True
    # The TP16 surface remains registered beside it (the reference is untouched).
    assert _tuning.SURFACES["PCIeIslandRSAllReduce.all_reduce"] == (16,)


def test_island_ar_query_rejects_non_eight_worlds(monkeypatch) -> None:
    from b12x.comm.pcie import _tuning

    query = _tuning.PcieQuery(
        surface="PCIeIslandARAllReduce.all_reduce",
        world_size=16,
        rank=0,
        topology="pcie_ipc",
        call=_tuning.FrozenMapping({}),
        setup=_tuning.FrozenMapping({}),
    )
    with pytest.raises(ValueError, match=r"supports world sizes \(8,\)"):
        _tuning.TUNING.validate_query(query, None)
