"""TP8 two-island all-reduce on GPU: bit-exact vs the pinned CPU reference.

The kernel's strict contract: kernel output must equal the composed-wire
emulation bit-for-bit on the recorded prefill shapes and on the small/odd
shapes below. Also exercises eager launches, graph capture/replay stability
and multi-generation double buffering, mirroring test_pcie_island_rs_gpu.py.

Run (8 GPUs behind two switches):

    B12X_RUN_ISLAND_AR_TEST=1 CUDA_VISIBLE_DEVICES=<8 GPUs> \
        python -m pytest tests/experimental/b12x/comm/test_pcie_island_ar_gpu.py
"""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.comm.pcie import _owner_preparation
from b12x.comm.pcie.pcie_island_ar import PCIeIslandARAllReduce

try:
    from tests.experimental.b12x.comm.test_pcie_island_ar_bitexact import (
        generate,
        island_composed_all_reduce,
    )
except ImportError:  # collected with the comm directory directly on sys.path
    from test_pcie_island_ar_bitexact import generate, island_composed_all_reduce

pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_ISLAND_AR_TEST") != "1",
    reason="set B12X_RUN_ISLAND_AR_TEST=1 to run the TP8 GPU bit-exact test",
)

WORLD_SIZE = 8
MAX_ROWS = 3072
# (rows, cols) coverage: the prefill shapes plus small and odd element counts
# so every _pick_blocks value (8/16/32 blocks) runs, and each small/odd shape
# has pairs = numel/2 with pairs % 4 != 0, which drives the uneven-quarter
# path (a quarter boundary not divisible into equal parts).
#   (3072, 7168)    22_020_096 elems -> 32 blocks, pairs % 4 == 0
#   (3072, 3584)    11_010_048 elems -> 32 blocks, pairs % 4 == 0
#   (1, 2)                  2 elems ->  8 blocks, pairs = 1
#   (1, 4098)            4_098 elems -> 16 blocks, pairs = 2_049
#   (3, 4098)           12_294 elems -> 16 blocks, pairs = 6_147 (odd rows)
#   (1, 1_048_578)   1_048_578 elems -> 32 blocks, pairs = 524_289
SHAPES = (
    (3072, 7168),
    (3072, 3584),
    (1, 2),
    (1, 4098),
    (3, 4098),
    (1, 1_048_578),
)
GRAPH_ROUNDS = 10


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _prepare(session, runtime, inp, out, name):
    query = _owner_preparation.query_from_runtime(
        runtime, surface="PCIeIslandARAllReduce.all_reduce", call=dict(inp=inp, out=out)
    )
    plan = _owner_preparation.plan(query, runtime=runtime)
    session.prepare(
        (
            plan.request(
                name=name,
                prepare_call=lambda state: _owner_preparation.prepared_call(
                    state, inp=inp, out=out
                ),
                collective=_owner_preparation.collective(runtime, key=name),
            ),
        ),
        coordinator=lambda state: (
            state.ready_collectives[0].key if state.ready_collectives else None
        ),
    )
    return plan


def _assert_bits(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    assert actual.dtype == torch.bfloat16
    assert torch.equal(
        actual.view(torch.uint16).cpu(), expected.view(torch.uint16).cpu()
    ), f"{what}: kernel output is not bit-exact vs the pinned CPU reference"


def _worker(rank: int, world: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
    )
    runtime = None
    session = None
    try:
        runtime = PCIeIslandARAllReduce.from_exchange_group(
            exchange_group=dist.group.WORLD,
            device=device,
            max_rows=MAX_ROWS,
            row_elems=7168,
        )
        assert runtime.mapped_peer_count == 4

        def barrier(key, ranks):
            assert tuple(ranks) == tuple(range(world))
            dist.barrier()

        from b12x.preparation import PreparationSession

        session = PreparationSession(
            device=device, autotune=False, compile_workers=1, collective_barrier=barrier
        )

        graph_inp = None
        for rows, cols in SHAPES:
            for cls in ("gaussian", "commonmode"):
                xs = generate(rows, cols, cls)
                expected = island_composed_all_reduce(xs)
                inp = xs[rank].to(device)
                out = torch.empty_like(inp)
                plan = _prepare(
                    session, runtime, inp, out, f"hidden.reduce{rows}x{cols}{cls}"
                )
                result = runtime.all_reduce(inp, plan=plan, out=out)
                assert result is out
                torch.cuda.synchronize(device)
                _assert_bits(out, expected, f"{rows}x{cols}/{cls}")
                if (rows, cols) == (3072, 7168) and cls == "gaussian":
                    graph_inp = (inp, out, plan, expected)

        # Graph capture: replay must reproduce the same bits with no drift in
        # buffer identity (allocation-free, double-buffered generations).
        inp, out, plan, expected = graph_inp
        stream = torch.cuda.Stream(device=device)
        input_ptr, output_ptr = inp.data_ptr(), out.data_ptr()
        slab_ptrs = tuple(runtime._slab_ptrs)
        graph = torch.cuda.CUDAGraph()
        with runtime.capture(stream=stream), torch.cuda.graph(graph, stream=stream):
            runtime.all_reduce(inp, plan=plan, out=out, stream=stream)
        out.fill_(float("nan"))
        torch.cuda.synchronize(device)
        for _ in range(GRAPH_ROUNDS):
            graph.replay()
            torch.cuda.synchronize(device)
            _assert_bits(out, expected, "graph replay")
        assert inp.data_ptr() == input_ptr and out.data_ptr() == output_ptr
        assert tuple(runtime._slab_ptrs) == slab_ptrs
        dist.barrier()
        if rank == 0:
            print("tp8 two-island all-reduce bit-exact OK", flush=True)
    finally:
        if session is not None:
            session.close()
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()


def test_tp8_two_island_all_reduce_bitexact_eager_and_graph() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    assert torch.cuda.device_count() == WORLD_SIZE, (
        "the TP8 bit-exact test requires 8 visible GPUs"
    )
    mp.spawn(_worker, args=(WORLD_SIZE, _free_port()), nprocs=WORLD_SIZE, join=True)
