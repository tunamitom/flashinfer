"""Measure prepared DCP LSE reduce-scatter with mutable graph inputs.

Launch with torch.distributed.run and an explicit CUDA_VISIBLE_DEVICES assignment.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist

from b12x.comm.pcie import _dcp_preparation as preparation
from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2APool, lse_reduce_scatter_reference
from b12x.preparation import CollectiveRequirement, PreparationSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,2,4,8,16,32,64")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=1381)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(minutes=3))
    world = dist.get_world_size()
    dtype = getattr(torch, args.dtype)
    pool = PCIeDCPA2APool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_batch_size=max(batches),
        total_heads=args.heads,
        head_dim=args.head_dim,
        query_head_dim=args.head_dim,
        single_channel=True,
    )
    runtime = pool.for_stream()
    session = PreparationSession(device=device, autotune=False, compile_workers=1)
    records, graph = [], None
    try:
        for batch in batches:
            partial = torch.ones(
                (batch, args.heads, args.head_dim), device=device, dtype=dtype
            )
            lse = torch.zeros((batch, args.heads), device=device)
            out = torch.empty(
                (batch, args.heads // world, args.head_dim), device=device, dtype=dtype
            )
            call = dict(partial_output=partial, partial_lse=lse, out=out)
            query = preparation.query_from_runtime(
                runtime, surface="DcpAllToAll.lse_reduce_scatter", call=call
            )
            plan = preparation.plan(query, runtime=runtime)
            name = f"lse-{batch}"
            session.prepare(
                (
                    plan.request(
                        name=name,
                        collective=CollectiveRequirement(
                            key=name, ranks=tuple(range(world))
                        ),
                        prepare_call=lambda state: preparation.prepare_call(
                            state, **call
                        ),
                    ),
                ),
                coordinator=lambda progress: progress.ready_collectives[0].key
                if progress.ready_collectives
                else None,
            )
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with session.capture(), pool.capture(), torch.cuda.graph(graph):
                for _ in range(16):
                    runtime.lse_reduce_scatter(
                        partial, lse, out=out, plan=plan, lse_base="natural"
                    )
            for step in range(5):
                generator = torch.Generator().manual_seed(args.seed + step)
                values = torch.randn((world, *partial.shape), generator=generator).to(
                    dtype
                )
                scores = torch.randn((world, *lse.shape), generator=generator)
                if step == 1:
                    scores[:, 0, 0] = float("-inf")
                    scores[0, 0, 1] = float("nan")
                elif step == 2:
                    values.zero_()
                elif step == 3:
                    scores *= 64
                partial.copy_(values[rank])
                lse.copy_(scores[rank])
                expected = lse_reduce_scatter_reference(
                    values.to(device), scores.to(device), rank, lse_base="natural"
                )
                out.fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for counter in (
                    "allocation.all.allocated",
                    "allocation.all.freed",
                    "segment.all.allocated",
                    "segment.all.freed",
                ):
                    assert before[counter] == after[counter], counter
                torch.testing.assert_close(
                    out,
                    expected,
                    rtol=2e-2 if dtype == torch.bfloat16 else 2e-3,
                    atol=2e-2 if dtype == torch.bfloat16 else 2e-3,
                )
            for _ in range(100):
                graph.replay()
            torch.cuda.synchronize()
            samples = []
            for _ in range(7):
                dist.barrier()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(50):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / (50 * 16))
            rank_samples = [None] * world
            dist.all_gather_object(rank_samples, samples)
            record = dict(
                batch=batch,
                world_size=world,
                rank_samples_us=rank_samples,
                max_rank_median_us=max(median(samples) for samples in rank_samples),
            )
            records.append(record)
            if rank == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(
                        {
                            "arguments": {**vars(args), "output": str(args.output)},
                            "timing": {
                                "samples": 7,
                                "replays": 50,
                                "chain": 16,
                                "warmup": 100,
                            },
                            "results": records,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(json.dumps(record), flush=True)
            graph.reset()
            graph = None
    finally:
        if graph is not None:
            graph.reset()
        torch.cuda.synchronize()
        session.close()
        pool.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
