"""Timing harness for the iHC operator (defect 26: "no timing").

Decode shapes are 1-16 rows and the prefill shape is 4096 rows, all at Hy4's
hidden 6144. The boundary measurement is the point of the fusion work: the
three launches the sublayer boundary used to need (``run_post`` -> ``run_pre``
-> RMSNorm) against the one fused ``run_post_pre`` launch.

Use standalone::

    python -m b12x.norm.mhc._ihc_timing --rows 1,2,4,8,16,4096 --out record.json

or from tests (``test_ihc_timing.py``). Times are milliseconds per call,
median-of-min over rounds of batched iterations on the current CUDA stream
(CUDA events around each batch).
"""

from __future__ import annotations

import json
import statistics
import time

import torch

from ._ihc_impl import IHC_MULT, run_head, run_post, run_post_pre, run_pre
from ._ihc_tuning import IHC_DEFAULT_NORM_EPS

HC = IHC_MULT
MIXES = 2 * IHC_MULT


def _make_inputs(*, tokens: int, hidden: int, seed: int, device):
    gen = torch.Generator().manual_seed(seed)
    streams = (
        torch.randn((tokens, HC, hidden), generator=gen, dtype=torch.float32).to(device)
        / 3
    ).to(torch.bfloat16).contiguous()
    fn = (
        torch.randn((MIXES, HC * hidden), generator=gen, dtype=torch.float32).to(device)
        / 64
    ).contiguous()
    hc_scale = (
        torch.randn((2,), generator=gen, dtype=torch.float32).to(device) / 3
    ).contiguous()
    hc_base = (
        torch.randn((MIXES,), generator=gen, dtype=torch.float32).to(device) / 5
    ).contiguous()
    branch = (
        torch.randn((tokens, hidden), generator=gen, dtype=torch.float32).to(device) / 4
    ).to(torch.bfloat16).contiguous()
    norm_weight = (
        torch.randn((hidden,), generator=gen, dtype=torch.float32).to(device) / 2
    ).contiguous()
    head_fn = (
        torch.randn((HC, HC * hidden), generator=gen, dtype=torch.float32).to(device)
        / 64
    ).contiguous()
    head_scale = (
        torch.randn((1,), generator=gen, dtype=torch.float32).to(device) / 3
    ).contiguous()
    head_base = (
        torch.randn((HC,), generator=gen, dtype=torch.float32).to(device) / 5
    ).contiguous()
    return {
        "streams": streams,
        "fn": fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "branch": branch,
        "norm_weight": norm_weight,
        "head_fn": head_fn,
        "head_scale": head_scale,
        "head_base": head_base,
    }


def time_call(
    fn, *, warmup: int = 10, rounds: int = 5, iters: int = 50, graph: bool = False
) -> float:
    """Median-of-min milliseconds per call on the current stream.

    ``graph=True`` times CUDA-graph replays of the warmed call — device time
    only, which is what a decode loop sees; the eager mode additionally carries
    the operator's Python validation (~60 us) at decode row counts.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            fn()
        samples = []
        for _ in range(rounds):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                captured.replay()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / iters)
        captured.reset()
        return float(statistics.median(samples))
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return float(statistics.median(samples))


def time_rows(
    rows: int,
    *,
    hidden: int = 6144,
    device: torch.device | None = None,
    iters: int | None = None,
) -> dict:
    """Milliseconds per call for every iHC step at one row count."""
    device = torch.device("cuda") if device is None else torch.device(device)
    t = _make_inputs(tokens=rows, hidden=hidden, seed=1000 + rows, device=device)
    iters = iters if iters is not None else (20 if rows >= 4096 else 100)
    out_post = torch.empty((rows, HC), dtype=torch.float32, device=device)
    out_y = torch.empty((rows, hidden), dtype=torch.bfloat16, device=device)
    out_streams = torch.empty((rows, HC, hidden), dtype=torch.bfloat16, device=device)
    post, _ = run_pre(t["streams"], t["fn"], t["hc_scale"], t["hc_base"])

    def pre():
        run_pre(
            t["streams"],
            t["fn"],
            t["hc_scale"],
            t["hc_base"],
            out_post=out_post,
            out_y=out_y,
        )

    def post_update():
        run_post(t["branch"], t["streams"], post, out=out_streams)

    def pre_norm():
        run_pre(
            t["streams"],
            t["fn"],
            t["hc_scale"],
            t["hc_base"],
            norm_weight=t["norm_weight"],
            out_post=out_post,
            out_y=out_y,
        )

    def boundary_three_launch():
        run_post(t["branch"], t["streams"], post, out=out_streams)
        run_pre(
            out_streams,
            t["fn"],
            t["hc_scale"],
            t["hc_base"],
            norm_weight=t["norm_weight"],
            out_post=out_post,
            out_y=out_y,
        )

    def boundary_fused():
        run_post_pre(
            t["branch"],
            t["streams"],
            post,
            t["fn"],
            t["hc_scale"],
            t["hc_base"],
            norm_weight=t["norm_weight"],
            out_streams=out_streams,
            out_post=out_post,
            out_y=out_y,
        )

    def head():
        run_head(
            t["streams"],
            t["head_fn"],
            t["head_scale"],
            t["head_base"],
            out=out_y,
        )

    return {
        "rows": rows,
        "hidden": hidden,
        "eager": {
            "pre": time_call(pre, iters=iters),
            "post": time_call(post_update, iters=iters),
            "pre_norm_fold": time_call(pre_norm, iters=iters),
            "boundary_three_launch": time_call(boundary_three_launch, iters=iters),
            "boundary_fused": time_call(boundary_fused, iters=iters),
            "head": time_call(head, iters=iters),
        },
        "device": {
            "pre": time_call(pre, iters=iters, graph=True),
            "post": time_call(post_update, iters=iters, graph=True),
            "pre_norm_fold": time_call(pre_norm, iters=iters, graph=True),
            "boundary_three_launch": time_call(
                boundary_three_launch, iters=iters, graph=True
            ),
            "boundary_fused": time_call(boundary_fused, iters=iters, graph=True),
            "head": time_call(head, iters=iters, graph=True),
        },
    }


def run_timing(
    rows_list=(1, 2, 4, 8, 16, 4096), *, hidden: int = 6144, out_path: str | None = None
) -> dict:
    record = {
        "op": "b12x.norm.mhc.ihc",
        "hidden": hidden,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "date": time.strftime("%Y-%m-%d"),
        "units": "ms per call (median of rounds of batched iterations)",
        "results": [time_rows(rows, hidden=hidden) for rows in rows_list],
    }
    if out_path:
        with open(out_path, "w") as handle:
            json.dump(record, handle, indent=2)
    return record


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="1,2,4,8,16,4096")
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    rows_list = tuple(int(item) for item in args.rows.split(","))
    record = run_timing(rows_list, hidden=args.hidden, out_path=args.out)
    header = (
        f"{'rows':>6} | {'mode':>6} | {'pre':>9} {'post':>9} {'pre+norm':>9} "
        f"{'3-launch':>9} {'fused':>9} {'head':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in record["results"]:
        for mode in ("eager", "device"):
            m = row[mode]
            print(
                f"{row['rows']:>6} | {mode:>6} | {m['pre']:>9.4f} {m['post']:>9.4f} "
                f"{m['pre_norm_fold']:>9.4f} {m['boundary_three_launch']:>9.4f} "
                f"{m['boundary_fused']:>9.4f} {m['head']:>9.4f}"
            )


if __name__ == "__main__":
    _main()
