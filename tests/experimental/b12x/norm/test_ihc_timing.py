"""Timing harness test: decode 1-16 rows and prefill 4096 rows at hidden 6144.

Records per-call milliseconds (eager call and CUDA-graph device time) for
``run_pre`` / ``run_post`` / the pre+norm fold / ``run_head``, and compares the
three-launch sublayer boundary (``run_post`` -> ``run_pre`` -> RMSNorm) with
the one-launch fused ``run_post_pre``. Numbers are printed for the test log and
written to ``$IHC_TIMING_OUT`` when set; the recorded reference run lives in
``ihc_timing_record.json`` next to this test.
"""

from __future__ import annotations

import json
import math
import os

import pytest
import torch

from b12x.norm.mhc import _ihc_timing

if not torch.cuda.is_available():
    pytest.skip("iHC timing tests require CUDA", allow_module_level=True)

RECORD_ROWS = (1, 16, 4096)


@pytest.mark.parametrize("rows", RECORD_ROWS)
def test_timing_recorded(rows: int) -> None:
    result = _ihc_timing.time_rows(rows, hidden=6144)
    for mode in ("eager", "device"):
        for name, value in result[mode].items():
            assert math.isfinite(value) and value > 0, (rows, mode, name, value)
    print(
        f"\niHC timing rows={rows} hidden=6144 (ms/call): "
        + " ".join(
            f"{mode}.{name}={value:.4f}"
            for mode in ("eager", "device")
            for name, value in result[mode].items()
        )
    )
    out = os.environ.get("IHC_TIMING_OUT")
    if out:
        records = []
        if os.path.exists(out):
            with open(out) as handle:
                records = json.load(handle)
        records.append(result)
        with open(out, "w") as handle:
            json.dump(records, handle, indent=2)


@pytest.mark.parametrize("rows", RECORD_ROWS)
def test_fused_boundary_is_not_slower_than_three_launches(rows: int) -> None:
    """One launch must not regress the boundary: at prefill it is faster, at
    decode it is within noise of the three launches it replaces (measured
    1.00x-1.03x; the 1.25x bound absorbs clock noise only)."""
    result = _ihc_timing.time_rows(rows, hidden=6144)
    for mode in ("eager", "device"):
        fused = result[mode]["boundary_fused"]
        three = result[mode]["boundary_three_launch"]
        assert fused <= 1.25 * three, (rows, mode, fused, three)
