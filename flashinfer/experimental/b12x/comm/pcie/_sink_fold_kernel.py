"""Triton kernel for the prepared post-merge sink fold (defect 13).

Imported lazily (``import triton`` only when a GPU fold is prepared).
"""

import triton
import triton.language as tl


@triton.jit
def sink_fold_kernel(
    out_ptr, lse_ptr, sink_ptr,
    batch, heads,
    stride_ob, stride_oh,
    stride_lb,
    LN2_F64: tl.constexpr,
    BASE2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OUT_IS_BF16: tl.constexpr,
    OUT_IS_FP16: tl.constexpr,
):
    """Fold the per-head sink in place: out *= sigmoid(lse_e - sink) and
    lse = logaddexp(lse_e, sink), reported in the pinned base (FP64 math)."""
    pid = tl.program_id(0)
    row = pid // heads
    head = pid % heads
    if row >= batch:
        return
    lse_val = tl.load(lse_ptr + row * stride_lb + head).to(tl.float64)
    sink_val = tl.load(sink_ptr + head).to(tl.float64)
    lse_e = lse_val * LN2_F64 if BASE2 else lse_val
    diff = lse_e - sink_val
    factor64 = 1.0 / (1.0 + tl.exp(-diff))
    m = tl.maximum(lse_e, sink_val)
    folded_e = m + tl.log(tl.exp(lse_e - m) + tl.exp(sink_val - m))
    folded = folded_e / LN2_F64 if BASE2 else folded_e
    tl.store(lse_ptr + row * stride_lb + head, folded.to(tl.float32))
    # The factor is rounded to the output dtype BEFORE the multiply, matching
    # the eager reference (``out.mul_(factor.to(out_dtype))``) exactly.
    if OUT_IS_BF16:
        factor_o = factor64.to(tl.bfloat16).to(tl.float32)
    elif OUT_IS_FP16:
        factor_o = factor64.to(tl.float16).to(tl.float32)
    else:
        factor_o = factor64.to(tl.float32)
    offs = tl.arange(0, BLOCK_D)
    for d in range(0, HEAD_DIM, BLOCK_D):
        mask = d + offs < HEAD_DIM
        ptr = out_ptr + row * stride_ob + head * stride_oh + d + offs
        x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        y = x * factor_o
        if OUT_IS_BF16:
            tl.store(ptr, y.to(tl.bfloat16), mask=mask)
        elif OUT_IS_FP16:
            tl.store(ptr, y.to(tl.float16), mask=mask)
        else:
            tl.store(ptr, y.to(tl.float32), mask=mask)
