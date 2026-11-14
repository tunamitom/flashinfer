"""Triton kernels for the Hy4 iHC operator (b12x norm.mhc iHC variant).

iHC = DeepSeek-V4 mHC minus the comb/Sinkhorn step. These kernels implement the
three iHC steps at hidden size 6144 (and the other whitelisted sizes) with the
8-row ``fn`` layout (4 pre + 4 post rows):

- ``_ihc_pre_kernel``: per token, FP32 RMS over the flattened ``4H`` row plus the
  8 projection dots (FP32 accumulation, no TF32), sigmoid gates
  (``pre = sigmoid(..) + hc_eps``, ``post = magnitude * sigmoid(..) + hc_eps``),
  then the weighted stream collapse ``y = sum_i pre_i * streams_i`` (FP32
  accumulation, one BF16 rounding). One program per token; the projection
  weight is shared across the grid through L2.
- ``_ihc_post_kernel``: ``out[i] = post_i * x + residual[i]`` per stream,
  FP32 accumulation, one BF16 rounding.
- ``_ihc_head_kernel``: final stream merge with the ``[4, 4H]`` head weights,
  a scalar scale and a ``[4]`` bias.

The geometry follows the mHC fused kernel's tile rules (``IHC_SOURCE_TILE_H`` /
``IHC_GRAM_BLOCK_H`` divisibility), which hidden 6144 already satisfies.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _ihc_pre_kernel(
    streams_ptr,
    fn_ptr,
    scale_ptr,
    base_ptr,
    post_ptr,
    y_ptr,
    H,
    K,
    magnitude,
    hc_eps,
    norm_eps,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Collapse 4 streams with pre gates and emit post gates (one token/program)."""
    t = tl.program_id(0)

    # Pass 1: FP32 square sum and the 8 projection dots over the flat 4H row.
    sq = 0.0
    a0 = 0.0
    a1 = 0.0
    a2 = 0.0
    a3 = 0.0
    a4 = 0.0
    a5 = 0.0
    a6 = 0.0
    a7 = 0.0
    for k in range(0, K, BLOCK_K):
        offs = k + tl.arange(0, BLOCK_K)
        xk = tl.load(streams_ptr + t * K + offs).to(tl.float32)
        sq += tl.sum(xk * xk)
        a0 += tl.sum(xk * tl.load(fn_ptr + 0 * K + offs).to(tl.float32))
        a1 += tl.sum(xk * tl.load(fn_ptr + 1 * K + offs).to(tl.float32))
        a2 += tl.sum(xk * tl.load(fn_ptr + 2 * K + offs).to(tl.float32))
        a3 += tl.sum(xk * tl.load(fn_ptr + 3 * K + offs).to(tl.float32))
        a4 += tl.sum(xk * tl.load(fn_ptr + 4 * K + offs).to(tl.float32))
        a5 += tl.sum(xk * tl.load(fn_ptr + 5 * K + offs).to(tl.float32))
        a6 += tl.sum(xk * tl.load(fn_ptr + 6 * K + offs).to(tl.float32))
        a7 += tl.sum(xk * tl.load(fn_ptr + 7 * K + offs).to(tl.float32))

    # mixes = dots * rsqrt(mean(x^2) + norm_eps), then the sigmoid gates.
    rms = tl.rsqrt(sq / K + norm_eps)
    s_pre = tl.load(scale_ptr + 0)
    s_post = tl.load(scale_ptr + 1)
    pre0 = tl.sigmoid((a0 * rms) * s_pre + tl.load(base_ptr + 0)) + hc_eps
    pre1 = tl.sigmoid((a1 * rms) * s_pre + tl.load(base_ptr + 1)) + hc_eps
    pre2 = tl.sigmoid((a2 * rms) * s_pre + tl.load(base_ptr + 2)) + hc_eps
    pre3 = tl.sigmoid((a3 * rms) * s_pre + tl.load(base_ptr + 3)) + hc_eps
    post0 = magnitude * tl.sigmoid((a4 * rms) * s_post + tl.load(base_ptr + 4)) + hc_eps
    post1 = magnitude * tl.sigmoid((a5 * rms) * s_post + tl.load(base_ptr + 5)) + hc_eps
    post2 = magnitude * tl.sigmoid((a6 * rms) * s_post + tl.load(base_ptr + 6)) + hc_eps
    post3 = magnitude * tl.sigmoid((a7 * rms) * s_post + tl.load(base_ptr + 7)) + hc_eps
    tl.store(post_ptr + t * 4 + 0, post0)
    tl.store(post_ptr + t * 4 + 1, post1)
    tl.store(post_ptr + t * 4 + 2, post2)
    tl.store(post_ptr + t * 4 + 3, post3)

    # Pass 2: y = sum_i pre_i * streams_i over the stream axis (FP32 in, one
    # BF16 rounding out, matching the reference).
    for d in range(0, H, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        y = pre0 * tl.load(streams_ptr + t * K + 0 * H + offs).to(tl.float32)
        y += pre1 * tl.load(streams_ptr + t * K + 1 * H + offs).to(tl.float32)
        y += pre2 * tl.load(streams_ptr + t * K + 2 * H + offs).to(tl.float32)
        y += pre3 * tl.load(streams_ptr + t * K + 3 * H + offs).to(tl.float32)
        tl.store(y_ptr + t * H + offs, y.to(tl.bfloat16))


@triton.jit
def _ihc_post_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    out_ptr,
    H,
    BLOCK_D: tl.constexpr,
):
    """out[t, i, d] = post[t, i] * x[t, d] + residual[t, i, d] (FP32, BF16 out)."""
    t = tl.program_id(0)
    dblk = tl.program_id(1)
    offs = dblk * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < H
    xv = tl.load(x_ptr + t * H + offs, mask=mask).to(tl.float32)
    p0 = tl.load(post_ptr + t * 4 + 0)
    p1 = tl.load(post_ptr + t * 4 + 1)
    p2 = tl.load(post_ptr + t * 4 + 2)
    p3 = tl.load(post_ptr + t * 4 + 3)
    o0 = p0 * xv + tl.load(residual_ptr + t * 4 * H + 0 * H + offs, mask=mask).to(
        tl.float32
    )
    o1 = p1 * xv + tl.load(residual_ptr + t * 4 * H + 1 * H + offs, mask=mask).to(
        tl.float32
    )
    o2 = p2 * xv + tl.load(residual_ptr + t * 4 * H + 2 * H + offs, mask=mask).to(
        tl.float32
    )
    o3 = p3 * xv + tl.load(residual_ptr + t * 4 * H + 3 * H + offs, mask=mask).to(
        tl.float32
    )
    tl.store(out_ptr + t * 4 * H + 0 * H + offs, o0.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + t * 4 * H + 1 * H + offs, o1.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + t * 4 * H + 2 * H + offs, o2.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + t * 4 * H + 3 * H + offs, o3.to(tl.bfloat16), mask=mask)


@triton.jit
def _ihc_head_kernel(
    streams_ptr,
    fn_ptr,
    scale_ptr,
    base_ptr,
    y_ptr,
    H,
    K,
    hc_eps,
    norm_eps,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Final stream merge (one token/program): 4 dots -> pre gates -> collapse."""
    t = tl.program_id(0)

    sq = 0.0
    a0 = 0.0
    a1 = 0.0
    a2 = 0.0
    a3 = 0.0
    for k in range(0, K, BLOCK_K):
        offs = k + tl.arange(0, BLOCK_K)
        xk = tl.load(streams_ptr + t * K + offs).to(tl.float32)
        sq += tl.sum(xk * xk)
        a0 += tl.sum(xk * tl.load(fn_ptr + 0 * K + offs).to(tl.float32))
        a1 += tl.sum(xk * tl.load(fn_ptr + 1 * K + offs).to(tl.float32))
        a2 += tl.sum(xk * tl.load(fn_ptr + 2 * K + offs).to(tl.float32))
        a3 += tl.sum(xk * tl.load(fn_ptr + 3 * K + offs).to(tl.float32))

    rms = tl.rsqrt(sq / K + norm_eps)
    s = tl.load(scale_ptr + 0)
    pre0 = tl.sigmoid((a0 * rms) * s + tl.load(base_ptr + 0)) + hc_eps
    pre1 = tl.sigmoid((a1 * rms) * s + tl.load(base_ptr + 1)) + hc_eps
    pre2 = tl.sigmoid((a2 * rms) * s + tl.load(base_ptr + 2)) + hc_eps
    pre3 = tl.sigmoid((a3 * rms) * s + tl.load(base_ptr + 3)) + hc_eps

    for d in range(0, H, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        y = pre0 * tl.load(streams_ptr + t * K + 0 * H + offs).to(tl.float32)
        y += pre1 * tl.load(streams_ptr + t * K + 1 * H + offs).to(tl.float32)
        y += pre2 * tl.load(streams_ptr + t * K + 2 * H + offs).to(tl.float32)
        y += pre3 * tl.load(streams_ptr + t * K + 3 * H + offs).to(tl.float32)
        tl.store(y_ptr + t * H + offs, y.to(tl.bfloat16))
