"""MXFP4-activation (A4) prefill MoE GEMMs over W4A16 packed FP4 weights.

Dynamic-E8M0-group-quant path (PLAN-w4a16-port.md §5): prefill contracts
MXFP4-quantized activations with the W4A16 packed E2M1 weights through the
SM120 block-scaled QMMA ``kind::mxf4nvf4`` ``scale_vec::2X`` m16n8k64
(UE8M0 per-32 scales), so decode keeps the exact W4A16 path while prefill runs
at A4 speed.  No calibrated activation scales: activations are quantized
per-32-group on the fly (exponent-aligned, ``6 * 2**e >= max|x|``).

Layout contracts (validated bit-exact vs a float64 dequantize-and-matmul
oracle across multi-atom K loops and N>8 layouts -- see W2-fc1-ladder.md):

- A/B/accumulator fragment mapping identical to the 4X variant.  Activations
  keep the PI permutation + WORD_ORDER storage of ``prefill_a4.py``; weights
  keep the W4A16 packed layout + PRMT B-register build.  Both operands permute
  K identically within 16-groups, so K32 scale groups are preserved.
- SFA: one u32 per row per K64 slice; byte 0 = UE8M0 scale of the slice's K32
  group 0, byte 1 = group 1; loaded directly as the MMA ``sfa`` word.
- SFB: one u32 per (atom column) holding that column's (group0, group1) UE8M0
  byte-pair.  Weight scale bytes come from the packed e8m0_k32 grid
  (``prepared.w13_scale``/``w2_scale``), whose stored-byte layout is
  ``stored[e, kg, 64*blk + 4*(r//4) + inv4[r%4]] = source[e, n, kg]`` with
  ``r = 8*(w%8) + w//8``, ``w = n%64``, ``blk = n//64``, ``inv4 = [0,2,1,3]``.
- Requantize (FC1 epilogue -> FC2 A operand): per 32-value group, scale byte
  = ``e + 127`` where ``e`` = smallest integer in [-127,127] with
  ``6*2**e >= max|x|`` (``max|x|==0`` -> ``e=0``); E2M1 RNE codes, saturation
  at 6, sign-magnitude (``-0.0`` -> ``+0``).  Byte 255 never emitted.

FC1 runs the gate and up halves of a 128-column intermediate tile in one K
sweep, applies ``silu(alpha g)*(alpha u)`` (``prefill_a4`` semantics:
``bf16(bf16(g)*sigmoid(bf16(g))*bf16(u))``) and requantizes to MXFP4.  FC2
stores unweighted per-route BF16 rows for the FP32 top-k sum.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import cuda.bindings.driver as cuda
import torch
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Float32, Int32, Int64, T, Uint32, Uint64, dsl_user_op

from b12x._lib.intrinsics import (
    bfloat2_to_float2_scaled,
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    fabs_f32,
    fmax_f32,
    f32_to_raw_bits,
    get_ptr_as_int64,
    ld_global_nc_v4_u32,
    ld_shared_u32,
    ld_shared_v2_u32,
    ld_shared_v4_u32,
    mxfp4_mma_m16n8k64_f32_e2m1_ue8m0,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_global_u32,
    st_global_v4_u32,
    st_shared_u32,
    u32_as_f32,
)

# Position m of a 16-group holds physical K offset PI[m] (both operands permute
# K identically, so K32 scale-group boundaries survive).
PI = (0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15)
WORD_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
# Inverse of the 8x8 packed-scale column shuffle (self-inverse).
_INV4 = (0, 2, 1, 3)

BLOCK_M = 128
TILE_K = 64
NUM_WARPS = 8
A_STAGE_BYTES = BLOCK_M * 32
SFA_STAGE_BYTES = BLOCK_M * 4
PLANE_STAGE_BYTES = A_STAGE_BYTES + SFA_STAGE_BYTES
B_STAGE_BYTES = (TILE_K // 16) * 4 * 512
# SFB: per K64 tile, 4 chunk slots x 64 columns x 4 B (one u32 byte-pair word
# per column: byte 0/1 = K32 groups 2*kg0/2*kg0+1 of the tile).
S_STAGE_BYTES = 4 * 64 * 4
ROUTE_BYTES = BLOCK_M * 4
EPI_ROW_BYTES = 256 * 2 + 16


@dsl_user_op
def prmt_b32(a: Uint32, b: Uint32, sel: Uint32, *, loc=None, ip=None) -> Uint32:
    """``prmt.b32`` with an immediate selector."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(a).ir_value(loc=loc, ip=ip),
                Uint32(b).ir_value(loc=loc, ip=ip),
                Uint32(sel).ir_value(loc=loc, ip=ip),
            ],
            "prmt.b32 $0, $1, $2, $3;",
            "=r,r,r,n",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def prmt_b32_reg(a: Uint32, b: Uint32, sel: Uint32, *, loc=None, ip=None) -> Uint32:
    """``prmt.b32`` with a register selector (byte LUT lookups)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(a).ir_value(loc=loc, ip=ip),
                Uint32(b).ir_value(loc=loc, ip=ip),
                Uint32(sel).ir_value(loc=loc, ip=ip),
            ],
            "prmt.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_global_nc_u8(addr: Int64, *, loc=None, ip=None) -> Uint32:
    """Non-coherent byte load (packed scale grid bytes sit at arbitrary offsets)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.b8 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------- requantize --
@dsl_user_op
def _f32_exponent(x: Float32, *, loc=None, ip=None) -> Int32:
    """IEEE-754 biased exponent of a finite, nonzero f32 (``(bits>>23)&0xFF``)."""
    bits = f32_to_raw_bits(x)
    return Int32((bits >> Uint32(23)) & Uint32(0xFF))


@cute.jit
def _ldexp2(e: Int32) -> Float32:
    """``2**e`` for ``e`` in [-127, 127] via bit construction (exact).

    ``e = -127`` is subnormal in f32 (``2**-127 = 0x00400000``); every other
    exponent in range is normal (``biased << 23``).
    """
    biased = e + Int32(127)
    result = u32_as_f32(Uint32(biased) << Uint32(23))
    if e == Int32(-127):
        result = u32_as_f32(Uint32(0x00400000))
    return result


@cute.jit
def _mxfp4_scale_exponent(mx: Float32) -> Int32:
    """Smallest ``e`` in [-127, 127] with ``6 * 2**e >= mx`` (``mx == 0`` -> 0).

    Matches the validated CPU rule (W2 ladder): ``e0 = floor(log2 mx) - 2``
    from the f32 exponent field, walk up with exact ``6 * 2**e`` comparisons,
    walk down for minimality, clamp at -127 (subnormal ``mx`` -> -127).
    """
    e = Int32(0)
    if mx != Float32(0.0):
        be = _f32_exponent(mx)
        e = Int32(-127)
        if be != Int32(0):
            e = (be - Int32(127)) - Int32(2)
            for _ in cutlass.range_constexpr(8):
                if Float32(6.0) * _ldexp2(e) < mx:
                    e = e + Int32(1)
            for _ in cutlass.range_constexpr(8):
                if e > Int32(-127) and Float32(6.0) * _ldexp2(e - Int32(1)) >= mx:
                    e = e - Int32(1)
    return e


@cute.jit
def _mxfp4_pack_words(codes) -> tuple[Uint32, Uint32, Uint32, Uint32]:
    """Pack 32 E2M1 codes (LOGICAL k order) into a row K64 slice half.

    Word ``j`` nibble ``i`` holds the code of logical position
    ``16 * (j // 2) + PI[8 * (j % 2) + i]`` (the PI permutation of the
    tensor-core lane layout, identical to ``prefill_a4``'s 16-group packing
    generalized to two 16-groups per K32 scale).
    """
    w0 = Uint32(0)
    w1 = Uint32(0)
    w2 = Uint32(0)
    w3 = Uint32(0)
    for j in cutlass.range_constexpr(4):
        for i in cutlass.range_constexpr(8):
            code = codes[16 * (j // 2) + PI[8 * (j % 2) + i]]
            word = code << Uint32(4 * i)
            if j == 0:
                w0 = w0 | word
            elif j == 1:
                w1 = w1 | word
            elif j == 2:
                w2 = w2 | word
            else:
                w3 = w3 | word
    return w0, w1, w2, w3


@cute.jit
def _mxfp4_quantize32(vals, codes):
    """MXFP4-quantize 32 f32 values in LOGICAL k order (one K32 group).

    ``vals[m]`` is the logical value at position ``m`` of the group (the K64
    slice's positions ``32 g .. 32 g + 31``).  Fills ``codes[m]`` (logical
    order) and returns the four packed payload words plus the group's UE8M0
    scale byte (the MMA's scale-pair byte for this K32 group).
    """
    mx = Float32(0.0)
    for m in cutlass.range_constexpr(32):
        mx = fmax_f32(mx, fabs_f32(vals[m]))
    e = _mxfp4_scale_exponent(mx)
    two_e = _ldexp2(e)
    for m in cutlass.range_constexpr(32):
        av = fabs_f32(vals[m]) / two_e
        code = Uint32(7)
        if av <= Float32(0.25):
            code = Uint32(0)
        elif av < Float32(0.75):
            code = Uint32(1)
        elif av <= Float32(1.25):
            code = Uint32(2)
        elif av < Float32(1.75):
            code = Uint32(3)
        elif av <= Float32(2.5):
            code = Uint32(4)
        elif av < Float32(3.5):
            code = Uint32(5)
        elif av <= Float32(5.0):
            code = Uint32(6)
        if vals[m] < Float32(0.0):
            code = code | Uint32(8)
        codes[m] = code
    w0, w1, w2, w3 = _mxfp4_pack_words(codes)
    return w0, w1, w2, w3, (Uint32(e) + Uint32(127)) & Uint32(0xFF)


@cute.jit
def _mxfp4_dequant(code: Uint32, scale_byte: Uint32) -> Float32:
    """Dequantized value of one E2M1 code under a K32 group scale byte."""
    half = prmt_b32_reg(
        Uint32(0x03020100), Uint32(0x0C080604), code & Uint32(7)
    ) & Uint32(0xFF)
    value = Float32(half) * Float32(0.5) * _ldexp2(Int32(scale_byte) - Int32(127))
    if (code & Uint32(8)) != Uint32(0):
        value = -value
    return value


@cute.jit
def _store_slice_mxfp4(q_ptr: Int64, sf_ptr: Int64, words, scale_word: Uint32):
    """One row K64 slice: eight payload words in WORD_ORDER + its scale word.

    ``words[2g], words[2g+1]`` hold K32 group ``g`` (stored positions
    32g..32g+31), so ``scale_word`` byte ``g`` is that group's UE8M0 byte --
    byte 0/1 of the MMA's per-atom scale pair.
    """
    st_global_v4_u32(
        q_ptr,
        words[WORD_ORDER[0]],
        words[WORD_ORDER[1]],
        words[WORD_ORDER[2]],
        words[WORD_ORDER[3]],
    )
    st_global_v4_u32(
        q_ptr + Int64(16),
        words[WORD_ORDER[4]],
        words[WORD_ORDER[5]],
        words[WORD_ORDER[6]],
        words[WORD_ORDER[7]],
    )
    st_global_u32(sf_ptr, scale_word)


class Mxfp4A4Quantize:
    """BF16 ``[rows, K]`` -> permuted MXFP4 ``x_q`` + per-K64 UE8M0 scale words.

    One thread per (row, K64 slice); per 32-value group (two 16-groups) the
    scale byte is ``e + 127`` with ``6*2**e >= max|x|`` (dynamic E8M0 group
    quant; no calibrated scales, no shared global scale).  ``terms=2`` also
    writes the residual term (``x_q2``/``x_sf2``, same layout): ``x ~ q1 + q2``.
    """

    THREADS = 256

    def __init__(self, *, size_k: int, terms: int = 1):
        if size_k % 64:
            raise ValueError("MXFP4 A4 quantization needs K % 64 == 0")
        if terms not in (1, 2):
            raise ValueError("terms must be 1 or 2")
        self.size_k = int(size_k)
        self.slices = self.size_k // 64
        self.terms = int(terms)

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (self.size_k, self.terms)

    @cute.jit
    def __call__(
        self,
        x_bf16: cute.Tensor,
        x_q: cute.Tensor,
        x_sf: cute.Tensor,
        x_q2: cute.Tensor,
        x_sf2: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        items = rows * Int32(self.slices)
        self.kernel(x_bf16, x_q, x_sf, x_q2, x_sf2, rows).launch(
            grid=((items + Int32(self.THREADS - 1)) // Int32(self.THREADS), 1, 1),
            block=[self.THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x_bf16: cute.Tensor,
        x_q: cute.Tensor,
        x_sf: cute.Tensor,
        x_q2: cute.Tensor,
        x_sf2: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        item = Int32(bidx) * Int32(self.THREADS) + Int32(tidx)
        if item < rows * Int32(self.slices):
            row = item // Int32(self.slices)
            sl = item - row * Int32(self.slices)
            src = get_ptr_as_int64(
                x_bf16, Int64(row) * Int64(self.size_k) + Int64(sl * Int32(64))
            )
            words = cute.make_rmem_tensor((8,), Uint32)
            words2 = cute.make_rmem_tensor((8,), Uint32)
            scale_word = Uint32(0)
            scale_word2 = Uint32(0)
            for g in cutlass.range_constexpr(2):
                # K32 group g = slice positions 32g..32g+31 (logical k order);
                # four 16-byte loads per group (8 BF16 each).
                vals = cute.make_rmem_tensor((32,), Float32)
                codes = cute.make_rmem_tensor((32,), Uint32)
                for v in cutlass.range_constexpr(4):
                    raw = ld_global_nc_v4_u32(src + Int64(64 * g + 16 * v))
                    for i in cutlass.range_constexpr(4):
                        lo, hi = bfloat2_to_float2_scaled(raw[i], Float32(1.0))
                        vals[8 * v + 2 * i] = lo
                        vals[8 * v + 2 * i + 1] = hi
                out = _mxfp4_quantize32(vals, codes)
                words[4 * g + 0] = out[0]
                words[4 * g + 1] = out[1]
                words[4 * g + 2] = out[2]
                words[4 * g + 3] = out[3]
                scale_word = scale_word | (out[4] << Uint32(8 * g))
                if cutlass.const_expr(self.terms == 2):
                    # Residual of the dequantized first term: res = x - deq(q1).
                    res = cute.make_rmem_tensor((32,), Float32)
                    codes2 = cute.make_rmem_tensor((32,), Uint32)
                    for m in cutlass.range_constexpr(32):
                        res[m] = vals[m] - _mxfp4_dequant(codes[m], out[4])
                    out2 = _mxfp4_quantize32(res, codes2)
                    words2[4 * g + 0] = out2[0]
                    words2[4 * g + 1] = out2[1]
                    words2[4 * g + 2] = out2[2]
                    words2[4 * g + 3] = out2[3]
                    scale_word2 = scale_word2 | (out2[4] << Uint32(8 * g))
            q_off = Int64(row) * Int64(self.size_k // 8) + Int64(sl * Int32(8))
            sf_off = Int64(row) * Int64(self.slices) + Int64(sl)
            _store_slice_mxfp4(
                get_ptr_as_int64(x_q, q_off),
                get_ptr_as_int64(x_sf, sf_off),
                words,
                scale_word,
            )
            if cutlass.const_expr(self.terms == 2):
                _store_slice_mxfp4(
                    get_ptr_as_int64(x_q2, q_off),
                    get_ptr_as_int64(x_sf2, sf_off),
                    words2,
                    scale_word2,
                )


# ------------------------------------------------------- packed scale grid ---
def _e8m0_byte_off(base: Int64, n: Int32, kg: Int32, n_padded: Int32) -> Int64:
    """Byte address of column ``n``'s UE8M0 scale for K32 group ``kg`` in the
    packed e8m0_k32 grid (``prepared.w13_scale``/``w2_scale``).

    ``stored[kg, 64*blk + s] = source[n, kg]`` where ``q, jj = divmod(s % 64, 4)``,
    ``p = 4*q + [0,2,1,3][jj]``, ``n = 64*blk + 8*(p % 8) + (p // 8)`` (verified
    bit-exact against ``_pack_e8m0_k32_scales`` for multi-block N).  Inverse:
    ``w = n % 64``, ``r = 8*(w % 8) + (w // 8)``, ``s = 4*(r // 4) + INV4[r % 4]``.
    """
    blk = n >> Int32(6)
    w = n & Int32(63)
    r = Int32(8) * (w & Int32(7)) + (w >> Int32(3))
    rr = r & Int32(3)
    inv4 = ((rr & Int32(1)) << Int32(1)) | (rr >> Int32(1))
    s = Int32(4) * (r >> Int32(2)) + inv4
    return base + Int64(kg) * Int64(n_padded) + Int64((blk << Int32(6)) + s)


class Mxfp4A4PrefillGemm:
    """One FC phase of the MXFP4-A4 prefill MoE over W4A16 packed FP4 weights.

    Structural twin of ``prefill_a4.A4PackedPrefillGemm`` (staging, PRMT B
    build, K-loop, epilogues, tile indexing all follow it); the differences
    are the operand contract: ``a_q``/``a_sf`` are the dynamic-E8M0 MXFP4
    activations from ``Mxfp4A4Quantize``, ``scales_u8`` is the packed e8m0_k32
    weight scale grid, and the MMA is the 2X/UE8M0 m16n8k64.  ``a_sf`` holds
    one u32 per row per K64 slice (byte 0/1 = K32 group 0/1); the SFB words
    are staged as per-column byte-pairs from the packed grid.
    """

    def __init__(
        self,
        *,
        phase: str,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        stages: int = 4,
        fast_math: bool = True,
        terms: int = 1,
        warps: int = NUM_WARPS,
    ):
        if phase not in ("fc1", "fc2"):
            raise ValueError(f"unknown MXFP4 A4 prefill phase {phase!r}")
        if hidden_size % 256 or intermediate_size % 128:
            raise ValueError("MXFP4 A4 prefill needs H % 256 == 0 and I % 128 == 0")
        if terms not in (1, 2):
            raise ValueError("terms must be 1 or 2")
        if warps not in (8, 16):
            raise ValueError("warps must be 8 or 16")
        self.phase = phase
        self.fc1 = phase == "fc1"
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.top_k = int(top_k)
        self.stages = int(stages)
        self.fast_math = bool(fast_math)
        self.terms = int(terms)
        self.warps = int(warps)
        self.threads = self.warps * 32
        self.m_warps = self.warps // 4
        self.rows_per_warp = BLOCK_M // self.m_warps
        self.mb_per_warp = self.rows_per_warp // 16
        self.acc_size = self.mb_per_warp * 8 * 4
        self.size_k = self.hidden_size if self.fc1 else self.intermediate_size
        self.size_n = 2 * self.intermediate_size if self.fc1 else self.hidden_size
        self.n_tiles = (
            self.intermediate_size // 128 if self.fc1 else self.hidden_size // 256
        )
        self.k_tiles = self.size_k // TILE_K
        self.scales_kg = self.size_k // 32
        self.scales_n_padded = self.size_n
        self.b_off = self.terms * PLANE_STAGE_BYTES
        self.s_off = self.b_off + B_STAGE_BYTES
        self.stage_bytes = self.s_off + S_STAGE_BYTES
        self.pipeline_bytes = self.stages * self.stage_bytes
        self.shared_bytes = (
            max(self.pipeline_bytes, BLOCK_M * EPI_ROW_BYTES) + ROUTE_BYTES
        )

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            self.phase,
            self.hidden_size,
            self.intermediate_size,
            self.top_k,
            self.stages,
            self.fast_math,
            self.terms,
            BLOCK_M,
            TILE_K,
            self.warps,
        )

    @cute.jit
    def __call__(
        self,
        a_q: cute.Tensor,
        a_sf: cute.Tensor,
        a_q2: cute.Tensor,
        a_sf2: cute.Tensor,
        weights_i32: cute.Tensor,
        scales_u8: cute.Tensor,
        w_global: cute.Tensor,
        a_gscale: cute.Tensor,
        q_gscale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        out_q: cute.Tensor,
        out_sf: cute.Tensor,
        out_q2: cute.Tensor,
        out_sf2: cute.Tensor,
        out_bf16: cute.Tensor,
        live_routes: cutlass.Int32,
        gs_stride: cutlass.Int32,
        grid_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            a_q,
            a_sf,
            a_q2,
            a_sf2,
            weights_i32,
            scales_u8,
            w_global,
            a_gscale,
            q_gscale,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            out_q,
            out_sf,
            out_q2,
            out_sf2,
            out_bf16,
            live_routes,
            gs_stride,
        ).launch(
            grid=(grid_ctas, 1, 1),
            block=[self.threads, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.jit
    def _chunk_n64(self, n_tile: Int32, chunk: Int32) -> Int32:
        """Packed N64 chunk index of tile ``n_tile`` slot ``chunk`` (0..3)."""
        result = n_tile * Int32(4) + chunk
        if cutlass.const_expr(self.fc1):
            result = n_tile * Int32(2) + (chunk & Int32(1))
            if chunk >= Int32(2):
                result += Int32(self.intermediate_size // 64)
        return result

    # -- staging -------------------------------------------------------------------

    @cute.jit
    def _issue(
        self,
        tid: Int32,
        stage_base: Int32,
        k_tile: Int32,
        a_desc,
        sf_desc,
        b_desc,
        s_base: Int64,
        n_tile: Int32,
    ):
        """One K64 slice: live A rows (2 x 16 B per plane) and their scale words,
        packed B chunks (16 B vectors), and the SFB byte-pairs (64 x 4 B).

        The packed e8m0_k32 grid stores each column's two K32-group scale
        bytes at non-adjacent offsets, so the SFB words (byte-pairs) are
        assembled here via byte loads: thread ``tid`` < 256 stages one u32
        (byte 0 = K32 group ``2 k_tile``, byte 1 = ``2 k_tile + 1``) per
        (chunk slot, column) of this tile.
        """
        for src, dst, live in a_desc:
            if live != Int32(0):
                cp_async4_shared_global(
                    stage_base + dst, src + Int64(k_tile) * Int64(32)
                )
        for src, dst, live in sf_desc:
            if live != Int32(0):
                cp_async_u32_shared_global(
                    stage_base + dst, src + Int64(k_tile) * Int64(4)
                )
        for src, dst in b_desc:
            cp_async4_shared_global(
                stage_base + dst,
                src + Int64(k_tile) * Int64((TILE_K // 16) * (self.size_n // 64) * 512),
            )
        if tid < Int32(256):
            wslot = tid >> Int32(6)
            wcol = tid & Int32(63)
            n_col = self._chunk_n64(n_tile, wslot) * Int32(64) + wcol
            kg0 = Int32(2) * k_tile
            b0 = ld_global_nc_u8(
                _e8m0_byte_off(s_base, n_col, kg0, Int32(self.scales_n_padded))
            )
            b1 = ld_global_nc_u8(
                _e8m0_byte_off(
                    s_base, n_col, kg0 + Int32(1), Int32(self.scales_n_padded)
                )
            )
            st_shared_u32(
                stage_base
                + Int32(self.s_off)
                + (wslot << Int32(8))
                + (wcol << Int32(2)),
                b0 | (b1 << Uint32(8)),
            )

    # -- mainloop ------------------------------------------------------------------

    @cute.jit
    def _mainloop(
        self,
        a_q,
        a_sf,
        a_q2,
        a_sf2,
        weights_i32,
        scales_u8,
        smem_base: Int32,
        tid: Int32,
        lane: Int32,
        m_warp: Int32,
        slot: Int32,
        expert: Int32,
        n_tile: Int32,
        live_blocks: Int32,
        acc,
    ):
        route_base = smem_base + Int32(self.shared_bytes - ROUTE_BYTES)
        k16_rows = self.size_k // 16
        n64_chunks = self.size_n // 64
        q = lane >> Int32(2)
        c = lane & Int32(3)
        h = c & Int32(1)
        live_rows = live_blocks * Int32(16)
        a_desc = []
        for i in cutlass.range_constexpr(
            (self.terms * 256 + self.threads - 1) // self.threads
        ):
            idx = tid + Int32(i * self.threads)
            plane = idx >> Int32(8)
            row = (idx & Int32(255)) >> Int32(1)
            vec = idx & Int32(1)
            live = Int32(0)
            if idx < Int32(self.terms * 256) and row < live_rows:
                live = Int32(1)
            src_row = Int32(ld_shared_u32(route_base + (row << Int32(2))))
            elem = Int64(src_row) * Int64(self.size_k // 8) + Int64(vec * Int32(4))
            src = get_ptr_as_int64(a_q, elem)
            if cutlass.const_expr(self.terms == 2) and plane != Int32(0):
                src = get_ptr_as_int64(a_q2, elem)
            dst = plane * Int32(PLANE_STAGE_BYTES) + row * Int32(32) + (vec << Int32(4))
            a_desc.append((src, dst, live))
        sf_desc = []
        for i in cutlass.range_constexpr(
            (self.terms * 128 + self.threads - 1) // self.threads
        ):
            idx = (tid + Int32(i * self.threads + 256)) & Int32(self.threads - 1)
            if cutlass.const_expr(self.terms * 128 > self.threads):
                idx = tid + Int32(i * self.threads)
            plane = idx >> Int32(7)
            row = idx & Int32(127)
            live = Int32(0)
            if idx < Int32(self.terms * 128) and row < live_rows:
                live = Int32(1)
            src_row = Int32(ld_shared_u32(route_base + (row << Int32(2))))
            elem = Int64(src_row) * Int64(self.size_k // 64)
            src = get_ptr_as_int64(a_sf, elem)
            if cutlass.const_expr(self.terms == 2) and plane != Int32(0):
                src = get_ptr_as_int64(a_sf2, elem)
            dst = (
                plane * Int32(PLANE_STAGE_BYTES)
                + Int32(A_STAGE_BYTES)
                + (row << Int32(2))
            )
            sf_desc.append((src, dst, live))
        b_desc = []
        for i in cutlass.range_constexpr(512 // self.threads):
            idx = tid + Int32(i * self.threads)
            b_k16 = idx >> Int32(7)
            b_chunk = (idx >> Int32(5)) & Int32(3)
            b_vec = idx & Int32(31)
            src = get_ptr_as_int64(
                weights_i32,
                (
                    (Int64(expert) * Int64(k16_rows) + Int64(b_k16)) * Int64(n64_chunks)
                    + Int64(self._chunk_n64(n_tile, b_chunk))
                )
                * Int64(128)
                + Int64(b_vec * Int32(4)),
            )
            dst = (
                Int32(self.b_off)
                + ((b_k16 * Int32(4) + b_chunk) << Int32(9))
                + ((b_vec ^ (b_k16 & Int32(1))) << Int32(4))
            )
            b_desc.append((src, dst))
        # SFB: one u32 byte-pair word per (chunk slot, column) is staged by
        # _issue from the packed e8m0_k32 grid (bytes at non-adjacent offsets).
        s_base = get_ptr_as_int64(
            scales_u8,
            Int64(expert) * Int64(self.scales_kg) * Int64(self.scales_n_padded),
        )

        b_frag = []
        for j in cutlass.range_constexpr(2):
            g = Int32(2 * j) + (c >> Int32(1))
            row_base = Int32(self.b_off) + ((g * Int32(4) + slot) << Int32(9))
            pair = []
            for e in cutlass.range_constexpr(2):
                vec = (q << Int32(2)) + (h << Int32(1)) + Int32(e)
                pair.append(row_base + ((vec ^ (g & Int32(1))) << Int32(4)))
            b_frag.append(pair)
        s_frag = Int32(self.s_off) + (slot << Int32(8)) + (q << Int32(2))
        row0 = m_warp * Int32(self.rows_per_warp) + q
        a_frag = row0 * Int32(32) + (c << Int32(3))
        sfa_frag = Int32(A_STAGE_BYTES) + ((row0 + (h << Int32(3))) << Int32(2))
        warp_live = live_blocks - m_warp * Int32(self.mb_per_warp)

        for p in cutlass.range_constexpr(self.stages - 1):
            if Int32(p) < Int32(self.k_tiles):
                self._issue(
                    tid,
                    smem_base + Int32(p * self.stage_bytes),
                    Int32(p),
                    a_desc,
                    sf_desc,
                    b_desc,
                    s_base,
                    n_tile,
                )
            cute.arch.cp_async_commit_group()
        rd_base = smem_base
        wr_base = smem_base + Int32((self.stages - 1) * self.stage_bytes)
        stage_end = smem_base + Int32(self.stages * self.stage_bytes)
        k_tile = Int32(0)
        while k_tile < Int32(self.k_tiles):
            cute.arch.cp_async_wait_group(self.stages - 2)
            cute.arch.sync_threads()
            nxt = k_tile + Int32(self.stages - 1)
            if nxt < Int32(self.k_tiles):
                self._issue(tid, wr_base, nxt, a_desc, sf_desc, b_desc, s_base, n_tile)
            cute.arch.cp_async_commit_group()
            if warp_live > Int32(0):
                self._compute_stage(
                    rd_base, b_frag, s_frag, a_frag, sfa_frag, warp_live, acc
                )
            rd_base += Int32(self.stage_bytes)
            if rd_base == stage_end:
                rd_base = smem_base
            wr_base += Int32(self.stage_bytes)
            if wr_base == stage_end:
                wr_base = smem_base
            k_tile += Int32(1)

    # -- mainloop compute -----------------------------------------------------------

    @cute.jit
    def _compute_stage(
        self,
        rd_base: Int32,
        b_frag,
        s_frag: Int32,
        a_frag: Int32,
        sfa_frag: Int32,
        warp_live: Int32,
        acc,
    ):
        # B registers [jj][t][j]: one PRMT of the e = 0 / 1 words per register
        # (the packed W4A16 B layout is identical to the 4X variant).
        breg = cute.make_rmem_tensor((16,), Uint32)
        for j in cutlass.range_constexpr(2):
            w0 = ld_shared_v4_u32(rd_base + b_frag[j][0])
            w1 = ld_shared_v4_u32(rd_base + b_frag[j][1])
            for jj in cutlass.range_constexpr(4):
                breg[(jj * 2 + 0) * 2 + j] = prmt_b32(w0[jj], w1[jj], Uint32(0x6420))
                breg[(jj * 2 + 1) * 2 + j] = prmt_b32(w0[jj], w1[jj], Uint32(0x7531))
        # SFB [jj][t]: the (group0, group1) byte-pair of column 16 jj + 8 t + q
        # of this slot -- already the MMA's u32 scale word (no lift, no PRMT).
        sfb = cute.make_rmem_tensor((8,), Uint32)
        for jj in cutlass.range_constexpr(4):
            for t in cutlass.range_constexpr(2):
                sfb[jj * 2 + t] = ld_shared_u32(
                    rd_base + s_frag + Int32((16 * jj + 8 * t) << Int32(2))
                )
        self._mma_block(rd_base, a_frag, sfa_frag, breg, sfb, acc, 0)
        for mb in cutlass.range_constexpr(1, self.mb_per_warp):
            if warp_live > Int32(mb):
                self._mma_block(rd_base, a_frag, sfa_frag, breg, sfb, acc, mb)

    @cute.jit
    def _mma_block(
        self,
        rd_base: Int32,
        a_frag: Int32,
        sfa_frag: Int32,
        breg,
        sfb,
        acc,
        mb: cutlass.Constexpr[int],
    ):
        for plane in cutlass.range_constexpr(self.terms):
            base = rd_base + a_frag + Int32(plane * PLANE_STAGE_BYTES + mb * 16 * 32)
            a0, a2 = ld_shared_v2_u32(base)
            a1, a3 = ld_shared_v2_u32(base + Int32(8 * 32))
            sfa = ld_shared_u32(
                rd_base + sfa_frag + Int32(plane * PLANE_STAGE_BYTES + mb * 16 * 4)
            )
            for jj in cutlass.range_constexpr(4):
                for t in cutlass.range_constexpr(2):
                    idx = ((mb * 4 + jj) * 2 + t) * 4
                    d0, d1, d2, d3 = mxfp4_mma_m16n8k64_f32_e2m1_ue8m0(
                        acc[idx],
                        acc[idx + 1],
                        acc[idx + 2],
                        acc[idx + 3],
                        a0,
                        a1,
                        a2,
                        a3,
                        breg[(jj * 2 + t) * 2 + 0],
                        breg[(jj * 2 + t) * 2 + 1],
                        sfa,
                        sfb[jj * 2 + t],
                    )
                    acc[idx] = d0
                    acc[idx + 1] = d1
                    acc[idx + 2] = d2
                    acc[idx + 3] = d3

    # -- epilogues -----------------------------------------------------------------

    @cute.jit
    def _stage_epilogue(
        self,
        acc,
        smem_base: Int32,
        lane: Int32,
        m_warp: Int32,
        slot: Int32,
        alpha: Float32,
    ):
        """[128, 256] BF16 tile of ``acc * alpha`` (columns in chunk-slot order)."""
        q = lane >> Int32(2)
        col0 = (slot << Int32(6)) + ((lane & Int32(3)) << Int32(1))
        for mb in cutlass.range_constexpr(self.mb_per_warp):
            for half_row in cutlass.range_constexpr(2):
                row = (
                    m_warp * Int32(self.rows_per_warp)
                    + Int32(16 * mb + 8 * half_row)
                    + q
                )
                for jj in cutlass.range_constexpr(4):
                    for t in cutlass.range_constexpr(2):
                        idx = ((mb * 4 + jj) * 2 + t) * 4 + 2 * half_row
                        col = col0 + Int32(16 * jj + 8 * t)
                        st_shared_u32(
                            smem_base + row * Int32(EPI_ROW_BYTES) + (col << Int32(1)),
                            pack_f32x2_to_bfloat2(
                                acc[idx] * alpha, acc[idx + 1] * alpha
                            ),
                        )
        cute.arch.sync_threads()

    @cute.jit
    def _round(self, x: Float32) -> Float32:
        lo, _ = bfloat2_to_float2_scaled(pack_f32x2_to_bfloat2(x, x), Float32(1.0))
        return lo

    @cute.jit
    def _activate(self, gate: Float32, up: Float32) -> Float32:
        sigmoid = cute.arch.rcp_approx(
            Float32(1.0) + cute.math.exp(-gate, fastmath=self.fast_math)
        )
        return self._round(gate * sigmoid * up)

    @cute.jit
    def _store_fc2(
        self,
        acc,
        out_bf16,
        smem_base: Int32,
        tid: Int32,
        lane: Int32,
        m_warp: Int32,
        slot: Int32,
        n_tile: Int32,
        live_routes: Int32,
        alpha: Float32,
    ):
        self._stage_epilogue(acc, smem_base, lane, m_warp, slot, alpha)
        route_base = smem_base + Int32(self.shared_bytes - ROUTE_BYTES)
        for i in cutlass.range_constexpr(BLOCK_M * 32 // self.threads):
            idx = tid + Int32(i * self.threads)
            row = idx >> Int32(5)
            vec = idx & Int32(31)
            route = Int32(ld_shared_u32(route_base + (row << Int32(2))))
            if route < live_routes:
                v0, v1, v2, v3 = ld_shared_v4_u32(
                    smem_base + row * Int32(EPI_ROW_BYTES) + (vec << Int32(4))
                )
                dst = Int64(route) * Int64(self.hidden_size) + Int64(
                    n_tile * Int32(256) + (vec << Int32(3))
                )
                st_global_v4_u32(get_ptr_as_int64(out_bf16, dst), v0, v1, v2, v3)

    @cute.jit
    def _store_fc1(
        self,
        acc,
        out_q,
        out_sf,
        out_q2,
        out_sf2,
        smem_base: Int32,
        tid: Int32,
        lane: Int32,
        m_warp: Int32,
        slot: Int32,
        n_tile: Int32,
        live_routes: Int32,
        alpha: Float32,
    ):
        # Stage BF16 gate (cols 0..127) and up (128..255) after alpha.
        self._stage_epilogue(acc, smem_base, lane, m_warp, slot, alpha)
        route_base = smem_base + Int32(self.shared_bytes - ROUTE_BYTES)
        # Thread -> (row, K64 half of the 128 intermediate columns).
        if tid < Int32(2 * BLOCK_M):
            row = tid & Int32(BLOCK_M - 1)
            half = tid >> Int32(7)
            route = Int32(ld_shared_u32(route_base + (row << Int32(2))))
            if route < live_routes:
                row_base = smem_base + row * Int32(EPI_ROW_BYTES) + (half << Int32(7))
                words = cute.make_rmem_tensor((8,), Uint32)
                words2 = cute.make_rmem_tensor((8,), Uint32)
                scale_word = Uint32(0)
                scale_word2 = Uint32(0)
                for g in cutlass.range_constexpr(2):
                    # K32 group g = intermediate columns 32g..32g+31 of the
                    # 64-column half (gate and up read at the same column).
                    vals = cute.make_rmem_tensor((32,), Float32)
                    codes = cute.make_rmem_tensor((32,), Uint32)
                    for v in cutlass.range_constexpr(4):
                        gw = ld_shared_v4_u32(row_base + Int32(64 * g + 16 * v))
                        uw = ld_shared_v4_u32(row_base + Int32(256 + 64 * g + 16 * v))
                        for i in cutlass.range_constexpr(4):
                            g0, g1 = bfloat2_to_float2_scaled(gw[i], Float32(1.0))
                            u0, u1 = bfloat2_to_float2_scaled(uw[i], Float32(1.0))
                            vals[8 * v + 2 * i] = self._activate(g0, u0)
                            vals[8 * v + 2 * i + 1] = self._activate(g1, u1)
                    out = _mxfp4_quantize32(vals, codes)
                    words[4 * g + 0] = out[0]
                    words[4 * g + 1] = out[1]
                    words[4 * g + 2] = out[2]
                    words[4 * g + 3] = out[3]
                    scale_word = scale_word | (out[4] << Uint32(8 * g))
                    if cutlass.const_expr(self.terms == 2):
                        res = cute.make_rmem_tensor((32,), Float32)
                        codes2 = cute.make_rmem_tensor((32,), Uint32)
                        for m in cutlass.range_constexpr(32):
                            res[m] = vals[m] - _mxfp4_dequant(codes[m], out[4])
                        out2 = _mxfp4_quantize32(res, codes2)
                        words2[4 * g + 0] = out2[0]
                        words2[4 * g + 1] = out2[1]
                        words2[4 * g + 2] = out2[2]
                        words2[4 * g + 3] = out2[3]
                        scale_word2 = scale_word2 | (out2[4] << Uint32(8 * g))
                k64 = n_tile * Int32(2) + half
                q_off = Int64(route) * Int64(self.intermediate_size // 8) + Int64(
                    k64 * Int32(8)
                )
                sf_off = Int64(route) * Int64(self.intermediate_size // 64) + Int64(k64)
                _store_slice_mxfp4(
                    get_ptr_as_int64(out_q, q_off),
                    get_ptr_as_int64(out_sf, sf_off),
                    words,
                    scale_word,
                )
                if cutlass.const_expr(self.terms == 2):
                    _store_slice_mxfp4(
                        get_ptr_as_int64(out_q2, q_off),
                        get_ptr_as_int64(out_sf2, sf_off),
                        words2,
                        scale_word2,
                    )

    # -- kernel --------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        a_q,
        a_sf,
        a_q2,
        a_sf2,
        weights_i32,
        scales_u8,
        w_global,
        a_gscale,
        q_gscale,
        packed_route_indices,
        block_expert_ids,
        packed_route_count,
        out_q,
        out_sf,
        out_q2,
        out_sf2,
        out_bf16,
        live_routes: cutlass.Int32,
        gs_stride: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        lane = tid & Int32(31)
        warp = tid >> Int32(5)
        m_warp = warp >> Int32(2)
        slot = warp & Int32(3)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_bytes // 4], 1024
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        route_base = smem_base + Int32(self.shared_bytes - ROUTE_BYTES)

        blocks = packed_route_count[0].to(Int32) // Int32(BLOCK_M)
        tiles = blocks * Int32(self.n_tiles)
        tile = Int32(bidx)
        while tile < tiles:
            block = tile // Int32(self.n_tiles)
            n_tile = tile - block * Int32(self.n_tiles)
            expert = block_expert_ids[block].to(Int32)
            if expert >= Int32(0):
                # Gather rows (token for FC1, route for FC2); live routes form a
                # prefix of the block.
                if tid < Int32(BLOCK_M):
                    route = packed_route_indices[block * Int32(BLOCK_M) + tid].to(Int32)
                    src = Int32(0)
                    if route < live_routes:
                        src = route
                        if cutlass.const_expr(self.fc1):
                            src = route // Int32(self.top_k)
                    st_shared_u32(route_base + (tid << Int32(2)), Uint32(src))
                cute.arch.sync_threads()
                live_blocks = Int32(0)
                for probe in cutlass.range_constexpr(BLOCK_M // 16):
                    r = packed_route_indices[
                        block * Int32(BLOCK_M) + Int32(16 * probe)
                    ].to(Int32)
                    if r < live_routes:
                        live_blocks = Int32(probe + 1)
                # alpha: the e8m0_k32 dequant convention is
                #   value = e2m1_code * 2^(scale_byte - 127) * global_scale
                # with global_scale = the raw per-expert source global (the
                # 2**119 lift belongs to the DECODE unpack path, which
                # materializes E8M0 as BF16 halves -- not to this MMA, whose
                # operands are native E2M1 with UE8M0 scale pairs).  The
                # dynamic E8M0 quantizer represents activations with local
                # per-K32 scales only and never applies activation globals,
                # so dividing by the input global scale would be unsound
                # compensation: weight globals alone scale the result.
                # Activation globals must therefore be unit (enforced in
                # run_w4a16_mxfp4_prefill).
                alpha = w_global[expert].to(Float32)

                acc = cute.make_rmem_tensor((self.acc_size,), Float32)
                acc.fill(0.0)
                self._mainloop(
                    a_q,
                    a_sf,
                    a_q2,
                    a_sf2,
                    weights_i32,
                    scales_u8,
                    smem_base,
                    tid,
                    lane,
                    m_warp,
                    slot,
                    expert,
                    n_tile,
                    live_blocks,
                    acc,
                )
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                # The epilogue reads route ids, not gather rows.
                if tid < Int32(BLOCK_M):
                    route = packed_route_indices[block * Int32(BLOCK_M) + tid].to(Int32)
                    st_shared_u32(route_base + (tid << Int32(2)), Uint32(route))
                cute.arch.sync_threads()
                if cutlass.const_expr(self.fc1):
                    self._store_fc1(
                        acc,
                        out_q,
                        out_sf,
                        out_q2,
                        out_sf2,
                        smem_base,
                        tid,
                        lane,
                        m_warp,
                        slot,
                        n_tile,
                        live_routes,
                        alpha,
                    )
                else:
                    self._store_fc2(
                        acc,
                        out_bf16,
                        smem_base,
                        tid,
                        lane,
                        m_warp,
                        slot,
                        n_tile,
                        live_routes,
                        alpha,
                    )
                cute.arch.sync_threads()
            tile += Int32(gdimx)


# -- host: plan-time launch set and the per-call pipeline ------------------------

MXFP4_PREFILL_ROUTE_BLOCK = BLOCK_M
_FAKE_ELEMENTS = 1 << 30
_LAUNCH_CACHE: dict[tuple, tuple] = {}


def mxfp4_prefill_scale_format_supported(scale_format: str | None) -> bool:
    """Whether a prepared W4A16 scale format is the MXFP4-A4 payload.

    Admission requires the e8m0_k32 (MXFP4/X4T) scales; other W4A16 formats
    (e.g. e4m3_k16) belong to NVFP4-style weights and never dispatch A4.
    Used as a cheap eligibility short-circuit before any device work.
    """
    return scale_format == "e8m0_k32"


def mxfp4_prefill_min_tokens() -> int:
    """Plan-time opt-in: W4A16 calls with at least this many tokens run MXFP4
    activations over the packed weights (``B12X_W4A16_MXFP4_PREFILL_MIN_TOKENS``,
    0 = off)."""
    value = int(os.environ.get("B12X_W4A16_MXFP4_PREFILL_MIN_TOKENS", "0") or 0)
    return max(value, 0)


def _terms2_allowed() -> bool:
    """Explicit opt-in for the unqualified terms=2 residual plane."""
    return os.environ.get("B12X_W4A16_MXFP4_PREFILL_ALLOW_TERMS2", "") == "1"


def mxfp4_prefill_terms() -> int:
    """Activation planes: 1 = MXFP4, 2 = MXFP4 value + MXFP4 residual;
    ``B12X_W4A16_MXFP4_PREFILL_TERMS``.

    terms=2 is gated: the residual plane is unqualified, so it additionally
    requires the explicit ``B12X_W4A16_MXFP4_PREFILL_ALLOW_TERMS2=1`` opt-in.
    """
    value = int(os.environ.get("B12X_W4A16_MXFP4_PREFILL_TERMS", "1") or 1)
    if value not in (1, 2):
        raise ValueError("B12X_W4A16_MXFP4_PREFILL_TERMS must be 1 or 2")
    if value == 2 and not _terms2_allowed():
        raise ValueError(
            "B12X_W4A16_MXFP4_PREFILL_TERMS=2 is gated (the MXFP4 residual "
            "plane is unqualified); set B12X_W4A16_MXFP4_PREFILL_ALLOW_TERMS2=1 "
            "to opt in"
        )
    return value


def mxfp4_prefill_warps() -> int:
    """GEMM CTA size (``B12X_W4A16_MXFP4_PREFILL_WARPS``): 8 warps with 64-row
    warp tiles (default) or 16 warps with 32-row tiles."""
    value = int(os.environ.get("B12X_W4A16_MXFP4_PREFILL_WARPS", "8") or 8)
    if value not in (8, 16):
        raise ValueError("B12X_W4A16_MXFP4_PREFILL_WARPS must be 8 or 16")
    return value


def mxfp4_prefill_supported(
    *,
    prepared_layout: str,
    scale_format: str,
    activation: str,
    is_gated: bool,
    dtype: torch.dtype,
    hidden_size: int,
    intermediate_size: int,
    terms: int = 1,
    swiglu_limit: float | None = None,
) -> bool:
    """Plan-time admission.

    Only the repacked K16/N64 payload (``'packed'``) is admitted: the GEMM has
    no layout branch and always reads that payload's B addressing, while the
    ``'modelopt'``/native preparation keeps the original weights without
    ``_repack_weight`` (prepare.py native/X4T-native branches), so a native
    object would produce incorrect B operands.  terms=2 is gated behind
    ``B12X_W4A16_MXFP4_PREFILL_ALLOW_TERMS2=1`` (unqualified residual plane).

    Clamped SiLU (non-null ``swiglu_limit``) is rejected: the W4A16 kernel
    clamps gate/up to the limit before activation, while the A4 epilogue
    computes a plain SiLU — crossing the A4 threshold would silently change
    activation semantics.  Clamped models retain W4A16.
    """
    if terms not in (1, 2):
        return False
    if terms == 2 and not _terms2_allowed():
        return False
    if swiglu_limit is not None:
        return False
    _dt = str(dtype).removeprefix("torch.")
    return (
        prepared_layout == "packed"
        and scale_format == "e8m0_k32"
        and activation == "silu"
        and bool(is_gated)
        and _dt == "bfloat16"
        and hidden_size % 256 == 0
        and intermediate_size % 128 == 0
    )


@dataclass(frozen=True)
class W4A16Mxfp4PrefillLaunches:
    """Compiled MXFP4 prefill pipeline for one W4A16 capacity."""

    tokens: int
    min_tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    topk: int
    grid: int
    terms: int
    quant: object
    fc1: object
    fc2: object
    topk_sum: object
    route_pack: object

    def carriers(self) -> tuple[object, ...]:
        return (
            self.quant,
            self.fc1,
            self.fc2,
            self.topk_sum,
            *self.route_pack.carriers(),
        )

    def scratch_bytes(self, tokens: int) -> int:
        """Bytes of ``intermediate_cache2`` the pipeline carves for ``tokens``."""
        return _carve_layout(self, int(tokens))[-1]


def _fake(dtype, align=16):
    return cute.runtime.make_fake_compact_tensor(
        dtype, (_FAKE_ELEMENTS,), assumed_align=align
    )


def _compile_mxfp4_kernels(
    *, hidden_size: int, intermediate_size: int, topk: int, fast_math: bool, terms: int
):
    from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
    from b12x._lib.utils import current_cuda_stream

    warps = mxfp4_prefill_warps()
    device = torch.cuda.current_device()
    key = (
        int(device),
        int(hidden_size),
        int(intermediate_size),
        int(topk),
        bool(fast_math),
        4,
        int(terms),
        warps,
    )
    cached = _LAUNCH_CACHE.get(key)
    if cached is not None:
        return cached
    stream = current_cuda_stream()
    quant = b12x_compile(
        Mxfp4A4Quantize(size_k=hidden_size, terms=terms),
        _fake(cutlass.BFloat16),
        _fake(cutlass.Uint32),
        _fake(cutlass.Uint32, 4),
        _fake(cutlass.Uint32),
        _fake(cutlass.Uint32, 4),
        Int32(1),
        stream,
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.mxfp4_prefill.quant", 2, key
        ),
    )
    gemms = []
    for phase in ("fc1", "fc2"):
        gemms.append(
            b12x_compile(
                Mxfp4A4PrefillGemm(
                    phase=phase,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    top_k=topk,
                    fast_math=fast_math,
                    terms=terms,
                    warps=warps,
                ),
                _fake(cutlass.Uint32),
                _fake(cutlass.Uint32, 4),
                _fake(cutlass.Uint32),
                _fake(cutlass.Uint32, 4),
                _fake(cutlass.Int32),
                _fake(cutlass.Uint8),
                _fake(cutlass.Float32, 4),
                _fake(cutlass.Float32, 4),
                _fake(cutlass.Float32, 4),
                _fake(cutlass.Int32, 4),
                _fake(cutlass.Int32, 4),
                _fake(cutlass.Int32, 4),
                _fake(cutlass.Uint32),
                _fake(cutlass.Uint32, 4),
                _fake(cutlass.Uint32),
                _fake(cutlass.Uint32, 4),
                _fake(cutlass.BFloat16),
                Int32(1),
                Int32(1),
                Int32(1),
                stream,
                compile_spec=KernelCompileSpec.from_key(
                    f"moe.w4a16.mxfp4_prefill.{phase}", 2, key
                ),
            )
        )
    result = (quant, gemms[0], gemms[1])
    _LAUNCH_CACHE[key] = result
    return result


def compile_w4a16_mxfp4_prefill(
    *,
    tokens: int,
    min_tokens: int,
    topk: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    sms: int,
    ordinal: int,
    fast_math: bool,
    terms: int = 1,
) -> W4A16Mxfp4PrefillLaunches:
    from b12x.moe._shared.kernels.w4a16.kernel import compile_w4a16_topk_sum
    from b12x.moe._shared.kernels.w4a16.route_pack import (
        compile_w4a16_route_pack_launches,
    )

    if terms not in (1, 2):
        raise ValueError("MXFP4 prefill terms must be 1 or 2")
    if terms == 2 and not _terms2_allowed():
        raise ValueError(
            "MXFP4 prefill terms=2 is gated (the residual plane is "
            "unqualified); set B12X_W4A16_MXFP4_PREFILL_ALLOW_TERMS2=1 to opt in"
        )

    quant, fc1, fc2 = _compile_mxfp4_kernels(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        topk=topk,
        fast_math=fast_math,
        terms=terms,
    )
    topk_sum = compile_w4a16_topk_sum(
        m=tokens,
        topk=topk,
        hidden_size=hidden_size,
        element_dtype="bf16",
        apply_topk_weights=True,
    )
    route_pack = compile_w4a16_route_pack_launches(
        tokens=tokens,
        topk=topk,
        block_size=MXFP4_PREFILL_ROUTE_BLOCK,
        num_experts=num_experts,
        ordinal=ordinal,
    )
    return W4A16Mxfp4PrefillLaunches(
        tokens=int(tokens),
        min_tokens=int(min_tokens),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_experts=int(num_experts),
        topk=int(topk),
        grid=int(sms),
        terms=int(terms),
        quant=quant,
        fc1=fc1,
        fc2=fc2,
        topk_sum=topk_sum,
        route_pack=route_pack,
    )


def _carve_layout(launches: W4A16Mxfp4PrefillLaunches, tokens: int):
    """Byte offsets of the pipeline buffers inside ``intermediate_cache2``.

    Identical to the A4 carve: x_q/x_sf are packed MXFP4 (u8 pairs of nibbles,
    1/16 of the K dimension per row) and act_q/act_sf the per-route
    intermediate.  ``x_sf`` holds one u32 per row per K64 slice; ``act_sf``
    one u32 per route per K64 slice of the intermediate.
    """
    h, i, e = launches.hidden_size, launches.intermediate_size, launches.num_experts
    routes = tokens * launches.topk
    planes = launches.terms
    sizes = (
        planes * tokens * h // 2,  # x_q (per plane)
        planes * tokens * (h // 64) * 4,  # x_sf (u32 per K64 slice)
        planes * routes * i // 2,  # act_q
        planes * routes * (i // 64) * 4,  # act_sf
        launches.route_pack.max_packed_routes * 4,  # packed_route_indices
        launches.route_pack.max_route_blocks * 4,  # block_expert_ids
        4,  # packed_route_count
        (e + 1) * 4,  # expert_offsets
        e * 4,  # expert_counts
    )
    offsets = []
    offset = 0
    for size in sizes:
        offsets.append(offset)
        offset = (offset + size + 255) // 256 * 256
    return tuple(zip(offsets, sizes)), offset


def _planes(region: torch.Tensor, terms: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an activation region into its planes (plane 2 aliases plane 1 when
    there is a single term; the kernels never touch it then)."""
    if terms == 1:
        return region, region
    half = region.numel() // 2
    return region.narrow(0, 0, half), region.narrow(0, half, half)


def mxfp4_prefill_fits(
    launches: W4A16Mxfp4PrefillLaunches,
    *,
    tokens: int,
    intermediate_cache13: torch.Tensor,
    intermediate_cache2: torch.Tensor,
) -> bool:
    routes = int(tokens) * launches.topk
    return (
        intermediate_cache13.numel() * intermediate_cache13.element_size()
        >= routes * launches.hidden_size * 2
        and intermediate_cache2.numel() * intermediate_cache2.element_size()
        >= launches.scratch_bytes(tokens)
    )


# --- activation-global admission (capture-safe) -------------------------------
# The unit-global contract (a_gscale all 1.0) is validated ONCE per scale-tensor
# contract and the trusted host-side verdict is cached; the captured execution
# path never inspects device values through Python.  The old per-call
# ``bool(torch.all(...))`` check copied a CUDA reduction to the host (a device
# sync on every eager call) and cannot run under CUDA graph capture.

# fingerprint -> _ActivationGlobalAdmissionEntry (trusted verdict + STRONG
# references to the exact validated tensors), keyed by
# (data_ptr, _version, numel, dtype) of each scale tensor.  The fingerprint is
# only a fast first filter: a cache HIT additionally requires the cached tensor
# objects to BE the queried tensors (identity check).  Otherwise a freed
# tensor whose data_ptr the CUDA caching allocator reuses for a different
# tensor (same numel/dtype, coincident _version) could inherit a stale verdict
# (ABA).  The strong references keep the validated tensors alive for the
# cached lifetime, so their addresses cannot be reused while the entry is
# live; the identity check defends whatever the allocator still does.  The
# cache is bounded (_ACTIVATION_GLOBAL_ADMISSION_MAX entries, oldest evicted)
# so validation cannot pin unbounded device memory.
# 512: the serving checkpoint has 69 MoE layers; with distinct scale-tensor
# pairs per layer a 64-entry cache evicted early layers before graph capture
# (Codex A3). 512 covers worst-case distinct pairs with headroom.
_ACTIVATION_GLOBAL_ADMISSION_MAX = 512
_ACTIVATION_GLOBAL_ADMISSION: dict[tuple, "_ActivationGlobalAdmissionEntry"] = {}


class _ActivationGlobalAdmissionEntry:
    """Cached admission verdict plus strong refs to the validated tensors."""

    __slots__ = ("a1_gscale", "a2_gscale", "verdict")

    def __init__(
        self, a1_gscale: torch.Tensor, a2_gscale: torch.Tensor, verdict: str
    ) -> None:
        self.a1_gscale = a1_gscale
        self.a2_gscale = a2_gscale
        self.verdict = verdict

    def matches(self, a1_gscale: torch.Tensor, a2_gscale: torch.Tensor) -> bool:
        """Identity check: the cached tensors must BE the queried ones."""
        return self.a1_gscale is a1_gscale and self.a2_gscale is a2_gscale

    def __eq__(self, other) -> bool:
        # Verdict-string comparison keeps the historical
        # ``cache.get(fingerprint) == "unit"`` assertions meaningful.
        if isinstance(other, str):
            return self.verdict == other
        if isinstance(other, _ActivationGlobalAdmissionEntry):
            return (
                self.verdict == other.verdict
                and self.a1_gscale is other.a1_gscale
                and self.a2_gscale is other.a2_gscale
            )
        return NotImplemented


def _cache_activation_global_verdict(
    fingerprint: tuple, a1_gscale: torch.Tensor, a2_gscale: torch.Tensor, verdict: str
) -> None:
    _ACTIVATION_GLOBAL_ADMISSION[fingerprint] = _ActivationGlobalAdmissionEntry(
        a1_gscale, a2_gscale, verdict
    )
    while len(_ACTIVATION_GLOBAL_ADMISSION) > _ACTIVATION_GLOBAL_ADMISSION_MAX:
        # dict preserves insertion order: pop the oldest entry.
        _ACTIVATION_GLOBAL_ADMISSION.pop(next(iter(_ACTIVATION_GLOBAL_ADMISSION)))


def _activation_global_fingerprint(*tensors: torch.Tensor) -> tuple:
    return tuple(
        (int(t.data_ptr()), int(t._version), int(t.numel()), t.dtype) for t in tensors
    )


def validate_activation_globals(
    a1_gscale: torch.Tensor, a2_gscale: torch.Tensor
) -> bool:
    """Validate the unit activation-global contract once (bind-time entry).

    Runs the all-ones device check a single time and records the trusted
    host-side verdict in ``_ACTIVATION_GLOBAL_ADMISSION`` keyed by a
    fingerprint of the scale tensors ``(data_ptr, _version, numel, dtype)``.
    Call this at bind time, before CUDA graph capture; ``run_w4a16_mxfp4_prefill``
    consults the cache afterwards and performs no device inspection on a
    validated contract.  Returns True when both globals are all 1.0.  Mutating
    a scale tensor in place bumps its ``_version`` and invalidates the cached
    verdict.
    """
    entry = _ACTIVATION_GLOBAL_ADMISSION.get(
        _activation_global_fingerprint(a1_gscale, a2_gscale)
    )
    if (
        entry is not None
        and entry.a1_gscale is a1_gscale
        and entry.a2_gscale is a2_gscale
    ):
        return entry.verdict == "unit"  # cached: no device sync
    verdict = (
        "unit"
        if bool(torch.all(a1_gscale == 1.0)) and bool(torch.all(a2_gscale == 1.0))
        else "nonunit"
    )
    _cache_activation_global_verdict(
        _activation_global_fingerprint(a1_gscale, a2_gscale),
        a1_gscale,
        a2_gscale,
        verdict,
    )
    return verdict == "unit"


def _admit_activation_globals(a1_gscale: torch.Tensor, a2_gscale: torch.Tensor) -> None:
    """Trusted admission of the unit-global contract; no device sync on hit.

    - cached verdict (fingerprint match AND identity match on the exact
      validated tensor objects): proceed with NO device inspection and NO
      sync.  A fingerprint match with different tensor objects (address
      reuse: freed tensors' data_ptr recycled by the CUDA caching allocator)
      is a MISS -- the stale verdict must never be served (ABA safety);
    - no cached verdict, not capturing: run the check once and cache it;
    - no cached verdict, capturing (``torch.cuda.is_current_stream_capturing()``):
      raise RuntimeError telling the caller to validate at bind time --
      never silently bypass the contract check.
    """
    entry = _ACTIVATION_GLOBAL_ADMISSION.get(
        _activation_global_fingerprint(a1_gscale, a2_gscale)
    )
    if entry is not None and entry.matches(a1_gscale, a2_gscale):
        if entry.verdict == "nonunit":
            raise ValueError(
                "mxfp4 prefill: nonunit activation globals are unsupported on the "
                "dynamic-E8M0 path (a_gscale must be all 1.0)"
            )
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "mxfp4 prefill: activation globals are not validated for CUDA graph "
            "capture; call validate_activation_globals(a1_gscale, a2_gscale) at "
            "bind time (before capture). The runner must not inspect device "
            "values through Python under capture."
        )
    if not validate_activation_globals(a1_gscale, a2_gscale):
        raise ValueError(
            "mxfp4 prefill: nonunit activation globals are unsupported on the "
            "dynamic-E8M0 path (a_gscale must be all 1.0)"
        )


def mxfp4_prefill_activation_globals_admitted(
    a1_gscale: torch.Tensor, a2_gscale: torch.Tensor
) -> bool:
    """Capture-safe bind-time admission query for the unit-global contract.

    Namesafe host hook for the serving dispatch (``fused_moe._impl`` bind):
    True when the contract is admitted — a trusted cached ``"unit"`` verdict,
    or (only when NOT capturing) a fresh validation.  Under CUDA graph capture
    with a cold cache this returns False instead of inspecting device values,
    so bind stays allocation-free and capture-safe and the call keeps its
    W4A16 path; ``run_w4a16_mxfp4_prefill`` re-checks through
    ``_admit_activation_globals`` and never launches on an unvalidated
    contract.  No numerics, no launches: host-side admission only.
    """
    entry = _ACTIVATION_GLOBAL_ADMISSION.get(
        _activation_global_fingerprint(a1_gscale, a2_gscale)
    )
    if entry is not None and entry.matches(a1_gscale, a2_gscale):
        return entry.verdict == "unit"
    if torch.cuda.is_current_stream_capturing():
        return False
    return validate_activation_globals(a1_gscale, a2_gscale)


def _admit_x4t_payload(prepared):
    """X4T payload admission, independent of the decode guard (pre-launch).

    Plane presence is inspected independently of program presence: a prepared
    object carrying X4T scale planes (``x4t_w13_scale``/``x4t_w2_scale``) MUST
    also carry paired programs (the only supported decoder), otherwise it is
    rejected here before any launch instead of silently falling through to
    the GEMM.  Malformed plane metadata (exactly one plane present) is
    rejected regardless of the program guard.  Non-X4T prepared objects (no
    X4T attributes at all, e.g. plain ``prepare_w4a16_fp4_e8m0_k32_weights``
    outputs) pass unchanged.  Returns
    ``(x4t_packed_pair_programs, x4t_w13_scale, x4t_w2_scale)``.
    """
    x4t_programs = getattr(prepared, "x4t_packed_pair_programs", None)
    x4t_w13_scale = getattr(prepared, "x4t_w13_scale", None)
    x4t_w2_scale = getattr(prepared, "x4t_w2_scale", None)
    if x4t_w13_scale is not None or x4t_w2_scale is not None:
        if (x4t_w13_scale is None) != (x4t_w2_scale is None):
            raise ValueError("prepared X4T weights have incomplete scale metadata")
        if x4t_programs is None:
            raise ValueError(
                "mxfp4 prefill: X4T scale planes without paired programs are "
                "not supported: the runner reconstructs routed scale scratch "
                "only through the paired-program decoder "
                "(decode_x4t_packed_scale_pair); the non-program "
                "decode_x4t_tp12_w4a16_scales arm is not implemented here. "
                "Ship x4t_packed_pair_programs (prepare_w4a16_x4t_weights "
                "builds them for 64-row exception tasks) or use a non-X4T "
                "prepared object."
            )
    return x4t_programs, x4t_w13_scale, x4t_w2_scale


def run_w4a16_mxfp4_prefill(
    a: torch.Tensor,
    prepared,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    intermediate_cache13: torch.Tensor,
    intermediate_cache2: torch.Tensor,
    output: torch.Tensor,
    launches: W4A16Mxfp4PrefillLaunches,
    x4t_scales_expanded: bool = False,
) -> torch.Tensor:
    """Route pack (128) -> MXFP4 quant -> FC1 -> FC2 -> FP32-weighted top-k sum.

    Every buffer is a view of caller scratch: ``intermediate_cache2`` holds the
    MXFP4 activations and the route pack, ``intermediate_cache13`` the per-route
    BF16 FC2 rows.

    Only the repacked K16/N64 ``packed`` weight payload is admitted (see
    ``mxfp4_prefill_supported``); routed X4T scale scratch is reconstructed
    here after route packing and before FC1, matching stock ``run_w4a16_moe``.
    """
    from cutlass.cute.runtime import make_ptr

    from b12x._lib.utils import current_cuda_stream
    from b12x.moe._shared.kernels.w4a16.kernel import _w4a16_topk_sum_launch_flat
    from b12x.moe._shared.kernels.w4a16.route_pack import pack_topk_routes_by_expert

    weight_layout = getattr(prepared, "weight_layout", "packed")
    if weight_layout != "packed":
        raise ValueError(
            "run_w4a16_mxfp4_prefill admits only the repacked K16/N64 "
            f"'packed' weight payload; got weight_layout={weight_layout!r}. "
            "The GEMM has no layout branch and reads only the repacked "
            "payload's B addressing; a 'modelopt'/native payload keeps the "
            "original weights without _repack_weight and would produce "
            "incorrect B operands."
        )
    # Activation globals are unsupported on the dynamic-E8M0 path: the
    # quantizer absorbs activation scale per K32 group and never applies a
    # global, so nonunit globals would silently scale the output wrong.
    # Admission is capture-safe: a trusted cached verdict (bind-time
    # ``validate_activation_globals``) passes with no device inspection and no
    # sync; an unvalidated contract is checked once when not capturing and
    # rejected under capture with a bind-time RuntimeError (see
    # ``_admit_activation_globals``).
    _admit_activation_globals(a1_gscale, a2_gscale)
    # X4T payload admission, pre-launch: planes require paired programs;
    # malformed or unpaired X4T metadata is rejected before any kernel launch
    # (no silent fallthrough to the GEMM).  Non-X4T payloads pass unchanged.
    x4t_programs, x4t_w13_scale, x4t_w2_scale = _admit_x4t_payload(prepared)
    tokens = int(a.shape[0])
    routes = tokens * launches.topk
    regions, _ = _carve_layout(launches, tokens)
    raw = intermediate_cache2.view(-1).view(torch.uint8)
    views = [raw.narrow(0, off, size) for off, size in regions]
    x_q, x_sf, act_q, act_sf, pir, beid, prc, offsets, counts = (
        v.view(torch.int32) for v in views
    )
    x_q, x_q2 = _planes(x_q, launches.terms)
    x_sf, x_sf2 = _planes(x_sf, launches.terms)
    act_q, act_q2 = _planes(act_q, launches.terms)
    act_sf, act_sf2 = _planes(act_sf, launches.terms)
    y = intermediate_cache13.view(-1).narrow(0, 0, routes * launches.hidden_size)
    pack_topk_routes_by_expert(
        topk_ids,
        MXFP4_PREFILL_ROUTE_BLOCK,
        launches.num_experts,
        packed_route_indices=pir,
        block_expert_ids=beid,
        packed_route_count=prc,
        expert_offsets=offsets,
        expert_counts=counts,
        launches=launches.route_pack,
    )

    stream = current_cuda_stream()
    # --- X4T routed scale-scratch reconstruction ----------------------------
    # Equivalent of stock run_w4a16_moe's pre-compute X4T scale expansion
    # (kernel.py:13804-13847).  For X4T prepared objects the w13_scale/w2_scale
    # grids are caller-owned reusable scratch shared by every MoE layer on one
    # stream (prepare_w4a16.py:1349-1355 docstring: "caller-owned reusable scale
    # grids ... shared by every MoE layer because each layer's decode and GEMM
    # are ordered on one CUDA stream"), so each routed call must expand the
    # current layer's scale planes after route packing and before FC1 or the
    # GEMM reads another layer's (or uninitialized) scales.  Stock's block is
    # inline in kernel.py, so this is a line-equivalent mirror of its packed
    # route arm (this runner always packs routes, i.e. stock's
    # use_direct_topk_routes=False), reusing the same decode entry points;
    # nothing is factored out of kernel.py.  Plane/program presence was
    # already admitted pre-launch (``_admit_x4t_payload``): planes require
    # paired programs, so the decode below runs only for well-formed paired
    # X4T; unpaired X4T was rejected before any launch.
    # Skipped when the caller pre-expanded every expert's scales into this
    # payload's scale scratch (x4t_scales_expanded=True): the all-expert
    # expansion is a superset of this selective decode for both route arms
    # (counts and sorted blocks), so the inline launch would be duplicate work.
    if x4t_programs is not None and not x4t_scales_expanded:
        # kernel.py:13804-13833 with use_direct_topk_routes=False: start
        # from the caller-owned expert counts, or bound the sorted
        # block_expert_ids list (retained paired programs, indices 1/3).
        from b12x._lib.quant.x4t_packed_scales import decode_x4t_packed_scale_pair

        counts_mode = True
        active = counts
        sorted_ids = False
        block_bound = min(beid.numel(), topk_ids.numel())
        if block_bound < int(getattr(prepared, "num_experts", launches.num_experts)):
            # A nonempty packed block contains at least one routed row.
            # Its sorted expert list therefore needs no more entries than
            # the routed-row count; the packer fills unused entries with -1.
            # Bounding the grid avoids scheduling all experts for decode.
            active = beid[:block_bound]
            counts_mode = False
            sorted_ids = True
        if active is None:
            raise ValueError("Packed X4T routing requires caller-owned expert counts")
        if sorted_ids:
            program_index = 3
        elif counts_mode:
            program_index = 1
        else:
            program_index = 2 if active.dtype == torch.int64 else 0
        decode_x4t_packed_scale_pair(
            x4t_w13_scale,
            x4t_w2_scale,
            active,
            prepared.w13_scale,
            prepared.w2_scale,
            expert_counts=counts_mode,
            expert_ids_sorted=sorted_ids,
            program=x4t_programs[program_index],
            stream=stream,
        )
    # kernel.py's non-program arm (13834-13845, decode_x4t_tp12_
    # w4a16_scales) is not mirrored here and is unreachable in this
    # runner: paired programs are the only supported decoder, and an X4T
    # payload without them is rejected pre-launch (see above) rather than
    # decoded or silently skipped.

    def ptr(dtype, tensor, align=16):
        return make_ptr(
            dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
        )

    a1 = a1_gscale.view(-1)
    a2 = a2_gscale.view(-1)
    gs_stride = Int32(0 if a2.numel() == 1 else 1)
    w13 = prepared.w13.view(torch.int32)
    w2 = prepared.w2.view(torch.int32)
    s13 = prepared.w13_scale.view(torch.uint8)
    s2 = prepared.w2_scale.view(torch.uint8)
    u32 = cutlass.Uint32
    launches.quant(
        ptr(cutlass.BFloat16, a),
        ptr(u32, x_q),
        ptr(u32, x_sf, 4),
        ptr(u32, x_q2),
        ptr(u32, x_sf2, 4),
        Int32(tokens),
        stream,
    )
    launches.fc1(
        ptr(u32, x_q),
        ptr(u32, x_sf, 4),
        ptr(u32, x_q2),
        ptr(u32, x_sf2, 4),
        ptr(cutlass.Int32, w13),
        ptr(cutlass.Uint8, s13),
        ptr(cutlass.Float32, prepared.w13_global_scale, 4),
        ptr(cutlass.Float32, a1, 4),
        ptr(cutlass.Float32, a2, 4),
        ptr(cutlass.Int32, pir, 4),
        ptr(cutlass.Int32, beid, 4),
        ptr(cutlass.Int32, prc, 4),
        ptr(u32, act_q),
        ptr(u32, act_sf, 4),
        ptr(u32, act_q2),
        ptr(u32, act_sf2, 4),
        ptr(cutlass.BFloat16, y),
        Int32(routes),
        gs_stride,
        Int32(launches.grid),
        stream,
    )
    launches.fc2(
        ptr(u32, act_q),
        ptr(u32, act_sf, 4),
        ptr(u32, act_q2),
        ptr(u32, act_sf2, 4),
        ptr(cutlass.Int32, w2),
        ptr(cutlass.Uint8, s2),
        ptr(cutlass.Float32, prepared.w2_global_scale, 4),
        ptr(cutlass.Float32, a2, 4),
        ptr(cutlass.Float32, a2, 4),
        ptr(cutlass.Int32, pir, 4),
        ptr(cutlass.Int32, beid, 4),
        ptr(cutlass.Int32, prc, 4),
        ptr(u32, act_q),
        ptr(u32, act_sf, 4),
        ptr(u32, act_q2),
        ptr(u32, act_sf2, 4),
        ptr(cutlass.BFloat16, y),
        Int32(routes),
        gs_stride,
        Int32(launches.grid),
        stream,
    )
    _w4a16_topk_sum_launch_flat(
        y,
        output.view(-1),
        tokens,
        launches.topk,
        launches.hidden_size,
        "bf16",
        torch.cuda.current_stream().cuda_stream,
        topk_weights=topk_weights.view(-1),
        launcher=launches.topk_sum,
    )
    return output


__all__ = [
    "Mxfp4A4PrefillGemm",
    "Mxfp4A4Quantize",
    "PI",
    "WORD_ORDER",
    "W4A16Mxfp4PrefillLaunches",
    "mxfp4_prefill_fits",
    "mxfp4_prefill_min_tokens",
    "mxfp4_prefill_supported",
    "compile_w4a16_mxfp4_prefill",
    "run_w4a16_mxfp4_prefill",
    "validate_activation_globals",
]
