"""One-launch KDA speculative verification: short convolution + recovery verify + gated RMSNorm.

The production verify path launches three kernels per KDA layer: the speculative short
convolution (Triton), the CuTe recovery verification, and the Triton gated RMSNorm. This
module runs all three in one launch of 4- or 8-CTA clusters, one cluster per (request, value
head), one CTA per 32- or 16-row value tile, exactly the geometry of the verification kernel.

* Convolution: CTA ``t`` owns Q, K and V channels ``bt .. bt+b-1`` (b = block_v) of its head, so every
  channel's history is read and rewritten by one thread, as in the Triton kernel. V outputs
  stay in the CTA's shared memory; Q/K outputs are stored into every cluster CTA's shared
  memory (``st.shared::cluster``) before a cluster barrier. ``mixed_qkv`` is not written.
* Verification: the recovery recurrence of ``_recovery._Verify`` with its arithmetic
  unchanged; records (corrections, raw key/gate) are written as there.
* Output norm: every CTA stores its BF16 recurrence outputs into all cluster CTAs' shared
  memory; after a cluster barrier each CTA reduces the full 128 values of a (token, head)
  in the Triton kernel's order (warp butterfly 16..1 per 32 rows, then
  ``(w0 + w2) + (w1 + w3)``).
  ``div.full``/``rsqrt.approx.ftz``/``ex2.approx`` are those of the Triton PTX, so the
  gated output is bit-identical to the three-launch path.

No global scratch is used: every exchange goes through shared memory initialised in the
launch, so a shared, un-zeroed workspace cannot leak state between layers or calls.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics.memory import set_block_rank
from b12x._lib.intrinsics.reduction import warp_reduce
from b12x._lib.utils.device import current_cuda_stream

from ._cute_kernels import _add, _fake_pointer, _numeric_type, _pointer

_LOG2E_NEG = -1.4426950408889634  # 0fBFB8AA3B, the constant of Triton's exp lowering


def _asm_f32(text, constraint, *args):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value() for a in args],
            text,
            constraint,
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _ex2(x, *, loc=None, ip=None):
    return _asm_f32("ex2.approx.f32 $0, $1;", "=f,f", x)


@dsl_user_op
def _div_full(a, b, *, loc=None, ip=None):
    return _asm_f32("div.full.f32 $0, $1, $2;", "=f,f,f", a, b)


@dsl_user_op
def _rsqrt_ftz(x, *, loc=None, ip=None):
    return _asm_f32("rsqrt.approx.ftz.f32 $0, $1;", "=f,f", x)


@dsl_user_op
def _fma(a, b, c, *, loc=None, ip=None):
    return _asm_f32("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", a, b, c)


@dsl_user_op
def _mul(a, b, *, loc=None, ip=None):
    return _asm_f32("mul.rn.f32 $0, $1, $2;", "=f,f,f", a, b)


@dsl_user_op
def _addrn(a, b, *, loc=None, ip=None):
    return _asm_f32("add.rn.f32 $0, $1, $2;", "=f,f,f", a, b)


@dsl_user_op
def _st_cluster(smem_ptr, rank, value, *, loc=None, ip=None):
    address = set_block_rank(smem_ptr, rank, loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [address, Float32(value).ir_value()],
        "st.shared::cluster.f32 [$0], $1;",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _sigmoid_triton(x):
    """``1 / (1 + exp(-x))`` as Triton lowers it (mul by -log2 e, ex2.approx, div.full)."""
    return _div_full(Float32(1.0), _addrn(_ex2(_mul(x, Float32(_LOG2E_NEG))), Float32(1.0)))


class _FusedVerify:
    def __init__(self, query, conv, block_v, conv_strides, weight_type, norm_type):
        self.heads = query.value_heads
        self.window = query.state_index_columns
        self.block_v = block_v
        self.tiles = 128 // block_v
        self.null = query.null_state_index if query.null_state_index is not None else -1
        self.strides = query.kda_strides
        self.width = self.heads * 128
        self.spec = conv.max_speculative_tokens
        self.history = conv.state_length
        self.capacity_positions = conv.state_capacity
        self.kernel_size = conv.kernel_size
        self.x_stride = conv.input_row_stride
        self.channel_stride, self.position_stride = conv_strides
        self.weight_type = weight_type
        self.norm_type = norm_type
        self.nq = self.spec + 1

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        conv_weight: cute.Pointer,
        conv_slots: cute.Pointer,
        fresh: cute.Pointer,
        accepted: cute.Pointer,
        conv_state: cute.Pointer,
        gate: cute.Pointer,
        beta: cute.Pointer,
        a_log: cute.Pointer,
        bias: cute.Pointer,
        state: cute.Pointer,
        starts: cute.Pointer,
        indices: cute.Pointer,
        live_seqs: cute.Pointer,
        live_tokens: cute.Pointer,
        output: cute.Pointer,
        z: cute.Pointer,
        norm_weight: cute.Pointer,
        corrections: cute.Pointer,
        kg: cute.Pointer,
        conv_slot_stride: Int64,
        index_stride: Int64,
        state_stride: Int64,
        correction_stride: Int64,
        kg_stride: Int64,
        beta_row_stride: Int64,
        beta_head_stride: Int64,
        z_row_stride: Int64,
        z_head_stride: Int64,
        capacity: Int32,
        token_capacity: Int32,
        scale: Float32,
        lower: Float32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            x, conv_weight, conv_slots, fresh, accepted, conv_state, gate, beta, a_log, bias, state,
            starts, indices, live_seqs, live_tokens, output, z, norm_weight, corrections, kg,
            conv_slot_stride, index_stride, state_stride, correction_stride, kg_stride,
            beta_row_stride, beta_head_stride, z_row_stride, z_head_stride,
            capacity, token_capacity, scale, lower, eps,
        ).launch(
            grid=(self.tiles, capacity * self.heads, 1),
            block=(self.block_v * 8, 1, 1),
            cluster=(self.tiles, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Pointer,
        conv_weight: cute.Pointer,
        conv_slots: cute.Pointer,
        fresh: cute.Pointer,
        accepted: cute.Pointer,
        conv_state: cute.Pointer,
        gate: cute.Pointer,
        beta: cute.Pointer,
        a_log: cute.Pointer,
        bias: cute.Pointer,
        state: cute.Pointer,
        starts: cute.Pointer,
        indices: cute.Pointer,
        live_seqs: cute.Pointer,
        live_tokens: cute.Pointer,
        output: cute.Pointer,
        z: cute.Pointer,
        norm_weight: cute.Pointer,
        corrections: cute.Pointer,
        kg: cute.Pointer,
        conv_slot_stride: Int64,
        index_stride: Int64,
        state_stride: Int64,
        correction_stride: Int64,
        kg_stride: Int64,
        beta_row_stride: Int64,
        beta_head_stride: Int64,
        z_row_stride: Int64,
        z_head_stride: Int64,
        capacity: Int32,
        token_capacity: Int32,
        scale: Float32,
        lower: Float32,
        eps: Float32,
    ):
        tile, request_head, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        tile = Int32(tile)
        request = Int32(request_head) // self.heads
        head = Int32(request_head) % self.heads
        tid = Int32(tid)
        lane = tid % 32
        warp = tid // 32
        warps = self.block_v * 8 // 32
        threads = self.block_v * 8
        key_lane = tid % 8
        local_row = tid // 8
        row = tile * self.block_v + local_row

        smem = cutlass.utils.SmemAllocator()
        # V convolution outputs of this CTA's rows: [row][relative token]; Q/K go to every CTA's ``vectors``.
        conv_out = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.block_v * self.nq,)),
            byte_alignment=16,
        )
        vectors = smem.allocate_tensor(
            Float32,
            cute.make_layout((3, self.window, 128), stride=(self.window * 128, 128, 1)),
            byte_alignment=16,
        )
        # Every CTA receives the whole head's BF16 recurrence outputs, [token][row].
        decoded_all = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.window, 128), stride=(128, 1)),
            byte_alignment=16,
        )
        inverse_rms = smem.allocate_tensor(Float32, cute.make_layout((self.window,)), byte_alignment=16)

        # Rows past the live tokens: the norm kernel's zero fill, done once by request 0's clusters.
        bounded_tokens = cutlass.max(Int32(0), cutlass.min(Int32(live_tokens[0]), token_capacity))
        if request == 0:
            token = bounded_tokens + tid // self.block_v
            while token < token_capacity:
                off = (
                    Int64(token) * self.strides[9]
                    + head.to(Int64) * self.strides[10]
                    + (tile * self.block_v + tid % self.block_v).to(Int64)
                )
                output[off] = BFloat16(0.0)
                token += threads // self.block_v

        seqs = Int32(live_seqs[0])
        # Cluster-uniform: every CTA of the cluster shares request and head.
        if request < seqs and request < capacity:
            start = Int32(starts[request])
            end = Int32(starts[request + 1])
            count = end - start
            # Peers' shared memory may be written only once every CTA of the cluster runs.
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

            # ---- 1. speculative short convolution of this CTA's 3 x block_v channels ----
            if tid < 3 * self.block_v:
                part = tid // self.block_v
                column = tile * self.block_v + tid % self.block_v
                channel = part * self.width + head * 128 + column
                slot = Int64(conv_slots[request])
                fresh_value = Int32(fresh[request])
                rollback = cutlass.min(Int32(accepted[request]) - 1, Int32(self.spec))
                state_base = slot * conv_slot_stride + Int64(channel) * self.channel_stride
                history = cute.make_rmem_tensor((self.history,), Float32)
                for position in cutlass.range_constexpr(self.history):
                    history[position] = Float32(0.0)
                    source_position = Int32(position) + rollback
                    if slot >= 0 and fresh_value == 0 and source_position < self.capacity_positions:
                        history[position] = Float32(
                            conv_state[state_base + Int64(source_position) * self.position_stride]
                        )
                query = cute.make_rmem_tensor((self.nq,), Float32)
                for relative in cutlass.range_constexpr(self.nq):
                    query[relative] = Float32(0.0)
                    if relative < count:
                        query[relative] = Float32(
                            x[Int64(start + relative) * self.x_stride + Int64(channel)]
                        )
                weights = cute.make_rmem_tensor((self.kernel_size,), Float32)
                for tap in cutlass.range_constexpr(self.kernel_size):
                    weights[tap] = Float32(conv_weight[Int64(channel) * self.kernel_size + tap])
                for relative in cutlass.range_constexpr(self.nq):
                  if relative < count:
                    accumulator = Float32(0.0)
                    for tap in cutlass.range_constexpr(self.kernel_size):
                        lag = self.kernel_size - 1 - tap
                        if cutlass.const_expr(relative >= lag):
                            accumulator = _fma(query[relative - lag], weights[tap], accumulator)
                        elif cutlass.const_expr(self.history + relative >= lag):
                            accumulator = _fma(history[self.history + relative - lag], weights[tap], accumulator)
                    convolved = Float32(BFloat16(accumulator))
                    activated = Float32(BFloat16(_mul(_sigmoid_triton(convolved), convolved)))
                    if slot < 0:
                        activated = Float32(0.0)
                    if part == 2:
                        conv_out[(tid % self.block_v) * self.nq + relative] = activated
                    else:
                        if cutlass.const_expr(relative < self.window):
                            # Q/K columns are needed by every CTA of the cluster: push them.
                            target = vectors.iterator + ((part * self.window + relative) * 128 + column)
                            for peer in cutlass.range_constexpr(self.tiles):
                                _st_cluster(target, Int32(peer), activated)
                if slot >= 0 and count > 0:
                    for position in cutlass.range_constexpr(self.history):
                        value = Float32(0.0)
                        if cutlass.const_expr(position + 1 < self.history):
                            value = history[position + 1]
                        else:
                            value = query[position + 1 - self.history]
                        conv_state[state_base + Int64(position) * self.position_stride] = BFloat16(value)
                    for candidate in cutlass.range_constexpr(self.spec):
                        conv_state[
                            state_base + Int64(self.history + candidate) * self.position_stride
                        ] = BFloat16(query[candidate + 1])
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            # ---- 2. recovery verification (arithmetic of _recovery._Verify) ----
            source = Int64(indices[request.to(Int64) * index_stride])
            if source == self.null:
                for pos in cutlass.range_constexpr(self.window):
                    if pos < count and key_lane == 0:
                        for peer in cutlass.range_constexpr(self.tiles):
                            _st_cluster(decoded_all.iterator + (pos * 128 + row), Int32(peer), Float32(0.0))
            else:
                base = source * state_stride + head.to(Int64) * self.strides[7] + row.to(Int64) * self.strides[8]
                s = cute.make_rmem_tensor((16,), Float32)
                for j in cutlass.range_constexpr(16):
                    s[j] = Float32(state[base + Int64(key_lane + j * 8)])
                values = cute.make_rmem_tensor((self.window,), Float32)
                betas = cute.make_rmem_tensor((self.window,), Float32)
                for pos in cutlass.range_constexpr(self.window):
                    values[pos] = Float32(0.0)
                    betas[pos] = Float32(0.0)
                    if pos < count:
                        token = Int64(start + pos)
                        values[pos] = Float32(conv_out[local_row * self.nq + pos])
                        betas[pos] = Float32(beta[token * beta_row_stride + head.to(Int64) * beta_head_stride])
                a = cute.math.exp(Float32(a_log[head]), fastmath=False)
                for step in cutlass.range_constexpr((self.window + warps - 1) // warps):
                    pos = warp + step * warps
                    if pos < count:
                        kg_base = source * kg_stride + (head.to(Int64) * self.window + Int64(pos)) * 256
                        qsum = Float32(0.0)
                        ksum = Float32(0.0)
                        for j in cutlass.range_constexpr(4):
                            k = lane + j * 32
                            qv = Float32(vectors[0, pos, k])
                            kv = Float32(vectors[1, pos, k])
                            qsum += qv * qv
                            ksum += kv * kv
                            if tile == 0:
                                kg[kg_base + k.to(Int64)] = BFloat16(kv)
                        qinv = cute.math.rsqrt(warp_reduce(qsum, _add) + 1.0e-6, fastmath=False)
                        kinv = cute.math.rsqrt(warp_reduce(ksum, _add) + 1.0e-6, fastmath=False)
                        for j in cutlass.range_constexpr(4):
                            k = lane + j * 32
                            vectors[0, pos, k] = Float32(vectors[0, pos, k]) * qinv * scale
                            vectors[1, pos, k] = Float32(vectors[1, pos, k]) * kinv
                for step in cutlass.range_constexpr((self.window * 128 + threads - 1) // threads):
                    index = tid + step * threads
                    pos = index // 128
                    column = index % 128
                    if index < self.window * 128 and pos < count:
                        token = Int64(start + pos)
                        raw = Float32(gate[token * self.strides[1] + head.to(Int64) * self.strides[2] + column.to(Int64)])
                        biased = raw + Float32(bias[head.to(Int64) * self.strides[5] + column.to(Int64)])
                        activated = lower / (1.0 + cute.math.exp(-a * biased, fastmath=False))
                        vectors[2, pos, column] = cute.math.exp(activated, fastmath=False)
                        if tile == 0:
                            kg_base = source * kg_stride + (head.to(Int64) * self.window + Int64(pos)) * 256
                            kg[kg_base + 128 + column.to(Int64)] = BFloat16(raw)
                cute.arch.sync_threads()
                for pos in cutlass.range_constexpr(self.window):
                    if pos < count:
                        dot = Float32(0.0)
                        for j in cutlass.range_constexpr(16):
                            k = key_lane + j * 8
                            s[j] = s[j] * Float32(vectors[2, pos, k])
                            dot += s[j] * Float32(vectors[1, pos, k])
                        dot = warp_reduce(dot, _add, 8)
                        delta = (values[pos] - dot) / (1.0 + cute.math.exp(-betas[pos], fastmath=False))
                        decoded = Float32(0.0)
                        for j in cutlass.range_constexpr(16):
                            k = key_lane + j * 8
                            s[j] = s[j] + delta * Float32(vectors[1, pos, k])
                            decoded += s[j] * Float32(vectors[0, pos, k])
                        decoded = warp_reduce(decoded, _add, 8)
                        if key_lane == 0:
                            rounded = Float32(BFloat16(decoded))
                            for peer in cutlass.range_constexpr(self.tiles):
                                _st_cluster(decoded_all.iterator + (pos * 128 + row), Int32(peer), rounded)
                            record = (
                                source * correction_stride
                                + (head.to(Int64) * self.window + Int64(pos)) * 128
                                + row.to(Int64)
                            )
                            corrections[record] = delta
            # Norm operands of (token = key_lane, row), loaded before the exchange.
            norm_pos = key_lane
            gate_input = Float32(0.0)
            norm_value = Float32(norm_weight[row])
            if norm_pos < count:
                gate_input = Float32(
                    z[Int64(start + norm_pos) * z_row_stride + head.to(Int64) * z_head_stride + row.to(Int64)]
                )
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            # ---- 3. gated RMSNorm in the Triton kernel's reduction order: 4 warps x 32 lanes, butterfly
            # 16, 8, 4, 2, 1 inside a warp, then (w0 + w2) + (w1 + w3); one warp per token here.
            for step in cutlass.range_constexpr((self.window + warps - 1) // warps):
                pos = warp + step * warps
                if pos < count:
                    sums = cute.make_rmem_tensor((4,), Float32)
                    for w in cutlass.range_constexpr(4):
                        value = Float32(decoded_all[pos, w * 32 + lane])
                        square = _mul(value, value)
                        for level in cutlass.range_constexpr(5):
                            square = _addrn(square, cute.arch.shuffle_sync_bfly(square, offset=16 >> level))
                        sums[w] = square
                    total = _addrn(_addrn(sums[0], sums[2]), _addrn(sums[1], sums[3]))
                    if lane == 0:
                        inverse_rms[pos] = _rsqrt_ftz(_addrn(eps, _div_full(total, Float32(128.0))))
            cute.arch.sync_threads()
            if norm_pos < count:
                token = Int64(start + norm_pos)
                normalized = _mul(Float32(inverse_rms[norm_pos]), Float32(decoded_all[norm_pos, row]))
                weighted = _mul(normalized, norm_value)
                off = token * self.strides[9] + head.to(Int64) * self.strides[10] + row.to(Int64)
                output[off] = BFloat16(_mul(weighted, _sigmoid_triton(gate_input)))


class FusedVerifyBinding:
    """A prepared one-launch verify: the gdn recovery binding plus the speculative conv binding."""

    def __init__(self, binding, conv, launch, args):
        self.binding, self.conv, self._launch, self._args = binding, conv, launch, args

    def run(self, *, scale, lower_bound, eps):
        self._launch(*self._args, scale, lower_bound, eps, current_cuda_stream())


def prepare(binding, conv):
    """Compile (cached by geometry) and pre-build the launch arguments of the fused verify."""
    from b12x.preparation.types import require_prepared

    state = require_prepared(binding.plan, "attention.gdn")
    query = state.query
    layout = state.layout
    block_v = layout.recurrent_block_v
    cq = require_prepared(conv.plan, "sequence.short_conv").query
    if not query.recover_speculative_state:
        raise ValueError("fused KDA verify requires recover_speculative_state")
    if block_v not in (16, 32) or query.value_heads != query.key_heads:
        raise ValueError("fused KDA verify requires recurrent_block_v 16 or 32 and KDA heads")
    if cq.mode != "decode" or cq.max_speculative_tokens < 1 or cq.kernel_size != 4 or cq.dilation != 1:
        raise ValueError("fused KDA verify requires speculative decode convolution, kernel 4, dilation 1")
    if cq.channels != 3 * query.value_heads * 128 or not cq.round_before_silu or conv.conv_bias is not None:
        raise ValueError("fused KDA verify requires 3 x heads x 128 channels, BF16 rounding, no bias")
    if query.state_index_columns > 8:
        raise ValueError("fused KDA verify requires a verify window of at most 8 tokens")
    if cq.max_speculative_tokens + 1 < query.state_index_columns:
        raise ValueError("fused KDA verify requires a conv window covering the verify window")
    if binding.mixed_qkv.data_ptr() != conv.out.data_ptr():
        raise ValueError("fused KDA verify requires the conv output to be the recurrence input")
    if conv.x.stride(1) != 1 or binding.z.stride(2) != 1 or binding.output.stride(2) != 1:
        raise ValueError("fused KDA verify requires unit inner strides")
    if conv.conv_weight.stride() != (cq.kernel_size, 1) or binding.norm_weight.stride() != (1,):
        raise ValueError("fused KDA verify requires contiguous conv and norm weights")
    capacity = binding.state_indices.shape[0]
    if cq.max_seqs != capacity or binding.num_accepted_tokens.data_ptr() != conv.num_accepted_tokens.data_ptr():
        raise ValueError("fused KDA verify requires one request capacity and shared accepted counts")
    conv_strides = (conv.conv_state.stride(1), conv.conv_state.stride(2))
    weight_type = _numeric_type(conv.conv_weight.dtype)
    norm_type = _numeric_type(binding.norm_weight.dtype)
    a_type = _numeric_type(binding.A_log.dtype)
    bias_type = _numeric_type(binding.dt_bias.dtype)
    index_type = _numeric_type(binding.state_indices.dtype)
    vtypes = (
        BFloat16, weight_type, Int64, Uint8, Int32, BFloat16,
        BFloat16, BFloat16, a_type, bias_type, Float32, Int32, index_type, Int32, Int32,
        BFloat16, BFloat16, norm_type, Float32, BFloat16,
    )
    strides = (*query.kda_strides[:6], 0, *query.kda_strides[7:])
    key = (
        query.value_heads, query.state_index_columns, block_v, query.null_state_index,
        cq.max_speculative_tokens, cq.input_row_stride, conv_strides, str(conv.conv_weight.dtype),
        str(binding.norm_weight.dtype), query.a_log_dtype, query.dt_bias_dtype, query.state_indices_dtype, strides,
    )
    kernel = _FusedVerify(query, cq, block_v, conv_strides, weight_type, norm_type)
    launch = b12x_compile(
        kernel,
        *(_fake_pointer(t) for t in vtypes),
        *(Int64(1) for _ in range(9)),
        Int32(1),
        Int32(1),
        Float32(1.0),
        Float32(-5.0),
        Float32(1e-6),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("sequence.gdn_decode.kda_fused_verify", 1, key),
    )
    tensors = (
        conv.x, conv.conv_weight, conv.state_slot_ids, conv.state_is_fresh.view(torch.uint8),
        conv.num_accepted_tokens, conv.conv_state,
        binding.raw_g, binding.raw_beta, binding.A_log, binding.dt_bias, binding.recurrent_state,
        binding.query_start_loc, binding.state_indices, binding.num_seqs, binding.num_tokens,
        binding.output, binding.z, binding.norm_weight, binding.correction_cache, binding.kg_cache,
    )
    args = (
        *(_pointer(t, ty) for t, ty in zip(tensors, vtypes, strict=True)),
        conv.conv_state.stride(0),
        binding.state_indices.stride(0),
        binding.recurrent_state.stride(0),
        binding.correction_cache.stride(0),
        binding.kg_cache.stride(0),
        binding.raw_beta.stride(0),
        binding.raw_beta.stride(1),
        binding.z.stride(0),
        binding.z.stride(1),
        capacity,
        binding.output.shape[0],
    )
    return FusedVerifyBinding(binding, conv, launch, args)


def prepare_kda_fused_verify(binding, conv):
    """Public: prepare the one-launch verify for a KDA recovery binding and its speculative conv binding.

    Call at preparation/finalize time (it may compile); ``run_kda_fused_verify`` is graph-capturable.
    Replaces ``short_conv.run(conv)`` followed by ``run_kda(binding)``; ``conv.out`` (the
    recurrence input) is not written.
    """
    return prepare(binding, conv)


def run_kda_fused_verify(fused, *, lower_bound=-5.0, eps=1e-6, scale=None):
    import math

    lower_bound, eps = float(lower_bound), float(eps)
    scale = float(128**-0.5 if scale is None else scale)
    if not (math.isfinite(lower_bound) and lower_bound < 0 and math.isfinite(eps) and eps > 0 and scale > 0):
        raise ValueError("fused KDA verify requires a negative lower bound and positive eps/scale")
    fused.run(scale=scale, lower_bound=lower_bound, eps=eps)
    return fused.binding.output
