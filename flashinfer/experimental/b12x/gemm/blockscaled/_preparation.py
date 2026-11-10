"""Packed precision lowering with exact per-invocation native programs."""

from __future__ import annotations

import os

import b12x._lib.dense_gemm.buffers as _module__lib_dense_gemm_buffers
import b12x._lib.dense_gemm.compile as _module__lib_dense_gemm_compile
import b12x._lib.dense_gemm.config as _module__lib_dense_gemm_config
import b12x._lib.dense_gemm.preparation as _module__lib_dense_gemm_preparation


from b12x._lib.quant.block_codec import BLOCK_CODECS, IQ2_CODECS, block_codec

from dataclasses import dataclass, replace
from types import MappingProxyType

import torch
import triton

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import scratch_buffer_spec
from b12x._lib.utils.device import current_cuda_stream, cuda_stream_to_int
from b12x._lib.utils.pointers import make_ptr
from b12x.preparation import FrozenMapping, MemoryRequirements, PersistentMemory, Plan
from b12x.preparation.types import _CompositePlan

# Opt-in (default off): launch the A16 split-K reduce as a programmatic dependent
# (PDL) of its split GEMM, which releases it as soon as every GEMM CTA has started.
# The reduce still waits for GEMM completion; its arithmetic is unchanged.
_A16_SPLITK_REDUCE_PDL = os.environ.get("B12X_A16_SPLITK_REDUCE_PDL", "0") == "1"
from ._tuning import (
    MXFP8_PREFILL_MIN_ROWS as _MXFP8_PREFILL_MIN_ROWS,
    MXFP8_A16_DEQUANT_MIN_ROWS,
    BlockscaledConfig,
    BlockscaledQuery,
    TUNING,
    effective_a16_config,
    functional_mxfp8_quantization,
    mxfp8_a16_dequant,
    quantized_expected_m,
    quantized_query,
)


def _compile_query_payload(query):
    return {
        name: value
        for name, value in TUNING.encode_query(query).items()
        if name != "autotune_effort"
    }


def _decode_query(payload):
    values = dict(payload)
    values["autotune_effort"] = "default"
    values["codegen"] = FrozenMapping(values["codegen"])
    return BlockscaledQuery(**values)


def _lower_dense_with_hint(query, config, device, expected_m):
    from b12x.gemm._preparation import _configured_lowering, _default_lowering
    from b12x.gemm._tuning import TUNING as DENSE_TUNING

    inner = quantized_query(query, expected_m=expected_m)
    if config.dense_config is not None:
        return _configured_lowering(
            inner, DENSE_TUNING.decode_config(config.dense_config), device.identity
        )
    return _default_lowering(inner, device.identity)


def _has_mxfp8_prefill_regime(query):
    return (
        query.recipe == "mxfp8"
        and query.expected_m is None
        and query.num_tokens >= _MXFP8_PREFILL_MIN_ROWS
    )


def _dense_lowering(query, config, device):
    # Both programs are compiled for the declared capacity. The large-row hint
    # must not force underfilled short prefills onto a large M tile.
    expected_m = quantized_expected_m(query)
    return _lower_dense_with_hint(query, config, device, expected_m)


def _short_dense_lowering(query, config, device):
    if not _has_mxfp8_prefill_regime(query):
        return None
    return _lower_dense_with_hint(
        query, replace(config, dense_config=None), device, None
    )


def _pad_activation_scale_rows(query, lowering):
    # Compact staging reads scale rows 0-15; inactive output rows are discarded.
    return not (
        query.recipe == "nvfp4"
        and lowering is not None
        and lowering.m <= 16
        and lowering.mma_tiler_mn in ((16, 32), (16, 64))
        and lowering.tile_k == 1024
        and lowering.load_path == "tma"
        and not lowering.swap_ab
        and not lowering.fused_quant
        and not lowering.direct_m1_wo_a_inputs
        and not lowering.programmatic_serialization
        and not lowering.quantize_c
    )


@program_cache(scope="preparation")
def compile_packed(
    query_payload,
    config_payload,
    dense_payload,
    short_dense_payload,
    ordinal,
    sm_count,
    capability,
):
    import cutlass
    from . import _quantize, _reduce

    query = _decode_query(query_payload)
    config = BlockscaledConfig.from_config(FrozenMapping(config_payload))
    fp4 = query.recipe in ("nvfp4", "mxfp4")
    group = 16 if query.recipe == "nvfp4" else 32
    with torch.cuda.device(ordinal):
        activation_program = {}
        if query.source_activation == "swiglu_bf16" and not config.fuse_swiglu:
            from b12x.norm.hyperconnection import _cute as activation

            activation_program["swiglu"] = activation._compile(
                ("blockscaled_swiglu_bf16", query.in_features),
                activation._Pointwise(
                    "swiglu", cutlass.BFloat16, query.in_features, float("inf"), True
                ),
                3,
                torch.device("cuda", ordinal),
                runtime_int64s=1,
            )
        if config.mode == "a16":
            if config.tile_n == 4:
                from ._iq2_xs_gemv import compile_gemv

                return {
                    "gemm": compile_gemv(
                        ordinal,
                        query.out_features,
                        query.in_features,
                        config.tile_m,
                        query.recipe,
                    )
                }
            bn, bk, slices = effective_a16_config(query, config)
            gemm = _module__lib_dense_gemm_compile._get_compiled_dense_gemm(
                query.out_features,
                query.padded_in_features,
                1,
                slices,
                "k",
                "k",
                "n",
                cutlass.BFloat16,
                cutlass.Uint8,
                cutlass.Float32 if slices > 1 else cutlass.BFloat16,
                cutlass.Float32,
                16 if fp4 else 32,
                16,
                bk,
                (config.tile_m or 16, bn),
                (1, 1),
                _module__lib_dense_gemm_config._DenseGemmPolicy(
                    True,
                    True,
                    False,
                    slices,
                    False,
                    False,
                    split_k_pdl_reduce=slices > 1 and _A16_SPLITK_REDUCE_PDL,
                ),
                sm_count,
                f"sm_{capability[0]}{capability[1]}a",
                "tma",
                query.recipe in BLOCK_CODECS and config.tile_m == 8,
                False,
                False,
                alpha_is_one=not fp4,
                target_occupancy_override=1,
                weight_only=query.recipe,
                mma_warps=config.mma_warps,
                alpha_reciprocal=query.global_scale_kind == "reciprocal",
                input_k=query.in_features,
                device_ordinal=ordinal,
            )
            programs = {"gemm": gemm, **activation_program}
            if slices > 1:
                programs["reduce"] = _reduce.compile_reduce(
                    query.out_features, slices, ordinal, pdl=_A16_SPLITK_REDUCE_PDL
                )
            if mxfp8_a16_dequant(query, config):
                from ._dequant import DEQUANT_BLOCK_K, _dequantize_mxfp8

                k = query.padded_in_features
                programs["dequant"] = _dequantize_mxfp8.warmup(
                    torch.float8_e4m3fn,
                    torch.uint8,
                    torch.bfloat16,
                    K=k,
                    BLOCK_K=DEQUANT_BLOCK_K,
                    grid=(query.out_features, triton.cdiv(k, DEQUANT_BLOCK_K), 1),
                    num_warps=4,
                    num_stages=1,
                    enable_fp_fusion=False,
                )
            return programs
        programs = dict(
            _module__lib_dense_gemm_preparation._compile_dense_lowering(
                dense_payload, ordinal
            )
        )
        if short_dense_payload is not None:
            programs.update(
                {
                    "short_" + name: program
                    for name, program in _module__lib_dense_gemm_preparation._compile_dense_lowering(
                        short_dense_payload, ordinal
                    ).items()
                }
            )
        programs.update(activation_program)
        if functional_mxfp8_quantization(query, config):
            from b12x._lib.quant import mxfp8_rows

            rows = query.num_tokens if query.expected_m is None else query.expected_m
            subgroup = 8 if rows <= 8 else mxfp8_rows._WARP_SUBGROUP_WIDTH
            threads = 128 if rows <= 8 else mxfp8_rows._THREADS
            programs["quantize"] = mxfp8_rows._get_compiled_mxfp8_rows_quant(
                query.padded_in_features,
                torch.bfloat16,
                subgroup,
                threads,
                "linear",
                device_ordinal=ordinal,
                sm_count=sm_count,
            )
        else:
            programs["quantize"] = _quantize._quantize.warmup(
                torch.bfloat16,
                torch.uint8 if fp4 else torch.float8_e4m3fn,
                torch.uint8,
                torch.float32,
                torch.float32,
                torch.float32,
                query.num_tokens,
                INPUT_K=query.in_features,
                K=query.padded_in_features,
                FP4=fp4,
                RECIPROCAL=query.global_scale_kind == "reciprocal",
                GROUP=group,
                CHUNKS=16,
                SWIGLU=config.fuse_swiglu,
                PAD_SCALE_ROWS=_pad_activation_scale_rows(
                    query,
                    _module__lib_dense_gemm_preparation._DenseLowering.from_dict(
                        dense_payload
                    ),
                ),
                grid=(
                    query.num_tokens,
                    triton.cdiv(query.padded_in_features // group, 16),
                    1,
                ),
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
            )
        return programs


def _mxfp8_bases_nbytes(query):
    from b12x.gemm._shared.wo_mxfp8 import (
        MXFP8_SCALE_K_TILE,
        MXFP8_SCALE_ROW_TILE,
        MXFP8_SCALE_VEC_SIZE,
    )

    rows, k = query.num_tokens, query.padded_in_features
    scale_columns = k // MXFP8_SCALE_VEC_SIZE
    tiles = -(-rows // MXFP8_SCALE_ROW_TILE) * -(-scale_columns // MXFP8_SCALE_K_TILE)
    return rows * k + rows * scale_columns + tiles * 32 * 4 * 4


def _owned_bytes(query, config, device=None):
    if query.workspace_form != "owned":
        return 0
    owned = _workspace_bytes(query, config, device)
    if functional_mxfp8_quantization(query, config):
        owned += _mxfp8_bases_nbytes(query)
    return owned


def _activation_bytes(query, config):
    if query.source_activation == "swiglu_bf16" and not config.fuse_swiglu:
        return triton.cdiv(query.num_tokens * query.in_features * 2, 256) * 256
    return 0


def _workspace_bytes(query, config, device=None):
    return _activation_bytes(query, config) + _projection_workspace_bytes(
        query, config, device
    )


def _projection_workspace_bytes(query, config, device):
    if config.mode == "a16":
        slices = effective_a16_config(query, config)[2]
        partials = (
            slices * query.num_tokens * query.out_features * 4 if slices > 1 else 0
        )
        if mxfp8_a16_dequant(query, config):
            if query.reference_rows is not None:
                from b12x.gemm._shared.reference_matmul import MAX_WORKSPACE_BYTES

                weight_bytes = query.out_features * query.padded_in_features * 2
                return triton.cdiv(weight_bytes, 256) * 256 + MAX_WORKSPACE_BYTES
            if query.output_partition is not None:
                logical = query.output_partition[0]
                weight_bytes = logical * query.padded_in_features * 2
                output_offset = triton.cdiv(weight_bytes, 256) * 256
                return output_offset + query.num_tokens * logical * 2
            return max(partials, query.out_features * query.padded_in_features * 2)
        return partials
    if functional_mxfp8_quantization(query, config):
        if not query.codegen["split_k_atomic"] or not query.allow_bf16_atomic_reduction:
            slices = _quantized_workspace_slices(query, config, device)
            if slices > 1:
                return slices * query.num_tokens * query.out_features * 4
        return 0
    return _quantized_layout(query, config, device)[-1]


def _quantized_workspace_slices(query, config, device):
    if config.dense_config is not None:
        return config.dense_config["split_k_slices"] or 1
    if query.codegen["split_k_atomic"] and query.allow_bf16_atomic_reduction:
        return 1
    from b12x.gemm._preparation import _default_lowering

    inner = quantized_query(query, expected_m=quantized_expected_m(query))
    return _default_lowering(inner, device).policy.split_k_slices


def _quantized_layout(query, config, device):
    from ._a16 import _layout

    return _layout(
        query.num_tokens,
        query.out_features,
        query.padded_in_features,
        query.recipe in ("nvfp4", "mxfp4"),
        (64, 64, _quantized_workspace_slices(query, config, device)),
        group=16 if query.recipe == "nvfp4" else 32,
    )


@dataclass(frozen=True)
class _PackedExecutionState:
    query: BlockscaledQuery
    config: BlockscaledConfig
    device: torch.device
    programs: dict
    dense: object | None
    alpha_one: torch.Tensor | None
    offsets: tuple | None
    required_workspace: int
    workspace: torch.Tensor | None = None
    mxfp8_bases: tuple | None = None
    short_dense: object | None = None
    iq2_lut: torch.Tensor | None = None
    reference_matmul: object | None = None

    @property
    def owned_nbytes(self) -> int:
        owned = 0 if self.workspace is None else self.workspace.numel()
        if self.mxfp8_bases is not None:
            owned += sum(
                base.numel() * base.element_size() for base in self.mxfp8_bases
            )
        return owned

    def _mxfp8_rows(self, m):
        from b12x.gemm._shared.wo_mxfp8 import (
            MXFP8_SCALE_ROW_TILE,
            mxfp8_rows_from_bases,
        )

        values_base, scale_rows_base, scale_physical_base = self.mxfp8_bases
        tiles = -(-m // MXFP8_SCALE_ROW_TILE)
        return mxfp8_rows_from_bases(
            values_base[:m],
            scale_rows_base[:, :m],
            scale_physical_base[:, :tiles],
            m,
            self.query.padded_in_features,
            num_groups=1,
        )

    def run(
        self,
        source,
        values,
        scales,
        global_scale,
        *,
        activation_scale=None,
        out=None,
        workspace=None,
        stream=None,
    ):
        from ._a16 import (
            _check_tensor,
            _overlap,
            _stream_context,
            _validate_output,
            scale_storage,
        )
        from ._linear import _source_2d, _pad_k

        q, config = self.query, self.config
        nvfp4 = q.recipe == "nvfp4"
        fp4 = q.recipe in ("nvfp4", "mxfp4")
        group = 16 if nvfp4 else 32
        iq2 = q.recipe in BLOCK_CODECS
        if source.device != self.device or source.dtype != torch.bfloat16:
            raise ValueError("packed source differs from prepared dtype/device")
        source_width = q.in_features * (
            2 if q.source_activation == "swiglu_bf16" else 1
        )
        if source.ndim < 2 or source.shape[-1] != source_width:
            raise ValueError("packed source logical K differs from preparation")
        m = source.numel() // source_width
        if (q.output_partition is not None or q.reference_rows is not None) and m != q.num_tokens:
            raise ValueError("output partition execution requires its exact prepared M")
        if m and (m > q.num_tokens or (q.expected_m is not None and m != q.num_tokens)):
            raise ValueError(
                "packed execution exceeds its capacity or differs from its exact M"
            )
        if source.is_contiguous() != q.source_contiguous or (
            bool(source.data_ptr() % 16 == 0) != q.source_aligned and m
        ):
            raise ValueError("packed source layout differs from preparation")
        if (out is None) != (q.output_mode == "functional"):
            raise ValueError("packed output form differs from preparation")
        if q.workspace_form == "provided":
            if workspace is None and self.required_workspace:
                raise ValueError(
                    "packed execution requires its caller-provided workspace"
                )
        elif workspace is not None:
            raise ValueError(
                "owned-workspace execution does not accept a caller workspace"
            )
        else:
            workspace = self.workspace
        if (activation_scale is not None) != q.activation_scale_available:
            raise ValueError("activation-scale presence differs from preparation")
        stored_k = q.padded_in_features // (
            block_codec(q.recipe).pack_factor if iq2 else 2 if fp4 else 1
        )
        _check_tensor(
            "weight",
            values,
            self.device,
            torch.uint8 if fp4 or iq2 else torch.float8_e4m3fn,
        )
        if values.shape != (q.out_features, stored_k):
            raise ValueError("packed weight geometry differs from preparation")
        if iq2:
            if tuple(scales.shape) != (
                ((q.out_features + 127) // 128, q.in_features // 256, 1280)
                if q.recipe == "iq2_xs"
                else (
                    q.in_features // block_codec(q.recipe).block_weights,
                    q.out_features,
                    2,
                )
            ):
                raise ValueError("IQ2_XS metadata geometry differs from preparation")
            scale_bytes = scales
        else:
            scale_bytes = scale_storage(
                scales, q.out_features, q.padded_in_features, group
            )
        _check_tensor("weight scale", scale_bytes, self.device, torch.uint8)
        if nvfp4:
            _check_tensor(
                "weight global scale", global_scale, self.device, torch.float32
            )
            if global_scale.numel() != 1:
                raise ValueError("weight global scale must be scalar")
        elif global_scale is not None:
            raise ValueError("this weight recipe has no global scale")
        if m == 0:
            return _validate_output(source, out, q.out_features)
        core = (
            self.short_dense
            if self.short_dense is not None and m < _MXFP8_PREFILL_MIN_ROWS
            else self.dense
        )
        with torch.cuda.device(self.device), _stream_context(stream, self.device):
            if functional_mxfp8_quantization(q, config):
                contiguous = _pad_k(_source_2d(source), q.padded_in_features)
                _check_tensor(
                    "quantizer source", contiguous, self.device, torch.bfloat16
                )
                packed = self._mxfp8_rows(m)
                self.programs["quantize"](
                    contiguous, packed.values, packed.scale_rows, packed.scale_mma
                )
                result = core.run(
                    (packed.values.view(m, stored_k, 1), packed.scale_mma),
                    (
                        values.view(q.out_features, stored_k, 1),
                        scales.view(torch.float8_e8m0fnu),
                    ),
                    stream=stream,
                    split_k_workspace=None
                    if workspace is None
                    else workspace.view(torch.float32),
                )[:, :, 0]
                return result.view(*source.shape[:-1], q.out_features)
            _check_tensor("source", source, self.device, torch.bfloat16)
            out = _validate_output(source, out, q.out_features)
            reads = (source, values, scale_bytes) + ((global_scale,) if nvfp4 else ())
            if q.recipe in IQ2_CODECS:
                reads += (self.iq2_lut,)
            if any(_overlap(out, tensor) for tensor in reads):
                raise ValueError("output must not overlap packed inputs")
            if workspace is not None:
                _check_tensor("workspace", workspace, self.device, torch.uint8)
                if (q.output_partition is not None or q.reference_rows is not None) and workspace.ndim != 1:
                    raise ValueError(
                        "output partition workspace must be one-dimensional"
                    )
                if workspace.numel() < self.required_workspace:
                    raise ValueError(
                        "workspace is smaller than the prepared requirement"
                    )
                if any(_overlap(workspace, tensor) for tensor in (*reads, out)):
                    raise ValueError(
                        "workspace must not overlap packed operands/output"
                    )
            if nvfp4 and config.mode == "quantized":
                _check_tensor(
                    "activation scale", activation_scale, self.device, torch.float32
                )
                if activation_scale.numel() != 1:
                    raise ValueError("activation global scale must be scalar")
                if _overlap(out, activation_scale) or _overlap(
                    workspace, activation_scale
                ):
                    raise ValueError(
                        "activation scale must not overlap output/workspace"
                    )
            activation_bytes = _activation_bytes(q, config)
            if activation_bytes:
                from b12x.norm.hyperconnection._cute import _pointer

                activated = (
                    workspace[: m * q.in_features * 2]
                    .view(torch.bfloat16)
                    .view(m, q.in_features)
                )
                self.programs["swiglu"](
                    _pointer(source),
                    _pointer(source),
                    _pointer(activated),
                    m * q.in_features,
                    current_cuda_stream(),
                )
                source = activated
                workspace = workspace[activation_bytes:]
            if (
                config.mode == "a16"
                and "dequant" in self.programs
                and m >= MXFP8_A16_DEQUANT_MIN_ROWS
            ):
                from ._dequant import DEQUANT_BLOCK_K

                k = q.padded_in_features
                logical, first, end = (
                    (q.out_features, 0, q.out_features)
                    if q.output_partition is None
                    else q.output_partition
                )
                weight_bytes = logical * k * 2
                weight = workspace[:weight_bytes].view(torch.bfloat16).view(logical, k)
                destination = out.view(m, q.out_features)
                if q.output_partition is not None:
                    # Scratch is shared across serial operators. Padding must be
                    # overwritten on every invocation, not only during preparation.
                    weight.zero_()
                    output_offset = triton.cdiv(weight_bytes, 256) * 256
                    destination = (
                        workspace[output_offset : output_offset + m * logical * 2]
                        .view(torch.bfloat16)
                        .view(m, logical)
                    )
                self.programs["dequant"][
                    (q.out_features, triton.cdiv(k, DEQUANT_BLOCK_K), 1)
                ](values, scale_bytes, weight[first:end], k, DEQUANT_BLOCK_K)
                if self.reference_matmul is None:
                    torch.mm(source.view(m, q.in_features), weight.t(), out=destination)
                else:
                    scratch_offset = triton.cdiv(weight_bytes, 256) * 256
                    self.reference_matmul(source.view(m, q.in_features), weight, destination,
                                          workspace[scratch_offset:])
                if q.output_partition is not None:
                    out.view(m, q.out_features).copy_(destination[:, first:end])
                return out
            if config.mode == "a16":
                import cutlass
                import cutlass.cute as cute

                _, _, slices = effective_a16_config(q, config)
                target = (
                    out
                    if slices == 1
                    else workspace[: self.required_workspace].view(torch.float32)
                )
                self.programs["gemm"](
                    source.view(m, q.in_features),
                    values,
                    self.iq2_lut if q.recipe in IQ2_CODECS else scale_bytes,
                    scale_bytes,
                    target,
                    global_scale if fp4 else self.alpha_one,
                    cuda_stream_to_int(stream),
                )
                if slices > 1:
                    self.programs["reduce"](
                        make_ptr(
                            cutlass.Float32,
                            target.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        make_ptr(
                            cutlass.BFloat16,
                            out.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        m,
                        current_cuda_stream(),
                    )
                return out
            sf_start, alpha_start, partial_start, needed = self.offsets
            packed_values = workspace[: m * stored_k].view(
                torch.uint8 if fp4 else torch.float8_e4m3fn
            )
            scale_size = (
                triton.cdiv(m, 128)
                * triton.cdiv(q.padded_in_features // group, 4)
                * 512
            )
            packed_scales = workspace[sf_start : sf_start + scale_size]
            alpha = workspace[alpha_start : alpha_start + 4].view(torch.float32)
            partials = workspace[partial_start:needed].view(torch.float32)
            constants = (
                q.in_features,
                q.padded_in_features,
                fp4,
                q.global_scale_kind == "reciprocal",
                group,
                16,
                config.fuse_swiglu,
                _pad_activation_scale_rows(q, core.lowering),
            )
            grid = (m, triton.cdiv(q.padded_in_features // group, 16), 1)
            self.programs["quantize"][grid](
                source,
                packed_values,
                packed_scales,
                activation_scale if nvfp4 else alpha,
                global_scale if nvfp4 else alpha,
                alpha,
                m,
                *constants,
            )
            from b12x._lib.intrinsics.reference import (
                as_grouped_scale_view,
                as_grouped_scale_view_mx,
            )

            scale_view = (as_grouped_scale_view if nvfp4 else as_grouped_scale_view_mx)(
                packed_scales.view(1, -1), m, q.padded_in_features
            )
            core.run(
                (packed_values.view(m, stored_k, 1), scale_view),
                (values.view(q.out_features, stored_k, 1), scales),
                out=out.view(m, q.out_features, 1),
                alpha=alpha,
                stream=stream,
                split_k_workspace=partials,
            )
            return out


def _plan_bf16(query: BlockscaledQuery, *, invocation=FrozenMapping(), override=None):
    if not isinstance(query, BlockscaledQuery):
        raise TypeError("packed BF16 plan requires BlockscaledQuery")
    invocation = FrozenMapping(invocation)
    if invocation:
        raise ValueError("packed invocation semantics belong in BlockscaledQuery")
    lowerings = {}

    def inner(config, device):
        key = (config, device.identity)
        if key not in lowerings:
            lowerings[key] = (
                (None, None)
                if config.mode == "a16"
                else (
                    _dense_lowering(query, config, device),
                    _short_dense_lowering(query, config, device),
                )
            )
        return lowerings[key]

    def compile_jobs(config, device):
        p, short = inner(config, device)
        return (
            CompileJob.create(
                "b12x.gemm.blockscaled._preparation:compile_packed",
                _compile_query_payload(query),
                config.to_dict(),
                None if p is None else p.to_dict(),
                None if short is None else short.to_dict(),
                CompileJob.DEVICE_ORDINAL,
                device.identity.sm_count,
                device.identity.compute_capability,
            ),
        )

    def memory(config, device):
        from b12x.preparation import current_plan, current_prepared_state

        needed = _workspace_bytes(query, config, device.identity)
        scratch = (
            (
                scratch_buffer_spec(
                    "blockscaled.scratch",
                    nbytes=needed,
                    device=torch.device("cuda", device.ordinal),
                ),
            )
            if needed and query.workspace_form == "provided"
            else ()
        )
        persistent = []
        owned = _owned_bytes(query, config, device.identity)
        if owned:
            existing = current_prepared_state()
            resident = 0 if existing is None else min(owned, existing.owned_nbytes)
            persistent.append(
                PersistentMemory(("blockscaled.owned", current_plan()), owned, resident)
            )
        if query.recipe in ("mxfp8", "iq2_xs", "iq2_xxs", "q8_0") and (
            config.mode == "a16" or functional_mxfp8_quantization(query, config)
        ):
            resident = _module__lib_dense_gemm_buffers._ALPHA_ONE_CACHE.get(
                ("cuda", device.ordinal)
            )
            persistent.append(
                PersistentMemory(
                    ("dense.alpha_one", device.ordinal),
                    4,
                    0
                    if resident is None
                    else resident.numel() * resident.element_size(),
                )
            )
        if query.recipe in IQ2_CODECS:
            from b12x._lib.quant.iq2_xs import _DEVICE_TABLES

            resident = _DEVICE_TABLES.get(("cuda", device.ordinal, False, query.recipe))
            persistent.append(
                PersistentMemory(
                    ("iq2.execution_lut", device.ordinal, query.recipe),
                    block_codec(query.recipe).lut_bytes(),
                    0
                    if resident is None
                    else resident.numel() * resident.element_size(),
                )
            )
        return MemoryRequirements(scratch, tuple(persistent))

    def materialize(selection, device):
        config = selection.config
        p, short = inner(config, device)
        programs = compile_packed(
            _compile_query_payload(query),
            config.to_dict(),
            None if p is None else p.to_dict(),
            None if short is None else short.to_dict(),
            device.ordinal,
            device.identity.sm_count,
            device.identity.compute_capability,
        )
        resolved_device = torch.device("cuda", device.ordinal)
        core = (
            None
            if p is None
            else _module__lib_dense_gemm_preparation._DenseExecutionState(
                p,
                resolved_device,
                programs["gemm"],
                programs.get("reduce"),
                _module__lib_dense_gemm_buffers._cached_alpha_one(resolved_device)
                if p.alpha_is_one
                else None,
            )
        )
        short_core = (
            None
            if short is None
            else _module__lib_dense_gemm_preparation._DenseExecutionState(
                short,
                resolved_device,
                programs["short_gemm"],
                programs.get("short_reduce"),
                _module__lib_dense_gemm_buffers._cached_alpha_one(resolved_device)
                if short.alpha_is_one
                else None,
            )
        )
        offsets = (
            None
            if config.mode == "a16" or functional_mxfp8_quantization(query, config)
            else _quantized_layout(query, config, device.identity)
        )
        needed = _workspace_bytes(query, config, device.identity)
        workspace = bases = None
        if query.workspace_form == "owned":
            from b12x.preparation import current_prepared_state

            existing = current_prepared_state()
            if needed:
                previous = None if existing is None else existing.workspace
                if (
                    previous is not None
                    and previous.numel() >= needed
                    and previous.device == resolved_device
                ):
                    workspace = previous
                else:
                    workspace = torch.empty(
                        needed, dtype=torch.uint8, device=resolved_device
                    )
            if functional_mxfp8_quantization(query, config):
                from b12x.gemm._shared.wo_mxfp8 import empty_mxfp8_rows_bases

                previous = None if existing is None else existing.mxfp8_bases
                if (
                    previous is not None
                    and previous[0].shape
                    == (query.num_tokens, query.padded_in_features)
                    and previous[0].device == resolved_device
                ):
                    bases = previous
                else:
                    bases = empty_mxfp8_rows_bases(
                        query.num_tokens,
                        query.padded_in_features,
                        num_groups=1,
                        device=resolved_device,
                        initialize_scales=False,
                    )
        lut = None
        if query.recipe in IQ2_CODECS:
            from b12x._lib.quant.iq2_xs import iq2_xs_execution_lut

            lut = iq2_xs_execution_lut(
                resolved_device, prepare=True, codec=query.recipe
            )
        reference_matmul = None
        if query.reference_rows is not None:
            from b12x.gemm._shared.reference_matmul import ReferenceMatmul

            reference_matmul = ReferenceMatmul(rows=query.num_tokens, columns=query.out_features,
                inner=query.in_features, logical_rows=query.reference_rows,
                logical_columns=query.out_features, device=resolved_device)
        return _PackedExecutionState(
            query,
            config,
            resolved_device,
            programs,
            core,
            _module__lib_dense_gemm_buffers._cached_alpha_one(resolved_device)
            if query.recipe in ("mxfp8", "iq2_xs", "iq2_xxs", "q8_0")
            and config.mode == "a16"
            else None,
            offsets,
            needed,
            workspace,
            bases,
            short_core,
            lut,
            reference_matmul,
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        shared=True,
        _compile_jobs=compile_jobs,
        _memory_requirements=memory,
        _materialize=materialize,
    )


def _query_bf16_from_call(
    source,
    weight,
    *,
    allow_bf16_atomic_reduction=True,
    source_activation="identity",
    activation_mode="auto",
    activation_global_scale=None,
    out=None,
    workspace=None,
    expected_m=None,
    output_partition=None,
    reference_rows=None,
):
    from ._a16 import MXFP4LinearWeight, _weight_parts
    from ._iq2_xs import IQ2XSLinearWeight

    values, _, _, fp4 = _weight_parts(weight)
    if activation_global_scale is None:
        activation_global_scale = getattr(weight, "activation_global_scale", None)
    if source.ndim < 2 or source.dtype != torch.bfloat16:
        raise ValueError(
            "packed BF16 queries require a BF16 source with at least two dimensions"
        )
    source_width = weight.in_features * (2 if source_activation == "swiglu_bf16" else 1)
    if source.shape[-1] != source_width or values.shape[0] != weight.out_features:
        raise ValueError("packed source/weight logical geometry disagrees")
    if source.device != values.device:
        raise ValueError("packed source and weight devices differ")
    if workspace is not None and (
        workspace.dtype != torch.uint8
        or workspace.device != source.device
        or not workspace.is_contiguous()
    ):
        raise ValueError(
            "packed workspace must be contiguous uint8 storage on the source device"
        )
    return BlockscaledQuery(
        recipe=weight.codec
        if isinstance(weight, IQ2XSLinearWeight)
        else "mxfp4"
        if isinstance(weight, MXFP4LinearWeight)
        else "nvfp4"
        if fp4
        else "mxfp8",
        num_tokens=source.numel() // source_width,
        in_features=weight.in_features,
        padded_in_features=weight.padded_in_features,
        out_features=weight.out_features,
        activation_mode=activation_mode,
        source_activation=source_activation,
        allow_bf16_atomic_reduction=allow_bf16_atomic_reduction,
        activation_scale_available=activation_global_scale is not None,
        global_scale_kind=weight.global_scale_kind if fp4 else "none",
        source_contiguous=source.is_contiguous(),
        source_aligned=source.data_ptr() % 16 == 0,
        output_mode="functional" if out is None else "provided",
        workspace_form="owned" if workspace is None else "provided",
        workspace_nbytes=None if workspace is None else workspace.numel(),
        expected_m=expected_m,
        output_partition=output_partition,
        reference_rows=reference_rows,
    )


def _fixed_lowering(query, device):
    from b12x.gemm._preparation import _default_lowering
    from b12x.gemm._tuning import DenseGemmQuery
    from ._linear import _use_block_fp8_recipe

    recipe = query.recipe
    use_block = False
    if recipe == "tensor_fp8":
        use_block = _use_block_fp8_recipe(
            live_m=query.max_rows,
            expected_m=query.expected_m or query.max_rows,
            out_features=query.out_features,
            padded_in_features=query.padded_in_features,
            sm_count=device.identity.sm_count,
        )
        recipe = "block_fp8" if use_block else "tensor_fp8"
    expected_m = query.expected_m
    if query.recipe == "block_fp8" and expected_m is not None:
        # Dense scheduling uses this bound to select single-row load/store paths.
        expected_m = max(expected_m, query.max_rows)
    inner = DenseGemmQuery(
        recipe=recipe,
        entry_point="gemm.blockscaled.mm",
        weight_storage="native",
        output_dtype=query.output_dtype,
        batch=1,
        max_rows=query.max_rows,
        in_features=query.padded_in_features,
        out_features=query.out_features,
        output_mode="functional",
        alpha_mode=query.alpha_mode,
        expected_m=expected_m,
    )
    return _default_lowering(inner, device.identity), use_block


@program_cache(scope="preparation")
def compile_fixed(query_payload, lowering_payload, ordinal, sm_count):
    from ._tuning import FixedBlockscaledQuery

    query = FixedBlockscaledQuery(**dict(query_payload))
    with torch.cuda.device(ordinal):
        programs = dict(
            _module__lib_dense_gemm_preparation._compile_dense_lowering(
                lowering_payload, ordinal
            )
        )
        if (
            query.call_kind == "packed"
            and query.recipe == "mxfp8"
            and query.input_dtype == "float16"
        ):
            from b12x._lib.quant import mxfp8_rows

            rows = query.expected_m or query.max_rows
            subgroup = 8 if rows <= 8 else mxfp8_rows._WARP_SUBGROUP_WIDTH
            threads = 128 if rows <= 8 else mxfp8_rows._THREADS
            programs["quantize"] = mxfp8_rows._get_compiled_mxfp8_rows_quant(
                query.padded_in_features,
                torch.float16,
                subgroup,
                threads,
                "linear",
                device_ordinal=ordinal,
                sm_count=sm_count,
            )
        return programs


@dataclass(frozen=True)
class _FixedExecutionState:
    query: object
    device: torch.device
    dense: object
    quantizer: object | None
    unit_scale: torch.Tensor | None
    use_block: bool
    gemv: object | None = None

    def _check(self, source, weight, output_dtype, *, serialized=False):
        q = self.query
        if (
            source.ndim != 2
            or source.device != self.device
            or source.dtype != getattr(torch, q.input_dtype)
        ):
            raise ValueError(
                "fixed source layout/dtype/device differs from preparation"
            )
        expected_k = (
            q.in_features // 2
            if serialized and q.recipe in ("nvfp4", "mxfp4")
            else q.in_features
        )
        if source.shape[1] != expected_k or source.shape[0] > q.max_rows:
            raise ValueError(
                "fixed execution exceeds its planned M capacity or logical K"
            )
        if q.expected_m is not None and source.shape[0] not in (0, q.max_rows):
            raise ValueError(
                "fixed execution requires its exact planned nonempty M and logical K"
            )
        weight_k = (
            q.padded_in_features // 2
            if serialized and q.recipe in ("nvfp4", "mxfp4")
            else q.padded_in_features
        )
        if weight.device != self.device or weight.shape != (q.out_features, weight_k):
            raise ValueError("fixed weight geometry/device differs from preparation")
        if output_dtype != getattr(torch, q.output_dtype):
            raise ValueError("fixed output dtype differs from preparation")

    def run_serialized(
        self,
        lhs_values,
        lhs_scale_storage,
        rhs_values,
        rhs_scale_storage,
        alpha,
        *,
        ab_dtype,
        sf_dtype,
        c_dtype,
        sf_vec_size,
        block_fp8,
        stream,
    ):
        from ._a16 import _stream_context, scale_storage
        from b12x._lib.intrinsics.reference import (
            as_grouped_scale_view,
            as_grouped_scale_view_mx,
        )
        from b12x.gemm._tuning import DenseGemmQuery, operand_options

        q = self.query
        if q.call_kind != "serialized":
            raise ValueError("execution does not prepare serialized operands")
        self._check(lhs_values, rhs_values, getattr(torch, c_dtype), serialized=True)
        expected = {
            "nvfp4": ("float4_e2m1fn", "float8_e4m3fn", 16, False),
            "mxfp4": ("float4_e2m1fn", "float8_e8m0fnu", 32, False),
            "block_fp8": ("float8_e4m3fn", "float32", 128, True),
        }[q.recipe]
        if (ab_dtype, sf_dtype, sf_vec_size, bool(block_fp8)) != expected:
            raise ValueError("serialized operand recipe differs from preparation")
        m = lhs_values.shape[0]
        if m == 0:
            return lhs_values.new_empty(
                (0, q.out_features), dtype=getattr(torch, c_dtype)
            )
        if self.gemv is not None:
            if alpha is not None:
                raise ValueError("block-FP8 GEMV requires unit alpha")
            for name, tensor, dtype, alignment in (
                ("lhs", lhs_values, torch.float8_e4m3fn, 16),
                ("rhs", rhs_values, torch.float8_e4m3fn, 16),
                ("lhs_scale", lhs_scale_storage, torch.float32, 4),
                ("rhs_scale", rhs_scale_storage, torch.float32, 4),
            ):
                if tensor.device != self.device or tensor.dtype != dtype:
                    raise ValueError(
                        f"block-FP8 GEMV {name} differs from prepared dtype/device"
                    )
                if not tensor.is_contiguous():
                    raise ValueError(f"block-FP8 GEMV {name} must be contiguous")
                if tensor.data_ptr() % alignment:
                    raise ValueError(
                        f"block-FP8 GEMV {name} must be {alignment}-byte aligned"
                    )
        with torch.cuda.device(self.device), _stream_context(stream, self.device):
            if q.recipe == "block_fp8":
                sfa, sfb = lhs_scale_storage, rhs_scale_storage
                if sfa.shape != (m, q.in_features // 128) or sfb.shape != (
                    q.out_features // 128,
                    q.in_features // 128,
                ):
                    raise ValueError(
                        "block-FP8 scale geometry differs from preparation"
                    )
                if self.gemv is not None:
                    out = torch.empty(
                        (m, q.out_features), dtype=torch.bfloat16, device=self.device
                    )
                    self.gemv(
                        lhs_values,
                        sfa,
                        rhs_values,
                        sfb,
                        out,
                        cuda_stream_to_int(stream),
                    )
                    return out
            else:
                view = (
                    as_grouped_scale_view
                    if q.recipe == "nvfp4"
                    else as_grouped_scale_view_mx
                )
                sfa = view(
                    scale_storage(
                        lhs_scale_storage, m, q.in_features, sf_vec_size
                    ).view(1, -1),
                    m,
                    q.in_features,
                )
                sfb = view(
                    scale_storage(
                        rhs_scale_storage, q.out_features, q.in_features, sf_vec_size
                    ).view(1, -1),
                    q.out_features,
                    q.in_features,
                )
            return self.dense.run(
                (lhs_values.view(m, lhs_values.shape[1], 1), sfa),
                (rhs_values.view(q.out_features, rhs_values.shape[1], 1), sfb),
                alpha=alpha,
                stream=stream,
            )[:, :, 0]

    def run_mxfp8(
        self,
        source_values,
        weight_values,
        weight_scale_mma,
        *,
        source_scale=None,
        out_dtype,
        stream,
    ):
        from ._a16 import _stream_context, _check_tensor
        from ._linear import _pad_k, _mxfp8_scale_mma_from_input

        q = self.query
        if q.call_kind != "packed" or q.recipe != "mxfp8":
            raise ValueError("execution does not prepare this packed MXFP8 operation")
        self._check(source_values, weight_values, out_dtype)
        m, k = source_values.shape[0], q.padded_in_features
        if m == 0:
            return source_values.new_empty((0, q.out_features), dtype=out_dtype)
        with torch.cuda.device(self.device), _stream_context(stream, self.device):
            source = _pad_k(source_values, k)
            if q.input_dtype == "float16":
                if source_scale is not None:
                    raise ValueError("plain FP16 inputs do not carry activation scales")
                from b12x.gemm._shared.wo_mxfp8 import (
                    empty_mxfp8_rows_bases,
                    mxfp8_rows_from_bases,
                )

                bases = empty_mxfp8_rows_bases(
                    m, k, num_groups=1, device=self.device, initialize_scales=False
                )
                packed = mxfp8_rows_from_bases(*bases, m, k, num_groups=1)
                _check_tensor("FP16 source", source, self.device, torch.float16)
                self.quantizer(
                    source, packed.values, packed.scale_rows, packed.scale_mma
                )
                values, scales = packed.values, packed.scale_mma
            else:
                if source_scale is None:
                    raise ValueError("prequantized MXFP8 requires activation scales")
                form = (
                    "compact"
                    if source_scale.ndim == 2
                    and source_scale.shape == (m, q.in_features // 32)
                    else "mma"
                    if source_scale.ndim == 6
                    else "swizzled"
                )
                if form != q.source_scale_form:
                    raise ValueError("activation scale form differs from preparation")
                values = source
                scales = _mxfp8_scale_mma_from_input(
                    source_scale, rows=m, width=k, logical_width=q.in_features
                ).view(torch.float8_e8m0fnu)
            return self.dense.run(
                (values.view(m, k, 1), scales),
                (weight_values.view(q.out_features, k, 1), weight_scale_mma),
                stream=stream,
            )[:, :, 0]

    def run_tensor_fp8(
        self,
        source_values,
        weight_values,
        weight_scale_mma,
        weight_block_scale,
        output_scale,
        *,
        out_dtype,
        stream,
    ):
        from ._a16 import _stream_context
        from ._linear import _pad_k

        q = self.query
        if q.call_kind != "packed" or q.recipe != "tensor_fp8":
            raise ValueError("execution does not prepare tensor-FP8")
        self._check(source_values, weight_values, out_dtype)
        m, k = source_values.shape[0], q.padded_in_features
        if m == 0:
            return source_values.new_empty((0, q.out_features), dtype=out_dtype)
        with torch.cuda.device(self.device), _stream_context(stream, self.device):
            source = _pad_k(source_values, k)
            return self.dense.run(
                (
                    source.view(m, k, 1),
                    self.unit_scale[:m]
                    if self.use_block
                    else self.unit_scale[:, :, : ((m + 127) // 128)],
                ),
                (
                    weight_values.view(q.out_features, k, 1),
                    weight_block_scale if self.use_block else weight_scale_mma,
                ),
                alpha=output_scale,
                stream=stream,
            )[:, :, 0]


def _uses_block_fp8_gemv(query) -> bool:
    """Whether the small-row block-FP8 GEMV serves this fixed plan."""
    from ._block_fp8_gemv import supports

    return (
        query.recipe == "block_fp8"
        and query.call_kind == "serialized"
        and query.expected_m is not None
        and query.output_dtype == "bfloat16"
        and query.alpha_mode == "unit"
        and supports(query.max_rows, query.out_features, query.in_features)
    )


def _plan_fixed_block_fp8_gemv(query, *, invocation, override):
    from ._tuning import FIXED_TUNING

    def materialize(selection, device):
        from ._block_fp8_gemv import compile_block_fp8_gemv

        program = compile_block_fp8_gemv(
            device.ordinal, query.out_features, query.in_features
        )
        target = torch.device("cuda", device.ordinal)
        return _FixedExecutionState(query, target, None, None, None, True, gemv=program)

    return Plan(
        shared=True,
        contract=FIXED_TUNING,
        query=query,
        invocation=FrozenMapping(invocation),
        override=override,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x.gemm.blockscaled._block_fp8_gemv:compile_block_fp8_gemv",
                CompileJob.DEVICE_ORDINAL,
                query.out_features,
                query.in_features,
            ),
        ),
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=materialize,
    )


def _plan_fixed(query, *, invocation=FrozenMapping(), override=None):
    from ._tuning import FIXED_TUNING

    if invocation:
        raise ValueError("fixed packed semantics belong in FixedBlockscaledQuery")
    if _uses_block_fp8_gemv(query):
        return _plan_fixed_block_fp8_gemv(
            query, invocation=invocation, override=override
        )
    cache = {}

    def lower(device):
        if device.identity not in cache:
            cache[device.identity] = _fixed_lowering(query, device)
        return cache[device.identity]

    def memory(config, device):
        from . import _linear

        p, use_block = lower(device)
        entries = []
        if p.alpha_is_one:
            resident = _module__lib_dense_gemm_buffers._ALPHA_ONE_CACHE.get(
                ("cuda", device.ordinal)
            )
            entries.append(
                PersistentMemory(
                    ("dense.alpha_one", device.ordinal),
                    4,
                    0
                    if resident is None
                    else resident.numel() * resident.element_size(),
                )
            )
        if query.recipe == "tensor_fp8":
            key = ("cuda", device.ordinal, query.max_rows, query.padded_in_features)
            table = (
                _linear._UNIT_BLOCK_SCALE_CACHE
                if use_block
                else _linear._UNIT_SCALE_MMA_CACHE
            )
            resident = table.get(key)
            needed = (
                query.max_rows * (query.padded_in_features // 128) * 4
                if use_block
                else ((query.max_rows + 127) // 128)
                * ((query.padded_in_features + 127) // 128)
                * 512
            )
            entries.append(
                PersistentMemory(
                    ("tensor_fp8.unit_scale", use_block, *key),
                    needed,
                    0
                    if resident is None
                    else resident.numel() * resident.element_size(),
                )
            )
        return MemoryRequirements(persistent=tuple(entries))

    def materialize(selection, device):
        from . import _linear

        p, use_block = lower(device)
        programs = compile_fixed(
            FIXED_TUNING.encode_query(query),
            p.to_dict(),
            device.ordinal,
            device.identity.sm_count,
        )
        target = torch.device("cuda", device.ordinal)
        core = _module__lib_dense_gemm_preparation._DenseExecutionState(
            p,
            target,
            programs["gemm"],
            programs.get("reduce"),
            _module__lib_dense_gemm_buffers._cached_alpha_one(target)
            if p.alpha_is_one
            else None,
        )
        scale = None
        if query.recipe == "tensor_fp8":
            factory = (
                _linear._cached_unit_activation_block_scale
                if use_block
                else _linear._cached_unit_scale_mma
            )
            scale = factory(
                "cuda", device.ordinal, query.max_rows, query.padded_in_features
            )
        return _FixedExecutionState(
            query, target, core, programs.get("quantize"), scale, use_block
        )

    return Plan(
        shared=True,
        contract=FIXED_TUNING,
        query=query,
        invocation=FrozenMapping(invocation),
        override=override,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x.gemm.blockscaled._preparation:compile_fixed",
                FIXED_TUNING.encode_query(query),
                lower(device)[0].to_dict(),
                CompileJob.DEVICE_ORDINAL,
                device.identity.sm_count,
            ),
        ),
        _memory_requirements=memory,
        _materialize=materialize,
    )


def plan(query, *, invocation=FrozenMapping(), override=None):
    from ._tuning import FixedBlockscaledQuery
    from b12x.gemm._tuning import DenseGemmQuery

    if isinstance(query, BlockscaledQuery):
        return _plan_bf16(query, invocation=invocation, override=override)
    if isinstance(query, FixedBlockscaledQuery):
        return _plan_fixed(query, invocation=invocation, override=override)
    if isinstance(query, DenseGemmQuery):
        from b12x.gemm._preparation import plan as dense_plan

        return dense_plan(query, invocation=invocation, override=override)
    raise TypeError("unsupported blockscaled declaration query")


_HEURISTIC_PLANS = {}


def heuristic_plan(query, device):
    """Retain a default plan for a power-of-two row capacity on one device."""
    from dataclasses import replace

    if query.expected_m is not None and (
        type(query.expected_m) is not int or query.expected_m <= 0
    ):
        raise ValueError("expected_m must be positive or None")
    field = "num_tokens" if isinstance(query, BlockscaledQuery) else "max_rows"
    rows = max(1, getattr(query, field), query.expected_m or 1)
    capacity = 1 << (rows - 1).bit_length()
    query = replace(query, **{field: capacity, "expected_m": None})
    key = (torch.device(device), query)
    declaration = _HEURISTIC_PLANS.get(key)
    if declaration is None:
        declaration = replace(plan(query), _device=key[0])
        _HEURISTIC_PLANS[key] = declaration
    return declaration


@dataclass(frozen=True)
class _PackedRegimeState:
    """Exact static-shape states plus one bounded dynamic-row state."""

    capacity: _PackedExecutionState
    exact: MappingProxyType
    a16_capacity: _PackedExecutionState | None = None

    @property
    def required_workspace(self) -> int:
        return max(
            (
                self.capacity.required_workspace,
                *(state.required_workspace for state in self.exact.values()),
                *(
                    ()
                    if self.a16_capacity is None
                    else (self.a16_capacity.required_workspace,)
                ),
            )
        )

    def resolve(self, source: torch.Tensor) -> _PackedExecutionState:
        rows = source.numel() // source.shape[-1]
        if rows > self.capacity.query.num_tokens:
            raise ValueError(
                f"packed execution rows {rows} exceed capacity "
                f"{self.capacity.query.num_tokens}"
            )
        if rows in self.exact:
            return self.exact[rows]
        if self.a16_capacity is not None and rows <= self.a16_capacity.query.num_tokens:
            return self.a16_capacity
        return self.capacity


def plan_regimes(
    query: BlockscaledQuery,
    *,
    exact_m: tuple[int, ...] = (),
    a16_max_tokens: int = 0,
    invocation=FrozenMapping(),
    override=None,
):
    """Declare exact shapes and capacity fallbacks, forcing A16 up to a cutoff.

    ``a16_max_tokens`` is inclusive; zero leaves precision unconstrained by
    token count. Larger calls retain the query's activation mode.
    """
    if not isinstance(query, BlockscaledQuery):
        raise TypeError("packed regime planning requires BlockscaledQuery")
    if query.expected_m is not None:
        raise ValueError("the packed capacity query must leave expected_m unset")
    if type(a16_max_tokens) is not int or a16_max_tokens < 0:
        raise ValueError("a16_max_tokens must be a nonnegative integer")
    cutoff = min(a16_max_tokens, query.num_tokens)
    counts = tuple(
        sorted({int(rows) for rows in exact_m if 0 < int(rows) < query.num_tokens})
    )
    if len(counts) != len(tuple(exact_m)):
        raise ValueError("exact M values must be unique and below capacity")
    child_queries = {
        rows: replace(query, num_tokens=rows, expected_m=rows) for rows in counts
    }
    child_queries[query.num_tokens] = query
    if cutoff:
        child_queries[cutoff] = replace(query, num_tokens=cutoff)
        child_queries = {
            rows: replace(child, activation_mode="a16") if rows <= cutoff else child
            for rows, child in sorted(child_queries.items())
        }
    children = {
        rows: _plan_bf16(child, invocation=invocation, override=override)
        for rows, child in child_queries.items()
    }

    def assemble(states, device):
        del device
        capacity = states[query.num_tokens]
        exact = MappingProxyType({rows: states[rows] for rows in counts})
        return _PackedRegimeState(capacity, exact, states[cutoff] if cutoff else None)

    return _CompositePlan(
        component_id="gemm.blockscaled_precision",
        capacity_metadata=FrozenMapping(
            {
                "max_rows": query.num_tokens,
                "exact_m": counts,
                **({"a16_max_tokens": cutoff} if cutoff else {}),
            }
        ),
        variants=children,
        _assemble=assemble,
    )


def query_from_call(
    source,
    weight,
    *,
    source_activation="identity",
    activation_mode="auto",
    activation_global_scale=None,
    out=None,
    workspace=None,
    expected_m=None,
    out_dtype=None,
    alpha=None,
    output_partition=None,
    reference_rows=None,
    **options,
):
    """Describe a packed projection, optionally preserving a global output extent.

    ``output_partition=(logical_rows, first, end)`` declares the local weight's
    contiguous output-row range. It requires an exact ``expected_m >= 128``,
    MXFP8 weights, BF16 A16 arithmetic and caller-owned output/workspace. The
    plan expands only stored rows into zero-filled full-N scratch, computes
    the full-N BF16 GEMM and extracts the local output. Storage and execution
    reject packed-A16 or dynamic-row substitutes for that arithmetic contract.
    ``reference_rows`` retains the cuBLASLt algorithm selected for an enclosing
    row count while executing only the supplied independent rows. To preserve
    bitwise output, the caller must retain each row's placement within the MMA
    tile; retaining an arbitrary unaligned suffix at row zero does not do that.
    """
    from ._a16 import MXFP4LinearWeight, NVFP4LinearWeight
    from ._iq2_xs import IQ2XSLinearWeight
    from ._linear import MXFP8LinearWeight, TensorFP8LinearWeight
    from ._tuning import FixedBlockscaledQuery

    if (output_partition is not None or reference_rows is not None) and not (
        isinstance(weight, MXFP8LinearWeight)
        and isinstance(source, torch.Tensor)
        and source.dtype == torch.bfloat16
    ):
        raise ValueError(
            "output partitions require packed MXFP8 weights and BF16 source"
        )
    if isinstance(
        weight, (NVFP4LinearWeight, MXFP4LinearWeight, IQ2XSLinearWeight)
    ) or (
        isinstance(weight, MXFP8LinearWeight)
        and isinstance(source, torch.Tensor)
        and source.dtype == torch.bfloat16
    ):
        allow_bf16_atomic_reduction = options.pop("allow_bf16_atomic_reduction", True)
        if options or alpha is not None or out_dtype not in (None, torch.bfloat16):
            raise ValueError("unsupported packed BF16 declaration options")
        return _query_bf16_from_call(
            source,
            weight,
            activation_mode=activation_mode,
            source_activation=source_activation,
            allow_bf16_atomic_reduction=allow_bf16_atomic_reduction,
            activation_global_scale=activation_global_scale,
            out=out,
            workspace=workspace,
            expected_m=expected_m,
            output_partition=output_partition,
            reference_rows=reference_rows,
        )
    if source_activation != "identity":
        raise ValueError("source activation requires a packed BF16 NVFP4 projection")
    if isinstance(source, tuple) and isinstance(weight, tuple) and source[0].ndim == 3:
        from b12x.gemm._tuning import query_from_call as dense_query

        recipe = dict(options)
        recipe.update(alpha=alpha, expected_m=expected_m)
        recipe.setdefault(
            "c_dtype", str(out_dtype or torch.bfloat16).removeprefix("torch.")
        )
        if workspace is not None:
            recipe["_split_k_workspace"] = workspace
        return dense_query(
            source, weight, out, entry_point="gemm.blockscaled.mm", options=recipe
        )
    if (
        out is not None
        or workspace is not None
        or activation_mode != "auto"
        or activation_global_scale is not None
    ):
        raise ValueError(
            "fixed packed wrappers do not accept A16/output/workspace constraints"
        )
    if isinstance(weight, (MXFP8LinearWeight, TensorFP8LinearWeight)):
        if options or alpha is not None:
            raise ValueError("fixed packed weights do not accept raw GEMM options")
        values = source[0] if isinstance(source, tuple) else source
        if values.ndim < 2 or values.shape[-1] != weight.in_features:
            raise ValueError("fixed packed source geometry differs from its weight")
        m = values.numel() // weight.in_features
        recipe = "mxfp8" if isinstance(weight, MXFP8LinearWeight) else "tensor_fp8"
        dtype = (
            values.dtype
            if recipe == "mxfp8" and not isinstance(source, tuple)
            else torch.bfloat16
        )
        form = "none"
        if isinstance(source, tuple):
            scales = source[1]
            form = (
                "compact"
                if scales.ndim == 2 and scales.shape == (m, weight.in_features // 32)
                else "mma"
                if scales.ndim == 6
                else "swizzled"
            )
        return FixedBlockscaledQuery(
            recipe=recipe,
            call_kind="packed",
            max_rows=m,
            in_features=weight.in_features,
            padded_in_features=weight.padded_in_features,
            out_features=weight.out_features,
            input_dtype=str(values.dtype).removeprefix("torch."),
            output_dtype=str(out_dtype or dtype).removeprefix("torch."),
            expected_m=m if expected_m is None else expected_m,
            source_scale_form=form,
        )
    if not isinstance(source, tuple) or not isinstance(weight, tuple):
        raise TypeError("fixed serialized declarations require operand pairs")
    a, sa = source
    b, sb = weight
    recipe_key = (
        options.get("ab_dtype"),
        options.get("sf_dtype"),
        options.get("sf_vec_size"),
        bool(options.get("block_fp8", False)),
    )
    recipes = {
        ("float4_e2m1fn", "float8_e4m3fn", 16, False): "nvfp4",
        ("float4_e2m1fn", "float8_e8m0fnu", 32, False): "mxfp4",
        ("float8_e4m3fn", "float32", 128, True): "block_fp8",
    }
    if (
        recipe_key not in recipes
        or a.ndim != 2
        or b.ndim != 2
        or a.shape[1] != b.shape[1]
    ):
        raise ValueError("unsupported serialized operand recipe or storage")
    if set(options) - {"ab_dtype", "sf_dtype", "sf_vec_size", "block_fp8", "c_dtype"}:
        raise ValueError("serialized declarations reject raw GEMM launch overrides")
    recipe = recipes[recipe_key]
    k = a.shape[1] * (1 if recipe == "block_fp8" else 2)
    return FixedBlockscaledQuery(
        recipe=recipe,
        call_kind="serialized",
        max_rows=a.shape[0],
        in_features=k,
        padded_in_features=k,
        out_features=b.shape[0],
        input_dtype=str(a.dtype).removeprefix("torch."),
        output_dtype=options.get(
            "c_dtype", str(out_dtype or torch.bfloat16).removeprefix("torch.")
        ),
        expected_m=a.shape[0] if expected_m is None else expected_m,
        alpha_mode="unit" if alpha is None else "tensor",
    )
