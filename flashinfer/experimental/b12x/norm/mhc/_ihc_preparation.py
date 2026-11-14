"""Per-invocation iHC lowering; runtime consumes only resolved launchers.

Mirrors the mHC preparation flow (``_preparation.plan_mhc``) at iHC's scale:
``plan_ihc`` declares one operation with its numerical recipe, the
``TuningContract`` (``norm.mhc.ihc``) picks a launch configuration from the
``IHC_TUNING`` rows, and ``PreparationSession`` compiles (Triton-warms) exactly
the chosen branch before ``state.bind`` builds the caller-owned-buffer view.
"""

from __future__ import annotations

from functools import partial
from types import MappingProxyType

import torch

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from b12x.preparation._efficiency import capture_autotune_effort

from ._ihc_tuning import IHC_MULT, IhcConfig, IhcQuery, TUNING

_RECIPE_FIELDS = {
    "pre": ("magnitude", "hc_eps", "norm_eps"),
    "post_pre": ("magnitude", "hc_eps", "norm_eps"),
    "head": ("hc_eps", "norm_eps"),
    "post": (),
}
_QUERY_OPTIONS = frozenset(IhcQuery.__dataclass_fields__) - {
    "dtype",
    "max_tokens",
    "hidden_size",
    "autotune_effort",
}


def _decode_query(payload):
    return IhcQuery(**dict(payload))


def compile_ihc(query_payload, config_payload, ordinal):
    """Reuse the warmed Triton programs of one iHC operation and geometry."""
    return _compile_ihc(FrozenMapping(query_payload), FrozenMapping(config_payload), ordinal)


@program_cache(scope="preparation")
def _compile_ihc(query_payload, config_payload, ordinal):
    """Warm exactly the chosen branch without allocating tensor storage."""
    query = _decode_query(query_payload)
    config = IhcConfig.from_config(config_payload)
    if config.backend != "triton_fused":
        return {}
    from . import _ihc_kernels as kernels

    hidden, total_k = query.hidden_size, IHC_MULT * query.hidden_size
    warp = config.num_warps
    programs = {}
    # The norm-weight pointer slot must warm the dtype the run passes: the
    # folded operand when the query declares one (`require` pins its dtype),
    # else the fp32 `hc_base` dummy the launchers hand Triton. Warming any
    # other pointer dtype compiles a second specialization at the first real
    # forward — the plan's warmed program must match the launch exactly.
    norm_ptr_dtype = (getattr(torch, query.norm_weight_dtype)
                      if query.has_norm_weight else torch.float32)
    if query.operation == "pre":
        programs["pre"] = kernels._ihc_pre_kernel.warmup(
            torch.bfloat16,
            torch.float32,
            torch.float32,
            torch.float32,
            torch.float32,
            torch.bfloat16,
            hidden,
            total_k,
            float(query.magnitude),
            float(query.hc_eps),
            float(query.norm_eps),
            norm_ptr_dtype,
            float(query.layernorm_eps),
            BLOCK_K=config.block_k,
            BLOCK_D=config.block_d,
            FUSE_NORM=query.has_norm_weight,
            num_warps=warp,
            grid=(query.max_tokens,),
        )
    elif query.operation == "post":
        programs["post"] = kernels._ihc_post_kernel.warmup(
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            torch.bfloat16,
            hidden,
            total_k,
            BLOCK_D=config.block_d,
            num_warps=warp,
            grid=(query.max_tokens, (hidden + config.block_d - 1) // config.block_d),
        )
    elif query.operation == "post_pre":
        programs["post_pre"] = kernels._ihc_post_pre_kernel.warmup(
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            torch.float32,
            torch.float32,
            torch.float32,
            torch.bfloat16,
            torch.float32,
            torch.bfloat16,
            hidden,
            total_k,
            float(query.magnitude),
            float(query.hc_eps),
            float(query.norm_eps),
            norm_ptr_dtype,
            float(query.layernorm_eps),
            BLOCK_K=config.block_k,
            BLOCK_D=config.block_d,
            FUSE_NORM=query.has_norm_weight,
            num_warps=warp,
            grid=(query.max_tokens,),
        )
    else:
        programs["head"] = kernels._ihc_head_kernel.warmup(
            torch.bfloat16,
            torch.float32,
            torch.float32,
            torch.float32,
            torch.bfloat16,
            hidden,
            total_k,
            float(query.hc_eps),
            float(query.norm_eps),
            BLOCK_K=config.block_k,
            BLOCK_D=config.block_d,
            num_warps=warp,
            grid=(query.max_tokens,),
        )
    return programs


def _materialize(caps, query, selection, device):
    from ._ihc_impl import _IhcState, _launchers_for

    programs = compile_ihc(
        TUNING.encode_query(query), selection.config.to_dict(), device.ordinal
    )
    launchers = _launchers_for(selection.config)
    if programs:
        launchers = {
            op: attach_programs(launcher, programs)
            for op, launcher in launchers.items()
        }
    return _IhcState(
        caps=caps,
        query=query,
        config=selection.config,
        launchers=MappingProxyType(launchers),
    )


def plan_ihc(caps, *, invocation=FrozenMapping(), override=None):
    from ._ihc_impl import B12XIHCCaps

    if not isinstance(caps, B12XIHCCaps):
        raise TypeError("caps must be B12XIHCCaps")
    invocation = FrozenMapping(invocation)
    options = invocation.to_dict()
    operation = options.get("operation")
    if operation not in ("pre", "post", "post_pre", "head"):
        raise ValueError("iHC declarations require one explicit operation")
    if not set(_RECIPE_FIELDS[operation]) <= set(options):
        raise ValueError("iHC declarations require their numerical recipe")
    if set(options) - _QUERY_OPTIONS:
        raise ValueError(
            f"unknown iHC invocation fields: {sorted(set(options) - _QUERY_OPTIONS)}"
        )
    query = IhcQuery(
        dtype=str(caps.dtype).removeprefix("torch."),
        max_tokens=caps.max_tokens,
        hidden_size=caps.hidden_size,
        autotune_effort=capture_autotune_effort().value,
        **options,
    )
    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _device=caps.device,
        shared=True,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x.norm.mhc._ihc_preparation:compile_ihc",
                TUNING.encode_query(query),
                config.to_dict(),
                CompileJob.DEVICE_ORDINAL,
            ),
        ),
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=partial(_materialize, caps, query),
    )


__all__ = ["compile_ihc", "plan_ihc"]
