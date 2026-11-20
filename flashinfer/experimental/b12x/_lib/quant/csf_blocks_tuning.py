"""Session-prepared expansion of compact FP4 scale records.

The tuner sweeps work per CTA, exception placement, and expert-presence
construction. Scale payloads and native outputs remain caller-owned. Each
prepared binding uses one fixed expert-presence scratch tensor.

Direct duplicate checks are compared above the expert count only at max effort.
Routing capacities below 64 retain direct checks; capacities at least sixteen
times the expert count expand every expert at every effort.
"""

from dataclasses import asdict, dataclass, field, replace

import os

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import ScratchBufferSpec, scratch_tensor
from b12x.preparation import (
    FrozenMapping,
    Knob,
    MemoryRequirements,
    ParameterBinding,
    ParameterSpace,
    Plan,
    TuningContract,
)
from b12x.preparation._efficiency import (
    AutotuneEffort,
    capture_autotune_effort,
)
from b12x._lib.quant.csf_blocks import Nvfp4CompactBlockPlane
from b12x._lib.quant.csf_routing import compile_csf_active_experts
from b12x._lib.quant.nvfp4_csf import Nvfp4CsfDecoder, compile_nvfp4_csf_pair


@dataclass(frozen=True)
class CsfExpansionQuery:
    first: tuple[int, ...]
    second: tuple[int, ...]
    routes: int
    ids_dtype: str
    autotune_effort: str = field(default_factory=capture_autotune_effort)


@dataclass(frozen=True)
class CsfExpansionConfig:
    groups_per_task: int = 8
    shared_exceptions: bool = True
    bitmap_routes: bool = False
    direct_routes: bool = False

    @classmethod
    def from_config(cls, payload):
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError(
                "CSF expansion configuration fields differ from its contract"
            )
        return cls(**dict(payload))


def _uses_presence(query):
    return 64 <= query.routes < 16 * query.first[6]


def _default_direct_routes(query) -> bool:
    """Untuned default for direct per-slot CSF deduplication (Lever-2 M1).

    Route capacities 64..128 (Hy4 decode caps 8/16 = 64/128 routes) previously
    defaulted to the two-launch presence path (``mark_active_experts`` + a
    256-expert mode-2 expansion), which also silently disabled the fused
    ``CsfRoutePack`` at exactly the packed-GEMM route boundary. Direct dedup
    expands every active expert exactly once (first route occurrence), which is
    bit-identical to presence expansion, and shrinks the expansion grid to
    ``routes * tasks`` CTAs. Default it on for presence-routed capacities that
    fit the fused route pack (``routes <= 128``) and the direct-tuning
    predicate (``routes <= E``); autotune may still select either value.
    """
    return _uses_presence(query) and query.routes <= min(128, query.first[6])


def _default_groups_per_task() -> int:
    """Untuned work-per-CTA default.

    ``B12X_CSF_GROUPS_PER_TASK`` lets the M1 sweep harness price 16/32-group
    tasks (fewer CTAs, shorter duplicate scans) without code edits; the value
    participates in compiled geometry and must be set before preparation.
    """
    value = os.environ.get("B12X_CSF_GROUPS_PER_TASK", "8")
    if value not in ("1", "2", "4", "8", "16", "32", "64"):
        raise ValueError(
            "B12X_CSF_GROUPS_PER_TASK must be 1, 2, 4, 8, 16, 32, or 64"
        )
    return int(value)


def _validate_query(query, device):
    if not isinstance(query, CsfExpansionQuery):
        raise TypeError("CSF expansion requires a CsfExpansionQuery")
    if type(query.routes) is not int or query.routes <= 0:
        raise ValueError("CSF expansion requires a positive route capacity")
    if query.ids_dtype not in ("int32", "int64"):
        raise ValueError("CSF expansion routes require int32 or int64")
    AutotuneEffort(query.autotune_effort)
    for geometry in (query.first, query.second):
        if (
            len(geometry) != 9
            or geometry[2] not in (1, 4)
            or geometry[3] not in range(4, 10)
        ):
            raise ValueError(
                "Native expansion requires compact one-bit or four-bit scales"
            )
    if query.first[6] != query.second[6]:
        raise ValueError("CSF expansion expert counts must agree")


def _validate_config(query, config, device):
    if not isinstance(config, CsfExpansionConfig):
        raise TypeError("CSF expansion requires a CsfExpansionConfig")
    if type(config.groups_per_task) is not int or config.groups_per_task not in (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
    ):
        raise ValueError("CSF expansion tasks require 1, 2, 4, 8, 16, 32, or 64 groups")
    if (
        type(config.shared_exceptions) is not bool
        or type(config.bitmap_routes) is not bool
        or type(config.direct_routes) is not bool
    ):
        raise TypeError("CSF expansion switches must be boolean")
    if config.direct_routes and not _uses_presence(query):
        raise ValueError("Direct-route tuning requires a presence-routed capacity")
    if config.direct_routes and config.bitmap_routes:
        raise ValueError("Direct routes do not use an expert-presence bitmap")
    if config.bitmap_routes and not _uses_presence(query):
        raise ValueError("Expert-presence tuning requires a presence-routed capacity")


def _parameters(query, device):
    return ParameterSpace.create(
        TUNING.knobs,
        values={
            "direct_routes": (False, True) if _uses_presence(query) else (False,),
            "bitmap_routes": (False, True) if _uses_presence(query) else (False,),
        },
        predicates=(
            lambda p: not p["direct_routes"] or query.routes <= query.first[6],
        ),
        efficiency_predicates=(lambda p: p["groups_per_task"] not in (1, 64),),
        autotune_effort=query.autotune_effort,
    )


def _default_config(query, device):
    direct = _default_direct_routes(query)
    return CsfExpansionConfig(
        groups_per_task=_default_groups_per_task(),
        bitmap_routes=_uses_presence(query) and not direct,
        direct_routes=direct,
    )


TUNING = TuningContract(
    component_id="quantization.csf_block_expansion",
    query_schema_version=9,
    config_schema_version=2,
    semantic_version=2,
    query_fields=frozenset(CsfExpansionQuery.__dataclass_fields__),
    config_fields=frozenset(CsfExpansionConfig.__dataclass_fields__),
    encode_query=asdict,
    encode_config=asdict,
    decode_config=CsfExpansionConfig.from_config,
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=_default_config,
    knobs=(
        Knob(name="groups_per_task", values=(1, 2, 4, 8, 16, 32, 64)),
        Knob(name="shared_exceptions", values=(False, True)),
        Knob(
            name="direct_routes",
            values=(False, True),
            binding=ParameterBinding.RUNTIME,
        ),
        Knob(
            name="bitmap_routes",
            values=(False, True),
            when=FrozenMapping({"direct_routes": False}),
            otherwise=False,
        ),
    ),
    parameters=_parameters,
)


def _program_query_payload(query):
    payload = asdict(query)
    payload.pop("autotune_effort")
    return payload


@program_cache(scope="preparation")
def compile_expansion(query_payload, config_payload, ordinal):
    values = dict(query_payload)
    values.setdefault("autotune_effort", AutotuneEffort.DEFAULT)
    query = CsfExpansionQuery(**values)
    config = CsfExpansionConfig.from_config(config_payload)
    geometries = tuple(
        (
            *geometry[:7],
            config.groups_per_task,
            int(config.shared_exceptions),
            *geometry[7:],
        )
        for geometry in (query.first, query.second)
    )
    with torch.cuda.device(ordinal):
        programs = tuple(
            compile_nvfp4_csf_pair(*geometries, ids64) for ids64 in (False, True)
        )
        routing = tuple(
            compile_csf_active_experts(
                ids64, query.first[6] if config.bitmap_routes else 0
            )
            for ids64 in (False, True)
        )
        task_programs = tuple(
            compile_nvfp4_csf_pair(*geometries, ids64, fuse_tasks=True)
            for ids64 in (False, True)
        )
        mapped_programs = tuple(
            compile_nvfp4_csf_pair(*geometries, ids64, mapped=True)
            for ids64 in (False, True)
        )
        mapped_routing = tuple(
            compile_csf_active_experts(
                ids64, query.first[6] if config.bitmap_routes else 0, mapped=True
            )
            for ids64 in (False, True)
        )
    return programs, routing, task_programs, mapped_programs, mapped_routing


def _scratch(experts, device):
    return (
        ScratchBufferSpec(
            name="csf_active_experts",
            shape=(experts,),
            dtype=torch.int32,
            device=device,
        ),
    )


@dataclass(frozen=True)
class _Binding:
    decoder: Nvfp4CsfDecoder
    ids: torch.Tensor
    out13: torch.Tensor
    out2: torch.Tensor

    def run(self):
        self.decoder.decode(self.ids, self.out13, self.out2)


@dataclass(frozen=True)
class _ExpansionState:
    query: CsfExpansionQuery
    first: Nvfp4CompactBlockPlane
    second: Nvfp4CompactBlockPlane
    programs: tuple
    routing: tuple
    device: torch.device
    bitmap_routes: bool
    direct_routes: bool
    task_programs: tuple
    mapped_programs: tuple
    mapped_routing: tuple

    def decoder(self, scratch):
        active = scratch_tensor(
            scratch,
            _scratch(self.first.num_experts, self.device),
            owner="CSF expansion",
        )
        return Nvfp4CsfDecoder(
            self.first,
            self.second,
            self.programs,
            self.routing,
            active[: self.first.num_experts],
            bitmap_routes=self.bitmap_routes,
            direct_routes=self.direct_routes,
            mapped_programs=self.mapped_programs,
            mapped_routing_programs=self.mapped_routing,
        )

    def bind(self, *, ids, out13, out2, scratch):
        if (ids.numel(), str(ids.dtype).removeprefix("torch."), ids.device) != (
            self.query.routes,
            self.query.ids_dtype,
            self.device,
        ):
            raise ValueError(
                "CSF routes differ from the prepared capacity, dtype, or device"
            )
        return _Binding(self.decoder(scratch), ids, out13, out2)

    def run(self, binding):
        binding.run()


def plan_csf_block_expansion(
    first,
    second,
    *,
    routes,
    ids_dtype=torch.int32,
    invocation=FrozenMapping(),
    override=None,
):
    """Declare an expansion sweep without compiling or allocating GPU storage."""
    for plane in (first, second):
        if not isinstance(plane, Nvfp4CompactBlockPlane):
            raise TypeError("CSF expansion planning requires compact FP4 block planes")
        plane.validate()
    if first.fixed.device != second.fixed.device:
        raise ValueError("CSF expansion planes must be on one device")
    query = CsfExpansionQuery(
        first.geometry[:7] + first.geometry[9:],
        second.geometry[:7] + second.geometry[9:],
        routes,
        str(ids_dtype).removeprefix("torch."),
    )

    def compile_jobs(config, device):
        return (
            CompileJob.create(
                "b12x._lib.quant.csf_blocks_tuning:compile_expansion",
                _program_query_payload(query),
                TUNING.encode_config(config),
                CompileJob.DEVICE_ORDINAL,
            ),
        )

    def materialize(selection, device):
        if first.fixed.device != torch.device("cuda", device.ordinal):
            raise ValueError("CSF payload and preparation devices differ")
        config = selection.config
        planes = tuple(
            replace(
                p,
                groups_per_task=config.groups_per_task,
                shared_exceptions=config.shared_exceptions,
            )
            for p in (first, second)
        )
        programs, routing, task_programs, mapped_programs, mapped_routing = (
            compile_expansion(
                _program_query_payload(query),
                TUNING.encode_config(config),
                device.ordinal,
            )
        )
        return _ExpansionState(
            query,
            *planes,
            programs,
            routing,
            torch.device("cuda", device.ordinal),
            config.bitmap_routes,
            config.direct_routes,
            task_programs,
            mapped_programs,
            mapped_routing,
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _device=first.fixed.device,
        _compile_jobs=compile_jobs,
        _memory_requirements=lambda config, device: MemoryRequirements(
            scratch=_scratch(first.num_experts, torch.device("cuda", device.ordinal))
        ),
        _materialize=materialize,
    )
