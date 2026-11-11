"""Singleton declarations for caller-owned PCIe communication surfaces.

Each surface retains its existing communicator, transport precision, launch
settings and channel ownership. This contract does not select interchangeable
collective backends or qualify a topology from a visible-device count.
"""

from dataclasses import dataclass, replace

from b12x.preparation import BackendConfig, FrozenMapping, make_fixed_contract


# Native execution surfaces after the all-reduce manager's plan-time routing.
# Values are supported worlds, never a set of transports to race.
SURFACES = {
    "OneshotAllReduce.all_reduce": (2, 3, 4, 6, 8, 10),
    "OneshotAllReduce.all_reduce_fused_add_rms_norm": (2, 3, 4, 6, 8, 10),
    "OneshotAllReduce.all_reduce_fused_add": (2, 3, 4, 6, 8, 10),
    "OneshotAllReducePool.all_reduce": (2, 3, 4, 6, 8, 10),
    "OneshotAllReducePool.all_reduce_fused_add_rms_norm": (2, 3, 4, 6, 8, 10),
    "DmaAllReduce.all_reduce": (2, 3, 4, 6, 8, 10),
    "PCIeTwoShotBF16.all_reduce": (2, 3, 4, 6, 8, 10),
    "PCIeTwoShotBF16.reduce_scatter": (2, 3, 4, 6, 8, 10),
    "PCIeTwoShotBF16.all_gather": (2, 3, 4, 6, 8, 10),
    "TwoShotReduceScatter.reduce_scatter_fp8": (2, 4, 8),
    "TwoShotReduceScatter.all_gather_fp8": (2, 4, 8),
    "DcpAllToAll.lse_reduce_scatter": (2, 4, 8, 9, 10, 12, 16),
    "DcpAllToAll.all_gather_heads": (2, 4, 8, 9, 10, 12, 16),
    "DcpAllToAll.all_gather_pair": (2, 4, 8, 16),
    "DcpAllToAll.all_gather_pair_kimi_topk": (2, 4, 8, 16),
    "DcpAllToAll.kimi_topk16": (2, 4, 8, 16),
    "DcpAllToAllPool.lse_reduce_scatter": (2, 4, 8, 9, 10, 12, 16),
    "DcpAllToAllPool.all_gather_heads": (2, 4, 8, 9, 10, 12, 16),
    "DcpAllToAllPool.all_gather_pair": (2, 4, 8, 16),
    "DcpAllToAllPool.all_gather_pair_kimi_topk": (2, 4, 8, 16),
    "DcpAllToAllPool.kimi_topk16": (2, 4, 8, 16),
    "DcpTopKOwnerExchange.stage_candidates": (2, 3, 4, 6, 8),
    "VocabParallelArgmax.fused_add_argmax": (8, 12, 16),
    "PCIeHierarchicalAllReduce.all_reduce": (9, 10, 12, 16),
    "PCIeIslandRSAllReduce.all_reduce": (16,),
    "PCIeIslandARAllReduce.all_reduce": (8,),
    "kimi_topk16": (1,),
}


@dataclass(frozen=True, kw_only=True)
class PcieQuery:
    surface: str
    world_size: int
    rank: int
    topology: str
    # Capacity, tensor geometry/dtypes, codec, channel, and existing host launch
    # overrides are concrete caller metadata, never alternative tuning labels.
    call: FrozenMapping
    setup: FrozenMapping


PcieConfig = BackendConfig
TUNING = make_fixed_contract(
    component_id="comm.pcie",
    query_type=PcieQuery,
    backend="native",
)
_validate_fixed_query = TUNING.validate_query


def _validate_query(query: PcieQuery, device) -> None:
    _validate_fixed_query(query, device)
    worlds = SURFACES.get(query.surface)
    if worlds is None:
        raise ValueError(f"unknown PCIe execution surface {query.surface!r}")
    if query.world_size not in worlds:
        raise ValueError(f"{query.surface} supports world sizes {worlds}")
    if not 0 <= query.rank < query.world_size:
        raise ValueError("rank must belong to the caller's process group")
    if "joint_cluster_size" in query.call:
        from ._joint_profile import profile_index, validate_cluster_geometry

        call = query.call
        if (
            not query.surface.endswith(".all_reduce")
            or call.get("transport") not in ("joint_latency", "joint_bandwidth")
            or call.get("dtype") != "bfloat16"
            or not call.get("stage_input")
        ):
            raise ValueError(
                "joint cluster geometry requires a staged joint BF16 all-reduce"
            )
        validate_cluster_geometry(
            call["blocks"], call["threads"], call["joint_cluster_size"], device
        )
        if call["blocks"] > query.setup.get("joint_max_blocks", 0):
            raise ValueError("joint cluster grid exceeds the prepared channel capacity")
        profile_index(
            call["transport"],
            call["blocks"],
            call["submode"],
            call["active_blocks"],
            cluster_size=call["joint_cluster_size"],
        )
    if query.surface.endswith(".all_gather_heads"):
        transport = query.call.get("gather_transport", "barrier")
        from ._gather_preparation import GATHER_MODES, validate_query as validate_gather

        if transport in GATHER_MODES:
            validate_gather(query)
        elif transport not in ("barrier", "word_exchange"):
            raise ValueError("unknown DCP gather transport")
        if transport == "word_exchange":
            from math import prod

            element_bytes = 1 if query.call["dtype"] == "torch.float8_e4m3fn" else 2
            nbytes = prod(query.call["shape"]) * element_bytes
            chunks = query.setup.get("exchange_chunks", 0)
            if (
                query.world_size not in (2, 4)
                or not 1 <= chunks <= 64
                or not 0 < nbytes <= chunks * 64 * 4
                or nbytes % 4
                or not query.call.get("disjoint_output", False)
                or not query.call.get("word_aligned", False)
                or query.call["threads"] not in (32, 64, 128, 256)
                or not 1 <= query.call["block_limit"] <= 64
                or query.setup.get("exchange_offset", -1) < 0
                or query.setup["exchange_offset"] % 256
                or query.setup["exchange_offset"] + query.world_size * chunks * 64 * 8
                > query.setup["slot_bytes"]
                or query.setup.get("exchange_counter_offset", 0)
                != 2 * query.setup["slot_bytes"]
            ):
                raise ValueError(
                    "word exchange requires disjoint aligned data and owned TP2/TP4 state"
                )
    if query.surface == "kimi_topk16":
        if query.topology != "local":
            raise ValueError("stateless kimi_topk16 uses one local device")
    elif query.topology != "pcie_ipc":
        raise ValueError(
            "PCIe collectives require caller-established CUDA IPC topology"
        )


TUNING = replace(TUNING, query_schema_version=42, validate_query=_validate_query)

__all__ = ["PcieQuery", "PcieConfig", "SURFACES", "TUNING"]
