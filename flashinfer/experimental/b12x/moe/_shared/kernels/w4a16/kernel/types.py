"""W4A16 launch and prepared-weight records."""

from __future__ import annotations

from b12x.moe._shared.trellis_codebooks import LUT_E4M3
from typing import NamedTuple
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class W4A16GemmCompileResult:
    compiled: object
    tile_n: int
    tile_k: int
    moe_block_size: int
    max_m_blocks: int
    blocks_per_sm: int
    weight_layout: str = "packed"
    scale_format: str = "e4m3_k16"
    w13_layout: str = "w13"
    dense_route_fast_path: bool = False
    trellis_bits: int = 3
    trellis_codebook: str = LUT_E4M3
    trellis_pair_kind: str | None = None
    trellis_rate_axis: str | None = None


@dataclass(frozen=True)
class W4A16ActivationCompileResult:
    compiled: object
    rows: int
    intermediate_size: int
    activation: str
    swiglu_limit: float | None
    swiglu_clamp_after_silu: bool
    swiglu_alpha: float
    swiglu_beta: float


@dataclass(frozen=True)
class W4A16TopKSumCompileResult:
    compiled: object
    m: int
    topk: int
    hidden_size: int
    full_rotation: bool = False
    intermediate_hadamard: bool = False
    num_experts: int = 0
    route_num_experts: int = 0
    route_ids_dtype: torch.dtype = torch.int32
    use_expert_map: bool = False
    broadcast_svh: bool = False
    apply_topk_weights: bool = False
    output_dtype: torch.dtype | None = None


# B12X_TRELLIS_FLOW_FLAGS: kernel_workspace extension after the 4*sms+2
# barrier/lock words: [4 reserved, fc1_ctr[MAX_BLOCKS], act_ctr[MAX_BLOCKS],
# spare], leaving the last two words for the small-M direct barrier.  The
# kernel zeroes it at entry (no state carried between launches).
W4A16_FLOW_MAX_BLOCKS = 256
# The counters replace the barriers only up to this many route blocks: with
# more blocks (M >= 12 over the full expert set) every warp owns several
# activation units and waiting per unit convoys behind slow blocks
# (+1..+3.5 us measured), so larger launches keep the barriers at run time.
W4A16_FLOW_LIVE_BLOCKS = 64
W4A16_FLOW_WORKSPACE_EXTRA = 4 + 4 * W4A16_FLOW_MAX_BLOCKS + 4


def w4a16_flow_flags_requested() -> bool:
    import os

    return os.environ.get("B12X_TRELLIS_FLOW_FLAGS", "0") == "1"


@dataclass(frozen=True)
class W4A16FusedMoeCompileResult:
    compiled: object
    size_m: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    activation: str
    apply_router_weight_on_input: bool
    zero_fc2_output: bool
    element_dtype: str
    fast_math: bool
    swiglu_limit: float | None
    swiglu_clamp_after_silu: bool
    swiglu_alpha: float
    swiglu_beta: float
    fc1_tile_n: int
    fc1_tile_k: int
    fc2_tile_n: int
    fc2_tile_k: int
    moe_block_size: int
    max_m_blocks: int
    blocks_per_sm: int
    weight_layout: str = "packed"
    w13_layout: str = "w13"
    direct_topk_routes: bool = False
    use_expert_map: bool = False
    scale_format: str = "e4m3_k16"
    tc_decode_fused_sum: bool = False
    prefill_fused_sum_fp32: bool = False
    collect_activation_amax: bool = False
    schedule_whole_tiles: bool = False
    intermediate_rotation: bool = False
    dual_a: bool = False
    trellis_bits: int = 3
    trellis_codebook: str = LUT_E4M3
    fc1_trellis_pair_kind: str | None = None
    fc2_trellis_pair_kind: str | None = None
    full_rotation: bool = False
    intermediate_hadamard: bool = False
    rotation_input_dtype: str = "fp16"
    cta_threads: int = -1
    shared_memory_bytes: int = -1
    broadcast_suh: bool = False
    small_m_direct_launches: tuple[_W4A16SmallMDirectLaunch, ...] = ()
    trellis_direct_lut: bool = False
    # B12X_TRELLIS_INKERNEL_ROUTE_PACK: the kernel can build the packed route
    # layout itself when the host passes the top-k ids (route_ids).
    inkernel_route_pack: bool = False
    route_pack_max_routes: int = 128
    flow_flags: bool = False


@dataclass(frozen=True)
class _W4A16GemmLaunch:
    kernel: W4A16GemmCompileResult
    c_tmp: torch.Tensor


class _W4A16SmallMDirectLaunch(NamedTuple):
    compiled: object
    grid_x: int
    m: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    topk: int
    activation: str
    fast_math: bool
    topk_ids_dtype: torch.dtype


@dataclass(frozen=True)
class W4A16SmallMDirectCompileResult:
    moe_block_size: int
    small_m_direct_launches: tuple[_W4A16SmallMDirectLaunch, ...]


class _W4A16FC2DirectLaunch(NamedTuple):
    compiled: object
    grid_x: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    topk_ids_dtype: torch.dtype
