"""Opt-in Trellis prefill launch shapes for the full-rotation W4A16 fused MoE.

Kept outside moe/_shared/kernels so that no CuTe kernel module's source closure (compile-cache fingerprint)
includes it: with the switch off, every W4A16 kernel compile key is unchanged by this module.
"""

from __future__ import annotations

import os


# Trellis prefill launch shapes. Default OFF. Selected only when
# B12X_W4A16_PREFILL_SHAPES=1 (read at plan/compile time) and only for launch
# token counts M >= 128, so every decode / verify / draft launch (exact-M
# plans for capacities <= 32) keeps the shipped heuristic by construction.
#   M >= 2560:       3 pipeline stages, 64-row route blocks, compact table
#   2048 <= M < 2560: untouched (all variants lost in their sweep)
#   1024 <= M < 2048: 32-row blocks, 2 stages, direct ("full") decode table
#   128 <= M < 1024:  16-row blocks, 3 stages, direct ("full") decode table
# The launch plan consumes the MoeDecodeConfig.w4a16_* planner knobs (wired
# at the trellis compile sites by the previous commit); with this switch off
# and the knobs unset, the shipped heuristic plan is unchanged.
# B12X_W4A16_PREFILL_SHAPES_TILE=native also drops a forced
# B12X_TRELLIS_TILE_CONFIG for these (M >= 128) launches.
def trellis_prefill_launch_shape(m: int) -> dict | None:
    """Measured prefill launch shape for the Trellis fused kernel, or None."""
    table_shape = trellis_prefill_table_shape(m)
    if table_shape is not None:
        return table_shape
    if os.getenv("B12X_W4A16_PREFILL_SHAPES", "0") != "1":
        return None
    m = int(m)
    if m < 128:
        return None
    if m >= 2560:
        return {"block_size_m": 64, "pipeline_stages": 3, "trellis_decode_table": "compact"}
    if m >= 2048:
        return None
    if m >= 1024:
        return {"block_size_m": 32, "pipeline_stages": 2, "trellis_decode_table": "full"}
    return {"block_size_m": 16, "pipeline_stages": 3, "trellis_decode_table": "full"}


def trellis_prefill_native_tile(m: int) -> bool:
    """True when a prefill-shape launch should ignore the forced env tile."""
    return (
        trellis_prefill_launch_shape(m) is not None
        and os.getenv("B12X_W4A16_PREFILL_SHAPES_TILE", "") == "native"
    )



# Direct-table prefill geometry (B12X_W4A16_PREFILL_TABLE=1, default OFF).
# Consulted only for launch token counts M > 16 (it returns None below 1536),
# so decode / verify / draft plans never see it. Takes precedence over the
# prefill-shapes policy where it applies; meant to be used together with
# B12X_W4A16_PREFILL_SHAPES=1 (which keeps owning the other sizes).
# "full1" = the 64 KiB direct decode table at one resident CTA per SM, with
# the GEMM output/reduction tile laid over the dead B/S/A stages (fused.py,
# gemm.py reduction_over_stages) so the stages fit beside the table.
# Measured on a Kimi K3.1 TP8 shard (us/layer, vs prefill shapes):
#   1536 <= M <= 2560: 48-row blocks, 2 stages, tile 128,128,64,256
#       (2560: 3872 vs 4183; 2048: 3045 vs 3752; 1536: 2742 vs 2880). Same
#       numerics class as that tile (2048: bit-identical to the shapes
#       policy in 99.99 % of outputs).
#   Other M: None (the prefill-shapes policy decides).
def trellis_prefill_table_shape(m: int) -> dict | None:
    """Direct-table prefill launch shape (block, stages, table, tile), or None."""
    mode = os.getenv("B12X_W4A16_PREFILL_TABLE", "0")
    if mode != "1":
        return None
    m = int(m)
    if m <= 16:
        return None
    if 1536 <= m <= 2560:
        return {
            "block_size_m": 48,
            "pipeline_stages": 2,
            "trellis_decode_table": "full1",
            "tile_config": (128, 128, 64, 256),
        }
    return None
