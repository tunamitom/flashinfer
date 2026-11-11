"""B12X_W4A16_PREFILL_TABLE=1: direct decode table for large-M Trellis prefill blocks.

Policy table, decode-size gate, and the kernel geometry it selects (constructed, not compiled).
"""
from __future__ import annotations

import pytest

from b12x.moe.fused_moe._prefill_shapes import (
    trellis_prefill_launch_shape,
    trellis_prefill_table_shape,
)


@pytest.mark.parametrize("shapes", ["0", "1"])
@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 17, 32, 128, 1024, 1536, 2048, 2560, 3072])
def test_off_by_default(monkeypatch, shapes, m):
    monkeypatch.delenv("B12X_W4A16_PREFILL_TABLE", raising=False)
    monkeypatch.setenv("B12X_W4A16_PREFILL_SHAPES", shapes)
    assert trellis_prefill_table_shape(m) is None
    shape = trellis_prefill_launch_shape(m)
    assert shape is None or shape["trellis_decode_table"] != "full1"


@pytest.mark.parametrize("mode", ["1"])
@pytest.mark.parametrize("shapes", ["0", "1"])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8, 12, 16, 17, 32, 64, 127, 128, 512, 1024, 1535])
def test_small_sizes_untouched_when_on(monkeypatch, mode, shapes, m):
    monkeypatch.setenv("B12X_W4A16_PREFILL_TABLE", mode)
    monkeypatch.setenv("B12X_W4A16_PREFILL_SHAPES", shapes)
    assert trellis_prefill_table_shape(m) is None
    monkeypatch.setenv("B12X_W4A16_PREFILL_TABLE", "0")
    reference = trellis_prefill_launch_shape(m)
    monkeypatch.setenv("B12X_W4A16_PREFILL_TABLE", mode)
    assert trellis_prefill_launch_shape(m) == reference


_R48 = (48, 2, "full1", (128, 128, 64, 256))


@pytest.mark.parametrize(
    "mode,m,expected",
    [
        ("1", 1536, _R48), ("1", 2047, _R48), ("1", 2048, _R48), ("1", 2559, _R48), ("1", 2560, _R48),
        ("1", 2561, None), ("1", 3072, None), ("1", 8192, None),
        ("2", 1536, None), ("2", 2048, None), ("2", 3072, None),
    ],
)
def test_policy_table(monkeypatch, mode, m, expected):
    monkeypatch.setenv("B12X_W4A16_PREFILL_TABLE", mode)
    assert trellis_prefill_table_shape(m) == (
        None if expected is None else dict(zip(
            ("block_size_m", "pipeline_stages", "trellis_decode_table", "tile_config"), expected))
    )
    for shapes in ("0", "1"):
        monkeypatch.setenv("B12X_W4A16_PREFILL_SHAPES", shapes)
        shape = trellis_prefill_launch_shape(m)
        if expected is not None:
            assert (
                shape["block_size_m"],
                shape["pipeline_stages"],
                shape["trellis_decode_table"],
                shape["tile_config"],
            ) == expected
        else:
            assert shape is None or shape["trellis_decode_table"] != "full1"


def _kimi_fused(size_m, block, stages, table, tile):
    from b12x.moe._shared.kernels.w4a16.kernel.fused import W4A16FusedMoeKernel

    fk, fn, sk, sn = tile
    return W4A16FusedMoeKernel(
        size_m=size_m, hidden_size=3584, intermediate_size=384, num_experts=896, top_k=16,
        activation="situ", apply_router_weight_on_input=False, zero_fc2_output=False,
        fc1_tile_n=fn, fc1_tile_k=fk, fc2_tile_n=sn, fc2_tile_k=sk,
        moe_block_size=block, max_m_blocks=size_m * 16 // block + 896, element_dtype="fp16",
        weight_layout="trellis_t256", scale_format="e4m3_k32", w13_layout="trellis_t256_proj",
        trellis_bits=2, trellis_codebook="lut_e4m3", intermediate_rotation=True, full_rotation=True,
        intermediate_hadamard=True, rotation_input_dtype="bf16", broadcast_suh=True,
        pipeline_stages=stages, trellis_decode_table=table,
    )


@pytest.mark.parametrize("mode,m", [("1", 1536), ("1", 2048), ("1", 2560)])
def test_policy_geometry_holds_the_direct_table(monkeypatch, mode, m):
    monkeypatch.setenv("B12X_W4A16_PREFILL_TABLE", mode)
    shape = trellis_prefill_table_shape(m)
    kernel = _kimi_fused(m, shape["block_size_m"], shape["pipeline_stages"], "full1", shape["tile_config"])
    assert kernel.trellis_direct_lut
    assert kernel.blocks_per_sm == kernel.fc1.blocks_per_sm == kernel.fc2.blocks_per_sm == 1
    assert kernel.fc1.reduction_over_stages and kernel.fc2.reduction_over_stages
    assert kernel.lut_e4m3_smem_off + 65536 <= kernel.fc1.max_shared_mem
    # The same geometry is rejected without the prefill layout (why the switch exists).
    with pytest.raises(ValueError, match="decode table"):
        _kimi_fused(m, shape["block_size_m"], shape["pipeline_stages"], "full", shape["tile_config"])


@pytest.mark.parametrize("m", [1, 8, 16])
def test_full1_refused_at_decode_sizes(m):
    with pytest.raises(ValueError, match="prefill-only"):
        _kimi_fused(m, 16, 2, "full1", (128, 128, 64, 256))


@pytest.mark.parametrize("m,block", [(1, 8), (8, 8), (16, 16)])
@pytest.mark.parametrize("table", ["auto", "compact"])
def test_decode_cache_keys_have_no_prefill_layout(m, block, table):
    kernel = _kimi_fused(m, block, None, table, (128, 128, 64, 256))
    assert not kernel.fc1.reduction_over_stages and not kernel.fc2.reduction_over_stages
    assert "reduction_over_stages" not in kernel.fc1.__cache_key__
    assert "reduction_over_stages" not in kernel.fc2.__cache_key__
