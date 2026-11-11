"""B12X_W4A16_PREFILL_SHAPES policy boundaries: CPU-only."""
from __future__ import annotations

import pytest

from b12x.moe.fused_moe._prefill_shapes import (
    trellis_prefill_launch_shape,
    trellis_prefill_native_tile,
)


@pytest.mark.parametrize("m", [1, 2, 4, 8, 10, 16, 32, 64, 127, 128, 512, 1024, 2048, 3072])
def test_off_by_default(monkeypatch, m):
    monkeypatch.delenv("B12X_W4A16_PREFILL_SHAPES", raising=False)
    monkeypatch.delenv("B12X_W4A16_PREFILL_TABLE", raising=False)
    monkeypatch.setenv("B12X_W4A16_PREFILL_SHAPES_TILE", "native")
    assert trellis_prefill_launch_shape(m) is None
    assert not trellis_prefill_native_tile(m)


@pytest.mark.parametrize("m", [1, 2, 4, 8, 10, 16, 32, 64, 127])
def test_decode_sizes_untouched_when_on(monkeypatch, m):
    monkeypatch.setenv("B12X_W4A16_PREFILL_SHAPES", "1")
    monkeypatch.delenv("B12X_W4A16_PREFILL_TABLE", raising=False)
    monkeypatch.setenv("B12X_W4A16_PREFILL_SHAPES_TILE", "native")
    assert trellis_prefill_launch_shape(m) is None
    assert not trellis_prefill_native_tile(m)


@pytest.mark.parametrize(
    "m,expected",
    [
        (128, (16, 3, "full")),
        (512, (16, 3, "full")),
        (1023, (16, 3, "full")),
        (1024, (32, 2, "full")),
        (1536, (32, 2, "full")),
        (2047, (32, 2, "full")),
        (2048, None),
        (2559, None),
        (2560, (64, 3, "compact")),
        (3072, (64, 3, "compact")),
    ],
)
def test_policy_table(monkeypatch, m, expected):
    monkeypatch.setenv("B12X_W4A16_PREFILL_SHAPES", "1")
    monkeypatch.delenv("B12X_W4A16_PREFILL_TABLE", raising=False)
    monkeypatch.delenv("B12X_W4A16_PREFILL_SHAPES_TILE", raising=False)
    shape = trellis_prefill_launch_shape(m)
    if expected is None:
        assert shape is None
    else:
        assert (shape["block_size_m"], shape["pipeline_stages"], shape["trellis_decode_table"]) == expected
    assert not trellis_prefill_native_tile(m)
