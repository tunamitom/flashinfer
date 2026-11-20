"""Lever-2 M0: SMALL_M_OCCUPANCY probe hook resolves before env, CPU-only.

The probe override exists so the M0 harness can A/B
``B12X_W4A16_SMALL_M_OCCUPANCY=2`` without editing recipes; the occupancy
selection feeds pipeline-stage and blocks-per-sm decisions at plan/compile
time, so the hook must resolve deterministically and validate strictly.
"""

import pytest

torch = pytest.importorskip("torch")

from b12x.moe._shared.kernels.w4a16.kernel.config import (
    _w4a16_small_m_double_occupancy,
    _w4a16_small_m_occupancy,
    _w4a16_pipeline_stages,
    small_m_occupancy_probe,
)

pytestmark = pytest.mark.cpu_only

_PACKED = dict(weight_layout="packed", scale_format="e4m3_k16", uses_m_block_8=True)


def test_probe_overrides_env_and_restores(monkeypatch):
    monkeypatch.setenv("B12X_W4A16_SMALL_M_OCCUPANCY", "1")
    assert _w4a16_small_m_occupancy() == 1
    previous = small_m_occupancy_probe(2)
    assert previous is None
    try:
        assert _w4a16_small_m_occupancy() == 2
    finally:
        assert small_m_occupancy_probe(previous) == 2
    assert _w4a16_small_m_occupancy() == 1


def test_probe_release_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("B12X_W4A16_SMALL_M_OCCUPANCY", "2")
    small_m_occupancy_probe(1)
    try:
        assert _w4a16_small_m_occupancy() == 1
    finally:
        small_m_occupancy_probe(None)
    assert _w4a16_small_m_occupancy() == 2


def test_probe_validates_selection():
    with pytest.raises(ValueError, match="1, 2, or None"):
        small_m_occupancy_probe(3)
    with pytest.raises(TypeError):
        small_m_occupancy_probe("2")
    assert _w4a16_small_m_occupancy() in (1, 2)


def test_probe_drives_double_occupancy_and_stages(monkeypatch):
    monkeypatch.setenv("B12X_W4A16_SMALL_M_OCCUPANCY", "1")
    small_m_occupancy_probe(2)
    try:
        assert _w4a16_small_m_double_occupancy(**_PACKED)
        assert (
            _w4a16_pipeline_stages(
                tile_n=128,
                tile_k=64,
                small_m_occupancy=_w4a16_small_m_occupancy(),
                **{k: v for k, v in _PACKED.items()},
            )
            == 3
        )
    finally:
        small_m_occupancy_probe(None)
    assert not _w4a16_small_m_double_occupancy(**_PACKED)
    assert (
        _w4a16_pipeline_stages(
            tile_n=256,
            tile_k=64,
            small_m_occupancy=_w4a16_small_m_occupancy(),
            **{k: v for k, v in _PACKED.items()},
        )
        == 4
    )
