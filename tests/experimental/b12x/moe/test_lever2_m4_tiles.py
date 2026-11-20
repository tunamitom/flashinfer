"""Lever-2 M4: prefill tile/stage pin hooks + occupancy selection, CPU-only.

Encodes the plan's B3 diagnosis as regression assertions: the untuned large-M
selection really is (64, 256, 256) at 1 CTA/SM with 255 regs/thread, and the
M4 sweep pins reach the (64, 128, 128) @ stages=3 -> 2 CTAs/SM candidate and
the moe_block_size=32 block without editing code.
"""

import pytest

torch = pytest.importorskip("torch")

from b12x.moe._shared.kernels.w4a16.kernel.config import (
    _LARGE_BATCH_TILE_CONFIGS,
    _select_tile_config,
    _shared_memory_footprint,
    _w4a16_num_regs,
    _w4a16_pipeline_stages,
    _w4a16_pipeline_stages_pin,
    _w4a16_prefill_tile_pin,
)

pytestmark = pytest.mark.cpu_only

HY4_PREFILL = dict(
    problem_m=2048,
    problem_n=512,
    problem_k=6144,
    top_k=8,
    sms=188,
    max_shared_mem=101_376,
    scale_format="e4m3_k16",
    weight_layout="packed",
)


def test_pin_parsing(monkeypatch):
    assert _w4a16_prefill_tile_pin() is None
    assert _w4a16_pipeline_stages_pin() is None
    monkeypatch.setenv("B12X_W4A16_PREFILL_TILE", "64,128,128")
    monkeypatch.setenv("B12X_W4A16_PIPELINE_STAGES_PIN", "3")
    assert _w4a16_prefill_tile_pin() == (64, 128, 128)
    assert _w4a16_pipeline_stages_pin() == 3
    monkeypatch.setenv("B12X_W4A16_PREFILL_TILE", "64,128")
    with pytest.raises(ValueError, match="B12X_W4A16_PREFILL_TILE"):
        _w4a16_prefill_tile_pin()
    monkeypatch.setenv("B12X_W4A16_PREFILL_TILE", "1,2,3")
    with pytest.raises(ValueError, match="B12X_W4A16_PREFILL_TILE"):
        _w4a16_prefill_tile_pin()
    monkeypatch.setenv("B12X_W4A16_PIPELINE_STAGES_PIN", "7")
    with pytest.raises(ValueError, match="B12X_W4A16_PIPELINE_STAGES_PIN"):
        _w4a16_pipeline_stages_pin()


def test_untuned_large_m_is_the_b3_one_cta_cliff():
    # Plan 1.2 B3: all candidates tie at 1 CTA/SM and the selector picks
    # (64, 256, 256) at the architectural 255-register limit.
    assert _select_tile_config(moe_block_size=64, **HY4_PREFILL) == (64, 256, 256, 1)
    regs = _w4a16_num_regs(
        cta_threads=256,
        cta_m_blocks=4,
        cta_n_blocks=16,
        cta_k_blocks=4,
        uses_m_block_8=False,
    )
    assert regs == 255
    smem = _shared_memory_footprint(
        cta_m_blocks=4, tile_n=256, tile_k=64, weight_layout="packed"
    )
    assert smem == 75_776


def test_moe_block_size_32_reaches_two_ctas_per_sm():
    # Plan 1.2 B3: moe_block_size=32 falls to (64, 128, 128) with 212 regs.
    assert _select_tile_config(moe_block_size=32, **HY4_PREFILL) == (64, 128, 128, 2)
    regs = _w4a16_num_regs(
        cta_threads=256,
        cta_m_blocks=2,
        cta_n_blocks=16,
        cta_k_blocks=4,
        uses_m_block_8=False,
    )
    assert regs == 212


def test_tile_and_stage_pins_unlock_two_ctas_per_sm(monkeypatch):
    monkeypatch.setenv("B12X_W4A16_PREFILL_TILE", "64,128,128")
    monkeypatch.setenv("B12X_W4A16_PIPELINE_STAGES_PIN", "3")
    # (64, 128, 128) @ stages=3 fits two resident CTAs (46080 B SMEM).
    assert _select_tile_config(moe_block_size=64, **HY4_PREFILL) == (64, 128, 128, 2)
    assert (
        _shared_memory_footprint(
            cta_m_blocks=4,
            tile_n=128,
            tile_k=64,
            weight_layout="packed",
            pipeline_stages=_w4a16_pipeline_stages(
                weight_layout="packed", tile_n=128, tile_k=64, uses_m_block_8=False
            ),
        )
        == 46_080
    )


def test_pin_without_stage_change_stays_one_cta(monkeypatch):
    monkeypatch.setenv("B12X_W4A16_PREFILL_TILE", "64,128,128")
    # Four stages push the tile back to 1 CTA/SM: the stage pin matters.
    assert _select_tile_config(moe_block_size=64, **HY4_PREFILL) == (64, 128, 128, 1)


def test_explicit_stages_argument_beats_pin(monkeypatch):
    monkeypatch.setenv("B12X_W4A16_PIPELINE_STAGES_PIN", "3")
    assert (
        _w4a16_pipeline_stages(
            weight_layout="packed", tile_n=256, tile_k=64, uses_m_block_8=False,
            pipeline_stages=4,
        )
        == 4
    )
    assert (
        _w4a16_pipeline_stages(
            weight_layout="packed", tile_n=256, tile_k=64, uses_m_block_8=False
        )
        == 3
    )


def test_pinned_tile_must_fit_or_raise(monkeypatch):
    monkeypatch.setenv("B12X_W4A16_PREFILL_TILE", "128,64,128")
    # (128, 64, 128) needs K divisible by 128; 6144 qualifies, so it fits.
    assert _select_tile_config(moe_block_size=64, **HY4_PREFILL)[0] == 128
    monkeypatch.setenv("B12X_W4A16_PREFILL_TILE", "64,256,256")
    assert _select_tile_config(moe_block_size=64, **HY4_PREFILL) == (64, 256, 256, 1)
    assert all(pin in _LARGE_BATCH_TILE_CONFIGS for pin in ((64, 256, 256),))
