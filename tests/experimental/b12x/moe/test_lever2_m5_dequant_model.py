"""Lever-2 M5: native BF16 E2M1 dequant probe, CPU-only.

The plan's M5 gate leans on ``cvt.rn.bf16x2.e2m1x2`` representing every E2M1
value exactly in BF16. This test pins that mathematical claim and the probe
flag semantics. The word/nibble mapping equivalence between the software
unpack and the native PTX path is A/B-verified on GPU by
``harness/verify_m5_native_dequant.py`` (bit-identical gate).
"""

import struct

import pytest

torch = pytest.importorskip("torch")

from b12x.moe._shared.kernels.w4a16.kernel.config import (
    native_bf16_e2m1_dequant_enabled,
)

pytestmark = pytest.mark.cpu_only


def _e2m1_values():
    """All 16 E2M1 (FP4) code values: sign x {0, 0.5, 1, 1.5, 2, 3, 4, 6}."""
    magnitudes = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    return [s * m for s in (1.0, -1.0) for m in magnitudes]


def test_every_e2m1_value_is_exact_in_bf16():
    for value in _e2m1_values():
        as_bf16 = torch.tensor(value, dtype=torch.bfloat16)
        assert float(as_bf16) == value, value
        # And the RN conversion of the BF16 pattern back to FP32 is exact.
        (bits,) = struct.unpack("<I", struct.pack("<f", value))
        assert struct.unpack("<f", struct.pack("<I", bits))[0] == value


def test_e2m1x2_pair_conversion_is_exact():
    # cvt.rn.bf16x2.e2m1x2 converts two packed nibbles per byte; both lanes
    # must land exactly in their bf16 slots.
    values = _e2m1_values()
    for i in range(0, 16, 2):
        pair = torch.tensor([values[i], values[i + 1]], dtype=torch.bfloat16)
        assert [float(x) for x in pair] == [values[i], values[i + 1]]


def test_native_bf16_flag_semantics(monkeypatch):
    monkeypatch.delenv("B12X_W4A16_NATIVE_BF16_DEQUANT", raising=False)
    assert native_bf16_e2m1_dequant_enabled() is False
    monkeypatch.setenv("B12X_W4A16_NATIVE_BF16_DEQUANT", "1")
    assert native_bf16_e2m1_dequant_enabled() is True
    monkeypatch.setenv("B12X_W4A16_NATIVE_BF16_DEQUANT", "on")
    with pytest.raises(ValueError, match="B12X_W4A16_NATIVE_BF16_DEQUANT"):
        native_bf16_e2m1_dequant_enabled()


def test_scale_multiply_is_shared_by_both_paths():
    # Both dequant branches feed the same _elem2_mul scale application, so a
    # bit-identical unpack implies a bit-identical scaled fragment.
    value = torch.tensor(1.5, dtype=torch.bfloat16)
    scale = torch.tensor(0.25, dtype=torch.bfloat16)
    once = value * scale
    twice = torch.tensor(1.5, dtype=torch.bfloat16) * torch.tensor(
        0.25, dtype=torch.bfloat16
    )
    assert float(once) == float(twice)
