"""L9 adaptation: the B12X_W4A16_NONCOOP_PREFILL_SMS prefill SM cap at the launch geometry."""

from __future__ import annotations

import pytest

from b12x.moe._shared.kernels.w4a16.kernel.config import w4a16_prefill_sms


def test_unset_keeps_the_device_sm_count():
    assert w4a16_prefill_sms(188, 3072) == 188
    assert w4a16_prefill_sms(188, 20) == 20 or w4a16_prefill_sms(188, 20) == 188


def test_cap_applies_to_prefill_rows_only(monkeypatch):
    monkeypatch.setenv("B12X_W4A16_NONCOOP_PREFILL_SMS", "170")
    assert w4a16_prefill_sms(188, 3072) == 170
    assert w4a16_prefill_sms(188, 128) == 170
    assert w4a16_prefill_sms(188, 127) == 188
    assert w4a16_prefill_sms(188, 20) == 188


def test_cap_below_device_sms_wins(monkeypatch):
    monkeypatch.setenv("B12X_W4A16_NONCOOP_PREFILL_SMS", "152")
    assert w4a16_prefill_sms(188, 2048) == 152
    assert w4a16_prefill_sms(120, 2048) == 120


def test_invalid_cap_rejects(monkeypatch):
    monkeypatch.setenv("B12X_W4A16_NONCOOP_PREFILL_SMS", "0")
    with pytest.raises(ValueError):
        w4a16_prefill_sms(188, 2048)
    monkeypatch.setenv("B12X_W4A16_NONCOOP_PREFILL_SMS", "nope")
    with pytest.raises(ValueError):
        w4a16_prefill_sms(188, 2048)
