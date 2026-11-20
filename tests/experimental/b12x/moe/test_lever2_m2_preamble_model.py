"""Lever-2 M2: cooperative-preamble decision surface + index model, CPU-only.

The preamble kernel body is not authored this wave (entry gate pending M1's
measurement; Principle 14 specialist authoring). These tests pin the fail-closed
flag semantics and verify the pure work-unit/dedup model the body must
implement: identical expansion set to the legacy two-launch presence path.
"""

import random

import pytest

torch = pytest.importorskip("torch")

from b12x.moe._shared.kernels.w4a16.kernel.csf_preamble import (
    csf_preamble_enabled,
    presence_path_expanded_experts,
    preamble_expanded_experts,
    preamble_unit_assignment,
    preamble_work_units,
    require_preamble_support,
)

pytestmark = pytest.mark.cpu_only


def test_flag_is_fail_closed(monkeypatch):
    monkeypatch.delenv("B12X_W4A16_CSF_PREAMBLE", raising=False)
    assert csf_preamble_enabled() is False
    monkeypatch.setenv("B12X_W4A16_CSF_PREAMBLE", "1")
    assert csf_preamble_enabled() is True
    monkeypatch.setenv("B12X_W4A16_CSF_PREAMBLE", "yes")
    with pytest.raises(ValueError, match="B12X_W4A16_CSF_PREAMBLE"):
        csf_preamble_enabled()


def test_capability_guard_rejects_plain_launch():
    class PlainLaunch:
        pass

    with pytest.raises(RuntimeError, match="csf_preamble_geometry"):
        require_preamble_support(PlainLaunch())

    class PreambleLaunch:
        csf_preamble_geometry = (64, 8, 512, 64)

    require_preamble_support(PreambleLaunch())


def test_work_units_and_assignment():
    # Hy4 cap 8: 64 routes, 48 W13 + 24 W2 tasks, packed.
    assert preamble_work_units(64, 48, 24, packed=True) == 64 * 72 + 1
    assert preamble_work_units(64, 48, 24, packed=False) == 64 * 72
    assert preamble_unit_assignment(0, 64, 48, 24) == (0, 0, 0)
    assert preamble_unit_assignment(48, 64, 48, 24) == (0, 0, 1)
    assert preamble_unit_assignment(72, 64, 48, 24) == (1, 0, 0)
    assert preamble_unit_assignment(64 * 72, 64, 48, 24) == ("pack",)
    with pytest.raises(ValueError):
        preamble_unit_assignment(64 * 72 + 1, 64, 48, 24)
    with pytest.raises(ValueError):
        preamble_work_units(0, 48, 24, packed=False)


@pytest.mark.parametrize("live", [8, 64, 128])
def test_dedup_model_matches_presence_path(live):
    rng = random.Random(live)
    # Duplicate-heavy routes plus invalid entries.
    ids = [rng.randrange(-2, 16) for _ in range(live)]
    direct = preamble_expanded_experts(ids, 256)
    presence = presence_path_expanded_experts(ids, 256)
    assert set(direct) == set(presence)
    # First-occurrence expansion: the recorded slot is the lowest route slot.
    for e, slot in direct.items():
        assert ids[slot] == e
        assert all(i >= slot or ids[i] != e for i in range(slot))


def test_dedup_model_expands_each_expert_once():
    ids = [3, 3, 3, -1, 7, 7, 256, 7]
    expanded = preamble_expanded_experts(ids, 256)
    assert expanded == {3: 0, 7: 4}
