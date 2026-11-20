"""N7 frozen replay through PreparationSession, and the flag-off no-op.

The serve-time contract: with ``B12X_TUNE_CACHE_MODE=frozen`` a boot resolves
every tunable obligation from the persisted offline-tune cache -- zero
candidate enumeration, zero racing, zero candidate compilation -- and a stale
or missing cache fails the boot loudly. With the flag off (or the mode unset)
nothing about heuristic preparation changes.
"""

import json
from dataclasses import replace

import pytest

from b12x.preparation import (
    DetectedDevice,
    FrozenMapping,
    MemoryRequirements,
    Plan,
    PreparedCall,
    TuningContract,
)
from b12x.preparation import _tune_cache as tc
from b12x.preparation.session import PreparationSession, _Obligation
from b12x.preparation.types import DeviceIdentity

from .test_defaults import Config, Query, contract

IDENTITY = DeviceIdentity("nvidia", (12, 0), 148, "synthetic SM120")
OTHER_ARCH = DeviceIdentity("nvidia", (12, 0), 132, "synthetic SM120")


def session(tmp_path, **kwargs):
    """CPU session on a synthetic SM120 identity with isolated caches."""
    value = PreparationSession(
        device=DetectedDevice(None, IDENTITY),
        cache_dir=kwargs.pop("cache_dir", tmp_path / "selections"),
        **kwargs,
    )
    return value


def request(*, name, tuning, pin=None, dependencies=()):
    plan = Plan(
        contract=tuning,
        query=Query(3),
        override=pin,
        _compile_jobs=lambda config, device: (),
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=lambda selection, device: object(),
    )
    return plan.request(
        name=name,
        prepare_call=lambda state: PreparedCall(run=lambda: None),
        dependencies=dependencies,
    )


def seed(tmp_path, tuning, *, width=4, arch=IDENTITY, query=None, fingerprint=None):
    """Write one tuned record exactly as an offline pass would."""
    cache = tc.TuneCache(tmp_path, arch=tc.arch_identity(arch))
    cache.record(
        tc.build_provenance(
            tuning,
            encoded_query=tuning.configure(
                Query(3) if query is None else query, device=IDENTITY, override=None
            ).encoded_query,
            invocation={},
            pin=None,
            dependencies=(),
            device_identity=arch,
            fingerprint=fingerprint,
        ),
        {"width": width},
        assignment={"width": width},
    )
    cache.flush()
    return cache


def forbidden(*_args, **_kwargs):
    raise AssertionError("frozen replay entered autotuning infrastructure")


def test_flag_off_default_writes_and_reads_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("B12X_TUNE_CACHE_MODE", raising=False)
    monkeypatch.delenv("B12X_TUNE_CACHE_DIR", raising=False)
    item = request(name="plain", tuning=contract(default=7, values=(1, 2, 4)))
    engine = session(tmp_path, autotune=False, compile_workers=0)
    with engine:
        engine.prepare((item,))
        assert item.plan.selection.config == Config(7)
        assert item.plan.selection.source == "default"
        assert engine._tune_cache is None
        assert engine._tune_audit == []
    assert not (tmp_path / "tune").exists()
    assert not any(tmp_path.glob("b12x-tune-*"))


def test_frozen_hit_replays_the_tuned_winner_without_enumeration(
    tmp_path, monkeypatch
):
    root = tmp_path / "tune"
    seed(root, contract(default=7, values=(1, 2, 4)), width=4)
    monkeypatch.setenv("B12X_TUNE_CACHE_MODE", "frozen")
    monkeypatch.setenv("B12X_TUNE_CACHE_DIR", str(root))
    monkeypatch.setattr(TuningContract, "iterate", forbidden)
    item = request(name="replayed", tuning=contract(default=7, values=(1, 2, 4)))
    engine = session(tmp_path, autotune=False, compile_workers=0)
    engine._compiler = forbidden
    with engine:
        engine.prepare((item,))
        assert item.plan.selection.source == "frozen"
        assert item.plan.selection.config == Config(4)
        assert item.plan.selection.assignment == FrozenMapping({"width": 4})
    manifest = json.loads(
        next(root.glob("b12x-tune-audit-*.json")).read_text()
    )
    assert [entry["source"] for entry in manifest["selections"]].count("frozen") == 1


def test_frozen_miss_fails_the_boot_loudly(tmp_path, monkeypatch):
    root = tmp_path / "tune"
    root.mkdir()
    monkeypatch.setenv("B12X_TUNE_CACHE_MODE", "frozen")
    monkeypatch.setenv("B12X_TUNE_CACHE_DIR", str(root))
    item = request(name="miss", tuning=contract(default=7, values=(1, 2, 4)))
    with session(tmp_path, autotune=False, compile_workers=0) as engine:
        with pytest.raises(tc.TuneCacheMissError, match="HY4_MOE_AUTOTUNE"):
            engine.prepare((item,))


def test_frozen_stale_kernel_version_fails_with_the_drifted_field(
    tmp_path, monkeypatch
):
    root = tmp_path / "tune"
    seed(root, contract(default=7, values=(1, 2, 4)), width=4)
    monkeypatch.setenv("B12X_TUNE_CACHE_MODE", "frozen")
    monkeypatch.setenv("B12X_TUNE_CACHE_DIR", str(root))
    drifted = replace(contract(default=7, values=(1, 2, 4)), semantic_version=2)
    item = request(name="drifted", tuning=drifted)
    with session(tmp_path, autotune=False, compile_workers=0) as engine:
        with pytest.raises(tc.TuneCacheStaleError) as info:
            engine.prepare((item,))
    names = [name for name, _want, _have in info.value.mismatches]
    assert "kernel.semantic" in names


def test_frozen_stale_shape_fails_with_the_query_drift(tmp_path, monkeypatch):
    root = tmp_path / "tune"
    seed(root, contract(default=7, values=(1, 2, 4)), width=4)
    monkeypatch.setenv("B12X_TUNE_CACHE_MODE", "frozen")
    monkeypatch.setenv("B12X_TUNE_CACHE_DIR", str(root))
    tuning = contract(default=7, values=(1, 2, 4))
    plan = Plan(
        contract=tuning,
        query=Query(5),
        _compile_jobs=lambda config, device: (),
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=lambda selection, device: object(),
    )
    item = plan.request(
        name="other-shape", prepare_call=lambda state: PreparedCall(run=lambda: None)
    )
    with session(tmp_path, autotune=False, compile_workers=0) as engine:
        with pytest.raises(tc.TuneCacheStaleError) as info:
            engine.prepare((item,))
    names = [name for name, _want, _have in info.value.mismatches]
    assert "shape.query" in names


def test_frozen_foreign_arch_fails_the_boot_loudly(tmp_path, monkeypatch):
    root = tmp_path / "tune"
    seed(root, contract(default=7, values=(1, 2, 4)), width=4, arch=OTHER_ARCH)
    monkeypatch.setenv("B12X_TUNE_CACHE_MODE", "frozen")
    monkeypatch.setenv("B12X_TUNE_CACHE_DIR", str(root))
    item = request(name="foreign", tuning=contract(default=7, values=(1, 2, 4)))
    with session(tmp_path, autotune=False, compile_workers=0) as engine:
        with pytest.raises(tc.TuneCacheStaleError, match="arch"):
            engine.prepare((item,))


def test_frozen_leaves_single_product_and_pins_on_their_paths(tmp_path, monkeypatch):
    root = tmp_path / "tune"
    root.mkdir()
    monkeypatch.setenv("B12X_TUNE_CACHE_MODE", "frozen")
    monkeypatch.setenv("B12X_TUNE_CACHE_DIR", str(root))
    monkeypatch.setattr(TuningContract, "iterate", forbidden)
    single = request(name="single", tuning=contract(default=2, values=(2,)))
    pinned = request(
        name="pinned", tuning=contract(default=7, values=(1, 2, 4)), pin=Config(9)
    )
    with session(tmp_path, autotune=False, compile_workers=0) as engine:
        engine.prepare((single, pinned))
        assert single.plan.selection.config == Config(2)
        assert pinned.plan.selection.config == Config(9)
        assert pinned.plan.selection.source == "override"


def test_record_mode_persists_tuned_winners_and_an_audit(tmp_path, monkeypatch):
    monkeypatch.delenv("B12X_TUNE_CACHE_MODE", raising=False)
    monkeypatch.delenv("B12X_TUNE_CACHE_DIR", raising=False)
    root = tmp_path / "tune"
    item = request(name="tuned", tuning=contract(default=7, values=(1, 2, 4)))
    engine = session(
        tmp_path,
        autotune=True,
        compile_workers=0,
        cache_dir=root,
    )
    with engine:
        job = engine.begin((item,))
        configuration = item.plan.contract.configure(
            item.plan.query, device=IDENTITY, override=None, search=False
        )
        obligation = _Obligation(item, configuration)
        job._selection(obligation, Config(7), "default")
        job._selection(
            obligation, Config(4), "tuned", FrozenMapping({"width": 4})
        )
        job.close()
    payload = json.loads(
        (root / tc.FILE_NAME).read_text()
    )
    records = payload["records"]
    assert len(records) == 1
    record = next(iter(records.values()))
    assert record["config"] == {"width": 4}
    assert record["extra"]["source"] == "tuned"
    manifest = json.loads(next(root.glob("b12x-tune-audit-*.json")).read_text())
    assert [entry["source"] for entry in manifest["selections"]] == [
        "default",
        "tuned",
    ]


def test_frozen_replay_accepts_a_record_written_by_record_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("B12X_TUNE_CACHE_MODE", raising=False)
    root = tmp_path / "tune"
    tuning = contract(default=7, values=(1, 2, 4))
    item = request(name="tuned", tuning=tuning)
    engine = session(
        tmp_path, autotune=True, compile_workers=0, cache_dir=root
    )
    with engine:
        job = engine.begin((item,))
        configuration = tuning.configure(
            item.plan.query, device=IDENTITY, override=None, search=False
        )
        job._selection(
            _Obligation(item, configuration), Config(3), "tuned", {"width": 3}
        )
        job.close()
    monkeypatch.setenv("B12X_TUNE_CACHE_MODE", "frozen")
    monkeypatch.setenv("B12X_TUNE_CACHE_DIR", str(root))
    monkeypatch.setattr(TuningContract, "iterate", forbidden)
    replayed = request(name="replayed", tuning=contract(default=7, values=(1, 2, 4)))
    with session(tmp_path, autotune=False, compile_workers=0) as engine:
        engine.prepare((replayed,))
        assert replayed.plan.selection.source == "frozen"
        assert replayed.plan.selection.config == Config(3)


def test_invalid_mode_is_rejected_loudly(tmp_path, monkeypatch):
    monkeypatch.setenv("B12X_TUNE_CACHE_MODE", "bogus")
    item = request(name="bad", tuning=contract(default=7, values=(1, 2, 4)))
    with session(tmp_path, autotune=False, compile_workers=0) as engine:
        with pytest.raises(ValueError, match="B12X_TUNE_CACHE_MODE"):
            engine.prepare((item,))


# --- N7 fix: decode-op coverage. ``plan_decode_op`` sessions run with
# autotune=False and were never closed, so their heuristic selections (decode-cap
# shapes) never reached the tune cache and a frozen boot tripped the staleness
# guard. Record mode must capture those heuristic keys, frozen must replay them
# exactly, and real tampering must keep failing loudly.


def test_record_mode_persists_heuristic_selections_from_no_race_sessions(
    tmp_path, monkeypatch
):
    """A decode-op session (autotune=False) records its heuristic default,
    published at the job boundary even though the session is never closed."""
    root = tmp_path / "tune"
    monkeypatch.setenv("B12X_TUNE_CACHE_MODE", "record")
    monkeypatch.setenv("B12X_TUNE_CACHE_DIR", str(root))
    monkeypatch.delenv("HY4_MOE_AUTOTUNE", raising=False)
    item = request(name="decode-cap-shape", tuning=contract(default=7, values=(1, 2, 4)))
    engine = session(tmp_path, autotune=False, compile_workers=0)
    engine.prepare((item,))  # plan_decode_op never closes its session
    assert item.plan.selection.source == "default"
    assert item.plan.selection.config == Config(7)
    payload = json.loads((root / tc.FILE_NAME).read_text())  # job-boundary publish
    assert len(payload["records"]) == 1
    record = next(iter(payload["records"].values()))
    assert record["config"] == {"width": 7}
    assert record["extra"]["source"] == "default"
    assert record["extra"]["request"] == "decode-cap-shape"
    manifest = json.loads(next(root.glob("b12x-tune-audit-*.json")).read_text())
    assert manifest["selections"]
    assert {entry["source"] for entry in manifest["selections"]} == {"default"}
    assert all(entry["config"] == {"width": 7} for entry in manifest["selections"])
    engine.close()


def test_frozen_replays_recorded_heuristic_selections_exactly(tmp_path, monkeypatch):
    """Frozen replay of a decode-cap record is source=frozen with the exact
    heuristic config record mode captured -- zero enumeration, zero writes."""
    root = tmp_path / "tune"
    tuning = contract(default=7, values=(1, 2, 4))
    monkeypatch.setenv("B12X_TUNE_CACHE_MODE", "record")
    monkeypatch.setenv("B12X_TUNE_CACHE_DIR", str(root))
    recorded = request(name="decode", tuning=tuning)
    with session(tmp_path, autotune=False, compile_workers=0) as engine:
        engine.prepare((recorded,))
        assert recorded.plan.selection.config == Config(7)
    monkeypatch.setenv("B12X_TUNE_CACHE_MODE", "frozen")
    monkeypatch.setattr(TuningContract, "iterate", forbidden)
    before = (root / tc.FILE_NAME).read_bytes()
    replayed = request(name="decode-replay", tuning=tuning)
    with session(tmp_path, autotune=False, compile_workers=0) as engine:
        engine.prepare((replayed,))
        assert replayed.plan.selection.source == "frozen"
        assert replayed.plan.selection.config == Config(7)
        assert replayed.plan.selection.assignment == FrozenMapping({"width": 7})
    assert (root / tc.FILE_NAME).read_bytes() == before


def test_frozen_decode_shape_without_a_record_still_trips_staleness(
    tmp_path, monkeypatch
):
    """The pre-fix live failure, unit-pinned: a decode-cap shape that record
    mode never captured keeps tripping the loud guard with the query drift."""
    root = tmp_path / "tune"
    monkeypatch.setenv("B12X_TUNE_CACHE_MODE", "record")
    monkeypatch.setenv("B12X_TUNE_CACHE_DIR", str(root))
    recorded = request(name="main-op-shape", tuning=contract(default=7, values=(1, 2, 4)))
    with session(tmp_path, autotune=False, compile_workers=0) as engine:
        engine.prepare((recorded,))
    monkeypatch.setenv("B12X_TUNE_CACHE_MODE", "frozen")
    tuning = contract(default=7, values=(1, 2, 4))
    plan = Plan(
        contract=tuning,
        query=Query(5),
        _compile_jobs=lambda config, device: (),
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=lambda selection, device: object(),
    )
    other = plan.request(
        name="decode-cap-shape",
        prepare_call=lambda state: PreparedCall(run=lambda: None),
    )
    with session(tmp_path, autotune=False, compile_workers=0) as engine:
        with pytest.raises(tc.TuneCacheStaleError) as info:
            engine.prepare((other,))
    names = [name for name, _want, _have in info.value.mismatches]
    assert "shape.query" in names


def test_recorded_heuristic_tamper_is_stale_with_field_diff(tmp_path, monkeypatch):
    """Staleness rejection keeps firing on real tampering of a heuristic key."""
    root = tmp_path / "tune"
    monkeypatch.setenv("B12X_TUNE_CACHE_MODE", "record")
    monkeypatch.setenv("B12X_TUNE_CACHE_DIR", str(root))
    item = request(name="decode-cap-shape", tuning=contract(default=7, values=(1, 2, 4)))
    with session(tmp_path, autotune=False, compile_workers=0) as engine:
        engine.prepare((item,))
    path = root / tc.FILE_NAME
    payload = json.loads(path.read_text())
    (key,) = tuple(payload["records"])
    entry = payload["records"].pop(key)
    entry["provenance"]["kernel"]["semantic"] = "tampered-by-test"
    payload["records"]["tampered-" + key] = entry
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("B12X_TUNE_CACHE_MODE", "frozen")
    replayed = request(name="decode-cap-shape", tuning=contract(default=7, values=(1, 2, 4)))
    with session(tmp_path, autotune=False, compile_workers=0) as engine:
        with pytest.raises(tc.TuneCacheStaleError) as info:
            engine.prepare((replayed,))
    names = [name for name, _want, _have in info.value.mismatches]
    assert "kernel.semantic" in names


def test_tune_pass_joins_no_race_sessions_to_record_mode(tmp_path, monkeypatch):
    """An offline HY4_MOE_AUTOTUNE=1 pass records the autotune=False decode
    sessions too (implicit record mode, explicit root)."""
    monkeypatch.delenv("B12X_TUNE_CACHE_MODE", raising=False)
    root = tmp_path / "tune"
    monkeypatch.setenv("B12X_TUNE_CACHE_DIR", str(root))
    monkeypatch.setenv("HY4_MOE_AUTOTUNE", "1")
    item = request(name="decode-cap-shape", tuning=contract(default=7, values=(1, 2, 4)))
    engine = session(tmp_path, autotune=False, compile_workers=0)
    engine.prepare((item,))
    payload = json.loads((root / tc.FILE_NAME).read_text())
    assert len(payload["records"]) == 1
    record = next(iter(payload["records"].values()))
    assert record["extra"]["source"] == "default"
    engine.close()


def test_flag_off_world_stays_silent_even_with_a_cache_root(tmp_path, monkeypatch):
    """Flag-off touches nothing: a root on the environment is not enough."""
    monkeypatch.delenv("B12X_TUNE_CACHE_MODE", raising=False)
    root = tmp_path / "tune"
    monkeypatch.setenv("B12X_TUNE_CACHE_DIR", str(root))
    monkeypatch.setenv("HY4_MOE_AUTOTUNE", "0")
    item = request(name="plain", tuning=contract(default=7, values=(1, 2, 4)))
    with session(tmp_path, autotune=False, compile_workers=0) as engine:
        engine.prepare((item,))
        assert item.plan.selection.source == "default"
        assert engine._tune_cache is None
        assert engine._tune_audit == []
    assert not root.exists()
    assert not any(tmp_path.glob("b12x-tune-*"))


def test_record_cache_keeps_the_final_selection_of_a_race(tmp_path, monkeypatch):
    """Per-trial champion churn may write the same key repeatedly; the record
    must hold the obligation's final selection (the winner frozen replays)."""
    monkeypatch.delenv("B12X_TUNE_CACHE_MODE", raising=False)
    monkeypatch.delenv("B12X_TUNE_CACHE_DIR", raising=False)
    root = tmp_path / "tune"
    item = request(name="raced", tuning=contract(default=7, values=(1, 2, 4)))
    engine = session(tmp_path, autotune=True, compile_workers=0, cache_dir=root)
    with engine:
        job = engine.begin((item,))
        configuration = item.plan.contract.configure(
            item.plan.query, device=IDENTITY, override=None, search=False
        )
        obligation = _Obligation(item, configuration)
        for width in (1, 2):  # trials: like the live race's per-candidate churn
            job._selection(
                obligation, Config(width), "tuned", FrozenMapping({"width": width})
            )
        job._selection(obligation, Config(4), "tuned", FrozenMapping({"width": 4}))
        job.close()
    payload = json.loads((root / tc.FILE_NAME).read_text())
    assert len(payload["records"]) == 1
    record = next(iter(payload["records"].values()))
    assert record["config"] == {"width": 4}
    assert record["assignment"] == {"width": 4}
