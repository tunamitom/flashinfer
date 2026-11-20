"""N7 offline tune cache: versioned persistence with loud staleness rejection.

Every record is versioned by kernel + shape + arch. The hard contract under
test: a stale or foreign cache is never silently used -- loads and lookups
raise with a field-level explanation -- and the default (flag-off) world never
touches the cache at all (covered by ``test_n7_frozen_session``).
"""

import json

import pytest

from b12x.preparation import FrozenMapping
from b12x.preparation import _tune_cache as tc
from b12x.preparation.types import DeviceIdentity

from .test_defaults import Config, Query, contract

IDENTITY = DeviceIdentity("nvidia", (12, 0), 148, "synthetic SM120")
OTHER_ARCH = DeviceIdentity("nvidia", (12, 0), 132, "synthetic SM120")


def provenance(tuning=None, *, query=None, arch=None, fingerprint=None):
    tuning = contract() if tuning is None else tuning
    return tc.build_provenance(
        tuning,
        encoded_query=tuning.configure(
            query or Query(3), device=IDENTITY, override=None
        ).encoded_query,
        invocation={},
        pin=None,
        dependencies=(),
        arch=tc.arch_identity(arch or IDENTITY),
        fingerprint=fingerprint,
    )


def test_record_flush_load_round_trip(tmp_path):
    cache = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    want = provenance()
    cache.record(want, FrozenMapping({"width": 4}), assignment={"width": 4})
    cache.flush()
    replay = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    record = replay.resolve(want)
    assert record["config"] == {"width": 4}
    assert record["assignment"] == {"width": 4}
    assert record["key"] == tc.provenance_key(want)


def test_provenance_key_is_stable_and_order_independent(tmp_path):
    left, right = provenance(), provenance()
    assert tc.provenance_key(left) == tc.provenance_key(right)
    rebuilt = {
        "arch": left["arch"],
        "shape": left["shape"],
        "kernel": left["kernel"],
    }
    assert tc.provenance_key(rebuilt) == tc.provenance_key(left)


def test_missing_cache_raises_miss(tmp_path):
    cache = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    with pytest.raises(tc.TuneCacheMissError, match="no tuned config"):
        cache.resolve(provenance())


def test_unrelated_component_is_a_miss_not_a_stale_hit(tmp_path):
    cache = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    cache.record(provenance(), {"width": 4})
    cache.flush()
    other = provenance(contract(default=2, values=(1, 2, 4)))
    other["kernel"]["component"] = "test.other"
    replay = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    with pytest.raises(tc.TuneCacheMissError):
        replay.resolve(other)


def test_kernel_version_bump_is_stale_with_field_diff(tmp_path):
    cache = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    want = provenance(contract())
    cache.record(want, {"width": 4})
    cache.flush()
    bumped = provenance(contract())
    bumped["kernel"]["semantic"] += 1
    replay = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    with pytest.raises(tc.TuneCacheStaleError) as info:
        replay.resolve(bumped)
    names = [name for name, _want, _have in info.value.mismatches]
    assert "kernel.semantic" in names


def test_kernel_source_fingerprint_change_is_stale(tmp_path):
    cache = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    want = provenance(fingerprint={"mode": "closure", "digest": "a" * 64})
    cache.record(want, {"width": 4})
    cache.flush()
    edited = provenance(fingerprint={"mode": "closure", "digest": "b" * 64})
    replay = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    with pytest.raises(tc.TuneCacheStaleError) as info:
        replay.resolve(edited)
    names = [name for name, _want, _have in info.value.mismatches]
    assert "kernel.sources" in names


def test_shape_change_is_stale_with_query_diff(tmp_path):
    cache = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    cache.record(provenance(), {"width": 4})
    cache.flush()
    other_shape = provenance(query=Query(5))
    replay = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    with pytest.raises(tc.TuneCacheStaleError) as info:
        replay.resolve(other_shape)
    names = [name for name, _want, _have in info.value.mismatches]
    assert "shape.query" in names


def test_arch_mismatch_rejected_on_load_naming_arch(tmp_path):
    cache = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    want = provenance()
    cache.record(want, {"width": 4})
    cache.flush()
    foreign = provenance(arch=OTHER_ARCH)
    replay = tc.TuneCache(tmp_path, arch=tc.arch_identity(OTHER_ARCH))
    with pytest.raises(tc.TuneCacheStaleError) as info:
        replay.resolve(foreign)
    names = [name for name, _want, _have in info.value.mismatches]
    assert "arch" in names


def test_foreign_cache_file_rejected_on_load(tmp_path):
    cache = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    cache.record(provenance(), {"width": 4})
    cache.flush()
    foreign = tc.TuneCache(tmp_path, arch=tc.arch_identity(OTHER_ARCH))
    with pytest.raises(tc.TuneCacheStaleError, match="foreign tune cache"):
        foreign.load()


def test_format_version_mismatch_rejected_on_load(tmp_path):
    cache = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    cache.record(provenance(), {"width": 4})
    cache.flush()
    payload = json.loads(cache.path.read_text())
    payload["format_version"] = tc.FORMAT_VERSION + 1
    cache.path.write_text(json.dumps(payload))
    replay = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    with pytest.raises(tc.TuneCacheStaleError, match="format"):
        replay.load()


def test_tuning_cache_version_knob_invalidates(tmp_path, monkeypatch):
    monkeypatch.setenv("B12X_TUNING_CACHE_VERSION", "2")
    stale = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    stale.record(provenance(), {"width": 4})
    stale.flush()
    monkeypatch.setenv("B12X_TUNING_CACHE_VERSION", "3")
    replay = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    with pytest.raises(tc.TuneCacheStaleError, match="tuning_cache_version"):
        replay.load()


def test_invalid_document_shape_rejected_loudly(tmp_path):
    cache = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    cache.flush()
    cache.path.write_text(json.dumps({"records": {}}))
    replay = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    with pytest.raises(tc.TuneCacheStaleError):
        replay.load()


def test_concurrent_ranks_merge_without_lost_records(tmp_path):
    left = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    right = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    first = provenance(query=Query(3))
    second = provenance(query=Query(5))
    left.record(first, {"width": 1})
    right.record(second, {"width": 2})
    right.flush()
    left.flush()
    merged = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    assert merged.resolve(first)["config"] == {"width": 1}
    assert merged.resolve(second)["config"] == {"width": 2}


def test_flushing_into_a_foreign_cache_fails_loudly(tmp_path):
    cache = tc.TuneCache(tmp_path, arch=tc.arch_identity(IDENTITY))
    cache.record(provenance(), {"width": 4})
    cache.flush()
    foreign = tc.TuneCache(tmp_path, arch=tc.arch_identity(OTHER_ARCH))
    foreign.record(provenance(arch=OTHER_ARCH), {"width": 8})
    with pytest.raises(tc.TuneCacheStaleError):
        foreign.flush()


def test_audit_manifest_lists_every_selection(tmp_path):
    entries = [
        {"request": "a", "component": "test.arithmetic", "source": "tuned"},
        {"request": "b", "component": "test.arithmetic", "source": "default"},
    ]
    path = tc.write_audit(tc.audit_path(tmp_path, "sess"), entries, session="sess")
    payload = json.loads(path.read_text())
    assert [item["source"] for item in payload["selections"]] == ["tuned", "default"]
    assert payload["session"] == "sess"


def test_mode_resolution_keeps_flag_off_default_untouched():
    assert tc.resolve_mode(autotune=False) == "off"
    assert tc.resolve_mode(autotune=True, has_root=False) == "off"
    assert tc.resolve_mode(autotune=True, has_root=True) == "record"
    assert tc.resolve_mode(autotune=False, explicit="frozen") == "frozen"
    with pytest.raises(ValueError, match="record mode requires"):
        tc.resolve_mode(autotune=True, explicit="record", has_root=False)
    with pytest.raises(ValueError, match="B12X_TUNE_CACHE_MODE"):
        tc.resolve_mode(autotune=True, explicit="bogus")


def test_root_resolution_prefers_env_then_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("B12X_TUNE_CACHE_DIR", str(tmp_path / "env"))
    assert tc.resolve_root(mode="frozen", cache_dir=tmp_path / "dir") == tmp_path / "env"
    monkeypatch.delenv("B12X_TUNE_CACHE_DIR")
    assert tc.resolve_root(mode="frozen", cache_dir=tmp_path / "dir") == tmp_path / "dir"
    assert tc.resolve_root(mode="frozen", cache_dir=None) == tc.Path(tc.DEFAULT_ROOT)
    assert tc.resolve_root(mode="off", cache_dir=None) is None


def test_arch_identity_requires_the_silicon_fields():
    assert tc.arch_identity(IDENTITY) == {
        "vendor": "nvidia",
        "compute_capability": [12, 0],
        "sm_count": 148,
    }
    with pytest.raises(ValueError, match="device identity"):
        tc.arch_identity(None)
    with pytest.raises(ValueError, match="vendor"):
        tc.arch_identity({"vendor": None})


def test_real_csf_expansion_provenance_is_kernel_versioned():
    from b12x._lib.quant.csf_blocks_tuning import CsfExpansionQuery, TUNING

    query = CsfExpansionQuery(
        first=(256, 256, 4, 9, 256, 256, 32, 8, 1),
        second=(256, 256, 4, 9, 256, 256, 32, 8, 1),
        routes=128,
        ids_dtype="int32",
    )
    first = tc.kernel_provenance(TUNING)
    assert first["component"] == "quantization.csf_block_expansion"
    assert first["sources"]["digest"]
    assert first == tc.kernel_provenance(TUNING)
    encoded = TUNING.configure(query, device=IDENTITY, override=None).encoded_query
    key = tc.provenance_key(
        tc.build_provenance(
            TUNING,
            encoded_query=encoded,
            invocation={},
            pin=None,
            dependencies=(),
            device_identity=IDENTITY,
        )
    )
    assert len(key) == 64
