"""Offline autotune persistence: a frozen tune-config cache (NEON lever N7).

One bounded offline ``HY4_MOE_AUTOTUNE=1`` pass records every winning
per-query configuration to a persistent cache (production path
``/state/b12x-tune``); later boots replay those selections with zero candidate
enumeration, zero racing, and zero candidate compilation instead of paying the
autotune cost per boot or falling back to static heuristics.

Correctness contract (hard): every record is versioned by *kernel + shape +
arch*, and a stale or foreign cache is rejected loudly -- ``resolve()`` and
``load()`` raise :class:`TuneCacheStaleError` with a field-level diff or
:class:`TuneCacheMissError`, never a silent wrong config. The provenance is
kept verbatim beside each record precisely so a near miss can be explained
("kernel.sources changed", "arch.sm_count differs", ...) instead of degrading
into an opaque digest miss.

Modes (``B12X_TUNE_CACHE_MODE``):

* ``off`` -- no reads, no writes (the default whenever tuning is disabled:
  default behavior is unchanged when the autotune flag is off).
* ``record`` -- while tuning, persist each winning selection and an audit
  manifest of every selection the session made. Winners include the
  heuristic ``default`` selections of no-race sessions (decode-cap
  planning), so a frozen replay never falls back to a heuristic the record
  pass did not capture.
* ``frozen`` -- serve-time replay: resolve selections exclusively from the
  cache; anything stale or missing raises loudly.

The cache file is one JSON document merged under a file lock, so concurrent
tuning ranks on a shared ``/state`` directory cannot lose each other's
records. The module is pure standard library on purpose: cache correctness
must be testable without a GPU, torch, or the compiler stack.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

FORMAT_VERSION = 1
DEFAULT_ROOT = "/state/b12x-tune"
FILE_NAME = "b12x-tune-cache.json"
AUDIT_PREFIX = "b12x-tune-audit-"
MODES = ("off", "record", "frozen")
_TUNING_CACHE_VERSION_ENV = "B12X_TUNING_CACHE_VERSION"
_MODE_ENV = "B12X_TUNE_CACHE_MODE"
_DIR_ENV = "B12X_TUNE_CACHE_DIR"


class TuneCacheError(RuntimeError):
    """Base class for tune-cache correctness failures."""


class TuneCacheMissError(TuneCacheError):
    """No tuned record covers the requested kernel/shape/arch."""


class TuneCacheStaleError(TuneCacheError):
    """A cache or record exists but is versioned for something else."""

    def __init__(self, message, mismatches=()):
        super().__init__(message)
        self.mismatches = tuple(mismatches)


def _json(value):
    to_dict = getattr(value, "to_dict", None)
    if to_dict is not None and not isinstance(value, dict):
        value = to_dict()
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _tuning_cache_version():
    raw = os.environ.get(_TUNING_CACHE_VERSION_ENV, "1")
    try:
        version = int(raw)
    except ValueError:
        raise ValueError(
            f"{_TUNING_CACHE_VERSION_ENV} must be a positive integer"
        ) from None
    if version <= 0:
        raise ValueError(f"{_TUNING_CACHE_VERSION_ENV} must be a positive integer")
    return version


def tune_cache_mode(explicit=None):
    """Resolve the cache mode; an unset env keeps default behavior (off)."""
    raw = explicit if explicit is not None else os.environ.get(_MODE_ENV)
    if raw in (None, ""):
        return None
    if raw not in MODES:
        raise ValueError(f"{_MODE_ENV} must be one of {', '.join(MODES)}")
    return raw


def resolve_mode(*, autotune, explicit=None, has_root=True):
    """Resolve the effective mode for one session.

    ``record`` turns on implicitly only while a tune pass runs (the session is
    tuning, or the process runs an offline ``HY4_MOE_AUTOTUNE=1`` pass) and an
    explicit cache root is known, so a plain flag-off session writes and reads
    nothing. ``frozen`` is always explicit.
    """
    mode = tune_cache_mode(explicit)
    if mode is not None:
        if mode == "record" and not has_root:
            raise ValueError("record mode requires B12X_TUNE_CACHE_DIR or cache_dir")
        return mode
    return "record" if autotune and has_root else "off"


def resolve_root(*, mode, cache_dir=None):
    """Explicit dir wins; frozen replay defaults to the shipped /state cache."""
    root = os.environ.get(_DIR_ENV) or cache_dir
    if root is None and mode in ("frozen", "record"):
        root = DEFAULT_ROOT
    return None if root is None else Path(root)


def arch_identity(device_identity):
    """Version records by the silicon: vendor, compute capability, SM count.

    Product labels and reported memory are excluded for the same reason the
    selection cache excludes them: one part ships under several names.
    """
    if device_identity is None:
        raise ValueError("the tune cache requires a device identity")
    if isinstance(device_identity, dict):
        vendor = device_identity.get("vendor")
        capability = device_identity.get("compute_capability")
        sm_count = device_identity.get("sm_count")
    else:
        vendor = getattr(device_identity, "vendor", None)
        capability = getattr(device_identity, "compute_capability", None)
        sm_count = getattr(device_identity, "sm_count", None)
    if capability is not None:
        capability = [int(item) for item in capability]
    if vendor is None or not capability or sm_count is None:
        raise ValueError("the tune cache requires vendor, compute capability, SM count")
    return {
        "vendor": str(vendor),
        "compute_capability": capability,
        "sm_count": int(sm_count),
    }


def _contract_callables(contract):
    for name in (
        "encode_query",
        "encode_config",
        "decode_config",
        "validate_query",
        "validate_config",
        "default_config",
        "parameters",
        "materialize",
        "equivalence_key",
        "collective_signature",
        "collective_family_bucket",
    ):
        value = getattr(contract, name, None)
        if value is not None:
            yield value


def source_fingerprint(contract, source_root=None):
    """Fingerprint the in-repo source closure of the contract's callables.

    This is the cache's *kernel* version: any edit to the tuning contract's
    modules or their b12x import closure (kernel implementations included)
    changes the fingerprint and invalidates every recorded winner. Contracts
    outside the b12x tree fall back to hashing their own source files.
    """
    import inspect

    from b12x._lib.source_dependencies import (
        SourceDependencies,
        canonical_module,
        source_dependencies,
    )

    if source_root is None:
        source_root = Path(__file__).resolve().parents[1]
    roots = set()
    for target in _contract_callables(contract):
        try:
            unwrapped = inspect.unwrap(target)
            module = canonical_module(unwrapped.__module__)
            found = inspect.getsourcefile(unwrapped) or inspect.getsourcefile(target)
            if found is None:
                continue
            path = Path(found).resolve()
        except (TypeError, OSError):
            continue
        roots.add((module, path))
    if not roots:
        raise ValueError("the tuning contract exposes no source callables")
    in_tree = tuple(sorted(item for item in roots if item[1].is_relative_to(source_root)))
    if in_tree:
        try:
            return {
                "mode": "closure",
                "digest": source_dependencies(source_root).fingerprint(in_tree),
            }
        except Exception:  # noqa: BLE001 - fall back to direct file hashes
            pass
    digest = hashlib.sha256()
    for module, path in sorted(roots):
        if not path.is_file():
            continue
        digest.update(module.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return {"mode": "files", "digest": digest.hexdigest()}


def kernel_provenance(contract, *, fingerprint=None):
    """Kernel half of the version key: component, contract versions, sources."""
    return {
        "component": contract.component_id,
        "query_schema": contract.query_schema_version,
        "config_schema": contract.config_schema_version,
        "semantic": contract.semantic_version,
        "candidate_contract": contract.candidate_contract_version,
        "config_fields": sorted(contract.config_fields),
        "sources": fingerprint
        if fingerprint is not None
        else source_fingerprint(contract),
    }


def shape_provenance(*, encoded_query, invocation, pin, dependencies):
    """Shape half of the version key: query, invocation, pin, dependency pins."""
    return {
        "query": json.loads(_json(encoded_query)),
        "invocation": json.loads(_json(invocation or {})),
        "pin": None if pin is None else json.loads(_json(pin)),
        "dependencies": [json.loads(_json(item)) for item in dependencies or ()],
    }


def build_provenance(
    contract,
    *,
    encoded_query,
    invocation=None,
    pin=None,
    dependencies=(),
    device_identity=None,
    arch=None,
    fingerprint=None,
):
    """Compose the full kernel + shape + arch version record."""
    return {
        "kernel": kernel_provenance(contract, fingerprint=fingerprint),
        "shape": shape_provenance(
            encoded_query=encoded_query,
            invocation=invocation,
            pin=pin,
            dependencies=dependencies,
        ),
        "arch": dict(arch if arch is not None else arch_identity(device_identity)),
    }


def provenance_key(provenance):
    return _digest(provenance)


def _diff_provenance(want, have):
    """Explain a near miss field by field, kernel/shape/arch first."""
    mismatches = []
    for group in ("kernel", "shape", "arch"):
        left, right = want.get(group, {}), have.get(group, {})
        if left == right:
            continue
        if not isinstance(left, dict) or not isinstance(right, dict):
            mismatches.append((group, left, right))
            continue
        for name in sorted(set(left) | set(right)):
            if left.get(name) != right.get(name):
                mismatches.append((f"{group}.{name}", left.get(name), right.get(name)))
    return mismatches


class TuneCache:
    """Persistent, versioned winning configurations for one arch."""

    def __init__(self, root, *, arch, tuning_cache_version=None, session=None):
        self.root = Path(root)
        self.arch = dict(arch)
        self.tuning_cache_version = (
            _tuning_cache_version()
            if tuning_cache_version is None
            else int(tuning_cache_version)
        )
        if self.tuning_cache_version <= 0:
            raise ValueError("tuning_cache_version must be a positive integer")
        self.session = session
        self.records = {}
        self._pending = {}
        self._loaded = False

    @property
    def path(self):
        return self.root / FILE_NAME

    def _header(self):
        return {
            "arch": dict(self.arch),
            "tuning_cache_version": self.tuning_cache_version,
        }

    def _validate_header(self, payload):
        if not isinstance(payload, dict) or set(payload) != {
            "format_version",
            "identity",
            "records",
        }:
            raise TuneCacheStaleError(
                f"{self.path}: invalid tune-cache document shape",
                (("format", "tune-cache document", "unknown"),),
            )
        if payload["format_version"] != FORMAT_VERSION:
            raise TuneCacheStaleError(
                f"{self.path}: tune-cache format "
                f"{payload['format_version']} != {FORMAT_VERSION}",
                (("format_version", FORMAT_VERSION, payload["format_version"]),),
            )
        identity = payload["identity"]
        for name in ("arch", "tuning_cache_version"):
            want = self._header()[name]
            have = identity.get(name)
            if have != want:
                raise TuneCacheStaleError(
                    f"{self.path}: foreign tune cache ({name} differs)",
                    ((name, want, have),),
                )

    def load(self):
        """Load and validate the cache document; foreign caches raise loudly."""
        if self._loaded:
            return self
        if self.path.exists():
            payload = json.loads(self.path.read_text())
            self._validate_header(payload)
            self.records = dict(payload["records"])
        self._loaded = True
        return self

    def record(
        self,
        provenance,
        config,
        *,
        assignment=None,
        coverage=None,
        finalists=None,
        extra=None,
    ):
        """Stage one winning selection; it reaches disk at the next flush."""
        key = provenance_key(provenance)
        entry = {
            "key": key,
            "provenance": json.loads(_json(provenance)),
            "config": json.loads(_json(config)),
        }
        if assignment is not None:
            entry["assignment"] = json.loads(_json(assignment))
        if coverage is not None:
            entry["coverage"] = json.loads(_json(dict(coverage)))
        if finalists is not None:
            entry["finalists"] = json.loads(_json(list(finalists)))
        if extra is not None:
            entry["extra"] = json.loads(_json(dict(extra)))
        entry["recorded_at"] = time.time()
        self._pending[key] = entry
        self.records[key] = entry
        return key

    def flush(self):
        """Merge staged records into the file under its lock."""
        if not self._pending:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        pending, self._pending = self._pending, {}
        lock_path = self.path.with_suffix(".lock")
        with lock_path.open("a") as lock:
            _lock(lock.fileno())
            payload = (
                json.loads(self.path.read_text())
                if self.path.exists()
                else {
                    "format_version": FORMAT_VERSION,
                    "identity": self._header(),
                    "records": {},
                }
            )
            self._validate_header(payload)
            payload["records"].update(pending)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=self.root, suffix=".tmp", delete=False
                ) as stream:
                    temporary = stream.name
                    stream.write(_json(payload))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                if temporary is not None and os.path.exists(temporary):
                    os.unlink(temporary)
        self.records.update(pending)

    def resolve(self, provenance):
        """Return the tuned record for exactly this kernel/shape/arch.

        A miss with a near record is a staleness signal and raises
        :class:`TuneCacheStaleError` with the exact version fields that
        drifted; no record at all raises :class:`TuneCacheMissError`.
        """
        self.load()
        key = provenance_key(provenance)
        record = self.records.get(key)
        if record is not None:
            return record
        near = []
        for candidate in self.records.values():
            other = candidate.get("provenance", {})
            if other.get("kernel", {}).get("component") != provenance["kernel"].get(
                "component"
            ):
                continue
            mismatches = _diff_provenance(provenance, other)
            if mismatches:
                near.append(mismatches)
        if near:
            detail = "; ".join(
                f"{name}: want {want!r} have {have!r}"
                for name, want, have in near[0]
            )
            raise TuneCacheStaleError(
                f"stale tune cache for {provenance['kernel'].get('component')}: "
                f"{detail}",
                near[0],
            )
        raise TuneCacheMissError(
            f"no tuned config for {provenance['kernel'].get('component')} "
            f"(cache holds {len(self.records)} records; "
            f"run an offline HY4_MOE_AUTOTUNE=1 pass to populate {self.path})"
        )

    def close(self):
        self.flush()


def _lock(fileno):
    try:
        import _flashinfer_filelock as filelock

        filelock.lock(fileno)
        return
    except ImportError:
        pass
    import fcntl

    fcntl.flock(fileno, fcntl.LOCK_EX)


def write_audit(path, entries, *, session=None):
    """Write the human-auditable manifest of every selection a session made."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "session": session,
        "written_at": time.time(),
        "selections": list(entries),
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = stream.name
            stream.write(_json(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return path


def audit_path(root, session_id):
    return Path(root) / f"{AUDIT_PREFIX}{session_id}.json"
