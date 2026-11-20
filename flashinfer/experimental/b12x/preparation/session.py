"""One serial preparation lifecycle with cooperative compilation and tuning.

The session fills each plan's prepared slot in place. Plans that are already
prepared are skipped unless a job supplies a different selected configuration.
Plan handles remain stable when their prepared payloads change. Candidate races are
timed in bounded batches; the running best candidate is carried into every
later batch so each candidate is compared head to head with it.
Completed batches checkpoint their measurements and champion before releasing
trials. Restarts resume the same race and tuning shard from that checkpoint.
"""

from __future__ import annotations

import logging
import gc
import hashlib
import json
import math
import os
import pickle
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

from filelock import Timeout

from b12x._lib.compile_plan import (
    _RESIDENT_PROGRAMS,
    compiled_program_available,
    observe_programs,
    retain_compiled_programs,
)
from b12x._lib.compile_pool import CompilePool, describe_compilation, compile_in_process
from b12x._lib.compiler import discard_prefetched_cute_objects, prefetch_cute_objects
from b12x._lib.program_cache import PreparationProgramCache, evict_unretained
from b12x._lib.runtime_control import (
    KernelResolutionFrozenError,
    kernel_resolution_guard,
)
from ._cache import SelectionCache, _json, cache_identity, digest
from . import _tune_cache
from ._efficiency import AutotuneEffort, capture_autotune_effort
from ._discovery import discovery_key, discovery_record, restore_discovery
from ._measurement import DEFAULT_SAMPLES, COLLECTIVE_MAX_ROUNDS
from ._ledger import RaceLedger
from ._race_cache import RaceCache, RaceCheckpoint
from ._timing import PreparationTiming
from .device import DetectedDevice, detect_device
from .types import (
    CandidateResourceIneligibleError,
    CollectiveRequirement,
    CollectiveTuningRequirement,
    FrozenMapping,
    MemoryRequirements,
    Plan,
    PreparationProgress,
    PreparationRequest,
    PreparationResult,
    PreparedCall,
    Selection,
    TuningCacheRequirement,
    TuningRequirement,
    _CompositePlan,
    Finalist,
    FinalistRecord,
    _Prepared,
    _close_all,
    _plan_scope,
    _prime,
    _set_lazy_preparer,
    call_scope,
)

logger = logging.getLogger("b12x.preparation")


class _CandidateLaunchRejected(Exception):
    """A trial could not launch, but its resources were safely released."""


def _cooperative_launch_rejected(error):
    from cutlass.base_dsl.common import DSLCudaRuntimeError

    return isinstance(error, DSLCudaRuntimeError) and error.error_code == 720


@dataclass
class _Obligation:
    request: PreparationRequest
    configuration: object
    requests: tuple = ()
    candidates: list = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    selection: Selection | None = None
    finalists: tuple[FinalistRecord, ...] = ()
    key: str | None = None
    candidate_program_identity: str | None = None
    programs: dict = field(default_factory=dict)
    compiled: dict = field(default_factory=dict)
    jobs: dict = field(default_factory=dict)
    compile_assignments: dict = field(default_factory=dict)
    cache_pending: bool = False
    ready: bool = False
    planned_candidates: int = 0
    discovery: _Discovery | None = None
    compiles_submitted: bool = False
    payloads: dict = field(default_factory=dict)


@dataclass
class _Discovery:
    """Program discovery for one obligation, advanced in non-blocking steps."""

    candidates: tuple
    key: str | None
    cache: object | None
    lock: object | None = None
    batches: list = field(default_factory=list)
    submitted: bool = False
    done: bool = False


@dataclass(frozen=True)
class _TuningBatch:
    contributions: tuple[TuningRequirement, ...]


class _CallGuard:
    """Run a prepared call's restore and close callbacks at most once each."""

    def __init__(self, call):
        if not isinstance(call, PreparedCall):
            raise TypeError("call factory must return PreparedCall")
        self.call = call
        self._restored = self._closed = False

    def restore(self):
        if not self._restored:
            self._restored = True
            if self.call.restore is not None:
                with call_scope():
                    self.call.restore()

    def close(self):
        if not self._closed:
            self._closed = True
            if self.call.close is not None:
                self.call.close()

    def finish(self):
        try:
            _close_all((self.restore, self.close))
        finally:
            self.call = None


@dataclass
class _Trial:
    index: int
    assignment: object
    config: object
    call: PreparedCall
    guard: _CallGuard
    retained: object
    state: object
    latency_us: float | None = None

    def close(self):
        try:
            self.guard.finish()
        finally:
            self.call = self.state = self.retained = None


# Local metadata and GPU steps share one time slice. Collective readiness and
# pending compilation always return control before more work is admitted.
_ADVANCE_SECONDS = 0.1


# Cross-rank priming assumes co-scheduled ranks: a collective authorization is
# consumed in the round after the exchange, and the local work before the launch
# (autotune rounds) is unbounded, so the barrier converts the shared
# authorization into a launch lockstep. Generous by default: it only ever waits
# during preparation, and must cover the slowest rank's pre-launch round.
def _barrier_timeout_seconds():
    """Return the finite positive collective-barrier deadline from the environment."""
    try:
        timeout = float(os.environ.get("B12X_COLLECTIVE_BARRIER_TIMEOUT", "120"))
    except ValueError as error:
        raise ValueError(
            "B12X_COLLECTIVE_BARRIER_TIMEOUT must be a finite positive number of seconds"
        ) from error
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(
            "B12X_COLLECTIVE_BARRIER_TIMEOUT must be a finite positive number of seconds"
        )
    return timeout


class CollectiveBarrierTimeout(RuntimeError):
    """A collective barrier callback exceeded its deadline."""

    def __init__(
        self,
        key: str,
        ranks: tuple[int, ...],
        arrived: tuple[int, ...] | None = None,
        *,
        timeout: float = 120.0,
    ):
        """Record verified arrivals when the barrier backend supplies them."""
        if arrived is None:
            message = (
                f"collective barrier callback timed out for {key!r} after {timeout:g}s"
            )
        else:
            waiting = tuple(sorted(set(ranks) - set(arrived)))
            message = (
                f"collective barrier timed out for {key!r} after "
                f"{timeout:g}s: ranks {waiting} never entered "
                f"(arrived: {tuple(sorted(arrived))})"
            )
        super().__init__(message)
        self.key = key
        self.ranks = ranks
        self.arrived = None if arrived is None else tuple(sorted(arrived))
        self.timeout = timeout


def _declaration_key(plan):
    """Return the immutable declaration identity used to coalesce requests."""
    if isinstance(plan, _CompositePlan):
        return (
            plan.component_id,
            plan.composite_semantic_version,
            plan.capacity_metadata,
            tuple(
                (count, _declaration_key(child))
                for count, child in plan.variants.items()
            ),
        )
    contract = plan.contract
    return (
        contract.component_id,
        contract.query_schema_version,
        contract.config_schema_version,
        contract.semantic_version,
        contract.candidate_contract_version,
        FrozenMapping(contract.encode_query(plan.query)),
        plan.invocation,
        None if plan.override is None else contract.config_payload(plan.override),
        plan._device,
    )


def _topological(requests):
    by_name = {}
    for request in requests:
        if not isinstance(request, PreparationRequest):
            raise TypeError("preparation batches contain PreparationRequest values")
        if request.name in by_name:
            raise ValueError(f"duplicate preparation name {request.name!r}")
        by_name[request.name] = request
    missing = {
        name for request in requests for name in request.dependencies
    } - by_name.keys()
    if missing:
        raise ValueError(f"unknown preparation dependencies: {sorted(missing)}")
    ordered, visiting, visited = [], set(), set()

    def visit(name):
        if name in visiting:
            raise ValueError("preparation dependencies contain a cycle")
        if name in visited:
            return
        visiting.add(name)
        for dependency in by_name[name].dependencies:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)
        ordered.append(by_name[name])

    for name in by_name:
        visit(name)
    return ordered


def _child_requests(request):
    plan = request.plan
    return {
        count: child.request(
            name=f"{request.name}/m{count}",
            prepare_call=request.prepare_call[count],
            benchmark_call=None
            if request.benchmark_call is None
            else request.benchmark_call[count],
            dependencies=request.dependencies,
            collective=request.collective,
            retain_benchmark_call=request.retain_benchmark_call,
        )
        for count, child in plan.variants.items()
    }


def _expand_requests(requests):
    requests = list(_topological(requests))
    expanded, composites = [], {}
    for request in requests:
        if request.plan.shared and (
            request.dependencies or request.collective is not None
        ):
            raise ValueError(
                f"shared plan {request.name!r} cannot declare dependencies or collectives"
            )
        if isinstance(request.plan, _CompositePlan):
            children = _child_requests(request)
            composites[request.name] = (request, children)
            expanded.extend(children.values())
        else:
            expanded.append(request)

    from dataclasses import replace

    normalized = []
    for request in expanded:
        dependencies = []
        for name in request.dependencies:
            if name in composites:
                dependencies.extend(
                    child.name for child in composites[name][1].values()
                )
            else:
                dependencies.append(name)
        normalized.append(
            replace(request, dependencies=tuple(dict.fromkeys(dependencies)))
        )
    return _topological(normalized), composites


def _coalesce_requests(requests):
    """Group declaration-equivalent work while preserving its runtime bindings."""
    groups, keys, policies = {}, {}, {}
    for request in requests:
        if (
            request.plan in policies
            and policies[request.plan] != request.finalist_policy
        ):
            raise ValueError("requests for the same Plan must agree on finalist policy")
        policies[request.plan] = request.finalist_policy
        key = (
            _declaration_key(request.plan),
            tuple(dict.fromkeys(keys[name] for name in request.dependencies)),
            None if request.collective is None else request.collective.ranks,
            request.finalist_policy,
        )
        keys[request.name] = key
        groups.setdefault(key, []).append(request)
    return tuple(tuple(group) for group in groups.values())


def _plans_of(request):
    """Return a request plan and every child plan of a composite request."""
    plan = request.plan
    if isinstance(plan, _CompositePlan):
        return (plan, *plan.variants.values())
    return (plan,)


def _default_compile_workers(device):
    identity = device.identity
    spark = (
        identity is not None
        and identity.vendor == "nvidia"
        and identity.product_name in {"gb10", "nvidia gb10"}
    )
    return int(os.environ.get("B12X_COMPILE_WORKERS", "4" if spark else "8"))


class PreparationSession:
    def __init__(
        self,
        *,
        device=None,
        autotune=True,
        cache_dir=None,
        namespace=None,
        compile_workers=None,
        rounds=COLLECTIVE_MAX_ROUNDS,
        samples=DEFAULT_SAMPLES,
        cache_only=False,
        race_batch=32,
        race_budget=None,
        cache_policy="cold",
        collective_barrier=None,
        prime=True,
    ):
        """Initialize planning resources and an optional collective entry barrier.

        ``collective_barrier(key, ranks)`` runs in a daemon thread and must be
        safe to call there. A timeout does not cancel that callback; another
        preparation job cannot begin until the callback exits.

        ``race_batch`` bounds the candidates timed together; ``race_budget`` is
        accepted for compatibility and no longer limits a batch.

        ``prime=False`` materializes and binds selected plans without running
        their prepare calls. The caller must warm every execution path before
        capture or freeze. Candidate and retained benchmark calls still prime.
        Prepare-call factories, synchronization, program checks, resource
        retention, and restore/close hooks run regardless of this setting.
        """
        if type(prime) is not bool:
            raise TypeError("prime must be a bool")
        self.prime = prime
        if cache_policy not in ("cold", "warm"):
            raise ValueError("cache_policy must be cold or warm")
        self._cache_policy = cache_policy
        self.device = (
            device if isinstance(device, DetectedDevice) else detect_device(device)
        )
        if compile_workers is None:
            compile_workers = _default_compile_workers(self.device)
        for name, value in (
            ("compile_workers", compile_workers),
            ("rounds", rounds),
            ("samples", samples),
            ("race_batch", race_batch),
        ):
            minimum = 0 if name == "compile_workers" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if race_budget is not None and (
            type(race_budget) is not int or race_budget <= 0
        ):
            raise ValueError("race_budget must be a positive byte count or None")
        if type(autotune) is not bool or type(cache_only) is not bool:
            raise TypeError("autotune and cache_only must be boolean")
        self._autotune_effort = capture_autotune_effort()
        self.autotune = autotune and self._autotune_effort is not AutotuneEffort.OFF
        self.cache_only = cache_only and self._autotune_effort is not AutotuneEffort.OFF
        self.compile_workers, self.rounds, self.samples = (
            compile_workers,
            rounds,
            samples,
        )
        self.race_batch, self.race_budget = race_batch, race_budget
        self.namespace = FrozenMapping(namespace or {})
        if collective_barrier is not None and not callable(collective_barrier):
            raise TypeError("collective_barrier must be callable")
        self.collective_barrier = collective_barrier
        self._barrier_timeout = (
            _barrier_timeout_seconds() if collective_barrier is not None else 120.0
        )
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self._cache = None
        self._race_cache = None
        # N7: persistent offline-tune cache (B12X_TUNE_CACHE_MODE / _DIR). The
        # mode defaults to off, so flag-off sessions read and write nothing.
        self._tune_cache = None
        self._tune_cache_mode = None
        self._tune_cache_root = None
        self._tune_cache_resolved = False
        self._tune_audit = []
        self._stop = threading.Event()
        if os.environ.get("B12X_AUTOTUNE", "1") == "0":
            self._stop.set()
        self._thread = threading.get_ident()
        self._pool = None
        self._job = None
        self._pending_collective_barrier = None
        self._plans = {}
        self._scratch_limit = None
        self._scratch_limit_locked = False
        self._shared = {}
        self._guard = None
        self._tuning_rank = 0
        self._tuning_ranks = (0,)
        self._tuning_cache_synchronized = False
        self._collective_family_buckets = {}
        self.session_id = uuid.uuid4().hex
        self.state = "OPEN"

    def configure_compile_workers(self, workers: int | None = None) -> None:
        """Change compiler concurrency between preparation jobs."""
        self._check_thread()
        if self._job is not None or self._pool is not None:
            raise RuntimeError("compiler concurrency can only change between jobs")
        if workers is None:
            workers = _default_compile_workers(self.device)
        if type(workers) is not int or workers < 0:
            raise ValueError("compile_workers must be an integer of at least 0")
        self.compile_workers = workers

    def configure_tuning_shard(self, rank: int, ranks: tuple[int, ...]) -> None:
        """Assign this process a disjoint share of non-collective candidates."""
        self._check_thread()
        if self._job is not None:
            raise RuntimeError("cannot change tuning ranks during preparation")
        ranks = tuple(ranks)
        if (
            type(rank) is not int
            or rank < 0
            or not ranks
            or any(type(item) is not int or item < 0 for item in ranks)
            or ranks != tuple(sorted(set(ranks)))
            or rank not in ranks
        ):
            raise ValueError(
                "tuning ranks must be sorted, unique, and include this rank"
            )
        self._tuning_rank = rank
        self._tuning_ranks = ranks
        self._tuning_cache_synchronized = False

    def configure_scratch_limit(self, max_bytes: int | None) -> None:
        """Limit each candidate's scratch before starting preparation jobs.

        Autotuning excludes configurations above this byte count. Explicit,
        default, and retained selections must fit. ``None`` removes the limit.
        """
        self._check_thread()
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
            raise ValueError("scratch limit must be a nonnegative byte count or None")
        if self._scratch_limit_locked and max_bytes != self._scratch_limit:
            raise RuntimeError(
                "scratch limit must be configured before preparation begins"
            )
        self._scratch_limit = max_bytes

    def _candidate_memory(self, requests, config):
        requirements, seen, shared = [], set(), False
        for request in requests:
            plan = request.plan
            if plan in seen:
                continue
            seen.add(plan)
            if plan.shared and shared and self._scratch_limit is None:
                continue
            with _plan_scope(plan):
                requirement = plan._memory_requirements(config, self.device)
            requirements.append(
                replace(requirement, persistent=())
                if plan.shared and shared
                else requirement
            )
            shared |= plan.shared
        requirement = MemoryRequirements.sequential(requirements)
        if (
            self._scratch_limit is not None
            and requirement.scratch_nbytes > self._scratch_limit
        ):
            raise CandidateResourceIneligibleError(
                f"{requests[0].plan.component_id} configuration {config!r} requires "
                f"{requirement.scratch_nbytes} scratch bytes, exceeding the session "
                f"scratch limit of {self._scratch_limit} bytes"
            )
        return requirement

    @property
    def cache_policy(self):
        """Whether tuning races evict L2 or retain warmed data between samples."""
        return self._cache_policy

    def _check_thread(self):
        """Reject use from another thread or after the session closes."""
        if threading.get_ident() != self._thread:
            raise RuntimeError("only cancel_tuning may run on another thread")
        if self.state == "CLOSED":
            raise RuntimeError("preparation session is closed")

    def _selection_cache(self):
        """Return the session selection cache, creating it for CUDA devices."""
        if self._cache is None:
            if self.device.ordinal is None:
                raise RuntimeError("selection caching requires a CUDA device")
            root = self.cache_dir
            if root is None:
                from b12x._lib.compiler import _cute_compile_cache_dir

                root = _cute_compile_cache_dir() / "preparation"
            self._cache = SelectionCache(
                root,
                cache_identity(
                    self.namespace.to_dict(),
                    self.device.ordinal,
                    cache_policy=self.cache_policy,
                ),
            )
        return self._cache

    def _race_progress_cache(self):
        if self._race_cache is None:
            path = self._selection_cache().path.with_suffix(".progress.sqlite3")
            self._race_cache = RaceCache(path)
        return self._race_cache

    def _tune_cache_settings(self):
        """Resolve the N7 tune-cache mode and root once (env, then cache_dir)."""
        if not self._tune_cache_resolved:
            explicit = os.environ.get("B12X_TUNE_CACHE_MODE") or None
            has_root = bool(
                os.environ.get("B12X_TUNE_CACHE_DIR") or self.cache_dir is not None
            )
            # An offline HY4_MOE_AUTOTUNE=1 pass records every session in the
            # process, including the autotune=False sessions (decode-op
            # planning) whose selections are heuristic. A flag-off process
            # stays off; an explicit mode always wins.
            tune_pass = self._autotune_effort is not AutotuneEffort.OFF and (
                os.environ.get("HY4_MOE_AUTOTUNE") or ""
            ) not in ("", "0")
            self._tune_cache_mode = _tune_cache.resolve_mode(
                autotune=self.autotune or tune_pass,
                explicit=explicit,
                has_root=has_root,
            )
            self._tune_cache_root = _tune_cache.resolve_root(
                mode=self._tune_cache_mode, cache_dir=self.cache_dir
            )
            self._tune_cache_resolved = True
        return self._tune_cache_mode, self._tune_cache_root

    def _tune_cache_for(self, mode):
        """Return the persistent tune cache when this session runs in ``mode``."""
        active, root = self._tune_cache_settings()
        if active != mode or root is None:
            return None
        if self._tune_cache is None:
            self._tune_cache = _tune_cache.TuneCache(
                root,
                arch=_tune_cache.arch_identity(self.device.identity),
                session=self.session_id,
            )
        return self._tune_cache

    def _tune_selection_made(self, job, obligation, selection):
        """Audit every selection; persist every replayable winner in record mode.

        Record mode captures exactly what frozen mode must replay: tuned and
        cached race winners *and* the heuristic ``default`` selections a
        no-race session (``autotune=False`` -- e.g. ``plan_decode_op``'s
        decode-cap sessions) actually made. Pinned overrides replay from the
        declaration itself, and frozen-mode replays never record.
        """
        mode, _root = self._tune_cache_settings()
        if mode == "off":
            return
        plan = obligation.request.plan
        self._tune_audit.append(
            {
                "request": obligation.request.name,
                "component": plan.component_id,
                "query": json.loads(_json(obligation.configuration.encoded_query)),
                "config": json.loads(_json(plan.contract.config_payload(selection.config))),
                "source": selection.source,
            }
        )
        if mode == "record" and selection.source in ("tuned", "cached", "default"):
            job._record_tune(obligation, selection)

    def _close_tune_cache(self):
        """Publish staged records and the audit manifest.

        Runs at job boundaries as well as session close: ``plan_decode_op``
        sessions are never closed, so their heuristic selections must still
        reach disk when their preparation job completes (or unwinds).
        """
        from ._tune_cache import audit_path, write_audit

        if self._tune_cache is not None:
            self._tune_cache.flush()
        mode, root = self._tune_cache_settings()
        if mode != "off" and root is not None and self._tune_audit:
            write_audit(
                audit_path(root, self.session_id),
                self._tune_audit,
                session=self.session_id,
            )

    def _gpu_scope(self):
        """Return the CUDA device context for preparation operations."""
        if self.device.ordinal is None:
            return nullcontext()
        import torch

        return torch.cuda.device(self.device.ordinal)

    def _synchronize(self):
        """Synchronize the preparation device when it is CUDA-backed."""
        if self.device.ordinal is not None:
            import torch

            torch.cuda.synchronize(self.device.ordinal)

    def begin(self, requests, *, autotune=None, selected_configs=None):
        """Start a job, optionally replacing scalar plans' selected configurations.

        ``selected_configs`` maps original scalar plans to validated configurations.
        Choices apply to declaration-equivalent aliases in this batch, preserve
        plan handles, and cannot replace an explicit declaration override or
        supply a measured finalist family. Composite children are scalar plans.
        """
        self._check_thread()
        if self._job is not None:
            raise RuntimeError("one preparation job may be active")
        pending = self._pending_collective_barrier
        if pending is not None:
            if not pending.is_set():
                raise RuntimeError("previous collective barrier is still running")
            self._pending_collective_barrier = None
        if autotune is not None and type(autotune) is not bool:
            raise TypeError("job autotune override must be boolean or None")
        requests = tuple(requests)
        _topological(requests)
        selected_configs = dict(selected_configs or {})
        if selected_configs:
            if self.state == "FROZEN":
                raise KernelResolutionFrozenError(
                    "selected configurations cannot change after session freeze"
                )
            expanded, _ = _expand_requests(requests)
            if not selected_configs.keys() <= {item.plan for item in expanded}:
                raise ValueError(
                    "selected configurations must name scalar plans in this batch"
                )
            for group in _coalesce_requests(expanded):
                choices = [
                    (item.plan, selected_configs[item.plan])
                    for item in group
                    if item.plan in selected_configs
                ]
                if not choices:
                    continue
                plan, config = choices[0]
                payload = plan.contract.config_payload(config)
                if any(
                    other.contract.config_payload(value) != payload
                    for other, value in choices[1:]
                ):
                    raise ValueError(
                        "declaration-equivalent plans selected different configurations"
                    )
                for item in group:
                    plan = item.plan
                    plan.contract.configure(
                        plan.query,
                        device=self.device.identity,
                        override=config,
                        search=False,
                    )
                    if (
                        plan.override is not None
                        and plan.contract.config_payload(plan.override) != payload
                    ):
                        raise ValueError(
                            "selected configuration conflicts with the plan override"
                        )
                    if item.finalist_policy is not None:
                        raise ValueError(
                            "selected configurations do not supply measured finalist families"
                        )
                    selected_configs[plan] = config
        if self.state == "FROZEN":
            for request in requests:
                if any(plan.prepared is None for plan in _plans_of(request)):
                    raise KernelResolutionFrozenError(
                        "new preparation obligation after session freeze"
                    )
        job = PreparationJob(
            self,
            requests,
            autotune=(self.autotune if autotune is None else autotune)
            and self._autotune_effort is not AutotuneEffort.OFF,
            selected_configs=selected_configs,
        )
        self._scratch_limit_locked = True
        self._job = job
        if self.state != "FROZEN":
            self.state = "PREPARING"
        return job

    def prepare(
        self,
        requests,
        *,
        coordinator=None,
        progress=None,
        autotune=None,
        selected_configs=None,
    ):
        requests = tuple(requests)
        if coordinator is None and any(
            request.collective is not None for request in requests
        ):
            raise ValueError("collective preparation requires a coordinator")
        job = self.begin(requests, autotune=autotune, selected_configs=selected_configs)
        authorization = {}
        while True:
            state = job.advance(**authorization)
            if progress is not None:
                progress(state)
            if state.done:
                return job.result()
            if state.ready_collective_tuning is not None:
                authorization = {"collective_tuning": coordinator(state)}
            elif state.ready_cache is not None or state.ready_tuning:
                if coordinator is None:
                    raise ValueError("sharded preparation requires a coordinator")
                key = "cache" if state.ready_cache is not None else "tuning"
                authorization = {key: coordinator(state)}
            else:
                authorization = {
                    "collective_key": coordinator(state)
                    if coordinator is not None
                    else None
                }
            if state.pending_compilation and self._pool is not None:
                self._pool.wait_for_progress()

    def candidate_memory_envelope(self, requests):
        """Return the largest scratch envelope across eligible configurations."""
        self._check_thread()
        if self._job is not None:
            raise RuntimeError("cannot inspect memory while preparation is active")
        expanded, _ = _expand_requests(tuple(requests))
        requirements = []
        for requests in _coalesce_requests(expanded):
            plan = requests[0].plan
            configuration = plan.contract.configure(
                plan.query,
                device=self.device.identity,
                override=plan.override,
                search=self._autotune_effort is not AutotuneEffort.OFF,
            )
            configs = (
                (configuration.pinned,)
                if configuration.pinned is not None
                else (configuration.default,)
                if self._autotune_effort is AutotuneEffort.OFF
                else (
                    configuration.default,
                    *(config for _, config in plan.contract.iterate(configuration)),
                )
            )
            seen = set()
            admitted = 0
            for config in configs:
                payload = plan.contract.config_payload(config)
                if payload in seen:
                    continue
                seen.add(payload)
                try:
                    requirement = self._candidate_memory(requests, config)
                except CandidateResourceIneligibleError:
                    if (
                        self._scratch_limit is None
                        or configuration.pinned is not None
                        or self._autotune_effort is AutotuneEffort.OFF
                    ):
                        raise
                    continue
                admitted += 1
                requirements.append(requirement)
            if not admitted:
                raise CandidateResourceIneligibleError(
                    f"no eligible configurations for {plan.component_id} within "
                    f"the session scratch limit of {self._scratch_limit} bytes"
                )
        return MemoryRequirements.sequential(requirements)

    def cancel_tuning(self):
        self._stop.set()
        pool = self._pool
        if pool is not None:
            pool.wake()

    def _compiler(self):
        if self._pool is None:
            if (
                self.device.ordinal is None
                or self.device.identity is None
                or self.device.uuid is None
                or self.device.max_shared_memory_per_block is None
                or self.device.max_shared_memory_per_multiprocessor is None
            ):
                raise RuntimeError("kernel compilation requires a CUDA device")
            self._pool = CompilePool(
                device_ordinal=self.device.ordinal,
                compute_capability=self.device.identity.compute_capability,
                device_uuid=self.device.uuid,
                product_name=self.device.identity.product_name,
                sm_count=self.device.identity.sm_count,
                max_shared_memory_per_block=self.device.max_shared_memory_per_block,
                max_shared_memory_per_multiprocessor=(
                    self.device.max_shared_memory_per_multiprocessor
                ),
                workers=self.compile_workers,
            )
        return self._pool

    def _drain(self):
        if self._pool is not None:
            if self._stop.is_set():
                pool, self._pool = self._pool, None
                summary = pool.summary()
                pool.close(terminate=True)
                return summary
            while self._pool.pending:
                yield "compile"
            summary = self._pool.summary()
            self._pool.close()
            self._pool = None
            return summary
        return None

    def _release_payload(self, plan, prepared):
        if plan in prepared.users:
            prepared.users.remove(plan)
        if plan._prepared is prepared:
            plan._clear()
        if plan._prepared is None:
            self._plans.pop(plan, None)
        if plan.shared:
            key = _declaration_key(plan)
            if self._shared.get(key) is plan and plan.prepared is not prepared:
                if plan.prepared is None:
                    replacement = next(
                        (user for user in prepared.users if user.shared), None
                    )
                    if replacement is None:
                        self._shared.pop(key)
                    else:
                        self._shared[key] = replacement
        if prepared.users or prepared.closed:
            return
        prepared.closed = True
        closers, prepared.closers = prepared.closers, ()
        try:
            _close_all(closers)
        finally:
            prepared.retained = None
            prepared.state = None
            prepared.owners = ()

    def release(self, plan):
        """Drop a plan's prepared state; destroy graphs that replay it first."""
        self.release_many((plan,))

    def release_many(self, plans) -> None:
        """Release a batch after graph teardown, then reclaim unused programs once."""
        self._check_thread()
        if self._job is not None:
            raise RuntimeError("cannot release resources during preparation")
        pending = {}

        def collect(plan):
            if plan in pending:
                return
            pending[plan] = plan._prepared
            if isinstance(plan, _CompositePlan):
                for child in plan.variants.values():
                    collect(child)

        for plan in plans:
            collect(plan)
        closers = [
            lambda plan=plan, prepared=prepared: self._release_payload(plan, prepared)
            for plan, prepared in pending.items()
            if prepared is not None
        ]
        if not closers:
            return
        try:
            _close_all((self._synchronize, *closers))
        finally:
            _close_all((self._reclaim_programs,))

    def race_contexts(
        self,
        candidates,
        *,
        collective_exchange=None,
        collective_ranks=None,
        max_candidates=16,
        rounds=None,
        samples=None,
    ):
        """Race caller-captured complete computations over prepared plan tuples."""
        from .context import race_contexts

        return race_contexts(
            self,
            candidates,
            collective_exchange=collective_exchange,
            collective_ranks=collective_ranks,
            max_candidates=max_candidates,
            rounds=rounds,
            samples=samples,
        )

    def _reclaim_programs(self):
        timing = PreparationTiming("cleanup", rank=self._tuning_rank)
        keep = frozenset(
            program
            for plan in self._plans
            if plan.prepared is not None
            for program in plan.prepared.programs
        )
        with timing.span("program_reclamation"):
            removed = evict_unretained(keep)
            gc.collect()
        timing.record("complete", evicted_entries=removed)

    @contextmanager
    def capture(self):
        """Refuse kernel resolution for the duration of a CUDA graph capture."""
        self._check_thread()
        if self._job is not None or self._pool is not None:
            raise RuntimeError("capture requires drained preparation")
        with kernel_resolution_guard("prepared CUDA graph capture"):
            yield

    def freeze(self):
        """Forbid later kernel resolution after all preparation work drains."""
        self._check_thread()
        if self._job is not None or self._pool is not None:
            raise RuntimeError("freeze requires drained preparation")
        if self._guard is None:
            self._guard = kernel_resolution_guard("prepared serving session")
            self._guard.__enter__()
        self.state = "FROZEN"

    def close(self):
        """Release preparation resources and mark the session closed."""
        if self.state == "CLOSED":
            return
        self._check_thread()
        synchronize = bool(self._plans) or self._job is not None
        closers = []
        if self._job is not None:
            closers.append(self._job.close)
        if self._pool is not None:
            pool, self._pool = self._pool, None
            pool.cancel_optional()
            closers.append(pool.close)
        for plan in reversed(tuple(self._plans)):
            prepared = plan._prepared
            if prepared is not None:
                closers.append(
                    lambda plan=plan, prepared=prepared: self._release_payload(
                        plan, prepared
                    )
                )
        if self._guard is not None:
            guard, self._guard = self._guard, None
            closers.append(lambda: guard.__exit__(None, None, None))
        if self._cache is not None:
            closers.append(self._cache.flush)
        if self._tune_cache is not None or self._tune_audit:
            closers.append(self._close_tune_cache)
        if self._race_cache is not None:
            cache, self._race_cache = self._race_cache, None
            closers.append(cache.close)
        self.state = "CLOSED"
        try:
            _close_all(((self._synchronize,) if synchronize else ()) + tuple(closers))
        finally:
            self._plans.clear()
            self._shared.clear()
            _close_all((self._reclaim_programs,))

    def __enter__(self):
        """Enter the session after validating the owning thread."""
        self._check_thread()
        return self

    def __exit__(self, kind, value, traceback):
        """Close session resources and preserve a body exception if cleanup fails."""
        try:
            self.close()
        except BaseException as error:
            if value is None:
                raise
            value.add_note(f"session cleanup failed: {error!r}")


class PreparationJob:
    def __init__(self, session, requests, *, autotune, selected_configs=None):
        """Initialize state for one bounded sequence of preparation requests."""
        self.session, self.requests = session, requests
        self.autotune = autotune
        self._selected_configs = dict(selected_configs or {})
        self._steps = self._run()
        self._blocked = None
        self._result = None
        self._error = None
        self._closed = False
        self._temporaries = []
        self._new_plans = []
        self._benchmark_closers = []
        self._cache_hits = self._benchmarked = self._overlaps = 0
        self._restored = self._rejected = 0
        self._all_programs = set()
        self._dependency_programs = {}
        self._coverage = {}
        self._finalists = {}
        self._described_candidates = {}
        self._described_jobs = {}
        self._discovery_locks = set()
        self._available_programs = set()
        self._pipeline = ()
        self._selections = {}
        self._benchmark_calls = {}
        self._plans_by_name = {}
        self._started = time.monotonic()
        self._timing = PreparationTiming("job", rank=session._tuning_rank)
        self._program_cache = PreparationProgramCache(self._timing)
        self._last_advance_end = None
        self._phase = "planning"
        self._active_request = None
        self._total_requests = len(requests)
        self._completed_requests = 0
        self._planned_requests = 0
        self._candidates_described = 0
        self._selection_counts = {}
        self._candidate_count = self._candidates_prepared = 0
        self._global_candidate_count = 0
        self._total_candidates = None
        self._candidate_sharded = False
        self._batch_index = self._batch_candidates = 0
        self._completed_rounds = self._total_rounds = self._active_count = 0
        self._latest_round_us = ()
        self._race_eviction = None
        self._compilations = 0
        self._stack_limit = None
        self._direct_ready = False
        if session.device.ordinal is not None:
            from b12x._lib.program_cache import stack_limit_bytes

            with session._gpu_scope():
                self._stack_limit = stack_limit_bytes()

    def _progress(
        self,
        running,
        pending_compilation,
        ready_collectives,
        done,
        ready_tuning=(),
        ready_cache=None,
        ready_collective_tuning=None,
    ):
        """Build a preparation progress snapshot from the live job state."""
        active_compilations = 0
        if self.session._pool is not None:
            summary = self.session._pool.summary()
            self._compilations = summary.cute_compilations + summary.triton_compilations
            active_compilations = self.session._pool.active_compilations
        phase = (
            "ready"
            if done
            else "waiting for ranks"
            if ready_collectives
            or ready_tuning
            or ready_cache
            or ready_collective_tuning
            else "compiling"
            if pending_compilation and self._phase != "planning programs"
            else self._phase
        )
        request = self._active_request
        return PreparationProgress(
            running,
            pending_compilation,
            ready_collectives,
            done,
            phase=phase,
            component_id="" if request is None else request.plan.component_id,
            request_name="" if request is None else request.name,
            completed_requests=self._completed_requests,
            total_requests=self._total_requests,
            candidate_count=self._candidate_count,
            candidates_prepared=self._candidates_prepared,
            measured_candidates=self._benchmarked,
            completed_rounds=self._completed_rounds,
            total_rounds=self._total_rounds if phase == "autotuning" else 0,
            active_count=self._active_count,
            latest_round_us=self._latest_round_us,
            cache_hits=self._cache_hits,
            compilations=self._compilations,
            active_compilations=active_compilations,
            elapsed_seconds=time.monotonic() - self._started,
            tuning_stopped=self._warmup_only and self.session._stop.is_set(),
            ready_tuning=tuple(ready_tuning),
            candidate_sharded=self._candidate_sharded,
            batch_index=self._batch_index,
            batch_candidates=self._batch_candidates,
            tuning_rank=self.session._tuning_rank,
            total_candidates=self._total_candidates,
            global_candidate_count=self._global_candidate_count,
            selection_counts=tuple(sorted(self._selection_counts.items())),
            ready_cache=ready_cache,
            planned_requests=self._planned_requests,
            candidates_described=self._candidates_described,
            restored_candidates=self._restored,
            rejected_candidates=self._rejected,
            ready_collective_tuning=ready_collective_tuning,
        )

    def _collective_barrier(self, requirement: CollectiveRequirement) -> None:
        """Convert a shared collective authorization into a launch lockstep.

        All participant ranks were authorized for the same requirement in the
        same exchange round, but the local work before any rank launches is
        unbounded (autotune races). The embedder-supplied barrier waits for
        every participant rank to enter before this rank proceeds to the
        launch, so the kernel-side spin window only covers launch jitter.
        """
        barrier = self.session.collective_barrier
        completed = threading.Event()
        errors = []

        def wait_for_peers():
            """Run the embedder barrier and publish its terminal state."""
            try:
                barrier(requirement.key, requirement.ranks)
            except BaseException as error:
                errors.append(error)
            finally:
                completed.set()

        try:
            thread = threading.Thread(target=wait_for_peers, daemon=True)
            self.session._pending_collective_barrier = completed
            try:
                thread.start()
            except BaseException:
                self.session._pending_collective_barrier = None
                raise
            if not completed.wait(self.session._barrier_timeout):
                raise CollectiveBarrierTimeout(
                    requirement.key,
                    requirement.ranks,
                    timeout=self.session._barrier_timeout,
                )
            if errors:
                raise errors[0]
        except CollectiveBarrierTimeout:
            raise
        except Exception as error:
            raise RuntimeError(
                f"collective barrier for {requirement.key!r} failed: {error}"
            ) from error

    def advance(
        self, *, collective_key=None, tuning=None, cache=None, collective_tuning=None
    ):
        """Advance preparation until work completes or reaches a readiness boundary.

        For each ready tuning key without a finalist policy, ``tuning`` contains
        its consolidated winner. A key requesting finalists instead requires
        every shard's contribution in ``ranks`` order, including empty shards.
        Contributions may be interleaved across keys while preserving that
        per-key rank order. The session selects the ordinary winner separately
        from the globally bounded per-family finalists.
        """
        started = time.perf_counter()
        if self._last_advance_end is not None:
            self._timing.add("between_advances", started - self._last_advance_end)
        try:
            with self._program_cache.activate():
                return self._advance(
                    collective_key=collective_key,
                    tuning=tuning,
                    cache=cache,
                    collective_tuning=collective_tuning,
                )
        finally:
            if self._result is not None or self._closed:
                self._program_cache.clear()
            self._timing.add("advance", time.perf_counter() - started)
            self._timing.record(
                "complete" if self._result is not None else "progress",
                periodic=self._result is None and self._error is None,
                phase=self._phase,
                failed=self._error is not None,
                request=None
                if self._active_request is None
                else self._active_request.name,
                completed_requests=self._completed_requests,
            )
            self._last_advance_end = time.perf_counter()

    def _advance(
        self, *, collective_key=None, tuning=None, cache=None, collective_tuning=None
    ):
        """Run work until it reaches a readiness boundary or time slice."""
        self.session._check_thread()
        if self._error is not None:
            raise self._error
        if self._result is not None:
            return self._progress(False, False, (), True)
        if self._closed:
            raise RuntimeError("preparation job is closed")
        if self._locally_stopped and self.session._pool is not None:
            pool, self.session._pool = self.session._pool, None
            pool.cancel_optional()
            pool.close(terminate=True)
        send_value = None
        if isinstance(self._blocked, CollectiveTuningRequirement):
            if collective_tuning is None:
                return self._progress(
                    False, False, (), False, ready_collective_tuning=self._blocked
                )
            peers = tuple(collective_tuning)
            expected = self._blocked
            if len(peers) != len(expected.ranks) or any(
                not isinstance(item, CollectiveTuningRequirement)
                or (item.key, item.ranks, item.step)
                != (expected.key, expected.ranks, expected.step)
                for item in peers
            ):
                raise ValueError("collective tuning exchange differs from ready work")
            self._blocked = None
            send_value = peers
        elif collective_tuning is not None:
            raise ValueError("collective tuning exchange does not match ready work")
        elif isinstance(self._blocked, TuningCacheRequirement):
            if cache is None:
                return self._progress(
                    False, False, (), False, ready_cache=self._blocked
                )
            snapshots = tuple(cache)
            if not snapshots and self.session._stop.is_set():
                pass
            elif len(snapshots) != len(self._blocked.ranks) or any(
                not isinstance(item, TuningCacheRequirement)
                or item.ranks != self._blocked.ranks
                for item in snapshots
            ):
                raise ValueError(
                    "tuning cache agreement does not match participating ranks"
                )
            self._blocked = None
            send_value = snapshots
        elif cache is not None:
            raise ValueError("tuning cache agreement does not match ready work")
        elif isinstance(self._blocked, CollectiveRequirement):
            if collective_key != self._blocked.key:
                return self._progress(False, False, (self._blocked,), False)
            if self.session.collective_barrier is not None:
                try:
                    self._collective_barrier(self._blocked)
                except BaseException as error:
                    self._error = error
                    try:
                        self.close()
                    except BaseException as cleanup:
                        error.add_note(f"job cleanup failed: {cleanup!r}")
                    raise
            self._blocked = None
        elif isinstance(self._blocked, _TuningBatch):
            required = self._blocked.contributions
            if tuning is None:
                return self._progress(False, False, (), False, required)
            contributions = (
                (tuning,) if isinstance(tuning, TuningRequirement) else tuple(tuning)
            )
            expected = {item.key: item for item in required}
            grouped = {key: [] for key in expected}
            if not contributions and self.session._stop.is_set():
                pass
            else:
                for item in contributions:
                    if (
                        not isinstance(item, TuningRequirement)
                        or item.key not in expected
                        or item.ranks != expected[item.key].ranks
                        or item.finalist_policy != expected[item.key].finalist_policy
                    ):
                        raise ValueError(
                            "tuning consolidation does not match pending races"
                        )
                    grouped[item.key].append(item)
                for key, items in grouped.items():
                    requirement = expected[key]
                    if requirement.finalist_policy is None:
                        if len(items) != 1 or items[0].assignment is None:
                            raise ValueError(
                                "tuning consolidation does not match pending races"
                            )
                    else:
                        if len(items) != len(requirement.ranks):
                            raise ValueError(
                                "finalist consolidation requires every tuning shard"
                            )
                        for rank_index, item in enumerate(items):
                            indices = [
                                record.candidate_index for record in item.finalists
                            ]
                            if item.candidate_index is not None:
                                indices.append(item.candidate_index)
                            if any(
                                index % len(items) != rank_index for index in indices
                            ):
                                raise ValueError(
                                    "finalist contributions must follow tuning rank order"
                                )
            self._blocked = None
            send_value = contributions
        elif isinstance(self._blocked, TuningRequirement):
            if tuning is None:
                return self._progress(False, False, (), False, (self._blocked,))
            if (
                not isinstance(tuning, TuningRequirement)
                or tuning.key != self._blocked.key
                or tuning.ranks != self._blocked.ranks
                or tuning.assignment is None
            ):
                raise ValueError("tuning authorization does not match ready work")
            self._blocked = None
            send_value = tuning
        elif collective_key is not None:
            raise ValueError("collective authorization does not match ready work")
        elif tuning is not None:
            raise ValueError("tuning authorization does not match ready work")
        try:
            deadline = time.monotonic() + _ADVANCE_SECONDS
            while True:
                started = time.perf_counter()
                try:
                    signal = self._steps.send(send_value)
                finally:
                    self._timing.add(self._phase, time.perf_counter() - started)
                send_value = None
                if isinstance(signal, CollectiveTuningRequirement):
                    self._blocked = signal
                    return self._progress(
                        False, False, (), False, ready_collective_tuning=signal
                    )
                if isinstance(signal, TuningCacheRequirement):
                    self._blocked = signal
                    return self._progress(False, False, (), False, ready_cache=signal)
                if isinstance(signal, CollectiveRequirement):
                    self._blocked = signal
                    return self._progress(False, False, (signal,), False)
                if isinstance(signal, _TuningBatch):
                    self._blocked = signal
                    return self._progress(False, False, (), False, signal.contributions)
                if isinstance(signal, TuningRequirement):
                    self._blocked = signal
                    return self._progress(False, False, (), False, (signal,))
                if signal == "compile":
                    return self._progress(False, True, (), False)
                if time.monotonic() >= deadline:
                    break
        except StopIteration as finished:
            self._result = finished.value
            self.session._job = None
            if self.session.state != "FROZEN":
                self.session.state = "READY"
            return self._progress(False, False, (), True)
        except BaseException as error:
            self._error = error
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note(f"job cleanup failed: {cleanup!r}")
            raise
        return self._progress(True, False, (), False)

    def result(self):
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise RuntimeError("preparation is not complete")
        return self._result

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._release_race_resources()
        closers = [self._steps.close, *reversed(self._temporaries)]
        self._temporaries.clear()
        closers.extend(lock.release for lock in self._discovery_locks)
        self._discovery_locks.clear()
        if self.session._cache is not None:
            closers.append(self.session._cache.flush)
        if self.session._tune_cache is not None or self.session._tune_audit:
            closers.append(self.session._close_tune_cache)
        if self._result is None:
            closers.extend(reversed(self._benchmark_closers))
            self._benchmark_closers.clear()
            for plan in reversed(self._new_plans):
                prepared = plan._prepared
                if prepared is not None:
                    closers.append(
                        lambda plan=plan, prepared=prepared: (
                            self.session._release_payload(plan, prepared)
                        )
                    )
        if self.session._pool is not None:
            pool, self.session._pool = self.session._pool, None
            pool.cancel_optional()
            closers.append(lambda: pool.close(terminate=self._error is not None))
        self.session._job = None
        try:
            _close_all(closers)
        finally:
            self._program_cache.clear()
            if self._result is None:
                _close_all((self.session._reclaim_programs,))

    def _choice_key(self, obligation, selections):
        if obligation.candidate_program_identity is None:
            return None
        request, configuration = obligation.request, obligation.configuration
        dependencies = []
        for name in request.dependencies:
            if name not in selections:
                return None
            selection, contract = selections[name]
            dependencies.append(
                {
                    "component": selection.component_id,
                    "query": selection.query.to_dict(),
                    "config": contract.config_payload(selection.config).to_dict(),
                    "semantic_version": contract.semantic_version,
                }
            )
        contract = request.plan.contract
        identity = {
            "component": contract.component_id,
            "query_schema": contract.query_schema_version,
            "config_schema": contract.config_schema_version,
            "semantic_version": contract.semantic_version,
            "candidate_contract_version": contract.candidate_contract_version,
            "query": configuration.encoded_query.to_dict(),
            "invocation": request.plan.invocation.to_dict(),
            "pin": None
            if configuration.pinned is None
            else contract.config_payload(configuration.pinned).to_dict(),
            "dependencies": dependencies,
            "candidate_programs": obligation.candidate_program_identity,
            "measurement": {
                "rounds": self.session.rounds,
                "samples": self.session.samples,
            },
        }
        if request.finalist_policy is not None:
            policy = request.finalist_policy
            identity["finalist_policy"] = {
                "version": 2,
                "family_fields": policy.family_fields,
                "per_family": policy.per_family,
                "max_total": policy.max_total,
            }
        return digest(identity)

    def _selection(self, obligation, config, source, assignment=None):
        if self.session._scratch_limit is not None:
            self.session._candidate_memory(
                obligation.requests or (obligation.request,), config
            )
        selection = Selection(
            obligation.request.plan.component_id,
            obligation.configuration.encoded_query,
            config,
            source,
            assignment,
        )
        self.session._tune_selection_made(self, obligation, selection)
        return selection

    def _tune_provenance(self, obligation, selections):
        """Build the kernel + shape + arch version record for one obligation."""
        contract = obligation.request.plan.contract
        dependencies = []
        for name in obligation.request.dependencies:
            if name not in selections:
                return None
            selection, dep_contract = selections[name]
            dependencies.append(
                {
                    "component": selection.component_id,
                    "query": selection.query.to_dict(),
                    "config": dep_contract.config_payload(selection.config).to_dict(),
                    "semantic_version": dep_contract.semantic_version,
                }
            )
        configuration = obligation.configuration
        return _tune_cache.build_provenance(
            contract,
            encoded_query=configuration.encoded_query,
            invocation=obligation.request.plan.invocation,
            pin=None
            if configuration.pinned is None
            else contract.config_payload(configuration.pinned),
            dependencies=dependencies,
            device_identity=self.session.device.identity,
        )

    def _record_tune(self, obligation, selection):
        """Stage one winning selection for the persistent tune cache."""
        cache = self.session._tune_cache_for("record")
        if cache is None:
            return
        provenance = self._tune_provenance(obligation, self._selections or {})
        if provenance is None:
            logger.warning(
                "tune cache: not recording %s (%s); a dependency selection is "
                "unresolved",
                obligation.request.name,
                obligation.request.plan.component_id,
            )
            return
        contract = obligation.request.plan.contract
        cache.record(
            provenance,
            contract.config_payload(selection.config),
            assignment=selection.assignment,
            coverage=dict(getattr(obligation, "coverage", None) or {}),
            extra={
                "request": obligation.request.name,
                "source": selection.source,
                "candidate_programs": obligation.candidate_program_identity,
                "session": self.session.session_id,
            },
        )

    def _frozen_tune_selection(self, obligation, selections):
        """N7 frozen replay: resolve the offline winner with zero racing.

        Only single-product, heuristic-off, or pinned declarations bypass the
        cache; every tunable obligation must resolve from the cache, and a
        stale or missing record raises loudly instead of degrading silently.
        """
        cache = self.session._tune_cache_for("frozen")
        if cache is None:
            return None
        configuration = obligation.configuration
        if configuration.pinned is not None:
            return None
        plan = obligation.request.plan
        space = plan.contract.parameter_space(
            plan.query, self.session.device.identity
        )
        if (
            space is None
            or space.autotune_effort is AutotuneEffort.OFF
            or all(len(knob.values) == 1 for knob in space.knobs)
        ):
            return None
        provenance = self._tune_provenance(obligation, selections)
        if provenance is None:
            logger.warning(
                "frozen tune cache: unresolved dependency for %s (%s); falling "
                "back to the heuristic default",
                obligation.request.name,
                plan.component_id,
            )
            return None
        record = cache.resolve(provenance)
        payload = FrozenMapping(record.get("assignment") or record["config"])
        config = plan.contract._lower(
            configuration.query, configuration.device, payload
        )
        requests = obligation.requests or (obligation.request,)
        requirement = self.session._candidate_memory(requests, config)
        baseline = self.session._candidate_memory(requests, configuration.default)
        if requirement.scratch_nbytes > baseline.scratch_nbytes:
            raise _tune_cache.TuneCacheError(
                f"frozen tune cache: {plan.component_id} configuration needs "
                f"{requirement.scratch_nbytes} scratch bytes, above the "
                f"{baseline.scratch_nbytes} prepared for the heuristic default"
            )
        return self._selection(obligation, config, "frozen", payload)

    def _validate_retained_scratch(self, obligation, prepared):
        if self.session._scratch_limit is None:
            return
        requests = obligation.requests or (obligation.request,)
        self.session._candidate_memory(requests, prepared.selection.config)
        plan = obligation.request.plan
        for item in prepared.finalists:
            config = plan.contract._lower(
                plan.query, self.session.device.identity, item.assignment
            )
            self.session._candidate_memory(requests, config)

    def _validate_finalists(self, obligation, records):
        policy = obligation.request.finalist_policy
        if policy is None:
            if records:
                raise ValueError("unrequested finalist metadata in tuning result")
            return ()
        records = tuple(
            item if isinstance(item, FinalistRecord) else FinalistRecord(**dict(item))
            for item in records
        )
        if not records or len(records) > policy.max_total:
            raise ValueError("finalist record count differs from its bounded policy")
        indices = set()
        families = {}
        for item in records:
            index = item.candidate_index
            if index >= len(obligation.candidates) or index in indices:
                raise ValueError(
                    "finalist candidate index is outside the legal catalog"
                )
            indices.add(index)
            assignment, config = obligation.candidates[index]
            if item.assignment != FrozenMapping(assignment):
                raise ValueError("finalist assignment differs from its candidate index")
            if self.session._scratch_limit is not None:
                self.session._candidate_memory(obligation.requests, config)
            family = FrozenMapping(
                {name: assignment[name] for name in policy.family_fields}
            )
            if item.family != family:
                raise ValueError("finalist family differs from its legal assignment")
            key = _json(family)
            families[key] = families.get(key, 0) + 1
            if families[key] > policy.per_family:
                raise ValueError("finalist family exceeds its retention bound")
        return tuple(
            sorted(
                records,
                key=lambda item: (
                    _json(item.family),
                    item.latency_us,
                    item.candidate_index,
                ),
            )
        )

    def _public_finalists(self, obligation, source):
        contract = obligation.request.plan.contract
        configuration = obligation.configuration
        return tuple(
            Finalist(
                selection=self._selection(
                    obligation,
                    contract._lower(
                        configuration.query, configuration.device, item.assignment
                    ),
                    source,
                    item.assignment,
                ),
                family=item.family,
                candidate_index=item.candidate_index,
                latency_us=item.latency_us,
            )
            for item in obligation.finalists
        )

    def _lookup(self, obligation, selections):
        if obligation.selection is not None:
            return
        obligation.key = self._choice_key(obligation, selections)
        if obligation.key is None:
            return
        with self._timing.span("cache_lookup"):
            cache = self.session._selection_cache()
            record = cache.get(obligation.key)
        if record is None:
            return
        if obligation.request.finalist_policy is not None and not record.get(
            "finalists"
        ):
            # A winner-only record does not establish family coverage.
            return
        configuration, contract = (
            obligation.configuration,
            obligation.request.plan.contract,
        )
        assignment = FrozenMapping(record["assignment"])
        config = contract._lower(configuration.query, configuration.device, assignment)
        obligation.selection = self._selection(obligation, config, "cached", assignment)
        obligation.coverage = dict(record["coverage"])
        obligation.finalists = self._validate_finalists(
            obligation, record.get("finalists", ())
        )
        self._cache_hits += 1

    def _alias_shared(self, obligation):
        """Install an equal shared declaration's payload; True when aliased."""
        plan = obligation.request.plan
        if not plan.shared:
            return False
        owner = self.session._shared.get(_declaration_key(plan))
        if owner is None or owner is plan or owner.prepared is None:
            return False
        prepared = owner.prepared
        if obligation.selection is not None and plan.contract.config_payload(
            prepared.selection.config
        ) != plan.contract.config_payload(obligation.selection.config):
            return False
        if prepared.finalist_policy != obligation.request.finalist_policy:
            return False
        if (
            prepared.selection.source == "default"
            and self.autotune
            and not self.session._stop.is_set()
        ):
            return False
        self._validate_retained_scratch(obligation, prepared)
        previous = plan._prepared
        prepared.users.append(plan)
        plan._install(prepared)
        if previous is not None and previous is not prepared:
            self.session._release_payload(plan, previous)
        self.session._plans[plan] = None
        self._new_plans.append(plan)
        obligation.selection = prepared.selection
        obligation.finalists = prepared.finalists
        obligation.ready = True
        return True

    @property
    def _locally_stopped(self):
        request = self._active_request
        if (
            request is not None
            and request.collective is not None
            and request.plan.contract.collective_signature is not None
        ):
            # Collective cancellation is agreed at drained group boundaries.
            return False
        return self.session._stop.is_set()

    @property
    def _warmup_only(self):
        return not self.session.cache_only and (
            not self.autotune or self._locally_stopped
        )

    def _direct_compilation(self):
        if self._direct_ready:
            return
        if self.session._pool is not None:
            pool, self.session._pool = self.session._pool, None
            pool.close(terminate=True)
        from b12x._lib.compile_plan import evict_planning_artifacts

        evict_planning_artifacts()
        self._direct_ready = True

    def _configure(self, groups):
        obligations, enumerations, requirements = [], {}, []
        for requests in groups:
            if len({item.finalist_policy for item in requests}) != 1:
                raise ValueError("coalesced requests must agree on finalist policy")
            request = next(
                (
                    item
                    for item in requests
                    if item.plan.prepared is not None
                    and item.plan.prepared.finalist_policy == item.finalist_policy
                    and (
                        not self.autotune
                        or self._locally_stopped
                        or item.plan.selection.source != "default"
                    )
                ),
                None,
            )
            if request is None:
                request = next(
                    (item for item in requests if item.benchmark_call is not None),
                    requests[0],
                )
            requests = (request, *(item for item in requests if item is not request))
            self._active_request = request
            self._candidate_count = self._candidates_prepared = 0
            self._candidate_sharded = False
            plan = request.plan
            caps_device = plan._device
            if caps_device is None:
                caps_device = getattr(plan.query, "device", None)
            if caps_device is not None:
                import torch

                resolved = torch.device(caps_device)
                if (
                    resolved.type != "cuda"
                    or resolved.index != self.session.device.ordinal
                ):
                    raise ValueError("declaration and session devices differ")
            configuration = plan.contract.configure(
                plan.query,
                device=self.session.device.identity,
                override=self._selected_configs.get(plan, plan.override),
                search=not self._warmup_only,
            )
            if request.finalist_policy is not None and not self._locally_stopped:
                if (
                    configuration.space is None
                    or self._warmup_only
                    or configuration.space.autotune_effort is AutotuneEffort.OFF
                ):
                    raise ValueError(
                        "finalists require an enabled ordinary tuning race"
                    )
                available = {knob.name for knob in configuration.space.knobs}
                if set(request.finalist_policy.family_fields) - available:
                    raise ValueError(
                        "finalist family field is outside the plan's tuning space"
                    )
            obligation = _Obligation(request, configuration, requests=requests)
            obligations.append(obligation)
            prepared = plan.prepared
            if plan in self._selected_configs:
                obligation.selection = self._selection(
                    obligation, configuration.pinned, "override"
                )
                if prepared is not None and plan.contract.config_payload(
                    prepared.selection.config
                ) == plan.contract.config_payload(obligation.selection.config):
                    self._validate_retained_scratch(obligation, prepared)
                    obligation.selection = prepared.selection
                    obligation.ready = True
            elif prepared is not None and (
                prepared.finalist_policy == request.finalist_policy
                and (
                    not self.autotune
                    or self._locally_stopped
                    or prepared.selection.source != "default"
                )
            ):
                self._validate_retained_scratch(obligation, prepared)
                obligation.selection = prepared.selection
                obligation.finalists = prepared.finalists
                obligation.ready = True
            elif self._alias_shared(obligation):
                pass
            elif configuration.pinned is not None:
                obligation.selection = self._selection(
                    obligation, configuration.pinned, "override"
                )
            elif self._warmup_only or (
                configuration.space is not None
                and configuration.space.autotune_effort is AutotuneEffort.OFF
            ):
                obligation.selection = self._selection(
                    obligation, configuration.default, "default"
                )
            else:
                single_product = all(
                    len(knob.values) == 1 for knob in configuration.space.knobs
                )
                if (
                    obligation.selection is None
                    and not single_product
                    and not self.session.cache_only
                    and (not self.autotune or self._locally_stopped)
                ):
                    obligation.selection = self._selection(
                        obligation, configuration.default, "default"
                    )
                if obligation.selection is None:
                    declaration = _declaration_key(plan)
                    enumerated = enumerations.get(declaration)
                    if enumerated is not None:
                        candidates, coverage, payloads = enumerated
                        obligation.candidates.extend(candidates)
                        obligation.payloads.update(payloads)
                        complete = True
                    else:
                        iterator = plan.contract.iterate(configuration)
                        while not iterator.done:
                            if (
                                not single_product
                                and self._locally_stopped
                                and not self.session.cache_only
                            ):
                                break
                            try:
                                candidate = iterator.step()
                            except StopIteration:
                                break
                            if candidate is not None:
                                obligation.candidates.append(candidate)
                            if iterator.cartesian_count % 256 == 0:
                                yield "metadata"
                        complete = iterator.done
                        coverage = {
                            "cartesian_count": iterator.cartesian_count,
                            "legal_count": iterator.legal_count,
                            "effective_count": iterator.effective_count,
                            "measured_count": 0,
                        }
                        # Equal declarations enumerate to the same candidates; an
                        # interrupted enumeration describes the stop, not the space.
                        obligation.payloads.update(iterator.payloads)
                        if complete:
                            enumerations[declaration] = (
                                tuple(obligation.candidates),
                                coverage,
                                iterator.payloads,
                            )
                    if self.session._scratch_limit is not None:
                        eligible = []
                        for candidate in obligation.candidates:
                            try:
                                self.session._candidate_memory(requests, candidate[1])
                            except CandidateResourceIneligibleError:
                                continue
                            eligible.append(candidate)
                        excluded = len(obligation.candidates) - len(eligible)
                        obligation.candidates = eligible
                        coverage = dict(coverage)
                        coverage["scratch_ineligible_count"] = excluded
                        obligation.coverage = dict(coverage)
                    self._global_candidate_count = self._candidate_count = len(
                        obligation.candidates
                    )
                    if complete and not obligation.candidates:
                        if self.session._scratch_limit is not None and excluded:
                            raise CandidateResourceIneligibleError(
                                f"no eligible configurations for {plan.component_id} "
                                f"within the session scratch limit of "
                                f"{self.session._scratch_limit} bytes"
                            )
                        raise ValueError(
                            f"no eligible configurations for {plan.component_id}"
                        )
                    if (
                        complete
                        and len(obligation.candidates) == 1
                        and request.finalist_policy is None
                    ):
                        assignment, config = obligation.candidates[0]
                        obligation.selection = self._selection(
                            obligation, config, "fixed", assignment
                        )
                    elif self._locally_stopped and not self.session.cache_only:
                        obligation.selection = self._selection(
                            obligation, configuration.default, "default"
                        )
                    if obligation.selection is None:
                        obligation.coverage = dict(coverage)
            for item in requests:
                if (
                    item.plan.prepared is not None
                    and item.plan.selection == obligation.selection
                    or item is not request
                    and item.plan.shared
                    and plan.shared
                ):
                    continue
                config = (
                    configuration.default
                    if obligation.selection is None
                    else obligation.selection.config
                )
                with _plan_scope(item.plan):
                    requirements.append(
                        item.plan._memory_requirements(config, self.session.device)
                    )
            self._planned_requests += 1
            yield "metadata"
        MemoryRequirements.sequential(requirements)
        rank_index = self.session._tuning_ranks.index(self.session._tuning_rank)
        rank_count = len(self.session._tuning_ranks)
        for obligation in obligations:
            if obligation.selection is None:
                obligation.planned_candidates = len(
                    obligation.candidates
                    if obligation.request.collective is not None
                    else obligation.candidates[rank_index::rank_count]
                )
        self._total_candidates = sum(item.planned_candidates for item in obligations)
        return obligations

    def _describe_candidates(self, obligation):
        """Complete program discovery for an obligation, waiting as needed."""
        self._phase = "planning programs"
        yield from self._advance_discovery(obligation, wait=True)

    def _advance_discovery(self, obligation, *, wait):
        """Advance one obligation's discovery; with ``wait`` it runs to completion.

        Without ``wait`` the step never blocks on the pool or another rank's
        discovery lock, so later obligations can be described while the current
        race measures.
        """
        if obligation.candidate_program_identity is not None:
            return
        plan = obligation.request.plan
        contract = plan.contract
        state = obligation.discovery
        if state is None or state.done:
            catalog_key = self._candidate_catalog_key(obligation)
            cached = self._described_candidates.get(catalog_key)
            if cached is not None:
                (
                    obligation.candidate_program_identity,
                    obligation.compiled,
                    obligation.programs,
                ) = cached
                if wait:
                    self._candidates_described = len(obligation.candidates)
                return
            candidates = []
            for assignment, config in obligation.candidates:
                if self._warmup_only:
                    return
                payload = self._payload(obligation, config)
                jobs = tuple(plan._compile_jobs(config, self.session.device))
                obligation.jobs[payload] = jobs
                candidates.append((assignment, payload, jobs))
                if len(candidates) % 32 == 0:
                    yield "metadata"
            with self._timing.span("discovery_key"):
                key = discovery_key(candidates, self.session.device)
            cache = self.session._race_progress_cache() if key is not None else None
            state = obligation.discovery = _Discovery(tuple(candidates), key, cache)
        if wait:
            self._candidates_described = 0
        if state.cache is not None and state.lock is None:
            record = state.cache.load_discovery(state.key)
            if record is None:
                # Bound open discovery descriptors while compiler work is pending.
                # Required discovery bypasses admission so the active race can progress.
                if not wait and len(self._discovery_locks) >= max(
                    1, self.session.compile_workers
                ):
                    return
                lock = state.cache.discovery_lock(state.key)
                while True:
                    if self._warmup_only:
                        return
                    try:
                        lock.acquire(timeout=0)
                        break
                    except Timeout:
                        if not wait:
                            return
                        yield from self._pump_pipeline(obligation)
                        yield "compile"
                record = state.cache.load_discovery(state.key)
                if record is None:
                    state.lock = lock
                    self._discovery_locks.add(lock)
                else:
                    lock.release()
            if record is not None:
                (
                    obligation.candidate_program_identity,
                    obligation.compiled,
                    obligation.programs,
                ) = restore_discovery(state.candidates, record)
                self._finish_discovery(obligation, state, publish=False)
                return
        try:
            completed = yield from self._discover_candidates(obligation, state, wait)
        except BaseException:
            self._release_discovery_lock(state)
            raise
        if not completed:
            if self._warmup_only:
                self._release_discovery_lock(state)
            return
        self._finish_discovery(obligation, state, publish=True)

    def _release_discovery_lock(self, state):
        lock, state.lock = state.lock, None
        if lock is not None:
            self._discovery_locks.discard(lock)
            lock.release()

    def _finish_discovery(self, obligation, state, *, publish):
        if publish and state.cache is not None:
            state.cache.save_discovery(
                state.key,
                discovery_record(
                    state.candidates,
                    obligation.compiled,
                    obligation.candidate_program_identity,
                ),
            )
        self._release_discovery_lock(state)
        state.done = True
        if obligation.request is self._active_request:
            self._candidates_described = len(obligation.candidates)
        self._described_candidates[self._candidate_catalog_key(obligation)] = (
            obligation.candidate_program_identity,
            obligation.compiled,
            obligation.programs,
        )

    def _candidate_catalog_key(self, obligation):
        declaration = _declaration_key(obligation.request.plan)
        if self.session._scratch_limit is None:
            return declaration
        return (
            declaration,
            tuple(
                (FrozenMapping(assignment), self._payload(obligation, config))
                for assignment, config in obligation.candidates
            ),
        )

    def _discover_candidates(self, obligation, state, wait):
        """Describe programs through the pool; True once every candidate is known."""
        contract = obligation.request.plan.contract
        if self.session.compile_workers and self.session.device.ordinal is not None:
            pool = self.session._compiler()
            if not state.submitted:
                batch_size = 32 * self.session.compile_workers
                for start in range(0, len(obligation.candidates), batch_size):
                    jobs, configs = [], []
                    for _, config in obligation.candidates[start : start + batch_size]:
                        payload = self._payload(obligation, config)
                        candidate_jobs = obligation.jobs[payload]
                        configs.append((payload, len(candidate_jobs)))
                        jobs.extend(candidate_jobs)
                    state.batches.append((pool.describe(jobs) if jobs else (), configs))
                state.submitted = True
            while state.batches:
                if self._warmup_only:
                    return False
                keys, configs = state.batches[0]
                descriptions = pool.descriptions(keys) if keys else ()
                if descriptions is None:
                    if not wait:
                        return False
                    yield from self._pump_pipeline(obligation)
                    yield "compile"
                    continue
                offset = 0
                for payload, count in configs:
                    compiled = descriptions[offset : offset + count]
                    offset += count
                    obligation.compiled[payload] = compiled
                    obligation.programs[payload] = frozenset(
                        p for item in compiled for p in item.programs
                    )
                    if wait:
                        self._candidates_described += 1
                state.batches.pop(0)
                yield "metadata"
        checksum = hashlib.sha256()
        for count, (assignment, config) in enumerate(obligation.candidates, 1):
            programs = yield from self._describe_programs(obligation, config)
            if self._warmup_only:
                return False
            checksum.update(
                _json(
                    (
                        assignment.to_dict(),
                        self._payload(obligation, config).to_dict(),
                        sorted((p.dialect, p.key) for p in programs),
                    )
                ).encode()
            )
            if wait:
                self._candidates_described = max(self._candidates_described, count)
        obligation.candidate_program_identity = checksum.hexdigest()
        return True

    def _pump_pipeline(self, current):
        """Run discovery, cache lookups and compile submission ahead of the race.

        Later obligations are described while the current one measures, looked
        up once their dependencies are selected, and their local candidates'
        programs are queued behind the current race's compiles so the pool stays
        busy. Nothing here blocks on the pool.
        """
        if (
            self._warmup_only
            or self.session.cache_only
            or not self.autotune
            or not self.session.compile_workers
        ):
            return
        remaining = []
        pipeline = self._pipeline
        for position, obligation in enumerate(pipeline):
            if self._locally_stopped:
                self._pipeline = remaining + list(pipeline[position:])
                return
            if (
                obligation.ready
                or obligation.selection is not None
                or obligation.compiles_submitted
            ):
                continue
            remaining.append(obligation)
            if obligation is current:
                continue
            yield from self._advance_discovery(obligation, wait=False)
            if (
                obligation.candidate_program_identity is None
                or obligation.request.collective is not None
            ):
                continue
            if obligation.key is None:
                self._lookup(obligation, self._selections)
            if obligation.selection is None and obligation.key is not None:
                self._submit_candidate_compiles(obligation)
        self._pipeline = [
            item
            for item in remaining
            if not (item.ready or item.selection is not None or item.compiles_submitted)
        ]

    def _checkpoint_key(self, obligation, rank_index, rank_count):
        return digest(
            {
                "race": obligation.key,
                "shard_index": rank_index,
                "shard_count": rank_count,
                "rounds": self.session.rounds,
                "samples": self.session.samples,
            }
        )

    def _submit_candidate_compiles(self, obligation):
        """Queue this shard's unmeasured candidates and a restored champion."""
        rank_index = self.session._tuning_ranks.index(self.session._tuning_rank)
        rank_count = len(self.session._tuning_ranks)
        checkpoint = self.session._race_progress_cache().load(
            self._checkpoint_key(obligation, rank_index, rank_count)
        )
        measured = (
            frozenset()
            if checkpoint is None
            else frozenset(checkpoint.measurements) - {checkpoint.champion}
        )
        plans = []
        for index in range(rank_index, len(obligation.candidates), rank_count):
            if index in measured:
                continue
            payload = self._payload(obligation, obligation.candidates[index][1])
            compiled = obligation.compiled.get(payload)
            if compiled is None:
                continue
            self._all_programs.update(obligation.programs[payload])
            plans.extend(
                item for item in compiled if not self._programs_available(item.programs)
            )
        if plans:
            self.session._compiler().submit_plans(plans, required=False)
        obligation.compiles_submitted = True

    def _ready_first(self, obligation, pending):
        """Order remaining candidates ready first; return the order and cold count."""
        contract = obligation.request.plan.contract
        ready, cold = [], []
        for item in pending:
            programs = obligation.programs.get(
                self._payload(obligation, item[1][1]), frozenset()
            )
            (ready if self._programs_available(programs) else cold).append(item)
        return ready + cold, len(cold)

    def _payload(self, obligation, config):
        """A candidate's payload from enumeration; other configs are encoded."""
        payload = obligation.payloads.get(id(config))
        if payload is None:
            payload = obligation.request.plan.contract.config_payload(config)
        return payload

    def _prefetch(self, obligation, items):
        """Queue object loads for the next batch of ready candidates."""
        keys = []
        for _, (_, config) in items[: self.session.race_batch]:
            for program in obligation.programs.get(
                self._payload(obligation, config), ()
            ):
                if program.dialect == "cute" and program not in _RESIDENT_PROGRAMS:
                    keys.append(program.key)
        if keys:
            prefetch_cute_objects(keys)

    def _programs_available(self, programs):
        """Availability with a job-lifetime memo; published programs stay published."""
        available = self._available_programs
        for program in programs:
            if program not in available:
                if not compiled_program_available(program):
                    return False
                available.add(program)
        return True

    def _describe_programs(self, obligation, config):
        plan = obligation.request.plan
        payload = self._payload(obligation, config)
        if payload not in obligation.compiled:
            compiled = []
            if payload not in obligation.jobs:
                obligation.jobs[payload] = tuple(
                    plan._compile_jobs(config, self.session.device)
                )
            for job in obligation.jobs[payload]:
                if self._warmup_only:
                    return frozenset()
                key = pickle.dumps(job, protocol=5)
                description = self._described_jobs.get(key)
                if description is None:
                    with self.session._gpu_scope():
                        with self._timing.span("compile_plan"):
                            description = describe_compilation(job)
                    self._described_jobs[key] = description
                compiled.append(description)
                yield "metadata"
            obligation.compiled[payload] = tuple(compiled)
            obligation.programs[payload] = frozenset(
                p for item in compiled for p in item.programs
            )
        return obligation.programs[payload]

    def _compile(self, obligation, config, *, required):
        programs = yield from self._describe_programs(obligation, config)
        payload = self._payload(obligation, config)
        if payload not in obligation.compiled:
            return programs
        self._all_programs.update(programs)
        missing = [
            item
            for item in obligation.compiled[payload]
            if not self._programs_available(item.programs)
        ]
        if missing:
            if self._warmup_only:
                return programs
            if self.session.cache_only:
                raise LookupError(
                    f"missing required compiled artifact for {obligation.request.name}"
                )
            if self.session.compile_workers == 0:
                # No compiler workers: compile in this process, the way a
                # kernel compiles on first use without startup preparation.
                with self.session._gpu_scope():
                    compile_in_process(missing)
            else:
                pool = self.session._compiler()
                pool.submit_plans(missing, required=required)
        return programs

    def _wait_programs(self, programs, current=None):
        while not self._programs_available(programs):
            if self._warmup_only:
                return
            pool = self.session._pool
            if pool is None:
                raise LookupError("required compiler artifacts are unavailable")
            if pool.ready(tuple(programs)) and not self._programs_available(programs):
                raise RuntimeError(
                    "compiler completed without publishing required artifacts"
                )
            yield from self._pump_pipeline(current)
            yield "compile"
        if self.session._pool is not None:
            # Surface a factory failure even if another job published shared code.
            self.session._pool.pending

    def _instantiate(
        self,
        request,
        selection,
        factory,
        expected,
        *,
        synchronize=True,
        compile_directly=False,
        reject_unlaunchable=False,
        prime=True,
    ):
        import torch

        from ._measurement import no_compilation

        compilation = nullcontext() if compile_directly else no_compilation()
        with (
            _plan_scope(request.plan),
            self.session._gpu_scope(),
            compilation,
            retain_compiled_programs() as retained,
            observe_programs() as used,
        ):
            # Prepared resources outlive the caller's profiling scope and must
            # remain mutable during later serving warmup and graph replay.
            try:
                with torch.inference_mode(False), torch.no_grad():
                    try:
                        with self._timing.span("materialize"):
                            state = request.plan._materialize(
                                selection, self.session.device
                            )
                    except CandidateResourceIneligibleError as error:
                        if not reject_unlaunchable:
                            raise
                        self.session._synchronize()
                        raise _CandidateLaunchRejected(str(error)) from error
                    with self._timing.span("bind"):
                        call = factory(state)
                    guard = _CallGuard(call)
                    self._temporaries.append(guard.finish)
                rejected = None
                try:
                    if prime:
                        with self._timing.span("prime"):
                            _prime(call)
                except Exception as error:
                    if not reject_unlaunchable or not (
                        _cooperative_launch_rejected(error)
                        or isinstance(error, CandidateResourceIneligibleError)
                    ):
                        raise
                    rejected = error
                if rejected is not None:
                    # Drain preceding launches before restoring scratch. Cleanup
                    # runs outside the handler so its failures remain fatal.
                    self.session._synchronize()
                    guard.finish()
                    self._temporaries.remove(guard.finish)
                    reason = (
                        "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE (720)"
                        if _cooperative_launch_rejected(rejected)
                        else str(rejected)
                    )
                    raise _CandidateLaunchRejected(reason) from rejected
                if synchronize:
                    self.session._synchronize()
            except _CandidateLaunchRejected:
                raise
            except Exception as error:
                message = (
                    f"{request.name} failed to prepare with configuration "
                    f"{selection.config!r} ({selection.source}): {error}"
                )
                try:
                    wrapped = type(error)(message)
                except Exception:
                    wrapped = RuntimeError(message)
                raise wrapped from error
        if compile_directly:
            expected.update(used)
        if used - expected:
            raise RuntimeError(
                f"preparation used undeclared programs for {request.name}"
            )
        return state, call, guard, retained

    def _call(
        self,
        obligation,
        selection,
        factory,
        *,
        request=None,
        synchronize=True,
        reject_unlaunchable=False,
        prime=True,
    ):
        request = obligation.request if request is None else request
        plan = request.plan
        expected = obligation.programs[self._payload(obligation, selection.config)]
        expected = expected | frozenset(
            program
            for name in request.dependencies
            for program in self._dependency_programs[name]
        )
        with self._timing.span("materialize_prime" if prime else "materialize_bind"):
            return self._instantiate(
                request,
                selection,
                factory,
                expected,
                synchronize=synchronize,
                reject_unlaunchable=reject_unlaunchable,
                prime=prime,
            )

    def _trial(self, obligation, index, assignment, config):
        if self.session._stop.is_set():
            return None
        request = obligation.request
        programs = obligation.programs[self._payload(obligation, config)]
        if not self._programs_available(programs):
            yield from self._compile(obligation, config, required=True)
        while not self._programs_available(programs):
            if self.session._stop.is_set():
                return None
            pool = self.session._pool
            if pool is None:
                raise LookupError("required compiler artifacts are unavailable")
            if pool.ready(tuple(programs)) and not self._programs_available(programs):
                raise RuntimeError(
                    "compiler completed without publishing required artifacts"
                )
            yield from self._pump_pipeline(obligation)
            yield "compile"
        selection = self._selection(obligation, config, "tuned", assignment)
        state, call, guard, retained = self._call(
            obligation,
            selection,
            request.benchmark_call,
            synchronize=False,
            reject_unlaunchable=True,
        )
        self._candidates_prepared += 1
        yield "gpu"
        self._temporaries.remove(guard.finish)
        return _Trial(index, assignment, config, call, guard, retained, state)

    def _observe(self, race):
        self._completed_rounds = race.completed_rounds
        self._total_rounds = race.planned_rounds
        self._latest_round_us = race.latest_round_us
        self._active_count = race.active_count

    def _release_race_resources(self):
        self._race_eviction = None

    def _measure(
        self,
        trials,
        *,
        champion,
        sample_exchange=None,
        launch_exchange=None,
        sample_observer=None,
        eliminate=True,
        adaptive_repeats=True,
        adaptive_collective_pair=False,
        repetitions=1,
    ):
        from ._measurement import _l2_flush_fn, prepare_race_steps, measure_race_steps

        calls = [trial.call for trial in trials]
        race = None
        steps = measuring = None
        try:
            self._phase = "calibrating"
            if self._race_eviction is None and self.session.device.ordinal is not None:
                import torch

                with self.session._gpu_scope():
                    self._race_eviction = _l2_flush_fn(
                        torch.device("cuda", self.session.device.ordinal),
                        enabled=self.session.cache_policy == "cold",
                    )
            steps = prepare_race_steps(
                calls,
                device_ordinal=self.session.device.ordinal,
                samples=self.session.samples,
                primed=True,
                eviction=self._race_eviction,
                cache_policy=self.session.cache_policy,
                repetitions=repetitions,
            )
            while sample_exchange is not None or not self.session._stop.is_set():
                try:
                    next(steps)
                except StopIteration as finished:
                    race = finished.value
                    break
                yield "gpu"
            if race is None:
                return None
            self._phase = "autotuning"
            measuring = measure_race_steps(
                race,
                device_ordinal=self.session.device.ordinal,
                rounds=self.session.rounds,
                compilation_active=lambda: (
                    self.session._pool is not None
                    and self.session._pool.active_compilations
                ),
                eliminate=eliminate,
                adaptive_collective_pair=adaptive_collective_pair,
                champion=champion,
                adaptive_repeats=adaptive_repeats,
                sample_exchange=sample_exchange,
                launch_exchange=launch_exchange,
                sample_observer=sample_observer,
            )
            response = None
            while sample_exchange is not None or not self.session._stop.is_set():
                try:
                    signal = measuring.send(response)
                    self._observe(race)
                except StopIteration as finished:
                    self._observe(race)
                    return finished.value
                response = yield "gpu" if signal is None else signal
            return None
        finally:
            closers = []
            for generator in (steps, measuring):
                if generator is not None:
                    closers.append(generator.close)
            if race is not None:
                closers.append(race.close)
            with self._timing.span("race_cleanup"):
                _close_all(closers)

    def _race_finalists(self, obligation, measurements, champion, live):
        policy = obligation.request.finalist_policy
        if policy is None:
            return ()
        families = {}
        for index, latency in measurements.items():
            assignment, _ = obligation.candidates[index]
            family = FrozenMapping(
                {name: assignment[name] for name in policy.family_fields}
            )
            families.setdefault(_json(family), []).append((latency, index, family))
        if not families:
            return ()
        nominees = {}
        for values in families.values():
            for _, index, family in sorted(values)[: policy.per_family]:
                nominees[index] = family
        if champion is not None and champion.index not in nominees:
            assignment, _ = obligation.candidates[champion.index]
            family = FrozenMapping(
                {name: assignment[name] for name in policy.family_fields}
            )
            family_indices = [
                index for index, selected in nominees.items() if selected == family
            ]
            if len(family_indices) >= policy.per_family:
                worst = max(
                    family_indices, key=lambda index: (measurements[index], index)
                )
                nominees.pop(worst)
            nominees[champion.index] = family
        if len(nominees) > policy.max_total:
            raise ValueError(
                "finalist family coverage exceeds max_total; increase the declared bound"
            )
        trials = []
        for index in sorted(nominees):
            if champion is not None and index == champion.index:
                trial = champion
            else:
                assignment, config = obligation.candidates[index]
                trial = yield from self._trial(obligation, index, assignment, config)
                if trial is None:
                    return ()
                live.append(trial)
            trials.append(trial)
        # Tournament batches have different neighbors and cache histories.
        # Compare every nominated family member in one fresh balanced race.
        measured = yield from self._measure(
            trials, champion=False, eliminate=False, adaptive_repeats=False
        )
        if measured is None:
            return ()
        records = tuple(
            FinalistRecord(
                assignment=FrozenMapping(trial.assignment),
                family=nominees[trial.index],
                candidate_index=trial.index,
                latency_us=float(latency),
            )
            for trial, latency in zip(trials, measured.latencies_us, strict=True)
        )
        return self._validate_finalists(obligation, records)

    def _race(self, obligation):
        request = obligation.request
        self._phase = "preparing candidates"
        rank_index = self.session._tuning_ranks.index(self.session._tuning_rank)
        rank_count = len(self.session._tuning_ranks)
        indexed_candidates = tuple(enumerate(obligation.candidates))
        local_candidates = indexed_candidates[rank_index::rank_count]
        self._candidate_count = len(local_candidates)
        self._candidate_sharded = rank_count > 1
        self._candidates_prepared = self._completed_rounds = self._active_count = 0
        self._total_rounds = 0
        self._latest_round_us = ()
        if request.benchmark_call is None:
            raise ValueError(
                f"multi-candidate preparation requires a representative benchmark: {request.name}"
            )
        cache = checkpoint = checkpoint_key = None
        if obligation.key is not None:
            cache = self.session._race_progress_cache()
            checkpoint_key = self._checkpoint_key(obligation, rank_index, rank_count)
            checkpoint = cache.load(checkpoint_key)
        measurements = {} if checkpoint is None else dict(checkpoint.measurements)
        if not set(measurements).issubset(index for index, _ in local_candidates):
            raise ValueError("race checkpoint contains candidates outside its shard")
        self._restored += len(measurements)
        self._candidates_prepared = len(measurements)
        remaining = [item for item in local_candidates if item[0] not in measurements]
        best_index = None if checkpoint is None else checkpoint.champion
        ledger = RaceLedger(
            job=self,
            obligation=obligation,
            kind="local",
            shard_index=rank_index,
            shard_count=rank_count,
            local_indices=[index for index, _ in local_candidates],
            restored=measurements,
        )
        compile_candidates = list(remaining)
        if remaining and best_index is not None:
            compile_candidates.append((best_index, obligation.candidates[best_index]))
        required_programs = ()
        compile_started = time.perf_counter()
        if rank_count == 1 and remaining:
            required_programs = yield from self._compile(
                obligation, obligation.configuration.default, required=True
            )
        for index, (assignment, config) in compile_candidates:
            if self.session._stop.is_set():
                return None
            ledger.note_precompiled(
                index,
                obligation.programs.get(self._payload(obligation, config), frozenset()),
            )
            yield from self._compile(obligation, config, required=False)
            if self.session._stop.is_set():
                return None
            compile_assignment = obligation.configuration.space.compile_assignment(
                assignment
            )
            actual = obligation.programs[self._payload(obligation, config)]
            previous = obligation.compile_assignments.setdefault(
                compile_assignment, actual
            )
            if previous != actual:
                raise ValueError(
                    f"runtime parameter changed actual compile keys for {request.name}"
                )
        yield from self._pump_pipeline(obligation)
        yield from self._wait_programs(required_programs, obligation)
        ledger.add_phase("compile", time.perf_counter() - compile_started)
        if not local_candidates:
            return TuningRequirement(
                obligation.key,
                self.session._tuning_ranks,
                None,
                None,
                None,
                finalist_policy=request.finalist_policy,
            )
        pending = remaining
        champion = None
        live = []
        rejected_count = 0
        try:
            if pending and best_index is not None:
                assignment, config = obligation.candidates[best_index]
                try:
                    champion = yield from self._trial(
                        obligation, best_index, assignment, config
                    )
                except _CandidateLaunchRejected:
                    # A resource-dependent launch rejection can invalidate the
                    # carried winner. Its eliminated competitors must race again.
                    cache.delete(checkpoint_key)
                    self._restored -= len(measurements)
                    return (yield from self._race(obligation))
                if champion is None:
                    return None
                live.append(champion)
                self._candidates_prepared -= 1
            while pending:
                if self.session._stop.is_set():
                    return None
                self._phase = "preparing candidates"
                self._batch_index += 1
                self._batch_candidates = 0
                self._completed_rounds = self._total_rounds = self._active_count = 0
                self._latest_round_us = ()
                pending, cold = self._ready_first(obligation, pending)
                self._prefetch(obligation, pending[: len(pending) - cold])
                if cold and len(pending) - cold < self.session.race_batch:
                    for _, (_, config) in pending[len(pending) - cold :]:
                        yield from self._compile(obligation, config, required=True)
                batch = []
                while pending and len(batch) < self.session.race_batch:
                    index, (assignment, config) = pending.pop(0)
                    trial_started = time.perf_counter()
                    try:
                        trial = yield from self._trial(
                            obligation, index, assignment, config
                        )
                    except _CandidateLaunchRejected as error:
                        rejected_count += 1
                        self._rejected += 1
                        ledger.reject(index, config, str(error))
                        logger.warning(
                            "Skipping %s candidate %d on rank %d: %r: %s",
                            request.name,
                            index,
                            self.session._tuning_rank,
                            config,
                            error,
                        )
                        self._timing.record(
                            "candidate_rejected",
                            request=request.name,
                            candidate_index=index,
                            reason=str(error),
                        )
                        continue
                    if trial is None:
                        return None
                    ledger.prepared(
                        trial,
                        seconds=time.perf_counter() - trial_started,
                        batch=self._batch_index,
                    )
                    live.append(trial)
                    batch.append(trial)
                if not batch:
                    continue
                trials = ([] if champion is None else [champion]) + batch
                self._batch_candidates = len(trials)
                self._timing.record(
                    "batch_begin",
                    request=request.name,
                    batch=self._batch_index,
                    candidate_indices=[trial.index for trial in trials],
                    carried_champion=None if champion is None else champion.index,
                    local_candidates=len(local_candidates),
                    remaining=len(pending),
                )
                measure_started = time.perf_counter()
                measurement = yield from self._measure(
                    trials,
                    champion=champion is not None,
                    sample_observer=ledger.observer(trials),
                )
                if measurement is None:
                    return None
                # Global enumeration order breaks exact timing ties without
                # adding a second policy to the distributed selection.
                best = min(
                    range(len(trials)),
                    key=lambda position: (
                        measurement.latencies_us[position],
                        trials[position].index,
                    ),
                )
                best_index = trials[best].index
                measurements.update(
                    (trial.index, float(latency))
                    for trial, latency in zip(
                        trials, measurement.latencies_us, strict=True
                    )
                )
                ledger.batch(
                    self._batch_index,
                    trials,
                    measurement,
                    champion=champion,
                    winner=best_index,
                    seconds=time.perf_counter() - measure_started,
                )
                if cache is not None:
                    with self._timing.span("race_checkpoint"):
                        cache.save(
                            checkpoint_key, RaceCheckpoint(measurements, best_index)
                        )
                self._benchmarked += len(batch)
                self._overlaps += measurement.overlapped_samples
                with self._timing.span("trial_release"):
                    for position, trial in enumerate(trials):
                        if position != best:
                            trial.close()
                            live.remove(trial)
                champion = trials[best]
                champion.latency_us = float(measurement.latencies_us[best])
                with self._timing.span("trial_reclamation"):
                    self._program_cache.clear()
                    evict_unretained(())
                self._timing.record(
                    "batch_end",
                    request=request.name,
                    batch=self._batch_index,
                    candidate_indices=[trial.index for trial in trials],
                    latencies_us=list(measurement.latencies_us),
                    winner=champion.index,
                )
                yield from self._pump_pipeline(obligation)
            obligation.coverage["measured_count"] = len(measurements)
            if cache is not None and best_index is not None:
                ledger.commit(
                    lambda: cache,
                    measurements=measurements,
                    champion_index=best_index,
                    rejected_count=rejected_count,
                )
            if best_index is None:
                if rank_count > 1:
                    return TuningRequirement(
                        obligation.key,
                        self.session._tuning_ranks,
                        None,
                        None,
                        None,
                        rejected_count=rejected_count,
                        finalist_policy=request.finalist_policy,
                    )
                raise RuntimeError(
                    f"no launchable candidates for {request.name}: "
                    f"all {rejected_count} candidates rejected by resource or "
                    "cooperative launch limits"
                )
            obligation.finalists = yield from self._race_finalists(
                obligation, measurements, champion, live
            )
            if obligation.finalists:
                obligation.coverage["finalist_count"] = len(obligation.finalists)
            assignment, config = obligation.candidates[best_index]
            candidate_index, latency_us = best_index, measurements[best_index]
            if rank_count > 1:
                return TuningRequirement(
                    obligation.key,
                    self.session._tuning_ranks,
                    assignment,
                    latency_us,
                    candidate_index,
                    rejected_count=rejected_count,
                    cute_programs=tuple(
                        program.key
                        for program in obligation.programs[
                            request.plan.contract.config_payload(config)
                        ]
                        if program.dialect == "cute"
                    ),
                    finalists=obligation.finalists,
                    finalist_policy=request.finalist_policy,
                )
            selection = self._selection(obligation, config, "tuned", assignment)
            obligation.cache_pending = rejected_count == 0
            return selection
        finally:
            closers = [self.session._synchronize, discard_prefetched_cute_objects]
            closers.extend(trial.close for trial in live)
            _close_all(closers)

    def _publish(
        self,
        request,
        selection,
        state,
        call,
        retained,
        programs,
        variants=None,
        finalists=(),
    ):
        plan = request.plan
        prepared = _Prepared(
            state=state,
            selection=selection,
            programs=programs,
            retained=retained,
            owners=() if call is None else tuple(call.owners),
            closers=() if call is None or call.close is None else (call.close,),
            device=self.session.device,
            finalist_policy=request.finalist_policy if finalists else None,
            finalists=tuple(finalists),
            variants=variants,
        )
        prepared.users.append(plan)
        previous = plan._prepared
        plan._install(prepared)
        self.session._plans[plan] = None
        if plan.shared:
            self.session._shared[_declaration_key(plan)] = plan
        self._new_plans.append(plan)
        self._plans_by_name[request.name] = plan
        if previous is not None and not previous.closed:
            self.session._release_payload(plan, previous)
        return prepared

    def _retain_benchmark(self, request, selection, expected):
        compile_directly = self._warmup_only and self.session.state != "FROZEN"
        if compile_directly:
            self._direct_compilation()
            expected = set(expected)
        state, call, guard, retained = self._instantiate(
            request,
            selection,
            request.benchmark_call,
            expected,
            compile_directly=compile_directly,
        )
        self._all_programs.update(expected)
        self._temporaries.remove(guard.finish)
        holder = (state, retained)
        self._benchmark_closers.append(lambda holder=holder: guard.finish())
        self._benchmark_calls[request.name] = call
        yield "gpu"

    def _install_obligation(self, obligation, selections):
        request, plan = obligation.request, obligation.request.plan
        selection = obligation.selection
        validated_scratch = set()
        default_selection = None

        def validate_scratch(selected):
            if self.session._scratch_limit is not None:
                payload = plan.contract.config_payload(selected.config)
                if payload not in validated_scratch:
                    self.session._candidate_memory(obligation.requests, selected.config)
                    validated_scratch.add(payload)

        def warmup_selection(prepared):
            nonlocal default_selection
            if plan in self._selected_configs:
                return obligation.selection
            if prepared is not None:
                return prepared.selection
            if (
                obligation.selection is not None
                and obligation.selection.source == "frozen"
            ):
                # N7 frozen replay: the offline-tuned winner is authoritative.
                return obligation.selection
            if default_selection is None:
                configuration = obligation.configuration
                source = "override" if configuration.pinned is not None else "default"
                default_selection = self._selection(
                    obligation, configuration.default, source
                )
                if self.session._scratch_limit is not None:
                    validated_scratch.add(
                        plan.contract.config_payload(default_selection.config)
                    )
            return default_selection

        validate_scratch(selection)
        self._active_request = request
        if self._warmup_only:
            obligation.finalists = ()
        elif request.finalist_policy is not None and not obligation.finalists:
            raise RuntimeError("requested finalist families were not measured")
        if obligation.ready:
            programs = plan.prepared.programs
        else:
            programs = frozenset()
            if not self._warmup_only:
                programs = yield from self._compile(
                    obligation, selection.config, required=True
                )
                yield from self._wait_programs(programs, obligation)
            if self._warmup_only:
                selection = warmup_selection(None)
                obligation.selection = selection
                obligation.cache_pending = False
                obligation.coverage = {}
                programs = frozenset()
                self._direct_compilation()
            if obligation.cache_pending:
                with self._timing.span("cache_save"):
                    cache = self.session._selection_cache()
                    cache.deferred = True
                    cache.save(
                        obligation.key,
                        assignment=selection.assignment,
                        config=plan.contract.config_payload(selection.config),
                        coverage=obligation.coverage,
                        programs=programs,
                        finalists=(
                            obligation.finalists
                            if request.finalist_policy is not None
                            else None
                        ),
                    )
        for item in obligation.requests:
            self._active_request = item
            item_plan = item.plan
            prepared = item_plan.prepared
            dependencies = frozenset(
                program
                for name in item.dependencies
                for program in self._dependency_programs[name]
            )
            if self._warmup_only:
                selection = warmup_selection(prepared)
            validate_scratch(selection)
            if (
                prepared is None
                or prepared.selection != selection
                or prepared.finalist_policy
                != (item.finalist_policy if obligation.finalists else None)
            ):
                alias = _Obligation(item, obligation.configuration, selection=selection)
                if not self._alias_shared(alias):
                    if item.collective is not None:
                        yield item.collective
                    if self._warmup_only:
                        selection = warmup_selection(prepared)
                        validate_scratch(selection)
                    if prepared is None or prepared.selection != selection:
                        self._phase = (
                            ("warming heuristics" if self._warmup_only else "priming")
                            if self.session.prime
                            else "materializing"
                        )
                        if self._warmup_only:
                            self._direct_compilation()
                            with _plan_scope(item_plan):
                                item_plan._memory_requirements(
                                    selection.config, self.session.device
                                )
                            expected = set(dependencies)
                            with self._timing.span(
                                "materialize_prime"
                                if self.session.prime
                                else "materialize_bind"
                            ):
                                state, call, guard, retained = self._instantiate(
                                    item,
                                    selection,
                                    item.prepare_call,
                                    expected,
                                    compile_directly=True,
                                    prime=self.session.prime,
                                )
                            expected = frozenset(expected)
                            self._all_programs.update(expected)
                        else:
                            expected = programs | dependencies
                            state, call, guard, retained = self._call(
                                obligation,
                                selection,
                                item.prepare_call,
                                request=item,
                                prime=self.session.prime,
                            )
                        self._publish(
                            item,
                            selection,
                            state,
                            call,
                            retained,
                            expected,
                            finalists=obligation.finalists,
                        )
                        self._temporaries.remove(guard.finish)
                        guard.restore()
                        del call, guard
                        yield "gpu"
            selection = item_plan.selection
            expected = item_plan.prepared.programs | dependencies
            selections[item.name] = (selection, item_plan.contract)
            self._dependency_programs[item.name] = expected
            self._plans_by_name[item.name] = item_plan
            self._coverage[item.name] = obligation.coverage
            if obligation.finalists:
                self._finalists[item.name] = self._public_finalists(
                    obligation, selection.source
                )
            if item.retain_benchmark_call:
                if item.collective is not None:
                    yield item.collective
                yield from self._retain_benchmark(item, selection, expected)
        self._timing.record(
            "request_end",
            request=request.name,
            source=selection.source,
            coverage=obligation.coverage,
        )
        source = plan.selection.source
        self._selection_counts[source] = self._selection_counts.get(source, 0) + 1
        self._completed_requests += 1

    def _consolidate(self, pending, selections):
        if not pending:
            return
        contributions = tuple(item[1] for item in pending)
        self._phase = "consolidating"
        received = yield _TuningBatch(contributions)
        by_key = {}
        for contribution in received:
            by_key.setdefault(contribution.key, []).append(contribution)
        for obligation, contribution in pending:
            configuration = obligation.configuration
            if self.session._stop.is_set():
                obligation.selection = self._selection(
                    obligation, configuration.default, "default"
                )
            else:
                peers = by_key[contribution.key]
                if obligation.request.finalist_policy is not None:
                    winner = self._consolidate_finalists(obligation, peers)
                else:
                    (winner,) = peers
                if winner.finalist_policy != obligation.request.finalist_policy:
                    raise ValueError(
                        "consolidated finalist policy differs from declaration"
                    )
                configuration.space.validate(winner.assignment)
                config = obligation.request.plan.contract._lower(
                    configuration.query,
                    configuration.device,
                    winner.assignment,
                )
                obligation.selection = self._selection(
                    obligation, config, "tuned", winner.assignment
                )
                obligation.finalists = self._validate_finalists(
                    obligation, winner.finalists
                )
                if obligation.finalists:
                    obligation.coverage["finalist_count"] = len(obligation.finalists)
                if (
                    obligation.request.finalist_policy is not None
                    and winner.candidate_index
                    not in {item.candidate_index for item in obligation.finalists}
                ):
                    raise ValueError("consolidated finalists omit the ordinary winner")
                obligation.coverage["measured_count"] = (
                    len(obligation.candidates) - winner.rejected_count
                )
                obligation.cache_pending = winner.rejected_count == 0
            self._timing.record("request_resume", request=obligation.request.name)
            yield from self._install_obligation(obligation, selections)
        pending.clear()

    def _consolidate_finalists(self, obligation, contributions):
        policy = obligation.request.finalist_policy
        families = {}
        populated = []
        for rank_index, contribution in enumerate(contributions):
            if contribution.assignment is None:
                shard_count = len(
                    range(rank_index, len(obligation.candidates), len(contributions))
                )
                if contribution.rejected_count != shard_count:
                    raise ValueError(
                        "empty tuning shard did not reject all its candidates"
                    )
                continue
            records = self._validate_finalists(obligation, contribution.finalists)
            if contribution.candidate_index not in {
                item.candidate_index for item in records
            }:
                raise ValueError("tuning shard finalists omit its ordinary winner")
            if (
                contribution.candidate_index >= len(obligation.candidates)
                or FrozenMapping(obligation.candidates[contribution.candidate_index][0])
                != contribution.assignment
            ):
                raise ValueError("tuning shard winner differs from its candidate index")
            populated.append(contribution)
            for item in records:
                families.setdefault(_json(item.family), []).append(item)
        if not populated:
            raise RuntimeError("no launchable candidates across tuning shards")
        winner = min(
            populated, key=lambda item: (item.latency_us, item.candidate_index)
        )
        finalists = []
        for records in families.values():
            records.sort(key=lambda item: (item.latency_us, item.candidate_index))
            selected = records[: policy.per_family]
            champion = next(
                (
                    item
                    for item in records
                    if item.candidate_index == winner.candidate_index
                ),
                None,
            )
            if champion is not None and champion not in selected:
                selected[-1] = champion
            finalists.extend(selected)
        return replace(
            winner,
            finalists=self._validate_finalists(obligation, finalists),
            rejected_count=sum(item.rejected_count for item in contributions),
        )

    def _run(self):
        if not self.requests:
            return PreparationResult(plans={})
        if (
            self.session.state != "FROZEN"
            and not self._warmup_only
            and len(self.session._tuning_ranks) > 1
            and not self.session._tuning_cache_synchronized
        ):
            cache = self.session._selection_cache()
            snapshots = yield TuningCacheRequirement(
                self.session._tuning_ranks,
                cache.identity,
                cache.records,
            )
            if snapshots:
                cache.reconcile(snapshots)
                self.session._tuning_cache_synchronized = True
        if self.session.state == "FROZEN":
            requests, composites = self._expand()
            groups = _coalesce_requests(requests)
            self._total_requests = len(groups)
            for request in requests:
                plan = request.plan
                if self.session._scratch_limit is not None:
                    self._validate_retained_scratch(
                        _Obligation(request, None), plan.prepared
                    )
                self._plans_by_name[request.name] = plan
                if request.retain_benchmark_call:
                    if request.collective is not None:
                        yield request.collective
                    yield from self._retain_benchmark(
                        request, plan.selection, plan.prepared.programs
                    )
            for name, (request, children) in composites.items():
                self._plans_by_name[name] = request.plan
            for group in groups:
                source = group[0].plan.selection.source
                self._selection_counts[source] = (
                    self._selection_counts.get(source, 0) + 1
                )
            self._completed_requests = self._total_requests
            result = PreparationResult(
                plans=self._plans_by_name,
                benchmark_calls=self._benchmark_calls,
                benchmark_closers=self._benchmark_closers,
            )
            self._benchmark_closers = []
            closers, self._temporaries = self._temporaries, []
            _close_all(reversed(closers))
            return result
        requests, composites = self._expand()
        groups = _coalesce_requests(requests)
        self._total_requests = len(groups)
        obligations = yield from self._configure(groups)
        dependencies = {name for request in requests for name in request.dependencies}
        terminal_collectives = [
            obligation
            for obligation in obligations
            if obligation.request.collective is not None
            and not any(item.name in dependencies for item in obligation.requests)
        ]
        terminal_ids = {id(item) for item in terminal_collectives}
        from ._collective import calibration_order

        terminal_collectives = calibration_order(terminal_collectives)
        obligations = [
            item for item in obligations if id(item) not in terminal_ids
        ] + terminal_collectives
        selections = self._selections = {}
        self._pipeline = tuple(
            item for item in obligations if not item.ready and item.selection is None
        )
        yield from self._pump_pipeline(None)
        pending = []
        pending_names = set()
        for obligation in obligations:
            request = obligation.request
            plan = request.plan
            if pending_names.intersection(request.dependencies) or (
                pending and request.collective is not None
            ):
                yield from self._consolidate(pending, selections)
                pending_names.clear()
            self._active_request = request
            self._phase = "selecting"
            self._batch_index = self._batch_candidates = 0
            self._candidate_sharded = False
            self._timing.record(
                "request_begin",
                request=request.name,
                component=plan.component_id,
                candidates=len(obligation.candidates),
                ready=obligation.ready,
                query=obligation.configuration.encoded_query.to_dict(),
                bindings=len(obligation.requests),
            )
            self._global_candidate_count = self._candidate_count = len(
                obligation.candidates
            )
            self._candidates_prepared = self._completed_rounds = self._active_count = 0
            self._total_rounds = 0
            self._latest_round_us = ()
            if obligation.ready:
                selection = obligation.selection
                programs = plan.prepared.programs
                obligation.programs[plan.contract.config_payload(selection.config)] = (
                    programs
                )
            else:
                if self._warmup_only:
                    configuration = obligation.configuration
                    source = (
                        "override" if configuration.pinned is not None else "default"
                    )
                    frozen = None
                    if configuration.pinned is None:
                        frozen = self._frozen_tune_selection(obligation, selections)
                    obligation.selection = frozen or self._selection(
                        obligation, configuration.default, source
                    )
                completed_before_selection = self._benchmarked + self._restored
                if obligation.selection is None:
                    yield from self._describe_candidates(obligation)
                    self._phase = "selecting"
                    if request.collective is not None:
                        from ._collective import select

                        obligation.selection = yield from select(
                            self, obligation, selections
                        )
                    else:
                        self._lookup(obligation, selections)
                if obligation.selection is not None and obligation.planned_candidates:
                    completed_during_selection = (
                        self._benchmarked + self._restored - completed_before_selection
                    )
                    self._total_candidates -= (
                        obligation.planned_candidates - completed_during_selection
                    )
                    obligation.planned_candidates = 0
                if obligation.selection is None:
                    if self.session.cache_only:
                        raise LookupError(f"no completed selection for {request.name}")
                    if not self.session._stop.is_set() and self.autotune:
                        result = yield from self._race(obligation)
                        if isinstance(result, TuningRequirement):
                            pending.append((obligation, result))
                            pending_names.update(
                                item.name for item in obligation.requests
                            )
                            self._timing.record("local_race_end", request=request.name)
                            continue
                        obligation.selection = result
                    if obligation.selection is None:
                        obligation.selection = self._selection(
                            obligation, obligation.configuration.default, "default"
                        )
            yield from self._install_obligation(obligation, selections)
        yield from self._consolidate(pending, selections)
        if self.session._cache is not None:
            with self._timing.span("cache_save"):
                self.session._cache.flush()
        # N7: publish tune records and the audit at the job boundary too --
        # decode-op sessions (plan_decode_op) are never closed, so their
        # heuristic selections must not stay staged until a close that never
        # comes.
        self.session._close_tune_cache()
        for name, (request, children) in composites.items():
            plan = request.plan
            child_plans = {count: child.plan for count, child in children.items()}
            if (
                plan.prepared is not None
                and all(child.prepared is not None for child in child_plans.values())
                and plan.prepared.variants is not None
                and all(
                    plan.prepared.variants[count] is child
                    for count, child in child_plans.items()
                )
                and not any(child in self._new_plans for child in child_plans.values())
            ):
                self._plans_by_name[name] = plan
                continue
            states = {
                count: child.prepared.state for count, child in child_plans.items()
            }
            programs = frozenset(
                program
                for child in child_plans.values()
                for program in child.prepared.programs
            )
            state = plan._assemble(states, self.session.device)
            self._publish(
                request, None, state, None, None, programs, variants=plan.variants
            )
            yield "metadata"
        self._release_race_resources()
        self._phase = "finishing"
        summary = yield from self.session._drain()
        if summary is not None:
            self._compilations = summary.cute_compilations + summary.triton_compilations
        from b12x._lib.program_cache import evict_unretained

        keep = frozenset(
            program
            for plan in self.session._plans
            if plan.prepared is not None
            for program in plan.prepared.programs
        )
        self.session._synchronize()
        self._program_cache.clear()
        evict_unretained(keep)
        result = PreparationResult(
            plans=self._plans_by_name,
            coverage=self._coverage,
            finalists=self._finalists,
            benchmark_calls=self._benchmark_calls,
            benchmark_closers=self._benchmark_closers,
            cache_hits=self._cache_hits,
            benchmarked_candidates=self._benchmarked,
            restored_candidates=self._restored,
            rejected_candidates=self._rejected,
            elapsed_seconds=time.monotonic() - self._started,
            compilation=summary,
            overlapped_benchmark_samples=self._overlaps,
            program_counts={
                dialect: sum(
                    program.dialect == dialect for program in self._all_programs
                )
                for dialect in ("cute", "triton")
            },
        )
        self._benchmark_closers = []
        closers, self._temporaries = self._temporaries, []
        _close_all(reversed(closers))
        del closers
        if self._benchmarked:
            # Losing candidates were launched: their executables are now
            # unreferenced and their local memory growth is undone.
            from b12x._lib.program_cache import reclaim_device_memory

            self.session._synchronize()
            with self.session._gpu_scope():
                reclaim_device_memory(keep, stack_limit=self._stack_limit)
        return result

    def _expand(self):
        return _expand_requests(self.requests)


_LAZY_SESSIONS: dict[object, PreparationSession] = {}


def prepare_default(request):
    """Prepare one request with its default configuration, outside any driver.

    Allowed while no session is frozen and no stream is capturing. The
    request's own prepare call primes the state, so families whose state
    depends on the caller's storage (a KV cache, a pool) materialize it for
    that storage. Returns the plan's prepared payload.
    """
    from b12x._lib.runtime_control import kernel_resolution_frozen

    plan = request.plan
    name = plan.component_id
    if kernel_resolution_frozen():
        raise RuntimeError(
            f"{name} plan is not prepared and kernel resolution is frozen"
        )
    try:
        import torch

        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
    except Exception:
        capturing = False
    if capturing:
        raise RuntimeError(
            f"{name} plan {request.name!r} is not prepared; a shape used under CUDA "
            "graph capture must be declared and prepared before the capture"
        )
    first = plan if isinstance(plan, Plan) else next(iter(plan.variants.values()))
    device = (
        first._device
        if first._device is not None
        else getattr(first.query, "device", None)
    )
    detected = detect_device(device)
    session = _LAZY_SESSIONS.get(detected.ordinal)
    if session is None or session.state == "CLOSED" or session._job is not None:
        # Compiles in this process: a first-use default is a JIT kernel, not a
        # startup pass with compiler workers.
        session = PreparationSession(device=detected, autotune=False, compile_workers=0)
        _LAZY_SESSIONS[detected.ordinal] = session
    session.prepare((request,), autotune=False)
    return plan._prepared


@lru_cache(maxsize=256)
def _warn_unprepared_declaration(key, component_id, query_type, dimensions):
    shape = f" ({dimensions})" if dimensions else ""
    logger.warning(
        "%s: %s%s was not prepared before its first use; preparing its default "
        "configuration. Repeated matching declarations are logged at DEBUG.",
        component_id,
        query_type,
        shape,
    )


def _prepare_default(plan):
    """Materialize an unprepared plan with its default configuration, without priming.

    This is the path a plan takes when it is bound or run without having been
    prepared: the heuristic configuration, its scratch, and kernels that
    compile on first use. Warnings identify distinct declarations; layer-local
    plan handles and complete queries remain available at DEBUG.
    """
    if isinstance(plan, _CompositePlan):
        query_type, query = type(plan).__name__, plan.capacity_metadata
    else:
        query_type = type(plan.query).__name__
        query = plan.contract.encode_query(plan.query)
    fields = [
        (name, value)
        for name, value in query.items()
        if type(value) in (int, float, str)
        or isinstance(value, (tuple, list))
        and len(value) <= 8
        and all(type(item) is int for item in value)
    ]
    fields.sort(key=lambda item: not item[0].startswith(("max_", "planned_")))
    dimensions = ", ".join(f"{name}={value}" for name, value in fields[:8])
    if len(fields) > 8:
        dimensions += ", ..."
    _warn_unprepared_declaration(
        _declaration_key(plan),
        plan.component_id,
        query_type,
        dimensions,
    )
    logger.debug(
        "%s: unprepared plan %s#%s: %r",
        plan.component_id,
        query_type,
        plan.handle,
        query,
    )

    def noop(state):
        return PreparedCall(run=lambda: None)

    if isinstance(plan, _CompositePlan):
        request = plan.request(
            name=f"default:{plan.handle}",
            prepare_calls={count: noop for count in plan.token_counts},
        )
    else:
        request = plan.request(name=f"default:{plan.handle}", prepare_call=noop)
    return prepare_default(request)


_set_lazy_preparer(_prepare_default)
