"""Cooperative-preamble CSF expansion: decision surface and index model.

Lever-2 M2 of ``perf-campaign/LEVER2-MOE-PLAN.md`` moves active-expert CSF
scale expansion into a preamble phase of ``W4A16FusedMoeKernel._moe_body``
(standalone ``decode_nvfp4_csf_pair`` launch + cooperative-grid drain bubble
removed). The plan leaves two shape choices open until measurement (W2 scale
expansion timing; dedup mechanism) and gates the step behind M1's residual-tax
measurement, so this wave ships the *decision surface* — the strict feature
flag, the capability contract the fused launch must satisfy before the flag
may be enabled, and the pure-Python model of the preamble's work-unit
partition/dedup index math (unit-tested here; the oracle for
``harness/m2_preamble_probe.py`` and the eventual kernel body).

The kernel body itself is specialist CuTe-DSL work (Principle 14 M2 marking:
strictly non-Opus author + mandatory specialist review) and must NOT be
enabled until the M2 entry gate in ``lever2-kernel/README.md`` has fired.
"""

from __future__ import annotations

import os


def csf_preamble_enabled() -> bool:
    """Whether the cooperative-preamble CSF path is requested.

    ``B12X_W4A16_CSF_PREAMBLE`` is ``"0"`` (default) or ``"1"``. Setting it to
    ``"1"`` is only legal on a build whose fused launch advertises
    ``csf_preamble_geometry`` (see :func:`require_preamble_support`); any other
    value raises, so a typo can never half-enable the path.
    """
    value = os.environ.get("B12X_W4A16_CSF_PREAMBLE", "0")
    if value not in ("0", "1"):
        raise ValueError("B12X_W4A16_CSF_PREAMBLE must be 0 or 1")
    return value == "1"


def require_preamble_support(launch) -> None:
    """Validate that a compiled fused launch can host the CSF preamble.

    Raises ``RuntimeError`` when the launch has no ``csf_preamble_geometry``
    capability marker. The M2 kernel-body patch adds that marker; until then
    this guard keeps ``B12X_W4A16_CSF_PREAMBLE=1`` fail-closed instead of
    silently running the old two-launch path.
    """
    if getattr(launch, "csf_preamble_geometry", None) is None:
        raise RuntimeError(
            "B12X_W4A16_CSF_PREAMBLE=1 requires a fused W4A16 launch compiled "
            "with cooperative-preamble CSF expansion support "
            "(W4A16FusedMoeKernel.csf_preamble_geometry); this build does not "
            "provide it — see perf-campaign/lever2-kernel/README.md (M2)"
        )


def preamble_work_units(
    routes: int, first_tasks: int, second_tasks: int, *, packed: bool
) -> int:
    """Total preamble work units: one (slot, task) pair per route and task.

    ``packed`` adds the single CTA-0 fused ``CsfRoutePack`` unit.
    """
    routes = int(routes)
    if routes <= 0 or int(first_tasks) <= 0 or int(second_tasks) <= 0:
        raise ValueError("preamble work requires positive routes and task counts")
    return routes * (int(first_tasks) + int(second_tasks)) + (1 if packed else 0)


def preamble_unit_assignment(unit: int, routes: int, first_tasks: int, second_tasks: int):
    """Map a grid-strided preamble unit to ``(slot, task, plane)``.

    Units ``0 .. routes*(first+second)-1`` cover (slot, task) pairs slot-major;
    with ``packed`` the final unit is the CTA-0 route pack. Returns
    ``("pack",)`` for that unit.
    """
    unit = int(unit)
    routes = int(routes)
    per_route = int(first_tasks) + int(second_tasks)
    total = preamble_work_units(routes, int(first_tasks), int(second_tasks), packed=True)
    if not 0 <= unit < total:
        raise ValueError(f"preamble unit {unit} outside [0, {total})")
    if unit == routes * per_route:
        return ("pack",)
    slot, task = divmod(unit, per_route)
    plane = 0 if task < int(first_tasks) else 1
    return (slot, task if plane == 0 else task - int(first_tasks), plane)


def preamble_expanded_experts(ids, num_experts: int):
    """Model of the preamble's per-slot direct dedup (mode-0 semantics).

    Expert ``e`` is expanded once, at its lowest route slot; invalid routes
    (-1, >= num_experts) never expand. Bit-identical to the two-launch
    presence path (``mark_active_experts`` + mode-2 expansion), which expands
    the same set exactly once.
    """
    expanded = {}
    for slot, raw in enumerate(ids):
        e = int(raw)
        if 0 <= e < int(num_experts) and e not in expanded:
            expanded[e] = slot
    return expanded


def presence_path_expanded_experts(ids, num_experts: int):
    """Model of the legacy two-launch presence path's expansion set."""
    active = sorted({int(raw) for raw in ids if 0 <= int(raw) < int(num_experts)})
    return {e: None for e in active}
