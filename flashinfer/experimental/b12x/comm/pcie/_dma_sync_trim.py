"""N4b (NEON-LEVERS): DmaAllReduce eager-replay sync / staging trim.

Two independent, numerics-neutral trims of `PCIeDmaAllReduce._run_prepared`
(wire protocol, ring schedule and reduction order are untouched):

* the neighbor handshake's "reads done" publish fires without first draining
  the send streams (it is ordered by this rank's own scratch reads, which all
  run on the main stream), and the flag stream's redundant input-ready edge is
  dropped.  The receive-side wait still blocks return until the lagging
  neighbor consumed this rank's sends, which is what keeps the next call's
  scratch writes safe.

* `B12X_PCIE_DMA_REPLAY_BIND=1`: per-call-site eager-replay graphs captured
  over the CALLER's own input/output buffers — `graph.replay()` writes the
  result in place and the replay path's 2 x payload staging copies
  (`source[:elements].copy_(inp)` / `out.copy_(result[:elements])`, 2 x 24 MiB
  at the 2048-row AR) disappear.  A key binds only after it recurred
  (`repeat=2`) and only while under `cap`, so churning caller addresses fall
  back to the staged path untouched; every replay is gated on an exact
  (ptr, numel, dtype, contiguity) key match — that gate, not address pinning,
  is what makes replaying captured absolute addresses safe.

This module is pure Python (no torch) so CPU tests can exercise the policy
directly; `pcie_dma` owns the CUDA side.  Neither trim touches
`FusedWireScratch.release_preparation`, `wire_decode_ops` sizing, or any wire
slot layout.
"""
from __future__ import annotations

BIND_OFF = "off"
BIND_ON = "on"
_BIND_VALUES = {"": BIND_OFF, "0": BIND_OFF, "false": BIND_OFF, "no": BIND_OFF,
                "off": BIND_OFF, "1": BIND_ON, "true": BIND_ON, "yes": BIND_ON,
                "on": BIND_ON, "both": BIND_ON}


def parse_bind_mode(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if value not in _BIND_VALUES:
        raise ValueError(f"unrecognized PCIe DMA replay-bind mode: {raw!r}")
    return _BIND_VALUES[value]


def parse_sync_trim(raw: str | None, *, default: bool = True) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def replay_bind_key(in_ptr: int, out_ptr: int, elements: int, dtype: str) -> tuple:
    """The exact identity a captured replay may serve: the caller's two
    addresses, the flat element count and the dtype."""
    if int(elements) <= 0:
        raise ValueError("replay-bind key needs a positive element count")
    return (int(in_ptr), int(out_ptr), int(elements), str(dtype))


class ReplayBindPolicy:
    """Decides, per call, between the bound replay and the staged fallback.

    `observe(key)` returns "match" (replay the bound graph), "bind" (capture
    this call's buffers now, then replay — the capture slot IS the call, so
    the flag protocol just sees one run), or "fallback" (the staged private
    replay).  A key is only captured after it recurred `repeat` times:
    churning caller addresses never bind and never pin anything."""

    def __init__(self, mode: str = BIND_OFF, *, cap: int = 4, repeat: int = 2):
        if mode not in (BIND_OFF, BIND_ON):
            raise ValueError(f"replay-bind mode must be off or on, got {mode!r}")
        if int(cap) < 0 or int(repeat) < 1:
            raise ValueError("replay-bind cap/repeat must be non-negative / positive")
        self.mode = mode
        self.cap = int(cap)
        self.repeat = int(repeat)
        self.counts: dict = {}
        self.bound: dict = {}

    def observe(self, key: tuple) -> str:
        if self.mode == BIND_OFF:
            return "fallback"
        if key in self.bound:
            return "match" if self.bound[key] is not None else "fallback"
        if len(self.counts) >= 8 * max(1, self.cap):
            # bounded bookkeeping: churned one-off keys never recur anyway
            self.counts = {k: v for k, v in self.counts.items()
                           if k in self.bound or v >= self.repeat}
        self.counts[key] = self.counts.get(key, 0) + 1
        if self.counts[key] >= self.repeat and len(self.bound) < self.cap:
            self.bound[key] = None      # reserved until commit()
            return "bind"
        return "fallback"

    def commit(self, key: tuple, graph) -> None:
        if self.bound.get(key, "missing") is None and key in self.bound:
            self.bound[key] = graph
            return
        raise ValueError("replay-bind commit for an unreserved key")

    def release(self) -> None:
        self.bound.clear()
        self.counts.clear()


__all__ = ["BIND_OFF", "BIND_ON", "parse_bind_mode", "parse_sync_trim",
           "replay_bind_key", "ReplayBindPolicy"]
