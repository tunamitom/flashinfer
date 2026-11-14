"""Cross-rank top-k selection hand-back (consult2 defect 15b).

Under context parallelism a sharded index merges every rank's local top-k
candidates into ONE global top-k per query row — the same selection on every
rank, in GLOBAL positions (``dsa_indexer.tiled_topk.merge_tiled_topk_candidates``
over the planes ``comm.pcie.pcie_dcp_topk`` stages into each row owner).
A rank's sparse MLA then attends over the records it OWNS and combines the
rank partials by their LSE, so the merged selection must be taken BACK to
per-rank selections: the entries this rank owns, mapped to its local physical
slots, compacted to the front with a per-row live count and ``-1`` padded.

Hy4's shared layers need the hand-back too.  The 21 full layers run the
indexer and the merge; the 57 shared layers bind their owner full layer's
``selected`` buffer (``IndexCache``: the 4-layer group cell binds layer 1's
buffer), so the hand-back runs ONCE per group, into the owner's buffer, and
every shared layer reads the live result through its binding — never its own
copy and never the raw merged cross-rank selection.

Witness: kimi3 ``ignition/recipes/kimi3/dsa.py`` (``PreparedIndexMerge``,
``select()``) and ``kimi3/dcp.py`` (``ContextShards.token_slots``,
``compact``) — the ownership mask and the selection compaction are ordinary
fixed-shape tensor arithmetic inside the graph; no data-dependent shapes, no
allocations when the caller owns the outputs.

Mutants that must die here (see the tests):
  (a) no hand-back: a rank (or a shared layer) consuming the merged global
      selection as if it were local attends the wrong records entirely;
  (b) a stale or privatized hand-back for shared layers: a shared layer must
      see the live owner buffer after every hand-back.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "rank_local_slots",
    "compact_selection",
    "selection_handback_reference",
    "PreparedSelectionHandback",
    "prepare_selection_handback",
]


def rank_local_slots(
    positions: torch.Tensor,
    *,
    world_size: int,
    rank: int,
    slot_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """This rank's local physical slot of each merged GLOBAL position.

    Ownership is the interleaved shard (the kimi3 ``ContextShards`` rule): a
    global position ``p`` is owned by rank ``p % world_size`` at local index
    ``p // world_size``, so interleaving spreads consecutive positions across
    ranks.  ``slot_map`` optionally maps a local index to a physical cache
    slot (paged layouts pass the page-table expansion); without it the local
    index IS the slot.  The map is PER ROW: rank-2 ``[rows, capacity]``
    (``positions`` must then be rank-2 with one map row per position row) so a
    batch of rows with distinct cache row bases maps correctly — one shared
    rank-1 map would send every row's local indices into row zero's slots.
    A rank-1 ``slot_map`` is still accepted and shared by every row.
    Negative positions, positions owned by another rank
    and positions whose slot map entry is negative all become ``-1``.
    """
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"positions must be int32/int64, got {positions.dtype}")
    if not isinstance(world_size, int) or world_size < 1:
        raise ValueError(f"world_size must be a positive int, got {world_size!r}")
    if not isinstance(rank, int) or not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank!r}")
    pos = positions.to(torch.int64)
    local = torch.div(pos.clamp(min=0), world_size, rounding_mode="floor")
    live = (pos >= 0) & (pos % world_size == rank)
    if slot_map is not None:
        if slot_map.ndim not in (1, 2):
            raise ValueError(
                f"slot_map must be rank-1 (shared) or rank-2 (per-row), got {tuple(slot_map.shape)}"
            )
        if slot_map.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"slot_map must be int32/int64, got {slot_map.dtype}"
            )
        if slot_map.device != positions.device:
            raise ValueError("slot_map must live on positions' device")
        if slot_map.ndim == 2:
            if positions.ndim != 2 or slot_map.shape[0] != positions.shape[0]:
                raise ValueError(
                    f"per-row slot_map must have one row per merged row "
                    f"({int(positions.shape[0]) if positions.ndim else 0}), "
                    f"got {tuple(slot_map.shape)}"
                )
            table = slot_map.to(torch.int64)
        else:
            table = slot_map.to(torch.int64).reshape(1, -1)
        bounds = int(table.shape[-1])
        if bounds == 0:
            return torch.full_like(pos, -1, dtype=torch.int32)
        inside = local < bounds
        gathered = (
            table[0, local.clamp(0, bounds - 1)]
            if table.shape[0] == 1
            else table.gather(1, local.clamp(0, bounds - 1))
        )
        mapped = torch.where(inside, gathered, torch.full_like(local, -1))
        live = live & inside & (mapped >= 0)
        local = mapped
    return torch.where(live, local, torch.full_like(local, -1)).to(torch.int32)


def compact_selection(
    ids: torch.Tensor,
    live: torch.Tensor,
    scratch: torch.Tensor,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Live entries of ``ids`` compacted to the front of ``output``, stably.

    ``output`` is ``ids``-shaped: the live entries in input order, then ``-1``
    padding, and the return value adds each row's live count (int32).  This is
    kimi3's ``compact``: ``scratch`` is ``[rows, width + 1]`` and every dead
    entry writes column zero, which the result drops, so live entries land on
    distinct columns and no scatter order matters.  Give ``scratch`` and
    ``output`` caller-owned buffers to keep the op allocation-free inside a
    CUDA graph.
    """
    rows, width = ids.shape
    if live.shape != ids.shape or live.dtype != torch.bool:
        raise ValueError("live must be a bool mask matching ids")
    if (
        scratch.shape != (rows, width + 1)
        or scratch.dtype != ids.dtype
        or scratch.device != ids.device
    ):
        raise ValueError(
            f"scratch must be {(rows, width + 1)} {ids.dtype} on {ids.device}"
        )
    if output is None:
        output = torch.empty_like(ids)
    elif (
        output.shape != ids.shape
        or output.dtype != ids.dtype
        or output.device != ids.device
    ):
        raise ValueError(f"output must match ids {tuple(ids.shape)} {ids.dtype}")
    destination = live.cumsum(-1)
    destination = torch.where(live, destination, torch.zeros_like(destination))
    scratch.fill_(-1)
    scratch.scatter_(
        1, destination, torch.where(live, ids, torch.full_like(ids, -1))
    )
    output.copy_(scratch[:, 1:])
    return output, live.sum(-1).to(torch.int32)


def selection_handback_reference(
    merged: torch.Tensor,
    *,
    world_size: int,
    rank: int,
    scratch: torch.Tensor,
    output: torch.Tensor | None = None,
    counts: torch.Tensor | None = None,
    slot_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The full hand-back: merged global top-k -> this rank's local selection.

    ``merged`` is ``[rows, topk]`` int32 global positions (``-1`` padding
    allowed).  Returns ``(local_selected, counts)``: ``[rows, topk]`` local
    physical slots compacted to the front with ``-1`` padding and ``[rows]``
    int32 live counts — exactly the ``selected_indices`` /
    ``selected_lengths`` operands of ``sparse_mla.bind``.  Identical input
    bytes give the identical result on every rank, and the ranks' hand-backs
    partition the merged selection: every live merged entry is recovered by
    exactly the rank that owns it.
    """
    if merged.ndim != 2:
        raise ValueError(f"merged must be rank-2, got {tuple(merged.shape)}")
    if merged.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"merged must be int32/int64, got {merged.dtype}")
    slots = rank_local_slots(
        merged, world_size=world_size, rank=rank, slot_map=slot_map
    )
    local, row_counts = compact_selection(
        slots, slots >= 0, scratch, output if output is not None else None
    )
    if counts is None:
        return local, row_counts
    if counts.shape != (merged.shape[0],) or counts.dtype != torch.int32:
        raise ValueError("counts must be int32 with one entry per row")
    counts.copy_(row_counts)
    return local, counts


@dataclass(frozen=True)
class PreparedSelectionHandback:
    """A geometry-pinned hand-back: validated once, run with owned buffers.

    ``run`` is fixed-shape tensor arithmetic — CUDA-graph safe and
    allocation-free when the caller passes ``output``/``counts``/``scratch``.
    ``slot_map`` is the per-row local-index -> physical-slot table
    (``[rows, local_capacity]``; a shared rank-1 map is accepted too) — see
    :func:`rank_local_slots`.
    """

    world_size: int
    rank: int
    rows: int
    topk: int
    device: torch.device

    def run(
        self,
        merged: torch.Tensor,
        *,
        slot_map: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
        counts: torch.Tensor | None = None,
        scratch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(merged.shape) != (self.rows, self.topk):
            raise ValueError(
                f"PreparedSelectionHandback expects merged "
                f"{(self.rows, self.topk)}, got {tuple(merged.shape)}"
            )
        if merged.device != self.device:
            raise ValueError("merged must be on the prepared device")
        if scratch is None:
            scratch = torch.empty(
                (self.rows, self.topk + 1),
                dtype=torch.int32,
                device=self.device,
            )
        return selection_handback_reference(
            merged,
            world_size=self.world_size,
            rank=self.rank,
            scratch=scratch,
            output=output,
            counts=counts,
            slot_map=slot_map,
        )


def prepare_selection_handback(
    *,
    merged: torch.Tensor,
    world_size: int,
    rank: int,
) -> PreparedSelectionHandback:
    """Pin the hand-back geometry for one rank from a sample merged selection."""
    if merged.ndim != 2:
        raise ValueError(f"merged must be rank-2, got {tuple(merged.shape)}")
    if merged.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"merged must be int32/int64, got {merged.dtype}")
    rows, topk = int(merged.shape[0]), int(merged.shape[1])
    if rows <= 0 or topk <= 0:
        raise ValueError(f"merged extents must be positive, got {(rows, topk)}")
    prepared = PreparedSelectionHandback(
        world_size=int(world_size),
        rank=int(rank),
        rows=rows,
        topk=topk,
        device=merged.device,
    )
    # Validate the rank/world pair now, not on the first run.
    rank_local_slots(
        merged[:1], world_size=prepared.world_size, rank=prepared.rank
    )
    return prepared
