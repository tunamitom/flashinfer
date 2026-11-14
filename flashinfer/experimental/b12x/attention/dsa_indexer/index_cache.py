"""Shared-selection IndexCache cell (Hy4 plan item 5).

Hy4 runs the lightning indexer on 21 ``full`` layers (0, 1, 5, 9, … 77) and
shares each 4-layer group's top-k selection verbatim with the group's 57
``shared`` layers — on every forward, including every decode step.  The
kernels need no change: ``dsa_indexer`` selection with
``Caps(output_index_space="physical")`` emits flat physical cache slots,
which ``sparse_mla.bind(selected_indices=...)`` consumes directly.  A shared
layer binds its full layer's ``selected`` buffer through the same
``data_ptr`` and allocates no index-key pages, so the paged FP8 index-key
pages exist for the 21 full layers only (and under context parallelism the
cross-rank selection exchange runs 21 times per step, not 78).

The 4-layer group cell (layers 1-4 of the model): layer 1 runs the indexer;
layers 2-4 consume layer 1's ``selected`` buffer.  ``hy4_indexer_layers()``
builds the full 78-layer map with the same rule — a shared layer binds the
previous full layer's buffer.

Mutants that must die here (hy4-plan §1.5): stale or recomputed selection in
a shared layer.  A shared layer must read the live full-layer buffer, never
its own copy and never an earlier snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .paged import INDEX_HEAD_DIM, PAGED_INDEX_PAGE_SIZE

__all__ = [
    "IndexerLayer",
    "HY4_NUM_LAYERS",
    "HY4_FULL_LAYER_IDS",
    "hy4_indexer_layers",
    "hy4_group_cell_layers",
    "IndexCache",
]

_SCALE_BYTES = 4
HY4_NUM_LAYERS = 78
HY4_FULL_LAYER_IDS = (0, 1, *range(5, 78, 4))  # 0, 1, 5, 9, ..., 77


@dataclass(frozen=True)
class IndexerLayer:
    """One layer's indexer role inside the shared-selection cell."""

    layer_id: int
    kind: Literal["full", "shared"]
    owner_id: int  # the full layer whose selection buffer this layer binds

    def __post_init__(self):
        if self.kind not in ("full", "shared"):
            raise ValueError(f"kind must be full or shared, got {self.kind!r}")
        if self.kind == "full" and self.owner_id != self.layer_id:
            raise ValueError("a full layer owns its own selection buffer")
        if self.layer_id < 0 or self.owner_id < 0:
            raise ValueError("layer ids must be non-negative")


def hy4_indexer_layers(
    num_layers: int = HY4_NUM_LAYERS,
    full_layer_ids: tuple[int, ...] = HY4_FULL_LAYER_IDS,
) -> tuple[IndexerLayer, ...]:
    """The Hy4 layer map: 21 full (0, 1, 5, … 77) and 57 shared.

    Each shared layer binds the previous full layer's buffer.
    """
    if not full_layer_ids or full_layer_ids[0] != 0:
        raise ValueError("full_layer_ids must start at layer 0")
    full = set(full_layer_ids)
    layers: list[IndexerLayer] = []
    owner = 0
    for layer_id in range(num_layers):
        if layer_id in full:
            owner = layer_id
            layers.append(IndexerLayer(layer_id, "full", layer_id))
        else:
            layers.append(IndexerLayer(layer_id, "shared", owner))
    return tuple(layers)


def hy4_group_cell_layers(full_id: int = 1) -> tuple[IndexerLayer, ...]:
    """The 4-layer group cell: ``full_id`` runs the indexer, +1..+3 share it."""
    return (
        IndexerLayer(full_id, "full", full_id),
        IndexerLayer(full_id + 1, "shared", full_id),
        IndexerLayer(full_id + 2, "shared", full_id),
        IndexerLayer(full_id + 3, "shared", full_id),
    )


class IndexCache:
    """Allocates and shares per-layer selection buffers and index-key pages.

    Full layers own an ``[max_q_rows, topk]`` int32 ``selected`` buffer
    (initialised to ``-1`` — no selection) and ``index_pages_per_layer``
    paged FP8 index-key pages.  Shared layers alias their owner's ``selected``
    buffer (same ``data_ptr``) and allocate nothing.  ``selected`` buffers are
    the tensors passed straight to ``sparse_mla.bind(selected_indices=...)``;
    ``indexer_output`` is the ``output_indices`` operand of
    ``dsa_indexer.bind`` (full layers only, physical slot space).
    """

    def __init__(
        self,
        layers: tuple[IndexerLayer, ...],
        *,
        device: torch.device | str,
        topk: int,
        max_q_rows: int,
        index_pages_per_layer: int = 1,
        page_size: int = PAGED_INDEX_PAGE_SIZE,
    ) -> None:
        if not layers:
            raise ValueError("layers must not be empty")
        if topk <= 0 or max_q_rows <= 0:
            raise ValueError(f"topk and max_q_rows must be positive, got {topk}, {max_q_rows}")
        if index_pages_per_layer < 0:
            raise ValueError(
                f"index_pages_per_layer must be non-negative, got {index_pages_per_layer}"
            )
        seen: dict[int, IndexerLayer] = {}
        for layer in layers:
            if not isinstance(layer, IndexerLayer):
                raise TypeError(f"layers must hold IndexerLayer, got {type(layer)}")
            if layer.layer_id in seen:
                raise ValueError(f"duplicate layer id {layer.layer_id}")
            seen[layer.layer_id] = layer
        for layer in layers:
            if layer.kind == "shared" and layer.owner_id not in seen:
                raise ValueError(
                    f"shared layer {layer.layer_id} binds unknown owner {layer.owner_id}"
                )
            if layer.kind == "shared" and seen[layer.owner_id].kind != "full":
                raise ValueError(
                    f"shared layer {layer.layer_id} must bind a full owner, "
                    f"got {seen[layer.owner_id].kind}"
                )

        self._layers = seen
        self._order = tuple(layer.layer_id for layer in layers)
        self._device = torch.device(device)
        self._topk = int(topk)
        self._max_q_rows = int(max_q_rows)
        self._page_size = int(page_size)
        self._index_pages_per_layer = int(index_pages_per_layer)
        self._page_bytes = self._page_size * (INDEX_HEAD_DIM + _SCALE_BYTES)

        self._selected: dict[int, torch.Tensor] = {}
        self._pages: dict[int, torch.Tensor] = {}
        for layer in layers:
            if layer.kind == "full":
                self._selected[layer.layer_id] = torch.full(
                    (self._max_q_rows, self._topk),
                    -1,
                    dtype=torch.int32,
                    device=self._device,
                )
                if self._index_pages_per_layer:
                    self._pages[layer.layer_id] = torch.zeros(
                        (self._index_pages_per_layer, self._page_bytes),
                        dtype=torch.uint8,
                        device=self._device,
                    )

    # ─────────────────────────── layout introspection ───────────────────────

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return self._order

    def layer(self, layer_id: int) -> IndexerLayer:
        return self._layers[self._require_layer(layer_id)]

    def full_layer_ids(self) -> tuple[int, ...]:
        return tuple(i for i in self._order if self._layers[i].kind == "full")

    def shared_layer_ids(self) -> tuple[int, ...]:
        return tuple(i for i in self._order if self._layers[i].kind == "shared")

    def _require_layer(self, layer_id: int) -> int:
        if layer_id not in self._layers:
            raise KeyError(f"unknown layer id {layer_id}")
        return layer_id

    def _require_full(self, layer_id: int) -> int:
        layer = self.layer(layer_id)
        if layer.kind != "full":
            raise ValueError(
                f"layer {layer_id} is {layer.kind}; only full layers run the indexer"
            )
        return layer.layer_id

    # ─────────────────────────── buffer surfaces ────────────────────────────

    def selected(self, layer_id: int) -> torch.Tensor:
        """The layer's live ``selected`` buffer for ``sparse_mla.bind``.

        Shared layers return the owner full layer's buffer object: the same
        ``data_ptr``, no copy, no recompute.
        """
        layer = self.layer(layer_id)
        return self._selected[layer.owner_id]

    def indexer_output(self, layer_id: int) -> torch.Tensor:
        """The ``output_indices`` operand for ``dsa_indexer.bind`` (full only).

        The plan must declare ``Caps(output_index_space="physical")`` so the
        slots written here are physical cache slots that feed
        ``sparse_mla.bind(selected_indices=...)`` directly.
        """
        owner = self._require_full(layer_id)
        return self._selected[owner]

    def index_key_pages(self, layer_id: int) -> torch.Tensor | None:
        """The layer's paged FP8 index-key pages, or None for shared layers."""
        layer = self.layer(layer_id)
        if layer.kind != "full":
            return None
        return self._pages.get(layer.layer_id)

    def physical_caps(self, layer_id: int, **caps_kwargs):
        """``dsa_indexer.Caps`` for a full layer, pinned to physical slot space."""
        from .api import Caps

        owner = self._require_full(layer_id)
        forced = dict(
            device=self._device,
            topk=self._topk,
            max_q_rows=self._max_q_rows,
            output_index_space="physical",
        )
        overlap = forced.keys() & caps_kwargs.keys()
        if overlap:
            raise ValueError(f"caps fields owned by the IndexCache: {sorted(overlap)}")
        return Caps(**forced, **caps_kwargs)

    # ──────────────────────────── accounting ───────────────────────────────

    @property
    def topk(self) -> int:
        return self._topk

    @property
    def max_q_rows(self) -> int:
        return self._max_q_rows

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def num_index_key_pages(self) -> int:
        """Pages actually allocated: full layers only (shared allocate none)."""
        return len(self._pages) * self._index_pages_per_layer

    @property
    def index_key_bytes(self) -> int:
        return self.num_index_key_pages * self._page_bytes

    @property
    def selection_exchange_count(self) -> int:
        """Cross-rank selection exchanges per step under context parallelism."""
        return len(self.full_layer_ids())

    def shares_selection(self, layer_id: int, other: int) -> bool:
        """True when both layers read the same buffer (same ``data_ptr``)."""
        return self.selected(layer_id).data_ptr() == self.selected(other).data_ptr()
