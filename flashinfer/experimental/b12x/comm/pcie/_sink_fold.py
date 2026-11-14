"""Prepared post-merge attention-sink fold (consult2 defect 13).

The context-parallel merge folds the per-head sink into the merged output and
LSE exactly once, after the cross-rank LSE reduction::

    out  *= sigmoid(lse_e - sink)      # factor from the PRE-fold LSE
    lse_e = logaddexp(lse_e, sink)     # empty rows -> sink

This used to run as eager FP64 torch (``apply_attention_sink_after_merge``)
with fresh allocations per call — un-fusable, uncapturable and slow on the
decode hot path.  :class:`PreparedSinkFold` is the prepared operator: the
geometry and the log base are pinned once (``prepare_sink_fold`` compiles the
Triton kernel up front, before any CUDA-graph capture), and every ``run`` is
a single allocation-free kernel launch folding ``out`` and ``lse`` in place.

``apply_attention_sink_after_merge`` (``pcie_dcp_a2a``) stays as the FP64
torch reference; the prepared operator is numerically identical to it (FP64
math inside the kernel, one FP32 store rounding on the folded LSE).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

__all__ = [
    "PreparedSinkFold",
    "prepare_sink_fold",
]

_LN2 = math.log(2.0)
_LSE_BASES = ("base2", "natural")

def _load_kernel():
    from ._sink_fold_kernel import sink_fold_kernel

    return sink_fold_kernel


def _fold_sink_torch_reference(
    out: torch.Tensor,
    lse: torch.Tensor,
    sink: torch.Tensor,
    *,
    lse_base: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The eager FP64 fold (kept for CPU operands and cross-checking)."""
    from .pcie_dcp_a2a import apply_attention_sink_after_merge

    return apply_attention_sink_after_merge(
        out,
        lse,
        sink,
        lse_base=lse_base,
        partials_sink_folded=False,
    )


@dataclass(frozen=True)
class PreparedSinkFold:
    """A prepared post-merge sink fold: pinned geometry, one launch per run.

    Build with :func:`prepare_sink_fold` (compiles the kernel), then call
    :meth:`run` on matching operands — allocation-free and CUDA-graph safe.
    """

    batch: int
    heads: int
    head_dim: int
    out_dtype: torch.dtype
    lse_base: str
    device: torch.device

    def _check(
        self, out: torch.Tensor, lse: torch.Tensor, sink: torch.Tensor
    ) -> None:
        if tuple(out.shape) != (self.batch, self.heads, self.head_dim):
            raise ValueError(
                f"PreparedSinkFold expects out "
                f"{(self.batch, self.heads, self.head_dim)}, got {tuple(out.shape)}"
            )
        if tuple(lse.shape) != (self.batch, self.heads):
            raise ValueError(
                f"PreparedSinkFold expects lse {(self.batch, self.heads)}, "
                f"got {tuple(lse.shape)}"
            )
        if out.dtype != self.out_dtype:
            raise ValueError(
                f"PreparedSinkFold expects out dtype {self.out_dtype}, got {out.dtype}"
            )
        if lse.dtype != torch.float32:
            raise ValueError(f"lse must be float32, got {lse.dtype}")
        if lse.stride(1) != 1 or out.stride(2) != 1:
            raise ValueError("PreparedSinkFold operands must be dense in the last dim")
        if int(sink.numel()) != self.heads:
            raise ValueError(
                f"PreparedSinkFold expects {self.heads} sink logits, "
                f"got {tuple(sink.shape)}"
            )
        if (
            out.device != self.device
            or lse.device != self.device
            or sink.device != self.device
        ):
            raise ValueError("PreparedSinkFold operands must be on the prepared device")

    def run(
        self, out: torch.Tensor, lse: torch.Tensor, sink: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold the sink into ``out`` and ``lse`` in place; returns both."""
        self._check(out, lse, sink)
        if self.device.type != "cuda":
            return _fold_sink_torch_reference(out, lse, sink, lse_base=self.lse_base)
        sink = sink.reshape(-1).to(dtype=torch.float32, device=self.device)
        kernel = _load_kernel()
        block_d = 1
        while block_d < self.head_dim:
            block_d *= 2
        block_d = min(block_d, 1024)
        kernel[(self.batch * self.heads,)](
            out,
            lse,
            sink,
            self.batch,
            self.heads,
            out.stride(0),
            out.stride(1),
            lse.stride(0),
            LN2_F64=_LN2,
            BASE2=self.lse_base != "natural",
            HEAD_DIM=self.head_dim,
            BLOCK_D=block_d,
            OUT_IS_BF16=out.dtype is torch.bfloat16,
            OUT_IS_FP16=out.dtype is torch.float16,
            num_warps=1,
        )
        return out, lse


def prepare_sink_fold(
    *,
    out: torch.Tensor,
    lse: torch.Tensor,
    sink: torch.Tensor,
    lse_base: str,
) -> PreparedSinkFold:
    """Prepare (and compile) the post-merge sink fold for fixed operands.

    ``lse_base`` is the EXPLICIT log base of ``lse`` (``"base2"`` |
    ``"natural"``) — the same token the merge APIs require.  Call this before
    CUDA-graph capture with the exact operand geometry; :meth:`PreparedSinkFold.run`
    then folds matching tensors with one allocation-free launch.
    """
    if lse_base not in _LSE_BASES:
        raise ValueError(f"lse_base must be one of {_LSE_BASES}, got {lse_base!r}")
    if out.ndim != 3 or lse.ndim != 2 or tuple(lse.shape) != tuple(out.shape[:2]):
        raise ValueError("lse must match out without the head_dim")
    if lse.dtype != torch.float32:
        raise ValueError(f"lse must be float32, got {lse.dtype}")
    if int(sink.numel()) != out.shape[1]:
        raise ValueError(
            f"attention_sink must cover {out.shape[1]} heads, got {tuple(sink.shape)}"
        )
    prepared = PreparedSinkFold(
        batch=int(out.shape[0]),
        heads=int(out.shape[1]),
        head_dim=int(out.shape[2]),
        out_dtype=out.dtype,
        lse_base=lse_base,
        device=out.device,
    )
    if out.device.type == "cuda":
        # Compile now so no JIT work happens under later graph capture.
        prepared.run(out.detach().clone(), lse.detach().clone(), sink)
    return prepared
