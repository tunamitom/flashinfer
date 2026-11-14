"""Pure-torch reference for Hy4 iHC (identity Hyper-Connections).

iHC is DeepSeek-V4 mHC with the comb/Sinkhorn step removed: the mixing weight
``fn`` carries ``2 * hc_mult = 8`` rows (4 pre + 4 post) instead of mHC's 24
rows (4 pre + 4 post + 16 comb), and the gates are plain sigmoid gates instead
of a Sinkhorn-normalized 4x4 mix.

Numerics follow the tencent/Hy4-preview reference implementations exactly:

- transformers ``HYV4HyperConnection.forward`` / ``HYV4HyperHead.forward``
  (``modeling_hy_v4.py``),
- vLLM ``HYV4HCPreLayer`` / ``HYV4HCPostLayer`` (``vllm/models/hy_v4``).

Per sublayer (``ihc_pre_reference``), with ``streams`` of shape ``[..., 4, H]``::

    flat   = streams.flatten(-2).float()                     # [..., 4H]
    rsqrt  = rsqrt(mean(flat^2, -1, keepdim=True) + norm_eps)
    mixes  = linear(flat, fn) * rsqrt                        # [..., 8], fn [8, 4H]
    pre    = sigmoid(mixes[..., :4] * hc_scale[0] + hc_base[:4]) + hc_eps
    post   = magnitude * sigmoid(mixes[..., 4:] * hc_scale[1] + hc_base[4:]) + hc_eps
    y      = sum_i pre_i * streams_i                         # [..., H], bf16

and the sublayer update (``ihc_post_reference``)::

    out[..., i, :] = post_i * branch[..., :] + streams[..., i, :]     # fp32, bf16 out

The model defaults are ``magnitude=2.0``, ``hc_eps=1e-6``, ``norm_eps=1e-5``
(``hc_magnitude`` / ``hc_eps`` / ``rms_norm_eps`` in the Hy4 config). All math
is FP32 with exactly one BF16 rounding on the way out, matching the forced-FP32
reference forward. The projection is FP32 with FP32 accumulation (``fn``,
``hc_scale``, ``hc_base`` are FP32 parameters in the checkpoint).

This module is device-agnostic pure torch: it is the golden model for
``_ihc_impl`` (the GPU operator) and for the FLARE recipe until the recipe
switches to the operator. No validation beyond the 8-row ``fn`` contract lives
here; argument validation is the operator's job.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IHC_MULT = 4
IHC_MIXES = 2 * IHC_MULT  # 8 rows: 4 pre + 4 post; mHC's 16 comb rows are gone

__all__ = [
    "IHC_MULT",
    "IHC_MIXES",
    "ihc_pre_reference",
    "ihc_post_reference",
    "ihc_head_reference",
]


def _check_mix_fn(fn: torch.Tensor, *, hidden_size: int) -> None:
    expected = (IHC_MIXES, IHC_MULT * hidden_size)
    if tuple(fn.shape) != expected:
        raise ValueError(
            f"fn must have the iHC 8-row layout {expected} (4 pre + 4 post rows, "
            f"no comb rows); got {tuple(fn.shape)}"
        )


def ihc_pre_reference(
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    magnitude: float = 2.0,
    hc_eps: float = 1.0e-6,
    norm_eps: float = 1.0e-5,
    y_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse the 4 streams into the sublayer input and emit the post gates.

    Args:
        streams: ``[..., 4, H]`` BF16 (or FP32) residual streams.
        fn: FP32 ``[8, 4H]`` iHC mixing weights (4 pre rows + 4 post rows).
        hc_scale: FP32 ``[2]`` logit scales (pre, post).
        hc_base: FP32 ``[8]`` logit biases (4 pre + 4 post).
        magnitude: post-gate magnitude (``hc_magnitude``, default 2.0).
        hc_eps: gate epsilon added after sigmoid (``hc_eps``, default 1e-6).
        norm_eps: RMS epsilon over the flattened ``4H`` row
            (``rms_norm_eps``, default 1e-5).
        y_dtype: dtype of the collapsed output (default: ``streams.dtype``).

    Returns:
        ``(post, y)``: FP32 post gates ``[..., 4]`` and collapsed sublayer input
        ``[..., H]`` (same order as ``HYV4HyperConnection.forward``).
    """
    hidden_size = int(streams.shape[-1])
    _check_mix_fn(fn, hidden_size=hidden_size)
    flat = streams.flatten(-2).float()  # [..., 4H]
    mixes = F.linear(flat, fn.float()) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + norm_eps
    )
    pre_logits, post_logits = mixes.split(IHC_MULT, dim=-1)
    pre = (
        torch.sigmoid(pre_logits * hc_scale[0].float() + hc_base[:IHC_MULT].float())
        + hc_eps
    )
    post = (
        magnitude
        * torch.sigmoid(
            post_logits * hc_scale[1].float() + hc_base[IHC_MULT:].float()
        )
        + hc_eps
    )
    y = (pre.unsqueeze(-1) * streams.float()).sum(dim=-2)
    y = y.to(streams.dtype if y_dtype is None else y_dtype)
    return post, y


def ihc_post_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    """Redistribute the sublayer output over the streams.

    ``out[..., i, :] = post[..., i] * x[..., :] + residual[..., i, :]``,
    accumulated in FP32 with one BF16 rounding (as
    ``HYV4DecoderLayer`` applies ``attn_hc`` / ``ffn_hc``).

    Args:
        x: ``[..., H]`` sublayer output (attention or MLP branch output).
        residual: ``[..., 4, H]`` streams that entered the sublayer.
        post: FP32 ``[..., 4]`` post gates from :func:`ihc_pre_reference`
            (``[..., 4, 1]`` is also accepted).

    Returns:
        ``[..., 4, H]`` in ``x.dtype``.
    """
    if post.ndim == residual.ndim:  # [..., 4, 1] form also accepted
        if post.shape[-1] != 1:
            raise ValueError(
                f"post must have shape [..., {IHC_MULT}] or [..., {IHC_MULT}, 1]; "
                f"got {tuple(post.shape)}"
            )
        post = post.squeeze(-1)
    elif post.ndim != residual.ndim - 1:
        raise ValueError(
            f"post must have shape [..., {IHC_MULT}] or [..., {IHC_MULT}, 1]; "
            f"got {tuple(post.shape)}"
        )
    out = (
        post.float().unsqueeze(-1) * x.float().unsqueeze(-2) + residual.float()
    )
    return out.to(x.dtype)


def ihc_head_reference(
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    *,
    hc_eps: float = 1.0e-6,
    norm_eps: float = 1.0e-5,
    y_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Final stream merge before the output norm (``HYV4HyperHead``).

    Same reduction as the pre collapse but with its own ``[4, 4H]`` weights,
    a scalar scale ``[1]`` and per-stream bias ``[4]``.

    Args:
        streams: ``[..., 4, H]`` streams.
        fn: FP32 ``[4, 4H]`` head mixing weights (4 rows).
        hc_head_scale: FP32 ``[1]`` logit scale.
        hc_head_base: FP32 ``[4]`` logit biases.
        hc_eps, norm_eps: as in :func:`ihc_pre_reference`.
        y_dtype: dtype of the merged output (default: ``streams.dtype``).

    Returns:
        Merged ``[..., H]``.
    """
    hidden_size = int(streams.shape[-1])
    expected = (IHC_MULT, IHC_MULT * hidden_size)
    if tuple(fn.shape) != expected:
        raise ValueError(
            f"head fn must have shape {expected} (4 rows, no comb rows); "
            f"got {tuple(fn.shape)}"
        )
    flat = streams.flatten(-2).float()  # [..., 4H]
    mixes = F.linear(flat, fn.float()) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + norm_eps
    )
    pre = (
        torch.sigmoid(mixes * hc_head_scale[0].float() + hc_head_base.float()) + hc_eps
    )
    y = (pre.unsqueeze(-1) * streams.float()).sum(dim=-2)
    return y.to(streams.dtype if y_dtype is None else y_dtype)
