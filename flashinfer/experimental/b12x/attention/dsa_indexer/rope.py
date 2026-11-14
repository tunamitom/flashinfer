"""Hy4 indexer key RoPE: interleaved rotation applied at cache-write time.

Hy4's lightning indexer keys are 128-dim (``index_head_dim``) with the
rotary positional block in the LAST ``qk_rope_head_dim`` = 64 dims — the
checkpoint/PTM layout.  The rotation is the interleaved
(``is_neox_style=False``) adjacent-pair form with ``rope_theta = 1e7``, and
``k_norm`` (LayerNorm over the full 128 dims, ``eps=1e-6``, FP32) runs BEFORE
the nope/pe split.  Keys are rotated ONCE at cache-write time (the FP8 key
path: :func:`pack_index_k_cache_reference` writing the paged index cache);
the scorer reads post-rope keys and never rotates.

Witness: vLLM ``ref/vllm_hy4_attention.py`` ``Indexer.prepare_inputs`` —
``k = self.k_norm(k)`` -> ``torch.split(k, [head_dim - rope_dim, rope_dim])``
-> ``rotary_emb(positions, q_pe, k_pe)`` with ``is_neox_style=False`` ->
``torch.cat([k_nope, k_pe])`` reassembled nope-first.

The split-half (NeoX) rotation is the mutant and must stay dead: the
checkpoint stores q_pe/k_pe in interleaved layout, so the split-half form
rotates the wrong dim pairs, loses the trained relative-position geometry
and corrupts the DSA top-k selection.

QUERY SIDE (``prepare_indexer_queries``): the witness rotates the indexer
queries the same way — pe in the LAST ``qk_rope_head_dim`` = 64 dims,
adjacent-pair interleaved rotation, ``rope_theta = 1e7`` — but the queries
get NO ``k_norm`` (only keys pass ``self.k_norm`` in the witness) and they
are rotated per call, not at cache-write time: query RoPE depends on the
query's own position and cannot be baked into any cache.
"""

from __future__ import annotations

import torch

INDEX_HEAD_DIM = 128
INDEX_ROPE_DIM = 64
INDEX_ROPE_THETA = 1.0e7
INDEX_K_NORM_EPS = 1e-6

__all__ = [
    "INDEX_HEAD_DIM",
    "INDEX_ROPE_DIM",
    "INDEX_ROPE_THETA",
    "INDEX_K_NORM_EPS",
    "indexer_rope_inv_freq",
    "apply_indexer_rope",
    "indexer_k_norm",
    "prepare_indexer_keys",
    "prepare_indexer_queries",
]


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def indexer_rope_inv_freq(
    rope_dim: int = INDEX_ROPE_DIM,
    theta: float = INDEX_ROPE_THETA,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Per-pair inverse frequencies ``theta ** (-2i / rope_dim)``, i < dim/2."""
    if rope_dim <= 0 or rope_dim % 2:
        raise ValueError(f"rope_dim must be positive and even, got {rope_dim}")
    if not theta > 0:
        raise ValueError(f"theta must be positive, got {theta}")
    half = rope_dim // 2
    exponents = torch.arange(half, device=device, dtype=torch.float64) * (
        -2.0 / rope_dim
    )
    return (theta ** exponents).to(dtype)


def apply_indexer_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    rope_dim: int = INDEX_ROPE_DIM,
    theta: float = INDEX_ROPE_THETA,
) -> torch.Tensor:
    """Interleaved (``is_neox_style=False``) RoPE over the LAST ``rope_dim`` dims.

    ``x`` is ``[tokens, ..., head_dim]``; the leading ``head_dim - rope_dim``
    dims pass through untouched (the checkpoint keeps pe at the end).  Each
    adjacent pair ``(2i, 2i + 1)`` of the pe block rotates by
    ``positions[t] * theta ** (-2i / rope_dim)``:

    ``x'[2i] = x[2i] * cos - x[2i+1] * sin``;
    ``x'[2i+1] = x[2i] * sin + x[2i+1] * cos``.

    Math runs in FP32 (FP64 for FP64 input) and the result has that compute
    dtype; the rotary angle itself is always evaluated in FP64.
    """
    if x.ndim < 2:
        raise ValueError(f"x must be rank >= 2, got {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError(f"x must be floating point, got {x.dtype}")
    if rope_dim <= 0 or rope_dim % 2 or rope_dim > x.shape[-1]:
        raise ValueError(
            f"rope_dim must be positive, even and <= {x.shape[-1]}, got {rope_dim}"
        )
    pos = torch.as_tensor(positions, device=x.device).reshape(-1)
    if pos.numel() != x.shape[0]:
        raise ValueError(
            f"positions must cover {x.shape[0]} tokens, got {pos.numel()}"
        )

    dtype = _compute_dtype(x)
    head_dim = int(x.shape[-1])
    nope_dim = head_dim - rope_dim
    xf = x.to(dtype)
    nope = xf[..., :nope_dim]
    pe = xf[..., nope_dim:].reshape(*xf.shape[:-1], rope_dim // 2, 2)

    inv_freq = indexer_rope_inv_freq(
        rope_dim, theta, device=x.device, dtype=torch.float64
    )
    # The angle stays FP64 through cos/sin (positions grow with the sequence,
    # and a FP32 angle loses ~1e-5 rad at long range); only the final
    # multiply-add runs in the compute dtype.
    angle = pos.to(torch.float64).unsqueeze(1) * inv_freq.unsqueeze(0)  # [t, pairs]
    angle = angle.reshape(xf.shape[0], *([1] * (xf.ndim - 2)), rope_dim // 2, 1)
    cos = angle.cos().to(dtype)
    sin = angle.sin().to(dtype)
    x0 = pe[..., 0:1]
    x1 = pe[..., 1:2]
    rot0 = x0 * cos - x1 * sin
    rot1 = x0 * sin + x1 * cos
    pe_rot = torch.cat([rot0, rot1], dim=-1).reshape(*xf.shape[:-1], rope_dim)
    return torch.cat([nope, pe_rot], dim=-1)


def indexer_k_norm(
    k: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = INDEX_K_NORM_EPS,
) -> torch.Tensor:
    """LayerNorm over the full 128-dim key (vLLM ``nn.LayerNorm(head_dim)``).

    Runs BEFORE the nope/pe split, so the mean/variance span the whole key
    vector, not each half.
    """
    if not k.is_floating_point():
        raise TypeError(f"k must be floating point, got {k.dtype}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    dtype = _compute_dtype(k)
    kf = k.to(dtype)
    mean = kf.mean(dim=-1, keepdim=True)
    var = kf.var(dim=-1, keepdim=True, unbiased=False)
    out = (kf - mean) * torch.rsqrt(var + eps)
    if weight is not None:
        out = out * weight.to(dtype).reshape(*([1] * (k.ndim - 1)), -1)
    if bias is not None:
        out = out + bias.to(dtype).reshape(*([1] * (k.ndim - 1)), -1)
    return out


def prepare_indexer_keys(
    k: torch.Tensor,
    positions: torch.Tensor,
    *,
    k_norm_weight: torch.Tensor | None = None,
    k_norm_bias: torch.Tensor | None = None,
    k_norm_eps: float = INDEX_K_NORM_EPS,
    rope_dim: int = INDEX_ROPE_DIM,
    theta: float = INDEX_ROPE_THETA,
) -> torch.Tensor:
    """Full indexer-key write preparation: ``k_norm`` -> split -> interleaved RoPE.

    ``k`` is ``[tokens, INDEX_HEAD_DIM]`` raw projected keys.  Order matters
    and matches the witness: the norm spans all 128 dims, the split keeps
    ``[nope | pe]`` with pe in the last ``rope_dim`` dims, and only the pe
    block rotates.  The output is what the FP8 key path stores in the paged
    index cache (keys rotated at write time).
    """
    if k.ndim != 2 or k.shape[1] != INDEX_HEAD_DIM:
        raise ValueError(
            f"k must have shape (tokens, {INDEX_HEAD_DIM}), got {tuple(k.shape)}"
        )
    k_normed = indexer_k_norm(
        k, weight=k_norm_weight, bias=k_norm_bias, eps=k_norm_eps
    )
    return apply_indexer_rope(k_normed, positions, rope_dim=rope_dim, theta=theta)


def prepare_indexer_queries(
    q: torch.Tensor,
    positions: torch.Tensor,
    *,
    rope_dim: int = INDEX_ROPE_DIM,
    theta: float = INDEX_ROPE_THETA,
) -> torch.Tensor:
    """Indexer query preparation: interleaved RoPE over the trailing pe block,
    NO norm.

    ``q`` is ``[tokens, INDEX_HEAD_DIM]`` (or ``[tokens, heads,
    INDEX_HEAD_DIM]`` — every head rotates identically) raw projected queries,
    the witness's ``wq_b`` output before quantization.  Order matches
    ``Indexer.prepare_inputs``: the checkpoint layout keeps the rotary block in
    the LAST ``rope_dim`` dims (``torch.split(q, [head_dim - rope_dim,
    rope_dim])``), only that block rotates (adjacent-pair interleaved,
    ``is_neox_style=False``, ``rope_theta``), and the result is reassembled
    nope-first.

    Queries get NO ``k_norm``: the witness normalizes keys only
    (``k = self.k_norm(k)``; ``q`` goes straight from ``wq_b`` to the split).
    Applying the key norm here is as wrong as the NeoX rotation — it moves the
    query off the trained score geometry and corrupts the DSA top-k selection.
    """
    if q.ndim < 2:
        raise ValueError(f"q must be rank >= 2, got {tuple(q.shape)}")
    if int(q.shape[-1]) != INDEX_HEAD_DIM:
        raise ValueError(
            f"q must have {INDEX_HEAD_DIM} columns (got {int(q.shape[-1])}); "
            "indexer queries are INDEX_HEAD_DIM wide"
        )
    return apply_indexer_rope(q, positions, rope_dim=rope_dim, theta=theta)
