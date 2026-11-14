"""iHC operator for Hy4 (b12x norm.mhc variant without comb/Sinkhorn).

iHC is DeepSeek-V4 mHC with the comb/Sinkhorn step removed: the mixing weight is
an FP32 ``[8, 4H]`` matrix (4 pre rows + 4 post rows — mHC's 24-row layout with
its 16 comb rows does not apply), the gates are ``pre = sigmoid(..) + hc_eps``
and ``post = magnitude * sigmoid(..) + hc_eps`` (``magnitude`` is ``hc_magnitude
= 2.0`` in the Hy4 config), both input-dependent from the same RMS-normalized
projection, and the sublayer update is ``post_i * branch + stream_i`` with no
comb term. Four streams (``hc_mult=4``) are collapsed before and redistributed
after every sublayer; the streams are replicated on every rank, so the operator
does no collectives.

Geometry: hidden 6144 for Hy4 (the mHC whitelist covers 4096/5120/7168 only).
``IHC_SUPPORTED_HIDDEN_SIZES`` extends it with 6144, which already satisfies the
mHC kernel tile rules (``IHC_SOURCE_TILE_H`` / ``IHC_GRAM_BLOCK_H``); the
per-size launch geometry lives in the ``IHC_TUNING`` table, whose 6144 row is
what the Hy4 recipe serves.

The golden model is ``_ihc_torch_ref`` (pure torch, follows the transformers
``HYV4HyperConnection`` / ``HYV4HyperHead`` and vLLM ``HYV4HCPreLayer`` /
``HYV4HCPostLayer`` semantics of the tencent/Hy4-preview checkpoint). On CUDA
this operator dispatches to the fused Triton kernels in ``_ihc_kernels``
(``B12X_IHC_BACKEND=triton`` forces them, ``B12X_IHC_BACKEND=torch`` forces the
composable torch path, ``auto`` uses Triton when importable). The torch path is
written independently of the reference (einsum/addcmul formulation) so the
operator-versus-reference comparison is never comparing a file to itself.

Caller-owned ``out=`` buffers are supported outside ``torch.compile``; the
allocation-free path is what a decode loop binds. The functional (allocating)
forms run through registered ``b12x::ihc_*`` custom ops so they trace under
torch.compile and CUDA graph capture.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

import torch

IHC_MULT = 4
IHC_MIXES = 2 * IHC_MULT  # 8 rows: 4 pre + 4 post; mHC's comb rows are absent
# The mHC fused-kernel whitelist is (4096, 5120, 7168); Hy4 needs 6144, which
# already satisfies the kernel tile divisibility below.
IHC_SUPPORTED_HIDDEN_SIZES = (4096, 5120, 6144, 7168)
IHC_SOURCE_TILE_H = 128  # same source tile as the mHC Gram kernel
IHC_GRAM_BLOCK_H = 1024  # same Gram block as the mHC Gram kernel
IHC_DEFAULT_MAGNITUDE = 2.0  # hc_magnitude in the Hy4 config
IHC_DEFAULT_HC_EPS = 1.0e-6  # hc_eps in the Hy4 config
IHC_DEFAULT_NORM_EPS = 1.0e-5  # rms_norm_eps in the Hy4 config

__all__ = [
    "IHC_DEFAULT_HC_EPS",
    "IHC_DEFAULT_MAGNITUDE",
    "IHC_DEFAULT_NORM_EPS",
    "IHC_GRAM_BLOCK_H",
    "IHC_MIXES",
    "IHC_MULT",
    "IHC_SOURCE_TILE_H",
    "IHC_SUPPORTED_HIDDEN_SIZES",
    "IHC_TUNING",
    "IhcConfig",
    "is_supported",
    "run_head",
    "run_post",
    "run_pre",
    "tuning_for",
]


@dataclass(frozen=True, kw_only=True)
class IhcConfig:
    """Tuned launch geometry for one hidden size.

    ``block_k`` is the projection K-loop chunk (must divide ``4 * hidden``);
    ``block_d`` is the collapse/update D-loop chunk (must divide ``hidden``);
    ``num_warps`` is the Triton warp count. Tile divisibility mirrors the mHC
    fused Gram kernel (``IHC_SOURCE_TILE_H`` / ``IHC_GRAM_BLOCK_H``).
    """

    hidden_size: int
    block_k: int
    block_d: int
    num_warps: int
    backend: str = "triton_fused"

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden_size": self.hidden_size,
            "block_k": self.block_k,
            "block_d": self.block_d,
            "num_warps": self.num_warps,
            "backend": self.backend,
        }


# Tuning table. One row per whitelisted hidden size; the 6144 row serves Hy4.
IHC_TUNING: Mapping[int, IhcConfig] = {
    4096: IhcConfig(hidden_size=4096, block_k=1024, block_d=256, num_warps=4),
    5120: IhcConfig(hidden_size=5120, block_k=1024, block_d=256, num_warps=4),
    6144: IhcConfig(hidden_size=6144, block_k=1024, block_d=512, num_warps=8),
    7168: IhcConfig(hidden_size=7168, block_k=1024, block_d=512, num_warps=8),
}


def tuning_for(hidden_size: int) -> IhcConfig:
    """Return the tuned launch configuration for a whitelisted hidden size."""
    hidden_size = int(hidden_size)
    config = IHC_TUNING.get(hidden_size)
    if config is None:
        raise ValueError(
            f"iHC hidden size must be one of {IHC_SUPPORTED_HIDDEN_SIZES} "
            f"(mHC's 4096/5120/7168 plus Hy4's 6144); got {hidden_size}"
        )
    if (
        hidden_size % IHC_SOURCE_TILE_H
        or hidden_size % IHC_GRAM_BLOCK_H
        or (IHC_MULT * hidden_size) % config.block_k
        or hidden_size % config.block_d
    ):
        raise ValueError(
            f"hidden size {hidden_size} violates the iHC tile divisibility "
            f"({IHC_SOURCE_TILE_H} / {IHC_GRAM_BLOCK_H} / block_k / block_d)"
        )
    return config


def is_supported(device=None) -> bool:
    """True when the operator can serve on ``device`` (CUDA; Triton optional)."""
    if not torch.cuda.is_available():
        return False
    if device is not None and torch.device(device).type != "cuda":
        return False
    return True


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _resolve_backend(streams: torch.Tensor) -> str:
    mode = os.environ.get("B12X_IHC_BACKEND", "auto").strip().lower()
    if mode in ("torch", "eager"):
        return "torch"
    if mode == "triton":
        if not _triton_available():
            raise RuntimeError("B12X_IHC_BACKEND=triton but triton is not importable")
        return "triton"
    if mode not in ("auto", ""):
        raise ValueError(f"unknown B12X_IHC_BACKEND value {mode!r}")
    return "triton" if (streams.is_cuda and _triton_available()) else "torch"


def _require_contiguous(tensor: torch.Tensor, *, name: str) -> None:
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_eps(name: str, value: float) -> float:
    value = float(value)
    if not value >= 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _validate_streams(streams: torch.Tensor, *, name: str = "streams") -> int:
    if streams.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if streams.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be torch.bfloat16, got {streams.dtype}")
    if streams.ndim < 3 or streams.shape[-2] != IHC_MULT:
        raise ValueError(
            f"{name} must have shape [..., 4, H], got {tuple(streams.shape)}"
        )
    tuning_for(int(streams.shape[-1]))  # whitelist + tile check
    _require_contiguous(streams, name=name)
    return int(streams.shape[-1])


def _validate_fn(
    fn: torch.Tensor, *, hidden_size: int, rows: int, name: str = "fn"
) -> None:
    expected = (rows, IHC_MULT * hidden_size)
    if fn.dtype != torch.float32:
        raise ValueError(f"{name} must be torch.float32, got {fn.dtype}")
    if tuple(fn.shape) != expected:
        raise ValueError(
            f"{name} must have the iHC {rows}-row layout {expected} "
            f"(no mHC comb rows); got {tuple(fn.shape)}"
        )
    _require_contiguous(fn, name=name)


def _validate_gate_weights(
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    device: torch.device,
    name_scale: str = "hc_scale",
    name_base: str = "hc_base",
) -> None:
    if hc_scale.dtype != torch.float32 or tuple(hc_scale.shape) not in ((2,),):
        raise ValueError(
            f"{name_scale} must be float32 shape [2], got {hc_scale.dtype} "
            f"{tuple(hc_scale.shape)}"
        )
    if hc_base.dtype != torch.float32 or tuple(hc_base.shape) != (IHC_MIXES,):
        raise ValueError(
            f"{name_base} must be float32 shape [{IHC_MIXES}], got {hc_base.dtype} "
            f"{tuple(hc_base.shape)}"
        )
    if hc_scale.device != device or hc_base.device != device:
        raise ValueError(f"{name_scale} and {name_base} must be on the residual device")
    _require_contiguous(hc_scale, name=name_scale)
    _require_contiguous(hc_base, name=name_base)


def _flatten_tokens(tensor: torch.Tensor, tail: int) -> torch.Tensor:
    return tensor.reshape(-1, tail)


def _validate_out(
    out: torch.Tensor | None,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor | None:
    if out is None:
        return None
    if tuple(out.shape) != shape or out.dtype != dtype or out.device != device:
        raise ValueError(
            f"{name} must have shape {shape}, dtype {dtype}, and device {device}; "
            f"got shape={tuple(out.shape)}, dtype={out.dtype}, device={out.device}"
        )
    _require_contiguous(out, name=name)
    return out


# ---------------------------------------------------------------------------
# Torch backend (composable; written with an einsum/addcmul formulation that is
# independent of the reference's linear/split/sum formulation).
# ---------------------------------------------------------------------------


def _ihc_pre_torch(
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_n = streams.numel() // (IHC_MULT * streams.shape[-1])
    hidden = int(streams.shape[-1])
    flat = _flatten_tokens(streams, IHC_MULT * hidden).float()
    inv_rms = torch.rsqrt(flat.square().mean(dim=-1) + norm_eps)
    mixes = torch.einsum("mk,tk->tm", fn, flat) * inv_rms.unsqueeze(-1)
    pre = (
        torch.sigmoid(mixes[:, :IHC_MULT] * hc_scale[0] + hc_base[:IHC_MULT]) + hc_eps
    )
    post = (
        magnitude * torch.sigmoid(mixes[:, IHC_MULT:] * hc_scale[1] + hc_base[IHC_MULT:])
        + hc_eps
    )
    streams_flat = _flatten_tokens(streams, IHC_MULT * hidden).float()
    y = torch.einsum("tih,ti->th", streams_flat.view(t_n, IHC_MULT, hidden), pre)
    return (
        post.reshape(streams.shape[:-2] + (IHC_MULT,)),
        y.reshape(streams.shape[:-2] + (hidden,)).to(streams.dtype),
    )


def _ihc_post_torch(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor
) -> torch.Tensor:
    out = torch.addcmul(
        residual.float(),
        post.float().unsqueeze(-1),
        x.float().unsqueeze(-2),
    )
    return out.to(x.dtype)


def _ihc_head_torch(
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    *,
    hc_eps: float,
    norm_eps: float,
) -> torch.Tensor:
    t_n = streams.numel() // (IHC_MULT * streams.shape[-1])
    hidden = int(streams.shape[-1])
    flat = _flatten_tokens(streams, IHC_MULT * hidden).float()
    inv_rms = torch.rsqrt(flat.square().mean(dim=-1) + norm_eps)
    mixes = torch.einsum("mk,tk->tm", fn, flat) * inv_rms.unsqueeze(-1)
    pre = torch.sigmoid(mixes * hc_head_scale[0] + hc_head_base) + hc_eps
    streams_flat = _flatten_tokens(streams, IHC_MULT * hidden).float()
    y = torch.einsum("tih,ti->th", streams_flat.view(t_n, IHC_MULT, hidden), pre)
    return y.reshape(streams.shape[:-2] + (hidden,)).to(streams.dtype)


# ---------------------------------------------------------------------------
# Triton backend (fused kernels; see _ihc_kernels).
# ---------------------------------------------------------------------------


def _ihc_pre_triton(
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
    post: torch.Tensor,
    y: torch.Tensor,
    config: IhcConfig,
) -> None:
    from ._ihc_kernels import _ihc_pre_kernel

    hidden = int(streams.shape[-1])
    t_n = streams.numel() // (IHC_MULT * hidden)
    streams_flat = _flatten_tokens(streams, IHC_MULT * hidden)
    y_flat = _flatten_tokens(y, hidden)
    post_flat = _flatten_tokens(post, IHC_MULT)
    _ihc_pre_kernel[(t_n,)](
        streams_flat,
        fn,
        hc_scale,
        hc_base,
        post_flat,
        y_flat,
        hidden,
        IHC_MULT * hidden,
        float(magnitude),
        float(hc_eps),
        float(norm_eps),
        BLOCK_K=config.block_k,
        BLOCK_D=config.block_d,
        num_warps=config.num_warps,
    )


def _ihc_post_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    out: torch.Tensor,
    config: IhcConfig,
) -> None:
    from ._ihc_kernels import _ihc_post_kernel

    hidden = int(residual.shape[-1])
    t_n = residual.numel() // (IHC_MULT * hidden)
    x_flat = _flatten_tokens(x, hidden)
    residual_flat = _flatten_tokens(residual, IHC_MULT * hidden)
    post_flat = _flatten_tokens(post, IHC_MULT)
    out_flat = _flatten_tokens(out, IHC_MULT * hidden)
    grid = (t_n, triton_cdiv(hidden, config.block_d))
    _ihc_post_kernel[grid](
        x_flat,
        residual_flat,
        post_flat,
        out_flat,
        hidden,
        BLOCK_D=config.block_d,
        num_warps=config.num_warps,
    )


def _ihc_head_triton(
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    *,
    hc_eps: float,
    norm_eps: float,
    y: torch.Tensor,
    config: IhcConfig,
) -> None:
    from ._ihc_kernels import _ihc_head_kernel

    hidden = int(streams.shape[-1])
    t_n = streams.numel() // (IHC_MULT * hidden)
    streams_flat = _flatten_tokens(streams, IHC_MULT * hidden)
    y_flat = _flatten_tokens(y, hidden)
    _ihc_head_kernel[(t_n,)](
        streams_flat,
        fn,
        hc_head_scale,
        hc_head_base,
        y_flat,
        hidden,
        IHC_MULT * hidden,
        float(hc_eps),
        float(norm_eps),
        BLOCK_K=config.block_k,
        BLOCK_D=config.block_d,
        num_warps=config.num_warps,
    )


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Validated implementations.
# ---------------------------------------------------------------------------


def _ihc_pre_impl(
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
    out_post: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = _validate_streams(streams)
    _validate_fn(fn, hidden_size=hidden, rows=IHC_MIXES)
    _validate_gate_weights(hc_scale, hc_base, device=streams.device)
    magnitude = float(magnitude)
    if not magnitude > 0.0:
        raise ValueError(f"magnitude must be positive, got {magnitude}")
    hc_eps = _validate_eps("hc_eps", hc_eps)
    norm_eps = _validate_eps("norm_eps", norm_eps)
    outer = tuple(streams.shape[:-2])
    config = tuning_for(hidden)
    out_post = _validate_out(
        out_post,
        shape=outer + (IHC_MULT,),
        dtype=torch.float32,
        device=streams.device,
        name="out_post",
    )
    out_y = _validate_out(
        out_y,
        shape=outer + (hidden,),
        dtype=torch.bfloat16,
        device=streams.device,
        name="out_y",
    )
    t_n = streams.numel() // (IHC_MULT * hidden)
    if out_post is None:
        out_post = torch.empty(
            outer + (IHC_MULT,), dtype=torch.float32, device=streams.device
        )
    if out_y is None:
        out_y = torch.empty(
            outer + (hidden,), dtype=torch.bfloat16, device=streams.device
        )
    if t_n == 0:
        return out_post, out_y
    if _resolve_backend(streams) == "triton":
        _ihc_pre_triton(
            streams,
            fn,
            hc_scale,
            hc_base,
            magnitude=magnitude,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            post=out_post,
            y=out_y,
            config=config,
        )
    else:
        post, y = _ihc_pre_torch(
            streams,
            fn,
            hc_scale,
            hc_base,
            magnitude=magnitude,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
        )
        out_post.copy_(post)
        out_y.copy_(y)
    return out_post, out_y


def _ihc_post_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    hidden = _validate_streams(residual, name="residual")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"x must be torch.bfloat16, got {x.dtype}")
    if tuple(x.shape) != tuple(residual.shape[:-2]) + (hidden,):
        raise ValueError(
            f"x must have shape {tuple(residual.shape[:-2]) + (hidden,)}, "
            f"got {tuple(x.shape)}"
        )
    if x.device != residual.device:
        raise ValueError("x and residual must be on the same device")
    _require_contiguous(x, name="x")
    if post.dtype != torch.float32 or tuple(post.shape) != tuple(
        residual.shape[:-2]
    ) + (IHC_MULT,):
        raise ValueError(
            f"post must be float32 shape {tuple(residual.shape[:-2]) + (IHC_MULT,)}, "
            f"got {post.dtype} {tuple(post.shape)}"
        )
    if post.device != residual.device:
        raise ValueError("post must be on the residual device")
    _require_contiguous(post, name="post")
    outer = tuple(residual.shape[:-2])
    out = _validate_out(
        out,
        shape=outer + (IHC_MULT, hidden),
        dtype=torch.bfloat16,
        device=residual.device,
        name="out",
    )
    if out is None:
        out = torch.empty_like(residual)
    if residual.numel() == 0:
        return out
    config = tuning_for(hidden)
    if _resolve_backend(residual) == "triton":
        _ihc_post_triton(x, residual, post, out, config)
    else:
        out.copy_(_ihc_post_torch(x, residual, post))
    return out


def _ihc_head_impl(
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    *,
    hc_eps: float,
    norm_eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    hidden = _validate_streams(streams)
    _validate_fn(fn, hidden_size=hidden, rows=IHC_MULT, name="fn")
    if (
        hc_head_scale.dtype != torch.float32
        or tuple(hc_head_scale.shape) != (1,)
    ):
        raise ValueError(
            "hc_head_scale must be float32 shape [1], got "
            f"{hc_head_scale.dtype} {tuple(hc_head_scale.shape)}"
        )
    if hc_head_base.dtype != torch.float32 or tuple(hc_head_base.shape) != (IHC_MULT,):
        raise ValueError(
            f"hc_head_base must be float32 shape [{IHC_MULT}], got "
            f"{hc_head_base.dtype} {tuple(hc_head_base.shape)}"
        )
    if hc_head_scale.device != streams.device or hc_head_base.device != streams.device:
        raise ValueError("hc_head_scale and hc_head_base must be on the stream device")
    _require_contiguous(hc_head_scale, name="hc_head_scale")
    _require_contiguous(hc_head_base, name="hc_head_base")
    hc_eps = _validate_eps("hc_eps", hc_eps)
    norm_eps = _validate_eps("norm_eps", norm_eps)
    outer = tuple(streams.shape[:-2])
    out = _validate_out(
        out,
        shape=outer + (hidden,),
        dtype=torch.bfloat16,
        device=streams.device,
        name="out",
    )
    if out is None:
        out = torch.empty(
            outer + (hidden,), dtype=torch.bfloat16, device=streams.device
        )
    if streams.numel() == 0:
        return out
    config = tuning_for(hidden)
    if _resolve_backend(streams) == "triton":
        _ihc_head_triton(
            streams,
            fn,
            hc_head_scale,
            hc_head_base,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            y=out,
            config=config,
        )
    else:
        out.copy_(
            _ihc_head_torch(
                streams,
                fn,
                hc_head_scale,
                hc_head_base,
                hc_eps=hc_eps,
                norm_eps=norm_eps,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Registered custom ops (functional, compile/graph-safe) and public entry
# points; the caller-owned-buffer forms run the validated impl directly.
# ---------------------------------------------------------------------------


@torch.library.custom_op("b12x::ihc_pre", mutates_args=())
def _ihc_pre_op(
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _ihc_pre_impl(
        streams,
        fn,
        hc_scale,
        hc_base,
        magnitude=magnitude,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
    )


@_ihc_pre_op.register_fake
def _ihc_pre_op_fake(streams, fn, hc_scale, hc_base, magnitude, hc_eps, norm_eps):
    outer = tuple(streams.shape[:-2])
    hidden = int(streams.shape[-1])
    return (
        torch.empty(outer + (IHC_MULT,), dtype=torch.float32, device=streams.device),
        torch.empty(outer + (hidden,), dtype=torch.bfloat16, device=streams.device),
    )


@torch.library.custom_op("b12x::ihc_post", mutates_args=())
def _ihc_post_op(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor
) -> torch.Tensor:
    return _ihc_post_impl(x, residual, post)


@_ihc_post_op.register_fake
def _ihc_post_op_fake(x, residual, post):
    return torch.empty_like(residual)


@torch.library.custom_op("b12x::ihc_head", mutates_args=())
def _ihc_head_op(
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    hc_eps: float,
    norm_eps: float,
) -> torch.Tensor:
    return _ihc_head_impl(
        streams,
        fn,
        hc_head_scale,
        hc_head_base,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
    )


@_ihc_head_op.register_fake
def _ihc_head_op_fake(streams, fn, hc_head_scale, hc_head_base, hc_eps, norm_eps):
    outer = tuple(streams.shape[:-2])
    hidden = int(streams.shape[-1])
    return torch.empty(outer + (hidden,), dtype=torch.bfloat16, device=streams.device)


def run_pre(
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    magnitude: float = IHC_DEFAULT_MAGNITUDE,
    hc_eps: float = IHC_DEFAULT_HC_EPS,
    norm_eps: float = IHC_DEFAULT_NORM_EPS,
    out_post: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse the streams into the sublayer input and emit the post gates.

    ``(post, y)``: FP32 post gates ``[..., 4]`` and BF16 collapsed sublayer input
    ``[..., H]`` (same order as transformers ``HYV4HyperConnection.forward``;
    vLLM's ``HYV4HCPreLayer`` returns the reverse order). ``fn`` is the iHC
    8-row FP32 ``[8, 4H]`` weight (4 pre + 4 post rows). See the module
    docstring for the exact numerics; ``out_post`` / ``out_y`` accept
    caller-owned destinations outside ``torch.compile``.
    """
    if out_post is None and out_y is None:
        return torch.ops.b12x.ihc_pre(
            streams,
            fn,
            hc_scale,
            hc_base,
            float(magnitude),
            float(hc_eps),
            float(norm_eps),
        )
    if torch.compiler.is_compiling():
        raise RuntimeError("caller-owned iHC buffers are not supported inside Dynamo")
    return _ihc_pre_impl(
        streams,
        fn,
        hc_scale,
        hc_base,
        magnitude=magnitude,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        out_post=out_post,
        out_y=out_y,
    )


def run_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Redistribute the sublayer output: ``out[i] = post_i * x + residual[i]``.

    FP32 accumulation with one BF16 rounding; ``post`` comes from
    :func:`run_pre` (FP32 ``[..., 4]``, ``[..., 4, 1]`` accepted).
    """
    if post.ndim == residual.ndim and post.shape[-1] == 1:
        post = post.squeeze(-1)
    if out is None:
        return torch.ops.b12x.ihc_post(x, residual, post)
    if torch.compiler.is_compiling():
        raise RuntimeError("caller-owned iHC buffers are not supported inside Dynamo")
    return _ihc_post_impl(x, residual, post, out=out)


def run_head(
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    *,
    hc_eps: float = IHC_DEFAULT_HC_EPS,
    norm_eps: float = IHC_DEFAULT_NORM_EPS,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Final stream merge before the output norm (``HYV4HyperHead``)."""
    if out is None:
        return torch.ops.b12x.ihc_head(
            streams, fn, hc_head_scale, hc_head_base, float(hc_eps), float(norm_eps)
        )
    if torch.compiler.is_compiling():
        raise RuntimeError("caller-owned iHC buffers are not supported inside Dynamo")
    return _ihc_head_impl(
        streams,
        fn,
        hc_head_scale,
        hc_head_base,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        out=out,
    )
