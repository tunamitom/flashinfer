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
per-size launch geometry lives in the ``IHC_TUNING`` table (``_ihc_tuning``),
whose 6144 row is what the Hy4 recipe serves.

The golden model is ``_ihc_torch_ref`` (pure torch, follows the transformers
``HYV4HyperConnection`` / ``HYV4HyperHead`` and vLLM ``HYV4HCPreLayer`` /
``HYV4HCPostLayer`` semantics of the tencent/Hy4-preview checkpoint). On CUDA
this operator dispatches to the fused Triton kernels in ``_ihc_kernels``
(``B12X_IHC_BACKEND=triton`` forces them, ``B12X_IHC_BACKEND=torch`` forces the
composable torch path, ``auto`` uses Triton when importable). The torch path is
written independently of the reference (einsum/addcmul formulation) so the
operator-versus-reference comparison is never comparing a file to itself.

Numerical contract: every step accumulates in FP32 and rounds to BF16 exactly
once on the way out ("exactly one BF16 rounding"), including the fused
sublayer-input RMSNorm fold (``norm_weight`` / ``layernorm_eps``), and the
fused ``run_post_pre`` boundary is bit-comparable with the three-launch
``run_post`` -> ``run_pre`` -> RMSNorm pipeline: the gates see the BF16-rounded
updated streams in both.

b12x integration mirrors the mHC fused kernels: ``_ihc_tuning`` registers an
``IhcQuery`` / ``IhcConfig`` ``TuningContract`` (``norm.mhc.ihc``, catalog
variant ``ihc``), ``_ihc_preparation`` exposes ``plan_ihc`` for
``PreparationSession``, and ``B12XIHCCaps`` / ``B12XIHCBinding`` /
``_IhcState.bind`` provide the caller-owned-buffer binding surface. The
functional (allocating) forms run through registered ``b12x::ihc_*`` custom ops
so they trace under ``torch.compile`` and CUDA graph capture; bound forms run
through the mutating ``b12x::ihc_bound`` op. Caller-owned ``out=`` buffers are
also supported outside ``torch.compile``.

Token-row offsets are computed in int64 inside the Triton kernels: a token row
spans ``4 * hidden`` elements and int32 offsets overflow above ~87k rows at
hidden 6144.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from collections.abc import Callable, Mapping

import torch
from b12x._lib.scratch import ScratchBufferSpec
from b12x.preparation import Plan, plan_from_handle, require_prepared

from ._ihc_tuning import (
    IHC_BACKENDS,
    IHC_DEFAULT_HC_EPS,
    IHC_DEFAULT_MAGNITUDE,
    IHC_DEFAULT_NORM_EPS,
    IHC_GRAM_BLOCK_H,
    IHC_MIXES,
    IHC_MULT,
    IHC_SOURCE_TILE_H,
    IHC_SUPPORTED_HIDDEN_SIZES,
    IHC_TUNING,
    IhcConfig,
    IhcQuery,
    tuning_for,
)

IHC_SUPPORTED_RMS_EPS = (IHC_DEFAULT_NORM_EPS,)

__all__ = [
    "B12XIHCBinding",
    "B12XIHCCaps",
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
    "IhcQuery",
    "is_supported",
    "run_head",
    "run_post",
    "run_post_pre",
    "run_pre",
    "tuning_for",
]


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


def _validate_norm_weight(
    norm_weight: torch.Tensor | None,
    *,
    hidden_size: int,
    device: torch.device,
) -> None:
    if norm_weight is None:
        return
    if norm_weight.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            "norm_weight must be float32 or bfloat16, got "
            f"{norm_weight.dtype}"
        )
    if tuple(norm_weight.shape) != (hidden_size,):
        raise ValueError(
            f"norm_weight must have shape {(hidden_size,)}, "
            f"got {tuple(norm_weight.shape)}"
        )
    if norm_weight.device != device:
        raise ValueError("norm_weight must be on the residual device")
    _require_contiguous(norm_weight, name="norm_weight")


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


def _ihc_norm_torch(
    y: torch.Tensor, norm_weight: torch.Tensor, layernorm_eps: float
) -> torch.Tensor:
    yf = y.float()
    rms = torch.rsqrt(yf.square().mean(dim=-1, keepdim=True) + layernorm_eps)
    return (norm_weight.float() * yf * rms).to(y.dtype)


def _ihc_pre_torch(
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
    norm_weight: torch.Tensor | None = None,
    layernorm_eps: float = IHC_DEFAULT_NORM_EPS,
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
    y = y.reshape(streams.shape[:-2] + (hidden,)).to(streams.dtype)
    if norm_weight is not None:
        y = _ihc_norm_torch(y, norm_weight, layernorm_eps)
    return (
        post.reshape(streams.shape[:-2] + (IHC_MULT,)),
        y,
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


def _ihc_post_pre_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
    norm_weight: torch.Tensor | None = None,
    layernorm_eps: float = IHC_DEFAULT_NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = _ihc_post_torch(x, residual, prev_post)
    post, y = _ihc_pre_torch(
        out,
        fn,
        hc_scale,
        hc_base,
        magnitude=magnitude,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        norm_weight=norm_weight,
        layernorm_eps=layernorm_eps,
    )
    return out, post, y


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


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _ihc_pre_triton(
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
    norm_weight: torch.Tensor | None,
    layernorm_eps: float,
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
        norm_weight if norm_weight is not None else hc_base,
        float(layernorm_eps),
        BLOCK_K=config.block_k,
        BLOCK_D=config.block_d,
        FUSE_NORM=norm_weight is not None,
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
        IHC_MULT * hidden,
        BLOCK_D=config.block_d,
        num_warps=config.num_warps,
    )


def _ihc_post_pre_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
    norm_weight: torch.Tensor | None,
    layernorm_eps: float,
    out_streams: torch.Tensor,
    post: torch.Tensor,
    y: torch.Tensor,
    config: IhcConfig,
) -> None:
    from ._ihc_kernels import _ihc_post_pre_kernel

    hidden = int(residual.shape[-1])
    t_n = residual.numel() // (IHC_MULT * hidden)
    x_flat = _flatten_tokens(x, hidden)
    residual_flat = _flatten_tokens(residual, IHC_MULT * hidden)
    prev_post_flat = _flatten_tokens(prev_post, IHC_MULT)
    out_flat = _flatten_tokens(out_streams, IHC_MULT * hidden)
    y_flat = _flatten_tokens(y, hidden)
    post_flat = _flatten_tokens(post, IHC_MULT)
    _ihc_post_pre_kernel[(t_n,)](
        x_flat,
        residual_flat,
        prev_post_flat,
        fn,
        hc_scale,
        hc_base,
        out_flat,
        post_flat,
        y_flat,
        hidden,
        IHC_MULT * hidden,
        float(magnitude),
        float(hc_eps),
        float(norm_eps),
        norm_weight if norm_weight is not None else hc_base,
        float(layernorm_eps),
        BLOCK_K=config.block_k,
        BLOCK_D=config.block_d,
        FUSE_NORM=norm_weight is not None,
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


# ---------------------------------------------------------------------------
# Launchers: one closure per operation, backend and tuning row resolved. The
# preparation module attaches compiled programs to the same entry points.
# ---------------------------------------------------------------------------


def _torch_launcher(operation: str, config: IhcConfig) -> Callable:
    def launch_pre(
        streams,
        fn,
        hc_scale,
        hc_base,
        *,
        magnitude,
        hc_eps,
        norm_eps,
        norm_weight,
        layernorm_eps,
        post,
        y,
    ):
        got_post, got_y = _ihc_pre_torch(
            streams,
            fn,
            hc_scale,
            hc_base,
            magnitude=magnitude,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            norm_weight=norm_weight,
            layernorm_eps=layernorm_eps,
        )
        post.copy_(got_post)
        y.copy_(got_y)

    def launch_post(x, residual, post, *, out):
        out.copy_(_ihc_post_torch(x, residual, post))

    def launch_post_pre(
        x,
        residual,
        prev_post,
        fn,
        hc_scale,
        hc_base,
        *,
        magnitude,
        hc_eps,
        norm_eps,
        norm_weight,
        layernorm_eps,
        out_streams,
        post,
        y,
    ):
        got_out, got_post, got_y = _ihc_post_pre_torch(
            x,
            residual,
            prev_post,
            fn,
            hc_scale,
            hc_base,
            magnitude=magnitude,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            norm_weight=norm_weight,
            layernorm_eps=layernorm_eps,
        )
        out_streams.copy_(got_out)
        post.copy_(got_post)
        y.copy_(got_y)

    def launch_head(streams, fn, hc_head_scale, hc_head_base, *, hc_eps, norm_eps, y):
        y.copy_(
            _ihc_head_torch(
                streams,
                fn,
                hc_head_scale,
                hc_head_base,
                hc_eps=hc_eps,
                norm_eps=norm_eps,
            )
        )

    return {
        "pre": launch_pre,
        "post": launch_post,
        "post_pre": launch_post_pre,
        "head": launch_head,
    }[operation]


def _triton_launcher(operation: str, config: IhcConfig) -> Callable:
    if operation == "pre":

        def launch(
            streams,
            fn,
            hc_scale,
            hc_base,
            *,
            magnitude,
            hc_eps,
            norm_eps,
            norm_weight,
            layernorm_eps,
            post,
            y,
        ):
            _ihc_pre_triton(
                streams,
                fn,
                hc_scale,
                hc_base,
                magnitude=magnitude,
                hc_eps=hc_eps,
                norm_eps=norm_eps,
                norm_weight=norm_weight,
                layernorm_eps=layernorm_eps,
                post=post,
                y=y,
                config=config,
            )

        return launch
    if operation == "post":

        def launch(x, residual, post, *, out):
            _ihc_post_triton(x, residual, post, out, config)

        return launch
    if operation == "post_pre":

        def launch(
            x,
            residual,
            prev_post,
            fn,
            hc_scale,
            hc_base,
            *,
            magnitude,
            hc_eps,
            norm_eps,
            norm_weight,
            layernorm_eps,
            out_streams,
            post,
            y,
        ):
            _ihc_post_pre_triton(
                x,
                residual,
                prev_post,
                fn,
                hc_scale,
                hc_base,
                magnitude=magnitude,
                hc_eps=hc_eps,
                norm_eps=norm_eps,
                norm_weight=norm_weight,
                layernorm_eps=layernorm_eps,
                out_streams=out_streams,
                post=post,
                y=y,
                config=config,
            )

        return launch

    def launch(streams, fn, hc_head_scale, hc_head_base, *, hc_eps, norm_eps, y):
        _ihc_head_triton(
            streams,
            fn,
            hc_head_scale,
            hc_head_base,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            y=y,
            config=config,
        )

    return launch


def _launchers_for(
    config: IhcConfig, backend: str | None = None
) -> Mapping[str, Callable]:
    backend = config.backend if backend is None else backend
    factory = _torch_launcher if backend == "torch" else _triton_launcher
    return {op: factory(op, config) for op in ("pre", "post", "post_pre", "head")}


# ---------------------------------------------------------------------------
# Prepared plan surface: caps / binding / state (the mHC pattern).
# ---------------------------------------------------------------------------


def _canonical_device(device: torch.device | str) -> torch.device:
    result = torch.device(device)
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    return result


@dataclass(frozen=True, kw_only=True)
class B12XIHCCaps:
    """Token capacity and static iHC geometry for one prepared plan."""

    device: torch.device | str
    max_tokens: int
    hidden_size: int
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        device = _canonical_device(self.device)
        max_tokens = int(self.max_tokens)
        hidden_size = int(self.hidden_size)
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        tuning_for(hidden_size)  # whitelist + tile check
        if self.dtype != torch.bfloat16:
            raise ValueError(
                f"iHC scratch currently supports torch.bfloat16 outputs, got {self.dtype}"
            )
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "hidden_size", hidden_size)


@dataclass(frozen=True, kw_only=True)
class B12XIHCBinding:
    """Caller-owned output buffers bound to one prepared iHC state.

    Mutating custom ops target the capacity tensors directly; the accessor
    views expose only the live token prefix to downstream operators.
    """

    state: "_IhcState"
    tokens: int
    out_streams: torch.Tensor | None = None
    post_buffer: torch.Tensor | None = None
    y: torch.Tensor | None = None
    plan: Plan | None = None

    def _view(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor[: self.tokens]

    def pre(self, streams, fn, hc_scale, hc_base, **kwargs):
        return run_pre(
            streams, fn, hc_scale, hc_base, binding=self, **kwargs
        )

    def post(self, x, residual, post, **kwargs):
        return run_post(x, residual, post, binding=self, **kwargs)

    def post_pre(
        self, x, residual, prev_post, fn, hc_scale, hc_base, **kwargs
    ):
        return run_post_pre(
            x, residual, prev_post, fn, hc_scale, hc_base, binding=self, **kwargs
        )

    def head(self, streams, fn, hc_head_scale, hc_head_base, **kwargs):
        return run_head(
            streams, fn, hc_head_scale, hc_head_base, binding=self, **kwargs
        )


@dataclass(frozen=True)
class _IhcState:
    """One immutable, already-lowered iHC operation family and its launchers."""

    caps: B12XIHCCaps
    query: IhcQuery
    config: IhcConfig
    launchers: Mapping[str, Callable]

    def require(
        self,
        operation: str,
        tensor: torch.Tensor,
        *,
        norm_weight: torch.Tensor | None = None,
        magnitude: float | None = None,
        hc_eps: float | None = None,
        norm_eps: float | None = None,
        layernorm_eps: float | None = None,
    ) -> None:
        query = self.query
        if operation != query.operation:
            raise ValueError(f"execution prepares {query.operation}, not {operation}")
        if tensor.device != self.caps.device:
            raise ValueError("iHC tensor and prepared device differ")
        if (
            not 0 <= tensor.shape[0] <= query.max_tokens
            or tensor.shape[-1] != query.hidden_size
        ):
            raise ValueError(
                "iHC tensor exceeds its planned token capacity or differs in hidden size"
            )
        if (norm_weight is not None) != query.has_norm_weight:
            raise ValueError("iHC normalization operand presence changed")
        if norm_weight is not None and (
            str(norm_weight.dtype).removeprefix("torch.") != query.norm_weight_dtype
        ):
            raise ValueError("iHC normalization dtype changed")
        if magnitude is not None and float(magnitude) != query.magnitude:
            raise ValueError("iHC magnitude differs from the prepared invocation")
        if (hc_eps is not None and float(hc_eps) != query.hc_eps) or (
            norm_eps is not None and float(norm_eps) != query.norm_eps
        ):
            raise ValueError("iHC gate recipe differs from the prepared invocation")
        if layernorm_eps is not None and float(layernorm_eps) != query.layernorm_eps:
            raise ValueError(
                "iHC folded RMSNorm epsilon differs from the prepared invocation"
            )

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        """iHC launches need no anonymous scratch allocation."""
        return ()

    def _live_tokens(self, tokens: int | None) -> int:
        live = self.caps.max_tokens if tokens is None else int(tokens)
        if not 0 <= live <= self.caps.max_tokens:
            raise ValueError(f"tokens={live} exceeds capacity {self.caps.max_tokens}")
        return live

    def bind(
        self,
        *,
        out_streams: torch.Tensor | None = None,
        post: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        tokens: int | None = None,
        plan: Plan | None = None,
    ) -> B12XIHCBinding:
        live = self._live_tokens(tokens)
        caps = self.caps
        hidden = int(caps.hidden_size)
        for name, tensor, tail, dtype in (
            ("out_streams", out_streams, (IHC_MULT, hidden), torch.bfloat16),
            ("post", post, (IHC_MULT,), torch.float32),
            ("y", y, (hidden,), torch.bfloat16),
        ):
            if tensor is None:
                continue
            if tensor.ndim != len(tail) + 1 or tuple(tensor.shape[1:]) != tail:
                raise ValueError(
                    f"{name} must have tail shape {tail}, got {tuple(tensor.shape)}"
                )
            if int(tensor.shape[0]) < live:
                raise ValueError(
                    f"{name} capacity {int(tensor.shape[0])} is smaller than "
                    f"tokens={live}"
                )
            if tensor.dtype != dtype or tensor.device != caps.device:
                raise ValueError(
                    f"{name} must use dtype={dtype} and device={caps.device}; "
                    f"got dtype={tensor.dtype}, device={tensor.device}"
                )
            _require_contiguous(tensor, name=name)
        return B12XIHCBinding(
            state=self,
            plan=plan,
            tokens=live,
            out_streams=out_streams,
            post_buffer=post,
            y=y,
        )


def _prepared_state(plan, *, output_mode=None):
    state = require_prepared(plan, "norm.mhc.ihc")
    if output_mode is not None and state.query.output_mode != output_mode:
        raise ValueError("iHC output form differs from prepared invocation")
    return state


def _state_and_launcher(
    operation: str,
    tensor: torch.Tensor,
    hidden: int,
    *,
    state: _IhcState | None,
    norm_weight,
    magnitude,
    hc_eps,
    norm_eps,
    layernorm_eps,
) -> Callable:
    if state is not None:
        state.require(
            operation,
            tensor,
            norm_weight=norm_weight,
            magnitude=magnitude,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            layernorm_eps=layernorm_eps,
        )
        return state.launchers[operation]
    config = tuning_for(hidden)
    backend = _resolve_backend(tensor)
    if config.backend == "torch":
        backend = "torch"
    return _launchers_for(config, backend)[operation]


def _binding_outputs(
    binding: B12XIHCBinding | None,
    *,
    names: tuple[str, ...],
    state: _IhcState | None,
):
    if binding is None:
        return {name: None for name in names}
    if state is not None and binding.state is not state:
        raise ValueError("binding and plan have different prepared states")
    outputs = {}
    for name in names:
        value = getattr(binding, {"out_streams": "out_streams", "post": "post_buffer", "y": "y"}[name])
        if value is None:
            raise ValueError(
                f"iHC binding lacks the {name} buffer this operation writes"
            )
        outputs[name] = value[: binding.tokens]
    return outputs


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
    norm_weight: torch.Tensor | None = None,
    layernorm_eps: float = IHC_DEFAULT_NORM_EPS,
    out_post: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
    binding: B12XIHCBinding | None = None,
    _state: _IhcState | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = _validate_streams(streams)
    _validate_fn(fn, hidden_size=hidden, rows=IHC_MIXES)
    _validate_gate_weights(hc_scale, hc_base, device=streams.device)
    _validate_norm_weight(norm_weight, hidden_size=hidden, device=streams.device)
    magnitude = float(magnitude)
    if not magnitude > 0.0:
        raise ValueError(f"magnitude must be positive, got {magnitude}")
    hc_eps = _validate_eps("hc_eps", hc_eps)
    norm_eps = _validate_eps("norm_eps", norm_eps)
    layernorm_eps = _validate_eps("layernorm_eps", layernorm_eps)
    bound = _binding_outputs(
        binding, names=("post", "y"), state=_state
    )
    if binding is not None:
        extras = [
            name
            for name, value in (("out_post", out_post), ("out_y", out_y))
            if value is not None
        ]
        if extras:
            raise ValueError(
                "iHC binding owns output buffers; do not also pass "
                + ", ".join(extras)
            )
        out_post, out_y = bound["post"], bound["y"]
    outer = tuple(streams.shape[:-2])
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
    launcher = _state_and_launcher(
        "pre",
        streams,
        hidden,
        state=_state,
        norm_weight=norm_weight,
        magnitude=magnitude,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        layernorm_eps=layernorm_eps,
    )
    launcher(
        streams,
        fn,
        hc_scale,
        hc_base,
        magnitude=magnitude,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        norm_weight=norm_weight,
        layernorm_eps=layernorm_eps,
        post=out_post,
        y=out_y,
    )
    return out_post, out_y


def _ihc_post_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    binding: B12XIHCBinding | None = None,
    _state: _IhcState | None = None,
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
    bound = _binding_outputs(binding, names=("out_streams",), state=_state)
    if binding is not None:
        if out is not None:
            raise ValueError(
                "iHC binding owns output buffers; do not also pass out"
            )
        out = bound["out_streams"]
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
    launcher = _state_and_launcher(
        "post",
        residual,
        hidden,
        state=_state,
        norm_weight=None,
        magnitude=None,
        hc_eps=None,
        norm_eps=None,
        layernorm_eps=None,
    )
    launcher(x, residual, post, out=out)
    return out


def _ihc_post_pre_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    magnitude: float = IHC_DEFAULT_MAGNITUDE,
    hc_eps: float = IHC_DEFAULT_HC_EPS,
    norm_eps: float = IHC_DEFAULT_NORM_EPS,
    norm_weight: torch.Tensor | None = None,
    layernorm_eps: float = IHC_DEFAULT_NORM_EPS,
    out_streams: torch.Tensor | None = None,
    out_post: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
    binding: B12XIHCBinding | None = None,
    _state: _IhcState | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    if prev_post.dtype != torch.float32 or tuple(prev_post.shape) != tuple(
        residual.shape[:-2]
    ) + (IHC_MULT,):
        raise ValueError(
            "prev_post must be float32 shape "
            f"{tuple(residual.shape[:-2]) + (IHC_MULT,)}, "
            f"got {prev_post.dtype} {tuple(prev_post.shape)}"
        )
    if prev_post.device != residual.device:
        raise ValueError("prev_post must be on the residual device")
    _require_contiguous(prev_post, name="prev_post")
    _validate_fn(fn, hidden_size=hidden, rows=IHC_MIXES)
    _validate_gate_weights(hc_scale, hc_base, device=residual.device)
    _validate_norm_weight(norm_weight, hidden_size=hidden, device=residual.device)
    magnitude = float(magnitude)
    if not magnitude > 0.0:
        raise ValueError(f"magnitude must be positive, got {magnitude}")
    hc_eps = _validate_eps("hc_eps", hc_eps)
    norm_eps = _validate_eps("norm_eps", norm_eps)
    layernorm_eps = _validate_eps("layernorm_eps", layernorm_eps)
    bound = _binding_outputs(
        binding, names=("out_streams", "post", "y"), state=_state
    )
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("out_streams", out_streams),
                ("out_post", out_post),
                ("out_y", out_y),
            )
            if value is not None
        ]
        if extras:
            raise ValueError(
                "iHC binding owns output buffers; do not also pass "
                + ", ".join(extras)
            )
        out_streams, out_post, out_y = bound["out_streams"], bound["post"], bound["y"]
    outer = tuple(residual.shape[:-2])
    out_streams = _validate_out(
        out_streams,
        shape=outer + (IHC_MULT, hidden),
        dtype=torch.bfloat16,
        device=residual.device,
        name="out_streams",
    )
    out_post = _validate_out(
        out_post,
        shape=outer + (IHC_MULT,),
        dtype=torch.float32,
        device=residual.device,
        name="out_post",
    )
    out_y = _validate_out(
        out_y,
        shape=outer + (hidden,),
        dtype=torch.bfloat16,
        device=residual.device,
        name="out_y",
    )
    t_n = residual.numel() // (IHC_MULT * hidden)
    if out_streams is None:
        out_streams = torch.empty_like(residual)
    if out_post is None:
        out_post = torch.empty(
            outer + (IHC_MULT,), dtype=torch.float32, device=residual.device
        )
    if out_y is None:
        out_y = torch.empty(
            outer + (hidden,), dtype=torch.bfloat16, device=residual.device
        )
    if t_n == 0:
        return out_streams, out_post, out_y
    launcher = _state_and_launcher(
        "post_pre",
        residual,
        hidden,
        state=_state,
        norm_weight=norm_weight,
        magnitude=magnitude,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        layernorm_eps=layernorm_eps,
    )
    launcher(
        x,
        residual,
        prev_post,
        fn,
        hc_scale,
        hc_base,
        magnitude=magnitude,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        norm_weight=norm_weight,
        layernorm_eps=layernorm_eps,
        out_streams=out_streams,
        post=out_post,
        y=out_y,
    )
    return out_streams, out_post, out_y


def _ihc_head_impl(
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    *,
    hc_eps: float,
    norm_eps: float,
    out: torch.Tensor | None = None,
    binding: B12XIHCBinding | None = None,
    _state: _IhcState | None = None,
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
    bound = _binding_outputs(binding, names=("y",), state=_state)
    if binding is not None:
        if out is not None:
            raise ValueError(
                "iHC binding owns output buffers; do not also pass out"
            )
        out = bound["y"]
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
    launcher = _state_and_launcher(
        "head",
        streams,
        hidden,
        state=_state,
        norm_weight=None,
        magnitude=None,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        layernorm_eps=None,
    )
    launcher(
        streams,
        fn,
        hc_head_scale,
        hc_head_base,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        y=out,
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
    norm_weight: torch.Tensor | None,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
    layernorm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _ihc_pre_impl(
        streams,
        fn,
        hc_scale,
        hc_base,
        magnitude=magnitude,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        norm_weight=norm_weight,
        layernorm_eps=layernorm_eps,
    )


@_ihc_pre_op.register_fake
def _ihc_pre_op_fake(
    streams, fn, hc_scale, hc_base, norm_weight, magnitude, hc_eps, norm_eps, layernorm_eps
):
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


@torch.library.custom_op("b12x::ihc_post_pre", mutates_args=())
def _ihc_post_pre_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor | None,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
    layernorm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _ihc_post_pre_impl(
        x,
        residual,
        prev_post,
        fn,
        hc_scale,
        hc_base,
        magnitude=magnitude,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        norm_weight=norm_weight,
        layernorm_eps=layernorm_eps,
    )


@_ihc_post_pre_op.register_fake
def _ihc_post_pre_op_fake(
    x,
    residual,
    prev_post,
    fn,
    hc_scale,
    hc_base,
    norm_weight,
    magnitude,
    hc_eps,
    norm_eps,
    layernorm_eps,
):
    outer = tuple(residual.shape[:-2])
    hidden = int(residual.shape[-1])
    return (
        torch.empty_like(residual),
        torch.empty(outer + (IHC_MULT,), dtype=torch.float32, device=residual.device),
        torch.empty(outer + (hidden,), dtype=torch.bfloat16, device=residual.device),
    )


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


_IHC_BOUND_FIELDS = ("out_streams", "post_buffer", "y")


@torch.library.custom_op("b12x::ihc_bound", mutates_args=("buffers",))
def _ihc_bound_op(
    inputs: list[torch.Tensor],
    buffers: list[torch.Tensor],
    norm_weight: torch.Tensor | None,
    operation: str,
    plan_handle: int,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
    layernorm_eps: float,
) -> None:
    plan = plan_from_handle(plan_handle)
    state = _prepared_state(plan)
    live = state.caps.max_tokens
    for candidate in inputs:
        if candidate.ndim >= 2 and tuple(candidate.shape[-2:])[-1] == state.caps.hidden_size:
            live = int(candidate.shape[0])
            break
    binding = B12XIHCBinding(
        state=state,
        plan=plan,
        tokens=live,
        **{
            name: value[:live] if value.numel() else None
            for name, value in zip(_IHC_BOUND_FIELDS, buffers, strict=True)
        },
    )
    if operation == "pre":
        _ihc_pre_impl(
            *inputs,
            magnitude=magnitude,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            norm_weight=norm_weight,
            layernorm_eps=layernorm_eps,
            binding=binding,
            _state=state,
        )
    elif operation == "post":
        _ihc_post_impl(*inputs, binding=binding, _state=state)
    elif operation == "post_pre":
        _ihc_post_pre_impl(
            *inputs,
            magnitude=magnitude,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            norm_weight=norm_weight,
            layernorm_eps=layernorm_eps,
            binding=binding,
            _state=state,
        )
    elif operation == "head":
        _ihc_head_impl(
            *inputs,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            binding=binding,
            _state=state,
        )
    else:
        raise ValueError("bound iHC requires pre, post, post_pre, or head")


@_ihc_bound_op.register_fake
def _ihc_bound_op_fake(
    inputs, buffers, norm_weight, operation, plan_handle, magnitude, hc_eps, norm_eps, layernorm_eps
):
    pass


def _run_bound(
    binding: B12XIHCBinding,
    *,
    operation: str,
    inputs: list[torch.Tensor],
    norm_weight: torch.Tensor | None,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
    layernorm_eps: float,
):
    if binding.plan is None:
        raise ValueError("bound iHC execution requires a plan on the binding")
    required = {
        "pre": ("post_buffer", "y"),
        "post": ("out_streams",),
        "post_pre": ("out_streams", "post_buffer", "y"),
        "head": ("y",),
    }[operation]
    missing = [name for name in required if getattr(binding, name) is None]
    if missing:
        raise ValueError(
            "bound iHC execution requires the "
            + ", ".join(missing)
            + " buffer(s) on the binding"
        )
    device = binding.state.caps.device
    buffers = [
        getattr(binding, name)
        if getattr(binding, name) is not None
        else torch.empty(0, dtype=torch.bfloat16, device=device)
        for name in _IHC_BOUND_FIELDS
    ]
    torch.ops.b12x.ihc_bound(
        inputs,
        buffers,
        norm_weight,
        operation,
        binding.plan.handle,
        float(magnitude),
        float(hc_eps),
        float(norm_eps),
        float(layernorm_eps),
    )
    tokens = binding.tokens
    return tuple(
        value[:tokens]
        if value.numel() and value.shape[0] != tokens
        else value
        for value in buffers
    )


def run_pre(
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    magnitude: float = IHC_DEFAULT_MAGNITUDE,
    hc_eps: float = IHC_DEFAULT_HC_EPS,
    norm_eps: float = IHC_DEFAULT_NORM_EPS,
    norm_weight: torch.Tensor | None = None,
    layernorm_eps: float = IHC_DEFAULT_NORM_EPS,
    out_post: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
    plan: Plan | None = None,
    binding: B12XIHCBinding | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse the streams into the sublayer input and emit the post gates.

    ``(post, y)``: FP32 post gates ``[..., 4]`` and BF16 collapsed sublayer input
    ``[..., H]`` (same order as transformers ``HYV4HyperConnection.forward``;
    vLLM's ``HYV4HCPreLayer`` returns the reverse order). ``fn`` is the iHC
    8-row FP32 ``[8, 4H]`` weight (4 pre + 4 post rows). With ``norm_weight``
    the sublayer input RMSNorm is folded into the same launch and ``y`` is the
    normalized sublayer input. See the module docstring for the exact numerics.
    ``out_post`` / ``out_y`` accept caller-owned destinations outside
    ``torch.compile``; ``plan`` / ``binding`` route through a prepared
    ``plan_ihc`` state.
    """
    if binding is not None:
        if out_post is not None or out_y is not None:
            raise ValueError(
                "iHC binding owns output buffers; do not also pass out_post/out_y"
            )
        out_streams, post, y = _run_bound(
            binding,
            operation="pre",
            inputs=[streams, fn, hc_scale, hc_base],
            norm_weight=norm_weight,
            magnitude=magnitude,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            layernorm_eps=layernorm_eps,
        )
        return post, y
    if plan is not None:
        if torch.compiler.is_compiling():
            raise RuntimeError("planned iHC is not supported inside Dynamo")
        return _ihc_pre_impl(
            streams,
            fn,
            hc_scale,
            hc_base,
            magnitude=magnitude,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            norm_weight=norm_weight,
            layernorm_eps=layernorm_eps,
            out_post=out_post,
            out_y=out_y,
            _state=_prepared_state(plan),
        )
    if out_post is None and out_y is None:
        return torch.ops.b12x.ihc_pre(
            streams,
            fn,
            hc_scale,
            hc_base,
            norm_weight,
            float(magnitude),
            float(hc_eps),
            float(norm_eps),
            float(layernorm_eps),
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
        norm_weight=norm_weight,
        layernorm_eps=layernorm_eps,
        out_post=out_post,
        out_y=out_y,
    )


def run_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    plan: Plan | None = None,
    binding: B12XIHCBinding | None = None,
) -> torch.Tensor:
    """Redistribute the sublayer output: ``out[i] = post_i * x + residual[i]``.

    FP32 accumulation with one BF16 rounding; ``post`` comes from
    :func:`run_pre` (FP32 ``[..., 4]``, ``[..., 4, 1]`` accepted).
    """
    if post.ndim == residual.ndim and post.shape[-1] == 1:
        post = post.squeeze(-1)
    if binding is not None:
        if out is not None:
            raise ValueError("iHC binding owns output buffers; do not also pass out")
        (out_view, _, _) = _run_bound(
            binding,
            operation="post",
            inputs=[x, residual, post],
            norm_weight=None,
            magnitude=IHC_DEFAULT_MAGNITUDE,
            hc_eps=IHC_DEFAULT_HC_EPS,
            norm_eps=IHC_DEFAULT_NORM_EPS,
            layernorm_eps=IHC_DEFAULT_NORM_EPS,
        )
        return out_view
    if plan is not None:
        if torch.compiler.is_compiling():
            raise RuntimeError("planned iHC is not supported inside Dynamo")
        return _ihc_post_impl(x, residual, post, out=out, _state=_prepared_state(plan))
    if out is None:
        return torch.ops.b12x.ihc_post(x, residual, post)
    if torch.compiler.is_compiling():
        raise RuntimeError("caller-owned iHC buffers are not supported inside Dynamo")
    return _ihc_post_impl(x, residual, post, out=out)


def run_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    magnitude: float = IHC_DEFAULT_MAGNITUDE,
    hc_eps: float = IHC_DEFAULT_HC_EPS,
    norm_eps: float = IHC_DEFAULT_NORM_EPS,
    norm_weight: torch.Tensor | None = None,
    layernorm_eps: float = IHC_DEFAULT_NORM_EPS,
    out_streams: torch.Tensor | None = None,
    out_post: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
    plan: Plan | None = None,
    binding: B12XIHCBinding | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The sublayer boundary in ONE launch: post update + next pre (+ RMSNorm).

    ``(out_streams, post, y)``: the updated BF16 streams ``[..., 4, H]``, the
    next FP32 post gates ``[..., 4]``, and the next BF16 sublayer input
    ``[..., H]`` (RMSNorm-folded when ``norm_weight`` is given). Equivalent to
    ``run_post`` followed by ``run_pre`` — including the single BF16 rounding of
    the updated streams feeding the gates — at three-launch cost.
    """
    if prev_post.ndim == residual.ndim and prev_post.shape[-1] == 1:
        prev_post = prev_post.squeeze(-1)
    if binding is not None:
        if out_streams is not None or out_post is not None or out_y is not None:
            raise ValueError(
                "iHC binding owns output buffers; do not also pass out_streams/"
                "out_post/out_y"
            )
        out_view, post_view, y_view = _run_bound(
            binding,
            operation="post_pre",
            inputs=[x, residual, prev_post, fn, hc_scale, hc_base],
            norm_weight=norm_weight,
            magnitude=magnitude,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            layernorm_eps=layernorm_eps,
        )
        return out_view, post_view, y_view
    if plan is not None:
        if torch.compiler.is_compiling():
            raise RuntimeError("planned iHC is not supported inside Dynamo")
        return _ihc_post_pre_impl(
            x,
            residual,
            prev_post,
            fn,
            hc_scale,
            hc_base,
            magnitude=magnitude,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            norm_weight=norm_weight,
            layernorm_eps=layernorm_eps,
            out_streams=out_streams,
            out_post=out_post,
            out_y=out_y,
            _state=_prepared_state(plan),
        )
    if out_streams is None and out_post is None and out_y is None:
        return torch.ops.b12x.ihc_post_pre(
            x,
            residual,
            prev_post,
            fn,
            hc_scale,
            hc_base,
            norm_weight,
            float(magnitude),
            float(hc_eps),
            float(norm_eps),
            float(layernorm_eps),
        )
    if torch.compiler.is_compiling():
        raise RuntimeError("caller-owned iHC buffers are not supported inside Dynamo")
    return _ihc_post_pre_impl(
        x,
        residual,
        prev_post,
        fn,
        hc_scale,
        hc_base,
        magnitude=magnitude,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        norm_weight=norm_weight,
        layernorm_eps=layernorm_eps,
        out_streams=out_streams,
        out_post=out_post,
        out_y=out_y,
    )


def run_head(
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    *,
    hc_eps: float = IHC_DEFAULT_HC_EPS,
    norm_eps: float = IHC_DEFAULT_NORM_EPS,
    out: torch.Tensor | None = None,
    plan: Plan | None = None,
    binding: B12XIHCBinding | None = None,
) -> torch.Tensor:
    """Final stream merge before the output norm (``HYV4HyperHead``)."""
    if binding is not None:
        if out is not None:
            raise ValueError("iHC binding owns output buffers; do not also pass out")
        (_, _, y_view) = _run_bound(
            binding,
            operation="head",
            inputs=[streams, fn, hc_head_scale, hc_head_base],
            norm_weight=None,
            magnitude=IHC_DEFAULT_MAGNITUDE,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            layernorm_eps=IHC_DEFAULT_NORM_EPS,
        )
        return y_view
    if plan is not None:
        if torch.compiler.is_compiling():
            raise RuntimeError("planned iHC is not supported inside Dynamo")
        return _ihc_head_impl(
            streams,
            fn,
            hc_head_scale,
            hc_head_base,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            out=out,
            _state=_prepared_state(plan),
        )
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
