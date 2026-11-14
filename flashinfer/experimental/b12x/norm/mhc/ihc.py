"""Public surface for the Hy4 iHC operator (norm.mhc variant without comb).

iHC = DeepSeek-V4 mHC minus the comb/Sinkhorn step, for tencent/Hy4-preview:
FP32 ``fn`` with the 8-row iHC layout (4 pre + 4 post rows, ``[8, 4H]``),
``pre = sigmoid(..) + hc_eps``, ``post = magnitude * sigmoid(..) + hc_eps``
(``magnitude = 2.0``), sublayer update ``post_i * branch + stream_i``, four
streams collapsed before / redistributed after every sublayer, hidden 6144.

Typical sublayer boundary (streams ``[..., 4, H]`` BF16, ``fn`` FP32 ``[8, 4H]``)::

    post, y = ihc.run_pre(streams, fn, hc_scale, hc_base)   # y: collapsed input
    branch = sublayer(norm(y))                              # attention or MLP
    streams = ihc.run_post(branch, streams, post)           # redistribute

and the same boundary in ONE launch (post update + next pre + RMSNorm fold)::

    streams, post, y = ihc.run_post_pre(branch, streams, post, fn, hc_scale,
                                       hc_base, norm_weight=norm_weight)

Before the output norm: ``y = ihc.run_head(streams, head_fn, head_scale,
head_base)``. The exact numerics live in :mod:`._ihc_torch_ref` (pure torch,
device-agnostic — usable directly in a FLARE recipe before the kernel lands);
the CUDA operator dispatches to the fused Triton kernels of :mod:`._ihc_kernels`
with the torch fallback of :mod:`._ihc_impl` (``B12X_IHC_BACKEND`` selects).

``plan(Caps(...), invocation=...)`` is allocation-free and registers the
``norm.mhc.ihc`` tuning contract (:mod:`._ihc_tuning`); ``PreparationSession``
selects and primes the plan before ``bind`` builds caller-owned output views.
Bound forms mutate their buffers through the ``b12x::ihc_bound`` custom op and
trace under ``torch.compile`` and CUDA graph capture; the functional forms run
through the ``b12x::ihc_*`` ops.

The streams are replicated on every rank (no collectives here) and the operator
does no scheduler work: ``run_pre``/``run_post``/``run_post_pre`` are stateless
per call.
"""

from __future__ import annotations

from b12x.preparation import Plan
from b12x.preparation.types import require_prepared

from ._ihc_impl import (
    B12XIHCBinding as Binding,
    B12XIHCCaps as Caps,
)
from ._ihc_impl import (
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
    is_supported,
    run_head,
    run_post,
    run_post_pre,
    run_pre,
    tuning_for,
)
from ._ihc_preparation import plan_ihc as plan
from ._ihc_torch_ref import (
    ihc_head_reference,
    ihc_norm_reference,
    ihc_post_pre_reference,
    ihc_post_reference,
    ihc_pre_reference,
)


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind caller-owned output buffers to a prepared iHC plan."""
    state = require_prepared(plan, "norm.mhc.ihc")
    return state.bind(plan=plan, **kwargs)


__all__ = [
    "Binding",
    "Caps",
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
    "Plan",
    "bind",
    "ihc_head_reference",
    "ihc_norm_reference",
    "ihc_post_pre_reference",
    "ihc_post_reference",
    "ihc_pre_reference",
    "is_supported",
    "plan",
    "run_head",
    "run_post",
    "run_post_pre",
    "run_pre",
    "tuning_for",
]
