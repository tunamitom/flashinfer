"""Mutant implementations for the iHC mutant-kill tests.

Each mutant is what a wrong operator would return at the API boundary — the
outputs are computed from the same inputs as the real ``run_pre`` / ``run_post``
/ ``run_head`` but with the bug applied. Two families:

Gross semantic mutants (``SEMANTIC_MUTANTS``, killed with ``MUTANT_MARGIN``):

- ``swap``: pre/post gates swapped (the collapse is driven by the post gates and
  the carried ``post`` holds the pre gates),
- ``post1``: post magnitude 1.0 instead of the Hy4 ``hc_magnitude`` 2.0.

Rounding mutants (``ROUNDING_MUTANTS``, killed with ``ROUNDING_MUTANT_MARGIN``)
— these are the defects that a ~160x-loose BF16 tolerance let through, and
killing them is what makes the "exactly one BF16 rounding" claim a tested
property rather than a comment:

- ``double_round``: every product is rounded to BF16 before it is accumulated
  (the stream collapse terms, the post-update product, the head terms) — two
  roundings where the contract has exactly one,
- ``bf16_gates``: the sigmoid gates are quantized to BF16 and carried in BF16
  instead of FP32,
- ``gate_0p3pct``: every gate carries a 0.3 % relative error.

``install(monkeypatch, ihc, mutant)`` patches the public operator entry points
the way a buggy implementation would behave; ``IHC_TEST_MUTANT=<name>`` activates
it from ``conftest.py`` so the whole parity suite runs red against the mutant
(the red-run proof that the tests kill it).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

MUTANTS = ("swap", "post1", "double_round", "bf16_gates", "gate_0p3pct")
SEMANTIC_MUTANTS = ("swap", "post1")
ROUNDING_MUTANTS = ("double_round", "bf16_gates", "gate_0p3pct")
GATE_ERROR = 1.003  # gate_0p3pct: +0.3 % relative gate error


def _mixes(streams, fn, norm_eps):
    flat = streams.flatten(-2).float()
    return F.linear(flat, fn.float()) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + norm_eps
    )


def _gates(mixes, hc_scale, hc_base, *, magnitude, hc_eps):
    pre_logits, post_logits = mixes.split(4, dim=-1)
    pre = torch.sigmoid(pre_logits * hc_scale[0] + hc_base[:4]) + hc_eps
    post = magnitude * torch.sigmoid(post_logits * hc_scale[1] + hc_base[4:]) + hc_eps
    return pre, post


def _gate_mutant(mutant, pre, post):
    if mutant == "bf16_gates":
        return pre.to(torch.bfloat16).float(), post.to(torch.bfloat16).float()
    if mutant == "gate_0p3pct":
        return pre * GATE_ERROR, post * GATE_ERROR
    return pre, post


def _collapse(mutant, pre, streams):
    terms = pre.unsqueeze(-1) * streams.float()
    if mutant == "double_round":
        terms = terms.to(torch.bfloat16).float()
    return terms.sum(dim=-2).to(streams.dtype)


def mutated_pre(
    mutant: str,
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    magnitude: float = 2.0,
    hc_eps: float = 1.0e-6,
    norm_eps: float = 1.0e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``run_pre`` as a buggy implementation would compute it."""
    if mutant not in MUTANTS:
        raise ValueError(f"unknown mutant {mutant!r}; choose from {MUTANTS}")
    mixes = _mixes(streams, fn, norm_eps)
    if mutant == "post1":
        magnitude = 1.0  # the bug: post magnitude 1.0 instead of hc_magnitude
    pre, post = _gates(mixes, hc_scale, hc_base, magnitude=magnitude, hc_eps=hc_eps)
    if mutant == "swap":
        # pre/post swapped: the collapse uses the post gates and the carried
        # post gates are the pre gates.
        return pre, _collapse(mutant, post, streams)
    pre, post = _gate_mutant(mutant, pre, post)
    return post, _collapse(mutant, pre, streams)


def mutated_post(
    mutant: str,
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    """``run_post`` as a buggy implementation would compute it.

    ``gate_0p3pct`` is applied at gate construction only (``mutated_pre``), so
    the update does not re-scale gates that already carry the error.
    """
    if mutant not in MUTANTS:
        raise ValueError(f"unknown mutant {mutant!r}; choose from {MUTANTS}")
    if mutant == "bf16_gates":
        post = post.to(torch.bfloat16).float()
    product = post.float().unsqueeze(-1) * x.float().unsqueeze(-2)
    if mutant == "double_round":
        product = product.to(torch.bfloat16).float()
    return (product + residual.float()).to(x.dtype)


def mutated_head(
    mutant: str,
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    *,
    hc_eps: float = 1.0e-6,
    norm_eps: float = 1.0e-5,
) -> torch.Tensor:
    """``run_head`` as a buggy implementation would compute it."""
    if mutant not in MUTANTS:
        raise ValueError(f"unknown mutant {mutant!r}; choose from {MUTANTS}")
    mixes = _mixes(streams, fn, norm_eps)
    pre = torch.sigmoid(mixes * hc_head_scale[0] + hc_head_base) + hc_eps
    pre, _ = _gate_mutant(mutant, pre, pre)
    return _collapse(mutant, pre, streams)


def mutated_pipeline(
    mutant: str,
    *,
    streams,
    fn,
    hc_scale,
    hc_base,
    branch,
    head_fn,
    head_scale,
    head_base,
    magnitude: float = 2.0,
    hc_eps: float = 1.0e-6,
    norm_eps: float = 1.0e-5,
):
    """The full sublayer pipeline with the mutant numerics at every stage."""
    post, y = mutated_pre(
        mutant,
        streams,
        fn,
        hc_scale,
        hc_base,
        magnitude=magnitude,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
    )
    out = mutated_post(mutant, branch, streams, post)
    merged = mutated_head(
        mutant, out, head_fn, head_scale, head_base, hc_eps=hc_eps, norm_eps=norm_eps
    )
    return post, y, out, merged


def install(monkeypatch, ihc, mutant: str) -> None:
    """Patch the public operator entry points with the mutant behavior."""
    if mutant not in MUTANTS:
        raise ValueError(f"unknown mutant {mutant!r}; choose from {MUTANTS}")

    def _mutant_run_pre(streams, fn, hc_scale, hc_base, **kwargs):
        out_post = kwargs.pop("out_post", None)
        out_y = kwargs.pop("out_y", None)
        post, y = mutated_pre(mutant, streams, fn, hc_scale, hc_base, **kwargs)
        if out_post is not None:
            out_post.copy_(post)
            post = out_post
        if out_y is not None:
            out_y.copy_(y)
            y = out_y
        return post, y

    monkeypatch.setattr(ihc, "run_pre", _mutant_run_pre)
    if mutant in ROUNDING_MUTANTS:

        def _mutant_run_post(x, residual, post, *, out=None, **kwargs):
            got = mutated_post(mutant, x, residual, post)
            if out is not None:
                out.copy_(got)
                got = out
            return got

        def _mutant_run_head(streams, fn, scale, base, *, out=None, **kwargs):
            got = mutated_head(mutant, streams, fn, scale, base, **kwargs)
            if out is not None:
                out.copy_(got)
                got = out
            return got

        monkeypatch.setattr(ihc, "run_post", _mutant_run_post)
        monkeypatch.setattr(ihc, "run_head", _mutant_run_head)
