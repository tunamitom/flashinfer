"""Mutant implementations for the iHC mutant-kill tests (plan section 1.5).

Each mutant is what a wrong operator would return at the API boundary — the
outputs are computed from the same inputs as the real ``run_pre`` but with the
bug applied. The GPU tests must kill both:

- ``swap``: pre/post gates swapped (the collapse is driven by the post gates and
  the carried ``post`` holds the pre gates),
- ``post1``: post magnitude 1.0 instead of the Hy4 ``hc_magnitude`` 2.0.

``install(monkeypatch, ihc, mutant)`` patches the public operator entry points
the way a buggy implementation would behave; ``IHC_TEST_MUTANT=swap|post1``
activates it from ``tests/ihc/conftest.py`` so the whole parity suite runs red
against the mutant (the red-run proof that the tests kill it).
"""

from __future__ import annotations

import torch

MUTANTS = ("swap", "post1")


def _mixes(streams, fn, norm_eps):
    flat = streams.flatten(-2).float()
    return torch.nn.functional.linear(flat, fn.float()) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + norm_eps
    )


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
    mixes = _mixes(streams, fn, norm_eps)
    pre_logits, post_logits = mixes.split(4, dim=-1)
    pre = torch.sigmoid(pre_logits * hc_scale[0] + hc_base[:4]) + hc_eps
    post = magnitude * torch.sigmoid(post_logits * hc_scale[1] + hc_base[4:]) + hc_eps
    if mutant == "swap":
        # pre/post swapped: the collapse uses the post gates and the carried
        # post gates are the pre gates.
        y = (post.unsqueeze(-1) * streams.float()).sum(dim=-2).to(streams.dtype)
        return pre, y
    if mutant == "post1":
        # post magnitude 1.0 instead of 2.0.
        post = 1.0 * torch.sigmoid(post_logits * hc_scale[1] + hc_base[4:]) + hc_eps
        y = (pre.unsqueeze(-1) * streams.float()).sum(dim=-2).to(streams.dtype)
        return post, y
    raise ValueError(f"unknown mutant {mutant!r}")


def install(monkeypatch, ihc, mutant: str) -> None:
    """Patch ``ihc.run_pre`` (and only ``run_pre``) with the mutant behavior."""
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
