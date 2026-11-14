"""Shared tolerance gates for the iHC GPU tests.

The parity tests and the mutant-kill tests compare through these same
functions; a mutant is "killed" exactly when one of these gates rejects it.

Tolerances (measured against the pure-torch reference on a switch-B card at
hidden 6144: post max-abs ~5e-7, BF16 rel-norm ~3e-5):

- FP32 post gates: ``atol = rtol = 1e-5`` (only FP32 accumulation order and the
  sigmoid/exp instruction selection separate implementations).
- BF16 tensors (``y``, sublayer update out, head out): the contract rounds the
  FP32 accumulation to BF16 exactly once, so implementations agree to BF16
  rounding-boundary flips. ``rel-norm <= 5e-3`` plus
  ``assert_close(atol=2e-2, rtol=2e-2)`` (the repo's mHC BF16 convention).
"""

from __future__ import annotations

import torch

# FP32 gate outputs (post tensors).
POST_ATOL = 1e-5
POST_RTOL = 1e-5

# BF16 outputs (y / sublayer update / head merge).
BF16_ATOL = 2e-2
BF16_RTOL = 2e-2
BF16_REL_NORM = 5e-3

# A mutant must clear this multiple of the tolerance to count as separated
# from the reference (so kills are not tolerance knife-edges).
MUTANT_MARGIN = 20.0


def rel_norm(got: torch.Tensor, ref: torch.Tensor) -> float:
    denom = ref.float().norm().clamp_min(1e-6)
    return float((got.float() - ref.float()).norm() / denom)


def assert_close_bf16(got: torch.Tensor, ref: torch.Tensor) -> None:
    """The BF16 output gate used by the parity tests and the mutant tests."""
    assert got.dtype == torch.bfloat16, f"expected bf16, got {got.dtype}"
    err = rel_norm(got, ref)
    assert err < BF16_REL_NORM, f"output rel-norm diff {err:.4g} exceeds {BF16_REL_NORM}"
    torch.testing.assert_close(got, ref, atol=BF16_ATOL, rtol=BF16_RTOL)


def assert_close_post(got: torch.Tensor, ref: torch.Tensor) -> None:
    """The FP32 post-gate gate."""
    assert got.dtype == torch.float32, f"expected fp32, got {got.dtype}"
    torch.testing.assert_close(got, ref, atol=POST_ATOL, rtol=POST_RTOL)


def assert_close_pair(
    got: tuple[torch.Tensor, torch.Tensor],
    ref: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Gate for the ``(post, y)`` pair returned by ``run_pre``."""
    assert_close_post(got[0], ref[0])
    assert_close_bf16(got[1], ref[1])
