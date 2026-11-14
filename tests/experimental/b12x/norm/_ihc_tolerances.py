"""Shared tolerance gates for the iHC GPU tests.

The parity tests and the mutant-kill tests compare through these same
functions; a mutant is "killed" exactly when one of these gates rejects it.

Tolerance policy (measured against the pure-torch reference on a switch-B card,
hidden 6144 across the parity grid, Triton and torch backends):

- FP32 post gates: measured operator error <= 1.4e-6 max-abs. Gate:
  ``atol = rtol = 1e-5`` (only FP32 accumulation order and the sigmoid/exp
  instruction selection separate implementations).
- BF16 tensors (``y``, sublayer update out, head out): the contract rounds the
  FP32 accumulation to BF16 exactly once ("exactly one BF16 rounding").
  Measured operator-vs-reference rel-norm is <= 8.4e-5 (worst: the fused
  boundary at 384 rows) and typically ~3e-5. The rel-norm gate is **3e-4**:
  ~4x the worst measured error and ~10x the typical one — tight enough that a
  second rounding anywhere in the chain cannot hide. The elementwise envelope
  is one BF16 ulp (2^-7 = 7.8e-3 relative) of the reference element plus one
  ulp of the largest output element as the cancellation floor.
- A gross semantic mutant (pre/post swap, post magnitude 1.0) must clear
  ``MUTANT_MARGIN`` (20x) the gate it fails; a rounding mutant (double
  rounding, BF16 gates, 0.3 % gate error) must clear
  ``ROUNDING_MUTANT_MARGIN`` (3x) — measured separations are 6x-20x.
"""

from __future__ import annotations

import torch

# FP32 gate outputs (post tensors).
POST_ATOL = 1e-5
POST_RTOL = 1e-5

# BF16 outputs (y / sublayer update / head merge): one-rounding policy.
BF16_REL_NORM = 3e-4
BF16_RTOL = 8e-3  # one BF16 ulp (2^-7 = 7.8e-3) at the binade top + margin
BF16_ULP_FLOOR = 8e-3  # atol = BF16_ULP_FLOOR * ref.abs().max()

# A mutant must clear this multiple of the tolerance to count as separated
# from the reference (so kills are not tolerance knife-edges).
MUTANT_MARGIN = 20.0
ROUNDING_MUTANT_MARGIN = 3.0


def rel_norm(got: torch.Tensor, ref: torch.Tensor) -> float:
    denom = ref.float().norm().clamp_min(1e-6)
    return float((got.float() - ref.float()).norm() / denom)


def max_abs(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got.float() - ref.float()).abs().max())


def assert_close_bf16(got: torch.Tensor, ref: torch.Tensor) -> None:
    """The BF16 output gate used by the parity tests and the mutant tests."""
    assert got.dtype == torch.bfloat16, f"expected bf16, got {got.dtype}"
    err = rel_norm(got, ref)
    assert err < BF16_REL_NORM, f"output rel-norm diff {err:.4g} exceeds {BF16_REL_NORM}"
    torch.testing.assert_close(
        got,
        ref,
        atol=BF16_ULP_FLOOR * float(ref.float().abs().max()),
        rtol=BF16_RTOL,
    )


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
