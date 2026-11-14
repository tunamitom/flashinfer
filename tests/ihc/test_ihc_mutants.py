"""Mutant-kill tests for the iHC operator (plan section 1.5).

Required kills: (a) pre/post swapped, (b) post magnitude 1.0. Each test builds
the full sublayer pipeline (``run_pre`` -> branch -> ``run_post`` -> head) once
with the real operator and once with the mutant, runs the same tolerance gate
the parity tests use, and asserts that the gate rejects the mutant with a wide
margin (>= ``MUTANT_MARGIN`` x the tolerance) while accepting the real operator.
"""

import pytest
import torch

from b12x.norm.mhc import _ihc_torch_ref as ref
from b12x.norm.mhc import ihc

import ihc_mutants
from ihc_tolerances import (
    BF16_REL_NORM,
    MUTANT_MARGIN,
    POST_ATOL,
    assert_close_bf16,
    assert_close_pair,
    rel_norm,
)

if not torch.cuda.is_available():
    pytest.skip("iHC mutant tests require CUDA", allow_module_level=True)

HC = ihc.IHC_MULT
MIXES = ihc.IHC_MIXES
TOKENS, HIDDEN = 33, 6144


def _pipeline(*, seed: int):
    """Real-operator pipeline golden: pre -> branch -> update -> head."""
    gen = torch.Generator().manual_seed(seed)
    streams = (
        torch.randn((TOKENS, HC, HIDDEN), generator=gen, dtype=torch.float32).cuda() / 3
    ).to(torch.bfloat16).contiguous()
    fn = (
        torch.randn((MIXES, HC * HIDDEN), generator=gen, dtype=torch.float32).cuda() / 64
    ).contiguous()
    hc_scale = (
        torch.randn((2,), generator=gen, dtype=torch.float32).cuda() / 3
    ).contiguous()
    hc_base = (
        torch.randn((MIXES,), generator=gen, dtype=torch.float32).cuda() / 5
    ).contiguous()
    branch = (
        torch.randn((TOKENS, HIDDEN), generator=gen, dtype=torch.float32).cuda() / 4
    ).to(torch.bfloat16).contiguous()
    head_fn = (
        torch.randn((HC, HC * HIDDEN), generator=gen, dtype=torch.float32).cuda() / 64
    ).contiguous()
    head_scale = (
        torch.randn((1,), generator=gen, dtype=torch.float32).cuda() / 3
    ).contiguous()
    head_base = (
        torch.randn((HC,), generator=gen, dtype=torch.float32).cuda() / 5
    ).contiguous()
    return streams, fn, hc_scale, hc_base, branch, head_fn, head_scale, head_base


def _mutant_pipeline(mutant: str, seed: int):
    (streams, fn, hc_scale, hc_base, branch, head_fn, head_scale, head_base) = _pipeline(
        seed=seed
    )
    post, y = ihc_mutants.mutated_pre(
        mutant,
        streams,
        fn,
        hc_scale,
        hc_base,
        magnitude=2.0,
        hc_eps=1.0e-6,
        norm_eps=1.0e-5,
    )
    out = ihc.run_post(branch, streams, post)  # real update, mutant gates
    merged = ihc.run_head(out, head_fn, head_scale, head_base)
    return post, y, out, merged


def _real_pipeline(seed: int):
    (streams, fn, hc_scale, hc_base, branch, head_fn, head_scale, head_base) = _pipeline(
        seed=seed
    )
    post, y = ihc.run_pre(streams, fn, hc_scale, hc_base)
    out = ihc.run_post(branch, streams, post)
    merged = ihc.run_head(out, head_fn, head_scale, head_base)
    # The golden for every stage is the pure-torch reference on the same
    # inputs, so the real pipeline is checked against the reference too (the
    # gate must accept it before it is allowed to reject a mutant).
    rpost, ry = ref.ihc_pre_reference(streams, fn, hc_scale, hc_base)
    rout = ref.ihc_post_reference(branch, streams, rpost)
    rmerged = ref.ihc_head_reference(rout, head_fn, head_scale, head_base)
    return (post, y, out, merged), (rpost, ry, rout, rmerged)


def _kill(mutant: str, seed: int, stage: int) -> None:
    """The gate must accept the real pipeline and reject the mutant pipeline."""
    (real, golden) = _real_pipeline(seed=seed)
    # Sanity: the real operator passes the very gate used for the kill.
    assert_close_pair((real[0], real[1]), (golden[0], golden[1]))
    assert_close_bf16(real[2], golden[2])
    assert_close_bf16(real[3], golden[3])

    mutant_out = _mutant_pipeline(mutant, seed=seed)
    got, want = mutant_out[stage], golden[stage]

    # The tolerance gate rejects the mutant output ...
    if got.dtype == torch.float32:
        with pytest.raises(AssertionError):
            torch.testing.assert_close(got, want, atol=POST_ATOL, rtol=POST_ATOL)
    else:
        with pytest.raises(AssertionError):
            assert_close_bf16(got, want)

    # ... and it is not a tolerance knife-edge: the separation is at least
    # MUTANT_MARGIN x the rel-norm gate threshold.
    margin = rel_norm(got, want) / BF16_REL_NORM
    assert margin >= MUTANT_MARGIN, (
        f"mutant {mutant!r} stage {stage} sits only {margin:.3g}x the rel-norm "
        f"gate from the reference; the kill would be a knife-edge"
    )


@pytest.mark.parametrize("stage", [0, 1, 2, 3], ids=["post", "y", "update", "head"])
def test_mutant_pre_post_swap_is_killed(stage: int) -> None:
    """(a) pre/post swapped: the collapse uses the post gates and the carried
    post gates are the pre gates."""
    _kill("swap", seed=101, stage=stage)


@pytest.mark.parametrize("stage", [0, 2, 3], ids=["post", "update", "head"])
def test_mutant_post_magnitude_one_is_killed(stage: int) -> None:
    """(b) post magnitude 1.0 instead of the Hy4 ``hc_magnitude`` 2.0."""
    _kill("post1", seed=202, stage=stage)


def test_post_magnitude_one_is_invisible_in_y_alone() -> None:
    """Why the parity gate must compare the post tensor and not only y:

    the post-magnitude-1.0 mutant changes only the post gates, so the collapsed
    ``y`` is bit-identical to the reference. Comparing y alone would let this
    mutant through — the stage coverage above is exactly what kills it."""
    (real, golden) = _real_pipeline(seed=202)
    mutant_out = _mutant_pipeline("post1", seed=202)
    assert torch.equal(mutant_out[1], golden[1])  # y identical: no kill here ...
    assert not torch.equal(mutant_out[0], golden[0])  # ... post differs: killed


def test_mutant_names_match_plan_section_15() -> None:
    assert set(ihc_mutants.MUTANTS) == {"swap", "post1"}
