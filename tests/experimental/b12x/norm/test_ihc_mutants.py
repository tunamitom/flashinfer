"""Mutant-kill tests for the iHC operator.

Required kills: (a) pre/post swapped, (b) post magnitude 1.0 — gross semantic
mutants separated with ``MUTANT_MARGIN`` — and the three rounding mutants
(double rounding, BF16 gates, 0.3 % gate error) that the ~160x-loose BF16
tolerance used to let through; killing them is what makes "exactly one BF16
rounding" a tested property. Each test builds the full sublayer pipeline
(``run_pre`` -> branch -> ``run_post`` -> head) once with the real operator and
once with the mutant, runs the same tolerance gate the parity tests use, and
asserts that the gate rejects the mutant with the required margin while
accepting the real operator.
"""

import pytest
import torch

from b12x.norm.mhc import _ihc_torch_ref as ref
from b12x.norm.mhc import ihc

from . import _ihc_mutants
from ._ihc_tolerances import (
    BF16_REL_NORM,
    MUTANT_MARGIN,
    POST_ATOL,
    ROUNDING_MUTANT_MARGIN,
    assert_close_bf16,
    assert_close_pair,
    assert_close_post,
    max_abs,
    rel_norm,
)

if not torch.cuda.is_available():
    pytest.skip("iHC mutant tests require CUDA", allow_module_level=True)

HC = ihc.IHC_MULT
MIXES = ihc.IHC_MIXES
TOKENS, HIDDEN = 33, 6144


def _pipeline(*, seed: int):
    """Real-operator pipeline golden inputs: pre -> branch -> update -> head."""
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


def _mutant_pipeline(mutant: str, seed: int):
    (streams, fn, hc_scale, hc_base, branch, head_fn, head_scale, head_base) = _pipeline(
        seed=seed
    )
    return _ihc_mutants.mutated_pipeline(
        mutant,
        streams=streams,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        branch=branch,
        head_fn=head_fn,
        head_scale=head_scale,
        head_base=head_base,
        magnitude=2.0,
        hc_eps=1.0e-6,
        norm_eps=1.0e-5,
    )


def _gate_rejects(got, want) -> float:
    """The parity gate must reject; return the separation from the reference."""
    if got.dtype == torch.float32:
        with pytest.raises(AssertionError):
            assert_close_post(got, want)
        return max_abs(got, want) / POST_ATOL
    with pytest.raises(AssertionError):
        assert_close_bf16(got, want)
    return rel_norm(got, want) / BF16_REL_NORM


def _kill(mutant: str, seed: int, stages, *, margin: float) -> None:
    """The gate must accept the real pipeline and reject the mutant pipeline."""
    (real, golden) = _real_pipeline(seed=seed)
    # Sanity: the real operator passes the very gate used for the kill.
    assert_close_pair((real[0], real[1]), (golden[0], golden[1]))
    assert_close_bf16(real[2], golden[2])
    assert_close_bf16(real[3], golden[3])

    mutant_out = _mutant_pipeline(mutant, seed=seed)
    for stage in stages:
        got, want = mutant_out[stage], golden[stage]
        separation = _gate_rejects(got, want)
        assert separation >= margin, (
            f"mutant {mutant!r} stage {stage} sits only {separation:.3g}x the "
            f"gate threshold from the reference; the kill would be a knife-edge"
        )


# (a) pre/post swapped: the collapse uses the post gates and the carried post
# gates are the pre gates.
@pytest.mark.parametrize("stage", [0, 1, 2, 3], ids=["post", "y", "update", "head"])
def test_mutant_pre_post_swap_is_killed(stage: int) -> None:
    _kill("swap", seed=101, stages=[stage], margin=MUTANT_MARGIN)


# (b) post magnitude 1.0 instead of the Hy4 ``hc_magnitude`` 2.0.
@pytest.mark.parametrize("stage", [0, 2, 3], ids=["post", "update", "head"])
def test_mutant_post_magnitude_one_is_killed(stage: int) -> None:
    _kill("post1", seed=202, stages=[stage], margin=MUTANT_MARGIN)


def test_post_magnitude_one_is_invisible_in_y_alone() -> None:
    """Why the parity gate must compare the post tensor and not only y:
    the post-magnitude-1.0 mutant changes only the post gates, so the collapsed
    ``y`` is bit-identical to the reference. Comparing y alone would let this
    mutant through — the stage coverage above is exactly what kills it."""
    (real, golden) = _real_pipeline(seed=202)
    mutant_out = _mutant_pipeline("post1", seed=202)
    assert torch.equal(mutant_out[1], golden[1])  # y identical: no kill here ...
    assert not torch.equal(mutant_out[0], golden[0])  # ... post differs: killed


# Rounding mutants: the "exactly one BF16 rounding" contract. These sit at
# 6x-20x the tightened 3e-4 rel-norm gate (measured); the old 5e-3 gate passed
# all three.
@pytest.mark.parametrize(
    "stage", [1, 2, 3], ids=["y", "update", "head"]
)
def test_mutant_double_rounding_is_killed(stage: int) -> None:
    """Products rounded to BF16 before accumulation: two roundings, not one."""
    _kill("double_round", seed=303, stages=[stage], margin=ROUNDING_MUTANT_MARGIN)


@pytest.mark.parametrize(
    "stage", [0, 1, 2, 3], ids=["post", "y", "update", "head"]
)
def test_mutant_bf16_gates_are_killed(stage: int) -> None:
    """Gates quantized to BF16 and carried in BF16 instead of FP32."""
    _kill("bf16_gates", seed=404, stages=[stage], margin=ROUNDING_MUTANT_MARGIN)


@pytest.mark.parametrize(
    "stage", [0, 1, 2, 3], ids=["post", "y", "update", "head"]
)
def test_mutant_gate_error_is_killed(stage: int) -> None:
    """A 0.3 % relative gate error."""
    _kill("gate_0p3pct", seed=505, stages=[stage], margin=ROUNDING_MUTANT_MARGIN)


def test_rounding_mutants_would_pass_the_old_loose_gate() -> None:
    """The defect-25 claim made concrete: at the collapsed-y stage all three
    rounding mutants pass the old ~160x-loose BF16 gate (rel-norm 5e-3) and
    are only caught by the tightened policy."""
    old_rel_norm = 5e-3
    for mutant, seed in (
        ("double_round", 303),
        ("bf16_gates", 404),
        ("gate_0p3pct", 505),
    ):
        mutant_out = _mutant_pipeline(mutant, seed=seed)
        (_, golden) = _real_pipeline(seed=seed)
        err = rel_norm(mutant_out[1], golden[1])
        assert err < old_rel_norm, f"{mutant} escaped the old gate at y"
        assert err >= BF16_REL_NORM, f"{mutant} not separated at y"


def test_mutant_names_match_plan_defects_25() -> None:
    assert set(_ihc_mutants.MUTANTS) == {
        "swap",
        "post1",
        "double_round",
        "bf16_gates",
        "gate_0p3pct",
    }
    assert set(_ihc_mutants.ROUNDING_MUTANTS) == {
        "double_round",
        "bf16_gates",
        "gate_0p3pct",
    }
