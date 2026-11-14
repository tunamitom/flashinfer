"""Prepared plan / bind / session contracts for the b12x iHC operator.

Mirrors the mHC prepared-plan tests: ``plan_ihc`` declares one operation with
its numerical recipe, ``PreparationSession`` selects and primes the plan
(Triton-warming exactly the chosen branch), ``bind`` builds caller-owned output
views, and the bound forms execute through the mutating ``b12x::ihc_bound``
custom op — including under CUDA graph capture, with changed inputs replayed
through the same captured buffers.
"""

import pytest
import torch

from b12x.norm.mhc import _ihc_impl, _ihc_tuning
from b12x.norm.mhc import ihc
from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall

from ._ihc_tolerances import assert_close_bf16, assert_close_pair, assert_close_post

HC = ihc.IHC_MULT
MIXES = ihc.IHC_MIXES

if not torch.cuda.is_available():
    pytest.skip("iHC prepared-plan tests require CUDA", allow_module_level=True)


def _make_inputs(*, tokens: int, hidden: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    streams = (
        torch.randn((tokens, HC, hidden), generator=gen, dtype=torch.float32).cuda() / 3
    ).to(torch.bfloat16).contiguous()
    fn = (
        torch.randn((MIXES, HC * hidden), generator=gen, dtype=torch.float32).cuda() / 64
    ).contiguous()
    hc_scale = (torch.randn((2,), generator=gen, dtype=torch.float32).cuda() / 3).contiguous()
    hc_base = (torch.randn((MIXES,), generator=gen, dtype=torch.float32).cuda() / 5).contiguous()
    return streams, fn, hc_scale, hc_base


def _make_branch(*, tokens: int, hidden: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    return (
        torch.randn((tokens, hidden), generator=gen, dtype=torch.float32).cuda() / 4
    ).to(torch.bfloat16).contiguous()


_RECIPE = {"magnitude": 2.0, "hc_eps": 1.0e-6, "norm_eps": 1.0e-5}


def _prepare(session, plan, name="ihc"):
    """Materialize one plan through the session (no benchmark call)."""
    session.prepare(
        (
            plan.request(
                name=name, prepare_call=lambda state: PreparedCall(run=lambda: None)
            ),
        )
    )


def _plan(device, *, tokens: int, hidden: int, operation: str, **extra):
    invocation = {"operation": operation}
    if operation in ("pre", "post_pre"):
        invocation.update(_RECIPE)
    elif operation == "head":
        invocation.update({"hc_eps": 1.0e-6, "norm_eps": 1.0e-5})
    invocation.update(extra)
    return ihc.plan(
        ihc.Caps(device=device, max_tokens=tokens, hidden_size=hidden),
        invocation=FrozenMapping(invocation),
    )


def test_tuning_contract_is_registered() -> None:
    from b12x.preparation.catalog import get_tuning_contract

    contract = get_tuning_contract("norm.mhc", variant="ihc")
    assert contract is _ihc_tuning.TUNING
    assert contract.component_id == "norm.mhc.ihc"
    assert contract.query_schema_version == 1


def test_plan_requires_explicit_operation_and_recipe() -> None:
    device = torch.device("cuda")
    caps = ihc.Caps(device=device, max_tokens=4, hidden_size=6144)
    with pytest.raises(ValueError, match="explicit operation"):
        ihc.plan(caps, invocation=FrozenMapping({"hc_eps": 1e-6, "norm_eps": 1e-5}))
    with pytest.raises(ValueError, match="numerical recipe"):
        ihc.plan(caps, invocation=FrozenMapping({"operation": "pre"}))
    with pytest.raises(ValueError, match="unknown iHC invocation"):
        ihc.plan(
            caps,
            invocation=FrozenMapping(
                {"operation": "post", "sinkhorn_iters": 20}
            ),
        )


def test_prepared_state_validates_recipe_and_capacity() -> None:
    device, tokens, hidden = torch.device("cuda"), 8, 6144
    streams, fn, hc_scale, hc_base = _make_inputs(
        tokens=tokens, hidden=hidden, seed=81_001
    )
    plan = _plan(device, tokens=tokens, hidden=hidden, operation="pre")
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        _prepare(session, plan, "pre")
        got_post, got_y = ihc.run_pre(streams, fn, hc_scale, hc_base, plan=plan)
        want_post, want_y = ihc.run_pre(streams, fn, hc_scale, hc_base)
        assert_close_pair((got_post, got_y), (want_post, want_y))
        with pytest.raises(ValueError, match="recipe differs"):
            ihc.run_pre(
                streams, fn, hc_scale, hc_base, plan=plan, hc_eps=2.0e-6
            )
        with pytest.raises(ValueError, match="capacity"):
            big, bfn, bsc, bbase = _make_inputs(tokens=tokens + 1, hidden=hidden, seed=9)
            ihc.run_pre(big, bfn, bsc, bbase, plan=plan)


@pytest.mark.parametrize("tokens", [1, 8])
def test_prepared_post_pre_binding_matches_functional(tokens: int) -> None:
    device, hidden = torch.device("cuda"), 6144
    streams, fn, hc_scale, hc_base = _make_inputs(
        tokens=tokens, hidden=hidden, seed=91_450 + tokens
    )
    branch = _make_branch(tokens=tokens, hidden=hidden, seed=91_460 + tokens)
    plan = _plan(device, tokens=tokens, hidden=hidden, operation="post_pre")
    prev_post, _ = ihc.run_pre(streams, fn, hc_scale, hc_base)

    def prepare(state):
        out = torch.empty((tokens, HC, hidden), dtype=torch.bfloat16, device=device)
        post = torch.empty((tokens, HC), dtype=torch.float32, device=device)
        y = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
        binding = state.bind(out_streams=out, post=post, y=y, tokens=tokens)
        return PreparedCall(
            run=lambda: _ihc_impl._ihc_post_pre_impl(
                branch,
                streams,
                prev_post,
                fn,
                hc_scale,
                hc_base,
                magnitude=2.0,
                hc_eps=1.0e-6,
                norm_eps=1.0e-5,
                binding=binding,
                _state=state,
            ),
            owners=(binding,),
        )

    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare((plan.request(name="post-pre", prepare_call=prepare),))
        out = torch.empty((tokens, HC, hidden), dtype=torch.bfloat16, device=device)
        post = torch.empty((tokens, HC), dtype=torch.float32, device=device)
        y = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
        binding = ihc.bind(plan, out_streams=out, post=post, y=y, tokens=tokens)
        got = ihc.run_post_pre(
            branch,
            streams,
            prev_post,
            fn,
            hc_scale,
            hc_base,
            magnitude=2.0,
            hc_eps=1.0e-6,
            norm_eps=1.0e-5,
            binding=binding,
        )
        assert tuple(t.data_ptr() for t in got) == (
            out.data_ptr(),
            post.data_ptr(),
            y.data_ptr(),
        )
        want = ihc.run_post_pre(
            branch, streams, prev_post, fn, hc_scale, hc_base
        )
        assert torch.equal(got[0], want[0])
        assert torch.equal(got[1], want[1])
        assert torch.equal(got[2], want[2])


def test_prepared_pre_and_head_bindings_match_functional() -> None:
    device, tokens, hidden = torch.device("cuda"), 4, 6144
    streams, fn, hc_scale, hc_base = _make_inputs(
        tokens=tokens, hidden=hidden, seed=92_100
    )
    gen = torch.Generator().manual_seed(92_101)
    head_fn = (
        torch.randn((HC, HC * hidden), generator=gen, dtype=torch.float32).cuda() / 64
    ).contiguous()
    head_scale = (torch.randn((1,), generator=gen, dtype=torch.float32).cuda() / 3).contiguous()
    head_base = (torch.randn((HC,), generator=gen, dtype=torch.float32).cuda() / 5).contiguous()

    pre_plan = _plan(device, tokens=tokens, hidden=hidden, operation="pre")
    head_plan = _plan(device, tokens=tokens, hidden=hidden, operation="head")
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        _prepare(session, pre_plan, "pre")
        _prepare(session, head_plan, "head")
        post_buf = torch.empty((tokens, HC), dtype=torch.float32, device=device)
        y_buf = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
        pre_binding = ihc.bind(pre_plan, post=post_buf, y=y_buf, tokens=tokens)
        got_post, got_y = ihc.run_pre(
            streams,
            fn,
            hc_scale,
            hc_base,
            magnitude=2.0,
            hc_eps=1.0e-6,
            norm_eps=1.0e-5,
            binding=pre_binding,
        )
        want_post, want_y = ihc.run_pre(streams, fn, hc_scale, hc_base)
        assert_close_post(got_post, want_post)
        assert_close_bf16(got_y, want_y)

        head_buf = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
        head_binding = ihc.bind(head_plan, y=head_buf, tokens=tokens)
        got_h = ihc.run_head(
            streams,
            head_fn,
            head_scale,
            head_base,
            hc_eps=1.0e-6,
            norm_eps=1.0e-5,
            binding=head_binding,
        )
        want_h = ihc.run_head(streams, head_fn, head_scale, head_base)
        assert_close_bf16(got_h, want_h)


def test_prepared_post_pre_capture_replays_changed_inputs() -> None:
    """The bound op is capture-safe: one captured boundary replays on new
    inputs through the same caller-owned buffers without reallocating."""
    device, tokens, hidden = torch.device("cuda"), 2, 6144
    streams, fn, hc_scale, hc_base = _make_inputs(
        tokens=tokens, hidden=hidden, seed=93_200
    )
    branch = _make_branch(tokens=tokens, hidden=hidden, seed=93_201)
    plan = _plan(device, tokens=tokens, hidden=hidden, operation="post_pre")
    prev_post, _ = ihc.run_pre(streams, fn, hc_scale, hc_base)
    out = torch.empty((tokens, HC, hidden), dtype=torch.bfloat16, device=device)
    post = torch.empty((tokens, HC), dtype=torch.float32, device=device)
    y = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)

    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        _prepare(session, plan, "post-pre")
        binding = ihc.bind(plan, out_streams=out, post=post, y=y, tokens=tokens)
        call = lambda: ihc.run_post_pre(  # noqa: E731
            branch,
            streams,
            prev_post,
            fn,
            hc_scale,
            hc_base,
            magnitude=2.0,
            hc_eps=1.0e-6,
            norm_eps=1.0e-5,
            binding=binding,
        )
        call()  # warm the exact branch outside capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        for magnitude in (1.0, 7.0):
            branch.normal_(0, magnitude)
            streams.normal_(0, magnitude)
            prev_post.copy_(
                torch.sigmoid(prev_post) * 2.0
            )  # changed gates, same buffers
            out.fill_(float("nan"))
            post.fill_(float("nan"))
            y.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            want_out, want_post, want_y = ihc.run_post_pre(
                branch, streams, prev_post, fn, hc_scale, hc_base
            )
            assert torch.equal(out, want_out)
            assert torch.equal(post, want_post)
            assert torch.equal(y, want_y)
        graph.reset()


def test_binding_rejects_bad_buffers_and_plan_mismatch() -> None:
    device, tokens, hidden = torch.device("cuda"), 4, 6144
    plan = _plan(device, tokens=tokens, hidden=hidden, operation="post_pre")
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        _prepare(session, plan, "post-pre")
        with pytest.raises(ValueError, match="tail shape"):
            ihc.bind(
                plan,
                out_streams=torch.empty(
                    (tokens, HC, hidden + 8), dtype=torch.bfloat16, device=device
                ),
                tokens=tokens,
            )
        with pytest.raises(ValueError, match="smaller than"):
            ihc.bind(
                plan,
                post=torch.empty((tokens - 1, HC), dtype=torch.float32, device=device),
                tokens=tokens,
            )
        with pytest.raises(ValueError, match="dtype"):
            ihc.bind(
                plan,
                y=torch.empty((tokens, hidden), dtype=torch.float32, device=device),
                tokens=tokens,
            )
        binding = ihc.bind(
            plan,
            out_streams=torch.empty(
                (tokens, HC, hidden), dtype=torch.bfloat16, device=device
            ),
            post=torch.empty((tokens, HC), dtype=torch.float32, device=device),
            y=torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device),
            tokens=tokens,
        )
        # A binding that owns outputs must not also receive caller outputs.
        streams, fn, hc_scale, hc_base = _make_inputs(
            tokens=tokens, hidden=hidden, seed=31
        )
        branch = _make_branch(tokens=tokens, hidden=hidden, seed=32)
        prev_post, _ = ihc.run_pre(streams, fn, hc_scale, hc_base)
        with pytest.raises(ValueError, match="owns output buffers"):
            ihc.run_post_pre(
                branch,
                streams,
                prev_post,
                fn,
                hc_scale,
                hc_base,
                binding=binding,
                out_y=torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device),
            )
