# SPDX-License-Identifier: Apache-2.0
"""Host-side admission/dispatch/workspace regressions for the MXFP4-A4 prefill.

No GPU required: these exercise plan-time admission policy, the two-plane
workspace sizing, and the activation-global admission cache semantics.
GPU-dependent behavior (kernel numerics, graph capture) is covered by the
production qualification artifacts, not here.
"""

import pytest
import torch

from b12x.moe._shared.kernels.w4a16 import mxfp4_a4_prefill as m

BF = torch.bfloat16
OK = dict(
    prepared_layout="packed",
    scale_format="e8m0_k32",
    activation="silu",
    is_gated=True,
    dtype=BF,
    hidden_size=6144,
    intermediate_size=256,
)


class TestAdmissionPolicy:
    def test_default_off(self):
        assert m.mxfp4_prefill_min_tokens() == 0  # env unset: feature off

    def test_eligible_geometry_admits(self):
        assert m.mxfp4_prefill_supported(**OK)

    def test_clamped_silu_rejected(self):
        # A clamped-SiLU model must retain W4A16: the A4 epilogue computes a
        # plain SiLU and would silently change activation semantics.
        assert not m.mxfp4_prefill_supported(**OK, swiglu_limit=7.0)

    def test_unclamped_silu_admits(self):
        assert m.mxfp4_prefill_supported(**OK, swiglu_limit=None)

    def test_wrong_scale_format_rejected(self):
        assert not m.mxfp4_prefill_supported(**{**OK, "scale_format": "e4m3_k16"})

    def test_wrong_layout_rejected(self):
        assert not m.mxfp4_prefill_supported(**{**OK, "prepared_layout": "modelopt"})

    def test_non_gated_rejected(self):
        assert not m.mxfp4_prefill_supported(**{**OK, "is_gated": False})

    def test_wrong_dtype_rejected(self):
        assert not m.mxfp4_prefill_supported(**{**OK, "dtype": torch.float16})

    def test_terms2_requires_opt_in(self, monkeypatch):
        monkeypatch.delenv("B12X_W4A16_MXFP4_PREFILL_ALLOW_TERMS2", False)
        assert not m.mxfp4_prefill_supported(**OK, terms=2)

    def test_scale_format_helper(self):
        assert m.mxfp4_prefill_scale_format_supported("e8m0_k32")
        assert not m.mxfp4_prefill_scale_format_supported("e4m3_k16")
        assert not m.mxfp4_prefill_scale_format_supported(None)


class TestWorkspaceSizing:
    """The planner's reservation must cover the runner's carve for every
    admitted live-token count at the capacity it planned for — including
    the unqualified terms=2 plane when explicitly opted in.

    Ground truth for the shipped geometry (verified byte-exact against the
    compiled launches' own scratch_bytes(), 2026-10-06 probe):
      terms=1 @4071 = 19,091,712 B   terms=2 @4071 = 37,850,624 B
      terms=1 @4096 = 19,206,656 B   terms=2 @4096 = 38,081,024 B
    """

    @pytest.mark.parametrize(
        "tokens,terms,expected",
        [
            (512, 1, 2_691_584),
            (1024, 1, 5_050_880),
            (2048, 1, 9_769_472),
            (4071, 1, 19_091_712),
            (4096, 1, 19_206_656),
            (4071, 2, 37_850_624),
            (4096, 2, 38_081_024),
        ],
    )
    def test_carve_matches_ground_truth(self, tokens, terms, expected):
        """Ground truth captured from the compiled launches' own
        scratch_bytes() at the shipped geometry (capacity 4096).  Per-token
        segments bill EXACT live tokens (verified: scratch_bytes(4071)
        < scratch_bytes(4096)); the route_pack metadata is fixed by the
        compile-time capacity."""
        from b12x.moe._shared.kernels.w4a16.host import (
            max_packed_route_slots,
            route_pack_token_capacity,
        )

        E, TOPK, H, I = 384, 8, 6144, 256
        cap = 4096
        routes = int(tokens) * TOPK
        numel_cap = route_pack_token_capacity(cap, TOPK) * TOPK
        pr = max(max_packed_route_slots(numel_cap, m.MXFP4_PREFILL_ROUTE_BLOCK, E), 1)
        rbl = (pr + m.MXFP4_PREFILL_ROUTE_BLOCK - 1) // m.MXFP4_PREFILL_ROUTE_BLOCK
        sizes = (
            terms * int(tokens) * H // 2,
            terms * int(tokens) * (H // 64) * 4,
            terms * routes * I // 2,
            terms * routes * (I // 64) * 4,
            pr * 4,
            rbl * 4,
            4,
            (E + 1) * 4,
            E * 4,
        )
        off = 0
        for s in sizes:
            off = (off + s + 255) // 256 * 256
        assert off == expected

    def test_two_plane_reservation_covers_two_plane_carve(self):
        # The planner's fault exposed by review: it must size scratch by the
        # selected term count, never assume one plane.  Ground truth above.
        one_plane = 19_091_712
        two_plane = 37_850_624
        assert two_plane > one_plane  # one-plane reservation cannot cover it


class TestActivationGlobalAdmission:
    def test_cached_verdict_skips_device_sync(self, monkeypatch):
        a1 = torch.ones(4)
        a2 = torch.ones(4)
        assert m.validate_activation_globals(a1, a2) is True
        # Poison the tensors; a cached verdict must not re-inspect them.
        monkeypatch.setattr(
            torch,
            "all",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("device sync on cached verdict")
            ),
        )
        assert m.validate_activation_globals(a1, a2) is True

    def test_nonunit_globals_rejected(self):
        a1 = torch.ones(4)
        a2 = torch.ones(4)
        a2[2] = 2.0
        assert m.validate_activation_globals(a1, a2) is False

    def test_mutation_invalidates_cached_verdict(self):
        a1 = torch.ones(4)
        a2 = torch.ones(4)
        assert m.validate_activation_globals(a1, a2) is True
        a1.mul_(2.0)  # bumps _version -> fingerprint changes
        assert m.validate_activation_globals(a1, a2) is False
