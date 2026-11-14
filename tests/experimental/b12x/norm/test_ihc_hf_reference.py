"""The iHC reference checked against the REAL transformers Hy4 HC modules.

The pure-torch reference (``_ihc_torch_ref``) is a transcription; this test runs
the actual ``transformers.models.hy_v4.modeling_hy_v4`` ``HYV4HyperConnection``
and ``HYV4HyperHead`` modules — pinned to /opt/models/quant-toolkit/.venv
transformers 5.23.0 (the provenance of /home/tom/hy4/reference/
modeling_hy_v4_patched.py, whose HC classes are byte-identical to stock; only
the RoPE helpers differ) — and requires the transcription and the operator to
agree with the real module, not just with each other.

The pinned interpreter runs the HF forward in a subprocess (the test venv has a
different transformers); tensors cross as float32 ``.npz`` (no torch
pickle across torch versions).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from b12x.norm.mhc import _ihc_torch_ref as ref
from b12x.norm.mhc import ihc

from ._ihc_tolerances import assert_close_bf16, assert_close_pair

HC = ihc.IHC_MULT
MIXES = ihc.IHC_MIXES
PINNED_PY = "/opt/models/quant-toolkit/.venv/bin/python"
PINNED_TRANSFORMERS = "5.23.0"
BATCH, TOKENS, HIDDEN = 2, 5, 6144

if not torch.cuda.is_available():
    pytest.skip("iHC HF-reference tests require CUDA", allow_module_level=True)

_HF_FORWARD = textwrap.dedent(
    """
    import numpy as np
    import torch
    import transformers

    assert transformers.__version__ == "5.23.0", transformers.__version__
    from transformers.models.hy_v4.configuration_hy_v4 import HYV4Config
    from transformers.models.hy_v4.modeling_hy_v4 import (
        HYV4HyperConnection,
        HYV4HyperHead,
    )

    data = np.load("inputs.npz")
    streams = torch.from_numpy(data["streams"]).to(torch.bfloat16)
    streams2 = torch.from_numpy(data["streams2"]).to(torch.bfloat16)

    try:
        config = HYV4Config(
            hidden_size=int(data["hidden"]),
            hc_mult=4,
            hc_eps=1.0e-6,
            hc_magnitude=2.0,
            rms_norm_eps=1.0e-5,
        )
    except Exception:
        from types import SimpleNamespace

        config = SimpleNamespace(
            hidden_size=int(data["hidden"]),
            hc_mult=4,
            hc_eps=1.0e-6,
            hc_magnitude=2.0,
            rms_norm_eps=1.0e-5,
        )

    hc = HYV4HyperConnection(config)
    head = HYV4HyperHead(config)
    with torch.no_grad():
        hc.fn.copy_(torch.from_numpy(data["fn"]))
        hc.base.copy_(torch.from_numpy(data["hc_base"]))
        hc.scale.copy_(torch.from_numpy(data["hc_scale"]))
        head.hc_fn.copy_(torch.from_numpy(data["head_fn"]))
        head.hc_base.copy_(torch.from_numpy(data["head_base"]))
        head.hc_scale.copy_(torch.from_numpy(data["head_scale"]))
        post, out = hc(streams)
        merged = head(streams2)
    np.savez(
        "outputs.npz",
        post=post.float().numpy(),
        out=out.float().numpy(),
        merged=merged.float().numpy(),
    )
    """
)


@pytest.fixture(scope="module")
def hf_outputs(tmp_path_factory):
    import shutil

    if shutil.which(PINNED_PY) is None:
        pytest.skip(f"pinned transformers venv missing at {PINNED_PY}")
    probe = subprocess.run(
        [PINNED_PY, "-c", "import transformers; print(transformers.__version__)"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if probe.returncode != 0 or probe.stdout.strip() != PINNED_TRANSFORMERS:
        pytest.skip(
            f"pinned transformers {PINNED_TRANSFORMERS} unavailable "
            f"(got {probe.stdout.strip() or probe.stderr.strip()!r})"
        )

    tmp = tmp_path_factory.mktemp("ihc_hf")
    gen = torch.Generator().manual_seed(705)
    streams = (
        torch.randn((BATCH, TOKENS, HC, HIDDEN), generator=gen, dtype=torch.float32) / 3
    ).to(torch.bfloat16).contiguous()
    streams2 = (
        torch.randn((BATCH, TOKENS, HC, HIDDEN), generator=gen, dtype=torch.float32) / 3
    ).to(torch.bfloat16).contiguous()
    fn = (
        torch.randn((MIXES, HC * HIDDEN), generator=gen, dtype=torch.float32) / 64
    ).contiguous()
    hc_scale = (torch.randn((2,), generator=gen, dtype=torch.float32) / 3).contiguous()
    hc_base = (torch.randn((MIXES,), generator=gen, dtype=torch.float32) / 5).contiguous()
    head_fn = (
        torch.randn((HC, HC * HIDDEN), generator=gen, dtype=torch.float32) / 64
    ).contiguous()
    head_scale = (torch.randn((1,), generator=gen, dtype=torch.float32) / 3).contiguous()
    head_base = (torch.randn((HC,), generator=gen, dtype=torch.float32) / 5).contiguous()
    np.savez(
        tmp / "inputs.npz",
        streams=streams.float().numpy(),
        streams2=streams2.float().numpy(),
        fn=fn.numpy(),
        hc_scale=hc_scale.numpy(),
        hc_base=hc_base.numpy(),
        head_fn=head_fn.numpy(),
        head_scale=head_scale.numpy(),
        head_base=head_base.numpy(),
        hidden=np.int64(HIDDEN),
    )
    (tmp / "hf_forward.py").write_text(_HF_FORWARD)
    result = subprocess.run(
        [PINNED_PY, "hf_forward.py"],
        cwd=tmp,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"pinned HF forward failed:\n{result.stdout}\n{result.stderr}"
    )
    outputs = dict(np.load(tmp / "outputs.npz"))
    inputs = dict(np.load(tmp / "inputs.npz"))

    def back(name):
        return torch.from_numpy(inputs[name])

    return {
        "streams": back("streams").to(torch.bfloat16).contiguous(),
        "streams2": back("streams2").to(torch.bfloat16).contiguous(),
        "fn": back("fn").contiguous(),
        "hc_scale": back("hc_scale").contiguous(),
        "hc_base": back("hc_base").contiguous(),
        "head_fn": back("head_fn").contiguous(),
        "head_scale": back("head_scale").contiguous(),
        "head_base": back("head_base").contiguous(),
        "post": torch.from_numpy(outputs["post"]),
        "out": torch.from_numpy(outputs["out"]).to(torch.bfloat16),
        "merged": torch.from_numpy(outputs["merged"]).to(torch.bfloat16),
    }


def test_transcription_matches_real_hf_module(hf_outputs) -> None:
    """The pure-torch reference is the real module's math, exactly."""
    post, out = ref.ihc_pre_reference(
        hf_outputs["streams"],
        hf_outputs["fn"],
        hf_outputs["hc_scale"],
        hf_outputs["hc_base"],
        magnitude=2.0,
        hc_eps=1.0e-6,
        norm_eps=1.0e-5,
    )
    torch.testing.assert_close(post, hf_outputs["post"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(out, hf_outputs["out"], rtol=0.0, atol=0.0)
    merged = ref.ihc_head_reference(
        hf_outputs["streams2"],
        hf_outputs["head_fn"],
        hf_outputs["head_scale"],
        hf_outputs["head_base"],
        hc_eps=1.0e-6,
        norm_eps=1.0e-5,
    )
    torch.testing.assert_close(merged, hf_outputs["merged"], rtol=0.0, atol=0.0)


def test_operator_matches_real_hf_module(hf_outputs) -> None:
    """The GPU operator serves the real HF module's contract within policy."""
    streams = hf_outputs["streams"].cuda()
    fn = hf_outputs["fn"].cuda()
    hc_scale = hf_outputs["hc_scale"].cuda()
    hc_base = hf_outputs["hc_base"].cuda()
    post, out = ihc.run_pre(streams, fn, hc_scale, hc_base)
    want_post = hf_outputs["post"].cuda()
    want_out = hf_outputs["out"].cuda()
    assert tuple(post.shape) == (BATCH, TOKENS, HC)
    assert_close_pair((post, out), (want_post, want_out))

    streams2 = hf_outputs["streams2"].cuda()
    merged = ihc.run_head(
        streams2,
        hf_outputs["head_fn"].cuda(),
        hf_outputs["head_scale"].cuda(),
        hf_outputs["head_base"].cuda(),
    )
    assert_close_bf16(merged, hf_outputs["merged"].cuda())
