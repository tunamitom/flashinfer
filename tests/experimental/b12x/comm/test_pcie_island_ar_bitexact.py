"""Bit-exact contract: the composed TP8 island wire == the CPU emulation of record.

HARD equality, not tolerance. The emulation of record is
``island_composed_all_reduce`` below (kept verbatim: seed 1234, generator seed
``seed + rows + cols``, independent-Gaussian and common-mode classes). The
kernel's wire (``pcie_island_ar.composed_wire_reference``, the stage arithmetic
the CuTeDSL kernel in ``_island_ar_cute`` executes) must match it bit-for-bit
on the recorded prefill shapes (3072, 7168) and (3072, 3584).

On device, kernel output must match the same emulation bit-for-bit (see
test_pcie_island_ar_gpu.py, which requires 8 GPUs).
"""

from __future__ import annotations

import pytest
import torch

from b12x.comm.pcie.pcie_island_ar import composed_wire_reference

WORLD = 8
ISLAND = 4


def island_composed_all_reduce(xs):
    """TP8 2x4-island composed reduction — the emulation of record, kept verbatim.

    Phase 1 -- intra-island FP32 reduce-scatter, one BF16 island partial.
    Phase 2 -- cross-uplink combine of the island quarter-sums (BF16 wire),
               FP32 2-way accumulate over islands, one BF16 store.
    Phase 3 -- island all-gather: exact BF16 copies, elementwise a no-op.
    """
    parts = []
    for i in range(WORLD // ISLAND):
        acc = torch.zeros_like(xs[0], dtype=torch.float32)
        for lane in range(ISLAND):  # kernel accumulates p = 0..3 in lane order
            acc += xs[i * ISLAND + lane].float()
        parts.append(acc.to(torch.bfloat16))  # pack_f32x2_to_bf16x2
    total = torch.zeros_like(xs[0], dtype=torch.float32)
    for i in range(WORLD // ISLAND):
        total += parts[i].float()
    out = total.to(torch.bfloat16)
    return out


def generate(rows, cols, cls, seed=1234):
    """Input generation of the emulation of record, kept verbatim."""
    gen = torch.Generator().manual_seed(seed + rows + cols)
    if cls == "gaussian":
        xs = [torch.randn(rows, cols, generator=gen) for _ in range(WORLD)]
    else:
        common = torch.randn(rows, cols, generator=gen)
        xs = [common + 0.05 * torch.randn(rows, cols, generator=gen) for _ in range(WORLD)]
    return [x.to(torch.bfloat16) for x in xs]


@pytest.mark.parametrize("rows,cols", [(3072, 7168), (3072, 3584)])
@pytest.mark.parametrize("cls", ["gaussian", "commonmode"])
def test_composed_wire_matches_the_recorded_emulation_bit_for_bit(rows, cols, cls):
    xs = generate(rows, cols, cls)
    expected = island_composed_all_reduce(xs)
    actual = composed_wire_reference(xs)
    assert actual.dtype == torch.bfloat16
    # HARD: bit-for-bit equality on the recorded shapes, never a tolerance.
    assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16)), (
        "composed island wire diverged from the recorded emulation"
    )


def test_wire_reference_rejects_non_eight_inputs():
    with pytest.raises(ValueError, match="expects 8 inputs"):
        composed_wire_reference([torch.zeros(2, 2, dtype=torch.bfloat16)] * 4)
    with pytest.raises(ValueError, match="BF16"):
        composed_wire_reference([torch.zeros(2, 2)] * 8)
