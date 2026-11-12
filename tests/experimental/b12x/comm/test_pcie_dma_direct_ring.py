"""CPU random-interleaving emulation of the zero-copy BF16 direct ring and reduce-scatter.

Ported from g0san's flashinfer chain (91859f79f `test_pcie_dma_direct_fp24.py`,
with the emulator from 996df5678's `test_pcie_dma_tree.py`) for the L9 dual-wave
transplant. The source ran nine ranks only; this version is parameterized at
world 8 and 9 to pin that the transport is world-generic (the "8-hop" in the
source commit title is world 9's hop count, not a constant). The fp24 case is
kept minus the ring all-gather section (that gather rides `run_gather`, which
this port does not take — it belongs to the ring column gathers we do not use).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from . import _ring_emulator as emulator


class _Tensor:
    def __init__(self, space, elems, dtype):
        self.dtype = dtype
        self.elems = elems
        self.size = torch.empty((), dtype=dtype).element_size()
        self.address = space.allocate(elems * self.size)

    @property
    def nbytes(self):
        return self.elems * self.size

    def data_ptr(self):
        return self.address

    def numel(self):
        return self.elems

    def element_size(self):
        return self.size

    def is_contiguous(self):
        return True


class _DirectMachine(emulator._Machine):
    def dma_direct_step(
        self,
        in_mode,
        out_mode,
        local_bf16_ptr,
        wire_in_ptr,
        scales_in_ptr,
        out_ptr,
        scales_out_ptr,
        elems,
        *,
        wait,
    ):
        self.emit("wait_flag", wait[0], wait[1])
        space = self.space

        def step():
            a = space.bf16(local_bf16_ptr, elems).float()
            if in_mode == "bf16":
                b = space.bf16(wire_in_ptr, elems).float()
            elif in_mode == "fp32":
                b = space.f32(wire_in_ptr, elems).clone()
            elif in_mode == "fp24":
                hi16 = torch.frombuffer(space.view(wire_in_ptr, elems * 2), dtype=torch.uint16).clone()
                lo8 = torch.frombuffer(space.view(scales_in_ptr, elems), dtype=torch.uint8).clone()
                b = _fp24_decode_py(hi16, lo8)
            else:
                raise AssertionError(in_mode)
            v = b + a
            if out_mode == "fp32":
                space.f32(out_ptr, elems).copy_(v)
            elif out_mode == "bf16":
                space.bf16(out_ptr, elems).copy_(v.to(torch.bfloat16))
            elif out_mode == "fp24":
                hi_out, lo_out = _fp24_encode_py(v)
                space.view(out_ptr, elems * 2)[:] = hi_out.view(torch.uint8).numpy()
                space.view(scales_out_ptr, elems)[:] = lo_out.numpy()
            else:
                raise AssertionError(out_mode)

        self.emit("host", step)

    def dma_direct_wait_copy(self, dst_ptr, src_ptr, elems, *, wait):
        self.emit("wait_flag", wait[0], wait[1])
        space = self.space
        self.emit("host", lambda: space.bf16(dst_ptr, elems).copy_(space.bf16(src_ptr, elems)))


def _fp24_encode_py(vals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact bit-level model of _pack_fp24x4 (IEEE binary32 -> RN binary24: hi16 + lo8)."""
    assert vals.dtype == torch.float32 and vals.numel() % 4 == 0
    bits = vals.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    abs_bits = bits & 0x7FFFFFFF
    is_fin = abs_bits < 0x7F800000
    is_nan = abs_bits > 0x7F800000
    lsb = (bits >> 8) & 1
    r = lsb + 0x7F
    rounded = torch.where(is_fin, (bits + r) & 0xFFFFFFFF, bits)
    rounded = torch.where(is_nan, rounded | 0x00400000, rounded)
    hi16 = ((rounded >> 16) & 0xFFFF).to(torch.uint16)
    lo8 = ((rounded >> 8) & 0xFF).to(torch.uint8)
    return hi16, lo8


def _fp24_decode_py(hi16: torch.Tensor, lo8: torch.Tensor) -> torch.Tensor:
    """Exact bit-level model of _unpack_fp24x4."""
    u = ((hi16.to(torch.int64) & 0xFFFF) << 16) | ((lo8.to(torch.int64) & 0xFF) << 8)
    return u.to(torch.uint32).view(torch.float32)


def _build_world(monkeypatch, direct_mode: str, shard_elems: int, pieces: int, world: int,
                 shard_cap_override: int | None = None):
    from b12x.comm.pcie import pcie_dma

    space = emulator._Space()
    flags_bytes = pcie_dma.FLAG_SLOTS * pcie_dma.FLAG_STRIDE
    if shard_cap_override is not None:
        capacity = shard_cap_override
    elif direct_mode == "fp24":
        capacity = ((shard_elems * 3) + 255) // 256 * 256
    else:
        capacity = ((shard_elems * 4) + 255) // 256 * 256
    slabs = [space.allocate(flags_bytes + 2 * (world - 1) * capacity) for _ in range(world)]
    machines, runtimes = [], []
    for rank in range(world):
        machine = _DirectMachine(space)
        machine.streams = {name: emulator._Stream(machine, name) for name in ("copy", "flag", "ring")}
        rt = pcie_dma.PCIeDmaAllReduce.__new__(pcie_dma.PCIeDmaAllReduce)
        rt.world_size, rt.rank, rt.device = world, rank, "fake"
        rt._closed = False
        rt._direct_mode = direct_mode
        rt.shard_capacity = capacity
        rt._direct_max_shard_elems = shard_elems
        rt._flags_base = list(slabs)
        rt._scratch_base = [base + flags_bytes for base in slabs]
        for name in ("_send_counters", "_wait_counters", "_wait_done"):
            setattr(rt, name, _Tensor(space, pcie_dma.FLAG_SLOTS, torch.int32))
        rt._direct_stage = _Tensor(space, 2 * capacity, torch.uint8)
        rt._copy_stream = machine.streams["copy"]
        rt._flag_stream = machine.streams["flag"]
        rt._ring_stream = machine.streams["ring"]
        rt._piece_events = [emulator._Event(machine) for _ in range(pcie_dma.MAX_PIECES)]
        rt._copied_events = [emulator._Event(machine) for _ in range(2 * (world - 1) * pcie_dma.MAX_PIECES)]
        rt._input_ready = emulator._Event(machine)
        rt._kernels = machine
        rt._pieces_override = pieces
        machines.append(machine)
        runtimes.append(rt)
    return pcie_dma, space, machines, runtimes


def _reference_ring_bf16(inputs: torch.Tensor, world: int) -> torch.Tensor:
    """Exact ordered FP32 ring accumulation rounded once to BF16 per chunk (all-reduce)."""
    chunks = []
    for chunk in range(world):
        total = inputs[chunk, chunk].float().clone()
        for step in range(1, world):
            total = total + inputs[(chunk + step) % world, chunk].float()
        chunks.append(total.to(torch.bfloat16))
    return torch.stack(chunks).reshape(-1)


def _reference_rs_bf16(inputs: torch.Tensor, world: int) -> torch.Tensor:
    """Reduce-scatter ring order: rank r's shard sums partials (r+1) .. r in FP32, one BF16 round."""
    chunks = []
    for owner in range(world):
        total = inputs[(owner + 1) % world, owner].float().clone()
        for step in range(2, world + 1):
            total = total + inputs[(owner + step) % world, owner].float()
        chunks.append(total.to(torch.bfloat16))
    return torch.stack(chunks)


@pytest.mark.parametrize("world", [8, 9])
@pytest.mark.parametrize("pieces", [1, 2])
@pytest.mark.parametrize("in_place", [True, False])
@pytest.mark.parametrize("multi_interval", [False, True])
def test_bf16_direct_is_bit_identical_to_ordered_fp32_ring(monkeypatch, world, pieces, in_place, multi_interval):
    shard_elems = 512
    cap = 256 * 4 if multi_interval else None
    pcie_dma, space, machines, runtimes = _build_world(monkeypatch, "fp32", shard_elems, pieces, world, cap)
    gen = torch.Generator().manual_seed(42 + pieces + int(in_place) * 7 + world)
    rounds = 2
    inputs = (
        torch.randn((rounds, world, world, shard_elems), generator=gen)
        * torch.pow(2.0, torch.randint(-10, 11, (rounds, world, world, shard_elems), generator=gen).float())
    ).to(torch.bfloat16)
    inputs[0, 4 % world, 2, 13] = float("inf")
    inputs[0, 6 % world, 7, 29] = float("nan")

    results = [[None] * world for _ in range(rounds)]
    for rank, (machine, rt) in enumerate(zip(machines, runtimes)):
        src = _Tensor(space, world * shard_elems, torch.bfloat16)
        dst = src if in_place else _Tensor(space, world * shard_elems, torch.bfloat16)
        monkeypatch.setattr(pcie_dma.torch.cuda, "current_stream", lambda device=None, m=machine: m.main)
        monkeypatch.setattr(pcie_dma.torch.cuda, "stream", machine.stream)
        for r in range(rounds):
            machine.main.operations.append(
                ("host", lambda r=r, rank=rank, src=src: space.bf16(src.address, world * shard_elems).copy_(inputs[r, rank].reshape(-1)))
            )
            rt._run_direct_bf16(src, dst, reduce_scatter_only=False)
            machine.main.operations.append(
                ("host", lambda r=r, rank=rank, dst=dst: results[r].__setitem__(rank, space.bf16(dst.address, world * shard_elems).clone()))
            )
    emulator._execute(space, machines, seed=11)

    for r in range(rounds):
        expected = _reference_ring_bf16(inputs[r], world)
        for rank in range(world):
            got = results[r][rank]
            nan_mask = torch.isnan(expected)
            assert torch.equal(torch.isnan(got), nan_mask)
            assert torch.equal(got[~nan_mask].view(torch.int16), expected[~nan_mask].view(torch.int16)), (r, rank)


@pytest.mark.parametrize("world", [8, 9])
@pytest.mark.parametrize("pieces", [1, 2])
@pytest.mark.parametrize("in_place", [True, False])
def test_reduce_scatter_is_bit_identical_to_ordered_ring_owned_shards(monkeypatch, world, pieces, in_place):
    shard_elems = 512
    pcie_dma, space, machines, runtimes = _build_world(monkeypatch, "fp32", shard_elems, pieces, world)
    gen = torch.Generator().manual_seed(1701 + world + pieces)
    rounds = 2
    inputs = (
        torch.randn((rounds, world, world, shard_elems), generator=gen)
        * torch.pow(2.0, torch.randint(-9, 10, (rounds, world, world, shard_elems), generator=gen).float())
    ).to(torch.bfloat16)
    inputs[0, 3 % world, 1, 7] = float("inf")
    inputs[0, 5 % world, 2, 21] = float("nan")

    results = [[None] * world for _ in range(rounds)]
    for rank, (machine, rt) in enumerate(zip(machines, runtimes)):
        src = _Tensor(space, world * shard_elems, torch.bfloat16)
        dst = _Tensor(space, shard_elems, torch.bfloat16)
        monkeypatch.setattr(pcie_dma.torch.cuda, "current_stream", lambda device=None, m=machine: m.main)
        monkeypatch.setattr(pcie_dma.torch.cuda, "stream", machine.stream)
        for r in range(rounds):
            machine.main.operations.append(
                ("host", lambda r=r, rank=rank, src=src: space.bf16(src.address, world * shard_elems).copy_(inputs[r, rank].reshape(-1)))
            )
            rt._run_direct_bf16(src, dst, reduce_scatter_only=True)
            machine.main.operations.append(
                ("host", lambda r=r, rank=rank, dst=dst: results[r].__setitem__(rank, space.bf16(dst.address, shard_elems).clone()))
            )
    emulator._execute(space, machines, seed=23)

    for r in range(rounds):
        expected = _reference_rs_bf16(inputs[r], world)
        for rank in range(world):
            got = results[r][rank]
            nan_mask = torch.isnan(expected[rank])
            assert torch.equal(torch.isnan(got), nan_mask), (r, rank)
            assert torch.equal(got[~nan_mask].view(torch.int16), expected[rank][~nan_mask].view(torch.int16)), (r, rank)


@pytest.mark.parametrize("world", [8])
@pytest.mark.parametrize("pieces", [1, 2])
def test_fp24_ring_preserves_bound_across_binades_and_special_values(monkeypatch, world, pieces):
    shard_elems = 512
    pcie_dma, space, machines, runtimes = _build_world(monkeypatch, "fp24", shard_elems, pieces, world)
    gen = torch.Generator().manual_seed(109 + pieces)
    inputs = (
        (torch.rand((world, world, shard_elems), generator=gen, dtype=torch.float64) - 0.5)
        * 12.0
        * torch.pow(2.0, torch.randint(-8, 8, (world, world, shard_elems), generator=gen).double())
    ).to(torch.bfloat16)
    inputs[:, 0, 0] = 10000.0
    inputs[:, 0, 1] = 1e-4
    inputs[4 % world, 1, 5] = float("inf")
    inputs[2, 3, 10] = float("-inf")
    inputs[5 % world, 6, 17] = float("inf")
    inputs[7 % world, 6, 17] = float("-inf")
    inputs[6 % world, 8 % world, 31] = float("nan")

    ar_out = [None] * world
    for rank, (machine, rt) in enumerate(zip(machines, runtimes)):
        src = _Tensor(space, world * shard_elems, torch.bfloat16)
        dst = src
        monkeypatch.setattr(pcie_dma.torch.cuda, "current_stream", lambda device=None, m=machine: m.main)
        monkeypatch.setattr(pcie_dma.torch.cuda, "stream", machine.stream)
        machine.main.operations.append(
            ("host", lambda rank=rank, src=src: space.bf16(src.address, world * shard_elems).copy_(inputs[rank].reshape(-1)))
        )
        rt._run_direct_bf16(src, dst, reduce_scatter_only=False)
        machine.main.operations.append(
            ("host", lambda rank=rank, dst=dst: ar_out.__setitem__(rank, space.bf16(dst.address, world * shard_elems).clone()))
        )
    emulator._execute(space, machines, seed=23)

    exact = inputs.double().sum(dim=0).reshape(-1)
    mag = inputs.double().abs().sum(dim=0).reshape(-1)
    rounded = exact.to(torch.bfloat16)
    finite = torch.isfinite(rounded) & torch.isfinite(mag)
    bound = (8 * 2.0 ** -16 * mag + 2.0 ** -8 * exact.abs() + 1e-30)[finite]
    for rank in range(world):
        res = ar_out[rank]
        assert torch.equal(torch.isnan(res), torch.isnan(rounded)), rank
        assert torch.equal(torch.isposinf(res), torch.isposinf(rounded)), rank
        assert torch.equal(torch.isneginf(res), torch.isneginf(rounded)), rank
        if rank > 0:
            nm = ~torch.isnan(res)
            assert torch.equal(res[nm].view(torch.int16), ar_out[0][nm].view(torch.int16))
        err = (res.double() - exact).abs()[finite]
        worst = float((err / bound).max())
        assert worst <= 1.01, f"worst error ratio {worst}"
        rel_l2_fp24 = float((res[finite].double() - exact[finite]).norm() / exact[finite].norm())
        rel_l2_fp32 = float((rounded[finite].double() - exact[finite]).norm() / exact[finite].norm())
        assert abs(rel_l2_fp24 - rel_l2_fp32) / rel_l2_fp32 < 0.01, (rel_l2_fp24, rel_l2_fp32)
        agree = float((res[finite].view(torch.int16) == rounded[finite].view(torch.int16)).float().mean())
        assert agree > 0.99, f"agreement {agree}"
