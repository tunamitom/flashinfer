"""CPU stand-in for one rank's CUDA streams and the DMA kernels (L9 ring port).

Extracted from the tree-ring test's emulator (996df5678) so the direct-ring and
reduce-scatter tests can drive the real `_run_direct_bf16` scheduling without
GPUs; operations execute in random interleavings and the runtime's stream logic
must never stall. `_collapse` + `_execute_collapsed` add the second execution
model: one serial queue per rank in a fixed topological order that prefers the
ring stream (a captured graph's executor may serialize independent nodes), the
same order on every rank — the adversarial shape the direct ring's missing
publish-to-wait edge died in.
"""

from __future__ import annotations

import random

import torch

class _Space:
    def __init__(self):
        self.blocks, self.cursor = [], 1 << 24

    def allocate(self, nbytes):
        import numpy as np

        base = (self.cursor + 4095) // 4096 * 4096
        self.cursor = base + nbytes + 65536
        self.blocks.append((base, np.zeros(nbytes, dtype=np.uint8)))
        return base

    def view(self, address, nbytes):
        for base, block in self.blocks:
            if base <= address and address + nbytes <= base + block.size:
                return block[address - base : address - base + nbytes]
        raise AssertionError(f"access outside every allocation: {address:#x} + {nbytes}")

    def bf16(self, address, elems):
        import numpy as np

        return torch.from_numpy(self.view(address, elems * 2).view(np.int16)).view(torch.bfloat16)

    def f32(self, address, elems):
        import numpy as np

        return torch.from_numpy(self.view(address, elems * 4).view(np.float32))


class _FakeTensor:
    def __init__(self, space, nbytes, dtype=torch.bfloat16):
        self.space, self.nbytes, self.dtype = space, nbytes, dtype
        self.address = space.allocate(nbytes)

    def data_ptr(self):
        return self.address

    def numel(self):
        return self.nbytes // (2 if self.dtype == torch.bfloat16 else 4 if self.dtype == torch.int32 else 1)


class _Stream:
    def __init__(self, machine, name):
        self.machine, self.name, self.operations = machine, name, []

    def wait_event(self, event):
        self.operations.append(("wait_event", event.last))

    def wait_stream(self, other):
        self.operations.append(("join", other, len(other.operations)))


class _Event:
    def __init__(self, machine):
        self.machine, self.last = machine, None

    def record(self, stream):
        self.machine.records += 1
        # Events belong to one rank: an identity includes its machine.
        self.last = (id(self.machine), self.machine.records)
        stream.operations.append(("record", self.last))


class _Machine:
    """Stands in for torch.cuda and the DMA kernels of one rank while its operations are issued."""

    def __init__(self, space):
        self.space, self.records, self.current = space, 0, []
        self.main = _Stream(self, "main")

    def stream(self, stream):
        machine = self

        class Scope:
            def __enter__(self):
                machine.current.append(stream)

            def __exit__(self, *exc):
                machine.current.pop()

        return Scope()

    def emit(self, *operation):
        (self.current[-1] if self.current else self.main).operations.append(operation)

    # DmaKernels
    def dma_copy(self, destination, source, nbytes):
        self.emit("copy", destination, source, nbytes)

    def dma_set_flag(self, flag, counter):
        self.emit("set_flag", flag, counter)

    def dma_wait_flag(self, flag, counter):
        self.emit("wait_flag", flag, counter)

    def dma_tree_accum(self, destination, values, partials, elems, final):
        self.emit("accumulate", destination, tuple(values), tuple(partials), elems, final)


def _execute(space, machines, seed):
    """Run every rank's streams to completion in a random interleaving."""
    import numpy as np

    order = random.Random(seed)
    done_records = set()
    streams = [(rank, stream) for rank, machine in enumerate(machines)
               for stream in (machine.main, *machine.streams.values())]
    cursor = {id(stream): 0 for _, stream in streams}
    armed = {}

    def u32(address):
        return space.view(address, 4).view(np.uint32)

    def step(stream):
        operation = stream.operations[cursor[id(stream)]]
        kind = operation[0]
        if kind == "wait_event":
            if operation[1] is not None and operation[1] not in done_records:
                return False
        elif kind == "record":
            done_records.add(operation[1])
        elif kind == "join":
            if cursor[id(operation[1])] < operation[2]:
                return False
        elif kind == "copy":
            _, destination, source, nbytes = operation
            space.view(destination, nbytes)[:] = space.view(source, nbytes).copy()
        elif kind == "set_flag":
            counter = u32(operation[2])
            counter[0] += 1
            u32(operation[1])[0] = counter[0]
        elif kind == "wait_flag":
            key = (id(stream), cursor[id(stream)])
            if key not in armed:
                counter = u32(operation[2])
                counter[0] += 1
                armed[key] = int(counter[0])
            if int(u32(operation[1])[0]) - armed[key] < 0:
                return False
        elif kind == "accumulate":
            _, destination, values, partials, elems, final = operation
            total = space.bf16(values[0], elems).float()
            for address in values[1:]:
                total = total + space.bf16(address, elems).float()
            for address in partials:
                total = total + space.f32(address, elems)
            if final:
                space.bf16(destination, elems).copy_(total.to(torch.bfloat16))
            else:
                space.f32(destination, elems).copy_(total)
        elif kind == "place":
            _, destination, source, nbytes = operation
            space.view(destination, nbytes)[:] = space.view(source, nbytes).copy()
        elif kind == "host":
            operation[1]()
        else:
            raise AssertionError(kind)
        cursor[id(stream)] += 1
        return True

    remaining = sum(len(stream.operations) for _, stream in streams)
    while remaining:
        candidates = [stream for _, stream in streams if cursor[id(stream)] < len(stream.operations)]
        order.shuffle(candidates)
        progressed = 0
        for stream in candidates:
            if step(stream):
                progressed += 1
                remaining -= 1
                if order.random() < 0.4:
                    break
        assert progressed, "the runtime's streams stall"


def _collapse(machines, prefer=("ring", "flag", "copy", "main")):
    """Collapse every rank's streams into ONE serial queue: a fixed topological
    order (per-stream order, wait_stream joins and event record->wait edges all
    preserved) that prefers the ring stream over the flag and copy streams —
    the captured-graph executor's freedom to serialize independent nodes, taken
    adversarially. Every rank issues the same operations, so the queues are the
    same order on every rank.
    """
    orders = []
    for machine in machines:
        streams = [machine.main] + list(getattr(machine, "streams", {}).values())
        by_name = {}
        for stream in streams:
            by_name.setdefault(stream.name, stream)
        cursor = {id(stream): 0 for stream in streams}
        records, queue = set(), []
        remaining = sum(len(stream.operations) for stream in streams)

        def placeable(stream):
            if cursor[id(stream)] >= len(stream.operations):
                return False
            operation = stream.operations[cursor[id(stream)]]
            kind = operation[0]
            if kind == "wait_event":
                return operation[1] is None or operation[1] in records
            if kind == "join":
                return cursor[id(operation[1])] >= operation[2]
            return True

        while remaining:
            for name in prefer:
                stream = by_name.get(name)
                if stream is not None and placeable(stream):
                    operation = stream.operations[cursor[id(stream)]]
                    if operation[0] == "record":
                        records.add(operation[1])
                    queue.append(operation)
                    cursor[id(stream)] += 1
                    remaining -= 1
                    break
            else:
                raise AssertionError("the collapsed order has no placeable operation")
        orders.append(queue)
    return orders


def _execute_collapsed(space, orders, seed):
    """Run every rank's collapsed queue serially — its serialized graph order —
    with ranks interleaved in random order. A blocked wait holds its rank's whole
    queue, exactly as a serialized node order would; if no rank can advance, the
    queues deadlock and this raises.
    """
    import numpy as np
    order = random.Random(seed)
    done_records = set()
    cursor = {rank: 0 for rank in range(len(orders))}
    armed = {}

    def u32(address):
        return space.view(address, 4).view(np.uint32)

    while any(cursor[rank] < len(orders[rank]) for rank in cursor):
        ranks = [rank for rank in cursor if cursor[rank] < len(orders[rank])]
        order.shuffle(ranks)
        progressed = 0
        for rank in ranks:
            index = cursor[rank]
            operation = orders[rank][index]
            kind = operation[0]
            if kind == "wait_event":
                if operation[1] is not None and operation[1] not in done_records:
                    continue
            elif kind == "record":
                done_records.add(operation[1])
            elif kind == "join":
                pass  # the collapse already placed the joined work ahead of this op
            elif kind == "copy" or kind == "place":
                _, destination, source, nbytes = operation
                space.view(destination, nbytes)[:] = space.view(source, nbytes).copy()
            elif kind == "set_flag":
                counter = u32(operation[2])
                counter[0] += 1
                u32(operation[1])[0] = counter[0]
            elif kind == "wait_flag":
                key = (rank, index)
                if key not in armed:
                    counter = u32(operation[2])
                    counter[0] += 1
                    armed[key] = int(counter[0])
                if int(u32(operation[1])[0]) - armed[key] < 0:
                    continue
            elif kind == "accumulate":
                _, destination, values, partials, elems, final = operation
                total = space.bf16(values[0], elems).float()
                for address in values[1:]:
                    total = total + space.bf16(address, elems).float()
                for address in partials:
                    total = total + space.f32(address, elems)
                if final:
                    space.bf16(destination, elems).copy_(total.to(torch.bfloat16))
                else:
                    space.f32(destination, elems).copy_(total)
            elif kind == "host":
                operation[1]()
            else:
                raise AssertionError(kind)
            cursor[rank] += 1
            progressed += 1
            if order.random() < 0.4:
                break
        assert progressed, "every rank's collapsed queue is stalled"


