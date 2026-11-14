# `test_pcie_dcp_a2a.py` — the 31-failure baseline, explained (W6 fix-on-the-way)

W6 D8 asks for an explanation of the 31-failure baseline in
`tests/experimental/b12x/comm/test_pcie_dcp_a2a.py` before the CP merge is
built on this file. Measured state (2026-11-14, CPU, the
`/opt/flare-local-inference/.venv` interpreter): **31 failed, 35 passed**.

## What the 35 passing cover — the merge math itself is green

The reference arithmetic the Hy4 CP merge contracts for passes:

- `test_reference_selects_destination_heads_and_combines_lse_weights` —
  `lse_reduce_scatter_reference` picks each destination rank's contiguous head
  shard and combines partials by their LSE weights (`pcie_dcp_a2a.py:308-417`);
- `test_reference_ignores_nan_output_from_empty_shard` — empty shards
  contribute zero weight and `-inf` LSE, never NaN;
- `test_lse_log_base_is_a_runtime_kernel_argument` and the surrounding
  kernel-source/compile tests — the base-2/natural switch is a real runtime
  argument (`resolve_lse_base`, nothing defaults);
- the graph-epoch, barrier, slot-delta and channel-pool tests.

So the 31 failures do **not** implicate the LSE merge math that the CP
reduce-scatter (and W6 gate a) rides on.

## Why the 31 fail — the test file predates the prepared-`Plan` API refactor

Every failure is test-surface drift against the current `PCIeDCPA2A` /
preparation API, not a math or transport defect. The failure signatures
(`--tb=line`, grouped):

| Count | Failure | Cause |
|---|---|---|
| 21 | `TypeError: PCIeDCPA2A.{lse_reduce_scatter,all_gather_pair_kimi_topk,kimi_topk16}() missing 1 required keyword-only argument: 'plan'` (plus 1 `kimi_topk16() ... 'plan'` at module level) | The tests call the runtime methods pre-`Plan` style; the runtime now requires the prepared `plan=` (`_prepared_state(plan)`), and the prepared surface is `plan.request(...)`/`DcpExecutionState.run` (`_dcp_preparation.py`) |
| 4 | `AttributeError: '_FakeKimiRuntime' object has no attribute 'prepare_graph_all_gather_pair_kimi_topk'` (1 × `_FakeRuntime ... 'prepare_graph_lse_reduce_scatter'`) | The test doubles implement the old `prepare_graph_*` hooks; the renamed hooks moved with the same refactor |
| 2 | `AssertionError: 'generation = ld_relaxed_gpu_u32(' in ...` / `'_a2a_graph_epoch_arrive(' in ...` | Kernel-source string pins on renamed internals of the all-gather head kernel |
| 1 | `assert 407040 == (209664 + 197120)` | A stale byte-accounting expectation from an older staging layout |
| 1 | `ValueError: not enough values to unpack (expected 2, got 1)` | A return-shape change at the same refactor |

None of these touch `lse_reduce_scatter_reference`,
`apply_attention_sink_after_merge`, `resolve_lse_base` or the `_require_single_fold`
double-sink guard — the contract the W6 CP merge and `PreparedSinkFold` use, and
what W6 gate a validates end to end in CPU simulation.

## Consequence for the W6 schedule

Per `w6-serving-world.md` §8, this explanation clears the dependency: the
merge itself is reliable (the math is tested and the kernels compile), so the
CP work stands on it and **no** "bake a smaller context without CP" fallback
is triggered. The stale test file itself should be ported to the prepared
`plan=`/`DcpExecutionState` surface in a separate cleanup (out of W6 scope;
the 31-failure count is expected to stay at 31 until then, and W6 changes
nothing in it — verified unchanged after the `_dcp_preparation`
`partials_sink_folded` passthrough fix).
