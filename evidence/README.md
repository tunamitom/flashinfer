# A4 prefill arena/admission fix — evidence

**Commit:** `0b042d3e8` ("Plan-time A4 prefill workspace reservation for the
fused MoE planner"), cherry-picked onto `2f5210dd1` (current `main` at PR
time). Original qualification commit: `d16e05bc8` in our integration tree
(byte-identical content; `git range-diff` shows 1:1 with a cherry-pick
trailer and one documented conflict resolution — the pick's diff context
carried an unrelated MXFP4 block that does not exist on `main`; dropped in
the conflict hunk, recorded in the commit body).

This document is a data summary. All numbers below come from the artifacts
in `evidence/raw/` (junit XMLs, console logs, patch, provenance). Test and
measurement methodology is stated with each result, including what was
*not* measured.

---

## 1. Problem

The stock planner sizes `intermediate_cache2` as
`routed_capacity × intermediate_size`. The A4 prefill pipeline carves that
buffer differently (quantized input + intermediate planes, E4M3 scales,
padded route metadata), so at near-full chunks the A4 carve does not fit.
`fits_buffers()` / `select_a4()` then **reject A4 silently** and the call
falls back to W4A16 with no warning — i.e. the hybrid A4-prefill /
W4A16-decode mode never engages at the chunk sizes that matter (8k tokens
in our serving geometry).

Observed on the unpatched planner in production geometry (probe evidence,
§4): `cache2` planned 33,554,432 bytes (stock formula), `a4_selected=False`
on plain-path binds even when A4 was requested.

## 2. Fix

One commit, 6 files, +979/−8:

| File | Change |
|---|---|
| `moe/fused_moe/_impl.py` | `_plan_core_workspace` now takes `w4a16_a4_prefill_enabled/_terms` and reserves `max(stock, carve)` for `intermediate_cache2`, plus per-route FC2 rows for `intermediate_cache13`. Sizes come from the compiled launches' own `_carve_layout` (`a4_prefill_sizing_launches` → `a4_prefill_workspace_requirements`) — no second hand-maintained formula. Flag-off plans keep stock sizes. Reservation is plan-level: an enabled hybrid plan reserves even when a given call binds A16. |
| `moe/_shared/kernels/w4a16/prefill_a4.py` | exposes the workspace carve (`a4_prefill_workspace_requirements`, `_carve_layout` wiring) the planner consumes. |
| `moe/test_w4a16_a4_prefill.py` | expectation updates for the plan-level reservation. |
| `moe/test_w4a16_a4_prefill_admission.py` (new) | plan/bind contract, 6 GPU cases. |
| `moe/test_w4a16_a4_prefill_arena.py` (new) | serving-path byte layout, terms 1/2, fused-sum cache13, frozen controls; asserts the recorded growth equals exactly the cache2 carve delta at GLM geometry. |
| `moe/test_w4a16_a4_prefill_fullgeom.py` (new) | E=256/H=6144/topk=8, capacity 8192: full/near-full execution, CSF inline vs expanded reader parity, graph replay with poisoned scratch, explicit-A16 control. |

No kernel arithmetic, no calibration-scale, no tolerance change. The fix is
plan-level sizing only.

## 3. Regression evidence (three-baseline attribution)

Method: (P) patched source + patched tests; (B1) pristine source + the same
patched tests; (B2) pristine source + pristine tests. B1's failures *are* the
attribution evidence; B2 proves the pristine tree is green on its own tests
so P-side outcomes are patch-caused. GPU 0, image
`kk-unified-20261008-r13` (`sha256:fe71cf02…`), source trees
hash-verified before every run (`evidence/raw/provenance.txt`).

| Run | Result | Failures/errors (exact nodes) |
|---|---|---|
| P — patched + patched tests | **83 passed, 1 skipped** (84 cases; skip = `test_a4_prefill_launchers_are_owned_by_each_device`, needs 2 GPUs — later run on 2 GPUs, passed) | none. Admission nodes: 6/6 passed |
| B1 — pristine + same patched tests | 31 passed, 1 failed, 1 skipped, 1 error | `test_w4a16_a4_prefill_admission` — **collection error** (new test file imports patch-added symbols; registers as 1 error node, not N failing tests) · `test_w4a16_a4_prefill::test_csf_prefetch_query_matches_exact_variant_and_precision[True]` — expectation flip |
| B2 — pristine + pristine tests | **32 passed, 1 skipped** | none |

Honest note: the qualification run pre-registered an expectation of ≥15
failing admission nodes on B1; the observed count was 1 (the collection
error above). A collection `ImportError` consumes the whole new test file as
one error node in JUnit, so the per-test failure count is not a measure of
how much B1 differs. The failure *set* (2 nodes, both patch-dependent) is
the attribution evidence, and it is exactly the set of nodes whose behavior
the patch changes. The pre-registered expectation itself is recorded as
violated in the run log (`evidence/raw/console/console-pristine-same-tests.log`
cited from `results.txt`).

Patched-suite evidence files: `evidence/raw/junit/junit-patched.xml`,
`junit-pristine-same-tests.xml`, `junit-pristine-head.xml` (full node IDs).

## 4. Serving engagement (probe evidence)

Method: in-process bind probe (`B12X_CANARY_PROBE_LOG`) recording every
`TPMoEScratchPlan.bind` on a live 8×TP serving instance (GLM geometry,
E=256/H=6144/topk=8, prepared capacity 8192), cold full-chunk 8,192-token
prefill request; A16 baseline arm and A4 candidate arm on the same image +
source (candidate = patched source mounted read-only).

- **A4 selected on every force=True call**: 60,888 full-chunk binds +
  4,768×2 metadata-split binds (rows=80/1), symmetric across all 8 TP ranks
  (1,915/2,140 per rank).
- **Fallback reasons: only `a4_not_requested`** (600 + 2,384
  decode/verification calls legitimately stayed W4A16). **Zero buffer-fit or
  capacity declines** — the failure mode the patch fixes.
- **Workspace: required == available exactly** — cache2 38,146,560 B,
  cache13 805,306,368 B, capacity 8192. Stock formula plans cache2 at
  33,554,432 B (baseline arm confirms `a4_selected=False` there).
- Scheduler metadata live (`prefill_ranges [[0,8192]]`); CSF readers both
  live (expanded 60,472 / inline 9,952 binds); prepared plan ladder
  [1,2,4,8,16,24,32,40,48,56,64,8192].
- Decode/MTP requests stay W4A16 by design (force=False → never selected).

## 5. Performance (cold-request, 30 trials, 5 samples × 3 contexts)

Method: A16 baseline vs A4-prefill candidate on identical image/source,
cold requests (prefix cache hits = 0 on every trial), medians with min–max
bands. Decode path identical in both arms (W4A16).

| Metric | Context | A16 | A4 prefill | Δ |
|---|---|---|---|---|
| ttft (s) | 8k | 2.275 (2.273–2.315) | 2.152 (2.149–2.195) | **−5.4%** |
| | 64k | 18.871 (18.834–18.902) | 17.882 (17.841–17.903) | **−5.2%** |
| | 128k | 38.280 (38.269–38.310) | 36.293 (36.250–36.330) | **−5.2%** |
| prefill tok/s | 8k | 3600.7 (3538.0–3604.5) | 3806.5 (3731.8–3812.7) | **+5.7%** |
| | 64k | 3472.9 (3467.2–3479.7) | 3664.9 (3660.7–3673.4) | **+5.5%** |
| | 128k | 3424.0 (3421.3–3425.0) | 3611.5 (3607.9–3615.8) | **+5.5%** |
| decode tok/s | 8k | 97.4 | 97.2 | −0.2% (IQRs overlap) |
| | 64k | 101.2 | 99.1 | −2.1% (IQRs overlap) |
| | 128k | 97.4 | 97.5 | +0.1% (IQRs overlap) |

Min–max bands do not overlap between arms on ttft/prefill at any context.
Decode and MTP rate differences are within sampling noise (overlapping
IQRs) — decode is W4A16 in both arms, as designed. Peak memory −18 MiB.

## 6. Quality cost of the A4 quantization mode (not of this patch)

Method: per-token NLL of held-out tokens via completions echo logprobs,
18 passages, **84,617 scored positions**, A16 vs A4 arms, full input-ID
coverage per prompt, tokenization identity recorded, operands ordered
A4−A16.

| | A16 | A4 | Δ |
|---|---|---|---|
| mean NLL/token | 0.88837 | 0.91935 | **+0.03098** |
| perplexity | 2.4312 | 2.5077 | **+3.1% relative** |

Per-token: 36,602 improved / 47,856 worsened / 159 unchanged. Total-loss
delta +2,621.06 nats. Degradation is larger on the two long documents in
the set. English-only corpus; calibration overlap with the corpus unknown.

This cost is inherent to the A4 activation-quantization mode (4-bit
activations in prefill). The present patch does not change arithmetic — it
only makes the planner admit the A4 path; without it the mode silently does
nothing. Whether +3.1% perplexity is acceptable for the prefill speedup is
an operator decision.

## 7. Numerical validation of the A4 kernel path

We separately validated the A4 prefill numerical path (kernel vs FP64
emulation reference, 6 captured layer states at TP0, 256 tokens, terms=1,
bound 4e-3): FC1 GEMM, intermediate quantization, FC2 GEMM, and the
reduction each match FP64 controls within BF16 rounding of the same
operands. Two initial end-to-end discrepancies were traced to defects in
the *test references*, not the kernel: (1) exact-tie handling at input
quantization in the original divide/midpoint reference; (2) device-dependent
E4M3 block-scale rounding in the GPU-executed reference quantizer (adjacent
E4M3 scale values vs the CUDA kernel's). With corrected references, all six
captures sit at the final-BF16-cast floor (~1.7e-3 rel-L2, within the 4e-3
bound). Full data: internal evidence tree (available on request); this PR
ships only the plan-level fix, which contains no arithmetic change.

## 8. Scope and limitations

- Tested at TP0 / full TP8 serving engagement, GLM-5.3 geometry
  (E=256/H=6144/topk=8), prepared capacity 8192, terms=1, 256-token
  captures for the numerical checks, 8k/64k/128k contexts for perf.
- Performance and quality numbers come from our serving instance
  (8×GPU, LMCache-enabled, specific checkpoint); absolute numbers will
  differ elsewhere — the plan-level sizing change is geometry-generic.
- The two-device ownership test is the only skip in the suite (needs 2
  GPUs); it passed when run with 2 GPUs.
- Corpus is English-only (18 passages); the +3.1% perplexity figure may not
  generalize.
- Flag-off behavior unchanged: with the A4 prefill flag unset, plans keep
  stock sizes (asserted in the arena suite).

## 9. Evidence index (`evidence/raw/`)

| File | What it is |
|---|---|
| `patch-of-record.diff` | the exact qualified diff (sha256 `e5cfb17af4fe79460518a1962f1a8c54d6e2ade58183164d5b36c32c8642224e`, 1,112 lines; equals `git diff a087dfbf..d16e05bc` for the six files) |
| `provenance.txt` | source commit/file hashes at qualification time, pristine image hashes |
| `junit/junit-patched.xml` | P run, full node results |
| `junit/junit-pristine-same-tests.xml` | B1 run (attribution failures) |
| `junit/junit-pristine-head.xml` | B2 run (pristine green) |
| `console/console-*.log` | full console output per run |
| `loss/loss-comparison-corpus-84617tokens.json` | the §6 corpus measurement (18 passages, 84,617 scored positions, per-prompt and per-token data included) |