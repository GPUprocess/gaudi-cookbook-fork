# ACCURACY.md — evaluation methodology and results

## Policy

Accuracy gate: **maximum 0.5 absolute percentage points of regression** versus the established
baseline on the agreed metric, measured with a fixed, versioned evaluation suite under
`docs/BENCHMARK_POLICY.md`. Correctness failures reject an optimization regardless of speed.

## Current status: evaluation suite NOT YET ESTABLISHED

No formal accuracy evaluation (per the policy's requirements: fixed datasets, paired
control/treatment runs, per-category metrics, uncertainty estimates) has been run yet. The
results below are **correctness and equivalence evidence**, not an accuracy score. Until the
suite exists, every performance result is capped at BASELINE CANDIDATE regardless of correctness.

## Correctness and equivalence evidence collected so far

All of these were run against the served MTP k=1 configuration (and k=2 where noted) and passed:

- **Greedy golden-reference comparison** (12 recorded prompts, same config): 2/12
  token-identical; every divergence measured as a verify-shape FP near-tie (top-2 logprob gaps
  recorded at each divergence; e.g. gap 0.25 flipped vs gap 1.375 in the reference). Content
  comparison after reasoning stripping: divergences are word-choice equivalents (e.g.
  "recursive approach" vs "recursive version"); divergence positions identical at k=1 and k=2.
  A non-MTP run on the same window reproduces 6/6 token-identical, attributing divergences to
  the speculative path, not the config.
- **Greedy target-behavior preservation** (`docs/BENCHMARK_POLICY.md` speculative requirement):
  satisfied in the near-tie sense documented above; a formal tie-breaking tolerance statement
  is pending.
- **Feature gates under MTP**: tools (single/parallel/no-call), images (4/prompt), streaming,
  penalties (frequency/presence/repetition), sampling (top-p/top-k/temperature), JSON-schema
  and guided-regex structured output, top-5 logprobs, client disconnect mid-generation,
  multi-turn, soak-repeat (5 identical greedy requests, no drift).
- **Positional correctness**: prompts sized to straddle KV-block boundaries (639/640/641 and
  1279/1280/1281 tokens) — all correct.
- **Long-context**: 16/16 access codes retrieved at 4 depths × 3 codes (16K cap of the 32K
  window) via the port's needle gate.
- **Long generation**: 1918-token generation (k=2) and 2034-token generations (k=1) with zero
  degeneration loops and coherent tails.
- **k=2 content equivalence**: divergence positions identical to k=1; word-choice class.

## TODO: establish the accuracy suite (blocks VERIFIED status for all performance results)

1. Select a fixed evaluation set covering reasoning, instruction following, math, coding,
   general language (size and license suitable for redistribution of *results*, not data).
2. Version the suite in-repo (`benchmarks/accuracy/`) with prompts, scoring code, decoding
   settings, and expected-output/scoring method.
3. Define the aggregate metric and per-category reporting; define the uncertainty method.
4. Run paired control (non-MTP) vs treatment (MTP k=1) evaluations on the same suite.
5. Apply the 0.5-percentage-point gate; record INCONCLUSIVE when evidence is insufficient.
6. Record the suite version in every experiment report that claims an accuracy pass.

## Known numerical caveats

- Speculative verification changes the FP shape of the verify forward (padded rows, k+1
  positions); near-tie decisions can flip. Greedy output is therefore *not* bit-identical to
  non-MTP by design of the hardware path; the equivalence argument is the measured near-tie
  class plus content equivalence, and the formal sampled-distribution check is still pending
  (`docs/BENCHMARK_POLICY.md` sampled-decoding requirement).
- k=2 uses a 3-row verify and S=4 candidate columns; divergence positions matched k=1,
  supporting that the FP-shift class is depth-stable.
