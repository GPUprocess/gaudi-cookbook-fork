# RA-014 — Prefill 2-4K bucket-transition fix (config-only)

**Status:** PROPOSED (research architect, 2026-11-09) · **Priority:** 6 (config-only; slot into any boot family) · **Complexity:** Low
**Parent:** `BOTTLENECK_RESEARCH.md` (Section 3, RA-014) · **Target:** the one unmet sub-case of the prefill objective (3-6K t/s): the 2,538-2,787 t/s dip at 3.5-4K prompts

---

## 1. Executive summary

The prefill objective (3,000-6,000 t/s) is MET across most of the 0.5-8K range but DIPS at exactly the
2048→4096 lin-bucket transition: 2,538-2,787 t/s at 3.5-4K (MEASURED, scorecard) vs 3,338-5,697 at
0.5-2K and 3,782-5,663 at 6-8K. The lin bucketing steps the query dimension by 128 up to 2048, then by
2048 — so a 3.5K prompt lands in the 4096 bucket (~15% padding waste at the edge), and the first use of
the 4096 bucket mid-bench pays the 13-15 s lazy first-use compile if the shape was not pre-warmed. The
6-8K band being MET again confirms the dip is transition-specific, not a long-context cliff. The fix is
config-only: (a) finer query buckets in the 2-4K range (512-step: 2048/2560/3072/3584/4096 — edge waste
≤6%), and/or (b) pre-warming the intermediate shapes in the boot's warm client (the warm protocol
already exists for the decode ladder — extend the shape list). No code, no kernel, 1 boot.

## 2. Current limitation (evidence)

| Fact | Value | Source |
|---|---|---|
| The dip (MEASURED) | Prefill (warm, 3.5-4K): **2,538-2,787 t/s** — "slightly below — the 2048→4096 lin-bucket transition"; 0.5-2K: 3,338-5,697 t/s (MET); 6-8K: 3,782-5,663 t/s (MET again) | the scorecard (BASELINE.md / the diagnosis) |
| The bucketing | lin buckets: 128-step query up to 2048, then 2048-step (2048, 4096, 6144, …); a 3.5K prompt → the 4096 bucket (padding 3.5K→4096 ≈ 15%; a 2.1K prompt → the 4096 bucket, padding ≈ 49% — the worst edge case sits just above 2048) | the bucketing config (the `kb_staging/boot_*.sh` family; E015-A) |
| The cold-compile cost | 13-15 s per new graph shape at first use (lazy-mode first-use compilation — the adopted warm protocol exists precisely for this) | CURRENT_STATE.md (the verified-results header: "Lazy-mode first-use compiles 13-15 s per new graph shape; pre-warm served shapes") |
| The warm protocol precedent | the decode ladder is pre-warmed in the warm client (the warm protocol is adopted, RESEARCH-001 addendum-9 / the companion doc) | project practice |
| The bucketing mechanism is proven | E015-A: lin bucketing adopted (+1.4% decode, warmup 3× faster) | EXPERIMENTS.md |
| The KDA state-chunk interaction | the KDA recurrent state is per-128-token chunks; 128 divides 2048/2560/3072/3584/4096 → NO state-block alignment interaction with the proposed bucket sizes (the gpt-oss/GLM chunk-boundary lesson — G12 — does not bite here) | the KDA architecture (the per-128 state) + the bucket arithmetic |

**The gap:** the dip has been recorded in the scorecard but never attacked (no experiment in
EXPERIMENTS.md targets the 3.5-4K band); the 4096 bucket's cold-compile cost is unquantified in the
bench (the bench's warm-up set may or may not include it — check the warm client's shape list, which is
the first implementation step).

## 3. Proposed architecture

Config-only, two independent levers (test separately so the A/B attributes the effect):

**Lever A — finer query buckets in the 2-4K range:**
- The lin-bucket query step: 128 up to 2048 (unchanged), then **512** for 2048→4096: buckets
  2048, 2560, 3072, 3584, 4096 (then 2048-step beyond, unchanged: 6144, 8192, …).
- Edge waste: a 3.5K prompt → 3584 bucket (2.4% waste) instead of 4096 (15%); a 2.1K prompt → 2560
  (14%… precisely 2048+512=2560: 2.1K→2560 = 18%… the worst case drops from 49% to ≤25% at the
  2048/2560 boundary — and the 2048-step buckets below 2048 already handle ≤2K at 128-step).
- HBM cost: 3 extra graph shapes (2560/3072/3584 query rows). The prefill graph workspace per shape is
  row-scaled; at 2.5-3.5K rows the activation workspace is a fraction of the 8K-row prefill graphs
  (the 25.5 GiB first-reclaimable-block figure applies to the 8K-row shapes — the companion doc). The
  decode pool has ~28 GB free (96 − 68 GB used) — the 3 shapes fit with wide margin. Boot warm time:
  +3 shapes × 13-15 s ≈ +40-45 s of warm-up.
- The knobs: the `VLLM_PROMPT_*` bucketing args in the boot scripts (the Q04 bucket-pinning work
  already identified them — read the current values from the adopted boot script).

**Lever B — pre-warm the intermediate shapes:**
- Add 2560/3072/3584/4096 query shapes to the warm client's shape list (the decode ladder already
  pre-warms; extend the prefill shape list).
- HBM cost: same as Lever A if combined (the shapes are the same); if Lever A is not adopted, Lever B
  alone pre-warms only the 4096 shape (1 shape, ~13-15 s).

**A/B design (1 boot family, same-boot protocol):**
- Arm 0 = current (128-step ≤2048, 2048-step beyond; the current warm list).
- Arm A = Lever A only (512-step 2048-4096; the current warm list — isolates the padding effect,
  accepting the 3 cold compiles mid-bench).
- Arm B = Lever A + Lever B (the 512-step buckets + all 4 shapes pre-warmed — the full fix).
- (Optional Arm A' = Lever B only (4096 pre-warmed, coarse buckets) if a 4-arm boot is acceptable —
  it isolates the cold-compile effect at the current bucketing.)

Bench: the prefill ladder (0.5K / 2K / 2.5K / 3.5K / 4K / 6K / 8K, warm, gen=1, 3 reps/depth — the
scorecard protocol). The 2.5K depth is NEW (it sits at the worst edge of the current 2048→4096
transition — the 49%-padding case).

## 4. Technical feasibility

- **Pure serving config:** the bucketing + the warm list are boot args; no code, no kernel, no graph
  change beyond the 3 extra cached shapes (which the lazy compiler handles the same way as every
  adopted shape — E015-A is the precedent that the mechanism works on this tree).
- **The padding-waste mechanism is direct:** the lin GEMMs/attention at query length L run at the
  bucket length B ≥ L; the compute (and the KV writes) scale with B, so the t/s at the bucket edge is
  t/s_ideal × L/B. 3.5K at 4096: ×0.855; at 3584: ×0.977. The measured 2,538-2,787 band × the padding
  recovery (1.15-1.44 at the 2.1K-3.5K edges) lands at ~3,200-4,500 — inside the 3-6K objective.
- **The cold-compile mechanism is documented:** 13-15 s per new shape at first use (CURRENT_STATE.md);
  if the 4096 bucket is compiled MID-bench (its first use at the 3.5K depth), the rep that pays it
  drags the 3-rep median/mean — the warm protocol removes it (the adopted practice for the decode
  ladder; extending the list is mechanical).
- **No KDA interaction:** 128 divides all proposed bucket sizes (Section 2) — the state-block boundary
  alignment (the G12 caution from the corpus) is satisfied by construction.
- **No interaction with the decode levers:** prefill shapes and decode shapes are separate cache
  entries (different query/row geometry); RA-011/RA-012/Q21 touch the decode path only.

**Unsupported assumptions:** (A1) the dip is padding + cold-compile (the location evidence is strong —
exactly at the 2048→4096 edge, recovering by 6K — but an alternative: a KDA state-chunk alignment
effect at 2048 is ruled out by the divisibility (Section 2); a memory-pressure effect from the 4096
bucket's workspace at first use is possible — the warm list (Lever B) addresses it too, since the
workspace is allocated at warm time, not mid-bench). If the A/B shows no movement, the dip's cause goes
to a new gap (the fallback is a 3.5K-depth device trace — 1 boot).

## 5. Expected performance impact (all ESTIMATED)

| Depth | Current (MEASURED) | After (ESTIMATED) | Mechanism |
|---|---|---|---|
| 2.1K (worst current edge, NEW bench depth) | ~2,600-2,800 (extrapolated into the dip) | ~3,200-3,900 | padding 49%→18% (Lever A) + no cold compile (Lever B) |
| 3.5K | 2,538-2,787 | ~3,200-4,000 | padding 15%→2.4% + no cold compile |
| 4K | (top of the dip band) ~2,787-3,300 | ~3,300-4,500 | at-bucket (no padding at 4096) + no cold compile |
| 0.5-2K | 3,338-5,697 (MET) | unchanged (within noise) | bucketing unchanged below 2048 |
| 6-8K | 3,782-5,663 (MET) | unchanged (within noise) | bucketing unchanged beyond 4096 |

- **Conservative:** +15% in the dip band (Lever B only — the cold compile was the whole dip).
- **Optimistic:** +50-60% (Lever A + B — the padding + the cold compile both real).
- **Objective outcome:** the prefill row of the scorecard becomes MET across the entire 0.5-8K range.

## 6. Risks

| # | Risk | Class | Mitigation |
|---|---|---|---|
| R1 | The 3 extra shapes' HBM/workspace pushes a later allocation (the 25.5 GiB first-reclaimable block is the 8K-row prefill shape — the 2.5-3.5K shapes are smaller, but the TOTAL shape count grows) | memory | Check the pool headroom at boot (the ~28 GB free at decode — the shapes are reclaimed between phases anyway; the prefill workspace is the reclaimable block, not a permanent allocation); if the boot OOMs at warm, drop to 3584-only (the single highest-value shape) |
| R2 | The warm-up time grows (+40-45 s) | operational | One-time per boot; acceptable (the warm protocol already costs minutes) |
| R3 | The dip is NOT padding/cold-compile (A1) | scope | The A/B arms are separated (Lever A vs B) — a null result localizes the cause; the fallback is a 3.5K-depth device trace (1 boot); the scorecard row is annotated with the finding either way |
| R4 | The finer buckets change the decode-side cache (if a prefill bucket shape is shared with a decode shape) | compatibility | The prefill (query-length) and decode (row-count) shapes are distinct cache entries in this stack (the Q04 bucket-pinning work confirmed the separation); the A/B's decode bench (a standard bs=1 + @16 check in the same boot) verifies no decode regression |

## 7. Dependencies

- None (independent of RA-011/012/013, Q21, RA-004 — slot into ANY boot family; ideally share the
  RA-011 boot to avoid an extra boot).
- Read the current bucketing args from the adopted boot script (`kb_staging/boot_*.sh`) and the warm
  client's shape list (first step — the current warm list's contents decide whether Lever B is a
  no-op for the 4096 shape).

## 8. Implementation instructions

1. Read the current config: the lin-bucket query-step args + the warm client's prefill shape list (the
   adopted boot script + the warm client script in `kb_staging/`). Record the current values in the
   experiment record (the baseline for the A/B table).
2. Arm 0 = current (fresh boot).
3. Arm A = 512-step query buckets 2048-4096 (2048/2560/3072/3584/4096) — the bucketing args only.
4. Arm B = Arm A + 2560/3072/3584/4094… 4096 added to the warm client's prefill shape list.
5. Per arm: the prefill ladder (0.5K / 2K / **2.5K** / 3.5K / 4K / 6K / 8K, warm, gen=1, 3 reps/depth,
   the scorecard protocol) + a standard decode check (bs=1 + @16 — R4) + the boot warm log (record
   WHICH shapes compiled mid-bench vs at warm — the cold-compile signature).
6. Record EXP-RA014: the arm table (per-depth t/s), the warm log, the HBM headroom at boot, the decode
   check. Update the scorecard row.

**Diagnostics to capture:** the per-rep times at 3.5K (the cold-compile rep shows a 13-15 s outlier —
its presence/absence is the mechanism proof for Lever B), the boot warm log (shape capture order),
the pool headroom at boot (R1).

## 9. Validation criteria

1. **The objective:** the 3.5-4K band ≥ 3,000 t/s (the objective band's floor) in the best arm; the
   2.5K depth ≥ 3,000 t/s (the new worst-edge case).
2. **No regression:** 0.5-2K and 6-8K unchanged within run-to-run noise; the decode check (bs=1/@16)
   unchanged (R4).
3. **Mechanism (Lever B):** the warm log shows 2560/3072/3584/4096 captured at warm time; NO 13-15 s
   outlier in the mid-bench reps (the cold-compile signature removed).
4. **Mechanism (Lever A):** the per-depth t/s at 2.5K/3.5K recovers toward the bucket-ratio prediction
   (the padding recovery — the 3.5K rep approaches the 3.584K-bucket ideal).
5. A null result (no arm ≥ 3,000) is still a VALID result: record it, open the fallback gap (the 3.5K
   device trace), and annotate the scorecard.

## 10. Novelty assessment

- **Not previously attempted:** no experiment targeted the 3.5-4K dip (checked EXPERIMENTS.md — the dip
  exists only in the scorecard as a noted sub-case; E015-A established the bucketing mechanism but with
  the current 128/2048 step pattern — the 512-step refinement in the 2-4K range is new).
- **Closest prior work and the difference:** E015-A (lin bucketing adoption — the mechanism, at the
  current step pattern); the warm protocol (RESEARCH-001 addendum-9 — for the decode ladder; the
  prefill-shape extension is new). This proposal is a refinement of an adopted mechanism at the one
  edge where it misbehaves, plus an extension of an adopted practice (pre-warming) to the prefill
  shapes.
- **Why it deserves investigation:** it is the ONLY unmet sub-case of the prefill objective (everything
  else in the 0.5-8K range is MET); the mechanism is inferred from the dip's exact location (a strong
  prior); the fix is config-only (1 boot, days, zero correctness risk — R4's decode check covers the
  only interaction); and it closes the scorecard's last open row, which matters for the 1M-context /
  4-HPU story (the prefill capacity at the final deployment should not have a known 30% hole at the
  mid-context edge).
