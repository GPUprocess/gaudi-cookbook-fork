# RA-019 — Round-2 audit + new theories: the mHC-prep kernel, the per-phase tensor cache, and the gather_sum port defects

**Status:** PROPOSED (research architect, 2026-11-10, round 2) · **Parent:** `BOTTLENECK_RESEARCH.md` (rev 1.5) · **Handoff:** `EXTERNAL_RESEARCH_PROPOSALS.md` rev 8
**Round-2 scope:** audit of the implementation agent's work since the round-1 sync (2026-11-09 14:33 — commits through e74cfe2), results + reasons of failure, new strategies/theories with GaudiGrimoire support.

---

## 1. Audit of the work since round 1

| Item | Result (MEASURED) | Reason of failure / status |
|---|---|---|
| D1-3 (timing-only instrument) | WORKS: graphed 2048-bucket prefill 295 ms vs eager 2944-bucket 1,408 ms (4.8×) | The 4.8× = the lazy re-tracing of ~1,800 ops at ~0.7 ms/op; the eager host time IS the wall time |
| AIP-util comparison | graphed 2048-class: 25-33% AIP, UNIFORM across ranks; eager 2944-class: 4-19%, WILDLY UNEVEN | The eager per-rank Python-forward jitter desynchronizes the TP collectives; the graph capture re-locks the ranks (see N4) |
| In-worker torch.profiler | UNUSABLE on this stack | The profile STOP segfaults the worker (kineto flush, exit code None, both export paths) — the D1-2 instrument needs the timing-only fallback or the vendor profiler |
| EXP-PFG+MHC (combined fix) | OOM at the first 2,944-row capture — with FULL KV and with KV 640 | The 2,944-row prefill graph's workspace >39 GB/rank (vs ~18-20 GB at 2,048 rows — SUPERLINEAR scaling); the wedged cards needed driver-level AIP resets |
| ABOTSIS-REF (reference comparison) | The abotsis fork serves the SAME model with prefill graphs past 2048 in ~17 GiB/card TOTAL (VLLM_HPU_PROMPT_GRAPH_MAX_TOKENS=20480, GMU 0.35, 3,200-row cap) | The 3× workspace-accounting gap (their 17 GiB total vs our 36-39 GB for HALF their graph size) was named THE investigation |
| AB3-HYPA (hypothesis a) | REFUTED: the 2,432-row capture (SMALLER than the 2,944 failure) OOMs with the KV 768 | The batched-tokens-sized buffers are NOT the gap — the workspace overhead is STRUCTURAL to our port/v0.26.0 |
| MEMLOG (the allocation profile) | Volume 667.7 GB; peak watermark 101.35 GB (the OOM). Classes: **184-MiB mHC-prep FP32 ×922 = 165.7 GB** ([rows,4,4096] FP32, ~20/layer), 8-MiB Sinkhorn intermediates ×9,929 = 77.6 GB, 128-MiB padded-row FP32 ×766 = 95.8 GB | **THE WORKSPACE HOG CONFIRMED = the mHC-prep FP32 chain** — the existing Sinkhorn kernel covers only the 8-MiB loop class; the prep (the FP32 conversion/mixing around the loop) is still torch |
| **ABCONFIG-WIN (the adopted fix)** | **THE DIP IS FIXED:** tensor cache OFF (`GLM_TENSOR_CACHE_OFF=1`) + prefill threshold 8192 → prefill 5,546-7,723 t/s across 2.0-4.9K (+209-277% vs eager) — the PREFILL OBJECTIVE MET ACROSS 0.5-8K. **Cost: decode bs=1 37.3 (−12% — the RA-001 win forfeited); @16 259.5 PARITY; suite 97.5% (R003 — the 4TH flip source: the tensor-cache state itself)** | ADOPTED AS THE PREFILL-OPTIMIZED PROFILE. The next-session item is the per-phase tensor cache (the decode shapes cached, the prefill shapes not) |
| CRASH-INCIDENT (ops) | The repeated PT_DEVMEM OOMs cascaded into a host-level failure: the journal daemon bloated to tens-of-GB RSS on the crash spam → the card resets failed ("Failed to allocate memory for host-resident page pool" → MMU init failure) → the cards wedged → the reboot hung → two hard power recoveries | The host RAM (125 GB) is a first-class constraint: the driver's reset path allocates a host-resident MMU page pool. Ops lessons recorded (journal monitoring, the reset-failure signature, the escalation order, pre-campaign BMC config) |
| **gather_sum port (RA-018/T1)** | **APPLIED to the venv** (`hpu_communicator.py`: `GLM_ALLREDUCE_GATHER_SUM=1` → `all_gather` + fixed-rank-order sum; default hccl = inert). **The gsum probe returned 0.0% with wall_s 0.0 — an instrumentation artifact** (probed a non-serving server — the current boot started 4.5 min before this audit, mid-load). The real A/B is pending | See §3 — the port as written has THREE deviations from the corpus patch, one of them a known correctness hazard |

**The round-2 bottleneck model (post-ABCONFIG-WIN):** the prefill objective is now MET (5.5-7.7K t/s across 0.5-8K); the new visible gap is **decode bs=1 37.3 (−12%)** — the tensor-cache forfeiture — plus the R003 97.5% (the 4th flip source). The decode stack (37.3) is now BELOW the round-1 adopted stack (42.4) on single-stream; the aggregate (259.5) is unchanged. **The single highest-value visible win is recovering the 5.1 t/s decode regression without losing the prefill fix** — and the structural fix (the mHC-prep kernel) is bigger than that: it removes the OOM class that caused the crash incident and the profile split in the first place.

## 2. The four R003 flip sources (the taxonomy, updated)

1. Q01v2 draft graphs — the captured-region fusion drift (the mode-independent 1/12 golden).
2. RA-011 async replay — NOT a ULP: the in-flight consumption of the cached outputs (a correctness bug; the D2H redesign fixes it).
3. MHC-ROWS (+19% at the 2944 bucket) — the TPC Sinkhorn kernel changes the loop's reduction order vs the torch loop (a real, bounded ULP).
4. **ABCONFIG-WIN (new): the tensor-cache state itself** — turning the cache OFF changes the decode path's allocation/layout pattern → a recipe-numerics shift (the "inherent shape/context numerics" class).

The pattern: **every structural change to the execution path that touches the decode residual stream flips R003.** The gate policy (RA-018/T7) is now more urgent — and the adopted profile itself sits at 97.5%, so the policy must define the BASELINE (the current adopted profile's suite profile = the reference; "regression" = new flips vs it), not assume the historical 100% as the bar.

## 3. The gather_sum port: three defects (RA-021 — fix BEFORE the pending A/B)

The venv port (`hpu_communicator.py::all_reduce`, env-gated `GLM_ALLREDUCE_GATHER_SUM`, verified in the server venv 2026-11-10) replaces `dist.all_reduce(input_)` with `all_gather(input_, dim=0)` + `gathered.reshape(ws, -1, *shape[1:]).sum(dim=0)`. Against the corpus patch (abotsis `LAYER0_DELTA_FINDINGS` §11 + gaudi-llm-notes patch 51) and the DSV4.1 port docs, it deviates in three ways:

1. **No size cap.** The corpus patch falls back to hccl for buffers >16 MiB — because captured-graph sites RETAIN the working buffers. As written, our port would all_gather the prefill ARs: at 8,192 rows the AR input is [8192, 4096] bf16 = 64 MB → the gathered buffer is 512 MB × 162 ARs per prefill step, retained in every captured prefill graph → **guaranteed OOM at the >2,048-row captures** — the exact failure class the CAP-8192/AB3 experiments just recorded. The cap is not optional; it is the difference between "decode ARs invariant, prefill ARs hccl (a wash)" and "prefill OOMs again."
2. **The sum is in the input dtype (bf16).** The corpus sums in fp32 ("all_gather the partials, sum in fixed rank order in fp32"). The rank-order invariance is preserved either way (that is the dominant effect), but the fp32 accumulation is the corpus-validated behavior and is free at decode sizes ([2, 4096] gathered → 16 KB).
3. **The all_gather stale-value hazard (the DSV4.1 warning).** The DSV4.1 port docs (gaudi-cookbook, tier D) record: *"vllm-gaudi's `all_gather` calls `mark_step`; inside a captured HPU graph that gave stale values for batches above 1. An all-reduce replaced it."* The gather_sum port makes every AR an all_gather — inside captured decode graphs, at every batch. The abotsis fork runs gather_sum in production on Gaudi2 (12/12 parity) on v0.27.2rc0 — so either their all_gather fixed the hazard or their version doesn't have it. **Our v0.26.0 (b059f9c) is the DSV4.1-era all_gather.** The pending A/B MUST include the concurrency>1 parity: the 8-identical-prompt rowdep probe is exactly the detector (stale all_gather values would show as divergent/garbage outputs at batch >1, not just at bs=1), plus the suite run at concurrency. If the hazard is live, the fix options are (a) the all_gather without the mark_step inside the captured region (a venv patch to the `all_gather` method — the mark_step is there for the lazy-collective workaround; the corpus patch presumably handled it), or (b) keep the AR but change its REDUCTION ORDER (a custom collective op in the TPC/HCCL layer — heavier).

**RA-021 (the hardening, a venv patch + the A/B protocol):** (1) add the size cap (the corpus 16 MiB default, env-tunable; prefill ARs fall back to hccl — the prefill first-token parity degrades to today's behavior, a wash); (2) fp32 sum for the in-cap path; (3) before enabling `GLM_ALLREDUCE_GATHER_SUM=1` in any boot: the rowdep probe (8 identical prompts, 1 distinct text expected) + a bs=8 parity check (the stale-value detector); (4) the A/B on the CURRENT adopted profile (cache-off, prefill 8192): suite (prediction below), decode parity bs=1/@16 (expect parity — the corpus measured 14.7 vs 13.6 = restart noise), the prefill ladder (the cap behavior: the >16 MiB ARs stay hccl). **Prediction:** on the current cache-off profile, gather_sum may NOT move R003 — the 4th flip source says the current flip is the cache-state class, orthogonal to the AR row order. The full 100% may require gather_sum + the per-phase cache (RA-020, which restores the decode retention state of the 100% profile). The A/B result either way is the empirical test of the T1 theory on our tree: R003 fixed → the row-order source is live here; R003 unchanged → the sources are (at least) the cache-state class, and T7's gate policy is the remaining unlock.

## 4. New theories

### RA-019 — The full mHC-prep TPC kernel (the structural fix for the workspace OOM class)

**Hypothesis:** the prefill-graph OOMs (CAP-8192, PFG+MHC, AB3) and the adopted profile's decode regression (−12%) share ONE root cause: the mHC-prep chain runs in torch and materializes ~20 × [rows, 4, 4096] FP32 tensors per layer (the 184-MiB MEMLOG class, 165.7 GB of the 667.7 GB capture volume) + the 128-MiB padded-row class. A TPC/MME kernel that fuses the ENTIRE mhc_pre chain (the FP32 prep + the 20-iteration Sinkhorn + the epilogue) into one program — the DSV4.1 `control_mme` family pattern, in the corpus and on this host — removes the materialization at the source: the capture workspace collapses (the agent's own estimate: 101 GB peak → ~20-30 GB), the >2,048-row prefill graphs become capturable WITH the tensor cache ON and the FULL KV pool, and the decode regression is recovered WITHOUT the profile split.

**Evidence:** (1) MEMLOG (the allocation profile — the hog is named and quantified; the existing kernel covers only the 8-MiB loop class); (2) the port's own `mhc.py` docstring: "90 `mhc_pre` calls per decode step... The wrapper repeats the stock ops in the stock order and swaps only the loop" — the prep is still torch, by construction; (3) the DSV4.1 control-op family (1Cat fork, tier D, in the corpus: `deepseek_v41_control_mme_f32`, `deepseek_v41_control_rrms_unpack/parallel/swizzled/bf16_weight`, `deepseek_v41_control_gemv`, `deepseek_v41_mhc_gates_f32/positive`, `deepseek_v4_sinkhorn4` — the mHC branch as a few custom ops, PT2-registered — `control_mme.cpp` includes the same PT2 pattern the agent's AB-DEQ dequant op already runs in production); (4) the DSV4.1 port lesson: "a 4x4 Sinkhorn loop cost about 3,400 nodes per step, and a tiny TPC kernel removed them bit-exactly" — the node-count win generalizes to the prep ops; (5) the existing GLM infrastructure: `GC_KERNEL_PATH` registration, the op lib (the B7 load-order recipe), the MAX_ROWS env (the MHC-ROWS experiment proved the kernel runs at 8192 rows).

**Expected (ESTIMATED):** (a) the capture workspace 101 GB peak → ~20-30 GB (the agent's estimate; my refinement: the 184-MiB class is 165.7 GB of the 667.7 GB volume — the peak concurrent is a handful of the ~193 MB blocks, so the peak drop is ~15-25 GB, and the 128-MiB padded class drops if the kernel absorbs the padded-row prep too — the memlog attribution decides); (b) prefill graphs >2,048 rows capturable with cache ON + full KV → decode back to 42.4 (the RA-001 win) AND the prefill fix kept — both wins, one profile; (c) the 2,944/4,096-bucket prefill speedup extends beyond MHC-ROWS's +19% (the full prep ops leave the graph — D1-3 says the eager re-tracing is ~1,800 ops; the captured graph with the kernel has a fraction of the nodes → faster replay too); (d) the 4-HPU HBM headroom (the workspace is part of the 4-HPU budget — 87.5 GB used at Q14v2); (e) the OOM-crash class (the CRASH-INCIDENT trigger) shrinks — fewer large transient allocations, no capture-time watermarks past the budget.

**Complexity:** MEDIUM (weeks — the kernel work; the pattern, the build system, the op registration, and the load-order recipe all exist). **Risks:** the kernel must be bit-identical or the ULP class expands (the existing kernel is bit-identical by construction — "the device's own reduction order and divide"; the prep ops' FP32 arithmetic must be preserved exactly — the DSV4.1 control ops carry explicit f32 variants for exactly this reason); the >1024-row path must be re-validated (the current kernel is capped at MAX_ROWS — the MHC-ROWS experiment raised it to 8192 with the +19% signal, so the row scaling is known-good); the 128-MiB padded-row class may be a DIFFERENT op (the memlog attribution — the agent's "THE WORKSPACE HOG CONFIRMED = the mHC prep FP32 chain" covers it, but the kernel scope should be set from the attribution, not assumed).

**Validation:** the memlog re-capture at the 2,944-row shape (the 184-MiB class gone; the peak watermark < 40 GB); the capture with cache ON + full KV (the OOM gone); the prefill ladder (≥ the ABCONFIG-WIN numbers); the decode bench (42.4 recovered); the suite (bit-identical expected — the kernel is in the decode path only above MAX_ROWS... it replaces the torch prep in BOTH paths — the suite A/B decides); the 4-HPU boot (the headroom).

### RA-020 — The per-phase tensor cache (the quick decode recovery; days)

**Hypothesis:** the vendor wrapper reads `PT_HPUGRAPH_DISABLE_TENSOR_CACHE` **on every call** (tier B, `wrapped_hpugraph_forward` in the 1.24.x bridge: `env_tensor_cache = os.environ.get(...)` inside the per-call path, not at wrap time). The runner knows the phase (prompt vs decode) at every call. So the staged `GLM_TENSOR_CACHE_OFF` patch (currently global-OFF) becomes per-phase: decode calls set the env to "0" (cache ON — the decode shapes retain their tensors, the RA-001 win), prefill calls set it to "1" (cache OFF — the prefill tensors free after replay, the ABCONFIG-WIN fix). The prefill graphs are still captured and replayed (the cache-OFF capture is the dry-run mode that ABCONFIG-WIN already proved works at +209-277%); only the retention differs.

**Expected (ESTIMATED):** decode bs=1 back to ~42.4 (+12% over the current 37.3 — the RA-001 win recovered); prefill unchanged (the ABCONFIG-WIN numbers); @16 ~259.5 (the decode path at @16 also gains the cache — expect parity-to-slightly-better). **Falsifiable accuracy prediction:** the decode path's retention state is bit-identical to the pre-ABCONFIG-WIN adopted profile (which ran 100% on the suite) — the per-phase config restores exactly that state for the decode shapes → **the suite should return to 100% (R003 unflips)**. If it doesn't, the 4th flip source is deeper than the retention state (a new data point for T7 either way).

**Complexity:** LOW (the 3-line change to the staged patch: set the env per call in the runner's execute_model where the phase is already computed — the `use_graphs`/`phase` site at l.3737; the env write is process-global but the call is single-threaded per worker — safe). **Risks:** the phase must be determined BEFORE the wrapped forward call (the env is read inside the wrapper); the MTP verify shapes are decode-class (they must take the cache-ON path — they do, they are decode calls); a mixed prompt+decode step (chunked prefill) takes the prefill env for that step (the decode shapes in the same step replay with cache-OFF for that one call — a transient, not a state change — the next pure-decode call restores the cache-ON replay... actually the cache ENTRY is per-shape and the retention policy applies at replay time — the decode entry, replayed during a chunked-prefill step with env=1, frees its tensors that step; the next decode step re-caches. The cost: one re-cache per mixed step — the chunked-prefill regime is rare at decode batch sizes; measure it in the A/B).

**Validation:** the decode bench (42.4 ± noise), the prefill ladder (the ABCONFIG-WIN numbers), the suite (100% predicted — the R003 unflip), the HBM per card (the decode retention is back, the prefill retention is gone — expect ~66-68 GB at decode, the pre-ABCONFIG level), the chunked-prefill A/B (the mixed-step re-cache cost).

**Sequencing:** RA-020 first (days — it recovers the −12% now and is a strict improvement over both current profiles), RA-019 second (weeks — the structural fix that subsumes RA-020: with the workspace collapsed, the cache can stay ON for everything and the profile split disappears). RA-020 is not wasted work if RA-019 lands (the per-phase toggle stays as the HBM-budget knob for the 4-HPU config).

### RA-021 — The gather_sum hardening + A/B protocol (see §3)

Rank: FIRST (it is a prerequisite for the pending A/B — the port is already in the venv and the server is mid-boot; the three defects (the missing size cap, the bf16 sum, the all_gather stale-value hazard) must be fixed before `GLM_ALLREDUCE_GATHER_SUM=1` is enabled in any boot — the size-cap absence is an OOM waiting to happen at the prefill captures, and the stale-value hazard is a correctness risk at batch >1). Effort: hours (the venv patch) + the existing A/B protocol.

### N4 (finding) — The eager prefill rank-desync mechanism: graph capture is a synchronization mechanism

The AIP-util comparison (the hl-smi sampling under sustained load) shows the dip has TWO mechanisms, not one: (1) the 4.8× host dispatch (D1-3: the ~1,800-op re-tracing), and (2) the **rank desync** — the eager 2,944-class runs at 4-19% AIP WILDLY UNEVEN across the 8 ranks (the per-rank Python-forward jitter), while the graphed 2,048-class runs at 25-33% UNIFORM. The eager ranks arrive at the 162 per-step AR boundaries at different times; the fast ranks spin-wait at the collectives. The device is idle not just because the host is slow, but because the ranks are desynchronized. Implications: (a) **capture-first is the fix hierarchy for every hot path** — the >8,192-token prefills (still eager under the current threshold) and the 4-HPU config (TP4 — 4 ranks, smaller desync surface, but the same mechanism) carry the same penalty; (b) the **MTP draft (eager, every step) has the same signature at the AR boundaries** — the draft's device time is partly AR spin-wait, and the T2 chain graph (RA-018) re-locks the ranks there as a SECONDARY win beyond the host dispatch saving (a new rationale for T2, measurable as the AIP-uniformity improvement under the chain graph); (c) the desync explains why the fix gain (+209-277%) EXCEEDS the dispatch saving alone (the 4.8× host ratio would predict less) — the re-locking is the excess.

### N5 (finding) — The Q21 k=4 expert-union sizing: the bs=1 k=4 weight floor is 3-5× the k=1 floor

BG-2's histogram (22M routes) gives the per-request expert coverage (168-182 of 288 at prefill scale; near-uniform, max/min 8×). At bs=1 k=4 the 9 rows (5 verify + 4 draft) are the SAME request's consecutive positions — strongly correlated routing (the same context). The 9-row union is far below the 9×8=72 coupon-collector bound: estimated 40-60 distinct experts globally per layer → 5-8 per rank (36 local, the contiguous sharding) → **the k=4 bs=1 Q21 weight floor ≈ 5.3-8.5 GB/card/step → 2.2-3.5 ms at 2.45 TB/s** (vs 0.86 ms at k=1 — the 2.1 GB figure). The end-state physics (RA-018 §2.6) holds (the 8-12 ms step envelope) but the device term at k=4 is the 2.2-3.5 ms weight floor, not the 0.86 — the Q21 kernel's k=4 design must target that. **The exact number is a no-HPU measurement:** a one-line hook in the CPU router that archives the per-row top-8 sets during a 100-request decode run (the 22M-route npz is per-REQUEST, not per-ROW — the decode hook is the missing piece); compute the 9-row union per layer from the archive.

## 5. Updated execution order (round 2, supersedes the round-1 order)

1. **RA-021 — the gather_sum hardening** (hours, the venv patch: the 16 MiB size cap + the fp32 sum) BEFORE the pending A/B enables `GLM_ALLREDUCE_GATHER_SUM=1`; the A/B protocol adds the rowdep probe + the bs=8 parity (the stale-value detector) as pre-enabling gates.
2. **RA-020 — the per-phase tensor cache** (days; the 3-line change to the staged patch; recovers the −12% decode NOW; the suite prediction is 100% — the R003 unflip).
3. **Q21 continues** (the fused MoE GEMV — unchanged; add the N5 per-row route hook to the next boot — no HPU time for the union measurement).
4. **RA-019 — the full mHC-prep TPC kernel** (weeks; the structural fix: the OOM class, the profile split, the 4-HPU headroom; the DSV4.1 control_mme family is the pattern, the source is on this host).
5. **RA-018/T2 — the chain graph** (after the gather_sum A/B result — the accuracy regime first; the N4 desync re-lock is the secondary win to measure).
6. **RA-018/T7 — the gate policy** (more urgent after the 4th flip source; add the baseline definition: the current adopted profile's suite profile = the reference).
7. **Ops (CRASH-INCIDENT lessons, cheap, now):** the systemd journal rate-limit (RateLimitIntervalSec/RateLimitBurst or the journal vacuum policy) + the host-RAM headroom check in the pre-flight (the driver reset path needs the host-resident page pool; the 125 GB host with 8×~7 GB workers + the crash spam has no margin).
8. RA-009/016/RA-018-T4 (the drafter scoping) — unchanged, parallel.

## 6. Evidence index (round 2)

**Project data (MEASURED):** the commit log e74cfe2..b101646 (the round-2 arc); `docs/current_bottleneck.md` (the AB3-HYPA/ABOTSIS-REF/PFG+MHC/ABCONFIG-WIN/MEMLOG/CRASH-INCIDENT entries + the D1-3/AIP-util findings); `benchmarks/accuracy/results/gsum_probe.json` (the 0.0% artifact), `abotsis_cacheoff.json` (97.5% ×3), `memlog_pfg_rank0.log` (the 654,829-line allocation profile); the venv inspection 2026-11-10 (`hpu_communicator.py` the gather_sum port, `hpu_model_runner.py` l.1131 `GLM_TENSOR_CACHE_OFF` + l.5212 `GLM_PREFILL_GRAPH_MAX_TOKENS` + the l.3737 phase site, `hpu_worker.py` l.410 `VLLM_GRAPH_RESERVED_MEM` — the KV sizing only, no graph cap); the port `mhc.py` (the 90 mhc_pre calls, the loop-only kernel, the MAX_ROWS cap, the bit-identical construction); the live server state (mid-boot at audit time; the journal healthy at 32 MB; the host RAM 57 GB used / 68 GB available — no current bloat).
**GaudiGrimoire (this round's queries):** (1) DSV4.1 mHC control family — `1cat-vllm-gaudi` `vllm_gaudi/compilation/deepseek_v41_overlap.py` (`independent_mhc_nodes`/`split_mhc_consumers` — the mHC branch extraction; the control-op list: `control_mme_f32`, `control_rrms_unpack/parallel/swizzled/bf16_weight`, `control_gemv`, `control_batch4_f32`, `control_prefetch_f32`, `mhc_gates_f32/positive`, `deepseek_v4_sinkhorn4`) + `csrc/deepseek_v41_unique/pytorch/control_mme.cpp` (includes `hpu_dsv41_control_mme_pt2.cpp` — the PT2 registration the AB-DEQ op already uses) — tier D; (2) the DSV4.1 port lessons — gaudi-cookbook `docs/ports/deepseek-v4-1-flash.md`: the 3,400-node Sinkhorn removal, **the all_gather stale-value hazard in captured HPU graphs at batches >1**, the graph-cache view-state re-capture — tier D; (3) the tensor cache mechanism — gaudi-pytorch-bridge `graphs.py::wrapped_hpugraph_forward` (tier B): the `PT_HPUGRAPH_DISABLE_TENSOR_CACHE` env is read PER CALL; the cache-OFF capture is the dry-run mode; the tensors are "replaced with empty tensors after replaying"; (4) the stock graph-memory management — vllm-fork `README_GAUDI.md` + habana-docs: `VLLM_GRAPH_RESERVED_MEM` (default 0.1 — 10% of usable memory reserved for graphs vs KV) + `VLLM_GRAPH_PROMPT_RATIO` (default 0.5 — the prefill/decode graph memory split) — **present in our v0.26.0 (`hpu_worker.py` l.410, verified) but a KV-sizing budget only — it does NOT cap graph capture** (the OOM watermarks prove the captures run past any reservation).
**DERIVED:** the N5 k=4 union estimate (40-60 distinct global, correlated rows — the coupon-collector bound with the correlation assumption); the RA-020 suite prediction (100% — the retention-state argument); the RA-019 peak-watermark refinement (the 184-MiB class is 165.7 GB of 667.7 GB volume; the peak concurrent is a handful of blocks → ~15-25 GB peak drop, the agent's 20-30 GB estimate is the working figure pending the memlog attribution of the 128-MiB class).

---

## 7. Post-audit update (2026-11-10 ~04:40 — the overnight results landed while this note was being written)

The implementation agent ran the remaining round-2 experiments overnight (boot sequence:
perphase 23:41 → restored 01:37 → gsum 03:13 → final 04:00 → k4 04:13 → k4b 04:20 in
progress). Two commits landed after this note's audit (ef7aad6 GSUM-T1, 6f45c1d T1.2):

1. **GSUM-T1 (T1.1 A/B) — REFUTED as the unlock, NEUTRAL otherwise.** The rowdep probe (4
   batch positions + the numeric-fillers batch) shows NO R003 flip on the HCCL BASELINE —
   the position-invariance holds on BOTH configs → the row-order ULP mechanism (the corpus
   G14 theory) is REFUTED as the R003 flip cause on our tree. The gather_sum arm: R003 STILL
   flips (suite 97.5%) with decode PARITY (43.1/43.7 — the corpus no-cost claim confirmed on
   our stack). The flip is re-characterized as CONFIG-NUMERICS SENSITIVITY — the 5th flip
   source: the AR order itself (the gather_sum change, even with the fixed-rank-order sum,
   shifts the residual stream enough to flip the knife-edge item). Consequences: T1.3
   (captured-core alignment) is downgraded to whack-a-mole; **T7 (the gate policy) is now the
   primary unlock** for the three banked optimizations (Q01v2 +20%, RA-011 +29%, MHC-ROWS
   +19%). The DSV4.1 all_gather stale-value hazard (RA-021 defect 3) was TESTED at batch >1
   (the rowdep batches + the suite under gather_sum) and did NOT manifest on our v0.26.0 —
   consistent with the abotsis production evidence. RA-021's size-cap caution remains valid
   for the ONE future case: gather_sum re-enabled on a cache-ON >2,048-row prefill profile
   (the RA-019 end-state) — the retained gathered buffers (512 MB × 162 at 8,192 rows) would
   OOM; on the current cache-off-prefill profile the buffers are freed per replay (no cap
   needed).
2. **T1.2 (pinned decode buckets) — DEFERRED as the predictable null** (the rowdep probe
   already refuted the shape-ladder mechanism: the positions map to different decode-bucket
   shapes and the invariance held). Secondary value only (the graph count, the warmup) for a
   separate adoption decision.
3. **The per-phase tensor cache (RA-020) — ATTEMPTED, boot failed on infra, and the deployed
   patch has three defects** (the implementation agent deployed their own mechanism: the
   runner sets `self._glm_no_tensor_cache_this_call =
   scheduler_output.total_num_scheduled_tokens > 0` at `hpu_model_runner.py:4266`; the
   patched vendor wrapper ORs it into `mark_user_inputs` at `graphs.py:706`):
   - **D1 (the phase signal):** `total_num_scheduled_tokens = sum(num_scheduled_tokens.values())`
     (`vllm/v1/core/sched/scheduler.py:1065` — verified in our tree) includes the DECODE
     requests (each ≥1; MTP verify ≥1+k/request) → the flag is True on EVERY non-empty step →
     the "per-phase" boot would run cache-OFF on all calls (the ABCONFIG-WIN behavior,
     decode 37.3), not the claimed both-wins.
   - **D2 (the lookup chain):** the flag is set on the `HpuModelRunner` but read from
     `orig_fwd.__self__` = the `HpuModelAdapter` (`orig_fwd = module.forward`, graphs.py:889;
     the adapter has no `__getattr__` proxy — verified) → `getattr(..., False)` → the flag is
     **always False** → the patch is INERT → the boot runs cache-ON everywhere → OOM at the
     2,944-row capture (the exact PFG+MHC failure — the per-phase boot config is PFG+MHC
     minus the mHC-rows change, which does not reduce the workspace).
   - **D3 (the capture mode):** `dry_run = disable_tensor_cache if dry_run is None else
     dry_run` (graphs.py:699-700) — the capture's allocation mode derives from the
     WRAP-TIME flag (+ the env override), NOT the per-call flag. The MEMLOG OOM class is the
     capture-time/retained INTERMEDIATE workspace (the 184-MiB [rows,4,4096] FP32); the
     vendor's own cache-OFF path does dry_run=True + mark_user_inputs — the deployed patch
     does only mark_user_inputs (the input-retention half). The >2,048-row captures would
     still OOM with it.
   - **The boot itself failed before any of this was testable:** `synStatus=26 [Generic
     failure] Device acquire failed` at worker init (wedged cards — the CRASH-INCIDENT-class
     infra failure, not a patch fault).
   - **Predicted outcome of `boot_perphase.sh` as deployed:** OOM at the 2,944-row capture
     (most likely, via D2) or decode 37.3 (via D1) — NOT both-wins. **Do not re-run as-is.**
4. **RA-020 CORRECTED IMPLEMENTATION (no vendor patch needed):** the vendor wrapper reads
   `PT_HPUGRAPH_DISABLE_TENSOR_CACHE` on EVERY call (graphs.py:697 — tier B, verified this
   round, G20) and that per-call value drives BOTH the disable_tensor_cache semantics AND
   the derived dry_run (l.699-700) — the full vendor cache-OFF path per call. The entire
   fix is in the RUNNER: at the l.4266 site, replace the attribute with
   `os.environ["PT_HPUGRAPH_DISABLE_TENSOR_CACHE"] = "1" if
   scheduler_output.total_num_scheduled_tokens > 2048 else "0"` (set before the wrapped
   forward; the env write is process-global, the call single-threaded per worker — safe),
   and REVERT the graphs.py:706 patch (unnecessary via the env route). The signal is the
   ROW-COUNT CLASS, not the phase: the demonstrated OOM boundary is between 2,048 rows
   (captured fine cache-ON) and 2,944 (OOM); the decode class is ≤ ~64 rows at ns32 (2
   rows/request × 32) even under MTP k=1; the 2,048-row prefill class was fine cache-ON.
   Expectation: bs=1 ~42.4 (+12%), prefill unchanged (the ABCONFIG-WIN ladder), and the
   FALSIFIABLE prediction: the decode retention state becomes bit-identical to the
   pre-ABCONFIG-WIN 100% profile → **the suite returns to 100% (the R003 unflip)**; a miss
   is a T7 data point (the flip is deeper than retention), not a RA-020 failure.
5. **k4b (the MTP k=4 experiment) — in progress at update time** (weights loading). The
   deliverables: the acceptance vs the 3.33/5 reference (±10% calibrates the stack delta —
   the RA-004 T0 probe's stated purpose), the candidate-slot capture log (the T5 HBM budget),
   the decode bs=1/@16.
6. **Updated do-first order (supersedes §5 above; the rev-8 handoff carries the same order):**
   1. let k4b finish; 2. RA-020 revised (the runner one-liner + the graphs.py revert + one
   boot family — the 100% prediction is the gate); 3. T7 (the gate policy + the baseline
   definition — the current adopted profile's suite profile = the reference; the adopted
   profile itself is at 97.5%); 4. RA-019 (the full mHC-prep TPC kernel — weeks); 5. Q21
   continues (+ the N5 per-row route hook); 6. T2 (the chain graph, after the k4b data; the
   N4 desync re-lock is the secondary win to measure); 7. ops (the journal rate-limit + the
   host-RAM headroom pre-flight); 8. the rest (BG-1, RA-012A, T6, RA-009/016/T4, RA-017 P0,
   RA-014 subsumed).
