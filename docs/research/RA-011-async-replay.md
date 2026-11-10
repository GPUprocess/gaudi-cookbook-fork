# RA-011 — Asynchronous HPU-graph replay with the vendor launch thread pool

**Status:** PROPOSED (research architect, 2026-11-09) · **Priority:** 1 · **Complexity:** Low change / Medium verification
**Parent:** `BOTTLENECK_RESEARCH.md` (Section 3, RA-011) · **Targets:** B2, B3, the host/device serialization (C7)
**Companion:** `docs/research/EXTERNAL_RESEARCH_PROPOSALS.md` RA-001 (tensor cache — precondition, adopted)

---

## 1. Executive summary

The decode step is executed as: host builds inputs → host replays the 322-recipe HPU-graph chain
**synchronously** (blocking in the vendor C++ per-recipe loop, 14 ms bs=1 / 42 ms @16) → host runs the
sampler + eager MTP draft dispatch + metadata → next step. The measured step time equals host + device
with ~zero overlap (45.1 ≈ 24 + 21 bs=1; 114 ≈ 61 + 53 @16). The vendor bridge already contains an
offload path for exactly this: `replay(..., asynchronous=True)` + `PT_HPU_ENABLE_HPUGRAPH_THREAD=1` +
`PT_HPU_QUEUE_SYNLAUNCHES=1` moves the entire recipe launch loop onto the vendor's internal thread pool,
and `PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK=1` skips the per-replay input-hash recomputation. This
proposal is a one-line wrapper change plus three env vars, with a strict same-boot A/B + parity + soak +
trace verification protocol. Estimated: bs=1 +50-100%, @16 +70-85% (all ESTIMATED until the A/B lands).

## 2. Current limitation (evidence)

| Fact | Value | Source |
|---|---|---|
| `wrap_in_hpu_graph` call in the deployed tree | `htorch.hpu.wrap_in_hpu_graph(HpuModelAdapter(...), disable_tensor_cache=False)` — **no `asynchronous` argument → vendor default `False` (synchronous replay)** | `vllm_gaudi/v1/worker/hpu_model_runner.py:1118-1120` (deployed tree, read on g2); the wrapper's signature `wrap_in_hpu_graph(..., asynchronous=False, ...)` in `habana_frameworks/torch/hpu/graphs.py` (deployed venv, read on g2) |
| Host time in `execute_model` (D1, py-spy 200×5ms) | `torch/hpu/graphs.py:75 replay` **93.3%** of the bucket; staging ~1.6%; draft ops ~21% of the remainder; sampler 6-8 ms; meta ~2 ms | D1 (rev-3 doc) |
| What graphs.py:75 actually is | a single native call: `_hpu_C.replay(self.hpu_graph, [], asynchronous)` — the 14/42 ms is inside vendor C++ `SingleHPUGraph::replay` → `replayGraph` → `HbLazyTensor::ExecuteCachedGraph` (322 regions × ~43 µs) | deployed venv `graphs.py` (read on g2); GaudiGrimoire tier B `HPUGraph.cpp` |
| Vendor thread-pool branch | `bool queue_in_thread_pool = async && GET_ENV_FLAG_NEW(PT_HPU_ENABLE_HPUGRAPH_THREAD) && GET_ENV_FLAG_NEW(PT_HPU_QUEUE_SYNLAUNCHES);` → `ExecuteCachedGraph` enqueued on `SingleTonExecThreadPool`; in this branch the output tensors are NOT nulled (tensor-cache-friendly) | GaudiGrimoire tier B `backend/habana_device/HPUGraph.cpp` (read directly) |
| Vendor docs on the flag | "If True, replays for the HPU Graphs will be launched asynchronously… This flag can be used along with stream and async D2H to copy outputs for reducing host overheads" | docs.habana.ai HPU Graph APIs (tier B) |
| The serialization fact | step 45.1 ms bs=1 = host 24 + device ~21 (DERIVED; sum reproduces the MEASURED step to 0.1 ms → ~zero overlap); @16: 61 + 53 ≈ 114 | D1 + rev-3 device model (C7) |
| Runner design intent | "Everything is done asynchronously - the only sync point is the place where we copy the generated tokens back to the host" | `hpu_model_runner.py:4381` comment (deployed tree) |
| Tensor cache (the precondition) | adopted (RA-001, +10.7%); the replay path is `copy_to(cached.graph_inputs, inputs)` (stream-ordered H2D) → `replay(asynchronous)` → returns cached `graph_outputs`; cache keyed by `input_hash(inputs)` | D1; deployed venv `graphs.py:635-770` |
| In-graph collectives are the supported path | `PT_HPU_ENABLE_LAZY_COLLECTIVES` "enables usage of collective operations (mostly allgather and allreduce in vLLM's case) in HPUGraphs… required for using tensor parallelism with HPUGraphs" (auto-enabled, `platform.py:432`) | GaudiGrimoire tier D (kzawora-intel) + companion doc rev 4 |

**The gap:** the wrapper is called without `asynchronous`, no thread-pool env vars are set, and the
hashcheck env is at its default. Every decode step pays the full host launch loop on the critical path
AND blocks on the device before any host post-work starts.

## 3. Proposed architecture

Execution flow after the change (per decode step, per rank):

```
host thread:                      device stream:
build inputs (meta)
copy_to(cached.graph_inputs)  ──► H2D staging (stream-ordered)
replay(async=True) ────────────► enqueue: thread pool runs the 322-recipe
  (returns after enqueue;                    ExecuteCachedGraph loop (322 × 43µs)
   the C++ loop runs in the                  on the stream; HCCL/AR work
   vendor pool thread)                       executes in stream order
sampler host work              ──► (overlaps device execution of step N)
eager draft dispatch           ──► draft ops enqueued on stream (overlaps)
metadata work                  ──► (overlaps)
token D2H (the designed sync)  ◄── step N device done
next step
```

Step time → `max(host_residual, device) + sync_ε` instead of `host + device`.

**Change list:**
1. `hpu_model_runner.py:1118-1120` (`_maybe_wrap_in_hpu_graph`): add `asynchronous=True` to the
   `wrap_in_hpu_graph` call, env-gated (`GLM_HPU_GRAPH_ASYNC=1`) for the A/B. Apply to the MAIN-model
   wrap (l.5021) only in iteration 1; leave the drafter wrap (l.5060) synchronous (the draft is eager
   under Q01/RA-004 anyway; graphing it is a separate item).
2. Serving boot env: `PT_HPU_ENABLE_HPUGRAPH_THREAD=1`, `PT_HPU_QUEUE_SYNLAUNCHES=1`.
3. Arm 3 only: `PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK=1` (justified: the graph-cache key
   `input_hash(inputs)` is already computed per step for the cache lookup; the inner
   `RecipeArgumentSpec` recomputation per region is the part this skips — stable shapes/layouts per
   bucket make it the intended use).
4. No runner logic changes. Ordering is preserved: the staging H2D is stream-ordered before the
   replayed work; the eager draft ops enqueue after the verify work on the same stream; the token D2H
   is the step-boundary sync (the runner's designed sync point).

## 4. Technical feasibility (why it should work)

- The mechanism is READ FROM THE VENDOR SOURCE (tier B), not inferred: the `replayGraph` thread-pool
  branch exists in the 1.24.1 bridge on g2; the env var names are exact.
- The vendor documents the flag for host-overhead reduction (tier B docs).
- The D1 profile shows exactly what gets overlapped: 24 ms of host (of which the replay loop is the
  bulk) serialized against ~21 ms of device.
- The tensor-cache interaction is analyzed (Section 6 risk R1) and is safe by stream ordering: the
  staging `copy_to` is an H2D into the cached device tensors enqueued on the stream AFTER step N's work
  — step N+1's staging cannot overwrite tensors step N still reads, because both are stream-ordered.
  (The unsafe pattern would be a host write into mapped device memory — not what this path does;
  verified in `graphs.py:635-770`.)
- In-graph collectives: the async flag changes WHEN the launch loop runs (pool thread), not the HCCL
  semantics — the ARs still execute on the device in stream order (G7: lazy collectives are the
  supported TP path).
- The corpus has no counter-example of this exact configuration (async replay + tensor cache + in-graph
  collectives on the 1.24.1 bridge); the cautionary corpus entries involve DIFFERENT mechanisms (the
  engine-level `--async-scheduling` — RJ-18 — and compile-mode AR-split — RJ-13). Their failure
  signatures (hang at first request; stale values at batch>1; HBM growth) are included in the
  verification watch list.

**Unsupported assumptions (explicit):**
- A1: the thread-pool branch handles 322 regions × 8 ranks concurrently without pool starvation
  (the pool is per-device; 8 ranks on 1 machine share the process — the pool sizing is vendor-internal;
  the A/B measures it).
- A2: `SetExecutionInProgress`/`SetTensorDataNullOpt` bookkeeping from the pool thread is race-free
  against the host thread preparing the next step (vendor-shipped path; untested in the corpus for our
  config — BG-5 vendor question).
- A3: the hashcheck env interacts cleanly with the per-step `input_hash` cache lookup (the outer key is
  still computed; only the inner per-region spec recompute is skipped).

## 5. Expected performance impact

All ESTIMATED (the A/B converts them to MEASURED; BG-3 records the actual overlap ratio).

| Metric | Current (MEASURED/DERIVED) | After RA-011 (ESTIMATED) | Δ |
|---|---|---|---|
| bs=1 step | 45.1 ms | max(host ~4, device ~21-28) + ε ≈ 22-29 ms | **−35 to −50 ms→22-29: +50-100% t/s (42.4 → ~65-85)** |
| @16 step (8 HPU) | 114 ms | max(host ~10, device ~53) + ε ≈ 55-60 ms | **+70-85% (257 → ~430-470 t/s)** |
| @16 step (4 HPU) | 182.7 ms | max(~10, ~85) + ε ≈ 88-95 ms | **+65-75% (153 → ~255-285 t/s)** |
| bs=1 (4 HPU) | 33.3 t/s | ≈ 50-65 t/s (host term rank-local) | +50-95% |
| Host post-work visibility | serialized | sampler/draft/meta (~12 ms bs=1, ~20 ms @16) overlap device | the mechanism proof (trace) |

- **Conservative:** +30% bs=1 / +40% @16 (thread-pool enqueue re-adds some host cost; partial overlap).
- **Best case:** +100% bs=1 / +85% @16 (full overlap + the eager draft's device work extending the
  tail productively).
- **Worst case (still acceptable):** parity (the pool path serializes internally) — in that case the
  env-var-only variant is reverted at zero cost and the hashcheck env + RA-012's loop-shrinking remain.
- Composition: with Q21 (device −6 ms @16) and RA-012 (device −3-5, host loop −30-50%), the Section 7
  combined model of the parent doc reaches 800@16 inside the envelope. RA-011 is the largest single
  contributor to that model.

## 6. Risks

| # | Risk | Class | Mitigation |
|---|---|---|---|
| R1 | Tensor-cache race: step N+1's staging overwrites inputs step N still reads | correctness | Stream ordering makes it safe (Section 4); gate: greedy bit-identity vs arm 0 on the accuracy-suite subset (same-boot protocol). Any divergence → stop, bisect (wrap-only vs wrap+env vars) |
| R2 | Vendor-internal race in the thread-pool bookkeeping (`SetExecutionInProgress` from the pool thread vs host prep) | stability | Soak: 30-60 min at 16 streams; watch for hangs, garbage/`!!!!`, HBM growth >1 GB, graph-count steps (re-capture storm — the DSV4.1 view-state signature); BG-5 vendor question in parallel |
| R3 | Stale-value pattern at batch>1 (the DSV4.1 all_gather-stale lesson) | correctness | The A/B bench at @16 and @32 is part of the gate (not just bs=1); the garbage signature is in the watch list |
| R4 | Thread-pool starvation at 8 ranks (A1) | performance | Measured in the A/B (the @16 arm is the test); if the pool is the bottleneck, the variant without the pool env (async only) is still a partial win |
| R5 | Hashcheck env + a shape change mid-serve (a new bucket appears) | correctness | The outer `input_hash` cache lookup still runs per step (the cache key); a new shape creates a new cache entry (re-capture, correct path); the risk window is only if the INNER spec is stale for a same-key different-layout input — the per-bucket layout stability makes this a non-case in steady state; the soak covers it |
| R6 | Numerical | — | None expected: pure scheduling, no math change |

## 7. Dependencies

- Precondition: the tensor-cache-on config (RA-001) — in place.
- Independent of: Q21, RA-012, RA-013, RA-004, RA-014 (composes with all; sequence the A/B FIRST so
  every later item is measured on the re-baselined stack).
- Boot: 1 (arms 0+2; arm 3 optional). No code beyond the gated line.

## 8. Implementation instructions (for the implementation agent)

**Files (all verified to exist in the deployed tree on g2):**
- `vllm_gaudi/v1/worker/hpu_model_runner.py` — `_maybe_wrap_in_hpu_graph` (l.1118-1120): add
  `asynchronous=True` to the `wrap_in_hpu_graph` call, env-gated. Call sites: main model l.5021
  (iteration 1), drafter l.5060 (keep synchronous until RA-004 lands).
- Serving boot script (the `kb_staging/boot_*.sh` family): the three env vars, arm-gated.

**Sequence:**
1. Arm 0 = current config (fresh boot, the same-boot protocol).
2. Arm 2 = `asynchronous=True` (gated on) + `PT_HPU_ENABLE_HPUGRAPH_THREAD=1` +
   `PT_HPU_QUEUE_SYNLAUNCHES=1` (fresh boot).
3. (Optional) Arm 3 = arm 2 + `PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK=1`.
4. Per arm, in order:
   a. Greedy parity vs arm 0 on the accuracy-suite subset (same-boot protocol; bit-identity gate).
   b. bs=1 bench (standard reps: 3 runs, median) + @16 + @32.
   c. 30-60 min soak at 16 streams: watch hang / garbage / HBM / graph count.
   d. Torch profiler timeline (CPU+HPU) — record the host-post-work-vs-device overlap ratio (this
      doubles as BG-3's measurement).
   e. HPUGraph cache count + HBM watch during the soak (the wrapper's `verbose=True,
      log_frequency=100` logging is available — re-capture signature: count steps + HBM growth).
5. Record EXP-RA011 (the arm table, the parity result, the t/s, the overlap ratio, the soak result);
   update `BOTTLENECK_RESEARCH.md` Section 7 with the MEASURED overlap (replacing the ESTIMATED
   range).

**Diagnostics to capture (all in the experiment record):**
- The D1-style py-spy split on arm 2 (the `replay` frame share should drop dramatically — the loop is
  off the calling thread; expect the sampler/draft frames to become the top host consumers).
- The per-step host wall time (the `execute_model` host span) vs the device step time from the trace.
- The HPUGraph cache count over the soak (stable = no re-capture storm).

## 9. Validation criteria (success definition)

1. **Parity:** greedy bit-identical vs arm 0 on the suite subset. (Any flip = FAIL, bisect.)
2. **Performance:** bs=1 ≥ +30% AND @16 ≥ +40% vs arm 0. (Below that = the overlap is not
   materializing — inspect the timeline BEFORE iterating; record the measured overlap ratio.)
3. **Stability:** 30-60 min soak @16: no hang, no garbage output, HBM growth ≤1 GB, HPUGraph count
   stable.
4. **Mechanism:** the timeline trace shows sampler/draft host frames overlapping the device execution of
   step N (qualitative confirmation that the thread-pool offload works as the source says).

## 10. Novelty assessment

- **Not previously attempted:** no project experiment touched the wrapper's `asynchronous` argument or
  the thread-pool env vars (checked TASKS.md, EXPERIMENTS.md, DECISIONS.md; the companion doc's
  RA-001..RA-010). The B2 work so far (RA-001 tensor cache) attacked per-replay ALLOCATION, not the
  serialization boundary.
- **Closest prior work and the difference:**
  - Q18 levers (early-draft enqueue): compatible, orthogonal — they add device work after host
    dispatch; RA-011 moves the host loop itself off the path.
  - vLLM `--async-scheduling` (RJ-18): a different mechanism (engine-loop scheduling); known broken on
    this stack class; RA-011 does not use it.
  - DSV4.1's async-mode lesson: it documents async replay as a supported path (the corpus entry) —
    RA-011 is the first application of it in this project.
- **Why it deserves investment now:** it is the largest single expected step in the whole program
  (Section 7 combined model), it is days not weeks, it re-baselines every later measurement, and its
  failure mode (parity) is fully gated and cheap to revert.
