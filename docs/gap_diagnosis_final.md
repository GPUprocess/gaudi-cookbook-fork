# Per-step pipeline gaps — GLM-5.3-Flash on Gaudi2 (worker TP0, bs=3 decode, 23 ms/step)

Read-only diagnosis, 2026-11-11. Evidence: py-spy record --idle 35 s @100 Hz (worker main thread = 3499
samples = 1536 steps ≈ 23 ms/step; 1 sample = 22.8 µs/step), the part-1/part-2 notes, live /proc checks,
and the deployed sources (vllm @ `/opt/glm53/vllm` clean upstream `568afb3a1`; vllm-gaudi @ `/opt/glm53/vllm-gaudi`;
bridge venv torch @ `/opt/glm53/venv/lib/python3.12/site-packages/habana_frameworks/torch`; bridge .so
`libhabana_pytorch_backend.so`). Live env (worker /proc/<pid>/environ): `PT_HPU_LAZY_MODE=1`,
`PT_HPU_LAZY_ACC_PAR_MODE=0`, `PT_HPU_ENABLE_LAZY_COLLECTIVES=true`, `PT_HPU_MEMORY_POOL=none`,
`PT_HPU_WEIGHT_SHARING=0`, `GLM_ROUTER_FAST=1`, `GLM_EARLY_DRAFT=1`, `GLM_SPEC_DEVICE_REJECT=1`,
`GLM_FP8_MXK=0`; `GLM_HPU_GRAPH_ASYNC` **not set** (async replay OFF in production).
`VLLM_ENABLE_V1_MULTIPROCESSING` default True (envs.py:148,1327). No `--async-scheduling` on the CLI.

## 0. Corrections to the earlier gap breakdown (important)

Re-classifying every worker main-thread stack by which branch of `wrapped_hpugraph_forward`
(`habana_frameworks/torch/hpu/graphs.py`) it sits in changes the story:

| bucket (main thread, % of 3499) | ms/step | was reported as | actually |
|---|---:|---|---|
| input wait, `SpinCondition.wait:198 → Poller.poll` (1046) | **6.8** | "zmq input wait 29.4%" | shm-broadcast **idle-mode** wait (zmq SUB notify poll), the pipeline bubble |
| `HPUGraph.replay` sync C++ (929, steady replay path graphs.py:75→760) | **6.0** | "graph replay" | unchanged — the 322-recipe sync launch loop (B2) |
| **graph CAPTURE branch (graphs.py:697-733, 399 samples)** | **2.6** | — | **new-shape re-captures** (dry-run trace) — contains the "kda_attention 32", "mark_step 264 of 321", most `dynamic_quant`, `mhc_pre_torch`, `BasevLLMParameter` samples |
| `RejectionSampler.parse_output` (232) | **1.5** | "CPU spec-decode parse" | **blocking D2H drain** (`.cpu()` at rejection_sampler.py:273) — numpy work is µs at bs=3 |
| eager-island dispatch, steady (OpOverload 83 MoE + 18 RMSNorm + 15 fp8-linear + 14 dyn-quant …) | **0.8** | "OpOverload dispatch tax" | per-step **eager islands** (42× monolithic MoE op etc.), see §4 |
| early-draft host dispatch (`decode_lean.early_draft` → `HpuEagleProposer.propose`, 525) | **~3.4** | "draft dispatch" | unchanged |
| `mark_step` steady (57) | **0.37** | "2 ms/step flush" | only 0.37 ms/step is steady; the "2 ms" was capture-path flushes |
| input prep (63) | 0.4 | — | — |
| `BasevLLMParameter.__torch_function__` (17 total) | **~0.05** | "14 samples" | **negligible** — deprioritize |
| take_draft_token_ids (13) | 0.09 | — | + its own RPC round trip (see §1) |

Two prior claims are wrong and must not carry into the research queue:
1. "kda_attention is an eager per-step op" — **all 32 kda_attention samples are under the capture branch
   (graphs.py:709 `orig_fwd`), zero under replay (graphs.py:760)**. KDA is traced into the HPU graph at
   capture; in steady state no KDA Python runs per step. The 32 samples are the *capture* cost.
2. "mark_step ~2 ms/step flush" — 264 of 321 mark_step samples are capture-path (dry-run trace forces the
   ~162 AR boundaries per forward). Steady-state flush is 0.37 ms/step.

Also: bs=1 "45.1 ms/step" (doc rev-3) vs bs=3 "23 ms/step" (this profile) is irreconcilable on a
weight-stream-bound device (≥15.5 ms/step at any bs); the bs=3 window likely includes inter-request gaps
(the worker sits in idle-mode `Poller.poll` 30% of the time, which requires >1 s since the last broadcast
read — impossible in gap-free 23 ms steps). Treat 23 ms/step as "steps-per-wallclock under this load", and
use the *percentages* (they are the stable signal) when extrapolating to bs=1/bs=8.

## 1. The per-step round trip (Q1: why is the worker idle ~30%?)

The pipeline is **fully serial: schedule → 3× RPC round trips → output processing, with the worker idle
the whole time**. Per step (all-decode, MTP k=1):

**Engine main thread** (`/opt/glm53/vllm/vllm`):
1. `run_busy_loop` v1/engine/core.py:1358-1364 → `_process_engine_step` :1397 → `step_fn`
   (`step_fn` = serial `step` because `batch_queue` is None: core.py:199-206,224-225;
   `batch_queue_size = vllm_config.max_concurrent_batches`, config/vllm.py:512-521 → PP=1 and
   `async_scheduling=False` → **1** → no double buffering).
2. `step()` core.py:576-606: `schedule` :587 → **RPC1** `execute_model(non_block=True)` :588 →
   `get_grammar_bitmask` :589 → **`future.result()` :594 BLOCKS** → (output is None) **RPC2**
   `sample_tokens` :596 → `future.result()` again → `update_from_output` :601.
3. `post_step` core.py:608-614: **RPC3** `take_draft_token_ids` :613 (`check_for_draft_tokens`
   core.py:163-165 = True for spec decode).

So **three broadcast/response round trips per step**, and the engine blocks on each before the next can
start. The block is `FutureWrapper.result` (v1/executor/multiproc_executor.py:85-92) which runs
`_wait_for_response` :93-100 **inline on the engine main thread** → `get_response` →
`mq.dequeue` :390 → shm SpinCondition wait. The engine main thread was 100% busy in the capture
(py-spy --idle saw only background threads: Condition.wait/selector/Poller = idle threads, 5 unique stacks).

**Worker** (vllm/v1/executor/multiproc_executor.py):
- `worker_busy_loop` :999-1032: `rpc_broadcast_mq.dequeue(indefinite=True)` :1003 (the 6.8 ms/step idle)
  → `func(*args)` :1012 → `handle_output` :1014/:974 → `enqueue_output` :952-972 (pickle +
  `response_mq.enqueue` :970).

**Why 3 RPCs:** the gaudi runner defers all execution into the 2nd RPC. `HPUModelRunner.execute_model`
(vllm-gaudi/v1/worker/hpu_model_runner.py:4214) does only `_update_states` and stashes the scheduler
output (l.4341 `self.scheduler_output = scheduler_output`; returns None) — 3 profile samples. The real
step is `sample_tokens` (l.4424): input prep → replay → sampler → early draft → parse → bookkeeping →
`_draft_token_ids = propose_draft_token_ids` (l.4915). RPC3 then fetches those draft ids via
`take_draft_token_ids` (l.7322-7331, `.tolist()` D2H at l.7327).

**Transport** (vllm/distributed/device_communicators/shm_broadcast.py): shm ring buffer
(default **max_chunks=10**, l.351 — the buffer itself is NOT the bottleneck) + zmq IPC PUB/SUB notify
(`enqueue` → `_spin_condition.notify` l.767; reader wake `SpinCondition.wait` l.183-201: hybrid —
`sched_yield()` spin while within `busy_loop_s` (default **1.0 s**, l.126, l.195-196) of the last read,
else blocking `poller.poll` idle mode l.198). Serialization: `pickle.dumps` per broadcast
(SchedulerOutput is pickled **twice** per step — once per RPC — `collective_rpc` l.374-378;
`enqueue` l.728-770 with PickleBuffer OOB for ≥1 MiB) and per response (`enqueue_output`).

**Mirror check:** worker idle 6.8 ms/step == engine serial time (schedule + 3 RPC pickup tails +
`update_from_output` + post_step). Part-2 engine capture confirms the engine burns CPU spinning on the
response MQ (sched_yield 1249 samples, `memory_fence` 361, `acquire_read` checks ~360).

## 2. What accumulates between mark_step calls (Q2)

`mark_step` = `htcore._mark_step` pybind C++ (torch/core/step_closure.py:66-69); 318/321 samples are the
C++ leaf — the flush is C++ (recipe compile/submit of the accumulated eager ops). Steady-state decode
accumulates very little (0.37 ms/step): the device-reject grid ops (`decode_lean.py:_accept_grid` 135-150,
`make_device_reject` 153-238: argmax/compare/cumsum/where/cat), `mhc_pre_torch` (mhc.py:153),
`dynamic_quant` (vllm-gaudi/extension/ops.py:1006-1016), the 42× monolithic MoE eager op
(ops.py:1372 `torch.ops.hpu.mixture_of_experts.fp8_fused_weights_dynamic`), RMSNorm (ops/hpu_layernorm.py:24),
early-draft H2D + draft aten ops. Boundaries: explicit `htorch.core.mark_step()` in
`sample_tokens` (hpu_model_runner.py:~4664/4677/4700) + AR-forced boundaries
(`PT_HPU_ENABLE_LAZY_COLLECTIVES=true`).

`PT_HPU_LAZY_ACC_PAR_MODE` is the **only** accumulation knob (vendor `scripts/env_flags.yaml`: default **1**;
we run **0** — the adopted config deliberately). There is **no accumulation batch-size knob**; the
accumulation unit is the mark_step/collective boundary, so the lever is "fewer eager ops between
flushes" (i.e. graph the islands / Q21 fused MoE), not a bridge setting. ACC_PAR_MODE=0 vs 1: with only
~0.37 ms/step of steady accumulation the parallel-accumulation thread buys nothing and adds a join —
consistent with the adopted 0. **Do not chase ACC_PAR_MODE; the 2 ms/step figure was capture-path.**

## 3. RejectionSampler.parse_output (6.6%, 1.5 ms/step) — confirmed blocking D2H

- Production: `output_token_ids_np = output_token_ids.cpu().numpy()` (vllm/v1/sample/rejection_sampler.py:273)
  — a **blocking D2H that drains the whole device**: the target verify + sampler + the **early draft**
  (the draft is enqueued *before* the parse: hpu_model_runner.py:4711-4713 `early_draft` →
  decode_lean.py:69-121, draft `propose` at :112; parse at :4715-4718). The numpy mask + per-row
  `.tolist()` (rejection_sampler.py:276-289) is µs at bs=3 — the whole 1.5 ms is the drain.
- `output_token_ids` is produced device-side: `RejectionSampler.forward` (rejection_sampler.py:93-203) →
  `rejection_sample` monkey-patched to the device walk (vllm-gaudi/v1/sample/hpu_rejection_sampler.py:298-300 →
  decode_lean.make_device_reject; `_accept_grid` returns `[n_req, k+1]` on HPU).
- The list-of-lists IS the wire format: `ModelRunnerOutput.sampled_token_ids: list[list[int]]`
  (vllm/v1/outputs.py:244) → `EngineCoreOutput.new_token_ids: list[int]` (v1/engine/__init__.py:182) →
  `Detokenizer.update(new_token_ids: list[int])` (v1/engine/detokenizer.py:41-42). **You cannot skip the
  list; you can only make the D2H async.**
- Async-D2H feasibility on this stack: pinned memory unsupported (vllm `platform.py:369`
  "Pin memory is not supported on HPU") → use pageable host double-buffer + the bridge's copy threads
  (`PT_HPU_ENABLE_D2H_ASYNC_THREAD` default true, `PT_HPU_ENABLE_H2D_COPY_ASYNC_THREAD` default true —
  vendor env_flags.yaml). Record a `torch.hpu.Event` right after the sampler output (before early_draft
  enqueues), enqueue a non-blocking copy of the tiny `[bs, k+1] int32` into the double buffer, and consume
  step N's buffer in step N+1's parse (wait on event N there — the device is a full step ahead, so the wait
  is a fast path, not a drain).
- Existing machinery in the tree: the RA-011 wrapper gate **already exists** in the deployed tree:
  `_maybe_wrap_in_hpu_graph` (hpu_model_runner.py:1118-1134, `GLM_HPU_GRAPH_ASYNC` env gate →
  `asynchronous=True` into `wrap_in_hpu_graph`, graphs.py:70 `replay(asynchronous)`; thread-pool branch
  needs `PT_HPU_ENABLE_HPUGRAPH_THREAD` (default true per env_flags.yaml) + `PT_HPU_QUEUE_SYNLAUNCHES`).
  EXP-RA011 measured **+29% bs=1** but **flipped R003 deterministically** (host consumes the tensor-cached
  graph outputs in-flight). **No `GLM_ASYNC_D2H` exists anywhere in the fork** (grep of
  /home/tmpai2/publish_work/fork: zero hits) — the async-D2H output redesign is un-staged. vllm-gaudi does
  have a partial precedent: its `use_async_scheduling` branch keeps tokens on HPU and defers the copy
  (hpu_model_runner.py:4813-4834, `prev_sampled_token_ids`) but requires `--async-scheduling` (known
  broken on this stack class, RJ-18) and the comment flags the spec-decode device mismatch.

## 4. Bridge dispatch angle (Q4)

The steady-state OpOverload tax (0.8 ms/step) is the **Python torch dispatcher for the per-step eager
islands**, dominated by the monolithic MoE op: `VllmMixtureOfExpertsOpFP8PerChannel.forward`
(vllm-gaudi/extension/ops.py:1372) 83 samples ≈ 0.55 ms/step (42 MoE layers × `fp8_fused_weights_dynamic`
+ `dynamic_quant` + casts). These ops are bridge-native `torch.ops.hpu.*` calls that the lazy executor
executes eagerly (one op at a time, per vendor ToO "Eager/Lazy" docs); each pays OpOverload/
OpOverloadPacket Python dispatch (torch/_ops.py:865/1269) + lazy boundary handling. Note the existing
`[gaudi-fix:moe-explicit-overload]` comment at ops.py:1368: the overload packet used to try a mismatching
overload whose error message **formatted the device tensor (a hidden D2H sync)** — named-overload calls
are already a micro-optimization in this file.

Bridge-level fast path that exists but is **OFF**: `PT_HPU_ENABLE_EAGER_CACHE` (vendor
`scripts/env_flags.yaml`: "separate flag to control recipe caching in eager backend", default **false**;
consumed in `backend/kernel/hpu_habana_launch_op_pt.cpp:259-264` — `enable_eager_caching_ =
((execution_mode_ == EAGER) && (!is_eager_compiler_supported && PT_HPU_PGM_ENABLE_CACHE)) ||
PT_HPU_ENABLE_EAGER_CACHE`). With it off, each eager-island op re-derives its launch/JIT setup per call.
`PT_HPU_LAZY_EAGER_OPTIM_CACHE` (default true) and `PT_HPU_EAGER_SHAPE_AGNOSTIC_GRAPH` (default true) are
already at their fast defaults. `nm` on the .so shows no separate "dispatch cache" beyond these; the
dispatch key machinery is the stock `c10::Dispatcher` HPU backend — the Python OpOverload layer is the
only per-op Python cost, and it only disappears by (a) caching the compiled op (EAGER_CACHE), (b) fewer
island ops (Q21 fused MoE replaces ~50 ops with 1), or (c) graphing the island (the MoE op would have to
be bridge-compiled as a graph node — unproven; the log's "index_select … might result in accuracy issues"
warnings at each capture show the bridge does trace some aten ops into graphs, but the monolithic MoE C++
kernel is not among the traced ones in steady state).

`kda_attention` (vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py:44-71,
`direct_register_custom_op` l.64-69, op_func l.44-62 calling `no_compile_layers[layer_name]._forward`):
already inside the graph (capture-traced; §0). No steady-state cost.

## 5. Ranked gaps (root cause + evidence)

| # | Gap (ms/step, bs=3 profile) | Root cause | Key evidence |
|---|---|---|---|
| G1 | **6.8** worker input-wait bubble | Serial engine loop: `step()` blocks on each of 3 RPC responses before the next schedule; gaudi defers the whole step into RPC2; RPC3 (draft ids) adds a 3rd round trip; SchedulerOutput pickled+broadcast twice | core.py:576-606,588,594,596,601,613; config/vllm.py:512-521; multiproc_executor.py:85-100,374-405,999-1032; hpu_model_runner.py:4214-4342,4424,4915,7322-7331; profile: 1046 samples at shm_broadcast.py:198 |
| G2 | **6.0** sync `HPUGraph.replay` C++ launch loop | 322-recipe launch loop runs on the host thread (`asynchronous=False` default) and blocks the step; host+device serialize (step ≈ host+device, ~zero overlap) | graphs.py:70-81 (`_hpu_C.replay(..., asynchronous)`), 756-767 (replay path), 697-733; hpu_model_runner.py:1118-1134 (async gate exists, OFF); B2/D1: 93.3% of execute host time; profile 929 samples at graphs.py:75 |
| G3 | **~3.4** early-draft host dispatch | The MTP draft forward runs ~15-20 aten ops eagerly every step, each paying Python dispatch + lazy accumulation; serialized before the step response can be sent | decode_lean.py:69-121 (early_draft, propose :112); hpu_model_runner.py:4711-4713; profile 525 samples under early_draft/HpuEagleProposer; B doc: 31.4% of host at bs=1 |
| G4 | **1.5** parse_output blocking D2H drain | `.cpu().numpy()` (rejection_sampler.py:273) drains verify+sampler+draft; host cannot send the response or start next-step work until the device is idle | hpu_model_runner.py:4715-4718; decode_lean.py:153-238; profile 232 samples at rejection_sampler.py:273 |
| G5 | **2.6** new-shape graph captures (amortized) | Every new (bs, ctx-bucket) decode shape pays a full dry-run trace: 34× KDA trace, 42× MoE trace, ~162 mark_step flushes, dry-run device exec; ctx ladder 64-step ⇒ frequent crossings with 3+ growing sequences | graphs.py:697-733; profile: 399 samples at :709 (kda 32, mark_step 264, dispatch 21, dyn-quant 12, mhc 12); serve log: `WarnIfOpsIncompatibleWithHPU` per capture (hpu_lazy_tensors.cpp:1293) |
| G6 | **0.8** eager-island dispatch tax (42× MoE op + norms + fp8 linears) | Per-step eager islands pay Python OpOverload dispatch + per-call launch setup because `PT_HPU_ENABLE_EAGER_CACHE` is OFF (default false) | ops.py:1372,1368 comment; hpu_layernorm.py:24; ops.py:992,1009; env_flags.yaml (PT_HPU_ENABLE_EAGER_CACHE default false); hpu_habana_launch_op_pt.cpp:259-264 |
| G7 | **0.4** mark_step steady flush + input prep 0.4 + H2D/take_draft/misc ~0.5 | Residual host work | step_closure.py:66-69 (57 steady samples); hpu_model_runner.py:3487-3717 |

## 6. Ranked fix proposals (expected ms/step saved; bs=1 vs bs=8)

Host gaps are ~batch-size independent; the engine-side bubble (G1) grows with bs (scheduler + more
requests in update_from_output). "bs=1" extrapolation anchored on the doc's 45.1 ms/step (host ~24 ms);
"bs=8" assumes host ~21-24 ms and device ~20-25 ms (weight-stream floor 15.5 ms + GEMV).

**F1. Async-D2H output-path redesign that unblocks RA-011** — expected **−3…6 ms/step bs=1, −5…10 ms/step bs=8** (largest).
Root fixes G2+G3+G4 together: with `asynchronous=True` (hpu_model_runner.py:1132-1134, gate
`GLM_HPU_GRAPH_ASYNC`) the 322-recipe loop moves to the vendor pool (`PT_HPU_QUEUE_SYNLAUNCHES=1`,
`PT_HPU_ENABLE_HPUGRAPH_THREAD`=true default) and the sampler/draft host work overlaps device N; the
R003 flip in EXP-RA011 came from the host consuming tensor-cached graph outputs in-flight — fixed by
making the one step-boundary D2H (parse_output, G4) truly async: `torch.hpu.Event` after the reject
sampler (hpu_model_runner.py:4704-4710) → non-blocking copy of `[bs,k+1]` into a 2-stage pageable host
buffer (bridge D2H async thread, default on) → step N+1's parse (hpu_model_runner.py:4715) consumes
stage[N-1] after `event[N-1].wait()` (device is a step ahead ⇒ fast path). Draft ids (RPC3 payload)
piggyback a second tiny async D2H after early_draft (decode_lean.py:120-121). Files:
hpu_model_runner.py:4704-4746,7322-7331; rejection_sampler.py:273 (make parse_output take a pre-copied
host buffer); hpu_model_runner.py:1118-1134. Risk: **R003-class** (measured flip with flag alone — the
redesign is precisely the mitigation; gate on same-boot parity + 30-60 min soak per RA-011 §9); HBM +2
tiny host buffers (negligible); no numerics change (pure scheduling). Precedent in tree:
hpu_model_runner.py:4813-4834 (async-scheduling token deferral) shows the deferral pattern works on HPU
except for the spec-decode device mismatch, which the staged-buffer design sidesteps.

**F2. Collapse the 3 RPC round trips into 1** — expected **−0.6…1.2 ms/step** (bs=1 and bs=8; slightly more at bs=8).
Fixes the transport half of G1. (a) `HPUModelRunner.execute_model` (hpu_model_runner.py:4214-4342) already
only stashes state — move `_update_states` into the start of `sample_tokens` and make
`HPUWorker.execute_model` (hpu_worker.py:569) a no-op returning None **without** a response
(`output_rank=None` path: multiproc_executor.py:1014-1017 already skips the response for non-output
ranks — set `unique_reply_rank` handling so RPC1 needs no response) — or, vllm-side, patch
`step()` (core.py:588-596) to skip the `execute_model` RPC when the executor defers (a one-line
executor capability flag). (b) Embed draft ids in the `ModelRunnerOutput` (new optional field
`draft_token_ids: list[list[int]] | None`, vllm/v1/outputs.py:244 vicinity) computed by
`sample_tokens` itself (it already owns `_draft_token_ids` at hpu_model_runner.py:4915; the device is
already drained there by the parse, so `.tolist()` at l.7327 moves into sample_tokens for free), and
patch `post_step` (core.py:612-614) to consume it instead of RPC3. Saves 2 broadcast+notify+wake cycles,
2 response round trips, and one SchedulerOutput re-pickle per step. Risk: **low** (wire-format add +
engine patch; no device semantics change; parity-trivially bit-identical).

**F3. Kill the re-capture tax (G5)** — expected **−2…2.6 ms/step bs=3, −2.5…4 ms/step bs=8** (more sequences ⇒ more bucket crossings).
Coarsen the decode ctx ladder (64-step → 256/512-step buckets in `_prepare_decode_inputs`,
hpu_model_runner.py:3487-3717) so shapes stop crossing during a session, and/or pre-capture the expected
bs×ctx decode grid at startup (warmup already walks buckets — extend it). Device cost of padding to a
coarser bucket is ~0 at decode (rows are padded to the bucket anyway; extra ctx rows in KDA/MLA decode
are masked — verify per-bucket). Alternatively shape-agnostic decode graphs (the bridge has
`PT_HPU_EAGER_SHAPE_AGNOSTIC_GRAPH`, default true, for eager; graph-side needs vendor support). Risk:
**low-medium** (HBM for extra captured graphs; padding compute; startup time). This is the real fix for
the "kda_attention" and "mark_step 2 ms" ghosts — both are capture costs.

**F4. `PT_HPU_ENABLE_EAGER_CACHE=1`** — expected **−0.3…0.6 ms/step** (bs=1 and bs=8; scales with the 42 MoE island calls).
One env var (vendor flag, `settable: true`, default false — env_flags.yaml; consumed at
hpu_habana_launch_op_pt.cpp:259-264). Caches the JIT/recipe launch setup of the eager-island ops
(42× `fp8_fused_weights_dynamic`, RMSNorm, fp8 linears). Risk: **low** (vendor-shipped, numerics-neutral;
A/B + parity gate as with any bridge env).

**F5. `VLLM_USE_SPINLOOP_EXT=1`** — expected **−0.2…0.5 ms/step** on the engine pickup tail (bs=1 and bs=8).
The engine main thread dequeues responses on the critical path (multiproc_executor.py:85-100) through the
Python-level spin (`memory_fence` lock + byte checks + `sched_yield`, shm_broadcast.py:667-724, part-2
engine profile: 1249 sched_yield + 361 fence samples). The C spinloop extension
(shm_broadcast.py:44-53, `vllm.spinloop`, `SPINLOOP_TIMEOUT_SECONDS=0.1` l.51, used at l.684-688) replaces
the Python check loop. No env knob exists for `busy_loop_s` (constructor default 1.0 s, l.126) — the only
spin-policy knobs are `VLLM_USE_SPINLOOP_EXT` (envs.py:2013) and `VLLM_RINGBUFFER_WARNING_INTERVAL`
(envs.py:710). Risk: **low** (vendor extension; verify the .so builds/loads in the venv first).

**F6. `--async-scheduling` (upstream batch-queue double buffer)** — expected **−4…6 ms/step bs=1, −6…9 ms/step bs=8** *if made to work* (fixes G1 at the control-flow level: schedule N+1 while N executes — core.py:617-703, config/vllm.py:512-521 → max_concurrent_batches=2).
**Ranked below F1+F2 because it is known broken on this stack class (RJ-18)** and re-opens the R003
hazard at the scheduler level (provisional scheduling on draft tokens; KV over-reservation via
`max_in_flight_tokens`, config/vllm.py:526-532; draft-token flow changes, core.py:608-611,714-716).
Revisit only after F1+F2 land and the residual bubble is still material.

**F7. Deprioritized (measured negligible):** `BasevLLMParameter.__torch_function__` post-load conversion
to plain tensors (vllm/model_executor/parameter.py:126-129) — 17 samples ≈ 0.05 ms/step;
`PT_HPU_LAZY_ACC_PAR_MODE=1` — steady accumulation is 0.37 ms/step, the parallel-acc thread adds a join
for nothing (vendor default is 1; adopted config's 0 is correct for this op mix).

## 7. Direct answers to the assigned questions

1. **Where the engine waits for all worker outputs before scheduling:** `step()` core.py:594
   (`future.result()` → multiproc_executor.py:85-100 → `mq.dequeue` :390), again after RPC2 (same
   `future.result` path), and RPC3 at core.py:613; then `update_from_output` core.py:601 — all before the
   next `schedule` (core.py:587) of the next `run_busy_loop` iteration (core.py:1358-1364). No
   double-buffering because `max_concurrent_batches==1` (config/vllm.py:512-521; no `--async-scheduling`).
2. **Can the broadcast be double-buffered:** the *transport* already can (shm ring `max_chunks=10`,
   shm_broadcast.py:351) — the serialization is the engine control flow (core.py:576-606) plus the gaudi
   2-RPC deferral split (hpu_model_runner.py:4214-4342) and the 3rd draft RPC (core.py:613). Fix F2
   (merge RPCs) is the cheap version; F6 (batch queue) is the full version.
3. **Shm spin policy knobs:** `VLLM_USE_SPINLOOP_EXT` (envs.py:2013; shm_broadcast.py:44-53,684-688,
   timeout 0.1 s l.51), `VLLM_RINGBUFFER_WARNING_INTERVAL` (envs.py:710); `busy_loop_s` (spin→idle
   window, default 1.0 s) has **no** env knob (shm_broadcast.py:126). Hybrid behavior: `sched_yield`
   within 1 s of last read (l.195-196), else zmq `Poller.poll` idle (l.198) — the worker's 29.9% is the
   idle branch (live check: worker `wchan=do_poll` when idle).
4. **kda_attention graph-inclusion path:** already included — all kda samples are capture-branch
   (graphs.py:709), none replay (l.760). The actionable item is F3 (capture frequency), not KDA.
5. **Parameter post-load conversion:** parameter.py:126-129 passthrough; measured 0.05 ms/step —
   deprioritized (F7).
6. **Async D2H (parse_output):** mechanism + files in §3/F1: event after hpu_model_runner.py:4704-4710,
   2-stage pageable host buffer (pinned memory unsupported: vllm platform.py:369; bridge D2H async
   thread default on), consume one step later at :4715. `GLM_ASYNC_D2H` does not exist in the fork —
   nothing staged; only the RA-011 `GLM_HPU_GRAPH_ASYNC` wrapper gate (hpu_model_runner.py:1118-1134)
   and the measured EXP-RA011 result (+29% bs=1, R003 flip).
7. **mark_step flush:** 0.37 ms/step steady (57 samples) vs 1.63 ms/step capture (264 samples); no
   accumulation-batch knob exists (`PT_HPU_LAZY_ACC_PAR_MODE` only, default 1 per env_flags.yaml, live 0);
   reduce by reducing eager islands (F1/F4/Q21), not by a bridge setting.

## 8. Live-state notes

- The server **restarted between the profile and this diagnosis**: profiled PIDs 645253/645732/645946 are
  gone; current = API 1052708 / EngineCore 1053529 / Worker_TP0 1053832 (same CLI, log
  /data/glm53/serve_restored.log). Live wchan while idle: worker `do_poll`, engine `futex_wait` —
  consistent with the profiled wait branches. py-spy not runnable as tmpai2 (ptrace_scope=1, no sudo).
- Steady-state decode at bs=3 should show the worker *spinning* (sched_yield, within the 1 s
  `busy_loop_s` window), not idle-polling; the profile's 30% idle-mode samples indicate the bs=3 load
  window contained inter-request gaps — see §0 caveat on the 23 ms/step figure.
