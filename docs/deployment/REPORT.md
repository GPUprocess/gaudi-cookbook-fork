# GLM-5.3-Flash deployment on g2 (192.168.0.143) — report

Date: 2026-10-31. Operator: autonomous session. Target: 8× Intel Gaudi 2 node
(`g2`, Ubuntu 24.04.5, Gaudi SW 1.24.1-482, 160 cores, 3.7 TB NVMe).

## 1. What was running, and what was stopped

| Instance | How it ran | Resources | Action taken |
| --- | --- | --- | --- |
| **DeepSeek-V4.1-Flash** (`/opt/dsv41/venv/bin/python -m vllm_gaudi.entrypoints.deepseek_v41`) | tmux session `dsv41` → `/data/dsv41/launch-watchdog.sh` → watchdog `dsv41-watchdog.service` (auto-relaunch on crash) | TP 2 × PP 2 on cards 4–7, port 8000, `--max-num-seqs 1` | **Stopped and disabled** |
| `strata-hpu` (user `mainmaster`, Qwen3.8-Flash-Next IQ4_XS) | bare processes (`python3 -m serve.server` + native binary), no systemd unit | card 2, port 8080 | **Killed** (user-approved; GLM-5.3-Flash needs all 8 cards) |

Stop/disable sequence and verification:

```
sudo systemctl stop dsv41-watchdog.service     → inactive (dead)
sudo systemctl disable dsv41-watchdog.service  → Removed "/etc/systemd/system/multi-user.target.wants/dsv41-watchdog.service"
                                                 is-enabled → disabled
tmux kill-session -t dsv41 + pkill of the two deepseek command lines
```

Post-check: no `deepseek`/`dsv41`/`VLLM::Worker*` processes left, port 8000
closed, `hl-smi` shows cards 4–7 released (768 MiB idle each). The service had
consumed 4 d 3 h CPU time since Oct 28. The `dsv41-watchdog.service` unit file
remains on disk at `/etc/systemd/system/dsv41-watchdog.service` (disabled, so it
does not start at boot); delete it with
`sudo rm /etc/systemd/system/dsv41-watchdog.service` if you want it gone.

Nothing else on the node was modified: the strata checkout, the `/opt/dsv41`
stack and the DeepSeek model files under `/data/dsv41` are untouched (only the
processes were stopped). The watchdog stays disabled — re-enable with
`sudo systemctl enable --now dsv41-watchdog.service` if you want DeepSeek back
(it will contend for cards 4–7 with the new server if both run).

## 2. Deployment: GLM-5.3-Flash (zai-org, 320B MoE, block-FP8)

Followed the gaudi-cookbook recipe `recipes/zai-org/GLM-5.3-Flash/recipe.yaml`
through the `glm-5-3-flash` port, with one structural deviation: **no Docker**.
The documented path builds a port image on a stock `vllm-gaudi v0.26.0` base
image; this host has no container runtime and `vault.habana.ai` no longer
serves its pip index, so the exact component set of that base image was
installed natively in a venv, from the official Habana wheel bundle already on
the host (`/opt/dsv41/pt/wheels`, SW `1.24.1.482`, torch `2.11.0a0+git009b5f6`
— the same wheel set the `pytorch-installer-2.11.0` image ships, matching the
host driver build).

Components, all verified at the release-pinned commits the port demands:

| Component | Version / commit |
| --- | --- |
| vllm-gaudi | `v0.26.0`, `b059f9c95d6a` ("Release v0.26.0 Updates (#1705)"), editable install, `VLLM_TARGET_DEVICE=hpu` |
| vLLM | `568afb3a1380` (= vllm-gaudi v0.26.0 `VLLM_STABLE_COMMIT`), editable, `VLLM_TARGET_DEVICE=empty` |
| PyTorch | `2.11.0a0+git009b5f6` (Habana fork, from the official bundle) |
| Port | `ports/glm-5-3-flash` at gaudi-cookbook main: 4 source patches applied, mHC Sinkhorn TPC kernel built, `glm5_next_hpu` plugin installed |
| Fixes | `fixes/apply_all.py --profile v0.26.0 glm47-streaming-text-markers` — 24 profile fixes + the streaming tool-marker fix, 25 applied |
| Checkpoint | `/data/models/GLM-5.3-Flash`, 306 GiB block-FP8 as published, downloaded from HF (Xet disabled) |

Two host-specific patches were required on the way:

1. **`vllm/env_override.py`** — vLLM at `568afb3a` carries the
   `GraphCaptureOutput.get_runtime_env` monkeypatch for torch < 2.12. The
   cookbook's getting-started says to comment it out for the Habana PyTorch
   2.11 fork (it lacks the class). Disabled with an `if False:` guard,
   `_apply_constrain_to_fx_strides_patch` kept as required.
2. **Home directory** — `/home/tmpai2` was root-owned and empty: pip had no
   cache, `hf` xet downloads hit `Permission denied`, and the TPC library gate
   aborted because Habana's logger could not create `~/.habana_logs`. Fixed by
   creating `~/.cache` and `~/.habana_logs` owned by `tmpai2` (and taking
   ownership of the home dir).

Serving: `server/serve_glm.sh` = the recipe launch block verbatim (env
switches `GLM5_NEXT_VISION/KDA_SCAN/DECODE_LAYER_GROUP/MHC_SINKHORN`,
`vllm serve /data/models/GLM-5.3-Flash --served-model-name
zai-org/GLM-5.3-Flash --tensor-parallel-size 8 --enable-expert-parallel
--max-model-len 262144 --num-gpu-blocks-override 1280 ...`, port 8000 — the
port DeepSeek gave up). Full commands: see `REPLICATE.md`.

## 3. Verification and benchmarks

The port's device gates (`ports/glm-5-3-flash/tests/hpu/run_gates.sh`, the
cookbook's acceptance card) all **PASS** against the running server:

| Gate | Result | Detail |
| --- | --- | --- |
| golden | **PASS** | 12/12 greedy answers token-identical to `golden_reference.json` (64 s) |
| needles | **PASS** | 3/3 codes retrieved at 16K, 63K, 127K and 239K depth (188 s) |
| tools | **PASS** | 8/8 (streamed + not, single, parallel, multi-turn, no-call, after a 32K document) (18 s) |
| images | **PASS** | 8/8 (colours, quadrants, counting, text around images, 5-image prompt rejected) (11 s) |
| decode | **PASS** | single-stream 30.7 tok/s @ 2K prompt (floor 15), 29.7 tok/s @ 128K (floor 10) (137 s) |
| prefill | **PASS** | warm TTFT 2.6 s @ 16K, 11.6 s @ 64K, 27.9 s @ 128K (limit 120 s) (86 s) |

Against the recipe's published reference on the documented (container) path:

| Metric | This deploy | Recipe reference |
| --- | --- | --- |
| decode @ 2K | 30.7 tok/s | 43.5 tok/s |
| decode @ 128K | 29.7 tok/s | 38.6 tok/s |
| TTFT @ 16K | 2.6 s | 2.6 s |
| TTFT @ 64K | 11.6 s | 10.3 s |
| TTFT @ 128K | 27.9 s | 23.8 s |

Correctness is bit-exact; throughput is 70–80% of the reference. [INFERENCE]
The remaining gap is likely the native-stack equivalent of the container's
tuning (no `sys_nice` isolation, NUMA pinning of the stock image's runtime
env) — all floors pass with 2× headroom, so this is performance headroom, not a
defect. During serving all 8 cards hold ~66.3 GiB / 98.3 GiB HBM (weights +
KV pool sized by `--num-gpu-blocks-override 1280`).

Server: `zai-org/GLM-5.3-Flash` on port 8000 (the port DeepSeek vacated),
max context 262,144, `--max-num-seqs 32`, prefix caching off (the KDA layer
refuses it by design).

## 4. Deliverables

- `REPLICATE.md` — step-by-step replication instructions.
- `server/setup_venv.sh`, `server/install_stack.sh`, `server/serve_glm.sh` —
  the exact scripts used (also at `/data/glm53/` on the target);
  `server/env_override.py` — the patched vLLM file as deployed.
- Server state: GLM-5.3-Flash on port 8000 (started via
  `nohup bash /data/glm53/serve_glm.sh`, log `/data/glm53/serve.log`).

## 5. Post-deployment: the 4-card experiment and the scout directive (2026-10-31 late)

After this report was first written, two directives reshaped the card budget: cap GLM at 6 HPUs
(prefer 4), and run two `Qwen3.8-Flash-Next` 8B scout instances on 2 dedicated HPUs. Both were
tested to ground truth; results (full evidence in `glm53_optimize/OPTIMIZATION_LOG.md` E003-E005):

1. **TP6/TP5/TP3 are invalid** (64 attention heads not divisible; PP unproven on this port).
2. **TP4 + graph capture does not fit**: two configs (capture 128/64, batched-tokens 8192/4096)
   died identically at init — `PT_DEVMEM Allocation failed for 1440.12 MB` (the fixed KV-pool
   carve). Weights 76.4 GiB/card + comm + graph workspace + prefill transient ≈ the full
   91.2 GiB GMU 0.95 budget.
3. **TP4 + eager boots but delivers 1.4 tok/s** single-stream (21.5× regression; lazy-mode op
   dispatch without graph replay). Unusable for serving.
4. **Scouts + TP8 GLM cannot co-exist on shared cards**: strata accepts a 16 GiB HBM trim, but
   Gaudi device acquisition is exclusive — the TP8 workers for the scout-held cards fail with
   `synStatus=8 [Device not found] Device acquire failed`.

**Final state** (the only layout satisfying the original production requirements): GLM-5.3-Flash
TP8 restored on all 8 cards (restored + re-verified: health, chat smoke, fresh benchmark), scouts
dormant with one-command start (`/data/glm53/start_scout_http.sh scout1 8081` / `scout2 8082`,
verified live during the TP4 window). The "prefer 4 HPUs" condition fails on both branches
(graphs: memory; eager: performance) — that tradeoff is the user's to re-visit, not silently
resolved; starting the scouts requires stopping GLM (commands in FINAL_REPORT.md §5).

Detailed optimization record: `../glm53_optimize/` (FEASIBILITY, BASELINE, PROFILING,
ENGINE_COMPARISON, OPTIMIZATION_LOG, DEPLOYMENT, FINAL_REPORT).

## 6. The optimization program (2026-11-01 → 11-13): MTP serving, the measurement program, and the research fork

After deployment, the program moved to performance: MTP speculative decoding, the
serving levers, and a measurement-first diagnosis of what limits decode. All work is
committed to the research fork `GPUprocess/gaudi-cookbook-fork` (branch
`glm53-flash`; ~120 commits total, 19 on 11-13), with the research docs in
`docs/research/` (RA-037 decode-commercial, BOTTLENECK_RESEARCH rev 1.5+,
EXTERNAL_RESEARCH_PROPOSALS rev 7) and `docs/current_bottleneck.md`.

### 6.1 Production config (adopted, verified)

Serving: `ports/glm-5-3-flash/serving/serve_mtp.sh` (the fork) with:

| Lever | Effect |
| --- | --- |
| `--speculative-config {"method":"mtp","num_speculative_tokens":1}` | MTP k=1: +15.1% decode, bit-identical greedy (A001) |
| `GLM_ROUTER_FAST=1` | the grouped top-k skips the identity group stage (the Q18 lever) |
| `GLM_SPEC_DEVICE_REJECT=1` | the rejection walk on device |
| `GLM_EARLY_DRAFT=1` | the draft for step N+1 enqueued during step N (the pipelined draft; the Q19 lever) |
| `GLM_F2_DRAFT_PIGGYBACK=1` | the draft ids piggyback on the model output (the take_draft_token_ids RPC eliminated) |
| the vectorized parse (in-tree) | the sampler output processing vectorized |
| `--max-num-seqs 32`, `--num-gpu-blocks-override 1280` | ns32 adopted (the target workload: 16 streams) |

**Production numbers (2026-11-13, verified)**: decode bs=1 **40–43 tok/s** sustained
(the MTP acceptance length ~1.95, the per-position acceptance 0.833); @16 streams
**250–258 tok/s** bench-aggregate (**~415–430 tok/s steady** — the bench dilutes with
the EOS early-finishes at ~155 tokens/request); the accuracy suite **97.5%** (39/40;
the one miss = the R003 bat-and-ball reasoning knife-edge — the model's intuitive
"10" vs the retrieved "5" at the low reasoning effort, measured at a **0.75-logit
top-2 gap**, deterministic, survives reboots and the config changes — the honest
baseline, not a regression); the prefill **5,546–7,723 tok/s** (0.5–8K, +209–277%
vs eager, via the prefill-optimized profile: the tensor cache off + the prefill
graph threshold 8,192 — the deploy-time tradeoff, see the docs).

### 6.2 The measurement program (2026-11-13): the @16 step anatomy

A four-boot instrumented session decomposed the @16 decode step completely. All
instruments are deployed in the tree **env-gated default-inert** (the production
behavior unchanged):

| Instrument | What it measures |
| --- | --- |
| `[F4TS]` (GLM_PHASE_TS=1) | the engine phase timestamps: schedule/broadcast/rpc/sample_tokens |
| `[LOOP]` | the engine `_process_engine_step` total: step_fn/outq/post_step |
| `[LAUNCH]` (GLM_STEP_TS=1) | the runner launch path: prep/forward(replay)/logits |
| `[T1B]` | the runner decode branch: launch/sync_wait/post |
| `[SPEC-TS]` (GLM_SPEC_SCHED_TS=1) | the scheduler spec-scheduling census |

**The @16 anatomy (all instruments reconciled)**: the iteration **~72.7 ms** =
the engine loop 61.5 ms (the launch/graph-replay 28.3 — prep 0.00, logits 0.47 —
+ the device drain 27.3 + the sampler/parse 15.6) + the worker post tail ~11 ms.
The acceptance AL ~1.95 confirmed by three independent instruments (the streaming
chunk counts, the drafted metric, the per-position rates). The routed-expert
histogram: **16.1 distinct local experts per rank per layer** (p50 16, max 27,
rank-balanced) — the routing statistics. **The corrections (11-13 late): the AL is
prompt-dependent — 1.95 on the predictable text (the count prompts), 1.68 on the
natural text (the bench's paragraph prompts); the bench arithmetic closes with the
AL ~1.65 and the EOS-drain average running ~10.6.** **The "dense-pass" op
attribution was wrong** (the cited function is MiniMax-M3's SwiGLU-OAI path; GLM
takes the standard fused MoE op), and the 43.6 GB/rank/step figure was the B1
all-read *assumption*, never directly measured — the T1a's own bs=1 device-work
reading (~1.26 ms) implies only ~3 GB, i.e. **the deployed op likely already reads
the routed experts only**. The Q21's remaining value is the fused dequant+GEMV,
not the routed-only read; **the MoE device time and bytes must be measured
directly (one boot) before the Q21 investment decision.** The measurement recipes
(the nonblocking py-spy for the lazy-mode workers; the streaming chunk count = the
per-request step count; the spec-metric windows unreliable for the cadence) are
recorded in `docs/research/RA-037-decode-commercial.md`.

### 6.3 What limits decode, and the path forward

The bs=1 decode is **host-dispatch-bound** (the device 78–100% idle, the AIP-Util
0–22%): the optimization target is the host dispatch rate, not the device. At @16
the step is ~65% worker (the launch replay + the drain + the sampler). The ranked
path (the doc RA-037 §6, re-scoped after the 11-13 corrections):

**⚠ THE Q21 RE-PRIORITIZATION (2026-11-14, the node-count analysis):** the
ranking below is superseded — **the Q21 fused MoE kernel is now THE RANK-1
LEVER**, not for the byte savings (zero, confirmed) but for the **node-count
reduction**: the step is launch-bound at the recipe level (20,485 nodes ×
~1.3 µs = the 26.6 ms device + the 28.3 ms replay, both ∝ the node count);
the MoE region = ~89% of the nodes; **the Q21 fused kernel collapses the MoE
nodes to ~500 = the −86% recipe → the iteration 72.7 → ~34 ms → ~768 tok/s
@16 → the 800 target with the trims.** The items 1-2 below are superseded by
this analysis; the items 3-6 remain valid.

1. **The MoE device-time/bytes measurement first** (one boot: the env-gated event
   timers around the moe_op call at bs=1 vs @16) — it decides whether the Q21
   kernel has any byte-saving value at all (the stock op may already be
   routed-only; the 43.6 GB figure was an unverified assumption). **The 11-13
   inference: the BG-1 = fused (the unfused model requires ~78 ms of pure
   traffic — impossible inside the 72.7 ms iteration), so the stock op reads the
   FP8 once, routed-only, and the Q21 byte case collapses; the boot-window
   benchmark remains as the confirmation instrument.**
2. **Q21 routing-aware fused MoE kernel** (conditional on 1; **deprioritized —
   the byte case collapsed**): the kernel step-1 (the single-expert block-FP8
   GEMV, bit-exact) is committed; the step-2 integration is weeks-class for the
   expected single-digit-percent gain (the dequant compute, already near-free per
   the AB-DEQ parity).
3. **The launch/sampler trims** (the replay 28.3 + the parse 15.6): the graph
   population and the sampler vectorization — **the primary @16 levers**.
   **The parse_output characterization (the py-spy, the final): all 95 parse_output
   samples are the leaf frame itself — the `.cpu()` D2H sync at its first line
   (rejection_sampler.py:273), where the host waits for the rejection sampler's
   device work. The 15.6 ms post term = the sampler's device time + the sync + the
   host parse (already vectorized). The trim paths are bounded: the F1-async class
   was refuted (−5%); the remaining is the `_accept_grid` tiny-kernel chain fusion
   (weeks-class for ~10% @16). The launch/replay term (28.3 ms) is the larger and
   more tractable lever (the graph population).**
4. **The acceptance-side** (the AL 1.68 on the natural text vs 1.95 on the
   predictable): the draft quality/the sampler — the MTP k=2 probe.
   **THE MTP K=2 PROBE RAN (11-13, the serve_mtp_k2ns32.sh, the ns32 + the k=2):
   the verify rows 48, the boot ~35 min (the +48-row buckets). THE RESULTS: the
   bs=1 decode 46.9 t/s (+10-17% vs the k=1's 40-43 — the latency WIN); the @16
   aggregate 200-204 tok/s (−20% vs the 250-258 — the throughput LOSS); the AL
   2.38 (the per-position 0.825/0.559 — the position-1 acceptance decays as
   expected). The mechanism: the tokens +30% (the AL 2.38/1.83) vs the step cost
   +50% (the 48-row verify) = the net −13-20% at the batch. **THE VERDICT: the
   workload-dependent config — the k=1 stays production (the throughput priority);
   the k=2 is the latency-mode option for the low-concurrency/single-user
   serving** (the boot script staged: `serve_mtp_k2ns32.sh` + `boot_k2ns32.sh`).
5. **The MTP k=4 chain graph or a DFlash2-class drafter** for the 800-tok/s
   target (the k=4: the RA-018 chain graph; the drafter: gated on the
   licensing/training-infra decision — the DFlash2 checkpoint is CC BY-NC-ND).
6. **The bucketing strategies: TESTED AND REFUTED** — the pad strategy
   (`VLLM_BUCKETING_STRATEGY=pad`) generated **384 buckets vs the exponential's
   195 (~2× more, not fewer)** with the decode parity (42.3 t/s, the sanity
   391); the bucket-file pinning (the T12) also refuted (the spec-decode seeds
   add buckets). **The bucket-strategy knob does not reduce the graph
   population** — the launch-path lever needs a different mechanism.

Combined (the trims + the acceptance-side): **the iteration ~50–55 ms → ~450–500
tok/s @16** (the 657 estimate assumed the Q21 byte savings — retracted;
proportionally less for the EOS-mixed; the AL 1.68 on the natural text vs 1.95 on
the predictable).

### 6.4 Infrastructure hardened

- **BMC/IPMI remote power control** (Supermicro ATEN at 192.168.0.144): the
  wedged-card recovery validated (the HL-225B cards can wedge at the 50W power cap
  after the rapid kill/reset cycles; `ipmitool chassis power off` + the 30 s drain
  + the power on restores the 600W baseline; `hl-smi -r` cannot).
- **The boot-script library** at `/data/glm53/` (boot_*.sh for every tested config)
  + `kill_glm.sh` (the graceful stop).
- **The research fork** synced to GitHub (`GPUprocess/gaudi-cookbook-fork`,
  the branch `glm53-flash`): the docs, the patches, the kernels, the accuracy
  result artifacts (30+ JSON arms), the boot scripts.

### 6.5 What was tried and dropped (the evidence-first rule)

Every optimization was adopted or dropped on measured evidence, recorded in the
commit messages and the docs: the F1 async-D2H (−5% decode, dropped), the F3
ctx-ladder coarsening (the re-capture tax is bs=3-specific, refuted for bs=1),
the F4/F5 numerics-neutral flags (no measurable win, dropped), the gather_sum
all-reduce invariance patch (the row-order mechanism refuted on this tree — the
A/B ran, neutral), the DFlash2 drafter port (staged ~40%: the boot reaches the
engine init; **dropped per the owner's licensing call** — CC BY-NC-ND; the
self-trained drafter path remains open, gated on the training-infra decision).

### 6.6 Operational notes

- The one production incident (11-13): a debug probe sent SIGUSR1 to the
  EngineCore process (USR1's default action is terminate) — recovered via the
  documented boot procedure; the lesson recorded in the docs. Never signal the
  engine processes.
- The serving restart procedure: `bash /data/glm53/kill_glm.sh` (the graceful
  stop) → `bash /data/glm53/boot_f4ts.sh` (the adopted config) → the health in
  ~20–25 min (the checkpoint load ~14 min + the graph capture).
- The research sessions run on the same node concurrently with serving; the
  measurement boots trade ~25 min of downtime each (documented per boot in the
  git history).
- **The bare HPU tests need the FULL production env** (11-13 late): after
  `kill_glm.sh`, a bare `python -c "matmul"` failed with synStatus 26 (Can not
  compile graph) persisting through the AIP reset, the driver reload, and even a
  full host reboot — **the root cause: the bare python lacked the serving env
  (PT_HPU_LAZY_MODE/LAZY_ACC_PAR_MODE/MEMORY_POOL/OMPI set by the boot scripts);
  the production boot post-reboot compiled with ZERO failures**. The BG-1
  benchmark procedure: run it with the boot-script env exported. The host reboot
  is the clean-slate recovery (5 min boot + the 25 min production boot).
