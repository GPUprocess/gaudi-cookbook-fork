# RA-058..RA-062 — Round 9 (2026-11-23): the Direct-SCaL round

Companion detail to `EXTERNAL_RESEARCH_PROPOSALS.md` revision 18. Owner directive of the
day: *"Look into the direct SCaL that JChen-318 implemented in his engine. Look at the
latest release. He said there was a huge gain in performance. Do a full in-depth analysis
of how the engine works and write a report on it. Then look into how we can implement
SCaL and other potential performance enhancements that are in his engine. Then write
hypothesis and theories and the implementation agent will try to implement."*

Repo state analyzed: **JChen-318/GaudiInfer @ 4d2acbc (2026-11-23, HEAD)** — cloned and
read in full for the direct subsystem (`src/direct_mme.cpp` 1547 ln, `src/direct_attn.cpp`
347 ln, `src/direct_gdn.cpp` 603 ln, `include/g2e/direct_*.hpp`, `tools/scal-g2e-direct.json`,
`third_party/gaudi2_direct/` 14 headers). The direct subsystem landed in commit
`16c3867` (2026-11-21: "Unified 64L Direct MME+SRAM, chained SCAL scheduling, FP8 chunked
prefill, and S1-S4 optimizations") with same-day stability fixes (`a4085c9` the 131K+ HBM
OOM, `50b6032` parity-isolated SOBs) and a cross-path race fix at HEAD (`4d2acbc`:
"sync Synapse before Direct SCAL"). License: **no LICENSE file in the repo → Tier D,
all-rights-reserved: the mechanisms and facts below are analyzed for reference; NO code
from this repo is imported** (standing rule, round 6). Where a mechanism rests on vendor
open source (HCL, driver, SynapseAI_Core, arcboot ROM — all Tier B in the corpus), the
port is written against those Tier B sources, not against his code.

---

## 0. What "Direct SCaL" is (one paragraph)

**SCaL** is the driver's stream/command-queue management layer for the ARC (Advanced
Runtime Controller) scheduler CPUs on the device: host-visible command buffers, streams
owned by scheduler ARCs, memory pools, sync-object (SOB) pools, monitor pools, and host
fence counters (`hcl/dependencies/scal/scal.h`, Tier B — `scal_init`,
`scal_stream_set_commands_buffer`, `scal_stream_submit`, `scal_host_fence_counter_wait`,
`scal_get_handle_from_fd`). The Synapse runtime itself is a client of SCaL: every "recipe
replay" ultimately becomes scheduler-ARC commands in a SCAL stream. **Direct SCaL** is
what GaudiInfer does with that: it stops asking the Synapse runtime to replay compiled
graphs and instead **writes the scheduler-ARC command stream by hand** — from user space,
on a private SCAL stream that it registers in the driver's SCAL config file — dispatching
MME GEMMs (with the exact descriptor programs the compiler generated for them, captured
and replayed) and custom TPC kernels, synchronized purely in hardware via SM2 sync objects
+ monitors, with **on-device chaining** so that an entire 64-layer forward pass executes
as ONE submission with zero host round-trips between layers. "Direct" = direct engine
dispatch, bypassing the graph compiler and the runtime replay path; the MME/TPC engines
and the ARC scheduler firmware are the same ones the Synapse runtime drives.

The measured claim (README, Tier D, same card + same model, his tuned vLLM-gaudi
baseline): **single-stream decode 180 → 357 tok/s (~2×, greedy)**; prefill 8K/32K/64K
1.46/6.49/14.89 s → 0.97/4.41/10.17 s (~1.5×); verification round at 105K context
37.5 → 29.5 ms; 8-lane concurrency ~650-800 tok/s.

Why this matters for us: the round-8 execution model found the wall is the **per-route
expert overhead (the expert term at 10-20× its HBM floor at bs1)** and the **host
boundary + per-layer host sync**. Direct SCaL is a proven-on-Gaudi2 mechanism that
attacks BOTH: the per-expert GEMM becomes a pre-uploaded descriptor dispatch (no fused-op
per-route staging, no per-expert graph node) and the whole step becomes one on-device
chain (no per-layer replay, no host wait). Our RA-055 (command ring) and RA-057 (driver-
level raw submission) were designs-in-theory for exactly these; this engine is the
working reference implementation. RA-057 is superseded by RA-058 below.

---

## 1. Full-depth engine analysis (GaudiInfer @ 4d2acbc)

### 1.1 The two execution paths

The engine (Qwen3.8-27B: 64 layers, 48 GDN (linear-attention) + 16 full-attention,
hidden 4096, FP8 weights, single Gaudi2 96 GiB) runs **two coexisting paths** on one
device:

| Path | Vehicle | Used for |
|---|---|---|
| **Synapse recipes** | Compiler-compiled graphs, disk-cached (`G2E_RECIPE_CACHE`), replayed via the Synapse C API on the runtime's own SCAL streams | embedding, **prefill** (FP8 chunked, FP8 SDPA attention + custom masked-softmax kernel), the **DSpine drafter** (5 Qwen3 layers), **state commit/accept** (the GDN state advance over accepted rows), **sampling** (two-stage topk2 over the 154K-class vocab + FP8-candidate verify head), and the **fallback decode** when the direct path can't build its plans |
| **Direct SCaL** | Hand-written scheduler-ARC command stream on a private SCAL stream; MME descriptors captured from compiled recipes + custom tpc-clang TPC kernels; SM2 SOBs + monitors for in-hardware sync | the hot decode/verify forward: **all 64 layers** at B=8×T=16 (batched verification of 8 requests × 16 draft tokens) and B=1×T=16 (single stream), as ONE chained submission |

The layer pass is split into **spans** ("17 spans + 16 attention cores"): each span = a
`DirectGdnBlock` (a run of GDN layers, each = in-norm → qkv/z GEMMs (MME) → causal-conv +
WY-form recurrence (TPC) → out-proj (MME) → gate/gate_up/dn MLP (MME), with the optional
attention pre-projection `apre` fused at the span's head) + the `DirectAttnCore` for the
span's full-attention layer (the tiled flash-attention engine, §1.5). A 6 MiB **persistent
SRAM zone** holds the inter-layer activations (residual [128, 4096] BF16 = 1.25 MiB, gate
1.5 MiB, packed Q 1.5 MiB, attention output 1.5 MiB) **across all 64 layers**, so the
decode path never round-trips the residual stream through HBM between layers.

Per-step host work (B=8 path, from `qwen35.cpp` `verify_batch`):
1. Build the ids/positions/last-token control block on the host staging buffer.
2. `update_args` on every span: retarget the dynamic 64-bit addresses (per-slot conv/SSM
   state pointers, per-slot K/V cache base addresses, attention output, positions)
   **inside the already-uploaded ECBs** (shadow-copy patch, §1.4).
3. `stream_.synchronize()` — wait for ALL pending Synapse work (prefix-cache load/save,
   prefill, commit, drafter) before the direct path touches shared HBM (the HEAD-commit
   fix; without it the direct path can read half-written KV — a real race they hit).
4. `flush_patches_and_upload`: ALL dirty ECBs of ALL spans + the ids control block in ONE
   coalesced PDMA upload (the MME ECBs are packed contiguously in HBM specifically so the
   whole dirty set is one `synMemCopyAsync` — "10-30× fewer PDMA calls").
5. `run_chain(all 64 layer-plans)`: ONE command-stream submission, on-device chaining,
   ONE host fence wait at the end. The host then does bookkeeping (accept/commit on the
   Synapse path).

### 1.2 The SCAL infrastructure (`DirectRuntime`)

- **Borrowed SCAL instance**: `DirectRuntime(Device)` takes the running Synapse device,
  reads `synDeviceGetInfoV2 → fd`, and calls `scal_get_handle_from_fd(fd, &scal)` — the
  driver's SCAL instance for that device. No second device open, no driver modification.
  (Tier B: `scal_get_handle_from_fd` is in `scal.h`.)
- **The config-file trick**: `tools/scal-g2e-direct.json` is a COMPLETE Gaudi2 SCAL
  config (9 engine clusters: 24 TPC + 2 MME + 5 EDMA + PDMA/ROT/NIC; 6 schedulers; the
  standard pools `host_shared`/`hbm_shared`/`hbm_binary`/`global_hbm`; the standard stream
  sets: compute×3, pdma×12, network GC + scaleup/scaleout×12 each; 4 sync managers) **plus
  one added stream set**: `mme_attention`, base_idx 15, **1 stream** on the
  `compute_media_scheduler`, plus private sync pools (`mme_attention_sos` ≥ 8 SOBs on SM2,
  `mme_attention_monitors` ≥ 64 on SM2) and a private host fence counter
  (`mme_attention_fence0` on SM2). The env var `SCAL_CONFIG_FILE_PATH` is set **before
  device construction** — the driver's HCL loads this config at device open, so the
  Synapse runtime gets its standard streams (unchanged) AND the private stream/pools
  exist for the direct path to borrow. The Synapse runtime is demonstrably unaffected:
  the same device runs prefill/drafter/commit recipes with this config loaded.
- **Its own command ring**: 64 KiB (`kOwnRing`) allocated from the `host_shared` pool and
  bound to the private stream via `scal_stream_set_commands_buffer`. The host writes
  scheduler-ARC commands (8-byte aligned packets) directly into this host-visible buffer
  and rings the doorbell with `scal_stream_submit(stream, pi, submission_alignment)`.
  Completion: `scal_host_fence_counter_wait(fence, credits, timeout)` on the mapped SM2
  host fence counter (no PCIe round-trip; the counter is a host-visible memory location).
- **The scheduler-ARC command language** (Tier B: `gaudi2_arc_sched_packets.h`, HCL —
  SPDX MIT): `FENCE_WAIT=0` (blocks the stream until the stream's fence counter reaches
  the target credits; the target is subtracted on wake — semantics documented in the
  header), `LBW_WRITE=1` (32-bit write to the device register/LBW space — used for
  monitor configuration and SOB manipulation from the command stream itself),
  `FENCE_INC_IMMEDIATE=4`, `NOP=5`, `DISPATCH_COMPUTE_ECB_LIST_V3=7` (dispatch an engine
  command block list to a compute engine group), `PDMA_BATCH_TRANSFER=8`,
  `UPDATE_RECIPE_BASE_V2=9` (set the engine group's recipe base slot that ECB
  `addr_index` references). His programs use NOP, LBW_WRITE, FENCE_WAIT,
  UPDATE_RECIPE_BASE_V2, DISPATCH_COMPUTE_ECB_LIST_V3 — all standard.
- **Monitors and SOBs** (SM2 sync manager, `mmDCORE2_SYNC_MNGR_OBJS_BASE`): a monitor is
  armed (via LBW_WRITE of the config + arm registers: sync-group id, mask, mode, target
  value) to watch an SOB bit; when the SOB reaches the target, the monitor **writes its
  payload to a configured destination**. Two payload destinations are used: (a) a device
  fence register (`kMme0FenceAddr` = the MME engine's fence0 in the sync manager space,
  `tpc_fence_addr(i)` for the TPCs) — the engine-side fence that the next engine's
  `FENCE` packet waits on; (b) **the scheduler ARC's DCCM message queue** — with a
  `MON_EXP_FENCE_UPDATE` message (Tier B: `sched_mon_exp_fence_t`, "Monitor expiration
  structure to update fence... used for updating global fence counters") that increments
  the stream's global fence counter, waking a blocked `FENCE_WAIT` in the command stream.
  Monitor config is cached host-side (`mon_cfg`/`mon_wr`) so steady-state re-arms write
  only the arm register (1 LBW instead of 5). Monitor pools are partitioned MME [0..95] /
  TPC [96..1023] so the cache hits 100%.
- **ECB format** (engine command block, Tier B: `gaudi2_arc_eng_packets.h`): a 512-byte
  meta header (two `ECB_CMD_LIST_SIZE` blocks, a dynamic `ECB_CMD_NOP` with
  switch_cq+yield, then per-engine `ECB_CMD_STATIC_DESC_V2` entries: cpu_index,
  addr_index=6, size, addr_offset, yield on last) + the concatenated per-engine static
  programs, each ≤ 8191 bytes, 8-byte aligned, per-engine 64-byte aligned, whole ECB
  256-byte aligned. The static program is a sequence of `WREG_BULK`/`WREG_32` (writing the
  MME descriptor into the MME QM register region 0xb008-0xb2dc, or the TPC
  `Gaudi2TpcDesc` into the TPC QM region 0xb5dc+, or the `tpc_cmd`/`tpc_execute` control
  registers), an optional `FENCE` (engine-side, id 3; MME also id 2), and a final
  `NOP swtc`. A hard validator (`validate_static_program`) restricts programs to exactly
  this subset — a safety property, not just style.
- **The TPC path**: kernels are compiled with tpc-clang to ELF64; the `.text` (≤ 64 KiB)
  is extracted, uploaded once to the `hbm_binary` pool (cached per object), and launched
  by writing the `Gaudi2TpcDesc` (Tier B: `gaudi2_tpc_descriptor.h` — 16 tensor
  descriptors with base/dims/strides/dtype, the sync object, and the non-tensor
  descriptor: kernel base, kernel_config with rate limits + SRF count, **32 SRFs** =
  scalar register file values passed to the kernel) into each of up to 24 TPCs' QM
  register regions, setting `tpc_cmd` then `tpc_execute=1`. Grid work is split across the
  TPCs by the descriptor's tid_base/tid_size fields (his `add_tpc_grid`: even split of
  `grid_sizes[split_dim]` over the active TPCs).
- **MME path**: the MME descriptor is the full 182-word (728-byte) `MME Desc` (Tier B:
  arcboot ROM `specs/gaudi2/mme.h` `_Desc` — baseAddrA/B/COut0/COut1, brains, header,
  ctrl, tensorA/B/COut, syncObject, aguIn[5×2]/aguOut[2×2], spatialSizeMinus1×3, conv,
  outerLoop, numIterationsMinus1 (the "tetrise" count), sbRepeat, fp8Bias, rateLimiter).
  **The descriptors are not hand-written: they are CAPTURED from the Synapse
  compiler's output** — a recipe compiled with `PRINT_MME_DESCRIPTORS=1` dumps
  `act_<act>_mme_descriptor_dump_<engine>.txt` per MME act; his `MmeTemplate::load_dir`
  parses those dumps (regex over the documented section layout: Brains/Header/Ctrl/
  TensorA/B/Cout/SyncObject/aguIn/aguOut + spatialSize/Conv/OuterLoop/numIterations/
  SBRepeat/Fp8Bias/rateLimiter). At plan-build time the base addresses (0x08-0x24) and
  so0Addr (0x14c) are zeroed and then patched per run. The MME computes from SRAM: the
  DMA engine prefetches HBM→SRAM and the MME slices the GEMM over the SRAM-resident tiles
  (vendor forum, Tier D: "we (Graph Compiler) usually use DMA engine to pre-fetch data
  from HBM to SRAM, meanwhile use MME engine doing GEMM... split GEMM into smaller one so
  that data fits into SRAM") — so the captured descriptor already encodes a
  compiler-chosen tiling/slicing for that shape.
- **Chaining (`run_chain`, the "chained SCAL scheduling")**: with N plans, the host
  writes into the command ring: [clear chain_sob] [arm chain monitor → fence-63 payload
  to the scheduler DCCM MQ] [dispatch plan i] [FENCE_WAIT 63] [clear chain_sob] [arm →
  next plan's fence] [dispatch plan i+1] ... [final plan armed to the host fence
  counter] — then ONE `scal_stream_submit`. The device scheduler ARC walks the stream:
  plan i's last engine sets `chain_sob` → the monitor fires → `MON_EXP_FENCE_UPDATE(63)`
  lands in the scheduler's DCCM MQ → the blocked `FENCE_WAIT(63)` wakes (its target is
  subtracted) → plan i+1 is dispatched. **No host involvement between plans.** The 64 KiB
  ring is pipelined in 20 KiB chunks (max 2 in flight, ≥ 24 KiB cushion) so even a
  64-layer plan set streams. `submit_chain`/`wait_chain` split this for double-buffering
  the CPU patch phase against the device chain (the `*_async` patch upload overlaps with
  the previous chain's execution).
- **The self-test in the constructor**: 2× NOP+SOB warmup submissions (validates the
  stream + fence counter immediately), then a 3-stage chaining self-test (chain_sob →
  monitor → FENCE_WAIT(63) → repeat → host fence) before `chain_enabled = true`.
  `G2E_DIRECT_NO_CHAIN=1` disables; without chaining, `run_chain` degrades to per-plan
  `run()` (submit + host wait per plan).

### 1.3 The plan model (`DirectPlan`)

A plan = an ordered list of **stages**; each stage is either an MME chunk or a TPC grid,
with `wait_sob`/`wait_target` (the stage's engines fence-wait on an SOB before starting)
and `sig_sob` (the completion SOB the stage's signaling descriptors increment; default =
the chain/host SOB). Stages within a plan execute back-to-back in dispatch order; the
MME stage chunks hold ≤ 9 descriptors per engine (one `DISPATCH_COMPUTE_ECB_LIST_V3` per
chunk — i.e., up to 9 MME GEMMs per engine per dispatch). `prepare()` uploads the meta +
code in ONE host→device DMA (with optional readback verify, `G2E_DIRECT_VERIFY_ECB`),
pre-configures all monitor payloads in batched LBW passes, and validates the pool usage
(SOBs ≤ 8, monitors ≤ 1024, MME/TPC partition). `run()` = [flush dirty patches] [clear +
arm the chain/host monitor] [emit: per-stage arm-monitors (re-arm only) + queue_ecb
(UPDATE_RECIPE_BASE_V2 + DISPATCH)] [submit] [one fence wait].

**The patch mechanism** (`patch_addr`/`patch_srf`): at plan build, tagged 64-bit address
locations (MME tensor bases: A at 0xb018, B at 0xb020, COut at 0xb010, COut1 at 0xb008;
TPC tensor bases; TPC SRF values) are recorded as (stage, chunk, engine, byte offset) in
the uploaded ECB's host shadow. Per step, `patch_addr(tag, addr)` memcpys the new
lo/hi words into the shadow; dirty ECBs are re-uploaded coalesced. **This is how a
static, uploaded, verified engine program becomes dynamic per step with zero rebuild and
zero per-step HBM write of program code** — only the changed 8-12 bytes per pointer,
batched.

### 1.4 The attention engine (`DirectAttnCore`) — "Direct MME+SRAM"

Tiled flash attention over up to 8 slots, KV lengths decomposed greedily into
{4096, 2048, 1024} tiles (per 1024 alignment; slot row counts are the padded context
lengths). Per tile:
- **QK^T**: MME GEMM, Q from the SRAM packed-Q zone, K from the HBM KV cache, S → SRAM
  (`s_S[parity]`, [384 rows × tile width × 2B] = up to 3 MiB/buffer; the 384 rows =
  4 KV-head groups × 96 heads per group for this model's GQA layout).
- **Online softmax + accumulate**: ONE TPC kernel (`direct_attn_tile_step`) over 384 rows
  × 24 TPCs: reads S (BF16) from SRAM, updates the running max/sum (F32 in SRAM:
  `s_m`/`s_l`/`s_res` rescale buffers), scales P in-place (to F8 if kv8), and
  accumulates PV^T contribution into the F32 running output `s_O[slot]`; on the last
  tile of a slot it writes the BF16 attention output to the SRAM att zone.
- **PV**: MME GEMM **dual-engine head-parallel** (`add_mme_pair`: MME0 runs KV-heads
  0,2; MME1 runs 1,3 simultaneously — "2× faster"), P from SRAM, V from HBM, PV → SRAM.
- **The pipeline**: double-buffered by parity with **parity-isolated SOBs** (sob 0/1 =
  MME parity completion, sob 2/3 = TPC parity completion): the prologue emits
  QK(0)→S[0] + QK(1)→S[1] back-to-back on the MME; then for k: TPC(k) waits sob[k&1]
  (MME done), does SM(k)+ACC(k-2), signals sob[2+(k&1)]; MME(k+1) waits that TPC sob,
  does PV(k)+QK(k+2), signals sob[(k+1)&1]. **MME and TPC run fully overlapped,
  zero-bubble, synchronized only by SOBs** — the `50b6032` commit added the parity
  isolation after a long-context race. Sub-plans hold ≤ 32 tiles (to fit ≤ 834
  monitors); the sub-plans are chained by `run_chain`.
- **FP8 KV** (`G2E_KV_FP8=1`): K/V/P in F8 (halves the KV HBM reads) — **not applicable
  to us (owner directive: KV stays BF16, RJ-29)**; the FP8 weight path is standard.
- The captured MME templates come in 6 shapes (qk/pv × {1024, 2048, 4096} tile width) ×
  {bf16, f8} — i.e., the Synapse compiler compiled 12 small attention GEMMs offline and
  the direct path re-issues them with patched K/V/S/P addresses.

### 1.5 The GDN engine (`DirectGdnBlock`)

The 48 GDN layers: per layer — in-norm (TPC `norm_mq` = fused RMSNorm + FP8 quant),
qkv/z GEMMs (MME, captured descriptors: `qkv` [14336, 4096] F8, `z`, `ba`), causal conv
+ WY-form gated-delta recurrence (TPC: `wyod` = the WY output; `ba_siluz`; the state
update over the double-buffered SSM/conv state), out-proj (MME `out`), MLP
(MME `guh`/`dn` + TPC `dq_silu`/`dq_mq`/`sig_mq` dequant/quant fusions). The attention
pre-projection (`apre`: in_norm → qkv GEMM → Q/K norm + RoPE (cos/sin tables in HBM) →
Q pack to SRAM + KV scatter to the cache, `kv_scatter_bf16/f8`) and post-projection
(`apost`: sigmoid(gate)·att → o_proj → post_norm → MLP) are fused into the block as
MME+TPC stages, so a "span" = GDN layers + attention core with NO HBM round-trip of the
residual between them. The **state commit** (after verification accepts a prefix) runs on
the direct path too for B=1 (`DirectGdnCommit`: advance the SSM/conv state over the
accepted rows only, `state_commit` + `conv_commit` TPC kernels — the deferred-commit
design from round 6), with the Synapse `spec_commit_recipe` as fallback.

### 1.6 The drafter (DSpine-b16) and the round protocol

The drafter is a **5-layer Qwen3 block** (BF16) whose context is the TARGET's residual
stream captured after layers {1, 16, 31, 46, 61} (5 × [H,H] hidden-feature taps,
hidden-normed, k/v-projected per draft layer, per-layer K/V cache, sliding-window
attention). It drafts a 16-token block [anchor, 15×mask] per round; the target verifies
all 16 in one forward (the B=8×T=16 direct path above). Greedy requests compare with the
target's choices; sampled requests use **rejection sampling** (the target's
distribution is preserved — the candidates come from a sampled draft table,
`dspine_mlx propose.sampled_table` style). Batched drafting across ≤ 8 slots in one
launch. This is the DSpine-class feature-drafter architecture recorded in round 6 (the
GaudiInfer reference for the 3090 self-trained drafter option, RA-049 option (c)) — now
with the concrete geometry: 5 layers, feature taps every 15 layers, 16-token blocks.

### 1.7 Scheduler, prefill, and the rest

- **Scheduler** (`g2e_server`): 8 fixed slots, decode-first steps (all decoding rounds
  first, then prefill FIFO), dynamic KV slots sharing one budget (~51 GiB ≈ 819K tokens
  auto), host-LRU prefix cache (KV chunks + GDN/drafter state snapshots, pinned pool,
  background pinning — the 285 ms 90K agent turn), OpenAI-compatible API.
- **Prefill**: Synapse recipes, FP8 chunked (4K chunks, FP8 SDPA attention + the custom
  masked-softmax kernel — the RJ-27-corroborating 1.8× vs SDPA at long verify width),
  "solo T=4096" direct attention for the single-request prefill tail (commit `50b6032`).
  Measured 7.3-8.5K tok/s at 2K-32K, 5.0K at 128K (vs our 2,857-4,928 — see §3.5).
- **Robustness**: the direct path catches plan-build failures (HBM OOM of the plan
  buffers at 131K+ — `a4085c9`) and falls back to the Synapse recipe path for that round
  ("Direct B=8 attn_core fallback to Synapse"); `G2E_DIRECT_VERIFY_ECB` readback-verifies
  uploaded programs; the static-program validator; the constructor self-test. The 11-23
  HEAD commit is the cross-path ordering fix (§1.1 step 3) — evidence that the hybrid
  (Synapse + direct) operation has real ordering hazards that must be handled explicitly.

---

## 2. GaudiGrimoire cross-check (every mechanism the theories rely on)

| # | Mechanism (as used by GaudiInfer) | Corpus evidence (tier) | Verdict |
|---|---|---|---|
| C1 | SCaL API: `scal_init` (config-file based), `scal_get_handle_from_fd`, `scal_stream_set_commands_buffer`, `scal_stream_submit`, `scal_host_fence_counter_wait`, pools (`host_shared`/`hbm_shared`/`hbm_binary`), SOB/monitor pools, `isDirectMode` fields | HCL `dependencies/scal/scal.h` (Tier B, open GitHub) + `hcl/src/infra/scal/gaudi2/stream_layout.cpp` (Tier B) | **CONFIRMED** — the API is the vendor's own; the config-file init is `scal_init(int fd, const char* config_file_path, ...)` |
| C2 | Scheduler-ARC command set: FENCE_WAIT=0, LBW_WRITE=1, FENCE_INC_IMMEDIATE=4, NOP=5, DISPATCH_COMPUTE_ECB_LIST_V3=7, PDMA_BATCH_TRANSFER=8, UPDATE_RECIPE_BASE_V2=9 | HCL `dependencies/qman_fw/engines-arc/include/gaudi2_arc_sched_packets.h` (Tier B, SPDX MIT) | **CONFIRMED** — all five commands he uses are standard; the header documents FENCE_WAIT semantics explicitly (blocks until credits, subtracts target on wake, suspends if negative) |
| C3 | On-device chaining: monitor expiry → `MON_EXP_FENCE_UPDATE` message to the scheduler ARC's DCCM message queue → global fence counter increment → blocked FENCE_WAIT wakes | HCL `gaudi2_arc_sched_packets.h` `sched_mon_exp_fence_t` (Tier B: "Monitor expiration structure to update fence... structure used for updating global fence counters") | **CONFIRMED** — the message type is a vendor-defined packet; the DCCM MQ is the scheduler control core's message queue (`scal_control_core_info_t.dccm_message_queue_address`, Tier B) |
| C4 | MME descriptor format (182 words: Brains/Header/Ctrl/tensorA/B/Cout/SyncObject/aguIn[5]/aguOut[2]/spatialSize/Conv/OuterLoop/numIterations/SBRepeat/Fp8Bias/rateLimiter); base addresses at 0x08-0x24, so0Addr 0x14c, QM region 0xb008 | arcboot Gaudi2 ROM source `specs/gaudi2/mme.h` `_Desc` + `mme_address_descriptor_regs.h` (Tier B, firmware) | **CONFIRMED** — field-for-field match with his parser |
| C5 | MME execution model: MME computes from SRAM, DMA prefetches HBM→SRAM, compiler slices large GEMMs into SRAM-fitting tiles, MME ∥ DMA pipelined; 48 MB SRAM (2 MB/TPC × 24), 2 logical MME = 4 physical dies | Intel-forum answer (Tier D, vendor employee) + our round-8 hardware facts (Tier B vendor) | **CONFIRMED** — the captured descriptors already encode the compiler's slicing; direct dispatch re-uses it verbatim |
| C6 | TPC descriptor: `Gaudi2TpcDesc` = 16 × `block_tpc_tensor` + `block_sync_object` + `block_tpc_non_tensor_descriptor` (kernel base, kernel_config, **srf[60]** in the ROM struct — 32 used via `num_valid_srfs`); TPC QM register region; `tpc_cmd`/`tpc_execute` | SynapseAI_Core `external_includes/gaudi2/gaudi2_tpc_descriptor.h` (Tier B) + arcboot ROM `specs/gaudi2/tpc_regs.h` `block_tpc` (Tier B: kernel_tensor_0..15 / qm_tensor_0..15 / qm_sync_object / qm layout) | **CONFIRMED** — his writes target the `qm_*` (QMan-programmed) regions, exactly the layout the ROM spec defines |
| C7 | The raw-submission precedent + the minimum path (map CB, configure QMAN PQ, init SM, arm monitor, write descriptor, ring PI) | Corpus finding `gaudi2-command-submission-and-completion.md` (Tier B content / D interpretation): "Engines driven by Synapse also go through the ARC schedulers... the scheduler-mediated path for compute was not traced here" | **CONFIRMED with scope note** — the corpus had documented the transport; GaudiInfer is now the public proof that the full scheduler-mediated compute path works from user space on production Gaudi2 + driver 1.24.x |
| C8 | `PRINT_MME_DESCRIPTORS=1` (the descriptor-dump env var of the Synapse runtime) | **NOT in the corpus** (no hits in Synapse runtime sources / TPC guides / docs) | **UNVERIFIED ASSUMPTION → gap D25.** If our 1.24.1 runtime lacks it, the fallbacks are: (a) the MME descriptors are in the compiled recipe binary (the corpus's TPC-object/recipe format work can extract them — the `DUMP_POST_GRAPHS` census already parses recipe artifacts), (b) request the dump feature from the vendor, (c) generate the descriptors from the Tier B MME spec directly (harder: the slicing/AGU setup is the compiler's value). The spike must establish which path is available. |
| C9 | The 2× decode / 1.5× prefill claims | README (Tier D, self-reported, his node, his tuned vLLM baseline; llm_speedtest harness, 54/54 runs) | **TIER D — plausible, unmeasured on our node.** The mechanism-level evidence (C1-C7) makes the direction credible; the magnitude is his node's. The spike (RA-058) converts the single-GEMM slice to MEASURED on our hardware. |
| C10 | Coexistence of the extended SCAL config with the Synapse runtime | Demonstrated in his engine (same device runs Synapse recipes + the direct stream under the extended config); the config preserves the full standard stream/pool layout and ADDS one stream set + private sync pools | **PLAUSIBLE-BY-CONSTRUCTION, unverified on our stack** → gap D26 (boot our vLLM worker with the extended config; verify the Synapse runtime is unaffected + the private stream is reachable). His config is for SynapseAI 1.24 — our driver is 1.24.1; the schema must be checked against our exact build. |

**What the cross-check does NOT cover:** the exact performance of the per-expert MME
dispatch at our shapes (his model is dense — no MoE; the per-expert dispatch cost at
M=1-16 tokens is inferred from the descriptor-dispatch structure, not measured by him)
→ hypothesis H2, gate in RA-059. And the multi-rank collective interplay (his engine is
single-card) → §3.3.

---

## 3. Transfer analysis: how we implement SCaL on our stack

### 3.1 The integration surface (what has to exist in our vLLM worker)

Our stack: vLLM v1 + vllm-gaudi v0.26.0 (torch-mlir graph capture → Synapse runtime →
HCL → driver 1.24.1), Python on the hot path, custom C++ ops already in-tree (AB-DEQ
TPC dequant op, Q21 GEMV core, the `fixes/` plugin machinery). The Direct-SCaL port
needs:

1. **The extended SCAL config** (D26): take the driver's default Gaudi2 SCAL layout for
   our exact 1.24.1 build (the SDK ships it / HCL generates it; the agent extracts it
   from a normal boot — the config is deterministic per driver+ASIC) and ADD: one
   stream set (1 stream on the compute scheduler, a free base_idx — his used 15), a
   private SOS pool (≥ 16 SOBs, SM2), a private monitor pool (≥ 1024, SM2), one private
   host fence counter (SM2). Set `SCAL_CONFIG_FILE_PATH` in the serve script env BEFORE
   the worker opens the device. Verify: vLLM boots, all recipes run, `scal_get_stream_
   handle_by_name` finds the private stream (from a test .so).
2. **The direct runtime as a custom op / C++ sidecar library** in the worker process
   (dlopen-able, linked against the HCL symbols already loaded by the Synapse runtime —
   `dlsym(RTLD_DEFAULT, "scal_get_handle_from_fd")` is the expected mechanism; if the
   symbols are hidden, link the HCL library explicitly — D26 verifies which). It exposes
   to Python (or directly to the C++ custom op): `init(device_fd)`, `load_mme_template
   (dump_dir)`, `load_tpc_elf(obj)`, `build_plan(...)`, `patch_addr(tag, addr)`,
   `flush_patches`, `run_chain(plans)`.
3. **The MME descriptor capture pipeline** (D25): for each GEMM shape the model needs,
   compile a minimal Synapse recipe (one GEMM) on our node and capture the MME
   descriptors (the `PRINT_MME_DESCRIPTORS` path if our runtime has it; else the
   recipe-binary extraction path). Shapes for the MoE program: the expert GEMM
   `[M, 4096] × [4096, 2048]` and `[M, 2048] × [2048, 4096]` (M ∈ {1, 2, 4, 8, 16, 32, 64,
   128} — the per-expert token counts at bs1/@16), the dense-MLP GEMMs, the attention
   projection GEMMs, and the QK/PV attention GEMMs (the DSA absorbed attention is exactly
   a QK^T + PV pair over the latent — his attention engine's shape, with our 512-dim
   latent instead of 256).
4. **The custom TPC kernels** (tpc-clang, the Q21 infrastructure already in-tree): the
   MoE `expert_sort_stage` (route sort + input-row staging — the RA-053 kernel), the
   `expert_combine` (unsort + residual + norm), the KDA recurrence kernel (RA-054 —
   the WY form is specified in RA-043; his `wyod` kernel is the same architecture,
   Tier D reference only), the DSA indexer + top-2048 + absorbed-attention tile kernel
   (RA-054), the fused norm+quant / silu+quant kernels (his `norm_mq`/`dq_silu` class —
   the Q21/AB-DEQ line already has the rounding-chain discipline), the KV scatter, the
   state commit.
5. **The step wiring** in the HPU model runner: replace the per-step graph replay with
   [host: patch the dynamic addresses (input rows, KV block-table-derived cache bases,
   MTP positions)] → [one flush_patches PDMA] → [run_chain(45 layer-plans)] → [host
   fence wait] → the existing accept/schedule bookkeeping. The MTP draft pass and the
   commit stay on the Synapse path in phase 1 (his engine does exactly this split).

### 3.2 What maps 1:1 and what changes

| His engine (single-card dense 27B) | Our port (8/4-HPU MoE 328B) | Delta |
|---|---|---|
| 64 layers = 48 GDN + 16 attn, all in ONE chain | 45 layers (34 KDA + 11 DSA) + 42 MoE layers; the MoE layer = router (TPC) + 288-expert GEMM batch (MME) + combine (TPC) instead of his dense MLP | The MoE layer is the new stage type — the descriptor-dispatch pattern applies per active expert (≤ 288 descriptors/layer; chunked 9/dispatch/engine as in his `add_mme`) |
| B=8×T=16 = 128 rows per verify | bs1 MTP = 2 rows (1 verify + 1 draft); @16 = 32 rows | The M dimension of the expert GEMMs is 1-2 (bs1) / 2-8 (per expert at @16) — his captured templates are at tile widths 1024-4096; ours need the small-M shapes (the D25 capture list above). The small-M MME GEMM is the HBM-bound regime (the floor we derived in round 8) |
| KV cache in HBM, per-slot contiguous (`k[s].address()`) | vLLM paged KV (block tables) | The per-step patch list is LARGER (per-block addresses from the block table, not per-slot base). His `cap-independent kv_scatter` (a4085c9) shows he hit the same table-size problem. The scatter kernel + the patch budget (64 KiB ring / ~20 KiB chunk) must be sized for our block granularity (128-token blocks: a 32K context = 256 blocks × 11 DSA layers × 2 (K,V) = 5,632 address patches — too many for per-step patching! **The fix is his attention tile structure itself: the tile base address = block_table[slot, tile/1024_blocks] — the TPC tile kernel reads the block table (in HBM, uploaded once per step) and computes the K/V address per tile at run time, so only the block-table base is patched.** This is exactly what his `last_tab` + per-tile address computation does (the `init_args.last_tab` [8, 32] position table + the per-head `k_cache[b][h]` bases). Our version: per-slot block-table base patch (11 layers × 2 = 22 patches) + the tile kernel walks the blocks. |
| Single card, no collectives | 8 ranks (TP8/EP8 production; TP4 4-HPU) — 162 ARs/step on the network engines | **The main architectural delta — §3.3** |
| DSpine 5-layer drafter (separate checkpoint) | MTP layer (in the checkpoint, k=1) | Phase 1: the draft pass stays on the Synapse path. RA-062 (the 16-token drafter) uses his DSpine geometry as the reference architecture for the 3090 training |
| FP8 KV option | **Prohibited (RJ-29)** | KV BF16 only — the tile kernel's kv8 branch is unused |
| 262K context max | 1M context | The plan buffer at 1M: 976 tiles/slot → 31 sub-plans/slot — the private pools (SOBs 8, monitors 1024) and the 64 KiB ring must be checked (his 131K OOM fix is the precedent) → D27 |

### 3.3 The multi-rank problem (the one thing his engine does not answer)

The direct path is per-device. Our 162 all-reduces per step run on the NIC/EDMA network
engines, driven by the Synapse runtime's own SCAL streams (the network scaleup/scaleout
stream sets — present in his config too, unused on his single card). Three designs:

- **(a) Hybrid spans (recommended first).** Split the 45-layer pass into compute spans
  (direct) and collective spans (a minimal Synapse recipe containing only the ARs —
  the 2 ARs per layer at the layer boundaries, or the ARs batched into per-group
  recipes). The direct chain and the recipe chain are interleaved ON-DEVICE: a recipe's
  completion writes its completion-queue SOB; a private monitor in the direct chain
  waits on that SOB before the next compute span; the recipe's launch is triggered by a
  direct-side SOB (the monitor → DCCM MQ mechanism works for the scheduler ARCs that
  own the network streams too — the same FENCE_WAIT chaining, per scheduler). Net: the
  host still submits once (the direct command stream + the tiny recipe launches), the
  per-layer host boundary is gone, the AR device time (3.4 ms) is unchanged but no
  longer forces 322 recipe boundaries. **Risk: the cross-scheduler ordering (compute
  scheduler ARC ↔ network scheduler ARC) is not demonstrated by his engine (single
  card) — the micro-test (RA-058 step 5) must prove the monitor→fence wake works
  across schedulers on our node.** Fallback if it fails: keep the ARs on the recipe
  path with the host submitting the two chains (compute-direct, collective-recipe) in
  one burst — still removes the per-layer host sync, keeps one host round-trip per
  AR-group.
- **(b) ARs inside the direct command stream.** The scheduler-ARC command set includes
  the PDMA/network opcodes (the scaleup send/recv schedulers have `NIC_COLL_OPS`,
  `ALLOC_NIC_BARRIER` — Tier B); the driver's own HCCL is built on these QMAN packets
  (the corpus's hl-thunk network CB tests show the raw construction). Moving the ARs
  into the direct chain removes the last Synapse dependency from the hot path — but it
  re-implements HCCL's barrier/collective logic in our own command streams: high effort,
  high risk (a wrong barrier = a hang or a corrupted reduction). **Recorded as the
  phase-2 option only if (a) leaves > 1 ms on the critical path.**
- **(c) The EP layout changes the collective count** (RA-056): EP4×TP2 halves the
  AR count; with design (a) the residual collective cost is the measured target.

The 4-HPU prototype (EP4×TP1 MoE + TP4 dense, the RA-056 topology) has FEWER and
SIMPLER collectives than 8-HPU TP8/EP8 — and 4-HPU is the production target. **The
port sequence is: 1-HPU dummy spike → 4-HPU model port → 8-HPU.**

### 3.4 What the port is NOT (scope discipline)

- Not a new engine: vLLM stays the server/scheduler/sampler/KV-manager; the direct path
  replaces the per-step model-forward execution only (the same split his engine makes
  with prefill/drafter/commit on recipes).
- Not a driver modification: the SCAL API is the vendor's user-space interface; the
  config file is a vendor-supported mechanism (D26 verifies our build honors it).
  RA-057's driver-rewrite question is answered: **no driver rewrite is needed — the
  right level is SCaL, and a working reference exists.**
- Not an import of his code (no license — Tier D). The port is written against the
  Tier B vendor sources (C1-C7); his repo is the architectural reference and the
  proof-of-existence.

### 3.5 The other performance enhancements in his engine (the owner's second ask)

1. **The 16-token drafter (DSpine-b16)** — the biggest one. His measured acceptance
   (222-272 tok/s decode at a ~23 ms ITL implies AL ≈ 5.5-6.5 at the target verify cost)
   comes from a 16-token draft block from a 5-layer feature drafter. Our MTP k=1 is
   AL 1.88 (MEASURED, the k=1 band). A 16-token drafter with even 40-60% per-token
   acceptance would be AL 7-10 — which makes EVERY step-length target trivial (a
   20 ms step at AL 8 = 400 t/s). This is RA-049 option (c) (the 3090 self-trained
   drafter) with the concrete reference architecture now pinned: 5 layers, feature taps
   at the target's layers {1, 15, 30, 45} (our 45-layer geometry, every ~15th layer),
   sliding-window attention, per-layer K/V cache, block drafting, rejection sampling for
   sampled requests. **RA-062 formalizes it.**
2. **FP8 KV (G2E_KV_FP8)** — 2× KV capacity + faster reads. **NOT APPLICABLE — owner
   directive RJ-29 (KV stays BF16).** Recorded as rejected-for-us, not re-proposed.
3. **The host-LRU prefix cache with state snapshots (285 ms 90K turn)** — corroborates
   RA-041 (the KDA prefix cache lattice): the design is KV chunks + GDN-state
   snapshots + a pinned pool + background pinning; his measured agent-turn TTFT is the
   target value for our TTFT work. No new proposal; RA-041's validation gains a
   reference number (285 ms at 90K / 300 new tokens, Tier D).
4. **The two-stage topk2 sampler (chunked chunk-max top-k over the full vocab,
   `kTopK`/`kTopkChunk` + the FP8-candidate verify head)** — corroborates RA-050
   (the top_p-nucleus-bounded gumbel): his design confirms the census finding that the
   full-vocab sampling work is the dominant TPC cost and that a chunked/bounded
   top-k is the fix. RA-050 stands, now with a second reference implementation of the
   idea.
5. **The masked-softmax verify kernel (1.8× vs SDPA at 131K verify width, Tier D)** —
   corroborates RJ-27's "recorded as an optional component of RA-043/RA-054": in the
   direct path the masked softmax is inside the tile TPC kernel (his `tile_step` does
   the online softmax) — it is part of RA-061, not a separate item.
6. **FP8 chunked prefill (1.5× vs his vLLM at 32K/64K)** — our prefill is 2,857-4,928
   t/s vs the 3,000-6,000 target; the prefill ladder (RA-007) is CLOSED at ~2,857 flat
   through 16K. His chunked-FP8 + direct-attention prefill is a plausible 1.3-1.5×
   prefill lever, but prefill is not the current bottleneck (decode is the priority —
   the owner's standing directive) and the port cost is non-trivial (the prefill tile
  kernels at T=4096). **Recorded as a phase-2 candidate (RA-062's sibling, not ranked
   now); the decode path first.**
7. **The disk recipe cache (~30 s startup, 10 min first compile)** — we already bank
   graph dumps (the 4,173-graph census); vLLM's lazy mode compiles per shape. An
   on-disk compiled-graph cache for our worker would cut cold-start; ops-class, no
   perf impact on warm decode. Not ranked.

---

## 4. Proposals

### RA-058 — The Direct-SCaL foundation spike (the gate for the whole program)

**Status:** PROPOSED (the first HPU action of the round, after the round-8
micro-benchmarks) · **Complexity:** Medium (C++ infra, no model changes) · **Effort:**
1-2 weeks, 1 HPU (no model weights needed for the core gates).

**1. Executive summary.** Bring the Direct-SCaL path into our stack at the minimal
provable scale: the extended SCAL config + the borrowed-handle runtime + the MME
descriptor capture + the TPC ELF upload + the plan/patch/flush/run_chain skeleton,
validated on a DUMMY GEMM (no model, 1 HPU): a single `[16, 4096] × [4096, 2048]` FP8
GEMM compiled as a Synapse recipe, its MME descriptors captured, replayed through the
direct path, and timed + parity-checked against the recipe replay. This spike converts
the Tier D "2×" claim and every C1-C10 assumption into MEASURED-or-refuted on our
node, and it is the prerequisite for RA-059..RA-061.

**2. Current limitation.** Our hot path is Synapse recipe replay: ~74 µs/recipe
submission + the per-node replay cost, 322 recipe boundaries/step (the lazy-collective
split), the host boundary 1-2.5 ms, and the per-route expert staging inside the fused
MoE op (the round-8 wall). The direct path exists as a working reference (his engine)
but not on our node/stack.

**3. Proposed architecture.** Per §3.1 items 1-3: the extended config (D26), the custom
op library (the `DirectRuntime` equivalent — written against the Tier B sources), the
descriptor capture pipeline (D25), and the dummy-GEMM parity test. The step-level wiring
(§3.1 item 5) is OUT of the spike.

**4. Technical feasibility.** C1-C7 (all Tier B) + C10 (his engine demonstrates the
config coexistence on SynapseAI 1.24). Unsupported assumptions: D25 (the capture
mechanism on our runtime), D26 (the config honored + symbols reachable in our worker),
the cross-scheduler chaining (his is single-scheduler; we need the network-scheduler
wake for §3.3(a) — tested in spike step 5 as a bonus, not a gate).

**5. Expected performance impact (the gate, not the payoff).**
- H1a: the single-GEMM direct replay is ≤ 2× the HBM floor at the decode shapes and
  its per-submission overhead is ≤ 50 µs (vs the 74 µs recipe submission) — expected
  PASS from the mechanism (the MME descriptor is the compiler's own program; the
  dispatch is one ECB list); the measured number replaces the round-8 "per-expert MME
  setup 1-10 µs" estimate (the RA-053 micro-benchmark) with the direct-path equivalent.
- H1b: `run_chain` of N GEMM plans shows ≥ 50% reduction in the inter-plan gap vs
  sequential submit/wait (the chaining works; his self-test pattern).
- If H1a/H1b FAIL: the recipe-path program (RA-053/054/055 as written in round 8)
  proceeds unchanged — the spike is a gate, not a commitment.

**6. Implementation complexity.** Medium: ~1.5-2.5 KLoC C++ (the runtime skeleton is
  the bulk; the Tier B packet structs are given), the config JSON (small), the capture
  script (Python, small). No model/numerics risk (dummy GEMM, the parity test is
  exact).

**7. Risks.** (1) D25/D26 failures (the capture mechanism or the config/symbol
  reachability) → the whole direct program is parked; the recipe program proceeds
  (explicit fallback, not a loss — the spike cost is 1-2 weeks of one engineer).
  (2) The extended config destabilizes the Synapse runtime on our 1.24.1 (his is 1.24;
  the schema delta) → the config must be built from OUR driver's default, not copied
  from his file. (3) The HCL symbol visibility (if `scal_*` is not exported from the
  loaded runtime) → link HCL explicitly (the SDK ships it; the driver's userspace
  library is on the node). (4) Watchdog: a stalled direct submission (a monitor that
  never fires) could trip the device watchdog — the constructor self-test + the
  `timeoutUs` monitor fields (Tier B: `scal_completion_group_infoV3_t.timeoutUs`) are
  the mitigation; the spike includes a hang-recovery test.

**8. Dependencies.** 1 HPU for 1-2 weeks (no model); the Q21 tpc-clang toolchain (in
  tree); the Tier B sources (in the corpus; the agent extracts the needed headers —
  they are public GitHub: HCL, SynapseAI_Core, the driver tree).

**9. Implementation instructions.**
1. **Step 0 (no HPU):** extract the needed Tier B headers from the corpus (scal.h,
   gaudi2_arc_sched_packets.h, gaudi2_arc_eng_packets.h, gaudi2_tpc_descriptor.h, the
   asic_reg_structs for sync_object + MME/TPC regions); write the runtime skeleton
   against them (the command emitters, the ECB builder, the validator, the patch
   shadow) — a host-only unit test builds a plan for a fake geometry and checks the
   byte layout (the validator is the test oracle).
2. **Step 1 (D25, no HPU or 1 HPU):** establish the descriptor capture on our
   1.24.1: (a) check the runtime for `PRINT_MME_DESCRIPTORS` (env scan + the recipe
   build logs with it set); (b) if absent, parse the compiled recipe binary for the MME
   descriptor blocks (the corpus's TPC-object work + the `DUMP_POST_GRAPHS` JSONs
   already banked show where the engine programs live); (c) fallback: generate one
   descriptor from the Tier B spec for a known-good GEMM and validate it against the
   recipe's numerical output (the GEMM output is the oracle). Record which path works.
3. **Step 2 (D26, 1 HPU boot):** dump our driver's default SCAL layout (from a normal
   Synapse init — the HCL logs / the config the driver generates); build the extended
   config (+1 stream set, +SOS pool, +monitor pool, +fence counter on SM2); boot a
   minimal Synapse client with `SCAL_CONFIG_FILE_PATH` set: verify (i) a recipe runs,
   (ii) `scal_get_handle_from_fd` + `scal_get_stream_handle_by_name("g2e_probe0")`
   succeed from a test .so loaded in the process, (iii) the 2× NOP warmup submit +
   fence wait completes.
4. **Step 3 (the dummy GEMM parity):** compile `[16, 4096] × [4096, 2048]` FP8 (E4M3,
   block scales as production) as a 1-GEMM recipe; capture its MME descriptor; build
   the direct plan (UPDATE_RECIPE_BASE + DISPATCH, the host-fence wait); run both
   paths ≥ 100×: (i) output parity (exact or ≤ 1 ULP — the descriptor is the same
   program, so exact is expected), (ii) timing: direct per-run vs recipe per-run,
   (iii) the 64-layer-equivalent: chain 42 dummy GEMM plans (the layer count), measure
   the chained total vs 42× sequential — the inter-plan gap is the number.
5. **Step 4 (the TPC side, in parallel):** one trivial tpc-clang kernel (e.g., the
   add1 from the g2-opt-manual, or the Q21 dequant core) launched via the direct TPC
   path: parity + timing vs the same kernel inside a recipe.
6. **Step 5 (bonus, if steps 1-4 pass):** the cross-scheduler test — a recipe that does
   one 2-rank all-reduce (2 HPUs) + a direct plan that fence-waits on the recipe's
   completion SOB via a private monitor (the §3.3(a) mechanism). GO/NO-GO for the
   hybrid-span design.
7. Record as EXP-DS1..DS5 per `docs/BENCHMARK_POLICY.md`.

**10. Validation criteria.** PASS = (i) D25 established (a working capture path,
recorded), (ii) D26 established (config honored, symbols reachable, warmup clean),
(iii) dummy GEMM exact parity, (iv) direct per-run ≤ recipe per-run × 1.5 AND
per-submission overhead ≤ 50 µs, (v) 42-plan chain gap ≤ 50% of sequential, (vi)
no watchdog trip / hang in 1 h of repeated runs. Any of (iv)/(v) failing → the direct
program is re-scoped to the recipe path (RA-053/054/055) with the measured numbers
recorded.

**11. Novelty assessment.** New to this project (RA-057 was the in-theory version of
the raw-submission spike; this is the same idea with a published working reference,
which changes the risk profile from "invent the mechanism" to "reproduce + integrate").
Not a retry of any closed experiment. It supersedes RA-057 (the corpus-only raw
submission design) — RA-057 is marked SUPERSEDED, its corpus building blocks carried
into this proposal's step 0.

---

### RA-059 — The Direct-path expert-streaming MoE (the RA-053 upgrade; the reservoir, attacked in the hardware's native form)

**Status:** PROPOSED (gated on RA-058) · **Complexity:** High · **Effort:** 2-4 weeks
after the spike (the TPC sort/combine kernels are the RA-053 kernels — shared work).

**1. Executive summary.** Re-target the round-8 expert-streaming kernel (RA-053) to the
Direct-SCaL vehicle: per MoE layer, one TPC `expert_sort_stage` (sort the 256 routes by
expert, stage input rows contiguously) → the active experts' GEMMs as **pre-uploaded MME
descriptor chunks** (the captured `[M, 4096]×[4096, 2048]` + `[M, 2048]×[2048, 4096]`
templates, M the per-expert token count; address-patched per step: the per-expert
input/output rows; the weight bases are STATIC — patched once at prepare) → one TPC
`expert_combine` (unsort + residual + norm). The per-expert cost becomes: a descriptor
already in HBM (the WREG program was uploaded at prepare) + one entry in a dispatched
ECB list (amortized 9 experts/dispatch/engine) + the MME's own weight stream. The
300-350 µs/route/layer fused-op staging (the round-8 reservoir) is structurally absent.

**2. Current limitation.** Expert term: 19.5 ms bs1 (slowest rank; 10-20× the 0.9-1.9 ms
HBM floor), ~130-150 ms @16 (5-7× the 19.6-28 ms floor) — per-route staging overhead
(round 8 §0.4). RA-053 (recipe vehicle) projects bs1 1.5-3 / @16 20-30 ms, contingent
on the per-expert MME node setup ≤ 10 µs (the micro-benchmark gate).

**3. Proposed architecture.** The RA-053 node sequence, with the MME GEMMs as direct
descriptor dispatches: the stage order per layer = [TPC sort (waits on the previous
layer's combine SOB)] [MME expert GEMM chunk 0 (engine 0: experts e0, e2, ...; engine 1:
e1, e3, ...) — up to 9 per engine per dispatch; the chunks for the 12-256 active experts
follow back-to-back, each chunk's first descriptor fence-waiting on the sort's SOB]
[TPC combine (waits on the last MME chunk's SOB)] [signal the layer's done SOB]. The
two MME engines run the two expert parity classes in parallel (his `add_mme_pair`
pattern, generalized). The shared expert (1 × 42 layers) rides as one more GEMM in the
chunk (its M is the full row count — the dense shape). The dense-MLP layers (3) and the
attention projections are the same pattern with their own captured templates (RA-061
folds the attention in; phase 1 of RA-059 can keep the attention on the recipe path).
EP layout: phase 1 = the current TP8/EP8 (each rank runs its 36 local experts'
templates; the inter-rank exchange stays on the recipe path per §3.3(a)); phase 2 =
EP4×TP2 (RA-056) with the exchange reduced.

**4. Technical feasibility.** C4 (the descriptor format), C5 (the MME-from-SRAM model —
the captured template encodes the compiler's tiling for the shape; at M=1-16 the GEMM
is weight-stream-bound and the tiling is trivially fine), his engine (the
descriptor-capture + patch + dispatch pattern works on Gaudi2, single card — the per-
expert generalization is the same dispatch, more of them; the ≤ 9/dispatch chunking is
his own bound). The TPC sort/combine kernels are the RA-053 kernels (the Q21
infrastructure). Unsupported assumptions: the small-M descriptor's MME efficiency at
our shapes (H2 — the spike's dummy GEMM at M=16 covers the regime; M=1-8 is
interpolated), and the cross-rank interplay (the spike step 5).

**5. Expected performance impact (DERIVED, the round-8 §0.7 table re-priced for the
direct vehicle).** Per-expert overhead → descriptor dispatch (≤ 1-2 µs amortized, vs
the recipe vehicle's 1-10 µs setup estimate) + the sort/combine TPC (0.5-2 ms, the
same kernel either way) + the patch (0.1-0.5 ms batched, shared with the attention
patches). Expert term: **bs1 1.0-3.0 ms** (floor 0.9-1.9 + sort/combine 0.5-1 + patch
share) vs RA-053-recipe 1.5-3; **@16 20-32 ms** (floor 19.6-28 + sort/combine 1-3) vs
RA-053-recipe 20-30. Step (8 HPU, with RA-060/061/056 at their round-8 values): bs1
8.2-11.2 ms → **168-229 t/s no-drafter** (vs the round-8 P3 176-204 — the expert term
improvement is inside the band; the big win is the mechanism is now PROVEN, not
projected); @16 27.5-38.5 ms → **780-1090 t/s no-drafter** (vs 760-1055). With a
drafter AL 4.0-4.5: 340-400+ bs1 / 1550-2200 @16. **The 8-HPU targets are met without a
drafter at the optimistic end, with a drafter everywhere — same conclusion as round 8,
now on a demonstrated mechanism.**

**6. Implementation complexity.** High (the TPC kernels are weeks-class; the MME wiring
is the spike's machinery reused). The parity gates are the R003/T1 discipline (the
MME numerics are the compiler's — exact; the TPC sort/combine must be bit-exact vs the
recipe-path reference — the contract tests per RA-053 §9).

**7. Risks.** (1) The small-M MME GEMM underperforms the descriptor's tiling at M=1-2
(the compiler tiled for a different M than the runtime per-expert count — the
`interleave_f32_scales`-class fix: capture the templates at the ACTUAL per-expert M
distribution, M ∈ {1..16}, one template per M-bucket; the routing histogram (Q13,
banked) gives the distribution) — the micro-benchmark (the spike's M sweep) is the
gate. (2) The EP imbalance (8.9 ms at bs1, the balls-in-bins) is UNCHANGED by this
proposal — it is a layout problem (RA-056), and the round-8 MTP-pair-overlap placement
recovers ~1 ms either way. (3) The per-step patch volume at @16 (32 rows × 256 routes'
row addresses — the row staging means the per-expert input addresses change per step:
~256 × 2 × 8B = 4 KiB/layer × 42 = 170 KiB/step of patches — that EXCEEDS the 64 KiB
ring budget for command bytes! **The fix: the sort kernel writes the per-expert row
pointers into a HBM table (the staging layout), and the MME descriptors' A-base is
patched to the TABLE base + expert_stride — i.e., the A operand is a 1-row-per-expert
pointer table the MME walks (the tensor descriptor's loopStride/ROI fields support
strided bases; if not, the A tensor is the staged contiguous block — the sort already
stages rows contiguously per expert, so the A base = stage_base + expert_offset,
computed at prepare from the KNOWN staging layout, and only the stage_base + the row
counts are patched). The staging layout is static per bucket shape → the per-expert
A/Cout addresses are prepare-time constants; the per-step patches are only the
stage_base + the row counts (SRFs).** The spike's patch-budget test (step 3, extended
to the staging pattern) must confirm this closes under the 20 KiB chunk. (4) Watchdog
on a stalled expert chunk (a missing descriptor signal) — the per-chunk monitor
timeout (the `timeoutUs` field) + the fallback to the recipe path (his engine's
pattern) is the mitigation.

**8. Dependencies.** RA-058 PASS; the Q13 routing histogram (banked); the RA-053 TPC
kernel designs (shared); the 4-HPU prototype topology for the phase-2 layout.

**9. Implementation instructions.** (1) After the spike: capture the expert GEMM
templates at M ∈ {1, 2, 4, 8, 16, 32, 64, 128} × {both directions} for ONE real expert
weight (the checkpoint's expert 0) — the template capture is the D25 machinery.
(2) The TPC `expert_sort_stage` + `expert_combine` kernels: the RA-053 §8 designs,
contract tests first (CPU references: the 10k-route sort, the combine vs the recipe-
path output on 100 real routing draws). (3) One MoE layer as a direct plan on 1 HPU
(dummy weights, the real shapes): parity vs the recipe-path layer output + the
per-layer timing (the expert term per layer × 42 is the projection). (4) The staging
patch-budget test (risk 3). (5) 42 layers on 1 HPU (the weights don't fit — use a
4-HPU boot: TP4, each rank its 72 experts... no: the 4-HPU TP4 layout = 72 full
experts/rank — the 4-HPU fit check is RA-056's boot). (6) The production A/B: the
direct MoE layers vs the recipe MoE layers, one layer class at a time (the RA-053 §9
gate stack), then 42.

**10. Validation criteria.** (i) per-layer parity (exact MME side; the TPC side ≤ 1
ULP with the R003 suite green), (ii) the expert term per layer MEASURED ≤ the
round-8 floor + 1.5 ms at bs1 and ≤ floor + 3 ms @16 (the 8-rank @16 trace, D24, is
the before/after instrument), (iii) the step-level A/B: bs1 step ≤ 12 ms and @16 step
≤ 40 ms with the 42 direct MoE layers (vs 34.9 / 166.5 current), (iv) the patch
budget closes (step 4), (v) 1 h soak, no watchdog.

**11. Novelty assessment.** RA-053 re-targeted to a proven vehicle; the TPC kernels are
shared, the MME vehicle changes from "nodes in a compiled graph" to "captured
descriptors dispatched raw." The per-expert dispatch at small M is new territory even
for his engine (dense model) — H2 is the open question, and the spike's M sweep prices
it before the kernel work starts.

---

### RA-060 — The full-step Direct chain (the RA-055 realization)

**Status:** PROPOSED (gated on RA-058; lands with RA-059's first layer class) ·
**Complexity:** Medium-High · **Effort:** 1-2 weeks of wiring after RA-059 step 3.

**1. Executive summary.** RA-055 (round 8) designed the command ring as a K-graph
pre-submitted ring + HBM mailbox handoff — a design-in-theory around the D14 vendor
gate. The direct chain makes it unnecessary in that form: the WHOLE step (45 layer
plans) is ONE command-stream submission with on-device FENCE_WAIT chaining (his
`run_chain`, proven), and the "mailbox" is the patched-address table (the next step's
input rows are written by the commit/sampler on the Synapse path into the fixed HBM
buffer the next chain reads — the same fixed-buffer mechanism the graph capture
already uses). The host per step = [patch] [one flush PDMA] [one submit] [one fence
wait] — no K, no ring size, no queue-depth question (the round-8 RA-055 risk #1,
"the runtime re-arms per graph," is dissolved: there is no runtime replay in the
loop).

**2. Current limitation.** The host boundary 1-2.5 ms (post-Q18) + the 322 recipe
boundaries/step (the per-layer host sync) + the D14 gate that forbids the engine-core
async scheduler (the round-7 refutation of the upstream path).

**3. Proposed architecture.** Per §3.1 item 5 + §3.3(a): the 45 layer plans (RA-059 MoE
layers + RA-061 attention layers) chained; the MTP draft pass (phase 1: the Synapse
recipe, launched after the verify chain's fence — its input is the verify's last
accepted row, available at the chain's end; the draft's output feeds the next
step's chain input buffer) and the commit (the Synapse recipe / his
`DirectGdnCommit`-class direct commit as the phase-2 optimization) complete the step
protocol. The device-reject (Q18, in production) stays where it is (the Synapse
sampler path) — it runs between the verify chain's end and the commit.

**4. Technical feasibility.** His `run_chain` (the mechanism, C2/C3) + his B=8×T=16
verify (the shape class: 128 rows; ours is 2-32 rows — smaller, easier) + the D14
gate analysis: the gate forbids the vLLM engine-core async SCHEDULER under spec
decoding; the direct chain changes the SUBMISSION CADENCE of the model forward, not
the scheduler — the vLLM scheduler still runs one step per call, synchronously.
**The gate is not engaged** (verify with the agent: the gate is at
`platform.py` l.276-278 on `async_scheduling and speculative_config is None` — the
direct path leaves `async_scheduling` off; no conflict).

**5. Expected performance impact (DERIVED).** The boundary term 1-2.5 ms → **0.2-0.5 ms**
(patch 0.1-0.3 + submit/fence 0.1) — vs RA-055's projected 0.3-0.7 (the direct chain
removes the poll-amortization term: there is nothing to poll, the fence counter is
memory-mapped). The per-layer host sync (the 322-boundary structure) is GONE from the
compute spans (the collective spans, §3.3(a), keep their boundaries but they are
few — 2 ARs/layer × 45 = the recipe spans, or batched per group). Step bs1:
expert 1.0-3.0 (RA-059) + attention 2.5-3.5 (RA-061) + M 2.5 + AR 3.4 (phase 1,
unchanged) + host 0.2-0.5 = **9.6-12.9 ms → 146-196 t/s no-drafter** (with RA-056's
AR 1.5-2: 7.7-11 ms → 171-244 t/s). @16: 20-32 + 3-4 + 2.5 + 2 + 0.5 = 28-39.5 ms →
**760-1075 t/s no-drafter**. (The round-8 P3 table: 176-204 / 760-1055 — the direct
vehicle lands at the SAME table with the boundary term tighter and the expert term
lower; the net is the optimistic half becomes the base case.)

**6. Implementation complexity.** Medium-High: the step wiring (the runner's forward
path), the span splitting, the cross-scheduler chaining (the spike step 5), the
commit/draft protocol. No new kernel work (RA-059/061's kernels).

**7. Risks.** (1) The cross-scheduler ordering (§3.3(a)) — the spike step 5 is the
gate; the fallback (host-submitted two-chain burst) costs ~0.3-0.5 ms of the boundary.
(2) The MTP draft on the Synapse path adds its own boundary (the draft pass ~1-2 ms
device + its recipe submission) — phase 2 moves the draft to the direct path (the MTP
module is 1 layer — a small plan set; his drafter-on-recipes pattern shows either
works). (3) The HEAD-commit race class (synapse-before-direct syncs) — the protocol
must enumerate every Synapse→direct and direct→Synapse handoff and put a
fence/sync at each (his 11-23 fix is the existence proof that this class of bug is
real). (4) The 1M context plan fit (D27).

**8. Dependencies.** RA-058 PASS + spike step 5; RA-059 step 3 (one direct layer class
working); the Q18 device-reject (in production).

**9. Implementation instructions.** (1) The span-split design doc (which layers are
compute spans, where the collective spans go, the SOB map — 8 SOBs may be tight for
45 layers × 2 spans: the per-layer done-SOB can be REUSED (the chain's fence semantics
consume it) — the design must show the SOB allocation closes over the 8 private SOBs +
the completion-queue SOBs the monitors can watch). (2) The step protocol state machine
(verify chain → device-reject → commit → draft → next-chain-input-write, with the
fence at each handoff). (3) The A/B: the full step direct vs the current recipe step,
bs1 and @16, the D24 8-rank trace as the instrument. (4) The 1 h soak + the
watchdog-recovery drill (kill a monitor's target; verify the timeout + the fallback).

**10. Validation criteria.** (i) step parity (the R003 suite + the 1M needle green),
(ii) bs1 step ≤ 13 ms (no drafter) and @16 ≤ 40 ms MEASURED, (iii) the host per-step
time (the patch+submit+fence window) ≤ 1 ms MEASURED (the py-spy class measurement of
round 7), (iv) no watchdog, no hang in the soak, (v) the fallback path (direct
disabled by env) runs production-identical (the kill switch).

**11. Novelty assessment.** RA-055 superseded in mechanism (the ring+mailbox design is
replaced by the proven chain+patched-table design; the GOAL — the boundary term to
≤ 1 ms — stands). The D14 gate is analyzed as not engaged (the scheduler is untouched)
— this must be re-verified by the agent against the gate's exact condition (it gates
`async_scheduling`, which stays off).

---

### RA-061 — The SRAM-resident residual + the attention tile engine (the RA-054 upgrade)

**Status:** PROPOSED (gated on RA-058; independent of RA-059, parallel track) ·
**Complexity:** Medium-High (2-3 TPC kernels) · **Effort:** 2-4 weeks.

**1. Executive summary.** Re-target the round-8 attention consolidation (RA-054) to the
direct vehicle: (a) the **6 MiB persistent SRAM zone** holds the residual stream
([rows, 4096] BF16), the packed Q, and the attention output ACROSS ALL 45 layers of the
decode pass — the inter-layer activation HBM round-trips disappear from the hot path;
(b) the **DSA attention** (the 11 layers) as his tiled MME∥TPC pipeline: QK^T (MME,
captured template over the latent [rows, 512] × [context, 512]) → the tile TPC kernel
(indexer scores + top-2048 + online softmax over the selected slots — the indexer read
is the only context-scaling term, the round-8 §0.5 1M design) → PV (MME, dual-engine)
→ accumulate (TPC) — S/P never touch HBM; (c) the **KDA layers** (the 34 layers) as
the RA-043/RA-054 WY-form recurrence in TPC (the state update over the double-buffered
SSM state — his `DirectGdnBlock` is the same architecture, Tier D reference) with the
projections as MME descriptor dispatches (RA-059's machinery).

**2. Current limitation.** Attention 6.9 ms/step (MEASURED, ~90% overhead — the
projection GEMMs are 1.2 ms of it); the KDA recurrence is a ~20-node graph loop per
layer (the RA-043 finding); the 1M decode is bounded but not yet built (the indexer
term 11 GB → 5.4 ms at 1M, DERIVED).

**3. Proposed architecture.** The round-8 RA-054 kernel designs (the KDA fat kernel, the
DSA fat kernel, the projection tails) as direct TPC stages + MME descriptor stages in
the layer plans; the SRAM zone allocation (the 6 MiB layout: residual 1.25 MiB at
rows=128 — at our @16 rows=32 it is 0.25 MiB, so the zone shrinks; the tile buffers
S[2]/P[2] at tile width 2048: 32 rows × 11 DSA layers... the tile buffers are PER
ATTENTION CORE (one at a time in the chain — his `kDirectSramWorkOff` scratch is
reused across cores; at 1M the tile width stays 4096 (the block size) and the buffer
count is the same — the CONTEXT length only changes the tile COUNT, not the buffer
SIZE: the 1M fit is structurally fine, D27 confirms the plan-buffer side).

**4. Technical feasibility.** His `DirectAttnCore` (the MME∥TPC tile pipeline, the
parity-isolated SOBs, the SRAM arena) + his GDN block (the WY recurrence) + C4/C5/C6 +
the round-8 RA-054 kernel designs (the math is specified there; the WY form is
RA-043's). The DSA absorbed attention is QK^T + PV over the 512-dim latent — the
EXACT shape class of his attention engine (his D=256; ours D=512 — the tile width
budget halves: 4096→2048 tiles, the same SRAM math). Unsupported assumption: the
indexer+top-2048 in the tile TPC kernel at 1M (the per-tile top-k over 2048 selected
slots from a 1M-scored context — the selection is GLOBAL over the context (top-2048 of
1M scores), not per-tile! **The design consequence: the DSA decode at 1M needs the
full 1M indexer score pass (11 GB read) + a global top-2048 + a GATHER of the 2048
latent slots — the "tile" is the 2048 gathered slots (constant size), and the indexer
pass is its own MME/TPC stage (the scores are [rows, 1M] — at rows=1 that is 1M
scores × 11 layers: the score pass is the 11 GB read, done as a streamed MME GEMV
(rows × 1M) × (1M × 128 indexer key) — the round-8 §0.5 arithmetic is unchanged; the
direct vehicle just executes it as descriptor dispatches + the TPC top-k. The
GATHERED 2048 slots go to HBM scratch (2048 × 512 × 2B × 11 layers = 23 MB — fine)
and the PV stage reads them tiled.** This is a design delta vs his engine (his
attention is dense-over-context; ours is sparse-select) — the report's §3.2 table
missed this; it is called out here as the one place the DSA is NOT a 1:1 port.

**5. Expected performance impact (DERIVED).** Attention term: 6.9 → **2.0-3.5 ms**
(projection GEMMs as MME dispatches 1.0-1.2 — the descriptor removes the node overhead
vs RA-054-recipe's 1.2 estimate; the KDA recurrence TPC 0.3-0.5 × 34/45 share; the DSA
tile pipeline 0.5-1.5 × 11/45 share at 2.8K; the residual SRAM-residency removes the
inter-layer activation round-trips (~0.1-0.3 ms at @16)). At 1M: +5.4 ms (the indexer
pass — unchanged from round 8) → the 1M bs1 step at the full program: 9.6-12.9 + 5.4
= 15-18.3 ms → **103-125 t/s at 1M** (the round-8 table: 101-125 — confirmed).

**6. Implementation complexity.** Medium-High: the KDA WY TPC kernel (the RA-043 spec —
the agent's staged spec `RA043_wy_spec.md` is the design input; his `wyod` is the
reference implementation, Tier D), the DSA tile kernel (the indexer + top-2048 +
gather — new), the SRAM zone allocator (small), the MME template captures (the
projection shapes + the QK/PV latent shapes).

**7. Risks.** (1) The KDA recurrence parity (the round-8 RA-054 risk #1, unchanged: the
rank-1 update rounding boundaries — the R003/T1 discipline, the 10k-token walk gate).
(2) The DSA top-2048 slot identity (the gate: 100 real sequences, exact slot set —
the indexer scores' near-ties at 1M are the R003-class knife-edge; the tie-break must
match the recipe path bit-exactly — the SRF-passed comparison constants). (3) The SRAM
zone contention: the direct path's 6 MiB + the TPC kernels' per-TPC 2 MB working
space + the Synapse runtime's own SRAM usage (the DMA prefetch buffers of the recipe
path — prefill/drafter run on the same device!) — **the SRAM budget must be audited
with the Synapse runtime present** (his single-engine process has no contention;
ours shares the card with the recipe path) → D28. (4) The FP8 weight path in the MME
descriptors (the production FP8 GEMM is native — the captured templates carry the
FP8 config; the round-8 "FP8 in place" property holds by construction).

**8. Dependencies.** RA-058 PASS; the RA-043 WY spec (staged); the D24 trace (the
attention-term instrument at @16).

**9. Implementation instructions.** (1) The SRAM zone audit (D28, no HPU time: the
runtime's SRAM allocation from a characterization boot + the zone layout). (2) The
KDA WY kernel per the RA-043 spec + contract tests (the CPU reference walk). (3) The
DSA tile kernel: the indexer score stage (the MME GEMV template at the 128-dim
indexer), the global top-2048 (TPC, the two-stage chunked top-k — the RA-050
corroborating design), the gather (TPC/EDMA), the PV tile stage (the MME dual-engine
pattern). (4) The projection templates (the qkv/o/gate/up/down shapes at the decode M
sweep). (5) One KDA layer + one DSA layer as direct plans (1 HPU, dummy weights):
parity + timing. (6) The 45-layer attention pass on 4 HPUs: parity vs the recipe
attention + the attention-term measurement.

**10. Validation criteria.** (i) the parity gates (the KDA walk, the DSA slot identity,
the projection exactness — per §7), (ii) the attention term ≤ 3.5 ms bs1 and ≤ 4.5 ms
@16 MEASURED (D24 instrument), (iii) the 1M decode: the indexer pass ≤ 6 ms MEASURED
(the 11 GB at ≥ 1.8 TB/s), (iv) the SRAM audit clean (no runtime contention, D28).

**11. Novelty assessment.** RA-054 re-targeted + the SRAM-residency addition (new —
his 6 MiB zone is the reference) + the DSA-sparse delta called out (§4 item 4) — the
one place where our model's attention is structurally different from his dense
attention and the design must be original (the top-2048 global select + gather).

---

### RA-062 — The 16-token drafter (the DSpine-class reference architecture for the RA-049 decision)

**Status:** PROPOSED (the owner-decision input; training on the 3090 is the owner's
hardware call, RA-049 option (c)) · **Complexity:** High (a trained model + the
draft/verify integration) · **Effort:** the integration 2-3 weeks; the training is the
3090's problem (his DSpine-b16 ships pre-trained for the 27B — the architecture is
pinned, the training data is our serving logs).

**1. Executive summary.** The round-8 table shows the drafter is the 4-HPU @16 answer +
the margin (AL ≥ 4.0). The measured AL of our MTP k=1 is 1.88 — the k=1 band (the W7
ceiling: even k=4 graphs at 1.87). The only way to AL ≥ 4 is a DEEPER draft: a
16-token block drafter. His engine ships exactly this (DSpine-b16: 5 layers, feature
taps at the target's layers {1,16,31,46,61}, sliding-window attention, rejection
sampling, his measured decode 222-272 tok/s implies AL ≈ 5.5-6.5). This proposal
formalizes the architecture for our model (45 layers: taps at {1, 15, 30, 44}; 16-token
blocks; the MTP layer's hidden state as the anchor feature) and prices it against the
round-8 table.

**2. Current limitation.** AL 1.88 (MEASURED, the k=1 band, temperature/prompt-
invariant — the W7 ceiling). 200 bs1 at AL 1.88 needs a 9.4 ms step (the round-8
arithmetic: unreachable even at the direct-program's 8.2-11.2 ms — the optimistic end
just reaches it); the 4-HPU @16 (800) at AL 1.88 needs ≤ 60 ms — the program lands
40-55 ms (borderline); the drafter is the robust answer.

**3. Proposed architecture.** A 5-layer BF16 drafter (his geometry, scaled to our
hidden 4096): input = the target's residual taps at layers {1, 15, 30, 44} (5 taps
including the embedding level, his {1,16,31,46,61} pattern) + the anchor token; per
draft layer: a k/v projection of the taps (the per-layer K/V cache over the context —
sliding window, his `window` parameter) + a Qwen3-class attention + MLP (his layers
are Qwen3 dense — ~[4096, 4096] × 3 + attn per layer ≈ 80-120 MB BF16 × 5 layers ≈
0.5-0.6 GB total — fits in the spare HBM); it drafts 16 tokens (greedy or the sampled
table); the target verifies all 16 in one forward (the direct chain at rows = 17: the
16 draft + 1 anchor — his B=1×T=16 shape exactly). Rejection sampling for sampled
requests (the target's distribution is preserved — the mechanism his README states and
the round-6 RJ-11 class analysis requires for any drafter). The draft cost: 5 layers ×
16 rows ≈ 5/45 of the step × 16/2 row ratio ≈ 0.9-1.5 ms device (DERIVED from the
program's step arithmetic) — the round-7 MTP-draft-cost finding (the 2nd draft pass is
the expensive one; a 5-layer drafter's marginal pass is ~1/9 of the target layer cost)
corroborates this being CHEAPER than MTP k=4 (4 full target layers).

**4. Technical feasibility.** His engine runs this architecture in production (Tier D,
the mechanism is demonstrated end-to-end: draft → verify → rejection → commit, on
Gaudi2, with the acceptance numbers implied by the tok/s table). The training data:
our serving logs (the target's hidden states at the tap layers + the accepted tokens —
the DFlash2-class "train on target features" pattern, RA-049 option (c), the 3090
route). The verify side is RA-060's direct chain at the 17-row shape (the bucket
exists: our MTP verify buckets already cover k+1 rows — the 17-row bucket is a new
capture, one shape). Unsupported assumptions: the acceptance of a 5-layer drafter on
GLM-5.3-Flash (his 27B numbers are the prior: AL ≈ 5.5-6.5; GLM-5.3's next-token
entropy is different — the honest band is AL 4-8; even AL 4.0 is the round-8 table's
"drafter class" row).

**5. Expected performance impact (DERIVED, the round-8 table's drafter column, now
with the drafter cost made explicit).** Step (8 HPU, program + drafter, AL 4.5):
bs1 8.2-11.2 + 1.0 (draft) = 9.2-12.2 ms × (4.5/1.88 row-equivalent...) — the
round-8 table's formula: t/s = AL × 1000 / step_ms → **410-489 t/s bs1** (the table's
P3 drafter row: 410-489 — the drafter cost is already inside the table's "the draft
is free" assumption only at ~1 ms — CONFIRMED consistent); @16 (AL 4.5 → 72 tok/step
/ 16 seqs... the table: 1820-2530). 4-HPU @16: **1315-1790** (the 800 target met with
2-2.2× margin). At AL 4.0 (the conservative drafter): bs1 330-390, @16 1470-1990,
4-HPU @16 1080-1470 — **every target met at AL ≥ 4.0.**

**6. Implementation complexity.** High: the drafter model (training — the 3090, the
owner's call), the draft/verify integration (the 17-row bucket, the rejection
sampling, the K/V management for the draft's sliding window), the draft-on-direct
(phase 2; phase 1 = the drafter on the Synapse recipe path, his pattern).

**7. Risks.** (1) The acceptance below 4.0 (the GLM-5.3 prior is unknown — his 27B
number is the only data point; the 3090 training + a 1K-prompt acceptance measurement
BEFORE the integration work is the de-risking gate — the round-7 W7 discipline:
measure the AL of the DRAFT before building the verify side). (2) The draft cost at
@16 (16 seqs × 16 draft rows = 256 draft rows/step — the draft pass at 256 rows is
5/45 × 256/32 of the step ≈ 12-18 ms — THAT DOUBLES THE STEP. **The drafter at @16
must be batched across the 16 seqs as ONE 256-row draft forward (his `draft_batch` —
one launch for all slots) and its cost re-priced: 5 layers × 256 rows ≈ 5/45 × 8 ×
(step/16)... the honest number: the draft forward at 256 rows ≈ the target forward at
32 rows × (5/45 layers) × (256/32 rows) = 1/9 × 8 = 0.89× the target step — i.e., the
draft nearly DOUBLES the @16 step. His engine lives with this (the 8-lane 650-800
tok/s IS with the drafter at 8×16 rows — his verify at 105K is 29.5 ms and his
8-lane decode is ~23 ms ITL at 16 tokens/round → the draft+verify ≈ 23 ms/16 tok
implying the draft is a minority — because his drafter is 5 layers of a 64-layer
target AND the draft rows attend only a sliding window (not the full context) — the
draft cost scales with the WINDOW, not the context. At 2.8K-32K context with a
4K-class window: the draft ≈ 5/45 × (256 window-rows / 32) × (window-attn vs
context-attn share ~0.3-0.5) ≈ 0.3-0.5× the step → the @16 step 28-39.5 + 8-18 =
36-57 ms → 530-1000 t/s at AL 4.5... which EATS the drafter's gain at @16. **The
hypothesis H6 must be measured, not assumed: the drafter's value at @16 depends on
the window size vs the context; at 2.8K context (the bench) the window ≈ the context
and the draft cost is ~0.4-0.6× the step — the net @16 gain at AL 4.5 is 72/
(step+draft) vs 30.08/step: (36+14) ms → 720 t/s vs the no-drafter 780-1075 — THE
DRAFTER CAN LOSE AT @16.** The drafter's clean win is BS1 (the 4-HPU bs1 200 target
+ the 8-HPU bs1 margin) and the LOW-BATCH @4-8 (the 4-HPU production shape at typical
load). The round-8 table's "@16 with drafter 1820-2530" row ASSUMES the draft is
cheap — this proposal corrects that assumption: the @16-with-drafter number must be
re-derived with the measured draft cost (H6). This is the round's most important
correction to the round-8 arithmetic.**
(3) The training quality (the 3090 on our logs — the DFlash2 B1-B5 blockers from
round 7 do NOT apply (no 1Cat GDN op needed — the drafter is a separate small
model; the feature taps are standard hidden states — the B1 blocker was the DFlash2
HPU port's compiled GDN op; a DSpine-class drafter on our MTP tap avoids it).
(4) The 0.5 pp accuracy budget: the rejection sampling preserves the distribution
EXACTLY (his README's stated property) — the risk is the drafter's degeneration under
the sampled table (the round-7 degeneration suite covers it).

**8. Dependencies.** The owner's 3090 call (RA-049); our serving logs (the feature
taps require the target's hidden states at the tap layers — the log capture is a small
runner patch, runs under the current production); RA-060 (the 17-row verify bucket).

**9. Implementation instructions.** (1) **The acceptance probe FIRST (the W7
discipline, 1-2 weeks, 3090 + 1 HPU):** train the 5-layer drafter on a 100K-step
slice of our logs (the tap features + next-16 tokens); measure its AL on 1K held-out
prompts (the greedy 16-block acceptance vs the target) — the gate: AL ≥ 4.0 before
any verify-side work. (2) The 17-row verify bucket (the capture + the parity suite).
(3) The draft/verify integration (phase 1: the drafter on the Synapse recipe path —
his pattern; the rejection sampling per the round-7 protocol). (4) The draft-cost
measurement at bs1 and @16 (the H6 numbers — the window-size sweep: 4K/16K/64K
windows). (5) The @16 net-gain re-derivation (the corrected table row).

**10. Validation criteria.** (i) the probe AL ≥ 4.0 (else: parked, recorded, the
MTP-only program stands — the round-8 no-drafter numbers are the floor), (ii) the
rejection-sampling distribution test (the round-7 protocol, exact), (iii) the
bs1 step with drafter ≤ 13 ms at AL ≥ 4.0 → ≥ 300 t/s, (iv) the @16 net gain
MEASURED (the corrected table), (v) the 1M needle + the R003 suite green.

**11. Novelty assessment.** RA-049 option (c) (the 3090 self-trained drafter) with the
architecture now Pinned to a shipped reference (his DSpine-b16 — the round-6 note
recorded the "DSpine-class feature drafter (5-18, the GaudiInfer reference
architecture)" — this round has the geometry, the measured performance, and the
rejection-sampling design). The correction to the round-8 @16-with-drafter arithmetic
(risk 2) is the new content: the drafter's @16 value is now an open measurement
(H6), not an assumption.

---

## 5. Hypotheses (the falsifiable set, for the implementation agent)

| ID | Hypothesis | Falsified if | Test |
|---|---|---|---|
| H1a | The single-GEMM direct replay (captured MME descriptor) is ≤ 2× the HBM floor at the decode shapes with ≤ 50 µs per-submission overhead | the direct per-run > 1.5× the recipe per-run OR > 2× the HBM floor | RA-058 step 3 |
| H1b | On-device chaining cuts the inter-plan gap ≥ 50% vs sequential submit/wait | the 42-plan chained total > 1.5× the 42× sequential | RA-058 step 3(iii) |
| H2 | The per-expert MME dispatch at M=1-16 is ≤ 1-2 µs amortized (the descriptor is pre-uploaded; the dispatch is amortized over the chunk) — i.e., the expert term reaches floor + sort/combine + patch (bs1 1.0-3.0 ms) | the per-expert dispatch > 5 µs (the small-M MME underperforms its tiling) | the spike M sweep (RA-058 step 3, M ∈ {1,2,4,8,16}) + the RA-059 one-layer A/B |
| H3 | The full-step direct chain removes ≥ 80% of the host boundary + the per-layer host sync (the host per step ≤ 1 ms) | the host per-step window > 1.5 ms (the patch upload or the fence wait dominates) | RA-060 step 3 (the py-spy-class measurement) |
| H4 | The DSA attention as the MME∥TPC tile pipeline (the gathered-2048 variant) is ≤ 1.5 ms/layer at 2.8K and the 1M indexer pass is ≤ 6 ms | the DSA layer > 2 ms OR the 1M pass > 7 ms | RA-061 step 6 + the D17 1M window |
| H5 | The cross-scheduler chaining (compute-scheduler ARC ↔ network-scheduler ARC, via the private monitor → the completion-queue SOB) works on our node with the extended config | the monitor never wakes the network scheduler's FENCE_WAIT (the §3.3(a) mechanism fails) | RA-058 step 5 (2 HPU, one 2-rank AR recipe + a direct plan waiting on its completion SOB) |
| H6 | The 16-token drafter's NET gain at @16 is positive: the AL gain (1.88 → 4.0+) outweighs the draft cost (the 256-row draft forward, window-limited). **Round-8 correction: this is NOT assumed anymore — the round-8 table's @16-with-drafter row (1820-2530) assumed the draft was cheap; the honest estimate is the draft ≈ 0.3-0.6× the @16 step, which can make the net @16 gain ZERO or NEGATIVE at AL 4.0-4.5.** The drafter's clean wins are bs1 (all 8-HPU/4-HPU bs1 targets) and @4-8 | the measured @16-with-drafter t/s < the @16-no-drafter t/s at the same step program | RA-062 step 4 (the draft-cost window sweep + the net re-derivation) |
| H7 | The extended SCAL config (our driver's default + 1 stream set + private pools) is transparent to the Synapse runtime in our 1.24.1 worker (all recipes run, no perf delta on the recipe path, the private stream is reachable) | any recipe failure / > 2% recipe-path perf delta / the stream unreachable | RA-058 step 2 (D26) |

---

## 6. New gaps (for the implementation agent's measurements)

| # | Question | Why it matters | Method | Effort |
|---|---|---|---|---|
| D25 | **The MME descriptor capture mechanism on our 1.24.1 runtime**: does `PRINT_MME_DESCRIPTORS=1` (or an equivalent) exist in our Synapse runtime build? If not: can the MME descriptor blocks be extracted from the compiled recipe binary (the banked `DUMP_POST_GRAPHS` artifacts / the corpus's TPC-object format)? | RA-058's step 1 — the ENTIRE direct program rests on getting the compiler's MME programs without the runtime's dump feature | (1) env scan of our runtime + a recipe build with the var set; (2) the recipe-binary format study (the census tooling already parses the artifacts); (3) the vendor inquiry (the TPC team) | 1-3 days, no HPU time |
| D26 | **The extended SCAL config on our 1.24.1 driver**: (a) is `SCAL_CONFIG_FILE_PATH` honored by our driver's device-open path (the HCL in our 1.24.1 reads it — check the loaded library's env scan); (b) does the Synapse runtime work unchanged with the extended config (the standard streams preserved + the private additions); (c) are the `scal_*` HCL symbols reachable from a custom op .so in the vLLM worker (exported from the loaded libHCL, or needing an explicit link)? | RA-058's step 2 — the integration surface. A failure here parks the whole direct program (the recipe program proceeds) | (1) dump the default config from a normal boot (the HCL logs / the generated config); (2) build the extended config (+1 stream set, +pools, +fence on SM2); (3) boot a minimal Synapse client with it set; (4) the test .so: `dlsym(RTLD_DEFAULT, "scal_get_handle_from_fd")` + the stream lookup + the warmup | 2-3 days, 1 HPU boot |
| D27 | **The 1M-context direct plan fit**: at 1M context, the DSA tile count per slot (244-976 tiles), the plan-code HBM (the sub-plan ECBs), the private pool usage (8 SOBs / 1024 monitors — his 131K+ OOM was the plan-buffer HBM), and the 64 KiB ring / 20 KiB chunk budget at the 1M patch set (the per-slot block-table bases: 11 DSA layers × 2 = 22 patches — small; the tile kernel walks the blocks) | The 1M objective under the direct program (the round-8 1M decode design's vehicle) | the plan-builder dry-run at the 1M shape (no execution: the buffer accounting) + the D17 window's measurement | included in RA-061 |
| D28 | **The SRAM budget with the Synapse runtime present**: the runtime's own SRAM usage (the DMA prefetch buffers of the recipe path — prefill/drafter/commit run on the same device) + the direct path's 6 MiB zone + the TPC kernels' per-TPC working space — does it fit in 48 MB with headroom, and is there a contention mode (the recipe path's DMA vs the direct path's MME reads from SRAM)? | RA-061's SRAM-residency is the new claim vs the recipe path; his single-engine process has no contention, ours shares the card | the characterization boot: the runtime's SRAM census (the `scal_get_sram_size` + the pool info) + the zone layout audit + a mixed prefill+decode soak with the SRAM usage telemetry | 1 boot window |

---

## 7. Execution order (the do-first sequence)

1. **RA-058 the spike (1-2 weeks, 1 HPU, no model)** — steps 0-5 above. This is the
   gate for the whole direct program; it is INDEPENDENT of the round-8 micro-
   benchmarks (the RA-053 MME-setup micro-benchmark and the RA-055 ring micro-test
   are now SUBSUMED by the spike: the spike measures the direct-path equivalents —
   the per-expert dispatch (H2) and the chaining (H1b) — so the agent should run the
   spike FIRST and the round-8 micro-benchmarks only for the recipe-path fallback
   decision if the spike fails).
2. **D24 the 8-rank @16 trace of current production (1 boot)** — the round-8 do-first
   stands: it is the before/after instrument for RA-059/060/061 and it is independent
   of the direct work (run it the same week as the spike, on a different HPU subset if
   needed).
3. **RA-059 the expert direct program (2-4 weeks after the spike PASS)** — the TPC
   sort/combine kernels (shared with RA-053-recipe) + the expert template captures +
   the one-layer A/B.
4. **RA-061 the attention direct program (parallel track, 2-4 weeks)** — the KDA WY
   kernel (the RA-043 spec) + the DSA tile kernel (the original design — the
   gathered-2048 variant) + the SRAM zone audit (D28).
5. **RA-060 the full-step chain (1-2 weeks after 3+4 have one layer class each)** —
   the span split, the protocol, the A/B.
6. **RA-056 the EP4×TP2 layout A/B (after 5, on the direct path)** — the round-8
   ordering stands (the layout after the kernel; the kernel is now the direct
   program).
7. **RA-062 the drafter probe (the owner's 3090 call first)** — the acceptance probe
   (H6's de-risk) is independent of the direct work and can start in parallel with
   the 3090; the integration waits for RA-060.
8. **The 4-HPU fit check + production topology (RA-056, 1 boot)** — the direct
   program at the production shape.
9. **8-HPU** — after the 4-HPU is green (the collective interplay is simpler at
   4-HPU; the 8-HPU TP8/EP8 is the harder collective case — the §3.3(a) design is
   validated at 4-HPU first).

**The kill-switch discipline:** every direct-path stage ships with the env kill
switch (the recipe path runs identically when the direct path is disabled — his
engine's fallback pattern), so the direct program is additive and revertible at every
step. The R003/T1 parity suite runs at every stage transition (the numerics
discipline is unchanged by the vehicle).
