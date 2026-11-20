# Q21-RECOVERY-001 — Session recovery report (2026-11-19)

New-session recovery per the anti-looping protocol. Everything below was re-derived from the
repository, the server state, and the experiment records — not taken from any session summary.

## 1. Exact starting state (verified 2026-11-19)

| Item | State |
|---|---|
| Repository | the project fork at `~/publish_work/fork` (server) |
| Branch / tip | `glm53-flash` @ `33c2002` ("Q21: the session close ...") |
| Working tree | 2 modified files, uncommitted: `kernels/mxfp4-moe/extra/block_fp8_moe_w13_glm/block_fp8_moe_w13_glm.c` (the staged v11) and `.../block_fp8_moe_w13_dbg/block_fp8_moe_w13_dbg.c` (the 3-block dbgF diagnostic) |
| Kernel dist | `/data/glm53/q21_dist` built 2026-11-15 16:55 (contains the staged v11 ELF `3f09f663…`) |
| Serving | vLLM DOWN; port 8000 no response; devices free (an earlier "RUNNING" pgrep hit was the probe's own command line) |
| Local mirror | fork working tree synced to the analysis workspace (`fork_recovery/`), `.git` excluded |

## 2. Commit map (the exact artifacts the brief asks to preserve)

| Artifact | Commit |
|---|---|
| Q21 two-pass kernel contract + 12-stage gate | `8a54682` |
| v3 (interleaved-chunk-pair layout) | `0cc582d` |
| v5 (ALL_LANES convert, bfloat256 pair) | `92ad318` |
| v6 (stateless x loads, one-hot probes) | `d81ffed` |
| MAC accumulator-layout investigation | `88227d7` (probe), `59f5faa` (de-interleave + gate fix), `b3547e6`/`3190efc` (store idioms) |
| BF16 load/store minimal reproduction (lanes 1–2) | `5e82b83` |
| H2D materialization fix (device-created input class) | `e0f0279` |
| Data-origin map (in-context loads → scrambled addresses) | `e00de44` |
| mxk8 structural reference decode | `4197db9` (store idiom), `34a078e` (MXK8_PT reference) |
| count-3 test / mxk8-port decision | `0212612` |
| Staged v11 (dbg9 structure + FP8 compute) | `33c2002` (committed), working tree (uncommitted refinements) |

## 3. Known-good references (verified exact, standalone, cache off)

The micro-probe ladder — every piece exact in isolation (each probe = one load class or one
compute op in its own tiny graph, `PT_HPU_RECIPE_CACHE_CONFIG=0,false`):

1. x dump: variable-dim1 bf16 load + store — max err 0.0.
2. fp8 load + `v_convert_f8_to_bf16_all_b` — max err 0.0, both `.v1`/`.v2` halves.
3. Lane id `read_lane_id_2b_b()` — identity 0..127.
4. `v_bf16_mac_acc32_b(a, b, acc)` — `acc == a*b` element-wise, 128/128 lanes (after the probe's
   own reference bug was fixed; the earlier "acc mismatch" was the probe's error, not the MAC's).
5. `v_f32_reduce_add(acc.v1 + acc.v2)` — the correct horizontal sum.
6. Partial stores with the last-index count convention (`v_bf16_st_tnsr_partial(co, ap, v, n, 0)`
   stores `n+1` values) — exact; scalar partial stores exact.

Structural reference: `kernels/mxfp4-moe/src/mxk8_w13.h` + `mxk8_core.h` (production-validated
MXFP4 MoE kernel on the same stack). Key idioms: 1-D index space over output column pairs;
route-outer walk with the x row staged in VLM once per route (`MXK8_STAGE_X`); 4-way nibble
unpack + 4 MACs per 512-chunk (`MXK8_DOT` — same MAC intrinsic Q21 uses); lane-packed result
collection via lane-id predicate moves (`v_f32_mov_vb` with `lid4 == m`), one dual-group move at
flush, one partial store per run.

Operational facts that were discovered the hard way and remain binding:

- **Recipe cache trap**: Synapse serves stale compiled graphs for identical shapes;
  `PT_HPU_RECIPE_CACHE_CONFIG=0,false` is mandatory in every gate/probe run.
- **H2D materialization rule**: device-created tensors lack proper buffers for the legacy
  custom-op loads; inputs must be CPU-created then transferred (`.to(dev)`).
- **Bridge dtype rules**: `float8_e4m3fnuz` device casts rejected; `float8_e4m3fn` accepted;
  i64→fp8 and device-side `.float()` of custom-op outputs rejected (pull bf16, cast on CPU).
- **Standalone HPU compile**: works for the tiny probe graphs; the earlier "standalone fails"
  finding was the synStatus 26 recipe-compile limitation for graphs containing add, and the
  late-session environment degradation (see 5).
- **Iteration speed**: ~11 s per standalone gate cycle with the devices free.

## 4. Known failures (the composed kernel)

Gate shape: E=8, T=4, H=512, I2=512, K=8; pass-1 = `block_fp8_moe_w13_glm` (raw [T,K,2I]);
pass-2 = `block_fp8_moe_w2_glm`. Reference = per-(t,k) GEMV against the dequantized weights.

Symptom class (stable across v3/v5/v6/v7/v8/v11): the one-hot-x probe shows the kernel **never
writes a wrong value** (spurious=0, wrong=0) but **fails to write ~89% of the output columns**
(dropped=446–454 of 512 on the k=0 row), with a scattered, non-modular ~11% correct subset.
The all-ones-x probe showed the whole token-0 output block unwritten and a constant broadcast
on tokens 1–3 in the v7-era store idiom (the per-row `v_bf16_mov_vb` insert chain — since
replaced by partial stores, drops persisted, so the store idiom was not the sole cause).

Eliminated defect classes (each with deterministic evidence — do NOT re-test without a new
variable): convert halving (v2), pair-split layout (v3/v5), VLM staging (v6), fp8 decode
convention (bias-7 e4m3fn verified against the corpus sweep), per-position convert/scale/MAC
(one-hot), cross-token state (zeroed-token probe), partial-store count (count-3 test), the
probe-side MAC layout (fixed reference).

**New finding this session (not previously recorded): the committed v11 cannot test the dbg9
structure.** The v11 kernel decodes the point id as `col = ip % I2; k = (ip/I2) % 8;
t = ip/(I2*8)` — an index space of `T*K*I2 = 16384` points — while the glue declares
`npts = (I2/128)*64*T = 1024` points. The kernel therefore processes only `t=0, k∈{0,1}` and
never writes tokens 1–3 or k≥2. Any v11 gate result is a measurement of ~25% coverage plus the
compute defect, confounded. **The v11 gate numbers (dropped=446) must not be read as a
dbg9-structure verdict.**

**Second new finding: the last-session dbgF diagnostic was self-invalidating.** Its three test
blocks (A: x dump at `{p*256,k,t}`, B: w-convert dump at `{0,k,t}`, C: acc dump at `{0,k,t}`)
write overlapping output regions; block A's p=0 store spans exactly the columns B and C target.
The "block B shows x-magnitude values" observation was block A's data winning the store race.
The dbgF conclusions are void; the kernel stays in the tree only as a record.

## 5. Environment degradation caveat

The previous session's final standalone runs produced deterministic-but-uncorrelated outputs
("deterministic garbage") after ~20 test cycles. Every pre-degradation result was reproduced
identically, so the recorded evidence stands; the degraded-state runs are void. This session
re-derives Layer A in fresh processes before anything else.

## 6. Layer A state and the open question

The minimal BF16 load/store reproduction (`extra/store_test` + `st_run.py`): one 128-lane load,
32 full stores to the [512,8,4]-layout output, plus 2 stride-probe stores. At `5e82b83` it
corrupted lanes 1–2; after the H2D fix the behavior changed (col-0 zero + scattered mask);
the probe-ladder verdict at session end was "stores exact". **The current fresh-process verdict
is unknown** (the last-session runs fell in the degraded window). Re-establishing it is the
first action of this session — it is the gate for every layer above.

## 7. Proposed next steps (the controlled path)

1. **Layer A re-baseline**: run `store_test` fresh (cache off, H2D inputs, new process).
   Outcome A (all 32 blocks exact): memory correctness holds for the simple pattern; the defect
   is in the composed-kernel structure → proceed to 2. Outcome B (corruption): the memory layer
   itself is broken → run the variable matrix (strides, widths, alignment, store forms) before
   touching any MoE code.
2. **The composed-structure isolation**: a load-progression ladder in the REAL kernel skeleton —
   (a) ids load only + x dump, (b) + w13 load dump, (c) + scales, (d) + full compute — one new
   load class per run, all stores to DISJOINT output regions (the dbgF lesson). The first rung
   where the x dump stops being exact localizes the interaction.
3. **The mxk8-exact structural port** (already agreed): mirror `mxk8_w13.h`'s schedule on the
   Q21 data path — 1-D column-pair space, route-outer walk, VLM x staging, fp8 loads + ALL_LANES
   convert in place of the nibble unpack, `MXK8_DOT`-style MACs, lane-packed collection, partial
   stores. One structural element per step, gate after each.
4. Layers B–H per the anti-looping protocol, with the ledger updated after every run.

## 8. Serving state

Production is down (devices free — required for the standalone iteration). Boot script
`/data/glm53/boot_f4ts.sh` restores the adopted config (MTP k=1, TP8+EP, ~42 t/s bs=1,
~250–258 t/s @16). Per the autonomy policy, restarts are autonomous; the engine will be
restored when Layer A–C work pauses or completes.
