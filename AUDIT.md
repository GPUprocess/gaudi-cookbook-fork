# Initial Audit — policy installation and project state

Per the pre-execution directive: the four policy files were located, installed, and
integrated; the repository was identified; the knowledge base was established; and the
project was audited. This file records the audit result at the time of the policy
installation (the Git commit that introduced this file).

## 1. Repository

`GPUprocess/gaudi-cookbook-fork`, branch `glm53-flash` — the existing fork with the
GLM-5.3-Flash MTP implementation. No new repositories created; remotes unchanged
(origin = the GPUprocess fork, upstream = thediymaker/gaudi-cookbook).

## 2. Branch and remote

Branch `glm53-flash`, origin `https://github.com/GPUprocess/gaudi-cookbook-fork.git`,
upstream `https://github.com/thediymaker/gaudi-cookbook.git`. Push workflow: the
GitHub CLI with the operator token (never embedded in URLs).

## 3. Policy files installed

| File | Location |
|---|---|
| BASELINE.md | repository root |
| SESSION_PROTOCOL.md | docs/ |
| BENCHMARK_POLICY.md | docs/ |
| HARDWARE_ROADMAP.md | docs/ |

## 4. Copy verification

All four files copied verbatim; md5 checksums match the operator originals at
`/home/[operator]/gaudi-agent-policies/` (the operator home on the Gaudi 2 server,
not the SSH user's home — found via a system-wide search after the advertised
`~/secrets/` path did not contain them).

## 5. Agent instructions updated

`AGENTS.md` carries a mandatory "Project policies (mandatory reading)" section with
the four-file table and the five requirements (session reading order, procedures,
the 0.5-pp accuracy gate, the 8-HPU development / 4-HPU deployment roadmap). All
upstream instructions preserved.

## 6. Knowledge-base state at installation

| File | State |
|---|---|
| TASKS.md | Created: the full protocol backlog (A001-K009 + additions), evidence-based statuses |
| EXPERIMENTS.md | Created: OPT-001..004 (pre-MTP), MTP-001..004, RESEARCH-001..002 |
| BASELINE.md | The policy template installed as-is: verified figures NOT YET VERIFIED (the treatment accuracy half is measured; the control half and the paired verdict await the Stage-1 window) |
| CURRENT_STATE.md | Created: the live MTP k=1 deployment, the verified baseline candidates, the active work, the blockers |
| ACCURACY.md | Created: the gate policy, the measured treatment half (200/200 x 5 reps), the remaining control-half steps |
| DECISIONS.md | Created: 13 accepted/rejected decisions with evidence |
| experiments/ | 11 reports (OPT + MTP + RESEARCH) |
| benchmarks/ | accuracy/ (suite v1 + evaluator + the measured treatment run), stage1_reproduction.sh, pre_flight.sh, diag_compare.py |

## 7. Missing or incomplete historical information

- **The ~21% MTP=1 batch-1 claim is PROVISIONAL**: the original control used the
  production window (256K/32seq) while the MTP window used 32K/8seq — non-equivalent
  conditions. The like-for-like single-run comparison is +14% (35.7 vs 31.4 tok/s).
  Neither has passed the repetition/variability requirements of BENCHMARK_POLICY.
- **The accuracy control half is unmeasured**: the treatment half (MTP k=1) is
  200/200 across 5 repetitions; the non-MTP control requires a server restart
  (the Stage-1 window).
- **Pre-MTP raw benchmark artifacts** are summarized, not published (they live in
  the operator's local working notes).

## 8. Production-safety concerns and resource constraints

- The GLM-5.3-Flash MTP k=1 server is the **production workload on all 8 HPUs**.
  Every Stage-1/E015/C015/B023/I015 measurement requires stopping and restarting it —
  explicit owner approval per the policies.
- The `HARDWARE_ROADMAP.md` reference to "the existing 1CatAI/vLLM-Gaudi DeepSeek
  V4.1 Flash service" is stale: that service was decommissioned at the owner's
  instruction before this project began; GLM-5.3-Flash is now the production service.
- The "200 tok/s decode" target is met under the aggregate interpretation (213-217
  tok/s at 16 streams, no MTP, replicated) and physically unachievable under the
  single-stream interpretation (the weight-read BW floor caps bs=1 at ~40-59 tok/s
  without MTP; ~150-190 with deep MTP at perfect utilization). Pending owner
  confirmation of the aggregate reading.

## 9. Recommended next task

**A001 — Stage-1 controlled MTP=1 reproduction** (the protocol's Stage 1). It converts
every PROVISIONAL figure to VERIFIED, is turnkey (`pre_flight.sh` 26/26 →
`stage1_reproduction.sh`), and unblocks the staged roadmap. Requires a maintenance
window (~2.5 h): the orchestrator stops and restarts the production engine twice.
