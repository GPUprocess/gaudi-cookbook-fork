<p align="center">
  <img src="docs/assets/logo.svg" alt="Gaudi Cookbook" width="96">
</p>

<h1 align="center">Gaudi Cookbook</h1>

<p align="center">
  Tested recipes, runtime fixes, kernels and fleet practices for serving open LLMs with
  <a href="https://github.com/vllm-project/vllm">vLLM</a> on Intel Gaudi 2, through the
  <a href="https://github.com/vllm-project/vllm-gaudi">vllm-gaudi</a> plugin.
</p>

<p align="center">
  <a href="https://github.com/thediymaker/gaudi-cookbook/actions/workflows/ci.yml"><img src="https://github.com/thediymaker/gaudi-cookbook/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License: Apache-2.0"></a>
</p>

## Fork: GLM-5.3-Flash native MTP speculative decoding on Gaudi 2

This fork of [thediymaker/gaudi-cookbook](https://github.com/thediymaker/gaudi-cookbook) adds one
verified extension, on the branch `glm53-flash`: **native MTP speculative decoding for the
GLM-5.3-Flash port**. The Apache-2.0 licence and the upstream content are untouched; all additions
live under `ports/glm-5-3-flash/` and `docs/ports/glm-5-3-flash-mtp.md`.

### What was done

The GLM-5.3-Flash checkpoint ships its own MTP draft head (layer 45 plus an `eh_proj` and a shared
head). This work binds that head to vllm-gaudi's speculative-decode machinery on one 8-card Gaudi 2
node:

- **Draft wrapper with no weight cost** (`ports/glm-5-3-flash/glm5_next_hpu/glm5_next_hpu/mtp.py`):
  the draft model is the target's already-loaded layer-45 modules held by reference — zero duplicate
  weight memory, one extra MLA latent cache and KDA candidate-state columns (~1.1 GiB/rank total).
  Checkpoint names are normalised on load, and startup refuses to proceed if speculative decoding is
  on and the MTP block did not fully load.
- **KDA candidate-slot state** (`kda_layer.py`, `kda.py`): the Kimi Delta Attention layers keep S+1
  recurrent-state columns per request, the verify forward scatters one state per row, and the
  post-verify load selects the column matching the accepted prefix. The slot algebra is shared with
  vllm-gaudi's compact Mamba state layout and unit-tested against request churn.
- **GPU-layout spec patches reused**: the eight speculative-decoding patches from
  `ports/glm-5-mxfp4/patches/` (already in this repository) were drift-checked against the
  `v0.26.0` release sources and apply cleanly; they cover padded-verify sampling, penalties,
  logprobs, structured output, chunked-prefill draft KV and per-verify-row KV block lists.

### Measured results (8-card Gaudi 2 node, block-FP8 checkpoint, greedy, natural prompts)

| Metric | k=1 MTP | baseline (no MTP) |
| --- | --- | --- |
| Decode, bs=1 | **36.8 tok/s** | 30.3 tok/s (+21%) |
| Decode after a 4K prompt | 37.6 tok/s | — |
| Acceptance length | 1.73 – 1.98 of 2.0 | 1.0 |
| HBM under load | 64.1 GiB / 96 GiB per card | ≈ 62 GiB |
| 16-stream aggregate | 42.7 tok/s | 59.4 tok/s (same window config) |

k=2 was swept and is correct (acceptance 2.04 – 2.52 of 3.0) but net-neutral on speed: the second
eager draft step costs more than the extra accepted tokens buy. **Enable MTP for single-stream /
interactive workloads; leave it off at high concurrency** (crossover measured at 16 streams with
the eager draft head; the draft HPU graph is the follow-up lever).

### How to replicate

1. **Hardware/software**: one 8-card Intel Gaudi 2 node (96 GB HBM per card), Gaudi software 1.24.1
   on the host, the base image
   `vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/pytorch-installer-2.11.0`.
2. **Build the port image** from this checkout as in `docs/getting-started.md` and
   `ports/glm-5-3-flash/README.md` (`ports/glm-5-3-flash/Dockerfile` runs `install.sh`, which
   checks the release commits, applies the port patches, builds the mHC Sinkhorn kernel and
   installs the plugin with the MTP module; fails closed).
3. **Apply the spec patches** to the release sources inside the image, in this order (each supports
   `--check` and exits 2 on anchor drift):
   `mtp_tuple`, `mtp_residual_shape`, `spec_sampling`, `spec_penalties`, `spec_logprobs`,
   `spec_structured_output`, `spec_draft_prefill`, `spec_verify_blocks` — from
   `ports/glm-5-mxfp4/patches/`.
4. **Download the checkpoint**: `zai-org/GLM-5.3-Flash` (block FP8 as published, ~328 GB).
5. **Launch**: `ports/glm-5-3-flash/serving/serve_mtp.sh` (set `COOKBOOK` and `GLM53_MODEL_PATH`).
   The deltas from the stock port launch: `--speculative-config '{"method": "mtp",
   "num_speculative_tokens": 1}'`, `--max-num-seqs 8` (verify rows = seqs x (k+1)), reduced
   context window and KV override for the validated shape.
6. **Validate**: `python3 ports/glm-5-3-flash/tests/hpu/mtp/mtp_validate.py --base
   http://127.0.0.1:8000` against the running server; run the port gates
   (`ports/glm-5-3-flash/tests/hpu/run_gates.sh`); run the CPU unit checks in
   `ports/glm-5-3-flash/tests/cpu/` on any torch-capable machine.

Expected behaviour: first boot compiles all graph shapes (slow; warmup graphs are pre-compiled with
`VLLM_SKIP_WARMUP=false` on production launches — the validation script keeps skip-warmup for
faster windows). Acceptance length ≈ 1.8 of 2.0 on natural prompts; single-stream decode ≈ 36–37
tok/s at bs=1.

### Known limitations and TODOs

- k=1 only is recommended; k=2 is correct but net-neutral with the eager draft.
- The draft head runs eagerly (shared storage cannot be captured in a second HPU graph); the
  documented levers to flip k>1 and high concurrency positive are the draft HPU graph and a
  vocab-parallel argmax in `get_top_tokens`.
- Verified in a 32K-context window at `--max-num-seqs 8`; the production 262K shape was not re-run
  under MTP.
- Golden parity under MTP is 2/12 token-identical (k=0 on the same window: 6/6); every divergence
  is a measured near-tie flip from the verify-shape FP shift, and the content is semantically
  equivalent.
- Prefix caching must stay off (KDA layer constraint, inherited from the base port).

---

Everything here was run on a large Gaudi 2 fleet serving real traffic. Everything uses public
container images and public checkpoints. Nothing depends on a private build.

> **New: MXFP4 MoE decode kernels.** Up to 2.5x faster decode for MXFP4 mixture-of-experts models on one 8-card
> Gaudi 2 node: GLM-5.3 single stream from 23.3 to 59.1 tokens/s, Kimi-K2.7 from 17.9 to 44.5, Muse Spark Open 2.4x
> aggregate at four streams, with lower numerical error than the stock op. See [the write-up](docs/kernels/mxfp4-moe.md).
>
> **Ports.** Nine ports bring up models that released images cannot serve on Gaudi 2: Muse Spark Open,
> GLM-5.3-Flash, DeepSeek-V4.1-Flash, GLM-5.x MXFP4, Beam-1.0, MiMo-V2.6 (Flash and Pro), Kimi-K2.7-Code,
> MiniMax-M3 and Inkling-Small. See [Ports](docs/ports/index.md) and [Results](docs/results/index.md).
>
> **Using a coding agent?** Point it at this repository and have it read [`AGENTS.md`](AGENTS.md)
> first. See [Use it with your coding agent](#use-it-with-your-coding-agent).

## What is inside

| Section | What you get |
| --- | --- |
| [`recipes/`](recipes/) | One `recipe.yaml` per model: cards, precision, buckets, parsers, environment, the checks that passed, measured throughput, and known pitfalls. Rendered pages live under [`docs/recipes/`](docs/recipes/). |
| [`fixes/`](fixes/) | Small runtime patches for bugs in released vllm-gaudi and vLLM images. Each one is tied to an upstream pull request or issue, has a `--check` mode, and refuses to start the server if the image does not match what it expects. |
| [`kernels/`](kernels/) | Custom TPC kernels. The MXFP4 MoE decode kernels read the packed E2M1 weights and E8M0 scales directly and make decode up to 2.5x faster. One build variant per model family (GLM-5.x, DeepSeek-V4.1, Muse Spark Open, Kimi, gpt-oss) with build scripts, parity tests and micro benchmarks; the ports hook them in. |
| [`docs/ports/`](docs/ports/index.md) | Write-ups of models brought up on Gaudi 2 that released images cannot serve: what was hard, how it was solved, the correctness gates and measured speed. The integration code is not published yet. |
| [`docs/results/`](docs/results/index.md) | Studies and findings that are not tied to one model: platform behaviours and workarounds, prefix caching on Gated DeltaNet hybrids, MoE decode-step anatomy, two-node pipeline parallel, fast prefill, text-to-speech and embeddings. |
| [`servers/`](servers/README.md) | Small HPU servers for models vLLM does not serve: Qwen3-TTS, Chatterbox and Orpheus text-to-speech behind an OpenAI `/v1/audio/speech` API, and EmbeddingGemma 2 multimodal embeddings, each with an image, CPU tests and device gates. |
| [`kubernetes/`](kubernetes/) | Running Gaudi at fleet scale: pod spec, bin-pack scheduling, start-up and warm-up, health checks, rollouts, autoscaling, routing and observability, with example manifests and small tools. |

## Quick start

1. Check the [software baseline](docs/getting-started.md): Gaudi software 1.24.1, the
   `pytorch-installer` base image, vllm-gaudi `v0.26.0` or `releases/v0.30.0`.
2. Pick a recipe, for example [`recipes/Qwen/Qwen3.5-397B-A17B-FP8`](recipes/Qwen/Qwen3.5-397B-A17B-FP8/).
3. Apply the fixes it lists, then start the server:

   ```bash
   python3 fixes/apply_all.py --check --profile v0.30.0   # report only, write nothing
   python3 fixes/apply_all.py --profile v0.30.0 || exit 1 # apply; exits 2 on source drift
   exec vllm serve <model> <recipe flags>
   ```

## Use it with your coding agent

This repository is written to be read by coding agents as much as by people. Point Claude Code,
Codex, Cursor, Copilot, Gemini CLI or a similar agent at it, and the agent can answer Gaudi
questions from tested configurations instead of guessing. [`AGENTS.md`](AGENTS.md) tells the agent
where everything is and what it must not do. `CLAUDE.md` loads the same file for Claude Code.

**How to point your agent at it**

- **Next to your work.** Clone it beside your project or deployment repo, and tell the agent:
  "Use `../gaudi-cookbook` as the reference for vLLM on Gaudi. Read its AGENTS.md first."
- **As the workspace.** Open the agent inside the clone. Most agents read `AGENTS.md` on start.
- **By URL.** Agents that browse can start from
  `https://github.com/thediymaker/gaudi-cookbook/blob/main/AGENTS.md`. The docs site serves an
  [`llms.txt`](docs/llms.txt) index.

**What the agent gets**

- **Recipes as data.** Each `recipe.yaml` holds the full verified `vllm serve` command,
  environment, card layout, required fixes, the checks that passed, and the settings that break it.
- **Symptom lookup.** Every fix has a `## Symptom` section with the error text it causes, so the
  agent can match a log line or a wrong output to a cause, the affected releases, and the upstream
  pull request.
- **Safe patches.** `apply_all.py --check` reports first, and the fixes refuse to apply to an
  image they do not match, so the agent cannot half-patch a server.
- **Fleet knowledge.** Kubernetes pages and manifests for scheduling, warm-up, health, autoscaling
  and routing, plus read-only diagnostic tools for nodes and cards.
- **Ground rules.** It must check the image before patching, keep verified and untested apart, and
  ask before touching shared hardware.

**Things to ask**

- "Serve Qwen3-235B-A22B-Instruct-2507-FP8 on four Gaudi 2 cards with the v0.26.0 image. Give me
  the container command."
- "My engine died with `tensor_data is empty` in `hccl::allgather_out`. What is it, and how do I
  fix it?"
- "Which fixes does my v0.30.0 image need for a Gemma 4 deployment with tool calling?"
- "Turn the Qwen3.5-397B recipe into a Kubernetes Deployment for our cluster."
- "Some nodes show 8 allocatable cards but every multi-card start fails with `Device acquire
  failed`. Help me find out why."

Agents make mistakes. Review what they propose, especially anything that changes a running
deployment.

## Documentation site

The documentation site is built from [`docs/`](docs/) with MkDocs Material:

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

## Principles

- **Stock first.** Recipes use released images. A fix is listed only when the stock image
  gets something wrong, and the recipe says which symptom it prevents.
- **Upstream first.** Every fix and kernel is offered upstream. When a release ships the
  change, the fix is marked superseded and recipes stop listing it.
- **Fail closed.** Patches match exact source anchors. If the code moved, they stop instead
  of guessing.
- **Show the evidence.** Claims come with the command that produced them and the numbers it
  printed.

## Hardware and scope

Intel Gaudi 2 (HL-225, 96 GB HBM per card, 8 cards per node), lazy mode unless a recipe sets
compile mode. Gaudi 3 is not
covered yet. Contributions with Gaudi 3 results are welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Status

This is an independent project. It is not affiliated with or endorsed by Intel or the vLLM
project. Names of models and products belong to their owners.

## License

[Apache License 2.0](LICENSE).
