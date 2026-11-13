# Replicating the GLM-5.3-Flash deployment on g2 (192.168.0.143)

What was done, end to end, so it can be redone from a clean state. Scripts live in
`server/` next to this file and on the target at `/data/glm53/`.

## 0. Target

| | |
| --- | --- |
| Host | `g2`, 192.168.0.143, user `tmpai2` (sudo ALL) |
| Hardware | 8× Intel Gaudi 2 (HL-225, 96 GB HBM), 160 CPU cores, 125 GB RAM, 3.7 TB NVMe |
| OS | Ubuntu 24.04.5, kernel 6.8.0-147-generic |
| Gaudi software | driver/firmware 1.24.1-482 (`habanalabs-dkms` et al.), hl-smi 1.24.0 |
| Source recipe | gaudi-cookbook `recipes/zai-org/GLM-5.3-Flash/recipe.yaml` (+ `docs/recipes/glm-5-3-flash.md`, port `ports/glm-5-3-flash/`) |

## 1. Free the node

The box came with two serving instances:

- **DeepSeek-V4.1-Flash** (the one this task targets): systemd unit
  `dsv41-watchdog.service` (auto-relaunch on crash) + tmux session `dsv41`
  launching `/opt/dsv41/venv/bin/python -m vllm_gaudi.entrypoints.deepseek_v41`
  (TP 2 × PP 2, cards 4–7, port 8000).

  ```bash
  sudo systemctl stop dsv41-watchdog.service
  sudo systemctl disable dsv41-watchdog.service   # removes multi-user.target.wants symlink
  tmux kill-session -t dsv41                      # session of user tmpai
  sudo pkill -f run_deepseek_v41.py
  sudo pkill -f vllm_gaudi.entrypoints.deepseek_v41
  ```

  Verified: `systemctl is-active` → inactive, `is-enabled` → disabled, port 8000
  closed, no `VLLM::Worker*` processes, cards 4–7 back to 768 MiB idle.

- **strata-hpu** (another user's Qwen3.8 service on card 2, port 8080): killed with
  user approval, since GLM-5.3-Flash needs all 8 cards (TP 8). No systemd unit or
  cron entry exists for it, so it stays stopped.

  ```bash
  sudo kill -9 <serve.server pid> <strata-hpu pid>
  ```

Check all 8 cards are free: `sudo hl-smi | grep 'MiB /'` → every row ~768 MiB / 98304 MiB.

## 2. Python stack (no Docker on this host)

The cookbook's documented path builds a port image on top of a stock
`vllm-gaudi v0.26.0` container. This host has no container runtime and
`vault.habana.ai` no longer serves its pip index, so the stack is installed
natively in a venv from the **official Habana wheel bundle** for SW 1.24.1 that
ships on the host (`/opt/dsv41/pt/wheels`: torch `2.11.0a0+git009b5f6`,
`habana_torch_plugin-1.24.1.482`, matching the host driver build `-482`). This is
the same component set the `pytorch-installer-2.11.0` base image installs.

```bash
# wheels must be copied to a writable dir (the bundle dir is not)
mkdir -p /data/glm53/ptw && cp -a /opt/dsv41/pt/wheels/. /data/glm53/ptw/
chmod -R u+w /data/glm53/ptw
bash server/setup_venv.sh          # → /opt/glm53/venv
```

`setup_venv.sh`: creates `/opt/glm53/venv`, installs the Habana wheels + the
`requirements-pytorch.txt` of the bundle, torchvision from pytorch.org — i.e. the
official installer's "fork" flow. Verifies `torch.__version__` (Habana fork, no
`+cpu`), `import habana_frameworks.torch`, and the plugin include/lib dirs.

## 3. vLLM + vllm-gaudi at the release commits

The port's `install.sh` fails closed unless `vllm` and `vllm_gaudi` are **git
checkouts** at `568afb3a` (vLLM) and `b059f9c` (vllm-gaudi, = tag `v0.26.0`,
whose `VLLM_STABLE_COMMIT` file pins 568afb3a). Both are installed editable so
the checks and the fix patches see the checkouts:

```bash
git clone https://github.com/vllm-project/vllm-gaudi.git /opt/glm53/vllm-gaudi
git -C /opt/glm53/vllm-gaudi checkout b059f9c          # Release v0.26.0
cat /opt/glm53/vllm-gaudi/VLLM_STABLE_COMMIT            # 568afb3a13806...
# vLLM: partial clone (a fetch-by-SHA is rejected by GitHub)
git clone --filter=blob:none --no-checkout https://github.com/vllm-project/vllm.git /opt/glm53/vllm
git -C /opt/glm53/vllm checkout 568afb3a
```

Then the Dockerfile's install steps (`server/install_stack.sh`, run once the
venv and clones exist — it waits for both):

```bash
source /opt/glm53/venv/bin/activate
cd /opt/glm53/vllm
pip install -r <(sed '/^torch/d' requirements/build/cuda.txt)   # keep the Habana torch
VLLM_TARGET_DEVICE=empty pip install -e . --no-build-isolation
cd /opt/glm53/vllm-gaudi
VLLM_TARGET_DEVICE=hpu pip install -e . --no-build-isolation --no-deps
```

`--no-deps` on vllm-gaudi prevents pip from replacing the editable vLLM with a
PyPI wheel. The stock image preinstalls a few packages the editable installs
don't pull in — add them (the worker import path needs them):

```bash
pip install numba "pandas>=2.2.3" "kaldi-native-fbank>=1.18.7" "tblib==3.1.0" "fastapi<0.137"
pip install transformers==5.14.1   # the stock image's version; newer renames
                                   # transformers.models.pixtral.PixtralRotaryEmbedding and breaks vllm's import
```

Finally the GraphCaptureOutput workaround from `docs/getting-started.md`
(step 4): vLLM at `568afb3a` monkeypatches
`torch._dynamo.convert_frame.GraphCaptureOutput.get_runtime_env` for torch
< 2.12; the Habana 2.11 fork has no such class, so disable that block in
`/opt/glm53/vllm/vllm/env_override.py` (guard it with `if False:`, keeping
`_apply_constrain_to_fx_strides_patch` — other modules import it).

## 4. Checkpoint

`zai-org/GLM-5.3-Flash` (MIT, block-FP8, 306 GiB / 74 files), public, no token
needed. ~90 min at ~70 MB/s:

```bash
HF_HUB_DISABLE_XET=1 hf download zai-org/GLM-5.3-Flash --local-dir /data/models/GLM-5.3-Flash
```

(Xet storage hit a permission error on this host; disabled it, plain CDN download.)

## 5. Port install + fixes

```bash
git clone --depth 1 https://github.com/thediymaker/gaudi-cookbook.git /opt/glm53/gaudi-cookbook
export COOKBOOK=/opt/glm53/gaudi-cookbook
export PATH=/opt/habanalabs/bin:$PATH        # tpc-clang for the Sinkhorn kernel
export PYTHON=/opt/glm53/venv/bin/python
bash $COOKBOOK/ports/glm-5-3-flash/install.sh
# → 4 source patches (fail closed), mHC Sinkhorn TPC kernel build, plugin install
/opt/glm53/venv/bin/python $COOKBOOK/fixes/apply_all.py --profile v0.26.0 glm47-streaming-text-markers
```

Host prerequisites for the kernel build (all present): `tpc-clang`
(`/opt/habanalabs/bin`, from `habanalabs-graph`), `/usr/include/habanalabs/tpc_kernel_lib_interface.h`,
`/usr/lib/habanalabs/libtpc_kernels.so`, g++/strip, and the venv's torch + plugin.

Gotcha: the op-library gate loads the op lib into a process that first imports
`habana_frameworks.torch.core`. With `PT_HPU_LAZY_MODE` unset that import loads
the PT2 plugin (`libhabana_pytorch2_plugin.so`); the op lib then pulls the legacy
`libhabana_pytorch_plugin.so` and both register `habana_d2d_memcpy_other` →
abort. The stock image carries `PT_HPU_LAZY_MODE=1` in its Docker ENV — export
it before `install.sh` (the scripts here do).

## 6. Serve

Two serving configs exist:

**(a) The original recipe config** — `server/serve_glm.sh` (the recipe's launch
block verbatim, no speculative decoding). Superseded for production but useful as
the baseline.

**(b) The adopted production config (current)** — MTP k=1 speculative decoding +
the decode-lean levers, via the research fork's launcher:

```bash
# the research fork (the optimization work + the MTP serving config)
git clone https://github.com/GPUprocess/gaudi-cookbook-fork.git /home/tmpai2/publish_work/fork -b glm53-flash
export COOKBOOK=/home/tmpai2/publish_work/fork
export GLM53_MODEL_PATH=/data/models/GLM-5.3-Flash

# stop + start (the graceful procedure)
bash /data/glm53/kill_glm.sh
nohup bash /data/glm53/boot_f4ts.sh > /data/glm53/serve_restored.log 2>&1 &
```

`serve_mtp.sh` (the fork) launches with: `--speculative-config
{"method":"mtp","num_speculative_tokens":1}`, `GLM_ROUTER_FAST=1`,
`GLM_SPEC_DEVICE_REJECT=1`, `GLM_EARLY_DRAFT=1`, ns32, and the F2
draft-ids piggyback (`GLM_F2_DRAFT_PIGGYBACK=1`, set by the boot wrapper). The
research fork also carries the measurement instrumentation ([LOOP]/[LAUNCH]/[T1B]/
[SPEC-TS] stamps, all env-gated default-inert) and the accuracy result artifacts.

Cold start: ~20–25 min (the checkpoint load ~14 min + the engine start + the
capture). Then `curl http://localhost:8000/health` → 200.

The production numbers with the adopted config: decode bs=1 **40–43 tok/s**
(MTP acceptance ~1.95), @16 streams 250–258 tok/s bench-aggregate, the prefill
5,546–7,723 tok/s (the prefill-optimized profile), the accuracy suite 97.5%.

## 7. Verify + benchmark

Full gate suite (`golden needles tools images decode prefill`, ~8 min against
the running server; `decode`/`prefill` are the benchmark numbers):

```bash
source /opt/glm53/venv/bin/activate
bash /home/tmpai2/publish_work/fork/ports/glm-5-3-flash/tests/hpu/run_gates.sh
```

Expected bars: single-stream decode ≥ 15 tok/s (2K prompt) and ≥ 10 tok/s (128K);
warm TTFT at 128K within 120 s. The adopted-config numbers: decode 40–43 tok/s
bs=1 sustained, @16 streams 250–258 tok/s (the bench; ~415–430 steady — the bench
dilutes with the EOS early-finishes), the accuracy suite 97.5% (the one miss =
the R003 bat-and-ball reasoning knife-edge, a 0.75-logit near-tie, deterministic
— see REPORT.md §6).

The throughput benchmark against the running server:

```bash
/opt/glm53/venv/bin/python3 /home/tmpai2/publish_work/fork/ports/glm-5-3-flash/tests/hpu/mtp/bench_concurrent.py --streams 16 --reps 2
```

## Notes / gotchas

- `hl-smi` as `tmpai2` aborts (core dump); use `sudo hl-smi`.
- `/data` and `/opt` are root-owned; create project dirs with sudo + chown.
- `~/.cache` may contain root-owned files from sudo runs → `chown` it back or pip
  silently disables caching.
- Don't re-enable prefix caching, don't drop `--num-gpu-blocks-override 1280`
  (KV autosizing over-commits on this hybrid model and fails), don't raise
  `GLM5_NEXT_IMAGE_MAX_TOKENS` without re-running the image gate — see the
  recipe's "What not to do".
- **Never signal the engine processes** (a SIGUSR1 to EngineCore terminates it —
  USR1's default action). Restart via `kill_glm.sh` + the boot script instead.
- **The wedged-card recovery**: after the rapid kill/reset cycles the HL-225B
  cards can wedge at the 50W power cap (the card shows no memory reading);
  `sudo hl-smi -r` cannot recover — use the BMC chassis cycle:
  `ipmitool -H 192.168.0.144 -U <user> -P <pass> chassis power off`, wait 30 s,
  `chassis power on` (the credentials in the owner record; validated 11-12).
- **The research fork is the serving source of truth** (the MTP serving config,
  the levers, the measurement instrumentation, the accuracy artifacts); the
  upstream `thediymaker/gaudi-cookbook` remains the recipe/patch source.
