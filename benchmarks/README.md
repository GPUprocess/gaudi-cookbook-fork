# benchmarks/ — sanitized benchmark results and reproducible configurations

Raw benchmark outputs are NOT committed here. This directory holds sanitized, reproducible
benchmark artifacts: per-run tables, configuration templates, and the workload scripts used
by the experiments indexed in `EXPERIMENTS.md`.

Contents (to be populated as experiments are re-run under `docs/BENCHMARK_POLICY.md`):

- `accuracy/` — the versioned accuracy evaluation suite (pending; see `ACCURACY.md`).
- Per-experiment raw per-run tables (median/mean/std/P95) once Stage-1 reproduction runs.

Reproducible launch configurations live in `ports/glm-5-3-flash/serving/` in the repository
root. Do not commit raw system logs, environment dumps, or host-identifying output.
