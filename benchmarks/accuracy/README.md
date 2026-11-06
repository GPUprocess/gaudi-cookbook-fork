# accuracy/ — versioned accuracy evaluation suite (v1)

Paired control/treatment accuracy evaluation per `docs/BENCHMARK_POLICY.md`
(gate: max 0.5 **absolute** percentage-point regression on the aggregate metric).

## Suite: `suite_v1.jsonl` (v1)

40 items, 5 categories × 8 items, all deterministically scoreable without a judge model:

| Category | What it measures | Scoring |
|---|---|---|
| math | arithmetic, rates, algebra | last-number exact match |
| reasoning | classic trick/inference problems | numeric/exact |
| instruction | format compliance (JSON, bullets, casing, ordering) | exact/contains/structure |
| coding | small Python semantics | exact/numeric |
| general | stable world facts | exact/contains |

The suite is FIXED once accepted: changes require bumping to `suite_v2.jsonl` and re-running
both control and treatment. Prompts are short (cheap prefill) so runs are fast and the accuracy
signal is not confounded by context length.

## Usage

```bash
# one condition (server must be running; greedy decoding):
python3 evaluate.py run --base http://127.0.0.1:8000 --label control --reps 5

# gate comparison after both conditions:
python3 evaluate.py gate --control results/control.json --treatment results/mtp1.json
```

Per-rep overall accuracy, pooled Wilson 95% CI, per-category pooled results, and the raw
per-item outputs (truncated) are saved to `results/<label>.json`.

## Interpretation rules (from docs/BENCHMARK_POLICY.md)

- Delta within ±0.5 pp with adequate evidence → eligible to pass.
- Delta worse than −0.5 pp → FAIL and investigate (report the largest per-category drop).
- Overlapping/insufficient evidence → INCONCLUSIVE, not PASS.
- Correctness failure → immediate rejection regardless of speed.
- Never modify the suite, scoring, or settings to make a result pass.

## Current status

Suite authored (v1) but **not yet run** — first paired run happens with Stage-1 (A001):
control (non-MTP) vs treatment (MTP k=1) on the same window/config, 5 reps per condition.
