#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paired accuracy evaluation runner for the GLM-5.3-Flash accuracy suite (v1).

Runs the versioned suite (suite_v1.jsonl) against a served endpoint N times under greedy
decoding, scores deterministically, and reports per-category and aggregate accuracy with
uncertainty. Two saved runs can be compared with the 0.5-absolute-percentage-point gate from
docs/BENCHMARK_POLICY.md.

Usage:
  evaluate.py run   --base http://127.0.0.1:8000 --label control --reps 5 [--suite suite_v1.jsonl]
  evaluate.py gate  --control control.json --treatment treatment.json

Writes one JSON result file per run: results/<label>.json (per-item raw + per-rep aggregates).
"""
import argparse
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

THINK = "</think>"
MAX_TOKENS = 128


def load_suite(path):
    items = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            items.append(json.loads(line))
    return items


def chat(base, prompt, timeout=280):
    body = {
        "model": "zai-org/GLM-5.3-Flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    msg = r["choices"][0]["message"]
    content = msg.get("content") or ""
    if THINK in content:
        content = content[content.index(THINK) + len(THINK):]
    return content.strip()


def last_number(text):
    m = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return m[-1] if m else None


def score(item, content):
    check, ans = item["check"], item["answer"]
    t = content.strip()
    if check == "numeric":
        got = last_number(t)
        try:
            return got is not None and abs(float(got) - float(ans)) < 1e-6
        except ValueError:
            return False
    if check == "exact":
        return t == ans
    if check == "contains":
        return ans.lower() in t.lower()
    if check == "contains_all":
        return all(w.lower() in t.lower() for w in ans.split())
    if check == "bullets":
        lines = [l for l in t.splitlines() if l.strip()]
        bullets = [l for l in lines if l.lstrip().startswith("- ")]
        return len(bullets) == int(ans)
    if check == "items2":
        parts = [p for p in re.split(r"[,;]", t) if p.strip()]
        return len(parts) == int(ans)
    raise ValueError(f"unknown check type {check}")


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def run(args):
    suite = load_suite(args.suite)
    out = {"label": args.label, "base": args.base, "suite": Path(args.suite).name,
           "reps": args.reps, "runs": []}
    for rep in range(args.reps):
        per_item, t0 = {}, time.time()
        for item in suite:
            try:
                content = chat(args.base, item["prompt"])
                ok = score(item, content)
                err = None
            except Exception as e:  # noqa: BLE001 - record and continue
                content, ok, err = "", False, str(e)
            per_item[item["id"]] = {"ok": ok, "content": content[:200], "error": err}
        wall = time.time() - t0
        cats = {}
        for item in suite:
            cats.setdefault(item["category"], []).append(per_item[item["id"]]["ok"])
        agg = {"overall": sum(all_ok for cat in cats.values() for all_ok in cat),
               "n": len(suite)}
        per_cat = {c: {"correct": sum(v), "n": len(v)} for c, v in cats.items()}
        out["runs"].append({"rep": rep, "per_item": per_item, "per_category": per_cat,
                            "aggregate": agg, "wall_s": round(wall, 1)})
        acc = 100.0 * agg["overall"] / agg["n"]
        print(f"rep {rep}: overall {agg['overall']}/{agg['n']} = {acc:.1f}% ({wall:.1f}s)")
    # pooled summary across reps (per-item majority is NOT used; report all reps)
    accs = [100.0 * r["aggregate"]["overall"] / r["aggregate"]["n"] for r in out["runs"]]
    pooled_correct = sum(r["aggregate"]["overall"] for r in out["runs"])
    pooled_n = sum(r["aggregate"]["n"] for r in out["runs"])
    lo, hi = wilson_ci(pooled_correct / pooled_n, pooled_n)
    out["summary"] = {"per_rep_overall_acc": [round(a, 2) for a in accs],
                      "mean_acc": round(sum(accs) / len(accs), 2),
                      "median_acc": round(sorted(accs)[len(accs) // 2], 2),
                      "pooled_acc": round(100 * pooled_correct / pooled_n, 2),
                      "wilson95": [round(lo * 100, 2), round(hi * 100, 2)],
                      "per_category_pooled": {}}
    cats = {}
    for r in out["runs"]:
        for c, v in r["per_category"].items():
            cats.setdefault(c, [0, 0])
            cats[c][0] += v["correct"]
            cats[c][1] += v["n"]
    out["summary"]["per_category_pooled"] = {
        c: {"acc": round(100 * a / n, 2), "n": n} for c, (a, n) in sorted(cats.items())}
    dest = Path(args.outdir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{args.label}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"saved {path}")
    print(f"summary: {out['summary']['pooled_acc']}% pooled "
          f"(mean {out['summary']['mean_acc']}%, wilson95 {out['summary']['wilson95']})")


def gate(args):
    c = json.loads(Path(args.control).read_text())["summary"]
    t = json.loads(Path(args.treatment).read_text())["summary"]
    dc = t["pooled_acc"] - c["pooled_acc"]
    print(f"control   : {c['pooled_acc']}% pooled (wilson95 {c['wilson95']}) "
          f"per-rep {c['per_rep_overall_acc']}")
    print(f"treatment : {t['pooled_acc']}% pooled (wilson95 {t['wilson95']}) "
          f"per-rep {t['per_rep_overall_acc']}")
    print(f"delta     : {dc:+.2f} percentage points")
    lo, hi = t["wilson95"]
    c_lo, c_hi = c["wilson95"]
    if dc <= -0.5:
        clear = hi < c_lo or abs(dc) > 1.0
        print("VERDICT: FAIL — regression beyond 0.5 pp" + ("" if clear else
              " (investigate: treatment CI overlaps control CI)"))
    elif abs(dc) > 0.5 or hi < c_lo or c_hi < lo:
        print("VERDICT: INCONCLUSIVE-ish — delta beyond 0.5 pp but check CIs; "
              "per policy, report per-category and rerun with more reps")
    else:
        print("VERDICT: within the 0.5 pp gate — eligible to pass (paired with performance gates)")
    worst = min(t["per_category_pooled"].items(),
                key=lambda kv: kv[1]["acc"] - c["per_category_pooled"].get(kv[0], {"acc": 0})["acc"])
    print(f"largest per-category drop: {worst[0]} "
          f"({c['per_category_pooled'].get(worst[0], {}).get('acc')} -> {worst[1]['acc']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base", default="http://127.0.0.1:8000")
    r.add_argument("--label", required=True)
    r.add_argument("--reps", type=int, default=5)
    r.add_argument("--suite", default=str(Path(__file__).parent / "suite_v1.jsonl"))
    r.add_argument("--outdir", default=str(Path(__file__).parent / "results"))
    g = sub.add_parser("gate")
    g.add_argument("--control", required=True)
    g.add_argument("--treatment", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        gate(args)


if __name__ == "__main__":
    sys.exit(main())
