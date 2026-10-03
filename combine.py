"""Merge the five cloud arms and the local arm into one frontier table.

Separates two different fail-closed outcomes that a naive grader conflates:
  invented_answer  - produced a concrete value for an unanswerable question
  budget_exhausted - produced no ANSWER line at all and burned the whole budget
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"

LOCAL_LABEL = "local Qwen3.8-27B Q4_K_M (3090)"


def from_cloud() -> dict:
    data = json.loads((RESULTS / "summary.json").read_text())
    return data["meta"], data["tiers"]


def from_local() -> dict:
    rows = [json.loads(l) for l in (RESULTS / "raw_local.jsonl").read_text().splitlines() if l]
    ans = [r for r in rows if r["expected"] is not None]
    una = [r for r in rows if r["expected"] is None]
    correct = sum(1 for r in ans if r["correct"])
    tokens = sum(r["completion_tokens"] for r in rows)
    lat = sorted(r["latency_s"] for r in rows)
    invented = sum(1 for r in una if r["got"] and not r["correct"])
    exhausted = sum(1 for r in una if not r["got"])
    return {
        "answerable_correct": f"{correct}/{len(ans)}",
        "answerable_accuracy": round(correct / len(ans), 3),
        "abstained_on_unanswerable": sum(1 for r in una if r["correct"]),
        "invented_answer": invented,
        "budget_exhausted": exhausted,
        "fail_closed_rate": round(sum(1 for r in una if r["correct"]) / len(una), 3),
        "total_completion_tokens": tokens,
        "correct_per_1k_tokens": round((correct / tokens) * 1000, 3) if tokens else None,
        "median_latency_s": round(lat[len(lat) // 2], 3),
        "errors": sum(1 for r in rows if not r["ok"]),
    }


def main() -> int:
    meta, tiers = from_cloud()
    tiers = dict(tiers)
    tiers[LOCAL_LABEL] = from_local()

    # Pareto on (maximise accuracy, minimise tokens)
    dominated = {}
    for m, s in tiers.items():
        dominated[m] = any(
            o != m
            and tiers[o]["answerable_accuracy"] >= s["answerable_accuracy"]
            and tiers[o]["total_completion_tokens"] <= s["total_completion_tokens"]
            and (
                tiers[o]["answerable_accuracy"] > s["answerable_accuracy"]
                or tiers[o]["total_completion_tokens"] < s["total_completion_tokens"]
            )
            for o in tiers
        )

    final = {
        "meta": {
            **meta,
            "arms_total": len(tiers),
            "cloud_arms": [t for t in tiers if t != LOCAL_LABEL],
            "local_arm": LOCAL_LABEL,
            "pareto_non_dominated": sorted(m for m, d in dominated.items() if not d),
            "pareto_dominated": sorted(m for m, d in dominated.items() if d),
        },
        "arms": tiers,
    }
    (RESULTS / "final_summary.json").write_text(json.dumps(final, indent=2) + "\n")

    print(f"{'arm':<34} {'acc':>5} {'tok':>7} {'cor/1k':>7} {'p50':>7} {'fail-closed':>11}")
    for m, s in tiers.items():
        fc = s.get("fail_closed_rate", s.get("abstentions_on_unanswerable", 0))
        print(f"{m[:33]:<34} {s['answerable_accuracy']:>5} "
              f"{s['total_completion_tokens']:>7} {s['correct_per_1k_tokens']:>7} "
              f"{s['median_latency_s']:>7} {str(fc):>11}")
    print("\nnon-dominated:", final["meta"]["pareto_non_dominated"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
