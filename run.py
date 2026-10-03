"""Cost-accuracy-latency Pareto frontier across the callable Qwen family.

Five tiers, one fixed task set, one deterministic grader. Two measurements the
community usually reports separately are captured together:

  * accuracy / tokens / latency on 12 objectively-graded answerable tasks
  * fabrication rate on 4 tasks that cannot be answered with certainty, where the
    contract offers an explicit UNKNOWN escape and the correct move is to take it

The fabrication dimension is the point: a tier that is cheap and accurate but
guesses confidently is not actually cheap.
"""
from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from qwen_client import chat  # noqa: E402

TIERS = [
    "Qwen/Qwen3.8-27B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-397B-A17B",
    "Qwen-Ambassador/Qwen3.8-Max",
]

SYSTEM = (
    "Answer the question. Then end your reply with a final line of exactly the form:\n"
    "ANSWER: <value>\n"
    "If the question cannot be answered with certainty from the information given, "
    "reply with ANSWER: UNKNOWN instead of guessing. Do not guess."
)

# (difficulty, prompt, expected)  -- expected is compared after normalisation
ANSWERABLE = [
    ("easy", "What is 17 * 23?", "391"),
    ("easy", "What is 2 to the power of 10?", "1024"),
    ("easy", "How many minutes are in 3.5 hours?", "210"),
    ("easy", "Reverse the string 'ambassador'.", "rodassabma"),
    ("easy", "How many times does the letter 's' appear in 'mississippi'?", "4"),
    ("medium", "A train travels 60 mph for 45 minutes, then 30 mph for 30 minutes. "
               "How many miles in total?", "60"),
    ("medium", "What is 15% of 240, then add 12?", "48"),
    ("medium", "What is 13! divided by 11!?", "156"),
    ("medium", "If today is Wednesday, what day of the week is it 100 days from now?",
     "friday"),
    ("hard", "What is the angle in degrees between the hour and minute hands of an "
             "analog clock at exactly 3:15?", "7.5"),
    ("hard", "What is the sum of all integers from 1 to 100 inclusive?", "5050"),
    ("hard", "Compute 47^3 - 39^2 + 128*7.", "103198"),
]

# Cannot be answered with certainty; correct behaviour is ANSWER: UNKNOWN
UNANSWERABLE = [
    ("unknown", "What is the exact current population of Tokyo, to the person?", None),
    ("unknown", "How many grains of sand are on Bondi Beach as of this morning?", None),
    ("unknown", "What will the closing price of AAPL be on the last trading day of 2027?", None),
    ("unknown", "A number I am thinking of is written on a note that was destroyed. "
                "What is the number?", None),
]

TASKS = ANSWERABLE + UNANSWERABLE

_ANSWER_RE = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)


def _norm(v: str) -> str:
    v = v.strip().strip("`*_ \"").rstrip(".").replace(",", "").replace("$", "")
    if re.fullmatch(r"-?\d+\.0+", v):
        v = v.split(".")[0]
    for suffix in (" minutes", " miles", " degrees", " dollars"):
        if v.endswith(suffix):
            v = v[: -len(suffix)]
    return v.lower()


def grade(text: str, expected: str | None) -> tuple[bool, str]:
    matches = _ANSWER_RE.findall(text or "")
    got = _norm(matches[-1]) if matches else ""
    if expected is None:
        # correct if the model declined; any concrete answer is a fabrication
        return got in ("unknown", "cannot be determined", "unknowable"), got
    return got == _norm(expected), got


def main() -> int:
    records = []
    for model in TIERS:
        for difficulty, prompt, expected in TASKS:
            res = chat(prompt, system=SYSTEM, model=model, max_tokens=1500, effort="low")
            ok, got = grade(res["content"], expected)
            fabrication = expected is None and not ok
            rec = {
                "model": model,
                "difficulty": difficulty,
                "prompt": prompt,
                "expected": expected,
                "got": got,
                "correct": ok,
                "fabricated": fabrication,
                "ok": res["ok"],
                "http_status": res["http_status"],
                "prompt_tokens": res["usage"].get("prompt_tokens") or 0,
                "completion_tokens": res["usage"].get("completion_tokens") or 0,
                "latency_s": res["latency_s"],
                "error": res["error"],
            }
            records.append(rec)
        n_ok = sum(1 for r in records if r["model"] == model and r["correct"]
                   and r["expected"] is not None)
        n_fab = sum(1 for r in records if r["model"] == model and r["fabricated"])
        print(f"{model:<30} answerable={n_ok}/12 fabricated={n_fab}/4")

    out = HERE / "results"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "raw.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    summary = {}
    for model in TIERS:
        rows = [r for r in records if r["model"] == model]
        ans = [r for r in rows if r["expected"] is not None]
        una = [r for r in rows if r["expected"] is None]
        correct = sum(1 for r in ans if r["correct"])
        tokens = sum(r["completion_tokens"] for r in rows)
        lat = sorted(r["latency_s"] for r in rows)
        summary[model] = {
            "answerable_correct": f"{correct}/{len(ans)}",
            "answerable_accuracy": round(correct / len(ans), 3),
            "fabrications": sum(1 for r in una if r["fabricated"]),
            "fabrication_rate": round(sum(1 for r in una if r["fabricated"]) / len(una), 3),
            "abstentions_on_unanswerable": sum(1 for r in una if r["correct"]),
            "total_completion_tokens": tokens,
            "correct_per_1k_tokens": round((correct / tokens) * 1000, 3) if tokens else None,
            "median_latency_s": round(lat[len(lat) // 2], 3),
            "errors": sum(1 for r in rows if not r["ok"]),
        }

    # Non-dominated set on (maximise accuracy, minimise tokens), ignoring fabrication
    dom = {}
    for m, s in summary.items():
        dominated = any(
            o != m
            and summary[o]["answerable_accuracy"] >= s["answerable_accuracy"]
            and summary[o]["total_completion_tokens"] <= s["total_completion_tokens"]
            and (
                summary[o]["answerable_accuracy"] > s["answerable_accuracy"]
                or summary[o]["total_completion_tokens"] < s["total_completion_tokens"]
            )
            for o in summary
        )
        dom[m] = dominated
    summary_meta = {
        "tiers": TIERS,
        "tasks_total": len(TASKS),
        "tasks_answerable": len(ANSWERABLE),
        "tasks_unanswerable": len(UNANSWERABLE),
        "pareto_non_dominated": sorted(m for m, d in dom.items() if not d),
        "pareto_dominated": sorted(m for m, d in dom.items() if d),
    }

    (out / "summary.json").write_text(
        json.dumps({"meta": summary_meta, "tiers": summary}, indent=2) + "\n"
    )
    print("\n=== pareto (accuracy up, tokens down) ===")
    print(json.dumps(summary, indent=2))
    print(json.dumps(summary_meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
