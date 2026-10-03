"""Sixth arm: the local Qwen3.8-27B Q4_K_M on the 3090, same tasks, same grader.

Imports the task set and grader from run.py so the comparison is apples-to-apples.
Talks to a llama-server OpenAI-compatible endpoint over Tailscale.
"""
from __future__ import annotations

import json
import os
import pathlib
import statistics
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).parent
import sys

sys.path.insert(0, str(HERE))
from run import SYSTEM, TASKS, grade  # noqa: E402

LOCAL_BASE = os.environ.get("LOCAL_BASE_URL", "http://100.77.225.85:8080/v1")
LOCAL_LABEL = "local Qwen3.8-27B Q4_K_M (3090)"


def local_chat(prompt: str, max_tokens: int = 1500) -> dict:
    body = {
        "model": "local",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"{LOCAL_BASE}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read().decode())
        status, err = resp.status, None
    except urllib.error.HTTPError as exc:
        payload, status, err = {}, exc.code, exc.read().decode()[:300]
    except Exception as exc:
        payload, status, err = {}, 0, f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - started
    choice = (payload.get("choices") or [{}])[0]
    return {
        "content": ((choice.get("message") or {}).get("content") or "").strip(),
        "ok": status == 200 and not err,
        "http_status": status,
        "usage": payload.get("usage") or {},
        "latency_s": round(elapsed, 3),
        "error": err,
    }


def main() -> int:
    records = []
    for difficulty, prompt, expected in TASKS:
        res = local_chat(prompt)
        ok, got = grade(res["content"], expected)
        rec = {
            "model": LOCAL_LABEL,
            "difficulty": difficulty,
            "prompt": prompt,
            "expected": expected,
            "got": got,
            "correct": ok,
            "fabricated": expected is None and not ok,
            "ok": res["ok"],
            "http_status": res["http_status"],
            "prompt_tokens": res["usage"].get("prompt_tokens") or 0,
            "completion_tokens": res["usage"].get("completion_tokens") or 0,
            "latency_s": res["latency_s"],
            "error": res["error"],
        }
        records.append(rec)
        print(f"{difficulty:<8} {'P' if ok else 'F'} {rec['completion_tokens']:>5}tok "
              f"{rec['latency_s']:>7}s  got={got[:40]}")

    out = HERE / "results"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "raw_local.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    ans = [r for r in records if r["expected"] is not None]
    una = [r for r in records if r["expected"] is None]
    correct = sum(1 for r in ans if r["correct"])
    tokens = sum(r["completion_tokens"] for r in records)
    lat = sorted(r["latency_s"] for r in records)
    summary = {
        "answerable_correct": f"{correct}/{len(ans)}",
        "answerable_accuracy": round(correct / len(ans), 3),
        "fabrications": sum(1 for r in una if r["fabricated"]),
        "fabrication_rate": round(sum(1 for r in una if r["fabricated"]) / len(una), 3),
        "abstentions_on_unanswerable": sum(1 for r in una if r["correct"]),
        "total_completion_tokens": tokens,
        "correct_per_1k_tokens": round((correct / tokens) * 1000, 3) if tokens else None,
        "median_latency_s": round(lat[len(lat) // 2], 3),
        "errors": sum(1 for r in records if not r["ok"]),
    }
    (out / "summary_local.json").write_text(
        json.dumps({LOCAL_LABEL: summary}, indent=2) + "\n"
    )
    print("\n=== local arm ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
