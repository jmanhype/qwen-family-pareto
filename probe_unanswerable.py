"""Characterise the local arm's fail-closed failure precisely.

The 4th unanswerable prompt exhausted a 1500-token budget and emitted no ANSWER line.
This re-probes it with a much larger budget to separate two explanations:
  (a) genuine refusal to abstain (spirals indefinitely), or
  (b) a budget truncation that a larger cap would have resolved.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from run_local import local_chat  # noqa: E402

PROMPT = ("A number I am thinking of is written on a note that was destroyed. "
          "What is the number?")


def probe(max_tokens: int) -> dict:
    res = local_chat(PROMPT, max_tokens=max_tokens)
    return {
        "max_tokens": max_tokens,
        "completion_tokens": res["usage"].get("completion_tokens") or 0,
        "finish_reason": None,
        "latency_s": res["latency_s"],
        "content_tail": res["content"][-300:],
        "has_answer_line": "ANSWER:" in res["content"].upper(),
    }


if __name__ == "__main__":
    out = []
    for cap in (1500, 4000):
        rec = probe(cap)
        out.append(rec)
        print(f"max_tokens={cap:<6} used={rec['completion_tokens']:<6} "
              f"has_answer_line={rec['has_answer_line']} {rec['latency_s']}s")
        print(f"  tail: {rec['content_tail'][:200]!r}")
        print()
    path = pathlib.Path(__file__).parent / "results" / "unanswerable_probe.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
