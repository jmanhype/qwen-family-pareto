# Six-arm Qwen Pareto frontier: accuracy is saturated, token efficiency is not

**Finding: on 12 objectively-graded answerable tasks, all six arms scored 12/12 —
five cloud tiers and a local 3090. So the frontier is not decided by accuracy at all;
it is decided by token efficiency, and there the spread is 7.9×. The flagship
`Qwen-Ambassador/Qwen3.8-Max` is the only non-dominated arm: tied-best accuracy with
the *fewest* tokens. And the local quant is the only arm that cannot fail closed.**

```
python3 run.py           # 5 cloud tiers x 16 tasks = 80 calls
python3 run_local.py     # local arm, same tasks, same grader
python3 combine.py       # merges both into results/final_summary.json
```

## The frontier

| arm | answerable | tokens | correct / 1k tok | p50 latency | fail-closed |
|-----|-----------|--------|------------------|-------------|-------------|
| `Qwen-Ambassador/Qwen3.8-Max` | **12/12** | **2 653** | **4.52** | 5.2 s | 4/4 |
| `Qwen/Qwen3.8-27B` | 12/12 | 3 056 | 3.93 | **4.2 s** | 4/4 |
| local Qwen3.8-27B Q4_K_M (3090) | 12/12 | 3 320 | 3.61 | 7.7 s | **3/4** |
| `Qwen/Qwen3.5-122B-A10B` | 12/12 | 15 031 | 0.80 | 7.1 s | 4/4 |
| `Qwen/Qwen3.5-397B-A17B` | 12/12 | 16 397 | 0.73 | 14.0 s | 4/4 |
| `Qwen/Qwen3.5-35B-A3B` | 12/12 | **20 912** | 0.57 | 9.1 s | 4/4 |

**Non-dominated: `Qwen-Ambassador/Qwen3.8-Max` only.** Every other arm is beaten on
accuracy (tied) *and* on tokens by Max.

## Four things worth taking from this

1. **Accuracy is the wrong axis for this task class.** Six very different models, three
   architecture families, one quantization level, and a 20× parameter spread all land on
   12/12. If you are choosing a tier by accuracy on verifiable short-answer work, you are
   choosing on noise. Choose on tokens.
2. **More parameters did not mean fewer tokens.** The 397B used 6.2× Max's tokens and was
   the slowest arm (14.0 s). The 35B-A3B MoE used the *most* tokens of any arm (20 912) —
   nearly 8× Max — for identical answers. Model size and token efficiency are separate
   axes, and the flagship wins on the one that matters here.
3. **The local quant is perfect until it is asked to stop.** 12/12 on answerable tasks —
   indistinguishable from the cloud tiers — but it fails closed on 1 of 4 unanswerable
   prompts. See below; the failure is worse than a wrong answer.
4. **Local is dominated, but only narrowly.** The 3090 arm used 25% more tokens than Max
   and ran 1.5× slower at p50 — while costing nothing per token. Token count is a proxy
   for API spend, not for local cost, so the honest read is "local is competitive on
   tokens, worse on wall-clock, and pays no per-token bill".

## The interesting failure: local can't refuse

Four prompts cannot be answered with certainty (an unknowable number, a lost note, a
future stock price, an exact city population). The contract offered an explicit
`ANSWER: UNKNOWN` escape. All five cloud tiers took it every time. The local arm took it
3 of 4 times and, on the fourth, **spiralled instead of answering**:

| budget | tokens used | ANSWER line emitted | wall-clock |
|--------|-------------|---------------------|------------|
| 1 500 | 1 500 | no | 43.5 s |
| 4 000 | 4 000 | no | 111.4 s |

Raising the cap 2.7× did not resolve it — it consumed the larger budget too, with empty
`content`, all of it spent in reasoning. This is not a truncation artefact; a bigger
budget buys nothing. In an agent loop that is worse than a wrong answer: an unbounded
latency and cost hole triggered by a question the model should have declined.

**A plausible cause, stated as a hypothesis:** the weights used are an *uncensored*
community quantization (`qwen38-27b-uncensored/Q4_K_M.gguf`), not the official
Qwen3.8-27B checkpoint. Refusal behaviour is exactly what such fine-tunes alter. I did not
test the official weights at 4-bit, so I cannot separate "Q4 damaged abstention" from
"this fine-tune damaged abstention" from "27B-class models in general spiral here".

## Honest caveats

- **12 answerable tasks, 4 unanswerable, one run each.** The accuracy result is saturated
  and therefore weak evidence about capability; the token ordering is the real signal and
  it is single-sample.
- **The local arm is not the same model as `Qwen/Qwen3.8-27B`.** It is a community
  **uncensored Q4_K_M** quantization. Any local-vs-cloud conclusion must carry that label.
- **Token counts come from different serving stacks** (llama.cpp vs the ModelScope
  gateway). Tokenizers differ, so cross-stack token counts are indicative, not exact.
- **Local latency includes a Tailscale hop** and ran unbatched on one 3090 behind
  `llama.cpp` (`-ngl 99`, 32 768 ctx, 4 slots). It is not a tuned deployment.
- **No dollar cost is measured.** Tokens proxy API spend; local inference costs
  electricity and GPU time instead.
- Nothing here says these models are equal in general — only on these 12 items.

## Reproduce

Cloud arms: `python3 run.py` (needs `MODELSCOPE_API_KEY`, or the secret file the bundled
client falls back to). Local arm: `python3 run_local.py` against a llama-server exposing
an OpenAI-compatible API; set `LOCAL_BASE_URL` to point it elsewhere.

## Files

- `run.py` — five cloud arms over 16 graded tasks
- `run_local.py` — the local arm, importing the same task set and grader
- `probe_unanswerable.py` — the budget-escalation probe behind the table above
- `combine.py` — merges arms into `results/final_summary.json`
- `results/raw.jsonl` — 80 cloud records with usage tokens
- `results/raw_local.jsonl` — 16 local records with usage tokens
- `results/final_summary.json` — the merged frontier
- `receipts.sha256` — hashes of harnesses, client, README and results
