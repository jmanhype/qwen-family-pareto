"""Minimal client for the Qwen Ambassador ModelScope route.

Pins the endpoint and model id so results are reproducible.
Never prints or logs the credential.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
import urllib.error
import urllib.request

BASE_URL = os.environ.get("QWEN_BASE_URL", "https://api-inference.modelscope.ai/v1")
MODEL = os.environ.get("QWEN_MODEL", "Qwen-Ambassador/Qwen3.8-Max")

_SECRET_FILE = pathlib.Path.home() / ".codex/codex-router/modelscope-api-key.secret"


def load_key() -> str:
    for env in ("MODELSCOPE_API_KEY", "MODELSCOPE_ACCESS_TOKEN"):
        v = os.environ.get(env)
        if v:
            return v.strip()
    if _SECRET_FILE.exists():
        return _SECRET_FILE.read_text().strip()
    raise RuntimeError("no ModelScope credential available")


def chat(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int = 512,
    effort: str | None = None,
    temperature: float | None = None,
    timeout: int = 180,
) -> dict:
    """Single non-streaming completion. Returns a normalized record."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body: dict = {"model": model or MODEL, "messages": messages, "max_tokens": max_tokens}
    if effort:
        body["reasoning_effort"] = effort
    if temperature is not None:
        body["temperature"] = temperature

    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {load_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        status, err = resp.status, None
    except urllib.error.HTTPError as exc:
        payload = {}
        status, err = exc.code, exc.read().decode()[:400]
    except Exception as exc:  # transport failure
        payload = {}
        status, err = 0, f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - started

    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {
        "ok": status == 200 and not err,
        "http_status": status,
        "error": err,
        "model": payload.get("model", model or MODEL),
        "content": (message.get("content") or "").strip(),
        "reasoning_present": bool(message.get("reasoning_content")),
        "usage": payload.get("usage") or {},
        "finish_reason": choice.get("finish_reason"),
        "requested_effort": effort,
        "latency_s": round(elapsed, 3),
    }


def append_jsonl(path: str | pathlib.Path, records) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
