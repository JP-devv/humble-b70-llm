#!/usr/bin/env python3
"""bigPP prompt-processing probe: single-prompt TTFT at exact prompt sizes.

Uses /v1/completions with raw token-id prompts (exact length control, no
template tokens). Each rep draws fresh random token ids so prefix caching
cannot hit (cached_tokens verified == 0 per request). rep 0 is shape-cold
(lazy kernel init may inflate it), rep 1 is shape-warm/content-cold.

Metrics: TTFT = request start -> first streamed chunk with text.
PP tok/s = prompt_tokens / TTFT.
"""
import argparse
import json
import random
import time
import urllib.request


def probe(base_url, model, size, max_tokens, rng):
    ids = [rng.randint(1000, 50000) for _ in range(size)]
    body = {
        "model": model,
        "prompt": ids,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base_url + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    usage = None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if (
                ttft is None
                and chunk.get("choices")
                and any(c.get("text") for c in chunk["choices"])
            ):
                ttft = time.perf_counter() - t0
            if chunk.get("usage"):
                usage = chunk["usage"]
    wall = time.perf_counter() - t0
    pt = usage["prompt_tokens"] if usage else None
    cached = (
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if usage
        else None
    )
    return {
        "target_size": size,
        "prompt_tokens": pt,
        "cached_tokens": cached,
        "ttft_ms": (ttft * 1000) if ttft else None,
        "wall_s": wall,
        "pp_tok_s": (pt / ttft) if (pt and ttft) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:19622")
    ap.add_argument("--model", default="Qwen3.8-27B-UNC-FP8")
    ap.add_argument("--sizes", default="633,8192,32768")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for size in [int(s) for s in args.sizes.split(",")]:
        for rep in range(args.reps):
            r = probe(args.base_url, args.model, size, args.max_tokens, rng)
            r["rep"] = rep
            rows.append(r)
            print(
                "size=%d rep=%d prompt_tokens=%s cached=%s ttft=%.1fms pp=%.0f tok/s"
                % (
                    size,
                    rep,
                    r["prompt_tokens"],
                    r["cached_tokens"],
                    r["ttft_ms"] or -1,
                    r["pp_tok_s"] or -1,
                ),
                flush=True,
            )
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
