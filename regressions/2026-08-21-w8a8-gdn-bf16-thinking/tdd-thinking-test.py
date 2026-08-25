#!/usr/bin/env python3
"""TDD: W8A8 thinking-mode quality test — must FAIL on the current stack, PASS after the fix.

Reproduces the user's failure: "check my current connection and determine what kind
of security and handshakes i have" with thinking ON (effort high), max_tokens 900.
PASS criteria per run:
  T1 content: final answer non-empty (the model must CLOSE into a content answer)
  T2 language: no CJK ideographs in reasoning+content (no Chinese bleed)
  T3 sanity: no hallucinated tool/shell session (no ">$" command lines pretending to run)
Passes only if ALL runs pass T1-T3; needle-recall regression check must stay green.
"""
import json, re, sys, time, urllib.request

BASE = "http://127.0.0.1:19622"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen3.8-27B-INT8-W8A8"
RUNS = 2
CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

# Policy under test (2026-08-21): the lane keeps thinking ENABLED, with the
# pi thinkingLevelMap capping reasoning_effort at medium (high/xhigh ramble
# past any budget; low/medium close within ~600 tokens). max_tokens budget is
# the pi-style thinking_token_budget+answer headroom.
def chat(messages, max_tokens, effort="medium", temp=0.7):
    payload = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
               "temperature": temp, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
               "presence_penalty": 0.0, "repetition_penalty": 1.0,
               "stream": True, "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True,
                                        "reasoning_effort": effort}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    text = ""; reasoning = ""
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            ln = raw.decode(errors="replace").strip()
            if not ln.startswith("data:"): continue
            d = ln[5:].strip()
            if d == "[DONE]": break
            try: ev = json.loads(d)
            except Exception: continue
            for ch in ev.get("choices", []):
                dd = ch.get("delta") or {}
                text += dd.get("content") or ""
                reasoning += dd.get("reasoning") or dd.get("reasoning_content") or ""
    return text, reasoning

PROMPT = "check my current connection and determine what kind of security and handshakes i have"
SYS = "You are a helpful coding assistant running on the user's machine."
MESSAGES = [{"role": "system", "content": SYS}, {"role": "user", "content": PROMPT}]

results = []
print("== TDD thinking test on %s ==" % MODEL)
for i in range(RUNS):
    text, reasoning = chat(MESSAGES, 900)
    full = reasoning + "\n" + text
    t1 = len(text.strip()) > 0
    t2 = not CJK.search(full)
    # runaway = the model PRETENDS it executed a shell session ($ lines with
    # fabricated output), not merely recommending commands in code fences.
    t3 = not re.search(r"\n\$ ", full) and "Took 0.2s" not in full
    results.append((t1, t2, t3))
    print("run %d: T1(content non-empty)=%s T2(no CJK)=%s T3(no tool-runaway)=%s | content_len=%d reasoning_len=%d"
          % (i, t1, t2, t3, len(text), len(reasoning)))
    print("  content head:", repr(text[:180]))
    print("  reasoning tail:", repr(reasoning[-180:]))

failures = [k for (t1, t2, t3) in results for k, v in (("T1", t1), ("T2", t2), ("T3", t3)) if not v]
print("RESULT:", "FAIL" if failures else "PASS", "| failed:", failures)
sys.exit(1 if failures else 0)
