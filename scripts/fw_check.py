"""Quick Fireworks check: does the API work with our credits, and which model IDs are valid?

Usage on the pod:
    export FIREWORKS_API_KEY=fw_your_key_here
    python scripts/fw_check.py
"""
import os

from openai import OpenAI

key = os.environ.get("FIREWORKS_API_KEY")
if not key:
    raise SystemExit("Set your key first:  export FIREWORKS_API_KEY=fw_...")

base = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
client = OpenAI(base_url=base, api_key=key)
print(f"Base URL: {base}\n")

print("=== Models your account can see (filtered) ===")
try:
    ids = sorted(m.id for m in client.models.list().data)
    keys = ("gemma", "minimax", "kimi", "llama", "qwen")
    relevant = [i for i in ids if any(k in i.lower() for k in keys)]
    for i in relevant:
        print(" ", i)
    print(f"(total {len(ids)} models; {len(relevant)} relevant)")
except Exception as e:
    print("  models.list() failed:", str(e)[:200])

print("\n=== Try a chat call on candidate Track-1 model IDs ===")
candidates = [
    "accounts/fireworks/models/gemma-4-31b-it-nvfp4",
    "gemma-4-31b-it-nvfp4",
    "accounts/fireworks/models/gemma-4-31b-it",
    "accounts/fireworks/models/minimax-m3",
    "accounts/fireworks/models/kimi-k2p7-code",
    "accounts/fireworks/models/llama-v3p1-8b-instruct",
]
for model in candidates:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with only: OK"}],
            max_tokens=5,
            temperature=0,
        )
        print(f"  OK   {model}  -> {r.choices[0].message.content!r}  (tokens={r.usage.total_tokens})")
    except Exception as e:
        print(f"  ERR  {model}  -> {str(e)[:120]}")
