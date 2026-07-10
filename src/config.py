"""Central config for the Track-1 routing agent.

SCORING REALITY: only tokens sent through Fireworks count against us.
Local (vLLM) inference is free. So the ladder is: LOCAL first (free), then the
cheapest Fireworks model that clears the accuracy bar.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-14B-Instruct")  # W7900 48GB fits 14B fp16 (~28GB)

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.75"))
SELF_CONSISTENCY_K = int(os.getenv("SELF_CONSISTENCY_K", "5"))

# Fireworks is only used when a key is present (the scoring env, or local testing
# with your own $50 key). Without it, the agent runs local-only (no escalation).
FIREWORKS_ENABLED = bool(FIREWORKS_API_KEY)


@dataclass(frozen=True)
class Tier:
    name: str
    model: str
    counted: bool          # True => tokens count against our leaderboard score
    price: float = 0.0     # relative $/1M tokens, cheapest first. CONFIRM live prices.


# Tier 0 — LOCAL, free. Everything below is Fireworks (counted) and is only
# reached when the verifier rejects the free local answer.
LOCAL_TIER = Tier(name="local", model=LOCAL_MODEL, counted=False, price=0.0)

# CONFIRM model IDs + live prices against the Fireworks pricing page / track rules.
FIREWORKS_LADDER = [
    Tier("fw-gemma2-9b",    "accounts/fireworks/models/gemma-2-9b-it",          counted=True, price=0.20),
    Tier("fw-llama3.1-8b",  "accounts/fireworks/models/llama-v3p1-8b-instruct",  counted=True, price=0.20),
    Tier("fw-llama3.3-70b", "accounts/fireworks/models/llama-v3p3-70b-instruct", counted=True, price=0.90),
    Tier("fw-qwen2.5-72b",  "accounts/fireworks/models/qwen2p5-72b-instruct",    counted=True, price=0.90),
]
