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

# Set USE_LOCAL=0 to disable the local tier entirely — used by the lean "scout"
# container that routes everything through Fireworks (no local model shipped).
LOCAL_ENABLED = os.getenv("USE_LOCAL", "1") == "1"


@dataclass(frozen=True)
class Tier:
    name: str
    model: str
    counted: bool          # True => tokens count against our leaderboard score
    price: float = 0.0     # relative $/1M tokens, cheapest first. CONFIRM live prices.


# Tier 0 — LOCAL, free. Everything below is Fireworks (counted) and is only
# reached when the verifier rejects the free local answer.
LOCAL_TIER = Tier(name="local", model=LOCAL_MODEL, counted=False, price=0.0)

# The Track-1 ALLOWED Fireworks models (AMD ACT II). Cheapest -> priciest here is
# a best guess — CONFIRM exact model-ID strings + prices from the Participant Guide
# / Fireworks pricing and reorder. Route code tasks straight to the code model.
FIREWORKS_LADDER = [
    # CONFIRMED callable via fw_check.py (need the full "accounts/fireworks/models/" id):
    Tier("fw-minimax-m3", "accounts/fireworks/models/minimax-m3",     counted=True, price=1.20),  # cheapest, general
    Tier("fw-kimi-code",  "accounts/fireworks/models/kimi-k2p7-code", counted=True, price=4.00),  # returns clean text; code
    # Allowed per the rules but 404 on the public API for this account — may exist
    # only on the judge's proxy. Kept as fallbacks (auto-skipped if unavailable).
    Tier("fw-gemma4-nvfp4", "accounts/fireworks/models/gemma-4-31b-it-nvfp4", counted=True, price=0.50),
    Tier("fw-gemma4-a4b",   "accounts/fireworks/models/gemma-4-26b-a4b-it",   counted=True, price=0.50),
    Tier("fw-gemma4-31b",   "accounts/fireworks/models/gemma-4-31b-it",       counted=True, price=1.00),
]
