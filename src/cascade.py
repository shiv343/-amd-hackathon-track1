"""The routing cascade.

- LOCAL mode  (USE_LOCAL=1, default): try the free local model first; if the
  verifier trusts it, submit it for 0 tokens; otherwise escalate to Fireworks
  cheapest-first.
- SCOUT mode  (USE_LOCAL=0): no local model shipped — route straight to
  Fireworks (cheapest-first). Used by the lean scout container.

The same function serves both, plus local-only pod testing (no Fireworks key).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import clients, config
from .verifier import Validator, verify


@dataclass
class Result:
    id: str
    answer: str
    tier: str            # which model answered ("local" is the win)
    counted_tokens: int  # Fireworks tokens spent on this task (0 if local)


SYSTEM = (
    "You are a precise assistant. Answer the task directly, correctly, and completely. "
    "Obey every format constraint stated in the task exactly — e.g. an exact number of "
    "sentences or bullet points, a required label, or a requested one-sentence reason. "
    "Do not add preamble, restate the question, or extra commentary beyond what is asked."
)


def answer_task(task: dict, validator: Optional[Validator] = None) -> Result:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": task["prompt"]}]
    best: Optional[Result] = None

    # Tier 0 — LOCAL (free), when enabled and reachable.
    if config.LOCAL_ENABLED:
        try:
            local = clients.local_chat(msgs, temperature=0.0, max_tokens=2048)
            accept, _conf = verify(task, local.text, validator)
            if accept:
                return Result(task["id"], local.text, config.LOCAL_TIER.name, 0)
            best = Result(task["id"], local.text, config.LOCAL_TIER.name + "*", 0)
        except Exception:
            best = None  # local unreachable (e.g. scout container) — use Fireworks

    # Fireworks — cheapest first, stop at the first verified answer. This is the
    # escalation path in local mode and the primary path in scout mode.
    if config.FIREWORKS_ENABLED:
        for tier in config.FIREWORKS_LADDER:
            try:
                start = clients.METER.total
                fw = clients.fireworks_chat(tier.model, msgs, temperature=0.0, max_tokens=2048)
                spent = clients.METER.total - start
            except Exception:
                continue  # bad id / transient error — try the next model
            best = Result(task["id"], fw.text, tier.name, spent)
            if validator is None or validator(task, fw.text):
                return best

    # Best effort: local fallback, last Fireworks attempt, or empty.
    return best or Result(task["id"], "", "none", 0)
