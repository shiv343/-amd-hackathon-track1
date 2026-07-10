"""Local-first cascade: try the free local model, verify, and escalate through
Fireworks cheapest-first only when the verifier can't trust the local answer.
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


def answer_task(task: dict, validator: Optional[Validator] = None) -> Result:
    msgs = [{"role": "user", "content": task["prompt"]}]

    # Tier 0 — LOCAL (free).
    local = clients.local_chat(msgs, temperature=0.0)
    accept, _conf = verify(task, local.text, validator)
    if accept:
        return Result(task["id"], local.text, config.LOCAL_TIER.name, 0)

    # Escalate through Fireworks, cheapest first — stop at the first verified answer.
    best = Result(task["id"], local.text, config.LOCAL_TIER.name + "*", 0)
    for tier in config.FIREWORKS_LADDER:
        before = clients.METER.total
        fw = clients.fireworks_chat(tier.model, msgs, temperature=0.0)
        spent = clients.METER.total - before
        best = Result(task["id"], fw.text, tier.name + "*", spent)
        if validator is None or validator(task, fw.text):
            return Result(task["id"], fw.text, tier.name, spent)

    # Nothing verified — return the strongest attempt (we must still answer).
    return best
