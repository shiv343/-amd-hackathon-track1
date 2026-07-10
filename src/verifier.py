"""Decide whether a FREE local answer can be trusted, or we must pay Fireworks.

The verifier is the whole ballgame: a well-calibrated "trust local" decision
maximizes the fraction answered at zero token cost while holding accuracy.

Signals (all run on the free local model, so they cost nothing):
  1. deterministic validator — task-specific, checkable correctness / format
  2. self-consistency        — sample k times; agreement = confidence

Extend `validator` per task type once the task set is known (CONFIRM guide):
e.g. code runs, math result checks, JSON parses, answer in the allowed set.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

from . import clients, config

Validator = Callable[[dict, str], bool]  # (task, answer) -> hard pass/fail


def _normalize(s: str) -> str:
    return " ".join(s.strip().lower().split())


def self_consistency(task: dict, k: Optional[int] = None) -> tuple[str, float]:
    """Sample the LOCAL model k times (free); return (majority answer, agreement)."""
    k = k or config.SELF_CONSISTENCY_K
    msgs = [{"role": "user", "content": task["prompt"]}]
    answers = [_normalize(clients.local_chat(msgs, temperature=0.7).text) for _ in range(k)]
    _, count = Counter(answers).most_common(1)[0]
    return answers[0], count / k


def verify(task: dict, answer: str, validator: Optional[Validator] = None) -> tuple[bool, float]:
    """Return (accept_local, confidence in [0, 1])."""
    # 1) a hard validator, if present, can reject outright.
    if validator is not None and not validator(task, answer):
        return False, 0.0
    # 2) otherwise, trust local when its own samples agree strongly.
    _, agreement = self_consistency(task)
    return agreement >= config.CONF_THRESHOLD, agreement
