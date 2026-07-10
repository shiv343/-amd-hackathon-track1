"""Local eval harness — measure accuracy AND Fireworks token cost before submitting.

Tasks file: JSONL of {"id", "prompt", "answer"} where "answer" is the gold label.
Reports accuracy, total Fireworks tokens (= your leaderboard cost), and the
local-vs-Fireworks split. Swap `score_one` for the track's real metric.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

from src import clients                 # noqa: E402
from src.cascade import answer_task     # noqa: E402


def _norm(s: object) -> str:
    return " ".join(str(s).strip().lower().split())


def score_one(pred: str, gold: str) -> float:
    # CONFIRM metric from guide (exact match / F1 / judge). Default: normalized EM.
    return 1.0 if _norm(pred) == _norm(gold) else 0.0


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "eval/sample_tasks.jsonl"
    with open(path, encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    total = 0.0
    tier_counts: dict[str, int] = {}
    for t in tasks:
        r = answer_task(t)
        total += score_one(r.answer, t.get("answer", ""))
        tier_counts[r.tier] = tier_counts.get(r.tier, 0) + 1

    n = len(tasks)
    local = tier_counts.get("local", 0)
    print("=" * 52)
    print(f"Tasks            : {n}")
    print(f"Accuracy         : {total / max(n, 1):.3f}")
    print(f"Tier usage       : {tier_counts}")
    print(clients.METER.report())
    print(f"Answered locally : {local}/{n} = {local / max(n, 1):.0%}  (these cost 0 tokens)")
    print("=" * 52)


if __name__ == "__main__":
    main()
