"""Container entrypoint.

I/O CONTRACT — CONFIRM against the Participant Guide. Assumed for now:
  input : JSONL of tasks   {"id": ..., "prompt": ...}   (path in argv[1] or $TASKS_PATH)
  output: JSONL of answers {"id": ..., "answer": ...}    (path in argv[2] or $OUTPUT_PATH)

Only these read/write adapters change to match the real harness — the
cascade/verifier logic stays identical.
"""
from __future__ import annotations

import json
import os
import sys

from . import clients
from .cascade import answer_task


def load_tasks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    tasks_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("TASKS_PATH", "eval/sample_tasks.jsonl")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.getenv("OUTPUT_PATH", "outputs/answers.jsonl")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    tasks = load_tasks(tasks_path)
    tier_counts: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as out:
        for t in tasks:
            r = answer_task(t)
            tier_counts[r.tier] = tier_counts.get(r.tier, 0) + 1
            out.write(json.dumps({"id": r.id, "answer": r.answer}) + "\n")

    n = len(tasks)
    local = tier_counts.get("local", 0)
    print(f"Answered {n} tasks. Tier usage: {tier_counts}")
    print(clients.METER.report())
    print(f"Local (free) share: {local}/{n} = {local / max(n, 1):.0%}")


if __name__ == "__main__":
    main()
