"""Container entrypoint for the AMD ACT II Track-1 judge.

Contract (from observed submissions — CONFIRM exact field names in the Participant Guide):
  INPUT : /input/tasks.json     — a JSON array (or {"tasks": [...]}) of task objects
  OUTPUT: /output/results.json  — a JSON array of {"id", "answer"}

Paths can be overridden by argv or the INPUT_PATH / OUTPUT_PATH env vars, so the
same entrypoint runs both locally and in the judge container.
"""
from __future__ import annotations

import json
import os
import sys

from . import clients
from .cascade import answer_task

DEFAULT_INPUT = os.getenv("INPUT_PATH", "/input/tasks.json")
DEFAULT_OUTPUT = os.getenv("OUTPUT_PATH", "/output/results.json")

# Field-name fallbacks — the real keys get pinned once we see the guide / a sample.
ID_KEYS = ("id", "task_id", "taskId", "uid")
PROMPT_KEYS = ("prompt", "question", "input", "text", "query")


def _load_tasks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    if isinstance(data, dict):  # unwrap {"tasks": [...]} / {"data": [...]} shapes
        data = data.get("tasks") or data.get("data") or data.get("items") or []
    return data


def _field(task: dict, keys: tuple[str, ...], default=None):
    for k in keys:
        if k in task:
            return task[k]
    return default


def main() -> None:
    in_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT

    tasks = _load_tasks(in_path)
    results = []
    tier_counts: dict[str, int] = {}
    for i, task in enumerate(tasks):
        tid = _field(task, ID_KEYS, default=i)
        prompt = _field(task, PROMPT_KEYS, default="")
        r = answer_task({"id": tid, "prompt": prompt})
        results.append({"id": tid, "answer": r.answer})
        tier_counts[r.tier] = tier_counts.get(r.tier, 0) + 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    n = len(tasks)
    local = tier_counts.get("local", 0)
    print(f"Wrote {n} answers to {out_path}. Tiers: {tier_counts}")
    print(clients.METER.report())
    print(f"Local (free) share: {local}/{n}")


if __name__ == "__main__":
    main()
