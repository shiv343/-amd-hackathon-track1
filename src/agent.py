"""Container entrypoint for the AMD ACT II Track-1 judge.

Contract (from observed submissions): the judge mounts tasks at /input/tasks.json
and reads answers from /output/results.json. Exact field names still vary, so we
read defensively and NEVER crash — a crash triggers RUNTIME_ERROR (a wasted,
hours-long scoring cycle). Better to emit whatever we can.
"""
from __future__ import annotations

import json
import os
import sys

from . import clients
from .cascade import answer_task

INPUT_CANDIDATES = [
    os.getenv("INPUT_PATH"),
    "/input/tasks.json",
    "/input/tasks.jsonl",
    "/data/tasks.json",
    "/data/input.json",
    "tasks.json",
]
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/output/results.json")

ID_KEYS = ("id", "task_id", "taskId", "uid")
PROMPT_KEYS = ("prompt", "question", "input", "text", "query", "instruction")


def _find_input(argv_path):
    for p in [argv_path] + INPUT_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return None


def _load(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    if isinstance(data, dict):  # unwrap {"tasks": [...]} / {"data": [...]}
        data = data.get("tasks") or data.get("data") or data.get("items") or []
    return data


def _field(task, keys, default=None):
    if isinstance(task, dict):
        for k in keys:
            if k in task:
                return task[k]
    return default


def _write(results, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
    except Exception as e:
        print(f"WARN: could not write {path}: {e}", file=sys.stderr)


def main() -> None:
    argv_in = sys.argv[1] if len(sys.argv) > 1 else None
    out_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PATH

    in_path = _find_input(argv_in)
    if not in_path:
        print(f"WARN: no input found; looked at {INPUT_CANDIDATES}", file=sys.stderr)
        _write([], out_path)
        return

    try:
        tasks = _load(in_path)
    except Exception as e:
        print(f"WARN: could not parse {in_path}: {e}", file=sys.stderr)
        _write([], out_path)
        return

    results, tier_counts = [], {}
    for i, task in enumerate(tasks):
        tid = _field(task, ID_KEYS, default=i)
        prompt = _field(task, PROMPT_KEYS, default="")
        try:
            r = answer_task({"id": tid, "prompt": prompt})
            answer, tier = r.answer, r.tier
        except Exception as e:  # one bad task must not sink the whole run
            answer, tier = "", "error"
            print(f"WARN: task {tid} failed: {e}", file=sys.stderr)
        results.append({"id": tid, "answer": answer})
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    _write(results, out_path)
    print(f"Wrote {len(results)} answers to {out_path}. Tiers: {tier_counts}", file=sys.stderr)
    print(clients.METER.report(), file=sys.stderr)


if __name__ == "__main__":
    main()
