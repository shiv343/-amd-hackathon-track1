"""Container entrypoint for the AMD ACT II Track-1 judge.

Judge mounts tasks at /input/tasks.json, reads answers from /output/results.json.
The exact results schema isn't published, so the OUTPUT SHAPE is switchable via
RESULTS_FORMAT (list | wrapped | map) and each item echoes the original task
(preserving the exact id field) plus the answer under several key aliases.
Never crash — a crash = RUNTIME_ERROR = a wasted scoring cycle.
"""
from __future__ import annotations

import json
import os
import sys

from . import clients
from .cascade import answer_task

INPUT_CANDIDATES = [
    os.getenv("INPUT_PATH"),
    "/input/tasks.json", "/input/tasks.jsonl",
    "/data/tasks.json", "/data/input.json", "tasks.json",
]
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/output/results.json")
RESULTS_FORMAT = os.getenv("RESULTS_FORMAT", "list")   # list | wrapped | map

ID_KEYS = ("id", "task_id", "taskId", "uid")
PROMPT_KEYS = ("prompt", "question", "input", "text", "query", "instruction")
ANSWER_KEYS = ("answer", "response", "output", "result", "prediction")


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
    if isinstance(data, dict):
        data = data.get("tasks") or data.get("data") or data.get("items") or []
    return data


def _field(task, keys, default=None):
    if isinstance(task, dict):
        for k in keys:
            if k in task:
                return task[k]
    return default


def _write(payload, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception as e:
        print(f"WARN: could not write {path}: {e}", file=sys.stderr)


def main() -> None:
    argv_in = sys.argv[1] if len(sys.argv) > 1 else None
    out_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PATH

    in_path = _find_input(argv_in)
    if not in_path:
        print(f"WARN: no input found; looked at {INPUT_CANDIDATES}", file=sys.stderr)
        _write([] if RESULTS_FORMAT != "map" else {}, out_path)
        return

    try:
        tasks = _load(in_path)
    except Exception as e:
        print(f"WARN: could not parse {in_path}: {e}", file=sys.stderr)
        _write([] if RESULTS_FORMAT != "map" else {}, out_path)
        return

    items, id_to_ans, tiers = [], {}, {}
    for i, task in enumerate(tasks):
        raw = task if isinstance(task, dict) else {"id": i, "prompt": str(task)}
        tid = _field(raw, ID_KEYS, default=i)
        prompt = _field(raw, PROMPT_KEYS, default="")
        try:
            r = answer_task({"id": tid, "prompt": prompt})
            ans, tier = r.answer, r.tier
        except Exception as e:
            ans, tier = "", "error"
            print(f"WARN: task {tid} failed: {e}", file=sys.stderr)

        item = dict(raw)                    # echo the input task -> preserves exact id field
        for k in ANSWER_KEYS:
            item[k] = ans
        items.append(item)
        id_to_ans[str(tid)] = ans
        tiers[tier] = tiers.get(tier, 0) + 1

    if RESULTS_FORMAT == "map":
        payload = id_to_ans
    elif RESULTS_FORMAT == "wrapped":
        payload = {"results": items}
    else:
        payload = items

    _write(payload, out_path)
    print(f"Wrote {len(items)} answers ({RESULTS_FORMAT}) to {out_path}. Tiers: {tiers}", file=sys.stderr)
    print(clients.METER.report(), file=sys.stderr)


if __name__ == "__main__":
    main()
