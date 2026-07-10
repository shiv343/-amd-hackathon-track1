"""Local eval harness — accuracy per category + Fireworks token cost.

Each JSONL item is {"id", "prompt", ...} plus a scoring spec:
  - metric "em"       (default): normalized exact match vs item["answer"]
  - metric "keywords": all of item["keywords"] appear in the answer (approx. summarization)
  - metric "exec"     : run the model's code, then item["test"] asserts (code generation)
An optional "category" field groups the per-category report.
Backward compatible with simple {"answer": ...} files.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

sys.path.insert(0, ".")

from src import clients                 # noqa: E402
from src.cascade import answer_task     # noqa: E402


def _norm(s: object) -> str:
    return " ".join(str(s).strip().lower().split())


def _extract_code(text: str) -> str:
    """Pull code out of ``` fences if the model wrapped it."""
    if "```" in text:
        block = text.split("```")[1]
        if "\n" in block:
            first, rest = block.split("\n", 1)
            if first.strip().lower() in ("python", "py", ""):
                return rest
        return block
    return text


def _score_em(item: dict, pred: str) -> float:
    return 1.0 if _norm(pred) == _norm(item.get("answer", "")) else 0.0


def _score_keywords(item: dict, pred: str) -> float:
    p = pred.lower()
    kws = [k.lower() for k in item.get("keywords", [])]
    return 1.0 if kws and all(k in p for k in kws) else 0.0


def _score_exec(item: dict, pred: str) -> float:
    ns: dict = {}
    try:
        exec(_extract_code(pred), ns)   # define the model's function
        exec(item["test"], ns)          # run the asserts against it
        return 1.0
    except Exception:
        return 0.0


METRICS = {"em": _score_em, "keywords": _score_keywords, "exec": _score_exec}


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "eval/eval_8cat.jsonl"
    with open(path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    total = 0.0
    cat_tot: dict = defaultdict(float)
    cat_n: dict = defaultdict(int)
    tier_counts: dict = {}

    for it in items:
        r = answer_task(it)
        metric = METRICS.get(it.get("metric", "em"), _score_em)
        s = metric(it, r.answer)
        total += s
        cat = it.get("category", "all")
        cat_tot[cat] += s
        cat_n[cat] += 1
        tier_counts[r.tier] = tier_counts.get(r.tier, 0) + 1

    n = len(items)
    print("=" * 56)
    print(f"Overall accuracy : {total / max(n, 1):.3f}  ({int(round(total))}/{n})")
    print("-" * 56)
    for cat in sorted(cat_n):
        print(f"  {cat:16s}: {cat_tot[cat] / cat_n[cat]:.2f}  ({int(round(cat_tot[cat]))}/{cat_n[cat]})")
    print("-" * 56)
    print(f"Tier usage       : {tier_counts}")
    print(clients.METER.report())
    local = tier_counts.get("local", 0)
    print(f"Answered locally : {local}/{n} = {local / max(n, 1):.0%}")
    print("=" * 56)


if __name__ == "__main__":
    main()
