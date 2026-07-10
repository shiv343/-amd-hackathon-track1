# Track 1 — Token-Efficient Routing Agent (AMD Developer Hackathon: ACT II)

A **local-first routing agent**. It answers each task with a *free* local model
(vLLM on the AMD GPU) and only escalates to a **Fireworks** model — cheapest
first — when a verifier can't trust the local answer.

Only tokens routed through Fireworks count toward the leaderboard, so:

> **answered locally = zero tokens = best possible rank.**

## How it works

```
task ─▶ local model (FREE) ─▶ verifier
              accept ─▶ submit answer            (0 Fireworks tokens)  ✅
              reject ─▶ Fireworks ladder, cheapest ─▶ priciest,
                        stop at the first answer that verifies
```

- `src/clients.py`  — OpenAI-compatible clients for vLLM + Fireworks; token meter
- `src/verifier.py` — the "trust the free local answer?" decision (validator + self-consistency)
- `src/cascade.py`  — the local-first escalation
- `src/agent.py`    — container entrypoint (JSONL tasks in → JSONL answers out)
- `eval/run_eval.py`— local accuracy **and** Fireworks-token scorecard

## Run locally

```bash
# 1. serve the free local model (on the AMD pod)
vllm serve $LOCAL_MODEL --port 8000

# 2. configure
cp .env.example .env      # fill FIREWORKS_API_KEY

# 3. run the agent
python -m src.agent tasks.jsonl outputs/answers.jsonl

# 4. eval (accuracy + token cost, before submitting)
python eval/run_eval.py eval/sample_tasks.jsonl
```

## Container

```bash
docker build -t track1-agent .
docker run --rm track1-agent
```

## Status / TODO — confirm against the Participant Guide

- [ ] **Does the scoring container get a GPU?** (VRAM → pick the largest local model that fits)
- [ ] Exact task set (domains, format, count)
- [ ] Accuracy metric + threshold (exact match / F1 / judge?)
- [ ] Token formula (prompt+completion vs. completion only; price-weighted?)
- [ ] Agent I/O contract (how the harness invokes the container)
- [ ] Fireworks model IDs + live prices for the escalation ladder
