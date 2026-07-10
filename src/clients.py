"""OpenAI-compatible clients + token metering.

Both vLLM (local) and Fireworks speak the OpenAI API, so one thin wrapper serves
both. The only difference that matters for scoring: Fireworks tokens are metered;
local tokens are free.
"""
from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from . import config


@dataclass
class Completion:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    counted: bool  # did this cost Fireworks tokens?

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class TokenMeter:
    """Accumulates Fireworks token usage — this IS our leaderboard cost."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, c: Completion) -> None:
        if c.counted:
            self.prompt_tokens += c.prompt_tokens
            self.completion_tokens += c.completion_tokens
            self.calls += 1

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def report(self) -> str:
        return (f"Fireworks tokens — prompt={self.prompt_tokens} "
                f"completion={self.completion_tokens} total={self.total} "
                f"({self.calls} calls)")


METER = TokenMeter()

_local = OpenAI(base_url=config.LOCAL_BASE_URL, api_key="EMPTY")
_fireworks = OpenAI(base_url=config.FIREWORKS_BASE_URL, api_key=config.FIREWORKS_API_KEY or "EMPTY")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _chat(client: OpenAI, model: str, messages, counted: bool, **kw) -> Completion:
    resp = client.chat.completions.create(model=model, messages=messages, **kw)
    u = resp.usage
    c = Completion(
        text=resp.choices[0].message.content or "",
        model=model,
        prompt_tokens=getattr(u, "prompt_tokens", 0) if u else 0,
        completion_tokens=getattr(u, "completion_tokens", 0) if u else 0,
        counted=counted,
    )
    METER.add(c)
    return c


def local_chat(messages, **kw) -> Completion:
    """Free tier — vLLM on the AMD GPU. Never counts toward score."""
    return _chat(_local, config.LOCAL_MODEL, messages, counted=False, **kw)


def fireworks_chat(model: str, messages, **kw) -> Completion:
    """Counted tier — every token here lowers our rank. Use sparingly."""
    return _chat(_fireworks, model, messages, counted=True, **kw)
