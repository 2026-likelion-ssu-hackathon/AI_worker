"""LLM 접근 레이어.

모든 LLM 호출은 여기를 거친다. `import openai` 직접 호출은 금지 — 트레이싱 일관성 때문이다.
judge 와 topic 이 같은 모델 설정을 공유하도록 한 곳에 모아둔다.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TypeVar

from langchain.chat_models import init_chat_model
from pydantic import BaseModel

from worker import PROMPT_DIR

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "openai:gpt-5"


@lru_cache(maxsize=4)
def _model(with_temperature: bool):
    name = os.getenv("KAKAPO_MODEL", DEFAULT_MODEL)
    raw = os.getenv("KAKAPO_TEMPERATURE", "0.3").strip()
    if with_temperature and raw:
        return init_chat_model(name, temperature=float(raw))
    return init_chat_model(name)


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    """`worker/prompts/{name}.md` 를 읽는다."""
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


def ask(schema: type[T], system: str, user: str) -> T:
    """구조화 출력 단발 호출.

    툴 루프가 없는 단발 분류/생성이라 create_agent 를 쓰지 않는다.
    reasoning 계열 모델은 temperature 를 거부하므로 한 번 재시도한다.
    """
    messages = [("system", system), ("human", user)]
    for with_temperature in (True, False):
        model = _model(with_temperature).with_structured_output(
            schema, method="json_schema", strict=True
        )
        try:
            return model.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            if with_temperature and "temperature" in str(exc).lower():
                continue
            raise
    raise RuntimeError("unreachable")
