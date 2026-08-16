"""LLM 접근 레이어.

모든 LLM 호출은 여기를 거친다. `import openai` 직접 호출은 금지 — 트레이싱 일관성 때문이다.
후보 3종과 기억 추출이 모델 설정·토큰 계량을 공유하도록 한 곳에 모아둔다.
지금 이 모듈을 거치는 호출은 7종이다 (`worker/prompts/*.md` 와 1:1).
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


class Usage:
    """이 프로세스에서 쓴 토큰. **유료는 OpenAI 뿐이라 여기만 센다.**

    카카오·유튜브는 무료 쿼터라 돈이 아니라 횟수 문제다.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.input = 0
        self.output = 0

    def add(self, meta: dict | None) -> None:
        self.calls += 1
        if not meta:
            return
        self.input += meta.get("input_tokens", 0)
        self.output += meta.get("output_tokens", 0)

    def reset(self) -> None:
        self.calls = self.input = self.output = 0

    def __str__(self) -> str:
        return f"LLM {self.calls}회 · 입력 {self.input:,} 토큰 · 출력 {self.output:,} 토큰"


USAGE = Usage()


def ask(schema: type[T], system: str, user: str) -> T:
    """구조화 출력 단발 호출.

    툴 루프가 없는 단발 분류/생성이라 create_agent 를 쓰지 않는다.
    reasoning 계열 모델은 temperature 를 거부하므로 한 번 재시도한다.

    `include_raw=True` 로 받는 이유는 **토큰 사용량을 세기 위해서다.** 파싱된 결과만
    받으면 usage_metadata 가 딸려오지 않아 비용을 알 수 없다.
    """
    messages = [("system", system), ("human", user)]
    for with_temperature in (True, False):
        model = _model(with_temperature).with_structured_output(
            schema, method="json_schema", strict=True, include_raw=True
        )
        try:
            result = model.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            if with_temperature and "temperature" in str(exc).lower():
                continue
            raise

        USAGE.add(getattr(result.get("raw"), "usage_metadata", None))
        if result.get("parsing_error") is not None:
            raise RuntimeError(f"구조화 출력 파싱 실패: {result['parsing_error']}")
        return result["parsed"]
    raise RuntimeError("unreachable")
