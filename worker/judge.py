"""LLM 판정 — 트리거 ④(일상 보고형 반복)와 바쁨 판별 + 기억 추출.

룰 게이트가 확정하지 못한 케이스(`needs_llm=True`)만 여기로 온다.
judge 는 **감지만** 한다. 사용자에게 보여줄 문구는 topic.py 에서 만든다.
"""

from __future__ import annotations

import hashlib

from worker.llm import ask, load_prompt
from worker.models import JudgeLLMOutput, JudgeResult, Memory, Message

MAX_MESSAGES = 60


def format_transcript(messages: list[Message]) -> str:
    """LLM 에 넘길 대화 로그. 날짜가 바뀌면 구분선을 넣어 반복 패턴을 보기 쉽게 한다."""
    lines: list[str] = []
    last_day = None
    for m in messages[-MAX_MESSAGES:]:
        day = m.ts.date().isoformat()
        if day != last_day:
            lines.append(f"--- {day} ---")
            last_day = day
        lines.append(f"[{m.ts:%H:%M}] {m.sender}: {m.content}")
    return "\n".join(lines)


def _memory_id(content: str, quote: str) -> str:
    digest = hashlib.sha1(f"{content}|{quote}".encode()).hexdigest()
    return f"m{digest[:8]}"


def judge(messages: list[Message]) -> JudgeResult:
    messages = sorted(messages, key=lambda m: m.ts)
    out = ask(JudgeLLMOutput, load_prompt("judge"), format_transcript(messages))

    # 원문에 실제로 존재하는 인용만 남긴다. 지어낸 기억을 저장소에 넣지 않기 위해서다.
    corpus = "\n".join(m.content for m in messages)
    memories: list[Memory] = []
    for ext in out.memories:
        if ext.source_quote and ext.source_quote not in corpus:
            continue
        occurred = next(
            (m.ts for m in messages if ext.source_quote and ext.source_quote in m.content),
            messages[-1].ts,
        )
        memories.append(
            Memory(
                id=_memory_id(ext.content, ext.source_quote),
                kind=ext.kind,
                content=ext.content,
                source_quote=ext.source_quote,
                occurred_at=occurred,
                used_at=None,
            )
        )

    return out.to_result(memories)
