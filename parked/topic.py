"""소재 산출 — 기억 기반 생성, 또는 오늘의 질문.

우선순위
1. 기억 기반 — 미소환 기억이 있으면 LLM 이 원문·시점을 살려 문장을 만든다
2. 오늘의 질문 — 사전 풀 30개에서 선택. **LLM 호출 없음**, 즉시 응답
"""

from __future__ import annotations

import json
import random
from datetime import datetime

from worker import DATA_DIR
from worker.llm import ask, load_prompt
from worker.models import Decision, Memory, Scope, Speaker, TopicLLMOutput
from worker.retrieve import REUSE_AFTER, now_kst

QUESTION_FILE = DATA_DIR / "daily_questions.json"


# --------------------------------------------------------------------------
# 1. 기억 기반
# --------------------------------------------------------------------------
def make_topic(
    memory: Memory,
    scope: Scope,
    target: Speaker | None,
    recent: str | None = None,
) -> Decision:
    payload = {
        "kind": memory.kind,
        "content": memory.content,
        "source_quote": memory.source_quote,
        "occurred_at": memory.occurred_at.isoformat() if memory.occurred_at else None,
        "recent": recent,
    }
    out = ask(
        TopicLLMOutput,
        load_prompt("topic"),
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    return Decision(
        scope=scope,
        target=target if scope == "individual" else None,
        content=out.content.strip(),
        reason=out.reason.strip() or None,
    )


# --------------------------------------------------------------------------
# 2. 오늘의 질문
# --------------------------------------------------------------------------
def _bucket(when: datetime) -> str:
    h = when.hour
    if 5 <= h < 11:
        return "morning"
    if 11 <= h < 17:
        return "afternoon"
    if 17 <= h < 23:
        return "evening"
    return "late_night"


def load_questions() -> list[dict]:
    return json.loads(QUESTION_FILE.read_text(encoding="utf-8"))


def _write_questions(questions: list[dict]) -> None:
    QUESTION_FILE.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _available(questions: list[dict], now: datetime) -> list[dict]:
    out = []
    for q in questions:
        used = q.get("used_at")
        if used and now - datetime.fromisoformat(used) < REUSE_AFTER:
            continue
        out.append(q)
    return out


def daily_question(
    scope: Scope,
    target: Speaker | None,
    now: datetime | None = None,
    persist: bool = True,
) -> Decision:
    now = now or now_kst()
    bucket = _bucket(now)
    questions = load_questions()

    pool = _available(questions, now)
    if not pool:  # 30개를 다 썼으면 중복 금지를 풀어준다
        pool = questions

    fit = [q for q in pool if bucket in q["time_tags"] or "any" in q["time_tags"]]
    pool = fit or pool

    # 답하기 부담 없는 질문 우선. 심야에는 무거운 질문을 아예 뺀다.
    light = [q for q in pool if not q.get("heavy")]
    if light:
        pool = light
    elif bucket == "late_night":
        pool = [q for q in questions if not q.get("heavy")] or pool

    chosen = random.choice(pool)

    if persist:
        for q in questions:
            if q["id"] == chosen["id"]:
                q["used_at"] = now.isoformat()
                break
        _write_questions(questions)

    # 기억이 없어 대체한 것이므로 근거 문구는 생략한다
    return Decision(
        scope=scope,
        target=target if scope == "individual" else None,
        content=chosen["text"],
        reason=None,
    )
