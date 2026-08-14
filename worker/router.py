"""개입 방향 결정 — 후보 기능 중 하나를 골라 실행한다.

**위젯 슬롯은 1개다.** 여러 후보가 동시에 발동해도 하나만 내보낸다.

CLAUDE.md 는 "정교한 스코어링 로직을 구현하지 않는다"고 못박아 뒀다. 후보가 2개인 지금은
**우선순위 체인**으로 충분하다. `CANDIDATES` 순서대로 시도하고, 먼저 결과를 내는 후보가 이긴다.

    1. 갈등 중재 (tone)  — 갈등이 감지된 순간에 대화 소재를 던질 때가 아니다
    2. 대화 소재 (topic)

후보를 추가하려면 `Candidate` 를 구현해 `CANDIDATES` 에 끼우면 된다.
후보가 늘어 우선순위만으로 못 정하게 되면 그때 점수 기반으로 바꾼다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from worker.filter import apply_filter, is_clean
from worker.gate import check_gate
from worker.judge import judge
from worker.models import (
    Decision,
    GateResult,
    JudgeResult,
    Memory,
    Message,
    Speaker,
    ToneGateResult,
    ToneJudgeLLMOutput,
)
from worker.retrieve import mark_used, recent_context, retrieve, save_memories, search
from worker.tone import check_tone_gate, tone_judge, tone_suggest
from worker.topic import daily_question, make_topic
from worker.profile import resolve_profile


class Trace:
    """--verbose 용 중간 단계 기록. 판정에는 관여하지 않는다."""

    def __init__(self) -> None:
        self.candidate: str | None = None
        # 대화 소재
        self.gate: GateResult | None = None
        self.judged: JudgeResult | None = None
        self.hits: list[Memory] = []
        self.memory: Memory | None = None
        self.saved: list[Memory] = []
        self.source: str | None = None  # memory | daily_question
        # 갈등 중재
        self.tone_gate: ToneGateResult | None = None
        self.tone_judged: ToneJudgeLLMOutput | None = None


@dataclass
class Context:
    messages: list[Message]
    now: datetime
    online: list[Speaker] = field(default_factory=lambda: ["A", "B"])
    persist: bool = True
    trace: Trace = field(default_factory=Trace)


class Candidate(Protocol):
    name: str

    def build(self, ctx: Context) -> Decision | None:
        """실행할 게 없으면 None 을 돌려준다. 라우터가 다음 후보로 넘어간다."""
        ...


# --------------------------------------------------------------------------
# 후보 1 — 갈등 중재 (말투 교정 제안)
# --------------------------------------------------------------------------
class ToneCandidate:
    name = "tone"

    def build(self, ctx: Context) -> Decision | None:
        gate = check_tone_gate(ctx.messages)
        ctx.trace.tone_gate = gate
        if not gate.triggered or gate.speaker is None:
            return None

        profile = resolve_profile(gate.speaker, ctx.messages)
        judged = tone_judge(ctx.messages, gate, profile)
        ctx.trace.tone_judged = judged
        if not judged.should_suggest:
            return None

        def _make() -> Decision:
            out = tone_suggest(ctx.messages, gate, profile, judged)
            return Decision(
                kind="tone",
                scope="individual",   # 보낸 사람에게만
                target=gate.speaker,
                content=out.alternative.strip(),   # 대체 문장
                reason=out.direction.strip(),      # 방향 문구
            )

        # 금지어에 걸리면 1회 재생성. 또 걸리면 아예 내보내지 않는다.
        # 여기서 오늘의 질문으로 폴백하면 맥락이 완전히 어긋난다.
        decision = _make()
        if is_clean(decision):
            return decision
        retry = _make()
        return retry if is_clean(retry) else None


# --------------------------------------------------------------------------
# 후보 2 — 대화 소재 제시
# --------------------------------------------------------------------------
class TopicCandidate:
    name = "topic"

    def build(self, ctx: Context) -> Decision | None:
        gate = check_gate(ctx.messages, now=ctx.now, online=ctx.online)
        ctx.trace.gate = gate

        if gate.needs_llm:
            judged = judge(ctx.messages)
            ctx.trace.judged = judged
            ctx.trace.saved = save_memories(judged.memories, persist=ctx.persist)
            if not judged.should_intervene:
                return None
            scope = judged.scope or "common"
            target = judged.target
        elif gate.triggered:
            scope = gate.scope or "common"
            target = gate.target
        else:
            return None

        if scope != "individual":
            target = None

        recent = recent_context(ctx.messages)
        ctx.trace.hits = search(recent, k=3)
        memory = retrieve(recent, now=ctx.now)
        ctx.trace.memory = memory

        def _fallback() -> Decision:
            return daily_question(scope, target, now=ctx.now, persist=ctx.persist)

        if memory is None:
            ctx.trace.source = "daily_question"
            return apply_filter(_fallback())

        ctx.trace.source = "memory"
        decision = apply_filter(
            make_topic(memory, scope, target, recent=recent),
            regenerate=lambda: make_topic(memory, scope, target, recent=recent),
            fallback=_fallback,
        )
        # 폴백으로 넘어갔으면 기억을 소모하지 않는다
        if decision.reason is not None:
            mark_used(memory.id, now=ctx.now, persist=ctx.persist)
        else:
            ctx.trace.source = "daily_question"
        return decision


# 우선순위 순. 앞에 있는 후보가 결과를 내면 뒤는 실행되지 않는다.
CANDIDATES: list[Candidate] = [ToneCandidate(), TopicCandidate()]


def route(ctx: Context) -> Decision | None:
    for candidate in CANDIDATES:
        decision = candidate.build(ctx)
        if decision is not None:
            ctx.trace.candidate = candidate.name
            return decision
    return None
