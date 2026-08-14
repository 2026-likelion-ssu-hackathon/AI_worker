"""Pydantic 스키마.

`Decision` / `Memory` / `GateResult` / `JudgeResult` 는 docs/worker-tasks.md 1단계 명세 그대로다.
`*LLMOutput` 은 LLM 에 넘기는 구조화 출력 전용 스키마다 — strict json_schema 모드가
nullable · date-time 포맷을 잘 다루지 못해서, LLM 에는 sentinel 문자열("none"/"unknown")로
받고 파이썬 쪽에서 명세 스키마로 변환한다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

MemoryKind = Literal["place", "activity", "promise", "wish", "interest"]
Scope = Literal["common", "individual"]
Speaker = Literal["A", "B"]


# --------------------------------------------------------------------------
# 입력
# --------------------------------------------------------------------------
class Message(BaseModel):
    sender: Speaker
    content: str
    ts: datetime


class Fixture(BaseModel):
    room_id: str
    messages: list[Message]
    # ⑤ 대화 중 정체 판정 기준 시각. 없으면 마지막 메시지 + 1분으로 본다(= 정체 아님).
    now: datetime | None = None
    # ⑤ 는 양쪽 다 접속 중일 때만 발동한다.
    online: list[Speaker] = Field(default_factory=lambda: ["A", "B"])


# --------------------------------------------------------------------------
# 출력
# --------------------------------------------------------------------------
class Decision(BaseModel):
    # 후보 기능 종류. 프론트는 이 값으로 렌더링 레이아웃을 고른다.
    #   topic — 상단 안내 / 중앙 content(소재) / 하단 reason(근거)
    #   tone  — 상단 안내 / reason(방향 문구) / content(대체 문장)
    # 기본값이 있어 기존 계약(대화 소재)은 그대로 동작한다.
    kind: Literal["topic", "tone"] = "topic"
    scope: Scope
    target: Speaker | None = None
    content: str
    reason: str | None = None


class Memory(BaseModel):
    id: str
    kind: MemoryKind
    content: str
    source_quote: str
    occurred_at: datetime | None = None
    used_at: datetime | None = None


class GateResult(BaseModel):
    triggered: bool
    trigger: str | None = None  # short_pingpong | no_question | one_sided | stall
    scope: Scope | None = None
    target: Speaker | None = None
    needs_llm: bool = False
    # 디버깅/시연용 근거. 판정에는 쓰이지 않는다.
    detail: str | None = None


class JudgeResult(BaseModel):
    should_intervene: bool
    trigger: Literal["routine_loop", "busy_excuse", "none"]
    scope: Scope | None = None
    target: Speaker | None = None
    memories: list[Memory] = Field(default_factory=list)


# --------------------------------------------------------------------------
# LLM 구조화 출력 전용
# --------------------------------------------------------------------------
class ExtractedMemory(BaseModel):
    """judge 호출에서 함께 뽑아내는 기억. id/시각은 파이썬 쪽에서 채운다."""

    kind: MemoryKind
    content: str
    source_quote: str


class JudgeLLMOutput(BaseModel):
    should_intervene: bool
    trigger: Literal["routine_loop", "busy_excuse", "none"]
    scope: Literal["common", "individual", "unknown"]
    target: Literal["A", "B", "none"]
    memories: list[ExtractedMemory]

    def to_result(self, memories: list[Memory]) -> JudgeResult:
        return JudgeResult(
            should_intervene=self.should_intervene,
            trigger=self.trigger,
            scope=None if self.scope == "unknown" else self.scope,
            target=None if self.target == "none" else self.target,
            memories=memories,
        )


class TopicLLMOutput(BaseModel):
    content: str
    reason: str


# --------------------------------------------------------------------------
# 후보 기능 2 — 갈등 중재 (말투 교정 제안)
# --------------------------------------------------------------------------
ToneFlagKind = Literal[
    "insult",          # 인신공격 · 욕설
    "generalization",  # 일반화 화법 ("넌 늘 그런 식이야")
    "sarcasm",         # 비꼼 · 반어 ("잘한다 ㅋㅋ")
    "abrupt_change",   # 평소 대비 급변 (마침표 종결, ㅋ 사라짐, 길이 급변)
    "repetition",      # 비슷한 말 반복 ("전화 받아 - 받아 - 받으라고")
    "address_change",  # 호칭 변화 (오빠 → 야)
]


class SpeakerProfile(BaseModel):
    """개인별 평소 말투 기준선.

    같은 문장이라도 사람마다 뜻이 다르다. 특정 단어를 절대 기준으로 잡지 않고
    **그 사람의 평소 대비 변화량**으로 판정하기 위한 값들이다.
    """

    speaker: Speaker
    avg_length: float = 0.0        # 평소 평균 메시지 길이
    period_rate: float = 0.0       # 마침표로 끝내는 비율
    laugh_per_msg: float = 0.0     # 메시지당 평균 ㅋ/ㅎ 개수
    emoji_rate: float = 0.0        # 이모지가 들어간 메시지 비율
    top_address: list[str] = Field(default_factory=list)  # 평소 호칭 상위 2개
    conflict_style: str | None = None  # 과거 갈등 때의 어휘 패턴 (자유 서술)


class ToneFlag(BaseModel):
    kind: ToneFlagKind
    detail: str


class ToneGateResult(BaseModel):
    triggered: bool
    speaker: Speaker | None = None
    message: str | None = None
    flags: list[ToneFlag] = Field(default_factory=list)


class ToneJudgeLLMOutput(BaseModel):
    """맥락 판정 — 진짜 갈등인지, 맥락상 장난인지."""

    should_suggest: bool
    is_playful: bool  # "와 미친 ㅋㅋ" 처럼 비속 표현이지만 공격 의도가 아닌 경우
    emotion: Literal["calm", "irritated", "angry", "hurt"]
    note: str  # 판정 근거 (내부용, 사용자에게 보여주지 않는다)


class ToneSuggestLLMOutput(BaseModel):
    direction: str    # 방향 문구 — 왜 다르게 읽힐 수 있는지 한 줄
    alternative: str  # 대체 문장 — 나 전달법으로 다시 쓴 문장 1개
