"""Pydantic 스키마 — **백엔드 연동 공통 규격 v1 기준**.

규격서(`docs/contract-v1.md`)의 요청·응답 구조를 그대로 모델로 옮겼다.
`model_dump(by_alias=True, exclude_none=True)` 하면 규격서 JSON 이 그대로 나온다.

- 필드명은 파이썬 쪽에서 snake_case, 직렬화할 때 camelCase (규격서 14장)
- 선택 필드에 값이 없으면 필드 자체를 생략 (`exclude_none=True`)
- 빈 목록은 `null` 이 아니라 `[]`

**화자 표기**: 규격서는 `USER_A`/`USER_B`, 내부 로직은 `A`/`B` 를 쓴다.
경계에서만 변환한다 (`to_speaker` / `to_key`). 기억 시드·말투 기준선 시드가
전부 `A`/`B` 로 쌓여 있어서 내부 표기를 바꾸면 시드를 전부 다시 써야 한다.

`*LLMOutput` 은 LLM 구조화 출력 전용 스키마다. strict json_schema 모드가 nullable ·
date-time 포맷을 잘 다루지 못해서, LLM 에는 sentinel 문자열("none"/"unknown")로 받고
파이썬 쪽에서 규격 스키마로 변환한다. 이쪽은 camelCase 로 바꾸지 않는다 — 규격이 아니라
프롬프트와 짝이 되는 내부 구조라서다.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

KST = timezone(timedelta(hours=9))

Speaker = Literal["A", "B"]
ParticipantKey = Literal["USER_A", "USER_B"]

ResultType = Literal[
    "TONE_CORRECTION",
    "DATE_RECOMMENDATION",
    "YOUTUBE_RECOMMENDATION",
]
VisibilityType = Literal["INDIVIDUAL", "COUPLE"]
ContentType = Literal["TEXT", "LINK", "MIXED"]
Status = Literal["COMPLETED", "SKIPPED", "FAILED"]

ErrorCode = Literal[
    "INVALID_REQUEST",
    "INVALID_PARTICIPANT",
    "MODEL_ERROR",
    "ANALYSIS_TIMEOUT",
    "YOUTUBE_SEARCH_FAILED",
    "EXTERNAL_API_ERROR",
    "INTERNAL_ERROR",
]

# 기억 종류. `schedule` 은 데이트 코스 추천 명세의 "일정"(이번주 토요일, 내일)이다.
# 기존 `promise`(지키지 못한 약속)와는 다르다 — promise 는 시점이 없고 schedule 은 시점이 전부다.
MemoryKind = Literal["place", "activity", "promise", "wish", "interest", "schedule"]


def to_speaker(key: str) -> Speaker:
    """`USER_A` → `A`. 이미 `A` 면 그대로."""
    return key.removeprefix("USER_")  # type: ignore[return-value]


def to_key(speaker: Speaker) -> ParticipantKey:
    """`A` → `USER_A`."""
    return f"USER_{speaker}"  # type: ignore[return-value]


def as_kst(value: datetime) -> datetime:
    """타임존 없는 시각은 KST 로 본다.

    규격서 요청 예시의 `sentAt` 은 오프셋이 없고(`2026-08-15T20:30:00`),
    기억 시드는 `+09:00` 이 붙어 있다. 섞이면 비교에서 TypeError 가 난다.
    """
    return value.replace(tzinfo=KST) if value.tzinfo is None else value


class Camel(BaseModel):
    """규격서 JSON 과 짝이 되는 모델의 공통 설정."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# --------------------------------------------------------------------------
# 요청 (규격서 5장)
# --------------------------------------------------------------------------
class Participant(Camel):
    participant_key: ParticipantKey


class Message(Camel):
    message_id: int
    sender: Speaker
    content: str
    sent_at: datetime

    @field_validator("sender", mode="before")
    @classmethod
    def _strip_prefix(cls, v: object) -> object:
        return to_speaker(v) if isinstance(v, str) else v

    @field_validator("sent_at")
    @classmethod
    def _kst(cls, v: datetime) -> datetime:
        return as_kst(v)


class AnalysisRequest(Camel):
    analysis_request_id: str
    chat_room_id: int
    participants: list[Participant]
    messages: list[Message]

    # ⚠️ 규격서에 없는 필드다. 워커가 "지금 몇 시인지"를 알아야 데이트 코스의 시간대
    # 판단(심야/주말)이 가능하고, 픽스처로 결과를 재현하려면 고정 시각이 필요하다.
    # 서버 담당자에게 추가를 요청한 상태이며, 없으면 마지막 메시지 시각으로 대체한다.
    requested_at: datetime | None = None

    @field_validator("requested_at")
    @classmethod
    def _kst(cls, v: datetime | None) -> datetime | None:
        return as_kst(v) if v is not None else None

    def now(self) -> datetime:
        if self.requested_at is not None:
            return self.requested_at
        return max(m.sent_at for m in self.messages) if self.messages else datetime.now(KST)


# --------------------------------------------------------------------------
# 기능별 resultData (규격서 8~10장)
# --------------------------------------------------------------------------
class ToneResultData(Camel):
    """말투 교정 (규격서 8장)."""

    situation_diagnosis: str   # 현재 표현에 대한 짧은 상황 진단
    guide_message: str         # 고정 안내 문구
    alternative_sentence: str  # 나 전달법 기반 대체 문장
    correction_reason: str     # 기존 표현이 다르게 읽힐 수 있는 이유


class Place(Camel):
    """데이트 코스의 장소 하나. `order` 는 코스 장소에만 붙는다 (mainPlace 는 생략)."""

    order: int | None = None
    name: str
    category: str
    summary: str
    external_url: str


class DateCourseResultData(Camel):
    """데이트 코스 추천 (규격서 9장)."""

    guide_message: str
    course_name: str
    course_summary: str
    main_place: Place
    course_places: list[Place]
    recommendation_reason: str


class YoutubeResultData(Camel):
    """유튜브 영상 추천 (규격서 10장)."""

    guide_message: str
    video_id: str
    title: str
    video_url: str
    thumbnail_url: str
    channel_name: str
    recommendation_reason: str
    # 선택 필드. 값이 없으면 직렬화에서 통째로 빠진다.
    video_summary: str | None = None


ResultData = ToneResultData | DateCourseResultData | YoutubeResultData


# --------------------------------------------------------------------------
# 응답 (규격서 6·7·11·12장)
# --------------------------------------------------------------------------
class AiResult(Camel):
    result_type: ResultType
    visibility_type: VisibilityType
    # INDIVIDUAL 이면 필수, COUPLE 이면 생략 (규격서 14장)
    target_participant: ParticipantKey | None = None
    content_type: ContentType
    trigger_message_ids: list[int] = Field(default_factory=list)
    result_data: ResultData


class EmotionAnalysis(Camel):
    """감정 분석 (규격서 11장).

    ⏸️ 별도 기능으로 명세가 아직 나오지 않았다. 지금은 항상 빈 배열을 반환한다.
    스키마만 규격서대로 잡아두고 채우지 않는다.
    """

    subject_participant: ParticipantKey
    viewer_participant: ParticipantKey
    emotion_type: str
    intensity_value: float
    should_show: bool
    trigger_message_ids: list[int] = Field(default_factory=list)
    expires_at: datetime | None = None


class AnalysisResponse(Camel):
    analysis_request_id: str
    status: Status
    results: list[AiResult] = Field(default_factory=list)
    emotion_analyses: list[EmotionAnalysis] = Field(default_factory=list)
    error_code: ErrorCode | None = None
    error_message: str | None = None

    def to_json_dict(self) -> dict:
        """규격서 그대로의 JSON 딕셔너리."""
        return self.model_dump(by_alias=True, exclude_none=True, mode="json")


# --------------------------------------------------------------------------
# 기억 저장소
# --------------------------------------------------------------------------
class Memory(BaseModel):
    id: str
    kind: MemoryKind
    content: str
    source_quote: str
    occurred_at: datetime | None = None
    used_at: datetime | None = None


# --------------------------------------------------------------------------
# 후보 기능 1 — 갈등 중재 (말투 교정 제안)
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
    message_id: int | None = None
    flags: list[ToneFlag] = Field(default_factory=list)


class ToneJudgeLLMOutput(BaseModel):
    """맥락 판정 — 진짜 갈등인지, 맥락상 장난인지."""

    should_suggest: bool
    is_playful: bool  # "와 미친 ㅋㅋ" 처럼 비속 표현이지만 공격 의도가 아닌 경우
    emotion: Literal["calm", "irritated", "angry", "hurt"]
    note: str  # 판정 근거 (내부용, 사용자에게 보여주지 않는다)


class ToneSuggestLLMOutput(BaseModel):
    situation_diagnosis: str   # 상황 진단 — 감정을 알아봄
    alternative_sentence: str  # 대체 문장 — 나 전달법, 1~2문장
    correction_reason: str     # 왜 다르게 읽힐 수 있는지


# --------------------------------------------------------------------------
# 후보 기능 2 — 데이트 코스 추천
# --------------------------------------------------------------------------
DateIntentKind = Literal[
    "plan_question",   # 명시적 계획 질문 ("어디 갈까", "뭐 먹지")
    "schedule_fixed",  # 일정 확정 발화 ("토요일에 보자")
    "category",        # 장소·음식 카테고리 언급 ("배고파", "카페", "~ 가보고 싶다")
    "recall",          # 이전 대화 기억의 재언급 ("저번에 그 카페")
]


class DateGateResult(BaseModel):
    triggered: bool
    kinds: list[DateIntentKind] = Field(default_factory=list)
    detail: str | None = None
    message_ids: list[int] = Field(default_factory=list)


class DatePlanLLMOutput(BaseModel):
    """대화 + 기억을 읽고 **무엇을 찾을지** 정한다. 장소를 지어내지 않는다.

    실제 장소는 카카오 로컬 API 로만 가져온다. LLM 이 상호명을 만들면 존재하지 않는
    가게가 나오고 externalUrl 이 죽은 링크가 된다.
    """

    should_recommend: bool
    scope: Literal["common", "individual"]
    target: Literal["A", "B", "none"]
    region: str          # 검색 지역 ("성수동", "연남동"). 모르면 "none"
    queries: list[str]   # 카카오 로컬 검색어. 코스 순서대로 2~4개
    course_name: str
    course_summary: str
    reason_seed: str     # 어떤 발화·시점을 근거로 삼았는지 (원문 인용 포함)


class DateReasonLLMOutput(BaseModel):
    """확정된 장소 목록을 보고 추천 이유 문구를 쓴다."""

    course_name: str
    course_summary: str
    recommendation_reason: str
    place_summaries: list[str]  # coursePlaces 순서대로. 장소 한 줄 설명


# --------------------------------------------------------------------------
# 후보 기능 3 — 유튜브 영상 추천
# --------------------------------------------------------------------------
ConcernType = Literal[
    "contact",       # 연락
    "reconcile",     # 다툼 후 화해
    "hurt",          # 서운함 표현
    "boredom",       # 권태
    "trust",         # 신뢰
    "understanding", # 이해 부족
    "apology",       # 사과
    "none",
]


class ConcernLLMOutput(BaseModel):
    """관계 고민 신호 분류 + 검색어 생성.

    룰로 잡을 수 없다. 명세도 "LLM 을 통한 스트림 내 대화 맥락 이해가 제일 중요"라고
    못박아 뒀다. 그래서 이 기능은 게이트가 LLM 이다.
    """

    should_recommend: bool
    concern: ConcernType
    scope: Literal["common", "individual"]
    target: Literal["A", "B", "none"]
    stage: str          # 관계 단계 추정 ("초반", "1년 이상", "unknown")
    queries: list[str]  # 한국어 유튜브 검색어 1~3개
    note: str           # 판정 근거 (내부용)


class VideoPickLLMOutput(BaseModel):
    """후보 영상 + 베스트 댓글을 읽고 1개를 고른다.

    제목·썸네일만으로 고르지 않는다. 댓글이 "영상이 실제로 무엇을 말하는지"를 알려준다.
    적합한 후보가 없으면 `picked_index = -1` 로 침묵한다.
    """

    picked_index: int  # 후보 목록에서 고른 인덱스. 없으면 -1
    recommendation_reason: str


# --------------------------------------------------------------------------
# 기억 추출 (모든 후보가 공유)
# --------------------------------------------------------------------------
class ExtractedMemory(BaseModel):
    """대화에서 뽑아낸 기억. id/시각은 파이썬 쪽에서 채운다."""

    kind: MemoryKind
    content: str
    source_quote: str


class ExtractLLMOutput(BaseModel):
    memories: list[ExtractedMemory]
