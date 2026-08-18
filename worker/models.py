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

# 실 상태 표현의 상태 라벨 (`docs/state-display-v4.md` 6장, 디자인 확정 5종).
#
# **규격서 11장 `emotionType` 에 이 값을 넣는다.** 필드 이름은 감정인데 값은 상태 라벨이다 —
# 디자인 확정안의 축이 이쪽이고 실 모양·글로우와 1:1 이라 프론트가 그대로 쓴다.
# 감정 이름 쪽("Calm / Neutral")은 슬래시가 들어가 enum 이 되지 않는다.
#
# 추이(쌓임·풀어짐)가 **라벨 안에 이미 들어 있다.** 별도의 trend 축을 만들지 않는다.
StateLabel = Literal[
    "STABLE",       # 평온·기본 — 아래로 축 늘어진 곡선 / 무채색
    "RESOLVED",     # 애정·설렘, 감정 풀어짐 — 하트 매듭 / 핑크·마젠타
    "ACCUMULATED",  # 서운함·오해, 감정 쌓임 — 엉킨 매듭 / 파랑
    "ENGAGED",      # 들뜸·활기, 대화 활발 — 크고 규칙적인 파동 / 초록
    "ESCALATED",    # 분노·격앙, 감정 격해짐 — 날카로운 스파이크 / 빨강
]

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


class RecentResult(Camel):
    """최근에 이미 내보낸 결과 (규격서 협의분, 2026-08-17 합의).

    규격서 10장의 "동일 영상을 최근에 추천한 경우 다른 후보 탐색"을 지킬 유일한 수단이다.
    워커는 동기 REST 요청마다 상태가 없어서, 서버가 실어주지 않으면 방법이 없다.

    `reference_key` 는 유튜브면 `videoId`, 데이트면 `mainPlace.name`.
    """

    result_type: ResultType
    reference_key: str
    created_at: datetime | None = None

    @field_validator("created_at")
    @classmethod
    def _kst(cls, v: datetime | None) -> datetime | None:
        """오프셋 없는 시각은 KST 로 본다.

        **`Message.sentAt` 과 같은 이유인데 여기는 터지는 방식이 다르다.** 이 값은
        `ctx.now`(타임존 인식)와 **빼기**를 한다 (`Context.minutes_since`). 규격서 예시가
        `"2026-08-14T21:00:00"` 처럼 오프셋 없이 오므로, 정규화하지 않으면 그 뺄셈이
        `TypeError` 로 터지고 요청 전체가 `MODEL_ERROR` 가 된다.
        """
        return as_kst(v) if v is not None else None


class SpeakerProfileInput(Camel):
    """서버가 집계해 실어주는 화자별 평소 말투 기준선 (2026-08-17 합의 — 우선 목데이터).

    다섯 값 전부 저장된 메시지에서 기계적으로 나오는 값이라 LLM 이 필요 없다.
    내부 계산은 `worker/profile.py` 가 같은 산식을 쓴다.
    """

    participant_key: ParticipantKey
    avg_length: float = 0.0
    period_rate: float = 0.0
    laugh_per_msg: float = 0.0
    emoji_rate: float = 0.0
    top_address: list[str] = Field(default_factory=list)


class AnalysisRequest(Camel):
    analysis_request_id: str
    chat_room_id: int
    participants: list[Participant]
    messages: list[Message]

    # 규격서 초안 v1 에는 없고 협의로 추가된 필드들. **둘 다 선택이다** —
    # 서버가 안 보내면 지금까지처럼 시드·폴백으로 동작한다 (`docs/server-handoff.md` 10장 1·2번).
    recent_results: list[RecentResult] = Field(default_factory=list)
    speaker_profiles: list[SpeakerProfileInput] = Field(default_factory=list)

    # 규격서에 없는 선택 필드다. **서버에 요청하지 않는다** — 시간대 판단은 LLM 에 넘기는
    # 대화 로그의 메시지별 `HH:MM` 으로 되고, 기억 중복 판정(30일)에는 마지막 메시지 시각과의
    # 차이가 의미가 없다. 픽스처로 결과를 재현할 때 시각을 고정하는 용도로만 남겨둔다.
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
    """감정 분석 = **실 상태 표현** (규격서 11장 · `docs/state-display-v4.md`).

    위젯 ①번 줄(상시)이 이 배열이고, ②번 줄(3종 개입)이 `results` 다.
    **화면이 두 줄이라 배열도 두 개다.** 서로 밀어내지 않는다.

    `subject` 는 감정의 주인, `viewer` 는 그걸 보는 사람이다. 두 값이 다르다 —
    **각자 상대방의 상태를 본다** (문서 4장①). 그래서 A 화면과 B 화면의 내용이 다르다.

    `state_text` 는 규격서에 없는 **선택 필드**다. 서버가 받아주면 실어 보내고, 아니면
    직렬화에서 빠진다(14장). 프론트가 라벨 → 문구 매핑을 갖는 경우에도 워커 로직은 같다.
    """

    subject_participant: ParticipantKey
    viewer_participant: ParticipantKey
    emotion_type: StateLabel
    intensity_value: float
    should_show: bool
    trigger_message_ids: list[int] = Field(default_factory=list)
    expires_at: datetime | None = None
    # 규격서에 없는 선택 필드 (문서 9장 🔵3)
    state_text: str | None = None


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
# 대화 분절 (`docs/segmentation-v3.md`)
# --------------------------------------------------------------------------
# 발화 하나가 직전까지의 맥락과 얼마나 이어지는가.
#
# ⚠️ **연속성 점수다. 변화량이 아니다.** 높을수록 **안 바뀐 것**이다 —
# 90~80점대면 "이전 맥락과 비슷하다". 방향을 뒤집지 말 것 (문서 3-3).
# 같은 설명이 `prompts/segment.md` 에 있다 — 모델은 그걸 읽는다.
#
# **LLM 은 여기까지만 한다. 자를지 말지는 `segment.py` 의 임계값이 정한다.**
# 경계가 LLM 안에 있으면 과분절이 나와도 조정할 손잡이가 없다.
#
# docstring 을 짧게 두는 이유는 `EmotionScores` 위 주석 참조 — 매 요청 전송된다.
class SegmentScore(BaseModel):
    """발화 하나의 맥락 연속성 점수. 높을수록 이어진다."""

    message_id: int
    same_context: bool
    topic_score: int   # 0~100 — 화제가 이어지는 정도
    tone_score: int    # 0~100 — 말투가 이어지는 정도

    # ⚠️ 근거 문구(`note`) 필드를 두지 않는다. 발화마다 한 줄씩 쓰게 하면 출력이 292 토큰까지
    # 늘어 분절 호출이 4.0초가 된다 — 빼면 2.3초다. 두 케이스를 5회씩 끝단으로 돌려
    # **경계가 완전히 같은 것**을 확인하고 뺐다 (문서 9장).
    #
    # "점수가 높을 때만 note 를 생략"하는 절충은 **하면 안 된다.** 조건부로 일을 줄여주면
    # 모델이 그쪽으로 쏠려서 점수를 전부 100 으로 매기고 경계를 통째로 놓친다. 실측했다.


class SegmentLLMOutput(BaseModel):
    scores: list[SegmentScore]


class Segment(BaseModel):
    """같은 화제로 이어지는 연속 메시지 묶음. 라우팅과 기억 추출의 단위.

    **화제 라벨(topic·mood)을 만들지 않는다.** 경계를 룰이 나중에 정하므로 채점 시점의
    LLM 은 세그먼트가 어디서 어디까지인지 모른다. 억지로 라벨을 받으면 "권태기 조짐"
    같은 문자열이 생기는데, `filter.py` 는 `AiResult` 의 화면 문자열만 검사해서 걸러지지
    않는다. **안 만드는 것이 가장 확실한 방어다** (문서 7장).
    디버깅에 필요한 정보는 `SegmentScore.note` 와 점수가 대신한다.
    """

    messages: list[Message]
    # 룰 컷(시간 공백)으로만 만들어졌는가. 채점으로 나뉜 것과 구분한다 (트레이스용).
    by_rule: bool = False

    @property
    def message_ids(self) -> list[int]:
        return [m.message_id for m in self.messages]


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


# 실제 장소는 카카오 로컬 API 로만 가져온다. LLM 이 상호명을 만들면 존재하지 않는
# 가게가 나오고 externalUrl 이 죽은 링크가 된다.
# (docstring 은 매 요청 전송되므로 짧게 — `EmotionScores` 위 주석 참조)
class DatePlanLLMOutput(BaseModel):
    """대화 + 기억을 읽고 무엇을 찾을지 정한다. 장소를 지어내지 않는다."""

    should_recommend: bool
    scope: Literal["common", "individual"]
    target: Literal["A", "B", "none"]
    region: str          # 검색 지역 ("성수동", "연남동"). 모르면 "none"

    # 코스는 **밥 → 구경 → 카페 세 자리로 고정**이다 (`date_course.SLOTS`).
    # 한 자리에 검색어를 2개씩 받는다 — 카카오가 0건을 주거나 카테고리가 안 맞으면
    # 두 번째를 쓴다. 슬롯이 없으면 카페가 두 곳 나오는 코스가 만들어진다 (실측).
    meal_queries: list[str]   # 밥 — 식당 검색어 2개
    sight_queries: list[str]  # 구경 — 문화시설·명소·소품샵 검색어 2개
    cafe_queries: list[str]   # 카페 — 검색어 2개

    # ⚠️ **코스명·요약을 여기서 받지 않는다.** `write_reason` 이 카카오가 준 실제 상호를
    # 보고 다시 쓰고 `to_result` 는 그쪽 값만 쓴다 — 여기서 만들면 그대로 버려진다.
    # 출력 토큰이 곧 지연이라(50%↓ → 최대 50%↓) 버릴 것을 만들게 하지 않는다.
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


# 룰로 잡을 수 없다. 명세도 "LLM 을 통한 스트림 내 대화 맥락 이해가 제일 중요"라고
# 못박아 뒀다. 그래서 이 기능은 게이트가 LLM 이다.
class ConcernLLMOutput(BaseModel):
    """관계 고민 신호 분류 + 검색어 생성."""

    should_recommend: bool
    concern: ConcernType
    scope: Literal["common", "individual"]
    target: Literal["A", "B", "none"]
    stage: str          # 관계 단계 추정 ("초반", "1년 이상", "unknown")
    queries: list[str]  # 한국어 유튜브 검색어 1~3개
    note: str           # 판정 근거 (내부용)


# 제목·썸네일만으로 고르지 않는다. 댓글이 "영상이 실제로 무엇을 말하는지"를 알려준다.
# 침묵이 기본값이라는 지시는 `prompts/yt_pick.md` 에도 있다.
# 유튜브 추천의 **두 번째 갈래** — 명세 "관계 고민 신호 **또는 공통 관심 주제**"
# (`docs/spec-v2.md` 3장). 고민 갈래만 먼저 구현했다가 나머지 절반을 채운 것이다.
#
# 고민 갈래와 **스키마도 프롬프트도 따로 둔다.** 한 프롬프트에 둘을 섞으면 고민 판정이
# 흐려진다 — 이 레포에서 이미 겪은 실패다(조건부로 일을 줄여주면 모델이 쉬운 쪽으로 쏠린다).
class TopicLLMOutput(BaseModel):
    """일상 대화의 화제가 **영상으로 이어질 만한가.**

    "먹방 얘기 → 쯔양", "왁뿌볼 얘기 → 왁뿌볼 ASMR" 처럼 **지금 하던 얘기의 연장**이어야
    한다. 관계·감정을 다루지 않는다 — 그건 고민 갈래가 한다.
    """

    should_recommend: bool
    topic: str          # 화제를 한 단어로 ("먹방", "캠핑"). 없으면 "none"
    queries: list[str]  # 한국어 유튜브 검색어 1~3개
    note: str           # 판정 근거 (내부용)


class VideoPickLLMOutput(BaseModel):
    """후보 영상과 댓글을 읽고 1개를 고른다. 적합한 후보가 없으면 picked_index = -1."""

    picked_index: int  # 후보 목록에서 고른 인덱스. 없으면 -1
    recommendation_reason: str


# --------------------------------------------------------------------------
# 실 상태 표현 (상시 — 후보가 아니다)
# --------------------------------------------------------------------------
# 화자 한 명의 감정 축 점수.
#
# ⚠️ **docstring 을 길게 쓰지 말 것.** 클래스 docstring 은 JSON 스키마의 `description`
# 으로 들어가 **매 요청 API 로 전송된다.** 여기 있던 설명 때문에 이 스키마만 640 토큰이었고
# 그중 414 가 사람용 주석이었다. 모델에게 필요한 지시는 `prompts/state.md` 에 있다.
#
# **LLM 은 점수까지만 낸다. 라벨도 문구도 만들지 않는다.**
# 라벨은 `state.pick_label()` 의 임계값이 정하고, 문구는 `copy.STATE_TEXT` 사전이 정한다.
#
# 처음에는 LLM 이 라벨을 직접 고르게 했다가 바꿨다. 서운함과 분노가 같이 높은 구간
# (다투는 중)에서 모델이 둘 중 하나를 임의로 골랐고, 점수가 없으니 왜 그쪽인지 알 수도
# 조정할 수도 없었다 — `case11_mixed` 에서 말투 교정은 공격 표현으로 잡은 발화를 상태
# 산출은 서운함으로 읽었다. **분절이 경계를 LLM 밖으로 뺀 이유와 같다**
# (`docs/state-display-v4.md` 6장). 명세의 산출 흐름과도 이쪽이 맞는다 —
# "감정 수치 스코어링(예: Sadness 3점) → 상태 라벨 매핑".
#
# ⚠️ **평온(`STABLE`)에 해당하는 축은 없다.** 처음에는 `calm` 을 두었는데 모델이 그걸
# 바닥값으로 깔아서(전 픽스처에서 3) 다른 축이 3 이하면 전부 `STABLE` 로 먹혔다.
# 평온은 다른 감정이 없는 상태지 경쟁하는 감정이 아니다. 네 축이 전부 임계 아래면 그게
# 평온이다 — `state.pick_label()` 이 그렇게 처리한다.
class EmotionScores(BaseModel):
    """화자 한 명의 감정 점수. 각 축 0~5."""

    speaker: Speaker
    affection: int   # 애정·설렘·화해. 쌓인 것이 풀리는 흐름     → RESOLVED
    hurt: int        # 서운함·오해·답답함. 안으로 쌓이는 흐름     → ACCUMULATED
    joy: int         # 들뜸·활기·웃음                          → ENGAGED
    anger: int       # 분노·격앙. 밖으로 터뜨림                 → ESCALATED
    confident: bool  # 근거가 뚜렷한가. false 면 파이썬 쪽에서 STABLE 로 내린다
    note: str        # 판정 근거 (내부용, 사용자에게 보여주지 않는다)


class StateLLMOutput(BaseModel):
    states: list[EmotionScores]


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
