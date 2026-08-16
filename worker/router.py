"""개입 방향 결정 — 후보 기능을 돌려 `results` 배열을 만든다.

규격서 6장은 한 번의 분석 요청에서 **여러 기능이 동시에 발동할 수 있다**고 정의한다.
그래서 라우터는 "하나를 고르는" 구조가 아니라 "발동한 것을 모으는" 구조다.

**후보들은 전체 스트림이 아니라 활성 세그먼트(`ctx.active`)만 본다.**
개입은 지금 벌어지고 있는 대화에 대해서 하는 것이다 — 두 시간 전에 끝난 화제에 지금
카드를 띄우는 건 늦은 게 아니라 틀린 것이다 (`docs/segmentation-v3.md` 5장).

맥락과 트리거를 갈라 쓴다:

    게이트 · 트리거 id · RAG · 기억 추출   ctx.active.messages   (활성 세그먼트만)
    말투 판정 프롬프트의 직전 대화         ctx.context           (부족분만 앞에서 채움)
    말투 기준선(profile)                  ctx.messages          (전체 — 표본이 넓을수록 정확)

    1. 갈등 중재 (TONE_CORRECTION)      — 방금 보낸 메시지 하나를 본다
    2. 데이트 코스 (DATE_RECOMMENDATION) — 데이트 계획 의도를 본다
    3. 유튜브 영상 (YOUTUBE_RECOMMENDATION) — 관계 고민 신호를 본다

> ⚠️ **위젯 슬롯 정책 미확정.** 기존 설계는 "위젯 슬롯 1개"였고 규격서는 배열을 허용한다.
> 프론트(민상)가 여러 개를 어떻게 배치할지 정해지지 않았다. 워커는 규격대로 전부 실어
> 보내고, 목록은 **우선순위 순으로 정렬**해 둔다. 하나만 쓰려면 `results[0]` 을 쓰면 된다.

각 후보는 `build(ctx) -> AiResult | None` 을 구현한다. `None` 은 "발동하지 않음"이다.
후보를 추가하려면 `CANDIDATES` 에 끼우고 규격서 `resultType` 에 값을 추가하면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from worker import date_course, places, youtube
from worker.copy import TONE_GUIDE
from worker.date_course import MEMORY_K, MEMORY_KINDS, MIN_PLACES
from worker.extract import extract_memories
from worker.filter import banned_in, find_banned, is_clean
from worker.models import (
    AiResult,
    AnalysisRequest,
    ConcernLLMOutput,
    DateGateResult,
    DatePlanLLMOutput,
    Memory,
    Message,
    Segment,
    ToneGateResult,
    ToneJudgeLLMOutput,
    ToneResultData,
    to_key,
)
from worker.places import KakaoPlace
from worker.profile import resolve_profile
from worker.retrieve import mark_used, recent_context, retrieve_many, save_memories
from worker.segment import active_context, segment
from worker.tone import check_tone_gate, tone_judge, tone_suggest
from worker.ytapi import Video

# 갈등 중재가 발동한 요청에서는 유튜브 추천을 건너뛴다.
#
# 둘 다 갈등 중재 성격이지만 개입 시점이 다르다. 말투 교정은 **지금 막 보낸 메시지**에
# 붙고, 영상 추천은 명세상 **냉각기**에 주는 기능이다. 방금 공격적인 메시지를 보낸
# 사람에게 교정 제안과 영상 카드가 동시에 뜨면 훈계처럼 읽힌다.
# LLM 판정(`yt_concern.md`)도 "한창 싸우는 중이면 false"로 막고 있지만, 규칙으로 한 번 더 막는다.
# 정책이 바뀌면 이 상수만 False 로 두면 된다.
SUPPRESS_YOUTUBE_WHEN_TONE = True


class Trace:
    """--verbose 용 중간 단계 기록. 판정에는 관여하지 않는다."""

    def __init__(self) -> None:
        self.fired: list[str] = []
        self.skipped: list[tuple[str, str]] = []  # (후보, 이유)
        # 대화 분절
        self.segments: list[Segment] = []
        # 기억
        self.extracted: list[Memory] = []
        self.saved: list[Memory] = []
        # 갈등 중재
        self.tone_gate: ToneGateResult | None = None
        self.tone_judged: ToneJudgeLLMOutput | None = None
        # 데이트 코스
        self.date_gate: DateGateResult | None = None
        self.date_memories: list[Memory] = []
        self.date_plan: DatePlanLLMOutput | None = None
        self.date_places: list[KakaoPlace] = []
        # 유튜브
        self.concern: ConcernLLMOutput | None = None
        self.yt_candidates: list[Video] = []
        self.yt_picked: Video | None = None

    def skip(self, name: str, reason: str) -> None:
        self.skipped.append((name, reason))


@dataclass
class Context:
    request: AnalysisRequest
    messages: list[Message]          # 전체 스트림. 말투 기준선 계산에만 쓴다
    now: datetime
    persist: bool = True
    trace: Trace = field(default_factory=Trace)

    # 분절 결과. `split()` 이 채운다.
    segments: list[Segment] = field(default_factory=list)
    context: list[Message] = field(default_factory=list)  # 말투 판정용 맥락

    @property
    def active(self) -> list[Message]:
        """활성 세그먼트의 메시지. 게이트·트리거·RAG·기억 추출이 보는 범위."""
        return self.segments[-1].messages if self.segments else self.messages


class Candidate(Protocol):
    name: str

    def build(self, ctx: Context) -> AiResult | None:
        """발동하지 않으면 None 을 돌려준다."""
        ...


# --------------------------------------------------------------------------
# 후보 1 — 갈등 중재 (말투 교정 제안)
# --------------------------------------------------------------------------
class ToneCandidate:
    name = "tone"

    def build(self, ctx: Context) -> AiResult | None:
        gate = check_tone_gate(ctx.active)
        ctx.trace.tone_gate = gate
        if not gate.triggered or gate.speaker is None:
            return None

        # 기준선은 전체 스트림으로 잡는다 — 표본이 넓을수록 "평소 대비"가 정확해진다.
        # 판정 프롬프트의 직전 대화는 맥락(ctx.context)을 쓴다.
        profile = resolve_profile(gate.speaker, ctx.messages)
        judged = tone_judge(ctx.context, gate, profile)
        ctx.trace.tone_judged = judged
        if not judged.should_suggest:
            ctx.trace.skip(self.name, "맥락 판정 — 갈등이 아님" + (" (장난)" if judged.is_playful else ""))
            return None

        def _make() -> AiResult:
            out = tone_suggest(ctx.context, gate, profile, judged)
            return AiResult(
                result_type="TONE_CORRECTION",
                visibility_type="INDIVIDUAL",  # 보낸 사람에게만
                target_participant=to_key(gate.speaker),
                content_type="TEXT",
                trigger_message_ids=[gate.message_id] if gate.message_id is not None else [],
                result_data=ToneResultData(
                    situation_diagnosis=out.situation_diagnosis.strip(),
                    guide_message=TONE_GUIDE,
                    alternative_sentence=out.alternative_sentence.strip(),
                    correction_reason=out.correction_reason.strip(),
                ),
            )

        # 금지어에 걸리면 1회 재생성. 또 걸리면 내보내지 않는다.
        result = _make()
        if is_clean(result):
            return result
        retry = _make()
        if is_clean(retry):
            return retry
        ctx.trace.skip(self.name, f"금지어 필터 — '{banned_in(retry)}'")
        return None


# --------------------------------------------------------------------------
# 후보 2 — 데이트 코스 추천
# --------------------------------------------------------------------------
class DateCandidate:
    name = "date"

    def build(self, ctx: Context) -> AiResult | None:
        gate = date_course.check_date_gate(ctx.active)
        ctx.trace.date_gate = gate
        if not gate.triggered:
            return None

        if not places.available():
            ctx.trace.skip(self.name, "KAKAO_REST_API_KEY 없음 — 장소를 지어내지 않고 미발동")
            return None

        recent = recent_context(ctx.active)
        memories = retrieve_many(recent, k=MEMORY_K, now=ctx.now, kinds=MEMORY_KINDS)
        ctx.trace.date_memories = memories

        plan = date_course.plan_date(ctx.active, memories, gate)
        ctx.trace.date_plan = plan
        if not plan.should_recommend:
            ctx.trace.skip(self.name, "LLM 판정 — 지금 제안할 상황이 아님")
            return None

        region = None if plan.region.strip().lower() in ("", "none") else plan.region.strip()
        course = date_course.build_course(plan.queries, region)
        ctx.trace.date_places = course
        if len(course) < MIN_PLACES:
            ctx.trace.skip(self.name, f"카카오 검색 결과 부족 ({len(course)}곳)")
            return None

        reason = date_course.write_reason(ctx.active, memories, plan, course)
        result = date_course.to_result(plan, reason, course, gate.message_ids)
        if not is_clean(result):
            ctx.trace.skip(self.name, f"금지어 필터 — '{banned_in(result)}'")
            return None

        # 근거로 삼은 기억을 소모 처리한다. 같은 소재가 매번 다시 나오지 않게 한다.
        if memories:
            mark_used(memories[0].id, now=ctx.now, persist=ctx.persist)
        return result


# --------------------------------------------------------------------------
# 후보 3 — 유튜브 영상 추천
# --------------------------------------------------------------------------
class YoutubeCandidate:
    name = "youtube"

    def build(self, ctx: Context) -> AiResult | None:
        trigger_ids = youtube.check_concern_gate(ctx.active)
        if not trigger_ids:
            return None

        if not youtube.available():
            ctx.trace.skip(self.name, "YOUTUBE_API_KEY 없음 — 영상을 지어내지 않고 미발동")
            return None

        concern = youtube.classify_concern(ctx.active)
        ctx.trace.concern = concern
        if not concern.should_recommend or not concern.queries:
            ctx.trace.skip(self.name, "LLM 판정 — 관계 고민 신호 아님")
            return None

        videos = youtube.find_candidates(concern.queries)
        # 제목·설명에 금지어가 든 영상은 LLM 에 보이기 전에 뺀다.
        # 우리가 쓴 문장이 아니어도 화면에는 그대로 뜬다.
        videos = [
            v for v in videos
            if find_banned(v.title) is None and find_banned(v.summary()) is None
        ]
        ctx.trace.yt_candidates = videos
        if not videos:
            ctx.trace.skip(self.name, "후보 영상 없음 — 침묵")
            return None

        pick = youtube.pick_video(ctx.active, concern, videos)
        if not 0 <= pick.picked_index < len(videos):
            ctx.trace.skip(self.name, "후보 전원 탈락 — 침묵")
            return None

        video = videos[pick.picked_index]
        ctx.trace.yt_picked = video
        result = youtube.to_result(concern, video, pick.recommendation_reason, trigger_ids)
        if not is_clean(result):
            ctx.trace.skip(self.name, f"금지어 필터 — '{banned_in(result)}'")
            return None
        return result


# 우선순위 순. 규격서상 여러 개가 동시에 나갈 수 있으므로 앞이 뒤를 막지 않는다.
CANDIDATES: list[Candidate] = [ToneCandidate(), DateCandidate(), YoutubeCandidate()]


# --------------------------------------------------------------------------
# 분절 + 기억 수확 + 라우팅
# --------------------------------------------------------------------------
def split(ctx: Context) -> None:
    """스트림을 화제 단위로 끊는다. 모든 것보다 **먼저** 돈다.

    이후 단계가 보는 범위가 여기서 정해진다. 설계는 `docs/segmentation-v3.md`.
    """
    ctx.segments = segment(ctx.messages)
    ctx.context = active_context(ctx.segments)
    ctx.trace.segments = ctx.segments


def harvest_memories(ctx: Context) -> None:
    """대화에서 기억을 뽑아 저장소에 넣는다.

    후보들보다 **먼저** 돈다. 방금 "마라탕 땡긴다"고 한 발화가 같은 요청의 데이트 코스
    추천에 바로 반영되게 하기 위해서다.

    **활성 세그먼트만 본다.** 화제가 섞인 덩어리에서 인용을 뽑으면 맥락이 어긋난 기억이
    저장되고, 그게 나중에 추천 이유로 화면에 그대로 나간다. 과거 세그먼트는 이전 요청에서
    이미 추출됐다.
    """
    extracted = extract_memories(ctx.active)
    ctx.trace.extracted = extracted
    ctx.trace.saved = save_memories(extracted, persist=ctx.persist)


def route(ctx: Context) -> list[AiResult]:
    results: list[AiResult] = []
    for candidate in CANDIDATES:
        if (
            SUPPRESS_YOUTUBE_WHEN_TONE
            and candidate.name == "youtube"
            and any(r.result_type == "TONE_CORRECTION" for r in results)
        ):
            ctx.trace.skip(candidate.name, "말투 교정이 발동한 요청 — 냉각기가 아니므로 보류")
            continue

        result = candidate.build(ctx)
        if result is not None:
            results.append(result)
            ctx.trace.fired.append(candidate.name)
    return results
