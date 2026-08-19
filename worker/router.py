"""개입 방향 결정 — 후보 기능을 돌려 `results` 배열을 만든다.

규격서 6장은 한 번의 분석 요청에서 **여러 기능이 동시에 발동할 수 있다**고 정의한다.
그래서 라우터는 "하나를 고르는" 구조가 아니라 "발동한 것을 모으는" 구조다.

> 위젯은 **두 줄**이다. 여기서 모으는 `results` 는 ②번 줄이고, 미발동이면 그냥 빈다.
> ①번 줄(실 상태 표현)은 후보가 아니라 상시라서 `read_state()` 가 따로 처리하고
> `emotionAnalyses` 로 나간다 (`docs/design.md` 2부).

**후보들은 전체 스트림이 아니라 활성 세그먼트(`ctx.active`)만 본다.**
개입은 지금 벌어지고 있는 대화에 대해서 하는 것이다 — 두 시간 전에 끝난 화제에 지금
카드를 띄우는 건 늦은 게 아니라 틀린 것이다 (`docs/design.md` 1부 5장).

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

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from worker import date_course, places, state, youtube
from worker.copy import TONE_GUIDE
from worker.date_course import COURSE_PLACES, MEMORY_K, MEMORY_KINDS
from worker.extract import extract_memories
from worker.filter import banned_in, banned_in_state, find_banned, is_clean
from worker.limits import enforce, over_limit
from worker.models import (
    AiResult,
    AnalysisRequest,
    ConcernLLMOutput,
    DateGateResult,
    DatePlanLLMOutput,
    EmotionAnalysis,
    EmotionScores,
    Memory,
    Message,
    Segment,
    SegmentScore,
    ToneGateResult,
    ToneJudgeLLMOutput,
    ToneResultData,
    TopicLLMOutput,
    to_key,
    to_speaker,
)
from worker.places import KakaoPlace
from worker.profile import resolve_profile
from worker.retrieve import mark_used, recent_context, retrieve_many, save_memories
from worker.segment import active_context, segment
from worker.tone import check_tone_gate, harsh_in, tone_judge, tone_suggest
from worker.ytapi import Video

# 갈등 중재가 발동한 요청에서는 유튜브 추천을 건너뛴다.
#
# 둘 다 갈등 중재 성격이지만 개입 시점이 다르다. 말투 교정은 **지금 막 보낸 메시지**에
# 붙고, 영상 추천은 명세상 **냉각기**에 주는 기능이다. 방금 공격적인 메시지를 보낸
# 사람에게 교정 제안과 영상 카드가 동시에 뜨면 훈계처럼 읽힌다.
# LLM 판정(`yt_concern.md`)도 "한창 싸우는 중이면 false"로 막고 있지만, 규칙으로 한 번 더 막는다.
# 정책이 바뀌면 이 상수만 False 로 두면 된다.
SUPPRESS_YOUTUBE_WHEN_TONE = True

# 유튜브 억제 — **같은 화제엔 영상 하나만** + 시간 백스톱 (분).
#
# **메시지 1건마다 워커가 한 번 불린다** (2026-08-18 백엔드 확정). 억제가 없으면 같은
# 화제가 이어지는 동안 메시지마다 검색이 나간다 — 대화 한 번에 쿼터가 통째로 마른다.
#
#     YouTube Data API 무료 쿼터 10,000 units/일 · 추천 1건 약 106 units → 하루 약 94회
#
# 소진되면 **오류가 아니라 조용한 미발동**으로 바뀐다 (`ytapi._get` 이 실패를 삼킨다).
# 화면에서는 "적절한 영상이 없었다"와 구분되지 않아서, 시연 중이면 원인을 못 찾는다.
#
# 주 억제는 **세그먼트 기준**이다 — 마지막 영상의 `createdAt` 이 활성 세그먼트 시작
# 뒤면 지금 화제에 이미 답한 것이라 보류한다. 처음엔 30분 고정 쿨다운이었는데 화제를
# 못 봐서 양쪽으로 틀렸다 — 새 화제엔 30분 안이라도 떠야 하고, 같은 화제엔 30분
# 지나도 두 번 뜨면 안 된다 (2026-08-19).
#
# 시간 백스톱이 남는 이유는 세그먼트 경계가 LLM 산출이라서다 — 요청마다 다시 긋기
# 때문에 같은 화제 중간에 경계가 흔들리면 그때마다 검색이 나간다. 백스톱이 그 폭주를
# 분당 상한으로 누른다. 백엔드 `createdAt` 과 메시지 `sentAt` 의 시계가 어긋나
# 세그먼트 비교가 빗나가는 경우도 같이 받친다.
#
# **근거는 서버가 실어주는 `recentResults[].createdAt` 이다.** 워커가 상태를 갖지 않는다는
# 원칙을 깨지 않는다. 그 필드가 안 오면 억제도 걸리지 않는다 — 지금까지와 같이 동작한다.
#
# 시연 리허설을 연달아 돌려야 하면 `KAKAPO_YOUTUBE_COOLDOWN_MIN=0` — 둘 다 꺼진다.
YOUTUBE_COOLDOWN_MIN = float(os.getenv("KAKAPO_YOUTUBE_COOLDOWN_MIN", "5"))

# --------------------------------------------------------------------------
# 시연 강제 트리거 — **환경변수로만 켠다. 기본은 꺼짐.**
#
# 방금 발화에 '데이트' / '유튜브' 낱말이 보이면 판단(게이트 · 억제 · LLM 보류)을 건너뛰고
# 해당 추천을 무조건 태운다. 시연에서 "말했는데 안 뜨는" 사고를 없애기 위한 것이다.
#
# **안 건너뛰는 것** — 절대 제약은 데모에서도 유지된다:
#   · 금지어 필터 (외부 문자열 포함)
#   · "이름·링크는 외부 API 가 준 것만" (카카오 0건이면 그 자리는 여전히 못 채운다)
#   · 화면 자수 한도
#
# **마지막 발화만 본다.** 세그먼트 전체를 보면 백엔드가 메시지 1건마다 부르는 구조상
# 키워드 이후 모든 메시지에서 재발동해 유튜브 쿼터가 마른다.
DEMO_TRIGGERS = os.getenv("KAKAPO_DEMO_TRIGGERS", "").strip().lower() in ("1", "true", "on")
DEMO_DATE_REGION = os.getenv("KAKAPO_DEMO_REGION", "성수").strip()


def demo_hit(ctx: "Context", word: str) -> Message | None:
    """시연 모드에서 방금 발화에 낱말이 있으면 그 발화. 아니면 None."""
    if not DEMO_TRIGGERS or not ctx.active:
        return None
    last = ctx.active[-1]
    return last if word in last.content else None

# 유튜브 **화제 갈래**가 비켜야 하는 데이트 신호.
#
# 4종 중 `category`("배고파" · "카페" · "맛집")는 뺐다 — 단어 하나로 걸려서 먹방 얘기가
# 데이트 의도로 잡힌다. 나머지 셋은 실제로 약속을 잡는 신호라 장소 추천이 맞다.
DATE_INTENT_STRONG = {"plan_question", "schedule_fixed", "recall"}


class Trace:
    """--verbose 용 중간 단계 기록. 판정에는 관여하지 않는다."""

    def __init__(self) -> None:
        self.fired: list[str] = []
        self.skipped: list[tuple[str, str]] = []  # (후보, 이유)
        # 결과는 내보내되 눈에 띄게 남길 것. **판정을 바꾸지 않는다** —
        # 재생성해도 안 고쳐지는 종류라 지연만 쓰기 때문이다 (`date_course.question_quote`).
        self.warnings: list[tuple[str, str]] = []
        # 대화 분절
        self.segments: list[Segment] = []
        self.scores: list[SegmentScore] = []  # 발화별 연속성 점수 (왜 잘렸는지)
        # 기억
        self.extracted: list[Memory] = []
        self.saved: list[Memory] = []
        # 실 상태 표현 (위젯 ①번 줄)
        self.states: list[EmotionAnalysis] = []
        self.state_scored: list[EmotionScores] = []  # 감정 5축 점수 + 근거 (내부용)
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
        self.topic: TopicLLMOutput | None = None  # 일상 화제 갈래
        self.yt_candidates: list[Video] = []
        self.yt_picked: Video | None = None

    def skip(self, name: str, reason: str) -> None:
        self.skipped.append((name, reason))

    def warn(self, name: str, note: str) -> None:
        self.warnings.append((name, note))


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

    def last_result_at(self, result_type: str) -> datetime | None:
        """그 기능이 **마지막으로 나간 시각.** 모르면 None.

        `recentResults` 는 지금까지 중복 제거(`recent_keys`)에만 쓰였다. 같은 배열의
        `createdAt` 을 보면 **언제 나갔는지**도 알 수 있다 — 워커가 상태를 갖지
        않고도 빈도를 조절할 수 있는 유일한 근거다. 세그먼트 시작과 비교하면
        "지금 화제에 이미 답했는가"가 되고, `now` 와 비교하면 시간 쿨다운이 된다.

        `createdAt` 이 없는 항목은 세지 않는다. 선택 필드라 서버가 안 보낼 수 있고,
        **모르는 것을 "오래됐다"로 치면 억제가 통째로 무력화된다.**
        """
        stamps = [
            r.created_at
            for r in self.request.recent_results
            if r.result_type == result_type and r.created_at is not None
        ]
        return max(stamps) if stamps else None

    def recent_keys(self, result_type: str) -> set[str]:
        """최근에 이미 내보낸 결과의 식별자 (규격서 10장 "동일 영상 재추천 방지").

        서버가 `recentResults` 를 안 보내면 빈 집합이고, 그러면 지금까지처럼 중복 방지가
        걸리지 않는다. **없는 것을 있는 척하지 않는다** — 워커는 상태를 갖지 않는다.
        """
        return {
            r.reference_key
            for r in self.request.recent_results
            if r.result_type == result_type and r.reference_key
        }


class Candidate(Protocol):
    name: str

    def build(self, ctx: Context) -> AiResult | None:
        """발동하지 않으면 None 을 돌려준다."""
        ...


def _fit(ctx: Context, name: str, result: AiResult, regenerate) -> AiResult:
    """화면 글자 수 한도를 지킨다 — **1회 재생성, 그래도 넘으면 절단.**

    금지어 필터와 같은 구조인데 **마지막 처리가 다르다.** 금지어는 절대 제약이라 못 지키면
    버리지만, 길이는 조금 넘었다고 기능을 통째로 미발동시킬 이유가 없다. 잘라서라도 내보낸다.

    프롬프트에 자수를 적어두는 것만으로는 안 지켜진다 — 실측에서 넘는 출력이 계속 나왔다
    (`worker/limits.py`).
    """
    hit = over_limit(result)
    if hit is None:
        return result

    field, actual, limit = hit
    ctx.trace.skip(name, f"글자 수 초과 — {field} {actual}자 > {limit}자, 재생성")

    retry = regenerate()
    if over_limit(retry) is None:
        return retry

    field, actual, limit = over_limit(retry)  # type: ignore[misc]
    ctx.trace.skip(name, f"재생성도 초과 — {field} {actual}자 > {limit}자, 잘라서 내보냄")
    return enforce(retry)


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

        # 기준선은 요청에 실려 온 값(`speakerProfiles`)이 있으면 그것을 쓴다. 없으면 시드,
        # 그것도 없으면 전체 스트림에서 계산한다 — 표본이 넓을수록 "평소 대비"가 정확해진다.
        # 판정 프롬프트의 직전 대화는 맥락(ctx.context)을 쓴다.
        profile = resolve_profile(gate.speaker, ctx.messages, ctx.request.speaker_profiles)
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
        if not is_clean(result):
            result = _make()
            if not is_clean(result):
                ctx.trace.skip(self.name, f"금지어 필터 — '{banned_in(result)}'")
                return None

        # 대체 문장에 거친 어휘가 남았으면 1회 재생성.
        #
        # **금지어와 달리 버리지 않는다.** 금지어는 절대 제약이지만 이건 품질이고,
        # 거친 말이 조금 남아도 원문(공격 표현)보다는 낫다. 길이 초과를 잘라서라도
        # 내보내는 것과 같은 판단이다.
        if (harsh := harsh_in(result.result_data.alternative_sentence, profile)) is not None:
            retry = _make()
            if harsh_in(retry.result_data.alternative_sentence, profile) is None:
                result = retry
            else:
                ctx.trace.warn(self.name, f"대체 문장에 거친 표현이 남음 — '{harsh}'")

        return _fit(ctx, self.name, result, _make)


# --------------------------------------------------------------------------
# 후보 2 — 데이트 코스 추천
# --------------------------------------------------------------------------
class DateCandidate:
    name = "date"

    def build(self, ctx: Context) -> AiResult | None:
        demo = demo_hit(ctx, "데이트")
        gate = date_course.check_date_gate(ctx.active)
        if demo is not None and not gate.triggered:
            gate = DateGateResult(
                triggered=True,
                kinds=["plan_question"],
                detail="시연 강제 트리거 — '데이트'",
                message_ids=[demo.message_id],
            )
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
            if demo is None:
                ctx.trace.skip(self.name, "LLM 판정 — 지금 제안할 상황이 아님")
                return None
            ctx.trace.warn(self.name, "시연 모드 — LLM 보류를 무시하고 진행")

        region = None if plan.region.strip().lower() in ("", "none") else plan.region.strip()

        # **지역을 못 잡으면 발동하지 않는다.**
        #
        # `date_plan.md` 에 이미 "지역을 전혀 알 수 없고 기억에도 장소 단서가 없다 → false"
        # 라고 적혀 있지만 **LLM 이 안 지킨다.** 실측에서 `region=none` 인 채로
        # `should_recommend=True` 가 나왔고, 지역이 없으니 카카오가 전국 1위를 물어와
        # **서울 중구 / 충남 천안**이 코스로 나갔다 ("주말에 뭐 할까 / 영화나 볼까" 대화).
        #
        # 코스 안에서 동선은 이어진다(첫 장소를 앵커로 삼는다). **문제는 그 동선이 커플과
        # 아무 상관이 없다는 것이다.** 어디 사는지 모르는 사람에게 천안 빵집을 제안하면
        # 추천이 아니라 오작동으로 읽힌다.
        #
        # 프롬프트로 안 되는 것을 코드가 막는 자리다 — 자리별 카테고리 강제(`SLOTS`),
        # 화면 자수 한도(`limits.py`)와 같은 패턴이다. **애매하면 개입하지 않는다.**
        if region is None:
            if demo is None:
                ctx.trace.skip(self.name, "지역 단서 없음 — 아무 데나 추천하지 않는다")
                return None
            # 시연 모드의 기본 지역. 지어낸 지역이 화면에 가는 위험(위 주석의 천안 사고)은
            # 시연 대본이 지역을 말하면 사라진다 — 이 폴백은 대본이 빗나갔을 때의 안전망이다.
            region = DEMO_DATE_REGION
            ctx.trace.warn(self.name, f"시연 모드 — 지역 단서 없음, 기본 지역 '{region}'")

        if demo is not None:
            # LLM 이 보류하면서 검색어를 비워 보냈을 때만 채운다. 준 검색어는 건드리지 않는다.
            plan = plan.model_copy(update={
                "meal_queries": plan.meal_queries or ["맛집", "파스타"],
                "sight_queries": plan.sight_queries or ["소품샵", "전시"],
                "cafe_queries": plan.cafe_queries or ["카페", "디저트 카페"],
            })

        course = date_course.build_course(plan, region)

        # 최근에 이미 추천한 장소는 뺀다. 같은 커플에게 매번 같은 코스를 주지 않는다.
        # 서버가 `recentResults` 를 안 보내면 빈 집합이라 아무것도 안 걸러진다.
        seen = ctx.recent_keys("DATE_RECOMMENDATION")
        if seen:
            dropped = [p.name for p in course if p.name in seen]
            course = [p for p in course if p.name not in seen]
            if dropped:
                ctx.trace.skip(self.name, f"최근 추천한 장소 제외 — {', '.join(dropped)}")

        ctx.trace.date_places = course
        # 코스는 **항상 3곳**이다. 요청마다 2곳/3곳이 섞이면 화면이 달라 보인다.
        # 검색어를 예비까지 받아도 못 채우면(지역에 그 업종이 없는 경우) 미발동한다 —
        # 억지로 먼 곳을 끼워 넣으면 코스가 아니게 된다.
        if len(course) < COURSE_PLACES:
            ctx.trace.skip(
                self.name, f"카카오 검색 결과 부족 ({len(course)}곳 / {COURSE_PLACES}곳 필요)"
            )
            return None

        def _make() -> AiResult:
            reason = date_course.write_reason(ctx.active, memories, plan, course)
            return date_course.to_result(plan, reason, course, gate.message_ids)

        result = _make()
        if not is_clean(result):
            ctx.trace.skip(self.name, f"금지어 필터 — '{banned_in(result)}'")
            return None

        # 글자 수만 넘은 것이면 문구 생성(`write_reason`)만 다시 돈다.
        # 카카오 검색과 계획은 이미 확정이라 다시 부르지 않는다.
        result = _fit(ctx, self.name, result, _make)

        # 인용에 묻는 말이 남았으면 트레이스에만 남긴다 (재생성하지 않는 이유는 검출기 주석).
        reason_text = result.result_data.recommendation_reason  # type: ignore[union-attr]
        if (asked := date_course.question_quote(reason_text)) is not None:
            ctx.trace.warn(self.name, f"인용에 묻는 말이 남음 — '{asked}'")

        # 근거로 삼은 기억을 소모 처리한다. 같은 소재가 매번 다시 나오지 않게 한다.
        if memories:
            mark_used(memories[0].id, now=ctx.now, persist=ctx.persist)
        return result


# --------------------------------------------------------------------------
# 후보 3 — 유튜브 영상 추천
# --------------------------------------------------------------------------
class YoutubeCandidate:
    """명세의 두 갈래를 모두 태운다 — **관계 고민 신호 또는 공통 관심 주제**
    (`docs/spec.md` 3장).

        ① 고민 갈래   싸우고 서먹할 때. 노출 범위를 LLM 이 정한다 (개별/공통)
        ② 화제 갈래   평범한 일상 대화에서 화제가 뚜렷할 때. 항상 COUPLE

    **고민이 우선이다.** 둘 다 신호가 있으면 고민 갈래를 탄다 — 감정이 걸린 대화에
    먹방 영상을 띄우면 눈치가 없다.

    두 갈래는 **프롬프트도 화면 문구도 따로**다. 판정 기준이 다르기 때문이다 —
    고민은 "이 상황에 도움이 되는가", 화제는 "그 소재를 실제로 다루는가".
    """

    name = "youtube"

    def build(self, ctx: Context) -> AiResult | None:
        # 시연 강제 트리거 — 억제·쿨다운·게이트를 전부 건너뛰고 화제 갈래를 태운다.
        if (demo := demo_hit(ctx, "유튜브")) is not None:
            return self._topic(ctx, demo=demo)

        # **억제를 게이트보다 먼저 본다.** 뒤에 두면 이미 LLM 분류를 부른 뒤라
        # 아낀 게 검색 쿼터뿐이다. 여기서 끊으면 LLM 호출까지 같이 준다.
        # 두 갈래에 똑같이 건다 — 쿼터는 갈래를 가리지 않고 한 통에서 나간다.
        last = ctx.last_result_at("YOUTUBE_RECOMMENDATION")
        if last is not None and YOUTUBE_COOLDOWN_MIN > 0:
            # ① 같은 화제엔 하나만 — 마지막 영상이 활성 세그먼트 시작 뒤에 나갔으면
            #    지금 이어지는 화제에 이미 답한 것이다. 화제가 바뀌면(새 세그먼트) 풀린다.
            if ctx.active and last >= ctx.active[0].sent_at:
                ctx.trace.skip(self.name, "이 화제에 이미 영상을 냈다 — 새 화제까지 보류")
                return None
            # ② 시간 백스톱 — 세그먼트 경계가 흔들리거나 두 시계가 어긋날 때의 폭주 방지
            #    (상수 주석 참조). 음수는 0으로 눌러 둔다 — `createdAt` 은 백엔드 시계라
            #    미래로 들어올 수 있는데, 로그에 "-50분 전"이 남으면 읽는 사람이 헤맨다.
            since = max(0.0, (ctx.now - last).total_seconds() / 60)
            if since < YOUTUBE_COOLDOWN_MIN:
                ctx.trace.skip(
                    self.name,
                    f"{since:.0f}분 전에 영상을 냈다 — 백스톱 {YOUTUBE_COOLDOWN_MIN:.0f}분",
                )
                return None

        concern_ids = youtube.check_concern_gate(ctx.active)
        if concern_ids:
            return self._concern(ctx, concern_ids)
        return self._topic(ctx)

    # ---------------------------------------------------------------- 공통
    def _candidates(self, ctx: Context, queries: list[str]) -> list[Video] | None:
        """검색 → 금지어·중복 제외. 쓸 후보가 없으면 None."""
        videos = youtube.find_candidates(queries)

        # 제목·설명에 금지어가 든 영상은 LLM 에 보이기 전에 뺀다.
        # 우리가 쓴 문장이 아니어도 화면에는 그대로 뜬다.
        videos = [
            v for v in videos
            if find_banned(v.title) is None and find_banned(v.summary()) is None
        ]

        # 규격서 10장 — "동일 영상을 최근에 추천한 경우 다른 후보 탐색".
        # **LLM 에 보이기 전에 뺀다.** 후보 목록에 남겨두고 "고르지 말라"고 하면
        # 지시를 어길 여지가 생기고, 후보가 줄어든 만큼 검색을 더 하지도 않는다.
        seen = ctx.recent_keys("YOUTUBE_RECOMMENDATION")
        if seen:
            before = len(videos)
            videos = [v for v in videos if v.video_id not in seen]
            if len(videos) < before:
                ctx.trace.skip(self.name, f"최근 추천한 영상 제외 — {before - len(videos)}건")

        ctx.trace.yt_candidates = videos
        if not videos:
            ctx.trace.skip(self.name, "후보 영상 없음 — 침묵")
            return None
        return videos

    def _ready(self, ctx: Context) -> bool:
        if youtube.available():
            return True
        ctx.trace.skip(self.name, "YOUTUBE_API_KEY 없음 — 영상을 지어내지 않고 미발동")
        return False

    # ---------------------------------------------------------------- ① 고민
    def _concern(self, ctx: Context, trigger_ids: list[int]) -> AiResult | None:
        if not self._ready(ctx):
            return None

        concern = youtube.classify_concern(ctx.active)
        ctx.trace.concern = concern
        if not concern.should_recommend or not concern.queries:
            ctx.trace.skip(self.name, "LLM 판정 — 관계 고민 신호 아님")
            return None

        videos = self._candidates(ctx, concern.queries)
        if videos is None:
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

        # 재생성은 `pick_video` 만 다시 돈다. 검색(쿼터 100 units)은 다시 부르지 않는다.
        def _again() -> AiResult:
            again = youtube.pick_video(ctx.active, concern, videos)
            if not 0 <= again.picked_index < len(videos):
                return result
            return youtube.to_result(
                concern, videos[again.picked_index], again.recommendation_reason, trigger_ids
            )

        return _fit(ctx, self.name, result, _again)

    # ---------------------------------------------------------------- ② 화제
    def _topic(self, ctx: Context, demo: Message | None = None) -> AiResult | None:
        # **데이트 의도가 뚜렷하면 화제 갈래를 타지 않는다.**
        # "주말에 뭐 할까" 에는 장소 추천이 나가야 한다. 영상이 끼어들면 자리를 뺏는다.
        #
        # 데이트 **결과**가 아니라 **게이트**(룰)를 본다. 결과를 기다리면 데이트 체인
        # (계획 → 카카오 → 문구, 약 6초)이 끝날 때까지 유튜브가 묶여서 병렬이 깨진다.
        # 게이트는 정규식이라 공짜다.
        #
        # ⚠️ **`category` 단독은 막지 않는다.** 그 신호는 "배고파" · "맛집" 같은 단어
        # 하나로 걸려서, **먹방 얘기가 통째로 데이트 의도로 잡힌다** (실측 — case13).
        # 약속을 잡는 신호(계획 질문 · 일정 확정 · 이전 약속 재언급)일 때만 비킨다.
        # 약한 신호로 데이트가 실제로 발동하면 `run()` 이 조립 시점에 뺀다.
        if demo is None:
            kinds = set(date_course.check_date_gate(ctx.active).kinds)
            if kinds & DATE_INTENT_STRONG:
                ctx.trace.skip(self.name, "약속을 잡는 대화 — 장소 추천에 자리를 넘김")
                return None

            trigger_ids = youtube.check_topic_gate(ctx.active)
            if not trigger_ids:
                return None
        else:
            trigger_ids = [demo.message_id]

        if not self._ready(ctx):
            return None

        topic = youtube.classify_topic(ctx.active)
        ctx.trace.topic = topic
        if not topic.should_recommend or not topic.queries:
            if demo is None:
                ctx.trace.skip(self.name, "LLM 판정 — 영상으로 이어 붙일 화제가 아님")
                return None
            # 시연 모드 — 검색어가 없으면 방금 발화에서 키워드를 뺀 나머지로 검색한다.
            fallback = demo.content.replace("유튜브", "").strip() or "커플 브이로그"
            topic = topic.model_copy(update={"should_recommend": True, "queries": [fallback]})
            ctx.trace.warn(self.name, f"시연 모드 — LLM 보류 무시, 검색어 '{fallback}'")

        videos = self._candidates(ctx, topic.queries)
        if videos is None:
            return None

        pick = youtube.pick_topic_video(ctx.active, topic, videos)
        if not 0 <= pick.picked_index < len(videos):
            if demo is None:
                ctx.trace.skip(self.name, "후보 전원 탈락 — 침묵")
                return None
            # 시연 모드 — 침묵 대신 첫 후보를 쓴다. 문구는 효과를 약속하지 않는 고정문.
            pick = pick.model_copy(update={
                "picked_index": 0,
                "recommendation_reason": "말씀 나눈 주제와 이어지는 영상이에요",
            })
            ctx.trace.warn(self.name, "시연 모드 — LLM 이 후보를 전원 탈락시켜 첫 후보로 대체")

        video = videos[pick.picked_index]
        ctx.trace.yt_picked = video
        result = youtube.to_topic_result(video, pick.recommendation_reason, trigger_ids)
        if not is_clean(result):
            ctx.trace.skip(self.name, f"금지어 필터 — '{banned_in(result)}'")
            return None

        def _again() -> AiResult:
            again = youtube.pick_topic_video(ctx.active, topic, videos)
            if not 0 <= again.picked_index < len(videos):
                return result
            return youtube.to_topic_result(
                videos[again.picked_index], again.recommendation_reason, trigger_ids
            )

        return _fit(ctx, self.name, result, _again)


# 우선순위 순. 규격서상 여러 개가 동시에 나갈 수 있으므로 앞이 뒤를 막지 않는다.
CANDIDATES: list[Candidate] = [ToneCandidate(), DateCandidate(), YoutubeCandidate()]


# --------------------------------------------------------------------------
# 분절 + 기억 수확 + 라우팅
# --------------------------------------------------------------------------
def split(ctx: Context) -> None:
    """스트림을 화제 단위로 끊는다. 모든 것보다 **먼저** 돈다.

    이후 단계가 보는 범위가 여기서 정해진다. 설계는 `docs/design.md` 1부.
    """
    result = segment(ctx.messages)
    ctx.segments = result.segments
    ctx.context = active_context(ctx.segments)
    ctx.trace.segments = result.segments
    ctx.trace.scores = result.scores


def read_state(ctx: Context) -> list[EmotionAnalysis]:
    """위젯 ①번 줄 — 실 상태 표현. **게이트가 없다. 매 요청 돈다.**

    후보 기능이 아니라서 `CANDIDATES` 에 넣지 않는다. `route()` 는 "발동한 것을 모으는"
    함수인데 이건 발동 여부가 없다. 결과도 `results` 가 아니라 `emotionAnalyses` 로 나간다
    (`docs/design.md` 2부 5장).

    **`ctx.context` 를 본다.** 활성 세그먼트만 보면 "쌓임 → 풀어짐" 추이를 볼 수 없다 —
    최소 두 시점이 필요하다. 말투 판정이 같은 이유로 쓰는 범위를 그대로 쓴다.
    전체 스트림을 쓰지 않는 이유는 분절 설계 그대로다. 세 시간 전에 끝난 다툼의 감정을
    지금 화면에 띄우면 늦은 게 아니라 **틀린 것**이다.
    """
    speakers = [to_speaker(p.participant_key) for p in ctx.request.participants]
    result = state.read_state(ctx.context, speakers, ctx.now)
    ctx.trace.state_scored = result.scored

    # 지금은 사전이 상수라 걸릴 일이 없다. 문구를 LLM 이 만들게 바뀌면 여기가 방어선이 된다.
    clean: list[EmotionAnalysis] = []
    for analysis in result.analyses:
        if (hit := banned_in_state(analysis)) is not None:
            ctx.trace.skip("state", f"금지어 필터 — '{hit}'")
            continue
        clean.append(analysis)

    ctx.trace.states = clean
    return clean


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
    """후보를 순서대로 돌린다. **`run()` 이 병렬로 도는 게 기본이고 이건 폴백이다.**

    동작이 같아야 하므로 `run()` 과 규칙을 공유한다 — 우선순위 순서, 유튜브 보류 조건,
    `trace.fired` 내용이 전부 동일하다. 병렬 실행을 끄고 원인을 좁힐 때 쓴다.
    """
    results: list[AiResult] = []
    for candidate in CANDIDATES:
        if (
            SUPPRESS_YOUTUBE_WHEN_TONE
            and candidate.name == "youtube"
            and any(r.result_type == "TONE_CORRECTION" for r in results)
            and demo_hit(ctx, "유튜브") is None
        ):
            ctx.trace.skip(candidate.name, "말투 교정이 발동한 요청 — 냉각기가 아니므로 보류")
            continue

        result = candidate.build(ctx)
        if result is not None:
            results.append(result)
            ctx.trace.fired.append(candidate.name)
    return results


# --------------------------------------------------------------------------
# 병렬 실행
# --------------------------------------------------------------------------
# 분절 이후 단계는 대부분 서로 독립인데 순차로 돌아서 시간이 그냥 더해지고 있었다.
# 실측 14.0초짜리 요청에서 LLM 에만 12.0초를 썼다.
#
#     분절 ─┬─ 상태 산출                                  (독립)
#           ├─ 기억 추출                                  (독립)
#           ├─ 데이트 계획 → 카카오 → 데이트 문구           (독립)
#           └─ 말투 판정 → 말투 생성 → 유튜브 (보류 판단)
#
# **의존이 둘 있고 둘 다 지킨다.**
#
# **의존은 하나뿐이다.** 말투 → 유튜브 — `SUPPRESS_YOUTUBE_WHEN_TONE` 판단에 말투 결과가
# 필요하다. 미리 돌려놓고 버리는 방법도 있지만 유튜브는 쿼터가 하루 95회라 버리는 호출을
# 만들지 않는다.
#
# **기억 추출 → 데이트 의존은 끊었다.** 오래 "방금 한 발화가 같은 요청에 반영되어야 한다"는
# 이유로 붙여뒀는데, 코드를 대조해 보니 **이미 만족하고 있었다** — `plan_date` 는
# `format_transcript(messages)` 로 대화 원문을 통째로 받는다. "마라탕 땡긴다"는 프롬프트
# 안에 그대로 있다. 기억 저장소를 거치는 것은 **과거 요청**에서 쌓인 기억을 찾기 위한
# 경로고, 이번 요청의 발화는 거기 없어도 된다.
#
# 끊고 나니 데이트 체인이 기억 추출(1~2.6초)만큼 앞당겨진다. 설계 가치는 그대로다.
#
# LLM·HTTP 대기가 전부라 스레드로 충분하다 (GIL 이 문제되지 않는다).
# 동시에 던지는 작업 수. **5개까지 나갈 수 있다** — 상태 · 기억 추출 · 데이트 · 말투 ·
# 유튜브. 4로 두면 유튜브가 슬롯을 기다리느라 병렬이 반쯤 무너진다 (실측에서 확인).
MAX_WORKERS = 6


def _build(candidate: Candidate, ctx: Context) -> AiResult | None:
    return candidate.build(ctx)


def run(ctx: Context) -> tuple[list[EmotionAnalysis], list[AiResult]]:
    """분절 이후 전체를 돌린다. 독립인 것은 동시에.

    **결과 순서는 `CANDIDATES` 우선순위 그대로다.** 완료 순서로 담으면 요청마다 배열
    순서가 바뀌고, 프론트가 `results[0]` 을 쓰기로 하면 화면이 달라진다.
    """
    tone, date, youtube = CANDIDATES

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        f_state = pool.submit(read_state, ctx)
        f_tone = pool.submit(_build, tone, ctx)
        f_memory = pool.submit(harvest_memories, ctx)
        f_date = pool.submit(_build, date, ctx)

        # 유튜브는 말투 결과를 알아야 한다 (의존 ②)
        tone_result = f_tone.result()
        demo_yt = demo_hit(ctx, "유튜브")
        if SUPPRESS_YOUTUBE_WHEN_TONE and tone_result is not None and demo_yt is None:
            ctx.trace.skip(youtube.name, "말투 교정이 발동한 요청 — 냉각기가 아니므로 보류")
            f_youtube = None
        else:
            f_youtube = pool.submit(_build, youtube, ctx)

        by_name = {
            tone.name: tone_result,
            date.name: f_date.result(),
            youtube.name: f_youtube.result() if f_youtube is not None else None,
        }
        states = f_state.result()

    # 약한 데이트 신호(`category` 단독)로 화제 갈래가 돌았는데 데이트가 실제로 발동했다면
    # 화제 갈래는 뺀다. **"데이트 코스가 떠야 할 때는 영상을 띄우지 않는다"** 가 규칙이다.
    # 여기서 빼는 것은 지연을 늘리지 않는다 — 둘 다 이미 병렬로 끝나 있다.
    # (고민 갈래는 빼지 않는다. 갈등과 데이트는 성격이 달라 같이 떠도 어색하지 않다.)
    if by_name[date.name] is not None and ctx.trace.topic is not None and demo_yt is None:
        if by_name[youtube.name] is not None:
            ctx.trace.skip(youtube.name, "데이트 코스가 발동한 요청 — 화제 갈래는 자리를 넘김")
        by_name[youtube.name] = None

    results: list[AiResult] = []
    for candidate in CANDIDATES:
        result = by_name[candidate.name]
        if result is not None:
            results.append(result)
            ctx.trace.fired.append(candidate.name)
    return states, results
