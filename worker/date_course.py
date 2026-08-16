"""후보 기능 — 데이트 코스 추천.

대화에서 데이트 계획 의도가 감지되면, **두 사람이 과거에 남긴 발화**(먹고 싶다 · 가보고
싶다 · 저장한 링크)를 최우선 근거로 장소를 구성해 카드로 제시한다.

    [룰 트리거] → [기억 검색] → [LLM 계획] → [카카오 로컬 검색] → [LLM 이유 문구]

**LLM 과 카카오의 역할이 엄격히 갈린다.**

    LLM  → 무엇을 찾을지 (지역, 검색어, 코스 의도, 추천 이유)
    카카오 → 실제로 무엇이 있는지 (상호명, 카테고리, place_url)

LLM 이 상호명을 만들면 존재하지 않는 가게가 나온다. 규격서 9장의 `externalUrl` 은 필수
필드라 환각이 그대로 죽은 링크로 나간다. 그래서 장소 이름은 **카카오가 준 것만** 쓴다.

명세의 적합도 스코어링 5가지 중 구현한 것은 ①과거 기억 ②현재 스트림 ⑤취향 셋이다.
③영업시간은 카카오 로컬 API 가 주지 않고(크롤링 영역), ④날씨·예산도 데이터 소스가 없다.
"""

from __future__ import annotations

import re

from worker.copy import DATE_GUIDE
from worker.llm import ask, load_prompt
from worker.models import (
    AiResult,
    DateCourseResultData,
    DateGateResult,
    DateIntentKind,
    DatePlanLLMOutput,
    DateReasonLLMOutput,
    Memory,
    Message,
    Place,
    to_key,
)
from worker.places import KakaoPlace, search_places
from worker.text import format_transcript

# 코스에 넣을 장소 수
MIN_PLACES = 2
MAX_PLACES = 4

# 기억을 몇 건까지 근거로 넘길지
MEMORY_K = 6

# 데이트 코스 근거가 되는 기억 종류. 갈등 관련은 애초에 저장하지 않지만 명시해 둔다.
MEMORY_KINDS = ("place", "wish", "activity", "promise", "interest", "schedule")


# --------------------------------------------------------------------------
# 룰 트리거 — 명세 [데이트 의도 감지] 4종
# --------------------------------------------------------------------------
# ① 명시적 계획 질문
_PLAN_RE = re.compile(
    r"(어디\s*(갈까|가지|가면|서\s*볼까|서\s*만날)|뭐\s*(먹지|먹을까|하지|할까|해먹)|"
    r"데이트\s*(어디|뭐|코스)|오늘\s*뭐\s*(하|할)|주말에\s*뭐|만나서\s*뭐|"
    r"뭐\s*하고\s*싶|어디\s*좋을까|계획\s*(짤|세울|있어))"
)

# ② 일정 확정 발화
_SCHEDULE_RE = re.compile(
    r"((월|화|수|목|금|토|일)요일에?\s*(보자|만나|볼까|봐)|"
    r"(내일|모레|이번\s*주|다음\s*주|주말|담주)에?\s*(보자|만나|볼까|봐|시간)|"
    r"몇\s*시에?\s*(봐|보자|만나)|시간\s*(돼|되|있어)\?*|"
    r"\d{1,2}일에?\s*(보자|만나|볼까|어때))"
)

# ③ 장소·음식 카테고리 언급
_CATEGORY_RE = re.compile(
    r"(배고파|배고프|맛집|카페|전시|영화|산책|드라이브|공원|방탈출|팝업|브런치|"
    r"[가-힣]+\s*(먹고\s*싶|마시고\s*싶|가보고\s*싶|해보고\s*싶|가고\s*싶)|"
    r"먹고\s*싶|가보고\s*싶|해보고\s*싶|땡긴다|땡겨)"
)

# ④ 이전 대화 기억의 재언급
_RECALL_RE = re.compile(
    r"(저번에\s*그|지난번에?\s*그|우리\s*가보자던|가보자고\s*했던|"
    r"네가\s*말한\s*그|니가\s*말한\s*그|먹고\s*싶다고\s*했던|가고\s*싶다고\s*했던|"
    r"그때\s*그|말했던\s*(그|데|곳))"
)

_PATTERNS: list[tuple[DateIntentKind, re.Pattern[str], str]] = [
    ("plan_question", _PLAN_RE, "명시적 계획 질문"),
    ("schedule_fixed", _SCHEDULE_RE, "일정 확정 발화"),
    ("category", _CATEGORY_RE, "장소·음식 카테고리 언급"),
    ("recall", _RECALL_RE, "이전 대화 기억의 재언급"),
]


def check_date_gate(messages: list[Message]) -> DateGateResult:
    """데이트 의도가 있는가. 확정하지 않고 후보만 잡는다 — 확정은 LLM 이 한다.

    **활성 세그먼트를 받는다.** 예전에는 최근 12개를 스스로 잘랐는데, 그 12개 안에 몇 개의
    화제가 들어 있는지 알 수 없어서 두 시간 전에 끝난 데이트 얘기가 지금 싸우는 요청에서
    코스를 발동시켰다 (`docs/segmentation-v3.md` 1장). 범위는 이제 분절이 정한다.
    """
    if not messages:
        return DateGateResult(triggered=False)

    recent = sorted(messages, key=lambda m: m.sent_at)
    kinds: list[DateIntentKind] = []
    details: list[str] = []
    hit_ids: list[int] = []

    for kind, pattern, label in _PATTERNS:
        for m in recent:
            hit = pattern.search(m.content)
            if hit is None:
                continue
            if kind not in kinds:
                kinds.append(kind)
                details.append(f"[{kind}] {label} — '{hit.group(0).strip()}'")
            if m.message_id not in hit_ids:
                hit_ids.append(m.message_id)
            break

    return DateGateResult(
        triggered=bool(kinds),
        kinds=kinds,
        detail=" / ".join(details) or None,
        message_ids=sorted(hit_ids),
    )


# --------------------------------------------------------------------------
# LLM — 무엇을 찾을지 정한다
# --------------------------------------------------------------------------
def _memory_block(memories: list[Memory]) -> str:
    if not memories:
        return "(없음 — 기억이 비어 있다)"
    lines = []
    for m in memories:
        when = m.occurred_at.strftime("%Y-%m-%d") if m.occurred_at else "시점 미상"
        lines.append(f'- [{m.kind}] {m.content} ({when}) ← "{m.source_quote}"')
    return "\n".join(lines)


def plan_date(
    messages: list[Message], memories: list[Memory], gate: DateGateResult
) -> DatePlanLLMOutput:
    body = "\n".join(
        [
            "## 최근 대화",
            format_transcript(messages),
            "",
            "## 두 사람이 과거에 남긴 발화 (기억 저장소)",
            _memory_block(memories),
            "",
            "## 룰이 잡은 데이트 의도 신호",
            gate.detail or "(없음)",
        ]
    )
    return ask(DatePlanLLMOutput, load_prompt("date_plan"), body)


# --------------------------------------------------------------------------
# 카카오 로컬 — 실재하는 장소로 코스를 채운다
# --------------------------------------------------------------------------
_QUERY_TOKEN_RE = re.compile(r"[가-힣A-Za-z]+")


def _intent_tokens(query: str) -> list[str]:
    """검색어에서 업종 대조에 쓸 조각들.

    한국어 합성어는 뒤쪽이 업종이다 — `평양냉면`의 업종은 `냉면`, `크림파스타`는 `파스타`.
    카카오 카테고리는 `음식점 > 한식 > 냉면` 이라 뒷조각으로 대조해야 걸린다.
    """
    tokens: list[str] = []
    for word in _QUERY_TOKEN_RE.findall(query):
        if len(word) < 2:
            continue
        tokens.append(word)
        # 합성어 뒤쪽 2~3글자를 업종 후보로 함께 본다
        for size in (3, 2):
            if len(word) > size:
                tokens.append(word[-size:])
    return tokens


def _fits_intent(place: KakaoPlace, query: str) -> bool:
    """검색 의도와 업종이 맞는가.

    카카오는 유사 매칭이 후해서 `평양냉면` 으로 검색해도 고기집이 1위로 온다.
    그대로 쓰면 "먹어보고 싶다던 평양냉면집" 이라는 **근거가 통째로 날아간다** —
    추천 이유가 이 기능의 전부인데 그게 무너진다.
    """
    haystack = f"{place.name} {place.category_name}"
    return any(t in haystack for t in _intent_tokens(query))


def build_course(queries: list[str], region: str | None) -> list[KakaoPlace]:
    """검색어 순서대로 장소를 하나씩 확정한다. 같은 장소가 겹치지 않게 한다.

    검색어 의도에 맞는 결과를 우선하고, 하나도 없으면 1위 결과로 물러선다
    (`카페` 처럼 업종명이 상호·카테고리에 안 뜨는 검색어가 있다).

    **지역명을 못 잡았을 때는 첫 장소를 코스의 중심으로 삼는다.** 안 그러면 검색어마다
    전국에서 독립적으로 고르게 되고, 실측에서 `영화관 / 필름 현상 / 카페` 가
    **용산 → 미상 → 남양주 북한강**으로 흩어졌다. 차로 한 시간 넘는 곳들이라 코스가 아니다.
    `search_places` 의 반경 방어는 지역명이 있을 때만 걸리므로, 없을 때는 여기서 건다.

    지역명이 있으면 기존대로 지역 중심 반경을 쓴다 — 그쪽은 이미 의도대로 동작한다.
    """
    course: list[KakaoPlace] = []
    taken: set[str] = set()
    anchor: tuple[str, str] | None = None  # 지역명이 없을 때만 쓴다

    for query in queries[:MAX_PLACES]:
        found = [
            p for p in search_places(query, region=region, center=anchor)
            if p.name not in taken
        ]
        if not found:
            continue
        picked = next((p for p in found if _fits_intent(p, query)), found[0])
        course.append(picked)
        taken.add(picked.name)

        # 첫 장소가 잡히면 그 좌표를 중심으로 고정한다. 지역명이 있으면 이미
        # 지역 중심 반경이 걸려 있으므로 손대지 않는다.
        if region is None and anchor is None and picked.x and picked.y:
            anchor = (picked.x, picked.y)

    return course


def write_reason(
    messages: list[Message],
    memories: list[Memory],
    plan: DatePlanLLMOutput,
    course: list[KakaoPlace],
) -> DateReasonLLMOutput:
    """확정된 장소를 보고 코스명·요약·추천 이유·장소 설명을 쓴다.

    계획 단계와 나눈 이유: 계획 시점에는 어떤 가게가 잡힐지 모른다. 카카오가 준 실제
    상호를 보고 나서 문구를 써야 "성수다락에서 브런치 먹고" 같은 문장이 나온다.
    """
    place_lines = "\n".join(
        f"{i}. {p.name} — {p.category_name} ({p.address})" for i, p in enumerate(course, 1)
    )
    body = "\n".join(
        [
            "## 최근 대화",
            format_transcript(messages),
            "",
            "## 근거로 삼은 기억",
            _memory_block(memories),
            "",
            "## 계획 단계에서 정한 방향",
            f"- 코스명(임시): {plan.course_name}",
            f"- 요약(임시): {plan.course_summary}",
            f"- 근거: {plan.reason_seed}",
            "",
            "## 확정된 장소 (카카오 로컬 검색 결과 — 이름을 바꾸지 말 것)",
            place_lines,
        ]
    )
    return ask(DateReasonLLMOutput, load_prompt("date_reason"), body)


# --------------------------------------------------------------------------
# 조립
# --------------------------------------------------------------------------
def to_result(
    plan: DatePlanLLMOutput,
    reason: DateReasonLLMOutput,
    course: list[KakaoPlace],
    trigger_message_ids: list[int],
) -> AiResult:
    summaries = list(reason.place_summaries) + [""] * len(course)

    places = [
        Place(
            order=i,
            name=p.name,
            category=p.category,
            summary=(summaries[i - 1] or p.category_name).strip(),
            external_url=p.url,
        )
        for i, p in enumerate(course, 1)
    ]

    # mainPlace 는 코스의 첫 장소다. 규격서상 order 가 없어야 해서 따로 만든다.
    head = places[0]
    main = Place(
        name=head.name,
        category=head.category,
        summary=head.summary,
        external_url=head.external_url,
    )

    individual = plan.scope == "individual" and plan.target in ("A", "B")
    return AiResult(
        result_type="DATE_RECOMMENDATION",
        visibility_type="INDIVIDUAL" if individual else "COUPLE",
        target_participant=to_key(plan.target) if individual else None,  # type: ignore[arg-type]
        content_type="MIXED",
        trigger_message_ids=trigger_message_ids,
        result_data=DateCourseResultData(
            guide_message=DATE_GUIDE,
            course_name=reason.course_name.strip(),
            course_summary=reason.course_summary.strip(),
            main_place=main,
            course_places=places,
            recommendation_reason=reason.recommendation_reason.strip(),
        ),
    )
