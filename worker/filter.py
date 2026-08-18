"""금지어 하드 필터.

프롬프트에도 "관계 상태 언급 금지"를 넣지만, 그것만 믿지 않는다.
LLM 판단에 맡기지 않고 **문자열 검사로 강제**한다. 이 프로젝트의 절대 제약이기 때문이다.

검사 대상은 **사용자 화면에 실제로 나가는 문자열 전부**다. 위젯 ②번 줄(`AiResult`)과
①번 줄(`EmotionAnalysis`)을 둘 다 본다. 우리가 생성한 문구뿐 아니라
유튜브 영상 제목·설명처럼 **외부에서 가져온 문자열도 포함한다.** "권태기 극복법"이라는
영상 제목이 위젯에 뜨면, 그 문장을 우리가 쓴 게 아니어도 사용자는 화면에서 '권태기'를
읽는다. 절대 제약의 근거는 "그 말을 들으면 의식이 심해진다"이지 "우리가 쓰면 안 된다"가
아니다.
"""

from __future__ import annotations

import re

from worker.models import (
    AiResult,
    DateCourseResultData,
    EmotionAnalysis,
    ToneResultData,
    YoutubeResultData,
)

# docs/worker-tasks.md 5단계 목록 + 같은 뉘앙스의 변형
BANNED = [
    "권태기",
    "대화가 줄",
    "대화가 뜸",
    "서먹",
    "사이가",
    "요즘 뜸",
    "소원해",
    "멀어지",
    "멀어졌",
    "데면데면",
    "어색해지",
    "소홀",
    "관계가",
    "예전보다",
    "요즘 들어",
    "말수가 줄",
    "대화가 없",
    "무뚝뚝해",
]

# 금지어 세트는 후보 기능과 무관하게 하나다. 절대 제약이 "관계 상태 언급 금지" 하나이기 때문이다.
#
# 갈등 중재(TONE_CORRECTION)에서 **감정 상태를 언급하는 것은 금지가 아니다.**
# 절대 제약의 근거는 "너네 권태기다" 같은 **관계 규정**을 들으면 의식이 심해진다는 것이다.
# 갈등 중재는 지금 감정이 올라온 걸 짚어주는 게 기능의 목적이다.
# "지금 감정이 올라와 있어요" 는 통과해야 하고, "요즘 두 분 사이가" 는 여기서도 막힌다.
#
# 사람을 평가하는 표현("무례하시네요")은 프롬프트에서 다룬다. 하드 필터로 잡으면
# "공격적으로 들릴 수 있어요"(정상)와 "공격적이시네요"(문제)를 구분하지 못한다.

_BANNED_RE = re.compile("|".join(re.escape(w) for w in BANNED))


def find_banned(text: str | None) -> str | None:
    """걸린 금지어를 돌려준다. 없으면 None."""
    if not text:
        return None
    hit = _BANNED_RE.search(text)
    return hit.group(0) if hit else None


def visible_texts(result: AiResult) -> list[str]:
    """사용자 화면에 나가는 문자열 전부."""
    data = result.result_data
    if isinstance(data, ToneResultData):
        return [
            data.situation_diagnosis,
            data.guide_message,
            data.alternative_sentence,
            data.correction_reason,
        ]
    if isinstance(data, DateCourseResultData):
        return [
            data.guide_message,
            data.course_name,
            data.course_summary,
            data.recommendation_reason,
            data.main_place.name,
            data.main_place.summary,
            *(p.name for p in data.course_places),
            *(p.summary for p in data.course_places),
        ]
    if isinstance(data, YoutubeResultData):
        # title / video_summary 는 유튜브가 준 값이다. 우리가 안 썼어도 화면에는 뜬다.
        return [
            data.guide_message,
            data.title,
            data.recommendation_reason,
            data.video_summary or "",
        ]
    return []


def banned_in(result: AiResult) -> str | None:
    for text in visible_texts(result):
        hit = find_banned(text)
        if hit is not None:
            return hit
    return None


def is_clean(result: AiResult) -> bool:
    return banned_in(result) is None


# --------------------------------------------------------------------------
# 실 상태 표현 — 위젯 ①번 줄
# --------------------------------------------------------------------------
# `visible_texts()` 는 `AiResult` 만 본다. 상태 문구는 `emotionAnalyses` 에 실려 나가므로
# **그 경로로는 검사되지 않는다.** 세그먼테이션에서 화제 라벨을 만들지 않기로 한 것과
# 같은 함정이다 (`docs/design.md` 2부 3-4).
#
# 지금 문구는 `copy.STATE_TEXT` 의 상수 5개고 임포트 시점에 이미 검사된다. 여기 걸릴
# 일이 없다는 뜻인데, 그래도 둔다 — 나중에 문구를 LLM 이 만들게 바꾸는 순간 이 함수가
# 유일한 방어선이 된다.
def banned_in_state(state: EmotionAnalysis) -> str | None:
    return find_banned(state.state_text)


def is_clean_state(state: EmotionAnalysis) -> bool:
    return banned_in_state(state) is None
