"""화면 글자 수 한도 — **디자인 실측값이다.**

넘으면 화면에서 잘린다. 잘린 문장은 읽히지 않는 게 아니라 **뜻이 바뀐다** —
"'맨날'은 그동안 전부로 들릴 수 있어…" 처럼 근거가 사라진 채로 남는다.

띄어쓰기를 포함한 `len()` 기준이다 (디자인이 그렇게 쟀다).

## 프롬프트에 적는 것만으로는 안 지켜진다

실측에서 넘는 출력이 실제로 나왔다 — 데이트 근거 52자(지시 60자였는데도 화면 한도 초과),
유튜브 근거 54자, 말투 근거 55자. 그래서 파이썬 쪽에서 한 번 더 잡는다.
금지어 필터와 같은 구조다: **검사 → 1회 재생성 → 그래도 넘으면 절단.**

미발동으로 떨어뜨리지 않는다. 글자가 조금 긴 것 때문에 데이트 추천이 통째로 사라지는 건
과한 대응이다 — 금지어는 절대 제약이라 버리지만 길이는 그렇지 않다.

## 목표와 상한이 다른 자리가 있다

`alternative_sentence` 는 화면이 73자까지 받지만 **프롬프트 목표는 45자**로 둔다.
복사 버튼이 없어서 사용자가 보고 외워서 직접 타이핑해야 하기 때문이다 (`CLAUDE.md`).
여기 있는 값은 "화면이 허용하는 최대"고, "얼마나 짧게 쓸지"는 프롬프트가 정한다.
"""

from __future__ import annotations

from worker.models import (
    AiResult,
    DateCourseResultData,
    ToneResultData,
    YoutubeResultData,
)

__all__ = ["MAX", "over_limit", "enforce", "shorten"]

# 디자인 실측 (2026-08-17). 화면에 나가는 것 중 한도를 받은 것만 있다 —
# **워커 출력이 전부 화면에 들어가는 게 아니다.** 받지 않은 필드는 여기 넣지 않는다.
MAX = {
    # 갈등 중재 (감정 상태 조정)
    "situation_diagnosis": 35,    # 디자인 미지정 — 기존 워커 기준 유지
    "alternative_sentence": 73,   # 화면 상한. 프롬프트 목표는 45자 (위 설명)
    "correction_reason": 52,
    # 데이트 코스 — 한 줄 27자 × 두 줄
    "date_recommendation_reason": 54,
    # 유튜브
    "youtube_recommendation_reason": 50,
}

# 자를 때 뒤에 붙이는 글자. 이것도 한도 안에 들어가야 한다.
ELLIPSIS = "…"


def shorten(text: str, limit: int) -> str:
    """한도 안으로 줄인다. **어절 경계에서 자른다.**

    글자 중간에서 자르면 "공격적으로 들릴 수 있어" 가 "공격적으로 들릴 수 있" 이 된다.
    어절 경계가 너무 앞이면(한도의 절반 이전) 그냥 글자 수로 자른다 — 한 어절이
    통째로 긴 경우다.
    """
    if len(text) <= limit:
        return text

    body = text[: limit - len(ELLIPSIS)]
    cut = body.rfind(" ")
    if cut >= limit // 2:
        body = body[:cut]
    return body.rstrip(" ,.·、") + ELLIPSIS


def _fields(result: AiResult) -> list[tuple[str, str]]:
    """(한도 키, 문자열) 목록. 한도를 받은 필드만 돌려준다."""
    data = result.result_data
    if isinstance(data, ToneResultData):
        return [
            ("situation_diagnosis", data.situation_diagnosis),
            ("alternative_sentence", data.alternative_sentence),
            ("correction_reason", data.correction_reason),
        ]
    if isinstance(data, DateCourseResultData):
        return [("date_recommendation_reason", data.recommendation_reason)]
    if isinstance(data, YoutubeResultData):
        # title 은 유튜브가 준 값이라 워커가 줄이지 않는다. 프론트에서 말줄임 처리한다.
        return [("youtube_recommendation_reason", data.recommendation_reason)]
    return []


def over_limit(result: AiResult) -> tuple[str, int, int] | None:
    """한도를 넘은 첫 필드. `(필드, 실제 길이, 한도)`. 없으면 None."""
    for key, text in _fields(result):
        limit = MAX[key]
        if len(text) > limit:
            return key, len(text), limit
    return None


def enforce(result: AiResult) -> AiResult:
    """한도를 넘은 필드를 잘라서 돌려준다. **재생성이 실패했을 때의 마지막 수단이다.**

    원본을 건드리지 않고 사본을 만든다 — 트레이스에 남은 값이 조용히 바뀌면
    "왜 잘렸지"를 되짚을 수 없다.
    """
    if over_limit(result) is None:
        return result

    data = result.result_data
    patch: dict[str, str] = {}
    if isinstance(data, ToneResultData):
        patch = {
            "situation_diagnosis": shorten(data.situation_diagnosis, MAX["situation_diagnosis"]),
            "alternative_sentence": shorten(data.alternative_sentence, MAX["alternative_sentence"]),
            "correction_reason": shorten(data.correction_reason, MAX["correction_reason"]),
        }
    elif isinstance(data, DateCourseResultData):
        patch = {
            "recommendation_reason": shorten(
                data.recommendation_reason, MAX["date_recommendation_reason"]
            )
        }
    elif isinstance(data, YoutubeResultData):
        patch = {
            "recommendation_reason": shorten(
                data.recommendation_reason, MAX["youtube_recommendation_reason"]
            )
        }

    return result.model_copy(update={"result_data": data.model_copy(update=patch)})
