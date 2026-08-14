"""금지어 하드 필터.

프롬프트에도 "관계 상태 언급 금지"를 넣지만, 그것만 믿지 않는다.
LLM 판단에 맡기지 않고 **문자열 검사로 강제**한다. 이 프로젝트의 절대 제약이기 때문이다.
"""

from __future__ import annotations

import re
from typing import Callable

from worker.models import Decision

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
# 갈등 중재(kind="tone")에서 **감정 상태를 언급하는 것은 금지가 아니다.**
# 절대 제약의 근거는 "너네 권태기다" 같은 **관계 규정**을 들으면 의식이 심해진다는 것이고,
# 그건 대화 소재 쪽 얘기다. 갈등 중재는 지금 감정이 올라온 걸 짚어주는 게 기능의 목적이다.
# "지금 감정이 올라와 있어요" 는 통과해야 하고, "요즘 두 분 사이가" 는 여기서도 막힌다.
#
# 사람을 평가하는 표현("무례하시네요")은 프롬프트에서 다룬다. 하드 필터로 잡으면
# "공격적으로 들릴 수 있어요"(정상)와 "공격적이시네요"(문제)를 구분하지 못한다.

_BANNED_RE = re.compile("|".join(re.escape(w) for w in BANNED))


def find_banned(text: str | None, kind: str = "topic") -> str | None:
    """걸린 금지어를 돌려준다. 없으면 None.

    `kind` 는 지금 판정에 쓰이지 않는다. 후보별로 금지어가 갈릴 때를 위해 자리만 열어둔다.
    """
    if not text:
        return None
    hit = _BANNED_RE.search(text)
    return hit.group(0) if hit else None


def is_clean(decision: Decision) -> bool:
    kind = decision.kind
    return (
        find_banned(decision.content, kind) is None
        and find_banned(decision.reason, kind) is None
    )


def apply_filter(
    decision: Decision,
    regenerate: Callable[[], Decision] | None = None,
    fallback: Callable[[], Decision] | None = None,
) -> Decision:
    """걸리면 1회 재생성, 또 걸리면 오늘의 질문으로 폴백한다."""
    if is_clean(decision):
        return decision

    if regenerate is not None:
        retry = regenerate()
        if is_clean(retry):
            return retry

    if fallback is not None:
        safe = fallback()
        if is_clean(safe):
            return safe
        # 폴백까지 걸리면 내보내지 않는다 — 절대 제약이 우선이다
        raise RuntimeError(f"금지어 필터 폴백 실패: {safe.content!r}")

    raise RuntimeError(f"금지어 감지, 대체 수단 없음: {decision.content!r}")
