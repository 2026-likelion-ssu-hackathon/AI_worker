"""대화 분절 — 스트림을 화제 단위로 끊는다.

설계와 실측 근거는 `docs/segmentation-v3.md` 에 있다. 요지만 옮기면:

    ① 룰 컷    3시간 이상 공백에서 자른다 (LLM 에게 묻지 않는다)
    ② LLM 분절  마지막 조각만 화제로 나눈다 (요청당 1회)

**경계 신호로 쓸 수 있는 룰은 시간 공백 하나뿐이다.** 화제 전환 표지어("그건 그렇고"),
어휘 겹침, 임베딩 거리를 전부 재봤고 셋 다 실측에서 무너졌다 (문서 3-1).

    표지어    같은 화제인 case9 에 '근데' 가 있고, 진짜 경계엔 표지어가 없었다
    어휘 겹침  단일 화제 안에서도 겹침이 0 이다. case4 는 어휘가 같은데 다른 대화다
    임베딩    경계에서 오히려 거리가 낮았다. 분포가 통째로 겹친다

**LLM 과 룰이 서로 다른 구멍을 메운다.** LLM 은 화제는 보는데 시간을 못 본다 —
`case4_routine`(사흘치, 공백 1432분)을 3회 모두 세그먼트 1개로 봤다. 화제가 "일상 대화"로
같으니 LLM 입장에선 틀린 판단이 아니다. 룰 컷이 정확히 이 구멍을 메운다.

**LLM 을 마지막 조각에만 거는 이유**: 라우팅도 기억 추출도 활성 세그먼트만 쓴다.
앞 조각을 세밀하게 나눌 이유가 없고, 이 설계라야 LLM 호출이 요청당 1회로 고정된다.
"""

from __future__ import annotations

from datetime import timedelta

from worker.llm import ask, load_prompt
from worker.models import Message, Segment, SegmentLLMOutput

__all__ = ["segment", "active_context"]

# 이 이상 침묵하면 LLM 에게 묻지 않고 자른다.
#
# ⚠️ **실측으로 정한 값이 아니다.** 단일 화제 안 최대 공백이 9분이고 경계가 140분 이상이라
# 그 사이 구간에 표본이 없다. 그래서 "틀렸을 때 복구 가능한 쪽"으로 넉넉하게 잡았다 —
# 룰 컷은 LLM 없이 확정이라 잘못 자르면 고칠 기회가 없지만, 관대하면 LLM 이 고친다.
GAP_HARD = timedelta(hours=3)

# 조각이 이보다 짧으면 LLM 을 부르지 않는다. 나눌 것이 없다.
MIN_FOR_LLM = 3

# 말투 판정에 최소한 확보해야 하는 메시지 수 (마지막 1개 + `tone.CONTEXT_TURNS` 3턴).
# 활성 세그먼트가 이보다 짧으면 앞 세그먼트에서 뒤에서부터 채운다.
CONTEXT_MIN = 4


# --------------------------------------------------------------------------
# ① 룰 컷 — 시간 공백
# --------------------------------------------------------------------------
def _rule_cut(messages: list[Message]) -> list[list[Message]]:
    """3시간 이상 벌어진 지점에서 자른다. 입력은 시간순으로 정렬돼 있다고 본다."""
    chunks: list[list[Message]] = []
    current: list[Message] = []
    for m in messages:
        if current and m.sent_at - current[-1].sent_at >= GAP_HARD:
            chunks.append(current)
            current = []
        current.append(m)
    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------
# ② LLM 분절 — 화제
# --------------------------------------------------------------------------
def _transcript(messages: list[Message]) -> str:
    lines = [f"[{m.message_id}] {m.sender}: {m.content}" for m in messages]
    return f"## 대화 ({len(messages)}개)\n" + "\n".join(lines)


def _whole(messages: list[Message], by_rule: bool = True) -> list[Segment]:
    """조각 전체를 세그먼트 1개로. 폴백 경로이자 '나눌 게 없음' 경로다."""
    return [Segment(messages=messages, by_rule=by_rule)] if messages else []


def _llm_split(chunk: list[Message]) -> list[Segment]:
    if len(chunk) < MIN_FOR_LLM:
        return _whole(chunk)

    try:
        out = ask(SegmentLLMOutput, load_prompt("segment"), _transcript(chunk))
    except Exception:  # noqa: BLE001 — 분절 실패는 오류가 아니라 '안 나눔'이다
        return _whole(chunk)

    by_id = {m.message_id: m for m in chunk}
    segments: list[Segment] = []
    seen: list[int] = []

    for span in out.segments:
        picked = [by_id[i] for i in span.message_ids if i in by_id]
        if not picked:
            continue
        seen.extend(m.message_id for m in picked)
        segments.append(
            Segment(
                messages=picked,
                topic=span.topic.strip(),
                mood=span.mood,
                by_rule=False,
            )
        )

    # 검증 — 모든 메시지가 정확히 한 번씩, 순서대로 들어갔는가.
    # 하나라도 어긋나면 통째로 폴백한다. 반쯤 맞은 경계는 안 나눈 것보다 나쁘다.
    if seen != [m.message_id for m in chunk]:
        return _whole(chunk)

    return segments


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------
def segment(messages: list[Message]) -> list[Segment]:
    """스트림을 세그먼트 목록으로. 마지막 원소가 **활성 세그먼트**다."""
    if not messages:
        return []

    ordered = sorted(messages, key=lambda m: (m.sent_at, m.message_id))
    chunks = _rule_cut(ordered)

    segments: list[Segment] = []
    for chunk in chunks[:-1]:
        segments.extend(_whole(chunk))
    segments.extend(_llm_split(chunks[-1]))
    return segments


def active_context(segments: list[Segment]) -> list[Message]:
    """말투 판정 프롬프트에 넣을 맥락.

    **게이트·트리거·RAG 는 활성 세그먼트만 쓴다.** 이건 말투 판정 전용이다 —
    활성 세그먼트가 1개짜리면 "와 미친 ㅋㅋ" 가 장난인지 갈등인지 구분할 수가 없다.
    부족한 만큼만 앞 세그먼트에서 뒤에서부터 끌어온다.
    """
    if not segments:
        return []

    messages = list(segments[-1].messages)
    for seg in reversed(segments[:-1]):
        if len(messages) >= CONTEXT_MIN:
            break
        need = CONTEXT_MIN - len(messages)
        messages = seg.messages[-need:] + messages
    return messages
