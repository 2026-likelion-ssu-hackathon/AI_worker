"""대화 분절 — 스트림을 화제 단위로 끊는다.

설계와 실측 근거는 `docs/design.md` 1부. 구조만 옮기면:

    ① 룰 컷     3시간 이상 공백에서 자른다 (LLM 에게 묻지 않는다)
    ② LLM 채점   마지막 조각의 발화마다 "직전 맥락과 얼마나 이어지는가" (호출 1회)
    ③ 룰 판정    임계값으로 붙일지 자를지 결정한다

**LLM 은 경계를 정하지 않는다. 점수만 낸다.** 자르는 판단은 전부 `_should_cut()` 에 있다.
경계가 LLM 안에 있으면 과분절이 나와도 프롬프트를 다시 쓰는 것 말고 할 수 있는 게 없고,
그건 재현 가능한 조정이 아니다. 임계값으로 빼두면 숫자로 만질 수 있고 왜 잘렸는지가
트레이스에 남는다.

**경계 신호로 쓸 수 있는 룰은 시간 공백 하나뿐이다.** 화제 전환 표지어("그건 그렇고"),
어휘 겹침, 임베딩 거리를 전부 재봤고 셋 다 실측에서 무너졌다 (문서 5장).

    표지어    같은 화제인 case9 에 '근데' 가 있고, 진짜 경계엔 표지어가 없었다
    어휘 겹침  단일 화제 안에서도 겹침이 0 이다. case4 는 어휘가 같은데 다른 대화다
    임베딩    경계에서 오히려 거리가 낮았다. 분포가 통째로 겹친다

**룰 컷과 채점이 서로 다른 구멍을 메운다.** LLM 은 화제는 보는데 시간을 못 본다 —
`case4_routine`(사흘치, 공백 1432분)을 한 덩어리로 본다. 룰 컷이 그걸 메운다.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta

from worker.llm import ask, load_prompt
from worker.models import Message, Segment, SegmentLLMOutput, SegmentScore

__all__ = ["SegmentResult", "segment", "active_context"]

# ① 룰 컷 — 이 이상 침묵하면 채점하지 않고 자른다.
#
# ⚠️ **실측으로 정한 값이 아니다.** 단일 화제 안 최대 공백이 9분이고 경계가 140분 이상이라
# 그 사이 구간에 표본이 없다. 그래서 "틀렸을 때 복구 가능한 쪽"으로 넉넉하게 잡았다 —
# 룰 컷은 LLM 없이 확정이라 잘못 자르면 고칠 기회가 없지만, 관대하면 채점이 고친다.
GAP_HARD = timedelta(hours=3)

# ③ 룰 판정 임계값.
#
# ⚠️ **실측으로 정한 값이 아니다. 프롬프트 앵커에서 역산한 값이다** (문서 11장).
#
# 프롬프트가 100 / 80 / 50 / 20 / 0 다섯 개를 기준점으로 준다. 실측해 보니 모델이
# **거의 기준점 값만 쓴다** — case11 에서 100, 100, 100, 80, 80, 80, 80 이 나왔다.
# 그래서 임계값은 기준점 사이에 놓아야 의미가 있다.
#
#     100 같은 것에 대해 계속       → 무조건 유지
#      80 이어지는 이야기           → **회색.** 이어진다고 한 대화라는 뜻은 아니다
#      50 느슨하게 연결             → 회색
#      20 다른 이야기               → 무조건 자름
#
# 처음에 KEEP_SOFT 를 80 으로 뒀더니 case11 의 진짜 경계(2시간 20분 뒤 다툼 시작)가
# 80 을 받아 통째로 안 잘렸다. "이어지는 이야기"는 유지 근거가 아니라 회색이다.
CUT_HARD = 35    # 이 아래면 무조건 자른다 (앵커 20 을 잡는다)
KEEP_SOFT = 90   # 이 위면 무조건 붙인다 (앵커 100 만 잡는다)
TONE_CUT = 40    # 회색지대에서만 쓰는 보조 기준
GAP_SOFT = timedelta(minutes=30)  # 회색지대에서만 쓰는 보조 기준

# 조각이 이보다 짧으면 채점하지 않는다. 나눌 것이 없다.
MIN_FOR_LLM = 3

# 말투 판정에 최소한 확보해야 하는 메시지 수 (마지막 1개 + `tone.CONTEXT_TURNS` 3턴).
# 활성 세그먼트가 이보다 짧으면 앞 세그먼트에서 뒤에서부터 채운다.
CONTEXT_MIN = 4


@dataclass
class SegmentResult:
    """분절 결과. `scores` 는 트레이스용이고 판정에는 이미 반영돼 있다."""

    segments: list[Segment] = field(default_factory=list)
    scores: list[SegmentScore] = field(default_factory=list)


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
# ② LLM 채점
# --------------------------------------------------------------------------
def _lines(messages: list[Message]) -> str:
    """`[HH:MM]` 을 붙인다.

    안 붙이면 **LLM 이 시간 공백을 아예 못 본다.** 2시간 50분이 벌어져도 안 보이고,
    3시간 룰 컷 바로 아래 구간이 통째로 사각지대가 된다 (문서 3-7).
    다른 프롬프트는 `text.format_transcript()` 가 같은 일을 한다.
    """
    return "\n".join(
        f"[{m.message_id}] ({m.sent_at:%m-%d %H:%M}) {m.sender}: {m.content}"
        for m in messages
    )


def _transcript(context: list[Message], targets: list[Message]) -> str:
    """채점 대상과 앞 맥락을 구획으로 나눠 준다. 맥락이 없으면 기존 단일 형식 그대로다."""
    body = f"## 대화 ({len(targets)}개)\n{_lines(targets)}"
    if not context:
        return body
    return (
        f"## 앞 맥락 (채점하지 않는다)\n{_lines(context)}\n\n"
        f"## 채점 대상 ({len(targets)}개)\n{_lines(targets)}"
    )


# 한 호출이 채점하는 최대 발화 수. **출력이 발화 수에 비례해서 지연도 비례한다** —
# 생산 창(메시지 30개)을 한 호출로 채점하면 출력 ~470토큰에 7~8초다.
# 넘으면 배치로 나눠 **동시에** 부른다. 각 배치 호출은 자기 앞의 전체 맥락을 '앞 맥락'
# 구획으로 그대로 보므로 **판단 재료는 한 호출일 때와 같다** — 출력만 나뉜다.
SCORE_BATCH = 15


def _ask_scores(context: list[Message], targets: list[Message]) -> list[SegmentScore]:
    out = ask(SegmentLLMOutput, load_prompt("segment"), _transcript(context, targets))
    return out.scores


def _score(chunk: list[Message]) -> list[SegmentScore] | None:
    """발화별 연속성 점수. 호출 전부가 실패하면 None → 자르지 않는다.

    **점수가 빠지거나 엉뚱한 id 가 섞여도 통째로 버리지 않는다.** 아는 id 만 남기고
    나머지는 없는 대로 둔다 — 빠진 발화는 `_cut_by_score()` 에서 "안 자름"으로 처리된다.
    배치 호출 하나가 실패해도 같다 — 그 구간만 "안 자름"이 되고 나머지 경계는 산다.

    전량 대조로 하면 실패 반경이 너무 크다. 점수 하나가 어긋났다고 조각 전체를 세그먼트
    1개로 되돌리면 **길게 나눠 놨던 경계가 통째로 사라진다.** 대화가 길수록 어긋날 확률은
    올라가는데 잃는 것도 같이 커진다 — 가장 나쁜 조합이다.
    """
    to_score = chunk[1:]  # 첫 발화는 비교할 앞이 없다
    if not to_score:
        return None
    n_batches = -(-len(to_score) // SCORE_BATCH)
    size = -(-len(to_score) // n_batches)

    raw: list[SegmentScore] = []
    if n_batches == 1:
        try:
            raw = _ask_scores([], chunk)
        except Exception:  # noqa: BLE001 — 채점 실패는 오류가 아니라 '안 나눔'이다
            return None
    else:
        jobs = [
            (chunk[: 1 + i], to_score[i : i + size])
            for i in range(0, len(to_score), size)
        ]
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [pool.submit(_ask_scores, ctx, tgt) for ctx, tgt in jobs]
            for future in futures:
                try:
                    raw.extend(future.result())
                except Exception:  # noqa: BLE001 — 이 배치만 '안 자름'이 된다
                    continue

    known = {m.message_id for m in chunk[1:]}
    seen: set[int] = set()
    scores: list[SegmentScore] = []
    for s in raw:
        if s.id in known and s.id not in seen:
            seen.add(s.id)
            scores.append(s)
    return scores or None


# --------------------------------------------------------------------------
# ③ 룰 판정 — 임계값
# --------------------------------------------------------------------------
def _should_cut(score: SegmentScore, gap: timedelta) -> bool:
    """이 발화 앞에서 자를 것인가.

    회색지대(CUT_HARD ~ KEEP_SOFT)에서는 **붙이는 쪽이 기본값이다.** 잘못 자르면 뒤
    단계가 맥락을 잃지만, 안 자르면 분절 전과 같아질 뿐이다 — 되돌릴 수 있는 실수를 택한다.
    """
    if score.topic >= KEEP_SOFT:
        return False
    if score.topic < CUT_HARD:
        return True

    # 회색지대 — 보조 신호로만 결정한다.
    #
    # ⚠️ `tone_score` 는 **단독으로 자르지 않는다.** case7_tone 은 호칭이 "오빠 → 야"로
    # 바뀌지만 처음부터 끝까지 저녁 약속 얘기 하나다. 말투로 자르면 말투 판정에서 "왜
    # 화가 났는지"(야근으로 약속이 깨짐)가 다른 세그먼트로 넘어가고, 대체 문장이 근거
    # 없는 일반론이 된다 — 말투 교정이 자기 발밑을 판다 (문서 3-5).
    if gap >= GAP_SOFT:
        return True
    return score.tone < TONE_CUT and not score.same


def _cut_by_score(chunk: list[Message], scores: list[SegmentScore]) -> list[Segment]:
    """점수를 id 로 찾아 붙인다. **점수가 없는 발화는 자르지 않는다.**

    순서대로 zip 하지 않는 이유: 점수가 하나라도 빠지면 그 뒤가 전부 한 칸씩 밀려서
    엉뚱한 자리에서 잘린다. id 로 맞추면 빠진 것만 조용히 넘어간다.
    """
    by_id = {s.id: s for s in scores}
    segments: list[Segment] = []
    current: list[Message] = [chunk[0]]

    for message in chunk[1:]:
        score = by_id.get(message.message_id)
        gap = message.sent_at - current[-1].sent_at
        if score is not None and _should_cut(score, gap):
            segments.append(Segment(messages=current, by_rule=False))
            current = [message]
        else:
            current.append(message)

    segments.append(Segment(messages=current, by_rule=False))
    return segments


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------
def _whole(messages: list[Message]) -> list[Segment]:
    """조각 전체를 세그먼트 1개로. 폴백이자 '나눌 게 없음' 경로다."""
    return [Segment(messages=messages, by_rule=True)] if messages else []


def segment(messages: list[Message]) -> SegmentResult:
    """스트림을 세그먼트 목록으로. 마지막 원소가 **활성 세그먼트**다."""
    if not messages:
        return SegmentResult()

    ordered = sorted(messages, key=lambda m: (m.sent_at, m.message_id))
    chunks = _rule_cut(ordered)

    segments: list[Segment] = []
    for chunk in chunks[:-1]:
        segments.extend(_whole(chunk))

    last = chunks[-1]
    if len(last) < MIN_FOR_LLM:
        return SegmentResult(segments=segments + _whole(last))

    scores = _score(last)
    if scores is None:
        # 폴백 = 점수 전부 100 = 안 자름 = 분절 전 동작. 더 나빠지지 않는다.
        return SegmentResult(segments=segments + _whole(last))

    return SegmentResult(segments=segments + _cut_by_score(last, scores), scores=scores)


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
