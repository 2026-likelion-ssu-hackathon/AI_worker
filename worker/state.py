"""실 상태 표현 — 위젯 ①번 줄. **상시 작동한다.**

설계와 근거는 `docs/design.md` 2부. 구조만 옮기면:

    ① LLM 채점   화자마다 **감정 5축 점수**(0~5) (호출 1회). 라벨도 문구도 만들지 않는다
    ② 룰 판정    `pick_label()` 의 임계값이 점수에서 라벨을 정한다
    ③ 문구 선택   `copy.STATE_TEXT` 사전에서 라벨로 찾는다

**후보 기능이 아니다.** 게이트가 없고 `route()` 를 타지 않는다. 3종은 트리거가 걸릴 때만
발동하고 미발동이면 ②번 줄이 비지만, 이건 매 요청 돌고 ①번 줄은 비지 않는다.

**LLM 은 점수까지만 낸다. 라벨은 룰이, 문구는 사전이 정한다.** 분절과 같은 패턴이다.

라벨을 LLM 이 직접 고르게 했다가 바꿨다. 다투는 구간은 서운함과 분노가 같이 높은데
모델이 둘 중 하나를 임의로 골랐고, **점수가 없으니 왜 그쪽인지 알 수도 조정할 수도
없었다.** `case11_mixed` 에서 말투 교정은 공격 표현으로 잡은 발화를 상태 산출은 서운함으로
읽었다 — 한 발화를 두 기능이 다르게 읽는 것이라 맞춰야 했다. 지금은 `ANGER_WINS` 를
올리고 내리는 것으로 조정한다.

화면 문구를 생성시키지 않는 이유는 상시 노출이기 때문이다 — 위반 확률이 1000분의 1이어도
하루 수백 번 뜨는 자리면 반드시 나온다. 사전에서 고르면 위반 확률이 0 이고, 10자·어미
규격이 구조적으로 보장되며, 같은 대화에 같은 문구가 나온다.

**애매하면 중립이다.** 근거 없이 감정을 지어내면 없던 갈등을 만든다 — B 가 "ㅇㅇ" 한 마디
한 요청에서 "서운해 보여요"가 뜨면 A 는 있지도 않은 서운함에 반응하기 시작한다.
데이트 코스가 없는 발화를 인용하지 않는 것과 같은 이유다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from worker.copy import STATE_TEXT
from worker.llm import ask, load_prompt
from worker.models import (
    EmotionAnalysis,
    EmotionScores,
    Message,
    Speaker,
    StateLabel,
    StateLLMOutput,
    to_key,
)
from worker.segment import GAP_HARD
from worker.text import format_transcript

__all__ = ["StateResult", "read_state", "pick_label", "NEUTRAL", "EXPIRE_AFTER"]

# 판단이 서지 않을 때의 기본 라벨.
NEUTRAL = "STABLE"

# --------------------------------------------------------------------------
# 라벨 판정 — **여기가 룰이다. LLM 은 점수만 낸다.**
# --------------------------------------------------------------------------
# 감정 축 → 상태 라벨. 이 매핑은 여기 하나뿐이다.
# **`STABLE` 은 축이 없다.** 네 축이 전부 임계 아래인 상태가 평온이다.
LABEL_BY_AXIS: dict[str, StateLabel] = {
    "affection": "RESOLVED",
    "hurt": "ACCUMULATED",
    "joy": "ENGAGED",
    "anger": "ESCALATED",
}

# 최고점이 이 아래면 아무 신호도 없는 것으로 보고 STABLE 로 둔다.
#
# **프롬프트 앵커에 맞춰 3 으로 잡았다.** 프롬프트가 3 을 "뚜렷하다 — 로그에서 문장으로
# 짚을 수 있다"로 정의한다. 2 로 내렸더니 리액션("ㅇㅇ", "ㅋㅋ")만 하는 화자가 활기 2 를
# 받아 `ENGAGED` 로 떴다 — 그건 신호가 아니라 잡음이다.
MIN_SCORE = 3

# **분노는 다른 축보다 낮아도 이 점수 이상이면 이긴다.**
#
# 분노는 혼자 오지 않는다 — 서운해서 화가 난다. 그래서 다투는 구간은 `hurt` 와 `anger`
# 가 같이 높고, 최고점만 보면 서운함이 이겨서 파란 매듭이 뜬다. 실측(`case11_mixed`)에서
# 말투 교정은 공격 표현으로 잡은 발화를 상태 산출은 서운함으로 읽었다 — **한 발화를 두
# 기능이 다르게 읽는 것**이라 맞춰야 한다.
#
# 화면에 필요한 것은 "지금 격해졌다"는 신호고, 그게 빨간 스파이크의 자리다.
# 값을 올리면 분노 판정이 보수적이 되고 내리면 공격적이 된다 — **손잡이가 여기 있다.**
ANGER_WINS = 3

# 동점일 때의 순서.
#
# 갈등 신호(`anger` · `hurt`)를 앞에 둔다 — 즐거운 대화에 섞인 서운함이 묻히면 안 된다.
# `joy` 가 `affection` 보다 앞인 이유: `RESOLVED`(감정 풀어짐)는 쌓인 것이 있었을 때만
# 뜻이 있는 라벨이다. 데이트 계획처럼 애정과 활기가 같이 높은 자리에서 "다정해 보여요"가
# 뜨면 화해한 것처럼 읽힌다. `affection` 이 더 높을 때만 `RESOLVED` 로 간다.
PRIORITY = ("anger", "hurt", "joy", "affection")


def pick_label(scores: EmotionScores) -> tuple[StateLabel, float]:
    """감정 점수에서 라벨과 강도를 정한다. **판정은 전부 여기 있다.**

    경계가 LLM 안에 있으면 결과가 이상해도 프롬프트를 다시 쓰는 것 말고 할 수 있는 게
    없다. 임계값으로 빼두면 숫자로 만질 수 있고 왜 그 라벨인지 트레이스에 남는다
    (`worker/segment.py` 와 같은 이유).
    """
    axes = {
        "affection": scores.affection,
        "hurt": scores.hurt,
        "joy": scores.joy,
        "anger": scores.anger,
    }

    if scores.anger >= ANGER_WINS:
        return "ESCALATED", float(scores.anger)

    top = max(axes.values())
    if top < MIN_SCORE:
        # 네 축이 전부 임계 아래 = 평온. `STABLE` 은 별도 축이 아니라 이 바닥값이다.
        return NEUTRAL, 0.0

    winner = next(axis for axis in PRIORITY if axes[axis] == top)
    return LABEL_BY_AXIS[winner], float(top)

# 상태 문구의 유효 기간.
#
# **새로 정한 값이 아니다.** `segment.GAP_HARD` 를 그대로 쓴다 — 분절이 이미 "3시간 이상
# 벌어지면 다른 대화"로 보고 있다. 기준이 어긋나면 워커가 "다른 대화"로 자른 구간에
# 프론트는 이전 문구를 계속 띄우게 된다.
EXPIRE_AFTER = GAP_HARD


@dataclass
class StateResult:
    """산출 결과. `scored` 는 트레이스용이고 판정에는 이미 반영돼 있다.

    분절의 `SegmentResult` 와 같은 모양이다 — 화면에 나가는 것과 왜 그렇게 됐는지를
    갈라 둔다. `scored[].note` 는 **트레이스 밖으로 나가지 않는다.**
    """

    analyses: list[EmotionAnalysis] = field(default_factory=list)
    scored: list[EmotionScores] = field(default_factory=list)


def _partner(speaker: Speaker, speakers: list[Speaker]) -> Speaker | None:
    return next((s for s in speakers if s != speaker), None)


def read_state(
    messages: list[Message],
    speakers: list[Speaker],
    now: datetime,
) -> StateResult:
    """화자별 상태를 산출한다. **보는 사람 기준으로 한 건씩** 돌려준다.

    `subject` 는 감정의 주인이고 `viewer` 는 그걸 보는 사람이다 — 각자 상대방의 상태를
    본다. 그래서 A 화면과 B 화면의 ①번 줄 내용이 다르다.

    실패는 오류가 아니라 **"이번엔 갱신 없음"** 이다. 빈 목록을 돌려주면 프론트는 직전
    문구를 그대로 두면 된다. 여기서 예외를 던지면 요청 전체가 `FAILED` 가 되고, 상태 한 줄
    때문에 데이트 코스 추천까지 죽는다.
    """
    if not messages or len(speakers) < 2:
        return StateResult()

    try:
        out = ask(StateLLMOutput, load_prompt("state"), format_transcript(messages))
    except Exception:  # noqa: BLE001 — 채점 실패는 오류가 아니라 '갱신 없음'이다
        return StateResult()

    scored = {s.speaker: s for s in out.states if s.speaker in speakers}
    last_at = max(m.sent_at for m in messages)

    analyses: list[EmotionAnalysis] = []
    for subject in speakers:
        viewer = _partner(subject, speakers)
        if viewer is None:
            continue

        own_ids = [m.message_id for m in messages if m.sender == subject]
        state = scored.get(subject)

        if not own_ids:
            # 이 사람의 발화가 활성 맥락에 없다 → 판단할 근거가 없다.
            # 문구를 지어내지 않고 갱신도 하지 않는다. 화면은 직전 값을 유지한다.
            label, intensity, should_show = NEUTRAL, 0.0, False
        elif state is None or not state.confident:
            # 신호가 약하다 → 중립을 **띄운다**. 갱신은 한다.
            # 여기서 갱신을 멈추면 다툼이 끝난 뒤에도 직전의 격한 문구가 계속 남는다.
            label, intensity, should_show = NEUTRAL, 0.0, True
        else:
            label, intensity = pick_label(state)
            should_show = True

        analyses.append(
            EmotionAnalysis(
                subject_participant=to_key(subject),
                viewer_participant=to_key(viewer),
                emotion_type=label,  # type: ignore[arg-type]
                intensity_value=intensity,
                should_show=should_show,
                trigger_message_ids=own_ids,
                expires_at=last_at + EXPIRE_AFTER,
                state_text=STATE_TEXT[label],
            )
        )
    return StateResult(analyses=analyses, scored=list(scored.values()))
