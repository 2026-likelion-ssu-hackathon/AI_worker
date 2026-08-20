"""실 상태 표현 — 위젯 ①번 줄. **상시 작동한다.**

설계와 근거는 `docs/design.md` 2부. 구조만 옮기면:

    ① LLM 채점   화자마다 **감정 4축 점수**(0~5) (호출 1회). 라벨도 문구도 만들지 않는다
    ②' 병합      화자별 유효 점수의 **축별 최댓값** → 커플 공통 점수 (PM 기획 변경 2026-08-19)
    ② 룰 판정    `pick_label()` 의 임계값이 점수에서 라벨을 정한다
    ③ 문구 선택   `copy.STATE_TEXT` 사전에서 라벨로 찾는다

**두 화면의 ①번 줄이 같다.** 원래 각자 상대방의 상태를 보는 개인화 설계였는데 PM 이
"둘의 상황 하나"로 바꿨다. 규격(`subject`/`viewer` 두 항목)은 그대로 두고 내용만
동일하게 싣는다 — 백엔드·프론트 무변경 (`read_state` docstring 참조).

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

# --------------------------------------------------------------------------
# 애정 표현 룰 — 명시적 애정 낱말이 있으면 애정 축의 바닥값을 보장한다
# --------------------------------------------------------------------------
# "사랑해"라고 말했는데 STABLE 이 뜨는 것을 막는다 (2026-08-19 연동 실측).
# 짧은 애정 발화는 LLM 이 애정을 2 이하로 깔 때가 있어 MIN_SCORE 에 걸려 평온이 됐다.
#
# **바닥값이지 강제 라벨이 아니다.** affection 을 MIN_SCORE 까지만 올리고 판정은
# 그대로 `pick_label()` 이 한다 — 분노가 ANGER_WINS 이상이면 여전히 ESCALATED 다.
# "사랑 같은 소리 하네"처럼 애정 낱말이 반어로 쓰인 발화는 분노 점수가 막는다.
# 라벨을 직접 찍으면 그 방어가 통째로 사라진다.
#
# 부분 문자열이라 "사랑해/사랑해요/사랑스러워", "보고싶다/보고싶었어"를 다 잡는다.
# "사랑니"는 애정이 아니라서 지우고 본다. "영화 보고싶어"(콘텐츠)는 못 가른다 —
# 알려진 한계인데, 틀려도 긍정 방향(다정)이라 없던 갈등을 만들지는 않는다.
AFFECTION_WORDS = ("사랑", "보고싶", "보고 싶")

# 화자의 **직전 발화 몇 개**까지 볼지. 창 전체를 보면 대화 초반의 "사랑해" 하나가
# 창에서 밀려날 때까지 계속 바닥값을 만든다 — 상태는 지금을 보여주는 자리다.
AFFECTION_RECENT = 3


def has_affection_words(text: str) -> bool:
    cleaned = text.replace("사랑니", "")
    return any(w in cleaned for w in AFFECTION_WORDS)


# 상태 문구의 유효 기간.
#
# **새로 정한 값이 아니다.** `segment.GAP_HARD` 를 그대로 쓴다 — 분절이 이미 "3시간 이상
# 벌어지면 다른 대화"로 보고 있다. 기준이 어긋나면 워커가 "다른 대화"로 자른 구간에
# 프론트는 이전 문구를 계속 띄우게 된다.
EXPIRE_AFTER = GAP_HARD

# 채점 대상은 **마지막 이 개수**만. 앞은 흐름을 읽는 맥락으로만 넘긴다.
#
# 배포 실측(2026-08-20, QA.md QA8): 격한 싸움 뒤 화해 6건 + 화제 전환 4건을 쌓아도
# `ESCALATED` 가 안 내려왔다. 싸움 발화가 채점 가능한 로그에 남아 있는 한 프롬프트
# 앵커("3 = 로그에서 문장으로 짚을 수 있다")에 분노가 계속 걸린다 — "뒤쪽이 근거"라는
# 지침만으로는 같은 세그먼트 안에서 안 지켜졌다. 그래서 분절 채점(`segment._transcript`)과
# 같은 방식으로 **구획을 갈라 인용 자체를 막는다.** 프롬프트로 안 되는 것을 구조가 막는
# 자리다 — 자리별 카테고리 강제·자수 한도와 같은 패턴.
#
# 6인 이유: 시연 대본의 장면이 4~6줄이라 장면 하나가 통째로 들어가고, 화해가 세 턴
# (사과 → 수용 → 다정) 쌓이면 싸움이 창 밖으로 완전히 밀려난다. 줄이면 전환이 빨라지고
# 늘리면 직전 감정이 오래 남는다 — 손잡이가 여기 있다.
SCORE_RECENT = 6


def _transcript(context: list[Message], targets: list[Message]) -> str:
    """채점 대상과 앞 맥락을 구획으로 나눠 준다. 맥락이 없으면 기존 단일 형식 그대로다."""
    body = format_transcript(targets)
    if not context:
        return body
    return (
        f"## 앞 맥락 (흐름만 — 채점 금지)\n{format_transcript(context)}\n\n"
        f"## 채점 대상 (지금 감정)\n{body}"
    )


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
    """**둘의 상황 하나**를 산출해 두 화면에 같은 내용으로 내보낸다.

    처음에는 각자 상대방의 상태를 보는 개인화 설계였는데 **PM 기획 변경(2026-08-19)** 으로
    커플 공통 상태로 바뀌었다. 규격은 안 바꿨다 — `subject`/`viewer` 쌍 두 건을 그대로
    내보내되 **라벨·강도·문구를 동일하게** 싣는다. 백엔드 분배도 프론트 렌더링도 그대로다.

    병합은 **축별 최댓값**이다. 평균을 쓰면 한쪽의 분노 4 가 상대의 평온 0 에 희석되어
    임계(MIN_SCORE) 아래로 내려간다 — 갈등 신호가 사라진다. 둘 중 누군가에게 뚜렷한
    감정이 있으면 그게 둘의 상황이다. 반어 방어(ANGER_WINS)도 최댓값이라야 산다.

    화자 단위 안전 규칙(애정 바닥값 · confident 게이트 · 미채점 무시)은 병합 **전에**
    화자마다 그대로 적용한다 — 근거 없는 화자의 점수를 지어내지 않는 원칙은 유지된다.

    실패는 오류가 아니라 **"이번엔 갱신 없음"** 이다. 빈 목록을 돌려주면 프론트는 직전
    문구를 그대로 두면 된다. 여기서 예외를 던지면 요청 전체가 `FAILED` 가 되고, 상태 한 줄
    때문에 데이트 코스 추천까지 죽는다.
    """
    if not messages or len(speakers) < 2:
        return StateResult()

    # 채점은 마지막 SCORE_RECENT 개만 — 앞은 흐름용 맥락으로만 넘긴다 (상수 주석 참조).
    targets = messages[-SCORE_RECENT:]
    try:
        out = ask(StateLLMOutput, load_prompt("state"),
                  _transcript(messages[:-SCORE_RECENT], targets))
    except Exception:  # noqa: BLE001 — 채점 실패는 오류가 아니라 '갱신 없음'이다
        return StateResult()

    scored = {s.speaker: s for s in out.states if s.speaker in speakers}
    last_at = max(m.sent_at for m in messages)

    # ① 화자별 유효 점수 — 기존 규칙을 화자 단위로 적용한 뒤에야 병합에 넣는다.
    #    발화 유무·애정 낱말도 **채점 대상 구간 기준**이다 — 상태는 지금을 보여주는
    #    자리라, 창 밖 발화로 기여 자격이나 바닥값을 만들면 채점 제한이 새로 샌다.
    merged = {"affection": 0, "hurt": 0, "joy": 0, "anger": 0}
    for subject in speakers:
        own = [m for m in targets if m.sender == subject]
        state = scored.get(subject)
        # 애정 낱말은 직전 발화만 본다 (상수 주석 참조).
        hinted = any(has_affection_words(m.content) for m in own[-AFFECTION_RECENT:])

        if not own or state is None:
            # 발화가 없거나 LLM 이 안 채점한 화자 → 기여 없음. 지어내지 않는다.
            # (미채점이면 애정 룰도 안 태운다 — 분노 점수 없이는 반어를 막을 가드가 없다.)
            continue
        if hinted and state.affection < MIN_SCORE:
            # 명시적 애정 낱말 → 바닥값 보정. 트레이스에도 보정 뒤 값이 남게 갈아 끼운다.
            state = state.model_copy(update={
                "affection": MIN_SCORE,
                "note": f"{state.note} · 룰: 애정 낱말 → affection 바닥값 {MIN_SCORE}",
            })
            scored[subject] = state
        if not state.confident and not hinted:
            # 신호가 약한 화자는 중립 기여 — 병합에 0 으로 들어간다.
            continue
        for axis in merged:
            merged[axis] = max(merged[axis], getattr(state, axis))

    # ② 병합 점수로 한 번만 판정한다. 판정 규칙(pick_label)은 그대로다.
    couple = EmotionScores(
        speaker=speakers[0],  # 스키마 자리 채움 — 트레이스에 안 싣고 라벨 판정에만 쓴다
        confident=True,
        note="커플 병합 — 화자별 유효 점수의 축별 최댓값",
        **merged,
    )
    label, intensity = pick_label(couple)

    # ③ 두 항목에 같은 내용을 싣는다. 근거는 둘의 대화 전체라 trigger 도 전체 id 다.
    #    갱신은 항상 한다(should_show=True) — 한쪽이 침묵해도 다른 쪽 발화가 근거가 된다.
    all_ids = [m.message_id for m in messages]
    analyses: list[EmotionAnalysis] = []
    for subject in speakers:
        viewer = _partner(subject, speakers)
        if viewer is None:
            continue
        analyses.append(
            EmotionAnalysis(
                subject_participant=to_key(subject),
                viewer_participant=to_key(viewer),
                emotion_type=label,  # type: ignore[arg-type]
                intensity_value=intensity,
                should_show=True,
                trigger_message_ids=all_ids,
                expires_at=last_at + EXPIRE_AFTER,
                state_text=STATE_TEXT[label],
            )
        )
    return StateResult(analyses=analyses, scored=list(scored.values()))
