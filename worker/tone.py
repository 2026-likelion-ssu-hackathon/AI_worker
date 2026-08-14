"""후보 기능 2 — 갈등 중재 (말투 교정 제안).

공격 표현·오해 유발 표현이 감지되면 **보낸 사람에게만** 개별로
① 왜 이 표현이 다르게 읽힐 수 있는지(방향 문구)
② 지금 이어서 보낼 수 있는 대체 문장 1개
를 제시한다.

기존 트리거(①~⑤)가 "대화 흐름"을 보는 것과 달리, 이 기능은 **방금 전송된 메시지 하나**를 본다.

    [메시지 전송] → [룰 트리거] → [LLM 맥락 판정] → [방향 문구 + 대체 문장] → [보낸 사람에게만]

판정과 생성을 따로 호출하는 이유: 이 기능의 최대 리스크는 "와 미친 ㅋㅋ" 같은 장난을
갈등으로 오인하는 것이다. 판정 프롬프트를 판정에만 집중시키고, 통과한 경우에만 생성한다.
대부분의 메시지는 판정에서 걸러지므로 호출 비용도 오히려 줄어든다.
"""

from __future__ import annotations

import re

from worker.llm import ask, load_prompt
from worker.models import (
    Message,
    SpeakerProfile,
    ToneFlag,
    ToneGateResult,
    ToneJudgeLLMOutput,
    ToneSuggestLLMOutput,
)
from worker.profile import (
    HARSH_ADDRESS,
    addresses_in,
    describe,
    resolve_profile,
)

# 직전 몇 턴을 맥락으로 넘길지 (명세: 직전 3턴)
CONTEXT_TURNS = 3

# 평소 대비 급변 판정 임계
SHORT_RATIO = 0.4
LONG_RATIO = 2.5
LOW_PERIOD_RATE = 0.15
LAUGH_BASELINE = 1.5
EMOJI_BASELINE = 0.3

# 다른 신호 없이 "평소 대비 급변"만으로 발동하려면 필요한 하위 신호 개수
ABRUPT_ALONE_SIGNALS = 3

# 인신공격 · 욕설. 맥락상 장난일 수 있으므로 룰은 후보만 잡고 확정은 LLM 이 한다.
_INSULT_RE = re.compile(
    r"(미친|또라이|돌+았|바보|멍청|한심|어이없|찌질|재수\s*없|밥맛|꺼져|닥쳐|"
    r"시끄러|짜증|정\s*떨어|질린다|역겹|쓰레기같)"
)

# 일반화 화법
_GENERAL_RE = re.compile(r"(늘|맨날|만날|항상|매번|평생|한\s*번을|하나같이|늘상|또\s*그러)")

# 비꼼 · 반어법
_SARCASM_RE = re.compile(r"(잘한다|잘하는\s*짓|잘\s*났|대단하다|훌륭하다|자랑이다|장하다|기특하다)")

_LAUGH_RE = re.compile(r"[ㅋㅎ]")
_EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF]")
_PERIOD_END_RE = re.compile(r"[^.]\.\s*$")
_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[^\s.,!?~…]+")


def _norm_len(text: str) -> int:
    return len(_SPACE_RE.sub("", text))


# --------------------------------------------------------------------------
# 룰 트리거
# --------------------------------------------------------------------------
def _abrupt_flags(text: str, profile: SpeakerProfile) -> list[str]:
    """평소 대비 급변 신호. 하나만으로는 약해서 2개 이상일 때만 트리거한다."""
    signals: list[str] = []
    length = _norm_len(text)

    if _PERIOD_END_RE.search(text) and profile.period_rate < LOW_PERIOD_RATE:
        signals.append(f"평소 마침표 종결 {profile.period_rate:.0%} → 이번엔 마침표로 끝남")

    if profile.laugh_per_msg >= LAUGH_BASELINE and not _LAUGH_RE.search(text):
        signals.append(f"평소 ㅋ/ㅎ {profile.laugh_per_msg:.1f}개 → 이번엔 0개")

    if profile.emoji_rate >= EMOJI_BASELINE and not _EMOJI_RE.search(text):
        signals.append(f"평소 이모지 {profile.emoji_rate:.0%} → 이번엔 없음")

    if profile.avg_length >= 8:
        if length <= profile.avg_length * SHORT_RATIO:
            signals.append(f"평소 {profile.avg_length:.0f}자 → 이번 {length}자로 짧아짐")
        elif length >= profile.avg_length * LONG_RATIO:
            signals.append(f"평소 {profile.avg_length:.0f}자 → 이번 {length}자로 길어짐")

    return signals


def _repetition_flag(messages: list[Message], speaker: str) -> str | None:
    """비슷한 말 반복 ("전화 받아 - 받아 - 받으라고")."""
    mine = [m for m in messages if m.sender == speaker][-3:]
    if len(mine) < 2:
        return None
    token_sets = [{t for t in _TOKEN_RE.findall(m.content) if len(t) >= 2} for m in mine]
    shared = set.intersection(*token_sets) if token_sets else set()
    if shared and all(_norm_len(m.content) <= 15 for m in mine):
        return f"최근 {len(mine)}개 메시지에 '{sorted(shared)[0]}' 반복"
    return None


def check_tone_gate(
    messages: list[Message], profile: SpeakerProfile | None = None
) -> ToneGateResult:
    """방금 전송된 메시지 하나를 본다. 확정하지 않고 후보만 잡는다."""
    if not messages:
        return ToneGateResult(triggered=False)

    ordered = sorted(messages, key=lambda m: m.ts)
    last = ordered[-1]
    profile = profile or resolve_profile(last.sender, ordered)
    text = last.content
    flags: list[ToneFlag] = []

    if hit := _INSULT_RE.search(text):
        flags.append(ToneFlag(kind="insult", detail=f"'{hit.group(0)}' 표현"))

    if hit := _GENERAL_RE.search(text):
        flags.append(ToneFlag(kind="generalization", detail=f"일반화 화법 '{hit.group(0)}'"))

    if hit := _SARCASM_RE.search(text):
        flags.append(ToneFlag(kind="sarcasm", detail=f"반어 표현 '{hit.group(0)}'"))

    # 호칭 변화 — 평소 쓰던 호칭이 아닌 거친 호칭이 등장
    used = set(addresses_in(text)) & HARSH_ADDRESS
    unusual = used - set(profile.top_address)
    if unusual:
        flags.append(
            ToneFlag(
                kind="address_change",
                detail=f"평소 '{', '.join(profile.top_address) or '없음'}' → '{', '.join(sorted(unusual))}'",
            )
        )

    if (rep := _repetition_flag(ordered, last.sender)) is not None:
        flags.append(ToneFlag(kind="repetition", detail=rep))

    # 평소 대비 급변은 **약한 증거**다. 단답 핑퐁이나 대화가 잦아든 상황에서도 그대로 걸린다.
    # 다른 신호와 함께일 때만 세고, 혼자서는 신호 3개 이상일 때만 인정한다.
    abrupt = _abrupt_flags(text, profile)
    if abrupt and (flags or len(abrupt) >= ABRUPT_ALONE_SIGNALS):
        flags.append(ToneFlag(kind="abrupt_change", detail=" / ".join(abrupt)))

    # 급변 신호만 남았는데 근거가 약하면 발동하지 않는다
    if flags and all(f.kind == "abrupt_change" for f in flags) and len(abrupt) < ABRUPT_ALONE_SIGNALS:
        flags = []

    return ToneGateResult(
        triggered=bool(flags),
        speaker=last.sender if flags else None,
        message=text if flags else None,
        flags=flags,
    )


# --------------------------------------------------------------------------
# LLM 맥락 판정 + 생성
# --------------------------------------------------------------------------
def _context_block(messages: list[Message], profile: SpeakerProfile, gate: ToneGateResult) -> str:
    ordered = sorted(messages, key=lambda m: m.ts)
    prior = ordered[-(CONTEXT_TURNS + 1) : -1]
    prior_lines = [f"{m.sender}: {m.content}" for m in prior] or ["(없음)"]
    lines = [
        "## 문제로 감지된 문장",
        f"{gate.speaker}: {gate.message}",
        "",
        "## 직전 대화",
        *prior_lines,
        "",
        f"## {gate.speaker}의 평소 말투 기준선",
        describe(profile),
        "",
        "## 룰이 잡은 신호",
        *(f"- [{f.kind}] {f.detail}" for f in gate.flags),
    ]
    return "\n".join(lines)


def tone_judge(
    messages: list[Message], gate: ToneGateResult, profile: SpeakerProfile
) -> ToneJudgeLLMOutput:
    """진짜 갈등인지, 맥락상 장난인지 판정한다. 문구는 만들지 않는다."""
    return ask(
        ToneJudgeLLMOutput,
        load_prompt("tone_judge"),
        _context_block(messages, profile, gate),
    )


def tone_suggest(
    messages: list[Message],
    gate: ToneGateResult,
    profile: SpeakerProfile,
    judged: ToneJudgeLLMOutput,
) -> ToneSuggestLLMOutput:
    """방향 문구 + 대체 문장을 만든다."""
    body = _context_block(messages, profile, gate)
    body += f"\n\n## 판정된 감정 온도\n{judged.emotion}"
    return ask(ToneSuggestLLMOutput, load_prompt("tone_suggest"), body)
