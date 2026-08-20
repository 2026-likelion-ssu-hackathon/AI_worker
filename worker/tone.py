"""후보 기능 1 — 갈등 중재 (말투 교정 제안).

공격 표현·오해 유발 표현이 감지되면 **보낸 사람에게만** 개별로 제시한다.
출력 3종은 백엔드 규격 8장 `resultData` 와 화면 명세가 그대로 대응된다.

    situationDiagnosis   상황 진단      "지금 감정이 조금 올라와 있는 것 같아요"
    alternativeSentence  대체 문장      "오늘 못 온다고 하니까 좀 서운했어"
    correctionReason     그렇게 읽히는 이유  "'맨날'은 그동안 전부로 들릴 수 있어요"

(`guideMessage` 는 고정 안내 문구라 LLM 이 만들지 않는다. `copy.TONE_GUIDE` 상수를
`router.py` 가 결과에 실어 보낸다.)

다른 후보가 "대화 흐름"을 보는 것과 달리, 이 기능은 **방금 전송된 메시지 하나**를 본다.

    [메시지 전송] → [룰 트리거] → [LLM 맥락 판정] → [방향 문구 + 대체 문장] → [보낸 사람에게만]

판정과 생성을 따로 호출하는 이유: 이 기능의 최대 리스크는 "와 미친 ㅋㅋ" 같은 장난을
갈등으로 오인하는 것이다. 판정 프롬프트를 판정에만 집중시키고, 통과한 경우에만 생성한다.
대부분의 메시지는 판정에서 걸러지므로 호출 비용도 오히려 줄어든다.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

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
from worker.text import norm_len as _norm_len

# 직전 몇 턴을 맥락으로 넘길지 (명세: 직전 3턴)
CONTEXT_TURNS = 3

# 평소 대비 급변 판정 임계
SHORT_RATIO = 0.4
LONG_RATIO = 2.5
LOW_PERIOD_RATE = 0.15
LAUGH_BASELINE = 1.5
EMOJI_BASELINE = 0.3

# 다른 신호 없이 "평소 대비 급변"만으로 발동하려면 필요한 하위 신호 개수.
#
# **3 → 2 로 내렸다.** 예전에는 "짧아짐 / ㅋ 없음 / 이모지 없음" 이 따로 세어져 3이
# 그냥 찼는데, 셋이 사실 한 가지라서 `_abrupt_flags` 에서 묶었다(거기 설명 참조).
# 묶고 나니 3은 사실상 도달 불가라 명세의 `abrupt_change` 트리거가 통째로 죽는다 —
# "됐어." "몰라." 같은 **진짜 냉랭한 단답**이 2개에서 막혔다.
ABRUPT_ALONE_SIGNALS = 2

# 대체 문장에 남으면 안 되는 거친 감정 어휘.
#
# **원문이 아니라 우리가 만든 문장만 검사한다.** 대체 문장은 사용자가 **그대로 보낼**
# 것이라, 여기 "짜증나"가 남으면 말투 교정이 거친 말을 다시 쓰라고 시키는 꼴이 된다.
# 실측: `짜증나` → **"나 지금 좀 짜증나서 그래"** — 주어만 '나'로 바꾸고 어휘는 그대로였다.
# 원하는 모양은 **"그런 게 아니라 나 지금 좀 속상해서 그래"** 다.
#
# ⚠️ **감정을 지우자는 게 아니다.** 세기와 색만 낮춘다 — 짜증 → 속상. 감정을 빼버리면
# 나 전달법이 성립하지 않고, 사용자가 하지 않은 말("괜찮아")을 시키는 게 된다.
#
# `화나`·`서운`·`속상` 은 **넣지 않았다.** 나 전달법에서 정당한 감정 표현이라 막을 이유가
# 없다. 여기 목록은 상대를 깎거나 대화를 끊는 쪽만 담는다.
_HARSH_RE = re.compile(
    r"(짜증|열\s*받|빡치|빡쳐|미치겠|어이없|황당|지겹|지긋지긋|최악|꼴\s*보기|"
    r"됐어|관심\s*없|말\s*섞기|질린다|정\s*떨어)"
)


def harsh_in(text: str, profile: SpeakerProfile | None = None) -> str | None:
    """대체 문장에 남은 거친 표현. 없으면 None.

    `profile` 을 주면 **평소 안 쓰던 거친 호칭**도 같이 본다. 게이트가 `address_change`
    로 잡은 바로 그 호칭이 교정 문장에 다시 들어가는 일이 있었다 —
    `야 너는 맨날 그런 식이야` → **"야, 오늘 또 못 온다니까 좀 서운해"** (실측).
    어휘는 부드러워졌는데 상대가 먼저 듣는 한 마디가 그대로다.

    **평소 호칭은 막지 않는다.** 늘 "야"라고 부르는 사이면 그건 거친 게 아니다 —
    이 기능 전체가 "그 사람의 평소 대비"로 판정하는 것과 같은 이유다.
    """
    if (hit := _HARSH_RE.search(text)) is not None:
        return hit.group(0)
    if profile is not None:
        unusual = (set(addresses_in(text)) & HARSH_ADDRESS) - set(profile.top_address)
        if unusual:
            return sorted(unusual)[0]
    return None


# correction_reason 이 언급하면 안 되는 형태 특징 — 원문에 그 문자가 없을 때.
#
# 실측(2026-08-20, 사용자 발견): 마침표 없는 메시지에 "마침표로 끝나 단호하게 들릴 수
# 있어요"가 나갔다. 프롬프트에 들어가는 **"평소 말투 기준선" 설명(마침표율·ㅋ 개수)을
# 이번 메시지의 특징으로 끌어다 쓴 것** — 사용자는 자기가 쓴 문장을 알고 있어서, 안 찍은
# 마침표를 찍었다고 하는 순간 위젯 전체의 신뢰가 무너진다. 프롬프트에 근거 실재성 규칙을
# 넣었지만(tone_suggest.md) 프롬프트만으로 안 지켜지는 것을 코드가 재검하는 자리다 —
# 자수 한도·거친 어휘 검사와 같은 패턴.
# ㅋ·이모지는 넣지 않는다 — 게이트 신호가 "평소보다 없음"(부재)이라, 원문에 없어도
# 언급하는 게 정당하다. 여기는 **있다고 주장하려면 실제로 있어야 하는** 특징만 담는다.
_FEATURE_CLAIMS = (("마침표", "."), ("물음표", "?"), ("느낌표", "!"))
_REASON_QUOTE_RE = re.compile(r"'([^']+)'")


# 대체 문장이 원문과 "사실상 같다"고 볼 유사도 임계.
#
# 실측 캘리브레이션(2026-08-20, 라이브 카드 + QA 배터리 출력 11쌍): 좋은 교정은 전부
# 0.40 이하("야 너는 맨날 그런 식이야" → "오늘 그런 식이라서 좀 서운했어" = 0.35),
# 가짜 교정은 0.86 이상("너 어제 … 누구야" → "어제 … 사람 누구야?" = 0.86)으로 갈렸다.
# 0.75 는 그 사이의 여유 있는 경계다 — 낮추면 살짝 다듬은 진짜 교정까지 죽는다.
# 비교 전에 공백·문장부호·ㅋㅎ 를 지운다 — 마침표 하나 바꾼 것은 교정이 아니다.
SAME_RATIO = 0.75
_SAME_STRIP_RE = re.compile(r"[\s.,!?~…ㅋㅎ]+")


def same_message(alternative: str, original: str) -> bool:
    """대체 문장이 원문과 사실상 같은가 — 같으면 교정이 아니다.

    "말투 교정" 카드가 원문과 똑같은 문장을 제시하면(실측: "너 어제 공원에서 술 마신 거
    누구야" → "어제 공원에서 술 마신 사람 누구야?") 사용자에게는 기능이 고장 난 것으로
    보인다. 프롬프트도 "바꿀 게 없으면 억지로 만들지 않는다"고 하므로, 바뀐 게 없다는
    것은 **교정할 것이 없었다**는 뜻이다 — 그 경우의 올바른 출력은 침묵이다.
    """
    a = _SAME_STRIP_RE.sub("", alternative)
    o = _SAME_STRIP_RE.sub("", original)
    if not a or not o:
        return not a  # 빈 대체 문장은 교정이 아니다
    return SequenceMatcher(None, o, a).ratio() >= SAME_RATIO


def ungrounded_in(reason: str, diagnosis: str, original: str) -> str | None:
    """진단·이유가 원문에 없는 특징을 주장하면 그 주장. 없으면 None.

    검증 가능한 주장만 본다 — 형태 특징(마침표류)과 따옴표 인용. "단호하게 들린다" 같은
    해석은 LLM 의 몫이라 건드리지 않는다.
    """
    for claimed in (reason, diagnosis):
        for word, char in _FEATURE_CLAIMS:
            if word in claimed and char not in original:
                return f"{word} 언급 (원문에 없음)"
    for quoted in _REASON_QUOTE_RE.findall(reason):
        if quoted not in original:
            return f"인용 '{quoted}' (원문에 없음)"
    return None


# 인신공격 · 욕설 · 도발. 맥락상 장난일 수 있으므로 룰은 후보만 잡고 확정은 LLM 이 한다.
# "장난해?" 류는 급변 신호 축소(가벼움 표지 묶음, 2026-08-20)로 놓치게 된 명백한 도발을
# 어휘로 되잡는 것이다 — 물음표까지 있어야 잡는다 ("장난해 놀자"는 도발이 아니다).
_INSULT_RE = re.compile(
    r"(미친|또라이|돌+았|바보|멍청|한심|어이없|찌질|재수\s*없|밥맛|꺼져|닥쳐|"
    r"시끄러|짜증|정\s*떨어|질린다|역겹|쓰레기같|장난(해|하냐|하니|이야|임)\s*\?)"
)

# 일반화 화법
#
# `늘` 을 그냥 넣으면 **"오늘"의 '늘'이 걸린다.** 채팅에서 "오늘"보다 흔한 단어가 없어서
# 상시 오발동한다. 앞뒤에 한글 음절이 붙지 않은 경우만 인정한다.
#
# `만날` 은 아예 뺐다. 경계를 잡아도 "토요일에 만날까"(만나다)와 "만날 늦어"(맨날)를
# 구분할 수 없다. 같은 뜻은 `맨날` 이 잡는다.
_GENERAL_RE = re.compile(
    r"((?<![가-힣])늘(?![가-힣])|맨날|항상|매번|평생|한\s*번을|하나같이|늘상|또\s*그러)"
)

# 비꼼 · 반어법
_SARCASM_RE = re.compile(r"(잘한다|잘하는\s*짓|잘\s*났|대단하다|훌륭하다|자랑이다|장하다|기특하다)")

_LAUGH_RE = re.compile(r"[ㅋㅎ]")
_EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF]")
_PERIOD_END_RE = re.compile(r"[^.]\.\s*$")
_TOKEN_RE = re.compile(r"[^\s.,!?~…]+")


# --------------------------------------------------------------------------
# 룰 트리거
# --------------------------------------------------------------------------
# 짧은 **동의·긍정 응답**은 "짧아짐"으로 세지 않는다.
#
# 명세의 `abrupt_change` 는 "맥락 없는 단답"을 말한다. 그런데 "홍대도 좋아" 같은 동의는
# 대화가 굴러가고 있다는 신호지 오해 유발이 아니다. **길이만 보면 둘이 구분되지 않는다.**
#
# 마침표 종결("알겠어.")은 그대로 센다 — 동의 어휘라도 마침표로 끊으면 차가워진다.
# ⚠️ `하자` · `가자` · `보자` 는 넣지 않는다. **동의가 아니라 제안이고 부정일 수 있다** —
# "그만하자" 가 동의로 잡혀서 급변 신호가 통째로 죽었다.
_AGREEMENT_RE = re.compile(
    r"(좋아|좋지|좋다|좋겠|좋은데|괜찮|그래|그러자|그러지|오케|오키|콜|찬성|맞아|맞네|동의)"
)
_DISAGREE_RE = re.compile(r"(안\s|말고|싫|별로|아니|못\s|그닥|글쎄|됐어|관심\s*없)")


def _is_agreement(text: str) -> bool:
    """상대 제안에 동의하는 짧은 응답인가."""
    return bool(_AGREEMENT_RE.search(text)) and not _DISAGREE_RE.search(text)


def _abrupt_flags(text: str, profile: SpeakerProfile) -> list[str]:
    """평소 대비 급변 신호. 하나만으로는 약해서 여러 개일 때만 트리거한다.

    ⚠️ **상관된 신호를 따로 세지 않는다.** 짧아진 메시지에는 ㅋ 도 이모지도 당연히 없다 —
    세 개를 각각 세면 **사실 하나를 세 번 세는 것**이라 `ABRUPT_ALONE_SIGNALS` 가 그냥
    찬다. 실측에서 "홍대도 좋아"(5자)가 신호 3개로 발동했다.
    """
    signals: list[str] = []
    length = _norm_len(text)
    shortened = profile.avg_length >= 8 and length <= profile.avg_length * SHORT_RATIO
    # 짧아진 것이 **동의 응답**이면 급변으로 세지 않는다 (위 `_is_agreement` 설명)
    count_short = shortened and not _is_agreement(text)

    if _PERIOD_END_RE.search(text) and profile.period_rate < LOW_PERIOD_RATE:
        signals.append(f"평소 마침표 종결 {profile.period_rate:.0%} → 이번엔 마침표로 끝남")

    # 짧아진 게 아닐 때만 센다 (위 설명)
    #
    # ⚠️ ㅋ 부재와 이모지 부재는 **하나의 신호다** (2026-08-20 실측). 진지한 메시지에는
    # 둘 다 없는 게 당연해서, 따로 세면 ㅋ·이모지를 즐겨 쓰는 커플의 **모든 평범한 진지
    # 메시지가 신호 2개**로 `ABRUPT_ALONE_SIGNALS` 를 채운다 — "점심은 집에 있는거
    # 먹었어"가 게이트를 지나 판정 LLM 편차에 노출됐고 교정 카드까지 떴다.
    # 짧아짐과 ㅋ·이모지 부재를 묶은 것과 같은 원리다("상관된 신호를 따로 세지 않는다").
    if not shortened:
        lightness_gone: list[str] = []
        if profile.laugh_per_msg >= LAUGH_BASELINE and not _LAUGH_RE.search(text):
            lightness_gone.append(f"ㅋ/ㅎ {profile.laugh_per_msg:.1f}개 → 0개")
        if profile.emoji_rate >= EMOJI_BASELINE and not _EMOJI_RE.search(text):
            lightness_gone.append(f"이모지 {profile.emoji_rate:.0%} → 없음")
        if lightness_gone:
            signals.append("평소의 가벼움 표지가 사라짐 — " + " · ".join(lightness_gone))

    if profile.avg_length >= 8:
        if count_short:
            signals.append(f"평소 {profile.avg_length:.0f}자 → 이번 {length}자로 짧아짐")
        elif length >= profile.avg_length * LONG_RATIO:
            signals.append(f"평소 {profile.avg_length:.0f}자 → 이번 {length}자로 길어짐")

    return signals


def _repetition_flag(messages: list[Message], speaker: str) -> str | None:
    """같은 말을 거듭 밀어붙이는 반복 ("전화 받아 - 받아 - 받으라고").

    ⚠️ **공유 단어가 있다는 것만으로는 반복이 아니다** (2026-08-20 실측 — "오늘 재택이라
    집에 있어" / "점심은 집에 있는거 먹었어"가 '집에' 하나로 잡혀 평온한 잡담에 교정
    카드가 떴다). 밀어붙임은 **메시지가 그 말로 채워져 있는 것**이다 — 인접한 두 짧은
    메시지에서 공유 토큰이 양쪽 모두의 절반 이상을 차지할 때만 인정한다.
    (예전 규칙은 전체 교집합이라 정작 "전화 받아 - 받아 - 받으라고"는 셋의 교집합이
    비어서 못 잡고, 흔한 단어를 공유한 잡담은 잡는 — 의도와 반대인 모양이었다.)
    """
    mine = [m for m in messages if m.sender == speaker][-3:]
    if len(mine) < 2:
        return None
    for a, b in zip(mine, mine[1:]):
        if _norm_len(a.content) > 15 or _norm_len(b.content) > 15:
            continue
        tokens_a = {t for t in _TOKEN_RE.findall(a.content) if len(t) >= 2}
        tokens_b = {t for t in _TOKEN_RE.findall(b.content) if len(t) >= 2}
        shared = tokens_a & tokens_b
        if shared and len(shared) * 2 >= len(tokens_a) and len(shared) * 2 >= len(tokens_b):
            return f"'{sorted(shared)[0]}' 를 거듭 밀어붙임"
    return None


def check_tone_gate(
    messages: list[Message], profile: SpeakerProfile | None = None
) -> ToneGateResult:
    """방금 전송된 메시지 하나를 본다. 확정하지 않고 후보만 잡는다."""
    if not messages:
        return ToneGateResult(triggered=False)

    ordered = sorted(messages, key=lambda m: m.sent_at)
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
        # 규격서 7장 triggerMessageIds — 실제 기능 발동과 관련된 기준 메시지.
        # 말투 교정은 "방금 전송된 메시지 하나"를 보므로 언제나 1개다.
        message_id=last.message_id if flags else None,
        flags=flags,
    )


# --------------------------------------------------------------------------
# LLM 맥락 판정 + 생성
# --------------------------------------------------------------------------
def _context_block(messages: list[Message], profile: SpeakerProfile, gate: ToneGateResult) -> str:
    ordered = sorted(messages, key=lambda m: m.sent_at)
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
    """상황 진단 + 대체 문장 + 그렇게 읽히는 이유를 만든다."""
    body = _context_block(messages, profile, gate)
    body += f"\n\n## 판정된 감정 온도\n{judged.emotion}"
    return ask(ToneSuggestLLMOutput, load_prompt("tone_suggest"), body)
