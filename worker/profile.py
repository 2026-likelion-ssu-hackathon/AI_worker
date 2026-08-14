"""개인별 평소 말투 기준선.

말투 교정은 **절대 기준으로 판정하면 안 된다.** 평소 "ㅇㅇ"을 자주 쓰는 커플에게 "ㅇㅇ"은
무례가 아니다. 특정 단어가 아니라 **그 사람의 평소 대비 변화량**을 본다.

기준선은 두 곳에서 온다.
1. `data/speaker_profiles.json` — 3개월치 대화에서 뽑은 시드 (기억 시드와 같은 논리)
2. 시드가 없으면 들어온 대화에서 직접 계산

시드가 필요한 이유는 기억 시드와 같다. 픽스처 한 개짜리 대화만으로는
"평소 ㅋ 3개 → 이번엔 0개" 같은 대비를 만들 수 없다.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from worker import DATA_DIR
from worker.models import Message, Speaker, SpeakerProfile

PROFILE_FILE = DATA_DIR / "speaker_profiles.json"

# 평소 호칭 후보. 상위 2개를 기준선으로 잡고, 여기서 벗어나면 호칭 변화로 본다.
ADDRESS_TERMS = [
    "오빠", "언니", "누나", "형", "자기야", "자기", "여보", "애기야", "애기",
    "야", "너", "니가", "네가", "당신", "그쪽",
]

# 공격적으로 읽히기 쉬운 호칭
HARSH_ADDRESS = {"야", "너", "니가", "네가", "당신", "그쪽"}

_LAUGH_RE = re.compile(r"[ㅋㅎ]")
_EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF]")
_PERIOD_END_RE = re.compile(r"[^.]\.\s*$")  # "..." 은 제외, 단일 마침표 종결만
_SPACE_RE = re.compile(r"\s+")


def _norm_len(text: str) -> int:
    return len(_SPACE_RE.sub("", text))


# 호칭 뒤에 붙을 수 있는 조사. 이것만 허용해서 "너무"가 "너"로, "거야"가 "야"로 잡히지 않게 한다.
_PARTICLES = {"", "는", "가", "도", "의", "한테", "랑", "이랑", "와", "과", "을", "를", "만", "은"}

_WORD_RE = re.compile(r"[가-힣]+")


def addresses_in(text: str) -> list[str]:
    """문장에 등장한 호칭.

    부분 문자열로 찾으면 "거야"의 '야', "너무"의 '너'까지 호칭으로 잡힌다.
    어절 단위로 보고 뒤에 조사만 붙은 경우까지만 인정한다.
    긴 것부터 대조해 '자기야'가 '자기'로 잘리지 않게 한다.
    """
    terms = sorted(ADDRESS_TERMS, key=len, reverse=True)
    found: list[str] = []
    for token in _WORD_RE.findall(text):
        for term in terms:
            if token.startswith(term) and token[len(term):] in _PARTICLES:
                if term not in found:
                    found.append(term)
                break
    return found


def compute_profile(messages: list[Message], speaker: Speaker) -> SpeakerProfile:
    """주어진 대화에서 한 사람의 말투 기준선을 계산한다."""
    mine = [m for m in messages if m.sender == speaker]
    if not mine:
        return SpeakerProfile(speaker=speaker)

    n = len(mine)
    address_counter: Counter[str] = Counter()
    for m in mine:
        address_counter.update(addresses_in(m.content))

    return SpeakerProfile(
        speaker=speaker,
        avg_length=sum(_norm_len(m.content) for m in mine) / n,
        period_rate=sum(bool(_PERIOD_END_RE.search(m.content)) for m in mine) / n,
        laugh_per_msg=sum(len(_LAUGH_RE.findall(m.content)) for m in mine) / n,
        emoji_rate=sum(bool(_EMOJI_RE.search(m.content)) for m in mine) / n,
        top_address=[t for t, _ in address_counter.most_common(2)],
    )


def load_seed_profiles() -> dict[str, SpeakerProfile]:
    if not PROFILE_FILE.exists():
        return {}
    raw = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    return {item["speaker"]: SpeakerProfile(**item) for item in raw}


def resolve_profile(speaker: Speaker, messages: list[Message]) -> SpeakerProfile:
    """시드가 있으면 시드를, 없으면 대화에서 계산한 값을 쓴다."""
    seed = load_seed_profiles().get(speaker)
    return seed if seed is not None else compute_profile(messages, speaker)


def describe(profile: SpeakerProfile) -> str:
    """LLM 프롬프트에 넣을 기준선 요약."""
    lines = [
        f"평소 평균 길이: {profile.avg_length:.0f}자",
        f"마침표로 끝내는 비율: {profile.period_rate:.0%}",
        f"메시지당 ㅋ/ㅎ 개수: {profile.laugh_per_msg:.1f}개",
        f"이모지 사용 비율: {profile.emoji_rate:.0%}",
        f"평소 호칭: {', '.join(profile.top_address) or '없음'}",
    ]
    if profile.conflict_style:
        lines.append(f"화났을 때의 평소 패턴: {profile.conflict_style}")
    return "\n".join(f"- {line}" for line in lines)
