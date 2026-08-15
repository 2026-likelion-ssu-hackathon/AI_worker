"""룰 게이트 — 트리거 ①②③⑤ 감지 + scope 결정.

여기서는 **감지만** 한다. 사용자에게 보여줄 문구는 절대 만들지 않는다.
④(일상 보고형 반복)와 바쁨 판별은 룰로 확정할 수 없어 `needs_llm=True` 로 넘긴다.

톤 판정은 전부 **화자 개인 베이스라인 대비**다. 원래 단답형인 사람에게 절대 기준을
적용하면 상시 트리거되기 때문이다.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import timedelta
from statistics import mean

from worker.models import GateResult, Message, Scope, Speaker

# 최근 대화 판정 윈도우 (메시지 개수). 6개 ≈ 3턴 왕복.
WINDOW = 6

# ① 종료형 단답이 몇 개 연속되면 트리거인가
PINGPONG_STREAK = 3

# ③ 한쪽 발화 비중 임계
ONE_SIDED_CHAR_RATIO = 0.75
# ③ 반대쪽이 리액션만 하는 비율 임계
ONE_SIDED_REACTION_RATIO = 0.8

# scope 결정 임계 (CLAUDE.md 표)
SCOPE_TALK_RATIO = 0.70
SCOPE_SHORT_RATIO = 0.70

# ⑤ 대화 중 정체 판정 시간
STALL_AFTER = timedelta(minutes=20)

# ④ 후보 — 날짜별 어휘 겹침이 이 이상이면 "동일 패턴 반복" 의심
ROUTINE_JACCARD = 0.30

_PUNCT_RE = re.compile(r"[\s.,!?？~…\-·^;:\"'()\[\]]+")
_JAMO_ONLY_RE = re.compile(r"^[ㄱ-ㅎㅏ-ㅣ]+$")
_EMOJI_ONLY_RE = re.compile(
    r"^[\U0001F000-\U0001FAFF\U00002600-\U000027BF←-⇿⬀-⯿]+$"
)

# 되묻는 문장 판별
_QUESTION_MARK_RE = re.compile(r"[?？]")
_INTERROGATIVE_RE = re.compile(
    r"(뭐|뭘|무슨|무엇|어디|언제|왜|누구|누가|어때|어땠|어떻|어떤|얼마|몇|그치|맞지)"
)
_QUESTION_ENDING_RE = re.compile(r"(니|냐|까|을래|ㄹ래|래|나요|남|는지|을지)$")

# 리액션 어휘 — 내용 없이 반응만 하는 말
_REACTIONS = {
    "ㅇㅇ", "ㅇㅋ", "ㅋㅋ", "ㅎㅎ", "ㅠㅠ", "ㅜㅜ", "ㄱㅅ", "ㅇㅎ",
    "응", "웅", "넹", "네", "예", "어", "음", "아", "오", "허",
    "그래", "그램", "그렇구나", "그렇군", "그런가", "그렇지", "그러네", "그랬구나",
    "알았어", "알겠어", "굿", "오키", "오케이", "ok", "okay", "ㅇㅈ", "헐", "대박",
}

# 종료형 단답에서만 쓰이는 어휘 (리액션 + 대화를 닫는 표현)
_CLOSERS = _REACTIONS | {"그러게", "그런듯", "몰라", "글쎄", "별로", "그냥", "ㅇㅇㅇ"}

# 바쁨을 알리는 표현 — 룰로 확정하지 않고 LLM 에 넘긴다
_BUSY_RE = re.compile(
    r"(회의|미팅|수업|강의|출근|퇴근길|운전\s*중|일\s*중|근무|바빠|바쁨|바쁘|정신없|"
    r"이따\s*(톡|연락|얘기|말)|나중에\s*(톡|연락|얘기)|끝나고\s*(톡|연락)|이따가)"
)


# --------------------------------------------------------------------------
# 문장 단위 판별
# --------------------------------------------------------------------------
def _norm(text: str) -> str:
    return _PUNCT_RE.sub("", text).strip()


def is_question(text: str) -> bool:
    """되묻는 문장인가."""
    if _QUESTION_MARK_RE.search(text):
        return True
    body = _norm(text)
    if not body:
        return False
    # "그래" 처럼 의문 어미와 겹치는 리액션 어휘를 되묻는 문장으로 세지 않는다
    if is_reaction(text):
        return False
    if _QUESTION_ENDING_RE.search(body):
        return True
    return bool(_INTERROGATIVE_RE.search(body))


def is_reaction(text: str) -> bool:
    """내용 없이 반응만 하는 말인가."""
    body = _norm(text).lower()
    if not body:
        return True
    if body in _REACTIONS:
        return True
    if _JAMO_ONLY_RE.match(body) or _EMOJI_ONLY_RE.match(body):
        return True
    # "ㅋㅋ 그래" 처럼 리액션 어휘만으로 이루어진 경우
    parts = [p for p in _PUNCT_RE.split(text.lower()) if p]
    return bool(parts) and all(p in _REACTIONS for p in parts)


def is_busy_signal(text: str) -> bool:
    return bool(_BUSY_RE.search(text))


# --------------------------------------------------------------------------
# 화자별 베이스라인
# --------------------------------------------------------------------------
def _baselines(messages: list[Message]) -> dict[str, float]:
    """화자별 평소 발화 길이. 최근 윈도우를 제외한 앞부분에서 잡는다."""
    history = messages[:-WINDOW] if len(messages) > WINDOW + 2 else messages
    lengths: dict[str, list[int]] = defaultdict(list)
    for m in history:
        lengths[m.sender].append(len(_norm(m.content)))
    return {s: max(mean(v), 1.0) for s, v in lengths.items() if v}


def _is_short(msg: Message, baselines: dict[str, float]) -> bool:
    """개인 베이스라인 대비 짧아진 발화인가."""
    if is_question(msg.content):
        return False
    if is_reaction(msg.content):
        return True
    base = baselines.get(msg.sender, 12.0)
    return len(_norm(msg.content)) <= max(4.0, base * 0.5)


def _is_short_closing(msg: Message, baselines: dict[str, float]) -> bool:
    """① 종료형 단답 — 짧고, 되묻지 않고, 대화를 닫는 말."""
    if not _is_short(msg, baselines):
        return False
    body = _norm(msg.content).lower()
    if is_reaction(msg.content):
        return True
    parts = [p for p in _PUNCT_RE.split(msg.content.lower()) if p]
    return body in _CLOSERS or (bool(parts) and all(p in _CLOSERS for p in parts))


# --------------------------------------------------------------------------
# scope 결정 (CLAUDE.md 표)
# --------------------------------------------------------------------------
def decide_scope(messages: list[Message]) -> tuple[Scope, Speaker | None, str]:
    window = messages[-WINDOW:] if len(messages) > WINDOW else messages
    baselines = _baselines(messages)

    chars: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    shorts: dict[str, int] = defaultdict(int)
    for m in window:
        chars[m.sender] += len(_norm(m.content))
        counts[m.sender] += 1
        if _is_short(m, baselines):
            shorts[m.sender] += 1

    speakers: list[Speaker] = ["A", "B"]
    total = sum(chars.values()) or 1
    talk = {s: chars[s] / total for s in speakers}
    short_ratio = {s: (shorts[s] / counts[s] if counts[s] else 0.0) for s in speakers}

    quiet = [s for s in speakers if counts[s] and short_ratio[s] >= SCOPE_SHORT_RATIO]

    # 양쪽 다 조용
    if len(quiet) == 2:
        return "common", None, "양쪽 다 단답"

    # 한쪽 발화 70% 이상 → 말 거는 쪽
    talker = max(speakers, key=lambda s: talk[s])
    if talk[talker] >= SCOPE_TALK_RATIO:
        return "individual", talker, f"{talker} 발화 {talk[talker]:.0%}"

    # 한쪽 단답 70% 이상 → 단답 보내는 쪽
    if len(quiet) == 1:
        s = quiet[0]
        return "individual", s, f"{s} 단답 {short_ratio[s]:.0%}"

    return "common", None, "발화 비중 균형"


# --------------------------------------------------------------------------
# ④ 후보 — 동일 패턴 반복 의심
# --------------------------------------------------------------------------
def _looks_routine(messages: list[Message]) -> str | None:
    """날짜별 어휘가 크게 겹치면 일상 보고형 반복을 의심한다.

    확정은 하지 않는다. LLM 에 넘길지 여부만 정한다.
    """
    by_day: dict[str, set[str]] = defaultdict(set)
    for m in messages:
        tokens = {t for t in _PUNCT_RE.split(m.content) if len(t) >= 2}
        by_day[m.ts.date().isoformat()] |= tokens

    days = sorted(by_day)
    if len(days) < 2:
        return None

    for prev, cur in zip(days, days[1:]):
        a, b = by_day[prev], by_day[cur]
        if not a or not b:
            continue
        jaccard = len(a & b) / len(a | b)
        if jaccard >= ROUTINE_JACCARD:
            return f"{prev}↔{cur} 어휘 겹침 {jaccard:.0%}"
    return None


# --------------------------------------------------------------------------
# 게이트 본체
# --------------------------------------------------------------------------
def check_gate(
    messages: list[Message],
    now=None,
    online: list[Speaker] | None = None,
) -> GateResult:
    if len(messages) < 2:
        return GateResult(triggered=False, detail="메시지 부족")

    messages = sorted(messages, key=lambda m: m.ts)
    window = messages[-WINDOW:] if len(messages) > WINDOW else messages
    baselines = _baselines(messages)

    # 바쁨 표현이 있으면 룰로 확정하지 않는다. 진짜인지 핑계인지는 LLM 판단.
    for m in window:
        if is_busy_signal(m.content):
            return GateResult(
                triggered=False,
                needs_llm=True,
                detail=f"바쁨 표현 감지: {m.sender} \"{m.content}\"",
            )

    scope, target, scope_detail = decide_scope(messages)

    # ① 단답 핑퐁 — 종료형 단답 3턴 연속
    streak = 0
    for m in reversed(messages):
        if _is_short_closing(m, baselines):
            streak += 1
        else:
            break
    if streak >= PINGPONG_STREAK:
        return GateResult(
            triggered=True,
            trigger="short_pingpong",
            scope=scope,
            target=target,
            detail=f"종료형 단답 {streak}턴 연속 / {scope_detail}",
        )

    # ③ 한쪽만 발화 — 비중 75% + 반대쪽은 리액션만
    chars: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    reactions: dict[str, int] = defaultdict(int)
    for m in window:
        chars[m.sender] += len(_norm(m.content))
        counts[m.sender] += 1
        if is_reaction(m.content):
            reactions[m.sender] += 1

    total = sum(chars.values()) or 1
    for talker, other in (("A", "B"), ("B", "A")):
        if counts[other] < 2:
            continue
        share = chars[talker] / total
        reaction_ratio = reactions[other] / counts[other]
        if share >= ONE_SIDED_CHAR_RATIO and reaction_ratio >= ONE_SIDED_REACTION_RATIO:
            return GateResult(
                triggered=True,
                trigger="one_sided",
                scope="individual",
                target=talker,  # type: ignore[arg-type]
                detail=f"{talker} 발화 {share:.0%} / {other} 리액션만 {reaction_ratio:.0%}",
            )

    # ② 질문 없는 대답 — 되묻는 문장 0개로 3턴 경과
    if len(window) >= WINDOW and not any(is_question(m.content) for m in window):
        return GateResult(
            triggered=True,
            trigger="no_question",
            scope=scope,
            target=target,
            detail=f"최근 {len(window)}개 메시지에 되묻는 문장 0개 / {scope_detail}",
        )

    # ⑤ 대화 중 정체 — 마지막 메시지 후 20분 경과, 양쪽 접속 중
    online = online or ["A", "B"]
    if now is not None and {"A", "B"} <= set(online):
        idle = now - messages[-1].ts
        if idle >= STALL_AFTER:
            return GateResult(
                triggered=True,
                trigger="stall",
                scope=scope,
                target=target,
                detail=f"마지막 메시지 후 {int(idle.total_seconds() // 60)}분 경과 / {scope_detail}",
            )

    # 룰은 정상으로 봤지만 동일 패턴 반복이 의심되면 LLM 에 넘긴다 (트리거 ④)
    routine = _looks_routine(messages)
    if routine:
        return GateResult(
            triggered=False,
            needs_llm=True,
            detail=f"동일 패턴 반복 의심: {routine}",
        )

    return GateResult(triggered=False, detail="룰 미발동")
