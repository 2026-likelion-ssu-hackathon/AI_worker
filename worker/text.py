"""한국어 채팅 문장 판별 유틸.

원래 `gate.py` 안에 있던 것들이다. 대화 소재 후보가 파킹되면서 게이트가 빠졌는데,
`retrieve.py`(검색 질의 구성)와 데이트 코스 트리거가 같은 판별을 필요로 해서 여기로 뺐다.

`parked/gate.py` 를 되살릴 때도 이 모듈을 import 하면 된다 — 로직은 그대로 옮겨왔다.
"""

from __future__ import annotations

import re

from worker.models import Message

# LLM 에 넘기는 대화 로그 최대 길이
MAX_TRANSCRIPT = 60

_PUNCT_RE = re.compile(r"[\s.,!?？~…\-·^;:\"'()\[\]]+")
_JAMO_ONLY_RE = re.compile(r"^[ㄱ-ㅎㅏ-ㅣ]+$")
_EMOJI_ONLY_RE = re.compile(r"^[\U0001F000-\U0001FAFF\U00002600-\U000027BF←-⇿⬀-⯿]+$")
_SPACE_RE = re.compile(r"\s+")

# 되묻는 문장 판별
_QUESTION_MARK_RE = re.compile(r"[?？]")
_INTERROGATIVE_RE = re.compile(
    r"(뭐|뭘|무슨|무엇|어디|언제|왜|누구|누가|어때|어땠|어떻|어떤|얼마|몇|그치|맞지)"
)
_QUESTION_ENDING_RE = re.compile(r"(니|냐|까|을래|ㄹ래|래|나요|남|는지|을지)$")

# 리액션 어휘 — 내용 없이 반응만 하는 말
REACTIONS = {
    "ㅇㅇ", "ㅇㅋ", "ㅋㅋ", "ㅎㅎ", "ㅠㅠ", "ㅜㅜ", "ㄱㅅ", "ㅇㅎ",
    "응", "웅", "넹", "네", "예", "어", "음", "아", "오", "허",
    "그래", "그램", "그렇구나", "그렇군", "그런가", "그렇지", "그러네", "그랬구나",
    "알았어", "알겠어", "굿", "오키", "오케이", "ok", "okay", "ㅇㅈ", "헐", "대박",
}


def norm(text: str) -> str:
    """구두점·공백을 걷어낸 본문."""
    return _PUNCT_RE.sub("", text).strip()


def norm_len(text: str) -> int:
    """공백을 뺀 글자 수. 메시지 길이 비교의 기준."""
    return len(_SPACE_RE.sub("", text))


def is_reaction(text: str) -> bool:
    """내용 없이 반응만 하는 말인가."""
    body = norm(text).lower()
    if not body:
        return True
    if body in REACTIONS:
        return True
    if _JAMO_ONLY_RE.match(body) or _EMOJI_ONLY_RE.match(body):
        return True
    # "ㅋㅋ 그래" 처럼 리액션 어휘만으로 이루어진 경우
    parts = [p for p in _PUNCT_RE.split(text.lower()) if p]
    return bool(parts) and all(p in REACTIONS for p in parts)


def is_question(text: str) -> bool:
    """되묻는 문장인가."""
    if _QUESTION_MARK_RE.search(text):
        return True
    body = norm(text)
    if not body:
        return False
    # "그래" 처럼 의문 어미와 겹치는 리액션 어휘를 되묻는 문장으로 세지 않는다
    if is_reaction(text):
        return False
    if _QUESTION_ENDING_RE.search(body):
        return True
    return bool(_INTERROGATIVE_RE.search(body))


def format_transcript(messages: list[Message], limit: int = MAX_TRANSCRIPT) -> str:
    """LLM 에 넘길 대화 로그.

    날짜가 바뀌면 구분선을 넣는다 — "3주 전에 가보고 싶다고 했던" 같은 시점 근거를
    쓰려면 모델이 언제 한 말인지 알아야 한다.
    """
    lines: list[str] = []
    last_day = None
    for m in sorted(messages, key=lambda x: x.sent_at)[-limit:]:
        day = m.sent_at.date().isoformat()
        if day != last_day:
            lines.append(f"--- {day} ---")
            last_day = day
        lines.append(f"[{m.sent_at:%H:%M}] {m.sender}: {m.content}")
    return "\n".join(lines)
