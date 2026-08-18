"""후보 기능 — 유튜브 영상 추천.

관계 고민 신호가 감지되면 상황에 맞는 영상 **1개**를 카드로 제시한다.

    [룰 프리필터] → [LLM 고민 유형 분류 + 검색어] → [유튜브 검색 + 베스트 댓글]
    → [LLM 적합도 검증 + 1개 선정] → [근거 문구]

**영상은 제목·썸네일만으로 고르지 않는다.** 베스트 댓글 3개를 함께 읽어 영상이 실제로
무엇을 말하는지 확인한 뒤 선택한다. 연애 유튜브는 제목이 자극적이고 내용이 딴판인 경우가
흔해서, 제목만 믿으면 "화해하고 싶은 사람"에게 "이런 남자는 걸러라" 영상을 던지게 된다.

**침묵이 기본값이다.** 검색 결과가 없거나 후보 전원이 부적합하면 아무것도 내보내지 않는다.
아무 영상이나 던지지 않는다 (명세 예외 처리).

룰 프리필터를 둔 이유: 명세는 "LLM 맥락 이해가 제일 중요"라고 했고 그건 맞다. 다만 분석
요청마다 무조건 LLM 을 태우면 평범한 잡담에도 분류 호출이 나간다. 갈등·고민의 흔적이
전혀 없으면 거기서 끊고, 흔적이 있을 때만 LLM 이 판단한다. 판정 자체는 여전히 LLM 이 한다.
"""

from __future__ import annotations

import re

from worker import ytapi
from worker.copy import YOUTUBE_GUIDE, YOUTUBE_TOPIC_GUIDE
from worker.llm import ask, load_prompt
from worker.models import (
    AiResult,
    ConcernLLMOutput,
    Message,
    TopicLLMOutput,
    VideoPickLLMOutput,
    YoutubeResultData,
    to_key,
)
from worker.text import format_transcript
from worker.ytapi import Video

# 관계 고민 신호. 확정이 아니라 "LLM 에게 물어볼 만한가"를 가른다.
_CONCERN_RE = re.compile(
    # ① 갈등 직후 · 냉각기
    r"(싸웠|다퉜|화났|화나|삐졌|삐짐|짜증|서운|속상|답답|섭섭|"
    # ② 사과 의향은 있는데 방법을 모름
    r"미안|사과|어떻게\s*풀|어떡하지|어떻게\s*해야|모르겠어|풀고\s*싶|"
    # ③ 반복되는 같은 갈등
    r"또\s*(그|이런|싸)|맨날|항상|매번|늘\s*그|"
    # ④ 관계 자체에 대한 회의
    r"우리\s*왜|이럴\s*거면|헤어|지친다|지쳤|힘들다|힘들어|"
    # ⑤ 상대를 이해 못 하겠다는 표현
    r"이해가\s*안|이해\s*못|왜\s*그러는지|무슨\s*생각|말이\s*안\s*통)"
)


def check_concern_gate(messages: list[Message]) -> list[int]:
    """고민 신호가 있는 메시지 id. 없으면 빈 목록 → LLM 을 부르지 않는다.

    **활성 세그먼트를 받는다.** 범위는 분절이 정한다 (`docs/design.md` 1부 7장).
    """
    if not messages:
        return []
    recent = sorted(messages, key=lambda m: m.sent_at)
    return [m.message_id for m in recent if _CONCERN_RE.search(m.content)]


# --------------------------------------------------------------------------
# 룰 프리필터 ② — 화제가 뚜렷한가 (명세 "공통 관심 주제")
# --------------------------------------------------------------------------
# 고민 어휘처럼 목록으로 열거할 수 없다. "먹방" · "왁뿌볼" · "캠핑" … 끝이 없다.
# 그래서 **어휘가 아니라 모양**을 본다 — 같은 낱말이 여러 발화에 걸쳐 반복되면
# 화제가 하나로 모여 있다는 뜻이다.
#
# 이것도 확정이 아니라 **LLM 에게 물어볼 만한가**만 가른다. 잡담마다 분류 호출이
# 나가지 않게 막는 자리다 (고민 갈래의 프리필터와 같은 역할).
TOPIC_MIN_REPEAT = 2   # 몇 개의 발화에 걸쳐 같은 낱말이 나와야 하는가
TOPIC_MIN_LEN = 2      # 낱말 최소 길이 (한 글자는 조사·감탄사가 섞인다)

_WORD_RE = re.compile(r"[가-힣A-Za-z0-9]+")

# 어느 대화에나 나오는 말. 이게 반복된다고 화제가 뚜렷한 건 아니다.
_TOPIC_STOP = {
    "그래", "그거", "그건", "그럼", "근데", "진짜", "완전", "너무", "그냥", "약간",
    "오늘", "내일", "어제", "지금", "아까", "이따", "나중", "요즘", "저번", "다음",
    "우리", "너는", "나는", "내가", "네가", "우린", "자기", "오빠",
    "생각", "얘기", "이야기", "느낌", "정도", "때문", "이제", "아직", "많이", "조금",
    "하는", "하고", "했어", "해서", "이거", "저거", "뭐야", "맞아", "그리고",
}


def _topic_words(text: str) -> set[str]:
    return {
        w for w in _WORD_RE.findall(text)
        if len(w) >= TOPIC_MIN_LEN and w not in _TOPIC_STOP
    }


def check_topic_gate(messages: list[Message]) -> list[int]:
    """화제가 뚜렷한 발화 id. 없으면 빈 목록 → LLM 을 부르지 않는다.

    같은 낱말이 `TOPIC_MIN_REPEAT` 개 이상의 **서로 다른 발화**에 나오면 뚜렷하다고 본다.
    한 발화 안에서 같은 말을 두 번 하는 것은 세지 않는다.
    """
    if not messages:
        return []

    recent = sorted(messages, key=lambda m: m.sent_at)
    seen: dict[str, list[int]] = {}
    for m in recent:
        for word in _topic_words(m.content):
            seen.setdefault(word, []).append(m.message_id)

    ids: set[int] = set()
    for word, hits in seen.items():
        if len(set(hits)) >= TOPIC_MIN_REPEAT:
            ids.update(hits)
    return sorted(ids)


def classify_topic(messages: list[Message]) -> TopicLLMOutput:
    return ask(TopicLLMOutput, load_prompt("yt_topic"), format_transcript(messages))


# --------------------------------------------------------------------------
# LLM — 고민 유형 분류 + 검색어 생성
# --------------------------------------------------------------------------
def classify_concern(messages: list[Message]) -> ConcernLLMOutput:
    return ask(ConcernLLMOutput, load_prompt("yt_concern"), format_transcript(messages))


# --------------------------------------------------------------------------
# LLM — 후보 + 베스트 댓글을 읽고 1개 선정
# --------------------------------------------------------------------------
def _candidate_block(videos: list[Video]) -> str:
    blocks: list[str] = []
    for i, v in enumerate(videos):
        comments = "\n".join(f"    · {c}" for c in v.comments) or "    · (없음)"
        blocks.append(
            "\n".join(
                [
                    f"### 후보 {i}",
                    f"- 제목: {v.title}",
                    f"- 채널: {v.channel_name}",
                    f"- 설명: {' '.join(v.description.split())[:200]}",
                    "- 베스트 댓글 3개:",
                    comments,
                ]
            )
        )
    return "\n\n".join(blocks)


def pick_video(
    messages: list[Message], concern: ConcernLLMOutput, videos: list[Video]
) -> VideoPickLLMOutput:
    body = "\n".join(
        [
            "## 최근 대화",
            format_transcript(messages),
            "",
            "## 판정된 고민 유형",
            f"- 유형: {concern.concern}",
            f"- 관계 단계: {concern.stage}",
            f"- 판정 근거: {concern.note}",
            "",
            "## 후보 영상",
            _candidate_block(videos),
        ]
    )
    return ask(VideoPickLLMOutput, load_prompt("yt_pick"), body)


# --------------------------------------------------------------------------
# 조립
# --------------------------------------------------------------------------
def _result(
    video: Video,
    reason: str,
    trigger_message_ids: list[int],
    guide: str,
    target: str | None,
) -> AiResult:
    return AiResult(
        result_type="YOUTUBE_RECOMMENDATION",
        visibility_type="INDIVIDUAL" if target else "COUPLE",
        target_participant=to_key(target) if target else None,  # type: ignore[arg-type]
        content_type="MIXED",
        trigger_message_ids=trigger_message_ids,
        result_data=YoutubeResultData(
            guide_message=guide,
            video_id=video.video_id,
            title=video.title,
            video_url=video.url,
            thumbnail_url=video.thumbnail_url,
            channel_name=video.channel_name,
            recommendation_reason=reason.strip(),
            video_summary=video.summary(),
        ),
    )



def pick_topic_video(
    messages: list[Message], topic: TopicLLMOutput, videos: list[Video]
) -> VideoPickLLMOutput:
    """일상 화제 갈래의 선정. 고민 갈래와 **프롬프트가 다르다.**

    보는 기준이 다르기 때문이다 — 고민은 "이 상황에 도움이 되는가", 화제는 "그 소재를
    실제로 다루는가". 한 프롬프트에 둘을 넣으면 판정이 흐려진다.
    """
    body = "\n".join(
        [
            "## 최근 대화",
            format_transcript(messages),
            "",
            "## 판정된 화제",
            f"- 화제: {topic.topic}",
            f"- 판정 근거: {topic.note}",
            "",
            "## 후보 영상",
            _candidate_block(videos),
        ]
    )
    return ask(VideoPickLLMOutput, load_prompt("yt_pick_topic"), body)


def to_result(
    concern: ConcernLLMOutput,
    video: Video,
    reason: str,
    trigger_message_ids: list[int],
) -> AiResult:
    """고민 갈래 — 노출 범위를 LLM 이 정한다 (개별/공통)."""
    individual = concern.scope == "individual" and concern.target in ("A", "B")
    return _result(
        video, reason, trigger_message_ids,
        guide=YOUTUBE_GUIDE,
        target=concern.target if individual else None,
    )


def to_topic_result(video: Video, reason: str, trigger_message_ids: list[int]) -> AiResult:
    """일상 화제 갈래 — **항상 COUPLE 이다.**

    화제는 둘이 같이 하던 얘기고, 감춰야 할 이유가 없다. 개별로 띄우면 상대만 모르는
    영상이 생겨서 "왜 나만 안 보이지"가 된다 — 개별 노출은 지적성 피드백을 감추기 위한
    장치지 취향 추천을 위한 게 아니다.
    """
    return _result(
        video, reason, trigger_message_ids,
        guide=YOUTUBE_TOPIC_GUIDE,
        target=None,
    )


def available() -> bool:
    return ytapi.available()


def find_candidates(queries: list[str]) -> list[Video]:
    return ytapi.find_candidates(queries)
