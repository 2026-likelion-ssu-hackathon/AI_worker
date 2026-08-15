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
from worker.copy import YOUTUBE_GUIDE
from worker.llm import ask, load_prompt
from worker.models import (
    AiResult,
    ConcernLLMOutput,
    Message,
    VideoPickLLMOutput,
    YoutubeResultData,
    to_key,
)
from worker.text import format_transcript
from worker.ytapi import Video

# 고민 신호를 찾을 최근 메시지 범위
WINDOW = 12

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
    """고민 신호가 있는 메시지 id. 없으면 빈 목록 → LLM 을 부르지 않는다."""
    if not messages:
        return []
    recent = sorted(messages, key=lambda m: m.sent_at)[-WINDOW:]
    return [m.message_id for m in recent if _CONCERN_RE.search(m.content)]


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
def to_result(
    concern: ConcernLLMOutput,
    video: Video,
    reason: str,
    trigger_message_ids: list[int],
) -> AiResult:
    individual = concern.scope == "individual" and concern.target in ("A", "B")
    return AiResult(
        result_type="YOUTUBE_RECOMMENDATION",
        visibility_type="INDIVIDUAL" if individual else "COUPLE",
        target_participant=to_key(concern.target) if individual else None,  # type: ignore[arg-type]
        content_type="MIXED",
        trigger_message_ids=trigger_message_ids,
        result_data=YoutubeResultData(
            guide_message=YOUTUBE_GUIDE,
            video_id=video.video_id,
            title=video.title,
            video_url=video.url,
            thumbnail_url=video.thumbnail_url,
            channel_name=video.channel_name,
            recommendation_reason=reason.strip(),
            video_summary=video.summary(),
        ),
    )


def available() -> bool:
    return ytapi.available()


def find_candidates(queries: list[str]) -> list[Video]:
    return ytapi.find_candidates(queries)
