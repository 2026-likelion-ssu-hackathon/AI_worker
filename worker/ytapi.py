"""YouTube Data API v3 — 실재하는 영상 + 베스트 댓글.

명세의 핵심은 **"제목·썸네일만으로 고르지 않는다"** 이다. 베스트 댓글 3개를 함께 읽어
영상이 실제로 무엇을 말하는지 확인한 뒤 고른다. 그래서 여기서 댓글까지 같이 가져온다.

댓글이 비활성화된 영상은 맥락 검증을 할 수 없으므로 **후보에서 제외한다** (명세 예외 처리).

## 쿼터

무료 할당량은 하루 10,000 units 다.

    search.list         100 units   ← 비싸다. 검색어당 1회만 쓴다
    videos.list           1 unit
    commentThreads.list   1 unit    ← 후보 수만큼

`search.list` 가 전체 비용을 지배한다. 추천 1건 비용은 검색어를 몇 개 쓰느냐로 결정된다.

    SEARCH_QUERIES=1 → 100 + 1 + 5 ≈ 106 units → 하루 약 94회
    SEARCH_QUERIES=2 → 200 + 1 + 5 ≈ 206 units → 하루 약 48회

기본값은 1이다. 검색어를 2개 쓰면 후보 다양성이 늘지만 하루 실행 횟수가 절반이 된다.
개발 중 반복 실행 + 시연 리허설까지 감안하면 48회는 하루 만에 소진될 수 있는 숫자다.
`YOUTUBE_API_KEY` 를 비워두면 이 기능만 조용히 미발동된다.

문서: https://developers.google.com/youtube/v3/docs
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx

API_ROOT = "https://www.googleapis.com/youtube/v3"
TIMEOUT = 8.0

# 후보를 몇 개까지 볼지. 늘리면 쿼터(commentThreads)와 프롬프트 길이가 같이 늘어난다.
MAX_CANDIDATES = 5
TOP_COMMENTS = 3

# 검색어를 몇 개 쓸지. **쿼터를 지배하는 값이다** — 하나 늘 때마다 100 units 씩 붙는다.
# LLM 이 검색어를 1~3개 만들지만 앞에서부터 이만큼만 쓴다.
SEARCH_QUERIES = 1

# 영상 설명은 통째로 넣으면 프롬프트를 잡아먹는다. 앞부분만 쓴다.
DESCRIPTION_CHARS = 300


@dataclass
class Video:
    video_id: str
    title: str
    channel_name: str
    thumbnail_url: str
    description: str
    comments: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    def summary(self) -> str | None:
        """규격서 10장 `videoSummary` (선택 필드).

        ⏸️ **화면에서는 더 이상 쓰지 않는다.** 2026-08-16 기능 명세 수정으로
        `[영상 요약 정보]` 자리가 없어지고 근거 문구(`recommendationReason`)가 그 역할을
        흡수했다. 규격서에서 필드를 빼자고 제안해 둔 상태다(`docs/contract-review.md` 4번).

        백엔드가 저장해 둘 수 있으니 값이 있으면 채운다. **유튜브가 주는 영상 설명을
        그대로 자른 것**이라 AI 생성 금지 규칙에 걸리지 않고 환각도 없다.
        설명이 빈 영상이 흔해서 실제로는 자주 None 이다.
        """
        body = " ".join(self.description.split())
        if not body:
            return None
        return body[:DESCRIPTION_CHARS].rstrip() + ("…" if len(body) > DESCRIPTION_CHARS else "")


def available() -> bool:
    return bool(os.getenv("YOUTUBE_API_KEY"))


def _get(path: str, **params) -> dict | None:
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        return None
    try:
        res = httpx.get(f"{API_ROOT}/{path}", params={**params, "key": key}, timeout=TIMEOUT)
        res.raise_for_status()
        return res.json()
    except Exception:  # noqa: BLE001 — 검색 실패는 오류가 아니라 미발동이다
        return None


def _search_ids(query: str, limit: int) -> list[str]:
    data = _get(
        "search",
        part="id",
        q=query,
        type="video",
        maxResults=limit,
        order="relevance",
        relevanceLanguage="ko",
        regionCode="KR",
        safeSearch="moderate",
        videoEmbeddable="true",
    )
    if not data:
        return []
    return [
        item["id"]["videoId"]
        for item in data.get("items", [])
        if item.get("id", {}).get("videoId")
    ]


def _details(video_ids: list[str]) -> list[Video]:
    if not video_ids:
        return []
    data = _get("videos", part="snippet", id=",".join(video_ids))
    if not data:
        return []

    videos: list[Video] = []
    for item in data.get("items", []):
        snippet = item.get("snippet", {})
        thumbs = snippet.get("thumbnails", {})
        thumb = (
            thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}
        ).get("url")
        if not thumb:
            continue
        videos.append(
            Video(
                video_id=item["id"],
                title=snippet.get("title", ""),
                channel_name=snippet.get("channelTitle", ""),
                thumbnail_url=thumb,
                description=snippet.get("description", ""),
            )
        )
    return videos


def _top_comments(video_id: str) -> list[str] | None:
    """베스트 댓글. 댓글이 꺼져 있으면 None (후보에서 제외해야 한다는 뜻)."""
    data = _get(
        "commentThreads",
        part="snippet",
        videoId=video_id,
        order="relevance",
        maxResults=TOP_COMMENTS,
        textFormat="plainText",
    )
    if data is None:
        return None
    comments = [
        item["snippet"]["topLevelComment"]["snippet"].get("textDisplay", "")
        for item in data.get("items", [])
    ]
    comments = [" ".join(c.split())[:200] for c in comments if c.strip()]
    return comments or None


def find_candidates(queries: list[str], limit: int = MAX_CANDIDATES) -> list[Video]:
    """검색어들로 후보 영상을 모으고 베스트 댓글까지 채운다.

    댓글을 못 읽는 영상은 버린다 — 맥락 검증이 이 기능의 안전장치이기 때문이다.
    """
    if not available() or not queries:
        return []

    seen: set[str] = set()
    ids: list[str] = []
    # 검색어 1개당 100 units. 넉넉히 뽑아 뒤에서 거른다.
    per_query = max(2, limit)
    for q in queries[:SEARCH_QUERIES]:
        for vid in _search_ids(q, per_query):
            if vid not in seen:
                seen.add(vid)
                ids.append(vid)

    candidates: list[Video] = []
    for video in _details(ids):
        comments = _top_comments(video.video_id)
        if comments is None:
            continue  # 댓글 비활성 → 맥락 검증 불가 → 제외
        video.comments = comments
        candidates.append(video)
        if len(candidates) >= limit:
            break
    return candidates
