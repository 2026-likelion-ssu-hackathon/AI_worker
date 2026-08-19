"""유튜브 시드 수집 — 쿼터 소진 대비 폴백 데이터 (`data/yt_seed.json`).

런타임에 YouTube API 가 죽으면(쿼터 소진 · 키 없음 · 네트워크) 이 시드에서 후보를
꺼낸다. **영상·링크·댓글 전부 실제 API 산출물이다** — "실재하는 것은 외부 API 가 준
것만 쓴다" 원칙을 빌드 타임으로 앞당긴 것이지 깨는 게 아니다. 기억 시드 27건
(`data/memories.json`)과 같은 패턴이다.

실행:  PYTHONPATH=. .venv/bin/python tools/build_yt_seed.py
비용:  검색어 14개 × 100 units ≈ 1,400 units + 댓글 소량 (하루 쿼터 10,000)

검색어를 여기 손으로 적는 이유 — 빌드 도구라 사람이 결과를 눈으로 검수하고 커밋한다.
'권태기' 같은 금지어는 검색어에도 안 쓴다 (제목이 따라 붙는다).
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from worker.filter import find_banned
from worker.ytapi import available, find_candidates

# 고민 유형(models.ConcernType) → 검색어 2개.
QUERIES: dict[str, list[str]] = {
    "contact": ["연인 연락 문제 해결", "커플 연락 빈도 갈등 조언"],
    "reconcile": ["커플 싸운 후 화해하는 법", "연인과 화해하는 대화법"],
    "hurt": ["연인에게 서운할 때 대처법", "서운한 마음 말하는 방법 연애"],
    "boredom": ["오래 만난 커플 설렘 유지법", "연애 매너리즘 극복"],
    "trust": ["연인 신뢰 쌓는 법", "연애 불안감 다루는 법"],
    "understanding": ["연인 마음 이해하는 법", "커플 대화법 소통"],
    "apology": ["진심으로 사과하는 법", "연인에게 사과 잘하는 방법"],
}
PER_TYPE = 3  # 유형당 최대 몇 개 (7유형 × 3 = 최대 21건)

OUT = Path("data/yt_seed.json")


def main() -> int:
    if not available():
        print("YOUTUBE_API_KEY 가 없다 — .env 확인", file=sys.stderr)
        return 1

    seen: set[str] = set()
    rows: list[dict] = []
    for concern, queries in QUERIES.items():
        kept = 0
        for query in queries:
            if kept >= PER_TYPE:
                break
            for video in find_candidates([query]):
                if kept >= PER_TYPE:
                    break
                if video.video_id in seen:
                    continue
                bad = find_banned(video.title) or find_banned(video.summary() or "")
                if bad:
                    print(f"  제외(금지어 '{bad}'): {video.title}")
                    continue
                seen.add(video.video_id)
                rows.append({"concern": concern, "video": asdict(video)})
                kept += 1
                print(f"[{concern}] {video.title}  ({video.channel_name})")
        if kept == 0:
            print(f"⚠️ {concern}: 후보 0건", file=sys.stderr)

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {OUT} — {len(rows)}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
