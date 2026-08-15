"""외부 API 키 점검.

    .venv/bin/python -m tools.check_keys

키를 `.env` 에 넣은 뒤 실제로 붙는지 확인한다. **쿼터를 거의 쓰지 않는다** —
유튜브는 100 units 짜리 `search.list` 대신 1 unit 짜리 `videos.list` 로 키만 검증한다.
"""

from __future__ import annotations

import os

import httpx

from worker import places

OK = "\033[32m✓\033[0m"
NO = "\033[31m✗\033[0m"
DIM = "\033[2m"
OFF = "\033[0m"

# 유튜브 키 검증용. 공개 영상이면 아무거나 상관없다 (Google 공식 채널 영상).
PROBE_VIDEO = "M7lc1UVf-VE"


def check_openai() -> bool:
    if not os.getenv("OPENAI_API_KEY"):
        print(f"{NO} OPENAI_API_KEY 없음 — 워커가 아예 동작하지 않는다")
        return False
    print(f"{OK} OPENAI_API_KEY 있음")
    return True


def check_kakao() -> bool:
    if not places.available():
        print(f"{NO} KAKAO_REST_API_KEY 없음 — 데이트 코스 미발동")
        print(f"  {DIM}developers.kakao.com → 내 애플리케이션 → 앱 키 → REST API 키{OFF}")
        return False

    # `places.search_places()` 는 실패를 조용히 빈 목록으로 삼킨다(운영에서는 그게 맞다).
    # 점검에서는 카카오가 준 오류 메시지를 그대로 보여줘야 원인을 찾을 수 있다.
    try:
        res = httpx.get(
            "https://dapi.kakao.com/v2/local/search/keyword.json",
            params={"query": "성수동 브런치", "size": 3},
            headers={"Authorization": f"KakaoAK {os.environ['KAKAO_REST_API_KEY']}"},
            timeout=8.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"{NO} 카카오 API 호출 실패: {exc}")
        return False

    if res.status_code != 200:
        body = res.json() if res.headers.get("content-type", "").startswith("application/json") else {}
        message = body.get("message", res.text[:200])
        print(f"{NO} 카카오 API 오류 {res.status_code} — {message}")
        if "OPEN_MAP_AND_LOCAL" in message:
            print(f"  {DIM}앱에서 카카오맵 서비스가 꺼져 있다. 키 문제가 아니다.{OFF}")
            print(f"  {DIM}developers.kakao.com → 내 애플리케이션 → 앱 설정 → 플랫폼 →{OFF}")
            print(f"  {DIM}  Web 플랫폼 등록 (http://localhost) → 제품 설정 → 카카오맵 → 활성화 ON{OFF}")
        elif res.status_code == 401:
            print(f"  {DIM}REST API 키가 아닐 수 있다 (Admin 키·JavaScript 키를 넣지 않았는지 확인){OFF}")
        return False

    found = places.search_places("성수동 브런치", size=3)
    print(f"{OK} 카카오 로컬 연결됨 — 검색 결과 {len(found)}건")
    for p in found:
        print(f"  {DIM}· {p.name} [{p.category}] {p.url}{OFF}")
    return bool(found)


def check_youtube() -> bool:
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        print(f"{NO} YOUTUBE_API_KEY 없음 — 유튜브 추천 미발동")
        print(f"  {DIM}console.cloud.google.com → YouTube Data API v3 사용 설정 → API 키{OFF}")
        return False

    try:
        res = httpx.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={"part": "snippet", "id": PROBE_VIDEO, "key": key},
            timeout=8.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"{NO} YouTube API 호출 실패: {exc}")
        return False

    if res.status_code != 200:
        # 쿼터 소진과 키 오류는 증상이 같아 보여서 구분해준다.
        body = res.json().get("error", {})
        reason = (body.get("errors") or [{}])[0].get("reason", "?")
        print(f"{NO} YouTube API 오류 {res.status_code} — reason={reason}")
        if reason == "quotaExceeded":
            print(f"  {DIM}쿼터 소진. 태평양 시간 자정에 초기화된다 (한국 시간 오후 4~5시){OFF}")
        return False

    items = res.json().get("items", [])
    if not items:
        print(f"{NO} YouTube API 응답이 비었다 — 키는 유효하나 확인 필요")
        return False

    print(f"{OK} YouTube API 연결됨 {DIM}(1 unit 사용){OFF}")
    print(f"  {DIM}· 응답 확인: {items[0]['snippet']['title']}{OFF}")
    return True


def main() -> int:
    print()
    results = [check_openai(), check_kakao(), check_youtube()]
    print()

    ready = sum(results)
    if ready == 3:
        print("전부 연결됨. 실측 가능하다.")
    else:
        print(f"{ready}/3 연결됨. 없는 키의 기능은 미발동으로 빠진다 (워커는 죽지 않는다).")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
