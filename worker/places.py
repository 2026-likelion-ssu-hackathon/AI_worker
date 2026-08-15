"""카카오 로컬 API — 실재하는 장소만 가져온다.

**LLM 에게 장소를 만들게 하지 않는다.** 상호명을 생성시키면 존재하지 않는 가게가 나오고
규격서 9장의 `externalUrl` 이 죽은 링크가 된다. 시연 중에 링크 한 번 눌리면 그대로 터진다.

역할 분담:
- LLM  → "무엇을 찾을지" (검색어, 지역, 코스 구성 의도)
- 카카오 → "실제로 무엇이 있는지" (상호명, 카테고리, place_url)

키가 없으면 예외를 던지지 않고 빈 목록을 돌려준다. 규격서 13장 기준으로
검색 결과 없음은 오류가 아니라 **기능 미발동**이다.

문서: https://developers.kakao.com/docs/latest/ko/local/dev-guide
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import httpx

SEARCH_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
TIMEOUT = 5.0

# 지역 중심에서 이만큼 안쪽만 코스에 넣는다 (미터, 카카오 최대 20000).
# 2km 면 걸어서 20~25분이라 데이트 동선으로 이어지는 범위다.
# 이보다 넓히면 성수동 검색에 건대·왕십리가 섞여 들어온다.
RADIUS = 2000

# 카카오 category_group_code → 규격서 `category`
#
# ⚠️ 규격서에 category enum 목록이 없다 (예시로 RESTAURANT / SHOP 만 등장).
# 서버 담당자와 확정 필요. 지금은 아래 값으로 내보낸다.
CATEGORY_MAP = {
    "FD6": "RESTAURANT",   # 음식점
    "CE7": "CAFE",         # 카페
    "CT1": "CULTURE",      # 문화시설
    "AT4": "ATTRACTION",   # 관광명소
    "AD5": "LODGING",      # 숙박
    "MT1": "SHOP",         # 대형마트
    "CS2": "SHOP",         # 편의점
    "HP8": "ETC",
    "PM9": "ETC",
}

# category_group_code 가 비어 있을 때 category_name 앞부분으로 때려잡는다.
# 소품샵·서점 같은 곳은 카카오가 그룹 코드를 안 준다.
_NAME_HINTS = (
    ("카페", "CAFE"),
    ("음식", "RESTAURANT"),
    ("술집", "RESTAURANT"),
    ("문화", "CULTURE"),
    ("여행", "ATTRACTION"),
    ("관광", "ATTRACTION"),
    ("공원", "ATTRACTION"),
    ("스포츠", "ACTIVITY"),
    ("가정", "SHOP"),
    ("생활", "SHOP"),
    ("판매", "SHOP"),
    ("소매", "SHOP"),
    ("서적", "SHOP"),
)


@dataclass(frozen=True)
class KakaoPlace:
    name: str
    category: str
    category_name: str   # 카카오 원문 ("음식점 > 카페 > 커피전문점") — LLM 설명 생성용
    address: str
    url: str


@lru_cache(maxsize=32)
def region_center(region: str) -> tuple[str, str] | None:
    """지역명 → 중심 좌표 (x=경도, y=위도). 못 찾으면 None.

    주소 문자열 대조로는 지역을 가둘 수 없다. "서울숲"의 주소는 `서울 성동구 뚝섬로`라
    '성수'가 들어가지 않는데 성수 한복판이다. 반대로 옆 동네인 건대는 걸러야 한다.
    **좌표를 잡고 반경으로 자르는 게 정공법이다.**

    호출 1회를 더 쓰지만 카카오는 일 100,000회라 부담이 없고, 같은 지역은 캐시된다.
    """
    key = os.getenv("KAKAO_REST_API_KEY")
    if not key:
        return None
    try:
        res = httpx.get(
            SEARCH_URL,
            params={"query": region, "size": 1},
            headers={"Authorization": f"KakaoAK {key}"},
            timeout=TIMEOUT,
        )
        res.raise_for_status()
        docs = res.json().get("documents", [])
    except Exception:  # noqa: BLE001
        return None
    if not docs:
        return None
    x, y = docs[0].get("x"), docs[0].get("y")
    return (x, y) if x and y else None


def _classify(group_code: str, category_name: str) -> str:
    if group_code in CATEGORY_MAP:
        return CATEGORY_MAP[group_code]
    for hint, value in _NAME_HINTS:
        if hint in category_name:
            return value
    return "ETC"


def available() -> bool:
    return bool(os.getenv("KAKAO_REST_API_KEY"))


def search_places(query: str, region: str | None = None, size: int = 5) -> list[KakaoPlace]:
    """키워드로 장소를 찾는다. 키가 없거나 실패하면 빈 목록.

    `region` 을 주면 그 지역 중심에서 **반경 안의 결과만** 돌려준다.
    질의에 지역명을 섞는 것만으로는 부족하다 — 실측에서 "성수동 분위기 좋은 카페" 가
    옆 동네인 건대 카페를 물어왔다. 카카오 정확도 정렬이 인접 지역을 같이 주기 때문이다.
    데이트 코스는 동선이 이어져야 의미가 있어서, 반경을 벗어난 결과는 아예 버린다.
    """
    key = os.getenv("KAKAO_REST_API_KEY")
    if not key:
        return []

    params: dict[str, object] = {"query": query, "size": size, "sort": "accuracy"}

    center = region_center(region) if region else None
    if center is not None:
        params["x"], params["y"] = center
        params["radius"] = RADIUS
    elif region:
        # 좌표를 못 잡았으면 질의에 지역명을 섞는 방식으로 물러선다
        params["query"] = f"{region} {query}".strip() if region not in query else query

    try:
        res = httpx.get(
            SEARCH_URL,
            params=params,
            headers={"Authorization": f"KakaoAK {key}"},
            timeout=TIMEOUT,
        )
        res.raise_for_status()
        documents = res.json().get("documents", [])
    except Exception:  # noqa: BLE001 — 검색 실패는 오류가 아니라 미발동이다
        return []

    places: list[KakaoPlace] = []
    for d in documents:
        name = d.get("place_name")
        url = d.get("place_url")
        if not name or not url:
            continue
        address = d.get("road_address_name") or d.get("address_name", "")
        category_name = d.get("category_name", "")
        places.append(
            KakaoPlace(
                name=name,
                category=_classify(d.get("category_group_code", ""), category_name),
                category_name=category_name,
                address=address,
                url=url,
            )
        )
    return places
