"""141 한국어 멀티세션 대화 → 기억/RAG 평가셋 빌더.

    .venv/bin/python -m tools.build_eval_set --download
    .venv/bin/python -m tools.build_eval_set --parquet /경로/0.parquet -n 2000

원본: https://huggingface.co/datasets/nayohan/141_korean_multi_session_dialogue
(AI Hub 141 주제별 텍스트 일상 대화 데이터. 76,000행)

## 왜 이 데이터셋인가

**두 사람이 시간을 두고 두 번 대화한다.** session1 에서 나온 이야기가 session2 에서
다시 언급된다. `extract.py` 가 뽑은 기억을 `retrieve.py` 가 제대로 찾아오는지 검증할
데이터가 지금 레포에 없다 — 시드 27건과 손으로 쓴 픽스처 10개가 전부다.

## 이 데이터로 하면 안 되는 것

**말투 기준선(`data/speaker_profiles.json`) 학습에는 쓰지 않는다.** 원본은 처음 만난
두 사람의 존댓말 대화다. 커플 반말 채팅과 마침표율·호칭·ㅋ 개수가 전부 다르다.
`profile.py` 는 "그 사람의 평소 대비 변화량"으로 판정하므로 기준선이 틀어지면
말투 교정 기능이 통째로 무너진다.

## 변환 규칙

| 원본 | 변환 결과 |
| --- | --- |
| `personaInfo_cl` | `USER_A` (짝수 인덱스 발화자, 먼저 말한다) |
| `personaInfo_cp` | `USER_B` (홀수 인덱스 발화자) |
| `session1` / `session2` | 규격서 5장 `AnalysisRequest` 2개 |
| session2 첫머리의 "3주만이네요" | `gapDays` — 두 세션의 시각 차이 |

발화자 배정은 원본에 라벨이 없어서 **짝수=cl 로 가정**한 것이다. 전량(76,000행)에
성별 대조를 돌려 73,027건 일치 / 192건 불일치로 확인했고, 불일치 행은 걸러낸다.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
import unicodedata
import urllib.request
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worker.models import KST, AnalysisRequest  # noqa: E402

PARQUET_URL = (
    "https://huggingface.co/api/datasets/nayohan/141_korean_multi_session_dialogue"
    "/parquet/default/train/0.parquet"
)

# 모든 케이스의 session2 를 같은 시각에 고정한다. 기억의 30일 중복 판정이 케이스마다
# 다른 "지금"을 보면 결과가 재현되지 않는다.
ANCHOR = datetime(2026, 8, 15, 20, 0, tzinfo=KST)

MIN_TURNS = 6          # 이보다 짧으면 재언급을 볼 게 없다
DF_CEILING = 0.02      # 문서 2% 초과 출현 어휘는 재언급 근거로 치지 않는다
MAX_GAP_DAYS = 365

# --------------------------------------------------------------------------
# 세션 간격 — 원문에서 읽는다
# --------------------------------------------------------------------------
# 지어낸 간격을 쓰면 "3주만이네요"라고 말하는 대화가 하루 뒤로 찍힌다. 기억 추출이
# 시점을 근거로 삼는 기능이라 이 불일치가 그대로 추천 이유로 나간다.
_NUM_KO = {
    "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6,
    "일곱": 7, "여덟": 8, "아홉": 9, "열": 10,
}
_UNIT_DAYS = {"일": 1, "주": 7, "주일": 7, "달": 30, "개월": 30, "년": 365}
_GAP_RE = re.compile(
    r"(\d{1,3}|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*(주일|개월|일|주|달|년)\s*만"
)

DEFAULT_GAP_DAYS = 7

# --------------------------------------------------------------------------
# 익명화 토큰
# --------------------------------------------------------------------------
# 원본은 고유명사를 `@이름1@` · `@브랜드명1@` 로 가려 놨다. 표본의 29% 에 들어 있다.
# `extract.py` 는 `source_quote` 를 원문 그대로 저장하고 데이트 코스가 그걸 추천 이유로
# 화면에 내보낸다 — "@브랜드명1@ 가보고 싶다고 하셨죠"가 그대로 나간다. 지어낸 이름으로
# 채우면 없던 사실을 만드는 것이라 **행 자체를 제외한다.** 적격 행은 5만 건 넘게 남는다.
_PLACEHOLDER_RE = re.compile(r"@[^@\s]{1,12}@")

# --------------------------------------------------------------------------
# 어휘 — 재언급 판정용
# --------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[가-힣]{2,}|[A-Za-z]{3,}")

# 서술어는 재언급의 근거가 못 된다. "속상하네요"가 양쪽에 나왔다고 같은 이야기를 다시
# 꺼낸 게 아니다. 형태소 분석기를 붙이지 않고 어미로만 거른다 (konlpy 의존성을 안 만든다).
_PREDICATE_END = (
    "습니다", "합니다", "입니다", "니다", "네요", "세요", "어요", "아요", "에요", "예요",
    "구요", "군요", "지요", "데요", "가요", "나요", "까요", "어서", "아서", "잖아",
    "거든", "더라", "는데", "은데", "지만", "면서", "려고", "니까",
    "요", "죠", "다",
)
# 길이 내림차순. "에게서"를 "서"로 자르면 안 된다.
_JOSA = (
    "으로써", "으로서", "이라고", "에게서", "한테서", "이라는", "라는",
    "에서는", "에게는", "한테는", "까지", "부터", "에서", "에게", "한테",
    "으로", "이랑", "라고", "이나", "에는", "도는",
    "은", "는", "이", "가", "을", "를", "에", "의", "도", "만", "로", "랑", "과", "와",
)


# 대명사·지시어는 아무 대화에나 나온다. 존댓말 데이터라 "저는"·"제가"는 DF 상한에 걸려
# 저절로 빠지지만 "너는"·"당신"은 드물게 나와서 살아남는다.
_STOPWORDS = frozenset({
    "너는", "너도", "나도", "나는", "당신", "그쪽", "우리", "저희", "본인", "서로",
    "여기", "거기", "저기", "이거", "그거", "저거", "무엇", "누구", "언제", "어디",
})


def _stem(token: str) -> str:
    """조사를 뗀다. 떼고 나서 2글자 미만이면 버린다 ("시를" → 근거로 못 쓴다)."""
    for josa in _JOSA:
        if token.endswith(josa):
            rest = token[: -len(josa)]
            return rest if len(rest) >= 2 else ""
    return token


def terms_of(text: str) -> set[str]:
    """원형과 어간을 **둘 다** 넣는다.

    조사 목록만으로는 어디까지가 어간인지 알 수 없다 — "고양이"는 "이"로 끝나서 "고양"이
    되는데 "고양이가"는 "고양이"가 된다. 두 형태를 모두 넣어야 같은 낱말이 서로 만난다.
    """
    out: set[str] = set()
    for token in _TOKEN_RE.findall(text):
        if token.endswith(_PREDICATE_END):
            continue
        out.add(token)
        stem = _stem(token)
        if len(stem) >= 2 and not stem.endswith(_PREDICATE_END):
            out.add(stem)
    return out - _STOPWORDS


# --------------------------------------------------------------------------
# 원본 파싱
# --------------------------------------------------------------------------
def parse_list(raw: str) -> list[str] | None:
    """원본의 각 칸은 파이썬 리스트 리터럴 **문자열**이다 — `"['안녕하세요', ...]"`."""
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        return None
    return [unicodedata.normalize("NFC", x).strip() for x in value]


def gender_of(text: str) -> str | None:
    female = ("여자" in text) or ("여성" in text)
    male = ("남자" in text) or ("남성" in text)
    if female and not male:
        return "F"
    if male and not female:
        return "M"
    return None


def gap_days_of(session2: list[str]) -> tuple[int, str]:
    """session2 첫 두 발화에서 "3주만이네요" 같은 표현을 읽는다."""
    head = " ".join(session2[:2])
    m = _GAP_RE.search(head)
    if not m:
        return DEFAULT_GAP_DAYS, "default"
    count, unit = m.group(1), m.group(2)
    n = int(count) if count.isdigit() else _NUM_KO[count]
    return min(n * _UNIT_DAYS[unit], MAX_GAP_DAYS), "phrase"


# --------------------------------------------------------------------------
# 규격서 형식으로 조립
# --------------------------------------------------------------------------
def build_request(
    row_index: int,
    session_no: int,
    utterances: list[str],
    start: datetime,
    rng: random.Random,
) -> dict:
    """규격서 5장의 공통 분석 요청 형식. `fixtures/*.json` 과 같은 모양이다."""
    base = row_index * 1000 + session_no * 100
    messages = []
    at = start
    for i, text in enumerate(utterances):
        messages.append({
            "messageId": base + i + 1,
            # 짝수 인덱스가 personaInfo_cl. 규격서 표기는 USER_A / USER_B 다.
            "sender": "USER_A" if i % 2 == 0 else "USER_B",
            "content": text,
            "sentAt": at.isoformat(),
        })
        at += timedelta(minutes=rng.randint(1, 5))

    return {
        "analysisRequestId": str(uuid5(NAMESPACE_URL, f"kakapo-msd/{row_index}/s{session_no}")),
        "chatRoomId": 900000 + row_index,
        "participants": [{"participantKey": "USER_A"}, {"participantKey": "USER_B"}],
        "messages": messages,
        "requestedAt": (at + timedelta(minutes=1)).isoformat(),
    }


def recall_hints(s1: list[dict], s2: list[dict], rare: set[str]) -> list[dict]:
    """session2 의 발화가 session1 의 어떤 발화를 다시 꺼냈는지 — **약한 라벨**이다.

    원본에 재언급 라벨이 없어서 희귀 어휘 겹침으로 근사한다. LLM 을 부르지 않으므로
    비용이 들지 않고, `retrieve.py` 가 가져온 기억이 최소한 같은 소재를 가리키는지
    자동으로 채점할 수 있다. **정답이 아니라 후보다** — 사람이 확인하고 쓴다.
    """
    s1_terms = [(m["messageId"], terms_of(m["content"]) & rare) for m in s1]
    hints = []
    for m2 in s2:
        t2 = terms_of(m2["content"]) & rare
        if not t2:
            continue
        matched_ids, shared = [], set()
        for mid, t1 in s1_terms:
            common = t1 & t2
            if common:
                matched_ids.append(mid)
                shared |= common
        if matched_ids:
            hints.append({
                "session2MessageId": m2["messageId"],
                "session1MessageIds": matched_ids,
                "terms": sorted(shared),
            })
    return hints


# --------------------------------------------------------------------------
# 파이프라인
# --------------------------------------------------------------------------
def eligible_rows(
    data: dict[str, list[str]], keep_placeholders: bool = False
) -> tuple[list[dict], Counter]:
    """품질 필터를 통과한 행만 남긴다. 탈락 사유는 세어서 보고한다."""
    rejected: Counter = Counter()
    seen: set[tuple[str, str]] = set()
    rows = []

    for i in range(len(data["session1"])):
        s1 = parse_list(data["session1"][i])
        s2 = parse_list(data["session2"][i])
        p_a = parse_list(data["personaInfo_cl"][i])
        p_b = parse_list(data["personaInfo_cp"][i])
        if s1 is None or s2 is None or p_a is None or p_b is None:
            rejected["파싱 실패"] += 1
            continue
        if len(s1) < MIN_TURNS or len(s2) < MIN_TURNS:
            rejected[f"발화 {MIN_TURNS}개 미만"] += 1
            continue
        if any(not x for x in s1 + s2):
            rejected["빈 발화 포함"] += 1
            continue
        if not keep_placeholders and any(_PLACEHOLDER_RE.search(x) for x in s1 + s2):
            rejected["익명화 토큰 포함"] += 1
            continue

        # 짝수=cl 가정이 깨진 행. 발화자가 뒤집히면 기억의 주인이 바뀐다.
        g_persona, g_first = gender_of(data["personaInfo_cl"][i]), gender_of(s1[0])
        if g_persona and g_first and g_persona != g_first:
            rejected["발화자 배정 불일치"] += 1
            continue

        key = (data["session1"][i], data["session2"][i])
        if key in seen:
            rejected["중복"] += 1
            continue
        seen.add(key)

        rows.append({
            "row_index": i,
            "topicType": data["topicType"][i],
            "topicTitle": data["topicTitle"][i],
            "personas": {"A": p_a, "B": p_b},
            "s1": s1,
            "s2": s2,
        })

    return rows, rejected


def build_cases(rows: list[dict], seed: int) -> list[dict]:
    cases = []
    for row in rows:
        rng = random.Random(f"{seed}/{row['row_index']}")
        gap, gap_source = gap_days_of(row["s2"])
        s1_start = ANCHOR - timedelta(days=gap)

        req1 = build_request(row["row_index"], 1, row["s1"], s1_start, rng)
        req2 = build_request(row["row_index"], 2, row["s2"], ANCHOR, rng)

        # 이 파일이 워커에 그대로 들어갈 수 있는지 여기서 보장한다.
        AnalysisRequest.model_validate(req1)
        AnalysisRequest.model_validate(req2)

        cases.append({
            "id": f"msd-{row['row_index']:06d}",
            "source": {
                "dataset": "nayohan/141_korean_multi_session_dialogue",
                "rowIndex": row["row_index"],
            },
            "topicType": row["topicType"],
            "topicTitle": row["topicTitle"],
            # 페르소나는 요청 규격에 없다. 채점할 때 사람이 보는 참고 자료로만 싣는다.
            "personas": row["personas"],
            "gapDays": gap,
            "gapSource": gap_source,
            "sessions": {"session1": req1, "session2": req2},
        })
    return cases


def attach_recall_hints(cases: list[dict]) -> None:
    """희귀 어휘를 표본 안에서 정한다 — 흔한 말("생각", "정말")은 근거가 못 된다."""
    df: Counter = Counter()
    for case in cases:
        terms: set[str] = set()
        for req in case["sessions"].values():
            for m in req["messages"]:
                terms |= terms_of(m["content"])
        df.update(terms)

    ceiling = max(2, int(len(cases) * DF_CEILING))
    rare = {t for t, c in df.items() if c <= ceiling}

    for case in cases:
        case["recallHints"] = recall_hints(
            case["sessions"]["session1"]["messages"],
            case["sessions"]["session2"]["messages"],
            rare,
        )


def download(dest: Path) -> None:
    print(f"내려받는 중… {PARQUET_URL}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(PARQUET_URL, dest)  # noqa: S310
    print(f"  → {dest} ({dest.stat().st_size / 1e6:.0f} MB)")


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        prog="tools.build_eval_set",
        description="141 멀티세션 대화 → 기억/RAG 평가셋",
    )
    parser.add_argument("--parquet", type=Path, default=root / "data" / "eval" / "raw" / "0.parquet")
    parser.add_argument("--download", action="store_true", help="원본 parquet 을 내려받는다 (122MB)")
    parser.add_argument("-n", "--count", type=int, default=2000, help="표본 개수. 0 이면 전량")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--keep-placeholders", action="store_true",
                        help="`@이름1@` 같은 익명화 토큰이 든 행도 남긴다")
    parser.add_argument("--out", type=Path, default=root / "data" / "eval" / "msd_sample.jsonl")
    args = parser.parse_args(argv)

    if args.download or not args.parquet.exists():
        if not args.download:
            print(f"원본이 없다: {args.parquet}\n  --download 를 붙이면 내려받는다.", file=sys.stderr)
            return 1
        download(args.parquet)

    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        print("pyarrow 가 필요하다 (개발 도구 전용): .venv/bin/pip install pyarrow", file=sys.stderr)
        return 1

    data = pq.read_table(args.parquet).to_pydict()
    total = len(data["session1"])
    rows, rejected = eligible_rows(data, keep_placeholders=args.keep_placeholders)

    if args.count and args.count < len(rows):
        rows = random.Random(args.seed).sample(rows, args.count)
    rows.sort(key=lambda r: r["row_index"])

    cases = build_cases(rows, args.seed)
    attach_recall_hints(cases)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    # 무엇이 빠졌는지 보이지 않으면 표본을 신뢰할 수 없다.
    print(f"\n원본 {total:,}행")
    for reason, count in rejected.most_common():
        print(f"  - {reason:<20} {count:>7,}")
    print(f"  = 적격 {total - sum(rejected.values()):,}행 → 표본 {len(cases):,}건")

    with_hint = sum(1 for c in cases if c["recallHints"])
    hints = sum(len(c["recallHints"]) for c in cases)
    phrase = sum(1 for c in cases if c["gapSource"] == "phrase")
    print(f"\n재언급 후보  {hints:,}건 / 케이스 {with_hint:,}건 ({with_hint / len(cases):.0%})")
    print(f"세션 간격    원문에서 읽음 {phrase:,}건 ({phrase / len(cases):.0%}), "
          f"나머지는 기본값 {DEFAULT_GAP_DAYS}일")
    print(f"\n{args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
