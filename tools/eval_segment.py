"""대화 분절 실측 검증 — 실제 한국어 대화 데이터로 잰다.

    .venv/bin/python -m tools.eval_segment --n 100
    .venv/bin/python -m tools.eval_segment --n 30 --show 5

`docs/segmentation-v3.md` 의 리스크 1(단일 실패 지점)·3(과분절)을 숫자로 확인한다.
데이터는 `data/eval/msd_sample.jsonl` (`tools/build_eval_set.py` 산출물, 실제 사람 대화).

## 이 데이터로 잴 수 없는 것

**시간 공백 임계값(`GAP_HARD = 3시간`)은 여기서 검증되지 않는다.**
빌더가 세션 내 `sentAt` 을 1~5분 랜덤으로 **생성**한다. 원본에 시각이 없기 때문이다.
`gapDays` 만 원문에서 읽은 진짜 값인데 일 단위라 3시간 컷은 무조건 걸린다.
임계값은 실서비스 로그가 쌓여야 정할 수 있다 (문서 3-3).

## 재는 것

    실험 A  과분절률   단일 세션 1개 = 한 대화. 세그먼트가 몇 개로 나오는가
    실험 B  경계 탐지   서로 다른 주제의 대화 2개를 이어붙인다. 그 지점을 찾는가

**실험 B 는 두 대화의 간격을 30분으로 둔다.** 3시간 컷 미만이라 룰이 안 걸리고,
**LLM 이 순수하게 화제만으로** 경계를 찾는지 격리해서 볼 수 있다. 실서비스에서 시간
공백까지 있으면 이보다 쉬워진다 — 여기 숫자는 하한선이다.

## 폴백을 구분해서 센다

`segment.py` 는 LLM 출력 검증에 실패하면 조각을 통째로 세그먼트 1개로 되돌린다.
결과만 보면 "경계 없음 판정"과 구별이 안 되는데, 폴백은 `by_rule=True` 로 남으므로
여기서 갈라서 센다. 리스크 1이 실제로 얼마나 터지는지가 이 숫자다.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worker import DATA_DIR  # noqa: E402
from worker.llm import USAGE  # noqa: E402
from worker.models import KST, Message  # noqa: E402
from worker.segment import segment  # noqa: E402

EVAL_FILE = DATA_DIR / "eval" / "msd_sample.jsonl"

# 이어붙인 두 대화 사이 간격. **3시간 컷 미만으로 둬서 룰을 끈다.**
JOIN_GAP = timedelta(minutes=30)

# 동시 실행. mini 라 넉넉하지만 레이트리밋을 건드릴 만큼 올리지 않는다.
WORKERS = 6

DIM = "\033[2m"
BOLD = "\033[1m"
OFF = "\033[0m"


# --------------------------------------------------------------------------
# 데이터
# --------------------------------------------------------------------------
def load_rows() -> list[dict]:
    if not EVAL_FILE.exists():
        print(f"평가셋이 없다: {EVAL_FILE}\n"
              f".venv/bin/python -m tools.build_eval_set --download 로 먼저 만든다.",
              file=sys.stderr)
        raise SystemExit(1)
    return [json.loads(line) for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()]


def to_messages(raw: list[dict], start: datetime, first_id: int) -> list[Message]:
    """규격 형식 메시지를 시각·id 를 다시 매겨 Message 로. 이어붙일 때 충돌을 없앤다."""
    out: list[Message] = []
    at = start
    for i, m in enumerate(raw):
        out.append(
            Message(
                messageId=first_id + i,
                sender=m["sender"],
                content=m["content"],
                sentAt=at.isoformat(),
            )
        )
        at += timedelta(minutes=2)
    return out


# --------------------------------------------------------------------------
# 실험 A — 과분절률
# --------------------------------------------------------------------------
def run_precision(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    picked = rng.sample(rows, min(n, len(rows)))

    def one(row: dict) -> dict:
        raw = row["sessions"]["session1"]["messages"]
        messages = to_messages(raw, datetime(2026, 8, 15, 20, 0, tzinfo=KST), 1)
        segs = segment(messages)
        return {
            "id": row["id"],
            "topic": row["topicTitle"],
            "n_msg": len(messages),
            "n_seg": len(segs),
            "fallback": bool(segs) and segs[-1].by_rule,
            "labels": [(s.topic, len(s.messages)) for s in segs],
            "messages": messages,
        }

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(one, picked))


# --------------------------------------------------------------------------
# 실험 B — 경계 탐지
# --------------------------------------------------------------------------
def run_recall(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """서로 **다른 대분류 주제**의 대화 둘을 이어붙인다.

    같은 주제끼리 붙이면 경계가 실제로 모호해져서 무엇을 재는지 알 수 없게 된다.
    """
    by_type: dict[str, list[dict]] = {}
    for row in rows:
        by_type.setdefault(row["topicType"].split(">")[0], []).append(row)
    types = [t for t, v in by_type.items() if len(v) >= 2]

    pairs = []
    for _ in range(n):
        t1, t2 = rng.sample(types, 2)
        pairs.append((rng.choice(by_type[t1]), rng.choice(by_type[t2])))

    def one(pair: tuple[dict, dict]) -> dict:
        a, b = pair
        # 각 대화의 앞부분만 쓴다. 너무 길면 프롬프트만 커지고 재는 것은 같다.
        raw_a = a["sessions"]["session1"]["messages"][:8]
        raw_b = b["sessions"]["session1"]["messages"][:8]

        start = datetime(2026, 8, 15, 19, 0, tzinfo=KST)
        first = to_messages(raw_a, start, 1)
        second = to_messages(raw_b, first[-1].sent_at + JOIN_GAP, 101)
        messages = first + second

        segs = segment(messages)
        join_id = second[0].message_id
        starts = {s.messages[0].message_id for s in segs}

        return {
            "topics": (a["topicTitle"], b["topicTitle"]),
            "n_msg": len(messages),
            "n_seg": len(segs),
            "hit": join_id in starts,
            "clean": len(segs) == 2 and join_id in starts,
            "fallback": bool(segs) and segs[-1].by_rule,
            "labels": [(s.topic, s.messages[0].message_id, len(s.messages)) for s in segs],
            "messages": messages,
            "join_id": join_id,
        }

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(one, pairs))


# --------------------------------------------------------------------------
# 출력
# --------------------------------------------------------------------------
def pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole * 100:.0f}%)" if whole else "-"


def show_case(case: dict, limit: int = 6) -> None:
    for m in case["messages"][:limit]:
        mark = " ←경계" if m.message_id == case.get("join_id") else ""
        print(f"      {DIM}{m.sender}: {m.content[:44]}{mark}{OFF}")
    if len(case["messages"]) > limit:
        print(f"      {DIM}…{OFF}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="tools.eval_segment", description="분절 실측 검증")
    parser.add_argument("--n", type=int, default=100, help="실험별 케이스 수")
    parser.add_argument("--show", type=int, default=3, help="실패 사례를 몇 건 보여줄지")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_rows()
    rng = random.Random(args.seed)
    print(f"평가셋 {len(rows)}건 · 실험별 {args.n}건 · seed={args.seed}\n")

    # ---------------- 실험 A ----------------
    print(f"{BOLD}실험 A — 과분절률{OFF}  단일 세션(한 대화) → 세그먼트 1개가 나와야 한다")
    a = run_precision(rows, args.n, rng)
    dist = Counter(c["n_seg"] for c in a)
    single = sum(1 for c in a if c["n_seg"] == 1 and not c["fallback"])
    fb_a = sum(1 for c in a if c["fallback"])
    print(f"  세그먼트 1개  {pct(single, len(a))}   {DIM}(폴백 제외, LLM 이 '경계 없음'으로 판정){OFF}")
    print(f"  폴백         {pct(fb_a, len(a))}   {DIM}(검증 실패 → 통째로 1개){OFF}")
    print(f"  개수 분포     {dict(sorted(dist.items()))}")
    over = [c for c in a if c["n_seg"] > 1]
    for c in over[: args.show]:
        print(f"\n    {BOLD}쪼개짐{OFF} [{c['topic']}] {c['n_msg']}개 → {c['n_seg']}개  {c['labels']}")
        show_case(c)

    # ---------------- 실험 B ----------------
    print(f"\n{BOLD}실험 B — 경계 탐지{OFF}  다른 주제 대화 2개를 30분 간격으로 이어붙임"
          f"  {DIM}(룰 컷 미만 — LLM 단독){OFF}")
    b = run_recall(rows, args.n, rng)
    hit = sum(1 for c in b if c["hit"])
    clean = sum(1 for c in b if c["clean"])
    miss = sum(1 for c in b if c["n_seg"] == 1)
    fb_b = sum(1 for c in b if c["fallback"])
    print(f"  경계 탐지     {pct(hit, len(b))}   {DIM}(그 지점에 경계가 있는가){OFF}")
    print(f"  정확히 2개    {pct(clean, len(b))}   {DIM}(경계 맞고 군더더기 없음){OFF}")
    print(f"  놓침(1개)     {pct(miss, len(b))}")
    print(f"  폴백         {pct(fb_b, len(b))}")
    print(f"  개수 분포     {dict(sorted(Counter(c['n_seg'] for c in b).items()))}")
    for c in [x for x in b if not x["hit"]][: args.show]:
        print(f"\n    {BOLD}놓침{OFF} {c['topics'][0]} + {c['topics'][1]} → {c['n_seg']}개  {c['labels']}")
        show_case(c, limit=4)

    print(f"\n{DIM}{USAGE}{OFF}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
