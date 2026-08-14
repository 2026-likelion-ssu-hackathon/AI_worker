"""CLI 러너.

    python -m tools.run fixtures/case3_one_sided.json
    python -m tools.run fixtures/case3_one_sided.json --verbose
    python -m tools.run fixtures/*.json --no-persist
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from worker.models import Decision
from worker.pipeline import Trace, run_traced

TRIGGER_LABEL = {
    "short_pingpong": "단답 핑퐁",
    "no_question": "질문 없는 대답",
    "one_sided": "한쪽만 발화",
    "stall": "대화 중 정체",
    "routine_loop": "일상 보고형 반복",
    "busy_excuse": "바쁨 표현",
}


def _label(trigger: str | None) -> str:
    if not trigger:
        return "-"
    return f"{trigger} ({TRIGGER_LABEL.get(trigger, '?')})"


def show(path: Path, decision: Decision | None, trace: Trace, verbose: bool) -> None:
    print(f"\n\033[1m▸ {path.name}\033[0m")

    if verbose and trace.tone_gate is not None:
        tg = trace.tone_gate
        print(f"  말투게이트 triggered={tg.triggered}")
        for f in tg.flags:
            print(f"           · [{f.kind}] {f.detail}")
        if trace.tone_judged is not None:
            tj = trace.tone_judged
            print(
                f"  말투판정 should_suggest={tj.should_suggest} "
                f"is_playful={tj.is_playful} emotion={tj.emotion}"
            )
            print(f"           {tj.note}")

    # 갈등 중재로 결정됐으면 여기서 끝. 대화 소재 경로는 아예 실행되지 않는다.
    if decision is not None and decision.kind == "tone":
        print("  후보     tone (갈등 중재)")
        print(f"  scope    individual → {decision.target}")
        print(f"  방향     {decision.reason}")
        print(f"  대체문장 \"{decision.content}\"")
        return

    gate = trace.gate
    judged = trace.judged
    if gate is None:
        print("  \033[2m개입하지 않음\033[0m")
        return

    trigger = (judged.trigger if judged and judged.trigger != "none" else None) or (
        gate.trigger if gate else None
    )

    if verbose:
        print(f"  게이트   triggered={gate.triggered} needs_llm={gate.needs_llm}")
        print(f"           {gate.detail}")
        if judged:
            print(
                f"  judge    should_intervene={judged.should_intervene} "
                f"trigger={judged.trigger} scope={judged.scope} target={judged.target}"
            )
            if judged.memories:
                print(f"           기억 추출 {len(judged.memories)}건 "
                      f"(신규 저장 {len(trace.saved)}건)")
                for m in judged.memories:
                    print(f"             · [{m.kind}] {m.content} ← \"{m.source_quote}\"")
        if trace.hits:
            print("  RAG top3")
            for i, m in enumerate(trace.hits, 1):
                mark = " " if m.used_at is None else "*"
                print(f"           {i}.{mark}[{m.kind}] {m.content}")
            print("           (* = 이미 소환된 기억)")

    if decision is None:
        print(f"  트리거   {_label(trigger)}")
        print("  \033[2m개입하지 않음\033[0m")
        return

    scope = decision.scope
    scope_line = scope if decision.target is None else f"{scope} → {decision.target}"
    print(f"  트리거   {_label(trigger)}")
    print(f"  scope    {scope_line}")
    print(f"  소재     \"{decision.content}\"")
    print(f"  근거     {decision.reason or '-'}"
          + ("" if trace.source == "memory" else "   \033[2m(오늘의 질문)\033[0m"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.run", description="kakapo 워커 러너")
    parser.add_argument("fixtures", nargs="+", type=Path)
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="게이트 판정, 검색된 기억 top-3 표시")
    parser.add_argument("--no-persist", action="store_true",
                        help="used_at / 기억 저장을 파일에 쓰지 않는다 (반복 시연용)")
    args = parser.parse_args(argv)

    for path in args.fixtures:
        if not path.exists():
            print(f"파일 없음: {path}", file=sys.stderr)
            return 1
        fixture = json.loads(path.read_text(encoding="utf-8"))
        decision, trace = run_traced(fixture, persist=not args.no_persist)
        show(path, decision, trace, args.verbose)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
