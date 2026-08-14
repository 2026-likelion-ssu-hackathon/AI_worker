"""전체 조립.

픽스처를 읽어 실행 맥락(`Context`)을 만들고 라우터에 넘긴다.
후보 기능별 로직은 `worker/router.py` 에 있다.
"""

from __future__ import annotations

from datetime import timedelta

from worker.models import Decision, Fixture
from worker.router import Context, Trace, route

__all__ = ["Context", "Trace", "run", "run_traced"]


def run_traced(fixture: dict, persist: bool = True) -> tuple[Decision | None, Trace]:
    fx = Fixture(**fixture)
    messages = sorted(fx.messages, key=lambda m: m.ts)

    # ⑤ 판정 기준 시각. 픽스처에 없으면 "정체 아님"으로 본다.
    now = fx.now or (messages[-1].ts + timedelta(minutes=1))

    ctx = Context(messages=messages, now=now, online=fx.online, persist=persist)
    return route(ctx), ctx.trace


def run(fixture: dict) -> Decision | None:
    decision, _ = run_traced(fixture)
    return decision
