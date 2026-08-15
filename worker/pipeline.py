"""전체 조립.

규격서(`docs/contract-v1.md`)의 **공통 분석 요청**을 받아 **공통 분석 응답**을 만든다.

    AnalysisRequest → [기억 수확] → [후보 기능 라우팅] → AnalysisResponse

> ⏸️ HTTP 서버(`POST /internal/v1/chat-analyses`)는 아직 만들지 않았다.
> 규격서가 초안 v1 이고 미확정 항목(감정 분석, 타임아웃, 오류 코드)이 남아 있어서,
> 지금은 **DTO 와 판정 로직만** 규격에 맞춰두고 CLI 로 검증한다.
> 서버 담당자와 규격이 확정되면 `analyze()` 를 FastAPI 핸들러에 그대로 물리면 된다.

`emotionAnalyses` 는 항상 빈 배열이다. 별도 기능으로 명세가 아직 나오지 않았다.
"""

from __future__ import annotations

from pydantic import ValidationError

from worker.models import AnalysisRequest, AnalysisResponse
from worker.router import Context, Trace, harvest_memories, route

__all__ = ["Context", "Trace", "analyze"]


def analyze(payload: dict, persist: bool = True) -> tuple[AnalysisResponse, Trace]:
    """규격서 요청 JSON 을 받아 응답을 만든다.

    예외를 밖으로 던지지 않는다. 규격서 12장의 `FAILED` 응답으로 바꿔서 돌려준다 —
    워커가 죽으면 채팅 서버가 타임아웃까지 기다리게 된다.
    """
    trace = Trace()

    try:
        request = AnalysisRequest(**payload)
    except ValidationError as exc:
        return _failed(payload.get("analysisRequestId", ""), "INVALID_REQUEST", str(exc)), trace

    if not request.messages:
        return _skipped(request.analysis_request_id), trace

    known = {p.participant_key.removeprefix("USER_") for p in request.participants}
    unknown = {m.sender for m in request.messages} - known
    if unknown:
        return (
            _failed(
                request.analysis_request_id,
                "INVALID_PARTICIPANT",
                f"참여자 목록에 없는 발화자: {', '.join(sorted(unknown))}",
            ),
            trace,
        )

    ctx = Context(
        request=request,
        messages=sorted(request.messages, key=lambda m: (m.sent_at, m.message_id)),
        now=request.now(),
        persist=persist,
        trace=trace,
    )

    try:
        harvest_memories(ctx)
        results = route(ctx)
    except Exception as exc:  # noqa: BLE001
        return _failed(request.analysis_request_id, "MODEL_ERROR", str(exc)), trace

    if not results:
        return _skipped(request.analysis_request_id), trace

    return (
        AnalysisResponse(
            analysis_request_id=request.analysis_request_id,
            status="COMPLETED",
            results=results,
            emotion_analyses=[],
        ),
        trace,
    )


def _skipped(request_id: str) -> AnalysisResponse:
    """분석은 정상 완료됐지만 발동할 기능이 없는 상태 (규격서 12장)."""
    return AnalysisResponse(
        analysis_request_id=request_id,
        status="SKIPPED",
        results=[],
        emotion_analyses=[],
    )


def _failed(request_id: str, code: str, message: str) -> AnalysisResponse:
    return AnalysisResponse(
        analysis_request_id=request_id,
        status="FAILED",
        results=[],
        emotion_analyses=[],
        error_code=code,  # type: ignore[arg-type]
        error_message=message,
    )
