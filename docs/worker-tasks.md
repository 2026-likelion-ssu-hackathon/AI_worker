# kakapo 워커 구현 작업 지시서 (MVP)

이 문서는 **이번 작업 지시**다. 프로젝트 상시 맥락은 `CLAUDE.md`를 본다.

Claude Code 세션 시작 시 이 문서를 읽히고 **단계 번호를 지정해서** 작업을 시킨다.
예: "worker-tasks.md 읽고 1단계 진행해줘"

한 세션에서 여러 단계를 몰아서 하지 않는다. 단계마다 검증하고 다음으로 넘어간다.

---

## 이번 MVP의 목표

**채팅 데이터를 넣으면 적절한 개입을 산출하는 엔진**을 만든다.

```bash
$ python -m tools.run fixtures/case3.json

트리거   one_sided (A 발화 75%)
scope    individual → A
소재     "작년 10월에 갔던 성수동 그 카페, 아직 그대로일까요?"
근거     "작년에 두 분이 함께 가셨던 곳이에요"
```

이게 돌아가면 데모의 대부분이 끝난다.

**구현하지 않는 것**: Redis, Postgres, 회신 API, LangGraph, 스코어링 라우터.
필요해 보여도 사용자에게 먼저 확인할 것.

---

## 0단계 — 세팅

- [ ] `requirements.txt`

  ```
  langchain>=1.3
  langchain-openai>=1.5
  pydantic>=2
  python-dotenv
  ```

  `openai`, `redis`, `psycopg`, `langgraph` 넣지 않는다

- [ ] 폴더 구조 생성 (`CLAUDE.md` 참조)
- [ ] 기존 스켈레톤에 `import openai` / `from openai` 있으면 전부 제거

  ```bash
  grep -rn "import openai\|from openai" worker/
  ```

  이 결과가 0이어야 한다

**검증**: `pip install -r requirements.txt` 통과

---

## 1단계 — 스키마 + 픽스처

### 스키마 (`worker/models.py`)

```python
class Decision(BaseModel):
    scope: Literal["common", "individual"]
    target: Literal["A", "B"] | None = None
    content: str
    reason: str | None = None

class Memory(BaseModel):
    id: str
    kind: Literal["place", "activity", "promise", "wish", "interest"]
    content: str
    source_quote: str
    occurred_at: datetime | None = None
    used_at: datetime | None = None

class GateResult(BaseModel):
    triggered: bool
    trigger: str | None = None   # short_pingpong | no_question | one_sided | stall
    scope: Literal["common", "individual"] | None = None
    target: Literal["A", "B"] | None = None
    needs_llm: bool = False

class JudgeResult(BaseModel):
    should_intervene: bool
    trigger: Literal["routine_loop", "busy_excuse", "none"]
    scope: Literal["common", "individual"] | None = None
    target: Literal["A", "B"] | None = None
    memories: list[Memory] = []
```

### 픽스처 (`fixtures/`)

명세서 예시 4개를 그대로 JSON으로 만든다.

| 파일 | 패턴 | 기대 결과 |
| --- | --- | --- |
| `case1_pingpong.json` | 단답 핑퐁 | `common` |
| `case2_no_question.json` | 질문 없는 대답 | `individual` → A |
| `case3_one_sided.json` | 한쪽만 발화 | `individual` → A |
| `case4_routine.json` | 일상 보고 반복 (LLM 필요) | `common` |
| `case5_busy.json` | 바쁨 표현 있음 | **트리거 안 됨** |

형식:

```json
{
  "room_id": "r1",
  "messages": [
    {"sender": "A", "content": "오늘 뭐했어?", "ts": "2026-08-14T21:03:00+09:00"},
    {"sender": "B", "content": "그냥 집에 있었어", "ts": "2026-08-14T21:05:00+09:00"}
  ]
}
```

**검증**: `python -c "from worker.models import *"` 통과, 픽스처 5개 존재

---

## 2단계 — 룰 게이트

`worker/gate.py`

- [ ] ① 단답 핑퐁 — 종료형 단답 3턴 연속
- [ ] ② 질문 없는 대답 — 되묻는 문장(`?` 또는 의문 어미) 0개로 3턴 경과
- [ ] ③ 한쪽만 발화 — 발화 비중 75% 이상 + 반대쪽은 리액션만
- [ ] ⑤ 대화 중 정체 — 마지막 메시지 후 20분 경과
- [ ] scope 결정 — 발화량 비율 (`CLAUDE.md` 표 참조)
- [ ] 바쁨 표현("회의", "이따 톡", "바빠") 감지 시 `needs_llm=True`. 룰로 확정하지 않는다
- [ ] 룰 미발동이지만 동일 패턴 반복이 의심되면 `needs_llm=True`

명세서의 ⑥ 장기 신호는 MVP에서 제외됐다. 구현하지 않는다.

**주의**: 화자별 톤은 **개인 베이스라인 대비**로 측정한다. 원래 단답형인 사람에게 절대 기준을 적용하면 상시 트리거된다.

**검증**: 픽스처 5개를 넣어 각각 기대한 트리거가 나오는지 확인. `case5_busy`는 반드시 `needs_llm=True`

---

## 3단계 — LLM judge

`worker/judge.py`. 룰로 판정 불가한 케이스만 처리한다.

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-5", temperature=0.3)
judge = model.with_structured_output(JudgeResult, method="json_schema", strict=True)
```

- [ ] 프롬프트는 `worker/prompts/judge.md`로 분리
- [ ] 프롬프트에 포함할 것
  - 트리거 ④(일상 보고형 반복) 판정 기준
  - 바쁨 표현이 진짜인지 핑계인지 구분 기준
  - 기억 추출 지시 (`place`/`activity`/`promise`/`wish`/`interest`)
  - scope 판정 (룰에서 못 정한 경우만)

**중요**: judge는 **감지만** 한다. 사용자에게 보여줄 문구를 여기서 생성하지 않는다.

**검증**: `case4_routine` → `trigger="routine_loop"`, `should_intervene=True`
`case5_busy` → `should_intervene=False`

---

## 4단계 — RAG 기억 검색

`worker/retrieve.py`

```python
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

store = InMemoryVectorStore(OpenAIEmbeddings(model="text-embedding-3-small"))
```

- [ ] `data/memories.json` 시드 작성 — 기억 12건, `kind` 골고루 분포
- [ ] 프로세스 시작 시 1회 인덱싱
- [ ] `retrieve(recent_context, k=3)` — 유사도 검색
- [ ] 검색 결과 중 `used_at is None`인 것 우선 선택
- [ ] 전부 사용됨 + 30일 이내면 `None` 반환 (오늘의 질문으로 폴백)
- [ ] `mark_used(memory_id)` — JSON 파일에 기록

**검증**: `"아 배고파 뭐 먹지"`를 넣었을 때 음식 관련 기억(`wish`)이 상위에 오는지 확인.
장소·활동 기억이 1위로 나오면 시드 데이터나 임베딩 대상 텍스트를 조정한다.
(`content`만 임베딩하지 말고 `source_quote`를 합쳐서 임베딩하면 맥락 매칭이 좋아진다)

---

## 5단계 — 소재 생성 + 필터

### `worker/topic.py`

- [ ] 기억 기반 — LLM이 원문·시점을 포함한 자연스러운 문장 생성. `content` + `reason` 둘 다
- [ ] 오늘의 질문 — `data/daily_questions.json` 30개에서 선택. **LLM 호출 없음**, `reason`은 `None`
- [ ] 질문 풀에 시간대 태그, 감정·관계 질문 후순위 태그

### `worker/filter.py`

- [ ] 금지어 하드 필터 — **문자열 검사로 강제**한다. LLM 판단에 맡기지 않는다

  ```python
  BANNED = ["권태기", "대화가 줄", "서먹", "사이가", "요즘 뜸", "소원해"]
  ```

- [ ] 걸리면 1회 재생성, 또 걸리면 오늘의 질문으로 폴백

**절대 제약**: 두 프롬프트 모두에 "관계 상태 언급 금지"를 명시하고 금지 문구 예시를 직접 넣는다.

**검증**: 기억 1건으로 10회 생성해서 금지어가 한 번도 안 나오는지 확인.
금지 문구가 든 가짜 소재를 필터에 넣어 반드시 차단되는지 확인.

---

## 6단계 — 조립 + CLI

### `worker/pipeline.py`

```python
def run(fixture: dict) -> Decision | None:
    gate = check_gate(fixture["messages"])

    if gate.needs_llm:
        judged = judge(fixture["messages"])
        if not judged.should_intervene:
            return None
        save_memories(judged.memories)
        scope, target = judged.scope, judged.target
    elif gate.triggered:
        scope, target = gate.scope, gate.target
    else:
        return None

    memory = retrieve(recent_context(fixture["messages"]))
    decision = make_topic(memory, scope, target) if memory else daily_question(scope, target)
    return apply_filter(decision)
```

`if` 세 개면 충분하다. LangGraph를 쓰지 않는다.

### `tools/run.py`

- [ ] 픽스처 경로를 인자로 받아 실행
- [ ] 결과를 사람이 읽기 좋게 출력 (위 예시 형식)
- [ ] `--verbose`로 중간 단계(게이트 판정, 검색된 기억 top-3) 표시 — 디버깅용이자 시연용

**검증**: 픽스처 5개 전부 실행. 기대 결과표와 대조.

---

## 완료 기준

- [ ] 픽스처 5개가 전부 기대한 결과를 낸다
- [ ] `case5_busy`는 개입하지 않는다
- [ ] 금지어가 출력에 한 번도 등장하지 않는다
- [ ] 첫 실행부터 기억 기반 소재가 나온다 (오늘의 질문 폴백이 아님)
- [ ] RAG 검색 결과가 대화 맥락과 연결된다

---

## 별도 처리 (코드 작업 아님)

- [ ] `docs/contract-v2.md` 폐기, 새 계약서 작성 — 방 구조가 커플방 1개로 바뀌고 봇 출력이 메시지에서 위젯 페이로드로 바뀌었다. 채팅 서버 담당자와 재합의 필요
- [ ] 서버 담당자에게 전달: 위젯은 **수신자별로 갈라서** 전송해야 한다. 방 단위 브로드캐스트만 하면 개별 코멘트가 양쪽에 다 뜬다
- [ ] PM에게 확인: 나머지 후보 기능 명세
