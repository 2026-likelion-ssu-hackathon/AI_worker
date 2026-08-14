# kakapo 워커 구현 작업 지시서

이 문서는 **작업 지시와 진행 상황**이다. 프로젝트 상시 맥락은 `CLAUDE.md`, 코드 설명은 `README.md`를 본다.

Claude Code 세션 시작 시 이 문서를 읽히고 **단계 번호를 지정해서** 작업을 시킨다.
예: "worker-tasks.md 읽고 8단계 진행해줘"

한 세션에서 여러 단계를 몰아서 하지 않는다. 단계마다 검증하고 다음으로 넘어간다.

---

## 진행 상황 (2026-08-14)

| 단계 | 내용 | 상태 |
| --- | --- | --- |
| 0 | 세팅 | ✅ 완료 |
| 1 | 스키마 + 픽스처 | ✅ 완료 |
| 2 | 룰 게이트 | ✅ 완료 |
| 3 | LLM judge | ✅ 완료 |
| 4 | RAG 기억 검색 | ✅ 완료 |
| 5 | 소재 생성 + 필터 | ✅ 완료 |
| 6 | 조립 + CLI | ✅ 완료 |
| 7 | 갈등 중재 (말투 교정 제안) | ✅ 완료 |
| 8 | 남은 것 — 아래 참조 | ⏸️ 대기 |

---

## MVP의 목표 — 달성함

**채팅 데이터를 넣으면 적절한 개입을 산출하는 엔진.**

```bash
$ python -m tools.run fixtures/case3_one_sided.json

▸ case3_one_sided.json
  트리거   one_sided (한쪽만 발화)
  scope    individual → A
  소재     "5월 9일 지하철 종점까지 갔던 날, 기억나세요?"
  근거     5월 9일 두 분이 지하철로 다녀오신 일이에요
```

**구현하지 않은 것**: Redis, Postgres, 회신 API, LangGraph, 점수 기반 스코어링.
필요해 보여도 사용자에게 먼저 확인할 것.

---

## 0단계 — 세팅 ✅

- [x] `requirements.txt`

  ```
  langchain>=1.3
  langchain-openai>=1.5
  pydantic>=2
  python-dotenv
  numpy          # 지시서에 없었지만 필요해서 추가
  ```

  `openai`, `redis`, `psycopg`, `langgraph` 넣지 않는다

  > **`numpy` 추가 사유**: `InMemoryVectorStore.similarity_search`가 cosine similarity 계산에
  > numpy를 요구한다. 없으면 4단계가 `ImportError`로 죽는다. 금지 목록에는 해당 없다.

- [x] 폴더 구조 생성 (`CLAUDE.md` 참조)
- [x] `import openai` / `from openai` 0건 확인

**검증 결과**: `pip install -r requirements.txt` 통과. `grep -rn "import openai" worker/` 결과 0건.

---

## 1단계 — 스키마 + 픽스처 ✅

`worker/models.py` — `Decision` / `Memory` / `GateResult` / `JudgeResult` 구현.

지시서와 달라진 점 2가지:

- **`Decision`에 `kind` 필드 추가** (`"topic"` | `"tone"`). 7단계에서 후보 기능이 2개가 되면서
  프론트 렌더링 레이아웃을 구분해야 했다. 기본값이 `"topic"`이라 기존 계약은 안 깨진다
- **`*LLMOutput` 스키마를 따로 둠.** strict json_schema 모드가 nullable · date-time을 잘 못 다뤄서,
  LLM에는 sentinel 문자열(`"none"` / `"unknown"`)로 받고 파이썬에서 명세 스키마로 변환한다

`Fixture`에 `now` / `online` 필드를 추가했다. 트리거 ⑤(20분 정체)가 실행 시각에 따라 결과가
바뀌면 재현이 안 되기 때문이다. 없으면 "마지막 메시지 +1분"(= 정체 아님)으로 본다.

**검증 결과**: `python -c "from worker.models import *"` 통과.

---

## 2단계 — 룰 게이트 ✅

`worker/gate.py`

- [x] ① 단답 핑퐁 — 종료형 단답 3턴 연속
- [x] ② 질문 없는 대답 — 되묻는 문장 0개로 3턴 경과
- [x] ③ 한쪽만 발화 — 발화 비중 75% 이상 + 반대쪽은 리액션만
- [x] ⑤ 대화 중 정체 — 마지막 메시지 후 20분 경과
- [x] scope 결정 — 발화량 비율
- [x] 바쁨 표현 감지 시 `needs_llm=True`
- [x] 동일 패턴 반복 의심 시 `needs_llm=True` (날짜별 어휘 Jaccard 30% 이상)

화자별 톤은 **개인 베이스라인 대비**로 측정한다. `_baselines()`가 최근 윈도우를 제외한
앞부분에서 화자별 평소 발화 길이를 잡는다.

**검증 결과** (LLM 없이 실측):

```
case1_pingpong    short_pingpong  common        종료형 단답 5턴 연속 / 양쪽 다 단답
case2_no_question no_question     individual→A  되묻는 문장 0개 / A 발화 82%
case3_one_sided   one_sided       individual→A  A 발화 91% / B 리액션만 100%
case4_routine     needs_llm=True                어휘 겹침 92%
case5_busy        needs_llm=True                바쁨 표현 감지 — 룰로 확정 안 함
case6_stall       stall           common        마지막 메시지 후 35분 경과
```

---

## 3단계 — LLM judge ✅

`worker/judge.py` + `worker/prompts/judge.md`

judge는 **감지만** 한다. 사용자에게 보여줄 문구를 여기서 생성하지 않는다.

`worker/llm.py`를 추가했다 (구조도에 없던 파일). judge와 topic이 모델 설정을 공유해야 해서
`init_chat_model` 호출을 한 곳에 모았다. reasoning 계열 모델이 `temperature`를 거부하면
한 번 재시도하는 처리도 여기 있다.

**검증 결과**: `case4_routine` → `trigger="routine_loop"`, `should_intervene=True`.
`case5_busy` → `should_intervene=False`.

---

## 4단계 — RAG 기억 검색 ✅

`worker/retrieve.py`

- [x] `data/memories.json` 시드 — **27건** (지시서는 12건이었으나 확장)
- [x] 프로세스 시작 시 1회 인덱싱
- [x] `retrieve(recent_context, k=3)` 유사도 검색
- [x] `used_at is None` 우선 선택
- [x] 전부 사용됨 + 30일 이내면 `None` 반환
- [x] `mark_used(memory_id)` JSON 기록

**검증 결과**: `"아 배고파... 뭐 먹지"` → `[wish] 연남동 크림파스타집` 1위, `[wish] 마라탕 맛집` 2위.

> **여기서 실제로 문제가 됐던 것**: 처음엔 `case1`·`case3`에서 RAG가 헛짚었다.
> 원인은 시드 부족이 아니라 **검색 질의**였다. `recent_context()`가 마지막 6개 메시지를
> 그대로 썼는데, 트리거가 걸리는 대화는 끝부분이 리액션으로 채워져 있어서 질의가
> `"다음에 같이 가자 ㅇㅇ 응 그래 ㅇㅇ"`가 되고 검색이 통째로 헛돌았다.
> **리액션을 제외하고 내용이 있는 발화만 모으도록 고쳤다.** 시드 확장은 그 다음 효과였다.

---

## 5단계 — 소재 생성 + 필터 ✅

`worker/topic.py` / `worker/filter.py` / `worker/prompts/topic.md`

- [x] 기억 기반 — LLM이 원문·시점을 포함한 문장 생성, `content` + `reason` 둘 다
- [x] 오늘의 질문 — `data/daily_questions.json` 30개, **LLM 호출 없음**, `reason`은 `None`
- [x] 질문 풀에 시간대 태그(`time_tags`), 감정·관계 질문 후순위 태그(`heavy`)
- [x] 금지어 하드 필터 — 문자열 검사로 강제
- [x] 걸리면 1회 재생성, 또 걸리면 오늘의 질문으로 폴백

**검증 결과**: 기억 1건으로 10회 생성 → **금지어 0건**.
금지 문구가 든 가짜 소재를 넣으면 재생성 → 폴백 경로가 정상 동작.

---

## 6단계 — 조립 + CLI ✅

`worker/pipeline.py` / `tools/run.py`

- [x] 픽스처 경로를 인자로 받아 실행 (여러 개 동시 가능)
- [x] 사람이 읽기 좋게 출력
- [x] `--verbose`로 중간 단계 표시
- [x] `--no-persist` 추가 — `used_at` 기록을 건너뛴다. **반복 시연용**

**완료 기준 대조**

- [x] 픽스처가 전부 기대한 결과를 낸다
- [x] `case5_busy`는 개입하지 않는다
- [x] 금지어가 출력에 한 번도 등장하지 않는다
- [x] 첫 실행부터 기억 기반 소재가 나온다 (오늘의 질문 폴백 0건)
- [x] RAG 검색 결과가 대화 맥락과 연결된다

---

## 7단계 — 갈등 중재 (말투 교정 제안) ✅

PM 명세 반영. 명세 전문은 `CLAUDE.md`의 "후보 기능 2" 절.

### 추가한 파일

```
worker/profile.py              개인 말투 기준선
worker/tone.py                 룰 트리거 6종 + 맥락 판정 + 문구 생성
worker/router.py               후보 기능 우선순위 체인
worker/prompts/tone_judge.md
worker/prompts/tone_suggest.md
data/speaker_profiles.json     A·B 기준선 시드
fixtures/case7_tone.json
fixtures/case8_banter.json
```

### 검증 결과

```
▸ case7_tone.json
  말투게이트 · [generalization] 일반화 화법 '맨날'
             · [address_change] 평소 '오빠, 자기' → '너, 야'
             · [abrupt_change]  마침표 3%→종결 / ㅋ 2.8개→0개 / 이모지 35%→없음
  말투판정   should_suggest=True  is_playful=False  emotion=angry
  scope      individual → A
  방향       '맨날'은 지금 한 번이 아니라 그동안 전부로 들릴 수 있어요
  대체문장   "오빠 오늘 못 온다니까 나 좀 서운했어, 다음엔 미리 말해줘"

▸ case8_banter.json                      ← 예외 처리 검증
  말투판정   should_suggest=False  is_playful=True
             "직전 대화가 가볍고 ㅋㅋ가 붙은 '야 이 바보야'는 장난투;
              B의 화난 패턴(존댓말·말수 증가)과 불일치"
  → 개입하지 않음
```

`case1`~`case6`은 말투 게이트 미발동, 기존 결과 그대로 (회귀 없음).

### 구현하면서 판단한 것

**`abrupt_change`만으로는 발동하지 않게 막았다.** 그대로 두면 단답 핑퐁·대화 정체가 전부
말투 교정으로 넘어가 후보 1의 케이스를 가로챈다. 다른 신호와 함께이거나 하위 신호 3개 이상일 때만.

**호칭 탐지를 어절 단위로 했다.** 부분 문자열로 찾으니 `"미안 금방 끝날 거야"`의 '야',
`"너무 배고파"`의 '너'가 호칭으로 잡혀 `case5`가 오발동했다.

**감정 상태 언급은 금지하지 않는다.** 처음엔 절대 제약을 확장 적용해 `TONE_BANNED`로
"감정적", "진정하" 등을 막았는데, 절대 제약의 근거는 **관계 규정**("너네 권태기다")이지
개인 감정 상태가 아니다. 갈등 중재는 감정이 올라온 걸 짚어주는 게 목적이므로 되돌렸다.
관계 상태 언급은 갈등 중재에서도 그대로 금지다.

---

## 8단계 — 남은 것 ⏸️

### 코드

- [ ] **방향 문구 (a)/(b) 분기 여부 결정** — `emotion`이 `angry`/`irritated`면 감정 짚기를
      우선하고, `hurt`/`calm`이면 표현 설명을 우선할지. 지금은 모델이 알아서 고른다.
      `tone_suggest.md`에 한 문단 추가하면 되고 코드 변경은 없다

### 코드 작업 아님

- [ ] `docs/contract-v2.md` 폐기, 새 계약서 작성 — 방 구조가 커플방 1개로 바뀌고 봇 출력이
      메시지에서 위젯 페이로드로 바뀌었다. 채팅 서버 담당자와 재합의 필요
- [ ] 서버 담당자에게 전달: 위젯은 **수신자별로 갈라서** 전송해야 한다.
      방 단위 브로드캐스트만 하면 개별 코멘트가 양쪽에 다 뜬다
- [ ] **프론트(민상)에게 전달: `Decision.kind` 필드 추가.** 후보 기능마다 화면 배치가 다르다

  | `kind` | 중앙 (크게) | 보조 (작게) |
  | --- | --- | --- |
  | `topic` | `content` — 대화 소재 | `reason` — 근거 문구, **하단** |
  | `tone` | `content` — 대체 문장 | `reason` — 방향 문구, **상단** |

- [ ] **디자인(혜원)에게 전달**: 갈등 중재는 복사 버튼을 만들지 않는다 (명세 제약 사항)
- [ ] PM(영환)에게 확인: 나머지 후보 기능 명세
