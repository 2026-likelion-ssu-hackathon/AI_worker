# kakapo AI 워커

커플의 채팅을 분석해 **대화 소재를 제안하는 판정 엔진**. 채팅 데이터를 넣으면 개입 여부와 소재를 산출한다.

프로젝트 상시 맥락은 [`CLAUDE.md`](CLAUDE.md), 작업 지시는 [`docs/worker-tasks.md`](docs/worker-tasks.md)를 본다.
이 문서는 **코드가 어떻게 생겼는지**만 설명한다.

---

## 절대 제약

**관계 상태를 언급하는 문구를 절대 출력하지 않는다.**

"요즘 대화가 줄었어요", "권태기 같아요", "두 분 사이가 서먹해 보여요" 같은 문구는 금지다.
누군가에게 "너네 권태기다"라는 말을 들으면 의식이 심해져서 오히려 관계가 나빠지기 때문이다.

그래서 **감지 로직과 발화 로직이 완전히 분리되어 있다.**

- `gate.py` / `judge.py` — 갈등·권태를 **감지**한다. 사용자에게 보여줄 문구는 만들지 않는다
- `topic.py` — 소재를 **생성**한다. 왜 개입하는지는 모른다. 그냥 기억 하나를 자연스럽게 꺼낼 뿐이다
- `filter.py` — 그럼에도 금지어가 새어나오면 **문자열 검사로 막는다**

프롬프트를 수정하거나 새 출력 문구를 만들 때 이 제약을 반드시 확인할 것.

---

## 빠른 시작

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env      # OPENAI_API_KEY 채우기

.venv/bin/python -m tools.run fixtures/case3_one_sided.json --verbose --no-persist
```

출력:

```
▸ case3_one_sided.json
  트리거   one_sided (한쪽만 발화)
  scope    individual → A
  소재     "5월 9일 지하철에서 졸다 종점까지 간 날, 기억나세요?"
  근거     2026년 5월 9일에 두 분이 다녀오신 당일치기예요
```

### CLI 옵션

| 옵션 | 설명 |
| --- | --- |
| `--verbose` / `-v` | 게이트 판정, judge 결과, RAG 검색 top-3 표시 |
| `--no-persist` | `used_at`·기억 저장을 파일에 쓰지 않는다. **반복 시연할 때 쓴다** |

`--no-persist` 없이 돌리면 소환한 기억에 `used_at`이 찍혀서 다음 실행 때 다른 소재가 나온다.
시연 리허설은 `--no-persist`로 돌리는 게 안전하다.

### 환경변수 (`.env`)

| 키 | 기본값 | 설명 |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | 필수 |
| `KAKAPO_MODEL` | `openai:gpt-5` | `init_chat_model` 형식. 느리면 `openai:gpt-5-mini` |
| `KAKAPO_TEMPERATURE` | `0.3` | 비우면 temperature 를 아예 안 넘긴다 |
| `KAKAPO_EMBEDDING_MODEL` | `text-embedding-3-small` | 기억 임베딩용 |

---

## 파이프라인

**위젯 슬롯은 1개다.** 후보 기능이 여러 개 발동해도 하나만 나간다.
`router.py` 가 우선순위 순으로 시도하고, 먼저 결과를 내는 후보가 이긴다.

```
fixtures/case.json
   ↓
router.route()
   │
   ├─ 후보 1. 갈등 중재 (tone) ─ 갈등이 감지된 순간에 대화 소재를 던질 때가 아니다
   │     check_tone_gate()   룰 — 공격·오해 유발 표현 6종
   │        ↓ 걸리면
   │     tone_judge()        LLM — 진짜 갈등인가, 맥락상 장난인가
   │        ↓ 진짜면
   │     tone_suggest()      LLM — 방향 문구 + 대체 문장
   │        ↓ 장난이면 아래 후보로 내려감
   │
   └─ 후보 2. 대화 소재 (topic)
         check_gate()        룰 — 트리거 ①②③⑤
            ↓ needs_llm 이면
         judge()             LLM — 트리거 ④, 바쁨 판별 + 기억 추출
            ↓
         retrieve()          RAG — 대화 맥락과 유사한 기억 검색
            ↓
         make_topic()        소재 생성 (기억 기반) 또는 오늘의 질문
   ↓
filter()      금지어 하드 필터 (후보별로 금지어 세트가 다르다)
   ↓
Decision      콘솔 출력
```

후보가 2개인 지금은 **우선순위 체인**으로 충분하다. CLAUDE.md 가 "정교한 스코어링 로직을
구현하지 않는다"고 못박아 뒀다. 후보가 늘어 우선순위만으로 못 정하게 되면 그때 점수 기반으로 바꾼다.
LangGraph 는 쓰지 않는다.

---

## 모듈별 설명

### `worker/models.py` — 스키마

Pydantic 모델 전부. LLM 을 태우지 않는다.

| 모델 | 용도 |
| --- | --- |
| `Message` / `Fixture` | 입력. `Fixture`는 `now`(⑤ 판정 기준 시각), `online`(접속 중인 사람)을 갖는다 |
| `Decision` | **최종 출력.** `kind` / `scope` / `target` / `content` / `reason` |
| `SpeakerProfile` | 개인별 평소 말투 기준선 |
| `ToneGateResult` / `ToneFlag` | 말투 룰 트리거 결과 |
| `Memory` | 기억 1건. `kind` 5종, `used_at` 이 "미소환 우선"과 "30일 중복 금지"를 동시에 처리 |
| `GateResult` | 룰 게이트 판정 결과 |
| `JudgeResult` | LLM 판정 결과 |
| `ExtractedMemory` / `JudgeLLMOutput` / `TopicLLMOutput` | **LLM 구조화 출력 전용** |

`*LLMOutput` 이 따로 있는 이유: strict json_schema 모드가 nullable·date-time 을 잘 못 다룬다.
LLM 에는 sentinel 문자열(`"none"` / `"unknown"`)과 id·시각 없는 형태로 받고,
`JudgeLLMOutput.to_result()` 가 명세 스키마로 변환한다.

**`Decision.kind` 는 프론트 렌더링 계약이다.** 후보 기능마다 화면 배치가 다르다.

| `kind` | 중앙 (크게) | 보조 (작게) |
| --- | --- | --- |
| `topic` | `content` — 대화 소재 | `reason` — 근거 문구, **하단** |
| `tone` | `content` — 대체 문장 | `reason` — 방향 문구, **상단** |

기본값이 `"topic"` 이라 기존 계약은 깨지지 않는다.

### `worker/gate.py` — 룰 게이트

**LLM 을 전혀 쓰지 않는다.** 정규식과 통계뿐이다.

```python
check_gate(messages, now=..., online=[...]) -> GateResult
```

판정 순서 (앞에서 걸리면 거기서 끝):

| 순서 | 트리거 | 조건 |
| --- | --- | --- |
| 0 | **바쁨 표현** | "회의", "이따 톡", "바빠" 등 감지 → `needs_llm=True`. **룰로 확정하지 않는다** |
| 1 | ① `short_pingpong` | 종료형 단답 3턴 연속 |
| 2 | ③ `one_sided` | 발화 비중 75% 이상 + 반대쪽은 리액션만 (80% 이상) |
| 3 | ② `no_question` | 최근 6개 메시지에 되묻는 문장 0개 |
| 4 | ⑤ `stall` | 마지막 메시지 후 20분 경과 + 양쪽 접속 중 |
| 5 | ④ 후보 | 날짜별 어휘 겹침(Jaccard) 30% 이상 → `needs_llm=True` |

**톤 판정은 전부 화자 개인 베이스라인 대비다.** `_baselines()` 가 최근 윈도우를 제외한 앞부분에서
화자별 평소 발화 길이를 잡고, 거기에 못 미치면 "짧아졌다"고 본다.
원래 단답형인 사람에게 절대 기준을 적용하면 상시 트리거되기 때문이다.

`decide_scope()` 는 발화량 비율로 노출 대상을 정한다. LLM 호출 불필요.

| 조건 | scope | target |
| --- | --- | --- |
| 양쪽 다 단답 70% 이상 | `common` | — |
| 한쪽 발화 70% 이상 | `individual` | 말 거는 쪽 |
| 한쪽 단답 70% 이상 | `individual` | 단답 보내는 쪽 |

주요 상수 — 튜닝은 여기서 한다:

```python
WINDOW = 6                      # 판정 윈도우 (메시지 개수)
PINGPONG_STREAK = 3             # ① 연속 단답 임계
ONE_SIDED_CHAR_RATIO = 0.75     # ③ 발화 비중
ONE_SIDED_REACTION_RATIO = 0.8  # ③ 반대쪽 리액션 비율
SCOPE_TALK_RATIO = 0.70         # scope 결정
SCOPE_SHORT_RATIO = 0.70
STALL_AFTER = timedelta(minutes=20)
ROUTINE_JACCARD = 0.30          # ④ 후보 판정
```

### `worker/llm.py` — LLM 접근 레이어

**모든 LLM 호출은 여기를 거친다.** `import openai` 직접 호출은 금지 (트레이싱 일관성).

```python
ask(schema, system, user) -> schema   # 구조화 출력 단발 호출
load_prompt("judge") -> str           # worker/prompts/judge.md 를 읽는다
```

- `init_chat_model` + `with_structured_output(schema, method="json_schema", strict=True)`
- 툴 루프가 없는 단발 분류/생성이라 `create_agent` 를 쓰지 않는다. 레이턴시와 비용만 늘어난다
- reasoning 계열 모델이 `temperature` 를 거부하면 한 번 재시도한다

### `worker/judge.py` — LLM 판정

룰로 확정 못 한 케이스(`needs_llm=True`)만 온다. **감지만 하고 문구는 만들지 않는다.**

```python
judge(messages) -> JudgeResult
```

- 트리거 ④(일상 보고형 반복) 판정
- 바쁨이 진짜인지 핑계인지 구분
- **기억 추출을 같은 호출에서 함께 한다.** 별도 배치 파이프라인을 만들지 않는다
- 룰이 못 정한 경우에만 scope 판정

`format_transcript()` 는 날짜가 바뀌면 `--- 2026-08-13 ---` 구분선을 넣는다. 반복 패턴을 모델이 보기 쉽게 하려는 것.

**지어낸 기억은 버린다.** `source_quote` 가 원문에 실제로 없으면 저장하지 않는다.

프롬프트는 [`worker/prompts/judge.md`](worker/prompts/judge.md).

### `worker/retrieve.py` — RAG 기억 검색

```python
recent_context(messages, n=6) -> str   # 검색 질의 생성
search(query, k=3) -> list[Memory]     # 유사도 상위 k건 (디버깅용)
retrieve(recent, k=3) -> Memory | None # 쓸 수 있는 기억 1건
mark_used(memory_id)                   # used_at 기록
save_memories(new)                     # judge 가 뽑은 기억 저장 (id 중복 무시)
```

`OpenAIEmbeddings` + `InMemoryVectorStore`. 기억이 30건 안쪽이라 pgvector·Chroma 를 띄우지 않는다.
프로세스 시작 시 `data/memories.json` 을 한 번 인덱싱한다.

**주의할 지점 두 개:**

1. **`recent_context()` 는 리액션을 제외한다.** 트리거가 걸리는 대화는 끝부분이 `ㅇㅇ 응 그래`로
   채워져 있어서, 단순히 마지막 6개를 쓰면 질의가 `"다음에 같이 가자 ㅇㅇ 응 그래 ㅇㅇ"` 가 되고
   검색이 통째로 헛돈다. 내용이 있는 발화만 모은다

2. **임베딩 대상은 `content + source_quote` 다.** `content` 만 넣으면 맥락 매칭이 약하다

선택 규칙: 유사도 상위 k건 중 `used_at is None` 인 것 우선 → 전부 소환됐으면 30일 지난 것만 재사용 → 없으면 `None`(오늘의 질문 폴백).

### `worker/topic.py` — 소재 산출

```python
make_topic(memory, scope, target, recent=None) -> Decision  # LLM 호출 O
daily_question(scope, target, now=None) -> Decision          # LLM 호출 X
```

**우선순위 1 — 기억 기반.** LLM 이 원문·시점을 살려 문장을 만든다. `content` + `reason` 둘 다 생성.

**우선순위 2 — 오늘의 질문.** `data/daily_questions.json` 30개에서 고른다. **LLM 호출 없음, 즉시 응답.**
`reason` 은 `None` — 기억이 없어 대체한 것이므로 근거 문구를 생략한다.

질문 선택 규칙:
1. 30일 내 사용한 것 제외
2. 현재 시간대(`morning` / `afternoon` / `evening` / `late_night`)에 맞는 것
3. `heavy`(감정·관계 질문) 후순위, **심야에는 아예 제외**

프롬프트는 [`worker/prompts/topic.md`](worker/prompts/topic.md).

### `worker/filter.py` — 금지어 하드 필터

```python
find_banned(text) -> str | None   # 걸린 금지어
is_clean(decision) -> bool
apply_filter(decision, regenerate=None, fallback=None) -> Decision
```

**문자열 검사로 강제한다. LLM 판단에 맡기지 않는다.** 프롬프트에도 금지 지시를 넣지만 그것만 믿지 않는다.

```python
BANNED = ["권태기", "대화가 줄", "서먹", "사이가", "요즘 뜸", "소원해",
          "멀어지", "데면데면", "예전보다", "요즘 들어", ...]
```

**금지어 세트는 후보와 무관하게 하나다.** 절대 제약이 "관계 상태 언급 금지" 하나이기 때문이다.

> **갈등 중재에서 감정 상태를 언급하는 것은 금지가 아니다.**
> 절대 제약의 근거는 "너네 권태기다" 같은 **관계 규정**을 들으면 의식이 심해진다는 것이고,
> 그건 대화 소재 쪽 얘기다. 갈등 중재는 지금 감정이 올라온 걸 짚어주는 게 기능의 목적이다.
> `"지금 감정이 올라와 있는 것 같아요"` 는 통과하고, `"요즘 두 분 사이가"` 는 여기서도 막힌다.

사람을 평가하는 표현("무례하시네요")은 하드 필터가 아니라 프롬프트에서 다룬다.
단어 목록으로는 `"공격적으로 들릴 수 있어요"`(정상)와 `"공격적이시네요"`(문제)를 구분할 수 없다.

폴백 규칙은 후보마다 다르다.

| kind | 걸렸을 때 |
| --- | --- |
| `topic` | 1회 재생성 → 또 걸리면 오늘의 질문으로 폴백 → 그것도 걸리면 예외 |
| `tone` | 1회 재생성 → 또 걸리면 **아무것도 내보내지 않는다** |

`tone` 에서 오늘의 질문으로 폴백하면 맥락이 완전히 어긋난다. 싸우는 중에 "요즘 듣는 노래 있어요?"가
뜨는 셈이다. 절대 제약이 응답 가용성보다 우선이라고 판단했다.

### `worker/profile.py` — 개인 말투 기준선

```python
resolve_profile(speaker, messages) -> SpeakerProfile
compute_profile(messages, speaker) -> SpeakerProfile   # 대화에서 직접 계산
addresses_in(text) -> list[str]                        # 문장에 등장한 호칭
describe(profile) -> str                               # 프롬프트에 넣을 요약
```

**말투 교정은 절대 기준으로 판정하면 안 된다.** 평소 "ㅇㅇ"만 보내는 사람의 "ㅇㅇ"은 무례가 아니고,
원래 호칭이 "야"인 커플에게 "야"는 호칭 변화가 아니다. 특정 단어가 아니라 **그 사람의 평소 대비 변화량**을 본다.

측정 항목: 평균 길이 / 마침표 종결 비율 / 메시지당 ㅋ·ㅎ 개수 / 이모지 사용률 / 평소 호칭 상위 2개 /
갈등 시 어휘 패턴(`conflict_style`, 시드에서만 제공).

`data/speaker_profiles.json` 시드가 있으면 그것을, 없으면 대화에서 계산한다.
시드가 필요한 이유는 기억 시드와 같다 — 픽스처 하나로는 "평소 ㅋ 3개 → 이번엔 0개" 같은 대비를 만들 수 없다.

**`addresses_in()` 주의**: 부분 문자열로 찾으면 "거야"의 '야', "너무"의 '너'까지 호칭으로 잡힌다.
어절 단위로 보고 뒤에 조사만 붙은 경우까지만 인정한다.

### `worker/tone.py` — 갈등 중재 (말투 교정 제안)

```python
check_tone_gate(messages, profile=None) -> ToneGateResult   # 룰, LLM 없음
tone_judge(messages, gate, profile) -> ToneJudgeLLMOutput   # 맥락 판정
tone_suggest(messages, gate, profile, judged) -> ToneSuggestLLMOutput
```

기존 트리거(①~⑤)가 "대화 흐름"을 보는 것과 달리 **방금 전송된 메시지 하나**를 본다.

룰 트리거 6종:

| kind | 내용 |
| --- | --- |
| `insult` | 인신공격 · 욕설 |
| `generalization` | 일반화 화법 ("넌 늘 그런 식이야", "한 번을") |
| `sarcasm` | 비꼼 · 반어 ("잘한다 ㅋㅋ") |
| `address_change` | 호칭 변화 (평소 '오빠' → '야') |
| `repetition` | 비슷한 말 반복 ("전화 받아" → "받아" → "받으라고") |
| `abrupt_change` | 평소 대비 급변 (마침표 종결 / ㅋ 사라짐 / 길이 급변 / 이모지 사라짐) |

**`abrupt_change` 는 약한 증거다.** 단답 핑퐁이나 대화가 잦아든 상황에서도 그대로 걸린다.
다른 신호와 함께일 때만 세고, 혼자서는 하위 신호 3개 이상(`ABRUPT_ALONE_SIGNALS`)일 때만 인정한다.
이 제한이 없으면 기존 `case1`~`case6` 이 전부 말투 교정에 가로채인다.

**판정과 생성을 따로 호출하는 이유**: 이 기능의 최대 리스크는 "와 미친 ㅋㅋ" 같은 장난을 갈등으로
오인하는 것이다. 판정 프롬프트를 판정에만 집중시키고, 통과한 경우에만 생성한다.
대부분의 메시지는 판정에서 걸러지므로 호출 비용도 오히려 줄어든다.

프롬프트는 [`tone_judge.md`](worker/prompts/tone_judge.md) / [`tone_suggest.md`](worker/prompts/tone_suggest.md).
대체 문장은 **나 전달법**(상황 / 영향 / 감정 / 바람)으로 쓴다. 주어를 '너'에서 '나'로 바꿔 행동과 감정만 남긴다.

> **화면에 복사 버튼을 만들지 않는다.** 직접 보고 타이핑해야 의미가 있다는 판단이라
> 대체 문장은 짧고 외우기 쉬워야 한다. 프롬프트에 명시되어 있다.

### `worker/router.py` — 개입 방향 결정

```python
route(ctx: Context) -> Decision | None
CANDIDATES = [ToneCandidate(), TopicCandidate()]   # 우선순위 순
```

후보를 순서대로 시도하고, 먼저 결과를 내는 후보가 이긴다.
후보가 `None` 을 돌려주면 다음 후보로 내려간다 — 말투 게이트가 걸렸지만 LLM 이 "장난"으로 판정하면
대화 소재 후보로 넘어가는 식이다.

후보를 추가하려면 `build(ctx) -> Decision | None` 을 구현해 `CANDIDATES` 에 끼우면 된다.

### `worker/pipeline.py` — 조립

```python
run(fixture: dict) -> Decision | None
run_traced(fixture, persist=True) -> (Decision | None, Trace)
```

픽스처를 읽어 `Context` 를 만들고 라우터에 넘기는 얇은 층이다. 후보별 로직은 `router.py` 에 있다.

`Trace` 는 `--verbose` 용 중간 기록(말투 게이트·판정, 대화 게이트·judge, RAG top-3, 소재 출처)이다.
판정에는 관여하지 않는다.

### `tools/run.py` — CLI 러너

픽스처 경로를 인자로 받아 실행하고 사람이 읽기 좋게 출력한다. 여러 개를 한 번에 넘길 수 있다.

---

## 데이터

### `data/memories.json` — 기억 시드 (27건)

```json
{
  "id": "m01",
  "kind": "place",
  "content": "성수동 카페 오르에르",
  "source_quote": "여기 분위기 진짜 좋다 다음에 또 오자",
  "occurred_at": "2025-10-14T15:20:00+09:00",
  "used_at": null
}
```

| kind | 의미 | 건수 |
| --- | --- | --- |
| `place` | 함께 간 장소 | 6 |
| `activity` | 함께 한 활동 | 6 |
| `promise` | 지키지 못한 약속 | 5 |
| `wish` | 저장만 하고 안 쓴 위시 | 5 |
| `interest` | 두 사람의 관심사 | 5 |

**시드가 필수다.** 실시간 추출만으로는 데모 시작 시점에 기억이 0건이라 오늘의 질문만 나온다.
`m02` 는 `used_at` 이 찍혀 있어 소환 메커니즘을 시연에서 보여준다.

### `data/daily_questions.json` — 오늘의 질문 풀 (30개)

```json
{"id": "q01", "text": "...", "time_tags": ["evening", "late_night"], "heavy": false, "used_at": null}
```

`time_tags` 는 `morning` / `afternoon` / `evening` / `late_night` / `any`.
`heavy: true` 는 감정·관계 질문으로 후순위이며 심야에는 제외된다.

**의도적으로 LLM 생성이 아니다.** 폴백 경로라 즉시 응답해야 하고, 사람이 검수한 고정 풀이어야
"폴백의 폴백" 문제(금지어 필터에 또 걸리는 것)가 원천적으로 없다.

### `fixtures/` — 시연용 입력 대화

```json
{
  "room_id": "r1",
  "now": "2026-08-14T22:41:00+09:00",
  "online": ["A", "B"],
  "messages": [
    {"sender": "A", "content": "오늘 뭐했어?", "ts": "2026-08-14T21:03:00+09:00"}
  ]
}
```

`sender` 는 `"A"` / `"B"` 만 받는다.
`now` 는 ⑤(20분 정체) 판정 기준 시각이다. 없으면 "마지막 메시지 + 1분"(= 정체 아님)으로 본다.
실행 시각에 따라 결과가 바뀌지 않게 하려고 픽스처에 박아둔다.

| 파일 | 패턴 | 기대 결과 |
| --- | --- | --- |
| `case1_pingpong.json` | 단답 핑퐁 | `short_pingpong` / `common` |
| `case2_no_question.json` | 질문 없는 대답 | `no_question` / `individual` → A |
| `case3_one_sided.json` | 한쪽만 발화 | `one_sided` / `individual` → A |
| `case4_routine.json` | 일상 보고 반복 (LLM 필요) | `routine_loop` / `common` |
| `case5_busy.json` | 바쁨 표현 있음 | **개입하지 않음** |
| `case6_stall.json` | 20분 정체 | `stall` / `common` |
| `case7_tone.json` | 일반화 + 호칭 변화 + 말투 급변 | `tone` / `individual` → A |
| `case8_banter.json` | "야 이 바보야 ㅋㅋ" (장난) | **개입하지 않음** ← 예외 처리 |

`case8` 이 이 기능의 안전장치다. 비속 표현이 있어도 맥락상 장난이면 개입하지 않아야 한다.
여기서 잔소리가 뜨면 사용자는 앱을 지운다.

### `data/speaker_profiles.json` — 개인 말투 기준선 시드

```json
{
  "speaker": "A",
  "avg_length": 21.0,
  "period_rate": 0.03,
  "laugh_per_msg": 2.8,
  "emoji_rate": 0.35,
  "top_address": ["오빠", "자기"],
  "conflict_style": "화가 나면 문장이 짧아지고 평소 안 쓰던 마침표를 찍는다"
}
```

`conflict_style` 은 계산으로 뽑을 수 없어 시드에만 있다. LLM 판정에 그대로 넘어가며,
실제로 `case8` 에서 "B의 화난 패턴(존댓말·말수 증가)과 불일치해 장난으로 보임" 이라는 근거로 쓰였다.

---

## 기술 스택

- Python 3.11+
- **LangChain 1.x** — `langchain`, `langchain-openai`
- `numpy` — `InMemoryVectorStore` 의 cosine similarity 계산에 필요

### LangChain 을 쓴 곳은 2개 파일뿐이다

```
worker/llm.py       init_chat_model + with_structured_output
worker/retrieve.py  OpenAIEmbeddings + InMemoryVectorStore
```

나머지(`gate` / `filter` / `topic` 의 오늘의 질문 / `pipeline`)는 전부 순수 파이썬이다.
LangChain 은 OpenAI 를 부르는 어댑터로만 쓰고, 판정 품질을 만드는 로직은 직접 짠 코드다.

### 금지 사항

- ❌ `langchain-classic` 설치
- ❌ `LLMChain`, `initialize_agent`, `AgentType`, `ConversationBufferMemory` 등 0.x 레거시 API
- ❌ `create_agent` — judge 는 툴 루프가 없는 단발 분류다
- ❌ `import openai` 직접 호출 — 모든 LLM 호출은 `worker/llm.py` 를 거친다
- ⏸️ `langgraph` — 분기가 3개뿐이라 `if` 로 충분하다

인터넷에서 찾은 LangChain 예제 대부분은 0.x 기준이다. 참고 전에 위 목록에 걸리는지 확인할 것.

---

## 구현하지 않은 것

MVP 범위 밖이다. **필요해 보여도 사용자에게 먼저 확인할 것.**

| | 지금 (MVP) | 후속 |
| --- | --- | --- |
| 입력 | `fixtures/*.json` | Redis 큐 |
| 기억 저장 | `data/memories.json` | Postgres |
| 출력 | 콘솔 | 채팅 서버로 POST |
| 배선 | 함수 호출 | LangGraph `StateGraph` |

- **회신 API (`/internal/widget`)** — 기존 `docs/contract-v2.md` 는 방 구조가 바뀌면서 폐기됐다.
  채팅 서버 담당자와 재합의가 필요하다
- **트리거 ⑥ 장기 신호** — 6주치 데이터가 필요해 해커톤 시연에서 재현 불가. 관련 통계 수집 코드도 넣지 않았다
- **점수 기반 스코어링** — 후보가 2개(갈등 중재 / 대화 소재)뿐이라 `router.py` 는 우선순위 체인이다.
  후보가 늘어 우선순위만으로 못 정하게 되면 그때 점수 기반으로 바꾼다

---

## 서버 담당자에게 전달할 것

**위젯은 수신자별로 갈라서 전송해야 한다.**

`Decision.scope` 가 `individual` 이면 `target` 인 사람에게만 보낸다.
방 단위 브로드캐스트만 하면 개별 코멘트가 양쪽에 다 뜬다.

B 에게 개별 코멘트가 갔을 때 A 는 그런 게 떴다는 사실 자체를 몰라야 한다.
지적성 피드백이 상대에게 노출되면 그 자체가 갈등 소재가 되기 때문이다.

---

## 비용

gpt-5 기준 판정 1회 약 $0.009. 픽스처 6개 전체 실행 1회에 약 $0.06.
임베딩(`text-embedding-3-small`)은 $0.02/1M 토큰이라 사실상 무시해도 된다.

비용의 90%는 gpt-5 의 내부 추론 토큰이다. 부담되면 `.env` 에서 `KAKAPO_MODEL=openai:gpt-5-mini` 로
바꾸면 코드 수정 없이 적용되고 체감 속도도 빨라진다.

이 워커는 무한 루프가 없다. `python -m tools.run` 한 번 = 호출 몇 번 하고 종료다.
