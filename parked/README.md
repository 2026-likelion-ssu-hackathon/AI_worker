# 파킹 — 대화 소재 제시 (`topic`)

우선순위에서 밀려 **현재 파이프라인에서 빠진** 후보 기능이다. 코드는 지우지 않고 여기 보관한다.
이 폴더는 파이썬 패키지가 아니라 **아카이브**다. `worker/` 에서 import 되지 않으며, 실행되지도 않는다.

파킹 시점: 백엔드 연동 규격 v1 반영 작업 (`docs/spec-v2.md` 참조)
파킹 직전 상태 태그: `git tag topic-parked`

---

## 무엇이 여기 있나

| 파일 | 원래 위치 | 내용 |
| --- | --- | --- |
| `gate.py` | `worker/gate.py` | 룰 게이트 — 트리거 ①단답 핑퐁 ②질문 없는 대답 ③한쪽만 발화 ⑤대화 중 정체 |
| `judge.py` | `worker/judge.py` | LLM 판정 — 트리거 ④일상 보고형 반복, 바쁨 판별 (+ 기억 추출) |
| `topic.py` | `worker/topic.py` | 소재 생성 (기억 기반) / 오늘의 질문 |
| `prompts/judge.md` | `worker/prompts/judge.md` | 판정 + 기억 추출 프롬프트 |
| `prompts/topic.md` | `worker/prompts/topic.md` | 소재 생성 프롬프트 |
| `daily_questions.json` | `data/daily_questions.json` | 오늘의 질문 30개 |

## 무엇이 남았나 — 중요

대화 소재 기능에 딸려 있었지만 **파킹하지 않은 것**이 있다. 데이트 코스 추천이 그대로 쓰기 때문이다.

| 남긴 것 | 이유 |
| --- | --- |
| `worker/retrieve.py` + `data/memories.json` | RAG 기억 검색. 데이트 코스 추천의 근거가 "3주 전에 성수 가보고 싶다고 한 내용"이라 기억 저장소가 그대로 재료다 |
| `worker/extract.py` | `judge.py` 의 **기억 추출 부분만** 떼어낸 것. 안 그러면 기억이 더 이상 쌓이지 않는다 |
| `worker/text.py` | `gate.py` 안에 있던 문장 판별 유틸(`is_reaction` / `is_question`). `retrieve.py` 와 데이트 코스 트리거가 쓴다 |

---

## 되살리는 방법

1. `gate.py` · `judge.py` · `topic.py` 를 `worker/` 로, 프롬프트를 `worker/prompts/` 로,
   `daily_questions.json` 을 `data/` 로 되돌린다
2. **import 를 고친다** — 파킹 시점 이후 아래가 바뀌었다
   - `gate.py` 안의 `_norm` / `is_reaction` / `is_question` / `_REACTIONS` 는
     `worker/text.py` 로 빠졌다. 중복 정의하지 말고 `from worker.text import ...` 로 바꾼다
   - `worker.models.Message` 의 `ts` 가 **`sent_at`** 으로 바뀌었고 `message_id` 가 생겼다
     (백엔드 규격의 `sentAt` / `messageId`). `m.ts` → `m.sent_at` 으로 전부 치환한다
   - `Decision` 스키마는 없어졌다. 후보는 이제 `AiResult` 를 반환한다
     (`resultType` / `visibilityType` / `targetParticipant` / `contentType` /
     `triggerMessageIds` / `resultData`). `topic.py` 의 반환부를 여기에 맞춰 다시 쓴다
   - `judge.py` 의 기억 추출은 `worker/extract.py` 로 이미 나가 있다. 되살릴 때는
     판정 부분만 남기고 추출을 중복 실행하지 않도록 한다
3. `worker/router.py` 의 `CANDIDATES` 에 `TopicCandidate()` 를 끼운다 (우선순위 마지막)
4. 백엔드 규격에 `resultType` 값을 추가해야 한다 — 지금 규격에는
   `TONE_CORRECTION` / `DATE_RECOMMENDATION` / `YOUTUBE_RECOMMENDATION` 세 개뿐이라
   **서버 담당자와 재합의가 필요하다.** 워커만 고치면 응답 검증에서 막힌다

## 왜 파킹했나

기능 자체에 문제가 있어서가 아니다. 후보 기능이 3개로 늘면서 우선순위가 밀렸고,
위젯 슬롯 정책이 확정되지 않은 상태라 동작하지 않을 코드를 파이프라인에 두는 것보다
아카이브가 낫다고 판단했다. 검증까지 끝난 코드이므로 되살릴 때 로직은 손댈 필요가 없다.
