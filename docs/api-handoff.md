# AI 워커 HTTP 연동 — 서버 담당자 전달본

요청 주신 구성 그대로 만들었습니다. **아래 JSON 은 전부 실제로 서버를 띄우고 받은 응답**이고,
손으로 쓴 예시가 아닙니다.

```
분석 요청  POST  http://ai-worker.railway.internal:8000/internal/v1/chat-analyses
헬스체크   GET   http://ai-worker.railway.internal:8000/health
```

같은 Railway 프로젝트 안의 사설망 주소입니다. **공개 도메인은 만들지 않았습니다.**

연동 규격 자체는 `docs/contract-v1.md`(초안 v1) 기준이고, 규격에 대한 답변은
`docs/contract-review.md` 에 있습니다. 이 문서는 **붙이는 방법**만 다룹니다.

---

## 요청 주신 항목

| 요청 | 결과 |
| --- | --- |
| `POST /internal/v1/chat-analyses` | ✅ |
| 기존 `worker/models.py` DTO 재사용 | ✅ 그대로 씁니다. 별도 DTO 를 만들지 않았습니다 |
| `analysisRequestId` 그대로 반환 | ✅ 검증 실패·타임아웃일 때도 실어 보냅니다 |
| 내부에서 `analyze()` 호출 | ✅ CLI·devui 와 같은 함수입니다 |
| 완성된 JSON 구조 그대로 반환 | ✅ CLI `--json` 출력과 바이트 단위로 같습니다 |
| `GET /health` | ✅ |
| `0.0.0.0` 으로 실행 | ⚠️ **IPv4·IPv6 둘 다 받게 했습니다.** 아래 "Railway 사설망" 참조 |
| 포트는 `PORT` 환경변수 | ✅ 없으면 8000 |
| `/docs` · `/openapi.json` | ✅ |
| 처리시간 30초 이내 | ✅ 실측 최장 11.3초. 25초에서 워커가 먼저 끊고 `ANALYSIS_TIMEOUT` 반환 |
| CORS | 설정하지 않았습니다 |
| 인증 헤더 | 없습니다 |

---

## 1. 엔드포인트

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| `POST` | `/internal/v1/chat-analyses` | 공통 분석 요청 → 공통 분석 응답 |
| `GET` | `/health` | 헬스체크 |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/openapi.json` | OpenAPI 원본 |

**포트** — `PORT` 환경변수. 없으면 **8000**.

### `/health`

```json
{ "status": "UP", "features": { "openai": true, "kakao": true, "youtube": true } }
```

`features` 는 **외부 API 키가 꽂혀 있는지**입니다. 키가 없어도 워커는 죽지 않고 그 기능만
조용히 미발동하는데, 그러면 화면에서는 "추천할 게 없었다"와 구분이 안 됩니다. 그래서
헬스체크에 같이 실었습니다. 값은 내보내지 않습니다.

---

## 2. 로컬 실행

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # OPENAI_API_KEY 필수, 나머지는 선택

.venv/bin/python -m worker.api            # 기본 8000
PORT=8123 .venv/bin/python -m worker.api  # 포트 지정
```

```
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/internal/v1/chat-analyses \
  -H 'Content-Type: application/json' \
  --data-binary @fixtures/case7_tone.json
```

> ⚠️ **`uvicorn worker.api:app --host 0.0.0.0` 으로 직접 띄우지 말아 주세요.**
> `python -m worker.api` 가 IPv4·IPv6 를 같이 받는 소켓을 만들어 넘깁니다. 이유는 아래.

### Railway 사설망 — `0.0.0.0` 을 못 쓰는 이유

Railway 의 프로젝트 내부 통신은 **IPv6 전용**입니다. `0.0.0.0` 으로 묶으면 IPv4 만 듣게
되어 같은 프로젝트의 채팅 서버가 연결하지 못합니다. 반대로 `--host ::` 만 주면 uvicorn 이
IPv6 전용 소켓을 만들어(asyncio 가 `IPV6_V6ONLY` 를 켭니다) `127.0.0.1` 접속이 전부
끊깁니다. **둘 다 실측으로 확인했습니다.**

그래서 `worker/api.py` 의 `main()` 이 `IPV6_V6ONLY` 를 끈 듀얼스택 소켓을 직접 만들어
uvicorn 에 넘깁니다. `127.0.0.1` 도 `[::1]` 도 같은 서버에 닿습니다.

호출은 이렇게 하시면 됩니다.

```
http://ai-worker.railway.internal:8000/internal/v1/chat-analyses
```

**IPv6 로 해석되는 주소입니다.** 자바 쪽에서 IPv4 를 강제하는 옵션
(`-Djava.net.preferIPv4Stack=true`)이 켜져 있으면 이 호스트를 못 찾습니다.

---

## 3. 요청 예시 (실제 페이로드)

`fixtures/case7_tone.json` — 규격서 5장 형식 그대로입니다.

```json
{
  "analysisRequestId": "b1287837-cc79-5cf9-86d1-10c6b81f8a92",
  "chatRoomId": 1,
  "participants": [
    { "participantKey": "USER_A" },
    { "participantKey": "USER_B" }
  ],
  "messages": [
    { "messageId": 101, "sender": "USER_A", "content": "오빠 오늘 몇 시에 와? 저녁 같이 먹으려고 기다리고 있었어", "sentAt": "2026-08-14T20:31:00+09:00" },
    { "messageId": 102, "sender": "USER_B", "content": "아 미안 오늘 야근 잡혀서 못 갈 것 같아", "sentAt": "2026-08-14T20:40:00+09:00" },
    { "messageId": 103, "sender": "USER_A", "content": "또? 지난주에도 그랬잖아", "sentAt": "2026-08-14T20:43:00+09:00" },
    { "messageId": 104, "sender": "USER_B", "content": "진짜 미안해 다음엔 꼭 갈게", "sentAt": "2026-08-14T20:47:00+09:00" },
    { "messageId": 105, "sender": "USER_A", "content": "야 너는 맨날 그런 식이야.", "sentAt": "2026-08-14T20:51:00+09:00" }
  ]
}
```

**선택 필드가 세 개 더 있습니다.** 안 보내셔도 지금처럼 동작합니다.

| 필드 | 용도 | 협의 |
| --- | --- | --- |
| `recentResults` | 같은 영상·장소 재추천 방지 | `contract-review.md` 1번 (합의) |
| `speakerProfiles` | 화자별 평소 말투 기준선 | `contract-review.md` 2번 (합의) |
| `requestedAt` | 분석 기준 시각. 없으면 마지막 메시지 시각 | 서버에 요청하지 않는 값 |

`sentAt` 에 오프셋이 없으면 **KST 로 해석**합니다.

---

## 4. 응답 예시 (실제 응답)

### ① 말투 교정 — 위 요청에 대한 실제 응답

```json
{
  "analysisRequestId": "b1287837-cc79-5cf9-86d1-10c6b81f8a92",
  "status": "COMPLETED",
  "results": [
    {
      "resultType": "TONE_CORRECTION",
      "visibilityType": "INDIVIDUAL",
      "targetParticipant": "USER_A",
      "contentType": "TEXT",
      "triggerMessageIds": [105],
      "resultData": {
        "situationDiagnosis": "지금 표현이 상대에게 공격적으로 들릴 수 있어요",
        "guideMessage": "대신 이렇게 상대방에게 말해보세요.",
        "alternativeSentence": "오늘 못 온다고 하니까 좀 서운했어, 다음엔 미리 말해줘",
        "correctionReason": "'맨날'과 마침표가 평소와 달라서 더 단호하게 들려요"
      }
    }
  ],
  "emotionAnalyses": [
    {
      "subjectParticipant": "USER_A",
      "viewerParticipant": "USER_B",
      "emotionType": "ESCALATED",
      "intensityValue": 4.0,
      "shouldShow": true,
      "triggerMessageIds": [101, 103, 105],
      "expiresAt": "2026-08-14T23:51:00+09:00",
      "stateText": "감정이 올라와요"
    },
    {
      "subjectParticipant": "USER_B",
      "viewerParticipant": "USER_A",
      "emotionType": "RESOLVED",
      "intensityValue": 3.0,
      "shouldShow": true,
      "triggerMessageIds": [102, 104],
      "expiresAt": "2026-08-14T23:51:00+09:00",
      "stateText": "다정해 보여요"
    }
  ]
}
```

> 같은 요청이라도 **LLM 이 만드는 문구는 매번 조금씩 다릅니다.** 구조·필드·enum 값은 고정입니다.

### ② 개입 없음 — **`results` 가 빈 배열인데 `COMPLETED`**

가장 흔한 응답이고, 여기만 따로 확인 부탁드립니다.

```json
{
  "analysisRequestId": "60b96a75-5e80-50e9-a2da-5bab5c173da5",
  "status": "COMPLETED",
  "results": [],
  "emotionAnalyses": [
    { "subjectParticipant": "USER_A", "viewerParticipant": "USER_B", "emotionType": "STABLE", "intensityValue": 0.0, "shouldShow": true, "triggerMessageIds": [105, 107], "expiresAt": "2026-08-15T00:40:00+09:00", "stateText": "평온해요" },
    { "subjectParticipant": "USER_B", "viewerParticipant": "USER_A", "emotionType": "STABLE", "intensityValue": 0.0, "shouldShow": true, "triggerMessageIds": [106, 108], "expiresAt": "2026-08-15T00:40:00+09:00", "stateText": "평온해요" }
  ]
}
```

**위젯이 두 줄이고 응답의 배열도 두 개입니다.** `results`(②번 줄, 3종 개입)는 비어도
정상이고, `emotionAnalyses`(①번 줄, 상시)는 거의 항상 찹니다. 규격서 12장이
`COMPLETED` 를 "기능 결과 **또는 감정 분석 결과**가 존재"로 정의하고 있어서
**`SKIPPED` 는 사실상 나오지 않습니다.**

### ③ 데이트 코스 (`resultData` 만)

```json
{
  "guideMessage": "카카포가 추천하는 장소와 데이트 코스를 가져왔어요.",
  "courseName": "성수 브런치 산책 코스",
  "courseSummary": "한식으로 든든히 먹고 서울숲 산책 후 카페에서 휴식",
  "mainPlace": {
    "name": "할머니의레시피",
    "category": "RESTAURANT",
    "summary": "성수동 골목의 한식당",
    "externalUrl": "http://place.map.kakao.com/27373628"
  },
  "coursePlaces": [
    { "order": 1, "name": "할머니의레시피", "category": "RESTAURANT", "summary": "성수동 골목의 한식당", "externalUrl": "http://place.map.kakao.com/27373628" },
    { "order": 2, "name": "서울숲", "category": "ATTRACTION", "summary": "도심 속 자연을 느낄 수 있는 공원", "externalUrl": "http://place.map.kakao.com/11331488" },
    { "order": 3, "name": "슈퍼말차 성수", "category": "CAFE", "summary": "말차 전문 카페에서 여유 즐기기", "externalUrl": "http://place.map.kakao.com/2077832392" }
  ],
  "recommendationReason": "성수 가보고 싶다고 하신 얘기 반영했어요"
}
```

`visibilityType` 은 `COUPLE`, `contentType` 은 `MIXED` 입니다.
`category` 는 8개 중 하나입니다 — `RESTAURANT` `CAFE` `CULTURE` `ATTRACTION` `LODGING`
`SHOP` `ACTIVITY` `ETC` (`contract-review.md` 3번에서 합의).

### ④ 유튜브 (`resultData` 만)

```json
{
  "guideMessage": "현재 겪고 있는 상황과 비슷한 내용을 다룬 영상이에요.",
  "videoId": "h4x4fXlIau0",
  "title": "평~생 써먹는 진심으로 사과하는 법",
  "videoUrl": "https://www.youtube.com/watch?v=h4x4fXlIau0",
  "thumbnailUrl": "https://i.ytimg.com/vi/h4x4fXlIau0/hqdefault.jpg",
  "channelName": "유어셀린 YourCeline",
  "recommendationReason": "사과하고 싶은데 어떻게 말할지 막막할 때, 구체적 사과법을 알려줘요"
}
```

`videoSummary` 는 선택 필드라 값이 없으면 **필드 자체가 빠집니다** (규격서 14장).

---

## 5. 오류 응답 — **HTTP 는 항상 200 입니다**

규격서에 HTTP 상태 코드 규정이 없고, 분석 결과가 `status` 와 `errorCode` 로 이미
표현됩니다. 그래서 잘못된 요청도 200 + `FAILED` 봉투로 돌려드립니다. 백엔드가 HTTP 예외
경로와 분석 실패 경로를 따로 짜지 않아도 되게 하려는 의도입니다.

**4xx 를 원하시면 말씀해 주세요.** 워커 쪽 한 줄이라 언제든 바꿉니다.

```json
{ "analysisRequestId": "bad-1", "status": "FAILED", "results": [], "emotionAnalyses": [],
  "errorCode": "INVALID_REQUEST", "errorMessage": "chatRoomId: Input should be a valid integer" }
```

```json
{ "analysisRequestId": "bad-2", "status": "FAILED", "results": [], "emotionAnalyses": [],
  "errorCode": "INVALID_PARTICIPANT", "errorMessage": "참여자 목록에 없는 발화자: B" }
```

| 상황 | `errorCode` |
| --- | --- |
| 본문 파싱·검증 실패 | `INVALID_REQUEST` |
| `participants` 에 없는 `sender` | `INVALID_PARTICIPANT` |
| 25초 초과 | `ANALYSIS_TIMEOUT` |
| 그 외 예외 | `MODEL_ERROR` |

**워커는 예외를 밖으로 던지지 않습니다.** 500 이 나가면 채팅 서버가 타임아웃까지 붙잡히게
되어서, 파이프라인 전체를 감싸 `FAILED` 로 바꿉니다.

유튜브 검색 결과가 없거나 적절한 후보가 없는 경우는 오류가 아니라 **미발동**입니다
(규격서 13장 그대로). `YOUTUBE_SEARCH_FAILED` 는 반환하지 않습니다.

---

## 6. 처리 시간 (실측)

| 요청 | LLM 호출 | 시간 |
| --- | --- | --- |
| 개입 없음 (고정 3회만) | 3~4 | 3.6s |
| 말투 교정 발동 | 5 | 5.6s |
| 데이트 코스 발동 | 5 | 9.5s |
| 데이트 발동 + 콜드 스타트 | 5 | 11.3s |
| **동시 3건 (데이트+말투+미발동)** | — | **8.0s** (전체 벽시계) |

동시 요청은 스레드풀로 처리해 서로 밀리지 않습니다. 위 8.0초는 세 건을 한꺼번에 던져
전부 받는 데 걸린 시간입니다.

**25초에서 워커가 먼저 끊습니다** (`KAKAPO_DEADLINE`). 30초 제한 안쪽에서 정상 응답이
나가게 하려는 것이고, 여기 걸리면 정상 지연이 아니라 OpenAI 쪽이 멈춘 경우입니다.

---

## 7. 환경변수

| 변수 | 필수 | 없으면 |
| --- | --- | --- |
| `OPENAI_API_KEY` | **필수** | 동작 불가 |
| `KAKAO_REST_API_KEY` | 선택 | 데이트 코스만 미발동 (장소를 지어내지 않습니다) |
| `YOUTUBE_API_KEY` | 선택 | 유튜브 추천만 미발동 (영상을 지어내지 않습니다) |
| `PORT` | 선택 | 8000 |
| `KAKAPO_DEADLINE` | 선택 | 25 (초) |
| `KAKAPO_PERSIST` | 선택 | 1. `0` 이면 추출한 기억을 파일에 쓰지 않습니다 |
| `KAKAPO_YOUTUBE_COOLDOWN_MIN` | 선택 | 30 (분). 최근에 영상을 낸 뒤 이 시간 안에는 유튜브 추천이 발동하지 않습니다 |

---

## 8. 리소스 — 모델을 컨테이너에서 돌리지 않습니다

**추론은 전부 외부 API 호출입니다.** 컨테이너 안에서 모델을 실행하지 않고, 가중치를
받아두지도 않습니다. `torch` · `transformers` 같은 로컬 추론 라이브러리가 아예 설치돼
있지 않습니다.

| 호출처 | 무엇을 |
| --- | --- |
| OpenAI API | 판정·생성 (`gpt-4.1-mini`) + 기억 임베딩 (`text-embedding-3-small`) |
| 카카오 로컬 API | 장소 검색 (데이트 코스) |
| YouTube Data API v3 | 영상·댓글 (유튜브 추천) |

파이썬 의존성은 `langchain` · `langchain-openai` · `pydantic` · `numpy` · `httpx` ·
`fastapi` · `uvicorn` 뿐입니다. `numpy` 는 기억 27건의 코사인 유사도 계산에만 씁니다.

### 실측 (로컬, 요청 1건 처리 전후)

| | |
| --- | --- |
| 기동 직후 RSS | **156 MB** (임베딩 인덱스 워밍 포함) |
| 분석 요청 1건 후 RSS | **163 MB** |
| 유휴 CPU | **0.1%** |
| 스레드 | **3개** (이벤트 루프 + 스레드풀). 요청 처리 중에만 잠깐 늘어납니다 |

**512 MB면 충분하고 1 GB면 여유입니다.** CPU 는 공유 1코어로 충분합니다 — 요청 시간의
대부분이 외부 API 응답을 기다리는 시간이라 CPU 를 거의 쓰지 않습니다.

### 상주하는 것 / 백그라운드 작업

**크론·큐·폴링·워커 프로세스가 없습니다.** 요청이 없으면 아무것도 돌지 않습니다.

| | 무엇 | 언제 |
| --- | --- | --- |
| 메모리 상주 | 기억 27건의 임베딩 벡터 (`InMemoryVectorStore`) | 프로세스 수명 내내. 수백 KB |
| 기동 시 1회 | 그 인덱스를 만드는 데몬 스레드 (약 2.6초, 임베딩 API 1회) | 컨테이너 시작 직후 |
| 요청 중에만 | 독립 단계를 동시에 돌리는 스레드풀 | 응답이 나가면 끝 |

예외가 하나 있습니다 — **25초 타임아웃으로 응답을 먼저 보낸 경우, 넘겨둔 스레드는 끝날
때까지 계속 돕니다.** 파이썬은 스레드를 밖에서 죽일 수 없습니다. 남은 외부 API 호출이
끝나면 스스로 종료되고(길어야 수십 초), 그 사이 응답에는 영향이 없습니다.

디스크는 `data/memories.json` 한 곳에 추출한 기억을 덧씁니다. **컨테이너 로컬이라
재배포하면 초기화**되고, 시드 27건은 이미지에 들어 있습니다. `KAKAPO_PERSIST=0` 으로
끌 수 있습니다.

> ⚠️ **앱 슬립(app sleeping)을 켜면 콜드 스타트마다 임베딩 인덱스를 다시 만듭니다.**
> 첫 요청이 2~3초 느려집니다. 켜셔도 동작에는 문제없지만 감안해 주세요.

---

## 9. 확인 부탁드립니다

| | 항목 | 없으면 |
| --- | --- | --- |
| 🔴 | **`results: []` + `COMPLETED` 를 정상으로 처리** | 가장 흔한 응답입니다. 개입 없는 대화가 오류로 잡힙니다 |
| 🔴 | **`emotionAnalyses` 를 `viewerParticipant` 로 갈라서 전송** | A 화면에 B 것이 뜹니다 (`docs/server-handoff.md`) |
| 🟢 | 호출 주소 `http://ai-worker.railway.internal:8000` 로 붙는지 | — |
| 🟢 | 오류 시 HTTP 200 유지할지, 4xx 로 바꿀지 | — |

`emotionAnalyses` 는 **개인화된 배열**입니다. `subjectParticipant` 는 감정의 주인,
`viewerParticipant` 는 그걸 보는 사람이고 **두 값이 다릅니다** — 각자 상대방의 상태를
봅니다. 방 단위로 브로드캐스트하면 이 설계가 통째로 깨집니다.
