# AI 워커 연동 공통 규격 초안 v1 (백엔드 제공)

채팅 서버 담당자가 작성한 연동 규격이다. **이 문서는 팀 합의 사항이라 워커가 임의로 바꾸지
않는다.** 워커 쪽 답변과 확인 요청은 `docs/server-handoff.md` 에 따로 정리했다.

기존 `docs/contract-v2.md` 는 방 구조가 커플방 1개로 바뀌면서 폐기됐고, 이 문서가 그 자리를
대체한다. 스키마는 `worker/models.py` 에 그대로 옮겨져 있으며,
`AnalysisResponse.to_json_dict()` 가 여기 정의된 JSON 을 그대로 만든다.

---

## 1. 목적

백엔드가 채팅 데이터를 AI 워커에 전달하고, AI 워커가 대화 맥락을 분석하여 필요한 기능과
노출 대상자를 판단한 뒤 약속된 형식으로 결과를 반환하기 위한 규격.

## 2. 역할 구분

**백엔드** — `analysisRequestId` 생성 / 참여자·메시지 전달 / AI 응답 형식 검증 /
AI 결과 ID 자동 생성 / 결과 및 감정 분석 결과 저장 / 대상 사용자의 채팅방 참여 여부 검증 /
사용자별 접근 권한 처리 / 프론트엔드에 결과 전달 / 타임아웃·오류·중복 응답 처리

**AI 워커** — 대화 맥락 분석 / 기능 발동 여부 판단 / 발동할 기능 종류 판단 /
결과를 보여줄 대상자 판단 / 결과 재노출 기준 메시지 판단 / 기능별 상세 결과 생성 /
감정 분석 결과 생성 / 약속된 JSON 형식으로 결과 반환

## 3. 전체 연동 흐름

```
백엔드의 메시지 저장 → 설정된 메시지 개수 누적 → AI 워커에 공통 분석 요청
→ AI 워커의 대화 맥락 분석 → 기능·대상자·기준 메시지 판단 → 기능별 결과 반환
→ 백엔드의 검증·저장 → 프론트엔드 노출
```

백엔드는 특정 기능을 지정하지 않고 대화 분석을 포괄적으로 요청한다.
한 번의 분석 요청에서 반환 가능한 결과: 기능 미발동 / 하나의 기능 발동 /
**여러 기능 동시 발동** / 감정 분석 결과 발생.

## 4. 통신 방식

```
POST /internal/v1/chat-analyses
Content-Type: application/json
```

REST API 기반 **동기** 요청·응답. AI 처리 시간이 길어 비동기 콜백이 필요할 경우 별도 협의.

---

## 5. 공통 분석 요청

```json
{
  "analysisRequestId": "c783ec30-8a45-4c65-a542-f865c71a2f01",
  "chatRoomId": 1,
  "participants": [
    { "participantKey": "USER_A" },
    { "participantKey": "USER_B" }
  ],
  "messages": [
    {
      "messageId": 101,
      "sender": "USER_A",
      "content": "왜 연락을 안 했어?",
      "sentAt": "2026-08-15T20:30:00"
    }
  ]
}
```

| 필드 | 타입 | 필수 | 설명 |
| --- | --- | --- | --- |
| `analysisRequestId` | UUID | 필수 | 백엔드가 생성한 분석 요청 식별값 |
| `chatRoomId` | Long | 필수 | 채팅방 식별값 |
| `participants[].participantKey` | String | 필수 | `USER_A` 또는 `USER_B` |
| `messages[].messageId` | Long | 필수 | 메시지 식별값 |
| `messages[].sender` | String | 필수 | 메시지 발화자 |
| `messages[].content` | String | 필수 | 메시지 내용 |
| `messages[].sentAt` | DateTime | 필수 | 메시지 전송 시각 |

메시지는 `sentAt` 과 `messageId` 기준 과거 → 최신 순서로 전달.

## 6. 공통 분석 응답

```json
{
  "analysisRequestId": "c783ec30-8a45-4c65-a542-f865c71a2f01",
  "status": "COMPLETED",
  "results": [ /* AI 결과 */ ],
  "emotionAnalyses": []
}
```

| 필드 | 타입 | 필수 | 설명 |
| --- | --- | --- | --- |
| `analysisRequestId` | UUID | 필수 | 요청으로 전달받은 값을 그대로 반환 |
| `status` | String | 필수 | 분석 처리 결과 |
| `results` | Array | 필수 | 발동된 AI 기능 결과 |
| `emotionAnalyses` | Array | 필수 | 사용자별 감정 분석 결과 |

## 7. 공통 AI 결과 구조

| 필드 | 타입 | 필수 | 설명 |
| --- | --- | --- | --- |
| `resultType` | String | 필수 | `TONE_CORRECTION` / `DATE_RECOMMENDATION` / `YOUTUBE_RECOMMENDATION` |
| `visibilityType` | String | 필수 | `INDIVIDUAL` / `COUPLE` |
| `targetParticipant` | String | 조건부 | `INDIVIDUAL` 이면 필수, `COUPLE` 이면 생략 |
| `contentType` | String | 필수 | `TEXT` / `LINK` / `MIXED` |
| `triggerMessageIds` | Long Array | 필수 | 결과를 다시 노출할 기준 메시지 목록 |
| `resultData` | Object | 필수 | 기능별 상세 결과 |

| 기능 | contentType |
| --- | --- |
| 말투 교정 | `TEXT` |
| 데이트 추천 | `MIXED` |
| 유튜브 추천 | `MIXED` |

`triggerMessageIds` 는 요청으로 전달받은 메시지 ID 만 사용하고, 중복을 제외한다.

## 8. 말투 교정 결과

`visibilityType = INDIVIDUAL` / `targetParticipant` = 공격적이거나 오해를 유발할 표현을
사용한 사용자 / `contentType = TEXT`

```json
{
  "situationDiagnosis": "현재 표현이 상대방에게 공격적으로 들릴 수 있어요.",
  "guideMessage": "대신 이렇게 상대방에게 말해보세요.",
  "alternativeSentence": "오늘 못 온다고 하니까 조금 서운했어.",
  "correctionReason": "'맨날'이라는 표현은 지금 한 번의 일이 아니라 그동안의 모든 일을 말하는 것처럼 들릴 수 있어요."
}
```

| 필드 | 필수 | 설명 |
| --- | --- | --- |
| `situationDiagnosis` | 필수 | 현재 표현에 대한 짧은 상황 진단 |
| `guideMessage` | 필수 | 사용자 안내 문구 |
| `alternativeSentence` | 필수 | 나 전달법 기반 대체 문장 |
| `correctionReason` | 필수 | 기존 표현이 다르게 읽힐 수 있는 이유 |

## 9. 데이트 코스 추천 결과

노출 범위는 AI 워커가 대화 맥락에 따라 판단.
`INDIVIDUAL` → 한 명이 상대방에게 데이트를 제안할 수 있도록 제공.
`COUPLE` → 두 사람이 함께 데이트 계획을 논의하는 상황에 제공.

```json
{
  "guideMessage": "카카포가 추천하는 장소와 데이트 코스를 가져왔어요.",
  "courseName": "성수 주민 픽 브런치 데이트 코스",
  "courseSummary": "브런치 식사 후 소품샵과 서울숲을 둘러보는 코스",
  "mainPlace": {
    "name": "성수다락",
    "category": "RESTAURANT",
    "summary": "골목길 건물 2~3층에 위치한 식당",
    "externalUrl": "https://..."
  },
  "coursePlaces": [
    { "order": 1, "name": "성수다락", "category": "RESTAURANT", "summary": "…", "externalUrl": "https://..." },
    { "order": 2, "name": "헤븐센스", "category": "SHOP", "summary": "…", "externalUrl": "https://..." }
  ],
  "recommendationReason": "3주 전에 상대방이 성수에 가보고 싶다고 한 내용과 내일 낮에 만나기로 한 대화를 반영했어요."
}
```

| 필드 | 필수 | 설명 |
| --- | --- | --- |
| `guideMessage` | 필수 | 사용자 안내 문구 |
| `courseName` | 필수 | 데이트 코스명 |
| `courseSummary` | 필수 | 한 줄 요약 |
| `mainPlace` | 필수 | 핵심 장소 |
| `coursePlaces` | 필수 | 순서가 있는 코스 장소 목록 |
| `recommendationReason` | 필수 | 대화 맥락 기반 추천 이유 |

**장소 필드** — `order`(코스 장소만 필수) / `name` / `category` / `summary` / `externalUrl`.
MVP 에서는 장소 이미지 필드 제외.

## 10. 유튜브 영상 추천 결과

노출 범위는 AI 워커가 대화 맥락에 따라 판단.

```json
{
  "guideMessage": "현재 겪고 있는 상황과 비슷한 내용을 다룬 영상이에요.",
  "videoId": "abc123",
  "title": "영상 제목",
  "videoUrl": "https://youtube.com/watch?v=abc123",
  "thumbnailUrl": "https://...",
  "channelName": "채널명",
  "recommendationReason": "현재 겪고 있는 연락 문제와 비슷한 상황을 다룬 영상이에요.",
  "videoSummary": "PM이 제공한 영상 요약"
}
```

`videoSummary` 는 **선택** 필드다. AI 가 새로 생성하거나 수정하지 않으며, PM 이 제공한
데이터셋에 존재하는 값만 반환한다. 데이터가 없는 경우 필드를 생략한다.

**예외 처리** — 검색 결과가 없는 경우 미발동 / 신뢰할 수 있는 후보가 없는 경우 미발동 /
동일 영상을 최근에 추천한 경우 다른 후보 탐색 / 임의 영상 추천 금지.

## 11. 감정 분석 결과

```json
{
  "subjectParticipant": "USER_A",
  "viewerParticipant": "USER_B",
  "emotionType": "ANGER",
  "intensityValue": 4.0,
  "shouldShow": true,
  "triggerMessageIds": [101],
  "expiresAt": null
}
```

감정 종류·강도 범위·표시 기준·유지시간은 추후 확정.

> ⏸️ **워커는 지금 이 배열을 항상 비워서 보낸다.** 별도 기능으로 명세가 아직 나오지 않았다.

## 12. 상태값

| 값 | 의미 |
| --- | --- |
| `COMPLETED` | 하나 이상의 기능 결과 또는 감정 분석 결과가 존재 |
| `SKIPPED` | 분석은 정상 완료됐으나 발동할 기능과 감정 결과가 없음 |
| `FAILED` | 요청 전체를 정상 처리하지 못함 (`errorCode` / `errorMessage` 동반) |

## 13. 오류 코드

`INVALID_REQUEST` / `INVALID_PARTICIPANT` / `MODEL_ERROR` / `ANALYSIS_TIMEOUT` /
`YOUTUBE_SEARCH_FAILED` / `EXTERNAL_API_ERROR` / `INTERNAL_ERROR`

YouTube 검색 결과가 없거나 적절한 후보가 없는 경우는 오류가 아니라 **기능 미발동** 처리.

## 14. 데이터 작성 규칙

- JSON 필드명은 `camelCase`
- 날짜·시간은 ISO 8601
- 필수 필드는 `null` 반환 금지
- 선택 필드에 값이 없으면 필드 생략
- 빈 목록은 `null` 대신 `[]`
- 요청에 없는 메시지 ID 를 `triggerMessageIds` 로 반환 금지
- `INDIVIDUAL` 이면 `targetParticipant` 필수, `COUPLE` 이면 생략
- 한 요청에서 동일 대상자에게 동일 `resultType` 결과는 최대 1개

## 15. AI 담당자 확인 요청 사항

→ **`docs/server-handoff.md` 에 항목별로 답변했다.**
