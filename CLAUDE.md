# CLAUDE.md — metalab

스택 통합 리포: 엔진 무관 sim(`sim/metalab`) + 학습/평가 파이프라인(`learning/`).
MetaLab sim 은 **계약서 하나가 단일 소스**이고, 엔진별 스포크(`backends/newton`, `backends/genesis`)가 그것을 읽는다.

## Conventions
- **커밋/PR 에 AI 저작 표기 금지** (`Co-Authored-By`, generated-by 푸터).
- **원격 git 작업은 그때그때 승인받는다** — push(force 포함)·히스토리 재작성·PR 머지/클로즈. 태스크가 아니라 push 단위다("PR 업데이트해줘"는 push 승인이 아니다). 커밋은 로컬에만, push 는 지시받았을 때만.
- **vendored/upstream 코드는 고치지 않는다** — 우리 코드에서 감싸거나 오버라이드한다.
- **조용히 실패하지 않는다** — 에러를 삼키는 try/except·fallback 금지. 불변식은 `assert`, 에러는 전파. 잡는 것은 시스템 경계에서만.
- **import 는 파일 최상단에**, 함수 안에 넣지 않는다.
- **`README.md` 는 사용자 매뉴얼** — 요청 없이 건드리지 않는다.

## Sim (`sim/metalab`)
- **단위는 SI, 각도는 rad** — 강성·게인 kp/kv/K_q 도 N·m/rad. 사람에게 보여지는 각도만 deg.
- **주석은 달지 않는다** — 주석은 리포에서 유일하게 검증 장치가 없어 틀려도 조용히 거짓이 된다. 필요한 주석은 사용자가 직접 요청하는 것만.
- **MDP(reward·obs·events·terminate)의 수식과 판정은 term 함수 안에만** 쓴다(여러 term 이 공유하는 계산은 `api/` 로 올린다).
- **엔진별 진입점은 스포크마다 한 곳**: `backends/<engine>/server.py` + `backend.py`.

## Training
- **실행은 `learning/scripts/local/` 의 스크립트로** (`metalab_train.sh`, `metalab_eval.sh`). 스크립트가 안 되면 스크립트를 고쳐 커밋한다 — 일회성 명령으로 우회하지 않는다.
- **긴 런 전에 스모크 테스트** — val/ckpt/sim-eval 한 사이클이 끝까지 도는지 먼저 확인한다.

## Testing
- **재구현이 아니라 실제 함수를 테스트한다** — 로직을 테스트 안에 옮겨 적지 않는다.

## 지식 축적 (메모리)
- 세션에서 **측정한** 엔진 지식과 설계 의도(왜 이 규약인지·어기면 나는 증상)를 메모리에 남긴다.
- 리포 코드·문서가 이미 기록하는 것은 메모리에 쓰지 않는다 — 메모리는 외부 엔진 소스처럼 세션이 끝나면 사라지는 것만 담는다.
- 태스크 하나에만 걸리는 지엽적 사실 대신, **다른 로봇·다른 엔진으로 옮겨가도 쓸모 있는 것**을 남긴다.
- 엔진마다 얇은 노트를 새로 만들지 말고 주제별 누적 노트에 append 한다. 저장 전 인덱스의 해당 섹션을 훑어 기존 파일 갱신인지 먼저 본다.
