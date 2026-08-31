# CLAUDE.md — metalab

스택 통합 리포: 엔진 무관 sim(`sim/metalab`) + 학습/평가 파이프라인(`learning/`).

## Conventions
- **커밋/PR 에 AI 저작 표기 금지** (`Co-Authored-By`, generated-by 푸터).
- **원격 git 작업은 그때그때 승인받는다** — push(force 포함)·히스토리 재작성·PR 머지/클로즈. 태스크가 아니라 push 단위다("PR 업데이트해줘"는 push 승인이 아니다). 커밋은 로컬에만, push 는 지시받았을 때만. AWS admin 자격증명도 마찬가지.
- **vendored/upstream 코드는 고치지 않는다** — 우리 코드에서 감싸거나 오버라이드한다.
- **조용히 실패하지 않는다** — 에러를 삼키는 try/except·fallback 금지. 불변식은 `assert`, 에러는 전파. 잡는 것은 시스템 경계에서만.
- **import 는 파일 최상단에**, 함수 안에 넣지 않는다.
- **`README.md` 는 사용자 매뉴얼** — 요청 없이 건드리지 않는다.

## Training
- **실행은 `learning/scripts/` 의 스크립트로** (`local/metalab_train.sh`, `local/metalab_eval.sh`, `aws/metalab_train.sh`). 스크립트가 안 되면 스크립트를 고쳐 커밋한다 — 일회성 `aws`/`ssm`/`scp` 로 우회하지 않는다.
- **긴 런 전에 스모크 테스트** — val/ckpt/sim-eval 한 사이클이 끝까지 도는지 먼저 확인한다.
- **wandb 런 이름 `[name]-[datetime]-[SHA]`** (UTC + short git SHA). 스크래치/스윕은 공용 dev 프로젝트에.

## Testing
- **재구현이 아니라 실제 함수를 테스트한다** — 로직을 테스트 안에 옮겨 적지 않는다.
