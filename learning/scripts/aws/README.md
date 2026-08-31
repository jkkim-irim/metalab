# learning/scripts/aws — GPU 노드에서 MetaLab 학습

`metalab_train.sh` 는 AWS GPU 노드에서 MetaLab 정책을 학습시킨다. 방식은
[`../local/metalab_train.sh`](../local/metalab_train.sh) 가 워크스테이션에서 하는 것과 같다 — 노드에서
바로 그 스크립트를 실행한다. 노드에는 **SSM** 으로 붙는다: SSH 키도, 인바운드 포트도 없다. 코드는 S3
tarball 로 올리고, 엔진 환경은 노드에서 커밋된 `uv.lock` 으로 `uv` 가 만든다(`../local/setup_env.sh` —
핀 고정된 sim 소스 clone + `uv sync`, conda 없음). 그래서 한 번 뜨고 나면 네 워크스테이션에 아무것도
의존하지 않는다.

## 노드 수명주기는 스크립트가 아니라 사용자의 몫

`--node <instance-id>` 는 **필수**다 — 이 스크립트는 인스턴스를 만들지도, 없애지도 않는다. GPU 를 띄우는
것은 이 스크립트가 할 수 있는 가장 비싼 일이라 절대 암묵적으로 일어나지 않는다: `--node` 가 없으면 지금
보이는 running GPU 인스턴스 목록을 출력하고 종료한다. 노드 생성은 이 리포 밖에 있는 gpu-launchers 킷으로
한다.

사전 준비(1회):
- **gpu-launchers** 그룹에 속한 `AWS_PROFILE` (미지정 시 `default` 프로파일)
- 노드가 이미 본인 전용 롤 `node-<your-IAM-username>`(SSM + 공용 prefix 읽기 + 본인 S3 prefix 읽기/쓰기)을
  달고 있고 `Owner=<your IAM username>` 로 태깅되어 있을 것 — 정책이 노드 제어를 본인 Owner 태그에
  묶으므로, 남의 태그가 붙은 노드에서는 학습을 돌릴 수 없다. 롤은 노드 **생성 시점**에 정해지므로 이
  스크립트가 붙여줄 수 없다.

## 실행

```bash
AWS_PROFILE=default bash learning/scripts/aws/metalab_train.sh \
    --node <id> --sim genesis --task hammer-lift-teacher --recipe privileged

… --node <id> --num-envs 8192 --max-iterations 5000 --record   # 실제 런 (detached; 노드는 계속 살아 있음)
… --node <id> --num-envs 512  --max-iterations 3    --smoke    # 짧은 테스트 (완료까지 대기)
```

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--node <id>` | — (**필수**) | 학습을 돌릴, 이미 running 인 노드 |
| `--sim {genesis\|newton}` | `genesis` | 엔진 스포크 |
| `--task <t>` / `--recipe <r>` | — (**task 필수**) | 계약서. task family 는 recipe 가 필요하며, 노드를 건드리기 전에 로컬에서 먼저 검증한다 |
| `--num-envs` / `--max-iterations` | `4096` / 스크립트 기본값 | 병렬 env 수 · 이터레이션 상한 |
| `--run-label <x>` | task 이름 | 런 이름의 label 구간 |
| `--record` | off | 체크포인트별 롤아웃 녹화 + 리포트 |
| `--smoke` | off | detach 하지 않고 런이 끝날 때까지 대기 |

환경변수 오버라이드: `AWS_PROFILE`, `AWS_REGION`(`us-east-1`), `AWS_USER`(본인 S3 prefix — 미지정 시 IAM
신원에서 해석), `METALAB_DEPLOY_S3`(repo tarball 스테이징 prefix), `RUN_NAME`, `SSM_TIMEOUT`.

## 결과물

- **체크포인트**는 런 도중 S3 로 미러링된다(`…/sim_rl/ckpts/<run_name>/model_*.pt`) — 노드를 terminate 해도
  남는다.
- **런 이름**은 로컬에서 만들어 노드로 export 한다 —
  `{yymmdd-HHMM}_{envs}_{engine}_{recipe}_{label}_{sha}_aws` — 그래서 실제 repo SHA 를 달고 있어 정확한
  코드로 추적된다.
- **wandb**: 로컬 `~/.netrc` 의 키를 SSM 으로 노드에 심는다(출력하지 않음). 키가 없으면 런은 offline 으로
  기록되고 스크립트가 그렇게 알려준다.
- 스크립트는 wandb URL 을 출력하고 노드 로그를 tail 한다.

**다 쓰면 노드를 stop 한다** — stop 전까지 계속 과금되고, 이 스크립트는 대신 꺼주지 않는다.
