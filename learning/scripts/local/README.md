# learning/scripts/local — 워크스테이션(로컬 GPU) 실행

[`../aws`](../aws) 의 로컬 짝. 트레이너와 엔진 sim-service(`sim/metalab/backends/<engine>/server.py`,
`<engine>` ∈ {genesis, newton})가 **RPC localhost 소켓 위의 두 프로세스**로, 둘 다 엔진의 uv venv 안에서
이 머신 GPU 로 돈다. AWS 노드도 SSM 도 conda 도 없다.

| 스크립트 | 하는 일 |
|---|---|
| `lib.sh` | 공용 설정 — venv 레이아웃(`METALAB_VENV_ROOT`, `engine_venv`), `RL_LOG_ROOT`, task·recipe 탐색. `aws/metalab_train.sh` 도 탐색 헬퍼 때문에 source 한다. |
| `setup_env.sh` | 커밋된 uv 프로젝트(`sim/metalab/setup/<engine>/`)를 venv 로 `uv sync`. 아래 둘이 자동 호출(lock 과 맞으면 no-op). |
| `metalab_train.sh` | 로컬 GPU 학습(`learning.train --trainer rl`). 체크포인트 S3 미러링 기본 ON. |
| `metalab_eval.sh` | 로컬 체크포인트 검증/관전 — SR + 에피소드별 S/F json. |

```
METALAB_VENV_ROOT  $HOME/.metalab/venvs/<repo-dir>   # worktree 별 격리: <root>/genesis, <root>/newton
RL_LOG_ROOT         <repo>/logs/rl                     # 체크포인트: <exp>/<run_name>/model_*.pt
```

## 학습

`--sim {genesis|newton}` 과 `--task <t>` 는 **필수**(fail-loud — 빠지면 목록을 출력한다). task 가
family(`sim/metalab/contract/tasks/<task>/`)면 `--recipe <r>` 도 필요하다. 기본 headless, `--viz gl` 이면
백엔드 GUI + 라이브 웹 대시보드가 뜬다. wandb 는 기본 ON (`--no_wandb` 로 끔).

```bash
T="--task hammer-lift-teacher --recipe privileged"
learning/scripts/local/metalab_train.sh --sim newton  $T --num_envs 4096 --max_iterations 30000
learning/scripts/local/metalab_train.sh --sim genesis $T --num_envs 4 --viz gl    # 라이브 GUI, env 4개
learning/scripts/local/metalab_train.sh --sim newton  $T --device cuda:1          # 엔진을 물리 GPU 1 에
```

런 노브(보상·커리큘럼·PPO·action scale·DR)는 `learning/rl/dexblind/<task>/experiment.py` 에 있다 —
스크립트가 아니라 그 파일을 고친다.

### 체크포인트 → S3 (기본 ON)

워크스테이션 런도 노드 런과 같은 위치·레이아웃으로 올라가므로, 어느 머신이 돌렸는지 몰라도 찾을 수 있다:

```
s3://wirobotics-internal/jkkim/sim_rl/ckpts/<run_name>/model_*.pt
```

```bash
SYNC_INTERVAL=120 ...                    # 미러 주기 (기본 300초)
CKPT_S3_ROOT=s3://.../my/ckpts ...       # 다른 곳에 올리기
KEEP_LOCAL_CKPTS=1 ...                   # 로컬 사본 전부 보존
--no-s3-sync                             # S3 에 안 올림 (로컬 전용, pruning 없음)
```

업로드가 성공하면 로컬 `model_*.pt` 는 **최신 하나만** 남기고 정리된다(최신 것은 eval·리포트 훅이 쓰므로
남긴다). 예전 것은 `aws s3 cp` 로 먼저 받아온다. S3 쪽은 절대 지우지 않는다.

로컬 자격증명(`aws login`)이 필요하다. 없으면 — S3 가 기본값이었을 땐 WARN 후 로컬 전용으로 계속(정리
안 함), 명시적으로 요청했을 땐(`--s3-sync`) hard error.

### 학습 중 녹화 + 리포트 (`--record`, 기본 OFF)

체크포인트마다 트레이너가 직접 짧은 sim-service 를 띄워 롤아웃을 rerun `.rrd` + 스텝별 시리즈로 녹화하고,
둘을 합친 `report.html`(왼쪽 rerun 뷰어 / 오른쪽 시간 동기화 플롯)을 체크포인트 옆에 올린다. W&B 에는
링크만(`val/report`) 기록된다 — **CloudFront** 로 연다.

```
s3://wirobotics-internal/jkkim/sim_rl/ckpts/<run_name>/report_<iter>/{report.html,rollout.rrd,data.json}
```

```bash
RECORD_ENVS=6 ... --record      # 시리즈 + 리포트 탭을 받을 env 수 (기본 4)
RECORD_STEPS=0 ... --record     # 녹화당 정책 스텝 (0 = 에피소드 전체; 기본 600)
```

- S3 목적지가 없으면 리포트는 `<log_dir>/report_<iter>/` 에 남는다.
- **BLOCKING**: 녹화는 학습 스레드에서 돌아 체크포인트마다 루프가 멈추고, 녹화용 env 가 학습 GPU 를 같이
  쓴다. 실패해도 WARN 만 남기고 학습은 계속된다.

## 검증/관전 (`metalab_eval.sh`)

RPC 로 체크포인트를 검증한다 → SR + 에피소드별 S/F json. 기본(headless)은 `.rrd` + `report.html` 을 그
json 옆에 **로컬로** 녹화하고, `--viz` 는 라이브 뷰어를 열며 녹화하지 않는다.

```bash
T="--task hammer-lift-teacher --recipe privileged"
learning/scripts/local/metalab_eval.sh --sim newton $T --viz                 # 라이브 관전 (∞, Ctrl-C)
learning/scripts/local/metalab_eval.sh --sim newton $T                       # headless: SR + .rrd/리포트
RECORD=0 EPISODES=64 learning/scripts/local/metalab_eval.sh --sim newton $T --num_envs 16   # 64 에피소드 SR
CHECKPOINT=logs/rl/.../model_800.pt learning/scripts/local/metalab_eval.sh --sim newton $T
GPU=1 learning/scripts/local/metalab_eval.sh --sim newton $T                 # 학습이 GPU 0 을 쓰는 동안
```

- **체크포인트**: 기본 = `RL_LOG_ROOT` 아래 그 task 의 최신 로컬 `model_*.pt`. S3 전용 체크포인트는 먼저
  `aws s3 cp` 해야 한다(이 도구는 S3 를 받아오지 않는다).
- **`EPISODES`**: `-1`(기본) 무한 관전 · `>0` N 에피소드 후 종료 · `0` 고정 `STEPS`.
- **`RECORD=0`** 녹화 생략, **`RECORD_ENVS`**(3) 탭 받을 env 수, **`GPU=N`** `CUDA_VISIBLE_DEVICES=N`.
- SR 은 모든 경로에서 커리큘럼 **END** 기준으로 판정한다.
