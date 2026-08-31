# learning/scripts/local — 플래그·환경변수 레퍼런스

트레이너(client)와 엔진 서버가 같은 uv venv 에서 두 프로세스로 뜨고 localhost RPC 로 통신한다.
`lib.sh` = 공용 설정·task 탐색, `setup_env.sh` = 엔진 venv `uv sync`(둘이 자동 호출).

```
METALAB_VENV_ROOT  $HOME/.metalab/venvs/<repo-dir>   # worktree 별 엔진 venv
RL_LOG_ROOT         <repo>/logs/rl                     # 체크포인트: <exp>/<run_name>/model_*.pt
```

## metalab_train.sh

| 플래그 | 뜻 |
|---|---|
| `--sim {genesis\|newton}` · `--task` | 필수. family 태스크는 `--recipe` 도 필수 |
| `--num_envs` · `--max_iterations` · `--seed` | 트레이너로 전달 |
| `--device cuda:N` | 런 전체를 물리 GPU N 에 고정 |
| `--viz gl` (newton 은 `rtx` 도) | 엔진 GUI + 라이브 웹 대시보드. 기본 headless |
| `--no_wandb` | wandb 끔 (기본 ON, 미로그인이면 실행 전에 실패) |
| `--record` | 체크포인트마다 rerun `.rrd` + `report.html` 을 체크포인트 옆 `report_<iter>/` 에 녹화(BLOCKING). `RECORD_ENVS`(4)=리포트 탭 env 수, `RECORD_STEPS`(600, 0=에피소드 전체) |

그 외 모르는 플래그는 트레이너로 그대로 전달. 리포트의 3D 패널은 http(s) 로 서빙해야 뜬다
(`file://` 는 플롯만; `rollout.rrd` 를 로컬 rerun 뷰어로 열어도 된다).

## metalab_eval.sh

SR + 에피소드별 S/F json. 기본은 headless 녹화(`.rrd`+리포트, 로컬), `--viz` 는 라이브 관전(녹화 없음).
SR 판정은 항상 커리큘럼 **END** 조건.

| 이름 | 뜻 |
|---|---|
| `--sim` · `--task` (+`--recipe`) | 필수 |
| `--checkpoint PATH` | 미지정 = 그 task 의 최신 로컬 `model_*.pt` |
| `--num_envs` | 관전은 보통 1 |
| `EPISODES` (env) | `-1`(기본) 무한 관전 · `>0` N 에피소드 후 SR · `0` 고정 `STEPS` |
| `RECORD=0` · `RECORD_ENVS`(3) | 녹화 끔 / 리포트 탭 env 수 |
| `GPU=N` (env) | 물리 GPU 고정 — 학습이 GPU 0 을 쓰는 동안 평가용 |
| `SEED` (env, 42) | 재현 시드 |
