# MetaLab

엔진 무관 로봇 학습 시뮬레이터. 계약서 하나로 환경을 정의하고 **Newton · Genesis** 두 물리 엔진에서
같은 환경을 학습·평가한다. 로컬 GPU 전용.

```
sim/metalab/       계약서(contract) · 엔진 스포크(backends) · 런타임 · 에셋
learning/          RL 트레이너 · 정책 · 평가 · 실행 스크립트
```

- 설계 개요 → [`sim/metalab/docs/00_project_overview.md`](sim/metalab/docs/00_project_overview.md)
- 플래그·환경변수 전체 → [`learning/scripts/local/README.md`](learning/scripts/local/README.md)
- 셋업 상세·재현성 → [`sim/metalab/README.md`](sim/metalab/README.md)

## 설치

요구사항: Ubuntu 22.04/24.04 · NVIDIA GPU + 드라이버(≥550) · git · build-essential · 디스크 ~30 GB

```bash
git clone <this repo> ~/metalab_ws/metalab && cd ~/metalab_ws/metalab
sim/metalab/setup.sh          # 엔진 소스 clone(형제 디렉터리, 커밋 핀 고정) + 엔진별 uv 환경
```

## 실행

```bash
sim/metalab/launchpad.sh      # 웹 콘솔: 엔진·태스크·학습/평가·노브 → Launch
```

CLI:

```bash
# 스모크 학습 — 4 envs + GUI
learning/scripts/local/metalab_train.sh --sim newton \
    --task hammer-lift-teacher --recipe privileged --num_envs 4 --viz gl

# 체크포인트 관전
learning/scripts/local/metalab_eval.sh --sim newton \
    --task hammer-lift-teacher --recipe privileged --viz --num_envs 1

# 본학습
nohup learning/scripts/local/metalab_train.sh --sim newton \
    --task hammer-lift-teacher --recipe privileged --num_envs 8192 --max_iterations 5000 \
    > train.log 2>&1 &
```

- `--sim` 과 `--task` 는 필수. task 가 family 면 `--recipe` 도 필수(빠뜨리면 목록 출력).
- wandb 미로그인이면 `--no_wandb`.
- GPU 지정: `--device cuda:N`(train) / `GPU=N`(eval).
- 평가는 항상 커리큘럼 **end** 조건(최종 난이도)으로 판정한다.

## 결과물

| | 위치 |
|---|---|
| 체크포인트 | `logs/rl/<experiment>/<run_name>/model_<iter>.pt` (run 이름에 git SHA) |
| 학습 커브 | wandb — `val/SR` 이 고정 tolerance 성공률 |
| 리포트 (`--record`) | 체크포인트 옆 `report_<iter>/report.html` (rerun `.rrd` + 플롯) |

## 노브

| 바꿀 것 | 파일 |
|---|---|
| 물리·접촉·보상·관측·종료·커리큘럼·DR | `sim/metalab/contract/tasks/rl/<task>/` (`_base.py` + 레시피) |
| PPO 하이퍼·네트워크·obs 라우팅 | `learning/rl/dexblind/<task>/experiment.py` |

스크립트는 편집하지 않는다 — 실험 노브는 위 두 곳에만 있다.

## 확장

- **새 태스크**: `sim/metalab/contract/tasks/rl/<name>/` 에 `_base.py` + `<name>_<recipe>.py` 를 두면
  자동 인식. 물체 에셋 88종은 `object_mjcf("<이름>")` 으로 사용.
- **새 엔진**: `sim/metalab/backends/<engine>/{parser,backend,server}` 스포크만 붙이면 같은 계약서가 돈다.