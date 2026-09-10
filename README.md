# MetaLab

엔진 무관 로봇 강화학습 시뮬레이터. 태스크를 **계약서 하나**(Python 파일)로 정의하면 **Newton · Genesis**
두 물리 엔진에서 같은 환경이 학습·평가된다. 로컬 GPU 한 대로 돈다.

```
sim/metalab/       계약서(contract) · 엔진 스포크(backends) · 런타임 · 에셋
learning/          RL 트레이너 · 평가 · 실행 스크립트
```

## 1. 요구사항

| 항목 | 값 |
|---|---|
| OS | Ubuntu 22.04 / 24.04 |
| GPU | NVIDIA + 드라이버 ≥ 550 (`nvidia-smi` 로 확인) |
| 패키지 | `git`, `build-essential` (`sudo apt install git build-essential`) |
| 디스크 | 약 30 GB (엔진 소스 + 두 개의 Python 환경) |

Python·conda 를 따로 설치하지 않는다. 셋업 스크립트가 `uv` 를 설치하고 엔진별 환경을 만든다.

## 2. 설치

```bash
mkdir -p ~/metalab_ws && cd ~/metalab_ws
git clone git@github.com:jkkim-irim/metalab.git
cd metalab
sim/metalab/setup.sh
```

`setup.sh` 가 하는 일:

1. 핀 고정된 엔진 소스(`newton`, `genesis-world`)를 **리포의 형제 디렉터리**에 clone 한다. 커밋은
   `sim/metalab/sim_versions.env` 가 정한다.
2. 엔진별 uv 환경을 `~/.metalab/venvs/metalab/<engine>` 에 만든다. 의존성은 커밋된 `uv.lock` 이 고정한다.

```
metalab_ws/
├── metalab/          ← 이 리포
├── newton/           ← setup.sh 가 clone
└── genesis-world/    ← setup.sh 가 clone
```

몇 번 다시 실행해도 안전하다. 실패하면 메시지에 빠진 것(컴파일러, 드라이버 등)이 그대로 나온다.

wandb 로 학습 곡선을 보려면 `wandb login`. 안 쓰면 실행 시 `--no_wandb`.

## 3. 첫 실행

가장 쉬운 길은 웹 콘솔이다.

```bash
sim/metalab/launchpad.sh
```

브라우저가 열리면 **Backends(엔진) → Mode → Task → Recipe** 를 고르고 Launch. 학습·평가·standalone 시뮬레이션·
엔진 parity 비교가 모두 여기서 된다. 처음 실행 시 GNOME 앱 목록에 아이콘도 등록된다.

CLI 로 같은 것을 하려면:

```bash
# 스모크: 작은 env 로 몇 iteration 만 (엔진·환경이 제대로 붙는지 확인)
learning/scripts/local/metalab_train.sh --sim newton --task franka_reach --recipe position \
    --num_envs 64 --max_iterations 3 --no_wandb

# GUI 로 보면서 학습
learning/scripts/local/metalab_train.sh --sim newton --task franka_reach --recipe position \
    --num_envs 4 --viz gl

# 본학습
nohup learning/scripts/local/metalab_train.sh --sim newton --task franka_reach --recipe position \
    --num_envs 4096 --max_iterations 5000 > train.log 2>&1 &

# 체크포인트 관전 (최신 체크포인트 자동 선택)
learning/scripts/local/metalab_eval.sh --sim newton --task franka_reach --recipe position --viz --num_envs 1
```

- `--sim {newton|genesis}` 와 `--task` 는 필수. 태스크가 family 면 `--recipe` 도 필수 (빠뜨리면 목록이 출력된다).
- 태스크 이름은 `-` 와 `_` 둘 다 받는다 (`franka-reach` = `franka_reach`).
- 다른 GPU 사용: 학습 `--device cuda:N`, 평가 `GPU=N`.
- 플래그 전체 → [`learning/scripts/local/README.md`](learning/scripts/local/README.md)

## 4. 결과물

| | 위치 |
|---|---|
| 체크포인트 | `_logs/rl/<experiment>/<run_name>/model_<iter>.pt` (run 이름에 git SHA) |
| 학습 곡선 | wandb. `val/SR` 이 계약서 GATE 기준 성공률 |
| 평가 리포트 | 체크포인트 옆 `report_<iter>/report.html` (rerun `.rrd` + 플롯) |
| parity 비교 | `_logs/parity/<task>/` (`.npz`, `.md`, `.png`) |

## 5. 어디를 고치나

| 바꿀 것 | 파일 |
|---|---|
| 물리 · 접촉 · 액션 · 보상 · 관측 · 종료 · DR · 커리큘럼 | `sim/metalab/contract/tasks/rl/<group>/<task>/` (`_base.py` + `<recipe>.py`) |
| PPO 하이퍼파라미터 · 네트워크 · obs 라우팅 | `learning/rl/<experiment>/<task>/experiment.py` |
| 로봇 정의 (관절 그룹 · 게인 · 프레임) | `sim/metalab/contract/robot/<family>/<robot>.yaml` |

실행 스크립트는 편집하지 않는다. 실험 노브는 위 세 곳에만 있다.

- **새 태스크**: `sim/metalab/contract/tasks/rl/<group>/<name>/` 에 `_base.py` + `<recipe>.py` 를 두고
  `learning/rl/<experiment>/<name>/experiment.py` 를 만들면 자동 인식된다. 물체 에셋은 `object_mjcf("<이름>")`.
- **새 로봇**: `contract/robot/<family>/<name>.yaml` + MJCF 를 `sim/metalab/assets/robots/` 에 둔다.
- **새 엔진**: `sim/metalab/backends/<engine>/{parser,backend,server}.py` 스포크만 붙이면 같은 계약서가 돈다.

## 6. 더 읽을 것

- 셋업 내부 · 재현성 · 엔진 버전 올리기 → [`sim/metalab/README.md`](sim/metalab/README.md)
- 스크립트 플래그 · 환경변수 → [`learning/scripts/local/README.md`](learning/scripts/local/README.md)
- 계약서 작성법 → [`sim/metalab/contract/tasks/README.md`](sim/metalab/contract/tasks/README.md)
