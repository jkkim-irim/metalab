# MetaLab

엔진 무관 로봇 학습 시뮬레이터. **계약서 한 장**으로 환경을 정의하고 **Newton · Genesis** 두 물리 엔진에서
같은 환경을 만들어 학습·평가한다. 로컬 GPU 전용이고, 클론과 `setup.sh` 외에 준비할 것이 없다.

```
sim/metalab/       계약서(contract) · 엔진 스포크(backends) · 런타임 · 에셋
learning/          RL 트레이너 · 정책 · 평가 · 실행 스크립트
```

- 설계 개요 → [`sim/metalab/docs/00_project_overview.md`](sim/metalab/docs/00_project_overview.md)
- 플래그·환경변수 전체 → [`learning/scripts/local/README.md`](learning/scripts/local/README.md)
- 셋업 상세·재현성 → [`sim/metalab/README.md`](sim/metalab/README.md)

---

## 0 · 준비

필요한 것: **Ubuntu 22.04/24.04 · NVIDIA GPU + 드라이버(≥550) · git · build-essential · 디스크 ~30 GB.**
`uv` 는 setup 이 알아서 설치한다(Python 도 uv 가 제공 — conda 불필요).

```bash
git clone <this repo> ~/metalab_ws/metalab && cd ~/metalab_ws/metalab
sim/metalab/setup.sh          # 핀 고정된 엔진 소스 clone + 엔진별 uv 환경
```

`setup.sh` 는 몇 번 돌려도 안전하다. 엔진 소스(newton, genesis-world)는 이 리포의 **형제 디렉터리**로
받아가고, 커밋은 `sim/metalab/sim_versions.env` 에 고정돼 있다.

- **GPU** 기본은 0번. 두 장이면 `--device cuda:1` 로 프로세스 전체를 1번에 고정할 수 있다.
- **wandb** 는 선택. 로그인 안 돼 있으면 `--no_wandb` 로 끈다(안 끄면 실행 전에 실패한다).

## 1 · 첫 실행

```bash
sim/metalab/launchpad.sh      # 웹 콘솔: 엔진·태스크·학습/평가·노브를 고르고 Launch
```

CLI 로 직접 돌려도 된다. `--sim`(엔진)과 `--task` 는 **항상 필수**고, task 가 family 면 `--recipe` 도
필요하다(빠뜨리면 목록을 출력하고 종료한다).

```bash
# ① 스모크 학습 — 4 envs + GUI 로 한 사이클 눈으로 확인
learning/scripts/local/metalab_train.sh --sim newton \
    --task hammer-lift-teacher --recipe privileged --num_envs 4 --viz gl

# ② 방금 만든 체크포인트를 라이브로 관전
learning/scripts/local/metalab_eval.sh --sim newton \
    --task hammer-lift-teacher --recipe privileged --viz --num_envs 1

# ③ 본학습 — 백그라운드
nohup learning/scripts/local/metalab_train.sh --sim newton \
    --task hammer-lift-teacher --recipe privileged --num_envs 8192 --max_iterations 5000 \
    > train.log 2>&1 &
```

첫 `metalab_train.sh` 는 그 엔진의 환경이 없으면 `setup.sh` 가 하는 일을 자동으로 한다 — 새 머신에서
곧바로 ①번 줄로 시작해도 된다.

엔진 이름만 바꾸면 genesis 에서도 똑같이 돈다(`--sim genesis`). 두 엔진을 두 GPU 에 동시에 올릴 수도 있다
(`--device cuda:0` / `--device cuda:1`).

## 2 · 결과물이 어디에 생기나

- **체크포인트** — `logs/rl/<experiment>/<run_name>/model_<iter>.pt`. `<run_name>` 은 datetime + git SHA 를
  달고 있어 코드와 1:1 추적된다.
- **wandb** — 스칼라 커브. `Reward/ · Termination/ · Curriculum/ · val/` 접두사로 그룹된다.
  `val/SR` 이 커리큘럼과 무관한 **고정 tolerance 성공률**(진짜 성능 지표)이다.
- **리포트** — `--record` 를 주면 체크포인트마다 rerun `.rrd` + 플롯을 합친 `report.html` 이 그 체크포인트
  옆(`report_<iter>/`)에 남고, W&B 에는 그 경로가 기록된다.

## 3 · 노브는 어디서 바꾸나

| 바꿀 것 | 파일 |
|---|---|
| 물리 타이밍·솔버·접촉·보상·관측·종료·커리큘럼·DR·목표 pose | `sim/metalab/contract/tasks/<task>/` (`_base.py` + 레시피) |
| PPO 하이퍼파라미터·네트워크·obs 라우팅·wandb 프로젝트 | `learning/rl/dexblind/<task>/experiment.py` |

**스크립트 자체는 편집하지 않는다** — 스크립트는 실행 방법만 담고, 실험 노브는 위 두 곳에 있다.

## 4 · 평가에서 알아둘 것

- **평가는 언제나 커리큘럼 *end* 조건**으로 판정한다. eval 시작 시 자동으로 `apply_curriculum_end()` 가
  걸려 목표 tolerance·유지 스텝·접촉 손가락 수가 최종 난이도로 고정된다(녹화든 관전이든 동일). 학습 중의
  느슨한 레벨이 아니다.
- **GUI 관전과 녹화를 동시에 켜지 말 것.** 한 프로세스에 GL 컨텍스트가 둘이면
  `GL_INVALID_OPERATION(0x502)` 로 죽는다. `--viz` 는 자동으로 녹화를 끈다.
- **크로스-엔진 평가**(newton 에서 학습 → genesis 에서 평가)도 `--sim` 만 바꾸면 된다. 다만 현재
  sim2sim 교차평가 SR 은 0% 다 — [`01_engine_parity`](sim/metalab/docs/01_engine_parity.md) 참고.

## 5 · 확장

- **새 태스크** = 계약서만 쓴다. `sim/metalab/contract/tasks/<name>/` 에 `_base.py`(공유 코어)와
  `<name>_<recipe>.py`(reward/gate/curriculum 트리오)를 두면 런치패드와 스크립트가 자동으로 인식한다.
  보상·관측·종료 계산식은 `terms/` 의 심볼을 조합한다. 엔진 코드는 손대지 않는다.
  물체 에셋은 `sim/metalab/assets/objects/` 에 88종이 들어 있어 `object_mjcf("<이름>")` 으로 바로 쓴다.
- **새 엔진** = 스포크만 붙인다. `sim/metalab/backends/<engine>/{parser,backend,server}` 가 계약서를 그
  엔진의 씬으로 번역하고 공통 인터페이스(read·control·step·reset)를 구현하면, 같은 계약서가 그대로 학습된다.

## 6 · 자주 겪는 문제

| 증상 | 원인 · 해결 |
|---|---|
| `uv: command not found` | `sim/metalab/setup.sh` 재실행(자동 설치) 후 `$HOME/.local/bin` 을 PATH 에 넣는다. |
| `uv sync` 가 소스 빌드에서 실패 | C 툴체인이 없다. `sudo apt install build-essential`. |
| `ModuleNotFoundError: genesis` / `newton` | 형제 엔진 소스가 없거나 핀과 다른 커밋이다 — `sim/metalab/setup.sh` 재실행. |
| `ModuleNotFoundError` (그 외) | 스크립트를 직접 부르면 엔진 uv venv 가 안에서 활성화된다. 손으로 `python -m ...` 을 돌렸다면 venv 를 먼저 활성화한다. |
| `GL_INVALID_OPERATION 0x502` | GUI 관전 + 녹화 동시. 둘 중 하나만 쓴다. |
| 다중 GPU 인데 전부 GPU 0 으로 몰림 | `--device cuda:N`(train) / `GPU=N`(eval). 인덱스만 넘기면 genesis/warp 가 무시하고 GPU 0 에 올라간다. |
| 리포트에서 플롯은 보이는데 3D 만 안 나옴 | `file://` 로 열었을 것이다. 3D 패널은 rerun 뷰어 번들을 받아와야 해서 http(s) 로 서빙해야 한다. 옆의 `rollout.rrd` 를 로컬 rerun 뷰어로 열어도 된다. |
