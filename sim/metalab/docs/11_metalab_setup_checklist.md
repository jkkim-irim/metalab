*MetaLab · 체크리스트 — 셋업 · 재현성 · uv 이전*

# MetaLab 셋업·재현성 체크리스트

팀원 누구든 **allex repo 만 clone → `sim/metalab/setup.sh` → 학습**이면 내가 쓰는 것과
동일한 환경으로 돈다. local·AWS 공통. 엔진 env 는 **conda 를 벗고 uv** 로 관리한다(`uv.lock` = 재현 산출물).
이 문서는 그 셋업의 **살아있는 체크리스트** — 재현성 체인, uv 이전 검증 결과, 산출물, 버전 bump 절차, 남은 작업까지 추적한다.

- 진입점 sim/metalab/setup.sh
- 매뉴얼 sim/metalab/README.md
- 버전 SSOT sim/metalab/sim\_versions.env
- env uv.lock (엔진별)

## 00 · 한 줄 요약

> 재현성은 **버전 통제되는 진실소스 3개**로 성립한다 — ① *코드*(allex repo, git SHA) ·
> ② *시뮬레이터 소스*(newton·genesis-world 형제 clone, `sim_versions.env` pin) ·
> ③ *파이썬 의존성*(`uv.lock`, 엔진별, git 버전관리). 에셋은 리포 내(LFS 아님). `sim/metalab/setup.sh` 가
> ①의 형제로 ②를 pin 커밋에 맞춰 clone 하고 ③을 `uv sync` 로 재현한다 — 수동 conda/pip 없음.

> **피벗: conda-pack(S3) → uv.lock.** 기존 5GB conda-pack 스냅샷(S3)은 폐기하고 **엔진별 `uv.lock`**
> 으로 대체한다. 실측 결과 conda env 는 *껍데기*였다 — torch·warp·CUDA·usd·mujoco 전부 pip wheel 이고 conda 는 python 만 제공.
> uv 는 그 python(3.12/3.11)까지 자체 제공하므로 **miniconda 도 더 이상 필수 아님**. 이점: S3 blob·AWS creds(env용)·chicken-and-egg 소멸.
> > **핵심 불변식.** pin(`sim_versions.env`) = 형제 repo HEAD. 나머지 deps = `uv.lock`(정확 버전+해시).
> > 버전 bump = 형제 re-checkout + `uv lock` 재생성, 둘 다 git 커밋(항상 함께).

## 01 · 재현성 체인 — 세 진실소스

| 조각 | 위치 | 고정 주체 |
| --- | --- | --- |
| 학습/env/task 코드 | 이 리포 | git SHA (run 이름에 각인) |
| 시뮬레이터 소스 | 형제 repo (newton, genesis-world) | `sim/metalab/sim_versions.env` |
| 파이썬 의존성 | `uv.lock` (엔진별, 리포 내) | uv (정확 버전+해시 핀) |
| 로봇/오브젝트 에셋 | 리포 내 `sim/metalab/assets/` (plain git) | git SHA |

> ⚠️ **결과의 bit-identical 은 보장 못 함.** *셋업*은 동일 재현되지만, 학습 *결과*는
> GPU 모델·드라이버·CUDA 차이로 인한 비결정성이 있어 기기 간 완전 동일하진 않다(절차·코드는 동일, 결과는 근사).

## 02 · uv 이전 — 프로토타입 검증 (완료)

conda→uv 로 확정하고 **양 엔진 end-to-end 검증**했다(RTX 5090). clone→`uv lock`→`uv sync`→**실제 PPO 학습**까지 conda 없이 통과.

| 단계 | genesis | newton |
| --- | --- | --- |
| uv lock (resolve) | ✅ 123 pkgs | ✅ 120 pkgs |
| uv sync (install) | ✅ | ✅ |
| torch cu128 · CUDA | ✅ 2.10.0+cu128 | ✅ 2.10.0+cu128 |
| 엔진 import (editable·pinned) | ✅ | ✅ +mujoco\_warp |
| warp on sm\_120 | — (미사용) | ✅ 1.16.0.dev20260713 |
| 학습 스모크 (PPO 2 iter) | ✅ | ✅ |

> **warp 결정.** newton(1.5.0.dev0)은 `warp-lang>=1.15.0.dev…` 를 명시해 *stable warp 로는 원천 불가*.
> uv 가 NVIDIA 인덱스(`pypi.nvidia.com`)에서 **최신 nightly `1.16.0.dev20260713` 을 자동 선택하고 uv.lock 에 핀** →
> "안정적으로 도는 최신 빌드, 재현 가능하게 고정". RTX 5090(Blackwell sm\_120)에서 warp init·학습 확인. genesis 는 warp 무관(자체 백엔드).

## 03 · 산출물

- [x] **`sim/metalab/sim_versions.env`** — 시뮬레이터 버전 pin **단일 진실소스**.newton `2ee010b` · genesis-world `491d41e` (최신, 업스트림 실존 확인).
- [x] **`sim/metalab/{genesis,newton}/pyproject.toml` + `uv.lock`** — 엔진별 uv 프로젝트(=lock 소유자).엔진 editable(pinned 형제) + torch cu128 + tensordict/wandb; nvidia·pytorch 인덱스 설정. 양쪽 검증 완료.
- [x] **`metalab_train.sh`** (수정) — wandb ON 인데 미로그인이면 **학습 시작 전 fail-loud** + 로그인/`--no_wandb` 안내.
- [x] **크롬 앱-창** (`launchpad.sh`·`server.py`·`telemetry.py`) — Launchpad/telemetry 를 격리 프로필의 standalone 앱 창으로.
- [x] **스크립트 uv 이전 완료** — `setup_env.sh`·`lib.sh`·`metalab_train.sh`·`metalab_eval.sh`·`sim/metalab/setup.sh`·`README.md` conda 제거, `snapshot_env.sh` 폐기. **fresh venv 에서 양 엔진 학습 스모크 통과** (아래 05).

## 04 · 전제조건 — 사람이 준비할 것 (uv 기준)

- [ ] **uv** — 엔진 env 관리자. python(3.12/3.11)까지 자체 제공.1회 설치: `curl -LsSf https://astral.sh/uv/install.sh | sh`. *miniconda 불필요.*
- [ ] **GPU + NVIDIA 드라이버** · **~30GB 디스크**실제 물리·학습 실행용. wheel 캐시 + 엔진 소스 + 리포 메시.
- [ ] **wandb 로그인** — 학습은 기본 wandb 로깅.`wandb login` 또는 `--no_wandb`(Hub "wandb 끄기"). 미로그인+ON = fail-loud.
- [ ] **AWS 연결** — **env 엔 불필요**(uv 가 PyPI/인덱스에서).AWS 노드 학습 · S3 체크포인트에만 `aws login`.
- [ ] **Chrome** (선택) — Hub 를 standalone 앱 창으로. 없으면 기본 브라우저 탭으로 폴백.

## 05 · 스크립트 이전 (conda → uv) — 완료 ✅

local 유지 스크립트를 uv 경로로 전면 이전하고, **fresh venv root 에서 `sim/metalab/setup.sh` → 양 엔진 `metalab_train.sh` 학습 스모크(PPO 2 iter, TRAIN_SERVICE_OK)** 로 검증했다.

- [x] **`setup_env.sh`** — conda-pack S3 복원 → **형제 clone(pin converge) + `uv sync`**. AWS creds·S3 blob 불필요.
- [x] **`lib.sh`** — `SIM_CONDA_SH`/conda activate 제거 → `METALAB_VENV_ROOT` + `engine_venv`/`engine_uv_project` 헬퍼.
- [x] **`metalab_train.sh` · `metalab_eval.sh`** — `conda activate` → uv venv 활성화(`source $venv/bin/activate`).
- [x] **`sim/metalab/setup.sh`** — preflight(uv 없으면 자동 설치) + `setup_env.sh`(clone+uv sync) 위임.
- [x] **`snapshot_env.sh` 폐기**(삭제) — uv.lock 이 재현 산출물로 대체.
- [x] **uv env deps 완성** — 영상: `imageio-ffmpeg`·`matplotlib`(newton은 `moviepy`·`opencv`도); **newton 뷰어/오프스크린 recorder: `pyglet`·`imgui-bundle`**(newton `examples`/`rtx` extra에만 있어 `[sim,importers]`로 안 딸려옴 → 명시 추가). conda↔uv 전체 diff로 render/media 갭 훑음. import 확인.
- [x] **`sim/metalab/README.md` · 이 체크리스트** uv 기준 갱신.

**다음 (별도 단계):**
- [ ] **AWS 경로** (`aws/metalab_train.sh`) — 노드도 `uv sync` → presigned-S3-blob 메커니즘 제거. 노드 부트스트랩에 uv 설치.
- [ ] ⚠️ **warp nightly 재현성** — uv.lock 이 버전+해시로 핀하지만 NVIDIA 인덱스가 그 nightly 를 지우면 재-sync 실패 가능. 필요 시 그 wheel 만 우리 S3 에 캐싱 + uv 인덱스 추가. (env owner)
- [ ] **push + main 머지** · **진짜 fresh 머신 검증** · 구 S3 conda-pack blob(`newton/genesis.tar.gz`) 삭제. (env owner)

## 06 · 버전 업데이트 절차 — env owner (uv)

새 newton/genesis 커밋이 **검증되면**, 아래를 **한 세트로**(pin 과 lock 은 항상 같이):

1. `sim/metalab/sim_versions.env` 의 해당 `*_REF` 를 새 커밋으로 설정.
2. 형제 소스를 새 커밋에 맞춤: `sim/metalab/setup.sh` 가 clean tree 를 자동 checkout(converge).
3. 엔진 uv 프로젝트에서 `uv lock` 재생성 → 새 deps 반영(필요 시 `uv sync` 로 로컬 검증).
4. commit(`sim_versions.env` + `uv.lock`) + merge → 팀원은 `sim/metalab/setup.sh`(=`uv sync`)로 수렴.

> **S3 스냅샷·`snapshot_env.sh` 불필요.** uv.lock 이 재현 산출물(git 안). conda 시절의 "스냅샷 먼저 올려야 복원 가능" chicken-and-egg 가 사라진다.

## 07 · 남은 항목 — 코드 밖

- **노드 IAM PassRole** — `metalab-node-role` 이 gpu-launchers PassRole 에 없어 AWS 노드 런치엔 admin 필요(PRD P3, 인프라 작업).
- **드라이버 최소버전** — README 는 ≥550 으로 표기. 팀 실제 최소 GPU 기준으로 조정 권장(마이너).

MetaLab · sim/ · 셋업·재현성 체크리스트 (uv)
매뉴얼 = sim/metalab/README.md · Hub = 20\_metalab\_hub\_prd · 사용법 = 10\_metalab\_tutorial
