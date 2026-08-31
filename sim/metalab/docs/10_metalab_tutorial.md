*MetaLab · Tutorial — 처음 쓰는 사람용 · hammer-lift*

# MetaLab 사용설명서 — hammer-lift 학습·평가

로컬 GPU에서 **hammer-lift** 태스크를 **newton**과 **genesis** 두 물리 엔진으로 학습·평가하는 법.
명령은 딱 둘 — `metalab_train.sh`(학습)와 `metalab_eval.sh`(평가). 나머지 설정은 전부 계약서(YAML)와 experiment.py에 있다.

- train local/rl\_train.sh
- eval local/rl\_eval.sh
- engines newton · genesis
- task hammer-lift

## 00 · 5분 요약 — 두 명령이 전부

스크립트는 `learning/scripts/local/` 아래에 있고, 리포 어디서 실행해도 된다. **`--sim`(엔진)과 `--task`는 항상 필수**(기본값 없음, 안 주면 목록을 뿌리고 종료). 로컬 학습은 언제나 *in-process*(RPC 없음) — 엔진의 conda env 안에서 sim과 트레이너가 한 프로세스로 돈다.

```
# ① 학습 (newton, 스모크 4 envs + 웹 대시보드)
learning/scripts/local/metalab_train.sh --sim newton --task hammer-lift --num_envs 4 --viz

# ② 평가 (방금 학습한 체크포인트를 라이브로 관전)
learning/scripts/local/metalab_eval.sh --sim newton --task hammer-lift --viz --num_envs 1
```

엔진만 바꾸면 genesis에서도 똑같이 돈다(`--sim genesis`). 아래는 이 두 명령의 모든 옵션을 처음 보는 사람 기준으로 풀어 쓴 것이다.

## 01 · 사전 준비

- **conda env 2개** — 엔진별로 하나씩. genesis → env `genesis`, newton → env `newton`. 스크립트가 `--sim`에 맞춰 **자동으로 activate**하므로 미리 activate할 필요는 없다(단, `nohup`/백그라운드 실행도 스크립트를 그대로 부르면 됨 — 스크립트 안에서 activate가 일어난다).
- **GPU** — 기본 GPU 0. 2장이면 `--device cuda:1`로 프로세스 전체를 GPU 1에 고정할 수 있다(엔진 두 개를 두 GPU에 동시에 올릴 때 유용).
- **wandb**(선택) — 학습 곡선/영상 로깅용. `wandb login` 안 돼 있으면 `--no_wandb`로 끄면 된다. 프로젝트는 `jkkim-dexblind-hammer`.
- **실행 위치** — 리포 루트 아무 데서나. 경로는 항상 리포 기준(`learning/scripts/local/…`).

> 예: hammer-lift 스탠드얼론(`sim/metalab/contract/tasks/standalone/manipulation/hammer_lift.py`). `--task`를 빼면 스크립트가 사용 가능한 태스크 목록을 출력한다.

## 02 · 학습하기 · rl\_train.sh

rl\_train.sh → 계약 YAML → EnvSpec → 엔진 parser → VecEnv → PPO (in-process) → model\_\*.pt

### 기본형과 대표 예시

```
# 스모크 — 4 envs + GUI + 웹 대시보드로 한 사이클 눈으로 확인 (버그 조기 발견)
learning/scripts/local/metalab_train.sh --sim newton  --task hammer-lift --num_envs 4 --viz

# 본학습 — 8192 envs, 백그라운드, 로그를 파일로
nohup learning/scripts/local/metalab_train.sh --sim newton --task hammer-lift \
      --num_envs 8192 --max_iterations 5000 > train.log 2>&1 &

# genesis 도 완전히 동일 — 엔진 이름만 교체
learning/scripts/local/metalab_train.sh --sim genesis --task hammer-lift --num_envs 8192

# 두 엔진을 두 GPU에 동시에 (터미널 2개)
learning/scripts/local/metalab_train.sh --sim newton  --task hammer-lift --device cuda:0
learning/scripts/local/metalab_train.sh --sim genesis --task hammer-lift --device cuda:1
```

### 주요 플래그

| 플래그 | 뜻 |
| --- | --- |
| --sim {newton｜genesis} | **필수.** 물리 엔진(=conda env) 선택. headless가 기본. |
| --task hammer-lift | **필수.** 태스크 계약(YAML) 선택. |
| --viz | 그 엔진의 GUI 창 + **라이브 웹 대시보드**를 연다(아래 참조). 느려지므로 관전/디버그용. |
| --num\_envs N | 병렬 환경 수(트레이너로 전달). 미지정 시 YAML 값(4096). |
| --max\_iterations N | PPO iteration 수(트레이너로 전달). 미지정 시 experiment.py 값(5000). |
| --device cuda:N | 프로세스 전체를 물리 GPU N에 고정(`CUDA_VISIBLE_DEVICES`). 다중-GPU 격리에 필수. |
| --no-s3-sync | S3 발행을 끈다(로컬 전용). **기본은 ON** — `model_*.pt`가 `s3://wirobotics-internal/jkkim/sim_rl/ckpts/<run_name>/`로 올라가고, 업로드 성공 후 로컬은 **최신 1개만** 남는다(AWS 노드와 동일. `KEEP_LOCAL_CKPTS=1`로 정리 끔). AWS creds(`aws login`) 필요 — 없으면 경고 후 로컬 전용으로 진행. |
| --record | 체크포인트마다 eval 영상을 **학습 중인 그 wandb run**의 `val/`에 녹화·업로드(별도 터미널 불필요). 동시에 영상+plot 싱크 리포트(`report.html`)를 그 체크포인트 폴더(`.../ckpts/<run_name>/report_<iter>/`)에 발행하고 CloudFront 링크를 `val/report`로 남긴다. wandb creds 필요. |
| --no\_wandb | wandb 로깅 전부 끔(`WANDB_MODE=disabled`). 로그인 없이 돌릴 때. |
| --seed N | 랜덤 시드(트레이너로 전달). |

그 밖의 알 수 없는 플래그는 트레이너(`learning.train --trainer rl`)로 그대로 넘어간다.

### 결과물이 어디에 생기나

- **체크포인트** — `logs/rl/<experiment>/<run_name>/model_<iter>.pt`. `<run_name>`은 `[name]-[datetime]-[SHA]` 형식이라 코드와 1:1 추적된다. 로컬 사본은 절대 자동 삭제되지 않으므로 `metalab_eval.sh`가 디스크에서 바로 읽는다.
- **wandb** — 프로젝트 `jkkim-dexblind-hammer`. 접두사 `Reward/ · Termination/ · Curriculum/ · val/`로 그룹. `val/SR`가 커리큘럼과 무관한 **고정 1cm 성공률**(진짜 성능 지표).

### 설정(노브)은 어디서 바꾸나 데이터=YAML · 학습=experiment.py

| 바꿀 것 | 파일 |
| --- | --- |
| 물리 타이밍 · 엔진별 솔버 · 접촉 파라미터 · 보상 · 관측 · 종료 · 커리큘럼 · DR(도메인 랜덤화) · 목표 pose | sim/metalab/contract/tasks/hammer\_lift\_teacher.py |
| PPO 하이퍼파라미터 · max\_iterations · 네트워크 · obs 라우팅 · wandb 프로젝트 | learning/rl/dexblind/hammer\_lift/experiment.py |

스크립트 자체는 절대 편집하지 않는다 — 실행 방법만 담고, 실험 노브는 위 두 파일에 있다.

### --viz 웹 대시보드

`--viz`를 주면 엔진 GUI와 함께 브라우저에 **라이브 대시보드**가 뜬다. 선택한 env의 obs·reward·termination 시계열 카드와, *Eval/SR* 카드(env별 **성공 횟수 / 시도 횟수** 누적표)를 볼 수 있다. Eval/SR의 성공은 `object_reached_goal` 종료(=현재 커리큘럼 tolerance에서의 성공) 그 자체를 센다.

## 03 · 평가·시각화 · rl\_eval.sh

학습된 체크포인트를 로컬에서 **in-process**로 굴린다. 기본은 headless로 영상을 녹화해 wandb에 올리고, `--viz`면 GUI로 라이브 관전한다. 두 경우 모두 끝에 **SR + per-episode 성공/실패 표**를 로컬 json으로 남긴다.

> ⚠️ **hammer-lift 평가는 언제나 커리큘럼 *end* 조건.** eval 시작 시 자동으로 `apply_curriculum_end()`가 걸려 목표 tolerance가 **1 cm**, 성공 연속 스텝 **20**, 접촉 손가락 **5**로 고정된다(녹화든 관전이든 동일). 학습 중 느슨한 레벨이 아니라 *최종 난이도*로 판정한다.

### 대표 예시

```
# 라이브 관전 — 무한 롤아웃, Ctrl-C로 종료 (녹화 off)
learning/scripts/local/metalab_eval.sh --sim newton --task hammer-lift --viz --num_envs 1

# 특정 체크포인트 지정
learning/scripts/local/metalab_eval.sh --sim newton --task hammer-lift \
      --checkpoint logs/rl/.../model_2000.pt --viz

# headless — 64 에피소드 SR 측정 (영상 없이, 빠르게 숫자만)
RECORD=0 EPISODES=64 learning/scripts/local/metalab_eval.sh --sim newton --task hammer-lift --num_envs 64

# 크로스-엔진 — newton 에서 학습한 정책을 genesis 에서 평가 (sim2sim)
learning/scripts/local/metalab_eval.sh --sim genesis --task hammer-lift \
      --checkpoint logs/rl/.../newton.../model_2000.pt

# 학습이 GPU 0을 쓰는 중 — 평가는 GPU 1에서
GPU=1 learning/scripts/local/metalab_eval.sh --sim newton --task hammer-lift --viz --num_envs 1
```

### 플래그 · 환경변수

| 이름 | 종류 | 뜻 |
| --- | --- | --- |
| --sim {newton｜genesis} | flag | **필수.** 평가할 엔진. |
| --task hammer-lift | flag | **필수.** 태스크. |
| --viz | flag | GUI 라이브 관전(녹화 자동 off). 미지정 → headless 녹화→wandb. |
| --checkpoint PATH | flag | 평가할 `.pt`. 미지정 → 해당 task의 **최신 로컬** 체크포인트 자동 선택. |
| --num\_envs N | flag | 병렬 env 수(관전은 보통 1). |
| EPISODES | env | `-1`(기본)=무한 · `>0`=N 에피소드 후 종료+SR · `0`=고정 STEPS. |
| RECORD | env | `1`(기본, --viz 아닐 때)=wandb 영상 녹화 · `0`=끔. |
| GPU | env | `GPU=N` → 그 물리 GPU에 고정(프로세스는 cuda:0으로 인식). |
| SEED | env | 기본 42(재현). 다른 상황을 뽑으려면 `SEED=<n>`. |
| EXPORT | env | `1`(기본) → `exported/{policy.pt, policy.onnx}` 굽기(raw obs→action, obs 정규화 접힘). sim2sim/sim2real용. `0`=skip. |

> ⚠️ **GUI 관전과 녹화를 동시에 켜지 말 것.** 한 프로세스에 GL 컨텍스트가 둘이면 `GL_INVALID_OPERATION(0x502)`로 죽는다. `--viz`는 자동으로 녹화를 끈다. 녹화는 **newton·genesis 모두 지원**(genesis=rasterizer, newton=headless offscreen ViewerGL; 둘 다 OpenGL이라 **RT 코어 불필요**, 아무 CUDA GPU).

## 04 · 확장 — 새 태스크 · 새 엔진

- **새 태스크** = 계약서(`.py`)만 쓴다. `sim/metalab/contract/tasks/<name>.py`의 `TaskSpec`에 로봇(이름→`robot/<name>.yaml`)·물체/픽스처(인라인)·보상·관측·종료·커리큘럼을 *심볼 + 튜너블*로 선언(보상/관측 계산식은 `sim/metalab/contract/{reward,obs,terminate}/common.py` 의 심볼을 조합 — 보상은 평평한 함수, 관측·종료는 factory). 엔진 코드는 손대지 않는다.
- **새 엔진** = 스포크만 붙인다. `sim/<engine>/{parser, backend}`가 계약서를 그 엔진의 씬으로 번역하고 공통 인터페이스(read·control·step·reset)를 구현하면, 같은 계약서가 그대로 학습된다.

## 05 · 자주 겪는 문제

| 증상 | 원인 · 해결 |
| --- | --- |
| `ModuleNotFoundError`로 즉사(백그라운드 실행) | 스크립트를 직접 부르면 내부에서 conda activate가 일어나므로 정상. 손으로 `python -m ...`을 돌렸다면 env를 먼저 activate. |
| `GL_INVALID_OPERATION 0x502` | GUI 관전 + 녹화 동시. `--viz`만 쓰거나 녹화만 쓴다. |
| 다중-GPU인데 전부 GPU 0으로 몰림 | `--device cuda:N`(train) / `GPU=N`(eval)으로 `CUDA_VISIBLE_DEVICES` 고정. 인덱스만 넘기면 genesis/warp가 무시하고 GPU 0에 올라감. |
| S3 업로드가 안 됨 | 워크스테이션엔 인스턴스 role이 없으니 AWS creds(`aws login`/프로파일) 필요. 없으면 경고 후 로컬 전용으로 계속되고(아무것도 삭제 안 함), `--no-s3-sync`로 명시적으로 끌 수도 있다. |
| 리포트에서 플롯은 보이는데 영상만 안 나옴 | presigned S3 링크로 열었을 것. presigned URL 은 키 하나에만 서명되므로 페이지의 상대참조 `env*.mp4` 가 서명 없이 요청되어 403. W&B `val/report` 의 **CloudFront** 링크로 열면 정상. |
| 체크포인트가 S3에만 있음 | 정상이다 — 업로드 후 로컬은 최신 1개만 남는다. eval은 S3를 안 당겨오므로 예전 것은 먼저 `aws s3 cp s3://.../ckpts/<run_name>/model_N.pt /tmp/` 후 `--checkpoint`로 넘긴다(최신 것은 그대로 동작). |

MetaLab · sim/ · local training tutorial
개요 = 00\_project\_overview · 엔진 parity = 01\_engine\_parity
