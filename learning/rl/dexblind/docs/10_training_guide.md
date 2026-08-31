# 학습 / 평가 / 테스트 실행법

2-venv 구조: **sim 서비스**(`sim/isaaclab`, `isaaclab` conda env — Isaac Lab + Newton)와 **트레이너**
(`learning/`, `rltrainer` venv — Isaac Lab 無, vendored rsl_rl). 트레이너가 sim 서버를 띄우고 실험 상수를
넘기면 서버가 env 를 빌드한다. 학습/play 는 **RTX GPU 필요**, 로직 테스트는 **GPU 불필요**.

## 학습 (train) — GPU 노드에 배포 + 실행
```bash
# 코드를 노드에 rsync(SSM-SSH) + sim 패키지 editable 재설치 + 학습.
# (프로비저닝된 노드 필요: node.sh up && node.sh provision — provision 은 env 스냅샷만 복원, 코드는 여기서 배포)
NODE=i-0abc... [NUM_ENVS=1024 MAX_ITERS=5000 SEED=42 WANDB_PROJECT=chrisryu-simrl] \
    sim/isaaclab/scripts/aws/train.sh
```
train.sh 가 노드의 `rltrainer` env 에서 실행하는 명령:
```bash
python -m learning.train --trainer rl --num_envs 1024 --max_iterations 5000 --seed 42 \
    --logger wandb --wandb_project chrisryu-simrl
```
- 실험(PPO + 보상/커리큘럼/성공/DR/노이즈 상수)은 `learning/rl/dexblind/hammer_lift/experiment.py` 한 곳.
  트레이너가 `ENV_TUNABLES` 를 sim 서버로 넘겨 env cfg 에 bind, PPO 로 `OnPolicyRunner` 를 만든다.
- wandb run 이름 = `[datetime(UTC)]_[task]_[iters]_[envs]-[short SHA]` 로 코드와 1:1 추적. 공용 dev 프로젝트 사용.

## 평가 (play)
```bash
NODE=i-0abc... sim/isaaclab/scripts/aws/eval.sh   # 최신 체크포인트로 play.py --video → headless Newton MP4
```
`play.py`(= `sim/isaaclab/play.py`) 는 gym 레지스트리 없이 env 를 직접 빌드하고 실험을 JSON 인자로 받는다
(`--experiment_file` / `--ppo_file` / `--checkpoint`). Play 변형 = 노이즈·외력 off + success 조건 고정
(시작=관대 값), 단일 env. (eval 런처의 실험-파일 배선 + 렌더 재검증은 follow-up.)

## 테스트 (CPU, GPU 불필요)
```bash
python -m pytest learning/rl/dexblind/hammer_lift/tests   # 실험 상수 불변식 (스케줄 부호/시작-끝, 단위 quat 등)
python -m pytest sim/isaaclab/tests                        # 커리큘럼/성공게이트/env-cfg 조립 (Isaac env 필요)
```
Isaac env 가 없는 경량 CI 에서는 `importorskip`/skip 으로 자동 skip. **테스트 통과 ≠ sim 동작 보증**
(end-to-end 는 GPU 학습 스모크로 확인).

## 운영 규칙 (팀)
- **val/ckpt/sim-eval 변경 후 장기 학습 전 반드시 스모크** — 짧은 간격으로 1 사이클 end-to-end 확인.
- 값 튜닝은 `learning/rl/dexblind/hammer_lift/experiment.py` 한 곳에서. 커리큘럼 스케줄도 동일.
- upstream(isaaclab/newton) 직접 수정 금지 — 래퍼/오버라이드로 우리 코드에서 처리.
