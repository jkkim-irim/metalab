# ALLEX Dexblind RL — 프로젝트 개요

ALLEX 휴머노이드 손의 **dexterous manipulation RL** — 망치를 잡아 목표 pose 로 들어올리는 과제.
**Isaac Lab (ManagerBasedRLEnv) + Newton (MuJoCo Warp) GPU 물리** 백엔드, **rsl_rl PPO** 학습.

> 값/스케줄의 단일 출처는 트레이너의 `learning/rl/dexblind/hammer_lift/experiment.py`. 이 문서는
> "무엇을 / 어디서" 만 설명하고, 구체 수치는 코드 상수를 참조한다(값이 자주 바뀌므로 여기 하드코딩하지 않음).

---

## 디렉토리 (sim 서비스 + 트레이너, 2-venv)

```
sim/isaaclab/                    # sim 서비스 (isaaclab env — Isaac Lab + Newton). 서버가 env 를 빌드/서빙.
├── server.py  transport.py      # env 를 직접 빌드해 RPC 로 서빙 / 와이어 프로토콜
├── play.py                      # 평가(체크포인트 렌더)
├── envs/hammer_lift/            # 태스크 env — env.py, env_cfg.py(make_env_cfg 팩토리), mdp/(보상·관측·
│   │                            #   이벤트·종료·커리큘럼), scene_cfg.py, simulation.py, visualizer
│   └── _experiment.py           # 트레이너가 넘긴 튜너블을 env cfg 에 bind (값은 저장하지 않음)
├── robot/                       # ALLEX 로봇: params, asset.py(ArticulationCfg), ik/, gains.json
├── assets/                      # 해머 USD + 텍스처
├── scripts/aws/                 # 배포 (train.sh / eval.sh / node.sh / provision — SSM-SSH, S3 에 코드 X)
└── tests/                       # 커리큘럼/보상/env-cfg 조립 (Isaac env 필요, GPU 불필요)

learning/rl/                     # 트레이너 (rltrainer venv — Isaac Lab 無, vendored rsl_rl PPO)
├── on_policy_runner.py ppo.py models/ nn/ ...   # PPO 엔진
├── client.py  service.py        # sim 서비스 클라이언트 + 서버 spawn
└── dexblind/hammer_lift/
    ├── experiment.py            # 실험 = PPO cfg + 태스크 튜너블(보상·커리큘럼·성공·DR·노이즈). 값의 단일 출처 ← 여기서 튜닝
    └── docs/                    # 이 문서 폴더
```

## 등록 환경 (Gym ID)

| ID | 로봇 | 설명 |
|----|------|------|
| `Isaac-Dexblind-Newton-Allex-Lift-v0` / `-Play-v0` | full-body | 허리+목+양팔+양손(오른쪽만 제어) |
| `Isaac-Dexblind-Newton-Allex-Dense-Lift-v0` / `-Play-v0` | dense | 오른팔+오른손만 (현재 주 학습 대상) |

`-Lift-v0` = 학습(노이즈·외력·커리큘럼 on), `-Play-v0` = 평가(노이즈/외력 off, success 조건 고정).

## 스택 요약

- **물리**: Newton(MuJoCo Warp), CUDA graph. 200Hz 물리 × decimation 4 → **50Hz policy**. (`simulation.py`)
- **학습**: rsl_rl PPO — LSTM actor + asymmetric MLP critic. (`config/allex/agents/rsl_rl_ppo_cfg.py`)
- **액션**: EMA joint-position-to-limits (arm + hand). (`learning.py` ARM/HAND_ACTION_*)
- **과제 정의·성공 조건·커리큘럼**: → `01_prd_dexblind_rl.md`

## 관련 문서

- `01_prd_dexblind_rl.md` — 과제 요구사항/설계(성공 조건, 커리큘럼, 보상, 종료)
- `02_commit_convention.md` — 커밋/브랜치/PR 규칙
- `10_training_guide.md` — 학습·평가·테스트 실행법
