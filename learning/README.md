# learning/

MetaLab 정책의 학습·평가. 실행 방법은 [`scripts/local/README.md`](scripts/local/README.md), 처음
보는 사람용 안내는 리포 루트 [`README.md`](../README.md).

## 레이아웃

```
learning/
├── train.py                 학습 진입점 (설정 파싱 → 트레이너 디스패치)
├── trainer/rl_trainer.py    RL 트레이너 — sim-service 를 띄우고 PPO/SAPG 를 돌린다
├── rl/                      정책·알고리즘 (ppo·sapg·models·nn), sim-service 클라이언트(client·service)
│   └── dexblind/<task>/     experiment.py — RL 설정(PPO 하이퍼·네트워크·obs 라우팅)
├── eval/                    체크포인트를 sim-service 경계 너머로 평가
├── metrics/ · utils/ · configs/
└── scripts/local/           유지되는 실행 스크립트 (metalab_train.sh · metalab_eval.sh)
```

`learning/` 은 패키지다(각 디렉터리에 `__init__.py`). 모듈은 패키지 경로로 import 한다 —
`from learning.rl.client import SimServiceVecEnv` 처럼.

## 설치

엔진 uv 환경은 `sim/metalab/setup.sh` 가 만든다 — 트레이너와 엔진 서버가 **같은 venv** 에서 뜨므로
`learning` 을 따로 설치할 필요는 없다.

## 노브가 있는 곳

| 바꿀 것 | 파일 |
|---|---|
| 보상·관측·종료·커리큘럼·물리·DR (태스크 노브) | `sim/metalab/contract/tasks/<task>/` |
| PPO 하이퍼파라미터·네트워크·obs 라우팅 | `learning/rl/dexblind/<task>/experiment.py` |

스크립트 자체는 편집하지 않는다 — 실행 방법만 담는다.
