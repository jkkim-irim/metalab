*MetaLab · Overview — 처음 보는 사람용 · sim/*

# MetaLab — 하나의 계약서, 여러 시뮬레이터

**MetaLab**(sim/)은 로봇 학습 환경을 **엔진과 무관하게 한 번만 정의**하고, 그 정의로 Genesis·Newton 등 여러 물리 엔진에서 **같은 환경**을 만들어내는 ALLEX 자체 시뮬레이션 계층이다. 엔진마다 환경을 새로 만들 필요가 없다.

- hub 계약서 = .py (태스크) · YAML (로봇·물체)
- spoke genesis · newton
- 목표 셋업 통일 · 물리차 계측

## 01 · 한 줄 요약 · 왜 필요한가

> **문제.** 같은 로봇 태스크를 엔진마다 따로 세팅하면 — 로봇 배치, 질량, 관절, 보상이 조금씩 어긋난다. 그러면 "Genesis에선 되는데 Newton에선 안 되는" 이유가 *환경 차이인지 물리 엔진 차이인지* 구분할 수 없다. sim2real 로 갈 때도 어디서 오차가 생겼는지 못 짚는다.
>
> **해법.** 환경을 *엔진-무관 계약서(contract)* 하나로 정의한다. 엔진별 *parser* 가 그 계약서를 읽어 각자의 씬을 만든다. 셋업이 같으므로, 두 엔진의 남는 차이는 **순수 물리 엔진 차이**뿐 — 이제 측정 가능한 값이 된다.

> **현재 상태 (2026-07-12)**
> · **hammer-lift-teacher** 태스크가 **newton·genesis 두 엔진에서 각각 재현** — MetaLab이 실사용 가능한 상태.
> · newton이 legacy·genesis 대비 느리고 VRAM 많이 쓰던 원인 규명·해소(진범 = `rigid_gap`) — 상세는 **01\_engine\_parity**.
> · *future work* — 한 엔진에서 학습한 정책을 다른 엔진에서 평가하는 **sim2sim 교차평가는 SR 0%**. 엔진 간 물리 격차 축소(또는 멀티-엔진 학습)가 다음 과제.
> · 학습·평가 방법은 **10\_metalab\_tutorial**(rl\_train.sh · rl\_eval.sh).

## 02 · 핵심 그림 — Hub & Spoke

**Hub**(바퀴 중심) = 계약서 한 장. **Spoke**(바퀴살) = 엔진별 어댑터. 중심을 한 번 정의하면 바퀴살이 각 엔진으로 뻗어 나간다.

계약서 (.py)
envs/ · 엔진 무관

genesis parser
→ gs.Scene

newton parser
→ newton.Model

동일한
환경

HUB
SPOKE (엔진별)

계약서 1장 → parser N개 → 시뮬레이터 N개. 구조는 통일, 물리 엔진 차이만 남는다.

같은 계약서를 두 엔진에 통과시켜 **구조는 똑같이** 만든다. 접촉·솔버 방식이 달라 **물리 거동**은 엔진마다 다르며, 그 차이 자체가 우리가 재고 싶은 값(=sim2real 오차 예산의 예측치)이다.

## 03 · 3개의 층으로 이해하기

sim/ 은 역할이 뚜렷한 세 층으로 나뉜다 — **무엇을**(계약) · **어떻게**(엔진) · **돌린다**(런타임).

### HUB 무엇을 envs/ · api/ · sim/\_assets/

환경이 무엇인지 — 로봇·물체·보상·관측·목표를 **엔진 모르게** 정의한다. 태스크는 선언적 **.py 계약서**(이름 + 튜너블 + 심볼 참조), 보상·관측 계산식은 api primitive를 조합한 .py, 로봇·물체 모델은 sim/\_assets 의 **MJCF(진실 소스)**. 이 층은 어떤 엔진도 import 하지 않는다.

### SPOKE 어떻게 genesis/ · newton/

계약서를 각 엔진의 씬으로 번역한다. parser 가 계약서 → 엔진 씬(gs.Scene / newton.Model)을 만들고, backend 가 그 씬을 읽고/제어하는 공통 인터페이스(read·control·step·reset)를 구현한다. **엔진 의존은 오직 이 층에만** 격리된다.

### RUNTIME 돌린다 runtime/

엔진-무관 실행 층. env\_driver 가 backend 를 학습용 **VecEnv**(관측·보상·액션 디코드·종료·로깅)로 감싼다. parity\_rollout 은 두 엔진을 같은 계약서로 굴려 궤적을 대조 — 물리차를 계측한다.

한 줄 요약: **envs**(무엇) 은 엔진을 모르고, **genesis/newton**(어떻게) 만 엔진을 알고, **runtime**(돌린다) 은 그 위에서 학습·검증을 조립한다.

## 04 · 디렉토리 한눈에

sim/metalab/

├─ envs/HUB — 환경 정의 (엔진 무관, 단일 출처)

│ ├─ tasks/태스크 계약서 = 선언적 .py (hammer\_lift\_teacher.py)

│ ├─ robot/ object/로봇·물체 계약 (마스크·액션·물리속성) .yaml

│ ├─ reward/ obs/ terminate/보상·관측·종료 계산식 (common.py)

│ └─ spec · loader · mjcf\_prep스키마 검증 · 계약서→EnvSpec · MJCF 준비

│

├─ api/엔진-무관 primitive (frames·keypoints·shaping·state)

├─ (sim/\_assets/)로봇·물체 모델 — MJCF 진실 소스 (robots/ · objects/; backend-중립이라 sim/ 레벨 공유)

│

├─ runtime/공유 런타임 (엔진 무관)

│ ├─ env\_driver.py계약서 → 학습용 VecEnv

│ ├─ backend.pySimBackend 인터페이스 (엔진이 구현)

│ └─ parity\_rollout.py두 엔진 대조 (물리차 계측)

│

├─ genesis/SPOKE — parser · backend · server (+ env/ = uv 프로젝트)계약서 → gs.Scene → 학습

└─ newton/SPOKE — parser · backend · server (+ env/ = uv 프로젝트)계약서 → newton.Model → 학습

프로세스 경계(RPC + CUDA-IPC 트랜스포트)는 **sim/service/** — sim 스택들이 공유하는 cross-sim 계층. MetaLab 은 **sim/metalab/** 아래 자립하며 sim/isaaclab · sim/libero 등의 **peer** 다. 엔진 스포크(genesis·newton)만 그 엔진을 import 한다.

## 05 · 태스크 하나가 학습이 되기까지

계약서 한 장이 두 경로로 소비된다 — **학습**과 **검증(parity)**. 둘 다 같은 계약서에서 출발한다.

학습 경로

계약서(.py) → loader (EnvSpec) → parser (엔진 씬) → backend → env\_driver · VecEnv → PPO 학습

검증 경로 (sim2sim)

계약서(.py) → genesis · newton parser → 롤아웃 ×2 → 궤적 대조 → 물리차 리포트

계약서가 실제로 어떻게 생겼는지 — 이름 + 튜너블만 적고, 기능은 **import 한 심볼**로 참조한다
(보상은 평평한 함수 `fn(env, **params)`, 관측·종료·이벤트는 factory 심볼):

```python
# _envs/tasks/hammer_lift_teacher.py — 선언적 계약서
from sim.metalab.terms.reward import lifting_reward, object_goal_keypoint_progress
from sim.metalab.contract.spec import TaskSpec, TermRef

TASK = TaskSpec(
    name="hammer_lift_teacher",
    robot="allex_right",        # robot/allex_right.yaml
    objects=[{                  # 인라인 object 계약 (asset MJCF + 물리 프로퍼티, task 소유)
        "name": "hammer", "mass": 0.55, "friction": 1.0, "variants": 3,
        "asset": {"mjcf": ["sim/metalab/assets/objects/hammer/hammer_cylinder.xml"]},  # (+2 variants)
        "init_pos": [0.5, -0.1, 0.91], "init_rpy": [0.0, 0.0, 0.0],
    }],
    reward=[                     # reward/common.py 의 함수(심볼) + weight. 크기는 weight 하나로만 정한다
        TermRef(fn=lifting_reward, weight=2.0, kwargs={"lift_height": 0.1}),
        TermRef(fn=object_goal_keypoint_progress, weight=200),
    ],
)
```

## 06 · 기억할 규칙 4가지

### 1로봇·물체 = YAML, 태스크·로직 = .py

로봇·물체 = YAML(값). 태스크 계약서·보상·관측 = .py. 계약서는 값과 심볼을 **조합만** 하고 로직은 담지 않는다.

### 2MJCF = 물리 진실 소스

게인·질량·관절 한계 등 물리값은 MJCF 하나에. 로봇 YAML·태스크 .py 는 그 위의 얇은 오버라이드(마스크·액션·튜너블)일 뿐.

### 3엔진 import 금지 (계약·런타임)

계약서·부품·런타임은 엔진을 모른다. 엔진 의존은 parser·backend 에만. 그래서 엔진 추가 = parser 하나 더.

### 4조용히 넘어가지 않는다 (fail-loud)

모든 계약서·부품은 스키마(Pydantic)로 검증. 잘못된 값·누락은 즉시 에러 — 조용한 폴백은 없다.

> **한 문장** — 하나의 계약서(.py) → 엔진별 parser → 각 시뮬레이터. 구조는 통일하고, 남는 물리 엔진 차이는 계측한다. 새 태스크는 계약서만 쓰고, 새 엔진은 스포크만 붙인다.

MetaLab · sim/ · engine-agnostic simulation
사용법 = 10\_metalab\_tutorial · 엔진 parity = 01\_engine\_parity
