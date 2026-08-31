# MetaLab — 하나의 계약서, 여러 시뮬레이터

**MetaLab**(`sim/metalab/`)은 로봇 학습 환경을 **엔진과 무관하게 한 번만 정의**하고, 그 정의로
Genesis·Newton 등 여러 물리 엔진에서 **같은 환경**을 만들어내는 시뮬레이션 계층이다.

## 왜 필요한가

**문제.** 같은 태스크를 엔진마다 따로 세팅하면 로봇 배치·질량·관절·보상이 조금씩 어긋난다. 그러면
"Genesis 에선 되는데 Newton 에선 안 되는" 이유가 *환경 차이인지 물리 엔진 차이인지* 구분할 수 없고,
sim2real 로 갈 때 오차가 어디서 왔는지 짚지 못한다.

**해법.** 환경을 *엔진-무관 계약서(contract)* 하나로 정의하고, 엔진별 *스포크*가 그것을 읽어 각자의 씬을
만든다. 셋업이 같으므로 두 엔진에 남는 차이는 **순수 물리 엔진 차이**뿐 — 측정 가능한 값이 된다.

## Hub & Spoke

```
계약서 (contract/) ──┬─► genesis 스포크 → gs.Scene    ─┐
     엔진 무관       └─► newton  스포크 → newton.Model ─┴─► 구조가 동일한 환경
```

계약서 1장 → 스포크 N개 → 시뮬레이터 N개. 구조는 통일하고, 물리 엔진 차이만 남긴다.

## 세 개의 층

| 층 | 어디 | 하는 일 |
|---|---|---|
| **HUB — 무엇을** | `contract/`, `terms/`, `api/` | 환경이 무엇인지를 **엔진 모르게** 정의한다. 태스크는 선언적 계약서(`.py`), 로봇은 `contract/robot/*.yaml`, 보상·관측·종료·이벤트·커리큘럼 계산식은 `terms/`, 그 아래 공용 primitive 는 `api/`. **이 층은 어떤 엔진도 import 하지 않는다.** |
| **SPOKE — 어떻게** | `backends/{genesis,newton}/` | 계약서를 그 엔진의 씬으로 번역한다. `parser` 가 씬을 만들고 `backend` 가 공통 인터페이스(read·control·step·reset)를 구현한다. **엔진 의존은 오직 이 층에만.** |
| **RUNTIME — 돌린다** | `runtime/` | 엔진-무관 실행 층. `env_driver` 가 backend 를 학습용 VecEnv 로 감싸고, `parity_rollout` 이 두 엔진을 같은 계약서로 굴려 궤적을 대조한다. |

프로세스 경계(RPC sim-service)는 `sim/service/` — 엔진 무관 공용 계층이다. 자세한 것은
**21_rpc_restore_plan**.

## 디렉터리

```
sim/metalab/
├─ contract/          HUB — 계약 (엔진 무관, 단일 출처)
│  ├─ tasks/<task>/   태스크 family. _base.py = 공유 코어, <task>_<recipe>.py = 튜너블 트리오
│  ├─ robot/          로봇 계약 (.yaml — 마스크·액션·물리 오버라이드)
│  ├─ spec.py         Pydantic 스키마 (TaskSpec·RobotSpec·PhysicsSpec …)
│  ├─ loader.py       계약서 → EnvSpec
│  └─ asset_path.py   계약서의 asset 경로 → 실제 파일
├─ terms/             보상·관측·종료·이벤트·커리큘럼·gate 의 계산식
├─ api/               엔진-무관 primitive (frames·keypoints·shaping·state·kinematics·contact)
├─ assets/            로봇·물체 MJCF 와 궤적 데이터 (전부 리포 안)
├─ runtime/           공유 런타임 — env_driver(VecEnv) · backend 인터페이스 · parity_rollout · 녹화/리포트
├─ setup/<engine>/    엔진별 uv 프로젝트 (pyproject.toml + uv.lock)
├─ backends/genesis/  SPOKE — parser · backend · server
└─ backends/newton/   SPOKE — parser · backend · server
```

## 태스크 하나가 학습이 되기까지

```
학습 : 계약서 → loader(EnvSpec) → 스포크 parser(엔진 씬) → backend → env_driver·VecEnv → PPO
검증 : 계약서 → genesis·newton 롤아웃 ×2 → 궤적 대조 → 물리차 리포트
```

계약서는 **값과 심볼을 조합만** 한다 — 로직은 담지 않는다. 태스크 family 는 모두가 공유하는 `_base.py`
(obs·scene·robot·sim·action·events·termination)와, 레시피마다 다른 트리오(`reward`/`gate`/`curriculum`)로
나뉜다:

```python
# contract/tasks/hammer_lift_teacher/hammer_lift_teacher_privileged.py — 레시피
from sim.metalab.contract.spec import Curr, Rew
from sim.metalab.terms import curriculum, gate, reward

class REWARD:
    lifting_reward                = Rew(reward.lifting_reward, weight=20.0, lift_height=0.1)
    object_goal_keypoint_progress = Rew(reward.object_goal_keypoint_progress, weight=378.0)

class GATE:                       # 무엇을 성공으로 볼 것인가
    predicate     = gate.object_at_goal
    goal_dist_tol = 0.01          # [m]
    hold_steps    = 20            # 연속 유지 스텝
    contact_count = 5             # 5지 파지
```

작성 규칙은 `contract/tasks/README.md` 에 있다.

## 기억할 규칙 4가지

1. **로봇 = YAML, 태스크·로직 = .py.** 계약서는 값과 심볼을 조합만 하고 로직을 담지 않는다.
2. **MJCF = 물리 진실 소스.** 게인·질량·관절 한계는 MJCF 하나에. 계약서는 그 위의 얇은 오버라이드다.
3. **계약·런타임은 엔진을 모른다.** 엔진 의존은 parser·backend 에만 — 그래서 엔진 추가 = 스포크 하나 더.
4. **조용히 넘어가지 않는다.** 모든 계약은 Pydantic 으로 검증하고, 잘못된 값·누락은 즉시 에러를 낸다.

또 하나: **태스크 노브는 전부 sim 쪽 계약서에 있다.** 보상 weight·임계값·액션 스케일·DR 범위·커리큘럼이
전부 계약서에 인라인으로 적히고, 트레이너의 `experiment.py` 는 RL 설정만 갖는다.

---
사용법 = 리포 루트 **README.md** · 엔진 parity = **01_engine_parity** · 런치패드 = **20_launchpad_prd**
