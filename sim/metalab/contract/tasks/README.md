# 태스크 계약서 작성법 — `sim/metalab/contract/tasks/`

태스크 하나 = Python 파일 하나. 로봇·물체·보상·관측·종료·DR 을 **선언만** 하고(`if`/`for`/계산 없음), 로더가
`TaskSpec` → `EnvSpec` 으로 풀어 **newton · genesis 양쪽에서 같은 환경**을 만든다. 스키마가 거부하는 값은 전부
**로드 시점에 크게 실패**한다(조용한 기본값 없음).

가장 빠른 길: 비슷한 태스크를 복사해서 고친다. 참고 순서는 `rl/manipulation/franka_reach/`(가장 짧음) →
`hammer_lift_teacher/` → `hammer_lift_student/`(비대칭 actor/critic, 노이즈, DR, 커리큘럼).

## 1. 폴더와 실행 이름

```
tasks/
  rl/<group>/<task>/        학습·평가용 family.  _base.py = 공통 뼈대, <recipe>.py = 튠 값
  standalone/<group>/*.py   씬만 있는 계약서 (런치패드 Standalone 모드, 학습 없음)
  parity/*.py               newton vs genesis 비교용
  _assets.py                물체 MJCF 경로 헬퍼 object_mjcf("<이름>")
```

- 실행은 `--task <task> --recipe <recipe>` (예: `--task franka_reach --recipe position`). `-` 와 `_` 는 같다.
- `<recipe>.py` 는 `TASK = build_task(...)` 하나를 내놓고, `build_task` 는 `_base.py` 가 정의한다.
- `<group>` 은 선반일 뿐 이름에 들어가지 않는다.

## 2. 뼈대 (franka_reach 축약)

```python
from sim.metalab.contract.spec import Done, Event, Obs, Rew, TaskSpec, values
from sim.metalab.terms import action, events, gate, obs, reward, terminate

class PHYSICS:                       # hz, substeps, decimation(정책 1스텝 = 물리 decimation 스텝)
    hz = 120; substeps = 2; decimation = 2

class SCENE:                         # "무엇이 있나" — 로봇·물체·카메라·goal·접촉 파라미터
    ground = True
    class robot:
        name = "franka"              # contract/robot/<family>/<name>.yaml
        base_pos = [0.0, 0.0, 0.0]
        init_pose = {"panda0_joint2": 20.0, ...}   # 유일하게 DEG 로 쓰는 곳
    class goal:
        pos = [0.5, 0.0, 0.4]; goal_dist_tol = 0.02

class ACTION:                        # 그룹 이름 = 로봇 yaml 의 action_groups
    arm = action.JointDeltaPosition(scale=0.5, ema_tau=0.1)

class OBS:
    joint_pos = Obs(obs.joint_positions, names="@joints.arm")
    goal_pos  = Obs(obs.goal_position)

class REWARD:
    reach = Rew(reward.body_goal_proximity, weight=1.0, body="@frames.palm", std=0.2)

class EVENTS:
    goal = Event(events.sample_goal_position, "interval", x_range=[0.35, 0.65], y_range=[-0.25, 0.25],
                 z_range=[0.2, 0.6], interval_range_s=[3.0, 5.0])

class GATE:                          # val/SR 이 재는 "성공" 정의
    predicate = gate.body_at_goal; goal_dist_tol = 0.02; hold_steps = 10

class TERMINATE:
    time_out = Done(terminate.time_out, time_out=True)   # MDP 계약서는 필수

TASK = TaskSpec(name="franka_reach_position", physics=PHYSICS, scene=values(SCENE), action=ACTION, obs=OBS,
                obs_groups={"actor": "all", "privileged": "all"}, reward=REWARD, events=EVENTS,
                gate=GATE, terminate=TERMINATE, episode_length_s=15.0)
```

## 3. 규칙 5개

1. **파일 하나 → `TASK` 하나.** 선언만. 계산이 꼭 필요하면 `build_task() -> TaskSpec` 함수로.
2. **`fn` 은 import 한 심볼**(문자열 아님). 카테고리 패키지에서 가져온다: `terms.obs / reward / terminate / events /
   curriculum / gate / action`. 오타는 ImportError 로 즉시 드러난다.
3. **`@`로 시작하는 문자열만 로더가 해석**한다(아래 표). 나머지는 전부 리터럴.
4. **단위는 SI + rad, 쿼터니언 wxyz, z-up.** 예외는 `robot.init_pose` 와 `range_deg`/`rot_deg` 처럼 이름에 deg 가 붙은 값.
5. **term 의 knob 은 이름을 붙여** 넘긴다(positional 금지). 로더가 함수 시그니처와 대조해 오타·누락을 로드 시 잡는다.

## 4. `@refs` — 로봇 yaml 에서 가져오는 값

| 문자열 | 뜻 |
|---|---|
| `@joints.<action group>` | 그 그룹의 관절 목록 |
| `@joints.<joint_group>` | 관측만 하고 명령하지 않는 관절 그룹 |
| `@joints.ctrl` | 명령하는 모든 관절, ACTION 순서 |
| `@frames.<key>` | body 하나 (`@frames.palm`, `@frames.chest_origin`) |
| `@bodies.<key>` | body 목록 (`@bodies.fingertips`, `@bodies.nail`) |

## 5. 블록별 요약

| 블록 | 핵심 |
|---|---|
| `physics` | `hz`(필수), `substeps`, `decimation`, `gravity`, `self_collision`. 엔진별 솔버 knob 은 `overrides` 로. |
| `scene.robot` | `name`(필수), `base_pos`(필수), `base_quat`, `fixed_base`, `init_pose`(deg). |
| `scene.objects` | 리스트. `{name, mass, asset:{mjcf:[…]} 또는 parts:[{shape,size,pos,quat}], fixed, variants, init_pos, init_rpy}`. `mjcf` 를 여러 개 주면 env i 가 i % variants 번째를 받는다. 첫 번째 non-fixed 물체가 "the object". |
| `scene.contact_params` | `{robot | <object> | <fixture>: {solref, solimp, solmix}}`. `solmix` 는 newton 전용. |
| `scene.camera` | `{eye, lookat, fov}` — 평가 녹화·뷰어 프레이밍. |
| `scene.goal` | `pos`(필수), `quat`, `keypoint_half_extent`, `goal_dist_tol`. goal 이 있으면 `gate` 도 필수. |
| `action` | 그룹별 매핑 클래스: `JointDeltaPosition(scale, ema_tau)` = spawn 기준 오프셋, `JointPositionToLimits(scale, range_deg)` = 관절 범위에 선형 매핑, `TaskDeltaPose(pos_m, rot_deg)` = spawn 기준 박스. `joints=` 로 부분집합 지정. 블록에 `min_delay/max_delay` 를 쓰면 명령 지연(스텝). |
| `gate` | `predicate` + 그 predicate 가 받는 bar 들(`goal_dist_tol`, `contact_count`, `contact_fingers`, `palm_distance`, `joint_pose_tolerance` …), `hold_steps`, `hold_mode`(consecutive/cumulative). 커리큘럼은 여기까지 올라온다. |
| `obs_groups` | `{group: "all" \| [이름…] \| Obs 블록}`. `obs_history_length={group: H}` 프레임 스택, `obs_noise_groups=[group…]` 노이즈 적용 그룹. |
| `terminate` | `Done(fn, time_out=True)` 는 truncation(부트스트랩). reward 가 있는 계약서는 `time_out` term 이 하나 있어야 한다. |
| `overrides` | `{"newton": {...}, "genesis": {...}}`. 가장 풀 세트는 `hammer_lift_student/_base.py`. |

PPO 하이퍼·네트워크·wandb 는 여기 없다 → `learning/rl/<experiment>/<task>/experiment.py`.

## 6. term 엔트리

```python
Obs(fn, scale=1.0, noise=ObsNoise(std=… | pos=…, rot=…), unit="", **knobs)
Rew(fn, weight=…, **knobs)              # weight 필수. term 은 물리량/[0,1]/0-1 이벤트만 반환하고 크기는 weight 가 정한다
Done(fn, time_out=False, **knobs)
Event(fn, "reset" | "interval", train_only=False, requires="<capability>", **knobs)
Curr(fn, **ctor_knobs)
```

- 클래스 블록에 쓰면 **속성 이름이 term 이름**(로그 키 `Reward/<name>`, `Termination/<name>`). 리스트로 쓰면 `name=`.
- `requires="object_scale"` 같은 능력을 엔진이 못 주면 그 이벤트만 빠지고 나머지는 돈다(로그에 찍힌다).
- `train_only=True` 이벤트는 평가·녹화에서 빠진다.

## 7. 쓸 수 있는 term (각 패키지 `__all__` 이 정본)

- **obs**: joint_positions · joint_velocities · joint_accelerations · joint_torque_obs · joint_pd_torque_obs ·
  joint_gravcomp_torque_obs · joint_state · prev_action_targets · last_action · action_delay · object_pose ·
  object_seen_pose_world · object_state_world · object_linear_velocity · object_angular_velocity · object_keypoints ·
  object_variant · body_pose_in_chest · palm_pose_in_chest · body_linear_velocity · body_angular_velocity ·
  goal_position · body_goal_error · goal_keypoints · goal_dist_error · palm_distance_error · joint_pose_error ·
  body_contact_flags · fingertip_contact_steps · hand_contact_force · hand_object_force_magnitude ·
  fingertip_penetration_depth · fingertip_relative_pos · fingertip_relative_pose · fingertip_relative_vel ·
  closest_keypoint_max_dist · object_goal_keypoint_success · object_lifed · episode_step · instantaneous_reward ·
  curriculum_state · curriculum_hold_progress · dr_params
- **reward**: body_goal_proximity · palm_object_proximity · fingertip_object_proximity · lifting_reward ·
  object_goal_keypoint_progress · object_goal_keypoint_tracking · object_goal_reach_bonus · joint_pose_convergence ·
  fingertip_object_contact · fingertip_object_pinch_contact · nail_object_contact · joint_vel_l1 ·
  joint_torque_penalty · action_rate_l2
- **terminate**: time_out · object_below_height · object_far_from_body · object_velocity_exceeded ·
  table_fingertip_contact_force_exceeded · body_contact_detected · curriculum_passed
- **events**: reset_object_pose · reset_joints_by_offset · sample_goal_position · set_shape_friction ·
  randomize_rigid_body_mass · randomize_object_scale(newton 전용, `requires="object_scale"`) ·
  randomize_fixed_base_root_height · record_object_spawn_z · apply_object_external_force ·
  apply_object_external_force_when_lifted
- **gate predicate**: object_at_goal · body_at_goal
- **curriculum**: hammer_lift_success_curriculum
- **action**: JointDeltaPosition · JointPositionToLimits · TaskDeltaPose

## 8. 없는 term 이 필요할 때

- `terms/<category>/common.py`(공용) 또는 `terms/<category>/<task>.py`(태스크 전용) 에 평평한 함수를 쓴다:
  `def <name>(env, <knob>=…) -> (N,)` / `(N, d)` / `(N,) bool`, 이벤트는 `def <name>(env, env_ids, <knob>=…)`.
  `env` 로 백엔드 읽기(`env.joint_pos(...)`, `env.object_pos()`)와 좌표 수학(`sim/metalab/api`) 만 쓴다. 엔진 import 금지.
- 에피소드 상태는 `env.buffer(key, shape, fill, dtype)` 로 갖는다. 리셋된 env 는 드라이버가 `fill` 로 되돌린다.
  같은 term 이 여러 그룹에 있어도 상태를 한 스텝에 한 번만 진행시키려면 `runtime/episode.step_edge(env)` 를 쓴다.
- 여러 term 이 공유하는 판정(성공 거리 등)은 `terms/gate/common.py` 에 한 번 두고 import 한다.
- 새 함수는 카테고리 `__init__.py` 의 import 와 `__all__` 에 추가한다.
- 새 로봇: `contract/robot/<family>/<name>.yaml` + MJCF 를 `sim/metalab/assets/robots/` 에. 물체는 yaml 없이
  `scene.objects` 에 인라인.

## 9. 확인

```bash
learning/scripts/local/metalab_train.sh --sim newton --task <task> --recipe <recipe> --num_envs 64 --max_iterations 3 --no_wandb
```

로드 실패는 첫 줄에 이유가 나온다(알 수 없는 필드, `@ref` 오타, knob 이름 불일치, goal 없는 gate 등).
