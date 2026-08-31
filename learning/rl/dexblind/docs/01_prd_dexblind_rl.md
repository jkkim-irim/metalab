# PRD — Dexblind Hammer-Lift RL

> 무엇을 학습시키는가에 대한 요구사항/설계. 구체 수치는 `learning/rl/dexblind/hammer_lift/experiment.py`
> (실험 상수) · `sim/isaaclab/envs/hammer_lift/simulation.py`(물리)에 있으며 자주 튜닝되므로 여기서는 **메커니즘**만 기술한다.

## 1. 목표
ALLEX 오른손이 테이블 위 망치를 **안정적으로 파지**하여 **목표 위치·자세로 들어올리고 유지**하도록
학습한다. 단순 "들어올림"이 아니라 *제대로 된 grasp* (여러 손가락 접촉 + 손바닥 근접 + 자세 정렬)를 요구한다.

## 2. 범위 / 비범위
- **범위**: Isaac Lab + Newton 시뮬레이션 학습. 도메인 랜덤화로 강건성 확보. 두 env 변형(full-body / dense).
- **비범위(현재)**: 실로봇 배포(sim2real)는 후속. 망치 외 오브젝트, 양손 협조는 비범위.

## 3. 관측 / 액션
- **Actor obs** (`ObservationsCfg.ActorObsCfg`, history_length=3): last_action, joint_pos(노이즈),
  hand_joint_torque(노이즈), right_hand_relative_pos(노이즈). play 모드는 노이즈 off.
- **Critic obs** (asymmetric, privileged): 위 + joint_vel/acc, object lin/ang vel, palm vel,
  fingertip-hammer contact, hammer relative pose, grasp 상태 등 특권 정보.
- **Action**: arm + hand **EMA joint-position-to-limits** (`EMAJointPositionToLimitsAction`).
  scale·EMA α 는 `learning.py`(ARM/HAND_ACTION_SCALE, *_EMA_ALPHA). EMA 가 적용 모션을 평활화.

## 4. 성공 정의 (`task_success` — reward + termination)
한 RL step에서 아래 **모든 순간 조건**이 충족되고, 이를 `success_hold_steps` **연속** step 유지하면
1회성(one-shot) 성공으로 bonus 지급 + 에피소드 종료:

1. 망치에 접촉한 fingertip 수 **≥ `contact_count`** (`flags.sum >= contact_count`)
2. 망치 높이 **≥ `lift_threshold`**
3. ‖hammer_pos(env-local) − `GOAL_POS`‖ **< `pos_threshold`**
4. quat 각거리(hammer, `GOAL_ROT`) = `2·acos(|⟨q,q_goal⟩|)` **< `rot_threshold`**
5. ‖palm − hammer grasp point‖ **< `palm_threshold`**

(값: `learning.py` TASK_SUCCESS_*. fingertip = Index/Middle/Ring/Little/Thumb 5개.)

## 5. 자동 커리큘럼 (success-rate 구동, `mdp/curriculums.py`)
단일 난이도 `level`. 최근 `task_success` 종료율이 `CURRICULUM_SUCCESS_THRESHOLD`를 넘으면
(`CURRICULUM_EVAL_INTERVAL_ITERATIONS` PPO iteration 마다 최대 1회 평가) level +1. level 이 오르면
**관대 → 엄격**으로 lockstep 진행 (각자 자기 end 값에서 clamp):

- `pos_threshold` ↓, `rot_threshold` ↓, `palm_threshold` ↓ (허용 오차 축소)
- `contact_count` ↑ (요구 접촉 손가락 수, → full grasp)
- `success_hold_steps` ↑ (유지 시간)
- 들린 해머 외력(`hammer_force_when_lifted`) ↓ (외란 강화)
- shaping 보상 weight(`fingertip_hammer_contact_reward`, `palm_hammer_proximity`) ↓ → 후반엔 sparse `task_success`가 주도

> noisy success-rate로 인한 폭주를 막으려 **iteration 단위**로 게이트한다. env는 iteration을 직접 모르므로
> `common_step_counter // num_steps_per_env`로 환산. (시작/끝/step 값 전부 `learning.py` 참조 → 주석 drift 방지)

## 6. 보상 (`mdp/rewards.py`, `RewardsCfg`)
- dense shaping: `fingertip_hammer_contact_reward`(접촉 손가락 비율), `palm_hammer_proximity`(palm↔grasp 근접, exp(−d/std)). 둘 다 커리큘럼이 weight를 줄임.
- penalty: `table_finger_penetration_penalty`(테이블↔손끝 침투 mm), `arm_joint_torque_penalty`.
- sparse: `task_success`(§4 충족 시 bonus).
- palm proximity는 **하이브리드** — 동일 거리(`_palm_grasp_distance`)를 dense 보상 + 성공 조건 양쪽에 공유.

## 7. 종료 (`mdp/terminations.py`, `TerminationsCfg`)
`time_out`, `task_success`(성공), `nan_detected`(NaN world → clean reset),
`hammer_dropped`(최소 높이 미만), `table_fingertip_contact_force_exceeded`(테이블 과접촉),
`hammer_velocity_exceeded`, `hammer_too_far_from_table`.

## 8. 도메인 랜덤화 (`EventCfg`, reset/interval)
해머 spawn pose(x/y/yaw), 해머 mass scale, 해머/로봇/테이블 마찰, arm/hand reset offset,
로봇 root 높이, 들린 해머 외력(interval). 범위: `learning.py` *_RANGE.

## 9. 학습 설정
- PPO (rsl_rl): LSTM actor + MLP asymmetric critic, obs normalization, adaptive LR(desired_kl).
- 물리 200Hz × decimation 4 → 50Hz policy. Newton CUDA graph(병목 회피).
- NaN-safe wrapper: NaN obs/reward를 on-GPU sanitize + 해당 env reset → 학습 무중단(`training/nan_safe_env_wrapper.py`).
- env 수: `scene.py` NUM_ENVS.

## 10. 검증
- 로직/설정 테스트: `python -m pytest learning/rl/tests` (Isaac env 필요, **CPU·GPU 불필요**). 커리큘럼 게이팅·성공 게이트·config 불변식·env-cfg 조립 검증.
- **테스트 통과 ≠ sim 동작 보증.** end-to-end 정상성은 GPU 학습/play 스모크로 별도 확인.

## 11. 미해결 / 후속
- sim2real 파이프라인.
- MDP silent fallback(`if x is None: return zeros`)을 fail-loud로 전환 + 테스트 (현재 동작 유지 결정).
- 전체 `learning/rl/` ruff 정리(레거시) + Isaac 가능 러너 기반 자동 CI.
