---
name: gravcomp-both-engines
description: "중력보상 양 엔진 구현 — actuator/passive 구분, obs folding, COM Jacobian 필수"
type: project
---

- 계약: `RobotSpec.gravcomp`(`actuator_joints`/`passive_joints`, YAML 소유). **actuator(팔)** = g(q)를 `joint_torque` obs 에 fold(실 토크센서가 보고하므로). **passive(pitch)** = 기계 스프링, obs 미포함. 손가락 무보상.
- newton: `body_gravcomp=1` + Newton source `model.mujoco.gravcomp`(re-sync clobber 우회) + 캡처 전 설정. obs = `qfrc_gravcomp` fold.
- genesis: per-joint native 없음 → child link 에 반중력 외력 `m·(-g)@COM` 매 step. obs = `get_jacobian(local_point=inertial_pos)`.
- ⚠️ **Jacobian 은 반드시 COM 기준** — origin 은 3× 과소보고.
- **`passive_bodies`**: passive pitch **위의** 질량은 관절 단위 보상에서 빠져 float 시 Neck 43° 처짐 → leaf link 리스트로 추가 보상(43.6°→0.3°). Waist 4-bar equality 저항을 gravity 로 오해 말 것.
- Torque(float) 모드의 PD 중립화 = **러너에서 target=현재 pose 매 스텝**(엔진 런타임 PD-off 는 양 경로 다 안 먹힘). kd 는 남음("damped float"). 끄기 `METALAB_GRAVCOMP=0`.
- settle: 렌더 끄고 물리만 + adaptive 조기종료. `METALAB_SETTLE_S=0.4`, 최초실행 `METALAB_SETTLE_FIRST_S=2.0`(cold 솔버).
- ⚠️ 물리·obs 계약 변경 → **기존 policy 무효**. genesis Jacobian×14/step 은 대규모 env 부하 후보.

관련: [[motor-to-joint-coupled-pd]], [[engine-api-semantic-traps]]
