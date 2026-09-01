---
name: genesis-spawn-velocity-kick
description: "genesis 만 spawn 첫 스텝에 속도 임펄스(~17 rad/s), 안 가라앉음 — 미해결"
type: project
---

접촉-free 씬, 초기상태 엔진 간 비트 동일인데 genesis 만 hold 첫 스텝에 **~17 rad/s** 튀고 step49 에도
1~2.5 잔존(newton 0.003). 접촉 있는 태스크에서도 동일 방향(0.139 vs 0.017).

- 배제: 초기 q/v, armature, coupled PD 첫 launch, 중력보상, 모터 커플링(`METALAB_MOTOR_COUPLING=0` 재현).
- ⇒ genesis 가 첫 substep 에 지시 없는 토크(~2.4 N·m) 주입. 후보: genesis 자체 equality spawn 해소
  (`eq_solref/eq_solimp` 가 newton 블록에만 있음), reset 경로.
- 재현: `parity_rollout holddump --engine genesis --task motor-parity --n_steps 50`.
- ⚠️ genesis 학습은 **매 리셋마다 이 임펄스를 맞는다**. 미해결.

관련: [[motor-to-joint-coupled-pd]], [[engine-parity-settings]]
