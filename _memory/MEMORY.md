# Memory Index

<!-- 각 줄 = "언제 이 파일을 열어야 하는가". 축적 규약 = 루트 CLAUDE.md "지식 축적". -->

## ⚙️ 엔진 물리·API (Newton / Genesis)

- [엔진 API 의미 함정](engine-api-semantic-traps.md) — 새 엔진·새 에셋 1순위 체크리스트. armature 주입 / angle=degree / 선속도 COM vs 원점 / 형상 DR 불가 / broadphase AABB / μ≤1e-6 NaN.
- [parity 확정 설정](engine-parity-settings.md) — newton 이 느리거나 물렁하거나 두 엔진 값이 다를 때. rigid_gap=0(양 경로) · nconmax 64 · CUDA graph · 배제 가설 · sim2sim SR 0%.
- [모터-관절 coupled PD](motor-to-joint-coupled-pd.md) — `control_mode:motor`. rated_torque ≠ torque_limit, 게인은 모터 레벨 N·m/rad.
- [중력보상 양 엔진](gravcomp-both-engines.md) — actuator/passive 구분, obs Jacobian 은 COM 기준.
- [genesis spawn 속도 킥](genesis-spawn-velocity-kick.md) — genesis 만 초기 속도가 튈 때. 미해결.
