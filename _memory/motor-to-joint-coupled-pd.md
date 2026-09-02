---
name: motor-to-joint-coupled-pd
description: "control_mode:motor 모터공간 coupled PD — rated_torque ≠ torque_limit, k_pos/k_vel 은 모터 레벨 N·m/rad"
type: project
---

`sim/metalab/control/`(커널·로더) + `sim/metalab/contract/robot/allex/`(맵·게인 데이터) — 실HW 모터공간 coupled PD, 양 엔진, 16그룹/44관절.

- 구조: `loaders.py`(numpy-only) / `motor_coupling.py`(newton warp, substep-rate) / `coupled_pd_torch.py`(genesis 미러). parity 는 **단일 numpy 오라클**로 고정(warp≡torch≡oracle). genesis 는 warp fast path(`_GenesisWarpOwner`, torch 스트림 정렬 필수)로 커플링 40→2.3ms.
- fold: `τ_m = k_φΔφ − k_dφ̇ + G⁻ᵀτ_g → clamp → τ_q = Gᵀτ_m`(arm-slice 만). readout = τ_q(중력 포함 = 모터전류 의미) → **obs 분포 변화, 재학습 필요**.
- ⚠️ genesis 에서 coupled PD 를 control-rate 로 돌리면 저관성 축 limit-cycle(29 rad/s) → **substep-rate 필수**(scene 을 dt/substeps 로 빌드).
- ⚠️ **토크 2개**: `rated_torque`(datasheet 참고) ≠ `torque_limit_pos/neg_abs`(clamp 실사용, ~3×rated). `_clamp_env` = stall droop ∩ ±torque_limit.
- ⚠️ **게인 단위**: `robot_model.json` 의 `k_pos`/`k_vel` = **모터 레벨 N·m/rad**(JSON 에 단위 필드 없음, 2026-08-10 확인). deg 환산 제시 금지(57.3×로 대조 깨짐).
- f64 pow 는 컨슈머 GPU 1/64 처리량 → 거듭제곱 테이블+gather.
- 남은 것: waist/neck 모터레벨(cpp 는 allex 리포), elbow·shoulder ratio 는 allex `ref_m2j.py` 에 확보.

관련: [[gravcomp-both-engines]], [[engine-api-semantic-traps]]
