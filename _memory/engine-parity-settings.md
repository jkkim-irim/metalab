---
name: engine-parity-settings
description: "newton↔genesis parity 확정 설정 — rigid_gap=0(양 경로), nconmax 64, CUDA graph, solref 는 Newton source, 배제 가설 목록"
type: reference
---

- **`rigid_gap=0`** — 기본 0.1(10cm)이 접촉 727/world(99.9% near-touching, 비용 선형 3.5× 과금) 유발. gap=0: 21/world, env.step 124→56.5ms, self-collision 보존. ⚠️ `use_mujoco_contacts` **양 경로 모두** 적용(한쪽만 넣고 당한 적 있음).
- **`nconmax` 1024→64** — gap=0 이면 grasp peak ~22/world. VRAM 30→10GB @8192. OOM 레버는 nconmax(→rigid_gap)지 `ccd_iterations` 아님.
- **CUDA graph = backend 가 직접 캡처**(코어에 헬퍼 없음). `step_n(decimation)` 전체 1 graph → 물리 2.21×, util 100%. ⚠️ torch 쓰기는 캡처 불가 — 이벤트 wrench 가 latch 로 graph 영구 비활성(실학습이 graph 안 쓰던 버그). wrench 는 graph-상주 warp 커널+고정 버퍼. **검증 = EnvDriver 경유 + replay-only 계측.**
- **contact 파라미터는 Newton source**(`model.mujoco.*` + `SOLREF_MODE_RAW` + notify 1회) — mjw 직접 쓰기는 첫 DR reset 이 덮음. 예외: `eq_*` 는 mjw 직접 쓰기로 생존.
- **friction cone = pyramidal**(genesis parity + MuJoCo 기본 + 2.3× 빠름). integrator `implicitfast` 유지.
- 마찰 결합 = 양 엔진 동시 패치 → [[engine-api-semantic-traps]].
- **jacobian auto(sparse)** — nv 33, sparse 가 2.3× 빠름.
- 기타: genesis massless Xform drop / `use_newton_actuators=True` 면 Lab actuator `compute()` 스킵 / native-contacts + mesh = trimesh NaN(retype 시 GJK 80×) → mesh 물체 있으면 True 유지.

**접촉 강성 격차(newton 물렁)**: 진범 = mjwarp face-face 접촉의 **1점 퇴화(영구 미복구)** ← init 에서 follower 미시딩 → eq 스윕이 물체를 스침. genesis 는 멀티포인트로 즉시 회복. 픽스: `use_mujoco_contacts:false`(mesh 함정 주의) / follower polycoef 시딩 / parity 레시피(solref 0.01·solimp 0.9/0.95·solmix 1). 동일 관통 접촉력 newton 5.88 vs genesis 12.98 N(2.2×)은 미해결. 손가락 정상상태 차이의 진범은 armature 주입.

**성능 조사 방법론**: ① 단일 빌드에서 계측 ② 접촉은 dist<0 vs ≥0 분리 ③ 접촉수-physics 선형곡선 + margin A/B ④ graph 는 replay-only 계측.
**배제된 가설(다시 파지 말 것)**: LS tolerance(−4%) · 드라이버 · reset notify · obs/reward(8%) · timing · nconmax capacity(3%) · 손 collision hull.

**미해결**: sim2sim 교차평가 **SR 0%** — 접촉·마찰 미세차 추정. 격차 축소 or 멀티-엔진 학습이 다음 과제.

관련: [[engine-api-semantic-traps]], [[gravcomp-both-engines]]
