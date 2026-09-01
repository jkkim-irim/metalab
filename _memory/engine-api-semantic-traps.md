---
name: engine-api-semantic-traps
description: "엔진별 \"같은 이름, 다른 의미\" 실측 함정 — 새 엔진·새 에셋 1순위 체크리스트"
type: reference
---

## 같은 이름, 다른 물리량

- genesis 는 armature 미기재 관절에 **0.1 을 조용히 주입**(타 엔진 0) → `MJCF(default_armature=None)` 필수. 증상: 손가락 plant ~16×, 토크 1.7~5.8×.
- MJCF `<compiler angle>` 기본값 = **degree**. 에셋마다 `angle="radian"` 명시 — 빠지면 57.3× 오차, 에러 없음.
- 선속도 기준점: newton=**COM**, genesis 기본=**링크 원점**(차이 ω×r_com) → genesis `ref="link_com"`. 각속도는 무관. 단일 free body 는 동일.
- MuJoCo mesh 충돌 = convex hull(오목부 메워짐) → CoACD 다분할, 비용 증가 없음(실측).
- `<mesh scale>`/`texturedir` 상대경로는 /tmp 사본 파싱에서 깨짐 → 절대경로 고정.
- mjwarp **μ≤1e-6 = 첫 접촉 NaN**(0·1e-6 NaN / 1e-3↑ 정상). ⚠️ 현행 `runtime/physics/friction.py:32` `FRICTION_EPS=1e-6` 이 정확히 발산점 — 모순 미해소. 파라미터 스윕은 값마다 새 프로세스(NaN 오염).
- 마찰 결합 순정 = `max(μa,μb)`. 우리는 **양 엔진 기하평균 패치**(`backends/*/friction.py`) — 바꿀 땐 반드시 양쪽 동시.

## 호출은 성공, 효과는 없음

- genesis `gs.init` 은 device index **무시** → 2-GPU CUDA-IPC 불가, 단일 GPU 고정.
- newton ViewerGL ImGui = ASCII 전용 → 뷰어 UI 문자열은 영문.
- genesis 버퍼는 **genesis DSL 로만 write**(`to_torch()` 는 호스트 복사) → 수식을 DSL 로 한 번 더 쓴다.
- mjwarp 반영 경로 이원화: `solref/solimp` = 텐서 직접 쓰기, `mu/gap` = notify 필요.
- genesis "build-time 속성" 두 부류: `entities_info` 류 런타임 필드(setter 없어도 써면 먹음 — gravcomp 등) vs 진짜 baked(geometry scale). 판정 = `*_info` 필드 grep + 커널이 매 스텝 읽는지.
- newton 브리지는 **colliding shape 만** mj geom 화 → visual-only body 는 ngeom 0. geom 세는 코드 전부에 "0개 가능" 방어.
- newton 네이티브 broadphase 는 build 때 구운 AABB 를 읽음 → **런타임 확대 시 접촉 조용히 소실**(1.2×에서 정착 0/32; 축소는 무해). 픽스 = `shape_collision_aabb_lower/upper` 도 ×s. 교훈: 형상 DR 은 narrow/broadphase 가 같은 배열을 읽는지 먼저 확인.

## newton 구조 제약 (회피만 가능)

- env 간 **topology 동일 강제** → 형상 DR 불가, 질량·마찰 랜덤화로 대체.
- variant 리스트로도 불가 — mesh registry 는 template world 만 컴파일, 전 world 가 world-0 형상 사본(실측). variant 의 실효 = **관성 다양성뿐**. genesis 는 진짜 heterogeneous.
- genesis 형상 DR 유일 경로 = pre-scaled MJCF k개 + round-robin.
- 한쪽 엔진만 되는 기능 = **capability 게이트**(`Event(requires=...)`), 하향 평준화 금지.

parity 확인(실측): frictionloss·중력·질량·breakaway 마찰·root height·케이던스·equality(≤0.05°)·패시브 평형·접촉 규약.
미해결: `apply_object_force` 임펄스 ≠ F·dt/m, 엔진별 상이(newton 1.71× / genesis 1.40×).

관련: [[engine-parity-settings]], [[motor-to-joint-coupled-pd]]
