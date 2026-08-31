# 엔진 parity — newton ↔ genesis

같은 계약서로 같은 태스크를 두 엔진에서 돌릴 때 실제로 문제가 됐던 것들과, 그때 확정한 설정값.
**엔진 소스를 읽어야만 알 수 있는 것들이라 여기 남긴다** — 코드의 주석이나 기본값만 봐서는 안 보인다.

## 왜 parity 가 목표인가

MetaLab 의 전제는 *계약서 한 장 → 엔진별 스포크 → 구조가 동일한 씬*이다. 구조가 같아야 두 엔진 사이에
남는 차이가 **순수 물리 엔진 차이**가 되고, 그게 sim2real 오차 예산의 예측치가 된다. "newton 이 느리다"가
엔진 본질 차이인지 우리 셋업 버그인지 구분되지 않으면 이 전제가 무너진다.

## 확정된 설정과 그 근거

### 1. `rigid_gap = 0` — 가장 크게 물렸던 곳
`newton.ModelBuilder.rigid_gap` 기본값 **0.1(10 cm)** 이 add-time 에 모든 shape 의 `geom_gap` 으로
구워지고, mjwarp broadphase 가 서로 10 cm 이내인 모든 geom 쌍을 접촉 후보로 잡는다. 다지 손처럼 조밀한
구조에서는 거의 모든 비인접 링크쌍이 걸린다.

- 실측: 접촉 **727개/world** 중 99.9% 가 실제 침투가 아닌 near-touching. 물리 비용은 접촉 수에 **선형**이라
  3.5× 과금인데 **fidelity 이득은 0**(gap=0 에서도 침투 접촉 집합이 동일).
- `rigid_gap=0` 적용 후 @5120: 접촉 727 → **21/world**, full `env.step` 124 → **56.5 ms**.
  gap=0 은 MuJoCo 기본값이자 genesis 실질값이고, **self-collision(손가락 관통 방지)은 그대로 보존**된다.
- ⚠️ **두 경로 모두에 적용해야 한다.** native-contacts 경로에만 넣고 실제 학습이 쓰는
  `use_mujoco_contacts=True` 경로에 빠뜨리면 폭발이 그대로 남는다.

### 2. `nconmax` 는 1의 파생 — 1024 → 64
gap=0 이면 실제 grasp peak 가 **~22 접촉/world**(firm full-close worst case 실측)라 `nconmax` 를 64(≈3× 여유,
overflow 0)로 내릴 수 있다. 접촉/EPA 버퍼가 ~16× 줄어 **newton 8192 VRAM 이 ~30 → ~10 GB**가 된다.

EPA 워크스페이스 = `naccdmax × (6+5·ccd) · vec3`, `naccdmax = nconmax × nworld`. 즉 OOM 의 진짜 레버는
부풀려진 `nconmax`(→ `rigid_gap`)이지 `ccd_iterations` 가 아니다. ccd 를 35→16 으로 줄여 OOM 을 막았던 것은
**증상 치료**였다(무해해서 유지 중).

### 3. CUDA graph 는 backend 에서 직접 잡는다
newton 코어에는 graph capture 헬퍼가 없다(예제 idiom 일 뿐). 그래서 `backend.step_n(decimation)` 에서
`wp.ScopedCapture` 로 **decimation×substeps 전체를 graph 1개**로 캡처한다 — policy step 당 `capture_launch`
1회(물리 2.21×, GPU util 100%).

- **torch 쓰기는 graph 에 캡처되지 않는다.** 이벤트에서 torch 로 wrench 를 쓰면 그 플래그가 latch 돼 graph 가
  영구 비활성화된다(실측 graph_n=0). wrench 는 graph-상주 warp 커널 + 고정 버퍼로 써야 한다.
- graph 검증은 "돈다"가 아니라 **replay-only 인지**(캡처 1 / replay N-1)를 계측으로 확인한다.

### 4. contact 파라미터는 mjw 가 아니라 Newton source 에 쓴다
robot·table 은 MJCF 에 contact 블록이 없어 MuJoCo 기본 solref 0.02(≈4× soft)가 적용돼 침투한다. 그렇다고
`mjw_model` 에 직접 쓰면 **첫 friction-DR reset 에서 덮어써진다** — `SHAPE_PROPERTIES` notify 의 재sync 커널이
`model.mujoco.*`(custom attr)에서 매번 다시 쓰기 때문.

→ Newton source(`model.mujoco.solref/solimp/solmix`)에 쓰고 `solref_mode = SOLREF_MODE_RAW` 로 authored
마킹 + notify 1회 push 하면 재sync 에도 살아남는다. 예외: `eq_solref`/`eq_solimp` 는 Newton 쪽 재sync 가 없어
mjw 직접 쓰기로 생존한다.

### 5. friction cone = `pyramidal`
genesis 마찰은 4-edge pyramid 다. elliptic 을 쓰면 **오히려 엔진 간 불일치**가 된다. pyramidal =
genesis parity + MuJoCo 기본 + **elliptic 대비 2.3× 빠름**(109 vs 253 ms @4096 grasp). 오차는 대각방향
마찰 최대 −29%(허용). integrator 는 `implicitfast` 유지(euler 이득 −3%뿐, stiff PD 발산 위험).

### 6. friction 결합 = `max`, geometric-mean 이식 금지
genesis 결합 규칙은 `max(μa, μb)` 로 MuJoCo 기본과 일치한다. geometric-mean 을 이식하면 cross-engine
parity 가 깨진다 — **하지 말 것.**

### 7. jacobian = `auto`(sparse) 유지
오른팔 7 + 오른손 20 + 해머 free 6 = nv 33 이 auto 경계(nv>32 → sparse)를 1 DOF 차로 넘는다. 실측
**sparse 가 dense 보다 2.3× 빠르다** — auto 유지.

### 그 밖의 함정
- Genesis 는 massless Xform 프레임을 drop 한다(MJCF 전환 후엔 팜이 네이티브 body 라 대체로 무관).
- `use_newton_actuators=True` 면 Lab actuator 의 `compute()` 가 스킵된다 — 커스터마이즈는 action-term /
  articulation-write 레벨에서.
- native-contacts(`use_mujoco_contacts=False`)는 mesh 를 trimesh 로 등록해 삼각형-쌍 폭발(NaN)을 만든다.
  `CONVEX_MESH` retype 으로 물리는 정상화되지만 GJK 비용이 80× 라, mesh 물체가 있으면 True 를 유지한다.

## 측정 방법론

성능 문제를 다시 만났을 때 이 순서로 본다.

1. 속도·graph 계측은 반드시 **자체일관 단일 빌드**에서 한다(빌드마다 조건이 달라지면 비교가 무의미).
2. 접촉은 **`dist<0`(실제 침투)** 과 **`dist≥0`(near-touching)** 을 **분리해서** 본다. 총 접촉 수만 보면
   위 1번을 놓친다.
3. 물리 비용은 접촉 수에 선형이다 → 접촉수-physics 곡선 + margin A/B 면 진범이 드러난다.
4. graph 는 replay-only 인지를 계측으로 확인한다.

이미 배제된 가설들(다시 파지 말 것): 솔버 LS tolerance(A/B −4%), NVIDIA 드라이버, reset notify 재sync,
obs/reward 파이프라인 무게(driver 오버헤드 8%), timing 120 vs 200 Hz, `nconmax` capacity 자체(3%),
손 collision hull 재작업(원인은 hull 이 아니라 `rigid_gap`).

## 남은 과제 — sim2sim 교차평가

한 엔진에서 학습한 정책을 다른 엔진에서 평가하면 **SR 0%** 다. 엔진 간 접촉·마찰 미세차가 grasp 에 크게
작용하는 것으로 보인다. 이 격차를 좁히거나, 멀티-엔진 학습으로 robust 하게 만드는 것이 다음 과제다.

---
개요 = **00_project_overview** · 사용법 = 리포 루트 **README.md**
