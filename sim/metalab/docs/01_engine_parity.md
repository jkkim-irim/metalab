*MetaLab · Engine Parity — 인수인계 · 2026-07-12 · jkkim*

# 엔진 parity — newton ↔ genesis 디버깅 기록

같은 MJCF 계약서로 hammer-lift를 두 엔진에서 **동일 품질로 재현**하기까지 밟은 함정과 그 해법.
목적은 인수인계 — *같은 실수를 두 번 하지 않기 위해*, 무엇이 진범이었고 무엇이 곁가지 오진이었는지 근거와 함께 남긴다.

- 엔진: newton (mujoco\_warp) · genesis
- 계약: 단일 MJCF + hammer\_lift.yaml
- 기준선: legacy isaaclab-newton

## §0 · 결론 요약

2 / 2

hammer-lift 엔진 재현 (newton · genesis)

124 → 56.5 ms

newton full env.step (legacy 74보다 빠름)

~30 → ~10 GB

newton 8192 VRAM

SR 0%

sim2sim 교차평가 (= future work)

**한 문장.** newton이 genesis·legacy 대비 2배 이상 느리고 VRAM을 2배 쓰던 근본 원인은 화려한 곳(솔버·드라이버·PPO)이 아니라 *단 하나의 기본값 — `ModelBuilder.rigid_gap = 0.1 (10 cm)`*이 접촉 수를 33배 부풀린 것이었다. 이걸 0으로 고치자 물리·메모리가 동시에 풀렸고, self-collision(손가락 관통 방지)은 그대로 보존됐다. 아래 §2가 그 진범, §3이 헛다리 짚은 오진들과 그때 배운 **측정 방법론**이다.

## §1 · 왜 parity가 목표였나

MetaLab의 전제는 **계약서 한 장(MJCF + YAML) → 엔진별 parser → 구조가 동일한 씬**이다. 구조가 같으면 두 엔진 사이에 남는 차이는 *순수 물리 엔진 차이*(접촉·솔버 편향)뿐이고, 그게 sim2real 오차 예산의 예측치가 된다.

그런데 "newton이 genesis보다 느리다/무겁다/학습이 다르다"가 **엔진 본질 차이인지, 우리 셋업 버그인지** 구분되지 않으면 이 전제가 무너진다. 그래서 legacy isaaclab-newton(newton으로 hammer를 실제 학습시켰던 코드)을 기준선으로 두고 20단계 프로세스를 하나씩 대조해, "IsaacLab을 지워도 MetaLab + newton만으로 같은 서비스가 되는가"를 검증했다. 결과: **구조·기능은 전 단계 parity, 남은 격차는 전부 설정 버그였고 규명됨.**

## §2 · 핵심 발견 — 진짜 원인들

F1rigid\_gap 0.1(10 cm)이 접촉을 33배 부풀렸다 ★ 진범

증상 — newton hammer가 legacy·genesis 대비 ~2.2× 느림. 접촉 **727개/world**인데 그중 *99.9%가 실제 침투가 아닌 near-touching*(penetrating 10,767 vs near-only 734,109 실측).

원인 — `newton.ModelBuilder.rigid_gap` 기본값 0.1이 add-time에 모든 shape의 `geom_gap`으로 **구워지고**, mjwarp broadphase가 서로 10 cm 이내인 모든 geom 쌍을 접촉 후보로 검출. 조밀한 다지 손에선 거의 모든 비인접 링크쌍이 잡힌다. 물리 비용은 접촉 수에 **선형**(886→149 ms, 21→32 ms 실측)이라 3.5× 과금인데 *fidelity 이득은 0*(gap=0에서도 penetrating 접촉 집합이 동일).

함정 — 처음엔 native-contacts(`use_mujoco_contacts=False`) 경로에만 `rigid_gap=0`을 넣고, **실제 학습이 쓰는 True 경로엔 빠뜨려서** 폭발이 남아 있었다. 두 경로 모두에 적용해야 한다.

**수정 (4c136a6)** — `overrides.newton.rigid_gap`(기본 0.0) 노출 + base/object/top 빌더 **양 경로 모두** 적용. 결과 @5120: 접촉 727 → **21/world**(genesis급), t\_phys ~114 → 32 ms, full env.step 124 → **56.5 ms**(legacy 74 ms보다 빠름). gap=0 = MuJoCo 기본값 = genesis 실질값. **self-collision 완전 보존**(손가락 안 뚫림).

sim/metalab/backends/newton/newton\_parser.py · memory: newton-rigid-gap-contact-explosion

F2nconmax는 gap 수정의 파생 — 1024 → 64로 축소

gap=0으로 실제 grasp peak가 **~22 접촉/world**(firm full-close worst case 실측)에 불과해졌다. 그러니 `nconmax`를 1024 → **64**(≈3× 여유, overflow 0)로 내릴 수 있고, 이게 접촉/EPA 버퍼를 ~16× 줄여 **newton 8192 VRAM ~30 → ~10 GB**가 됐다.

**교훈** — 앞서 8192 OOM을 `ccd_iterations` 35→16으로 막았던 건 *증상 치료*였다. EPA 워크스페이스 = `naccdmax × (6+5·ccd) · vec3`이고 `naccdmax = nconmax × nworld`이므로, 진짜 레버는 부풀려진 nconmax(→gap)였다. ccd=16은 무해하니 그대로 둠.

tasks/standalone/hammer\_lift.py overrides.newton · memory: newton-8192-oom-ccd-iterations

F3CUDA graph — 코어에 헬퍼 없음, backend에 직접 + latch 버그

newton 코어엔 graph capture 헬퍼가 없다(예제 idiom일 뿐). 그래서 `backend.step_n(decimation)`에서 `wp.ScopedCapture`로 **decimation×substeps 전체를 graph 1개**로 캡처 — policy step당 `capture_launch` 1회로 isaaclab과 동형(물리 2.21×, GPU util 100%).

숨은 버그 — lifted-force interval 이벤트의 **torch wrench 쓰기가 `_has_wrench`를 latch**해, 실학습 경로에서 graph를 *영구 비활성화*하고 있었다(실측 graph\_n=0). torch write는 graph에 캡처 불가라 매번 eager로 떨어진 것.

**수정** — wrench를 graph-상주 warp 커널(`_write_body_wrench`, 고정 버퍼)로 전환. + DECIMATION이 graph에 1 physics step만 구워지던 별개 버그도 수정(capture 전에 decimation 주입). graph 검증은 반드시 **replay-only**임을 계측으로 확인(캡처 1 / replay N-1).

sim/metalab/backends/newton/backend.py step\_n/\_write\_body\_wrench · memory: newton-cuda-graph-in-backend, decimation-not-propagated-substep-count

F4contact 파라미터는 mjw가 아니라 Newton source에 써야 생존

robot·table은 MJCF에 contact 블록이 없어 MuJoCo 기본 solref 0.02(≈4× soft)가 적용 → 침투. legacy 값(solref 0.01 등)을 mjw\_model에 직접 쓰면 **첫 friction-DR reset에서 클로버**된다 — `SHAPE_PROPERTIES` notify의 재sync 커널이 `model.mujoco.*`(custom attr)에서 매번 다시 쓰기 때문.

**수정** — Newton source(`model.mujoco.solref/solimp/solmix`)에 쓰고 `solref_mode = SOLREF_MODE_RAW`로 authored 마킹 + notify 1회 push → 재sync에도 **영구 생존(실측)**. 예외: `eq_solref/eq_solimp`는 Newton측 재sync가 없어 mjw 직쓰기로 생존(legacy도 동일).

sim/metalab/backends/newton/newton\_parser.py \_apply\_contact\_params · memory: newton-legacy-parity-contact-params, mjw-model-writes-clobbered-by-newton-resync

F5friction cone = pyramidal (elliptic이 오히려 cross-engine 불일치)

genesis 마찰은 4-edge **pyramid**(constraint/solver.py "4\*max\_contacts", noslip.py "friction-pyramid pair")다. legacy hammer는 elliptic을 썼지만 MetaLab 기준선은 genesis-검증 계약이므로, elliptic이 오히려 엔진 간 불일치였다.

**수정** — `cone: pyramidal` 채택 = genesis parity + MuJoCo 기본 + **elliptic 대비 2.3× 빠름**(109 vs 253 ms @4096 grasp). 오차 = 대각방향 마찰 최대 −29%(허용). integrator는 implicitfast 유지(euler 이득 −3%뿐, stiff PD 발산 위험).

F6friction 결합 = max, geometric-mean 이식 금지

legacy SolverJK는 geometric-mean friction을 썼지만, genesis 결합 규칙은 `max(μa, μb)`(collider/contact.py:338) = MuJoCo 기본과 일치한다. geomean을 이식하면 *오히려 cross-engine parity가 깨진다* — **하지 말 것.**

F7jacobian = auto(sparse) 유지 — dense가 2.3× 느림

오른팔 7 + 오른손 20 + 해머 free 6 = nv 33이 auto 경계(nv>32 → sparse)를 1 DOF 차로 넘는다. 실측 **sparse가 dense보다 2.3× 빠름** — auto 유지 확정. (왼팔·왼손·목·허리를 YAML에서 미사용 처리해 nv가 이 경계 근처에 있음.)

> **기타 알아둘 함정.**
> ① *Genesis가 massless Xform 프레임을 drop*(RigidBodyAPI 없는 R\_Palm\_Link 등) — MJCF 전환 후엔 팜이 네이티브 body라 대체로 moot.
> ② *use\_newton\_actuators=True면 Lab actuator.compute()가 스킵*(DelayedPDActuator·effort clip 안 돎) — 커스터마이즈는 action-term/articulation-write 레벨에서.
> ③ *native-contacts(use\_mujoco\_contacts=False)는 hammer에 부적합* — import\_mjcf가 mesh를 trimesh로 등록해 삼각형-쌍 폭발(NaN); CONVEX\_MESH retype으로 물리는 정상화되나 GJK 비용이 80×라 hammer는 True 유지.

## §3 · 오진 이력 — 헛다리 짚은 곳과 측정 방법

F1을 찾기까지 여러 그럴듯한 가설이 **전부 곁가지**로 판명났다. 인수인계의 핵심은 이 목록 자체 — 다음에 "왜 느리지?"가 나오면 여기부터 배제하고 시작하라.

| 오진 가설 | 왜 틀렸나 (반증 방법) |
| --- | --- |
| LS tolerance 회귀 (1e-6 vs 0.01) | 직접 A/B → **−4%뿐**. 솔버가 아니라 collision이 지배. |
| NVIDIA 드라이버 업데이트가 느리게 함 | genesis는 같은 드라이버로 **영향 없음**(wandb 곡선) → newton 취급 문제로 확정. |
| reset의 notify가 매 스텝 재sync | early-episode reset 0회 — 매 스텝 아님. |
| obs/reward/driver 파이프라인이 무겁다 | 단일-빌드 분해: 물리 92%, **driver 오버헤드 8%**(obs 6.6 + reward 0.8 ms). 무죄. |
| timing(120 vs 200 Hz)이 원인 | A/B → legacy timing이 **오히려 2× 느림**(policy당 8 substeps). 현 계약 우위. |
| nconmax capacity(스레드) 낭비 | 768 vs 1024 = 88 vs 91 ms. **3%뿐**, 무죄. (capacity가 아니라 *탐지 접촉 수*가 문제였음 → F1) |
| 손 collision hull AABB 상시 겹침 (narrowphase) | 거의 맞았으나 원인 오귀속 — 그 ~700 후보쌍은 hull이 아니라 **rigid\_gap 10cm** 때문. hull 재작업 하지 말 것. |

> ⚠️ **측정 방법론(이게 진짜 인수인계).**
> ① 속도·graph 계측은 반드시 *자체일관 단일 빌드*에서(빌드마다 조건이 달라지면 비교 무의미).
> ② 접촉은 **`dist<0`(실제 침투)**과 **`dist≥0`(near-touching)**을 *분리*해서 볼 것 — 총 접촉 수만 보면 F1을 놓친다.
> ③ 물리 비용은 접촉 수에 선형 → 접촉수-physics 곡선 + margin A/B면 진범이 드러난다.
> ④ graph는 "돈다"가 아니라 *replay-only인지*(캡처 1/replay N-1)를 계측으로 확인.

## §4 · 의도적으로 재현하지 않은 것 (근거)

legacy와 다르게 둔 것들 — 전부 genesis-검증 계약을 기준으로 한 **의도적** 결정이다(버그 아님).

- **timing 200/2/4 → 120/2/2** — 양 엔진 공유 계약값, genesis SR 99%로 검증됨. newton이 이 계약에서 학습 실패할 때만 재검토.
- **geometric-mean friction** → 금지(F6). genesis=max와 일치가 parity.
- **gravcomp 1.0** — newton importer·genesis 모두 MJCF gravcomp 미지원 → 양 엔진 동일하게 무시(genesis 성공이 무해함을 반증).
- **obs Gaussian noise**(joint σ=0.1 등) → teacher policy이고 genesis도 noise 없이 SR 99% → 비채택 확정. sim2real 단계에서 계약에 추가.
- **ccd\_iterations 35 → 16** — 원래 8192 OOM 레버, 탐지 접촉 수 불변. F2 이후로는 메모리에 무관하나 무해하니 유지.

## §5 · 현재 상태 · 남은 일

- *완료* — hammer-lift가 newton·genesis **각 엔진에서 재현**. newton 물리·VRAM·graph 격차 규명 및 해소. 구조 parity 20/20.
- *future work* — **sim2sim 교차평가 SR 0%**: 한 엔진에서 학습한 정책을 다른 엔진에서 평가하면 성공 못 함. 엔진 간 접촉·마찰 미세차가 grasp에 크게 작용하는 것으로 보이며, 이 격차를 좁히는 것(또는 멀티-엔진 학습으로 robust하게 만드는 것)이 다음 과제.

MetaLab · sim/ · newton↔genesis parity handoff
개요 = 00\_project\_overview · 사용법 = 10\_metalab\_tutorial
