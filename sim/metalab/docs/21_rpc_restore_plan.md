# 21 — RPC sim-service 복구 계획 (MetaLab: genesis·newton / local+aws / train+eval)

> **살아있는 체크리스트.** 각 Stone 은 additive + 검증 가능 단위. 한 번에 다 하지 않고
> 돌다리 하나씩 두드리고 검증한 뒤 다음으로 간다. 완료 시 `[x]`.

## §0 현재 상태

- **현재 위치**: **P0·S0–S9 + Phase A 코드 완료** = RPC-only(in-process 완전 제거, main 철학). 로컬 train·eval·녹화(S3) 검증 완료.
- **Phase C (CUDA-IPC transport) 거의 완료 (C0–C6 ✅, C7 불필요)** — RPC 유지하며 payload를 GPU-only로 두어 오버헤드 제거. **결과: IPC 가 socket transport 오버헤드의 85–99.6% 제거**(newton 4096 socket +149%→ipc +0.6%; genesis 8192 socket +115%→ipc +0.6%), correctness ⭐PASS(genesis 실env socket-vs-IPC `max|Δ|=0`). 3-way(inproc/socket/ipc) 27 run wandb `rpc-overhead-ab`. **최종: RPC = RPC+CUDA-IPC 단일 세트** — socket 선택·`RPC_TRANSPORT` 플래그·폴백 **전부 제거**(사용자 결정). sim-service 는 **항상** IPC(핫패스 GPU-only 공유버퍼) + socket 제어채널 = 그냥 "RPC" 하나. same-GPU 는 스크립트가 트레이너+서버 같은 물리 GPU 핀(`CUDA_VISIBLE_DEVICES` 상속)으로 자동 충족. 비-CUDA/IPC불가면 **fail-loud**(server.main CUDA assert + share 핸들 assert). **클래스명은 main 과 동일하게 `RpcServer`/`RpcClient`**(팀원 혼동 방지) — 내부만 CUDA-IPC(설명은 `rpc_transport.py` `RpcServer` 주석). `sim/service/transport.py`(엔진-무관 단일 파일)로 통합, 옛 socket `transport.py`(genesis/newton) + server `make_handler` 삭제. **isaaclab 의 socket-RPC 는 main-parity 로 보존**(inert). **커밋 정리 완료 — 4 로컬 커밋(push 미승인)**: `5a69955` isaaclab restore · `629c1aa` uv env(retire conda) · `b3ea29f` RPC sim-service + CUDA-IPC · `a695613` docs(HTML→MD + plan). Launchpad(구 Hub; server.py·telemetry.py·launchpad.sh) + hammer_lift.yaml(물리 튜닝)은 별개 concern이라 미커밋 유지(jkkim 판단). +`88de2b5` aws rl_train `--transport` 플래그. **Stone7 AWS 스모크 ✅** — 노드 `jkkim-gpu-metalab`(L40S)에서 newton 512 RPC(socket)·RPC(IPC) 둘 다 학습+`TRAIN_SERVICE_OK`. **다음 = push 승인** → (옵션) end-to-end 오버헤드/실학습. 상세 → 문서 하단 Phase C. ⚠️ AWS 노드 아직 running — 안 쓰면 stop.
- **검증 유예 (환경 막힘 — 사용자가 나중에)**: (1) **AWS 노드 스모크**(노드 막힘) · (2) **Hub UI + `--viz gl` 라이브 watch**(워크스테이션 디스플레이 필요 — 이 자동화 env 헤드리스, RPC 무관). ⚠️ Phase A 를 이 둘 前에 진행함(사용자 지시) — 전부 미커밋이라 문제 시 되돌리기 쉬움.
- **저장 목적지 모델 (확정, aws·local 동일)**: 학습 스칼라(loss/SR 커브) → **wandb**(유지) · **체크포인트 + 모든 영상**(per-checkpoint 자동녹화 + 수동 eval) → **S3**(chris 레이아웃 `s3://…/jkkim/sim_rl/{ckpts,eval_videos}/…`, CloudFront 열람). Stone5 서버-측 mp4 recorder(producer) + `rl_eval.sh`→`aws s3 cp`(uploader); 학습 자동녹화 = `_make_record_callback`→`rl_eval.sh --record`(백그라운드)→S3. wandb-영상 경로(`_record_eval_to_wandb`/`log_videos_to_wandb`/`wandb_video`)는 미사용 → Phase A 정리. (chris식 train→wandb val-hook 시도했으나 offline media 미영속 + S3 결정으로 철회.)
- **S6 성과**: `rl_train.sh`/`rl_eval.sh`에서 `SIM_INPROCESS=1` 제거 → RPC default. rl_eval.sh 녹화 = 서버 mp4 → `aws s3 cp`(fail-loud creds). 검증: rl_eval.sh RECORD=0(RPC SR) · RECORD=1(mp4 2개 S3 업로드 확인 후 정리) · rl_train.sh(`sim service on port`+`TRAIN_SERVICE_OK`) 전부 PASS. stale in-process/wandb 로그·주석 정리.
- **미커밋**: 이 문서 + `sim/isaaclab/`(P0) + `sim/{newton,genesis}/transport.py`(S0) + `sim/{newton,genesis}/server.py`(S1+S5 녹화) + `learning/rl/client.py`(S2, +apply_curriculum_end) + `learning/rl/service.py`(S3, +video forward/teardown) + `learning/trainer/rl_trainer.py`(S4) + `learning/eval/eval_service.py`(S5) + `sim/metalab/runtime/env_driver.py`(S5 task_success) + `sim/metalab/runtime/mp4_recorder.py`(S5 신규).
- **브랜치**: `dev/jkkim-irim/metalab`.
- **핵심 성과**:
  - **RPC 학습 default 동작** (newton 3-iter: `sim service on port N`→PPO loss→ckpt→`TRAIN_SERVICE_OK`). in-process HATCH 도 OK.
  - **RPC eval default 동작** — plain eval SR + `apply_curriculum_end` RPC 메서드(RPC에서도 curriculum-END SR 정확). RPC↔in-process parity(SR·reward 동일).
  - **녹화 = chris식 서버-측 mp4(RPC 경유) → S3** — genesis/newton `render_record_frames`로 서버가 per-env `env<NN>_{success,fail}.mp4` 생성(600 프레임 검증) → `aws s3 cp`(rl_eval.sh RECORD=1 실측 S3 업로드 확인). local·aws 동일.
  - **latent 버그 수정**: `env_driver.step()`이 `task_success`(object_reached_goal)를 extras에 노출 → in-process·RPC eval SR 둘 다 정상(전엔 항상 0). 녹화 success/fail 태깅도 이 신호 사용.
  - single-venv(sys.executable) 확정.

---



## 배경 / 왜 복구하나

- **팀 규칙**: MetaLab 학습·검증은 **RPC sim-service 경유로만**. 이는 곧 `main`**(chris) 의 정본 상태**다.
- **히스토리 근거**:
  - `9bca97d` (#19, Chris Ryu, 07-05): RPC sim-service **최초 도입**(isaaclab).
  - `origin/main` `b5abc75`: `SIM_INPROCESS` **전무** = **순수 RPC-only**. `service.py`/`client.py`/`transport.py` 보유.
  - `8544670` (jkkim, 07-12): MetaLab engine-agnostic(genesis/newton) — **여기서 in-process(**`SIM_INPROCESS`**) 최초 추가**. main 에 올라간 적 없음.
  - `4a0f8a7` (jkkim, 07-13): RPC 제거, in-process only. **← 이걸 되돌린다.**
- **train·eval 둘 다** RPC 경로가 실존했고, **AWS 는 RPC 가 default** 였다(`aws/rl_eval.sh` 사용).
- **토폴로지**: RPC 는 처음부터 **same-machine localhost(127.0.0.1)** 전용 — client·server 가 한 GPU 공유.
AWS 도 트레이너+서버 **둘 다 노드 위**에서 127.0.0.1 통신 → **로컬과 동일 토폴로지**(WAN 없음).



## 소스 오브 트루스 (어디서 복원하나)


| 대상                                                            | 소스                                  | 비고                                 |
| ------------------------------------------------------------- | ----------------------------------- | ---------------------------------- |
| genesis/newton RPC 코드 (transport/server-serve/client/service) | `4a0f8a7^`                          | byte-restore. genesis/newton 직접 조상 |
| isaaclab 전체                                                   | `origin/main:sim/isaaclab/`         | staged-merge 정합. 내 브랜치선 미사용        |
| **유일한 재작성**                                                   | service.py `_spawn`: **conda → uv** | 현 브랜치가 conda→uv 로 이관돼서             |




## 설계 결정 (확정)

- **single-venv RPC (v1)**: 트레이너(client) + sim server **둘 다 엔진 uv venv**(`~/.metalab/venvs/<engine>`)에서
**2 프로세스**로 127.0.0.1 통신. 팀규칙(소켓 경계 + 별도 sim 프로세스) 충족 + **uv 셋업 무변경**(엔진 venv 에
이미 `learning` import 가능). ⇒ `setup_env.sh` 손 안 댐.
  - *two-venv(린 트레이너 venv + 무거운 엔진 서버 venv)* = env 격리 이득까지 얻는 v2. 지금은 범위 밖.
- **in-process 처분**:
  - **Phase B** = 숨긴 escape hatch 로만 보존. `SIM_INPROCESS=1` 을 **명시**할 때만 동작. **default 아님**,
  문서화 안 함, **학습 스크립트는 절대 안 씀**. RPC vs sim 자체 고장 격리용 진단 도구.
  - **Phase A** = B 완전 검증 후 **완전 제거**(main 과 100% 동일 철학).



## 잃으면 안 되는 현재 기능 (매 Stone 회귀 체크)

uv 엔진 venv 부트스트랩 · AllexHub(앱창/네임카드/telemetry 탭/eval 폴더트리/DEVICE combo/EC2 combo/Stop=선택run만)
· keyless AWS(SSM, `--node` 재사용 fail-loud) · S3 ckpt sync · wandb fail-loud preflight · `--viz gl`
· 녹화(genesis raster / newton headless ViewerGL) · 앱창 telemetry iframe.

---



## Phase 0 — isaaclab 복원 (staged-merge 대비, 기능 무관)

- [x] **P0.** `git checkout origin/main -- sim/isaaclab/` (54 파일, RPC transport/server 포함)
  - ✅ 검증: 54파일 복원 · `import sim` OK · 활성 코드(newton/genesis/learning) 실제 `import sim.isaaclab` 0건(주석/문서만) → 순수 inert.

---



## Phase B — RPC 복구 (in-process 는 hatch 로 보존)



### Stone 0 — 와이어 프로토콜 (transport.py) 복원 · 순수 additive/무위험

- [x] `sim/metalab/backends/newton/transport.py`, `sim/metalab/backends/genesis/transport.py` 를 `4a0f8a7^` 에서 byte-restore
  - ✅ 검증: 3 엔진 byte-identical(md5 `bef57076...`) · 양 venv 에서 `from transport import RpcServer, RpcClient` OK · 아무도 아직 import 안 함 → 기존 동작 0 변화.



### Stone 1 — server.py serve-mode 재추가 (build_env 는 이미 있음)

- [x] `sim/metalab/backends/newton/server.py` + `sim/metalab/backends/genesis/server.py` 에 `make_handler(env)` + `main()` (argparse→build_env→RpcServer.accept/serve) 재추가.
  - 7 메서드: `attrs/get_observations/reset/step/get_ep_len/set_ep_len/seed`.
  - ✅ 검증: 양 엔진 모듈 로드 + `build_env/make_handler/main` 심볼 + `RpcServer` top-import OK · `make_handler` 7-메서드 디스패치 + unknown-method guard mock 테스트 PASS.
  - ⏳ 실제 포트 바인딩/serve → **S2 에서 서버 1회 기동 시 확정**. in-process(`build_env`) 무손상.



### Stone 2 — client 프록시 (SimServiceVecEnv) 복원 · NaNSafeVecEnv 중복 금지

- [x] `learning/rl/client.py` 재생성 — `SimServiceVecEnv` **만**. `NaNSafeVecEnv` 는 이미 `vec_env.py` 에 있으므로
  거기서 import (`from learning.rl.vec_env import VecEnv`).
  - 검증: Stone 1 서버 띄운 채 tiny smoke — `SimServiceVecEnv("127.0.0.1", port)` → `attrs`/`reset`/`step` round-trip 텐서 형상 OK.



### Stone 3 — service.py (sim_server) 복원 + **conda→uv 재배선** ⚠️ 유일한 재작성

- [x] `learning/rl/service.py` 재생성: `sim_server(args, tunables)` ctx + `ensure_transport_importable()`.
  - `_spawn` 를 uv 로: `source $(engine_venv <engine>)/bin/activate && python sim/<engine>/server.py <fwd>`.
  - isaaclab-ism 제거: `isaaclab.sh`/`ISAACLAB_DIR`/`ALLEX_DESCRIPTION_DIR` default 강제 안 함(genesis/newton 는 YAML 계약서로 asset 자체 해석).
  - `SIMULATOR` 로 엔진 선택. per-PID `/tmp` port/experiment 파일 유지(동시 run 안전).
  - 검증: `with sim_server(ns, {}) as port:` 가 엔진 uv venv 서버를 스폰하고 포트 yield → 종료.



### Stone 4 — 트레이너 RPC 분기 재연결 (RPC=default, in-process=hatch)

- [x] `learning/trainer/rl_trainer.py`: `if SIM_INPROCESS==1: build_env in-process(hatch) else: sim_server + SimServiceVecEnv(RPC, default)`.
  - `NaNSafeVecEnv` 래핑·seed·`_make_record_callback` 은 RPC/in-process 공통 경로로.
  - 검증: newton RPC 스모크 학습 몇 iter — 체크포인트 기록 + `[rl-trainer] sim service on port N` 로그 + 손실 감소 신호.



### Stone 5 — eval RPC 분기 재연결 (+ 서버-측 mp4 녹화, +task_success 수정)

- [x] `eval_service.py`: RPC default (`sim_server`+`SimServiceVecEnv`), `SIM_INPROCESS=1`=in-process hatch.
  - **SR 정확도**: `apply_curriculum_end` RPC 메서드(server make_handler + client) → RPC에서도 curriculum-END SR.
  - **녹화 = chris식 서버-측 mp4/RPC**(사용자 결정): `sim/metalab/runtime/mp4_recorder.py`(신규, `render_record_frames`) → server가 per-env `env<NN>_{success,fail}.mp4` → `log_videos_to_wandb.py`가 wandb 직접(S3 무관). `--record`=`--video` alias. in-process hatch는 jkkim식 frame→wandb(`_record_eval_to_wandb`) 보존.
  - **latent 버그 수정**: `env_driver.step()` extras에 `task_success`(object_reached_goal) 노출 — 없으면 in-process·RPC eval SR 둘 다 항상 0였음.
  - ✅ 검증: plain RPC eval(SR+curriculum-end) · RPC `--video`(env00/01_fail.mp4 각 600프레임 유효) · in-process hatch eval(RPC와 SR·reward parity) 전부 PASS.



### Stone 6 — 로컬 스크립트를 RPC 모드로

- [x] `rl_train.sh`/`rl_eval.sh` `SIM_INPROCESS=1` 제거 → RPC default. rl_eval.sh 녹화 = 서버 mp4 → `aws s3 cp`(chris 레이아웃, fail-loud creds, scratch 삭제). stale in-process/wandb 문구 정리.
  - ✅ 검증: RECORD=0(RPC SR) · RECORD=1(env00/01_fail.mp4 → S3 업로드 확인·정리) · rl_train.sh(`sim service on port`+`TRAIN_SERVICE_OK`) PASS.



### Stone 7 — AWS 스크립트를 RPC 모드로 (노드 위 localhost)

- [x] **코드**: `aws/rl_train.sh`가 노드에서 `local/rl_train.sh`를 호출 → Stone 6로 **이미 RPC**(aws 흐름에 `SIM_INPROCESS` 전무). `--record`→rl_eval.sh→S3, `--s3-sync`→ckpts→S3. 헤더 stale "in-process (NO RPC)" 정정. `--transport socket|ipc` 추가(remote `RPC_TRANSPORT` export, 커밋 `88de2b5`).
  - [x] ✅ **검증 완료 (노드 `jkkim-gpu-metalab` i-03c8be9cf53869d95, g6e.4xlarge L40S)**: worktree→S3 tar 배포 + 노드 uv sync(conda 無) + newton 512 × 3 iter, **RPC(socket)·RPC(IPC) 둘 다 PASS** — wandb URL 발급 + loss/reward 학습(both Mean reward ~0.18) + `TRAIN_SERVICE_OK` exit 0. IPC 는 AWS L40S 에서도 CUDA-IPC 핸들 교환 성공(단일 GPU=same-GPU 전제 충족, AssertionError 無). Steps/sec socket 6446→7301 / ipc 6706→7868.



### Stone 8 — Hub 점검 (대체로 agnostic)

- [x] Launchpad(구 Hub) 는 rl_train/rl_eval **shell-out + 로그 tail + 엔드포인트**라 **RPC-agnostic → 코드 변경 불필요**. telemetry-over-RPC 서피싱은 **설계상 보장**: service.py `_spawn` 의 `Popen` 이 서버 stdout+env 상속 → 서버의 "live dashboard → http://.." 가 트레이너 로그(Launchpad tail)에 오르고 `METALAB_LAUNCHPAD_EMBED` 도 서버로 전파.
  - ⏳ 검증 유예: 전체 Hub UI(telemetry 탭 렌더·네임카드·Stop·`--viz gl` 라이브 watch)는 워크스테이션 디스플레이+브라우저 필요(이 env 헤드리스 → 인터랙티브 GL 불가, RPC 무관).



### Stone 9 — Phase B 완료 게이트

- [ ] local train+eval, aws train+eval **4 조합 모두 RPC 로** 완주 + 위 "잃으면 안 되는 기능" 전부 회귀 없음.
- [ ] 커밋 정리(논리 단위) — **push 는 별도 승인**.

---



## Phase A — in-process 완전 제거 (B 완전 검증 후에만)

- [x] `rl_trainer.py`/`eval_service.py` 의 `SIM_INPROCESS` 분기 + in-process 코드 제거 (RPC 단일 경로). `build_env` 는 서버가 쓰므로 유지.
- [x] 스크립트/문서(server.py·rl_train.sh·rl_eval.sh·mp4_recorder.py) `SIM_INPROCESS`/hatch 주석 스크럽.
- [x] **wandb-영상 경로 삭제**: `_record_eval_to_wandb`(+orphan `import numpy`) 제거 · `learning/eval/log_videos_to_wandb.py` + `learning/rl/utils/wandb_video.py` 삭제(S3로 대체). 스칼라 `wandb_writer` 는 유지.
- [x] ✅ 검증: 활성 코드 `SIM_INPROCESS` 0건 · 10개 변경 py 구문 OK · RPC-only train(`TRAIN_SERVICE_OK`)+eval(`EVAL_OVER_SERVICE_OK`) 스모크 PASS.

---



## 열린 항목 / 나중에

- two-venv RPC(린 트레이너 venv) — env 격리 이득 필요 시 v2.
- CUDA-IPC 공유 GPU 메모리 transport — **→ 아래 Phase C** (설계 확정 + feasibility 실측 완료).

---

## Phase C — CUDA-IPC transport (RPC 유지 + payload GPU-only → 오버헤드 제거)

> 목표: `step()` 핫패스에서 obs/action 을 **GPU 에 유지**(공유 GPU 버퍼 + CUDA-IPC)해 `torch.save`+GPU↔CPU↔socket **payload 복사를 제거**. 같은 `RpcServer`/`RpcClient` 계약 뒤에 드롭인 → 팀 RPC-only 유지. 그리고 **in-process / RPC(socket) / RPC(CUDA-IPC) 3-way 비교**로 효과 정량화.

### 왜 (실측 근거)
- socket RPC 오버헤드는 **payload(=num_envs×obs) 복사에 비례** — genesis/newton 실측: 512 ~8%, 2048 ~28%, 4096 ~48%, **8192 +63%** (엔진 무관, `rpc-overhead-ab` wandb). 진범 = 매 스텝 GPU→CPU→socket→CPU→GPU.
- 그 payload 복사만 없애면 잔여 = **스텝당 신호+동기화(프로브 0.32ms/step)** → 곡선 평탄화(대규모 폭증 소멸).

### 검증된 전제 (feasibility, 실측 — C0)
- **POC ✅**: torch CUDA-IPC 크로스-프로세스 공유 GPU 텐서 동작 (RTX 5090, torch 2.10.0+cu128, **same GPU**). 자식이 on-GPU write → 부모가 같은 물리 GPU 메모리에서 [7,7,7,7] 봄 = zero-copy 공유. (`scratchpad/cuda_ipc_poc.py`)
- **same-GPU 전용**: 이 박스 GPU0↔1 **P2P 불가**(`can_device_access_peer=False`, `nvidia-smi topo=CNS`). single-venv RPC 는 트레이너+서버가 **같은 GPU** → OK. 크로스-GPU 스플릿은 이 박스선 불가(우린 안 씀).

### 설계 결정
- **핫패스만 IPC**: `step` / `get_observations` / `reset`(returns obs) 만 공유버퍼. cold-path(`attrs`/`seed`/`get_ep_len`/`set_ep_len`/`apply_curriculum_end`)는 **기존 socket RPC 유지**(드묾, 무시 가능).
- **고정 shape 공유버퍼**(env 빌드 후 불변): obs 그룹별(N×dim)·action(N×A)·reward(N)·dones(N)·time_outs(N)·task_success(N). 셋업 1회 할당, **IPC 핸들을 부트스트랩 소켓으로 교환**.
- **payload write = on-GPU copy**: 서버가 `shared.copy_(wp.to_torch(obs))` (warp 내부와 무관, 싸다). 트레이너는 공유버퍼에서 GPU로 직접 read.
- **동기화**: 우선 스텝당 `torch.cuda.synchronize()`(단순·안전) → 검증 후 **CUDA event IPC**(`cudaIpcGetEventHandle`)로 미세 최적화. ⚠️ 동기화 틀리면 **silent 데이터 오염**(그래디언트) → C5 게이트 필수.
- **extras["log"]**(host float dict): 소량 → 소켓으로 동반 전송(또는 N스텝마다).
- **최종: 단일 세트 (플래그·socket·폴백 제거)** — 처음엔 `RPC_TRANSPORT` 로 socket↔ipc 토글(default socket)로 구현했으나, 사용자 결정으로 **RPC=RPC+CUDA-IPC 하나로 고정**. `transport_mode`/`RPC_TRANSPORT`/socket `RpcServer`·`RpcClient`(genesis/newton `transport.py`)·server `make_handler` 전부 삭제. sim-service 는 항상 `IpcServer`+`serve_vec_env`(server) / `IpcClient`(client). GPU-only fail-loud(server.main CUDA assert). same-GPU 는 스크립트 GPU 핀으로 자동. (3-way 벤치 때는 socket/inproc 를 임시로 썼고 지금은 제거 — 히스토리는 git.)
- **allocator 주의**: 공유버퍼는 IPC-safe 하게 keep-alive(caching allocator 재활용 회피; 필요시 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`).

### Stones
- [x] **C0** feasibility POC (위 실측 — same-GPU IPC 공유 동작 확인, P2P off 확인).
- [x] **C1** 공유버퍼 transport 프리미티브 — 신규 모듈 `sim/service/transport.py`(엔진-무관). `IpcServer`/`IpcClient`(소켓 bind/accept/connect + 부트스트랩 핸들교환 + control 채널) + `serve_vec_env`(양 엔진 server 공용 서버 루프, sim-aware) + 키 규약(`obs::<group>`/action/rew/dones/time_outs/task_success) + `build_layout`. hot-path=공유버퍼, cold-path(attrs/seed/get_ep_len/set_ep_len/apply_curriculum_end)=control 채널(소켓, RPC와 동일). fail-loud: IPC 핸들 None(expandable_segments)이면 share 시 assert.
  - ✅ 검증(돌다리): (1) **소켓 핸들교환 POC** — 독립 `Popen` 2프로세스 + 소켓 + `reduce_tensor` pickle 전송으로 zero-copy 공유 PASS(mp.Process 아닌 실제 토폴로지). 기본 allocator IPC 핸들 present=True(이 박스 expandable_segments 미설정; RL 스크립트도 미설정 — `train_groot.sh`만 True=BC 경로 무관). (2) **프리미티브 자체 테스트** — MockEnv + 실제 Popen+소켓 토폴로지로 40스텝 N=2048/A=30, obs 2그룹+rew+dones+time_outs+task_success **bit-exact**(mismatch=0) + cold-path 동작. newton·genesis 양 venv PASS. (`scratchpad/ipc_socket_poc.py`, `ipc_transport_selftest.py`)
- [x] **C2** server 측: `sim/{newton,genesis}/server.py` main() 에 `--transport socket|ipc` 인자 + 분기 추가(default socket). ipc 시 `IpcServer` + 공용 `serve_vec_env`(step 핸들러가 obs 그룹·rew·dones·time_outs·task_success 를 공유버퍼에 on-GPU write → `torch.cuda.synchronize()` → ready 신호; action 공유버퍼에서 read. get_observations/reset 도 obs 공유버퍼로. cold-path=control 채널). recorder.after_step 보존. 양 엔진 모듈 로드 OK.
- [x] **C3** client 측: `SimServiceVecEnv(transport="ipc")` — action write → `synchronize` → `_ctl("step")` 신호 → obs/rew/dones/time_outs/task_success 공유버퍼에서 read(clone). **같은 VecEnv 인터페이스**(trainer/eval 무변경). socket 경로는 그대로.
- [x] **C4** wire 선택: `service.transport_mode()`(env `RPC_TRANSPORT`, default socket, 검증) 단일 소스 — `_spawn` 이 `--transport` 포워딩 + `rl_trainer`/`eval_service` 가 `SimServiceVecEnv(transport=...)` 로 사용(양 끝 항상 일치). 7개 파일 py_compile OK.
- [x] **C5** ⭐ **correctness 게이트 PASS**: (1) **transport 무손실** — C1 mock(결정적) bit-exact mismatch=0. (2) **실 env** — 실제 server.py(`--transport ipc`)+client 를 Popen+소켓으로 띄워 같은 seed/action stream 25스텝 구동: **genesis socket-vs-IPC `max|Δ|=0`(obs 전그룹+rew+dones+task_success 완전 일치)**; newton 은 엔진 자체가 프로세스간 비결정적(MuJoCo 접촉 atomic 합산 순서)이라 socket-vs-socket 도 max|Δ|~57 인데 **IPC 발산은 그 밴드 아래**(25.7). dones/task_success 는 양 엔진 모든 비교에서 동일. ⇒ sync race/aliasing 없음(있으면 genesis max|Δ|=0 불가). (`scratchpad/ipc_e2e_parity.py`)
- [x] **C6** **3-way 벤치 완료** (harness `scratchpad/bench_transport.py`+`bench_sweep.sh` — 모드별 별도 프로세스, collection 루프 = num_steps_per_env(24)×`env.step`, warmup 8·측정 40 iter). num_envs 512–4096(genesis+8192) × genesis/newton × {inproc,socket,ipc} = 27 run 전부 **wandb `rpc-overhead-ab`** 온라인 기록(`<engine>-N<N>-<mode>`, mean_s_per_iter). **결과: IPC 가 socket transport 오버헤드의 85–99.6% 제거, 대규모에서 in-process 하한과 사실상 동일** (아래 표).
- [ ] **C7** (opt, **불필요 판단**) CUDA event IPC — 잔여가 이미 무시할 수준(4096 newton +0.6%, 8192 genesis +0.6%)이라 `torch.cuda.synchronize()` 최적화 이득 없음. 필요 시에만.

### 3-way 벤치 방법 (harness = `scratchpad/bench_transport.py` + `bench_sweep.sh`)
- **격리 측정 = collection 단계만**: 한 "iter" = `num_steps_per_env`(24) × `env.step`. transport 오버헤드는 100% 여기에 있음(학습단계 SGD는 client-side GPU라 transport-무관·모드 동일). 그래서 in-process baseline 은 **임시 hatch 재활성 불필요** — worker 가 `build_env`(서버 자체 함수, Phase A 에서 유지됨)를 직접 in-process 로 구동. socket/ipc 는 실제 server.py(`--transport`)+client 스폰. **모드마다 별도 프로세스**(CUDA/env state 격리) + warmup 제외 평균 s/iter.
- ⚠️ 이 collection-only 오버헤드%는 이전 full-iter sweep(8192 +63%)보다 **절대값이 큼**(학습단계 희석 없음). 비교 포인트는 **socket→ipc 상대 개선**.
### 결과 (풀 sweep, collection s/iter, warmup8/측정40, GPU1 RTX5090; wandb `rpc-overhead-ab`)

```
newton                                    genesis
 N     inproc socket  ipc  soc%   ipc%     N     inproc socket  ipc  soc%    ipc%
 512   0.280  0.340  0.285 +21.2  +1.7     512   0.571  0.637  0.581 +11.5   +1.7
1024   0.308  0.394  0.309 +28.1  +0.3    1024   0.617  0.736  0.642 +19.3   +4.1
2048   0.351  0.665  0.358 +89.8  +2.1    2048   0.680  1.042  0.703 +53.2   +3.3
4096   0.444  1.107  0.446 +149.4 +0.6    4096   0.796  1.191  0.818 +49.7   +2.8
                                          8192   1.046  2.254  1.053 +115.4  +0.6
```

- **socket 오버헤드는 num_envs↑ 폭증**(newton 4096 **+149%**, genesis 8192 **+115%**) — payload 복사가 진범(실증).
- **IPC 는 전 구간 평탄 ~0–4%** = in-process 하한과 사실상 동일. **socket transport 오버헤드의 85–99.6% 제거**(대규모일수록 ↑: 4096/8192 에서 99%+).
- **절대 절감**: newton 4096 iter당 0.66s, genesis 8192 iter당 1.20s (learning 단계는 이 위에 transport-무관 상수로 더해짐 → 실제 학습 전체 오버헤드%는 이보다 작지만, **절대 절감 시간은 동일**).
- ⇒ Phase C 가설 완전 실증: **CUDA-IPC 가 대규모에서 RPC 경계를 거의 공짜로 만든다.** C5(무손실)+C6(속도)로 IPC 는 socket 의 드롭인 상위호환(단, same-GPU 전제).

### 결과 (AWS end-to-end, 실제 trainer full-iter = collection+learning, newton g6e.4xlarge L40S; wandb `rpc-overhead-ab` `-aws`)

실제 `learning.train` 100 iter × {inproc,socket,ipc} × {512,1024,2048,4096} (12 run). full iter = collection + learning(=transport-무관 상수). warmup 제외 평균:

```
 N     inproc  socket   ipc   | iter_ovh% (soc/ipc) | coll_ovh% (soc/ipc) | learn_s(모드동일)
 512   1.821   1.966   1.850  |  +7.9 / +1.6        |  +8.5 / +1.7        | ~0.116
1024   2.248   2.465   2.332  |  +9.6 / +3.7        | +10.6 / +4.1        | ~0.172
2048   2.814   3.285   2.806  | +16.7 / -0.3        | +18.6 / -0.3        | ~0.274
4096   3.199   4.026   3.351  | +25.9 / +4.8        | +30.7 / +5.7        | ~0.493
```
(iter s = 모드별 full-iter wall; learn_s 는 세 모드 거의 동일 → 학습단계 transport-무관 확인.)

- **실제 학습 오버헤드(end-to-end)**: socket +7.9%→+25.9%(num_envs↑), **IPC 는 ≤+5%** (2048 은 노이즈로 -0.3%) = in-process 사실상 동일.
- collection-only 실측(위 로컬 표)과 정합: L40S 에선 learning 이 작아 iter-오버헤드 ≈ coll-오버헤드. 더 무거운 학습/느린 GPU 면 learning 이 희석해 % 는 더 작아짐(절대 절감은 동일).
- **Stone7 AWS + Phase C 실노드 실증 완료**: RPC(socket)·RPC(IPC) 둘 다 노드에서 학습, IPC 가 실학습 오버헤드를 socket 대비 5배↓.
- 운영 메모: 노드 root 73G 는 DLAMI base + 1회성 uv env(torch/newton/warp 휠 ~9G)로 이미 빠듯 → sweep 중 로그 누적으로 100% → miniconda3(12G, conda retire 후 dead) 제거로 해소. 학습 자체는 디스크 거의 안 씀(100 iter=ckpt 0).

**genesis 도 동일 (같은 노드, 50 iter):**
```
 N     inproc  socket   ipc   | iter_ovh% (soc/ipc) | coll_ovh% (soc/ipc)
 512   1.469   1.574   1.488  |  +7.2 / +1.3        |  +9.2 / +2.8
1024   1.659   1.874   1.732  | +13.0 / +4.4        | +15.7 / +6.2
2048   1.931   2.276   1.949  | +17.9 / +0.9        | +22.2 / +2.2
4096   2.362   2.928   2.368  | +24.0 / +0.3        | +31.6 / +1.6
```
→ 양 엔진 완전 정합: **socket +7%→+24% (num_envs↑), IPC ≤+4.4%**. genesis 도 same-GPU CUDA-IPC 로 실학습 오버헤드가 socket 대비 ~5배↓. (genesis wandb `rpc-overhead-ab` `*_genesis_*-aws` 12 run.)

### 리스크 (정직)
- 구현 난이도·공수(며칠) + sync **correctness**(C5)가 핵심 — 실패 시 조용히 학습 망가짐.
- torch caching-allocator ↔ IPC 마찰(공유버퍼 keep-alive 관리).
- same-GPU 전용(이 박스 P2P off) — single-venv 는 충족, 크로스-GPU 스플릿 불가.

