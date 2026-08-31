*MetaLab · PRD — 제품 기획 · 초안 v0.1*

# MetaLab Launchpad — 원클릭 학습·검증 웹 콘솔

터미널·플래그·conda env 를 몰라도, **브라우저에서 버튼 몇 번**으로 원하는 엔진에서 원하는 태스크를 학습/검증하고,
누구나 **그대로 재현**할 수 있게 만드는 상시 실행형 웹 콘솔. 지금의 관전용 telemetry 대시보드를 프로세스 밖으로 **끄집어내 확장**한다.

- 진입점 sim/metalab\_hub.sh
- 대상 처음 쓰는 사람도
- 철학 원클릭 실행 · 완전 재현
- 범위 v1 로컬

## 00 · 한 줄 요약 · 왜

> 문제. 학습/검증을 돌리려면 지금은 *터미널을 열고* → 올바른 conda env 를 알고 → `rl_train.sh --sim … --task … --num_envs …` 플래그를 외워 치고 → 로그를 눈으로 tail 해야 한다. 처음 온 사람에겐 진입장벽이 높고, "그때 뭘로 돌렸더라"가 흩어져 *재현이 어렵다*. 웹 대시보드는 있지만 *실행 중인 한 프로세스를 관전만* 할 뿐, 실행을 *시작*하지도, 이력을 남기지도 못한다.
>
> 해법. `sim/launchpad.sh` 한 줄로 뜨는 *MetaLab Launchpad* — 엔진·태스크·작업(학습/검증)·노브를 **버튼과 드롭다운으로 고르고 Launch**. 실행은 검증된 `rl_train.sh`/`rl_eval.sh` 를 그대로 부른다. 실행되는 **정확한 명령을 화면에 그대로 보여주고 기록**하며, 라이브 telemetry·로그·체크포인트를 한 화면에서 본다. 어떤 실행이든 **"다시 실행" 한 번으로 재현**된다.

> **설계 원칙 3줄.** ① Launchpad 는 실행 로직을 새로 만들지 않는다 — **유지 스크립트를 그대로 shell-out**(단일 진실 소스). ② Launchpad 서버는 **stdlib-only** → conda env 무관하게 뜨고, env activate 는 스크립트가 한다. ③ 실행은 **명령 + git SHA + 설정**이 함께 남는다.
> > **구현 현황 (2026-07-12)**
> > · **P0 완료(로컬)** — 앱 아이콘 → 자기 **최대화 창** → 엔진·태스크·작업·노브 → **Launch**(rl\_train/eval shell-out) + 실행 전 명령 미리보기.
> > · **라이브 콘솔 + Stop** — 우측 패널이 현재 실행 로그를 **라이브 tail**(ANSI 컬러·폰트 zoom). **Stop** = 실행 중 프로세스 전체를 Ctrl-C(SIGINT→SIGKILL)로 종료(Launchpad 유지). Exit·탭닫기 = 전체 종료(프로세스 트리 kill → wandb *"killed"*).
> > · **env 자동 프로비저닝** — engine conda env 없으면 S3 스냅샷에서 자동 복원(`setup_env.sh`/`snapshot_env.sh`) → clone 후 `rl_train.sh` 만으로 로컬·AWS 모두 학습.
> > · *피벗* — 실행이력 목록·과거 로그 뷰어는 **제거**(버그원 + 전체 로그는 wandb). 구조화 telemetry 플롯 흡수(P1)는 **보류**.
> > · *다음* — **P3 AWS: RPC 없이 노드에서 in-process 실행**(env 자동복원이 핵심 인에이블러). 실제 노드 검증 대기.

## 01 · 성공 기준 — 이게 되면 성공

- **터미널 0줄.** 처음 온 사람이 문서 없이, `sim/launchpad.sh` 실행 후 **브라우저 클릭만으로** hammer-lift 스모크 학습을 띄운다.
- **실행 = 재현.** 아무 과거 실행에서 *"다시 실행"* → 같은 엔진·태스크·노브·시드로 재현되고, 원본과의 코드 차이(git SHA)가 화면에 표시된다.
- **한 화면 관측.** 실행 목록(상태·진행률·SR) + 선택 실행의 라이브 obs/reward/termination 플롯 + 로그 tail + 체크포인트가 한 곳에.
- **투명성.** Launch 전에 **실제로 실행될 명령 전문**을 보여준다 — 버튼이 뒤에서 뭘 하는지 항상 보이므로, 원하면 그대로 복사해 터미널에서도 쓸 수 있다(러닝 브릿지).

## 02 · 사용자 여정 — 클릭 흐름

metalab\_hub.sh → 브라우저 자동 오픈 → 엔진 고르기 → 태스크 고르기 → 학습 / 검증 → 노브(선택) → Launch → 라이브 관측 · 재현

### 모형 — 런처 Launcher

MetaLab Launchpadlocalhost:8770

1 · 시뮬레이터 엔진

newtongenesis

2 · 태스크 (자동 탐색: sim/metalab/contract/tasks/)
hammer-lift

3 · 작업

학습 · Train검증 · Eval

4 · 노브 (기본값이 이미 채워져 있음 — 그냥 Launch 해도 됨)

num\_envs

4096

max\_iter

5000

seed

42

device

cuda:0

3D 뷰어 창 (엔진 GUI)
 wandb 끄기
 S3 체크포인트 미러
 체크포인트별 영상 녹화

실행될 명령 (미리보기 · 그대로 복사 가능)

```
# 이 버튼이 뒤에서 실제로 부르는 것:
learning/scripts/local/rl_train.sh
--sim
newton
--task
hammer-lift
--num_envs
4096
--max_iterations
5000
```

▶ Launch
명령 복사
Enter 로도 실행

엔진·태스크·작업은 버튼/드롭다운, 노브는 기본값 프리필 — 처음 온 사람은 **클릭 3번 + Launch**면 끝.

### 모형 — 라이브 콘솔 + Stop right pane

MetaLab Launchpad▶ Launch  ■ Stop

[train] in-process (no RPC): trainer+sim in env=newton
 Learning iteration 42/5000
Mean reward: 12.34
Mean episode length: 187.2
…

우측 패널은 **현재 실행의 라이브 로그**만 tail(ANSI 컬러 렌더·폰트 zoom). 실행이력 목록/과거 로그 뷰어는 없음 — 전체 로그·영상·메트릭은 **wandb**. **Stop** = 실행 중 프로세스를 Ctrl-C 로 종료(Launchpad 는 유지), Exit = 전체 종료.

## 03 · 현재 → 목표 (무엇을 끄집어내나)

지금의 `sim/metalab/runtime/telemetry.py` 는 훌륭한 **자체 완결 HTTP+SSE 대시보드**(stdlib-only, 캔버스 라이브 플롯)지만, `env_driver` **안에서** `--viz` 일 때만 뜨고 그 프로세스와 생사를 같이한다. 관전만 가능하다. 이 렌더링 자산은 **버리지 않고 재활용**하되, 소유권을 프로세스 → Launchpad 로 옮긴다.

#### 지금 (in-process 관전)

- `--viz` → `EnvDriver` 안에서 `TelemetryServer` 기동
- 한 실행의 env-0..3 을 *관전만*
- 프로세스 끝 = 대시보드 소멸, 이력 없음
- 실행은 여전히 터미널에서 손으로

#### 목표 (standalone Launchpad)

- `launchpad.sh` → **상시** Launchpad 서버
- 여러 실행을 **런치·나열·관측·재현**
- 실행 이력·명령·SHA 영속
- telemetry 는 Launchpad 로 **스트림**되어 실행과 분리

> ⚠️ **`--viz` 의 두 역할을 분리한다.** 현재 `--viz` 는 ⓐ**엔진 네이티브 3D 창**과 ⓑ**웹 telemetry 브라우저**를 동시에 연다. Launchpad 는 **ⓑ만 흡수**(telemetry 는 항상 Launchpad 로 흐름) 하고, ⓐ(엔진 GL 창)는 그대로 두되 Launchpad 폼의 *"3D 뷰어 창"* 토글로 노출한다. 즉 telemetry ↔ 3D 창이 **독립 토글**이 된다.

## 04 · 아키텍처

Launchpad 서버 → rl\_train.sh / rl\_eval.sh → conda activate + 실행 → EnvDriver (telemetry 스트림) → Launchpad 로 등록·중계

| 구성요소 | 역할 |
| --- | --- |
| metalab\_hub.sh | 진입점. stdlib-only Launchpad 서버를 띄우고 브라우저를 자동으로 연다(현 `_open_browser` 재사용). conda env 불필요. |
| Launchpad 서버 (신규) | `telemetry.py` 의 HTTP 골격을 확장. 태스크/엔진 자동 탐색, 런치 요청 처리(스크립트 shell-out), 실행 레지스트리, telemetry 중계, 로그 tail 스트림, 정적 UI 서빙. |
| 런처 | UI 폼 → `rl_train.sh`/`rl_eval.sh` 인자 조립 → `subprocess` 로 실행. **스크립트는 절대 재구현하지 않음.** 실행 전 명령 전문을 UI 로 회신. |
| 실행 레지스트리 | 실행마다 *명령·엔진·태스크·노브·git SHA·PID·로그 경로·telemetry 포트·상태*를 JSON 으로 기록·영속. "다시 실행"·이력·재현의 근거. |
| telemetry 중계 | 실행이 자기 telemetry 스냅샷을 Launchpad 로 흘린다(핸드셰이크로 포트/실행ID 등록 → Launchpad 가 proxy, 또는 실행이 Launchpad 로 push). 현 캔버스 플롯 UI 를 Launchpad 안에서 재사용. |
| 유지 스크립트 | **변경 없음.** Launchpad 는 이들의 얇은 GUI 프론트일 뿐 — 그래서 CLI 와 Launchpad 가 언제나 동일 결과(진실 소스 1개). |

> 엔진/태스크 **자동 탐색**은 스크립트의 `_list_tasks` 규칙을 그대로 쓴다(`sim/metalab/contract/tasks/*.py|yaml`). 새 태스크·새 엔진이 추가되면 Launchpad UI 에 **자동으로** 나타난다 — Launchpad 코드 수정 불필요.

## 05 · 핵심 기능

| 기능 | 내용 |
| --- | --- |
| 원클릭 런치 | 엔진·태스크·작업 3택 + 노브 프리필 → Launch. 클릭 3번. *구현* |
| 명령 투명성 / 재현 | 실행 전 실제 명령 전문 표시 + 복사(= 수동 재현). *구현* · 원클릭 "다시 실행"은 미구현(실행이력 제거). |
| 라이브 콘솔 + Stop | 우측 패널이 현재 실행 로그를 라이브 tail(ANSI·zoom). Stop = 실행 중 프로세스 Ctrl-C(SIGINT→SIGKILL) 종료, Launchpad 유지. 전체 로그는 wandb. *구현* |
| 종료 수명주기 | Exit·탭닫기·브라우저 종료 → 실행 프로세스 트리까지 정리(wandb "killed"). 아이콘 = 자기 최대화 창, telemetry(viz)는 같은 창 새 탭. *구현* |
| env 자동 프로비저닝 | engine conda env 없으면 S3 스냅샷에서 자동 복원(genesis 는 genesis-world clone + editable 재작성) → clone 후 rl\_train.sh 만으로 학습(로컬·AWS). *구현(genesis, 노드 검증 대기)* |

## 06 · 단계 · 로드맵

**P0 = 로컬 완료**(newton·genesis, in-process). 이후 실제 진전은 **P3 AWS = 노드에서 동일 in-process 실행(RPC 없음)** — env 자동복원이 이미 그 인에이블러다.

#### P0런처 + 라이브 콘솔 + 종료 — ✅ 완료 (피벗 포함)

`launchpad.sh` → Launchpad 서버·최대화 창. 엔진/태스크 자동 탐색. 폼 → `rl_train.sh`/`rl_eval.sh` shell-out + 명령 미리보기. **피벗:** 실행이력 목록·telemetry 플롯 대신 **현재 실행 라이브 콘솔 + Stop** + Exit/탭닫기 수명주기(프로세스 트리 kill→wandb "killed"). 레지스트리(`logs/launchpad/runs.jsonl`)는 Stop 대상 추적용으로만.

#### P1구조화 telemetry 흡수 — 보류 (로그는 wandb)

obs/reward/termination + Eval/SR 곡선을 Launchpad 로 흡수하는 계획. wandb 가 이미 커버 → **보류.** 필요해지면 재개.

#### P2재현·편의 폴리시 — 부분/드롭

명령 복사 = 수동 재현 ✅. 원클릭 "다시 실행"·체크포인트 브라우저·프리셋·딥링크는 실행이력 제거로 **드롭/보류.**

#### P3AWS in-process launch — 다음 (노드 검증 대기)

**RPC 불필요**(MetaLab 스포크는 in-process). 노드에서 로컬과 **동일한 `rl_train.sh`** 를 돌리고(env 는 `setup_env.sh` 가 S3 스냅샷에서 자동복원) SSM 으로 로그 tail. 남은 일: genesis env 스냅샷 업로드 · 노드 base(miniconda/git)+런치 · additive `aws/` 배포·실행·tail 스크립트 + Launchpad "AWS" 토글. 기존 chrisryu 의 RPC/isaaclab 경로는 **안 건드림**(additive).

## 07 · 범위 · 비목표 · 열린 질문

### local vs AWS 둘 다 MetaLab in-process

핵심 재검토: **RPC 는 IsaacLab 때문에 필요했던 것**이다 — Isaac Sim 은 자체 앱 프로세스라 트레이너와 한 프로세스에 못 담아 별도 sim-service + 소켓(RPC)이 불가피했다. 반면 **MetaLab 의 newton/genesis 는 import 해서 step 하는 라이브러리**라 트레이너와 **같은 프로세스(in-process)**로 돈다 → **RPC 불필요.** 그래서 AWS 도 노드에서 **로컬과 동일한 in-process 경로**를 그대로 돌리면 된다(소켓 telemetry 터널도 불필요 — 로그는 SSM tail + wandb).

| 능력 | local (newton·genesis) | AWS (동일 in-process, 계획) |
| --- | --- | --- |
| Launch / Stop | *완료* | 노드에서 같은 `rl_train.sh` 실행 (SSM) |
| env 준비 | *자동* (있으면 그대로) | *자동 복원* (S3 스냅샷 → setup\_env) |
| 로그 | *라이브 콘솔* | SSM tail → 콘솔 + wandb |
| 재현 | 명령 복사 (동일 스크립트·SHA) | 동일 |

> **env 자동 프로비저닝 (구현됨).** `setup_env.sh` — engine conda env 없으면 S3 스냅샷(`s3://…/metalab/conda/envs/<engine>.tar.gz`, conda-pack)에서 복원(genesis 는 genesis-world clone + editable 경로 재작성). `snapshot_env.sh` 로 env 소유자가 1회 업로드. rl\_train/eval 이 activate 직전 호출 → **clone 후 그냥 rl\_train.sh 하면 env 자동 설치·학습**(로컬 온보딩 + AWS 노드 env 를 한 번에 해결).
> > ⚠️ **AWS P3 남은 일 (실제 노드 필요).** ① genesis env 스냅샷 S3 업로드(`snapshot_env.sh --sim genesis`). ② 노드 base(miniconda·git) + gpu-launcher 프로파일로 노드 런치. ③ additive `aws/` 스크립트: repo rsync 배포 → 노드에서 `rl_train.sh --sim genesis`(env 자동복원) → SSM 로그 tail → Launchpad "AWS" 토글. 기존 chrisryu 의 RPC/isaaclab 경로는 **안 건드림**(additive).
> >
> > ### 비목표 out of scope
> >
> > - **실행이력 / 과거 로그 뷰어** — 제거(버그원). 콘솔은 현재 실행만 tail, 전체 로그·영상·메트릭은 wandb.
> > - **스크립트/실험 노브 재구현** — Launchpad 는 실행 방법만 조립. 보상·커리큘럼·PPO 는 YAML · experiment.py.
> > - **다중 사용자/인증** — 로컬 개발 도구. 127.0.0.1 바인드.
> >
> > ### 확정된 결정 was: 열린 질문
> >
> > - Launchpad 서버 위치 = `sim/metalab/runtime/launchpad/` ✓ · 실행 레지스트리 = `logs/launchpad/runs.jsonl`(gitignored) ✓
> > - 노브 = 핵심 + "고급" 접이식 ✓ · 종료 = SIGINT(Ctrl-C → wandb "killed") → 8s → SIGKILL, 프로세스 **트리** kill ✓
> > - 구조화 telemetry 플롯 흡수(P1) = 보류(로그는 wandb) · AWS = in-process(RPC 없음) ✓

MetaLab · sim/ · MetaLab Launchpad PRD (draft)
개요 = 00\_project\_overview · 사용법 = 10\_metalab\_tutorial
