# sim-service — 트레이너와 엔진을 가르는 RPC 경계

MetaLab 의 학습·검증은 **항상 sim-service 를 거친다.** 트레이너(client)와 엔진 서버(server)가 같은 노드에서
**두 프로세스**로 뜨고 `127.0.0.1` 로 통신한다. in-process 경로는 없다.

```
learning.train (client)  ──control: socket──►  backends/<engine>/server.py (server)
        │                ◄─hot path: 공유 GPU 버퍼 (CUDA-IPC)─┘
   SimServiceVecEnv                                    serve_vec_env
```

- **single-venv**: 트레이너와 서버가 **같은 엔진 uv venv**에서 뜬다(엔진 venv 안에서 `learning` 이 import
  가능하므로 환경 셋업은 추가로 필요 없다).
- **토폴로지는 언제나 same-machine localhost** — 트레이너와 서버가 한 머신에 있다(WAN 경유 없음).
- 구현: `sim/metalab/transport.py`(단일 파일) — `RpcServer`/`RpcClient`/`serve_vec_env`,
  클라이언트 프록시는 `learning/rl/client.py` 의 `SimServiceVecEnv`.

## 핫패스는 GPU 를 벗어나지 않는다

`step` / `get_observations` / `reset` 은 **공유 GPU 버퍼**(CUDA-IPC)로 주고받는다. obs·action 이 매 스텝
GPU→CPU→socket→CPU→GPU 로 복사되지 않는다. 드물게 불리는 cold-path(`attrs`·`seed`·`get_ep_len`·
`set_ep_len`·`apply_curriculum_end`)만 socket 제어채널을 쓴다.

버퍼는 env 빌드 후 **고정 shape** 로 한 번 할당하고 IPC 핸들을 부트스트랩 소켓으로 교환한다
(obs 그룹별 N×dim · action N×A · reward/dones/time_outs/task_success N).

**제약과 fail-loud**
- **same-GPU 전용.** 트레이너와 서버가 같은 물리 GPU 를 봐야 한다 — 스크립트가 `CUDA_VISIBLE_DEVICES`
  상속으로 자동 충족시킨다. 비-CUDA 이거나 IPC 핸들을 못 얻으면 **크게 실패한다**(server 의 CUDA assert +
  share 핸들 assert). 조용한 폴백은 없다.
- torch caching allocator 와 마찰이 있다 — 공유버퍼는 keep-alive 로 관리한다
  (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` 가 필요할 수 있다).

## 왜 이렇게 하나 — 실측

socket 으로 payload 를 나르면 오버헤드가 **num_envs 에 비례해 폭증**한다. CUDA-IPC 는 그 payload 복사를
없애 오버헤드를 in-process 하한 수준으로 되돌린다.

collection 단계만 격리 측정(s/iter, RTX 5090):

```
newton                                     genesis
 N     inproc socket  ipc   soc%   ipc%     N     inproc socket  ipc   soc%    ipc%
 512   0.280  0.340  0.285  +21.2  +1.7     512   0.571  0.637  0.581  +11.5   +1.7
2048   0.351  0.665  0.358  +89.8  +2.1    2048   0.680  1.042  0.703  +53.2   +3.3
4096   0.444  1.107  0.446 +149.4  +0.6    8192   1.046  2.254  1.053 +115.4   +0.6
```

실제 학습 전체(collection + learning)로 보면 L40S 에서 socket 은 +7.9% → +25.9%(num_envs↑),
**IPC 는 ≤ +5%** 로 in-process 와 사실상 같다. 양 엔진에서 동일한 경향이 나왔다.

정확성은 별도로 검증했다 — genesis 실 env 에서 같은 seed·action stream 으로 socket 과 IPC 를 비교해
**`max|Δ| = 0`**(obs 전 그룹 + reward + dones + task_success 완전 일치). newton 은 엔진 자체가 프로세스 간
비결정적(MuJoCo 접촉 합산 순서)이라 socket-vs-socket 도 값이 흔들리는데, IPC 의 발산은 그 밴드 아래였다.

> ⚠️ 이 구조에서 동기화를 틀리면 **조용히 학습이 망가진다**(잘못된 그래디언트). 공유버퍼 쪽을 손댈 때는
> 위와 같은 bit-exact 비교를 반드시 다시 돌린다.

## 알아둘 것

- 학습 스칼라(loss/SR 커브)는 **wandb**, 체크포인트와 리포트는 `logs/rl/` 아래 로컬에 남는다.
- 녹화는 서버 쪽에서 만든다(genesis rasterizer / newton headless offscreen). 둘 다 OpenGL 이라 RT 코어가
  필요 없다.
- 런치패드는 스크립트를 shell-out 하고 로그를 tail 할 뿐이라 이 경계와 무관하다. 서버 stdout 이
  트레이너 로그로 상속되므로 서버가 찍는 것도 콘솔에 보인다.

---
개요 = **00_project_overview** · 엔진 parity = **01_engine_parity**
