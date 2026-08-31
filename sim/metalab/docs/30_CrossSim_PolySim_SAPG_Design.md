Research & System Design Note

# CrossSim = PolySim + SAPG

다중 simulator의 **dynamics diversity**와 다중 policy의 **exploration diversity**를 결합하는 학습 프레임워크에 대한 설계 가능성, 위험요인, 권장 아키텍처, 실험계획 총정리

**PolySim 축**IsaacSim · Genesis · Newton · MuJoCo 등 이질적인 transition dynamics

**SAPG 축**Leader–Follower policy population과 importance-sampled aggregation

**핵심 원칙**Policy 축과 Simulator 축을 직교시켜 원인 분리와 안정성을 확보

인쇄 / PDF 저장
모든 세부사항 펼치기
모든 세부사항 접기

**목차**

[1. 한눈에 보는 결론](#summary)
[2. 두 논문의 핵심](#papers)
[3. 결합 원리](#principle)
[4. 가능한 아키텍처](#design-space)
[5. 권장 구조](#recommended)
[6. 학습 알고리즘](#algorithm)
[7. Actor–Critic 설계](#network)
[8. 혼란·상쇄 위험](#risk)
[9. 환경 수 배치](#allocation)
[10. 실험·Ablation](#experiments)
[11. 진단 지표](#metrics)
[12. 구현 로드맵](#roadmap)
[13. 논문화 포인트](#paper)
[14. 근거 자료](#refs)
[15. 최종 제안](#bottomline)

## 1. 한눈에 보는 결론

**CrossSim은 설계상 충분히 타당하다.** PolySim과 SAPG는 같은 문제를 중복 해결하는 것이 아니라 서로 다른 두 종류의 다양성을 만든다.

PolySim이 만드는 다양성

환경 축

Simulator마다 다른 contact solver, integrator, actuator model, numerical convention으로부터 발생하는 **dynamics-level diversity**.

SAPG가 만드는 다양성

정책 축

여러 follower가 서로 다른 탐색 분포와 trajectory를 만들고, leader가 유용한 데이터를 importance sampling으로 흡수하는 **policy-level diversity**.

> **가장 권장하는 기본형**
> IsaacSim, Genesis, Newton, MuJoCo 각각에서 A·B·C·D policy가 모두 rollout을 수행한다. 즉, **각 policy가 동일한 simulator mixture를 경험**한다.

```
             IsaacSim   Genesis   Newton   MuJoCo
Leader A        ✓          ✓         ✓        ✓
Follower B      ✓          ✓         ✓        ✓
Follower C      ✓          ✓         ✓        ✓
Follower D      ✓          ✓         ✓        ✓
```

**핵심 구현 문장:** 동일 simulator 안에서는 SAPG로 policy 데이터를 aggregate하고, simulator 사이에서는 PolySim mixture loss로 aggregate한다.

## 2. PolySim과 SAPG의 핵심 아이디어

### 2.1 PolySim

- 하나의 policy를 여러 heterogeneous simulator에서 동시에 학습한다.
- 단일 simulator의 inductive bias를 줄이는 것을 목표로 한다.
- Simulator Router가 physics harmonization, API translation, numerical normalization을 담당한다.
- 병렬 multi-simulator training은 sequential training보다 catastrophic forgetting을 줄이고 unseen simulator generalization을 높이는 방향으로 보고되었다.
- 이론적 장점은 simulator mixture의 convex hull이 실제 dynamics에 더 가까울 수 있다는 조건부 직관에 기반한다.

근거: PolySim PDF pp.1–7.

### 2.2 SAPG

- 대규모 parallel environments에서 단일 PPO policy의 batch saturation을 완화한다.
- 환경을 여러 policy block으로 나누어 trajectory 다양성을 만든다.
- Leader는 자신의 on-policy 데이터와 follower의 off-policy 데이터를 함께 사용한다.
- Follower는 보통 자신의 on-policy 데이터만 사용해 다양성을 보존한다.
- Symmetric aggregation과 과도한 off-policy 비율은 성능을 악화시킬 수 있다.

근거: SAPG PDF pp.1–9.

## 3. 결합 원리: 두 축을 직교시키기

CrossSim의 가장 중요한 설계 원칙은 **Simulator axis**와 **Policy axis**가 서로 얽히지 않도록 하는 것이다.

> **얽힌 구조**
> IsaacSim에는 A·B만, Genesis에는 C·D만 배치하면 A와 C의 차이가 policy 때문인지 simulator 때문인지 구분하기 어렵다.

> **직교 구조**
> A·B·C·D가 모두 모든 simulator를 경험하면, policy 차이와 dynamics 차이를 독립적으로 분석할 수 있다.

**Policy별 공통 mixture objective**
J(πi) = Σm∈Sim wm · Jm(πi)

**Leader의 계층적 loss**
Lleader = Σm wm[Lon,m + λoffLoff,m]

이 구조에서 simulator mixture는 PolySim이 담당하고, 동일 simulator 내부의 여러 policy 간 데이터 결합은 SAPG가 담당한다.

## 4. 지금까지 논의한 모든 아키텍처 가능성

A. Simulator-fixed policies

비권장 기본형

```
IsaacSim : A, B
Genesis  : C, D
Newton   : E, F
MuJoCo   : G, H
```

**장점:** 구현이 단순하고 각 policy가 simulator-specific specialist가 될 수 있다.

**문제:** policy diversity와 dynamics diversity가 confounded된다. Importance sampling은 policy mismatch를 보정하지만 transition kernel mismatch는 보정하지 않는다.

**용도:** 본 구조가 아니라 specialist ablation 또는 ensemble baseline.

B. Full factorial / Orthogonal CrossSim

최우선 권장

```
IsaacSim : A, B, C, D
Genesis  : A, B, C, D
Newton   : A, B, C, D
MuJoCo   : A, B, C, D
```

**장점:** 모든 policy가 같은 MDP mixture를 보므로 SAPG 가정에 가장 가깝고 원인 분리가 쉽다.

**주의:** policy 수 × simulator 수로 환경이 나뉘므로 cell당 batch가 너무 작아질 수 있다.

C. Global leader + Simulator-local followers

강력한 대안

```
Leader L : 모든 simulator
F_I      : IsaacSim 전용
F_G      : Genesis 전용
F_N      : Newton 전용
F_M      : MuJoCo 전용
```

**장점:** Leader는 generalist, follower는 simulator-specific explorer 역할을 한다.

**주의:** local follower 데이터는 반드시 같은 simulator loss 항으로만 들어가야 한다.

D. Two-level hierarchy

연구 확장형

```
Level 1: simulator 내부 SAPG
Level 2: simulator 간 PolySim aggregation
```

**장점:** 문제 구조가 명확하고 분산 시스템으로 확장하기 좋다.

**주의:** optimizer synchronization, stale policy, straggler 문제가 커질 수 있다.

E. Symmetric all-to-all aggregation

비권장

모든 policy가 다른 모든 policy의 데이터를 받아 업데이트한다.

**문제:** policy들이 같은 행동으로 수렴해 SAPG의 탐색 다양성이 사라질 수 있다. SAPG ablation에서도 symmetric variant가 불리했다.

F. Sequential multi-simulator + SAPG

보조 baseline

Simulator를 순서대로 바꾸며 SAPG를 적용한다.

**문제:** 이전 simulator에서 획득한 행동을 잊는 catastrophic forgetting 가능성이 있다. PolySim의 핵심 장점인 동시 mixture exposure를 잃는다.

G. Adaptive simulator weighting

후속 확장

고정된 25% 비율 대신 robustness, gradient conflict, real-data similarity에 따라 wm을 조정한다.

**위험:** 특정 simulator를 과도하게 낮추면 dynamics diversity가 다시 줄어든다.

H. Curriculum CrossSim

안정화 옵션

초기에는 1–2개 simulator와 2개 policy로 시작하고, 안정화 후 simulator/policy 수를 늘린다.

**장점:** 초기 optimization noise를 줄일 수 있다.

## 5. 권장 아키텍처

### 5.1 기본형: 4 × 4 Factorial CrossSim

각 policy가 동일한 simulator mixture를 경험

IsaacSimGenesisNewtonMuJoCo

A · LeaderBCD

### 5.2 업데이트 규칙

```
A (Leader):
  - A의 on-policy data: 4개 simulator 모두 사용
  - B/C/D의 off-policy data: 같은 simulator끼리만 사용
  - simulator별 loss를 계산한 뒤 weighted sum

B/C/D (Followers):
  - 각자 4개 simulator에서 수집
  - 자신의 on-policy PPO loss로만 업데이트
  - 서로의 데이터를 직접 사용하지 않음
```

> ⚠️
> **중요:** Genesis에서 수집된 B의 trajectory를 IsaacSim advantage나 IsaacSim critic target에 넣지 않는다. Off-policy correction은 action-policy mismatch를 다루지만 simulator transition mismatch를 자동 보정하지 못한다.

### 5.3 대안형: Global leader + Local followers

Factorial 구조가 계산량상 부담이 크거나 simulator-specific exploration을 더 강하게 유도하고 싶다면 다음을 사용한다.

```
Global Leader L : IsaacSim + Genesis + Newton + MuJoCo
Follower F_I    : IsaacSim only
Follower F_G    : Genesis only
Follower F_N    : Newton only
Follower F_M    : MuJoCo only
```

이때 FG의 데이터는 leader의 Genesis 항에만, FM의 데이터는 MuJoCo 항에만 들어간다.

## 6. 권장 학습 알고리즘

### 6.1 전체 pipeline

1. Policy A·B·C·D가 각 simulator에서 rollout을 병렬 수집한다.
2. Simulator별 reward scale과 termination semantics를 정렬한다.
3. 각 simulator 전용 critic 또는 simulator-conditioned critic으로 value와 GAE를 계산한다.
4. Advantage를 simulator별로 normalization한다.
5. Follower는 각자의 on-policy PPO loss로 업데이트한다.
6. Leader는 동일 simulator 내부에서 follower 데이터를 importance sampling으로 사용한다.
7. Leader의 on-policy:off-policy minibatch 비율을 우선 1:1로 제한한다.
8. Simulator별 leader loss를 wm으로 weighted sum한다.
9. Shared actor backbone을 업데이트한다.

### 6.2 Pseudocode

```
for iteration in training:
    for policy i in {A, B, C, D}:
        for simulator m in {IsaacSim, Genesis, Newton, MuJoCo}:
            D[i,m] = rollout(policy=i, simulator=m)

    for simulator m:
        for policy i:
            V[i,m], A[i,m] = compute_GAE(D[i,m], critic_head=m)
            A[i,m] = normalize_within_simulator(A[i,m])

    # Followers: on-policy only
    for follower i in {B, C, D}:
        L_i = sum_m w[m] * PPO_on(D[i,m], A[i,m])
        update(follower_i, L_i)

    # Leader: simulator-wise SAPG
    L_A = 0
    for simulator m:
        D_off = subsample_equal_amount(D[B,m] ∪ D[C,m] ∪ D[D,m])
        L_on  = PPO_on(D[A,m], A[A,m])
        L_off = SAPG_importance_sampled(D_off, leader=A, simulator=m)
        L_A  += w[m] * (L_on + lambda_off * L_off)

    update(leader_A, L_A)
```

### 6.3 Importance weight 안정화

- Ratio clipping 또는 truncated importance sampling 사용.
- Per-simulator ESS(Effective Sample Size) 모니터링.
- Leader–Follower KL이 임계값을 넘으면 해당 follower의 off-policy 기여를 줄임.
- 훈련 후반에는 λoff를 낮추는 schedule 고려.

## 7. Actor–Critic 네트워크 설계

Actor

Simulator-agnostic 권장

최종 real-world deployment를 위해 actor에는 simulator ID를 직접 입력하지 않는 것이 기본값이다.

πi(a|o) = π(a|o, φi)

Shared backbone + policy-specific latent φi 구조를 사용하면 parameter sharing과 diversity를 동시에 확보할 수 있다.

Critic

Simulator-conditioned 권장

동일 observation이라도 simulator에 따라 미래 return이 다르므로 critic은 simulator 정보를 사용하는 편이 안정적이다.

```
Shared critic trunk
 ├─ V_Isaac(o)
 ├─ V_Genesis(o)
 ├─ V_Newton(o)
 └─ V_MuJoCo(o)
```

### 7.1 가능한 critic 설계

| 설계 | 장점 | 위험 | 권장도 |
| --- | --- | --- | --- |
| 단일 shared value head | 단순 | 서로 다른 return을 평균내 value aliasing 가능 | 낮음 |
| Shared trunk + simulator heads | 안정성·효율성 균형 | head 수 증가 | 높음 |
| Simulator ID conditioned critic | 유연함 | ID embedding에 과적합 가능 | 높음 |
| Policy × Simulator 별도 critic | 가장 분리됨 | 메모리·학습량 증가 | 중간 |

### 7.2 Policy diversity 유지 장치

- Follower별 entropy coefficient 차등 적용.
- Policy-specific latent φi.
- 초기 log-std 또는 action noise schedule 차등.
- Follower끼리 all-to-all off-policy update 금지.
- State visitation diversity를 PCA/representation reconstruction으로 측정.

## 8. 서로의 장점이 상쇄될 수 있는 지점

| 위험 | 증상 | 원인 | 대응 |
| --- | --- | --- | --- |
| Gradient conflict | Simulator별 성능이 번갈아 악화 | 한 dynamics에 유리한 update가 다른 dynamics에 불리 | Per-sim normalization, loss weighting, PCGrad류 ablation |
| Value aliasing | Critic explained variance 저하 | 동일 observation의 return이 simulator마다 다름 | Simulator-conditioned critic |
| Off-policy drowning | Leader 성능이 follower 데이터 추가 후 하락 | Noisy off-policy gradient가 on-policy gradient를 압도 | 1:1 subsampling, λoff 축소 |
| Policy collapse | A·B·C·D의 행동·state coverage가 유사 | Symmetric aggregation 또는 shared gradient 과도 | Leader–Follower 비대칭 유지 |
| Simulator exploit | 특정 simulator에서만 고성능 | Contact solver artifact 활용 | Leave-one-sim-out, unseen sim, real test |
| Cell batch shortage | 학습 variance 증가 | Policy × simulator로 지나치게 세분화 | 2-policy부터 시작, cell당 env 보장 |
| Reward dominance | 특정 simulator gradient만 지배 | Reward scale·termination 차이 | Per-sim reward/advantage normalization |
| Straggler effect | Iteration time이 느린 simulator에 묶임 | Backend throughput 차이 | Async buffer 또는 bounded staleness 검토 |

> ⚠️
> **불확실성:** PolySim 논문은 주로 IsaacGym·IsaacSim·Genesis 조합과 MuJoCo unseen evaluation을 다뤘고, SAPG는 대규모 manipulation 환경에서 검증되었다. Humanoid whole-body control에서 두 기법의 결합 성능은 직접 검증된 결과가 아니므로 실험이 필요하다.

## 9. 환경 수 배치 예시

### 9.1 총 8,192 environments

| 설정 | Policy 수 | Simulator 수 | Cell당 env | 해석 |
| --- | --- | --- | --- | --- |
| PolySim-PPO baseline | 1 | 4 | 2,048 | 가장 안정적인 기준선 |
| CrossSim-SAPG-2 | 2 | 4 | 1,024 | 첫 결합 실험으로 권장 |
| CrossSim-SAPG-4 | 4 | 4 | 512 | 최종 제안형, cell batch 확인 필요 |
| CrossSim-SAPG-8 | 8 | 4 | 256 | 초기 실험에는 과도할 가능성 |

### 9.2 총 16,384 environments

| 설정 | Policy 수 | Simulator 수 | Cell당 env | 권장 |
| --- | --- | --- | --- | --- |
| CrossSim-SAPG-4 | 4 | 4 | 1,024 | 강하게 권장 |
| CrossSim-SAPG-8 | 8 | 4 | 512 | 후속 실험 |

**판단 기준:** SAPG의 장점은 대규모 병렬 환경에서 단일 PPO가 포화될 때 나타난다. 따라서 총 env 수뿐 아니라 **기존 PolySim-PPO가 실제로 batch saturation을 보이는지** 먼저 확인해야 한다.

## 10. 필수 실험 및 Ablation 계획

### 10.1 단계별 핵심 실험

1. **SingleSim-PPO:** 각 simulator 단독 정책.
2. **PolySim-PPO:** 하나의 policy가 여러 simulator에서 병렬 학습.
3. **SingleSim-SAPG:** SAPG 자체가 현재 task에서 이득인지 확인.
4. **CrossSim-SAPG-2:** 1 leader + 1 follower, 모든 simulator 공유.
5. **CrossSim-SAPG-4:** 1 leader + 3 followers, 권장 구조.
6. **Simulator-fixed specialists:** A/B, C/D 식 분리 구조의 ablation.
7. **Global leader + local followers:** specialist exploration이 유효한지 비교.

### 10.2 Ablation matrix

| 실험 | Multi-sim | Multi-policy | Leader–Follower | Per-sim critic | Off-policy ratio | 목적 |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | ✗ | ✗ | ✗ | ✗ | 0 | SingleSim 기준선 |
| A1 | ✓ | ✗ | ✗ | ✓/✗ | 0 | PolySim 효과 |
| A2 | ✗ | ✓ | ✓ | ✗ | 1:1 | SAPG 효과 |
| A3 | ✓ | ✓ | ✓ | ✓ | 1:1 | CrossSim main |
| A4 | ✓ | ✓ | Symmetric | ✓ | 1:1 | Policy collapse 확인 |
| A5 | ✓ | ✓ | ✓ | ✓ | All data | Off-policy drowning 확인 |
| A6 | ✓ | ✓ | ✓ | 단일 head | 1:1 | Value aliasing 확인 |
| A7 | Sequential | ✓ | ✓ | ✓ | 1:1 | Forgetting 비교 |

### 10.3 Evaluation protocol

- **Seen simulators:** 학습에 사용된 각 backend의 성능.
- **Leave-one-simulator-out:** 4개 중 하나를 빼고 학습한 뒤 제외된 simulator에서 평가.
- **Unseen fifth simulator:** 가능하면 training set에 없는 simulator 사용.
- **Real robot:** 최종 zero-shot 또는 최소 calibration 조건.
- **Worst-domain metric:** 평균뿐 아니라 simulator 중 최저 성능을 보고.

> MuJoCo를 학습에 포함하면 더 이상 unseen 평가 simulator가 아니다. 이 경우 제5의 simulator 또는 real robot 평가가 필요하다.

## 11. 반드시 기록할 진단 지표

성능

- Success rate
- Tracking error
- Energy / torque
- Fall rate
- Worst-simulator score

Policy diversity

- Pairwise policy KL
- Action distribution distance
- State visitation coverage
- PCA reconstruction error
- Trajectory novelty

Off-policy health

- Importance weight ESS
- Clipping fraction
- Leader–Follower KL
- Accepted off-policy ratio
- λoff sensitivity

Gradient

- Per-sim gradient norm
- Pairwise cosine similarity
- Dominant simulator ratio
- Actor/critic loss balance

Critic

- Explained variance
- Value error by simulator
- GAE variance
- Return calibration

System

- Steps/sec
- Iteration wall time
- GPU utilization
- Straggler delay
- Communication overhead

### 11.1 실패 신호

Leader–Follower KL이 계속 증가한다

Follower 데이터가 leader에 너무 off-policy가 되어 대부분 clipping되거나 ESS가 낮아질 수 있다. λoff를 줄이고 policy synchronization 주기를 짧게 한다.

특정 simulator의 gradient norm만 매우 크다

Reward scale, horizon, termination 또는 advantage variance가 다를 가능성이 높다. Per-sim normalization과 clipping을 우선 적용한다.

A·B·C·D의 state coverage가 비슷해진다

Policy collapse 가능성이다. Follower entropy, local latent, 독립 minibatch, all-to-all aggregation 제거를 검토한다.

Seen 성능은 높지만 unseen에서 급락한다

Simulator mixture memorization 또는 solver exploit 가능성이 있다. Leave-one-sim-out과 real evaluation이 필요하다.

## 12. 구현 로드맵

> 📌
>
> ### Phase 0 — 정합성
>
> - Observation/action/reward semantics 통일
> - Time step, control decimation, actuator limits 정렬
> - Simulator별 reset/termination 검증
> - Single-action replay 테스트

> 📌
>
> ### Phase 1 — PolySim baseline
>
> - 1 policy × 4 simulators
> - Per-sim critic 비교
> - Leave-one-sim-out baseline 확보

> 📌
>
> ### Phase 2 — SAPG baseline
>
> - 1 simulator × 2/4 policies
> - Leader–Follower, entropy, off-policy ratio 검증
> - PPO saturation 존재 확인

> 📌
>
> ### Phase 3 — CrossSim
>
> - 2 policies × 4 simulators부터 시작
> - 안정화 후 4 policies로 확장
> - Per-sim SAPG aggregation 적용

> 📌
>
> ### Phase 4 — 확장
>
> - Adaptive simulator weighting
> - Gradient conflict mitigation
> - Global leader + local followers
> - Asynchronous rollout

> 📌
>
> ### Phase 5 — Deployment
>
> - Unseen simulator
> - Real robot zero-shot
> - Safety envelope와 torque limit 검증

## 13. 논문으로 발전시킬 때의 핵심 주장

### 13.1 가능한 연구 질문

1. Dynamics diversity와 policy diversity는 단순 합이 아니라 상호보완 효과를 만드는가?
2. Simulator 축과 policy 축을 직교시킨 factorial design이 specialist design보다 일반화에 유리한가?
3. Cross-simulator off-policy aggregation에서 simulator-wise importance sampling이 필요한가?
4. Simulator-conditioned critic이 multi-dynamics value aliasing을 줄이는가?
5. 어떤 simulator weighting이 real-world transfer에 가장 유리한가?

### 13.2 기여점 후보

- **Algorithmic:** simulator-wise SAPG aggregation.
- **Architectural:** orthogonal policy × simulator factorial rollout.
- **Optimization:** per-simulator critic/advantage normalization과 off-policy gating.
- **Systems:** heterogeneous backend를 위한 distributed leader–follower orchestration.
- **Empirical:** unseen simulator 및 real robot generalization.

### 13.3 제목 후보

- **CrossSim:** Factorized Policy and Dynamics Diversity for Sim-to-Real Reinforcement Learning
- **CrossSim:** Split-and-Aggregate Policy Learning Across Heterogeneous Simulators
- **CrossSim:** Orthogonalizing Policy Exploration and Simulator Dynamics for Robust Robot Control

> **논문화 시 가장 중요한 비교:** “환경 수가 많아서 좋아졌다”가 아니라, 동일한 총 환경 수와 동일한 compute budget에서 PolySim-PPO, SAPG, CrossSim을 비교해야 한다.

## 14. 근거 자료

**PolySim: Bridging the Sim-to-Real Gap for Humanoid Control via Multi-Simulator Dynamics Randomization**

핵심 참고 위치: Abstract 및 motivation pp.1–2, system design pp.3–4, theoretical analysis pp.4–5, experiments and ablations pp.5–7.

[로컬 PDF 열기](PolySim.pdf)

**SAPG: Split and Aggregate Policy Gradients**

핵심 참고 위치: motivation pp.1–3, leader–follower formulation pp.3–5, large-scale experiments pp.5–7, ablations and diversity analysis pp.8–9.

[로컬 PDF 열기](SAPG.pdf)

Newton을 포함한 4-simulator 학습은 현재 설계 제안이며, 위 두 논문에서 동일한 조합으로 직접 검증된 것은 아니다.

## 15. 최종 제안

CrossSim의 첫 번째 main architecture는 다음과 같이 두는 것이 가장 설득력 있다.

```
Simulator set = {IsaacSim, Genesis, Newton, MuJoCo}
Policy set    = {A(Leader), B, C, D(Followers)}

모든 policy는 모든 simulator에서 rollout
Followers는 on-policy PPO만 수행
Leader는 simulator별로 on-policy + follower off-policy를 결합
Critic은 simulator-conditioned
Actor는 simulator-agnostic
Simulator loss는 weighted mixture로 합산
Off-policy minibatch는 우선 on:off = 1:1
```

> **핵심 문장**
> CrossSim은 “여러 simulator에서 여러 policy를 돌린다”가 아니라, **동일한 simulator mixture 위에서 policy diversity를 만들고, 동일 simulator 내부에서만 off-policy 데이터를 결합하는 계층적 학습 구조**로 정의하는 것이 좋다.

**권장 실험 시작점:** 8,192 environments 기준으로 PolySim-PPO → CrossSim-SAPG-2 → CrossSim-SAPG-4 순서로 확장한다.

**불확실성:** 더 좋아질 가능성은 충분하지만, 실제 이득의 크기는 PPO saturation, task 난이도, simulator 간 mismatch, critic 설계, cell당 batch 크기에 달려 있다. 따라서 단계별 ablation이 필수다.

CrossSim Design Note · PolySim + SAPG · 생성일: 2026-07-12
