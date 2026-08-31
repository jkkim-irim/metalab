# `tracking` — whole-body motion tracking (WBT) for hammer-lift

Learn a policy that **reproduces recorded hammer-lift motions**: a DeepMimic-style tracking objective
where every reward term compares the live sim against **the reference's recorded value at the same
frame** — never against a fixed goal. Sim-side terms: `sim/isaaclab/envs/hammer_lift/mdp/wbt.py`
(env variants `HammerLiftEnvCfg_WBT` / `_WBTCOLLECT`); driven by
`learning/train.py --trainer rl --experiment tracking --wbt`. Algorithm: PPO (the in-repo runner).

## Pipeline

```
dexblind lift policy (blind, LSTM)                 learning/rl/tracking/
  └─ rollouts in the WBT-collect env  ──────────►  collect_trajectories.py   (keeps SUCCESS episodes)
       traj_*.npz  (tracking_state / privileged / action / setup per episode)
  └─ reference.py  --trim_head 2 --min_lift_cm 8 ─►  ref_*.npz  (state / contact / setup / action)
  └─ track_train.sh  (EXPERIMENT=tracking VARIANT_FLAG=--wbt) ─► PPO in HammerLiftEnvCfg_WBT
  └─ track_eval.sh / wbt_replay.sh  ─► per-env videos + breakdown → S3 reports + W&B
```

## Why a learned tracker — and not just replaying the recorded actions

The references store the source policy's actions (`action [T,18]`), so the obvious question:
why not replay them? **Measured answer: open-loop replay drifts even in the same simulator,
starting from the recorded setup** (~2.1/5 tracking reward, lifts not reproduced — it's why the
reference *videos* are rendered by kinematic pose-writes, not by replaying actions). Three layers
of perturbation force a feedback policy:

1. **Intrinsic divergence — sufficient by itself.** Actions are PD position targets; the torque
   they produce depends on the current state. Contact-rich dynamics (five frictional fingertips +
   a free rigid body) amplify solver warm-start, contact-ordering and float-level differences
   exponentially: one stick↔slip flip a frame early and every later recorded action addresses a
   world it wasn't recorded in. No noise needs to be injected for playback to fail.
2. **Initial-state gap.** RSI reproduces the recorded `setup` approximately, never exactly — the
   setup snapshot is captured pre-settle, and env↔reference hammer shapes must be matched (the
   `variant` field exists for this). Any ε at t=0 is seed capital for layer 1.
3. **Deliberate training noise — the training aid, not the obstacle.** Rollouts act through PPO's
   exploration noise and noised observations, so the policy constantly lands slightly
   off-reference and learns to steer back. Evals turn both off (mean actions, noise-free obs).
   The one perturbation the original lift training had — the external force on the lifted
   hammer — is intentionally OFF here, mirroring collection.

**Measured magnitudes** (protocol and lineage in `EXPERIMENTS.md`):

- **Even the source policy cannot replay itself: 0.8668.** The canonical-826 references are the
  source's own successful episodes, yet the source re-run from those same inits under the
  noise-free protocol fails 13% of them. If trajectories were reproducible this would be 1.0 by
  construction — the 13% is the intrinsic sensitivity of contact-rich rollouts, with no learning
  involved.
- **ε at t=0, at two magnitudes.** A ~15 cm / 9° spawn offset (the RSI init bug) scored the same
  tracker checkpoint at 0.0703; fixing the spawn to 0.21 cm p50 residual scored it 0.7734. The
  divergence layer turns centimeters at t=0 into success-or-failure.
- **Batch composition alone forks outcomes.** Half of the champion tracker's residual failures
  succeed on a re-roll with checkpoint, protocol and seed unchanged — only which envs run
  alongside differs (solver scheduling). Failures are knife-edge events, not a fixed hard set.
- **The residual failure mode is contact flicker**, not kinematic error: near-reference tracking
  (~3 cm / 15°) broken by a one-frame fingertip drop inside the gate's 10-step hold window —
  grasp maintenance is mode-switching stabilization, the regime playback dies in fastest.
- **Feedback-controller fingerprints move the score.** Start diversity beats start fidelity
  (random starts over recorded starts ×1.56–1.72; the `WBT_NO_RSI_PROB` dose curve has an
  interior optimum at 0.25), and cold-start deaths (the zero-hidden first action tripping the
  force guard) anti-correlate with SR — properties of a stabilizing controller, meaningless for
  a tape.

So the references solve **exploration** (where in state space a solution lives) and RL still has
to solve **control**: a stabilizing feedback law — a basin of attraction — around each of the
demonstrated paths, robust to the policy's own past errors, one network across all references and
their per-episode mass/friction draws. Phase-RSI exists to make training visit that neighborhood
(episodes start mid-reference, including imperfectly-reached states), and the success gate's
consecutive-hold requirement lives in exactly the regime where open-loop playback dies fastest:
sustained micro-correction against gravity and slip. The deployed actor additionally never
observes the object (privilege boundary: proprioception + reference goal only) — it must infer
the hammer's state through contact, which playback would not and could not do. The recorded
actions are kept for provenance and replay tooling only — they are not a training target.
End state of the comparison: tracker 0.8729 vs source 0.8668 on the same protocol, with
nearly disjoint failure sets (Jaccard 0.068) — the residual gap is the task's knife edge,
shared by both policies, not a tracking deficit.

## The tracked state (`tracking_state`, [T, 59], 50 Hz)

Single source of truth in `mdp/wbt.py`; everything chest-relative (`Chest_Origin_Link`, xyzw quats):
`hammer pose 0:7 | hammer vel 7:13 | palm pose 13:20 | palm vel 20:26 | 5 fingertip pos 26:41 |
18 arm+hand joints 41:59`. The palm keypoint reads `R_Wrist_Pitch_Link` — **not** `R_Palm_Link`, which
hangs on a fixed joint that the physics merges away (reads return garbage constants; found via the
per-kernel breakdown). References additionally carry `contact [T,5]` (fingertip↔hammer flags, from the
recorded privileged obs), `setup [84]` (exact initial env state for RSI), `action [T,18]` (provenance +
replay tooling).

## Reward — per-frame comparison against the reference timeline (flat, object-dominant)

$$r(t) = 1.5\,k_{objpos} + 1.0\,k_{objori|grip} + 1.0\,k_{c} + 0.5\,k_{hold} + 0.5\,k_{kp} + 0.5\,k_{joint}$$

Each kernel is 0..1 (Gaussian `exp(−e²/σ²)` except the contact terms); max r = 5. Registered as
**separate reward terms** so each logs its own `Episode_Reward/wbt_*` curve (an aggregate hides which
component fails — that's how the dead palm read was found). A hierarchical variant (object terms ×
`k_c`) was measured worse and reverted — see revision rows 7–8.

| term | w | σ | live quantity vs reference at frame t |
|---|---|---|---|
| `wbt_obj_pos` | 1.5 | 0.08 m | hammer position vs the reference hammer's (which **rises** through the lift) |
| `wbt_obj_ori` | 1.0 | 0.2 rad | hammer orientation vs the reference's — scored only while the reference grips (constant 1.0 pre-grasp) |
| `wbt_contact` (`k_c`) | 1.0 | — | grip quality = **per-finger** match fraction of the reference contact pattern (3/5-finger hold = 0.6) |
| `wbt_hold` | 0.5 | — | sustained grasp: `min(consecutive 5/5-contact steps, 10)/10`, scored inside the reference's own ≥10-frame full-grasp window (neutral outside) |
| `wbt_keypoint` | 0.5 | 0.1 m | palm + 5 fingertip positions vs the reference hand path |
| `wbt_joint` | 0.5 | 0.5 rad | 18 joint angles vs the reference joint trajectory |

### Reward design revisions & why (each driven by a measured failure)

| # | revision | evidence that forced it |
|---|---|---|
| 1 | **Object-dominant weights + task-scaled σ** (obj_pos w2.0, σ 0.3→0.08 m) | equal weights + locomotion σ let the policy score ~2.5/5 by miming the arm with the hammer **never touched** (σ=0.3 m pays ~85% for a hammer left on the desk, ~12 cm off) |
| 2 | **Contact term added** | nothing else demanded the hand actually grip; the object terms give almost no action-gradient until force closure exists |
| 3 | **Episodes end at reference end** (`wbt_ref_end`) | ~80% of a 250-step episode was a static held tail on a ~55-frame reference — diluted experience |
| 4 | **Reference quality filter + head trim** (≥8 cm lift, first 2 frames dropped) | the source policy's lenient success gate let ~4 cm nudges count as "lifts"; post-reset obs are stale for ~2 frames (garbage targets, visible pose jump) |
| 5 | **obj_ori gated to reference-grip frames** (w 1.0→0.5, freed → contact) | an untouched hammer collected ~0.54 of the ungated kernel — free credit diluting the grasp gradient; orientation only discriminates post-grasp (dangling/swinging grips) |
| 6 | **Per-finger contact match** (was `exp(−Σ(Δ)²)`) | a functional 3/5-finger hold scored `exp(−2)` ≈ 0.14 — partial grips got no gradient toward full ones |
| 7 | **Hierarchical soft gate** (object terms × `k_c`) | policies reached near-perfect kinematics (joints 7°, keypoints 3–4 cm) while the hammer stayed on the desk: object reward must be *reachable only through* the grasp, and the product pays extra for closing the grip while on-reference. Soft (multiplicative), not a hard switch — contact flags flicker on marginal grips and a hard gate puts reward cliffs at the exact boundary being learned |
| 8 | **Hierarchical gate REVERTED** (object terms back to flat; `contact_gate` stays available as a param) | measured worse on the KPI in two independent 5000-iter runs: best hierarchical checkpoint scored ~half the flat config's eval/SR on the identical 64-reference protocol — pre-grasp, the product zeroes the object-tracking gradient exactly when the policy most needs to learn *where the hammer should go*. The mime failure of row 7 is instead closed by the success gate itself (full 5-finger contact is a claim condition) and by references that demonstrate full grips |
| 9 | **obj_ori sharpened, obj_pos trimmed** (σ_ori 0.4→0.2, w_ori 0.5→1.0, w_pos 2.0→1.5) | gate autopsy on the canonical-reference tracker: GIVEN a full 5-finger grip, rot<0.5 rad was met only **27%** of steps (the hammer tilts in the grasp) while position was met 100% — orientation was the #2 streak killer (28% of hold breaks) yet earned 10% of the reward at a σ that barely penalizes a 0.5–0.7 rad tilt. Position is non-binding at the 0.2 m gate and already tracked to ~8 cm |
| 10 | **+15-frame frozen-target tail on training episodes** (`wbt_ref_end` `train_tail`; eval keeps its longer tail) | canonical references end at their own claim frame, so training episodes ended exactly where the gate's 10-step hold must happen: the hold was unrehearsable, best streaks truncated at the episode boundary (p10 of frames-remaining = 1), contact flicker killed 47% of holds, and in-train claims could barely register. The frozen final frame is a full-grasp goal-pose target — tail steps train precisely the sustained hold the gate scores |
| 11 | **Sustained-grasp term** (`wbt_hold` w 0.5, funded by instantaneous contact 1.5→1.0): ramp `min(streak,10)/10` of consecutive full-5/5-contact steps, scored inside the reference's own ≥10-frame full-grasp window | after the row-9/10 fixes, contact FLICKER became the dominant hold killer (59% of streak breaks; rot fell to 12%): the instantaneous match pays a grip that drops a finger every few frames almost the same as an unbroken one — only a trailing-streak ramp makes the 10-consecutive requirement visible to the reward. Still pure tracking: every canonical reference demonstrates the sustained window by construction |
| 12 | **Partial-credit hold blend** (`wbt_hold` `partial_weight`/`partial_power` via `WBT_HOLD_PARTIAL_*`, default off): in-window value `(1−w)·ramp + w·(n_contacts/5)^p` | the pure ramp pays EXACTLY 0 for a stable 4/5 carry — zero gradient toward the 5th finger, and the measured terminal failure mode IS the 4-finger carry (per-env gate trace: lifted/pos/rot/palm all green, 5-finger contact met on ~1% of frames; the thumb never loads). `p=4` keeps a steep 5th-finger premium (0.8^4≈0.41 → 1.0), the blend keeps the unbroken-streak incentive dominant, and a sustained 5/5 hold still earns exactly 1.0 |
| 14 | **OmniRetarget-shaped reward redesign (2026-07-11, user-directed)**: body tracking = OBJECT-ANCHORED keypoints (demonstrated hand↔object offsets composed with the LIVE hammer pose) w1.0 + joints w0.5 + joint velocities w0.25 (new); object pos w1.5 + gated ori w1.0; action-rate −0.002 (new). REMOVED: contact match, hold streak (+partial blend), task-success bonus | absolute keypoint targets chase the recorded ghost once contact chaos diverges live physics from the recording — the reward then pays the hand to miss the real object. Anchoring preserves the interaction (OmniRetarget's core logic, collapsed to one rigid object) and replaces explicit contact-pattern shaping; their remaining terms are structural here (to-limits actions can't violate joint limits; destructive contact is termination-owned). RISK, stated: v1's mime failure was closed by contact terms — anchored fingertip targets must now carry that signal |
| 13 | **Phase-guard RSI** (`wbt_phase_rsi` `phase_guard` via `WBT_PHASE_GUARD`, default off): phase sampling capped at the reference's sustained-hold window start | in-window starts SPAWN already-held (hammer grasped at goal) — free reward. Hold-10 references keep those windows narrow, but on hold-30 references they are wide enough to dominate the phase mixture, and the policy exploits the free holds instead of learning approach+grasp (measured: full-task SR collapsed to ~0.05). The guard makes long-hold sets trainable; in-window experience still arrives by flowing in from pre-window starts |


Non-reward levers with the same evidence discipline: **phase-RSI** (contact kernel flat for 750+ iters of
frame-0 starts → episodes start at sampled phases incl. grasped states) and the **reference-conditioned
critic** (the value function couldn't know which motion/phase an env tracks). Empirically (t=0 evals,
32 episodes): mime → 0/32 → 1/32 → **~25% full lift reproduction** across these revisions.

**Peak harvesting, not peak fine-tuning.** The entropy bonus drives an explore/collapse arc: the
learned action std grows past the SR peak and sampled rollouts degrade the policy afterwards. The
supported path to the best artifact is dense checkpointing (`SAVE_INTERVAL=50`) + full-dataset SR
probes + the watcher's confirmed-decline kill; selection is always by t=0 eval. Fine-tuning a
resumed peak is measurably unstable — with entropy zeroed, with the poisoned std parameter reset,
and with the LR pinned small, continued PPO still walked a 0.36 checkpoint down within 150-800
iterations (the peaks are sharp optima; nearby checkpoints vary several-fold). Improving past a
harvested peak needs objective-level changes, not in-place polishing.

Two knobs schedule the arc instead of riding it (both annotated in the run's W&B config under
`wbt_sched`; `learning/trainer/rl_trainer.py`):

- `WBT_ENTROPY_SCHED="e0:e1:N"` — linear `entropy_coef` anneal `e0→e1` over the first `N` iterations.
  Flat 0.005 ignites grasp discovery but fuels the std spiral; flat 0.001 never spirals but plateaus
  ~2.5× lower — the anneal takes the ignition, then removes the fuel.
- `WBT_STD_MAX` — hard ceiling on the learned action std, clamped after every update: caps the spiral
  mechanism itself so an entropy bonus can stay on without noise-death.

Every run's recipe is self-documenting on W&B: display name from `WBT_RUN_TAG`, free-text
`experiment_notes` from `WBT_NOTES`, reward knobs under `wbt`, schedule knobs under `wbt_sched`,
dataset under `reference_s3`, eval protocol (gate values included) under `eval_cfg`.

**Distill-then-continue** (`distill_collect.py` → `distill_bc.py` → `--init_actor_ckpt`;
`learning/scripts/aws/distill_continue.sh` drives all three): BC-clone a harvested peak's MEAN
behavior into a fresh actor, then continue PPO from the clone — every component the direct resumes
carried poisoned (exploded std, stale Adam moments, a critic fit to the collapsed late policy) is
replaced fresh, while the behavior stays. Collection rolls the teacher on the TRAINING-style env
(phase-RSI, obs noise) with optional DART action noise — noised execution, mean labels — so the
student learns recoveries around the teacher's tube. `WBT_ACTOR_FREEZE=N` holds the cloned actor
fixed for the first N iterations while the fresh critic warms up — implemented as GRADIENT HOOKS
that zero the actor's grads (a `requires_grad`-off actor crashes the recurrent PPO update: cuBLAS
`INVALID_VALUE` in the batched forward); pin the LR alongside (frozen actor → kl≈0 → the adaptive
controller ramps to its ceiling). **The obs normalizer is part of the contract**: `--init_actor_ckpt`
also freezes the actor's `EmpiricalNormalization` (`until=0`) — BC students train against the
identity stats (only PPO's rollout path calls `update_normalization`), so the trainer's first
rollout would install live stats and rescale every input under weights that never saw scaling.
Measured (v15c/v15d): clone 0.25 → 0.00 at probe@0 with byte-identical weight tensors — the
normalizer buffers alone had shifted (std → ~17, one 24×1024 rollout).
Measured clone-gate scaling (64-ep evals, teacher 0.345): BC val MSE 0.50 → SR 0.094; 0.37 → 0.188;
0.21 → 0.250 (rounds: +2nd 512×400 collection at noise 0.25, then 120 epochs cosine LR) — BC
fidelity transfers steeply, so cheap BC polish buys real SR before any GPU goes to the continue.

**Asymmetric, reference-conditioned critic:** the actor sees `wbt_goal` (reference state + contact
pattern + phase); the critic additionally gets `wbt_goal` **and** `wbt_errors` (the reward's own kernels +
error magnitudes) on top of the privileged state — the value function predicts the tracking reward, so it
is fed the reward's components rather than made to re-derive them. (Standard asymmetric-critic practice
for motion tracking; without it the critic can't know which motion/phase an env is tracking.)

**Privilege boundary (enforced, two chokepoints):** the actor group is the deployed policy's complete
input — proprioception + task input (reference goal) ONLY; live object/sim state is critic-only. Term
placement is validated against `_ACTOR_DEPLOYABLE_TERMS` at every env build (`env_cfg.py`,
`_validate_actor_privilege`), and the trainer refuses any actor wired to a group other than `actor`
(`rl_trainer.py`). Adding a non-deployable term to the actor fails loudly at launch; extending the actor
legitimately requires editing the allowlist — a deliberate, reviewable act. Runtime confirmation appears
in every log: `Resolved observation sets: actor : ['actor']`.

**Reading the curves (two non-obvious semantics):**
- `wbt_contact` reads **1.0 pre-grasp** — the reference isn't touching yet either, so "match" is perfect;
  it drops exactly at the reference's grip onset if the policy doesn't follow. The video overlay draws
  the reference grip fraction (white `refgrip` line) to make this visible.
- `wbt_obj_pos/ori` **decay even if the hammer is never touched** — the live hammer is static but the
  reference target moves away with the phase (the chest measurement frame is static; verified: an
  untouched hammer's chest-relative y is constant to 4 decimals across frames).

## Episodes, RSI, and phase-RSI

- **RSI**: each env is reset to its assigned reference's exact recorded `setup` (hammer pose+vel, all 33
  joints, root z, mass, frictions) — the reference is reachable from the start state by construction.
- **Episode ends at the reference's last frame** (`wbt_ref_end`, time_out=True): without it, a 250-step
  episode tracking a ~55-frame reference spends ~80% of experience on a static held tail.
  `wbt_bad_tracking` terminates early at >0.25 m object drift.
- **Phase-RSI** (`wbt_phase_rsi`): a fraction of episodes start at a **uniformly sampled phase t** of the
  reference — including mid/post-grasp states with the hammer already in the hand (state written from
  `state[t]`: hammer world pose+vel via chest FK, joints, finite-difference joint velocities) — the rest
  start at t=0. Rationale: grasp closure is a needle-in-a-haystack for exploration (observed: contact
  kernel flat for 750+ iters); starting inside the grasped state lets the policy experience and maintain
  it, and learning propagates backward. Evals/replay (`--fixed_refs` / `--replay`) always start at t=0.

## Metrics & logging conventions

- `Episode_Reward/X` (framework convention) = episodic sum × dt ÷ **max_episode_length_s (5 s)** — a
  time-normalized rate, NOT the kernel mean. With ~1.2 s episodes each term's ceiling ≈ w × 0.23 (and
  lower with phase-RSI, whose episodes are shorter). For kernel means read the eval `WBT_BREAKDOWN`.
- Evals print `WBT_BREAKDOWN` **mean** (per-kernel 0..1 + hammer cm/deg, keypoint cm, joint deg — tracking
  fidelity) and **final-frame** stats with `track_success` = final hammer error < 5 cm (the lift outcome).
  Mean = how faithfully it tracks; final = whether the lift is reproduced.
- `eval/SR` (the KPI) = the hammer-lift task's SUCCESS GATE — the project's fixed success definition,
  independent of any curriculum level (`TASK_SUCCESS_*_GATE` tunables: full 5-finger contact + lift +
  goal pos/rot + palm distance, held `TASK_SUCCESS_HOLD_STEPS_GATE` consecutive steps). The gate state
  machine is owned by the `task_success` termination term (WBT registers it claim-only, so episodes
  still run to the reference end); the same gate scores the source policy, so tracker SR and source SR
  are directly comparable. Claims are latched server-side across the in-step auto-reset; every eval
  cross-checks the claim count against an independently computed all-conditions streak count
  (`GATE_VERIFY`).
- `Train/task_success_rate_t0`, `Train/contact_match_finger_1..5`, `Train/palm_dist_cm` — gate
  statistics over the TRAINING distribution, logged every iteration (dense trend + per-finger
  diagnosis). The success rate counts **zero-start episodes only**: phase-RSI episodes can begin
  already grasping at the goal, where the training tail makes a claim near-free — mixing them in
  makes the rate a phase-mixture artifact that moves with `zero_start_prob`. The t0 cohort is the
  training-distribution analog of eval/SR (full approach→grasp→lift→hold, but stochastic actions and
  the short `train_tail`), so it reads as a noisy lower bound of the eval curve. The in-run `eval/SR`
  (val hook, eval protocol, small n) and the 32-episode probes carry the calibrated level.
- Videos carry the breakdown as **curves** (per-term time-series up to the current frame, including the `wbt_hold` ramp — a sawtooth there is contact flicker on screen);
  W&B uploads are 5×-slowed re-encodes (no player speed control there); S3 reports slow client-side.

## Reports / visual validation

- `simrl_tracking_trajectories.html` — the reference set, kinematically replayed (RSI to frame 0, then
  recorded pose written + FK each frame; **no physics, zero drift** — open-loop *action* replay drifts
  and is not used for visualization).
- `simrl_tracking_crosscheck.html` — row i pairs reference i with the policy tracking that same
  reference (`--fixed_refs`: env i → ref i), same RSI start, breakdown overlay burned in.
Both on `s3://wirobotics-internal/chrisryu/sim_rl/reports/` (CloudFront, SSO-gated).
