# Tracking experiments — the trail

Every training/eval run behind the hammer-lift WBT tracker, in order, with what changed, what it
scored, and what it taught. Companion to `README.md` (which documents what the method IS; this file
documents how we got here). W&B project: `wiin2-wirobotics-inc/chrisryu-simrl`.

**KPI**: `eval/SR` under the canonical success gate (`TASK_SUCCESS_*_GATE`: 5/5 fingertip contact +
pos<0.2 m + rot<0.5 rad + palm<0.084 m, held 10 consecutive steps), noise-free, deterministic
references (env i → ref i), t=0 starts, eval hold-tail, latched claims, `GATE_VERIFY`-checked.
Since 2026-07-09 every run is self-describing on W&B: display name from `WBT_RUN_TAG`, recipe knobs
under `wbt`/`wbt_sched`, dataset under `reference_s3`, free-text `experiment_notes`, eval protocol
under `eval_cfg`.

## PROTOCOL CORRECTION (2026-07-10) — read before the numbers

Every WBT policy eval before this date measured the FIRST post-spawn wave, in which all envs carry
the attach-time default `_ref_idx = 0` and a non-RSI start (the env is built before references
attach, and no policy-eval path issued a reset afterward — caught via the video header stamps
reading `ref 0000` on every env). **Pre-fix `eval/SR` therefore means "chase reference 0 from a
default reset", not "reproduce the reference set."** Replay reports (own post-attach reset) and
the dexblind source baseline (no references) were correct. Relative checkpoint standings survive
(every eval was identically broken); absolute numbers do not. Fix: server post-attach reset.

First fixed-protocol measurements (826 refs, RSI'd t=0 starts, noise-free, canonical gate):

| checkpoint | ref-0 proxy (pre-fix) | TRUE SR | true failure profile |
|---|---|---|---|
| v14b `model_2700` | 0.345 | **0.0944** own-canonical / **0.0847** cross-matched | 30% of episodes die <15 steps (fingertip-force dominant) |
| v17 `model_2050` (variant-matched training) | ≤0.140 probes | **0.0920** own-matched (arc 0.039@1400 → 0.092@2050 → 0.036@2999) | under-15 count anti-correlates with SR across its checkpoints (441 dead → 0.041; 259 → 0.092) |
| v9b `model_1600` | 0.345 / 0.311 | **0.0230** | 70% die <15 steps, median length 5 — cold-start first-action jerk |
| source dexblind | — | **0.83** (its own task protocol; unaffected) | — |

Two corollaries: variant-matched training (v17) bought no clear edge — v14b's CROSS-set 0.085
≈ v17's own-set 0.092; and the pre-fix "cross-set transfer cost" (0.345→0.141) was mostly a ref-0
artifact — true cross degradation is ~10%. Cold-start robustness is the SR differentiator
(→ `WBT_ZERO_START_PROB`, v18).

The champion margin is real and larger than the proxy showed (4×, not +0.03): the scheduled
recipe's robustness advantage dominates under RSI'd multi-reference starts. The dominant true
failure is the zero-hidden LSTM's first-action jerk at an RSI'd state tripping
`table_fingertip_contact_force_exceeded` within 1–5 steps.

## THE PROTOCOL CONTROL (2026-07-11) — the decisive experiment

jkkim's task policy (`hammer_lift_2026-06-30-12-50/model_1600`, trained with the task objective,
zero references) evaluated under the tracker's EXACT protocol (`--rsi_play`: 826 reference-RSI'd
starts, fixed refs, canonical gate): **SR 0.8838** (730/826; 677 distinct references claimed;
median episode 60 steps; under-15 deaths 7% vs the trackers' 19–30%).

Consequences: (1) the protocol is fair — RSI'd starts are, if anything, easier than native random
inits (0.88 vs 0.83); (2) the trackers' cold-start deaths are their own first-action jerks, not the
states'; (3) **82% of references are claimable from their starts** — vs the trackers' 112-ref
union — so reference difficulty was never the wall; (4) same env, same starts, same gate:
task-objective 0.88 vs tracking-objective ≤0.12 — the objective is the entire gap, measured in one
experiment.

## THE STRAITJACKET RESULT (2026-07-11 night) — the reference conditioning suppresses claims

Definitive ref-0 measurement (champion, 256 trials per condition, refs {0,1} pair-dir): from
**random starts SR = 0.2578** (the old 0.345-era condition reproduces); from the demonstration's
**own recorded start SR = 0.0000** — zero in 256 full-length episodes. The source policy claims
0.88 from those same recorded starts. Mechanism: at an RSI start the tracker begins at zero error,
phase-locked to the demonstrated timeline, and must reproduce the grasp at the reference's exact
frames — where contact chaos compounds; from a random start the phase-lock is immediately broken
and the policy falls back on its learned general grasping competence, claiming 4× better. The
conditioning suppresses skill the policy demonstrably has (also retro-explains v14 transfer >
own-set). Levers: object-anchored guidance (v26, spatial half) and GOAL-DROPOUT training (v27
candidate, phase half). Ops: small-ref-dir evals need ≥2 refs; ref files are `ref_%04d.npz`.

## THE 0.345 DECOMPOSITION (2026-07-11) — the old number, fully accounted

Same champion checkpoint, conditions varied one at a time (2k matched set): ref-0-only from random
starts (the broken eval's condition) = 0.345; ALL 2000 refs from random starts = **0.1325**
(per-reference spread — ref 0 is an easy target, ~2.6× the mean); all 2000 refs from RSI'd recorded
starts = **0.0847** (the recorded-start tax, ×0.64 — trackers are mildly brittle at the demos' own
starts; the source policy finds the same starts easy, 0.8838). 0.345 → 0.085 reconciles with no
residual. Eval standard from here: the full 2000-reference matched set. (Also fixed en route:
`num_envs=1` eval sims are broken — deterministic 2-step bad_tracking, diagnostics must use ≥2
envs; and eval env tunables now follow `--experiment` — tracking-side knobs previously never
reached eval envs via the static dexblind import.)

## Scoreboard (canonical gate — pre-fix numbers below are the ref-0 proxy; see the correction)

| policy | eval/SR | note |
|---|---|---|
| source: full-curriculum dexblind (`jkkim .../hammer_lift_2026-06-30-12-50/model_1600`) | **0.83** | the ceiling |
| **v14b `model_2700`** ([ha4ivju9](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/ha4ivju9)) | **0.345** | **champion** — scheduled recipe, no collapse, plateau tail; leads row-identical |
| v9b `model_1600` ([3iz6wgss](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/3iz6wgss)) | 0.345 / **0.311** | first record (on its min-lift-filtered 826) / on the current slice — lucky-tail peak, knife-edge neighbors |
| v14c `model_1400` ([wphzj5o7](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/wphzj5o7)) | 0.277 | anneal-length lesson (below) |
| best BC clone of v9b (student3, 64-ep gate) | 0.250 | distillation groundwork |
| any tracker of partial-curriculum references | 0.00 | structural — reference feasibility |

Row-parity (RESOLVED): v9b's 826 references were built with `--min_lift_cm 8` (826/1000 kept);
2026-07-09 rebuilds keep all 1000 (`--trim_head 2` only), and probes use the first-826 slice.
On that identical slice: **v14b 0.3450 vs v9b 0.3111** — the scheduled recipe leads +0.034
row-for-row, and its 0.345 was earned on the harder unfiltered slice.

## Era 1 — original (lenient) gate: reward/method iteration

All SRs below used the curriculum's lenient START gate (contact trivial, pos 0.1, rot 1.0, hold 30
play-style) — **obsolete after the canonical pivot, not comparable** to the scoreboard. Each ran
~1000–5000 iters, PPO, 1024 envs, LSTM-512 + MLP actor (never grown).

| exp | run | recipe delta | result | takeaway |
|---|---|---|---|---|
| (prehistory) | `8x5mxfr3` | object-pose-only DeepMimic (`dexdeepmimic`, since removed) | reward 0.39/2.0 plateau | full-state WBT replaces it |
| v1 | `o37837me` | HoloSoma kernels, equal-ish weights, broad σ | 2.5/5 reward, hammer never touched | broad σ pays the mime; object terms must dominate |
| v3 | `x56pbxyw` | object-dominant σ0.08 w2 + contact term + ref-end episodes + ≥8 cm ref filter + palm fix | approach tracks (keypoint 6.8 cm); contact flat 750 iters; 0/32 lifts | grasp closure is a needle — exploration never finds it from t=0 |
| v4 | `qcynplqh` | + phase-RSI (zero_start 0.3) | holds work, skill flows to t=0; still 0/32 | phase-RSI necessary, not sufficient |
| v5 | `vvxzddnp` | + reference-conditioned critic; obj_ori contact-gated w0.5; contact w1.5 | kinematics near-perfect; first from-scratch lift 1/32 | asymmetric critic unlocks value learning |
| v5-long | `obodkzwo` | v5 × 5000 iters | model_800–1000 = 8/32, then collapse to 2/32 by 5000; best `model_1000` = 0.34 old-gate (11/32, [812zu60o](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/812zu60o)) | select by t=0 eval, never last ckpt; long PPO destabilizes; grasp-discovery variance is large |
| v6 | `wx943rgn` | hierarchical soft-gated reward (object × contact kernel) | 0.19@1600 → 0.31@2400, watcher-killed on a confounded proxy | watch/kill on the KPI only; freeze eval code mid-run |
| v6-resume | `ome1odkb` | resume v6 `model_2400` | plateau 0.09–0.28 | resumes don't recover peaks (first hint) |
| v6b | `trjomqhk` | v6 config from scratch, SAVE=100 | peak 0.19@2700; 64-ref eval 0.25 vs v5's 0.53 | hierarchical gating loses on the KPI across two runs — REVERTED (README row 8) |

## Canonical-gate pivot (2026-07-08, jkkim review)

The measured gate was the curriculum's lenient start values. Canonical = jkkim's fixed conditions
(scoreboard header). Under it: the old partial-curriculum source lifts with a partial grip —
**0/888 old references ever sustain a 10-frame full grasp → any tracker of them scores 0.00
structurally.** New references collected from his full-curriculum checkpoint under the canonical
claim (dataset `collect-6893081`, 1000 kept of 1160 episodes). Gate ownership moved into the
`task_success` termination term (claim-only for WBT); every eval self-verifies (`GATE_VERIFY`).

## Era 2 — canonical gate

| exp | run | recipe delta | eval/SR (own set) | takeaway |
|---|---|---|---|---|
| v7 | [f4o4mg1u](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/f4o4mg1u) | v5 config × canonical refs | 0.17 (`model_1600`; videos [zfeny3tt](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/zfeny3tt)) | tracks worse kinematically than v5 yet scores — the gate is all about the demonstrated full grasp |
| v8 | `dn0p3uzm`/`o3kyrzhn` (paired launch) | + obj_ori sharpened σ0.2 w1.0, obj_pos w1.5 (row 9); +15-frame train tail (row 10) | 0.30 (64-slice, pre-refactor) | autopsy-driven reward surgery pays |
| v9a | `meuzf1t0` | aborted early relaunch | — | — |
| **v9b** | [3iz6wgss](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/3iz6wgss) | + `wbt_hold` sustained-grasp ramp (row 11), contact 1.5→1.0 | **0.345** (`model_1600`; probe 0.362@1600; videos [3a32wo5o](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/3a32wo5o)) | the record — but a sharp transient peak inside an explore/collapse arc |
| v10 | `k5968ujr`/`qgz4znyv` (paired) | hold-30 references (n2000) | 0.138, culled | free in-hold phase-RSI starts hijack the mixture (README row 13 rationale) |
| v11 | `zmv992i0` | entropy 0.001 flat | 0.139 plateau | no spiral but no ignition — explore beats safe |
| v12 | `wy8w1e2b` | full package on hold-30 n2000 | 0.138 | confounded by data; variance suspected |
| v13 | `iz8ps4nu` | exact v9b repro (control) | 0.213 | run-to-run lottery dominates config deltas; v9b was a lucky tail |
| v13b | [e6k9cghn](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/e6k9cghn) | resume v9b peak, entropy 0 | 0.318→0.065 in 150 iters | σ≈3 rollouts erode the intact mean |
| v13c | `wtj2vc0u` | + `WBT_RESUME_STD=0.4` | held at load → 0.20 plateau | std reset insufficient |
| v13d | `k7u84gdc` | + `WBT_LR=1e-5` pinned | 0.133→0.00 | resumed-peak fine-tuning closed as negative (README: peak harvesting) |
| v14 | [jds5el2e](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/jds5el2e) | improvement package (partial-credit hold w0.3 p4 + entropy anneal 0.005→0.001:1000 + std cap 1.5 + phase guard) on hold-30 n2000 | own 0.104@1950 (killed 2550); **cross-eval on canonical set 0.206** with best-yet fidelity (keypoint 4.0 cm) | hold-30 demos are harder tracking targets (transfer > own-set!); guard+anneal starved ignition; the reward/process levers themselves looked healthy |
| **v14b** | [ha4ivju9](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/ha4ivju9) | package minus data change: partial hold + anneal :1500 + cap, canonical set | **0.345** (`model_2700`; monotone 0.099@650 → 0.345@2700, no collapse through 3000; tail sweep 2750–2999 = 0.27–0.34 plateau; videos [at7lzy67](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/at7lzy67), 10/16, hammer 7.1 cm/14°) | the champion, by DESIGN: scheduled arc, plateau instead of knife-edge, and it leads v9b +0.034 on row-identical evals — fine-tune from a ridge, not a needle |
| v14c | [wphzj5o7](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/wphzj5o7) | v14b recipe × 6000 iters, anneal :2500 | 0.277 (`model_1400`; watcher-killed @2650; peak zone JAGGED: 0.017–0.277 within 150 iters) | stretched anneal backfired — exploration pressure through the peak zone prevents consolidation (v14b's plateau formed AFTER its anneal ended). End the anneal before the expected peak zone; also more same-recipe lottery (0.345 vs 0.277) |
| v16 | [zylg9paa](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/zylg9paa) | champion recipe + per-group init noise (`WBT_STD_GROUPS=7:1.0,11:0.35` — finger-scale exploration) | 0.189 (`model_1350`; killed @2050) | NEGATIVE: quiet fingers delayed ignition (0.000 through 350) AND the emerged policy collapsed in the post-anneal phase where v14b consolidated — coarse finger exploration appears load-bearing for finding a ROBUST grasp. Uniform init std stays |
| v17 | [ayhgwgz6](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/ayhgwgz6) | champion recipe on the VARIANT-MATCHED `collect-f1c353b` n2000 set (env shape == reference shape by attach interleaving) | TRUE 0.0920 (`model_2050`; arc 0.039→0.092→0.036) | variant matching bought no clear edge (champion cross-evals 0.0847 on the same set); under-15 deaths anti-correlate with SR across its checkpoints — cold-start robustness IS the differentiator |
| v18 | [i1q25fcy](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/i1q25fcy) | + `WBT_ZERO_START_PROB=0.5` + anneal to EXACTLY 0 by 1500, matched set | TRUE 0.0363 (`model_1750`; killed @2400) | HALF-validated: zero-start exposure trains survival (under-15 deaths 823→346, median length 1→103) but anneal-to-zero collapses σ (deflation: survivors 449→62, SR →0.000). Third point on the entropy-floor curve: 0.005 spiral / 0.001 cap-park / 0.0 deflation |
| v18b | [n5rp7vdq](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/n5rp7vdq) | + entropy floor 0.0005 + cap 1.2, zero-start 0.5 kept | TRUE 0.0097 (killed @2200) | dead end: cap 1.2 strangles the exploration both survival and grasping need under the heavy zero-start mixture. v18-family verdict: cold-start exposure costs more grasp learning than it buys robustness; best operating point stays zero-start 0.3 / cap 1.5 / residual 0.001 |
| v19 | [235ux2tr](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/235ux2tr) | proven base (v17 recipe) + SELF-IMITATION anchor v1 (`WBT_SELF_IMIT=0.5:8:16`) | TRUE 0.034@1050 → 0.000@1700 (killed) | anchor v1 BUG: banked SAMPLED actions — BC toward samples pulls the mean into σ-scale noise (SI loss parked at 2σ²=2.0; train t0 rose to 0.46 while eval collapsed). Fixed → bank the policy MEAN |
| v19b | [vnkfqcvj](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/vnkfqcvj) | anchor banks the policy MEAN (fix) | TRUE 0.034 (flattest mid-arc yet: 0.028–0.034 held 1850–2500; late slide) | the anchor STABILIZES — but too early: the bank's crude first successes become gravity before good behavior exists, capping ignition at a third of v17's peak. Line closed (2 runs, both sub-plateau); a weight-curriculum v3 is plausible but the evidence now points at the OBJECTIVE tier |

**Direction (2026-07-11, user):** repro the champion (`ha4ivju9`) and change ONE variable at a
time; pack the L40S (measured: 54% util solo → 94–99% with two concurrent 1024-env runs; safety kit:
run-scoped watcher kills via `WBT_RUN_TAG` environ match, one-eval-at-a-time probe guard,
`SKIP_REF_BUILD`).

| exp | run | single variable vs v20 | true SR | verdict |
|---|---|---|---|---|
| v20 (control) | [avf77ogn](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/avf77ogn) | none — exact champion recipe | **0.0872** @1350, plateau 0.068–0.082 through 2750 | REPRO CONFIRMED: the recipe is real at ~0.09 (champion 0.0944 within process variance); plateau signature reproduces |
| v21 | [d4fl0axr](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/d4fl0axr) | zero-start 0.3 → 0.4 | **0.1102** @2000 | **NEW RECORD** (+17% over champion, +26% over control) — the cold-start dose that works (0.5 was the measured overdose). Ladder base updates to zs04 |
| v22 | [cfdea1re](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/cfdea1re) | hold partial weight 0.3 → 0.5 | 0.0230, died to 0.000 by 1750 | NEGATIVE — stronger instantaneous credit undercuts the streak incentive; 0.3 stays |
| v23 | [7c5ek9k3](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/7c5ek9k3) | (on zs04 base) 1024 → 2048 envs | 0.0969 @1550 | NEUTRAL on ceiling, positive on speed (fastest ignition, peak 450 iters earlier) — lottery tickets buy iteration-speed, not height; 1024 stays |
| v24 | `v24-zs04-anneal2000` | (zs04 base) anneal end 1500 → 2000 | in flight | consolidation-window length test (v21 peaked 500 post-anneal) |
| v25 | `v25-zs045` | (zs04 base) zero-start 0.4 → 0.45 | in flight | dose refinement on the record variable |

**Wave-2/3 results (2026-07-11 night):**

| exp | run | protocol | best | verdict |
|---|---|---|---|---|
| v24 | [jkkcz0by](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/jkkcz0by) | n1000-slice | **0.1235** @3350, plateau 0.11–0.12 across 3050–3950 | RECORD on its protocol — anneal:2000 validated on the zs04 base (+12% over v21, plateau not needle) |
| v25 | [dexa2bk5](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/dexa2bk5) | n1000-slice | 0.1150 @3650 | zs dose curve flat across 0.4–0.45 |
| v26 (redesign pure) | [xyf2sxv0](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/xyf2sxv0) | 2k matched | 0.0555 @3400 | anchored keypoints alone can't carry the grasp signal |
| **v26b (redesign + contact)** | [51s0ravl](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/51s0ravl) | 2k matched | **0.1130** @3450 | **2k-protocol best** (champion cross = 0.0847); contact stays; new standing base |

**Wave-4 results (2026-07-12 morning):**

| exp | run | protocol | best | verdict |
|---|---|---|---|---|
| v27 (v26b + goal-dropout 0.25) | [gtliiu4c](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/gtliiu4c) | 2k matched | 0.0910 @4450 | NEGATIVE (−19% vs base): blinding the conditioning costs more than the autonomous mode buys |
| v27b (v26b + anneal:2000) | [twbqjvpy](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/twbqjvpy) | 2k matched | 0.1060 @5400 | v24's lever did NOT transfer to the redesign base — anneal :1500 stays |
| v26b-r2 (replica) | node1 ckpt sweep | 2k matched | **0.1015 @3450** (arc 0.015 / 0.043 / **0.1015** / 0.084 / 0.050 / 0.049) | base REPRODUCES: same peak iteration as the original 0.1130, Δ ≈ 0.011 seed noise; the post-peak decline reproduces too |

**Straitjacket, quantified on the redesign base (the decisive pair):** the SAME replica checkpoint
(r2 `model_3450`), all-2k, 2000 episodes — RSI protocol **0.1015** vs random-start
(`WBT_EVAL_RANDOM_START=1`) **0.1750**, ×1.72 (the v14b-era pair was 0.0847 / 0.1325 = ×1.56).
Random-start is the highest all-2k number measured on ANY of our checkpoints under any protocol:
the recorded start the objective trains hardest is the policy's WORST start.

**Wave 5 (2026-07-12): contact dose-response** over the v26b base — its biggest lever
(w0 = 0.0555, w1.0 = 0.1130). v28 = contact 2.0 (abl2; relaunched once — first boot died in an
Isaac Sim malloc/tcache abort, rc=250), v28c = contact 0.5 (abl3). **KILLED pre-probe (~iter
900-1000) with wave 6 per user directive — the redesign-base line is closed in favor of the v14b
pivot; no 2k probes recorded** ([845ck1gf](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/845ck1gf),
[rjhahxdd](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/rjhahxdd),
[tczhbd58](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/tczhbd58),
[ty0jrtow](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/ty0jrtow)).

**Wave 6 (in flight): `WBT_NO_RSI_PROB`** — the countermeasure the ×1.72 pair calls for (user:
"use random start now as well"): a per-episode fraction of TRAINING resets skip recorded-setup RSI
+ phase-RSI and keep the env's own randomized reset, so the policy learns to converge INTO the
reference (eval protocol and `bad_tracking` untouched — the 0.1750 eval ran under both). v29 =
no-RSI 0.25 + v29b = no-RSI 0.5, packed on node1.

**Per-reference random-start map (2026-07-12, user-directed step-back: "back to the 0.345, one
variable at a time — different reference points"):** v14b `model_2700`
([ha4ivju9](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/ha4ivju9), the 0.345
checkpoint), each reference evaluated ALONE (2-copy dir), 256 random-start trials each
(`WBT_EVAL_RANDOM_START=1`), n2000 set, current protocol:

| ref | SR | ref | SR |
|---|---|---|---|
| 0 | 0.2461 | 250 | 0.1328 |
| 1 | 0.2617 | 500 | **0.3555** |
| 2 | 0.2461 | 750 | 0.2695 |
| 5 | 0.2539 | 1000 | 0.2500 |
| 10 | 0.1055 | 1250 | 0.2070 |
| 25 | 0.2070 | 1500 | 0.0391 |
| 50 | 0.1211 | 1750 | 0.0000 |
| 100 | 0.3320 | 1999 | 0.1562 |

Mean 0.199, median ≈0.227, max 0.356. **ref-0 (0.246) is NOT special — 5/16 sampled references
score higher.** The 0.345-era number is v14b's TYPICAL random-start capability on a mid-pack
reference, not an easy-reference artifact; there is also a hard motion tail (ref_1500/1750 ≈ 0,
consistent with the plateau anatomy's 86%-unclaimed finding). Caveats: the sample is
low-index-heavy (8/16 from the set's first 5%); map dirs make every env chase the one motion
regardless of its own hammer variant (cross-variant tracking — if anything this UNDERSTATES
matched capability). Reading: the binding constraint is the START CONDITION, not reference
selection — converges with the ×1.72 straitjacket pair, and makes no-RSI training the natural
first single-variable step on the v14b line too (`dev/chrisryu0/dexdeepmimic-v14b-line` @485446c:
7beac8b training code + forward-ported measurement stack).

**The v14b line + fixed measurement (2026-07-12 afternoon, user-directed "fix everything for the
old model; repro 20+% SR in training and in-trainer eval"):** the line now carries @07f58f7:
protocol-fixed server, random-start in-train eval (the eval/SR≡0 incident fix — an RSI slice at
small n flatlines; random-start is where skill is visible at small n), sha provenance, and a
3-attempt sim-boot retry (the malloc/tcache boot flake no longer kills launches). Canonical-826
recovered exactly: `--trim_head 2 --min_lift_cm 8` on the 20260708 collection → 826/1000.

Preflight calibration (random-start, canonical-826, the fixed in-train-eval condition):

| checkpoint | n=256 slice | full 826 |
|---|---|---|
| v14b model_2500 | **0.2109** | — |
| v14b model_2700 (the 0.345-era champion) | 0.1367 | 0.1525 |
| v14b model_2900 | 0.1680 | — |
| (RSI contrast, model_2700, n=128) | 0.0703 | — |

Random-start capability peaks at a DIFFERENT checkpoint (2500) than the old-protocol champion
(2700), with wide checkpoint-to-checkpoint spread — the repro's eval/SR curve samples this band
every 100 iters.

**First 12k verdict + mid-flight trends (2026-07-12 ~22:30):**

| arm | eval/SR (random-start n=256) so far | read |
|---|---|---|
| v31 contact-0 ([rmsr8emc](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/rmsr8emc)) | **peak 0.3164@2400** (highest fixed-eval point yet), volatile 0.23–0.32 through 800–3000, then COLLAPSE (0.08–0.11 @4000+, t0→0.009) — watcher-killed @6276, both signals agreed; model_2400 saved (cadence alignment paid off) | pure omniretarget = high peak, unstable arc |
| **v32 omninorm (uniform-sum)** | **max 0.3086@5200**, still 0.246@6200 — peaks far LATER than any weighted arm | the user's combination + length working together; strongest live arm |
| v30 contact-1.0 | max ~0.25@2000-2400, holding 0.20-0.21 @6000 | stable mid band |
| v31b contact-2.0 | max 0.2656@3800, 0.223@5200 | stable |
| v14b-s43 (old set) | max 0.1914@4000, 0.125@6200 | old set trails at length |

Contact at 12k reads as a STABILIZER (0 = peak-then-collapse; 1.0/2.0 = stable bands), not a
height lever — contrast the v26-era "essential for height" conclusion. v31's freed slot →
**v33-norsi025-12k** (no-RSI 0.25 over the v30 recipe, the straitjacket-countermeasure dose).

**THE RSI INIT BUG (2026-07-13, user-driven hunt: "46% vs 5% — env init has a bug"; "RSI
initializes one set of env variables while leaving other parts inconsistent"): FOUND, FIXED,
×11.** Audit harness `sim/isaaclab/diag_rsi_init.py` (v14b line @a4b36ec+). Two mechanisms:

1. *Stale FK*: `apply_env_setup` wrote joints/root/hammer but never flushed body transforms —
   t=0 observations mixed new joint state with the previous pose's chest/palm/fingertip poses
   (up to 4.98 cm chest error per reset; `phase_rsi` always flushed for its own t>0 reads).
2. *Frame-inconsistent references (the big one)*: every reference's SETUP hammer pose disagrees
   with its own tracking frame 0 by ~14 cm-Y + ~9° — `capture_env_setup` read the per-step hammer
   caches at reset-event time, when they still held the PREVIOUS episode's pose. RSI'd t=0
   episodes therefore spawned the object one inter-episode displacement from its target —
   systematically OOD vs phase-RSI'd (t>0) training starts, which use the observation-convention
   formula and are exactly on-reference. Eval = 100% t=0 starts → the whole RSI-protocol history
   was measured through the corrupted spawn. Random starts skip the setup path → the entire
   RSI-vs-random gap ("the straitjacket") was mostly THIS, not conditioning psychology.

Fix (read-side, works with ALL existing reference sets): FK flush + spawn the t=0 hammer from
reference frame 0 via the phase_rsi formula + refresh the per-step caches. Post-fix open-loop:
hammer on-reference at 0.21 cm p50 from t=2 (was 15 cm).

**The payoff: v14b `model_2500` fixed-RSI n=256 = 0.7734** (was 0.0703 — ×11; random-start
0.2109; source control 0.8838). 240/257 episodes run the full reference. The tracker was always
this good; the eval was broken.

**Family sweep under the FIXED protocol (n=256, canonical-826) — a reshuffle, not a rescale:**

| checkpoint | broken-RSI | random-start | **fixed-RSI** |
|---|---|---|---|
| v14b model_2500 | 0.0703 | 0.2109 | **0.7734** |
| v14b model_2700 (the old "0.345 champion") | ~0.07 | 0.1367 | **0.1289** |
| v14b model_2900 | — | 0.1680 | **0.4961** |
| repro model_1600 | — | 0.266 | **0.7578** |

Three reads: (1) v14b-class training genuinely reaches **~0.77**, 0.11 from the source — twice,
independently (2500 and the repro's 1600). (2) Checkpoint-to-checkpoint variance under truth is
enormous (0.77 → 0.13 → 0.50 across 400 iters) — harvesting needs the FIXED probes, which v35/v36
now run. (3) The old protocol's champion (2700) is nearly the WORST of its family under truth —
broken measurement didn't just deflate numbers, it selected the wrong checkpoints all along.

**FIXED-INIT WAVE (2026-07-13 ~01:00, user: "launch new training jobs, kill old stale ones") —
the full fleet on truth (@8bf10ac):** broken-era arms closed (v30 natural finish; v31b killed
@8511, 43 ckpts; v32 killed @9974, 50 ckpts — all published + annotated).

| run | node | question it answers |
|---|---|---|
| v35-fixedinit-v14b-3000 | abl2 | the true baseline: v14b recipe trained on consistent starts |
| v36-fixedinit-omnicontact-12k | abl5 | redesign set on truth (+ direct broken-vs-fixed pair with v30) |
| v37-fixedinit-omninorm-12k | node1 | the uniform-sum combination on truth |
| v38-fixedinit-norsi025-12k | flakelab | was no-RSI's record real start-diversity value or corrupted-zero-start avoidance? (vs v36) |
| v39-fixedinit-v14b-s43-3000 | abl3 | seed band → the checkpoint-harvest noise floor under truth |

Watchers at MIN_ITER=1000 (fixed probes = honest RSI numbers; peaks may come early and HIGH —
the 0.77-class regime). Target: source 0.8838 (itself due a fixed-build re-baseline).

**v35 COMPLETE — the true baseline's arc ([yt3nf59k](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/yt3nf59k)):**
honest probes (fixed-RSI, n=1000): **0.0810 @1000 → 0.6220 @2000 → 0.7770 @2999** — best =
FINAL, still rising, 951/1003 lifts. The v14b recipe on fixed init matches the best legacy
checkpoint (0.7734) at its natural end without peaking. 3000 iters too short on truth →
**v35b-fixedinit-v14b-12k** relaunched (same recipe/seed, full horizon); dense fixed-RSI sweep of
v35's 2200–2800 checkpoints interleaving on abl2. The broken-era "process levers hit a ~0.09
plateau" doctrine is dead: the plateau was the measurement.

**Seed band on truth + the 0.77 pattern (2026-07-13 ~04:00):** v39 (seed 43,
[flw61nkw](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/flw61nkw)) probes
0.229 → **0.324@2000** → 0.262 — vs seed 42's 0.081 → 0.622 → **0.777**, same recipe, same
horizon: the lottery dominates on truth too, and arcs differ in KIND (early-peak-decline vs
late-ignition-rising). AND a discovery from v32's dying breath: its final watcher probe (it=9800)
ran after the fixed eval landed on abl3 — **broken-init omninorm model_9800 = 0.7710 fixed-RSI**.
Scoreboard under truth: v35-final 0.7770 · v14b_2500 0.7734 · v32_9800 0.7710 · repro_1600
0.7578 — **four 0.77-class checkpoints from different reward sets and both init regimes**. The
0.77 band looks recipe-independent; the remaining 0.11 to the source (0.8838, itself pending
fixed-build re-baseline) is the real object of study. Next: source control re-baseline on abl3
(fix ported to the main line), then seed-44 at 12k on that slot.

**THE DEFINITIVE GAP (2026-07-13 ~05:00, both sides honest for the first time):** source
(jkkim `hammer_lift_2026-06-30-12-50/model_1600`, rsi_play, FIXED build, full 826 set) =
**0.8668** (682/826 claimable; 717/827 clean gate terminations; pre-fix it read 0.8838 — the
source was robust to the corrupted spawn, as expected). Tracker best = **0.7770** (v35 final,
curve still rising) · 0.7734 (v14b_2500) · 0.7710 (v32_9800) · 0.7578 (repro_1600). **The real
gap is ~0.09**, not 0.75 — and the baseline hadn't peaked at its horizon. The program is now:
close 0.09 with honest probes, full horizons, and seed bands (v35b/v36/v37/v38 running; v40
seed-44 next).

**QUEUED (user-directed): v41-fixedinit-v14b-n2000-12k** — data scale as the single variable on
the fixed regime: v35b's exact recipe with ONLY the reference set changed to n2000 (20260709
`collect-f1c353b`, 2000 refs, `--trim_head 2`). Launches at the first freed slot. Companion
measurement once its refs are on-node: the source control re-run on n2000-fixed (the old
n2000-era 0.8838 was broken-spawn; tonight's 0.8668 is 826-fixed — the n2000-fixed target is
needed for apples-to-apples). NOTE: 826-based and n2000-based SRs are different scales — never
mix in one table without labels.

**v38 VERIFIED — the first dual-axis policy (2026-07-13 ~06:30,
[lbeby5x8](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/lbeby5x8)):** the
"exceptional" random-start curve (0.7891@5600, n=256) verified out-of-band on model_5600 — full
826 set, seed 7: **random-start 0.6949 (550/826) + fixed-RSI 0.7094 (584/826)**, terminations
sane, claims spread. Every prior policy chose an axis (baseline 0.777 RSI / 0.21 random; v33 0.28
/ 0.40); v38 (no-RSI 0.25 on FIXED init) holds both at half horizon with probes still rising
(0.759@6000). Start diversity is a real lever that needed honest starts underneath. Source
comparison: 0.8668 RSI.

**v37 harvested + v41 launched (2026-07-13 ~07:30):** omninorm-on-truth peaked **0.7180 @ 5000**
(honest probes) then confirmed-declined to 0.499@8000 — watcher killed + harvested as designed
([69lc05pm](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/69lc05pm), model_5000 among
41 ckpts). Fourth 0.7-class result, THIRD reward set: combination style matters less than honest
starts + harvesting; and without no-RSI episodes there is no dual-axis profile (its random-start
axis peaked 0.2617). Meanwhile **v38's random-start curve hit 0.8047 @ 7600**. v37's slot →
**v41-fixedinit-v14b-n2000-12k** (the user-directed data-scale arm; n2000 refs building on node1;
n2000-fixed source control to follow there).

**v36 harvested at 0.8200 + the v40 incident (2026-07-13 ~08:30):** v36 (omniretarget weighted
+ contact, fixed init) probes peaked **0.8200 @ 5000** — the best harvested checkpoint of the
program (gap to source ~0.047) — legit confirmed decline to 0.679@11000, watcher kill correct
([pabxl81j](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/pabxl81j)). Its slot →
**v42-fixedinit-norsi05-12k** (no-RSI dose 0.5; the v38-family expansion the user asked for).
Dose curve on truth: {0 = v36 0.820 RSI / 0.31 rand · 0.25 = v38 ~0.76 RSI / 0.80 rand rising ·
0.5 = v42}. AND the confessed error: **v40 (seed 44) was wrongly killed at ~7600 while its probes
read 0.8270 @ 7000 RISING** — killed off a stale random-start snapshot without reading its RSI
axis. Recovery: resume from model_7400 (v40r) + full-set harvest sweep of 7000/7200/7400, both
running. Seed band now: 42 = 0.777 · 43 = 0.324 · 44 = 0.827+. Doctrine 12: NO kill without
reading every axis of the run's own instruments in the same sitting.

**SOURCE-LEVEL TRACKING REACHED (2026-07-13 ~09:00): v40 model_7400 = 0.8729 fixed-RSI, full
826 set, out-of-band seed** — vs the source's 0.8668: statistically matched (±~0.02 at this n),
nominally above. The wrongly-killed arc was still steepening (0.765 → 0.846 → 0.873 across its
last three saves); v40r resumes from model_7400. Its random-start is 0.1634 — an RSI SPECIALIST;
the program now holds BOTH profiles: v40_7400 (source-level tracking) and v38 (~0.76 RSI + 0.80
random, dual-axis, still training). Seed band final: 42 = 0.777 · 43 = 0.324 · 44 = **0.873** —
run-to-run lottery is the dominant factor on truth, and dense saves + honest probes + harvest
(+ resume-on-wrongful-kill) are how peaks are actually captured.

**v38 COMPLETE + the user's continuation picks (2026-07-13 ~10:00):** v38 finished 12000/12000 —
probes best **0.7680 @ 10000**, random-start max **0.8047 @ 7600** (the dual-axis run; 61 ckpts;
final full-set verification of model_10000/11800 interleaving). The user flagged
[w4nax0bt](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/w4nax0bt) (v33) and
[f8xst0im](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/f8xst0im) (v34) as having the
best INIT MOVEMENTS — the no-RSI signature; v34 adds the fast-ignition datum (random-start 0.27 by
iter 400 at dose 0.5 — v42 now tests that on truth). Continuation: **v33r-resume2600-fixedinit**
launched on v38's slot (warm-start hybrid: v33's cold-start-skilled weights + consistent starts +
honest probes); v34 not continued (5 young ckpts; v42 supersedes).

**Two L40S released to a coworker (2026-07-13 ~11:00, user-directed):** abl5
(i-0f7ba403728f86864; v42 killed young — its datum: v34's fast ignition did NOT transfer to truth,
0.122@1000 vs broken-era 0.27@400) and abl2 (i-036ada650b8d54751; v35b killed @~9000 after a
decision probe read model_8800 = 0.4758 — declined past its 3000-era 0.777; watcher had never
been re-armed, probe gap 3000–9000 logged as a miss; user kill order on
[94xg5vzs](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/94xg5vzs) converged). Both
nodes handed over RUNNING with baked envs, USAGE.lock rewritten. Fleet now 3: abl3 = v40r
(champion line), node1 = v41 (n2000), flakelab = v33r (+v38 final verification interleaving).

**v38 final verification (full 826, seed 7): model_10000 = 0.7857 RSI / 0.7131 random-start** —
the dual-axis artifact, both axes verified out-of-band; model_11800 = 0.615 (the late dip was
real). The program's two flagship checkpoints: **v38_10000 (0.79/0.71 dual-axis)** and **v40_7400
(0.873/0.163 specialist)** vs the source control 0.8668. v33r (resume of the user's init-movement
pick on truth) confirmed training on flakelab.

**PRE-REGISTERED MATRIX (2026-07-13 ~14:00, after user: "so many successes but not doing correct
ablations"):** base = v38's recipe (omniretarget weighted + contact 1.0 + no-RSI 0.25, fixed init,
canonical-826, seed 42, 12k). Arms: v42r = dose 0.5 (completes {0=v36 0.820 · 0.25=v38 0.786/0.713
· 0.5}); **v43** = seed 43; **v44** = seed 44 (dual-axis band). Exploratory (user-ordered
continuations, excluded from lever tables): v33-lineage resumed PROPERLY from ilbexrsp model_3200
into w4nax0bt (flakelab); v34-lineage into f8xst0im (abl7, launching). Rules in force: no manual
kills — watcher confirmed-decline only; every arm to completion or harvest; instruments verified
by iter ~1200. Killed-mid-rise ledger to honor later: v41 (n2000, RSI 0.679@4000 rising, 26 ckpts)
= the Q3 data-scale rerun at the next natural completion. v40r post-mortem: resume degraded the
champion line (0.873 -> 0.6768@9200) — resume-negative doctrine reconfirmed on truth; champion
artifact stays v40 model_7400.

**OVERNIGHT PROGRAM 2 (2026-07-13 ~14:40, user hypotheses: residual init bugs / more references;
"i'll leave it to you"):** the decisive cheap test landed first — **failure-set overlap, source vs
champion (v40_7400): Jaccard 0.068, only 16/826 refs fail BOTH**; the tracker claims 719/826
(MORE than the source's 682) and 91 of its 107 failures are refs the source handles; union
coverage 810/826 = 98%. Residual failures are tracker/episode-specific, NOT data infeasibility —
the init-residual hypothesis stays alive, the 826-set is not the binding constraint. In flight
overnight (interleaved on abl3): the 826-env settle-drift map (AUDIT_I CSV) correlated against
the champions' failure sets, and v36 model_5000's never-run full-set verification. Q3 (n2000)
queued: resume v41 from model_5000 INTO 48a3b8wb at the first natural completion. Matrix
untouched: v43/v44 (dual-axis seed band), dose-0.5 on bgou6xmw; exploratory continuations on
w4nax0bt (from 3200) and f8xst0im (from 800, abl7). Known residual init items on the ledger:
1-frame stale t=0 observation; settling minority (~10-25% episodes, world-drift p90 7-11cm);
trim-head 2-frame lag; capture-side stale-cache (bypassed read-side, unfixed at source).

**THE RESIDUAL-13% AUTOPSY, CLOSED (2026-07-13 ~15:30):** three probes, one picture. (1) NOT
data: champion-vs-source failure sets nearly disjoint (Jaccard 0.068), union coverage 810/826 =
98%. (2) NOT a fixed hard set: 4/8 sampled champion-failures SUCCEEDED on a re-roll (same ckpt,
protocol, seed; different batch composition) — knife-edge numeric fragility. (3) The mechanism,
on video (failauto clips, abl3): near-perfect tracking (3cm/15deg) broken by CONTACT FLICKER
inside the 10-step hold window — the documented terminal mode, now isolated as THE residual.
RETRACTIONS logged: the "settle-drift" instrument was invalid (zero-action stepping strikes the
hammer with the arm — 16cm on every env); the 826-audit "regression" was a t<=1 read inside the
known 2-frame cache lag. The RSI spawn fix STANDS (open-loop 0.22cm p50 from t=2). Remaining init
ledger: the first-2-frames observation lag (last real init item; plausible flicker aggravator),
trim-lag, capture-side source fix. Next registered round (after the matrix): hold-stability
levers (partial-credit dose, contact-stability shaping) + the obs-lag infra fix.

**v33-lineage COMPLETE on w4nax0bt + Q3 launched (2026-07-13 evening):** the continuation
finished 12000/12000 — one page now holds the whole story (broken-era 0.4688@2600 record; fixed-
init peak **0.7461 @ 4400** random-start; final 0.6016). Exploratory-class, 45 ckpts published.
Its slot → **Q3: v41 resumed from model_5000 INTO 48a3b8wb** (n2000 rebuilt on flakelab, setsid-
hardened). Infra hardening this cycle: track_train setsids the trainer (SSM group-teardowns had
killed the abl7 continuation at ~2800 — resumed from its published segment — and earlier the
first w4nax0bt attempt's server).

**DOSE-0.5 COMPLETE — the dose curve has an interior optimum (2026-07-13 night):** v42-fixedinit-
norsi05-12k finished 12000/12000 on
[bgou6xmw](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/bgou6xmw) (arc-completion
resume of the wrongly-killed v42 from model_1200 — young-run resume, allowed class). Fixed-
protocol verdict (canonical-826, seed 7): **model_7200 = 0.6586 RSI / 0.6356 random-start** (its
own dual-axis peak; 10600 = 0.533/0.565, final 11999 = 0.484/0.493 — the RSI axis decays past
7200, peak-harvest again). The dose curve on the v38 base, both axes:
{0 = v36 0.749/0.225 · 0.25 = v38 **0.786/0.713** · 0.5 = 0.659/0.636} — **0.25 dominates 0.5 on
BOTH axes**; more start diversity past 0.25 costs RSI competence without buying random-start
robustness. Single-seed caveat at 0.5 (v43/v44 band the 0.25 point, not this one). In-train
random-start val peak 0.7305 @ 7200 agrees with the verdict pick. 53 ckpts published
(`.../ckpts/2026-07-13-13-32_v42-fixedinit-norsi05-12k_12000it_1024envs-bfc1277/`); W&B summary
carries the verdict block.

**THE DUAL-AXIS SEED BAND, COMPLETE (2026-07-14):** v43 and v44 finished 12000/12000 naturally;
fixed-protocol verdicts (canonical-826, seed 7) on
[ipzswst1](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/ipzswst1) and
[5ou909r8](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/5ou909r8). The band for the
v38 recipe (omniretarget wts + contact 1.0 + no-RSI 0.25, fixed init), final-region checkpoints:

| seed | RSI | random-start |
|---|---|---|
| 42 (v38 model_10000) | 0.7857 | 0.7131 |
| 43 (v43 model_11999; 10000 = 0.7349/0.6852) | 0.7337 | 0.7276 |
| 44 (v44 model_11999) | **0.8087** | **0.7312** |

Two findings. (1) **The recipe kills the seed lottery**: random-start spread 0.018 (0.713–0.731),
RSI spread 0.075 (0.734–0.809) — vs the v14b-set band's 0.32–0.87 chaos. The banded claim now
stands: this recipe reliably lands ~0.73–0.81 RSI / ~0.72 random-start. (2) **No late decay at
dose 0.25**: v43's final ≈ its peak, v44's final IS its best measured — the RSI-axis decay seen
at dose 0.5 (0.659→0.484) and in the specialist line (v40) is absent; the 12k arc is stable here.
Caveat: v44's probe-best (9000) went unswept (watcher line truncated at kill — orchestrator
corner case); all 61 ckpts/seed are on S3 for densification if needed. Both nodes terminated
post-closeout (fleet cap 3, user-directed): track (169h 8xlarge) and abl3 gone; abl7 closes out
v34-lineage next, leaving abl6 (v45 hold-lever, carrying the annotation contract) as the fleet.

**THE ANNOTATION CONTRACT'S FIRST CATCH — v45 retired as a config error (2026-07-14 ~01:00):**
the user asked for a live demonstration that annotations are fixed; the demo render (8-env
eval_service off v45's model_800, v45 tunables exported) produced a derived legend of
`1.5*obj_pos + 1*obj_ori + 1*keypoint_obj + 0.5*joint + 0.25*joint_vel + 1*contact  reg:
-0.002*action_rate_l2` — **hold missing**. Root cause: `WBT_HOLD_WEIGHT` was wired in env_cfg
and track_train.sh but not in the `ENV_TUNABLES` whitelist (`hammer_lift/experiment.py`) — the
missed third link — so v45 trained the plain v38 recipe with ZERO delta for ~900 iters. The
contract exposed it precisely because channels derive from the live reward manager (a hand-kept
overlay would have happily drawn a hold curve). Fix @fd42ca3; v45 killed + W&B-annotated
([opxy6dff](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/opxy6dff), no ckpts
published — duplicate recipe); relaunched as **v45b-holdpartial-s42-12k** with the knob wired
end-to-end; after-render verification owed: hold must appear in the derived legend before the
arm counts as registered. Knob-wiring checklist now three links: env_cfg getattr + launcher
forward + ENV_TUNABLES whitelist.
**Verification landed (pixels + clips on CloudFront `reports/annot_contract_demo/`):** the v45b
after-render legend ends `+ 0.5*hold  reg: -0.002*action_rate_l2` with the hold curve drawn only
inside the reference's sustained-hold window (gap-segments elsewhere) — the arm counts as
registered. v45b = [yhxg6f4s](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/yhxg6f4s).

**v34-LINEAGE COMPLETE + FLEET SHED-DOWN DONE (2026-07-14 ~01:40):** the exploratory omninorm ×
no-RSI-0.5 continuation ([f8xst0im](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/f8xst0im))
finished 12000/12000; fixed-protocol verdict: model_11999 = **0.6731 RSI / 0.7155 random-start**,
and — unique among all 12k arms — **still rising at the horizon** (11200 = 0.633/0.690 < final on
both axes; the uniform-sum set's slow-arc signature, v32's pattern on truth). Random-start lands
inside the dual-axis band while RSI trails it — consistent with dose 0.5 trading precision for
robustness. Exploratory-class (multi-confound lineage), excluded from lever tables; 47 ckpts
published; abl7 terminated post-closeout. Fleet end-state after the user's cap directives: 8
running L40S → **1** (abl6, v45b) — abl2/abl5 (unclaimed coworker hand-offs), groot-hf-free2,
flakelab (Q3 harvested mid-rise at user direction), track (v43), abl3 (v44), abl7 (v34c) all
terminated with artifacts published + runs annotated first.

**CURRENT STANDINGS — what survives, and the simplest of the best (2026-07-14; all
fixed-protocol canonical-826 seed 7, RSI / random-start):**

| line | best artifact | RSI / rand | status |
|---|---|---|---|
| source control | jkkim full-curr iter1600 | 0.8668 / — | the bar |
| **dual-axis recipe** | **v44 model_11999** | **0.8087 / 0.7312** | **THE survivor** — 3-seed band 0.734–0.809 / 0.713–0.731 |
| RSI specialist | v40 model_7400 | 0.8729 / 0.1630 | artifact only; recipe dead (see below) |
| dose 0.5 | bgou6xmw model_7200 | 0.6586 / 0.6356 | dominated by dose 0.25 on both axes |
| omninorm × 0.5 | f8xst0im model_11999 | 0.6731 / 0.7155 | exploratory; only still-rising 12k arc |

**FRAMING CORRECTION (user, 2026-07-15): "source level" is the source's REPEATABILITY, not the
tracker's ceiling.** A trajectory-conditioned policy is handed a known-feasible demonstration and
its exact init — its target is ~100% (modulo residual sim stochasticity), so 87%≈87% means the
tracker has matched the source's fragility, not exhausted the task. The residual 13% (knife-edge
hold flicker) is the robustness gap the DR/hold levers exist to close. PR results bullet updated.

**One recipe survives — and it is also the simplest of the best.** The dual-axis recipe
(v38 = v43 = v44, delta only the seed): omniretarget object-anchored reward — obj_pos 1.5 +
gated obj_ori 1.0 + keypoint_obj 1.0 + joint 0.5 + joint_vel 0.25 + contact 1.0 + action-rate
regularizer — with fixed-init RSI, phase-RSI, and `WBT_NO_RSI_PROB=0.25`; train 12k; **take the
final checkpoint**. Everything else the workstream ever added is ABSENT from it: no hold term,
no partial-credit blend, no entropy schedule, no std cap, no peak-harvest race (the arc doesn't
decay), no seed retries (the band is tight). Its two levers with measured mechanisms: object
anchoring keeps the dense reward task-consistent at displaced starts, and the 25% default-start
dose converts that into the funnel (v36 = same reward at dose 0 → 0.225 random-start).
**v40 survives as an artifact, not a recipe**: 0.8729 beats the source bar itself, proving
tracking reaches source parity — but its recipe (v14b set, dose 0) is a seed lottery
(0.32–0.87 across 3 seeds), collapses at random start, and needs peak harvesting; nothing to
rerun. Pending against the survivor: **v45b** (the one live run — survivor + hold ramp w0.5
partial(0.3,4), the anti-flicker lever aimed at the residual 13%); registered next cells:
keypoint DE-anchoring at dose 0.25 (is anchoring necessary, or only the dose?) and the
spawn-distance decomposition of random-start success (coverage vs true re-registration).

**REFRAME (user, 2026-07-14): no-RSI is a hammer-lift accident; the general lever is DR around
the trajectory.** The task-init fallback only implements start diversity because THIS task's
init distribution is narrow — every task-init draw is already an ε-ball around every
reference's start (the same narrowness that lets the blind source score 0.87). An arbitrary
motion library has no shared task init to borrow, so `WBT_NO_RSI_PROB` does not generalize.
Corollary spotted in the same pass: RSI'd episodes replay the RECORDED physics too (mass +
frictions ride the setup vector), so the no-RSI fraction was also the recipe's only physics
diversity. The transferable form: **the RSI tube** — RSI to the reference, then perturb inside
a canonical tube. Implemented @2aee0b1 (three-link checklist honored): `_rsi_tube_noise` in
wbt.py runs after `apply_env_setup` (rewrites joints + hammer xy/yaw off the fresh caches,
re-flushes FK, re-syncs caches; hammer z untouched — PenExceed guard; jvel exact), canonical σs
jpos 0.05 rad / obj-xy 2 cm / obj-yaw 0.15 rad, master-scaled by `WBT_RSI_NOISE` (training) and
`WBT_EVAL_RSI_NOISE` (fixed-refs graded-ε robustness evals — the NEW third axis; the legacy
random-start eval stays as the task-specific axis). REGISTERED NEXT ARM — **v46-rsitube-s42-12k**
(launches on abl6 when v45b closes): base = survivor recipe, single conceptual delta = the
start-diversity SOURCE (no-RSI 0.25 → 0, tube 1.0). Readouts: RSI axis, legacy random-start,
graded-ε tube curve (0.5×/1×/2×) — v44 gets the same tube evals from S3 for the head-to-head.
If v46 holds the band on RSI and matches/beats v44 on the tube curve, the task-init trick is
retired from the recipe. Physics-tube (mass/friction jitter on RSI'd episodes) queued as the
follow-up lever, separately.

**FLEET BACK TO 3 (user-directed, 2026-07-14 ~02:30) — two user-picked arms take the new
nodes; the tube/de-anchor cells requeue behind them (tube arm renumbered v48).** Registered:
- **v46-n2000-dualaxis-s42-12k** (abl8): base = survivor recipe byte-for-byte; single delta =
  the reference set, canonical-826 → **n2000** (`20260709/…collect-f1c353b_n2000`,
  `--trim_head 2`, full-grasp-verified at collection). The data-scale question at the current
  best recipe — completes what Q3 started (v14b-set n2000 was killed rising at 0.733@6000) and
  tests the overnight hypothesis "more references might help". Verdicts stay on canonical-826
  (out-of-band seed 7) for cross-arm comparability, n2000-protocol probes logged alongside.
  NOTE (surfaced by the v14b-line variant warning on v45b's boot, 32 hits): the canonical-826
  collection (20260708) PREDATES variant recording — every fixed-init arm so far trained with
  env↔ref hammer shapes unmatched on ~2/3 of rows, uniformly (cross-arm comparability intact;
  matching alone measured ~within-noise on the old champion). The 20260709 n2000 set IS
  variant-recorded, so v46's delta inseparably bundles data-scale + variant-matching restored.
- **v47-batch2048-s42-12k** (abl9): base = survivor recipe; single delta = **NUM_ENVS
  1024 → 2048** (batch doubles; minibatch scales with it under full-batch updates). 2× samples
  per update at the same 12k update count — the gradient-quality/batch-size lever. VRAM watch
  on the first 100 iters (48GB L40S; fallback 1536 relabeled if OOM).

**PROTOCOL v2 — the task-init random-start axis is RETIRED as a headline metric (user-directed,
2026-07-14 ~08:45: "random init isn't very meaningful at this point"):** it measured transfer
within this task's accidentally-narrow init distribution — the same narrowness that made
`WBT_NO_RSI_PROB` work at all — an artifact axis, not general robustness. Replacement: the
**graded-ε RSI-tube curve** (`WBT_EVAL_RSI_NOISE` at 0.5/1/2 under fixed refs). In-train val now
fires tube-ε=1.0 (@71aec27; deploys with launches after v48). Verdict protocol v2 = fixed-RSI
axis + tube curve; task-init random-start demoted to a legacy/secondary readout (kept cheap for
band-table continuity). v48's boot crash (tube read `_ref_setup` at the pre-attach reset — 3
deterministic rc=1 attempts) fixed @b130c67; **v48 relaunched and training on
[m0jl3iv6](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/m0jl3iv6)**.

**THE n2000/TUBE FORENSICS — both new arms carried defects; fixed + relaunched (2026-07-14
~19:00, user: "n2000 doesn't work as good — this is suspicious"):** trajectory forensics on the
BUILT sets settled it. (1) **v46's n2000 build was unfiltered** (meta `min_lift_cm=0.0`):
348/2000 refs move the hammer <8 cm (canonical: 0) — 17% infeasible-ish training targets. (2)
**Collection off-by-one**: 1825/2000 refs end their full-grasp run ON the final recorded frame
with streak exactly 9 — one short of the 10-step gate — so 92% of the set exposed NO hold window
in-file (canonical: 100% have ≥10). Fix at the derivation (@44e86aa): the grasp streak now GROWS
through the frozen-target tail when a ref ends grasped (the tail repeats the grasped frame, so
the window is physically real). (3) **v48's tube broke ignition**: in-hand phase entries got
±0.05 rad finger jitter → grasped spawns broke instantly → contact-avoidance learned (lifts
76→9 by it2000, probes 0.000 flat, reward rising). Fix: ALL tube noise contact-gated (the
"joints everywhere is useful roughening" call RETRACTED — measured wrong). v46/v48 jobs killed
(nodes kept), latest ckpts published, runs annotated
([1oa4mh0n](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/1oa4mh0n),
[m0jl3iv6](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/m0jl3iv6)); relaunched as
**v46b-n2000filtered** (min-lift build + tail-aware windows) and **v48b-rsitube**
(contact-gated). v47 (batch-2048) is healthy — 0.115@3000, the family's strongest ignition yet.

**v47c VERDICT — uniform tube fails on the batch arm too, with late collapse (2026-07-15
~01:30):** complete 12000. Peak model_10000 = 42.7% fixed / 46.6% @ε0.25 / 36.6% @ε1.0; final
model_11999 COLLAPSED to 13.9/25.1/21.4 — the std-spiral endgame reasserted itself on the
2048-env arm (peak-harvest applies again there). Uniform-tube family verdict, both members:
flat-low curves, no robustness bought, precision destroyed. Surviving tube hypothesis =
v50-tubesplit (init-heavy). **v51-n2000trim launched** on abl9: shipped-default recipe, one
delta vs v49 = the post-claim-trimmed 1778-ref set — the data-scale question on clean footing.
Annotated on [1mcvpgr1](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/1mcvpgr1).

**v48c VERDICT — the uniform tube recipe FAILS the bar (2026-07-15 ~00:30):** complete 12000
(first run with a captured exit code: TRAINER_EXIT rc=0). Fixed protocol: model_11999 = 34.3%
exact / 47.3% @ε0.25 / 35.8% @ε1.0 (model_9400 = 26.0/49.4/35.4) vs the v44 incumbent's
80.9/82.0/37.5. Readings: (1) full-strength uniform tube noise DESTROYED precision-axis
competence (34% vs the 73–81% band) and bought no robustness (ε1.0 ≈ incumbent); flat-but-low
curve. (2) The curve PEAKS at ε=0.25, not ε=0 — exact recorded starts are off the policy's
training distribution — evidence the dose/profile, not the tube concept, is the failure.
(3) The user-observed "80%" curves were the n=256 val instrument; the 826-set out-of-band
verdict is the truth (standing rule reaffirmed: probes/verdicts are the KPI). 20 resume-segment
ckpts published; annotated on
[b0i5nfjb](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/b0i5nfjb). **v50-tubesplit
launched** in the slot (one lever: mid-entry noise 1.0 → 0.25×; spike-crash-tolerant watcher
DROP=0.4 fleet-wide).

**REWARD MINUS-ONE QUEUE REGISTERED (user: "simplify the reward further?", 2026-07-15):**
sequenced subtraction, one delta each, behind the current round: (1) −joint_vel (weakest
evidence — never individually ablated, arrived with the port); (2) −joint (plausibly redundant
with object-anchored keypoints); (3) −obj_ori last (it fixed a measured failure — in-grip tilt,
rot met 27% pre-fix — so it must prove removable). Core stays: obj_pos + keypoint_obj + contact
(the one term with a clean measured ablation: 5.6% without vs 11.3% with) + the action-rate
regularizer. Endpoint if all fall: a three-kernel reward. Each removal judged on the fixed axis
+ tube curve vs the then-current recipe of record.

**NEXT ROUND REGISTERED — init-heavy tube profile + the clean data arm (2026-07-15, user:
"more init DR, not too much mid DR"; PR sent for review by the user with their own visuals):**
the tube gains split scales (@9cb7064): `WBT_RSI_NOISE` = init (t=0) scale, `WBT_RSI_NOISE_MID`
= phase-entry scale (default 0.25× init). Succession plan as slots free:
1. abl6 (after v48c completes): **v50-tubesplit-s42-12k** — base v48c, one lever = noise profile
   uniform(1.0/1.0) → init-heavy (init 1.0, mid 0.25× default). Judged on the fixed axis + the
   tube curve vs v48c and v44 (37.5% at ε=1.0 is the incumbent bar).
2. abl9 (after v47c completes): the **n2000-trimmed data arm** — base = the shipped default
   (exact reference-start, no tube), one delta = the reference set, canonical-826 → n2000
   post-claim-trimmed (1778 refs, validated).
3. abl8 (after v49): v49 stamps the PR default headline; slot then follows results.
Closeouts switch to `closeout_keep.sh`: publish → verdicts (fixed + tube 0.25/1.0) → annotate,
nodes KEPT (jobs and nodes are different layers).

**n2000 SOLVED — the post-claim tail was source-degradation footage (2026-07-14 ~18:00,
user's side-by-side ask):** head frames of both sets are structurally identical (same layout,
same setup convention). The fresh set's tails are NOT static holding: past its own claim the
source policy is out-of-distribution and DEGRADES — final frames read 3/5-finger contact with
the hammer sagged near-table (t=186: c=[1 1 1 0 0], z back to −0.415). Training on those refs
rewards tracking a sag-and-drop — the real reason the n2000 arms never ignited. Fix at build
(@c1ed3f7): `reference.py --max_post_claim N` cuts each ref at its first gate-crossing hold + N
(collect_references.sh passes 5). Full rebuild validated: **1778 refs, lengths p5/50/95 =
59/64/81 (canonical-shaped), zero validator violations**; 222 dropped by min-lift now measured
post-trim (decay-only "lifts" excluded). The data arm requeues on this build at the next free
slot. Side note logged: fresh trajs carry action[250] vs tracking_state[~189] — collector
length quirk, harmless to refs, on the ledger.

**THE "70%" SIGHTING FORENSICS — real crests on a fast, oscillating batch arm; instruments
verified sane (2026-07-14 ~17:00):** the user saw v47c val crests up to ~0.7. Controls run: (1)
zero-control — v47c model_0 (random weights) offline exact-RSI = 0.0000 → the abl9 eval stack is
sane; (2) cross-node discriminator — v48c model_800 = 0.0000 (normal from-scratch); (3) v47c
model_800 offline = **0.3945 exact-RSI / 0.375 @tube-ε0.25** — the early competence is REAL at
the weights level; (4) old-v47 model_800 (no tube) = 0.1055 → batch-2048 alone is a ~5× faster
igniter than the 1024 family; v47c's additional ~4× (tube on, task-init off) is UNEXPLAINED —
no contamination vector found (clean iter-0 start, no resume flags, refs verified canonical-826
min-lift-8). The violent swings (vals 0.64→0.28 across 100 iters; probes 0.25↔0.33↔0.19) match
the measured batch-2048 std-spiral (4.81 on old v47) — fast arc + oscillation, not instrument
noise. OPEN: cause of v47c's ignition speed; registered next-round lever for the batch arm =
entropy tame (WBT_ENTROPY_COEF 0.001, the documented spiral fix) for stability, not height.
v46e/n2000 remains UNRESOLVED (killed at 5000 for the v49 slot while the slow-igniting tube
family was still pre-arc; requeued).

**PR DEFAULT PINNED: exact-RSI + canonical n1000, tube stays experimental (2026-07-14 ~16:00,
user-directed "meet the 80% SR"):** the shipped zero-knob default = exact RSI + phase-RSI on the
canonical gate-feasible build (now the launcher default), omniretarget + contact 1.0, NO tube.
**v49-default-s42-12k** launched on abl8 (v46e's slot — the watch-flagged weakest arm; its
data-scale question requeues) to stamp the default's own headline number: the banded 0.73–0.81
results carried the since-removed task-init lever, and the nearest pure-exact-RSI prior is v36
(0.7494 full-set / 0.8200 probe, single seed). Fleet: v48c + v47c continue the tube matrix.

**THE TASK-INIT LEVER IS REMOVED; THE FLEET IS THE TUBE-RECIPE MATRIX (2026-07-14 ~15:00,
user-directed: "totally unnecessary"):** WBT_NO_RSI_PROB machinery deleted end-to-end (mask,
wrapper filter, phase filter, knob, launcher forward) along with WBT_EVAL_RANDOM_START — setting
either is now a hard error (@ffd14f4; silent-ignore is how zero-delta runs happen). Every episode
starts ON its reference, slightly DR'ed inside the contact-gated tube; phase-RSI supplies entry
diversity; physics-tube remains the registered follow-up. WBT_CONTACT_WEIGHT now defaults 1.0
(knob-less evals used to silently build 5-term rewards vs training's 6 — the derived legend's
second catch). Fleet = one recipe, three single-delta arms vs v48c:
[v48c base](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/wqbgyikb-successor) —
see m0jl3iv6 lineage — canonical-826/1024;
**v46e** data arm (fresh n2000) [pabmjcxw](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/pabmjcxw);
**v47c** batch arm (2048) [1mcvpgr1](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/1mcvpgr1).
Verdict protocol unchanged: fixed-RSI + graded-ε tube; v44 (0.809/0.375) is the incumbent bar.

**CLEAN-SLATE RELAUNCHES of the n2000 and batch arms (2026-07-14 ~14:30, user-directed):**
v46c and v47 killed (jobs only) and relaunched as **v46d-n2000fresh** / **v47b-batch2048** — same
single deltas, now on the fully-fixed stack with honest ε=0.25 val curves from iteration 0 and
clean single-segment pages. For the record: both trainings were clean (tube knob 0; the zero
vals were the frozen pre-fix eval servers), and v47 was killed at its strongest — probe
**0.514@5000**, the best family arc at 5k yet measured (its ckpts incl. the 0.514 era are
published; the relaunch will confirm or beat it on a clean page). v46c's 3k segment published.

**THE TUBE QUAT BUG — "all SR 0%" traced to a convention flip (2026-07-14 ~12:40, user-flagged):**
every arm's in-train val (tube-ε protocol) read 0.000 while probes read normal — the
discriminating contradiction: v44 tolerates 15cm task-init draws (0.71 random-start) but scored
0.000 under 2cm/0.7° tube noise, with 218/258 episodes tracking to full length unclaimed and a
3× step-reward crash. Video showed the mechanism: **hammer ~156° flipped at spawn, magnitude
independent of ε** — the tube's yaw quat was built wxyz (`[cos,0,0,sin]`) in an xyzw stack,
whose "identity" reads x=1 = a constant 180° X-flip on every noised hammer. Convention bugs
don't scale with the knob. Consequences unwound: v48/v48b trained against flipped objects (the
"contact-avoidance" reading was DOWNSTREAM of this — the joint-gating fix remains sensible but
was not the root cause); every arm's tube-val zeros are this, not policy; v44's tube-eval cliff
was this, not fragility. Fix @c1dc4ea (both noise sites, xyzw identity); v48b killed + annotated
([wqbgyikb](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/wqbgyikb)); **v48c**
relaunched on the corrected tube. v46c/v47 trainings were never tube-noised (knob 0) — only
their val curves carry frozen-code zeros; probes stay the KPI. Post-fix tube-eval of v44
pending as the protocol's sanity gate (expect ε=0.25 ≈ baseline).

**SILENT DEATH #2 — v47 vanished at ~3076 (2026-07-14, caught ~12:00):** same signature as
v45b@3097: process gone between iteration prints, no traceback, no dmesg OOM, no systemd-oomd
entry, GPU cleanly released. Both on post-contract code, both in the 3.0–3.1k region — pattern
suspicious but cause UNIDENTIFIED (2 cases). Instrumentation queued for the next occurrence:
core-dump ulimit + a trainer heartbeat wrapper. v47 resumed from model_3000 into
[19g10fsn](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/19g10fsn) (no delta; 76
iters lost). Also on the record at the death point: **action std 4.81** — the entropy spiral is
live on the batch-2048 recipe (the dual-axis 1024-env arms never spiraled at this coef; bigger
batch appears to feed the arc — watch, don't intervene: one delta).

**FRESH n2000 COLLECTED VIA THE NEW PIPELINE, v46c LAUNCHED (2026-07-14 ~11:40, user-directed):**
`collect_references.sh` (in the PR: claim-only collection so the hold window records fully →
build with min-lift → `reference.py --validate` fail-loudly) produced
`20260714/…collect-ca59965_n2000`: 2000 successes (source SR ~0.84 in-collection), 1983 refs
kept (17 low-lift filtered at build), **validation zero-bad on all four checks** — the
off-by-one is fixed at the source. v46b (interim: old set + filters) superseded before its
first probe; **v46c-n2000fresh** trains on the clean set (delta vs v44: the reference set,
now genuinely just data-scale + variant-matching). The 20260709 f1c353b set is retired. "not very useful, run the
next one" at 3369/12000 (probe 0.0640@3200 — family-normal slow ignition, but 3.4k of this
family's arc is not a verdict). JOB killed, node kept (the layer rule). The hold-lever question
closes UNRESOLVED; segments published (fd42ca3 seg incl. model_3000 + 5cfaa2d resume seg);
[yhxg6f4s](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/yhxg6f4s) annotated. Its
run also delivered two artifacts regardless: the annotation contract's live catch (v45
zero-delta) and the silent-death + PenExceed close-window sightings. **v48-rsitube-s42-12k**
launched in the slot (abl6): base = v38 recipe at dose 0 + `WBT_RSI_NOISE=1.0` — the
start-diversity-source swap, task-init fallback → per-trajectory tube.

**v33 post-mortem under truth (user: "you killed the best one"):** its record checkpoint
(model_2600 = the last saved; the kill landed ~2700, right at the 0.4688 random-start peak)
re-evals at **fixed-RSI 0.2812 / random-start 0.4023** (model_2400: 0.1797 RSI). Verdict: a
random-start SPECIALIST — its RSI'd 75% trained on the corrupted spawn, so it never saw a true
t=0 start — far below the 0.77-class under the true protocol. No hidden champion was lost; the
open question it leaves is exactly v38's: does no-RSI + TRUE starts give both axes (RSI-high +
random-robust)?
Remaining control to re-baseline: the source's 0.8838 was ALSO measured through the corrupted
spawn (rsi_play uses apply_env_setup) — re-run on the fixed build once the fix is ported to the
main line; source may read higher. Fleet rebased: v33/v34 (no-RSI arms, trained on corrupted
zero-starts) killed + published; **v35-fixedinit-v14b-3000** (abl2, the true baseline) and
**v36-fixedinit-omnicontact-12k** (abl5) launched on the fixed code; v30/v31b/v32 12k arms finish
naturally (their random-start val curves remain valid). Fixed-RSI re-evals of v14b_2700/2900 +
repro model_1600 running. Residual known issue: the t=0 OBSERVATION still reads the pre-spawn
hammer for 1 frame (a step-cached body-pose buffer; 0.7734 shows it's near-harmless) — capture-
side fix for future collections + the 1-frame read both queued.

**NO-RSI BREAKTHROUGH (2026-07-12 ~23:15, user-confirmed from videos):** v33-norsi025-12k
([w4nax0bt](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/w4nax0bt)) — no-RSI 0.25 on
the v30 base — hit **eval/SR 0.449 @ 1800** (0.277→0.422→0.289→0.449), the highest number this
workstream has measured, at 1/6th of the horizon. The inversion that proves the mechanism: v33
Train-t0 = 0.173 with eval 0.449, vs s43 (old set) Train-t0 = **0.912** with eval 0.13–0.20 —
training-start overfitting traded for transfer, exactly the straitjacket prediction. s43 killed
@~8400 (RSI-overfit exemplar, 44 ckpts published,
[vvndkibj](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/vvndkibj)); its slot runs
**v34-norsi05-12k** — the dose curve on the SAME base: {0 = v30, 0.25 = v33, 0.5 = v34}.
Queued next (on the following free slot): omninorm × no-RSI interaction. NOTE (user question):
no env task-success REWARD exists in any set — `rewards.task_success = None` in every WBT cfg;
all sets are pure tracking kernels (+ regularizer); the canonical gate is measurement-only
(claim latch, `terminate=False`). Nothing so far suggests it is needed.

**LENGTH DIRECTIVE (2026-07-12 evening, user: "2k, 6k are not long enough"):** all overnight arms
relaunched at **MAX_ITERS=12000** (val every 200, watchers MIN_ITER=4000 / gap 1800 / 1000-env
probes — lenient, because the historical "decline after ~3.5k" kills keyed on RSI probes and the
long arc has never been measured under the FIXED random-start eval). The <1.5h short launches were
killed + annotated; the 3k repro finishes naturally as the short-baseline. 12k arms:
v30-omnireward-12k (node1), v31-contact0-12k (abl2), v32-omninorm-12k (abl3), v14b-s43-12k (abl5);
flakelab takes a 12k arm when the repro ends. ETA per run ≈ 40-45h.

**Overnight trend fleet (2026-07-12 evening, user: "up to 5 L40S, understand TRENDS not points"):**
all on the v14b line, single-variable vs the repro, fixed random-start eval n=256, canonical-826,
3000 it, seed 42 unless stated. Two curves + a band:

| run | node | delta | purpose |
|---|---|---|---|
| v30 `omnireward` ([contact 1.0]) | node1 | reward set → omniretarget (@4edff1f knob) | reward-set head-to-head vs repro |
| v31 `contact0` | abl2 | omniretarget, contact 0 | contact dose arm |
| v31b `contact2p0` | abl3 | omniretarget, contact 2.0 | contact dose arm — PREEMPTED ~iter 150 for v32; re-queues |
| **v32 `omninorm`** | abl3 | UNIFORM-SUM combination (user-directed: weight-1.0 add, σ-internal normalization, action-rate as kernel; @aa3e610) | combination-style head-to-head vs the weighted arms |
| v14b-repro-s43 | abl5 (new 5th node) | seed 43 | seed band on the base (the noise floor) |
| v31c `contact0p5` | flakelab, after the repro ends | omniretarget, contact 0.5 | completes the {0, 0.5, 1.0, 2.0} curve |
| v32 no-RSI arms | freed nodes | `WBT_NO_RSI_PROB` 0.25/0.5 (@00a8aba port) | straitjacket-countermeasure dose |

**Repro CRITERION MET + COMPLETE (run [1mm9xov6](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/1mm9xov6),
finished 3000/3000 — first natural completion on the fixed stack):** eval/SR (random-start,
n=256) 30-point curve — rise to **0.3008 @ 1100**, band 0.16–0.27 through 1100–2100, slump
2200–2500 (min 0.066 @ 2500), recovery to 0.172 @ 3000; Train t0 max 0.6402. Both dials > 0.20
with headroom. Under the fixed eval, capability arrives by ~600–1100 — the old protocol's
"0.345 @ 2700" peak-late story was partly measurement artifact — and the best random-start
checkpoint of this replica (model_1100, 0.301) exceeds every sampled point of the original v14b
family (max 0.211 @ 2500). The repro's slot now runs v31b-contact2p0-12k, completing the 12k
contact curve {0, 1.0, 2.0}.

**Fingertip-guard grace diagnostic (the "1-second episodes"; user-flagged):** the instant enders
under random starts are the `table_fingertip_contact_force_exceeded` safety guard tripped by
cold-start flails (58/132 force ends, 38/132 under-15-step episodes; `wbt_bad_tracking` fired
ZERO times). Paired eval, v14b model_2500, n=256: grace-15 exempts the first 15 steps →
under15 54→0, force ends 55→11, SR 0.1836→**0.2031** (+0.02). Verdict: the guard is NOT
meaningfully suppressing capability — ~90% of the spared episodes fail anyway; the flail is a
trainable deficiency (start-diversity lever), not eval harshness. Canonical KPI keeps the guard
from step 0; `WBT_EVAL_FORCE_GRACE` stays diagnostic-only. The unfiltered-1000 slice reads HIGHER than canonical-826 (0.2109 vs 0.1641 at
n=128, model_2700): the low-lift refs the canonical filter drops are easier to claim from random
starts. REPRO LAUNCHED on the flake-lab node (i-08d381e5470e968f3, node slot 4/5): exact v14b
recipe (hold-partial 0.3/pow4, anneal 0.005→0.001:1500, cap 1.5, zs 0.3 hardcoded, 826 refs,
3000 it, seed 42), VAL_EVAL_ENVS=256, watcher at PROBE_GAP=900/PROBE_ENVS=1000 (probes must not
contend the criterion-bearing val fires).

**Reward REDESIGN (2026-07-11, user-directed — new base, breaks the single-variable chain on
purpose):** OmniRetarget-shaped set (README row 14): object-anchored keypoints w1.0 + joints w0.5 +
joint-vel w0.25 + obj pos 1.5 / gated ori 1.0 + action-rate −0.002; contact/hold/task-bonus removed.
Fleet expanded to 3 L40S nodes (abl2/abl3 g6e.2xlarge; fresh-node gotchas hit+fixed: pubkey via SSM,
EBS root 150G — use `ROOT_GB` on future launches).

| exp | node | delta | status |
|---|---|---|---|
| v26 | abl2 | the redesign, pure | launching |
| v26b | abl3 | + contact term w1.0 (mime-risk hedge; the pair decides if explicit contact matching is still needed under anchored guidance) | launching |

**Plateau anatomy (CLAIMED_REFS instrument):** record ∩ champion = 54 refs (Jaccard 0.48), union
112/826 = 13.6% — 86% of references claimed by NEITHER; 2-checkpoint retrieval bound ≈ 0.136.
Feature forensic is WEAK (claimed refs only mildly shorter; medians equal): difficulty is
motion-content-level. Next data-tier levers: content-targeted augmentation (OmniRetarget mapping,
`papers/README.md`) or the task-reward stage.

True-protocol scoreboard after the first ablation round: **v21 `model_2000` = 0.1102 / 0.1053
re-eval — record CONFIRMED** (peak is checkpoint-sharp: neighbors 1900–2150 at 0.065–0.075) >
v14b 0.0944 > v17 0.0920 (own-matched) > v20 control 0.0872 > source 0.83 still the ceiling.
Mechanism verified end-to-end: the record checkpoint's under-15-step death rate is **19% vs the
champion's 30%** — zero-start 0.4 bought the designed cold-start robustness, and it converts to SR.

**Where the process-lever search stands (2026-07-10 evening):** eight recipe variants (v14–v19b)
all landed at or under the ~0.09 true-SR plateau (v14b 0.0944 / v17 0.0920). Training-distribution
t0 success reaches 0.4+ while eval sits at 0.09; training reward keeps rising past every SR peak;
schedules, exposure, caps and anchors each fixed their targeted failure and still didn't move the
plateau. The consistent reading: **the dense tracking objective, not the training process, is the
ceiling** — the gate is not what PPO optimizes. Next tier: task-reward fine-tuning (gate claim as
reward, tracking as regularizer) — Part-2 lever #1.

## Distillation line (distill-then-continue) — closed as a measured negative, with assets

Goal: bank the v9b peak into a fresh net (fresh critic/optimizer/σ), then continue PPO — the
mechanistic answer to v13b–d. Pipeline: `distill_collect` (teacher rollouts, training-style env,
DART noise — noised execution, mean labels) → `distill_bc` (TBPTT clone) → `--init_actor_ckpt`.

| step | run/artifact | result |
|---|---|---|
| BC scaling law | clone gates (64-ep evals, tags `v15-clone*-gate`) | val MSE 0.498 → SR 0.094; 0.368 → 0.188; 0.212 → 0.250 (more data + epochs + cosine LR; teacher 0.345) |
| v15 (first continue) | `79bwoinb` | crashed: `requires_grad`-off actor breaks recurrent PPO (cuBLAS INVALID_VALUE) → freeze reimplemented as gradient hooks |
| v15c | [to1el473](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/to1el473) | no warmup: clone 0.250 → **0.000 at probe@0**, re-climb to 0.121 only |
| v15d | [w18ldsv8](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/w18ldsv8) | grad-hook weight freeze alone: same collapse. Smoking gun: `model_0` weight tensors byte-identical to the student — the **obs-normalizer buffers alone** shifted (BC never trains them; PPO's first rollout installs live stats, std→~17, rescaling every input) |
| v15e | [couoxo3c](https://wandb.ai/wiin2-wirobotics-inc/chrisryu-simrl/runs/couoxo3c) | full init contract (normalizer frozen `until=0` + grad-hook freeze 150 + LR pin): **probe@0 = 0.253 — contract verified** — but post-unfreeze PPO still eroded it (0.149@250, 0.10@450; watcher-killed) |

Verdict: with std, optimizer, critic and normalizer all controlled, on-policy PPO still walks a
BC-initialized policy off its competence tube before value-guided improvement takes hold —
**BC is the endpoint for a student (→ Part-2 perceptive distillation), not a PPO restart.** The
scaling law and the init contract (`--init_actor_ckpt` freezes the actor's `EmpiricalNormalization`;
`WBT_ACTOR_FREEZE` grad-hooks) are the reusable assets.

## Datasets (S3, `s3://wirobotics-internal/chrisryu/sim_rl/trajectories/`)

| set | contents | used by |
|---|---|---|
| `20260707/...collect-955759b` | old-gate WBT refs (min-lift 8) | v4–v6b (obsolete) |
| `20260708/...fullcurr-iter1600_collect-6893081` | canonical n1000 (1000/1160 episodes pass the claim); v9b-era builds filtered to 826 with `--min_lift_cm 8`, later builds keep 1000 (`--trim_head 2`) | v7–v9b, v13*, v14b/c, v15* |
| `20260709/...iter1600-hold30_collect-0f60fd8_n2000` | hold-30 collection, variant-recorded | v10, v12, v14 |
| `20260709/...iter1600_collect-f1c353b_n2000` | default-hold recollect, variant-recorded (the shipped set) | future |

## Checkpoints of record (S3, `.../sim_rl/ckpts/`)

Fixed-protocol era (canonical-826, seed 7; RSI / random-start):
- **dual-axis champion**: v44 `2026-07-13-13-08_v44-dualaxis-s44-12k_12000it_1024envs-bfacb6b/model_11999.pt`
  — 0.8087 / 0.7312 (band-mates: v38 model_10000 0.7857/0.7131, v43 model_11999 0.7337/0.7276)
- **RSI-axis record (artifact)**: v40 `2026-07-13-04-08_v40-fixedinit-v14b-s44-12k_12000it_1024envs-8bf10ac/model_7400.pt`
  — 0.8729 / 0.1630 (vs source 0.8668; recipe non-reproducing, see CURRENT STANDINGS)
- dose curve point 0.5: bgou6xmw `2026-07-13-13-32_v42-fixedinit-norsi05-12k_12000it_1024envs-bfc1277/model_7200.pt` — 0.6586 / 0.6356

Broken-eval era (kept for lineage; scores not comparable):
- v9b: `2026-07-09-00-55_hammer-lift_3500it_1024envs-nogit/model_1600.pt`
- v14b: `2026-07-09-18-16_v14b-holdpartial-entsched-stdcap-canonical826_3000it_1024envs-nogit/model_2700.pt`
- BC students: node `/home/ubuntu/distill/student{,2,3}_v15-distill-v9b1600.pt` (+ v15e run ckpts on S3)

## Doctrine index (details in README)

1. Reference feasibility is decisive — demonstrations must satisfy the gate (0/888 → SR 0).
2. Run-to-run lottery dominates config deltas at this scale (v13 control).
3. The explore/collapse arc is schedulable: entropy anneal + std ceiling beat the lucky-tail record
   row-for-row with a plateau tail (v14b) instead of a knife-edge (v9b) — and the anneal must END
   before the expected peak zone (v14c: anneal :2500 kept the zone jagged and unconsolidated).
4. Peaks are harvested, not polished: dense checkpoints + KPI probes + confirmed-decline kill;
   direct resumes (v13b–d) and clone continues (v15c–e) are both measured negatives.
5. A checkpoint's obs-normalizer stats are part of its input contract.
6. Long-hold references are harder tracking targets — and free in-hold starts poison phase-RSI
   (guard exists, `WBT_PHASE_GUARD`).
7. The terminal failure mode is the 4-finger carry; partial-credit hold (`WBT_HOLD_PARTIAL_*`)
   gives the 5th finger a gradient.
8. The straitjacket is real and protocol-portable: random starts beat the reference's own recorded
   start ×1.56–1.72 on both reward bases (and 0.258-vs-0.000/256 on the ref-0 pair) — start
   diversity, not more tracking pressure, is the lever that gap points at (`WBT_NO_RSI_PROB`).
9. Infra: pgrep gap-wait guards belong in node-side script FILES. An inline `bash -c` whose cmdline
   carries the guarded pattern becomes a self-matching ghost (two such shells mutually deadlocked
   and starved node1's whole eval pipeline overnight — the bracket idiom can't save a cmdline that
   ALSO contains the plain string).
10. Ghost pattern #3 (same family): a launcher shell that writes a node-side script via an inline
    heredoc carries the script's text — including any guarded pattern — in its OWN cmdline; if the
    session lingers, every guard starves (cost the refmap sweep its first hour). Ship node-side
    scripts by scp, not inline heredocs; kill tools with plain-name text in their cmdline
    (pkill inside a heredoc) also self-match — split write and kill into separate sessions.
    Ghost pattern #4 (RETRACTED as stated, one sighting later): the Q3-kill PenExceed was first
    read as a launcher retry-loop resurrecting the killed trainer — then v44's node showed the
    IDENTICAL signature 4 min later after a NATURAL 12k completion (no kill, no resurrection
    possible). Refined again after a THIRD sighting (v34c, natural end — 3/3 run-ends: v44 +
    v34c natural, Q3 killed): **the PenExceed fires in the training server's close/teardown
    window**, always reporting exactly 1024 envs (the training server's own count), immediately
    after `[sim-service] closed` (= trainer `env.close()`). Sub-mechanism unresolved from
    severed session logs — dying server's uncontrolled drain-steps tripping the guard vs a
    stray post-close boot. Zero effect on results (wandb closes finished; verdict sweeps boot
    clean minutes later; no zombie steps). DEFERRED FORENSICS: inspect live at v45b's natural
    end (its node survives the completion) — pen_exceed JSON contents, uncut log ordering,
    GPU/process state. Kill-order hygiene (take down any launcher loop before hand-killing a
    trainer) stays as advice.
11. The save interval must divide the val cadence: peaks are harvested, and an unsaved peak is a
    lost peak — the repro's val-100/save-200 mismatch left its 0.3008@1100 with no checkpoint
    (nearest saved: 0.266@1600). The 12k arms align both at 200.
12. NO kill without reading every axis of the run's own instruments in the same sitting — and
    read the TRENDS, not points (v40 killed at 0.827-rising off a stale random-start snapshot;
    v42 killed while its random-start curve accelerated 0.07→0.20→0.33).
13. One delta per run, named base in the run notes; continuations carry no delta; multi-confound
    hybrids (warm-starts across regimes) are labeled EXPLORATORY and excluded from lever tables;
    no new levers until the running single-variable set concludes (user-directed 2026-07-13).
14. THE ANNOTATION CONTRACT (user-directed, v14b line @4b46954): overlay channels DERIVE from
    the live reward manager (every active non-regularizer term = one auto-labeled curve — no
    hand-kept mapping can be missing, structurally), and a BUILD-TIME validator probe-evaluates
    every term (must be a [N] 0..1 kernel, NaN allowed for gated windows, or a declared
    KNOWN_REGULARIZER rendered as a labeled 'reg:' segment) — anything else refuses to serve.
    [UNPLOTTED] is now a defensive dead path: a run that boots cannot miss an annotation.

