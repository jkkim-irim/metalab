# learning/scripts/local — workstation (local-GPU) sim-service runs

The **local** peer of [`../aws`](../aws). The trainer and the MetaLab engine's sim-service
(`sim/<engine>/server.py`, `<engine>` ∈ {genesis, newton}) run as **two processes over an RPC localhost
socket** — both in the engine's **uv venv** (`~/.metalab/venvs/<engine>`), on this machine's GPU. No AWS
node, no SSM, no conda. Use these when you train/eval on your own workstation; use `../aws` for a GPU node
(which just runs this same `metalab_train.sh` on the node over SSM).

| local | aws counterpart | what it does |
|---|---|---|
| `lib.sh` | `aws/lib.sh` | shared config — uv venv layout (`METALAB_VENV_ROOT`, `engine_venv`), `RL_LOG_ROOT`, `log`. Sourced by both scripts. |
| `metalab_train.sh` | `aws/metalab_train.sh` | train on the local GPU (`learning.train --trainer rl`, RPC). Checkpoints mirror to S3 by default; optional per-checkpoint eval recording + synced HTML report (`--record`). |
| `metalab_eval.sh` | (node runs this too) | eval/watch a **local** checkpoint over RPC — SR + per-episode S/F json; optional rerun `.rrd` + report recording (kept local). |

## Layout (override from the env; defaults in `lib.sh`)

```
METALAB_VENV_ROOT  $HOME/.metalab/venvs        # uv engine venvs: <root>/genesis, <root>/newton
RL_LOG_ROOT         <repo>/logs/rl               # checkpoints: <exp>/<run_name>/model_*.pt
```

The engine venvs are built from the committed uv projects (`sim/_setup/<engine>/{pyproject.toml,uv.lock}`)
by `setup_env.sh` (`uv sync` — clones the pinned sim source as a sibling repo, no conda, no S3 snapshot).
`metalab_train.sh`/`metalab_eval.sh` call `setup_env.sh` automatically (no-op if the venv already matches the lock).

## Train

`--sim {genesis|newton}` **and** `--task <t>` are **REQUIRED** (no defaults — fail-loud; a missing `--task`
lists the tasks for that backend). Headless by default; add `--viz gl` to open that backend's GUI + a live
web dashboard on this machine.

```bash
learning/scripts/local/metalab_train.sh --sim newton  --task hammer-lift --num_envs 4096 --max_iterations 30000
learning/scripts/local/metalab_train.sh --sim genesis --task hammer-lift --num_envs 4 --viz gl   # live GUI, 4 envs
learning/scripts/local/metalab_train.sh --sim newton  --task hammer-lift --device cuda:1         # engine on physical GPU 1
nohup learning/scripts/local/metalab_train.sh --sim newton --task hammer-lift --num_envs 4096 > train.log 2>&1 &
```

All run knobs (rewards, curriculum, PPO, action scales, DR) live in `learning/rl/dexblind/<task>/experiment.py`
— edit that, not the script. `--task` selects both the experiment and the sim env package.

RPC: `metalab_train.sh` activates the engine's uv venv and runs `learning.train`, which spawns
`sim/<engine>/server.py` as a second process in the SAME venv and drives it over `127.0.0.1` (single-venv
RPC — the team-required path). `wandb` (scalar curves: loss/SR) is ON by default; `--no_wandb` disables it.

### Checkpoints → S3 (ON by default)

**S3 is the checkpoint store** — a workstation run publishes to the same place, and in the same layout, as a
node run, so a checkpoint is findable without knowing which machine trained it:

```
s3://wirobotics-internal/jkkim/sim_rl/ckpts/<run_name>/model_*.pt
```

```bash
learning/scripts/local/metalab_train.sh --sim newton --task hammer-lift --num_envs 4096   # S3 mirror is already on
SYNC_INTERVAL=120 ...                    # sync every 2 min (default 300 s)
CKPT_S3_ROOT=s3://.../my/ckpts ...       # publish somewhere else
KEEP_LOCAL_CKPTS=1 ...                   # keep every local copy (see pruning below)
--no-s3-sync                             # keep this run off S3 entirely (local only, nothing pruned)
```

Each checkpoint goes up the moment it is written (`OnPolicyRunner._upload_ckpt_s3`), plus a mirror sweep
every `SYNC_INTERVAL`s and a final one on exit. S3 is never pruned (no `--delete`).

**Local retention — same on a workstation as on a node**: after a *successful* upload the local `model_*.pt`
are pruned to the **newest one**. The newest is kept deliberately: it may still be mid-write, the
per-checkpoint video/report hook loads it, and `metalab_eval.sh` picks it up — so re-evaluating the LATEST
checkpoint needs no S3 pull. An older one is fetched first:

```bash
aws s3 cp s3://wirobotics-internal/jkkim/sim_rl/ckpts/<run_name>/model_400.pt /tmp/
learning/scripts/local/metalab_eval.sh --sim newton --task <task> --checkpoint /tmp/model_400.pt
```

A workstation has no instance role, so this needs local creds (`aws login`). With creds **missing**: if S3 was
merely the default the run WARNS loudly and continues local-only (nothing is pruned); if you asked for it
explicitly (`--s3-sync` / `S3_SYNC=1`) that is a hard error instead.

### Training-time eval recording + report (`--record`)

`--record` (OFF by default) records a progress rollout **by itself** — no second terminal, no eval script.
After each checkpoint the trainer itself (`rl_trainer._make_record_callback`) spins up a short-lived
sim-service and rolls that checkpoint out. The rollout is saved as a **rerun `.rrd`** (not MP4), together with
the per-step series, and the checkpoint is never touched.

Both halves go into one **synced report** (`report.html`: an embedded rerun viewer on the left — orbit, zoom
and scrub the actual scene — with time-synced obs/reward/action/joint-state plots on the right, one top-level
tab per recorded env; see `sim/metalab/runtime/rollout_report.py`). It is published **next to that run's
checkpoints**, one subdir per checkpoint, and only its LINK is logged to W&B (`val/report`) on that
checkpoint's training step:

```
s3://wirobotics-internal/jkkim/sim_rl/ckpts/<run_name>/report_<iter>/{report.html,rollout.rrd,data.json}
    → https://d1iitptfxhu64e.cloudfront.net/jkkim/sim_rl/ckpts/<run_name>/report_<iter>/report.html
```

Open it through **CloudFront** — the link W&B logs. A *presigned* S3 URL signs ONE key, so the page's
relative `rollout.rrd` fetch comes back `403`: the plots render (the series are inlined) and the 3D pane
degrades to a message.

Without an S3 destination (`--no-s3-sync`, or no creds) the report is kept at
`<log_dir>/report_<iter>/` instead. Measured against the old MP4 path at 4 envs: the recording rollout went
from **118 s to 41 s** per checkpoint (no offscreen render, and the CUDA graph is no longer disabled by a
viewer), at ~13 MB of `.rrd` per checkpoint.

```bash
learning/scripts/local/metalab_train.sh --sim newton --task hammer-lift --num_envs 4096 --record   # + per-checkpoint report link
RECORD_ENVS=6 ... --record      # envs given a series + a report tab (default 4)
RECORD_STEPS=0 ... --record     # policy steps per recording (0 = full episode; default 600)
```

- The link goes to the **same W&B run** as the scalars (no sibling run), so a curve and its replay share one
  step axis. A failed record/publish logs a WARN and training continues (best-effort).
- One panel: `val/report`, this checkpoint's link as media, so it streams into the panel LIVE (no page
  reload; drag its step slider for the earlier checkpoints). A `val/reports` TABLE of all checkpoints was
  tried and reverted — its cell is a client-artifact reference the table panel serves from cache, so rows
  only appeared after a page reload.
- **BLOCKING**: the recording runs on the training thread, so the loop pauses at each checkpoint. Its few
  envs share the training GPU.
- Mobile: the rerun viewer is desktop-only (`Mobile OSes are not yet supported`); the plots still work.
- On a **display-less node** the offscreen render needs no X server: the newton parser routes pyglet to EGL
  when `DISPLAY` is unset (`backends/newton/parser.py`), the same switch Isaac Lab's video capture uses.

## Eval / watch a local checkpoint (`metalab_eval.sh`)

Evals a checkpoint over the RPC sim-service (RPC-only) → SR + a per-episode S/F json. Headless by default it
records the rollout as a rerun `.rrd` + `report.html` **next to that json (local — S3 publishing was
removed)**; `--viz` instead opens the backend's live viewer (no recording).

```bash
learning/scripts/local/metalab_eval.sh --sim newton --task hammer-lift --viz                 # live watch (∞, Ctrl-C) — no recording
learning/scripts/local/metalab_eval.sh --sim newton --task hammer-lift                        # headless: SR + .rrd/report (local)
RECORD=0 EPISODES=64 learning/scripts/local/metalab_eval.sh --sim newton --task hammer-lift --num_envs 16   # 64-ep SR, no recording
CHECKPOINT=logs/rl/.../model_800.pt learning/scripts/local/metalab_eval.sh --sim newton --task hammer-lift  # a specific local ckpt
GPU=1 learning/scripts/local/metalab_eval.sh --sim newton --task hammer-lift                  # on physical GPU 1 while training holds GPU 0
```

- **Checkpoint**: default = newest **local** `model_*.pt` for the task under `RL_LOG_ROOT`. `CHECKPOINT=<path>`
  picks one. To eval an S3-only checkpoint, `aws s3 cp` it under `logs/rl/.../` first (this tool never pulls S3).
- **Recording** (default, headless): the server writes `rollout.rrd` + `data.json` + `report.html` into the
  run's output dir beside the meta json — **kept local**, no S3 upload. `RECORD_ENVS` (3) sets how many envs
  get a series + a report tab. `RECORD=0` skips it, `RRD=0` keeps the SR run without a recording.
- **`EPISODES`**: `-1` (default) = infinite watch (Ctrl-C → prints SR + writes the meta json) · `>0` = stop
  after N episodes · `0` = a fixed `STEPS`.
- **`GPU=N`** sets `CUDA_VISIBLE_DEVICES=N` (seen as `cuda:0` in-process). Use `GPU=1` to eval while a training
  run holds GPU 0.
- **`--viz`** opens the `--sim` backend's live GL viewer (needs a display) and disables recording. SR is judged at the
  curriculum-END criteria on every path (over RPC via `SimServiceVecEnv.apply_curriculum_end`).
