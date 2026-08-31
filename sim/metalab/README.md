# MetaLab — setup & reproducibility

Engine-agnostic robot-learning sim (**Newton** / **Genesis**) with a one-click web console. This is the
onboarding manual: from a fresh clone to a training run — reproduced **identically** on a local
workstation or an AWS GPU node. Engine environments are **uv-managed** (`uv.lock` per engine) — no conda.

## TL;DR

First time on a fresh box? Skim [What you need first](#what-you-need-first) — `sim/metalab/setup.sh` installs
`uv` for you, but you still need a GPU + driver, `git`, and `build-essential`.

```bash
git clone git@github.com:wirobotics-rih/allex.git ~/allex_ws/allex && cd ~/allex_ws/allex
sim/metalab/setup.sh              # clone the pinned sims + `uv sync` each engine env (from uv.lock)
sim/metalab/launchpad.sh       # open the web console → pick engine / task / train|eval → Launch
```

That's the whole local setup. `sim/metalab/setup.sh` is idempotent — re-run it any time. It installs `uv`
automatically if missing (uv also provides Python — no miniconda needed).

## What you need first

| Requirement | Why | If missing |
|---|---|---|
| **Ubuntu** 22.04 / 24.04 | tested platform | — |
| **NVIDIA GPU + driver** (≥ 550) | runs the physics + training | install the driver; `nvidia-smi` must work |
| **C toolchain** (`build-essential`) | the sim sources build from source | `sudo apt install build-essential` |
| **git** | clones the repo + the sim sources | `sudo apt install git` |
| **wandb** login | runs log to wandb by default | `wandb login`, or train with `--no_wandb` |
| **~30 GB disk** | 2 uv envs + sim sources + in-repo meshes | free up space |
| `uv` | manages each engine env + Python | **auto-installed by `sim/metalab/setup.sh`** (or `curl -LsSf https://astral.sh/uv/install.sh \| sh`) |
| Chrome *(optional)* | the Launchpad opens as a standalone **app window**; without it, falls back to your default browser | — |

> No miniconda, and **no AWS account needed to set up the env** — deps come from public indexes (PyPI,
> PyTorch cu128, NVIDIA). AWS is only for training on a GPU node or S3 checkpoints (see [Connect AWS](#connect-aws)).

## What `sim/metalab/setup.sh` does

1. **Clones the pinned simulator sources** as *siblings* of this repo (at the exact pinned commit):

   ```
   allex_ws/
   ├── allex/           ← this repo
   ├── newton/          ← pinned commit
   └── genesis-world/   ← pinned commit
   ```

   Commits are pinned in [`sim/metalab/sim_versions.env`](sim_versions.env) — the single source of
   truth — so everyone builds against the exact same simulator revision.

2. **`uv sync`s each engine's env** (`newton`, `genesis`) from its committed `uv.lock`
   ([`sim/metalab/backends/<engine>/env/`](backends/genesis/env)) into a venv under `~/.metalab/venvs/<engine>`. The lockfile pins
   every dependency (torch cu128, warp, usd, …) by exact version + hash — reproducible from git, no
   conda-pack, no S3 snapshot. The engine source is installed *editable* so it stays hackable.

## Run

Web console (recommended):

```bash
sim/metalab/launchpad.sh       # pick engine · task · train/eval · knobs → Launch; live log in the right pane
```

Or straight from the CLI — the Launchpad just shells out to these (each activates the engine's uv venv):

```bash
# first-run smoke — tiny, no wandb needed (verifies the whole setup end-to-end):
learning/scripts/local/metalab_train.sh --sim genesis --task hammer-lift-teacher --num_envs 4 --max_iterations 5 --no_wandb

# real runs (log to wandb — see below):
learning/scripts/local/metalab_train.sh --sim genesis --task hammer-lift-teacher
learning/scripts/local/metalab_eval.sh  --sim genesis --viz --num_envs 1     # watch the newest local checkpoint
```

> The **first** `metalab_train.sh` for an engine auto-runs `sim/metalab/setup.sh`'s work (clone pinned source + `uv
> sync`) if you skipped it — so on a fresh box you can go straight to the smoke line above.

## Reproducibility model

A run is determined by four version-controlled pieces, so "clone + `setup.sh`" reproduces the setup
identically, local or AWS:

| Piece | Where | Pinned by |
|---|---|---|
| training / env / task code | this repo | git SHA (embedded in every run name) |
| simulator source (newton, genesis) | sibling repos | `sim/metalab/sim_versions.env` |
| python dependencies | `uv.lock` (per engine, in-repo) | uv (exact version + hash) |
| robot / object assets | in-repo `sim/metalab/assets/` (plain git, no LFS) | git SHA |

> Reproducible *setup* — yes. Bit-identical *results* — not guaranteed across different GPU models /
> driver / CUDA versions (GPU nondeterminism). Same procedure, same code; results match up to that.
>
> Newton pins a **warp-lang dev build** from NVIDIA's index (newton itself is a dev release — no stable
> warp satisfies it). It's pinned by hash in `uv.lock`, so it's reproducible as long as that nightly
> stays on the index.

## Connect AWS

Only needed to **train on a GPU node** or read/write **S3 checkpoints** — *not* for the local env. You
need access to the team AWS account. Sign in with the AWS CLI's console-based login (temporary
credentials, auto-refreshed while your console session is valid):

```bash
aws login                       # opens the browser → sign in to the AWS console
aws sts get-caller-identity     # verify you're connected
```

Credentials expire when your console session lapses (by design — no permanent keys); just re-run
`aws login` when that happens.

### Train on an AWS GPU node

Same engines, same trainer — reached over SSM. The node builds the env the same way (`uv sync` from the
committed lockfiles), so nothing depends on your workstation once it's running:

```bash
AWS_PROFILE=<gpu-launchers profile> \
  learning/scripts/aws/metalab_train.sh --sim genesis --task hammer-lift-teacher --num-envs 8192
```

You must be in the **gpu-launchers** group; the node role is a one-time setup (ask jkkim). Details in the
`learning/scripts/aws/metalab_train.sh` header. **Stop idle GPU nodes** — they dominate cost.

> ⚠️ **The AWS node path is still being migrated to uv** (node bootstrap + `uv sync` on the node). Local
> is done and verified; verify the AWS path before relying on it.

## wandb

Runs log to wandb **by default**. If you are not logged in, `metalab_train.sh` **fails loudly before the run
starts** (rather than dying deep inside `wandb.init`). Pick one:

```bash
wandb login                # log in (writes ~/.netrc), or
export WANDB_API_KEY=...    # provide a key, or
...  --no_wandb             # disable logging entirely (Launchpad: check "wandb 끄기")
```

## Updating a sim version (env owner)

When a new newton/genesis commit is validated:

1. set the matching `*_REF` in `sim/metalab/sim_versions.env`
2. re-checkout the sibling + re-lock: `sim/metalab/setup.sh` (or `uv lock` in `sim/metalab/backends/<engine>`)
3. commit `sim_versions.env` + the updated `uv.lock`, merge → teammates re-run `sim/metalab/setup.sh` to converge

No S3 snapshot to rebuild — the lockfile *is* the reproducible artifact.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `uv: command not found` | Re-run `sim/metalab/setup.sh` (auto-installs), or `curl -LsSf https://astral.sh/uv/install.sh \| sh` then add `$HOME/.local/bin` to PATH. |
| `uv sync` fails compiling a source dep | Install a C toolchain: `sudo apt install build-essential`. |
| `ModuleNotFoundError: genesis` / `newton` | Sibling source missing or on the wrong commit — re-run `sim/metalab/setup.sh`. |
| wandb error at launch | See [wandb](#wandb). |
| Launchpad opens as a normal Chrome tab, not an app window | Chrome isn't installed; it fell back to your default browser. |

## Newton RTX viewer — OVRTX (opt-in)

Newton's `--viz gl` opens an **OpenGL** window; `--viz rtx` uses **OVRTX** (NVIDIA's real-time path tracer),
presented in a pyglet window. (`--viz none`, the default, is headless. `--viz` with no value fails loud.)

This is **opt-in** — `ovrtx` is *not* in the default newton env (headless training / AWS never render, and it
needs an RTX GPU). `metalab_train.sh` syncs the `rtx` extra **automatically when you pass `--viz rtx`**:

```bash
learning/scripts/local/metalab_train.sh --sim newton --task hammer-lift-teacher --num_envs 1 --viz rtx
```

To install it into the newton env by hand (e.g. for `metalab_eval.sh`, which does not auto-sync it yet):

```bash
UV_PROJECT_ENVIRONMENT=~/.metalab/venvs/newton \
  uv sync --project sim/metalab/backends/newton/env --extra rtx
```

The default (`--viz none`) install is unaffected — newton training / AWS stay lean.
