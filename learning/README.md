# learning/

Training and evaluation for ALLEX policies — **start here** for setup, local + AWS
training/testing, and the layout.

## Layout

```
learning/
├── train.py                 # training entrypoint (config + accelerator + split; dispatches a trainer)
├── trainer/
│   ├── bc_trainer.py        # BC trainer (ACT / VLA): epoch loop + episode-wise validation
│   └── tests/               # test_bc_trainer.py
├── data/
│   ├── dataset.py           # episode-wise train/val split + EpisodeAwareSampler DataLoader
│   ├── conversion/          # raw -> LeRobot v3.0 (132-D) converter + canonical build recipe
│   └── tests/               # test_dataset.py
├── metrics/
│   └── validation.py        # compute_val_loss + compute_per_horizon_metrics
├── model/
│   └── act_policy.py        # ACT policy / pre+post processors / optimizer construction
└── scripts/                 # node-side recipes (run ON the node)
    ├── _setup_venv.sh       # (sourced) ensure uv + ffmpeg + the venv + pinned deps
    ├── train.sh             # ensure env, sync dataset from S3, run the recipe
    ├── run_ci_tests.sh      # ensure env, run the full gate (ruff + all tests)
    └── aws/                 # client-side AWS drivers (run on your laptop)
        ├── launch_aws.sh    # launch a managed GPU node (gpu-launchers profile, not admin)
        ├── train_aws.sh     # scp code to a node over SSM, launch training
        └── test_aws.sh      # scp code to a node over SSM, run the test gate
```

`learning/` is a package (each dir has `__init__.py`). `train.py` is a thin entrypoint that
dispatches to a trainer in `trainer/` (BC today; RL later); modules import package-qualified —
`from learning.data.dataset import …`, `from learning.metrics.validation import …`,
`from learning.model.act_policy import build` — so the split, policy build and validation path
are shared rather than re-implemented.

Module names are policy-specific where it matters (`act_policy.py`) so other policies — e.g. a
future `vla_policy.py` — can sit alongside without one monopolizing a generic name.

## Setup

Repo-wide dev setup (git hooks, lint) is in the [top-level README](../README.md). For training,
install the package once (editable) so `learning` imports cleanly:

```sh
pip install -e .               # from the repo root
```

**For AWS runs (optional)** — only if you'll use the `scripts/aws/*` drivers. They require a
**gpu-launchers** AWS profile (the scripts verify `gpu-launchers` IAM-group membership and refuse
otherwise) and an SSH key; training logs to Weights & Biases. Ask the account admin (**chrisryu**)
for a gpu-launchers profile, then set these once in your shell rc (`~/.zshrc`):

```sh
export AWS_PROFILE=<your-gpu-launchers-profile>   # node launch + SSM, IAM-scoped to your nodes
export SSH_KEY=~/.ssh/<your-key>                  # private key for scp to the node (ssh user: ubuntu)
wandb login                                       # writes the api.wandb.ai entry to ~/.netrc
```

## Running locally

```sh
python -m learning.train ...   # train + validate
python -m pytest learning      # all tests (co-located under each package's tests/ dir)
ruff check learning            # import order + lint
```

`scripts/run_ci_tests.sh` runs the full pre-PR gate (ruff + the whole test suite) in one go; the
pre-commit hook runs the ruff half on every commit.

## Running on AWS

Provision a single-GPU node, then deploy + run on it over SSM (no S3 for code; your `~/.netrc` is
copied to the node so the run logs to W&B). `AWS_PROFILE`/`SSH_KEY` come from your shell env, so you
only pass `INSTANCE_ID`:

```sh
bash learning/scripts/aws/launch_aws.sh                  # launch a 1x L40S node -> prints INSTANCE_ID
INSTANCE_ID=<id> bash learning/scripts/aws/test_aws.sh   # ruff + full test suite on the node
INSTANCE_ID=<id> bash learning/scripts/aws/train_aws.sh  # train (reproduces the 5m3dzdwe recipe)

# short smoke — intra-epoch validation every 5 steps for 100 steps:
INSTANCE_ID=<id> EXTRA_FLAGS="--epochs 1 --max_iters_per_epoch 100 --val_interval 5 \
  --val_max_batches 2 --log_every_n_steps 5" bash learning/scripts/aws/train_aws.sh
```

`launch_aws.sh` defaults to `g6e.2xlarge` (1× L40S; override `INSTANCE_TYPE`), names the node
`<profile>-<gpu>`, and tags it `ManagedBy=gpu-launcher`. Terminate when done:
`aws ec2 terminate-instances --instance-ids <id>`.

## Notes

- The custom epoch/split flags (`--epochs`, `--val_ratio`, `--val_freq_epochs`,
  `--save_freq_epochs`, `--split_seed`, `--early_stop_patience`,
  `--val_only_first_n_episodes`, `--max_iters_per_epoch`, `--rollout_horizons`,
  `--log_every_n_steps`, `--val_interval`, `--val_max_batches`) are stripped from `argv` **before** LeRobot's parser is imported,
  so LeRobot only sees its own flags. Every other flag is a standard LeRobot CLI flag.
```

