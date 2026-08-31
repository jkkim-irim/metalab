# Newton sim-service — node/deploy toolkit (learning-side)

Launch + set up a GPU node with the Newton / Isaac Lab env from a prebuilt **S3 snapshot**; the eval
(`rl_eval.sh`) then deploys + runs the sim service over **SSH-over-SSM** — no inbound ports, no shared
keys, and **no project code on S3**. (Training deploy `rl_train.sh` lands with the trainer, #18.) Node
provisioning is learning-driven; the sim (`sim/isaaclab/`) only owns its **env definition** (deps build).

```
learning/scripts/aws/       # node provisioning + deploy — runs LOCALLY (your machine)
  lib.sh       # shared config + helpers: AWS, SSH-over-SSM tunnel + rsync deploy (deploy_sim_and_learning). Sourced; override via env.
  setup_sim_node.sh     # node lifecycle:  up | launch | provision | sync | status | ssh | logs | stop | start | down
  rl_eval.sh        # provision (idempotent) -> deploy -> eval over the boundary
  remote/           # run ON the node (shipped there by setup_sim_node.sh)
    provision.sh    #   apt deps + restore the S3 env snapshot (conda env + IsaacLab fork)

sim/isaaclab/scripts/       # the SIM's env build (deps installation) — the sim owns its dependencies
  rebuild_env.sh    #   build the kitless IsaacLab(release/3.0.0-beta2)+Newton env from scratch (on a node)
  snapshot_env.sh   #   snapshot that built env -> S3 (the artifact provision.sh restores)
```

`setup_sim_node.sh` only manages the **node + its env** (launch → restore snapshot → lifecycle). The **eval run
script lives in [`learning/scripts/aws/`](../../../../learning/scripts/aws)** — driven by `learning/`
(which owns the experiment), it rsyncs **both** `sim/isaaclab/` and `learning/` to the node and runs the
eval through the service boundary:

- `learning/scripts/aws/rl_eval.sh` — deploy + eval a checkpoint over the client (`learning.eval.eval_service`).

(The training deploy — `rl_train.sh` — lands with the trainer in the follow-up PR.)

## Prerequisites (per teammate, one time)

1. **AWS CLI v2** + **session-manager-plugin**
   (`brew install awscli && brew install --cask session-manager-plugin`).
2. An **AWS profile** whose IAM allows, on `ManagedBy=gpu-launcher` instances:
   `ec2 Run/Start/Stop/Terminate/Describe*`, `ssm:StartSession` (document `AWS-StartSSHSession`),
   `ssm:Describe*`, and `iam:PassRole` for the SSM instance profile. The launched node needs an instance
   profile with **`AmazonSSMManagedInstanceCore`** (default `IAM_INSTANCE_PROFILE=project-x-ssm-profile`).
3. An **SSH keypair** (`ssh-keygen -t ed25519`, default `~/.ssh/id_ed25519`). Your *public* key is injected
   into the node via EC2 user-data — only you can log in.
4. The node's IAM role must have **S3 read** on `ENV_SNAPSHOT_S3` (the env snapshot restored at provision).

## Reproduce the eval on a bare node

Everything restores from the S3 env snapshot — no hand-built venv, no project code on S3:

```bash
export AWS_PROFILE=<your-gpu-launcher-profile>        # only if not your default profile

# from the repo root:
learning/scripts/aws/setup_sim_node.sh up                   # launch an L40S + restore the env snapshot (~10 min)
NODE=<instance-id> learning/scripts/aws/rl_eval.sh    # deploy sim+learning, pull the ref ckpt, eval over the boundary
learning/scripts/aws/setup_sim_node.sh stop                 # BILLABLE until stopped (EBS + checkpoints persist)
```

`setup_sim_node.sh up` prints the instance id (also saved to `~/.allex-node`). `rl_eval.sh` pulls the reference
checkpoint (`REF_CKPT_S3`, a run-name path `…/ckpts/<run_name>/model_<iter>.pt`; override with
`CHECKPOINT=<node path>` + `CHECKPOINT_S3=<s3 source>` for a node-local one) and prints
`EVAL_OVER_SERVICE_OK ... SR=...`.

### Env snapshot image

`ENV_SNAPSHOT_S3=s3://wirobotics-internal/chrisryu/sim_rl/envsnap` holds two tarballs, restored by
`remote/provision.sh` with the node's own IAM role. They are a **coupled pair** — the conda env
editable-installs the fork, so its site-packages point into `~/IsaacLab`; restore both or neither. The
prefix carries its own `README.md` (source: `sim/isaaclab/scripts/ENVSNAP_README.md`, re-uploaded by
`snapshot_env.sh`) so a future S3 browser doesn't need this repo to understand it:

| object | ~size | restores to | contents |
|---|---|---|---|
| `env_isaaclab.tar` | 5.2 GB | `~/miniconda3/envs/isaaclab` | the conda env: kitless Isaac Lab `release/3.0.0-beta2` + Newton + mujoco-warp, py3.12, **torch / tensordict / wandb** — a superset, so the Isaac-free trainer + eval run here too |
| `isaaclab.tar` | 109 MB | `~/IsaacLab` | the Isaac Lab fork source (+ `allex_description` baked in) |

The Newton stack is **kitless** (no Isaac Sim / Omniverse) and not cleanly pip-installable, so provisioning
restores this snapshot rather than building from scratch. Rebuild it from a known-good node with
`sim/isaaclab/scripts/snapshot_env.sh s3://.../envsnap`.

## Common overrides (all via env)

| var | default | meaning |
|---|---|---|
| `AWS_PROFILE` / `AWS_REGION` | _default chain_ / `us-east-1` | AWS creds + region |
| `INSTANCE_TYPES` | `g6e.4xlarge g6e.2xlarge g5.4xlarge g5.2xlarge` | capacity-first list (L40S / A10G — **need RT cores** for Newton) |
| `ROOT_GB` | `300` | gp3 root EBS (snapshot ~12 GB + assets + checkpoints) |
| `ENV_SNAPSHOT_S3` | `s3://wirobotics-internal/chrisryu/sim_rl/envsnap` | env snapshot (`env_isaaclab.tar` + `isaaclab.tar`) |
| `SSH_PUBKEY` | `~/.ssh/id_ed25519.pub` | key injected into the node |
| `NODE_ID` | _state file_ | target a specific instance |

## How it fits together

- **Transport:** nodes have no inbound ports. `lib.sh` writes a tiny ssh wrapper that tunnels through
  `aws ssm start-session` (`AWS-StartSSHSession`); `ssh`, `rsync`, and `scp` all ride it. No bastion, no public SSH.
- **Provisioning (self-contained):** `setup_sim_node.sh provision` runs `remote/provision.sh`, which restores the
  S3 env snapshot with the node's own IAM role — no clone, no `isaacsim`, no reference to any running box.
- **Code + assets:** `rl_eval.sh` rsyncs `sim/isaaclab/` → `/home/ubuntu/sim/isaaclab` and
  `learning/` → `/home/ubuntu/learning_repo/learning`, then editable-reinstall the sim packages in the
  isaaclab env. Task assets (~600 MB) transfer once; later runs send only deltas.
- **One env:** the isaaclab snapshot env is a **superset** (torch / tensordict / wandb), so both
  `server.py` and the Isaac-free trainer / eval run in it — nothing else to build on a fresh node.

## Notes / gotchas

- **Newton needs RT cores** — A100 / H100 / V100 cannot render it. Stick to L40S (g6e) / A10G (g5).
- Checkpoints persist on the node's EBS while **stopped**; `setup_sim_node.sh down` destroys them — fetch first.
- Capacity: GPU types are scarce; `setup_sim_node.sh launch` retries across the type list. If all fail, try later
  or widen `INSTANCE_TYPES`.
- Rebuild the env snapshot from a known-good node with `sim/isaaclab/scripts/snapshot_env.sh s3://.../envsnap`.
