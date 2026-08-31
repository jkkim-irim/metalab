# GR00T-N1.7 adapter

NVIDIA GR00T **N1.7** (3B VLA) wrapped as a policy inside `learning/`: `--policy.type groot` reuses the
same wir_v1 data, gpu_aug, validation, and `BCTrainer` as ACT. The model is **vendored in-repo** under
`learning/model/groot/` (the `learning.model.groot` package) — there is no external `gr00t` package
dependency. `transformers` (the Qwen3-VL backbone) is still a dependency here — dropping it is a later
internalization step. The pristine `allex_groot/Isaac-GR00T` checkout is kept only as an upstream
reference for diffing/re-syncing the vendored copy; it is never installed, cloned, or imported at runtime.

## Design

- **`GrootPolicy`** (`learning/model/groot_policy.py`) wraps `Gr00tN1d7` + its processor and exposes
  the ACT interface: `forward(batch) -> (loss, out)`, `predict_action_chunk(batch) -> (B, chunk, 44)`,
  `build`. Pre/post are **identity** — the GR00T processor normalizes state/action + resizes images
  internally, so the batch stays raw radians (`l1_all`/FK reuse unchanged, in radians).
- **Adapter boundary** (in-memory, per batch; `learning/data/conversion/groot_modality.py`):
  state 132-D → `q/dq/tau`; action 44-D → `r_arm_cmd[7] l_arm_cmd[7] r_hand_cmd[15] l_hand_cmd[15]`
  (**absolute**); video `(C,H,W) [0,1]` → HWC uint8; language = dataset `task`; embodiment
  `NEW_EMBODIMENT`. Stats come from the wir_v1 dataset.
- Frozen backbone + diffusion action head; **bf16** (fp32 weights + `autocast`). `GrootConfig` is in
  `configuration.py`.

## Base model + node env

- Base **`nvidia/GR00T-N1.7-3B`** and its VLM backbone **`nvidia/Cosmos-Reason2-2B`** are synced from
  `s3://wirobotics-internal/chrisryu/models/groot/` into an offline HF cache — no gated-HuggingFace
  access or HF token needed (`HF_HUB_OFFLINE=1`, resolved from the local snapshot).
- Dedicated venv `/opt/dlami/nvme/groot-venv`, built once from `learning/scripts/requirements-groot.txt`
  (torch 2.7.1 line — separate from the ACT venv's torch 2.10). `train_groot.sh` builds it on first run.

## Train

```bash
POLICY=groot bash learning/scripts/aws/train_aws.sh   # node recipe: learning/scripts/train_groot.sh
```

Multi-GPU = 8× A100 DDP (`accelerate launch`, eff_batch = batch × #GPUs). Needs
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set by the recipe) or it OOMs on a 40 GB A100.

## Equivalence

Verified bit-exact vs the pre-rebase tip (2026-06-26): identical batch/stats, forward loss
`1.4903576374053955` (full float64), and `predict_action_chunk`.
