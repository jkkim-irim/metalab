"""ALLEX learning — validation routines for ACT, used by the BC trainer (learning.trainer).

compute_val_loss            : validation loss (L1 + KL) over a held-out loader.
compute_per_horizon_metrics : per-horizon prediction error (loss_h0 + rollout_h{h}).
compute_group_metrics       : per-body-part + whole-chunk action L1, padding-masked over every
                              chunk step (arm/finger per side, plus l1_all over all dims).
compute_cartesian_gripper_metrics : physical-unit EE error (mm/deg) + gripper L1/sign-accuracy for
                              EE-delta embodiments (LibERO OSC_POSE) — no URDF/FK, err x controller
                              output_max scale; the embodiment-agnostic counterpart to compute_fk_metrics.
compute_fk_metrics          : Cartesian fingertip/wrist error (mm) via forward kinematics.
compute_prediction_metrics  : all three of the above predict-based metrics in ONE eval pass (one
                              predict_action_chunk/batch) — the PRODUCTION path the trainer calls.
                              Emits per-group/per-horizon L1 in BOTH ``_norm`` (preprocessor space)
                              and ``_unnorm`` (raw radians) — only ``_unnorm``/``fk_*`` compare
                              across policies (ACT normalizes, GR00T's pre/post-processor is identity).
log_fk_video                : pred-vs-GT 3D skeleton videos -> wandb.

NOTE: the trainer uses compute_prediction_metrics (the one-pass fold); compute_per_horizon_metrics /
compute_group_metrics / compute_fk_metrics are retained as the per-metric REFERENCE implementations
that compute_prediction_metrics is pinned equal to by test_validation — do not delete them as
"unused" or the equivalence test loses its oracle.

The trainer validates over a fixed, representative val SUBSET built at the loader level (a seeded-
random sample of the val set — see data.make_dataloader), not the whole set; the ``max_batches``
arg below is a plain trailing-batch cap (used by the tests + generic callers), independent of that.

compute_val_loss / compute_per_horizon_metrics are robot-agnostic. The task-space metrics are
robot-specific but parameterized, not hardcoded — so a future embodiment (e.g. LIBERO) plugs in
without touching this module:
  * compute_group_metrics takes a ``groups`` action-slice layout (ALLEX_ACTION_GROUPS is the
    default for the 44-D ALLEX action);
  * compute_fk_metrics / log_fk_video take an ``fk`` object and depend only on its interface
    (``tips`` / ``wrists`` dicts + ``tip_wrist_positions`` / ``render_motion``) — AllexFK today,
    a LiberoFK (or any URDF-backed FK with the same interface) tomorrow.
"""
import logging
from pathlib import Path

import torch

from learning.data.allex_modality import ACTION_METRIC_GROUPS
from learning.model.act.constants import ACTION

# Default action-group layout for the 44-D ALLEX action — the single source of truth is
# learning/data/allex_modality.py (so the layout can't drift across the GR00T modality.json
# generator, the ACT state-slice/reparam paths, and these metrics). "all" = the chunk-wide pure-L1
# over all 44 dims (the headline val L1, distinct from val_loss which adds KL, and from loss_h0 which
# is step 0 only). A different embodiment passes its own {name: (start, end)} layout to
# compute_group_metrics. compute_prediction_metrics emits each group in two spaces: l1_<g>_norm
# (preprocessor space) and l1_<g>_unnorm (raw radians).
ALLEX_ACTION_GROUPS = ACTION_METRIC_GROUPS


@torch.no_grad()
def compute_val_loss(policy, loader, preprocessor, accelerator, max_batches=0):
    """Validation loss (plain L1+KL via ACTPolicy.forward).

    ``max_batches`` > 0 caps the pass to that many (leading) batches; 0 runs the whole loader. The
    trainer passes a pre-sampled representative val subset and leaves this 0 (see module docstring).

    LeRobot's ACTPolicy.forward always computes a KL term when `use_vae=True`,
    but the inner ACT model only runs the VAE encoder in training mode. In eval
    mode mu_hat/log_sigma_x2_hat come back as None, so the KL line raises
    TypeError. Workaround: switch the model to training mode so the VAE encoder
    runs on val actions (we DO have ground-truth actions during val). To prevent
    BatchNorm running stats from being updated with val batches, we snapshot &
    restore them around the forward. Dropout + the VAE reparameterization also draw from the
    RNG, so the forward runs under ``torch.random.fork_rng`` — those draws never advance the
    training RNG, so step-cadence validation (``--val_every_n_steps``) stays monitoring-only.
    """
    # Snapshot BN stats
    bn_stats = {}
    for name, m in policy.named_modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                          torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)):
            bn_stats[name] = (
                m.running_mean.clone() if m.running_mean is not None else None,
                m.running_var.clone() if m.running_var is not None else None,
                m.num_batches_tracked.clone() if m.num_batches_tracked is not None else None,
            )

    was_training = policy.training
    policy.train()  # activates VAE encoder path inside ACT.forward

    total_loss = 0.0
    n_batches = 0
    # Fork the RNG so val-time dropout + VAE-reparameterization draws do NOT advance the training
    # RNG. Without this, step-cadence validation (--val_every_n_steps) would shift later training
    # dropout masks and change the training trajectory — validation must be monitoring-only.
    fork_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=fork_devices):
            for batch in loader:
                batch = preprocessor(batch)
                with accelerator.autocast():
                    loss, _ = policy.forward(batch)
                total_loss += loss.item()
                n_batches += 1
                if max_batches and n_batches >= max_batches:
                    break
    finally:
        # Restore eval mode and BN stats (no leak from val)
        if not was_training:
            policy.eval()
        for name, m in policy.named_modules():
            if name in bn_stats:
                mean, var, num = bn_stats[name]
                if mean is not None:
                    m.running_mean.copy_(mean)
                if var is not None:
                    m.running_var.copy_(var)
                if num is not None:
                    m.num_batches_tracked.copy_(num)

    assert n_batches > 0, "compute_val_loss: the val loader yielded no batches"
    return total_loss / n_batches


@torch.no_grad()
def compute_per_horizon_metrics(policy, loader, preprocessor, accelerator, horizons, max_batches=0):
    """Per-horizon prediction error via policy.predict_action_chunk (eval-mode).

    Returns {'loss_h0': float, 'rollout_h{h}': float, ...}. Padded steps masked out via
    batch['action_is_pad']. ``max_batches`` > 0 caps the pass (slow-predict policies like GR00T);
    0 = full val set.
    """
    policy.eval()

    K = policy.config.chunk_size
    horizons = sorted({h for h in horizons if 0 <= h < K})
    if not horizons:
        return {}

    sum_l1 = torch.zeros(K)
    cnt = torch.zeros(K, dtype=torch.long)
    for n, batch in enumerate(loader):
        if max_batches and n >= max_batches:
            break
        batch = preprocessor(batch)
        with accelerator.autocast():
            pred = policy.predict_action_chunk(batch)   # (B, K, A)
        gt = batch[ACTION]                                # (B, K, A)
        if pred.shape != gt.shape:
            kk = min(pred.shape[1], gt.shape[1])
            pred, gt = pred[:, :kk], gt[:, :kk]
        per = (pred - gt).abs().mean(dim=-1).float().cpu()
        if "action_is_pad" in batch:
            mask = (~batch["action_is_pad"][:, : per.shape[1]]).float().cpu()
        else:
            mask = torch.ones_like(per)
        sum_l1[: per.shape[1]] += (per * mask).sum(0)
        cnt[: per.shape[1]] += mask.sum(0).long()

    per_h = (sum_l1 / cnt.clamp(min=1)).numpy()
    out = {}
    for h in horizons:
        key = "loss_h0" if h == 0 else f"rollout_h{h}"
        out[key] = float(per_h[h])
    return out


@torch.no_grad()
def compute_group_metrics(policy, loader, preprocessor, accelerator, groups=ALLEX_ACTION_GROUPS,
                          max_batches=0):
    """Per-body-part action-prediction L1 over the WHOLE predicted chunk (all steps, padding-masked).

    Reports ``all`` (the headline val L1) + EE (arm) vs finger (hand) + per-side, in normalized
    action units. ``groups`` is a ``{name: (start, end)}`` action-slice layout (ALLEX default; a
    different embodiment passes its own). Cheap (pure slicing) — no FK, no extra deps. Averaged over
    every valid (sample, chunk-step) pair, not just the executed step.
    Returns ``{'l1_all': float, 'l1_arm': float, 'l1_finger': float, ...}``.
    """
    policy.eval()
    sums = {g: 0.0 for g in groups}
    cnt = 0
    for n, batch in enumerate(loader):
        if max_batches and n >= max_batches:
            break
        batch = preprocessor(batch)
        with accelerator.autocast():
            pred = policy.predict_action_chunk(batch)
        gt = batch[ACTION]
        kk = min(pred.shape[1], gt.shape[1])
        pred = pred[:, :kk].float().cpu()        # (B, K, A)
        gt = gt[:, :kk].float().cpu()
        mask = ((~batch["action_is_pad"][:, :kk]).float().cpu() if "action_is_pad" in batch
                else torch.ones(pred.shape[:2]))  # (B, K) over chunk steps
        valid = float(mask.sum())
        if valid == 0:
            continue
        diff = (pred - gt).abs()                 # (B, K, A)
        for g, (a, b) in groups.items():
            per_step = diff[:, :, a:b].mean(dim=2)         # (B, K) mean over the group's dims
            sums[g] += float((per_step * mask).sum())      # masked sum over all chunk steps
        cnt += int(valid)
    if cnt == 0:
        return {}
    return {f"l1_{g}": s / cnt for g, s in sums.items()}


@torch.no_grad()
def compute_cartesian_gripper_metrics(policy, loader, preprocessor, accelerator, *,
                                      cartesian_metrics, gripper_dim, max_batches=0):
    """Physical-unit end-effector + gripper action-prediction error for EE-delta embodiments (LibERO).

    The embodiment-agnostic counterpart to ``compute_fk_metrics``: no URDF, no forward kinematics —
    it turns a NORMALIZED action error directly into physical units via the robot controller's
    per-axis output_max, which is the right metric when the action already IS an end-effector delta
    (e.g. LibERO / robosuite ``OSC_POSE``: a 7-D ``[dpos(3), drot(3), gripper(1)]`` command in
    ``[-1, 1]``) rather than joint angles the ALLEX URDF-FK expects. (ALLEX's FK nan's out here — its
    URDF has no matching joints.) Same eval pass structure as ``compute_group_metrics``: one
    ``predict_action_chunk`` vs ``batch[ACTION]`` over the WHOLE chunk, padding-masked, averaged over
    every valid (sample, chunk-step) pair.

    ``cartesian_metrics`` is a list of ``{"key", "slice": (a, b), "scale", "unit"}`` entries; each
    reports ``{key: mean_L2(pred[..., a:b] - gt[..., a:b]) * scale}``. For LibERO the scales come from
    the ``OSC_POSE`` controller ``output_max = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]``:
      * position   ``mm_pos``  = err x 0.05 m x 1000 = err x 50.0   (mm),
      * orientation ``deg_rot`` = err x 0.5 rad x 180/pi = err x 28.6479 (deg).
    ``gripper_dim`` (or ``None`` to skip): the action index of the continuous gripper command; adds
      * ``gripper_l1``  = mean |pred - gt| on that dim,
      * ``gripper_acc`` = mean(sign(pred) == sign(gt)) — the fraction of chunk steps whose open/close
        command matches (``sign(0) == 0`` is treated as its own sign).

    Pure torch — no extra deps. Returns ``{}`` if the loader yields no valid (unmasked) chunk steps.
    """
    policy.eval()
    sums = {e["key"]: 0.0 for e in cartesian_metrics}
    if gripper_dim is not None:
        sums["gripper_l1"] = 0.0
        sums["gripper_acc"] = 0.0
    cnt = 0
    for n, batch in enumerate(loader):
        if max_batches and n >= max_batches:
            break
        batch = preprocessor(batch)
        with accelerator.autocast():
            pred = policy.predict_action_chunk(batch)
        gt = batch[ACTION]
        kk = min(pred.shape[1], gt.shape[1])
        pred = pred[:, :kk].float().cpu()        # (B, K, A)
        gt = gt[:, :kk].float().cpu()
        mask = ((~batch["action_is_pad"][:, :kk]).float().cpu() if "action_is_pad" in batch
                else torch.ones(pred.shape[:2]))  # (B, K) over chunk steps
        valid = float(mask.sum())
        if valid == 0:
            continue
        for e in cartesian_metrics:
            a, b = e["slice"]
            l2 = (pred[:, :, a:b] - gt[:, :, a:b]).norm(dim=2)   # (B, K) L2 over the slice
            sums[e["key"]] += float((l2 * float(e["scale"]) * mask).sum())
        if gripper_dim is not None:
            gp, gg = pred[:, :, gripper_dim], gt[:, :, gripper_dim]   # (B, K)
            sums["gripper_l1"] += float(((gp - gg).abs() * mask).sum())
            acc = (torch.sign(gp) == torch.sign(gg)).float()          # open/close command match
            sums["gripper_acc"] += float((acc * mask).sum())
        cnt += int(valid)
    if cnt == 0:
        return {}
    return {k: s / cnt for k, s in sums.items()}


@torch.no_grad()
def compute_fk_metrics(policy, loader, preprocessor, postprocessor, accelerator, fk, max_batches=0):
    """Cartesian wrist/fingertip POSE error over the WHOLE predicted chunk (all steps, masked),
    predicted vs GT action via forward kinematics:

      * ``fk_mm_*``        fingertip + wrist POSITION error (mm),
      * ``fk_deg_*_wrist`` wrist ORIENTATION error (geodesic angle, degrees) — the rotation half of
        the EE pose that the position metric ignores.

    The GT raw action (radians) is captured BEFORE normalization; the policy predicts a *normalized*
    action, which we map back to radians with the ``postprocessor`` (Unnormalize) — the exact inverse
    of the ``preprocessor``'s action normalization. FK runs on the flattened ``(B*chunk, 44)`` poses;
    errors are averaged over every valid (sample, chunk-step) pair, not just the executed step.
    """
    policy.eval()
    tip_keys = list(fk.tips.keys())  # R_finger_1..L_finger_5
    sums = {f"tip:{k}": 0.0 for k in tip_keys}
    sums.update({f"wrist:{s}": 0.0 for s in ("R", "L")})
    sums.update({f"deg:{s}": 0.0 for s in ("R", "L")})
    cnt = 0
    for n, batch in enumerate(loader):
        if max_batches and n >= max_batches:
            break
        raw_gt = batch[ACTION].clone()                 # (B,K,44) raw radians, pre-normalization
        pad = batch.get("action_is_pad")
        batch = preprocessor(batch)
        with accelerator.autocast():
            pred = policy.predict_action_chunk(batch)  # (B,K,44) normalized
        kk = min(pred.shape[1], raw_gt.shape[1])
        a_dim = pred.shape[2]
        raw_pr = postprocessor(pred[:, :kk].reshape(-1, a_dim).float())  # (B*kk,44) radians
        rg = raw_gt[:, :kk].reshape(-1, a_dim).float()                   # (B*kk,44) radians
        mask = ((~pad[:, :kk]).reshape(-1).float().cpu() if pad is not None
                else torch.ones(raw_pr.shape[0]))                        # (B*kk,) over chunk steps
        valid = float(mask.sum())
        if valid == 0:
            continue
        tp_p, wr_p, ro_p = fk.tip_wrist_poses(raw_pr.cpu())
        tp_g, wr_g, ro_g = fk.tip_wrist_poses(rg.cpu())
        for k in tip_keys:
            sums[f"tip:{k}"] += float(((tp_p[k] - tp_g[k]).norm(dim=-1) * 1000.0 * mask).sum())
        for s in ("R", "L"):
            sums[f"wrist:{s}"] += float(((wr_p[s] - wr_g[s]).norm(dim=-1) * 1000.0 * mask).sum())
            # geodesic angle (deg) between pred and GT wrist rotation: arccos((tr(Rp Rgᵀ)-1)/2)
            rel = ro_p[s] @ ro_g[s].transpose(-1, -2)                       # (N,3,3)
            cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
            ang_deg = torch.arccos(cos) * (180.0 / torch.pi)               # (N,)
            sums[f"deg:{s}"] += float((ang_deg * mask).sum())
        cnt += int(valid)
    if cnt == 0:
        return {}
    out = {f"fk_mm_{k}": sums[f"tip:{k}"] / cnt for k in tip_keys}
    for s in ("R", "L"):
        out[f"fk_mm_{s}_wrist"] = sums[f"wrist:{s}"] / cnt
        out[f"fk_deg_{s}_wrist"] = sums[f"deg:{s}"] / cnt
    rtips = [out[f"fk_mm_{k}"] for k in tip_keys if k.startswith("R")]
    ltips = [out[f"fk_mm_{k}"] for k in tip_keys if k.startswith("L")]
    out["fk_mm_R_finger_mean"] = sum(rtips) / len(rtips)
    out["fk_mm_L_finger_mean"] = sum(ltips) / len(ltips)
    out["fk_mm_finger_mean"] = (sum(rtips) + sum(ltips)) / (len(rtips) + len(ltips))
    out["fk_deg_wrist_mean"] = (out["fk_deg_R_wrist"] + out["fk_deg_L_wrist"]) / 2.0
    return out


@torch.no_grad()
def compute_prediction_metrics(policy, loader, preprocessor, postprocessor, accelerator, *,
                               horizons=(), groups=None, fk=None, max_batches=0, sample_seed=None):
    """All predict-based val metrics in ONE eval-mode pass (1 ``predict_action_chunk`` per batch):
    per-horizon (``loss_h0``/``rollout_h{h}``), per-group L1 (``l1_*``), and — if ``fk`` is given —
    task-space FK error (``fk_mm_*``/``fk_deg_*``).

    The per-horizon and per-group L1 are each emitted in TWO explicit action spaces, suffixed
    ``_norm`` and ``_unnorm`` (e.g. ``l1_all_norm`` / ``l1_all_unnorm``, ``rollout_h8_norm`` / ...):
    ``_norm`` is the trainer-preprocessor space (ACT: mean/std-normalized — matches the training L1);
    ``_unnorm`` is raw action units (radians, the FK input). Only ``_unnorm`` (and ``fk_*``) are
    comparable ACROSS policies — ACT's preprocessor normalizes while GR00T's is identity, so a bare
    ``l1_all`` would silently mix normalized (ACT) and radian (GR00T) scales. ``fk_*`` keys are
    unsuffixed (always raw-space mm/deg).

    Folds the former per-metric passes into one; the ``_norm`` values equal what the separate
    reference functions produced. ``horizons=()`` / ``groups`` falsy / ``fk=None`` each skip that
    block. ``max_batches`` caps the number of batches.

    ``sample_seed`` (not None): fork + seed the RNG so a generative policy's ``predict_action_chunk``
    (e.g. GR00T's flow sampler starts from ``randn``) is DETERMINISTIC/repeatable across evals — the
    metric becomes precise instead of jittering with the sampling seed. Forked so training RNG is
    untouched (mirrors ``compute_val_loss``). No-op for a deterministic policy (ACT).
    """
    policy.eval()
    K = policy.config.chunk_size
    horizons = sorted({h for h in horizons if 0 <= h < K})
    # per-horizon + per-group L1, accumulated in BOTH action spaces (reported side-by-side):
    #   norm   = trainer-preprocessor space (ACT: mean/std-normalized; matches the training L1)
    #   unnorm = raw action units / radians (== the FK input) — the ONLY space comparable ACROSS
    #            policies (GR00T's pre/post-processor is identity, so its norm == unnorm == radians).
    sum_l1 = {"norm": torch.zeros(K), "unnorm": torch.zeros(K)}        # per-horizon, per space
    cnt_h = torch.zeros(K, dtype=torch.long)                           # step count (space-agnostic)
    gsums = {sp: {g: 0.0 for g in (groups or {})} for sp in ("norm", "unnorm")}   # per-group L1
    if fk is not None:
        tip_keys = list(fk.tips.keys())
        fsums = {f"tip:{k}": 0.0 for k in tip_keys}
        fsums.update({f"wrist:{s}": 0.0 for s in ("R", "L")})
        fsums.update({f"deg:{s}": 0.0 for s in ("R", "L")})
    pcnt = 0   # valid (sample, chunk-step) pairs — shared by the group + FK averages

    # Optional deterministic sampler seed: repeatable predict across evals (precise sampled metric).
    # Save the global RNG and restore it after the loop so training is unaffected (like fork_rng).
    _saved_rng = None
    if sample_seed is not None:
        _saved_rng = (torch.get_rng_state(),
                      torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        torch.manual_seed(sample_seed)

    for n, batch in enumerate(loader):
        if max_batches and n >= max_batches:
            break
        raw_gt = batch[ACTION].clone()                 # (B,K,44) raw radians, pre-normalization
        pad = batch.get("action_is_pad")
        batch = preprocessor(batch)
        with accelerator.autocast():
            pred = policy.predict_action_chunk(batch)  # (B,K,A) in the policy's preprocessor space
        gt = batch[ACTION]
        kk = min(pred.shape[1], gt.shape[1])
        a_dim = pred.shape[2]
        # norm = trainer-preprocessor space; unnorm = raw radians (postprocessor-decoded pred vs the
        # pre-normalization GT). unnorm is the FK input and the cross-policy-comparable space.
        prn = pred[:, :kk].float().cpu()                                     # (B,kk,A) norm
        gtn = gt[:, :kk].float().cpu()
        # postprocess on pred's OWN device (Unnormalize's stats live there), then move to cpu
        pru = postprocessor(pred[:, :kk].reshape(-1, a_dim).float()).cpu().reshape(prn.shape)  # unnorm (rad)
        gtu = raw_gt[:, :kk].float().cpu()
        mask = ((~pad[:, :kk]).float().cpu() if pad is not None
                else torch.ones(prn.shape[:2]))        # (B,kk) over chunk steps
        valid = float(mask.sum())

        for sp, pr_, gt_ in (("norm", prn, gtn), ("unnorm", pru, gtu)):
            if horizons:                               # per-horizon (whole chunk, masked)
                sum_l1[sp][:kk] += ((pr_ - gt_).abs().mean(dim=-1) * mask).sum(0)
            if valid > 0 and gsums[sp]:                # per-group L1
                diff = (pr_ - gt_).abs()               # (B,kk,A)
                for g, (a, b) in groups.items():
                    gsums[sp][g] += float((diff[:, :, a:b].mean(dim=2) * mask).sum())
        if horizons:
            cnt_h[:kk] += mask.sum(0).long()
        if valid == 0:
            continue
        if fk is not None:                             # task-space FK (mm + deg) — raw radians
            raw_pr = pru.reshape(-1, a_dim)            # (B*kk,44) rad — reuse the unnorm decode
            rg = gtu.reshape(-1, a_dim)
            fmask = mask.reshape(-1)
            tp_p, wr_p, ro_p = fk.tip_wrist_poses(raw_pr)
            tp_g, wr_g, ro_g = fk.tip_wrist_poses(rg)
            for k in tip_keys:
                fsums[f"tip:{k}"] += float(((tp_p[k] - tp_g[k]).norm(dim=-1) * 1000.0 * fmask).sum())
            for s in ("R", "L"):
                fsums[f"wrist:{s}"] += float(((wr_p[s] - wr_g[s]).norm(dim=-1) * 1000.0 * fmask).sum())
                rel = ro_p[s] @ ro_g[s].transpose(-1, -2)
                cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
                fsums[f"deg:{s}"] += float((torch.arccos(cos) * (180.0 / torch.pi) * fmask).sum())
        pcnt += int(valid)

    if _saved_rng is not None:                                # restore global RNG (training untouched)
        torch.set_rng_state(_saved_rng[0])
        if _saved_rng[1] is not None:
            torch.cuda.set_rng_state_all(_saved_rng[1])

    out = {}
    cnt = cnt_h.clamp(min=1)
    for sp in ("norm", "unnorm"):                       # both action spaces, explicit suffixes
        if horizons:
            per_h = (sum_l1[sp] / cnt).numpy()
            for h in horizons:
                out[("loss_h0" if h == 0 else f"rollout_h{h}") + f"_{sp}"] = float(per_h[h])
        if gsums[sp] and pcnt > 0:
            out.update({f"l1_{g}_{sp}": s / pcnt for g, s in gsums[sp].items()})
    if fk is not None and pcnt > 0:
        out.update({f"fk_mm_{k}": fsums[f"tip:{k}"] / pcnt for k in tip_keys})
        for s in ("R", "L"):
            out[f"fk_mm_{s}_wrist"] = fsums[f"wrist:{s}"] / pcnt
            out[f"fk_deg_{s}_wrist"] = fsums[f"deg:{s}"] / pcnt
        rtips = [out[f"fk_mm_{k}"] for k in tip_keys if k.startswith("R")]
        ltips = [out[f"fk_mm_{k}"] for k in tip_keys if k.startswith("L")]
        out["fk_mm_R_finger_mean"] = sum(rtips) / len(rtips)
        out["fk_mm_L_finger_mean"] = sum(ltips) / len(ltips)
        out["fk_mm_finger_mean"] = (sum(rtips) + sum(ltips)) / (len(rtips) + len(ltips))
        out["fk_deg_wrist_mean"] = (out["fk_deg_R_wrist"] + out["fk_deg_L_wrist"]) / 2.0
    return out


@torch.no_grad()
def log_fk_video(wandb_logger, policy, loader, preprocessor, postprocessor, accelerator, fk,
                 step, video_dir, n_videos=3, fps=8):
    """Render up to ``n_videos`` pred-vs-GT 3D skeleton MP4s (spread across the val set) and log
    them to wandb under the Media section as ``media/fk_motion_{i}``, at the given global ``step``.
    Per clip: item 0's predicted action chunk vs the GT chunk over the horizon. Prediction ->
    radians via the ``postprocessor`` (same as the FK metric). ``video_dir`` holds the rendered files.
    """
    policy.eval()
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    try:
        stride = max(1, len(loader) // n_videos)
    except TypeError:
        stride = 1
    logged = 0
    for bi, batch in enumerate(loader):
        if logged >= n_videos:
            break
        if bi % stride != 0:
            continue
        raw_gt = batch[ACTION].clone()
        batch = preprocessor(batch)
        with accelerator.autocast():
            pred = policy.predict_action_chunk(batch)
        kk = min(pred.shape[1], raw_gt.shape[1])
        gt0 = raw_gt[0, :kk].float().cpu()                   # (K,44) radians
        pr0 = postprocessor(pred[0, :kk].float()).cpu()      # (K,44) radians
        out_path = str(video_dir / f"fk_motion_s{step:07d}_{logged}.mp4")
        fk.render_motion(gt0, pr0, out_path, fps=fps, title=f"step {step} sample {logged}")
        # Log AT the validation step (not stepless) so the videos align with their metrics and
        # don't nudge wandb's auto-step counter past the next explicit log.
        wandb_logger._wandb.log(
            {f"media/fk_motion_{logged}": wandb_logger._wandb.Video(out_path, format="mp4")},
            step=step)
        logged += 1
    logging.info(f"logged {logged} FK motion videos -> wandb")
