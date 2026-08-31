"""Reward terms — one flat function per reward (engine-agnostic).

    def <reward_name>(env, <knob>=<default>, ...) -> (N,)
    driver pays:  weight * fn(...)

1. **A term never scales itself** — magnitude lives ONLY in the contract's ``weight``, which is what ONE step
   of the term returning its maximum pays (see :class:`Rew`). A term paid every step therefore totals
   ``weight * max_episode_length`` over a full episode; a ONE-SHOT term totals ``weight``.
2. **Vocabulary matches GATE** — ``lift_height`` / ``force_threshold`` mean the same here and in ``GATE``.
   What counts as SOLVED is not a reward knob at all: the curriculum states that bar, the driver judges it,
   and the success term is paid for the latch (``object_goal_reach_bonus``).
3. **Check the budget shape before touching a weight** — PERSISTENT terms are capped by TIME (value x steps),
   TELESCOPING/one-shot ones by GEOMETRY; swapping the form REQUIRES re-picking the weight.

``env`` = :class:`~sim.metalab.runtime.env_driver.EnvDriver`: backend reads, the fixed goal and the robot's
fingertips/pad axes/palm, and per-env episode state via ``buffer()`` (reset-cleared — mutate IN PLACE). Terms
run in contract order; a term publishing on ``env`` precedes its readers. Shared math: ``sim/metalab/api``.
"""
from __future__ import annotations

import torch

from sim.metalab.api import frames, keypoints, shaping


# --- lift -----------------------------------------------------------------------
def lifting_reward(env, lift_height: float = 0.1) -> torch.Tensor:
    """Rise off the object's OWN spawn height, as a fraction of ``lift_height`` → [0, 1]. PERSISTENT.

        returns  clamp((object_z - object_init_z) / lift_height, 0, 1)

        lift_height  0.1 [m]  the rise that pays 1.0

    Spawn-relative, so table height and spawn DR do not change the scale.
    """
    assert lift_height > 0.0, f"lifting_reward: lift_height must be > 0 (it normalizes the ratio) — got {lift_height}"
    return ((env.object_pos()[:, 2] - env.object_init_z) / lift_height).clamp(0.0, 1.0)


def object_lift_ramp(env, z_start: float, z_end: float) -> torch.Tensor:
    """Linear rise over an ABSOLUTE-z band, latched off once the top is reached → [0, 1].

        returns   clamp((object_z - z_start) / (z_end - z_start), 0, 1) * (not latched)
        latched <- latched | (object_z >= z_end)          per episode

        z_start  [m]  world z the ramp starts paying at — the table line, so nothing pays for a resting object
        z_end    [m]  world z that pays 1.0 and LATCHES the ramp off, handing the carry terms the signal

    The lift phase's only gradient. PERSISTENT inside the band, so hovering just under ``z_end`` collects
    ``weight`` per remaining step — keep the weight below what the crossing bonus pays,
    or that hover is the better policy. Latching (rather than gating on z) means a later drop cannot restart it.

    episode state: ``buffer("latched")`` (N,) bool.
    """
    assert z_end > z_start, f"object_lift_ramp: z_end must be > z_start — got {z_end} <= {z_start}"
    latched = env.buffer("latched", dtype=torch.bool)
    z = env.object_pos()[:, 2]
    r = ((z - z_start) / (z_end - z_start)).clamp(0.0, 1.0)
    latched |= z >= z_end
    return r * (~latched).float()


def object_lift_bonus(env, lift_threshold: float) -> torch.Tensor:
    """Pays its whole budget on the step the object FIRST reaches ``lift_threshold``, 0 every other step.

        returns  1.0 the first step object_z >= lift_threshold, else 0.0

        lift_threshold  [m]  ABSOLUTE world z — pair it with ``object_lift_ramp``'s ``z_end``

    ONE-SHOT, so its whole payout is ``weight`` while a term paid every step totals ``weight *
    max_episode_length``: to compare the two, scale this weight by the episode length yourself.

    The rung between the lift phase and the carry phase: the ramp latches off here and the carry terms take
    over, so without a payout AT the crossing the two phases meet at a step that pays nothing. One-shot per
    episode, so bouncing across the bar cannot farm it.

    episode state: ``buffer("paid")`` (N,) bool.
    """
    paid = env.buffer("paid", dtype=torch.bool)
    up = env.object_pos()[:, 2] >= lift_threshold
    newly = up & ~paid
    paid |= up
    return newly.float()


# --- reach ----------------------------------------------------------------------
def palm_object_proximity(env, palm_body: str, std: float = 0.05, palm_offset=(0.0, 0.0, 0.0)) -> torch.Tensor:
    """Dense palm→grasp-point attraction → (0, 1]. Stateless, PERSISTENT, bounded.

        palm    = palm_pos   + R(palm_quat)   @ palm_offset
        returns   exp(-||palm - grasp|| / std)

        palm_body                   the hand body (MJCF R_Palm_Link)
        std             0.05  [m]   distance e-fold — larger = wider attraction
        palm_offset  (0,0,0)  [m]   palm-local; 0 when the palm is a real body (the usual case)
    """
    palm = frames.local_point(env.body_pos(palm_body), env.body_quat(palm_body), palm_offset)
    grasp = env.object_pos()
    return shaping.exp_kernel(torch.linalg.norm(palm - grasp, dim=-1), std)


def fingertip_object_proximity(env, std: float = 0.05) -> torch.Tensor:
    """Dense fingertip→grasp-point attraction, per tip → (0, 1]. Stateless, PERSISTENT.

        returns   mean_k exp(-||tip_k - grasp|| / std)

        std          0.05  [m]   distance e-fold — larger = wider attraction

    MEAN of per-tip exponentials, not ``exp`` of the mean distance: one tip arriving pays 1/K instead of the
    whole hand having to close together. K and WHICH bodies come from ``env.fingertips``. Not lift-gated —
    the tips staying ON the object is what a carry needs.
    """
    tips = torch.stack([env.body_pos(b) for b in env.fingertips], dim=1)      # (N, K, 3) world
    grasp = env.object_pos()
    d = (tips - grasp.unsqueeze(1)).norm(dim=-1)                              # (N, K)
    return shaping.exp_kernel(d, std).mean(dim=-1)


# --- carry ----------------------------------------------------------------------
def object_goal_keypoint_progress(env, lift_threshold: float = 0.0) -> torch.Tensor:
    """Pays the FRACTION of the opening goal distance newly closed → (N,).

        kp      = object_goal_dist(...)         [m] position+rotation fused (cage corners)
        d0     <- kp                            on the first step of the episode
        returns   max(0, best - kp) / d0 * (object_z >= lift_threshold)
        best   <- min(best, kp)                 per episode, whether or not the gate pays

        lift_threshold  0.0 [m]  ABSOLUTE world z the object must be at to be paid. 0 = always

    TELESCOPING: only new ground pays, so approach-retreat-approach cannot farm it, and the episode total is
    GEOMETRY-capped — closing the whole opening distance pays exactly ``weight``, NOT ``weight`` per step like
    a persistent term, so scale this weight by the episode length to compare the two.

    Normalised by the episode's OWN opening distance, so a spawn that starts far and one that starts near are
    worth the same for solving the task; the reward per metre is what differs. That is the trade against
    paying a fixed amount per metre, which hands the luckier spawn a bigger return for the same result.

    ``lift_threshold`` reads the CURRENT height, not a latch, so an object put back down stops being paid to
    carry. ``best`` ratchets through that gate either way, which is what makes ground closed by SLIDING the
    object along the table unpayable — then and later.

    episode state: ``buffer("best")`` (N,) — also read by the ``closest_keypoint_max_dist`` obs term —
    and ``buffer("d0")`` (N,) [m].
    """
    best = env.buffer("best", fill=float("inf"))
    d0 = env.buffer("d0")
    cur = keypoints.object_goal_dist(env.object_pos(), env.object_quat(),
                                     env.goal_pos, env.goal_quat, env.goal_half_extent)
    d0[:] = torch.where(torch.isinf(best), cur, d0)       # capture BEFORE min_dist_progress_ clears the inf
    r = shaping.min_dist_progress_(best, cur) / d0.clamp(min=1.0e-6)
    if lift_threshold > 0.0:
        r = r * (env.object_pos()[:, 2] >= lift_threshold).float()
    return r


def object_goal_keypoint_tracking(env, std: float = 0.1, lift_threshold: float = 0.0,
                                  lift_full: float = 0.0) -> torch.Tensor:
    """PERSISTENT goal attraction → [0, 1]: pays every step the object is NEAR the goal, including the hold
    the GATE requires, which ``object_goal_keypoint_progress`` pays 0 for. Stateless.

        height  = clamp((object_z - lift_threshold) / (lift_full - lift_threshold), 0, 1)   if lift_full > 0
                  (object_z > lift_threshold)                                               otherwise
        returns   exp(-object_goal_dist(...) / std) * height

        std          0.1 [m]  distance e-fold — larger = wider goal attraction
        lift_threshold   0.0 [m]  ABSOLUTE world z where this starts paying (0 = no height condition). Set it
                              just above the RESTING height — spawn-relative is not "off the table".
        lift_full        0.0 [m]  ABSOLUTE world z where the height factor reaches 1.0 (0 = STEP instead of a
                              ramp: full pay the instant the threshold is crossed)

    Prefer the RAMP. A step multiplier hands the policy the whole term in one frame the moment it crosses the
    bar — a cliff the value function has to fit and an incentive to jitter across it — while a linear band
    turns "lift it higher" into gradient the whole way up. The band ends where the carry phase begins, so
    above ``lift_full`` height stops paying and only the distance to the goal still moves the term.

    TIME-capped, so keep the weight SMALL (bounded 1.0/step). With no height condition it also pays for an
    object merely SITTING near the goal, so the goal must not be within ``std`` of the spawn pose.
    """
    assert lift_full <= 0.0 or lift_full > lift_threshold, (
        f"lift_full={lift_full} must be above lift_threshold={lift_threshold} — it is the top of the ramp "
        f"the height factor climbs over (0 = no ramp)")
    r = shaping.exp_kernel(keypoints.object_goal_dist(env.object_pos(), env.object_quat(),
                                                     env.goal_pos, env.goal_quat, env.goal_half_extent), std)
    if lift_threshold > 0.0:
        z = env.object_pos()[:, 2]
        r = r * (((z - lift_threshold) / (lift_full - lift_threshold)).clamp(0.0, 1.0) if lift_full > 0.0
                 else (z > lift_threshold).float())
    return r


def joint_pose_convergence(env, joint_pose: dict, std: float, lift_threshold: float = 0.0) -> torch.Tensor:
    """Dense attraction to a target posture → [0, 1], paid only while the object is UP. Stateless.

        err     = mean_j |q_j - joint_pose[j]|                  [rad] over the posture's joints
        returns   exp(-err / std) * (grasp_point_z >= lift_threshold)

        joint_pose {joint: rad}  the posture — pass the contract's ``GATE.joint_final_pose``
        std              [rad]   error e-fold
        lift_threshold  0.0 [m]  ABSOLUTE world z the grasp point must clear before this pays. 0 = always

    The MEAN, so every joint carries gradient on every step: ``exp`` of the mean error, not the mean of
    per-joint ``exp``, so a joint that is still far is pulled as hard as one that has nearly arrived
    (a mean of exponentials would hand a far joint ``exp(-far/std)`` ≈ 0 of the pull it needs).

    Bounding the WORST joint is the GATE's job (``joint_pose_tolerance``, which the curriculum ramps down to
    it); paying the max here as well left 11 of 12 joints with no signal at all. The trade is that one joint
    sitting wide of the target is diluted by 1/J, so this term alone no longer forbids it — the gate does,
    and through the gate the reach bonus.

    LIFT-GATED because the target is a CLOSED hand: ungated, curling the fingers in mid-air pays the same as
    closing on the handle.
    """
    q = env.joint_pos(list(joint_pose))
    err = (q - frames.const(tuple(joint_pose.values()), q)).abs().mean(dim=-1)
    r = shaping.exp_kernel(err, std)
    if lift_threshold > 0.0:
        grasp_z = env.object_pos()[:, 2]
        r = r * (grasp_z >= lift_threshold).float()
    return r


# --- success --------------------------------------------------------------------
def object_goal_reach_bonus(env) -> torch.Tensor:
    """The success term: pays its whole budget on the step the CURRICULUM's bar is first met, 0 otherwise.

        returns  1.0 on the step ``env.curriculum_passed`` first turns True, else 0.0
                 (ONE-SHOT: the whole payout is ``weight``, while a term paid every step totals
                  ``weight * max_episode_length``)

    NO knobs: what "solved" means at this level is the curriculum term's to state and the driver's to judge
    (``EnvDriver._advance_curriculum``), and this pays for exactly that — the same latch that ends the episode
    (``terminate.curriculum_passed``), so the bonus and the reset land on the same step by construction rather
    than by two bar-sets agreeing. ``val/SR`` is unaffected: the GATE is scored separately, at any level.

    episode state: ``buffer("paid")`` (N,) bool — the latch is per-EPISODE, so without it a solved env would
    be paid again on every remaining step of that episode.
    """
    paid = env.buffer("paid", dtype=torch.bool)
    newly = env.curriculum_passed & ~paid                       # one-shot
    paid |= env.curriculum_passed
    return newly.float()


# --- penalties (positive values — compose with a NEGATIVE weight) ---------------
def joint_torque_penalty(env, names: list[str]) -> torch.Tensor:
    """Joint torque L2 — discourages straining.

        returns  sum_j tau_j^2     [(N*m)^2], unbounded     (7 arm joints at 10 N*m each -> 700)

        names   the joints summed over
    """
    tau = env.joint_torque(names)
    return torch.sum(tau * tau, dim=-1)


def joint_vel_l1(env, names: list[str]) -> torch.Tensor:
    """Joint-velocity L1 — the motion/jerk penalty.

        returns  sum_j |qdot_j|    [rad/s], unbounded       (7 arm joints at 1 rad/s each -> 7)

        names   the joints summed over — the contract splits arm and hand so each gets its own weight
    """
    return env.joint_vel(names).abs().sum(dim=-1)


def action_rate_l2(env) -> torch.Tensor:
    """Action L2 rate — penalizes command jitter, and with it the policy's own sampling noise.

        returns  sum_i (a_i - a_i_prev)^2    unbounded; 0 on an episode's FIRST step

    Reads the RAW policy output (``env.last_action``), not the post-EMA joint target: the action EMA
    already absorbs the physical effect of the jitter, so this prices the noise without distorting what
    the robot actually does. For a Gaussian policy ``E[(a - a_prev)^2] = (mu - mu_prev)^2 + 2*sigma^2``,
    so it is a DIRECT quadratic cost on the action std, not an indirect one: the mean pays only for
    CHANGING, the noise pays always. Against an entropy bonus (gradient +coef per log-sigma) it gives
    log-sigma an equilibrium at ``sigma = sqrt(coef / (4*weight))`` instead of letting it grow without
    bound, which is what a sparse reward otherwise does — nothing else in the objective has a definite
    sign on sigma. PPO's advantage normalisation rescales the effective weight, so treat that equilibrium
    as the mechanism and the order of magnitude, not as a prediction.

    Keep it OUT of ``dense_reward_terms``: faded to zero it stops bounding sigma exactly when the sparse
    bar is hardest.

    episode state: ``buffer("prev")`` (N, num_actions), ``buffer("seen")`` (N,) bool — an episode's first
    step has no predecessor and buffers start at 0, so paying ``(a - 0)^2`` there would bill a fresh
    episode for its whole first command.
    """
    prev = env.buffer("prev", shape=(env.num_actions,))
    seen = env.buffer("seen", dtype=torch.bool)
    a = env.last_action
    r = ((a - prev) ** 2).sum(dim=-1) * seen
    prev[:] = a
    seen[:] = True
    return r
