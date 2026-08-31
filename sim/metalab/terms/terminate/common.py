"""Termination terms — one flat function per termination (engine-agnostic).

    def <termination_name>(env, <knob>=<default>, ...) -> (N,) bool

The driver calls ``t.fn(env, **t.params)`` once per policy step and ORs every term into ``done`` (together
with ``time_out``), logging each one separately as ``Termination/<name>`` — the fraction of envs whose LAST
episode ended by that cause, so the seven values are a population distribution. ``truncation=True`` on the
contract entry makes that done BOOTSTRAP (success / horizon) instead of being a true absorbing terminal.

``env`` (EnvDriver) gives every backend read plus two driver-latched values: ``lifted`` (at
``GATE.lift_height``) and ``curriculum_passed`` (at the curriculum's own bar), so a termination never
re-derives a predicate another owner already computed.

Knobs are named by the signature and written by the contract, and ``params`` stays MUTABLE at runtime — the
same shape as a reward term, see :class:`sim.metalab.contract.spec.TerminateTerm`.
Reference: ``sim/isaaclab/envs/hammer_lift/mdp/terminations.py``.
"""
from __future__ import annotations

import torch


def object_below_height(env, min_height: float) -> torch.Tensor:
    """Terminate when object z < min_height (drop). (N,) bool.

        min_height  [m]  ABSOLUTE world z — below the table top, not below the spawn.
    """
    return env.object_pos()[:, 2] < min_height


def object_far_from_body(env, body: str, max_distance: float) -> torch.Tensor:
    """Terminate when the object is further than max_distance from ``body`` (e.g. the palm). (N,) bool.

        body              the body the distance is measured to (MJCF name)
        max_distance [m]  3-D distance above which the episode is hopeless

    Distance to the HAND, in 3-D, not to a fixed point in the world: what makes an episode hopeless is the
    object being out of reach, and where "reach" is moves with the arm.

    Replaces an ``object_too_far(anchor_xy, max_distance)`` that measured the horizontal distance to a fixed
    world anchor and so missed both cases that matter here — an object flung straight up (height was not in
    the distance at all) and an object left behind while the arm travelled away (the anchor distance barely
    moves).
    """
    d = torch.linalg.norm(env.object_pos() - env.body_pos(body), dim=-1)
    return d > max_distance


def object_velocity_exceeded(env, max_lin_vel: float = 15.0, max_ang_vel: float = 30.0) -> torch.Tensor:
    """Terminate when object lin/ang velocity exceeds threshold. (N,) bool. Backend-read only.

        max_lin_vel  15.0 [m/s]    linear speed ceiling
        max_ang_vel  30.0 [rad/s]  angular speed ceiling      (defaults = legacy hammer_velocity_exceeded)

    When penetration causes runaway contact impulses, the solver diverges to Inf/NaN. The NaN
    sentinel (C1) resets NaN only and misses Inf / huge-finite velocities. Early terminate on
    velocity threshold absorbs it via clean re-spawn before divergence. Inf is caught by
    ``inf > max`` = True (NaN compares False -> sentinel handles it). Cheapest read-only guard.
    Reference: hammer_lift:hammer_velocity_exceeded, perceptive:object_velocity_exceeded.
    """
    lin = torch.linalg.norm(env.object_lin_vel(), dim=-1)
    ang = torch.linalg.norm(env.object_ang_vel(), dim=-1)
    return (lin > max_lin_vel) | (ang > max_ang_vel)


def table_fingertip_contact_force_exceeded(env, fingertips: list[str],
                                           force_threshold_n: float = 100.0) -> torch.Tensor:
    """Terminate when fingertip-**table** contact force exceeds threshold (N) (prevent table slamming).
    Counterpart-scoped (A5 contact_force_with(target="table")) so it doesn't mix with object contact.
    True if any fingertip exceeds. (N,) bool.

        fingertips               the tip bodies checked (``"@bodies.fingertips"``)
        force_threshold_n  100.0 [N]  per tip: above this ends the episode

    Reference: hammer_lift/mdp/terminations.py:table_fingertip_contact_force_exceeded."""
    f = env.contact_force_with(fingertips, "table")          # (N, K, 3)
    return f.norm(dim=-1).amax(dim=-1) > force_threshold_n


def body_contact_detected(env, bodies: list[str], force_threshold: float = 1.0) -> torch.Tensor:
    """Terminate when ANY of ``bodies`` touches ANYTHING → (N,) bool.

        bodies                       links with no business touching the world in this task
        force_threshold  1.0 [N]     net force that counts as a touch — above solver noise, below a graze

    NET force rather than counterpart-scoped (``contact_force``, not ``contact_force_with``): the whole point
    is that these links should not meet the table, the object, the ground OR each other, and scoping to one
    counterpart would wave the rest through. WHICH links are out of bounds is a TASK decision — a forearm may
    legitimately lean on a table in some other task — so it is a knob, not a robot fact.

    Only links that still carry a collision geom can ever trip this: the robot contract drops most of them
    (``collision: 0``), and a dropped link reads zero force forever, i.e. silently never terminates."""
    return (env.contact_force(bodies).norm(dim=-1) > force_threshold).any(dim=-1)


def curriculum_passed(env) -> torch.Tensor:
    """Terminate when the CURRICULUM's bar has been met — the episode latch ``EnvDriver.curriculum_passed``,
    set by ``_advance_curriculum`` once the level's conditions have held for its ``hold_steps``. (N,) bool.
    Fixed-goal task only. No knobs: every threshold that decides it belongs to the curriculum term.

    The CURRICULUM's bar, not the GATE's: the difficulty the policy is training at is the one that ends its
    episode, so a level that is already solved stops costing steps. ``object_goal_reach_bonus`` is paid for
    the same latch, so success and reset land on the same step. ``val/SR`` is unaffected — the driver latches
    the contract's ``GATE`` separately (``_advance_gate``), at any level."""
    return env.curriculum_passed


def grasp_lost_after_lift(env, fingertips: list[str], min_height: float = 0.9,
                          contact_threshold: float = 1.0) -> torch.Tensor:
    """Terminate when a once-lifted (latched) object drops below min_height with no fingertip contact (clear drop).

        fingertips                    the tip bodies whose object contact is checked
        min_height          0.9 [m]   ABSOLUTE world z the object must stay above once lifted
        contact_threshold   1.0 [N]   per tip: at or below this counts as off the object

    Three conditions: ``env.lifted`` (latched lifting-reward gate exposed by EnvDriver) & object_z
    < min_height & **all fingertip object-contact forces <= threshold** (grasp released). Unlike
    object_below_height (below table), this catches a post-latch descent near rest, distinguishing it from
    in-hand reorientation (height held) and momentary contact loss (height held). Contact is A5
    counterpart-scoped (``contact_force_with(target="object")``) so table contact doesn't mix in.
    Reference: perceptive/mdp/terminations.py:grasp_lost_after_lift."""
    z = env.object_pos()[:, 2]
    f = env.contact_force_with(fingertips, "object")         # (N, K, 3) object-only contact force
    no_contact = f.norm(dim=-1).amax(dim=-1) <= contact_threshold   # all fingertips off the object
    return env.lifted & (z < min_height) & no_contact
