"""hammer_lift task-specific reward terms (see common.py for the general/reusable ones).

Kept separate so ``common.py`` stays task-agnostic; the term contract itself (flat ``fn(env, **knobs) ->
(N,)``, never self-scaled — magnitude is the contract's ``weight``) is documented there.
"""
from __future__ import annotations

import torch

from sim.metalab.api.contact import contact_mask


def fingertip_object_contact(env, target: str = "object",
                             force_threshold: float = 0.1,
                             lift_threshold: float = 0.0) -> torch.Tensor:
    """Fraction of fingertips GRIPPING ``target``, optionally only once it is OFF THE TABLE → [0, 1], step 1/K.

        returns  #{k : ||force_k|| > force_threshold} / K  *  (object_z >= lift_threshold)

        target          "object"   contact counterpart the force is scoped to
        force_threshold  0.1 [N]   force on a tip that counts as gripping
        lift_threshold   0.0 [m]   ABSOLUTE world z the object must clear before this pays (0 = always).
                              Set it just above the RESTING height: with the hammer on the table a hand can
                              collect the full grip reward by draping fingers over it, and that pose carries
                              nothing — gating on the lift makes the term pay for a grasp that HOLDS.

    Contact on a PAD is what counts, and the hand says which surface that is by geometry — each pad is its
    own collision shell on its own body, so the tips this reads are pad shells and a nail press lands on a
    body nobody asks about.

    WHICH bodies are the fingertips comes from the ROBOT (``RobotSpec.fingertips``), not a knob. The GATE's
    ``contact_count`` counts the SAME call, so this dense reward and the success bar cannot drift apart.
    """
    f = env.contact_force_with(env.fingertips, target)                        # (N,K,3) world, target-scoped
    r = contact_mask(f, force_threshold).float().mean(dim=-1)
    if lift_threshold > 0.0:
        r = r * (env.object_pos()[:, 2] >= lift_threshold).float()
    return r


def nail_object_contact(env, bodies: list[str], target: str = "object",
                        force_threshold: float = 0.1) -> torch.Tensor:
    """Fraction of NAILS touching ``target`` → [0, 1], step 1/K. A PENALTY: compose with a NEGATIVE weight.

        returns  #{k in bodies : ||force_k|| > force_threshold} / len(bodies)

        bodies   (body names)   the nail shells (``"@bodies.nail"`` = what the robot pins in nail_friction)
        target       "object"   contact counterpart the force is scoped to
        force_threshold 0.1 [N] force on a nail that counts as touching

    The mirror image of :func:`fingertip_object_contact`: proto_v4 splits every fingertip into a PAD shell
    and a NAIL shell on separate bodies, so "gripping" and "raking it with the back of the finger" are told
    apart by which body the force lands on — the pad term pays for one and this charges for the other.

    Why charge at all when the nails are already pinned slick: friction only stops the nail from CARRYING the
    object, not from pushing it around. A policy can still shove the hammer into place with the backs of its
    fingers, which is a motion the real hand should not be learning.

    Counterpart-scoped to the object on purpose — a nail resting on the TABLE is contact too, and on a
    table-top task the approach puts it there constantly.
    """
    f = env.contact_force_with(bodies, target)                        # (N,K,3) world, target-scoped
    return contact_mask(f, force_threshold).float().mean(dim=-1)


def fingertip_object_pinch_contact(env, fingers: list[str], target: str = "object",
                                   force_threshold: float = 0.1,
                                   lift_max: float = 0.0) -> torch.Tensor:
    """Fraction of the NAMED fingers pressing ``target``, paid only while it is still LOW → [0, 1].

        returns  #{k in fingers : ||force_k|| > force_threshold} / len(fingers)  *  (grasp_point_z < lift_max)

        fingers   (body names)  the tips that form the pinch (``RobotSpec.fingertips`` names)
        target        "object"  contact counterpart the force is scoped to
        force_threshold 0.1 [N] force on a tip that counts as pressing
        lift_max      0.0 [m]   ABSOLUTE world z the grasp point must stay BELOW to be paid (0 = no cap)

    :func:`fingertip_object_contact` narrowed to a named pair and capped in height — a pinch bootstrap that
    stops paying at ``lift_max``, where the lift phase already hands over to the carry terms.
    """
    f = env.contact_force_with(fingers, target)                       # (N,K,3) world, target-scoped
    r = contact_mask(f, force_threshold).float().mean(dim=-1)
    if lift_max > 0.0:
        r = r * (env.object_pos()[:, 2] < lift_max).float()
    return r
