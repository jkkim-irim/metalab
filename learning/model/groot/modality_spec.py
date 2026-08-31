"""Per-embodiment GR00T modality spec (torch / gr00t / numpy-free — safe at parse time).

Declares, per embodiment name, how a flat wir_v1 batch maps onto GR00T's named state/action groups:

  * ``state_groups``   : ``{group: (start, end)}`` slices of ``observation.state``,
  * ``action_groups``  : ``{group: (start, end)}`` slices of the flat ``action`` vector,
  * ``action_keys``    : the STABLE order the action groups are concatenated back into the flat
                         action vector (drives the processor ``ModalityConfig.modality_keys`` and the
                         ``predict_action_chunk`` reassembly),
  * ``state_keys``     : the order the state groups are concatenated for the model's state encoder,
  * ``max_state_dim`` / ``max_action_dim`` : the width the GR00T processor zero-pads the concatenated
                         state / action up to before the model (see the padding note below).

This is the single source of truth for the embodiment layout: ``learning.model.groot.configuration``
reads it to fill ``state_keys`` / ``max_state_dim`` / ``max_action_dim`` on ``GrootConfig``, and
``learning.model.groot_policy`` reads it to slice/reassemble batches. ALLEX is the default and pulls
its slices straight from ``learning.data.conversion.groot_modality`` (the dataset-conversion source of
truth), so the two never drift.

CRITICAL — ``max_state_dim`` / ``max_action_dim`` are model-fixed, not embodiment-fixed.
The pretrained ``Gr00tN1d7`` action head is built at the CHECKPOINT's ``config.max_state_dim`` /
``config.max_action_dim`` (132 for groot-base-n1.7-3b): the state encoder's input width is
``config.max_state_dim * state_history_length`` and the action encoder/decoder width is
``config.max_action_dim``. The processor zero-pads the concatenated per-group state/action UP to
these dims, and the per-sample ``action_mask`` restricts the flow-matching loss to the real dims —
this is exactly how GR00T supports a low-D embodiment on a wide pretrained head. So a 7-D embodiment
(mikasa) MUST keep ``max_state_dim == max_action_dim == 132`` (pad 7 -> 132); setting them to 7 would
make the padded tensors narrower than the pretrained layers expect and raise a shape mismatch. Only
change these if you point ``base_model_path`` at a checkpoint whose ``config.max_*_dim`` differs.
"""
from dataclasses import dataclass

from learning.data.conversion.groot_modality import ACTION_GROUPS as _ALLEX_ACTION_GROUPS
from learning.data.conversion.groot_modality import STATE_GROUPS as _ALLEX_STATE_GROUPS

# Width the pretrained groot-base-n1.7-3b action head / state encoder are built at (== the vendored
# Gr00tN1d7Config default max_state_dim / max_action_dim). Every embodiment pads UP to this — see the
# module docstring. Named so it is obvious this is the base-checkpoint width, not an embodiment dim.
BASE_MODEL_MAX_DIM = 132


@dataclass(frozen=True)
class ModalitySpec:
    """Declarative state/action layout for one GR00T embodiment (see module docstring).

    The trailing OPTIONAL fields declare how VALIDATION scores this embodiment's action (consumed by
    ``learning.trainer.bc_trainer`` via ``learning.metrics.validation``). They default to ``None`` so
    ALLEX/mikasa are unaffected — ALLEX keeps its URDF-FK path, which these do not touch:

      * ``val_metric_groups`` : ``{name: (start, end)}`` action-slice groups for the group-L1 metric
        (``l1_<name>_norm``/``_unnorm``). Richer than the bare ``{"all": (0, flat_dim)}`` the trainer
        falls back to — e.g. LibERO splits its 7-D EE-delta action into pos/rot/gripper L1.
      * ``cartesian_metrics`` : a tuple of declarative physical-unit EE metrics, each
        ``{"key", "slice": (a, b), "scale", "unit"}``. The metric is the mean (over the masked chunk)
        of the L2 norm of ``(pred - gt)`` on that slice, times ``scale`` — turning a NORMALIZED action
        error into physical units (mm / deg) via the embodiment's controller output_max.
      * ``gripper_dim`` : action index of the (continuous) gripper command — enables ``gripper_l1``
        (mean |pred - gt|) and ``gripper_acc`` (open/close sign-match rate).
    """

    name: str
    state_groups: dict[str, tuple[int, int]]
    action_groups: dict[str, tuple[int, int]]
    action_keys: tuple[str, ...]   # concat order of action groups -> flat action vector
    state_keys: tuple[str, ...]    # concat order of state groups -> state-encoder input
    max_state_dim: int = BASE_MODEL_MAX_DIM
    max_action_dim: int = BASE_MODEL_MAX_DIM
    # ---- OPTIONAL validation-metric layout (default None -> ALLEX/mikasa unchanged; see docstring).
    val_metric_groups: dict[str, tuple[int, int]] | None = None
    cartesian_metrics: tuple[dict, ...] | None = None
    gripper_dim: int | None = None

    def __post_init__(self) -> None:
        # The concat orders must name exactly the declared groups (fail loud on a typo / omission).
        if set(self.action_keys) != set(self.action_groups):
            raise ValueError(
                f"[{self.name}] action_keys {self.action_keys} must be a permutation of the "
                f"action_groups {sorted(self.action_groups)}."
            )
        if set(self.state_keys) != set(self.state_groups):
            raise ValueError(
                f"[{self.name}] state_keys {self.state_keys} must be a permutation of the "
                f"state_groups {sorted(self.state_groups)}."
            )
        # The processor pads the concatenated state/action UP to max_*_dim, so the real width must
        # fit. State groups slice observation.state (may overlap); action groups are concatenated.
        state_width = max((e for _, e in self.state_groups.values()), default=0)
        if state_width > self.max_state_dim:
            raise ValueError(
                f"[{self.name}] state slices reach {state_width} > max_state_dim {self.max_state_dim}."
            )
        action_width = sum(e - s for s, e in self.action_groups.values())
        if action_width > self.max_action_dim:
            raise ValueError(
                f"[{self.name}] action groups total {action_width} > max_action_dim "
                f"{self.max_action_dim}."
            )
        for grp, (s, e) in {**self.state_groups, **self.action_groups}.items():
            if not (0 <= s < e):
                raise ValueError(f"[{self.name}] group {grp!r} has an invalid slice ({s}, {e}).")


# ---- ALLEX (default) — the wir_v1 132-D q/dq/tau state + 44-D r/l arm/hand action, identical to
# ---- the dataset-conversion contract (learning/data/conversion/groot_modality.py).
_ALLEX = ModalitySpec(
    name="allex",
    state_groups=dict(_ALLEX_STATE_GROUPS),    # {"q":(0,44),"dq":(44,88),"tau":(88,132)}
    action_groups=dict(_ALLEX_ACTION_GROUPS),  # {"r_arm_cmd":(0,7),"l_arm_cmd":(7,14),...(29,44)}
    action_keys=("r_arm_cmd", "l_arm_cmd", "r_hand_cmd", "l_hand_cmd"),
    state_keys=("q", "dq", "tau"),
    max_state_dim=BASE_MODEL_MAX_DIM,
    max_action_dim=BASE_MODEL_MAX_DIM,
)

# ---- mikasa — a 7-D single-group embodiment (7-D state, 7-D eef action, top+wrist cameras). One
# ---- state group and one action group; the processor zero-pads both 7 -> 132 (see docstring).
_MIKASA = ModalitySpec(
    name="mikasa",
    state_groups={"state": (0, 7)},
    action_groups={"eef_cmd": (0, 7)},
    action_keys=("eef_cmd",),
    state_keys=("state",),
    max_state_dim=BASE_MODEL_MAX_DIM,
    max_action_dim=BASE_MODEL_MAX_DIM,
)

# ---- libero — LibERO (panda, 8-D state, 7-D eef action, top+wrist cameras). One state group and
# ---- one action group; the processor zero-pads state 8 -> 132 and action 7 -> 132 (see docstring).
# ---- Note the dims differ from mikasa (8 vs 7 state), but max_state_dim / max_action_dim STAY 132:
# ---- they are the pretrained head widths, not the embodiment dims — setting them to 8/7 would make
# ---- the padded tensors narrower than the base checkpoint's layers expect and raise a shape error.
_LIBERO = ModalitySpec(
    name="libero",
    state_groups={"state": (0, 8)},
    action_groups={"eef_cmd": (0, 7)},
    action_keys=("eef_cmd",),
    state_keys=("state",),
    max_state_dim=BASE_MODEL_MAX_DIM,
    max_action_dim=BASE_MODEL_MAX_DIM,
    # ---- Validation metrics for LibERO's 7-D robosuite OSC_POSE EE-delta action
    # ---- [dpos(3), drot(3), gripper(1)], all normalized to [-1, 1]. The ALLEX URDF-FK metric does
    # ---- NOT apply here, so we report action-space group L1 plus physical-unit EE error instead.
    # ---- Physical scales come from the OSC_POSE controller output_max = [0.05 m, 0.05 m, 0.05 m,
    # ---- 0.5 rad, 0.5 rad, 0.5 rad]: position mm = err x 0.05 x 1000 = err x 50.0;
    # ---- orientation deg = err x 0.5 x 180/pi = err x 28.6479. Gripper is dim 6 (-1 open / +1 close).
    val_metric_groups={"all": (0, 7), "pos": (0, 3), "rot": (3, 6), "gripper": (6, 7)},
    cartesian_metrics=(
        {"key": "mm_pos", "slice": (0, 3), "scale": 50.0, "unit": "mm"},
        {"key": "deg_rot", "slice": (3, 6), "scale": 28.6479, "unit": "deg"},
    ),
    gripper_dim=6,
)

MODALITY_SPECS: dict[str, ModalitySpec] = {s.name: s for s in (_ALLEX, _MIKASA, _LIBERO)}


def get_modality_spec(name: str) -> ModalitySpec:
    """Return the ``ModalitySpec`` for ``name`` (e.g. "allex", "mikasa"); fail loud if unknown."""
    if name not in MODALITY_SPECS:
        raise ValueError(
            f"Unknown GR00T modality {name!r}. Known embodiments: {sorted(MODALITY_SPECS)}."
        )
    return MODALITY_SPECS[name]
