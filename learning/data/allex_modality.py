"""Canonical ALLEX state/action joint map — the single source of truth for how the flat
132-D ``observation.state`` and 44-D ``action`` vectors split into named joint groups.

Every consumer imports its spans from here instead of hardcoding indices, so the layout is
defined once and cannot drift: the GR00T ``modality.json`` generator
(``learning/data/conversion/groot_modality.py``), the per-body-part validation metrics
(``learning/metrics/validation.py``), and the ACT state-slice / action-reparameterization paths
all reference these constants. The FK joint order in ``learning/metrics/allex_fk.py`` is name-based
(no indices) but follows the same per-limb order as ``ACTION_GROUPS``.

Layout (mirrors ``learning/data/WIR_FORMAT.md``):

    observation.state (132-D) = q(0:44) | dq(44:88) | tau(88:132)   -- three 44-D proprio blocks
    action            (44-D)  = r_arm(0:7) | l_arm(7:14) | r_hand(14:29) | l_hand(29:44)

The 44 joints share ONE per-joint ordering across q/dq/tau and action, so ``action[i]`` and the
``q`` block's ``observation.state[i]`` are the same physical joint. The cross-limb spans are
therefore contiguous: arms = r_arm+l_arm = (0:14), hands = r_hand+l_hand = (14:44).
"""

Span = tuple[int, int]  # (start, end), half-open — usable directly as a Python slice

# --- observation.state (132-D): the three 44-D proprio blocks ---------------------------------
STATE_DIM = 132
STATE_GROUPS: dict[str, Span] = {"q": (0, 44), "dq": (44, 88), "tau": (88, 132)}
STATE_Q: Span = STATE_GROUPS["q"]  # pos-only slice (the ACT "S1" recipe consumes just this)

# --- action (44-D): per-limb command blocks ---------------------------------------------------
ACTION_DIM = 44
ACTION_GROUPS: dict[str, Span] = {
    "r_arm": (0, 7), "l_arm": (7, 14), "r_hand": (14, 29), "l_hand": (29, 44),
}
# Derived cross-limb spans (contiguous by construction — asserted below).
ARMS: Span = (ACTION_GROUPS["r_arm"][0], ACTION_GROUPS["l_arm"][1])    # (0, 14)
HANDS: Span = (ACTION_GROUPS["r_hand"][0], ACTION_GROUPS["l_hand"][1])  # (14, 44)
ALL: Span = (0, ACTION_DIM)

# Per-body-part metric layout: whole-chunk + per-limb + arm/finger split (the default the
# validation metrics report; ``learning/metrics/validation.py`` aliases this).
ACTION_METRIC_GROUPS: dict[str, Span] = {
    "all": ALL,
    **ACTION_GROUPS,
    "arm": ARMS, "finger": HANDS,
}


def span(name: str) -> Span:
    """(start, end) for a named action or state group. Fail loud on unknown names."""
    if name in ACTION_GROUPS:
        return ACTION_GROUPS[name]
    if name in STATE_GROUPS:
        return STATE_GROUPS[name]
    raise KeyError(
        f"unknown ALLEX group {name!r}; have {sorted({*STATE_GROUPS, *ACTION_GROUPS})}"
    )


def slice_(sp: Span) -> slice:
    """A Python ``slice`` for a ``(start, end)`` span."""
    return slice(sp[0], sp[1])


def validate_span(sp: Span, dim: int, what: str = "span") -> None:
    """Fail loud if ``sp`` does not fit within a tensor whose last dim is ``dim``."""
    s, e = sp
    if not (0 <= s < e <= dim):
        raise ValueError(f"{what} {sp} out of range for last-dim size {dim}")


def state_prefix_dim(keys: list[str]) -> int:
    """Resolve ``state_keys`` (names from ``STATE_GROUPS``: q/dq/tau) to the number of leading
    ``observation.state`` dims ``N`` they select — i.e. the state is consumed as ``[..., :N]``.

    All realistic selections are a contiguous ``[0:N]`` prefix (``q`` -> 44, ``q+dq`` -> 88, full ->
    132); order in ``keys`` doesn't matter. **Fails loud** on an unknown name, a selection that
    doesn't start at ``q`` (dim 0), or a non-contiguous one (e.g. ``["q", "tau"]`` skips ``dq``) —
    non-contiguous gather is not implemented (would need a real index-gather, not a leading slice).
    """
    if not keys:
        raise ValueError("state_keys must be a non-empty list of STATE_GROUPS names")
    spans = []
    for k in keys:
        if k not in STATE_GROUPS:
            raise KeyError(f"unknown state group {k!r}; have {sorted(STATE_GROUPS)}")
        spans.append(STATE_GROUPS[k])
    spans.sort()
    if spans[0][0] != 0:
        raise ValueError(
            f"state_keys {keys} must include the leading 'q' block (start 0); got spans {spans}")
    end = 0
    for s, e in spans:
        if s != end:
            raise ValueError(
                f"state_keys {keys} select a non-contiguous span {spans}; only a contiguous "
                f"[0:N] prefix is supported (non-contiguous gather not implemented)")
        end = e
    return end


# Consistency guards — fail at import if the spans are ever edited out of sync.
assert ACTION_GROUPS["r_arm"][0] == 0 and ACTION_GROUPS["l_hand"][1] == ACTION_DIM
assert ARMS == (0, 14) and HANDS == (14, ACTION_DIM)
assert STATE_GROUPS["q"] == (0, ACTION_DIM) and STATE_GROUPS["tau"][1] == STATE_DIM
