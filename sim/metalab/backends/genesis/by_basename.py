"""ByBasename — proxy to look up a Genesis entity's joints/links by contract basename.

A Genesis-loaded entity (e.g. MJCF robot) may have full prim-path joint/link names, so this wraps lookup
by the basename the contract uses (e.g. ``R_Elbow_Joint``). parser returns the robot entity via this
proxy and the backend reads by name. (USD-agnostic — also used for robot MJCF loading.)
"""
from __future__ import annotations


class ByBasename:
    """Entity joint/link names may be full prim paths — proxy lookup by contract basename."""

    def __init__(self, entity):
        object.__setattr__(self, "_entity", entity)

    def __getattr__(self, name):
        return getattr(self._entity, name)

    def _find(self, seq, name):
        matches = [x for x in seq if x.name.rsplit("/", 1)[-1] == name.rsplit("/", 1)[-1]]
        assert len(matches) == 1, f"'{name}' matched {len(matches)} — check name contract"
        return matches[0]

    def get_joint(self, name):
        return self._find(self._entity.joints, name)

    def get_link(self, name):
        return self._find(self._entity.links, name)
