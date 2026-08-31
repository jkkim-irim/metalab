"""Third-party plugin discovery (de-lerobot'd).

ALLEX calls ``register_third_party_plugins()`` once before ``main()``. In LeRobot this scans
installed distributions for ``lerobot_robot_`` / ``lerobot_camera_`` / ``lerobot_teleoperator_`` /
``lerobot_policy_`` packages and imports them so they can self-register. ALLEX installs no such
plugins (the policy — ACT — is wired in directly), so for us this is a no-op. Kept as a function so
``train.py`` can call it unchanged.
"""

import logging


def register_third_party_plugins() -> None:
    """No-op: ALLEX has no LeRobot third-party plugin packages to discover/import."""
    logging.debug("register_third_party_plugins: no third-party plugins to import (ALLEX no-op).")
