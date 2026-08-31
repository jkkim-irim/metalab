"""Engine-agnostic environment contract (hub).

:func:`sim.metalab.contract.loader.load_task` imports a declarative task contract
(``tasks/<task>.py``, which defines ``TASK: TaskSpec``) and resolves it into a
:class:`~sim.metalab.contract.spec.EnvSpec` via components (robot/object) and terms
(obs/reward/terminate factories, imported as symbols). This package imports no
engine (gs/newton). Conventions and units live in :mod:`sim.metalab.conventions`.
"""
from sim.metalab.contract.spec import EnvSpec

__all__ = ["EnvSpec"]
