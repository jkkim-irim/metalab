"""Task contracts — ``<task>.py`` (declarative: each module builds one ``TASK = TaskSpec(...)``).

:func:`sim.metalab.contract.loader.load_task` imports the task module and resolves components
(robot/object) and terms (obs/reward/terminate factories, referenced as imported symbols)
into a :class:`~sim.metalab.contract.spec.EnvSpec`. A contract is declarative — names, tunables, and
factory refs, no logic; behavior lives in ``sim/metalab/contract/{obs,reward,terminate}/common.py``.
See ``README.md`` in this directory for the authoring rules a new task must follow.
"""
