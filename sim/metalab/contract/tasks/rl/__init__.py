"""RL task contracts — the families Train/Eval run, one folder per family.

A family is ``rl/<task>/`` — or one group-shelf deeper, ``rl/<group>/<task>/`` (the group is a shelf,
not part of the task's identity): a shared ``_base.py`` core plus one ``<task>_<recipe>.py`` per recipe,
so a run names two axes (``--task`` + ``--recipe``). Sibling of ``standalone/``, which holds the scene-only
contracts that carry no learning; keeping them apart is what lets the launchpad list them separately.
"""
