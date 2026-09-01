"""hammer_lift_student — task-family folder.

``_base.py`` holds the shared core and each ``hammer_lift_student_<recipe>.py`` beside it is one thin
recipe. The family is NOT runnable by itself — a run names the pair, ``--task hammer-lift-student
--recipe only-ycb``; this package exposes no ``TASK``, so a bare family name fails in the loader with
the recipe list instead of silently resolving to whatever the default happened to be.
"""
