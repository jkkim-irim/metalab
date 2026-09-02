from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sim.metalab.runtime import service  # noqa: E402


def build_env(task: str, recipe: str | None = None, num_envs: int | None = None,
              device: str = "cuda:0", viz: str = "none",
              telemetry: bool | None = None, rrd_path: str | None = None):
    from sim.metalab.backends.newton import parser
    from sim.metalab.backends.newton.backend import NewtonBackend
    from sim.metalab.contract.loader import load_task
    from sim.metalab.runtime.env_driver import EnvDriver
    spec = load_task(task.replace("-", "_"), recipe, num_envs=num_envs)
    num_envs = spec.num_envs
    handles = parser.build_scene(
        spec, num_envs=num_envs, viz=(viz not in (None, "none")),
        viewer_kind=(viz if viz in ("gl", "rtx", "rerun") else "gl"), device=device,
        rrd_path=rrd_path)
    backend = NewtonBackend(spec, handles, num_envs=num_envs)
    max_ep = max(1, round(spec.episode_length_s / (spec.physics.dt * spec.physics.decimation)))
    tele = (viz not in (None, "none")) if telemetry is None else telemetry
    return EnvDriver(spec, backend, max_episode_length=max_ep, telemetry=tele)


if __name__ == "__main__":
    service.main(build_env, "newton-sim")
