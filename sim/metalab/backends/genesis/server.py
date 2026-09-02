from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sim.metalab.backends.genesis.viewer import GenesisRerunScene  # noqa: E402
from sim.metalab.runtime import rerun_recording, service  # noqa: E402


def build_env(task: str, recipe: str | None = None, num_envs: int | None = None,
              device: str = "cuda:0", viz: str = "none",
              telemetry: bool | None = None, rrd_path: str | None = None):
    import genesis as gs

    from sim.metalab.backends.genesis import parser
    from sim.metalab.backends.genesis.backend import GenesisBackend
    from sim.metalab.contract.loader import load_task
    from sim.metalab.runtime.env_driver import EnvDriver
    spec = load_task(task.replace("-", "_"), recipe, num_envs=num_envs)
    num_envs = spec.num_envs
    backend_dev = gs.cpu if str(device).startswith("cpu") else gs.gpu
    handles = parser.build_scene(
        spec, num_envs=num_envs, viz=(viz not in (None, "none")), backend=backend_dev)
    backend = GenesisBackend(spec, handles, num_envs=num_envs)
    max_ep = max(1, round(spec.episode_length_s / (spec.physics.dt * spec.physics.decimation)))
    tele = (viz not in (None, "none")) if telemetry is None else telemetry
    env = EnvDriver(spec, backend, max_episode_length=max_ep, telemetry=tele)
    if rrd_path:
        fps = 1.0 / (float(spec.physics.dt) * int(spec.physics.decimation))
        print(f"[genesis] rerun recording → {rerun_recording.open_recording(rrd_path, fps=fps)}", flush=True)
        ents = {"robot": backend.robot}
        ents.update({f"object{i}": o for i, o in enumerate(backend.objects)})
        ents.update(dict(backend.fixtures or {}))
        env.rerun_scene = GenesisRerunScene(backend.scene, ents, num_envs)
        print(f"[genesis] rerun scene: {env.rerun_scene.summary()}", flush=True)
        print(f"[genesis] rerun world labels: "
              f"{rerun_recording.log_world_labels(backend.scene.envs_offset[:num_envs])}", flush=True)
    return env


if __name__ == "__main__":
    service.main(build_env, "genesis-sim")
