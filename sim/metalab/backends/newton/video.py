from __future__ import annotations

from pathlib import Path

import imageio
from newton.viewer import ViewerGL

VIDEO_RES = (1280, 720)


class VideoRecorder:
    def __init__(self, backend, path: Path, fps: float):
        cam = backend.spec.camera
        assert cam is not None, f"{backend.spec.name}: video recording needs scene.camera (eye/lookat/fov) in the contract"
        self._backend = backend
        self._path = Path(path)
        self._fps = float(fps)
        self._frames = 0
        self._gl = ViewerGL(width=VIDEO_RES[0], height=VIDEO_RES[1], headless=True)
        self._gl.set_model(backend.model)
        self._gl.camera.pos = type(self._gl.camera.pos)(*(float(v) for v in cam.eye))
        self._gl.camera.look_at([float(v) for v in cam.lookat])
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(str(self._path), fps=self._fps, codec="libx264", quality=8,
                                          macro_block_size=8)

    def capture(self) -> None:
        self._gl.begin_frame(self._frames / self._fps)
        self._gl.log_state(self._backend.state_0)
        self._gl.end_frame()
        self._writer.append_data(self._gl.get_frame().numpy())
        self._frames += 1

    def close(self) -> Path:
        self._writer.close()
        self._gl.close()
        assert self._path.is_file(), f"newton viewer wrote no video at {self._path}"
        return self._path
