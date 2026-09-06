from __future__ import annotations

from pathlib import Path


class VideoRecorder:
    def __init__(self, backend, path: Path, fps: float):
        assert backend.camera is not None, "genesis VideoRecorder needs build_env(video=True) (offscreen camera)"
        self._cam = backend.camera
        self._path = Path(path)
        self._fps = float(fps)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._cam.start_recording()

    def capture(self) -> None:
        self._cam.render()

    def close(self) -> Path:
        self._cam.stop_recording(save_to_filename=str(self._path), fps=self._fps)
        assert self._path.is_file(), f"genesis camera wrote no video at {self._path}"
        return self._path
