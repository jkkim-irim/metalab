"""Small video helpers for the RL tooling (node-side)."""
from __future__ import annotations

import os
import subprocess


def slow_mp4(path: str, factor: float = 5.0) -> str:
    """Write a ``factor``× slowed re-encode of an MP4 next to it and return its path.

    For players with no speed control — the W&B web player — where a 50 Hz-recorded clip plays too fast
    to interpret (the S3 HTML reports don't need this; they slow playback client-side). Best-effort: on
    any failure returns the original path so an upload never breaks over cosmetics."""
    try:
        import imageio_ffmpeg  # ships the ffmpeg binary the recorder already depends on

        out = f"{path[:-4]}_slow{int(factor)}x.mp4"
        if not os.path.exists(out):
            subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", path,
                 "-filter:v", f"setpts={factor}*PTS", "-an", "-movflags", "+faststart", out],
                check=True, timeout=180)
        return out
    except Exception as e:
        # Cosmetic fallback, but never silent: a broken ffmpeg would otherwise quietly degrade
        # every uploaded clip to unwatchable speed with no trace.
        print(f"[video] WARN slow_mp4 failed for {path}: {e!r} — uploading at original speed", flush=True)
        return path
