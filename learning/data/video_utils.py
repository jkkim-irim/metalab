"""Video frame decoding — lerobot-free reimplementation of lerobot 0.4.4's decode path.

Faithful port of the *read* path of ``lerobot/datasets/video_utils.py`` (0.4.4):
``decode_video_frames`` dispatching to ``decode_video_frames_torchcodec`` (default) or
``decode_video_frames_torchvision`` ("pyav" / "video_reader"). The encoder, streaming encoder and
ffprobe helpers are intentionally omitted — the trainer only ever decodes.

Deviation from lerobot: lerobot wraps the path in ``fsspec.open(...)`` before handing the file
handle to ``torchcodec.decoders.VideoDecoder``. ``fsspec`` is not in the allowed dependency set, and
ALLEX datasets live on local disk, so we pass the local path string straight to ``VideoDecoder``.
torchcodec decodes identically whether given a path or an open file handle, so decoded pixels and
``pts_seconds`` are unchanged. Everything else (seek_mode="approximate", frame-index rounding via
``average_fps``, the ``torch.cdist`` nearest-timestamp selection, the /255 float32 cast, and the
decoder cache) is preserved verbatim.
"""
import importlib.util
import logging
from pathlib import Path
from threading import Lock
from typing import Any

import torch


def get_safe_default_codec() -> str:
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logging.warning(
            "'torchcodec' is not available in your platform, falling back to 'pyav' as a default "
            "decoder"
        )
        return "pyav"


class FrameTimestampError(ValueError):
    """Helper error to indicate the retrieved timestamps exceed the queried ones."""

    pass


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    """Decode video frames using the specified backend.

    Args:
        video_path: Path to the video file.
        timestamps: List of timestamps to extract frames.
        tolerance_s: Allowed deviation in seconds for frame retrieval.
        backend: Backend to use for decoding. Defaults to "torchcodec" when available; otherwise
            "pyav".

    Returns:
        torch.Tensor: Decoded frames, float32 in [0, 1], channel-first.
    """
    if backend is None:
        backend = get_safe_default_codec()
    if backend == "torchcodec":
        return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s)
    elif backend in ["pyav", "video_reader"]:
        return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")


def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Load frames associated to the requested timestamps of a video.

    The backend can be either "pyav" (default) or "video_reader". Both use the CPU. See the lerobot
    docs for the trade-offs; this is a verbatim port of lerobot 0.4.4's implementation.
    """
    import torchvision

    video_path = str(video_path)

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesn't support accurate seek

    # set a video stream reader
    reader = torchvision.io.VideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    # access closest key frame of the first requested frame
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        raise FrameTimestampError(
            "One or several query timestamps unexpectedly violate the tolerance "
            f"({min_[~is_within_tol]} > {tolerance_s=})."
            " It means that the closest frame that can be loaded from the video is too far away in "
            "time."
            " This might be due to synchronization issues with timestamps during data collection."
            " To be safe, we advise to ignore this item during training."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts}"
            f"\nvideo: {video_path}"
            f"\nbackend: {backend}"
        )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    if len(timestamps) != len(closest_frames):
        raise FrameTimestampError(
            f"Number of retrieved frames ({len(closest_frames)}) does not match "
            f"number of queried timestamps ({len(timestamps)})"
        )
    return closest_frames


class VideoDecoderCache:
    """Thread-safe cache for torchcodec video decoders to avoid expensive re-initialization."""

    def __init__(self):
        self._cache: dict[str, Any] = {}
        self._lock = Lock()

    def get_decoder(self, video_path: str):
        """Get a cached decoder or create a new one."""
        if importlib.util.find_spec("torchcodec"):
            from torchcodec.decoders import VideoDecoder
        else:
            raise ImportError("torchcodec is required but not available.")

        video_path = str(video_path)

        with self._lock:
            if video_path not in self._cache:
                decoder = VideoDecoder(video_path, seek_mode="approximate")
                self._cache[video_path] = decoder

            return self._cache[video_path]

    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Return the number of cached decoders."""
        with self._lock:
            return len(self._cache)


_default_decoder_cache = VideoDecoderCache()


def decode_video_frames_torchcodec(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
    decoder_cache: VideoDecoderCache | None = None,
) -> torch.Tensor:
    """Load frames associated with the requested timestamps of a video using torchcodec.

    Verbatim port of lerobot 0.4.4 ``decode_video_frames_torchcodec`` (apart from the local-path
    handling documented in the module docstring).
    """
    if decoder_cache is None:
        decoder_cache = _default_decoder_cache

    # Use cached decoder instead of creating a new one each time
    decoder = decoder_cache.get_decoder(str(video_path))

    loaded_ts = []
    loaded_frames = []

    # get metadata for frame information
    metadata = decoder.metadata
    average_fps = metadata.average_fps
    # convert timestamps to frame indices
    frame_indices = [round(ts * average_fps) for ts in timestamps]
    # retrieve frames based on indices
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=True):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
        if log_loaded_timestamps:
            logging.info(f"Frame loaded at timestamp={pts:.4f}")

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and loaded timestamps
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        raise FrameTimestampError(
            "One or several query timestamps unexpectedly violate the tolerance "
            f"({min_[~is_within_tol]} > {tolerance_s=})."
            " It means that the closest frame that can be loaded from the video is too far away in "
            "time."
            " This might be due to synchronization issues with timestamps during data collection."
            " To be safe, we advise to ignore this item during training."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts}"
            f"\nvideo: {video_path}"
        )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to float32 in [0,1] range
    closest_frames = (closest_frames / 255.0).type(torch.float32)

    if not len(timestamps) == len(closest_frames):
        raise FrameTimestampError(
            f"Retrieved timestamps differ from queried {set(closest_frames) - set(timestamps)}"
        )

    return closest_frames
