#!/usr/bin/env python3
"""LibERO -> wir_v1: convert a LibERO LeRobot **v2.0 (images-in-parquet)** dataset into a wir_v1 /
LeRobot **v3.0 video** dataset that passes ``validate_wir_contract`` and loads via ``LeRobotDataset``.

Why this is a *real* convert (unlike ``mikasa_to_wir``, which only re-tags an already-v3.0+video
export): the LibERO source is LeRobot **v2.0** with the two cameras stored as an ``image``-dtype
column (PNG bytes embedded in the per-episode parquet). ``learning/data/lerobot_dataset.py`` has
**no decode path for an ``image`` column** — it only decodes ``video`` cameras through torchcodec.
So we MUST re-encode the PNG frames to an mp4 per camera and emit v3.0 ``video`` features.

Source layout (``--src`` root, LeRobot v2.0):

  * ``meta/info.json``   — ``codebase_version "v2.0"``, ``robot_type "panda"``, ``fps 10``, 73 tasks.
  * ``meta/tasks.jsonl`` — one ``{"task_index", "task"}`` per line (task_index -> NL string).
  * ``meta/episodes.jsonl`` — one ``{"episode_index", "tasks":[str], "length"}`` per line.
  * ``data/chunk-{c:03d}/episode_{e:06d}.parquet`` — one file per episode, one row per frame, columns
    ``image`` / ``wrist_image`` (dtype image = a pyarrow struct ``{bytes: <PNG bytes>, path: str}``,
    256x256x3), ``state`` f32[8], ``actions`` f32[7], ``timestamp``, ``frame_index``,
    ``episode_index``, ``index``, ``task_index``.

Output layout (wir_v1 / LeRobot v3.0), what we produce:

  * ``observation.images.top``  (from ``image``) and ``observation.images.wrist`` (from
    ``wrist_image``) as dtype **video** (mp4, libx264, yuv420p, CFR at fps). Declared shape
    ``(3, 256, 256)`` — channel-first, because ``validate_wir_contract`` requires ``shape[0] == 3``
    and the reader's torchcodec decode returns CHW frames.
  * ``observation.state`` f32[8] (from ``state``), ``action`` f32[7] (from ``actions``), plus the
    v3.0 bookkeeping columns ``timestamp/index/episode_index/frame_index/task_index``.
  * ``meta/`` : ``info.json`` (codebase_version 3.0, video features, fps 10, stamped
    ``"wir_data_version": "wir_v1"``), ``stats.json`` (state/action min/max/mean/std computed from
    the data + ImageNet mean/std per camera at shape ``(3,1,1)``), ``tasks.parquet``,
    ``meta/episodes/chunk-000/file-000.parquet``.

Task selection: LibERO-90 has 73 tasks. Convert the FULL multi-task suite (``--all-tasks``: keep every
episode, each carrying its own NL task string; ``tasks.parquet`` holds all distinct tasks re-indexed
``0..K-1``), or ONE task (``--task-id`` by task_index / ``--task-name`` by exact NL string; neither ->
task 0). By default ALL episodes are kept; ``--max-episodes N`` (N>0) trims to the first N (of the
task, or overall for ``--all-tasks``) for a quick smoke subset. Kept frames are re-indexed to
``episode_index 0..N-1`` / global ``index 0..F-1``; each frame's ``task_index`` points into
``tasks.parquet``.

REUSE: the video encode (``make_video``), per-episode stat computation (``compute_feature_stats``) and
global aggregation (``aggregate_stats``) come straight from
``learning.data.conversion.convert_raw_to_v3_state_variant`` (single source of truth); the meta-write
and the ffmpeg concat frame-drop workaround mirror that converter. Only the source decode (v2.0
images-in-parquet), the state/action dims (8/7), the camera names and the task/episode selection are
LibERO-specific.

CLI::

    python -m learning.data.conversion.libero_to_wir \
        --src /root/data/libero --out /scratch/libero_wir/task0 \
        --task-id 0 --max-episodes 40

    # or select the task by its natural-language string, and pull straight from S3:
    python -m learning.data.conversion.libero_to_wir \
        --src s3://wirobotics-internal/chrisryu/datasets/libero_90_lerobot \
        --out /scratch/libero_wir/mug --task-name "put the white mug on the plate" --max-episodes 20

Validate the result (metadata + contract; no video decode needed)::

    python -c "from learning.data.lerobot_dataset import LeRobotDatasetMetadata; \
        from learning.data.contract import validate_wir_contract; \
        m = LeRobotDatasetMetadata('libero/task0', root='/scratch/libero_wir/task0'); \
        validate_wir_contract(m); print('OK', m.total_episodes, m.total_frames, m.fps)"
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from learning.data.contract import WIR_DATA_VERSION
from learning.data.conversion.convert_raw_to_v3_state_variant import (
    aggregate_stats,
    compute_feature_stats,
    make_video,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATE_DIM = 8
ACTION_DIM = 7
IMG_HW = (256, 256)  # LibERO frames are 256x256; declared channel-first as (3, 256, 256).
FPS = 10

# Source image-column name -> output video feature key.
CAMERA_MAP = {"image": "observation.images.top", "wrist_image": "observation.images.wrist"}

# ImageNet normalization stats at shape (c, 1, 1) — identical to learning.data.factory.IMAGENET_STATS
# (kept as a local literal so the converter stays torch-free). make_dataset overrides camera stats
# with these anyway; we write them so meta/stats.json is complete and self-consistent on its own.
IMAGENET_MEAN_CHW = [[[0.485]], [[0.456]], [[0.406]]]
IMAGENET_STD_CHW = [[[0.229]], [[0.224]], [[0.225]]]


# --------------------------------------------------------------------------------------------------
# Source resolution + selection
# --------------------------------------------------------------------------------------------------
def resolve_source(src: str, out_dir: Path) -> tuple[Path, str | None]:
    """Return ``(local_root, s3_root)`` — the dataset root containing ``meta/info.json``.

    For an ``s3://`` ``--src`` only ``meta/`` is synced up front (the per-episode parquet files are
    pulled on demand in :func:`episode_parquet_path`, so a small subset never downloads the whole
    dataset); ``s3_root`` is the returned S3 base for those on-demand copies. For a local ``--src``
    the path is used directly and ``s3_root`` is ``None``. Fails loud if no ``meta/info.json`` exists.
    """
    if src.startswith("s3://"):
        s3_root = src.rstrip("/")
        cache = out_dir.parent / ".libero_raw_src"
        (cache / "meta").mkdir(parents=True, exist_ok=True)
        print(f"[libero_to_wir] aws s3 sync {s3_root}/meta -> {cache}/meta", flush=True)
        subprocess.run(["aws", "s3", "sync", f"{s3_root}/meta", str(cache / "meta")], check=True)
        root = cache
    else:
        root = Path(src).expanduser().resolve()
        s3_root = None
    if not (root / "meta" / "info.json").exists():
        raise FileNotFoundError(
            f"No LeRobot dataset at {root}: {root}/meta/info.json is missing. Point --src at the "
            "LibERO dataset root (contains meta/info.json) or an s3:// URI of it."
        )
    return root, s3_root


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSON-lines file into a list of dicts (fail loud if missing)."""
    if not path.exists():
        raise FileNotFoundError(f"expected {path} not found")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def select_task(tasks: list[dict], task_id: int | None, task_name: str | None) -> tuple[int, str]:
    """Resolve the (task_index, task_string) to keep from ``--task-id`` / ``--task-name``.

    Exactly one selector may be given; neither defaults to task 0. Fails loud on an unknown id, an
    unmatched / ambiguous name, or if both selectors are passed.
    """
    if task_id is not None and task_name is not None:
        raise ValueError("pass at most one of --task-id / --task-name, not both")
    by_index = {int(t["task_index"]): t["task"] for t in tasks}
    if task_name is not None:
        matches = [i for i, s in by_index.items() if s == task_name]
        if len(matches) == 0:
            raise ValueError(
                f"--task-name {task_name!r} matches no task; available: {sorted(by_index.values())}"
            )
        if len(matches) > 1:
            raise ValueError(f"--task-name {task_name!r} is ambiguous (task_index {matches})")
        return matches[0], task_name
    tid = 0 if task_id is None else int(task_id)
    if tid not in by_index:
        raise ValueError(f"--task-id {tid} out of range; known task_index {sorted(by_index)}")
    return tid, by_index[tid]


def select_episodes(episodes: list[dict], task_str: str, max_episodes: int) -> list[int]:
    """Return the episode_index list for ``task_str`` (episodes.jsonl order), capped at ``max_episodes``.

    Membership uses the ``tasks`` NL-string list each episode carries in episodes.jsonl. Fails loud if
    no episode matches the task.
    """
    matched = [int(ep["episode_index"]) for ep in episodes if task_str in ep.get("tasks", [])]
    if not matched:
        raise ValueError(f"no episodes found for task {task_str!r}")
    if max_episodes > 0:
        matched = matched[:max_episodes]
    return matched


def select_all_tasks(episodes: list[dict], max_episodes: int) -> tuple[list[int], dict[int, str], list[str]]:
    """Full-suite selection: keep EVERY episode (optionally capped to the first ``max_episodes`` for a
    smoke) and build the multi-task registry from the episodes' own NL task strings.

    Returns ``(selected_episode_indices, per_episode_task_str, ordered_task_list)``, where each distinct
    task gets a contiguous index by first-appearance order in ``ordered_task_list``. Fails loud if any
    kept episode does not carry exactly one task string.
    """
    by_index = {int(ep["episode_index"]): ep for ep in episodes}
    selected = sorted(by_index)
    if max_episodes > 0:
        selected = selected[:max_episodes]
    per_ep: dict[int, str] = {}
    ordered: list[str] = []
    seen: set[str] = set()
    for ei in selected:
        ep_tasks = by_index[ei].get("tasks", [])
        if len(ep_tasks) != 1:
            raise ValueError(f"episode {ei} carries {len(ep_tasks)} task strings, expected exactly 1: {ep_tasks}")
        t = ep_tasks[0]
        per_ep[ei] = t
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return selected, per_ep, ordered


def episode_parquet_path(root: Path, src_info: dict, ep_idx: int, s3_root: str | None) -> Path:
    """Return the local path to episode ``ep_idx``'s parquet, copying it from S3 on demand if needed."""
    chunks_size = int(src_info.get("chunks_size", 1000))
    rel = src_info["data_path"].format(episode_chunk=ep_idx // chunks_size, episode_index=ep_idx)
    local = root / rel
    if not local.exists():
        if s3_root is None:
            raise FileNotFoundError(f"expected episode parquet missing: {local}")
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"[libero_to_wir] aws s3 cp {s3_root}/{rel} -> {local}", flush=True)
        subprocess.run(["aws", "s3", "cp", f"{s3_root}/{rel}", str(local)], check=True)
    return local


# --------------------------------------------------------------------------------------------------
# Image decoding (v2.0 image-dtype cell)
# --------------------------------------------------------------------------------------------------
def decode_image_cell(cell) -> np.ndarray:
    """Decode one LibERO ``image`` cell to an ``(H, W, 3)`` uint8 RGB array.

    A cell may be (LibERO's actual layout) a struct ``{"bytes": <PNG bytes>, "path": str}``, or raw
    PNG/JPEG ``bytes``, or an already-decoded ``(H, W, 3)`` array/nested list. Fails loud on anything
    else so a schema change is caught, not silently mangled.
    """
    if isinstance(cell, dict):
        data = cell.get("bytes")
        if data is None:
            raise ValueError(f"image struct cell has no 'bytes' (keys={list(cell)})")
        cell = data
    if isinstance(cell, (bytes, bytearray)):
        arr = np.asarray(Image.open(io.BytesIO(bytes(cell))).convert("RGB"))
    else:
        arr = np.asarray(cell)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"decoded image has shape {arr.shape}, expected (H, W, 3)")
    if arr.dtype != np.uint8:
        raise ValueError(f"decoded image dtype {arr.dtype}, expected uint8")
    return arr


# --------------------------------------------------------------------------------------------------
# Video frame-count probe (mirror of the convert_raw_to_v3 frame-drop workaround)
# --------------------------------------------------------------------------------------------------
def probe_nframes(path: Path, expected: int) -> int:
    """Return the encoded video's frame count via ffprobe, or ``expected`` (with a warning) if the
    probe is unavailable/fails — matching ``convert_raw_to_v3_state_variant``'s inline fallback."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        )
        return int(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logger.warning(f"could not probe frame count of {path} ({e}); assuming {expected}")
        return expected


# --------------------------------------------------------------------------------------------------
# Per-episode decode worker (runs in a process pool — the decode + PNG re-encode is the cost)
# --------------------------------------------------------------------------------------------------
def _decode_one_episode(job: dict) -> dict:
    """Pool worker: read one source episode, validate it, and decode its frames to PNGs under a
    per-episode temp dir; also compute its state/action stats. Self-contained + picklable (episodes
    are independent). No global frame offsets here — the parent stitches the per-episode frame lists
    together in order, and ``make_video`` honours that list order (not filenames).
    """
    out_ep_idx, src_ep_idx = job["out_ep_idx"], job["src_ep_idx"]
    root, tmp_dir, _fps = Path(job["root"]), Path(job["tmp_dir"]), job["fps"]
    src_info, s3_root = job["src_info"], job["s3_root"]

    pq_path = episode_parquet_path(root, src_info, src_ep_idx, s3_root)
    table = pq.read_table(str(pq_path))
    T = table.num_rows
    if T == 0:
        raise ValueError(f"episode {src_ep_idx} parquet has 0 rows")
    state = np.asarray(table.column("state").to_pylist(), dtype=np.float32)
    action = np.asarray(table.column("actions").to_pylist(), dtype=np.float32)
    if state.shape != (T, STATE_DIM):
        raise ValueError(f"episode {src_ep_idx} state shape {state.shape} != {(T, STATE_DIM)}")
    if action.shape != (T, ACTION_DIM):
        raise ValueError(f"episode {src_ep_idx} action shape {action.shape} != {(T, ACTION_DIM)}")

    frames: dict[str, list[str]] = {}
    for src_key, cam in CAMERA_MAP.items():
        col = table.column(src_key)
        cam_dir = tmp_dir / f"ep{out_ep_idx:06d}" / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for i in range(T):
            arr = decode_image_cell(col[i].as_py())
            if arr.shape != (IMG_HW[0], IMG_HW[1], 3):
                raise ValueError(
                    f"episode {src_ep_idx} {src_key} frame {i} shape {arr.shape} "
                    f"!= {(IMG_HW[0], IMG_HW[1], 3)}"
                )
            dst = cam_dir / f"frame_{i:05d}.png"
            Image.fromarray(arr).save(dst)
            paths.append(str(dst))
        frames[cam] = paths

    ep_stats = {
        "observation.state": compute_feature_stats(state),
        "action": compute_feature_stats(action),
    }
    return {"out_ep_idx": out_ep_idx, "src_ep_idx": src_ep_idx, "T": T,
            "state": state, "action": action, "ep_stats": ep_stats, "frames": frames}


# --------------------------------------------------------------------------------------------------
# Main conversion
# --------------------------------------------------------------------------------------------------
def convert(src: str, out: str, task_id: int | None, task_name: str | None, all_tasks: bool,
            max_episodes: int, vcodec: str, crf: int, workers: int) -> None:
    out_dir = Path(out).expanduser().resolve()
    root, s3_root = resolve_source(src, out_dir)

    src_info = json.loads((root / "meta" / "info.json").read_text())
    if str(src_info.get("codebase_version", "")).lstrip("v").split(".")[0] != "2":
        raise ValueError(
            f"source is not LeRobot v2.x: codebase_version={src_info.get('codebase_version')!r}"
        )
    fps = int(src_info["fps"])
    robot_type = src_info.get("robot_type", "panda")

    tasks = load_jsonl(root / "meta" / "tasks.jsonl")
    episodes = load_jsonl(root / "meta" / "episodes.jsonl")
    if all_tasks:
        selected, per_ep_task, task_list = select_all_tasks(episodes, max_episodes)
        print(
            f"[libero_to_wir] source={root} out={out_dir}\n"
            f"[libero_to_wir] ALL-TASKS: {len(task_list)} tasks, {len(selected)} episodes fps={fps}",
            flush=True,
        )
    else:
        sel_task_index, task_str = select_task(tasks, task_id, task_name)
        selected = select_episodes(episodes, task_str, max_episodes)
        per_ep_task = {i: task_str for i in selected}
        task_list = [task_str]
        print(
            f"[libero_to_wir] source={root} out={out_dir}\n"
            f"[libero_to_wir] task_index={sel_task_index} task={task_str!r} "
            f"episodes={len(selected)} (of the task) fps={fps}",
            flush=True,
        )
    task_to_index = {t: i for i, t in enumerate(task_list)}

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (out_dir / "data" / "chunk-000").mkdir(parents=True)

    all_data_rows: list[dict] = []
    episode_stats_list: list[dict] = []
    episode_metadata_rows: list[dict] = []
    total_frames = 0
    video_cumulative_time = {cam: 0.0 for cam in CAMERA_MAP.values()}

    # Decode frames to PNGs (per-episode temp dirs) IN PARALLEL — decode + PNG re-encode is the cost
    # and episodes are independent — then stitch each episode's frames/rows/meta together IN ORDER
    # (make_video honours the list order, not filenames) and feed ffmpeg's concat demuxer one
    # consolidated mp4 per camera.
    with tempfile.TemporaryDirectory(prefix="libero_frames_") as tmp:
        tmp_dir = Path(tmp)
        all_images = {cam: [] for cam in CAMERA_MAP.values()}
        jobs = [{"out_ep_idx": oi, "src_ep_idx": si, "root": str(root), "src_info": src_info,
                 "s3_root": s3_root, "tmp_dir": str(tmp_dir), "fps": fps}
                for oi, si in enumerate(selected)]
        n_jobs = len(jobs)
        print(f"[libero_to_wir] decoding {n_jobs} episodes with {workers} worker(s)...", flush=True)

        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(_decode_one_episode, jobs):  # yields in submission (episode) order
                out_ep_idx, src_ep_idx, T = res["out_ep_idx"], res["src_ep_idx"], res["T"]
                state, action, ep_stats = res["state"], res["action"], res["ep_stats"]
                episode_stats_list.append(ep_stats)
                for cam in CAMERA_MAP.values():
                    all_images[cam].extend(Path(p) for p in res["frames"][cam])

                timestamps = np.arange(T, dtype=np.float32) / fps
                for i in range(T):
                    all_data_rows.append({
                        "timestamp": float(timestamps[i]),
                        "frame_index": i,
                        "episode_index": out_ep_idx,
                        "index": total_frames + i,
                        "task_index": task_to_index[per_ep_task[src_ep_idx]],
                        "observation.state": state[i].tolist(),
                        "action": action[i].tolist(),
                    })

                ep_duration = float(T) / fps
                ep_meta = {
                    "episode_index": out_ep_idx,
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": 0,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": total_frames,
                    "dataset_to_index": total_frames + T,
                    "tasks": [per_ep_task[src_ep_idx]],
                    "length": T,
                }
                for cam in CAMERA_MAP.values():
                    from_ts = video_cumulative_time[cam]
                    to_ts = from_ts + ep_duration
                    video_cumulative_time[cam] = to_ts
                    ep_meta[f"videos/{cam}/chunk_index"] = 0
                    ep_meta[f"videos/{cam}/file_index"] = 0
                    ep_meta[f"videos/{cam}/from_timestamp"] = from_ts
                    ep_meta[f"videos/{cam}/to_timestamp"] = to_ts
                for feat_key, feat_stats in ep_stats.items():
                    for stat_key, stat_val in feat_stats.items():
                        ep_meta[f"stats/{feat_key}/{stat_key}"] = stat_val
                episode_metadata_rows.append(ep_meta)
                total_frames += T
                done += 1
                if done % 100 == 0 or done == n_jobs:
                    print(f"[libero_to_wir] assembled {done}/{n_jobs} episodes "
                          f"({total_frames} frames)", flush=True)

        num_episodes = len(episode_metadata_rows)

        # ── Write data parquet (may be re-sliced below if ffmpeg dropped tail frames) ──
        data_path = out_dir / "data" / "chunk-000" / "file-000.parquet"
        _write_data_parquet(all_data_rows, data_path)

        # ── Encode one consolidated mp4 per camera (CFR at fps, 256x256) + probe frame counts ──
        actual_frame_counts = {}
        for src_key, cam in CAMERA_MAP.items():
            video_path = out_dir / "videos" / cam / "chunk-000" / "file-000.mp4"
            logger.info(f"encoding {cam}: {len(all_images[cam])} frames -> {video_path}")
            make_video(all_images[cam], video_path, fps, vcodec, crf, size=(IMG_HW[1], IMG_HW[0]))
            actual_frame_counts[cam] = probe_nframes(video_path, len(all_images[cam]))

    # ── Frame-drop workaround: if any encode is short, trim parquet + last-episode meta to match ──
    min_actual = min(actual_frame_counts.values())
    shortfall = total_frames - min_actual
    if shortfall > 0:
        logger.warning(f"video frame shortfall {shortfall}; trimming parquet + metadata to {min_actual}")
        dt = pq.read_table(str(data_path)).slice(0, min_actual)
        pq.write_table(dt, str(data_path))
        last_ep = episode_metadata_rows[-1]
        last_ep["length"] -= shortfall
        last_ep["dataset_to_index"] -= shortfall
        for cam in CAMERA_MAP.values():
            last_ep[f"videos/{cam}/to_timestamp"] -= shortfall / fps
        total_frames = min_actual

    # ── Write meta ──
    _write_info(out_dir, robot_type, num_episodes, total_frames, fps, vcodec, len(task_list))
    _write_stats(out_dir, episode_stats_list, total_frames)
    _write_tasks(out_dir, task_list)
    _write_episodes(out_dir, episode_metadata_rows)

    print(
        f"[libero_to_wir] DONE tasks={len(task_list)} episodes={num_episodes} "
        f"frames={total_frames} fps={fps} cameras={list(CAMERA_MAP.values())} codec={vcodec} "
        f"-> {out_dir}",
        flush=True,
    )


def _write_data_parquet(rows: list[dict], path: Path) -> None:
    table = pa.table({
        "timestamp": pa.array([r["timestamp"] for r in rows], type=pa.float32()),
        "frame_index": pa.array([r["frame_index"] for r in rows], type=pa.int64()),
        "episode_index": pa.array([r["episode_index"] for r in rows], type=pa.int64()),
        "index": pa.array([r["index"] for r in rows], type=pa.int64()),
        "task_index": pa.array([r["task_index"] for r in rows], type=pa.int64()),
        "observation.state": [r["observation.state"] for r in rows],
        "action": [r["action"] for r in rows],
    })
    pq.write_table(table, str(path))


def _write_info(out_dir: Path, robot_type: str, num_episodes: int, total_frames: int, fps: int,
                vcodec: str, total_tasks: int) -> None:
    def _video_feature() -> dict:
        return {
            "dtype": "video",
            "shape": [3, IMG_HW[0], IMG_HW[1]],
            "names": ["channel", "height", "width"],
            "video.fps": fps,
            "video.codec": vcodec,
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        }

    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": fps,
        "splits": {"train": f"0:{num_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "wir_data_version": WIR_DATA_VERSION,
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": None, "fps": fps},
            "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": None, "fps": fps},
            "observation.images.top": _video_feature(),
            "observation.images.wrist": _video_feature(),
        },
    }
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2))


def _write_stats(out_dir: Path, episode_stats_list: list[dict], total_frames: int) -> None:
    stats = aggregate_stats(episode_stats_list)
    # Camera stats: ImageNet mean/std at (3,1,1); min/max the [0,1] float range the reader produces.
    for cam in CAMERA_MAP.values():
        stats[cam] = {
            "mean": IMAGENET_MEAN_CHW,
            "std": IMAGENET_STD_CHW,
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "count": [total_frames],
        }
    (out_dir / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))


def _write_tasks(out_dir: Path, task_list: list[str]) -> None:
    tasks_df = pd.DataFrame({"task_index": list(range(len(task_list)))},
                            index=pd.Index(task_list, name="task"))
    pq.write_table(pa.Table.from_pandas(tasks_df), str(out_dir / "meta" / "tasks.parquet"))


def _write_episodes(out_dir: Path, rows: list[dict]) -> None:
    cols = {}
    for col in ["episode_index", "length", "meta/episodes/chunk_index", "meta/episodes/file_index",
                "data/chunk_index", "data/file_index", "dataset_from_index", "dataset_to_index"]:
        cols[col] = pa.array([r[col] for r in rows], type=pa.int64())
    cols["tasks"] = pa.array([r["tasks"] for r in rows])
    for cam in CAMERA_MAP.values():
        for suffix in ["chunk_index", "file_index"]:
            col = f"videos/{cam}/{suffix}"
            cols[col] = pa.array([r[col] for r in rows], type=pa.int64())
        for suffix in ["from_timestamp", "to_timestamp"]:
            col = f"videos/{cam}/{suffix}"
            cols[col] = pa.array([r[col] for r in rows], type=pa.float64())
    for feat_key in ["observation.state", "action"]:
        for stat_key in ["mean", "std", "min", "max", "count", "q01", "q10", "q50", "q90", "q99"]:
            col = f"stats/{feat_key}/{stat_key}"
            cols[col] = pa.array([r[col] for r in rows])
    pq.write_table(pa.table(cols), str(out_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert a LibERO LeRobot v2.0 (images-in-parquet) dataset into a wir_v1 "
                    "(LeRobot v3.0 video) dataset: one task (--task-id/--task-name) or the full "
                    "multi-task suite (--all-tasks)."
    )
    p.add_argument("--src", required=True,
                   help="LibERO dataset root (has meta/info.json) or an s3:// URI of it.")
    p.add_argument("--out", required=True, help="Output wir_v1 dataset directory.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--task-id", type=int, default=None,
                   help="Select one task by its task_index (default: task 0 if no selector given).")
    g.add_argument("--task-name", type=str, default=None,
                   help="Select one task by its exact natural-language string.")
    g.add_argument("--all-tasks", action="store_true",
                   help="Convert ALL tasks into a single multi-task wir_v1 dataset (the full suite).")
    p.add_argument("--max-episodes", type=int, default=0,
                   help="Keep only the first N episodes (of the task, or overall for --all-tasks); "
                        "<=0 = keep ALL (default). Set >0 only for a quick smoke subset.")
    p.add_argument("--vcodec", default="libx264", help="Video codec (default libx264).")
    p.add_argument("--crf", type=int, default=23, help="Encode CRF (default 23).")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                   help="Parallel decode workers (default: all CPUs). Set 1 to force sequential.")
    args = p.parse_args()

    convert(
        src=args.src, out=args.out, task_id=args.task_id, task_name=args.task_name,
        all_tasks=args.all_tasks, max_episodes=args.max_episodes, vcodec=args.vcodec, crf=args.crf,
        workers=max(1, args.workers),
    )


if __name__ == "__main__":
    main()
