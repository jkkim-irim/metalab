#!/usr/bin/env python3
"""
ALLEX GR00T: Convert raw bilateral dataset to LeRobot V3.0 format.

Raw format:
  raw/episode_XXXXXX/
    state.npy    (T, 88)
    action.npy   (T, 44)
    camera_1/frame_XXXXXX.jpg
    camera_2/frame_XXXXXX.jpg
    episode_meta.json

Usage:
  python convert_to_lerobot_v3.py -i ~/pippo/data/allex_groot/groot_data/raw \
    -o ~/pippo/data/allex_groot/groot_data/groot_bilateral_torque \
    --fps 30 --task "cube_stacking" --include-torque
"""

import argparse
import json
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
import time as _time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import butter, filtfilt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATE_DIM = 88
ACTION_DIM = 44
OBS_STATE_DIM = 44  # pos only (44D)
IMG_SIZE = (224, 224)
FPS = 30
DEG2RAD = np.pi / 180.0


def _butter_lowpass(arr, cutoff_hz, fs_hz, order=4):
    """Zero-phase Butterworth low-pass, mirroring the WIRobotics preprocess."""
    nyq = fs_hz / 2.0
    if not 0 < cutoff_hz < nyq:
        raise ValueError(f"cutoff must be in (0, {nyq:.2f}) Hz; got {cutoff_hz}")
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, arr, axis=0)


def load_raw_episode(episode_dir: Path) -> dict | None:
    state_path = episode_dir / "state.npy"
    action_path = episode_dir / "action.npy"
    pos_target_path = episode_dir / "pos_target.npy"
    meta_path = episode_dir / "episode_meta.json"
    if not state_path.exists() or not action_path.exists():
        return None

    state = np.load(state_path)
    action = np.load(action_path)
    pos_target = np.load(pos_target_path) if pos_target_path.exists() else None
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    timestamps = []
    if "timestamps" in meta and "camera_1" in meta["timestamps"]:
        timestamps = meta["timestamps"]["camera_1"]

    images = {}
    for cam in ["camera_1", "camera_2"]:
        cam_dir = episode_dir / cam
        if cam_dir.exists():
            images[cam] = sorted(cam_dir.glob("frame_*.jpg"))

    return {"state": state, "action": action, "pos_target": pos_target,
            "meta": meta,
            "timestamps": timestamps, "images": images}


def make_video(image_paths: list[Path], output_path: Path, fps: int, vcodec: str, crf: int,
               size: tuple[int, int] = IMG_SIZE):
    """Encode a list of images into a single mp4.

    ``size`` is the ``(width, height)`` the frames are scaled to; it defaults to the ALLEX
    ``IMG_SIZE`` (224x224) so the existing converter is unchanged, and is overridable so other
    converters (e.g. ``libero_to_wir``, 256x256) can reuse this exact CFR encode.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Create temp file list for ffmpeg
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for p in image_paths:
            f.write(f"file '{p}'\n")
            f.write(f"duration {1.0/fps}\n")
        listfile = f.name

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
        "-vf", f"scale={size[0]}:{size[1]}",
        "-c:v", vcodec, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-r", str(fps), str(output_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    Path(listfile).unlink()


def compute_feature_stats(data: np.ndarray) -> dict:
    """Compute stats for a feature array (T, D)."""
    return {
        "mean": data.mean(0).tolist(),
        "std": data.std(0).tolist(),
        "min": data.min(0).tolist(),
        "max": data.max(0).tolist(),
        "count": [len(data)],
        "q01": np.quantile(data, 0.01, axis=0).tolist(),
        "q10": np.quantile(data, 0.10, axis=0).tolist(),
        "q50": np.quantile(data, 0.50, axis=0).tolist(),
        "q90": np.quantile(data, 0.90, axis=0).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).tolist(),
    }


def aggregate_stats(all_stats: list[dict]) -> dict:
    """Aggregate per-episode stats into global stats."""
    result = {}
    keys = all_stats[0].keys()
    for key in keys:
        feature_stats_list = [s[key] for s in all_stats]
        counts = np.array([s["count"][0] for s in feature_stats_list])
        total = int(counts.sum())
        means = np.array([s["mean"] for s in feature_stats_list])
        stds = np.array([s["std"] for s in feature_stats_list])
        weighted_mean = np.average(means, weights=counts, axis=0)

        # Aggregate std: sqrt(weighted_mean_of_variances + weighted_variance_of_means)
        vars_ = stds ** 2
        weighted_var = np.average(vars_, weights=counts, axis=0)
        mean_diff = means - weighted_mean
        var_of_means = np.average(mean_diff ** 2, weights=counts, axis=0)
        agg_std = np.sqrt(weighted_var + var_of_means)

        result[key] = {
            "mean": weighted_mean.tolist(),
            "std": agg_std.tolist(),
            "min": np.min([s["min"] for s in feature_stats_list], axis=0).tolist(),
            "max": np.max([s["max"] for s in feature_stats_list], axis=0).tolist(),
            "count": [total],
            "q01": np.quantile([s["q01"] for s in feature_stats_list], 0.01, axis=0).tolist(),
            "q10": np.quantile([s["q10"] for s in feature_stats_list], 0.10, axis=0).tolist(),
            "q50": np.median([s["q50"] for s in feature_stats_list], axis=0).tolist(),
            "q90": np.quantile([s["q90"] for s in feature_stats_list], 0.90, axis=0).tolist(),
            "q99": np.quantile([s["q99"] for s in feature_stats_list], 0.99, axis=0).tolist(),
        }
    return result


def convert_raw_to_lerobot_v3(
    input_raw_dir: Path,
    output_dir: Path,
    fps: int = 30,
    vcodec: str = "libx264",
    crf: int = 23,
    task_name: str = "unspecified",
    include_torque: bool = False,
    include_velocity: bool = False,
    include_pos_target: bool = False,        # append raw pos_target (rad)
    include_pos_target_delta: bool = False,  # append (pos_target - q) (rad)
    velocity_cutoff_hz: float = 10.0,
    torque_cutoff_hz: float = 0.0,  # 0 = no filtering
):
    input_raw_dir = Path(input_raw_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "meta").mkdir()
    (output_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (output_dir / "data" / "chunk-000").mkdir(parents=True)

    # obs_state layout = [pos(44)] + [vel(44)]? + [trq(44)]?
    obs_dim = OBS_STATE_DIM
    if include_velocity:
        obs_dim += 44
    if include_torque:
        obs_dim += 44
    if include_pos_target:
        obs_dim += 44
    if include_pos_target_delta:
        obs_dim += 44
    # Unit convention: when any of velocity/pos_target/pos_target_delta is on,
    # we use rad throughout (so q, dq, pos_target, delta share unit).
    state_use_rad = include_velocity or include_pos_target or include_pos_target_delta
    logger.info(
        "obs_state layout: pos(44)"
        + (" + vel(44)" if include_velocity else "")
        + (" + trq(44)" if include_torque else "")
        + (" + pos_target(44)" if include_pos_target else "")
        + (" + (pos_target-q)(44)" if include_pos_target_delta else "")
        + f" = {obs_dim}D · unit={'rad' if state_use_rad else 'deg'}"
    )

    # Find episodes
    episode_dirs = sorted([
        d for d in input_raw_dir.iterdir()
        if d.is_dir() and d.name.startswith("episode_") and "trash" not in d.name
    ])
    logger.info(f"Found {len(episode_dirs)} episodes in {input_raw_dir}")

    all_data_rows = []
    all_obs_states = []
    all_actions = []
    episode_stats_list = []
    episode_metadata_rows = []
    video_timestamps = {cam: [] for cam in ["observation.images.camera_1", "observation.images.camera_2"]}
    total_frames = 0
    video_cumulative_time = {"observation.images.camera_1": 0.0, "observation.images.camera_2": 0.0}

    # Temp dir for per-episode images (for video concat)
    all_images = {"camera_1": [], "camera_2": []}

    _t_start = _time.time()
    _total_eps = len(episode_dirs)
    print(f"[PROGRESS] 0/{_total_eps} starting conversion", flush=True)
    for ep_idx, ep_dir in enumerate(episode_dirs):
        _ep_t0 = _time.time()
        logger.info(f"Processing {ep_dir.name} ({ep_idx+1}/{_total_eps})")
        ep = load_raw_episode(ep_dir)
        if ep is None:
            logger.warning(f"Skip {ep_dir.name}: missing data")
            continue

        state_full = ep["state"]
        action = ep["action"]
        pos_target_full = ep.get("pos_target")
        T_raw = min(len(state_full), len(action))
        if pos_target_full is not None:
            T_raw = min(T_raw, len(pos_target_full))
        if T_raw == 0:
            continue

        state_full = state_full[:T_raw]
        action = action[:T_raw]
        if pos_target_full is not None:
            pos_target_full = pos_target_full[:T_raw]

        # Align to camera frame count (camera runs at ~30Hz, proprio at ~50Hz)
        cam1_count = len(ep["images"].get("camera_1", []))
        cam2_count = len(ep["images"].get("camera_2", []))
        T = min(cam1_count, cam2_count) if cam1_count > 0 else T_raw

        # Downsample state/action from T_raw to T (nearest-neighbor)
        if T < T_raw:
            indices = np.round(np.linspace(0, T_raw - 1, T)).astype(int)
            state_full = state_full[indices]
            action = action[indices]
            if pos_target_full is not None:
                pos_target_full = pos_target_full[indices]

        # Build obs_state = pos(44) + [vel(44)] + [trq(44)] based on flags.
        # Velocity/torque filtering runs on the downsampled signal at camera rate
        # (fps Hz), which is sufficient for 10Hz cutoff (Nyquist = fps/2).
        parts = []
        ts_T = np.arange(T, dtype=np.float64) / float(fps)
        state_fs = float(fps)

        q = state_full[:, :OBS_STATE_DIM].astype(np.float32)
        if state_use_rad:
            q = q * DEG2RAD
        parts.append(q)

        if include_velocity:
            dq = np.gradient(q, ts_T, axis=0).astype(np.float32)
            try:
                dq = _butter_lowpass(dq, velocity_cutoff_hz, state_fs).astype(np.float32)
            except ValueError as e:
                logger.warning(f"velocity filter skipped: {e}")
            parts.append(dq)

        if include_torque:
            tau = state_full[:, OBS_STATE_DIM:STATE_DIM].astype(np.float32)
            if torque_cutoff_hz > 0:
                try:
                    tau = _butter_lowpass(tau, torque_cutoff_hz, state_fs).astype(np.float32)
                except ValueError as e:
                    logger.warning(f"torque filter skipped: {e}")
            parts.append(tau)

        if include_pos_target or include_pos_target_delta:
            if pos_target_full is None:
                raise RuntimeError(
                    f"pos_target.npy missing in {ep_dir} but include_pos_target* requested"
                )
            # Raw pos_target is in degrees; convert to rad to match q.
            pt_rad = pos_target_full.astype(np.float32) * DEG2RAD
            if include_pos_target:
                parts.append(pt_rad)
            if include_pos_target_delta:
                parts.append((pt_rad - q).astype(np.float32))

        obs_state = np.concatenate(parts, axis=1).astype(np.float32)
        action = action[:T].astype(np.float32)

        # Timestamps — video-aligned (frame_idx / fps)
        timestamps = np.arange(T, dtype=np.float32) / fps

        # Per-episode stats
        ep_stats = {
            "observation.state": compute_feature_stats(obs_state),
            "action": compute_feature_stats(action),
        }
        episode_stats_list.append(ep_stats)

        all_obs_states.append(obs_state)
        all_actions.append(action)

        # Data rows
        for i in range(T):
            all_data_rows.append({
                "timestamp": float(timestamps[i]),
                "frame_index": i,
                "episode_index": ep_idx,
                "index": total_frames + i,
                "task_index": 0,
                "observation.state": obs_state[i].tolist(),
                "action": action[i].tolist(),
            })

        # Video timestamps tracking
        ep_duration = float(T) / fps
        for cam_key in ["observation.images.camera_1", "observation.images.camera_2"]:
            from_ts = video_cumulative_time[cam_key]
            to_ts = from_ts + ep_duration
            video_timestamps[cam_key].append((from_ts, to_ts))
            video_cumulative_time[cam_key] = to_ts

        # Collect images for video (trim to T frames)
        for cam in ["camera_1", "camera_2"]:
            imgs = sorted(ep["images"].get(cam, []))[:T]
            all_images[cam].extend(imgs)

        # Episode metadata
        ep_meta = {
            "episode_index": ep_idx,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": total_frames,
            "dataset_to_index": total_frames + T,
            "tasks": [task_name],
            "length": T,
        }
        # Video metadata
        for cam_key in ["observation.images.camera_1", "observation.images.camera_2"]:
            from_ts, to_ts = video_timestamps[cam_key][-1]
            ep_meta[f"videos/{cam_key}/chunk_index"] = 0
            ep_meta[f"videos/{cam_key}/file_index"] = 0
            ep_meta[f"videos/{cam_key}/from_timestamp"] = from_ts
            ep_meta[f"videos/{cam_key}/to_timestamp"] = to_ts

        # Flatten stats into episode metadata
        for feat_key, feat_stats in ep_stats.items():
            for stat_key, stat_val in feat_stats.items():
                ep_meta[f"stats/{feat_key}/{stat_key}"] = stat_val

        episode_metadata_rows.append(ep_meta)
        total_frames += T

        _elapsed = _time.time() - _t_start
        _done = ep_idx + 1
        _avg = _elapsed / _done
        _eta_s = _avg * (_total_eps - _done)
        print(
            f"[PROGRESS] {_done}/{_total_eps} {ep_dir.name} ({T} frames) "
            f"took {_time.time() - _ep_t0:.1f}s · elapsed {_elapsed:.0f}s · ETA {_eta_s:.0f}s",
            flush=True,
        )

    num_episodes = len(episode_metadata_rows)
    logger.info(f"Total: {num_episodes} episodes, {total_frames} frames")
    # Extend progress count: episode processing was the fast part.
    # Remaining: parquet write (1) + 2 video encodes + final meta write (1) = 4 steps.
    _total_phases = num_episodes + 4
    print(f"[PROGRESS] {num_episodes}/{_total_phases} episodes parsed, now writing parquet…",
          flush=True)

    # ── Write data parquet ──
    table = pa.table({
        "timestamp": pa.array([r["timestamp"] for r in all_data_rows], type=pa.float32()),
        "frame_index": pa.array([r["frame_index"] for r in all_data_rows], type=pa.int64()),
        "episode_index": pa.array([r["episode_index"] for r in all_data_rows], type=pa.int64()),
        "index": pa.array([r["index"] for r in all_data_rows], type=pa.int64()),
        "task_index": pa.array([r["task_index"] for r in all_data_rows], type=pa.int64()),
        "observation.state": [r["observation.state"] for r in all_data_rows],
        "action": [r["action"] for r in all_data_rows],
    })
    pq.write_table(table, output_dir / "data" / "chunk-000" / "file-000.parquet")
    logger.info(f"Wrote data parquet: {total_frames} rows")
    print(f"[PROGRESS] {num_episodes + 1}/{_total_phases} parquet written, encoding videos…",
          flush=True)

    # ── Write videos ──
    # ffmpeg's concat demuxer occasionally drops the final frame due to EOF
    # rounding. After encoding, probe the actual frame count and trim
    # parquet/metadata to match so LeRobot's video-backed __getitem__ never
    # tries to read past the end.
    actual_frame_counts = {}
    for _i, cam in enumerate(["camera_1", "camera_2"]):
        cam_key = f"observation.images.{cam}"
        video_dir = output_dir / "videos" / cam_key / "chunk-000"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / "file-000.mp4"
        if all_images[cam]:
            logger.info(f"Encoding {cam}: {len(all_images[cam])} frames -> {video_path}")
            _vid_t0 = _time.time()
            make_video(all_images[cam], video_path, fps, vcodec, crf)
            # Probe actual frame count
            try:
                result = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                     "-count_packets", "-show_entries", "stream=nb_read_packets",
                     "-of", "csv=p=0", str(video_path)],
                    capture_output=True, text=True, check=True,
                )
                actual = int(result.stdout.strip())
            except Exception as e:
                logger.warning(f"Could not probe {cam} frame count: {e}")
                actual = len(all_images[cam])
            actual_frame_counts[cam] = actual
            if actual != len(all_images[cam]):
                logger.warning(
                    f"{cam}: expected {len(all_images[cam])} frames, got {actual} "
                    f"(off by {len(all_images[cam]) - actual}). Will trim metadata."
                )
            print(
                f"[PROGRESS] {num_episodes + 2 + _i}/{_total_phases} encoded {cam} "
                f"({actual} frames in {_time.time()-_vid_t0:.1f}s)",
                flush=True,
            )

    # If any video dropped frames, trim global total_frames and the last
    # episode's length by the max shortfall across cameras.
    if actual_frame_counts:
        expected_total = total_frames
        min_actual = min(actual_frame_counts.values())
        shortfall = expected_total - min_actual
        if shortfall > 0:
            logger.warning(
                f"Video frame shortfall {shortfall} detected. "
                f"Trimming parquet + metadata to {min_actual} frames."
            )
            # Trim data parquet
            dp = output_dir / "data" / "chunk-000" / "file-000.parquet"
            dt = pq.read_table(dp)
            dt = dt.slice(0, min_actual)
            pq.write_table(dt, dp)
            # Adjust last-episode metadata in episode_metadata_rows
            # We drop frames from the tail of the last episode
            last_ep = episode_metadata_rows[-1]
            last_ep["length"] -= shortfall
            last_ep["dataset_to_index"] -= shortfall
            # Also adjust the video to_timestamp for the last episode
            for cam_key in ["observation.images.camera_1", "observation.images.camera_2"]:
                last_ep[f"videos/{cam_key}/to_timestamp"] -= shortfall / fps
            total_frames = min_actual

    # ── Get video codec info ──
    video_codec = vcodec
    video_pix_fmt = "yuv420p"

    # ── Write meta/info.json ──
    info = {
        "codebase_version": "v3.0",
        "robot_type": "allex",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": fps,
        "splits": {},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.state": {
                "dtype": "float32",
                "shape": [obs_dim],
                "names": None,
                "fps": fps,
            },
            "action": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": None,
                "fps": fps,
            },
            "observation.images.camera_1": {
                "dtype": "video",
                "shape": [3, IMG_SIZE[1], IMG_SIZE[0]],
                "names": ["channel", "height", "width"],
                "video.fps": fps,
                "video.codec": video_codec,
                "video.pix_fmt": video_pix_fmt,
                "video.is_depth_map": False,
                "has_audio": False,
            },
            "observation.images.camera_2": {
                "dtype": "video",
                "shape": [3, IMG_SIZE[1], IMG_SIZE[0]],
                "names": ["channel", "height", "width"],
                "video.fps": fps,
                "video.codec": video_codec,
                "video.pix_fmt": video_pix_fmt,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
    }
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # ── Write meta/stats.json ──
    global_stats = aggregate_stats(episode_stats_list)
    with open(output_dir / "meta" / "stats.json", "w") as f:
        json.dump(global_stats, f, indent=2)

    # ── Write meta/tasks.parquet ──
    tasks_df = pd.DataFrame({"task_index": [0]}, index=pd.Index([task_name], name="task"))
    tasks_table = pa.Table.from_pandas(tasks_df)
    pq.write_table(tasks_table, output_dir / "meta" / "tasks.parquet")

    # ── Write meta/episodes/chunk-000/file-000.parquet ──
    ep_columns = {}
    # Scalar columns
    for col in ["episode_index", "length"]:
        ep_columns[col] = pa.array([r[col] for r in episode_metadata_rows], type=pa.int64())
    for col in ["meta/episodes/chunk_index", "meta/episodes/file_index",
                "data/chunk_index", "data/file_index",
                "dataset_from_index", "dataset_to_index"]:
        ep_columns[col] = pa.array([r[col] for r in episode_metadata_rows], type=pa.int64())
    ep_columns["tasks"] = pa.array([r["tasks"] for r in episode_metadata_rows])

    # Video columns
    for cam_key in ["observation.images.camera_1", "observation.images.camera_2"]:
        for suffix in ["chunk_index", "file_index"]:
            col = f"videos/{cam_key}/{suffix}"
            ep_columns[col] = pa.array([r[col] for r in episode_metadata_rows], type=pa.int64())
        for suffix in ["from_timestamp", "to_timestamp"]:
            col = f"videos/{cam_key}/{suffix}"
            ep_columns[col] = pa.array([r[col] for r in episode_metadata_rows], type=pa.float64())

    # Stats columns (flattened)
    for feat_key in ["observation.state", "action"]:
        for stat_key in ["mean", "std", "min", "max", "count", "q01", "q10", "q50", "q90", "q99"]:
            col = f"stats/{feat_key}/{stat_key}"
            ep_columns[col] = pa.array([r[col] for r in episode_metadata_rows])

    ep_table = pa.table(ep_columns)
    pq.write_table(ep_table, output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    logger.info(
        f"Conversion complete: {num_episodes} episodes, {total_frames} frames -> {output_dir}\n"
        f"  Format: LeRobot v3.0\n"
        f"  State: {obs_dim}D {'(with torque)' if include_torque else '(pos only)'}\n"
        f"  Action: {ACTION_DIM}D"
    )
    print(
        f"[PROGRESS] {_total_phases}/{_total_phases} DONE · {num_episodes} episodes · "
        f"{total_frames} frames · {obs_dim}D obs_state · "
        f"total {_time.time() - _t_start:.0f}s · output: {output_dir}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Convert raw bilateral episodes to LeRobot V3.0")
    parser.add_argument("-i", "--input", required=True, help="Raw episodes directory")
    parser.add_argument("-o", "--output", required=True, help="Output LeRobot V3.0 directory")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--vcodec", default="libx264")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--task", default="unspecified")
    parser.add_argument("--include-torque", action="store_true",
                        help="Append raw torque (44D) to observation.state")
    parser.add_argument("--include-velocity", action="store_true",
                        help="Compute velocity via gradient + low-pass and append (44D). "
                             "When set, position is output in rad (WIRobotics convention).")
    parser.add_argument("--include-pos-target", action="store_true",
                        help="Append raw pos_target (44D rad).")
    parser.add_argument("--include-pos-target-delta", action="store_true",
                        help="Append (pos_target - q) delta (44D rad). Force-gap signal.")
    parser.add_argument("--velocity-cutoff-hz", type=float, default=10.0,
                        help="Butterworth low-pass cutoff for velocity (Hz). Default 10.")
    parser.add_argument("--torque-cutoff-hz", type=float, default=0.0,
                        help="Butterworth low-pass cutoff for torque (Hz). 0 = no filter. Default 0.")
    args = parser.parse_args()

    convert_raw_to_lerobot_v3(
        input_raw_dir=Path(args.input),
        output_dir=Path(args.output),
        fps=args.fps,
        vcodec=args.vcodec,
        crf=args.crf,
        task_name=args.task,
        include_torque=args.include_torque,
        include_velocity=args.include_velocity,
        include_pos_target=args.include_pos_target,
        include_pos_target_delta=args.include_pos_target_delta,
        velocity_cutoff_hz=args.velocity_cutoff_hz,
        torque_cutoff_hz=args.torque_cutoff_hz,
    )


if __name__ == "__main__":
    main()
