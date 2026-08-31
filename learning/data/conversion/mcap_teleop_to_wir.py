#!/usr/bin/env python3
"""Direct ROS2 MCAP teleop -> WIR_v1 (LeRobot v3.0) converter.

Reads the teleop-dashboard recorder's rosbag2 MCAP episodes
(``<episode>/bag/bag_0.mcap`` + ``meta.json``) and writes a WIR_v1 (LeRobot v3.0)
dataset **directly**: parquet frames + consolidated mp4 videos + meta/{info,stats,
tasks,episodes}. This fills the Phase-3 "offline raw -> LeRobot/GR00T converter" gap
(``apps/teleop_dashboard/docs/DATA_LOGGING.md``); until now only the older intermediate
``raw/*.npy`` path (``convert_raw_to_v3_state_variant.py``) existed, and nothing bridged
the MCAP bags.

It reconstructs the canonical ALLEX bilateral wir_v1 embodiment
(``embodiment/allex_modality.py``: ``STATE_GROUPS = {q:(0,44), dq:(44,88), tau:(88,132)}``):

  observation.state 132-D = [ q_rad(44) | dq(44) | tau(44) ]
      q   = r_arm(7), l_arm(7), r_hand(15), l_hand(15)   from ``.../joint_positions_deg`` (deg->rad)
      dq  = np.gradient(q) + Butterworth low-pass         (identical to the raw->v3 converter)
      tau = r_arm(7), l_arm(7), r_hand(15), l_hand(15)   from ``.../joint_torque`` (as-is)
  action 44-D = r_arm(7), l_arm(7), r_hand(15), l_hand(15) from ``robot_inbound/.../joint_command`` (rad)

Per hand: 5 fingers x their first 3 joints = 15 (matches the GR00T bilateral logger's
``[:HAND_FINGER_DIM]`` slicing). ``--policy.state_keys '["q"]'`` selects obs_state[:, 0:44].

Cameras: camera_1 = ZED left, camera_2 = ZED right (this rig has no realsense/orbbec).

Sampling: native ~60 Hz. The anchor grid is the ZED-left frame times; at each anchor we
zero-order-hold the latest value <= t for every joint topic and pair the nearest ZED-right
frame. Co-temporal by construction (no "assume uniform co-sampling" index resample), so full
60 Hz fidelity is kept. ``--fps`` sets the nominal rate stamped into the dataset (default 60).

The heavy numeric + writer helpers (stats, aggregation, video encode, Butterworth) are imported
from ``convert_raw_to_v3_state_variant.py`` so the validated parquet/stats/video path is reused;
only the MCAP front-end here is new.

Usage:
  python mcap_teleop_to_wir.py -i <dir with episode_*/bag/bag_0.mcap> -o <out> \
      --fps 60 --task "put the eggplant in or out of the plate" [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Reuse the validated numerics + writer bits from the raw->v3 converter (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_raw_to_v3_state_variant import (  # noqa: E402
    DEG2RAD,
    _butter_lowpass,
    aggregate_stats,
    compute_feature_stats,
)

from mcap.reader import make_reader  # noqa: E402
from mcap_ros2.decoder import DecoderFactory  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mcap_teleop_to_wir")

# ── Embodiment dims (must match embodiment/allex_modality.py) ──
ARM_DIM = 7
HAND_FINGER_DIM = 3
NUM_FINGERS = 5
HAND_DIM = HAND_FINGER_DIM * NUM_FINGERS  # 15
Q_DIM = (ARM_DIM + HAND_DIM) * 2          # 44 (r_arm, l_arm, r_hand, l_hand)
ACTION_DIM = Q_DIM                        # 44
STATE_DIM = Q_DIM * 3                     # 132 = q | dq | tau
IMG_SIZE = (224, 224)

FINGERS = ["thumb", "index", "middle", "ring", "little"]
NS = "/allex/p001"

# ZED stereo (this rig). camera_1 = left (primary), camera_2 = right.
CAM1_TOPIC = "/zed/zed_node/left/color/rect/image/compressed"
CAM2_TOPIC = "/zed/zed_node/right/color/rect/image/compressed"


def _pos_topics(side: str) -> list[str]:
    arm = f"{NS}/robot_outbound_data/{side}_arm/joint_positions_deg"
    fingers = [f"{NS}/robot_outbound_data/{side}_{f}/joint_positions_deg" for f in FINGERS]
    return [arm] + fingers


def _trq_topics(side: str) -> list[str]:
    arm = f"{NS}/robot_outbound_data/{side}_arm/joint_torque"
    fingers = [f"{NS}/robot_outbound_data/{side}_{f}/joint_torque" for f in FINGERS]
    return [arm] + fingers


def _cmd_topics(side: str) -> list[str]:
    arm = f"{NS}/robot_inbound/{side}_arm/joint_command"
    fingers = [f"{NS}/robot_inbound/{side}_{f}/joint_command" for f in FINGERS]
    return [arm] + fingers


# Ordered lists of (topic, take) where take = dims to slice from the front of the message.
def _block_spec(topics_fn) -> list[tuple[str, int]]:
    """[(arm_topic, 7)] + 5x [(finger_topic, 3)] for right then left = 7+15 per side, x2 sides."""
    spec: list[tuple[str, int]] = []
    for side in ("right", "left"):
        tops = topics_fn(side)
        spec.append((tops[0], ARM_DIM))
        for ft in tops[1:]:
            spec.append((ft, HAND_FINGER_DIM))
    return spec


POS_SPEC = _block_spec(_pos_topics)      # -> 44 dims (deg)
TRQ_SPEC = _block_spec(_trq_topics)      # -> 44 dims
CMD_SPEC = _block_spec(_cmd_topics)      # -> 44 dims (rad)

ALL_JOINT_TOPICS = sorted({t for spec in (POS_SPEC, TRQ_SPEC, CMD_SPEC) for t, _ in spec})
NEEDED_TOPICS = ALL_JOINT_TOPICS + [CAM1_TOPIC, CAM2_TOPIC]


class _Series:
    """Sorted (time_ns, vector) series with zero-order-hold lookup."""

    def __init__(self) -> None:
        self.t: list[int] = []
        self.v: list[np.ndarray] = []

    def add(self, t_ns: int, vec: np.ndarray) -> None:
        self.t.append(t_ns)
        self.v.append(vec)

    def finalize(self) -> None:
        if not self.t:
            self._t = np.zeros(0, dtype=np.int64)
            self._v = np.zeros((0, 0), dtype=np.float32)
            return
        order = np.argsort(np.asarray(self.t, dtype=np.int64), kind="stable")
        self._t = np.asarray(self.t, dtype=np.int64)[order]
        self._v = np.stack([self.v[i] for i in order], axis=0).astype(np.float32)

    @property
    def first_t(self) -> int | None:
        return int(self._t[0]) if len(self._t) else None

    def zoh(self, times_ns: np.ndarray) -> np.ndarray:
        """Latest value at-or-before each query time (zero-order hold)."""
        idx = np.searchsorted(self._t, times_ns, side="right") - 1
        idx = np.clip(idx, 0, len(self._t) - 1)
        return self._v[idx]


def read_episode_mcap(mcap_path: Path) -> dict | None:
    """Decode one episode bag into co-temporal 60 Hz-ish arrays anchored on ZED-left frames.

    Returns dict(q_deg (T,44), tau (T,44), action (T,44), cam1 list[bytes], cam2 list[bytes]).
    """
    joint_series: dict[str, _Series] = {t: _Series() for t in ALL_JOINT_TOPICS}
    cam1: list[tuple[int, bytes]] = []
    cam2: list[tuple[int, bytes]] = []
    cam_format: dict[str, str] = {}

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, ros_msg in reader.iter_decoded_messages(
            topics=NEEDED_TOPICS
        ):
            topic = channel.topic
            t = int(message.log_time)  # ns, receive time (single clock for all topics)
            if topic in joint_series:
                joint_series[topic].add(t, np.asarray(ros_msg.data, dtype=np.float32))
            elif topic == CAM1_TOPIC:
                cam1.append((t, bytes(ros_msg.data)))
                cam_format.setdefault("cam1", str(getattr(ros_msg, "format", "")))
            elif topic == CAM2_TOPIC:
                cam2.append((t, bytes(ros_msg.data)))
                cam_format.setdefault("cam2", str(getattr(ros_msg, "format", "")))

    for s in joint_series.values():
        s.finalize()
    cam1.sort(key=lambda x: x[0])
    cam2.sort(key=lambda x: x[0])
    if len(cam1) == 0 or len(cam2) == 0:
        logger.warning(f"{mcap_path.parent.parent.name}: missing ZED frames "
                       f"(cam1={len(cam1)}, cam2={len(cam2)}); skip")
        return None

    # Valid window: essential arm topics (pos+cmd, both sides) + both cameras must exist.
    # Missing fingers / torque are zero-filled (not fatal) — torque is not used by the
    # state_keys=["q"] recipe, and a task may leave one hand idle.
    ep_name = mcap_path.parent.parent.name
    missing = [t for t in ALL_JOINT_TOPICS if joint_series[t].first_t is None]
    if missing:
        logger.warning(f"{ep_name}: {len(missing)} joint topic(s) absent -> zero-filled: "
                       f"{missing[:4]}{'...' if len(missing) > 4 else ''}")
    essential = [_pos_topics("right")[0], _pos_topics("left")[0],
                 _cmd_topics("right")[0], _cmd_topics("left")[0]]
    if any(joint_series[t].first_t is None for t in essential):
        logger.warning(f"{ep_name}: an essential arm pos/cmd topic is missing; skip")
        return None
    present_firsts = [s.first_t for s in joint_series.values() if s.first_t is not None]
    start_t = max(max(present_firsts), cam1[0][0], cam2[0][0])

    cam1_t = np.array([t for t, _ in cam1], dtype=np.int64)
    cam2_t = np.array([t for t, _ in cam2], dtype=np.int64)
    # Anchor grid = ZED-left frames within the valid window.
    keep = np.nonzero(cam1_t >= start_t)[0]
    if len(keep) == 0:
        logger.warning(f"{mcap_path.parent.parent.name}: no anchor frames after warmup; skip")
        return None
    anchor_idx = keep
    anchor_t = cam1_t[anchor_idx]

    # Pair each anchor with the nearest ZED-right frame in time.
    r_pos = np.searchsorted(cam2_t, anchor_t)
    r_lo = np.clip(r_pos - 1, 0, len(cam2_t) - 1)
    r_hi = np.clip(r_pos, 0, len(cam2_t) - 1)
    pick_hi = np.abs(cam2_t[r_hi] - anchor_t) < np.abs(cam2_t[r_lo] - anchor_t)
    cam2_pick = np.where(pick_hi, r_hi, r_lo)

    def gather(spec: list[tuple[str, int]]) -> np.ndarray:
        cols = []
        for topic, take in spec:
            s = joint_series[topic]
            if len(s._t) == 0:  # absent topic -> zero-fill
                cols.append(np.zeros((len(anchor_t), take), dtype=np.float32))
                continue
            vecs = s.zoh(anchor_t)  # (T, D_msg)
            if vecs.shape[1] < take:
                pad = np.zeros((vecs.shape[0], take - vecs.shape[1]), dtype=np.float32)
                vecs = np.concatenate([vecs, pad], axis=1)
            cols.append(vecs[:, :take])
        return np.concatenate(cols, axis=1).astype(np.float32)

    q_deg = gather(POS_SPEC)     # (T, 44)
    tau = gather(TRQ_SPEC)       # (T, 44)
    action = gather(CMD_SPEC)    # (T, 44)

    cam1_bytes = [cam1[i][1] for i in anchor_idx]
    cam2_bytes = [cam2[i][1] for i in cam2_pick]

    return {
        "q_deg": q_deg, "tau": tau, "action": action,
        "cam1": cam1_bytes, "cam2": cam2_bytes,
        "cam_format": cam_format,
    }


def make_video(image_paths: list[Path], output_path: Path, fps: int, vcodec: str, crf: int,
               size: tuple[int, int] = IMG_SIZE) -> None:
    """Encode images -> one CFR mp4 scaled to ``size``, forced to EXACTLY len(image_paths) frames.

    Same as convert_raw_to_v3_state_variant.make_video, but adds ``-frames:v N`` so the concat
    demuxer's boundary rounding can't emit an extra tail frame (which would leave the video 1
    frame longer than the parquet). If ffmpeg emits fewer, the caller's shortfall-trim handles it.
    """
    import subprocess
    import tempfile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in image_paths:
            f.write(f"file '{p}'\n")
            f.write(f"duration {1.0 / fps}\n")
        listfile = f.name
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
        "-vf", f"scale={size[0]}:{size[1]}",
        "-c:v", vcodec, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-r", str(fps), "-frames:v", str(len(image_paths)), str(output_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    Path(listfile).unlink()


def _img_ext(fmt: str) -> str:
    f = (fmt or "").lower()
    if "png" in f:
        return ".png"
    return ".jpg"  # ZED image/compressed is jpeg


def convert(input_dir: Path, output_dir: Path, fps: int, task_name: str,
            vcodec: str, crf: int, velocity_cutoff_hz: float,
            limit: int | None, tmp_root: Path) -> None:
    import shutil

    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    tmp_root = Path(tmp_root).expanduser().resolve()

    ep_dirs = sorted(d for d in input_dir.iterdir()
                     if d.is_dir() and d.name.startswith("episode_"))
    if limit is not None:
        ep_dirs = ep_dirs[:limit]
    logger.info(f"Found {len(ep_dirs)} episode dirs (limit={limit}) in {input_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (output_dir / "data" / "chunk-000").mkdir(parents=True)
    frames_root = tmp_root / "frames"
    if frames_root.exists():
        shutil.rmtree(frames_root)
    (frames_root / "camera_1").mkdir(parents=True)
    (frames_root / "camera_2").mkdir(parents=True)

    all_rows: list[dict] = []
    episode_stats_list: list[dict] = []
    episode_metadata_rows: list[dict] = []
    all_images = {"camera_1": [], "camera_2": []}
    video_cumtime = {"observation.images.camera_1": 0.0, "observation.images.camera_2": 0.0}
    total_frames = 0
    kept = 0
    _t0 = _time.time()

    for ep_idx_all, ep_dir in enumerate(ep_dirs):
        mcap_files = sorted((ep_dir / "bag").glob("*.mcap"))
        if not mcap_files:
            logger.warning(f"{ep_dir.name}: no .mcap under bag/; skip")
            continue
        _et0 = _time.time()
        ep = read_episode_mcap(mcap_files[0])
        if ep is None:
            continue
        q_deg, tau, action = ep["q_deg"], ep["tau"], ep["action"]
        T = min(len(q_deg), len(tau), len(action), len(ep["cam1"]), len(ep["cam2"]))
        if T < 2:
            logger.warning(f"{ep_dir.name}: T={T} too short; skip")
            continue
        q_deg, tau, action = q_deg[:T], tau[:T], action[:T]

        # obs_state = [q_rad | dq | tau] (identical transform to convert_raw_to_v3 --incl-vel --incl-trq)
        ts_T = np.arange(T, dtype=np.float64) / float(fps)
        q_rad = (q_deg * DEG2RAD).astype(np.float32)
        dq = np.gradient(q_rad, ts_T, axis=0).astype(np.float32)
        try:
            dq = _butter_lowpass(dq, velocity_cutoff_hz, float(fps)).astype(np.float32)
        except ValueError as e:
            logger.warning(f"{ep_dir.name}: velocity filter skipped: {e}")
        obs_state = np.concatenate([q_rad, dq, tau], axis=1).astype(np.float32)
        assert obs_state.shape[1] == STATE_DIM, obs_state.shape
        assert action.shape[1] == ACTION_DIM, action.shape

        ep_idx = kept
        ext = _img_ext(ep["cam_format"].get("cam1", ""))
        for i in range(T):
            p1 = frames_root / "camera_1" / f"{ep_idx:04d}_{i:06d}{ext}"
            p2 = frames_root / "camera_2" / f"{ep_idx:04d}_{i:06d}{ext}"
            p1.write_bytes(ep["cam1"][i])
            p2.write_bytes(ep["cam2"][i])
            all_images["camera_1"].append(p1)
            all_images["camera_2"].append(p2)

        timestamps = (np.arange(T, dtype=np.float32) / fps)
        for i in range(T):
            all_rows.append({
                "timestamp": float(timestamps[i]),
                "frame_index": i,
                "episode_index": ep_idx,
                "index": total_frames + i,
                "task_index": 0,
                "observation.state": obs_state[i].tolist(),
                "action": action[i].tolist(),
            })

        ep_stats = {
            "observation.state": compute_feature_stats(obs_state),
            "action": compute_feature_stats(action),
        }
        episode_stats_list.append(ep_stats)

        ep_dur = float(T) / fps
        ep_meta = {
            "episode_index": ep_idx,
            "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
            "data/chunk_index": 0, "data/file_index": 0,
            "dataset_from_index": total_frames, "dataset_to_index": total_frames + T,
            "tasks": [task_name], "length": T,
        }
        for cam_key in ["observation.images.camera_1", "observation.images.camera_2"]:
            frm = video_cumtime[cam_key]
            to = frm + ep_dur
            video_cumtime[cam_key] = to
            ep_meta[f"videos/{cam_key}/chunk_index"] = 0
            ep_meta[f"videos/{cam_key}/file_index"] = 0
            ep_meta[f"videos/{cam_key}/from_timestamp"] = frm
            ep_meta[f"videos/{cam_key}/to_timestamp"] = to
        for feat_key, feat_stats in ep_stats.items():
            for stat_key, stat_val in feat_stats.items():
                ep_meta[f"stats/{feat_key}/{stat_key}"] = stat_val
        episode_metadata_rows.append(ep_meta)

        total_frames += T
        kept += 1
        logger.info(
            f"[{ep_idx_all + 1}/{len(ep_dirs)}] {ep_dir.name}: T={T} "
            f"q_deg[min={q_deg.min():.1f},max={q_deg.max():.1f}] "
            f"act[min={action.min():.2f},max={action.max():.2f}] "
            f"({_time.time() - _et0:.1f}s)"
        )

    if kept == 0:
        raise RuntimeError("No episodes converted.")
    logger.info(f"Converted {kept} episodes, {total_frames} frames. Writing parquet…")

    # ── data parquet ──
    table = pa.table({
        "timestamp": pa.array([r["timestamp"] for r in all_rows], type=pa.float32()),
        "frame_index": pa.array([r["frame_index"] for r in all_rows], type=pa.int64()),
        "episode_index": pa.array([r["episode_index"] for r in all_rows], type=pa.int64()),
        "index": pa.array([r["index"] for r in all_rows], type=pa.int64()),
        "task_index": pa.array([r["task_index"] for r in all_rows], type=pa.int64()),
        "observation.state": [r["observation.state"] for r in all_rows],
        "action": [r["action"] for r in all_rows],
    })
    pq.write_table(table, output_dir / "data" / "chunk-000" / "file-000.parquet")

    # ── videos (reuse validated encode + frame-count trim) ──
    actual_counts = {}
    for cam in ["camera_1", "camera_2"]:
        cam_key = f"observation.images.{cam}"
        vdir = output_dir / "videos" / cam_key / "chunk-000"
        vdir.mkdir(parents=True, exist_ok=True)
        vpath = vdir / "file-000.mp4"
        logger.info(f"Encoding {cam}: {len(all_images[cam])} frames -> {vpath}")
        make_video(all_images[cam], vpath, fps, vcodec, crf, size=IMG_SIZE)
        import subprocess
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-count_packets",
                 "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(vpath)],
                capture_output=True, text=True, check=True)
            actual_counts[cam] = int(r.stdout.strip())
        except Exception as e:
            logger.warning(f"ffprobe {cam} failed: {e}")
            actual_counts[cam] = len(all_images[cam])

    if actual_counts:
        min_actual = min(actual_counts.values())
        shortfall = total_frames - min_actual
        if shortfall > 0:
            logger.warning(f"Video frame shortfall {shortfall}; trimming parquet+meta to {min_actual}.")
            dp = output_dir / "data" / "chunk-000" / "file-000.parquet"
            pq.write_table(pq.read_table(dp).slice(0, min_actual), dp)
            last = episode_metadata_rows[-1]
            last["length"] -= shortfall
            last["dataset_to_index"] -= shortfall
            for cam_key in ["observation.images.camera_1", "observation.images.camera_2"]:
                last[f"videos/{cam_key}/to_timestamp"] -= shortfall / fps
            total_frames = min_actual

    # ── meta/info.json (WIR_v1 stamped) ──
    def _img_feat():
        return {
            "dtype": "video", "shape": [3, IMG_SIZE[1], IMG_SIZE[0]],
            "names": ["channel", "height", "width"],
            "video.fps": fps, "video.codec": vcodec, "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False, "has_audio": False,
        }
    info = {
        "codebase_version": "v3.0",
        "wir_data_version": "wir_v1",
        "robot_type": "allex",
        "total_episodes": kept,
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
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": None, "fps": fps},
            "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": None, "fps": fps},
            "observation.images.camera_1": _img_feat(),
            "observation.images.camera_2": _img_feat(),
        },
    }
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # ── meta/stats.json ──
    with open(output_dir / "meta" / "stats.json", "w") as f:
        json.dump(aggregate_stats(episode_stats_list), f, indent=2)

    # ── meta/tasks.parquet ──
    tasks_df = pd.DataFrame({"task_index": [0]}, index=pd.Index([task_name], name="task"))
    pq.write_table(pa.Table.from_pandas(tasks_df), output_dir / "meta" / "tasks.parquet")

    # ── meta/episodes/…parquet ──
    cols: dict[str, pa.Array] = {}
    for col in ["episode_index", "length", "meta/episodes/chunk_index", "meta/episodes/file_index",
                "data/chunk_index", "data/file_index", "dataset_from_index", "dataset_to_index"]:
        cols[col] = pa.array([r[col] for r in episode_metadata_rows], type=pa.int64())
    cols["tasks"] = pa.array([r["tasks"] for r in episode_metadata_rows])
    for cam_key in ["observation.images.camera_1", "observation.images.camera_2"]:
        for suffix in ["chunk_index", "file_index"]:
            c = f"videos/{cam_key}/{suffix}"
            cols[c] = pa.array([r[c] for r in episode_metadata_rows], type=pa.int64())
        for suffix in ["from_timestamp", "to_timestamp"]:
            c = f"videos/{cam_key}/{suffix}"
            cols[c] = pa.array([r[c] for r in episode_metadata_rows], type=pa.float64())
    for feat_key in ["observation.state", "action"]:
        for stat_key in ["mean", "std", "min", "max", "count", "q01", "q10", "q50", "q90", "q99"]:
            c = f"stats/{feat_key}/{stat_key}"
            cols[c] = pa.array([r[c] for r in episode_metadata_rows])
    pq.write_table(pa.table(cols), output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    logger.info(
        f"DONE: {kept} episodes · {total_frames} frames · {STATE_DIM}D obs_state · {ACTION_DIM}D action · "
        f"fps={fps} · {_time.time() - _t0:.0f}s -> {output_dir}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Direct MCAP teleop -> WIR_v1 (LeRobot v3.0) converter")
    p.add_argument("-i", "--input", required=True, help="dir containing episode_*/bag/bag_0.mcap")
    p.add_argument("-o", "--output", required=True, help="output WIR_v1 dataset dir")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--task", default="unspecified")
    p.add_argument("--vcodec", default="libx264")
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--velocity-cutoff-hz", type=float, default=10.0)
    p.add_argument("--limit", type=int, default=None, help="convert only the first N episodes (testing)")
    p.add_argument("--tmp-root", default="/opt/dlami/nvme/mcap_convert_tmp",
                   help="scratch dir for extracted frames before video encode")
    a = p.parse_args()
    convert(Path(a.input), Path(a.output), a.fps, a.task, a.vcodec, a.crf,
            a.velocity_cutoff_hz, a.limit, Path(a.tmp_root))


if __name__ == "__main__":
    main()
