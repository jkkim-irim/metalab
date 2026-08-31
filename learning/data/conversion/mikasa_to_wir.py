#!/usr/bin/env python3
"""Mikasa -> wir_v1: make a Mikasa LeRobot-v3.0 task dataset satisfy the wir_v1 contract.

The Mikasa GEM-VLA export at ``s3://wirobotics-internal/chrisryu/gem_vla/mikasa/lerobot/<task>/`` is
*already* a LeRobot v3.0 + video dataset (data parquet(s) + one or more mp4s per camera — larger
tasks split each camera's video across ``chunk-XXX/file-YYY.mp4`` independently — plus
``meta/{info.json,stats.json,tasks.parquet,episodes/...}``). It is essentially wir_v1 already, but
three things stop it from loading through ``learning/data/`` unchanged:

  1. **Image feature shape.** Its ``observation.images.{top,wrist}`` declare ``shape [128,128,3]``
     (H, W, C). ``learning/data/contract.py::validate_wir_contract`` requires every camera to be
     ``(3, H, W)`` (``shape[0] == 3``) — matching the channel-first tensor the reader actually
     produces (``LeRobotDataset._query_videos`` returns torchcodec CHW frames). We rewrite the
     declared shape to ``(3, H, W)`` and the names to ``["channel","height","width"]``.
  2. **av1 codec.** The videos are AV1. The allex reader decodes through torchcodec (see
     ``learning/data/video_utils.py``) and the allex converter emits **h264/libx264**; AV1 may not
     decode on every node. By default we transcode AV1 -> h264 (yuv420p, constant fps) and update
     ``video.codec`` in ``info.json``. ``--no-transcode`` copies the mp4 verbatim (keeps AV1).
  3. **``wir_data_version``.** Not stamped. ``validate_wir_contract`` asserts it matches ``wir_v1``
     when present and new datasets *should* stamp it, so we write ``"wir_data_version": "wir_v1"``.

Everything else (state/action float32 [7], stats.json already carrying per-feature stats *including*
both cameras at ``(3,1,1)``, tasks.parquet, episodes parquet, the v3.0 bookkeeping columns, fps=10)
is copied through unchanged.

Default (keep ALL episodes): a *structure-preserving* copy — ``data/`` and ``meta/`` are copied
verbatim (every per-episode data/video pointer intact) and each video file is transcoded in place, so
whatever chunk/file split the source uses (single- or multi-file) is mirrored exactly.

Subsetting (``--max-episodes N``, single-file sources only): because the data and video are single
consolidated files, a subset keeps the *first N episodes* by (a) slicing the data parquet to the first N episodes' frames,
(b) slicing the episodes parquet to the first N rows, (c) fixing ``total_episodes`` / ``total_frames``
/ ``splits`` in info.json. The consolidated video is transcoded/copied **whole** — episodes 0..N-1
occupy the prefix ``[0, to_timestamp[N-1])`` of the video and the reader only ever queries those
timestamps via each episode's ``videos/<key>/from_timestamp``; the unreferenced tail frames are
simply never decoded. This is correct without re-slicing the video and avoids frame-count drift.

Pulling the source: ``--src`` is a **local directory** (a task dataset root, i.e. contains
``meta/info.json``, or a parent directory containing ``<task>/``). Sync it first with, e.g.::

    aws s3 sync s3://wirobotics-internal/chrisryu/gem_vla/mikasa/lerobot/shell_game_touch_vla_v0 \
        /scratch/mikasa/shell_game_touch_vla_v0

An ``s3://`` ``--src`` is also accepted and synced automatically (requires the ``aws`` CLI).

CLI (single task)::

    python -m learning.data.conversion.mikasa_to_wir \
        --src /scratch/mikasa/shell_game_touch_vla_v0 --out /scratch/mikasa_wir/shell_game_touch \
        --task shell_game_touch_vla_v0            # all episodes; add --max-episodes N for a smoke subset

CLI (full suite — every task under a parent dir / s3 prefix, one wir_v1 dataset per task)::

    python -m learning.data.conversion.mikasa_to_wir \
        --src s3://wirobotics-internal/chrisryu/gem_vla/mikasa/lerobot \
        --out /scratch/mikasa_wir --all-tasks    # writes /scratch/mikasa_wir/<task>/; --workers = all CPUs

Validate the result (metadata + contract, no video decode needed)::

    python -c "from learning.data.lerobot_dataset import LeRobotDatasetMetadata; \
        from learning.data.contract import validate_wir_contract; \
        m = LeRobotDatasetMetadata('mikasa/shell_game_touch', root='/scratch/mikasa_wir/shell_game_touch'); \
        validate_wir_contract(m); print('OK', m.total_episodes, m.total_frames, m.fps)"
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from learning.data.contract import WIR_DATA_VERSION

# The video keys we expect (and the only camera dtypes we touch).
_IMAGE_DTYPES = {"video", "image"}
DEFAULT_TASK = "shell_game_touch_vla_v0"


# --------------------------------------------------------------------------------------------------
# Source resolution
# --------------------------------------------------------------------------------------------------
def resolve_source(src: str, task: str, out_dir: Path) -> Path:
    """Return the local task-dataset root (contains ``meta/info.json``), syncing from S3 if needed.

    ``src`` may be: the task dataset root itself, a parent dir containing ``<task>/``, or an
    ``s3://`` URI of either. An ``s3://`` source is synced into ``<out parent>/.mikasa_raw_src/<task>``
    (a reusable, incremental cache). Fails loud if no ``meta/info.json`` can be found.
    """
    if src.startswith("s3://"):
        cache = out_dir.parent / ".mikasa_raw_src" / task
        cache.mkdir(parents=True, exist_ok=True)
        # If the URI already points at a task root (ends with the task name) sync it directly,
        # otherwise sync the task subdir under it.
        s3_root = src.rstrip("/")
        if not s3_root.endswith(task):
            s3_root = f"{s3_root}/{task}"
        print(f"[mikasa_to_wir] aws s3 sync {s3_root} -> {cache}", flush=True)
        subprocess.run(["aws", "s3", "sync", s3_root, str(cache)], check=True)
        local = cache
    else:
        local = Path(src).expanduser().resolve()

    if (local / "meta" / "info.json").exists():
        return local
    candidate = local / task
    if (candidate / "meta" / "info.json").exists():
        return candidate
    raise FileNotFoundError(
        f"No LeRobot v3.0 dataset found: neither {local}/meta/info.json nor "
        f"{candidate}/meta/info.json exists. Point --src at the task dataset root or its parent."
    )


def list_task_names(src: str) -> list[str]:
    """Enumerate the MIKASA task-dataset names under a parent location (local dir or ``s3://`` URI).

    Each MIKASA task is its own LeRobot v3.0 dataset in a ``<task>/`` subdirectory. For a local parent
    we require ``<task>/meta/info.json`` to exist; for an ``s3://`` parent we list the immediate
    prefixes (each task's sync + contract check then happens per-task inside ``convert``). Returns the
    names sorted; fails loud if none are found.
    """
    if src.startswith("s3://"):
        base = src.rstrip("/") + "/"
        out = subprocess.run(["aws", "s3", "ls", base], capture_output=True, text=True, check=True)
        candidates = [ln.split()[-1].rstrip("/") for ln in out.stdout.splitlines()
                      if ln.strip().startswith("PRE")]
        # Keep only prefixes that are actually a LeRobot dataset (have meta/info.json) — drops helper
        # prefixes like ``.cache/`` that aren't tasks (parity with the local-dir branch below).
        names = []
        for c in candidates:
            probe = subprocess.run(["aws", "s3", "ls", f"{base}{c}/meta/info.json"],
                                   capture_output=True, text=True)
            if probe.stdout.strip():
                names.append(c)
    else:
        parent = Path(src).expanduser().resolve()
        names = [d.name for d in parent.iterdir() if (d / "meta" / "info.json").exists()]
    names = sorted(names)
    if not names:
        raise FileNotFoundError(f"no MIKASA task datasets found under {src!r}")
    return names


# --------------------------------------------------------------------------------------------------
# info.json rewrite
# --------------------------------------------------------------------------------------------------
def fix_image_feature_shape(feature: dict, key: str) -> None:
    """Rewrite an (H, W, C) camera feature shape to channel-first (C, H, W) in place.

    ``validate_wir_contract`` requires camera shape ``(3, H, W)``; the Mikasa export declares
    ``(H, W, 3)``. Fails loud if the shape is not a recognisable 3-channel image shape.
    """
    shape = list(feature["shape"])
    if len(shape) != 3:
        raise ValueError(f"camera {key!r} shape {shape} is not 3-D; cannot make channel-first")
    if shape[0] == 3:
        return  # already channel-first
    if shape[2] != 3:
        raise ValueError(
            f"camera {key!r} shape {shape} is neither (3,H,W) nor (H,W,3); refusing to guess"
        )
    h, w, c = shape
    feature["shape"] = [c, h, w]
    feature["names"] = ["channel", "height", "width"]


def build_info(
    src_info: dict, num_episodes: int, num_frames: int, num_tasks: int, transcode: bool, vcodec: str
) -> dict:
    """Return the rewritten ``info.json`` dict for the (possibly subset) wir_v1 dataset."""
    info = json.loads(json.dumps(src_info))  # deep copy

    for key, feature in info["features"].items():
        if feature.get("dtype") in _IMAGE_DTYPES:
            fix_image_feature_shape(feature, key)
            if transcode:
                # We re-encode to h264; reflect it so consumers see the real codec.
                nested = feature.get("info")
                if isinstance(nested, dict) and "video.codec" in nested:
                    nested["video.codec"] = vcodec
                    nested["video.pix_fmt"] = "yuv420p"

    info["total_episodes"] = num_episodes
    info["total_frames"] = num_frames
    info["total_tasks"] = num_tasks
    info["splits"] = {"train": f"0:{num_episodes}"}
    info["wir_data_version"] = WIR_DATA_VERSION
    return info


# --------------------------------------------------------------------------------------------------
# Parquet helpers
# --------------------------------------------------------------------------------------------------
def _sorted_parquet(pq_dir: Path) -> list[Path]:
    paths = sorted(pq_dir.glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files under {pq_dir}")
    return paths


def load_concat(pq_dir: Path) -> pa.Table:
    """Read all nested ``chunk-*/file-*.parquet`` in sorted order and concatenate (reader order)."""
    tables = [pq.read_table(str(p)) for p in _sorted_parquet(pq_dir)]
    return pa.concat_tables(tables, promote_options="default") if len(tables) > 1 else tables[0]


def assert_single_file_indices(ep_table: pa.Table, num_episodes: int) -> None:
    """Assert the subset episodes all live in chunk 0 / file 0 for data and every video key.

    We write a single consolidated ``file-000.parquet`` / ``file-000.mp4`` per stream, so a subset is
    only self-consistent if the source kept everything in chunk 0 / file 0 (true for the Mikasa
    export). Fail loud rather than silently emit a broken index.
    """
    cols = ep_table.column_names
    idx_cols = [c for c in cols if c.endswith("chunk_index") or c.endswith("file_index")]
    head = ep_table.slice(0, num_episodes)
    for c in idx_cols:
        vals = pc.unique(head.column(c)).to_pylist()
        if vals != [0]:
            raise ValueError(
                f"episode column {c!r} has values {vals} in the first {num_episodes} episodes; "
                "this adapter only supports a single-file (chunk 0/file 0) source layout"
            )


# --------------------------------------------------------------------------------------------------
# Video transcode / copy
# --------------------------------------------------------------------------------------------------
def _ffprobe_nframes(path: Path) -> int | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        )
        return int(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def transcode_video(src_mp4: Path, dst_mp4: Path, fps: int, vcodec: str, crf: int,
                    min_frames: int) -> None:
    """Transcode ``src_mp4`` -> ``dst_mp4`` at constant ``fps`` with ``vcodec``/``crf``, yuv420p.

    Forces CFR so torchcodec frame i lands at pts i/fps (matching the reader's
    ``round(ts * average_fps)`` indexing and the allex converter's output). Fails loud if the encoded
    video ends up with fewer than ``min_frames`` frames (the subset would be missing frames).
    """
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src_mp4),
        "-an", "-c:v", vcodec, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-r", str(fps), str(dst_mp4),
    ]
    subprocess.run(cmd, check=True)
    n = _ffprobe_nframes(dst_mp4)
    if n is not None and n < min_frames:
        raise RuntimeError(
            f"transcoded {dst_mp4} has {n} frames < {min_frames} required by the subset "
            f"(source={src_mp4}); refusing to emit a video missing frames"
        )


# --------------------------------------------------------------------------------------------------
# Main conversion
# --------------------------------------------------------------------------------------------------
def _convert_full(root: Path, out_dir: Path, src_info: dict, task: str, transcode: bool,
                  vcodec: str, crf: int) -> None:
    """Full (keep-ALL) conversion: copy ``data/`` + ``meta/`` verbatim (preserving the chunk/file
    layout and every per-episode data/video pointer), transcode each video file in place, rewrite
    ``info.json``.

    Unlike the subset path this makes NO single-file assumption — it mirrors whatever chunk/file split
    the source uses (MIKASA splits large per-camera videos across ``file-000.mp4``, ``-001``, …, and
    the two cameras can split differently), transcoding each file to h264 at its own relative path so
    the episodes' per-camera ``(chunk_index, file_index, from_timestamp)`` pointers stay valid (frame
    count + CFR preserved per file, so ``round(ts*fps)`` still lands on the same frame).
    """
    fps = int(src_info["fps"])
    num_episodes = int(src_info["total_episodes"])
    num_frames = int(src_info["total_frames"])
    num_tasks = pq.read_table(str(root / "meta" / "tasks.parquet")).num_rows
    info = build_info(src_info, num_episodes, num_frames, num_tasks, transcode, vcodec)

    # meta + data: copy the whole trees verbatim (episode/data pointers unchanged), then rewrite info.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shutil.copytree(root / "meta", out_dir / "meta")
    shutil.copytree(root / "data", out_dir / "data")
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    # videos: transcode (or copy) EVERY file under each camera, at its original relative path.
    video_keys = [k for k, ft in info["features"].items() if ft.get("dtype") == "video"]
    n_files = 0
    for key in video_keys:
        for src_mp4 in sorted((root / "videos" / key).rglob("*.mp4")):
            dst_mp4 = out_dir / src_mp4.relative_to(root)
            if transcode:
                src_n = _ffprobe_nframes(src_mp4)
                transcode_video(src_mp4, dst_mp4, fps, vcodec, crf, min_frames=src_n or 0)
            else:
                dst_mp4.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src_mp4, dst_mp4)
            n_files += 1
    if not video_keys or n_files == 0:
        raise FileNotFoundError(f"no video files found under {root / 'videos'} for task {task!r}")

    codec = info["features"][video_keys[0]]["info"]["video.codec"]
    print(f"[mikasa_to_wir] DONE task={task} episodes={num_episodes} frames={num_frames} fps={fps} "
          f"cameras={video_keys} video_files={n_files} codec={codec} -> {out_dir}", flush=True)


def convert(src: str, out: str, task: str, max_episodes: int, transcode: bool,
            vcodec: str, crf: int) -> None:
    out_dir = Path(out).expanduser().resolve()
    root = resolve_source(src, task, out_dir)
    print(f"[mikasa_to_wir] source={root}\n[mikasa_to_wir] out={out_dir}", flush=True)

    src_info = json.loads((root / "meta" / "info.json").read_text())
    if str(src_info.get("codebase_version", "")).lstrip("v").split(".")[0] != "3":
        raise ValueError(f"source is not LeRobot v3.x: codebase_version={src_info.get('codebase_version')!r}")
    fps = int(src_info["fps"])
    total_src_episodes = int(src_info["total_episodes"])

    if max_episodes <= 0:
        # Keep ALL episodes: structure-preserving copy + per-file transcode. Works for any chunk/file
        # layout (single- OR multi-file); the subset path below requires a single-file source.
        _convert_full(root, out_dir, src_info, task, transcode, vcodec, crf)
        return

    # ---- episodes: decide the subset ----
    ep_table = load_concat(root / "meta" / "episodes")
    if len(ep_table) != total_src_episodes:
        raise ValueError(
            f"episodes parquet rows {len(ep_table)} != info total_episodes {total_src_episodes}"
        )
    num_episodes = total_src_episodes if max_episodes <= 0 else min(max_episodes, total_src_episodes)
    assert_single_file_indices(ep_table, num_episodes)

    ep_subset = ep_table.slice(0, num_episodes)
    # Frames for the first N episodes = dataset_to_index of the last kept episode.
    num_frames = int(ep_subset.column("dataset_to_index")[num_episodes - 1].as_py())

    # ---- data: slice to the first N episodes' frames ----
    data_table = load_concat(root / "data")
    if len(data_table) != int(src_info["total_frames"]):
        raise ValueError(
            f"data parquet rows {len(data_table)} != info total_frames {src_info['total_frames']}"
        )
    data_subset = data_table.slice(0, num_frames)
    # The first `num_frames` rows must be exactly episodes 0..N-1, in global index order.
    ep_idx_vals = pc.unique(data_subset.column("episode_index")).to_pylist()
    if max(ep_idx_vals) >= num_episodes or min(ep_idx_vals) < 0:
        raise ValueError(
            f"first {num_frames} data rows span episode_index {sorted(ep_idx_vals)[:3]}..."
            f"{sorted(ep_idx_vals)[-1]}, expected 0..{num_episodes - 1}; source is not "
            "episode-contiguous in global index order"
        )
    first_index = int(data_subset.column("index")[0].as_py())
    last_index = int(data_subset.column("index")[num_frames - 1].as_py())
    if first_index != 0 or last_index != num_frames - 1:
        raise ValueError(
            f"data 'index' column is not 0..{num_frames - 1} (got {first_index}..{last_index}); "
            "global index does not match row position"
        )

    # ---- write meta ----
    (out_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    tasks_table = pq.read_table(str(root / "meta" / "tasks.parquet"))
    num_tasks = tasks_table.num_rows
    info = build_info(src_info, num_episodes, num_frames, num_tasks, transcode, vcodec)
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    # stats.json already carries per-feature stats for state/action AND both cameras (3,1,1);
    # they are full-dataset stats, which is exactly what normalization wants -> copy verbatim.
    shutil.copyfile(root / "meta" / "stats.json", out_dir / "meta" / "stats.json")
    # tasks.parquet: single task, unchanged by subsetting -> copy verbatim (preserves pandas index).
    shutil.copyfile(root / "meta" / "tasks.parquet", out_dir / "meta" / "tasks.parquet")
    # optional subtasks.parquet
    if (root / "meta" / "subtasks.parquet").exists():
        shutil.copyfile(root / "meta" / "subtasks.parquet", out_dir / "meta" / "subtasks.parquet")

    pq.write_table(ep_subset, out_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pq.write_table(data_subset, out_dir / "data" / "chunk-000" / "file-000.parquet")

    # ---- videos ----
    video_keys = [k for k, ft in info["features"].items() if ft.get("dtype") == "video"]
    video_path_tmpl = src_info["video_path"]
    for key in video_keys:
        # Source consolidated video (chunk 0 / file 0, asserted above).
        rel = video_path_tmpl.format(video_key=key, chunk_index=0, file_index=0)
        src_mp4 = root / rel
        dst_mp4 = out_dir / rel
        if not src_mp4.exists():
            raise FileNotFoundError(f"expected source video missing: {src_mp4}")
        if transcode:
            print(f"[mikasa_to_wir] transcode {key}: {src_mp4.name} -> {vcodec}", flush=True)
            transcode_video(src_mp4, dst_mp4, fps, vcodec, crf, min_frames=num_frames)
        else:
            dst_mp4.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_mp4, dst_mp4)

    codec = info["features"][video_keys[0]]["info"]["video.codec"] if video_keys else "none"
    print(
        f"[mikasa_to_wir] DONE task={task} episodes={num_episodes}/{total_src_episodes} "
        f"frames={num_frames} fps={fps} cameras={video_keys} codec={codec} -> {out_dir}",
        flush=True,
    )


def _convert_one(job: dict) -> dict:
    """Pool worker: convert one MIKASA task to ``<out_parent>/<task>/``. Returns a small summary dict.

    Module-level (picklable) so it works with ``ProcessPoolExecutor``; each task is an independent
    dataset, so the tasks parallelise cleanly with no shared state.
    """
    convert(
        src=job["src"], out=job["out"], task=job["task"], max_episodes=job["max_episodes"],
        transcode=job["transcode"], vcodec=job["vcodec"], crf=job["crf"],
    )
    return {"task": job["task"], "out": job["out"]}


def convert_all(src: str, out: str, max_episodes: int, transcode: bool, vcodec: str, crf: int,
                workers: int) -> None:
    """Convert EVERY MIKASA task under ``src`` (a parent dir / ``s3://`` prefix) to ``<out>/<task>/``.

    The tasks are independent datasets, so they run concurrently across ``workers`` processes.
    """
    out_parent = Path(out).expanduser().resolve()
    tasks = list_task_names(src)
    print(f"[mikasa_to_wir] all-tasks: {len(tasks)} tasks, workers={workers} -> {out_parent}",
          flush=True)
    src_parent = src.rstrip("/")
    jobs = [
        {"src": src_parent, "out": str(out_parent / t), "task": t, "max_episodes": max_episodes,
         "transcode": transcode, "vcodec": vcodec, "crf": crf}
        for t in tasks
    ]
    ok: list[str] = []
    failed: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_convert_one, j): j["task"] for j in jobs}
        for fut in as_completed(futs):
            t = futs[fut]
            n = len(ok) + len(failed) + 1
            try:
                fut.result()
                ok.append(t)
                print(f"[mikasa_to_wir] [{n}/{len(tasks)}] OK task={t}", flush=True)
            except Exception as e:  # boundary: one bad task must not abort the other 88
                failed.append((t, repr(e)))
                print(f"[mikasa_to_wir] [{n}/{len(tasks)}] FAIL task={t}: {e}", flush=True)
    print(f"[mikasa_to_wir] ALL DONE: {len(ok)}/{len(tasks)} ok, {len(failed)} failed -> {out_parent}",
          flush=True)
    if failed:
        for t, e in failed:
            print(f"[mikasa_to_wir]   FAILED {t}: {e}", flush=True)
        raise RuntimeError(f"{len(failed)}/{len(tasks)} MIKASA tasks failed to convert")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Adapt a Mikasa LeRobot v3.0 task dataset into a wir_v1 dataset (subset-capable)."
    )
    p.add_argument("--src", required=True,
                   help="Task-dataset root (has meta/info.json) or parent dir (for --all-tasks), "
                        "local path or s3:// URI.")
    p.add_argument("--out", required=True,
                   help="Output wir_v1 dataset dir (single task) or parent dir (--all-tasks: writes "
                        "<out>/<task>/ per task).")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--task", default=DEFAULT_TASK, help=f"Single task name (default: {DEFAULT_TASK}).")
    grp.add_argument("--all-tasks", action="store_true",
                     help="Convert EVERY task dataset under --src (a parent dir / s3 prefix), each to "
                          "<out>/<task>/. Tasks run in parallel (--workers).")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                   help="Parallel task conversions for --all-tasks (default: all CPUs).")
    p.add_argument("--max-episodes", type=int, default=0,
                   help="Keep only the first N episodes (<=0 = keep ALL, the default). Set >0 only to "
                        "make a quick smoke subset.")
    p.add_argument("--no-transcode", dest="transcode", action="store_false",
                   help="Copy source mp4 verbatim (keep AV1) instead of transcoding to h264.")
    p.add_argument("--vcodec", default="libx264", help="Transcode video codec (default libx264).")
    p.add_argument("--crf", type=int, default=23, help="Transcode CRF (default 23).")
    p.set_defaults(transcode=True)
    args = p.parse_args()

    if args.all_tasks:
        convert_all(
            src=args.src, out=args.out, max_episodes=args.max_episodes, transcode=args.transcode,
            vcodec=args.vcodec, crf=args.crf, workers=args.workers,
        )
        return

    convert(
        src=args.src, out=args.out, task=args.task, max_episodes=args.max_episodes,
        transcode=args.transcode, vcodec=args.vcodec, crf=args.crf,
    )


if __name__ == "__main__":
    main()
