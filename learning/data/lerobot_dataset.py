"""LeRobotDataset (v3.0) — lerobot-free reimplementation of the read path the ALLEX trainer uses.

This reproduces, without importing ``lerobot`` or HuggingFace ``datasets``, the subset of
``lerobot/datasets/lerobot_dataset.py`` (0.4.4, codebase version ``v3.0``) that the trainer touches:

  * ``LeRobotDatasetMetadata`` — reads ``meta/info.json``, ``meta/stats.json``,
    ``meta/tasks.parquet``, ``meta/subtasks.parquet`` (optional) and the nested
    ``meta/episodes/chunk-*/file-*.parquet``.
  * ``LeRobotDataset`` — loads the nested ``data/chunk-*/file-*.parquet`` frames, assembles
    ``delta_timestamps`` chunks (with ``*_is_pad`` masks), decodes video frames via torchcodec, and
    applies optional image transforms — matching lerobot's ``__getitem__`` field-for-field.

Storage backend deviation (documented, behaviourally inert): lerobot loads frames and episode
metadata through ``datasets.Dataset`` (a memory-mapped pyarrow wrapper) with a ``set_transform``
(``hf_transform_to_torch``) that converts each accessed batch to torch tensors. ``datasets`` is not
in the allowed dependency set, so we read the parquet files directly with pyarrow and apply the
*same* per-value conversion.

Critical dtype faithfulness note: HF ``datasets`` with the default (python) formatter feeds the
transform values produced by pyarrow ``Array.to_pylist()`` — i.e. *python* objects (python
``float`` / ``int`` / nested lists), not numpy. ``hf_transform_to_torch`` then does
``torch.tensor(x)``. So a ``float32`` parquet column ends up as a torch **float64** tensor (python
float -> float64), and an ``int64`` column as int64. We replicate this exactly by calling
``to_pylist()`` on each pyarrow column and feeding the python objects to ``torch.tensor`` — never
routing numeric frame data through pandas/numpy (which would silently produce float32 and break
equivalence). Frames are concatenated in sorted ``chunk-*/file-*`` parquet order, the same order
``datasets.Dataset.from_parquet`` uses, so the global ``index`` column lines up 1:1 with row
position.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from learning.data.video_utils import decode_video_frames, get_safe_default_codec

CODEBASE_VERSION = "v3.0"

INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"
EPISODES_DIR = "meta/episodes"
DATA_DIR = "data"
DEFAULT_TASKS_PATH = "meta/tasks.parquet"
DEFAULT_SUBTASKS_PATH = "meta/subtasks.parquet"

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}


# --------------------------------------------------------------------------------------------------
# Loading helpers (mirror lerobot.datasets.utils)
# --------------------------------------------------------------------------------------------------
def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dict by joining keys with ``sep`` (mirror of lerobot.utils.flatten_dict)."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    """Unflatten a dict with delimited keys (mirror of lerobot.utils.unflatten_dict)."""
    outdict: dict = {}
    for key, value in d.items():
        parts = key.split(sep)
        cur = outdict
        for part in parts[:-1]:
            if part not in cur:
                cur[part] = {}
            cur = cur[part]
        cur[parts[-1]] = value
    return outdict


def load_info(local_dir: Path) -> dict:
    with open(local_dir / INFO_PATH) as f:
        info = json.load(f)
    for ft in info["features"].values():
        ft["shape"] = tuple(ft["shape"])
    return info


def cast_stats_to_numpy(stats: dict) -> dict[str, dict[str, np.ndarray]]:
    stats = {key: np.array(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)


def load_stats(local_dir: Path) -> dict[str, dict[str, np.ndarray]] | None:
    if not (local_dir / STATS_PATH).exists():
        return None
    with open(local_dir / STATS_PATH) as f:
        stats = json.load(f)
    return cast_stats_to_numpy(stats)


def load_tasks(local_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(local_dir / DEFAULT_TASKS_PATH)


def load_subtasks(local_dir: Path) -> pd.DataFrame | None:
    subtasks_path = local_dir / DEFAULT_SUBTASKS_PATH
    if subtasks_path.exists():
        return pd.read_parquet(subtasks_path)
    return None


def _sorted_parquet_paths(pq_dir: Path) -> list[Path]:
    """Find nested parquet files {pq_dir}/chunk-xxx/file-xxx.parquet in sorted (load) order."""
    paths = sorted(pq_dir.glob("*/*.parquet"))
    if len(paths) == 0:
        raise FileNotFoundError(f"Provided directory does not contain any parquet file: {pq_dir}")
    return paths


def check_delta_timestamps(
    delta_timestamps: dict[str, list[float]], fps: int, tolerance_s: float
) -> bool:
    """Check delta timestamps are multiples of 1/fps +/- tolerance (mirror of lerobot.utils)."""
    outside_tolerance = {}
    for key, delta_ts in delta_timestamps.items():
        within_tolerance = [abs(ts * fps - round(ts * fps)) / fps <= tolerance_s for ts in delta_ts]
        if not all(within_tolerance):
            outside_tolerance[key] = [
                ts for ts, ok in zip(delta_ts, within_tolerance, strict=True) if not ok
            ]
    if len(outside_tolerance) > 0:
        raise ValueError(
            "The following delta_timestamps are found outside of tolerance range. Please make sure "
            f"they are multiples of 1/{fps} +/- tolerance and adjust their values accordingly.\n"
            f"{outside_tolerance}"
        )
    return True


def get_delta_indices(delta_timestamps: dict[str, list[float]], fps: int) -> dict[str, list[int]]:
    delta_indices = {}
    for key, delta_ts in delta_timestamps.items():
        delta_indices[key] = [round(d * fps) for d in delta_ts]
    return delta_indices


def _pyobj_to_torch(value):
    """Convert one ``to_pylist()`` cell to torch, mirroring ``hf_transform_to_torch`` exactly.

    The input is a *python* object (the pyarrow python formatter output): a python scalar for
    ``Value`` columns, a (possibly nested) python list for ``Sequence``/``ArrayND`` columns, a str
    for string columns, or ``None``. ``hf_transform_to_torch`` runs ``torch.tensor(x)`` on
    non-string / non-None values, so python ``float`` -> float64 and python ``int`` -> int64, which
    is what lerobot produces. Strings and ``None`` pass through unchanged.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return torch.tensor(value)


# --------------------------------------------------------------------------------------------------
# Episodes table — replicates the column- and row-access surface of the HF episodes Dataset
# --------------------------------------------------------------------------------------------------
class EpisodesTable:
    """Episode metadata exposing both column access (``episodes["dataset_from_index"]``) and row
    access (``episodes[ep_idx]`` -> dict), matching the HF ``datasets.Dataset`` surface that lerobot
    relies on. Built by concatenating the nested ``meta/episodes/*/*.parquet`` files in sorted order
    and dropping ``stats/*`` columns (as lerobot's ``load_episodes`` does)."""

    def __init__(self, df: pd.DataFrame):
        self._df = df.reset_index(drop=True)

    @classmethod
    def load(cls, local_dir: Path) -> "EpisodesTable":
        paths = _sorted_parquet_paths(local_dir / EPISODES_DIR)
        frames = [pd.read_parquet(p) for p in paths]
        df = pd.concat(frames, ignore_index=True)
        # Drop stats/* columns to mirror lerobot.load_episodes (speeds up access).
        keep = [c for c in df.columns if not c.startswith("stats/")]
        return cls(df[keep])

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, key):
        # Column access: episodes["dataset_from_index"] -> list[int]
        if isinstance(key, str):
            return self._df[key].tolist()
        # Row access: episodes[ep_idx] -> dict (python scalars, matching HF row dicts)
        if isinstance(key, (int, np.integer)):
            row = self._df.iloc[int(key)]
            return {col: _to_python_scalar(row[col]) for col in self._df.columns}
        raise TypeError(f"Unsupported episodes index type: {type(key)}")

    @property
    def columns(self) -> list[str]:
        return list(self._df.columns)


def _to_python_scalar(v):
    if isinstance(v, np.generic):
        return v.item()
    return v


# --------------------------------------------------------------------------------------------------
# Frame table — replicates the row / column / list access surface of the HF frames Dataset
# --------------------------------------------------------------------------------------------------
class HFFramesTable:
    """Frame data exposing the access patterns lerobot's ``__getitem__`` uses against
    ``self.hf_dataset``:

      * ``table[idx]`` (int) -> dict of single torch tensors / strings (one row),
      * ``table[key]`` (str) -> column object supporting ``column[list_of_indices]`` -> list,
      * ``table[list_of_indices]`` -> dict of lists,
      * ``table["index"]`` iteration and ``table.unique("episode_index")``.

    Backed by a single pyarrow Table. Per-cell python objects come from ``to_pylist()`` so the torch
    conversion matches HF's python formatter + ``hf_transform_to_torch`` (see module docstring).
    Video feature columns are absent from the parquet (lerobot stores only path/frame metadata for
    them and excludes them from the hf features), so they never appear here.
    """

    def __init__(self, table: pa.Table, video_keys: list[str]):
        drop = [c for c in table.column_names if c in video_keys]
        if drop:
            table = table.drop(drop)
        self._table = table
        self._columns = table.column_names
        # Cache python-list views per column lazily (to_pylist once, reuse on repeated access).
        self._pylists: dict[str, list] = {}

    @classmethod
    def load(cls, local_dir: Path, video_keys: list[str]) -> "HFFramesTable":
        paths = _sorted_parquet_paths(local_dir / DATA_DIR)
        tables = [pq.read_table(str(p)) for p in paths]
        table = pa.concat_tables(tables, promote_options="default")
        return cls(table, video_keys)

    def _col_pylist(self, col: str) -> list:
        cached = self._pylists.get(col)
        if cached is None:
            cached = self._table.column(col).to_pylist()
            self._pylists[col] = cached
        return cached

    def __len__(self) -> int:
        return self._table.num_rows

    def __getitem__(self, key):
        if isinstance(key, str):
            return _FrameColumn(self._col_pylist(key))
        if isinstance(key, (int, np.integer)):
            i = int(key)
            return {col: _pyobj_to_torch(self._col_pylist(col)[i]) for col in self._columns}
        if isinstance(key, (list, tuple, np.ndarray)):
            idxs = [int(i) for i in key]
            out = {}
            for col in self._columns:
                pylist = self._col_pylist(col)
                out[col] = [_pyobj_to_torch(pylist[i]) for i in idxs]
            return out
        raise TypeError(f"Unsupported frames index type: {type(key)}")

    def unique(self, column: str) -> list:
        return pc.unique(self._table.column(column)).to_pylist()

    @property
    def columns(self) -> list[str]:
        return list(self._columns)


class _FrameColumn:
    """A single frame column supporting ``column[list_of_indices] -> list[torch.Tensor]`` and
    iteration, matching ``hf_dataset[key]`` followed by ``[indices]`` in lerobot."""

    def __init__(self, pylist: list):
        self._pylist = pylist

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple, np.ndarray)):
            return [_pyobj_to_torch(self._pylist[int(i)]) for i in idx]
        if isinstance(idx, (int, np.integer)):
            return _pyobj_to_torch(self._pylist[int(idx)])
        raise TypeError(f"Unsupported column index type: {type(idx)}")

    def __iter__(self):
        for v in self._pylist:
            yield _pyobj_to_torch(v)

    def __len__(self) -> int:
        return len(self._pylist)


# --------------------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------------------
class LeRobotDatasetMetadata:
    """Read-only metadata for a local v3.0 LeRobot dataset (mirror of lerobot's class)."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
    ):
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        if root is None:
            raise ValueError(
                "This lerobot-free reimplementation only supports local datasets; pass `root`."
            )
        self.root = Path(root)
        self.load_metadata()

    def load_metadata(self):
        self.info = load_info(self.root)
        _check_version_compatibility(self.info["codebase_version"])
        self.tasks = load_tasks(self.root)
        self.subtasks = load_subtasks(self.root)
        self.episodes = EpisodesTable.load(self.root)
        self.stats = load_stats(self.root)

    @property
    def fps(self) -> int:
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def total_episodes(self) -> int:
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        return self.info["total_frames"]

    @property
    def data_path(self) -> str:
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        return self.info["video_path"]

    def get_data_file_path(self, ep_index: int) -> Path:
        ep = self.episodes[ep_index]
        chunk_idx = ep["data/chunk_index"]
        file_idx = ep["data/file_index"]
        return Path(self.data_path.format(chunk_index=chunk_idx, file_index=file_idx))

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        ep = self.episodes[ep_index]
        chunk_idx = ep[f"videos/{vid_key}/chunk_index"]
        file_idx = ep[f"videos/{vid_key}/file_index"]
        return Path(
            self.video_path.format(video_key=vid_key, chunk_index=chunk_idx, file_index=file_idx)
        )


def _check_version_compatibility(codebase_version: str) -> None:
    major = str(codebase_version).lstrip("v").split(".")[0]
    expected_major = CODEBASE_VERSION.lstrip("v").split(".")[0]
    if major != expected_major:
        raise ValueError(
            f"Dataset codebase version '{codebase_version}' is incompatible with this "
            f"reimplementation, which targets '{CODEBASE_VERSION}'."
        )


# --------------------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------------------
class LeRobotDataset(torch.utils.data.Dataset):
    """Read-only v3.0 LeRobotDataset reimplementation (local datasets only).

    Faithful to lerobot 0.4.4's ``__getitem__``: delta_timestamps chunk assembly + padding, video
    frame query/decode through torchcodec, and optional image transforms (applied per camera key, in
    ``camera_keys`` order, after video decode — matching lerobot)."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms=None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        video_backend: str | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        if root is None:
            raise ValueError(
                "This lerobot-free reimplementation only supports local datasets; pass `root`."
            )
        self.root = Path(root)
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None

        self.meta = LeRobotDatasetMetadata(self.repo_id, self.root, self.revision)

        self.hf_dataset = self.load_hf_dataset()

        # Map absolute global index -> relative row position when only a subset of episodes loads.
        self._absolute_to_relative_idx = None
        if self.episodes is not None:
            self._absolute_to_relative_idx = {
                int(abs_idx.item()): rel_idx
                for rel_idx, abs_idx in enumerate(self.hf_dataset["index"])
            }

        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def load_hf_dataset(self) -> HFFramesTable:
        if self.episodes is not None:
            table = _load_filtered_frames(self.root, self.episodes)
            return HFFramesTable(table, self.meta.video_keys)
        return HFFramesTable.load(self.root, self.meta.video_keys)

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        if self.episodes is not None and self.hf_dataset is not None:
            return len(self.hf_dataset)
        return self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    def __len__(self):
        return self.num_frames

    def _get_query_indices(self, abs_idx, ep_idx):
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(self, current_ts, query_indices=None):
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                if self._absolute_to_relative_idx is not None:
                    relative_indices = [
                        self._absolute_to_relative_idx[idx] for idx in query_indices[key]
                    ]
                    timestamps = self.hf_dataset[relative_indices]["timestamp"]
                else:
                    timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def _query_hf_dataset(self, query_indices):
        result = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            # column-then-indices, matching lerobot's hf_dataset[key][indices] (see HFFramesTable /
            # _FrameColumn). Let any KeyError/IndexError propagate — no silent fallback.
            result[key] = torch.stack(self.hf_dataset[key][relative_indices])
        return result

    def _query_videos(self, query_timestamps, ep_idx):
        ep = self.meta.episodes[ep_idx]
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(
                video_path, shifted_query_ts, self.tolerance_s, self.video_backend
            )
            item[vid_key] = frames.squeeze(0)
        return item

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks.iloc[task_idx].name

        if "subtask_index" in self.features and self.meta.subtasks is not None:
            subtask_idx = item["subtask_index"].item()
            item["subtask"] = self.meta.subtasks.iloc[subtask_idx].name

        return item


def _load_filtered_frames(root: Path, episodes: list[int]) -> pa.Table:
    """Load only rows whose ``episode_index`` is in ``episodes``, preserving global parquet order.

    Mirrors lerobot's ``load_nested_dataset(..., episodes=...)`` (pyarrow predicate pushdown). Files
    are read in sorted ``chunk-*/file-*`` order and concatenated, so row position keeps matching the
    global ``index`` column.
    """
    ep_value_set = pa.array(sorted({int(e) for e in episodes}))
    paths = _sorted_parquet_paths(root / DATA_DIR)
    tables = []
    for p in paths:
        table = pq.read_table(str(p))
        mask = pc.is_in(table["episode_index"], value_set=ep_value_set)
        filtered = table.filter(mask)
        if filtered.num_rows > 0:
            tables.append(filtered)
    if len(tables) == 0:
        raise FileNotFoundError(f"No frames found for requested episodes {episodes} under {root}")
    return pa.concat_tables(tables, promote_options="default")
