"""Unit tests for the trainer-side first-batch-aug visualization (learning/trainer/viz.py).

log_first_batch_aug logs annotated per-camera grids to wandb PLUS non-truncating info tables:
``media/first_batch/info`` (one row per sample: prompt + per-camera + state/action) and
``media/first_batch/model`` (a model-specific input summary if the policy owns one, else a generic
pre-normalization line). ``_aug_summary_lines`` describes the augmentation actually applied. Pure
presentation — no training/validation logic. Run in the node env from the repo root:
  python -m pytest learning -q
"""
import torch

from learning.configs.config import ImageAugConfig
from learning.trainer.viz import _aug_summary_lines, log_first_batch_aug


class _FakeTable:
    """Records the columns/data handed to ``wandb.Table`` so tests can assert what got logged."""

    def __init__(self, columns, data):
        self.columns = list(columns)
        self.data = [list(r) for r in data]


class _FakeWandb:
    """Minimal wandb stub WITHOUT ``.Table`` — exercises the graceful table-less fallback path
    (the real test double the trainer historically used: only image grids are supported)."""

    def __init__(self):
        self.logged = {}
        self._wandb = self

    def Image(self, arr, caption=""):
        return f"Image{tuple(arr.shape)}"

    def log(self, d):
        self.logged.update(d)


class _FakeWandbTable(_FakeWandb):
    """wandb stub WITH ``.Table`` — exercises the ``media/first_batch/{info,model}`` tables."""

    def Table(self, columns, data):
        return _FakeTable(columns, data)


class _FakePolicy:
    """A policy that owns a model-specific input summary (like GrootPolicy.input_summary): viz just
    renders whatever this returns (model-agnostic), so the exact lines are the model's business."""

    def input_summary(self, batch):
        return [
            "GR00T-N1.7 | modality=mikasa | embodiment=new_embodiment",
            "images: cameras ['cam_a', 'cam_b'] -> resized 256x256 (crop 230x230)",
        ]


def test_log_first_batch_aug_logs_per_camera_grids():
    """No-.Table stub -> only the per-camera grids are logged (table path skipped gracefully)."""
    batch = {
        "observation.images.cam_a": torch.rand(4, 3, 32, 32),
        "observation.images.cam_b": torch.rand(4, 3, 32, 32),
        "observation.state": torch.randn(4, 8),  # non-image keys don't produce grids
    }
    w = _FakeWandb()
    # Pass the real aug config (enabled) — the overlay renders its args; output stays an HWC array.
    log_first_batch_aug(w, batch, ImageAugConfig(), "gpu")  # real first batch; samples nothing
    assert set(w.logged) == {"media/first_batch_aug/cam_a", "media/first_batch_aug/cam_b"}
    for img in w.logged.values():
        assert img.startswith("Image(") and img.endswith(", 3)")   # annotated HWC-uint8 RGB array


def test_log_first_batch_info_tables_with_policy():
    """With a .Table client + a policy that owns input_summary: grids + a per-sample info table +
    the model-specific summary table are all logged; prompts come from batch["task"]."""
    batch = {
        "observation.images.cam_a": torch.rand(2, 3, 16, 16),
        "observation.images.cam_b": torch.rand(2, 3, 16, 16),
        "observation.state": torch.randn(2, 12),
        "action": torch.randn(2, 4, 9),                 # (B, chunk, action_dim)
        "task": ["pick up the cube", "stack the blocks"],
    }
    w = _FakeWandbTable()
    log_first_batch_aug(w, batch, ImageAugConfig(), "gpu", policy=_FakePolicy())

    # Grids are still logged alongside the tables.
    assert {"media/first_batch_aug/cam_a", "media/first_batch_aug/cam_b"} <= set(w.logged)

    # Per-sample info table: one row per sample, expected columns, prompts wired from batch["task"].
    info = w.logged["media/first_batch/info"]
    assert info.columns == ["sample_idx", "prompt", "cam_a", "cam_b", "state[:8]", "action[:7]"]
    assert len(info.data) == 2
    assert [r[0] for r in info.data] == [0, 1]
    assert [r[1] for r in info.data] == ["pick up the cube", "stack the blocks"]
    assert info.data[0][2].startswith("3x16x16 [")      # cam_a cell: "C x H x W [min,max]"

    # Model-owned summary rendered verbatim, with the aug description appended (relocated from pixels).
    model = w.logged["media/first_batch/model"]
    assert model.columns == ["model input summary"]
    assert model.data[0][0].startswith("GR00T-N1.7 | modality=mikasa")
    assert any("aug: ON" in r[0] for r in model.data)


def test_log_first_batch_model_table_generic_fallback():
    """policy=None -> the model table falls back to the current generic lines (ACT is unchanged);
    missing state/action columns degrade to "(n/a)"."""
    batch = {
        "observation.images.cam_a": torch.rand(2, 3, 16, 16),
        "task": ["do a", "do b"],                        # no state / no action in this batch
    }
    w = _FakeWandbTable()
    log_first_batch_aug(w, batch, ImageAugConfig(), "off", policy=None)

    model = w.logged["media/first_batch/model"]
    assert model.data[0][0].startswith("model input BEFORE normalization")
    assert model.data[-1][0].startswith("aug: OFF")     # generic fallback keeps the aug summary

    info = w.logged["media/first_batch/info"]
    assert info.columns == ["sample_idx", "prompt", "cam_a", "state[:8]", "action[:7]"]
    assert info.data[0][0] == 0
    assert info.data[0][1] == "do a"
    assert info.data[0][3] == "(n/a)"                    # state[:8] absent -> graceful
    assert info.data[0][4] == "(n/a)"                    # action[:7] absent -> graceful


def test_log_first_batch_no_table_stub_skips_tables():
    """Even with a policy, a wandb stub lacking .Table logs ONLY the grids (never crashes)."""
    batch = {
        "observation.images.cam_a": torch.rand(2, 3, 16, 16),
        "observation.state": torch.randn(2, 8),
        "action": torch.randn(2, 4, 7),
        "task": ["x", "y"],
    }
    w = _FakeWandb()   # no .Table attribute
    log_first_batch_aug(w, batch, ImageAugConfig(), "gpu", policy=_FakePolicy())
    assert set(w.logged) == {"media/first_batch_aug/cam_a"}   # tables skipped; grid still logged


def test_aug_summary_lines_on_and_off():
    assert _aug_summary_lines(ImageAugConfig(), "off")[0].startswith("aug: OFF")
    assert _aug_summary_lines(ImageAugConfig(), "off")[0].startswith("aug: OFF")
    on = _aug_summary_lines(ImageAugConfig(), "gpu")
    assert on[0].startswith("aug: ON")
    assert "gpu" in on[0]                                       # says WHERE it runs
    assert any("brightness" in s and "(0.8, 1.2)" in s for s in on)   # ranges are listed
    assert any("rotate_deg" in s for s in on)                  # geometric is listed too
    # cpu vs gpu describe the SAME ranges — only the "where" line differs
    cpu = _aug_summary_lines(ImageAugConfig(), "cpu")
    assert on[1:] == cpu[1:] and on[0] != cpu[0]
