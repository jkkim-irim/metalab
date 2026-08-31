"""Unit tests for publishing a checkpoint's rollout report (``rl_trainer._publish_report``).

Covers where the report lands: a copy beside that run's checkpoints, so it survives the deletion of the
temp dir the sim-service recorded into, and one subdir per checkpoint so two checkpoints never collide.

Run from the repo root:  python -m pytest learning -q
"""
from pathlib import Path

from learning.trainer.rl_trainer import _publish_report


def _recording(d: Path) -> Path:
    """A finished recording dir as the sim-service leaves it."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.html").write_text("<html>report</html>")
    (d / "data.json").write_text('{"meta":{}}')
    (d / "rollout.rrd").write_bytes(b"\x00")
    return d


def _ckpt(d: Path, it: int = 500) -> Path:
    """A checkpoint inside a run's log dir (its basename is the run name)."""
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"model_{it}.pt"
    p.write_bytes(b"\x00")
    return p


def test_report_lands_beside_the_checkpoint(tmp_path):
    """The recording dir is deleted right after this returns, so the whole thing must be copied out."""
    vdir = _recording(tmp_path / "val_record_x")
    ckpt = _ckpt(tmp_path / "logs" / "260730-1253_2envs_ppo_newton_abc")
    out = Path(_publish_report(str(vdir), str(ckpt), 500))
    assert out == ckpt.parent / "report_500"
    assert (out / "report.html").exists() and (out / "rollout.rrd").exists()
    assert (out / "data.json").exists()                          # the series ride along, page rebuildable


def test_each_checkpoint_gets_its_own_subdir(tmp_path):
    """The page addresses its recording by BASENAME, so a flat layout would repoint an older page."""
    ckpt_dir = tmp_path / "logs" / "260730-1253_2envs_ppo_newton_abc"
    for it in (400, 800):
        _publish_report(str(_recording(tmp_path / f"val_record_{it}")), str(_ckpt(ckpt_dir, it)), it)
    assert (ckpt_dir / "report_400" / "report.html").exists()
    assert (ckpt_dir / "report_800" / "report.html").exists()


def test_a_recording_without_a_page_publishes_nothing(tmp_path):
    """Auxiliary artifact: a missing report.html warns and returns "", it must not raise into training."""
    vdir = tmp_path / "val_record_empty"
    vdir.mkdir()
    ckpt = _ckpt(tmp_path / "logs" / "260730-1253_2envs_ppo_newton_abc")
    assert _publish_report(str(vdir), str(ckpt), 500) == ""
    assert not (ckpt.parent / "report_500").exists()
