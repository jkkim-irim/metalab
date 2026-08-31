"""Unit tests for publishing a checkpoint's rollout report (``rl_trainer._publish_report``).

Covers where the report lands and what link the training run gets: the copy beside the LOCAL checkpoints,
the ``$CKPT_S3_ROOT/<run_name>/report_<it>/`` key (the folder that already holds that run's ``model_*.pt``),
and the s3->CloudFront mapping the browser needs. The upload shells out to ``aws``, which is stubbed on PATH
here — these tests never touch S3 or W&B.

Run in the node env from the repo root:  python -m pytest learning -q
"""
import os
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


def _stub_aws(bin_dir: Path, *, rc: int = 0) -> None:
    """Put a fake ``aws`` on PATH that records its argv instead of uploading."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "aws").write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$@" >> "{bin_dir}/argv"\n'
        f"exit {rc}\n")
    (bin_dir / "aws").chmod(0o755)
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"


def test_local_copy_is_the_fallback_when_there_is_no_s3(tmp_path, monkeypatch):
    """No CKPT_S3_ROOT (--no-s3-sync, or no creds) — the report must still survive the temp dir's deletion."""
    monkeypatch.delenv("CKPT_S3_ROOT", raising=False)
    vdir = _recording(tmp_path / "val_record_x")
    ckpt = _ckpt(tmp_path / "logs" / "260730-1253_2envs_ppo_newton_abc_local")
    assert _publish_report(str(vdir), str(ckpt), 500) == ""      # nothing to link to
    out = ckpt.parent / "report_500"
    assert (out / "report.html").exists() and (out / "rollout.rrd").exists()
    assert (out / "data.json").exists()                          # the series ride along, page rebuildable


def test_s3_key_lands_in_the_checkpoint_folder_and_maps_to_cloudfront(tmp_path, monkeypatch):
    monkeypatch.setenv("CKPT_S3_ROOT", "s3://wirobotics-internal/jkkim/sim_rl/ckpts")
    _stub_aws(tmp_path / "bin")
    vdir = _recording(tmp_path / "val_record_y")
    run = "260730-1253_12288envs_newton_abc123_aws"
    ckpt = _ckpt(tmp_path / "logs" / run, it=800)
    url = _publish_report(str(vdir), str(ckpt), 800)
    assert url == ("https://d1iitptfxhu64e.cloudfront.net/"
                   f"jkkim/sim_rl/ckpts/{run}/report_800/report.html")
    # published to S3 → no local duplicate (S3 is where checkpoints live now, local or node)
    assert not (ckpt.parent / "report_800").exists()
    argv = (tmp_path / "bin" / "argv").read_text().split("\n")
    assert argv[:4] == ["s3", "cp", "--recursive", str(vdir)]
    assert argv[4] == f"s3://wirobotics-internal/jkkim/sim_rl/ckpts/{run}/report_800/"


def test_a_failed_upload_yields_no_link_but_keeps_the_local_copy(tmp_path, monkeypatch):
    """Publishing is auxiliary: a broken aws/S3 must not lose the artifact or raise into training."""
    monkeypatch.setenv("CKPT_S3_ROOT", "s3://wirobotics-internal/jkkim/sim_rl/ckpts")
    _stub_aws(tmp_path / "bin", rc=1)
    vdir = _recording(tmp_path / "val_record_z")
    ckpt = _ckpt(tmp_path / "logs" / "run-a", it=100)
    assert _publish_report(str(vdir), str(ckpt), 100) == ""
    assert (ckpt.parent / "report_100" / "report.html").exists()


def test_non_cloudfront_bucket_falls_back_to_the_s3_path(tmp_path, monkeypatch):
    """Only wirobotics-internal is fronted by CloudFront; anywhere else, a browser URL would be a lie."""
    monkeypatch.setenv("CKPT_S3_ROOT", "s3://some-other-bucket/team/ckpts")
    _stub_aws(tmp_path / "bin")
    vdir = _recording(tmp_path / "val_record_w")
    ckpt = _ckpt(tmp_path / "logs" / "run-b", it=7)
    assert _publish_report(str(vdir), str(ckpt), 7) == (
        "s3://some-other-bucket/team/ckpts/run-b/report_7/report.html")


def test_a_recording_without_a_report_is_skipped(tmp_path, monkeypatch):
    """A rollout that never stepped leaves a recording but no page — not an error."""
    monkeypatch.setenv("CKPT_S3_ROOT", "s3://wirobotics-internal/jkkim/sim_rl/ckpts")
    vdir = tmp_path / "val_record_v"
    vdir.mkdir()
    (vdir / "rollout.rrd").write_bytes(b"\x00")
    ckpt = _ckpt(tmp_path / "logs" / "run-c", it=1)
    assert _publish_report(str(vdir), str(ckpt), 1) == ""
    assert not (ckpt.parent / "report_1").exists()


def test_unnumbered_checkpoint_is_published_as_final(tmp_path, monkeypatch):
    monkeypatch.delenv("CKPT_S3_ROOT", raising=False)
    vdir = _recording(tmp_path / "val_record_u")
    ckpt = _ckpt(tmp_path / "logs" / "run-d")
    _publish_report(str(vdir), str(ckpt), None)
    assert (ckpt.parent / "report_final" / "report.html").exists()


def _main() -> int:
    """Direct run (no pytest): a tiny monkeypatch stand-in so this file is runnable anywhere."""
    import tempfile
    import types

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        with tempfile.TemporaryDirectory() as td:
            saved_env, saved_path = dict(os.environ), os.environ["PATH"]
            mp = types.SimpleNamespace(
                setenv=lambda k, v: os.environ.__setitem__(k, v),
                delenv=lambda k, raising=True: os.environ.pop(k, None),
                setattr=lambda *a, **k: None)
            try:
                fn(Path(td), mp)
                print(f"PASS {fn.__name__}")
            finally:
                os.environ.clear(); os.environ.update(saved_env); os.environ["PATH"] = saved_path
    print(f"{len(fns)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
