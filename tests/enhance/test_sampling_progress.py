"""Advancing decode progress, not elapsed time or repeated output, controls liveness."""

from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest

from video_enhance_lite import analyzer


def _python(code):
    return [sys.executable, "-u", "-c", code]


def test_advancing_frames_allow_work_longer_than_idle_budget():
    started = time.monotonic()
    analyzer._run_with_frame_progress(
        _python(
            "import time\nfor i in range(16):\n print('frame='+str(i),flush=True)\n time.sleep(.1)"
        ),
        idle_timeout_sec=0.7,
    )
    assert time.monotonic() - started > 1.4


@pytest.mark.parametrize("chatter", [False, True])
def test_stall_is_killed_even_when_identical_progress_keeps_arriving(monkeypatch, chatter):
    spawned = []
    real_popen = analyzer.subprocess.Popen

    def capture(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(analyzer.subprocess, "Popen", capture)
    code = "import time\nfor i in range(100):\n " + (
        "print('frame=1',flush=True); time.sleep(.1)" if chatter else "time.sleep(.1)"
    )
    with pytest.raises(analyzer.AnalyzerError, match="no decoding progress"):
        analyzer._run_with_frame_progress(_python(code), idle_timeout_sec=0.5)
    assert spawned[0].poll() is not None


def test_cancellation_kills_an_actively_progressing_child(monkeypatch):
    spawned = []
    real_popen = analyzer.subprocess.Popen

    def capture(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(analyzer.subprocess, "Popen", capture)
    cancel = threading.Event()
    timer = threading.Timer(0.5, cancel.set)
    timer.start()
    try:
        with pytest.raises(analyzer.AnalyzerError, match="cancelled"):
            analyzer._run_with_frame_progress(
                _python(
                    "import time\nfor i in range(100):\n print('frame='+str(i),flush=True)\n time.sleep(.1)"
                ),
                idle_timeout_sec=5,
                cancel_event=cancel,
            )
    finally:
        timer.cancel()
    assert spawned[0].poll() is not None


def test_progress_runner_drains_stderr_and_reports_nonzero_exit():
    with pytest.raises(subprocess.CalledProcessError) as error:
        analyzer._run_with_frame_progress(
            _python(
                "import sys\nfor i in range(10000): print('diagnostic',file=sys.stderr)\nprint('last error',file=sys.stderr)\nsys.exit(3)"
            ),
            idle_timeout_sec=3,
        )
    assert error.value.returncode == 3
    assert "last error" in error.value.stderr
    assert len(error.value.stderr) < 1000


@pytest.mark.parametrize("missing_index,raises", [(1, True), (3, False)])
def test_one_pass_keeps_missing_middle_fatal_and_final_warning(
    tmp_path, monkeypatch, missing_index, raises
):
    warnings = []

    def fake_pass(ffmpeg, source, duration, paths, cancel, progress):
        for i, path in enumerate(paths):
            if i != missing_index:
                path.write_bytes(b"image")

    monkeypatch.setattr(analyzer, "_extract_samples_one_pass", fake_pass)

    def extract():
        return analyzer._extract_samples(
            tmp_path / "ffmpeg",
            tmp_path / "source.ts",
            2,
            4,
            tmp_path / "samples",
            None,
            None,
            warning=warnings.append,
            per_stream_timestamp_rebase=True,
        )

    if raises:
        with pytest.raises(analyzer.AnalyzerError, match="produced no file"):
            extract()
    else:
        assert len(extract()) == 3
        assert len(warnings) == 1


def test_failed_one_pass_is_not_accepted_as_a_missing_final_sample(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["ffmpeg"], stderr="decode failed")

    monkeypatch.setattr(analyzer, "_run_with_frame_progress", fail)
    with pytest.raises(analyzer.AnalyzerError, match="decode failed"):
        analyzer._extract_samples(
            tmp_path / "ffmpeg", tmp_path / "source.ts", 2, 4, tmp_path / "samples", None, None
        )
