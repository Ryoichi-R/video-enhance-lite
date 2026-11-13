"""Analyzer subprocesses must stop on timeout or cancellation instead of hanging."""

from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest

from video_enhance_lite import analyzer

SLEEPER = [sys.executable, "-c", "import time; time.sleep(30)"]


def test_run_stops_a_process_that_exceeds_the_timeout():
    started = time.monotonic()
    with pytest.raises(analyzer.AnalyzerError, match="did not finish within"):
        analyzer._run(SLEEPER, timeout=0.5)
    assert time.monotonic() - started < 10


def test_run_stops_promptly_when_cancelled():
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(analyzer.AnalyzerError, match="cancelled"):
        analyzer._run(SLEEPER, timeout=60, cancel_event=cancel)
    assert time.monotonic() - started < 10


def test_run_reports_nonzero_exit_like_check_true():
    argv = [sys.executable, "-c", "import sys; sys.stderr.write('bad input'); sys.exit(3)"]
    with pytest.raises(subprocess.CalledProcessError) as info:
        analyzer._run(argv)
    assert info.value.returncode == 3
    assert "bad input" in info.value.stderr


def test_run_returns_output_of_a_successful_process():
    argv = [sys.executable, "-c", "print('ok')"]
    assert analyzer._run(argv).stdout.strip() == "ok"
