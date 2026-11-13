"""Regressions for exclusive publication and failed frame-window seeks."""

from __future__ import annotations

import errno
import json
import sys
import threading
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_enhance_lite import analyzer, pipeline


@pytest.mark.parametrize("existing_kind", ["file", "directory", "dangling_link"])
def test_exclusive_rename_keeps_existing_entries(tmp_path, existing_kind):
    source = tmp_path / "完成.partial.mp4"
    target = tmp_path / "完成.mp4"
    source.write_bytes(b"completed movie")
    if existing_kind == "file":
        target.write_bytes(b"other writer")
    elif existing_kind == "directory":
        target.mkdir()
    else:
        try:
            target.symlink_to(tmp_path / "absent.mp4")
        except OSError:
            pytest.skip("creating symlinks requires OS permission")
    with pytest.raises(FileExistsError):
        pipeline._rename_without_overwrite(source, target)
    assert source.read_bytes() == b"completed movie"
    if existing_kind == "file":
        assert target.read_bytes() == b"other writer"
    elif existing_kind == "directory":
        assert target.is_dir()
    else:
        assert target.is_symlink()


def test_no_link_publication_retries_atomic_collision_without_overwrite(tmp_path, monkeypatch):
    source = tmp_path / "output.partial.mp4"
    target = tmp_path / "output.mp4"
    source.write_bytes(b"completed movie")
    native_rename = pipeline._rename_without_overwrite
    attempts = []

    def no_links(*args):
        raise OSError(errno.ENOTSUP, "hard links unsupported")

    def concurrent_rename(src, dst):
        attempts.append(dst)
        if dst == target:
            dst.write_bytes(b"other writer")
        native_rename(src, dst)

    monkeypatch.setattr(pipeline.os, "link", no_links)
    monkeypatch.setattr(pipeline, "_rename_without_overwrite", concurrent_rename)
    saved = pipeline._publish_output_without_overwrite(source, target)
    assert saved != target
    assert target.read_bytes() == b"other writer"
    assert saved.read_bytes() == b"completed movie"
    assert not source.exists()
    assert len(attempts) == 2


def test_unsupported_atomic_publication_keeps_finished_partial(tmp_path, monkeypatch):
    source = tmp_path / "output.partial.mp4"
    target = tmp_path / "output.mp4"
    source.write_bytes(b"completed movie")

    def unsupported(*args):
        raise OSError(errno.ENOTSUP, "unsupported")

    monkeypatch.setattr(pipeline.os, "link", unsupported)
    monkeypatch.setattr(pipeline, "_rename_without_overwrite", unsupported)
    with pytest.raises(OSError):
        pipeline._publish_output_without_overwrite(source, target)
    assert source.read_bytes() == b"completed movie"
    assert not target.exists()


def test_windows_publication_uses_rename_not_replace(tmp_path, monkeypatch):
    source, target = tmp_path / "a", tmp_path / "b"
    calls = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(pipeline.os, "rename", lambda *args: calls.append(args))
    monkeypatch.setattr(pipeline.os, "replace", lambda *args: pytest.fail("replace can overwrite"))
    pipeline._rename_without_overwrite(source, target)
    assert calls == [(source, target)]


@pytest.mark.parametrize(
    "cadence,expected", [("constant", "cfr"), ("variable", "vfr_suspected"), ("missing", "unknown")]
)
def test_window_retry_uses_requested_absolute_region_and_preserves_classification(
    tmp_path, monkeypatch, cadence, expected
):
    calls = []
    cancel = threading.Event()
    # Nonzero video PTS, GOP longer than the video, variation only at the end.
    timestamps = [10 + i / 30 for i in range(60)]
    if cadence == "variable":
        timestamps = [t + (0.015 if i >= 48 else 0) for i, t in enumerate(timestamps)]

    def fake_probe(argv, *, cancel_event):
        assert cancel_event is cancel
        interval = argv[argv.index("-read_intervals") + 1]
        calls.append(interval)
        values = timestamps
        if not interval.startswith("%") or cadence == "missing":
            values = []
        else:
            values = [t for t in values if t < float(interval[1:])]
        return SimpleNamespace(
            stdout=json.dumps({"frames": [{"best_effort_timestamp_time": t} for t in values]})
        )

    monkeypatch.setattr(analyzer, "_run", fake_probe)
    status = analyzer._classify_framerate(
        Path("ffprobe"), tmp_path / "source.ts", 2, Fraction(30), Fraction(30), cancel, 10.0
    )
    assert status == expected
    assert calls == [
        "10.000000%+#64",
        "%11.566667",
        "10.750000%+#64",
        "%12.316667",
        "11.500000%+#64",
        "%13.066667",
    ]


def test_window_retry_does_not_swallow_cancellation(tmp_path, monkeypatch):
    calls = 0

    def fake_probe(argv, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(stdout='{"frames": []}')
        raise analyzer.AnalyzerError("cancelled while running ffprobe")

    monkeypatch.setattr(analyzer, "_run", fake_probe)
    with pytest.raises(analyzer.AnalyzerError, match="cancelled"):
        analyzer._probe_frame_timestamp_windows(Path("ffprobe"), tmp_path / "source.ts", 2)


def test_publication_skips_dangling_names_without_looping(tmp_path, monkeypatch):
    source = tmp_path / "output.partial.mp4"
    target = tmp_path / "output.mp4"
    source.write_bytes(b"finished")
    try:
        target.symlink_to(tmp_path / "missing")
        (tmp_path / "output~(1).mp4").symlink_to(tmp_path / "also-missing")
    except OSError:
        pytest.skip("creating symlinks requires OS permission")
    real_link = pipeline.os.link
    attempts = 0

    def bounded_link(*args):
        nonlocal attempts
        attempts += 1
        assert attempts <= 3, "publication repeatedly retried an occupied dangling-link name"
        return real_link(*args)

    monkeypatch.setattr(pipeline.os, "link", bounded_link)
    saved = pipeline._publish_output_without_overwrite(source, target)
    assert saved.name == "output~(2).mp4"
    assert saved.read_bytes() == b"finished"
    assert target.is_symlink()


def test_long_video_window_retry_decodes_from_nearby_earlier_point(tmp_path, monkeypatch):
    intervals = []

    def fake_probe(argv, **kwargs):
        interval = argv[argv.index("-read_intervals") + 1]
        intervals.append(interval)
        values = []
        if "#" not in interval:
            end = float(interval.split("%")[1])
            values = [end - 1 + i / 30 for i in range(10)]
        return SimpleNamespace(
            stdout=json.dumps({"frames": [{"best_effort_timestamp_time": t} for t in values]})
        )

    monkeypatch.setattr(analyzer, "_run", fake_probe)
    windows = analyzer._probe_frame_timestamp_windows(
        Path("ffprobe"), tmp_path / "source.ts", 120, video_start_sec=10
    )
    assert all(len(window) == 10 for window in windows)
    assert intervals == [
        "10.000000%+#64",
        "%12.000000",
        "69.000000%+#64",
        "64.000000%71.000000",
        "128.000000%+#64",
        "123.000000%130.000000",
    ]
