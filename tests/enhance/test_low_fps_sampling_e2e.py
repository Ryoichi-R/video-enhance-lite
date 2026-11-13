"""Real-media regressions for short/low-fps inputs and one-pass MPEG sampling."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from video_enhance_lite import analyzer, config
from tests.enhance.test_functional_regressions_e2e import FFMPEG, FFPROBE, _ffmpeg, _probe, _process

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("fps,duration", [(1, 10), (10, 1), (15, 1), (30, 1), (24, 0.25)])
def test_low_fps_and_short_cfr_complete_default_analysis_and_conversion(tmp_path, fps, duration):
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=96x64:rate={fps}:duration={duration}",
        "-c:v",
        "libx264",
        "-bf",
        "3",
        source,
    )
    meta, stats = analyzer.analyze(
        source, Path(FFPROBE), Path(FFMPEG), sample_dir=tmp_path / "samples"
    )
    assert meta.framerate_status == "cfr"
    assert stats.sample_count >= min(24, round(fps * duration)) - 1
    _process(tmp_path, source, output, meta)
    assert int(_probe(output)["streams"][0]["nb_read_frames"]) == round(fps * duration)


def test_too_few_timestamped_frames_remain_unknown(tmp_path):
    source = tmp_path / "tiny.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=30:duration=0.1", "-c:v", "libx264", source
    )
    assert analyzer.probe_metadata(Path(FFPROBE), source).framerate_status == "unknown"


@pytest.mark.parametrize(
    "container,suffix,codec", [("mpeg", ".mpg", "mpeg2video"), ("mpegts", ".ts", "libx264")]
)
def test_one_pass_samples_match_individual_video_relative_seeks(
    tmp_path, monkeypatch, container, suffix, codec
):
    source = tmp_path / f"source{suffix}"
    _ffmpeg(
        "-itsoffset",
        "0.5",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=96x64:rate=30:duration=2",
        "-f",
        "lavfi",
        "-i",
        "sine=duration=2.5",
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-fps_mode",
        "passthrough",
        "-c:v",
        codec,
        "-g",
        "300",
        "-c:a",
        "libmp3lame",
        "-f",
        container,
        source,
    )
    meta = analyzer.probe_metadata(Path(FFPROBE), source)
    calls = []
    original = analyzer._run_with_frame_progress

    def record(argv, **kwargs):
        calls.append(argv)
        return original(argv, **kwargs)

    monkeypatch.setattr(analyzer, "_run_with_frame_progress", record)
    events = []
    samples = analyzer._extract_samples(
        Path(FFMPEG),
        source,
        meta.duration_sec,
        24,
        tmp_path / "batch",
        None,
        lambda done, total: events.append(done),
        per_stream_timestamp_rebase=True,
    )
    assert len(calls) == 1
    assert len(samples) == 24
    assert events == sorted(events)
    assert events[-1] == 24
    for i, path in enumerate(samples):
        reference = tmp_path / f"reference-{i}.png"
        t = meta.duration_sec * (i + 0.5) / 24
        _ffmpeg("-i", source, "-ss", f"{t:.6f}", "-frames:v", "1", reference)
        assert np.array_equal(cv2.imread(str(path)), cv2.imread(str(reference))), i


def test_throttled_decode_outlives_old_total_timeout_with_one_invocation(tmp_path, monkeypatch):
    source = tmp_path / "slow.ts"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=96x64:rate=30:duration=4",
        "-c:v",
        "libx264",
        "-g",
        "15",
        source,
    )
    # Same idle budget as the old total budget, shortened for a deterministic
    # real-media test. Read at real time to keep decoding alive beyond it.
    monkeypatch.setattr(config, "DEFAULT_ANALYZE_PROCESS_TIMEOUT_SEC", 1.5)
    original = analyzer._run_with_frame_progress
    calls = []

    def throttled(argv, **kwargs):
        calls.append(argv)
        argv = argv[:]
        index = argv.index("-i")
        argv[index:index] = ["-readrate", "1"]
        return original(argv, **kwargs)

    monkeypatch.setattr(analyzer, "_run_with_frame_progress", throttled)
    start = time.monotonic()
    samples = analyzer._extract_samples(
        Path(FFMPEG),
        source,
        4,
        24,
        tmp_path / "samples",
        None,
        None,
        per_stream_timestamp_rebase=True,
    )
    assert time.monotonic() - start > 1.5
    assert len(samples) == 24
    assert len(calls) == 1


@pytest.mark.parametrize("fps,duration", [(10, 1), (15, 1)])
def test_low_fps_transport_stream_completes_analysis(tmp_path, fps, duration):
    source = tmp_path / "low.ts"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=96x64:rate={fps}:duration={duration}",
        "-c:v",
        "libx264",
        source,
    )
    meta, stats = analyzer.analyze(
        source, Path(FFPROBE), Path(FFMPEG), sample_dir=tmp_path / "samples"
    )
    assert meta.framerate_status == "cfr"
    assert stats.sample_count >= round(fps * duration) - 1


def test_one_pass_preserves_duplicate_selections_for_sparse_frames(tmp_path):
    source = tmp_path / "sparse.ts"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=96x64:rate=1:duration=10",
        "-c:v",
        "libx264",
        source,
    )
    samples = analyzer._extract_samples(
        Path(FFMPEG),
        source,
        9,
        24,
        tmp_path / "samples",
        None,
        None,
        per_stream_timestamp_rebase=True,
    )
    assert len(samples) == 24
    reference = tmp_path / "reference.png"
    for i, path in enumerate(samples):
        _ffmpeg("-i", source, "-ss", f"{9 * (i + 0.5) / 24:.6f}", "-frames:v", "1", reference)
        assert np.array_equal(cv2.imread(str(path)), cv2.imread(str(reference)))
    assert not list((tmp_path / "samples").glob(".sampling-*"))
