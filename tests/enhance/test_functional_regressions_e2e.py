"""Real media checks for paths, stream duration, exposure, and autorotation.

Only the AI upscaler in the rotation test is replaced by deterministic resizing;
analysis, extraction, clamping, encoding, and final mux use real FFmpeg.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from video_enhance_lite import analyzer, cli, config, decisions, pipeline, temp_workspace
from tests.enhance.test_pipeline import _make_decisions


pytestmark = pytest.mark.e2e
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
if not FFMPEG or not FFPROBE:
    pytest.skip("real FFmpeg/ffprobe required", allow_module_level=True)


def _run(argv):
    return subprocess.run(
        [str(arg) for arg in argv],
        capture_output=True,
        check=True,
        timeout=120,
    )


def _ffmpeg(*args):
    return _run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args])


def _probe(path):
    return json.loads(
        _run(
            [
                FFPROBE,
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                path,
            ]
        ).stdout
    )


def _process(tmp_path, source, output, meta, selected=None, runner=None):
    context = temp_workspace.create_temp_run_context(source, tmp_path / "temp")
    try:
        pipeline.run_pipeline(
            input_path=source,
            output_path=output,
            decisions=selected or _make_decisions(),
            meta=meta,
            ffmpeg=Path(FFMPEG),
            ffprobe=Path(FFPROBE),
            temp_run=context,
            log=lambda _message: None,
            chunk_seconds=1,
            runner=runner,
            realesrgan=Path("test-resizer") if runner else None,
        )
    finally:
        temp_workspace.cleanup_temp_run(context)


@pytest.mark.parametrize(
    "relative_input,new_directory", [(True, False), (False, True), (True, True)]
)
def test_cli_saves_relative_inputs_and_creates_default_output_directory(
    tmp_path, monkeypatch, relative_input, new_directory
):
    source = tmp_path / "source.mp4"
    destination = tmp_path / "Videos" / "video-enhance-lite"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=30:duration=1", "-c:v", "libx264", source
    )
    if not new_directory:
        destination.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(config.ENV_OUTPUT_ROOT, str(destination))
    monkeypatch.setenv(config.ENV_TEMP_ROOT, str(tmp_path / "temp"))
    monkeypatch.setattr(cli.deps, "check_ffmpeg", lambda: SimpleNamespace(path=Path(FFMPEG)))
    monkeypatch.setattr(cli.deps, "check_ffprobe", lambda: SimpleNamespace(path=Path(FFPROBE)))
    monkeypatch.setattr(cli.deps, "format_resolved_binary", lambda *_args: "test FFmpeg")
    disabled = ",".join(key for key in config.STAGE_KEYS if key != "encode")
    code = cli.main(
        [
            source.name if relative_input else str(source),
            "--auto-all",
            "--disable",
            disabled,
            "--encode-preset",
            "ultrafast",
            "--worker-count",
            "1",
        ]
    )
    assert code == 0
    outputs = list(destination.glob("*.mp4"))
    assert len(outputs) == 1
    stream = _probe(outputs[0])["streams"][0]
    assert int(stream["nb_read_frames"]) == 30
    assert float(stream["duration"]) == pytest.approx(1, abs=0.04)


@pytest.mark.parametrize("suffix", ["mp4", "mkv"])
def test_analysis_and_pipeline_preserve_audio_tail_without_sampling_past_video(tmp_path, suffix):
    source = tmp_path / f"source.{suffix}"
    output = tmp_path / "output.mp4"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=96x64:rate=30:duration=1",
        "-f",
        "lavfi",
        "-i",
        "sine=duration=2:sample_rate=48000",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        source,
    )
    meta, stats = analyzer.analyze(
        source,
        Path(FFPROBE),
        Path(FFMPEG),
        sample_count=24,
        sample_dir=tmp_path / "samples",
    )
    assert meta.duration_sec == pytest.approx(1, abs=0.04)
    assert meta.container_duration_sec == pytest.approx(2, abs=0.05)
    assert stats.sample_count >= 23
    _process(tmp_path, source, output, meta)
    streams = _probe(output)["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")
    audio = next(s for s in streams if s["codec_type"] == "audio")
    assert int(video["nb_read_frames"]) == 30
    assert float(video["duration"]) == pytest.approx(1, abs=0.04)
    assert float(audio["duration"]) == pytest.approx(2, abs=0.06)


@pytest.mark.parametrize("level,direction", [(20, 1), (235, -1)])
def test_recommended_exposure_changes_real_output_in_correct_direction(tmp_path, level, direction):
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=gray:size=96x64:rate=30:duration=1",
        "-vf",
        f"lutyuv=y={level}:u=128:v=128",
        "-c:v",
        "libx264",
        source,
    )
    meta, stats = analyzer.analyze(
        source,
        Path(FFPROBE),
        Path(FFMPEG),
        sample_dir=tmp_path / "samples",
    )
    selected = decisions.recommend(meta, stats)
    for dec in selected:
        dec.enabled = dec.key in {"color_correction", "encode"}
    _process(tmp_path, source, output, meta, selected)

    def mean_luma(path):
        pixels = _ffmpeg(
            "-i", path, "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"
        ).stdout
        return np.frombuffer(pixels, dtype=np.uint8).mean()

    assert direction * (mean_luma(output) - mean_luma(source)) > 10


class _ResizeOnlyRunner(pipeline.RealProcessRunner):
    def run_with_progress(self, argv, **kwargs):
        if kwargs.get("stage") != "upscale_frames":
            return super().run_with_progress(argv, **kwargs)
        source = Path(argv[argv.index("-i") + 1])
        output = Path(argv[argv.index("-o") + 1])
        scale = int(argv[argv.index("-s") + 1])
        for frame_path in source.glob("*.png"):
            frame = cv2.imdecode(np.frombuffer(frame_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            enlarged = cv2.resize(frame, None, fx=scale, fy=scale)
            ok, data = cv2.imencode(".png", enlarged)
            assert ok
            (output / frame_path.name).write_bytes(data.tobytes())
        return 0


@pytest.mark.parametrize("upscale", [False, True])
def test_rotated_video_summary_and_scaled_output_agree(tmp_path, upscale):
    base = tmp_path / "base.mp4"
    source = tmp_path / "rotated.mp4"
    output = tmp_path / "output.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=96x64:rate=30:duration=1", "-c:v", "libx264", base)
    _ffmpeg("-display_rotation", "90", "-i", base, "-c", "copy", source)
    meta = analyzer.probe_metadata(Path(FFPROBE), source)
    assert (meta.width, meta.height) == (64, 96)
    selected = _make_decisions(upscale=upscale)
    if upscale:
        decisions.decisions_by_key(selected)["upscale"].params.update(scale=2, target_width=192)
    summary = decisions.compute_output_summary(meta, selected)
    _process(tmp_path, source, output, meta, selected, _ResizeOnlyRunner())
    stream = _probe(output)["streams"][0]
    expected = (192, 288) if upscale else (64, 96)
    assert (summary.out_width, summary.out_height) == expected
    assert (stream["width"], stream["height"]) == expected


def _stream_start(probe, codec_type):
    stream = next(s for s in probe["streams"] if s["codec_type"] == codec_type)
    return float(stream["start_time"])


@pytest.mark.parametrize(
    "suffix,video_delay,audio_delay",
    [(".mp4", 0.5, 0.0), (".ts", 0.5, 0.0), (".ts", 0.0, 0.5), (".mkv", 0.0, 0.5)],
)
def test_output_keeps_source_av_offset_and_every_video_frame(
    tmp_path, suffix, video_delay, audio_delay
):
    source = tmp_path / f"offset{suffix}"
    output = tmp_path / "out.mp4"
    _ffmpeg(
        "-itsoffset", video_delay,
        "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=30:duration=2",
        "-itsoffset", audio_delay,
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-g", "15", "-pix_fmt", "yuv420p", "-c:a", "aac",
        source,
    )  # fmt: skip
    meta = analyzer.probe_metadata(Path(FFPROBE), source)

    _process(tmp_path, source, output, meta)

    before = _probe(source)
    after = _probe(output)
    expected = _stream_start(before, "video") - _stream_start(before, "audio")
    actual = _stream_start(after, "video") - _stream_start(after, "audio")
    assert actual == pytest.approx(expected, abs=1 / 30)
    frames = [
        int(next(s for s in probe["streams"] if s["codec_type"] == "video")["nb_read_frames"])
        for probe in (before, after)
    ]
    assert frames[1] == frames[0]
