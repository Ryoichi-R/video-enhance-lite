"""Media-correctness contracts for VFR timing and stream preservation.

Public-contract tests exercise the production builders; reference tests retain
the selected Phase 2 FFmpeg strategies as an independent correctness oracle.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from video_enhance_lite import analyzer, pipeline, stages
from video_enhance_lite.media_streams import (
    UnsupportedMediaStreamsError,
    resolve_mp4_stream_preservation,
)


pytestmark = pytest.mark.e2e

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
if _FFMPEG is None or _FFPROBE is None:
    pytest.skip("ffmpeg/ffprobe not on PATH", allow_module_level=True)


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=True,
        shell=False,
    )


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    return _run([_FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *args])


def test_short_cfr_analyzer_tolerates_missing_final_sample(tmp_path):
    source = tmp_path / "short-cfr.mp4"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=30:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ]
    )
    warnings: list[str] = []

    meta, stats = analyzer.analyze(
        input_path=source,
        ffprobe=Path(_FFPROBE),
        ffmpeg=Path(_FFMPEG),
        sample_count=24,
        sample_dir=tmp_path / "samples",
        warning=warnings.append,
    )

    assert meta.framerate_status == "cfr"
    assert 23 <= stats.sample_count <= 24
    if stats.sample_count == 23:
        assert len(warnings) == 1
        assert "final analyzer sample" in warnings[0]


def _probe(path: Path) -> dict:
    proc = _run(
        [
            _FFPROBE,
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=index,codec_type,codec_name,duration,nb_read_frames:stream_tags=language:"
            "format=duration:format_tags=title",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(proc.stdout)


def _duration(path: Path) -> float:
    return float(_probe(path)["format"]["duration"])


def _video_frame_count(path: Path) -> int:
    video = next(stream for stream in _probe(path)["streams"] if stream["codec_type"] == "video")
    return int(video["nb_read_frames"])


def _raw_rgb_frame(path: Path, frame_index: int) -> bytes:
    proc = subprocess.run(
        [
            _FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-fps_mode",
            "passthrough",
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        capture_output=True,
        timeout=120,
        check=True,
        shell=False,
    )
    return proc.stdout


@pytest.fixture(scope="module")
def vfr_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("vfr-correctness")
    source = root / "source-vfr.mp4"
    reference_frames = root / "reference-frames"
    reference_frames.mkdir()
    current_chunks = root / "current-chunks"
    current_chunks.mkdir()
    current_output = root / "current-output.mp4"
    reference_output = root / "reference-output.mp4"

    # Eight seconds on one timeline: 15 fps for the first four seconds and
    # 30 fps for the last four. There are 180 unique source frames.
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=30:duration=8",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=8",
            "-vf",
            r"select=if(lt(t\,4)\,not(mod(n\,2))\,1)",
            "-fps_mode",
            "vfr",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ]
    )
    meta = analyzer.probe_metadata(Path(_FFPROBE), source)
    assert meta.framerate_status == "vfr_suspected"

    # Public chunked behavior: every chunk uses the same nominal CFR target and
    # a global rounding-boundary frame budget.
    chunk_paths: list[Path] = []
    for start_sec, dur_sec, chunk_index in pipeline._compute_chunk_ranges(meta.duration_sec, 3):
        frames = root / f"current-frames-{chunk_index}"
        frames.mkdir()
        expected_frames = pipeline._normalized_frame_count(start_sec, dur_sec, meta.fps)
        _run(
            stages.build_extract_args(
                Path(_FFMPEG),
                source,
                start_sec,
                dur_sec,
                frames,
                normalization_fps=meta.fps,
                expected_frame_count=expected_frames,
            )
        )
        chunk = current_chunks / f"chunk_{chunk_index:03d}.mp4"
        _run(
            stages.build_mux_args(
                Path(_FFMPEG),
                frames,
                meta.fps,
                chunk,
                ["-c:v", "libx264", "-pix_fmt", "yuv420p"],
            )
        )
        chunk_paths.append(chunk)
    assert meta.streams is not None
    stream_plan = resolve_mp4_stream_preservation(
        meta.streams,
        primary_video_index=meta.primary_video_index,
    )
    pipeline._concat_chunks(
        runner=pipeline.RealProcessRunner(),
        ffmpeg=Path(_FFMPEG),
        chunks_dir=current_chunks,
        chunk_paths=chunk_paths,
        input_path=source,
        stream_plan=stream_plan,
        duration_sec=meta.duration_sec,
        output_path=current_output,
        log=lambda _message: None,
    )

    # Selected D11 reference: normalize on extraction to the explicit nominal
    # CFR. FFmpeg duplicates/drops before the PNG boundary, so downstream
    # engines receive a uniform sequence without losing the source timeline.
    fps_text = f"{meta.fps.numerator}/{meta.fps.denominator}"
    _run_ffmpeg(
        [
            "-i",
            str(source),
            "-vf",
            f"fps={fps_text}",
            "-start_number",
            "0",
            str(reference_frames / "%08d.png"),
        ]
    )
    _run(
        stages.build_mux_args(
            Path(_FFMPEG),
            reference_frames,
            meta.fps,
            reference_output,
            ["-c:v", "libx264", "-pix_fmt", "yuv420p"],
        )
    )
    return {
        "source": source,
        "current": current_output,
        "reference": reference_output,
        "fps": meta.fps,
    }


def test_reference_explicit_cfr_normalization_preserves_duration(vfr_case):
    source_duration = _duration(vfr_case["source"])
    reference_duration = _duration(vfr_case["reference"])
    expected_frames = round(source_duration * float(vfr_case["fps"]))

    assert reference_duration == pytest.approx(source_duration, abs=0.05)
    assert _video_frame_count(vfr_case["reference"]) == expected_frames


def test_public_contract_vfr_pipeline_preserves_duration(vfr_case):
    assert _duration(vfr_case["current"]) == pytest.approx(_duration(vfr_case["source"]), abs=0.05)
    expected_frames = round(_duration(vfr_case["source"]) * float(vfr_case["fps"]))
    assert _video_frame_count(vfr_case["current"]) == expected_frames


def test_public_contract_vfr_pipeline_keeps_av_end_within_50ms(vfr_case):
    payload = _probe(vfr_case["current"])
    durations = {
        stream["codec_type"]: float(stream["duration"])
        for stream in payload["streams"]
        if stream["codec_type"] in {"video", "audio"}
    }
    assert abs(durations["video"] - durations["audio"]) <= 0.05


def test_public_contract_vfr_pipeline_keeps_first_and_last_normalized_frames(vfr_case):
    frame_count = _video_frame_count(vfr_case["current"])
    for frame_index in (0, frame_count - 1):
        current = _raw_rgb_frame(vfr_case["current"], frame_index)
        reference = _raw_rgb_frame(vfr_case["reference"], frame_index)
        assert len(current) == len(reference) == 64 * 64 * 3
        mean_abs_error = sum(abs(a - b) for a, b in zip(current, reference)) / len(current)
        # Each chunk is independently H.264 encoded before concat, while the
        # reference is encoded as one sequence. Allow only minor codec drift.
        assert mean_abs_error <= 4.0


@pytest.fixture(scope="module")
def multi_stream_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("stream-preservation")
    subtitle = root / "captions.srt"
    subtitle.write_text(
        "1\n00:00:00,200 --> 00:00:01,800\nPhase 2 subtitle\n",
        encoding="utf-8",
    )
    source = root / "source.mp4"
    chunk = root / "chunk_000.mp4"
    chunks = root / "chunks"
    chunks.mkdir()
    current_output = root / "current-output.mp4"
    reference_output = root / "reference-output.mp4"

    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=10:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=48000:duration=2",
            "-i",
            str(subtitle),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map",
            "2:a:0",
            "-map",
            "3:s:0",
            "-metadata",
            "title=phase2-stream-contract",
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:1",
            "language=jpn",
            "-metadata:s:s:0",
            "language=eng",
            "-disposition:a:0",
            "default",
            "-disposition:a:1",
            "0",
            "-disposition:s:0",
            "forced",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-c:s",
            "mov_text",
            str(source),
        ]
    )
    _run_ffmpeg(["-i", str(source), "-map", "0:v:0", "-an", "-sn", "-c:v", "copy", str(chunk)])
    chunk_in_dir = chunks / chunk.name
    shutil.copy2(chunk, chunk_in_dir)
    meta = analyzer.probe_metadata(Path(_FFPROBE), source)
    assert meta.streams is not None
    stream_plan = resolve_mp4_stream_preservation(
        meta.streams,
        primary_video_index=meta.primary_video_index,
    )

    pipeline._concat_chunks(
        runner=pipeline.RealProcessRunner(),
        ffmpeg=Path(_FFMPEG),
        chunks_dir=chunks,
        chunk_paths=[chunk_in_dir],
        input_path=source,
        stream_plan=stream_plan,
        duration_sec=meta.duration_sec,
        output_path=current_output,
        log=lambda _message: None,
    )

    # Selected D12 reference for MP4-compatible streams.
    _run_ffmpeg(
        [
            "-i",
            str(chunk),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-map",
            "1:s?",
            "-map_metadata",
            "1",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-c:s",
            "copy",
            "-t",
            f"{_duration(source):.6f}",
            str(reference_output),
        ]
    )
    return {
        "source": source,
        "current": current_output,
        "reference": reference_output,
    }


def _stream_signature(path: Path) -> tuple[list[str], list[str], str | None]:
    payload = _probe(path)
    audio_languages = [
        stream.get("tags", {}).get("language", "und")
        for stream in payload["streams"]
        if stream["codec_type"] == "audio"
    ]
    subtitle_languages = [
        stream.get("tags", {}).get("language", "und")
        for stream in payload["streams"]
        if stream["codec_type"] == "subtitle"
    ]
    title = payload.get("format", {}).get("tags", {}).get("title")
    return audio_languages, subtitle_languages, title


def _stream_dispositions(path: Path) -> tuple[list[dict], list[dict]]:
    payload = _probe(path)
    audio = [
        stream.get("disposition", {})
        for stream in payload["streams"]
        if stream["codec_type"] == "audio"
    ]
    subtitle = [
        stream.get("disposition", {})
        for stream in payload["streams"]
        if stream["codec_type"] == "subtitle"
    ]
    return audio, subtitle


def test_reference_mp4_mapping_preserves_supported_streams_and_metadata(multi_stream_case):
    assert _stream_signature(multi_stream_case["reference"]) == (
        ["eng", "jpn"],
        ["eng"],
        "phase2-stream-contract",
    )
    assert _duration(multi_stream_case["reference"]) == pytest.approx(
        _duration(multi_stream_case["source"]), abs=0.08
    )


def test_public_contract_pipeline_preserves_supported_streams_and_metadata(multi_stream_case):
    assert _stream_signature(multi_stream_case["current"]) == _stream_signature(
        multi_stream_case["source"]
    )
    assert _stream_dispositions(multi_stream_case["current"]) == _stream_dispositions(
        multi_stream_case["source"]
    )


def test_real_inventory_rejects_subrip_before_processing(tmp_path):
    subtitle = tmp_path / "unsupported.srt"
    subtitle.write_text(
        "1\n00:00:00,100 --> 00:00:00,900\nunsupported in MP4 copy mode\n",
        encoding="utf-8",
    )
    source = tmp_path / "unsupported-subtitle.mkv"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=10:duration=1",
            "-i",
            str(subtitle),
            "-map",
            "0:v:0",
            "-map",
            "1:s:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:s",
            "subrip",
            str(source),
        ]
    )

    meta = analyzer.probe_metadata(Path(_FFPROBE), source)
    assert meta.streams is not None
    with pytest.raises(UnsupportedMediaStreamsError, match="index=1.*subrip"):
        resolve_mp4_stream_preservation(
            meta.streams,
            primary_video_index=meta.primary_video_index,
        )
