"""Real analysis-to-output coverage for MPEG-PS and long-GOP MPEG-TS."""

from pathlib import Path

import pytest

from video_enhance_lite import analyzer
from tests.enhance.test_functional_regressions_e2e import (
    FFMPEG,
    FFPROBE,
    _ffmpeg,
    _probe,
    _process,
    _stream_start,
)

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize(
    "container,suffix,delay,gop",
    [
        ("mpeg", ".mpg", 0.5, 15),
        ("mpeg", ".mpg", 5, 15),
        ("mpeg", ".bin", 0.5, 15),
        ("mpegts", ".ts", 0, 300),
        ("mpegts", ".ts", 5, 300),
        ("mpegts", ".bin", 0, 300),
    ],
)
def test_delayed_and_long_gop_video_survives_analysis_and_conversion(
    tmp_path, container, suffix, delay, gop
):
    source = tmp_path / f"source{suffix}"
    output = tmp_path / "output.mp4"
    video_codec = "mpeg2video" if container == "mpeg" else "libx264"
    audio_codec = "libmp3lame" if container == "mpeg" else "aac"
    _ffmpeg(
        "-itsoffset",
        delay,
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=96x64:rate=30:duration=2",
        "-f",
        "lavfi",
        "-i",
        f"sine=duration={delay + 2}",
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-fps_mode",
        "passthrough",
        "-g",
        gop,
        "-c:v",
        video_codec,
        "-c:a",
        audio_codec,
        "-f",
        container,
        source,
    )
    meta, stats = analyzer.analyze(
        source,
        Path(FFPROBE),
        Path(FFMPEG),
        sample_count=24,
        sample_dir=tmp_path / "samples",
    )
    assert meta.framerate_status == "cfr"
    assert stats.sample_count == 24
    _process(tmp_path, source, output, meta)
    before, after = _probe(source), _probe(output)
    for probe in (before, after):
        video = next(s for s in probe["streams"] if s["codec_type"] == "video")
        assert int(video["nb_read_frames"]) == 60
    expected = _stream_start(before, "video") - _stream_start(before, "audio")
    actual = _stream_start(after, "video") - _stream_start(after, "audio")
    assert actual == pytest.approx(expected, abs=1 / 30)
