"""Behavioral regressions found in the 0.2.0 functional review."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from video_enhance_lite import analyzer, config, decisions, pipeline, stabilize_cv
from tests.enhance.test_analyzer import _fake_ffprobe_factory
from tests.enhance.test_decisions import FakeMeta, FakeStats
from tests.enhance.test_pipeline import _make_decisions


def test_stabilization_reduces_measured_camera_translation(tmp_path):
    source = tmp_path / "frames"
    output = tmp_path / "stabilized"
    source.mkdir()
    base = np.random.default_rng(428).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    for i in range(60):
        dx = 4 if i % 2 else -4
        frame = cv2.warpAffine(
            base,
            np.float32([[1, 0, dx], [0, 1, 0]]),
            (256, 256),
            borderMode=cv2.BORDER_REFLECT,
        )
        assert cv2.imwrite(str(source / f"{i:08d}.png"), frame)
    stabilize_cv.stabilize_chunk(source, output, crop_border=False)
    reference = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def displacement_std(directory):
        positions = [
            cv2.phaseCorrelate(reference, cv2.imread(str(p), 0).astype(np.float32))[0][0]
            for p in sorted(directory.glob("*.png"))
        ]
        assert len(positions) == 60
        return np.std(positions[10:50])

    assert displacement_std(output) < displacement_std(source) * 0.25


def test_stabilization_follows_smoothed_rotation_about_the_image_centre(tmp_path):
    """Rotation shake is corrected about the centre, toward the smoothed camera path."""
    source = tmp_path / "frames"
    output = tmp_path / "stabilized"
    source.mkdir()
    rng = np.random.default_rng(7)
    canvas = cv2.GaussianBlur(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8), (5, 5), 0)

    def render(angle):
        matrix = cv2.getRotationMatrix2D((320, 240), angle, 1.0)
        rotated = cv2.warpAffine(canvas, matrix, (640, 480), borderMode=cv2.BORDER_REFLECT)
        return rotated[120:360, 160:480]

    angles = np.array([1.5 if i % 2 else -1.5 for i in range(90)])
    for i, angle in enumerate(angles):
        assert cv2.imwrite(str(source / f"{i:08d}.png"), render(angle))
    stabilize_cv.stabilize_chunk(source, output, crop_border=False)

    radius = stabilize_cv.DEFAULT_SMOOTHING_RADIUS
    padded = np.pad(angles, radius, mode="edge")
    smoothed = np.convolve(padded, np.ones(2 * radius + 1) / (2 * radius + 1), mode="valid")
    ideal = [render(angle).astype(np.float32)[40:-40, 40:-40] for angle in smoothed]

    def mean_error(directory):
        frames = [cv2.imread(str(p)).astype(np.float32) for p in sorted(directory.glob("*.png"))]
        assert len(frames) == 90
        return float(
            np.mean([np.mean(np.abs(f[40:-40, 40:-40] - r)) for f, r in zip(frames, ideal)])
        )

    # Measured: unstabilized 14.9, origin-based rotation 4.3, centre-based 2.7.
    assert mean_error(output) < 3.5


@pytest.mark.parametrize("blocked_by_file", [True, False])
def test_invalid_output_fails_before_processing(tmp_path, monkeypatch, fake_run, blocked_by_file):
    output = tmp_path / "output" / "video.mp4"
    if blocked_by_file:
        output.parent.write_bytes(b"existing user file")
    else:

        def deny_probe(*args, **kwargs):
            raise PermissionError("write denied")

        monkeypatch.setattr(config.tempfile, "TemporaryFile", deny_probe)
    with pytest.raises(pipeline.PipelineError, match="output_preflight"):
        pipeline.run_pipeline(
            input_path=tmp_path / "input.mp4",
            output_path=output,
            decisions=_make_decisions(),
            meta=FakeMeta(),
            runner=fake_run,
            ffmpeg=Path("ffmpeg"),
            ffprobe=Path("ffprobe"),
        )
    assert not fake_run.calls
    if blocked_by_file:
        assert output.parent.read_bytes() == b"existing user file"


def test_output_preflight_does_not_truncate_existing_output(tmp_path):
    output = tmp_path / "output.mp4"
    output.write_bytes(b"existing movie")
    assert config.prepare_output_path(output) == output
    assert output.read_bytes() == b"existing movie"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("luma,sign", [(20, 1), (240, -1)])
def test_exposure_recommendation_has_bounded_non_neutral_parameters(luma, sign):
    dec = decisions.decisions_by_key(decisions.recommend(FakeMeta(), FakeStats(luma_mean=luma)))[
        "color_correction"
    ]
    assert dec.enabled
    assert 0 < sign * dec.params["brightness"] <= 0.15


def test_saturation_recommendation_and_neutral_material():
    low = decisions.decisions_by_key(
        decisions.recommend(FakeMeta(), FakeStats(saturation_mean=10))
    )["color_correction"]
    assert low.enabled
    assert 1 < low.params["saturation"] <= 1.25
    for saturation in (0, 80):
        normal = decisions.decisions_by_key(
            decisions.recommend(FakeMeta(), FakeStats(saturation_mean=saturation))
        )["color_correction"]
        assert not normal.enabled
        assert normal.params == {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0}


@pytest.mark.parametrize(
    "rotation,expected",
    [(90, (64, 96)), (-90, (64, 96)), (270, (64, 96)), (180, (96, 64)), (0, (96, 64))],
)
@pytest.mark.parametrize("legacy_tag", [False, True])
def test_metadata_uses_autorotated_dimensions(
    monkeypatch, tmp_path, rotation, expected, legacy_tag
):
    stream = {"width": 96, "height": 64, "r_frame_rate": "30/1", "duration": "1"}
    if legacy_tag:
        stream["tags"] = {"rotate": str(rotation)}
    else:
        stream["side_data_list"] = [{"rotation": rotation}]
    monkeypatch.setattr(analyzer, "_run", _fake_ffprobe_factory({"streams": [stream]}))
    meta = analyzer.probe_metadata(Path("ffprobe"), tmp_path / "in.mp4")
    assert (meta.width, meta.height) == expected


def test_video_duration_does_not_follow_longer_audio(monkeypatch, tmp_path):
    payload = {
        "streams": [{"width": 96, "height": 64, "r_frame_rate": "30/1", "duration": "1"}],
        "format": {"duration": "2"},
    }
    monkeypatch.setattr(analyzer, "_run", _fake_ffprobe_factory(payload))
    meta = analyzer.probe_metadata(Path("ffprobe"), tmp_path / "in.mp4")
    assert meta.duration_sec == 1
    assert meta.container_duration_sec == 2


def test_missing_stream_duration_uses_video_packets_not_container(monkeypatch, tmp_path):
    payload = {
        "streams": [{"width": 96, "height": 64, "r_frame_rate": "30/1"}],
        "format": {"duration": "20"},
    }
    default_run = _fake_ffprobe_factory(payload)

    def probe(argv, **kwargs):
        if "-show_packets" in argv:
            # Out-of-order PTS (B frames) and a nonzero timestamp origin.
            return SimpleNamespace(
                stdout="pts_time=3.0|duration_time=0.5\n"
                "pts_time=2.0|duration_time=0.5\n"
                "pts_time=2.5|duration_time=0.5\n"
            )
        return default_run(argv)

    monkeypatch.setattr(analyzer, "_run", probe)
    meta = analyzer.probe_metadata(Path("ffprobe"), tmp_path / "in.mkv")
    assert meta.duration_sec == 1.5
    assert meta.container_duration_sec == 20
