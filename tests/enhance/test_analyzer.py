from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from video_enhance_lite import analyzer


# ---------- probe_metadata ----------


def _fake_ffprobe_factory(payload_video: dict, payload_audio: dict | None = None):
    """Return a fake subprocess.run that emits ffprobe-shaped JSON."""

    def _fake_run(argv, check=True, capture_output=False, text=False, shell=False, **kwargs):
        if "-select_streams" in argv:
            kind = argv[argv.index("-select_streams") + 1]
            if kind.startswith("v"):
                payload = payload_video
            else:
                payload = payload_audio or {"streams": []}
        else:
            payload = payload_audio or {"streams": []}
        return SimpleNamespace(stdout=json.dumps(payload), stderr="", returncode=0)

    return _fake_run


def test_probe_metadata_basic(monkeypatch, tmp_path):
    video = {
        "streams": [
            {
                "width": 1280,
                "height": 720,
                "r_frame_rate": "30/1",
                "avg_frame_rate": "30/1",
                "duration": "12.0",
                "codec_name": "h264",
                "bit_rate": "2500000",
                "pix_fmt": "yuv420p",
                "color_range": "tv",
                "color_space": "bt709",
                "field_order": "progressive",
            }
        ],
        "format": {"duration": "12.0", "bit_rate": "2500000"},
    }
    audio = {
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264"},
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "tags": {"language": "jpn"},
                "disposition": {"default": 1, "forced": 0},
            },
        ]
    }
    monkeypatch.setattr(analyzer, "_run", _fake_ffprobe_factory(video, audio))
    meta = analyzer.probe_metadata(Path("ffprobe"), tmp_path / "in.mp4")
    assert meta.width == 1280
    assert meta.height == 720
    assert int(meta.fps) == 30
    assert meta.duration_sec == pytest.approx(12.0)
    assert meta.codec == "h264"
    assert meta.bitrate_bps == 2_500_000
    assert meta.pix_fmt == "yuv420p"
    assert meta.field_order == "progressive"
    assert meta.has_audio is True
    assert meta.primary_video_index == 0
    assert meta.streams is not None
    assert meta.streams[1].language == "jpn"
    assert meta.streams[1].disposition == ("default",)


def test_probe_metadata_interlaced(monkeypatch, tmp_path):
    video = {
        "streams": [
            {
                "width": 720,
                "height": 480,
                "r_frame_rate": "30000/1001",
                "avg_frame_rate": "30000/1001",
                "duration": "60.0",
                "codec_name": "mpeg2video",
                "pix_fmt": "yuv420p",
                "field_order": "tt",
            }
        ],
        "format": {"duration": "60.0"},
    }
    monkeypatch.setattr(analyzer, "_run", _fake_ffprobe_factory(video, {"streams": []}))
    meta = analyzer.probe_metadata(Path("ffprobe"), tmp_path / "in.mp4")
    assert meta.field_order == "tt"
    assert meta.has_audio is False
    # 30000/1001 を Fraction として保持
    assert meta.fps.denominator == 1001


def test_framerate_classifier_accepts_bframe_presentation_timestamps(monkeypatch, tmp_path):
    windows = [
        [base + offset for offset in (0.0, 0.033, 0.067, 0.100, 0.133, 0.167)]
        for base in (0.0, 5.0, 9.0)
    ]
    monkeypatch.setattr(analyzer, "_probe_frame_timestamp_windows", lambda *args: windows)
    status = analyzer._classify_framerate(
        Path("ffprobe"),
        tmp_path / "in.mp4",
        10.0,
        Fraction(30, 1),
        Fraction(30, 1),
    )
    assert status == "cfr"


def test_framerate_classifier_rejects_variable_deltas(monkeypatch, tmp_path):
    windows = [[0.0, 0.033, 0.067, 0.117, 0.150, 0.183]] * 3
    monkeypatch.setattr(analyzer, "_probe_frame_timestamp_windows", lambda *args: windows)
    assert (
        analyzer._classify_framerate(
            Path("ffprobe"),
            tmp_path / "in.mp4",
            10.0,
            Fraction(30),
            Fraction(30),
        )
        == "vfr_suspected"
    )


def test_framerate_classifier_allows_small_rate_rounding(monkeypatch, tmp_path):
    windows = [[0.0, 0.033, 0.067, 0.100, 0.133, 0.167]] * 3
    monkeypatch.setattr(analyzer, "_probe_frame_timestamp_windows", lambda *args: windows)
    assert (
        analyzer._classify_framerate(
            Path("ffprobe"),
            tmp_path / "in.mp4",
            10.0,
            Fraction(30000, 1001),
            Fraction(2997, 100),
        )
        == "cfr"
    )


def test_framerate_classifier_unknown_for_non_monotonic_frame_timestamps(monkeypatch, tmp_path):
    windows = [[0.0, 0.033, 0.020, 0.067, 0.100, 0.133]] * 3
    monkeypatch.setattr(analyzer, "_probe_frame_timestamp_windows", lambda *args: windows)
    assert (
        analyzer._classify_framerate(
            Path("ffprobe"),
            tmp_path / "in.mp4",
            10.0,
            Fraction(30),
            Fraction(30),
        )
        == "unknown"
    )


def test_framerate_classifier_unknown_when_avg_rate_is_missing(monkeypatch, tmp_path):
    stable = [[0.0, 0.033, 0.067, 0.100, 0.133]] * 3
    monkeypatch.setattr(analyzer, "_probe_frame_timestamp_windows", lambda *_args: stable)
    assert (
        analyzer._classify_framerate(
            Path("ffprobe"),
            tmp_path / "in.mp4",
            3.0,
            Fraction(30),
            None,
        )
        == "unknown"
    )


def test_probe_metadata_no_video_stream(monkeypatch, tmp_path):
    monkeypatch.setattr(analyzer, "_run", _fake_ffprobe_factory({"streams": []}, {"streams": []}))
    with pytest.raises(analyzer.AnalyzerError, match="no video stream"):
        analyzer.probe_metadata(Path("ffprobe"), tmp_path / "in.mp4")


def test_probe_metadata_invalid_duration(monkeypatch, tmp_path):
    video = {
        "streams": [{"width": 100, "height": 100, "r_frame_rate": "30/1", "duration": "-1"}],
        "format": {"duration": "-1"},
    }
    monkeypatch.setattr(analyzer, "_run", _fake_ffprobe_factory(video, {"streams": []}))
    with pytest.raises(analyzer.AnalyzerError, match="invalid duration"):
        analyzer.probe_metadata(Path("ffprobe"), tmp_path / "in.mp4")


def test_probe_metadata_invalid_resolution(monkeypatch, tmp_path):
    video = {
        "streams": [{"width": 0, "height": 0, "r_frame_rate": "30/1", "duration": "5"}],
        "format": {"duration": "5"},
    }
    monkeypatch.setattr(analyzer, "_run", _fake_ffprobe_factory(video, {"streams": []}))
    with pytest.raises(analyzer.AnalyzerError, match="invalid resolution"):
        analyzer.probe_metadata(Path("ffprobe"), tmp_path / "in.mp4")


def test_probe_metadata_subprocess_failure(monkeypatch, tmp_path):
    def _fail(argv, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=argv, stderr="boom")

    monkeypatch.setattr(analyzer, "_run", _fail)
    with pytest.raises(analyzer.AnalyzerError, match="ffprobe.*failed"):
        analyzer.probe_metadata(Path("ffprobe"), tmp_path / "in.mp4")


# ---------- compute_frame_stats ----------


def test_extract_samples_skips_only_missing_final_sample(monkeypatch, tmp_path):
    warnings: list[str] = []
    progress: list[tuple[int, int]] = []

    def _fake_run(argv):
        out_path = Path(argv[-1])
        if out_path.name != "sample_0003.png":
            out_path.touch()

    monkeypatch.setattr(analyzer, "_run", _fake_run)

    samples = analyzer._extract_samples(
        ffmpeg=Path("ffmpeg"),
        input_path=tmp_path / "in.mp4",
        duration_sec=1.0,
        sample_count=4,
        out_dir=tmp_path / "samples",
        cancel_event=None,
        progress=lambda done, total: progress.append((done, total)),
        warning=warnings.append,
    )

    assert [path.name for path in samples] == [
        "sample_0000.png",
        "sample_0001.png",
        "sample_0002.png",
    ]
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert len(warnings) == 1
    assert "continuing with 3 samples" in warnings[0]


def test_extract_samples_keeps_nonfinal_missing_sample_fatal(monkeypatch, tmp_path):
    def _fake_run(argv):
        out_path = Path(argv[-1])
        if out_path.name != "sample_0001.png":
            out_path.touch()

    monkeypatch.setattr(analyzer, "_run", _fake_run)

    with pytest.raises(analyzer.AnalyzerError, match="produced no file"):
        analyzer._extract_samples(
            ffmpeg=Path("ffmpeg"),
            input_path=tmp_path / "in.mp4",
            duration_sec=1.0,
            sample_count=4,
            out_dir=tmp_path / "samples",
            cancel_event=None,
            progress=None,
            warning=None,
        )


def _solid_frame(value: int, h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _gradient_frame(h: int = 64, w: int = 64) -> np.ndarray:
    g = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return np.stack([g, g, g], axis=2)


def _noisy_frame(seed: int, h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_compute_frame_stats_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        analyzer.compute_frame_stats([])


def test_compute_frame_stats_solid_color_blur_indicator():
    # 単色フレームは Laplacian 分散が極めて低い（=ブラー大）
    frames = [_solid_frame(128) for _ in range(4)]
    stats = analyzer.compute_frame_stats(frames)
    assert stats.sample_count == 4
    assert stats.laplacian_var_mean == pytest.approx(0.0, abs=1e-6)
    assert stats.luma_std_over_time == pytest.approx(0.0, abs=1e-6)
    assert stats.interframe_diff_mean == pytest.approx(0.0, abs=1e-6)
    assert stats.luma_mean == pytest.approx(128.0, abs=1e-6)


def test_compute_frame_stats_clipping_detection():
    # 黒潰れフレーム: 全 pixel が 0
    frames = [_solid_frame(0) for _ in range(3)]
    stats = analyzer.compute_frame_stats(frames)
    assert stats.luma_clipping_ratio == pytest.approx(1.0)
    assert stats.luma_mean == pytest.approx(0.0)


def test_compute_frame_stats_flicker_detection():
    # 明るさが時間方向で変動 -> luma_std_over_time が大きい
    frames = [_solid_frame(50), _solid_frame(200), _solid_frame(50), _solid_frame(200)]
    stats = analyzer.compute_frame_stats(frames)
    assert stats.luma_std_over_time > 50.0
    # 隣接 absdiff も大きい
    assert stats.interframe_diff_mean > 100.0


def test_compute_frame_stats_gradient_high_frequency_low():
    # なめらかなグラデーション: high-pass エネルギーは比較的小さい
    frames = [_gradient_frame() for _ in range(3)]
    stats = analyzer.compute_frame_stats(frames)
    assert stats.highfreq_energy < 5.0


def test_compute_frame_stats_noise_high_frequency_high():
    frames = [_noisy_frame(seed) for seed in range(4)]
    stats = analyzer.compute_frame_stats(frames)
    # ノイズフレームでは high-pass エネルギーが顕著
    assert stats.highfreq_energy > 20.0


def test_compute_frame_stats_histogram_flat_ratio_solid_high():
    # 単色 -> 1 bin に集中 = 残り 255 bin が 0 -> flat ratio ≈ 255/256
    frames = [_solid_frame(100) for _ in range(2)]
    stats = analyzer.compute_frame_stats(frames)
    assert stats.histogram_flat_ratio > 0.99


def test_compute_frame_stats_histogram_flat_ratio_gradient_low():
    frames = [_gradient_frame() for _ in range(2)]
    stats = analyzer.compute_frame_stats(frames)
    # 64 列のグラデーションは多くの bin に分布する
    assert stats.histogram_flat_ratio < 0.9


# ---------- face restore candidates ----------


def test_build_face_restore_candidate_ranges_merges_nearby_samples():
    detections = (
        analyzer.FaceDetectionSample(10.0, 1, 0.03, "medium"),
        analyzer.FaceDetectionSample(13.0, 1, 0.04, "medium"),
    )
    ranges = analyzer._build_face_restore_candidate_ranges(
        detections,
        duration_sec=60.0,
        sample_count=12,
    )
    assert len(ranges) == 1
    assert ranges[0].start_sec >= 0.0
    assert ranges[0].end_sec <= 60.0
    assert ranges[0].face_sample_count == 2
    assert ranges[0].confidence == "high"


def test_build_face_restore_candidate_ranges_clamps_and_caps_long_sample_window():
    detections = (analyzer.FaceDetectionSample(1.0, 1, 0.01, "low"),)
    ranges = analyzer._build_face_restore_candidate_ranges(
        detections,
        duration_sec=2400.0,
        sample_count=24,
    )
    assert len(ranges) == 1
    assert ranges[0].start_sec == pytest.approx(0.0)
    assert ranges[0].end_sec <= 6.0
    assert ranges[0].confidence == "low"
    assert "サンプル間隔" in ranges[0].note


# ---------- analyze (integration with mocked subprocess) ----------


def test_analyze_end_to_end(monkeypatch, tmp_path):
    """analyze() を probe_metadata と _extract_samples をモックして検証."""
    video = {
        "streams": [
            {
                "width": 1280,
                "height": 720,
                "r_frame_rate": "30/1",
                "avg_frame_rate": "30/1",
                "duration": "10.0",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "field_order": "progressive",
            }
        ],
        "format": {"duration": "10.0"},
    }
    audio = {"streams": []}

    sample_dir = tmp_path / "sd"

    # _extract_samples をモックして実際に numpy 画像をファイルに書く
    def _fake_extract(
        ffmpeg, input_path, duration_sec, sample_count, out_dir, cancel_event, progress, warning
    ):
        import cv2

        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(sample_count):
            p = out_dir / f"s_{i:04d}.png"
            cv2.imwrite(str(p), _gradient_frame())
            paths.append(p)
            if progress:
                progress(i + 1, sample_count)
        return paths

    monkeypatch.setattr(analyzer, "_run", _fake_ffprobe_factory(video, audio))
    monkeypatch.setattr(analyzer, "_extract_samples", _fake_extract)

    meta, stats = analyzer.analyze(
        input_path=tmp_path / "in.mp4",
        ffprobe=Path("ffprobe"),
        ffmpeg=Path("ffmpeg"),
        sample_count=4,
        sample_dir=sample_dir,
        keep_samples=False,
    )
    assert meta.width == 1280
    assert stats.sample_count == 4
    # cleanup されている
    assert not sample_dir.exists()


def test_analyze_face_candidate_timestamps_survive_unreadable_sample(monkeypatch, tmp_path):
    video = {
        "streams": [
            {
                "width": 1280,
                "height": 720,
                "r_frame_rate": "30/1",
                "avg_frame_rate": "30/1",
                "duration": "40.0",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "field_order": "progressive",
            }
        ],
        "format": {"duration": "40.0"},
    }

    def _fake_extract(
        ffmpeg, input_path, duration_sec, sample_count, out_dir, cancel_event, progress, warning
    ):
        import cv2

        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(sample_count):
            p = out_dir / f"s_{i:04d}.png"
            if i != 1:
                cv2.imwrite(str(p), _gradient_frame())
            else:
                p.write_text("not an image", encoding="utf-8")
            paths.append(p)
        return paths

    captured_timestamps: list[float] = []

    def _fake_detect(timestamped_frames, cancel_event=None):
        captured_timestamps.extend(t for t, _frame in timestamped_frames)
        return (analyzer.FaceDetectionSample(timestamped_frames[1][0], 1, 0.03, "medium"),)

    monkeypatch.setattr(analyzer, "_run", _fake_ffprobe_factory(video, {"streams": []}))
    monkeypatch.setattr(analyzer, "_extract_samples", _fake_extract)
    monkeypatch.setattr(analyzer, "_detect_face_samples", _fake_detect)

    _meta, stats = analyzer.analyze(
        input_path=tmp_path / "in.mp4",
        ffprobe=Path("ffprobe"),
        ffmpeg=Path("ffmpeg"),
        sample_count=4,
        sample_dir=tmp_path / "sd_timestamp",
    )

    assert captured_timestamps == pytest.approx([5.0, 25.0, 35.0])
    assert stats.face_restore_candidates
    assert stats.face_restore_candidates[0].start_sec == pytest.approx(20.0)


def test_analyze_keep_samples(monkeypatch, tmp_path):
    video = {
        "streams": [
            {
                "width": 640,
                "height": 360,
                "r_frame_rate": "24/1",
                "avg_frame_rate": "24/1",
                "duration": "5.0",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "field_order": "progressive",
            }
        ],
        "format": {"duration": "5.0"},
    }
    monkeypatch.setattr(analyzer, "_run", _fake_ffprobe_factory(video, {"streams": []}))

    def _fake_extract(
        ffmpeg, input_path, duration_sec, sample_count, out_dir, cancel_event, progress, warning
    ):
        import cv2

        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "s_0.png"
        cv2.imwrite(str(p), _solid_frame(120))
        # 2枚以上必要 (sample_count >= 2)
        p2 = out_dir / "s_1.png"
        cv2.imwrite(str(p2), _solid_frame(130))
        return [p, p2]

    sd = tmp_path / "sd2"
    monkeypatch.setattr(analyzer, "_extract_samples", _fake_extract)
    analyzer.analyze(
        input_path=tmp_path / "in.mp4",
        ffprobe=Path("ffprobe"),
        ffmpeg=Path("ffmpeg"),
        sample_count=2,
        sample_dir=sd,
        keep_samples=True,
    )
    assert sd.exists()
    # ファイルが残っている
    assert any(sd.iterdir())
