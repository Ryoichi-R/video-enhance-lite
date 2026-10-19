from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import pytest

from video_enhance_lite import (
    config,
    decisions as decisions_mod,
    pipeline,
    runtime_estimator,
    stages,
)
from video_enhance_lite.decisions import StageDecision, decisions_by_key


@dataclass
class FakeMeta:
    width: int = 1280
    height: int = 720
    fps: Fraction = Fraction(30, 1)
    duration_sec: float = 60.0
    codec: str = "h264"
    bitrate_bps: int | None = 5_000_000
    pix_fmt: str = "yuv420p"
    color_range: str = "tv"
    color_space: str = "bt709"
    field_order: str = "progressive"
    has_audio: bool = True


@dataclass
class FakeStats:
    laplacian_var_mean: float = 200.0
    luma_std_over_time: float = 1.0
    interframe_diff_mean: float = 1.0
    interframe_diff_std: float = 1.0
    histogram_flat_ratio: float = 0.05
    luma_mean: float = 120.0
    luma_clipping_ratio: float = 0.001
    saturation_mean: float = 80.0
    highfreq_energy: float = 5.0
    sample_count: int = 24


def _decs(meta=None, stats=None, defaults=None) -> list[StageDecision]:
    return decisions_mod.recommend(meta or FakeMeta(), stats or FakeStats(), defaults or {})


def _all_off(decs: list[StageDecision]) -> list[StageDecision]:
    for d in decs:
        d.enabled = d.key == "encode"
    return decs


def _only(decs: list[StageDecision], *keys: str) -> list[StageDecision]:
    for d in decs:
        d.enabled = d.key in keys or d.key == "encode"
    return decs


# ---------- 基本ケース ----------


def test_estimate_runtime_upscale_only():
    decs = _decs(FakeMeta(width=1280, height=720, fps=Fraction(30, 1), duration_sec=60.0))
    by = decisions_by_key(decs)
    _all_off(decs)
    by["upscale"].enabled = True
    by["upscale"].params["scale"] = 2
    by["upscale"].params["target_width"] = 2560
    est = runtime_estimator.estimate_runtime(FakeMeta(), decs)
    assert "upscale_scale2" in est.per_stage_seconds
    # 想定範囲: 数十秒〜数分
    assert 10.0 < est.total_seconds < 600.0


def test_estimate_runtime_full_pipeline():
    meta = FakeMeta(width=640, height=480, field_order="tt", fps=Fraction(24, 1))
    decs = _decs(
        meta,
        FakeStats(
            interframe_diff_std=50.0,
            luma_std_over_time=50.0,
            highfreq_energy=50.0,
            histogram_flat_ratio=0.5,
            laplacian_var_mean=10.0,
            luma_mean=30.0,
        ),
    )
    for d in decs:
        d.enabled = d.recommended or d.key == "encode"
    by = decisions_by_key(decs)
    by["upscale"].enabled = True
    by["interpolate"].enabled = True
    est = runtime_estimator.estimate_runtime(meta, decs)
    for k in ("extract", "pre_filter", "post_filter", "mux", "interpolate"):
        assert k in est.per_stage_seconds, k
    assert any(k.startswith("upscale_scale") for k in est.per_stage_seconds)


def test_estimate_runtime_interpolate_doubles_post_frames():
    meta = FakeMeta(width=1920, height=1080, fps=Fraction(30, 1), duration_sec=30.0)
    base = _only(
        _decs(meta),
    )  # only encode -> mux
    est_no = runtime_estimator.estimate_runtime(meta, base)
    decs = _decs(meta)
    _all_off(decs)
    by = decisions_by_key(decs)
    by["interpolate"].enabled = True
    by["interpolate"].params["target_fps_num"] = 60
    by["interpolate"].params["target_fps_den"] = 1
    est_interp = runtime_estimator.estimate_runtime(meta, decs)
    # mux は f_out 基準なので ×2 になる
    assert est_interp.per_stage_seconds["mux"] == pytest.approx(
        est_no.per_stage_seconds["mux"] * 2, rel=0.01
    )


def test_estimate_runtime_resolution_clamp_uses_intermediate_px():
    # src=1920x1080, scale=4 (intermediate 7680), target_width=1920 -> 下方 clamp
    meta = FakeMeta(width=1920, height=1080, fps=Fraction(30, 1), duration_sec=10.0)
    decs = _decs(meta)
    _all_off(decs)
    by = decisions_by_key(decs)
    by["upscale"].enabled = True
    by["upscale"].params["scale"] = 4
    by["upscale"].params["target_width"] = 1920
    est = runtime_estimator.estimate_runtime(meta, decs)
    assert "resolution_clamp" in est.per_stage_seconds
    # intermediate_px (7680x4320) 基準なので src 基準より遥かに大きい比率
    clamp_sec = est.per_stage_seconds["resolution_clamp"]
    f_in = est.assumed_frames_in
    inter_ratio = (7680 * 4320) / config.STAGE_BASELINE_RESOLUTION_PX
    expected = f_in * config.STAGE_BASELINE_FRAME_COST_SEC["resolution_clamp"] * inter_ratio
    assert clamp_sec == pytest.approx(expected, rel=0.01)


def test_estimate_runtime_scale_factor_env(monkeypatch):
    meta = FakeMeta()
    decs = _decs(meta)
    base = runtime_estimator.estimate_runtime(meta, decs, scale_factor=1.0)
    monkeypatch.setenv(config.ENV_RUNTIME_SCALE, "2.0")
    scaled = runtime_estimator.estimate_runtime(meta, decs)
    assert scaled.total_seconds == pytest.approx(base.total_seconds * 2.0, rel=0.001)
    assert scaled.assumed_scale_factor == 2.0


def test_estimate_runtime_scale_factor_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv(config.ENV_RUNTIME_SCALE, "abc")
    est = runtime_estimator.estimate_runtime(FakeMeta(), _decs())
    assert est.assumed_scale_factor == 1.0
    assert est.warnings


def test_estimate_runtime_scale_factor_out_of_range_clamped(monkeypatch):
    monkeypatch.setenv(config.ENV_RUNTIME_SCALE, "999")
    est = runtime_estimator.estimate_runtime(FakeMeta(), _decs())
    assert est.assumed_scale_factor == 10.0
    assert any("範囲外" in w for w in est.warnings)


@pytest.mark.parametrize(
    "sec,expected",
    [
        (0, "0s"),
        (59, "59s"),
        (60, "1m0s"),
        (3599, "59m59s"),
        (3600, "1h0m"),
        (36000, "10h0m"),
        (9213, "2h33m"),
        (73, "1m13s"),
        (4, "4s"),
    ],
)
def test_format_eta_hms_boundaries(sec, expected):
    assert runtime_estimator.format_eta_hms(sec) == expected


# ---------- 仮想ステージ判定 (§1.2.0 マップ) ----------


def test_pre_filter_only_deinterlace():
    meta = FakeMeta(field_order="tt")
    decs = _only(_decs(meta), "deinterlace")
    est = runtime_estimator.estimate_runtime(meta, decs)
    assert "pre_filter" in est.per_stage_seconds
    assert "post_filter" not in est.per_stage_seconds


def test_pre_filter_only_stabilize():
    meta = FakeMeta()
    decs = _only(_decs(meta), "stabilize")
    est = runtime_estimator.estimate_runtime(meta, decs)
    assert "pre_filter" in est.per_stage_seconds
    assert "stabilize_detect" in est.per_stage_seconds


def test_post_filter_only_color_correction():
    meta = FakeMeta()
    decs = _only(_decs(meta), "color_correction")
    est = runtime_estimator.estimate_runtime(meta, decs)
    assert "post_filter" in est.per_stage_seconds
    assert "pre_filter" not in est.per_stage_seconds


def test_post_filter_only_grain():
    meta = FakeMeta()
    decs = _only(_decs(meta), "grain")
    est = runtime_estimator.estimate_runtime(meta, decs)
    assert "post_filter" in est.per_stage_seconds


def test_resolution_clamp_applied():
    meta = FakeMeta(width=1920, height=1080)
    decs = _decs(meta)
    _all_off(decs)
    by = decisions_by_key(decs)
    by["upscale"].enabled = True
    by["upscale"].params["scale"] = 4  # intermediate 7680
    by["upscale"].params["target_width"] = 3840
    est = runtime_estimator.estimate_runtime(meta, decs)
    assert "resolution_clamp" in est.per_stage_seconds


def test_resolution_clamp_skipped():
    meta = FakeMeta(width=1920, height=1080)
    decs = _decs(meta)
    _all_off(decs)
    by = decisions_by_key(decs)
    by["upscale"].enabled = True
    by["upscale"].params["scale"] = 2  # intermediate 3840
    by["upscale"].params["target_width"] = 3840
    est = runtime_estimator.estimate_runtime(meta, decs)
    assert "resolution_clamp" not in est.per_stage_seconds


def test_mux_default_on():
    meta = FakeMeta()
    decs = _decs(meta)
    _all_off(decs)  # encode は ON のまま
    est = runtime_estimator.estimate_runtime(meta, decs)
    assert "mux" in est.per_stage_seconds


def test_mux_counted_even_without_encode_decision():
    meta = FakeMeta()
    decs = [d for d in _decs(meta) if d.key != "encode"]
    _all_off(decs)
    est = runtime_estimator.estimate_runtime(meta, decs)
    assert "mux" in est.per_stage_seconds


def test_stabilize_detect_follows_stabilize_decision():
    meta = FakeMeta()
    on = _only(_decs(meta), "stabilize")
    assert "stabilize_detect" in runtime_estimator.estimate_runtime(meta, on).per_stage_seconds
    off = _all_off(_decs(meta))
    assert "stabilize_detect" not in runtime_estimator.estimate_runtime(meta, off).per_stage_seconds


def test_face_restore_eta_reflects_range_with_minimum_factor():
    meta = FakeMeta(duration_sec=100.0)
    full = _only(_decs(meta), "face_restore")
    ranged = _only(_decs(meta), "face_restore")
    decisions_by_key(ranged)["face_restore"].params["start_sec"] = 30.0
    decisions_by_key(ranged)["face_restore"].params["end_sec"] = 40.0

    est_full = runtime_estimator.estimate_runtime(meta, full)
    est_ranged = runtime_estimator.estimate_runtime(meta, ranged)

    assert est_ranged.per_stage_seconds["face_restore"] == pytest.approx(
        est_full.per_stage_seconds["face_restore"] * 0.25,
        rel=0.001,
    )
    assert any("範囲指定" in w for w in est_ranged.warnings)


def test_face_restore_eta_uses_one_sided_range_ratio():
    meta = FakeMeta(duration_sec=100.0)
    full = _only(_decs(meta), "face_restore")
    ranged = _only(_decs(meta), "face_restore")
    decisions_by_key(ranged)["face_restore"].params["start_sec"] = 40.0
    decisions_by_key(ranged)["face_restore"].params["end_sec"] = None

    est_full = runtime_estimator.estimate_runtime(meta, full)
    est_ranged = runtime_estimator.estimate_runtime(meta, ranged)

    assert est_ranged.per_stage_seconds["face_restore"] == pytest.approx(
        est_full.per_stage_seconds["face_restore"] * 0.60,
        rel=0.001,
    )


def test_face_restore_eta_invalid_range_falls_back_to_full():
    meta = FakeMeta(duration_sec=100.0)
    full = _only(_decs(meta), "face_restore")
    invalid = _only(_decs(meta), "face_restore")
    decisions_by_key(invalid)["face_restore"].params["start_sec"] = 90.0
    decisions_by_key(invalid)["face_restore"].params["end_sec"] = 80.0

    est_full = runtime_estimator.estimate_runtime(meta, full)
    est_invalid = runtime_estimator.estimate_runtime(meta, invalid)

    assert est_invalid.per_stage_seconds["face_restore"] == pytest.approx(
        est_full.per_stage_seconds["face_restore"],
        rel=0.001,
    )
    assert not est_invalid.warnings


def test_face_restore_range_does_not_change_peak_estimate():
    meta = FakeMeta(duration_sec=100.0)
    full = _only(_decs(meta), "face_restore")
    ranged = _only(_decs(meta), "face_restore")
    decisions_by_key(ranged)["face_restore"].params["start_sec"] = 30.0
    decisions_by_key(ranged)["face_restore"].params["end_sec"] = 40.0

    assert runtime_estimator.estimate_peak_for_decisions(meta, full, 10) == (
        runtime_estimator.estimate_peak_for_decisions(meta, ranged, 10)
    )


def test_virtual_stage_keys_track_implementation(monkeypatch):
    """PRE_FILTER_ORDER を書き換えても runtime_estimator が追従する (ハードコード防止)。"""
    meta = FakeMeta()
    decs = _only(_decs(meta), "color_correction")  # color_correction は本来 POST
    # POST_FILTER_ORDER から color_correction を外し、PRE 側に移す
    monkeypatch.setattr(
        stages,
        "PRE_FILTER_ORDER",
        ("deinterlace", "stabilize", "deflicker", "denoise", "debanding", "color_correction"),
    )
    monkeypatch.setattr(stages, "POST_FILTER_ORDER", ("sharpen", "grain"))
    est = runtime_estimator.estimate_runtime(meta, decs)
    # color_correction が PRE 扱いになったので pre_filter が計上され post_filter は計上されない
    assert "pre_filter" in est.per_stage_seconds
    assert "post_filter" not in est.per_stage_seconds


# ---------- estimate_peak_for_decisions 整合性 ----------


def test_estimate_peak_matches_cli_inline_logic():
    meta = FakeMeta(width=1920, height=1080, fps=Fraction(30, 1), duration_sec=120.0)
    decs = _decs(meta)
    by = decisions_by_key(decs)
    by["upscale"].enabled = True
    by["upscale"].params["scale"] = 2
    by["interpolate"].enabled = True
    by["interpolate"].params["target_fps_num"] = 60
    by["interpolate"].params["target_fps_den"] = 1

    chunk_seconds = 10
    png_ratio = 0.80
    worker_count = 2

    peak_gib, iw, eff = runtime_estimator.estimate_peak_for_decisions(
        meta,
        decs,
        chunk_seconds,
        png_compression_ratio=png_ratio,
        worker_count=worker_count,
    )
    # cli inline ロジック相当
    inline_iw = meta.width * 2
    inline_eff = Fraction(60, 1)
    inline_bytes = pipeline.estimate_peak_temp_bytes(
        meta.width,
        meta.height,
        meta.fps,
        chunk_seconds,
        intermediate_width=inline_iw,
        effective_fps=inline_eff,
        png_compression_ratio=png_ratio,
        worker_count=worker_count,
    )
    assert iw == inline_iw
    assert eff == inline_eff
    assert peak_gib == pytest.approx(inline_bytes / (1024**3))


def test_estimate_peak_matches_gui_inline_logic():
    meta = FakeMeta(width=1280, height=720, fps=Fraction(24, 1), duration_sec=30.0)
    decs = _decs(meta)
    _all_off(decs)  # upscale/interpolate OFF
    peak_gib, iw, eff = runtime_estimator.estimate_peak_for_decisions(meta, decs, 10)
    assert iw is None
    assert eff is None
    inline_bytes = pipeline.estimate_peak_temp_bytes(
        meta.width,
        meta.height,
        meta.fps,
        10,
        intermediate_width=None,
        effective_fps=None,
        png_compression_ratio=pipeline.DEFAULT_PNG_COMPRESSION_RATIO,
        worker_count=1,
    )
    assert peak_gib == pytest.approx(inline_bytes / (1024**3))


def test_estimate_peak_uses_stats_compression_ratio_when_no_explicit():
    meta = FakeMeta()
    decs = _all_off(_decs(meta))
    anime_stats = FakeStats(histogram_flat_ratio=0.5)  # -> 0.65
    peak_gib, _, _ = runtime_estimator.estimate_peak_for_decisions(
        meta,
        decs,
        10,
        stats=anime_stats,
    )
    inline_bytes = pipeline.estimate_peak_temp_bytes(
        meta.width,
        meta.height,
        meta.fps,
        10,
        png_compression_ratio=0.65,
        worker_count=1,
    )
    assert peak_gib == pytest.approx(inline_bytes / (1024**3))
