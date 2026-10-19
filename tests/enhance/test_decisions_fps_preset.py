from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


from video_enhance_lite import config, decisions


@dataclass
class FakeMeta:
    width: int = 1280
    height: int = 720
    fps: Fraction = Fraction(30, 1)
    duration_sec: float = 10.0
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


def _interp(decs):
    return next(d for d in decs if d.key == "interpolate")


def _explicit_defaults(target_fps: float) -> dict:
    return {
        "target_fps": Fraction(target_fps).limit_denominator(10000),
        "target_fps_explicit": True,
    }


# ---------- preset 明示時 (source < target) ----------


def test_source_30_target_60_doubles_on():
    decs = decisions.recommend(
        FakeMeta(fps=Fraction(30, 1)), FakeStats(), defaults=_explicit_defaults(60.0)
    )
    d = _interp(decs)
    assert d.recommended is True
    assert d.enabled is True  # プリセット明示 = 明示同意
    f = Fraction(d.params["target_fps_num"], d.params["target_fps_den"])
    assert f == Fraction(60, 1)


def test_source_24_target_60_rounds_to_48_with_warn():
    """24 -> 60: 60/24=2.5 -> 2倍=48 で丸め (warning)"""
    decs = decisions.recommend(
        FakeMeta(fps=Fraction(24, 1)), FakeStats(), defaults=_explicit_defaults(60.0)
    )
    d = _interp(decs)
    assert d.recommended is True
    assert d.enabled is True
    f = Fraction(d.params["target_fps_num"], d.params["target_fps_den"])
    assert f == Fraction(48, 1)
    assert "warn" in d.reason.lower()


def test_source_25_target_48_rounds_to_50_with_warn():
    """25 -> 48: 48/25=1.92 -> 2倍=50 で丸め (warning)"""
    decs = decisions.recommend(
        FakeMeta(fps=Fraction(25, 1)), FakeStats(), defaults=_explicit_defaults(48.0)
    )
    d = _interp(decs)
    assert d.recommended is True
    f = Fraction(d.params["target_fps_num"], d.params["target_fps_den"])
    assert f == Fraction(50, 1)
    assert "warn" in d.reason.lower()


# ---------- preset 明示時 (source >= target → OFF) ----------


def test_source_60_target_60_equal_off():
    """境界値 == も OFF (§2.4 / 利用者承認 #3)"""
    decs = decisions.recommend(
        FakeMeta(fps=Fraction(60, 1)), FakeStats(), defaults=_explicit_defaults(60.0)
    )
    d = _interp(decs)
    assert d.recommended is False
    assert d.enabled is False
    assert "補間しません" in d.reason


def test_source_60_target_48_source_greater_off():
    """source > target も OFF"""
    decs = decisions.recommend(
        FakeMeta(fps=Fraction(60, 1)), FakeStats(), defaults=_explicit_defaults(48.0)
    )
    d = _interp(decs)
    assert d.recommended is False
    assert d.enabled is False
    assert "補間しません" in d.reason


# ---------- keep (現状維持): FPS_KEEP_SOURCE -> source_fps に解決 ----------


def test_keep_source_fps_interpolate_off():
    """keep センチネル -> target_fps=source に解決され interpolate OFF (source<30 でも)."""
    decs = decisions.recommend(
        FakeMeta(fps=Fraction(24, 1)),
        FakeStats(),
        defaults=_explicit_defaults(config.FPS_KEEP_SOURCE),
    )
    d = _interp(decs)
    assert d.recommended is False
    assert d.enabled is False
    assert "補間しません" in d.reason


def test_keep_source_fps_differs_from_auto():
    """24fps 入力で auto は ×2 補間 ON、keep は OFF になることを対比で確認."""
    auto = _interp(decisions.recommend(FakeMeta(fps=Fraction(24, 1)), FakeStats()))
    keep = _interp(
        decisions.recommend(
            FakeMeta(fps=Fraction(24, 1)),
            FakeStats(),
            defaults=_explicit_defaults(config.FPS_KEEP_SOURCE),
        )
    )
    assert auto.recommended is True  # auto -> ×2 補間
    assert keep.recommended is False  # keep -> 現状維持
