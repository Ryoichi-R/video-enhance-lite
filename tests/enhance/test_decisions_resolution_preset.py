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


def _upscale(decs):
    return next(d for d in decs if d.key == "upscale")


# ---------- preset 解像度 (needed = ceil(target/src)) ----------


def test_720p_to_fullhd_scale_2_with_clamp_needed():
    """1280x720 -> Full HD(1920): needed=ceil(1920/1280)=2 -> scale=2, 中間 2560 (clamp 縮小)"""
    decs = decisions.recommend(
        FakeMeta(width=1280, height=720),
        FakeStats(),
        defaults={"target_width": 1920, "target_width_explicit": True},
    )
    up = _upscale(decs)
    assert up.recommended is True
    assert up.params["scale"] == 2
    assert up.params["target_width"] == 1920
    assert "target_width=1920" in up.reason


def test_720p_to_4k_scale_3_exact():
    """1280x720 -> 4K(3840): needed=3 -> scale=3, 中間 3840 (ぴったり)"""
    decs = decisions.recommend(
        FakeMeta(width=1280, height=720),
        FakeStats(),
        defaults={"target_width": 3840, "target_width_explicit": True},
    )
    up = _upscale(decs)
    assert up.recommended is True
    assert up.params["scale"] == 3
    assert up.params["target_width"] == 3840


def test_1080p_to_wqhd_scale_2_with_clamp():
    """1920x1080 -> WQHD(2560): needed=ceil(2560/1920)=2 -> scale=2, 中間 3840 (clamp 縮小)"""
    decs = decisions.recommend(
        FakeMeta(width=1920, height=1080),
        FakeStats(),
        defaults={"target_width": 2560, "target_width_explicit": True},
    )
    up = _upscale(decs)
    assert up.recommended is True
    assert up.params["scale"] == 2
    assert up.params["target_width"] == 2560


def test_1080p_to_4k_scale_2_exact():
    """1920x1080 -> 4K(3840): needed=2 -> scale=2, 中間 3840 (ぴったり)"""
    decs = decisions.recommend(
        FakeMeta(width=1920, height=1080),
        FakeStats(),
        defaults={"target_width": 3840, "target_width_explicit": True},
    )
    up = _upscale(decs)
    assert up.recommended is True
    assert up.params["scale"] == 2
    assert up.params["target_width"] == 3840


def test_4k_source_to_fullhd_upscale_off():
    """src >= target は upscale OFF (現行通り)"""
    decs = decisions.recommend(
        FakeMeta(width=3840, height=2160),
        FakeStats(),
        defaults={"target_width": 1920, "target_width_explicit": True},
    )
    up = _upscale(decs)
    assert up.recommended is False
    assert up.params["target_width"] == 1920


# ---------- keep (現状維持): RESOLUTION_KEEP_SOURCE -> meta.width に解決 ----------


def test_keep_source_sub_fullhd_upscale_off():
    """1280x720 + keep センチネル -> target_width=入力幅(1280) に解決され upscale OFF.

    auto (target_width 未指定) なら 1280<1920 で upscale ON になるのと対照的。
    """
    decs = decisions.recommend(
        FakeMeta(width=1280, height=720),
        FakeStats(),
        defaults={"target_width": config.RESOLUTION_KEEP_SOURCE, "target_width_explicit": True},
    )
    up = _upscale(decs)
    assert up.recommended is False
    assert up.params["target_width"] == 1280
    assert "upscale 不要" in up.reason


def test_keep_source_differs_from_auto():
    """同じ 1280p 入力でも auto は upscale ON、keep は OFF になることを対比で確認."""
    auto = _upscale(decisions.recommend(FakeMeta(width=1280, height=720), FakeStats()))
    keep = _upscale(
        decisions.recommend(
            FakeMeta(width=1280, height=720),
            FakeStats(),
            defaults={"target_width": config.RESOLUTION_KEEP_SOURCE},
        )
    )
    assert auto.recommended is True  # auto -> 1920 へ拡大
    assert keep.recommended is False  # keep -> 現状維持


# ---------- needed > 4 のクランプケース ----------


def test_640x480_to_4k_needed_6_clamp_to_4():
    """640x480 -> 4K: needed=ceil(3840/640)=6 -> scale=4 で頭打ち, 中間 2560 (上方 lanczos)"""
    decs = decisions.recommend(
        FakeMeta(width=640, height=480),
        FakeStats(),
        defaults={"target_width": 3840, "target_width_explicit": True},
    )
    up = _upscale(decs)
    assert up.recommended is True
    assert up.params["scale"] == 4
    # reason に「クランプ」or「needed>=5」が含まれる
    assert ("クランプ" in up.reason) or ("needed=" in up.reason)


def test_854x480_to_4k_needed_5_clamp_to_4():
    """854x480 -> 4K: needed=ceil(3840/854)=5 -> scale=4, 中間 3416 (上方 lanczos)"""
    decs = decisions.recommend(
        FakeMeta(width=854, height=480),
        FakeStats(),
        defaults={"target_width": 3840, "target_width_explicit": True},
    )
    up = _upscale(decs)
    assert up.params["scale"] == 4
    assert "クランプ" in up.reason or "needed=" in up.reason


def test_960x540_to_4k_needed_4_exact():
    """960x540 -> 4K: needed=ceil(3840/960)=4 -> scale=4, 中間 3840 (ぴったり、警告なし)"""
    decs = decisions.recommend(
        FakeMeta(width=960, height=540),
        FakeStats(),
        defaults={"target_width": 3840, "target_width_explicit": True},
    )
    up = _upscale(decs)
    assert up.params["scale"] == 4
    # needed=4 なので「クランプ」表記は出ない（needed>=5 ブランチに行かない）
    assert "クランプ" not in up.reason
