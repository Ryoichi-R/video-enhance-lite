from __future__ import annotations

import pytest

from video_enhance_lite import stages


def test_clamp_skipped_when_equal():
    assert stages.build_resolution_clamp_filter(target_width=2560, intermediate_width=2560) == ""


def test_clamp_downscale_3840_to_2560():
    """中間 3840 -> target 2560: lanczos で縮小"""
    assert (
        stages.build_resolution_clamp_filter(target_width=2560, intermediate_width=3840)
        == "scale=2560:-2:flags=lanczos"
    )


def test_clamp_upscale_2560_to_3840():
    """中間 2560 -> target 3840: lanczos で上方 scale (Real-ESRGAN scale=4 で頭打ちしたケース)"""
    assert (
        stages.build_resolution_clamp_filter(target_width=3840, intermediate_width=2560)
        == "scale=3840:-2:flags=lanczos"
    )


def test_clamp_rejects_non_positive():
    with pytest.raises(ValueError):
        stages.build_resolution_clamp_filter(target_width=0, intermediate_width=1920)
    with pytest.raises(ValueError):
        stages.build_resolution_clamp_filter(target_width=1920, intermediate_width=0)


# ---------- PR-G2: clamp flags (opt-in) ----------


def test_clamp_default_flags_is_lanczos():
    """flags 未指定は従来通り lanczos (既定挙動・不変)。"""
    assert (
        stages.build_resolution_clamp_filter(target_width=2560, intermediate_width=3840)
        == "scale=2560:-2:flags=lanczos"
    )


def test_clamp_bicubic_flag():
    assert (
        stages.build_resolution_clamp_filter(
            target_width=2560, intermediate_width=3840, flags="bicubic"
        )
        == "scale=2560:-2:flags=bicubic"
    )


def test_clamp_bilinear_flag():
    assert (
        stages.build_resolution_clamp_filter(
            target_width=2560, intermediate_width=3840, flags="bilinear"
        )
        == "scale=2560:-2:flags=bilinear"
    )


def test_clamp_rejects_unknown_flag():
    with pytest.raises(ValueError):
        stages.build_resolution_clamp_filter(
            target_width=2560, intermediate_width=3840, flags="nearest"
        )
