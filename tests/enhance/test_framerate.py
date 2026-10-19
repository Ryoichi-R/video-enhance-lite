from __future__ import annotations

from fractions import Fraction

import pytest

from video_enhance_lite.framerate import (
    UnsupportedFramerateError,
    resolve_framerate_normalization,
)


def test_cfr_keeps_current_timeline_without_normalization():
    plan = resolve_framerate_normalization("cfr", Fraction(30000, 1001))

    assert plan.requires_normalization is False
    assert plan.normalization_fps is None
    assert plan.nominal_fps == Fraction(30000, 1001)


def test_vfr_uses_nominal_r_frame_rate_for_explicit_cfr_normalization():
    plan = resolve_framerate_normalization("vfr_suspected", Fraction(30, 1))

    assert plan.requires_normalization is True
    assert plan.normalization_fps == Fraction(30, 1)


def test_unknown_framerate_fails_closed():
    with pytest.raises(UnsupportedFramerateError, match="処理を開始できません"):
        resolve_framerate_normalization("unknown", Fraction(30, 1))


@pytest.mark.parametrize("fps", [Fraction(0, 1), Fraction(-1, 1)])
def test_nonpositive_nominal_framerate_fails_closed(fps: Fraction):
    with pytest.raises(UnsupportedFramerateError, match="正数"):
        resolve_framerate_normalization("cfr", fps)


def test_missing_nominal_framerate_fails_closed():
    with pytest.raises(UnsupportedFramerateError, match="解釈できません"):
        resolve_framerate_normalization("cfr", None)


def test_missing_status_legacy_compatibility_is_explicit():
    plan = resolve_framerate_normalization(
        "cfr",
        Fraction(24, 1),
        legacy_status_missing=True,
    )

    assert plan.requires_normalization is False
    assert plan.legacy_status_missing is True
