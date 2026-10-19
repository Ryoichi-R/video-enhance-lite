from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

FramerateStatus = Literal["cfr", "vfr_suspected", "unknown"]

__all__ = [
    "FramerateNormalizationPlan",
    "UnsupportedFramerateError",
    "resolve_framerate_normalization",
]


@dataclass(frozen=True)
class FramerateNormalizationPlan:
    status: FramerateStatus
    nominal_fps: Fraction
    normalization_fps: Fraction | None
    legacy_status_missing: bool = False

    @property
    def requires_normalization(self) -> bool:
        return self.normalization_fps is not None


class UnsupportedFramerateError(ValueError):
    pass


def resolve_framerate_normalization(
    status: str,
    nominal_fps: object,
    *,
    legacy_status_missing: bool = False,
) -> FramerateNormalizationPlan:
    """Resolve the D11 CFR/VFR/unknown policy without side effects."""
    try:
        fps = Fraction(nominal_fps)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise UnsupportedFramerateError(
            f"nominal r_frame_rateを解釈できません: {nominal_fps!r}"
        ) from exc
    if fps <= 0:
        raise UnsupportedFramerateError(f"nominal r_frame_rateは正数が必要です: {fps}")

    if legacy_status_missing:
        return FramerateNormalizationPlan(
            status="cfr",
            nominal_fps=fps,
            normalization_fps=None,
            legacy_status_missing=True,
        )
    if status == "cfr":
        return FramerateNormalizationPlan("cfr", fps, None)
    if status == "vfr_suspected":
        return FramerateNormalizationPlan("vfr_suspected", fps, fps)
    if status == "unknown":
        raise UnsupportedFramerateError(
            "フレームレートを安全に判定できないため処理を開始できません。"
            "r_frame_rate/avg_frame_rateまたはframe PTSを確認してください"
        )
    raise UnsupportedFramerateError(f"未対応framerate statusです: {status!r}")
