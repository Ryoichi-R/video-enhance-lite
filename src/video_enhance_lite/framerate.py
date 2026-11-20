from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil
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


def output_frame_count(meta) -> int:
    count = getattr(meta, "frame_count", None)
    if count and getattr(meta, "framerate_status", None) == "cfr":
        return count
    return max(1, int(Fraction(str(meta.duration_sec)) * Fraction(meta.fps) + Fraction(1, 2)))


def frame_ranges(meta, chunk_seconds: int) -> list[tuple[int, int]]:
    """Half-open normalized frame ranges. Round cumulative boundaries only."""
    if chunk_seconds < 1 or meta.duration_sec <= 0 or meta.fps <= 0:
        raise ValueError("duration, fps and chunk_seconds must be positive")
    total = output_frame_count(meta)
    step = Fraction(meta.fps) * chunk_seconds
    ranges = []
    first = 0
    index = 1
    while first < total:
        end = min(total, int(index * step + Fraction(1, 2)))
        if end == first:
            index = ceil((first + Fraction(1, 2)) / step)
            continue
        ranges.append((first, end))
        first = end
        index += 1
    return ranges
