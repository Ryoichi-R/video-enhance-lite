"""OpenCV-based video stabilization (vidstabtransform 代替).

ffmpeg N-120725+ で vidstabtransform が FAIL する既知問題 (TODO.md V1-1) の回避策。
依存は cv2 (= opencv-python-headless) のみ。

アルゴリズム:
  1. frames_in/*.png を順に読み込み、cv2.goodFeaturesToTrack + calcOpticalFlowPyrLK で
     前フレームとの affine 変換 (dx, dy, da) を推定
  2. 全フレームの累積軌跡を smoothing_radius の moving average で平滑化
  3. 各フレームに「smoothed - raw」の補正 affine を適用 (cv2.warpAffine)
  4. crop_border=True なら ~5% 内側クロップ + 元サイズへ scale 戻し
  5. frames_out/<同名>.png に書き出し
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import cv2  # type: ignore
import numpy as np

__all__ = [
    "stabilize_chunk",
    "DEFAULT_SMOOTHING_RADIUS",
]

DEFAULT_SMOOTHING_RADIUS = 30


def _sorted_pngs(frames_in: Path) -> list[Path]:
    return sorted(p for p in frames_in.iterdir() if p.suffix.lower() == ".png")


def _estimate_transforms(
    files: list[Path],
    cancel_event: threading.Event | None,
) -> np.ndarray:
    """Return Nx3 ndarray of (dx, dy, da) raw inter-frame transforms.

    First row is (0,0,0) by convention (no previous frame).
    """
    if len(files) < 2:
        raise ValueError(f"need at least 2 frames, got {len(files)}")

    transforms = np.zeros((len(files), 3), dtype=np.float64)

    prev = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    if prev is None:
        raise RuntimeError(f"failed to read {files[0]}")

    for i in range(1, len(files)):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("cancelled")

        curr = cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE)
        if curr is None:
            raise RuntimeError(f"failed to read {files[i]}")

        prev_pts = cv2.goodFeaturesToTrack(
            prev, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3
        )
        if prev_pts is None or len(prev_pts) < 4:
            transforms[i] = transforms[i - 1]
            prev = curr
            continue

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, prev_pts, None)
        if curr_pts is None or status is None:
            transforms[i] = transforms[i - 1]
            prev = curr
            continue

        idx = np.where(status.ravel() == 1)[0]
        if len(idx) < 4:
            transforms[i] = transforms[i - 1]
            prev = curr
            continue

        prev_good = prev_pts[idx]
        curr_good = curr_pts[idx]

        m, _ = cv2.estimateAffinePartial2D(prev_good, curr_good)
        if m is None:
            transforms[i] = transforms[i - 1]
            prev = curr
            continue

        dx = float(m[0, 2])
        dy = float(m[1, 2])
        da = float(np.arctan2(m[1, 0], m[0, 0]))
        transforms[i] = (dx, dy, da)
        prev = curr

    return transforms


def _smooth_trajectory(trajectory: np.ndarray, radius: int) -> np.ndarray:
    """Moving-average smoothing on cumulative trajectory."""
    if radius < 1:
        return trajectory.copy()
    window = 2 * radius + 1
    pad = np.pad(trajectory, ((radius, radius), (0, 0)), mode="edge")
    smoothed = np.zeros_like(trajectory)
    for i in range(trajectory.shape[1]):
        smoothed[:, i] = np.convolve(pad[:, i], np.ones(window) / window, mode="valid")
    return smoothed


def _apply_transform_to_frame(
    frame: np.ndarray,
    dx: float,
    dy: float,
    da: float,
    *,
    crop_border: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]
    cos_a = np.cos(da)
    sin_a = np.sin(da)
    m = np.array([[cos_a, -sin_a, dx], [sin_a, cos_a, dy]], dtype=np.float64)
    stabilized = cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if crop_border:
        # 約 5% 内側クロップ + 元サイズへ scale back
        crop_x = int(w * 0.04)
        crop_y = int(h * 0.04)
        if crop_x > 0 and crop_y > 0 and w - 2 * crop_x > 0 and h - 2 * crop_y > 0:
            stabilized = stabilized[crop_y : h - crop_y, crop_x : w - crop_x]
            stabilized = cv2.resize(stabilized, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return stabilized


def stabilize_chunk(
    frames_in: Path,
    frames_out: Path,
    *,
    smoothing_radius: int = DEFAULT_SMOOTHING_RADIUS,
    crop_border: bool = True,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Stabilize a chunk of PNG frames in `frames_in`, writing results to `frames_out`.

    Args:
        frames_in: directory containing input PNG frames (named in lex order).
        frames_out: directory to write stabilized PNGs (created if missing).
        smoothing_radius: moving-average radius for trajectory smoothing.
        crop_border: if True, crop ~5% inner border and rescale to original size
                     (avoids visible black borders after warp).
        cancel_event: optional Event to abort processing.
        on_progress: optional callback (frame_index, total) for progress reporting.

    Raises:
        ValueError if fewer than 2 PNGs in `frames_in`.
        InterruptedError if cancel_event is set.
    """
    files = _sorted_pngs(frames_in)
    if len(files) < 2:
        raise ValueError(f"need at least 2 PNG frames in {frames_in}, got {len(files)}")
    frames_out.mkdir(parents=True, exist_ok=True)

    raw = _estimate_transforms(files, cancel_event)
    trajectory = np.cumsum(raw, axis=0)
    smoothed = _smooth_trajectory(trajectory, smoothing_radius)
    diff = smoothed - trajectory  # 各フレームに加算すべき補正
    transforms_smoothed = raw + diff

    for i, p in enumerate(files):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("cancelled")
        frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"failed to read {p}")
        dx, dy, da = transforms_smoothed[i]
        stabilized = _apply_transform_to_frame(
            frame,
            dx,
            dy,
            da,
            crop_border=crop_border,
        )
        out_path = frames_out / p.name
        cv2.imwrite(str(out_path), stabilized)
        if on_progress is not None:
            try:
                on_progress(i + 1, len(files))
            except Exception:
                pass
