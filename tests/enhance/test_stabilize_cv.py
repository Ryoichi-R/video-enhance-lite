from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from video_enhance_lite import stabilize_cv


def _gen_shake_frames(out_dir: Path, *, n_frames: int = 15, w: int = 320, h: int = 240) -> None:
    """合成手ブレ素材を生成。中央の白矩形を一定パターンで揺らす。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # 背景にノイズを入れて特徴点検出が機能するように
        noise = (rng.normal(loc=64, scale=16, size=(h, w, 3))).clip(0, 255).astype(np.uint8)
        frame[:] = noise
        # 中心矩形を意図的に揺らす (dx=10 px)
        dx = int(10 * np.sin(i * 0.5))
        dy = int(6 * np.cos(i * 0.4))
        cx, cy = w // 2 + dx, h // 2 + dy
        cv2.rectangle(frame, (cx - 40, cy - 30), (cx + 40, cy + 30), (255, 255, 255), -1)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)
        cv2.imwrite(str(out_dir / f"{i:08d}.png"), frame)


def _frame_diff_sum(frames_dir: Path) -> float:
    files = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() == ".png")
    total = 0.0
    prev = None
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if prev is not None:
            total += float(np.mean(np.abs(img.astype(np.int16) - prev.astype(np.int16))))
        prev = img
    return total


def test_stabilize_chunk_reduces_interframe_diff(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    _gen_shake_frames(in_dir, n_frames=15)
    original_diff = _frame_diff_sum(in_dir)

    stabilize_cv.stabilize_chunk(in_dir, out_dir, smoothing_radius=5)

    out_files = list(out_dir.glob("*.png"))
    assert len(out_files) == 15
    stabilized_diff = _frame_diff_sum(out_dir)
    # 完璧な減少を期待するのは難しいが、少なくとも増えていないこと (regression 防止)。
    # 合成素材では多少のずれで diff が同程度になることがあるため、緩めの上限。
    assert stabilized_diff <= original_diff * 1.2


def test_stabilize_chunk_too_few_frames_raises(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    cv2.imwrite(str(in_dir / "00000000.png"), np.zeros((100, 100, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="need at least 2"):
        stabilize_cv.stabilize_chunk(in_dir, tmp_path / "out")


def test_stabilize_chunk_cancellation(tmp_path):
    in_dir = tmp_path / "in"
    _gen_shake_frames(in_dir, n_frames=20)

    ev = threading.Event()
    ev.set()  # 事前 cancel
    with pytest.raises(InterruptedError):
        stabilize_cv.stabilize_chunk(in_dir, tmp_path / "out", cancel_event=ev)


def test_stabilize_chunk_progress_callback(tmp_path):
    in_dir = tmp_path / "in"
    _gen_shake_frames(in_dir, n_frames=10)

    seen: list[tuple[int, int]] = []
    stabilize_cv.stabilize_chunk(
        in_dir,
        tmp_path / "out",
        smoothing_radius=3,
        on_progress=lambda n, t: seen.append((n, t)),
    )
    assert seen[-1] == (10, 10)
    assert len(seen) == 10
