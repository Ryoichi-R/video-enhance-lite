from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest

from video_enhance_lite import analyzer, cli, config, deps as ven_deps


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


@pytest.fixture
def patched_env(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(
        ven_deps,
        "check_ffmpeg",
        lambda: ven_deps.ResolvedBinary(path=Path(r"C:\bin\ffmpeg.exe"), version="n", sha256=None),
    )
    monkeypatch.setattr(
        ven_deps,
        "check_ffprobe",
        lambda: ven_deps.ResolvedBinary(path=Path(r"C:\bin\ffprobe.exe"), version="n", sha256=None),
    )
    monkeypatch.setattr(analyzer, "analyze", lambda **kw: (FakeMeta(), FakeStats()))
    input_path = tmp_path / "in.mp4"
    input_path.write_bytes(b"")
    return input_path


def test_analyze_only_reason_includes_target_width_and_target_fps(patched_env, capsys):
    """--analyze-only --target-resolution 4k --target-fps-preset 60 → reason に target_width=3840 と target_fps=60"""
    rc = cli.main(
        [
            str(patched_env),
            "--analyze-only",
            "--target-resolution",
            "4k",
            "--target-fps-preset",
            "60",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # upscale の reason に target_width=3840 が含まれる (preset 4K)
    assert "target_width=3840" in out
    # interpolate の reason に target_fps=60 が含まれる (preset)
    # 30fps source < 60fps preset → ON (×2=60)
    assert "target_fps=60" in out
