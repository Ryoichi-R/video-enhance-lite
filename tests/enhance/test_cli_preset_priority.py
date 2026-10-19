from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest

from video_enhance_lite import analyzer, cli, config, deps as ven_deps


@dataclass
class FakeMeta:
    width: int = 1920
    height: int = 1080
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


def test_explicit_target_width_overrides_preset(patched_env, capsys):
    """--target-width 2048 + --target-resolution wqhd → 2048 採用 + WARN"""
    rc = cli.main(
        [
            str(patched_env),
            "--analyze-only",
            "--target-width",
            "2048",
            "--target-resolution",
            "wqhd",
        ]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "--target-width=2048 overrides --target-resolution=wqhd" in err


def test_explicit_scale_overrides_autoscale(patched_env, capsys):
    """1080p input + --target-resolution wqhd (auto-scale=2) + --scale 3 → scale=3 + WARN"""
    rc = cli.main(
        [str(patched_env), "--analyze-only", "--target-resolution", "wqhd", "--scale", "3"]
    )
    assert rc == 0
    err = capsys.readouterr().err
    # auto-scale=2 と明示 --scale=3 の差で WARN
    assert "--scale=3" in err and "scale=2" in err


def test_preset_alone_autoscale(patched_env, capsys):
    """--target-resolution wqhd 単体 → target_width=2560, scale=2 自動"""
    rc = cli.main([str(patched_env), "--analyze-only", "--target-resolution", "wqhd"])
    assert rc == 0
    out = capsys.readouterr().out
    # decision table の upscale 行に target_width=2560 が出る
    assert "target_width=2560" in out


def test_fps_preset_overrides_warning(patched_env, capsys):
    """--target-fps 48.5 + --target-fps-preset 60 → 48.5 採用 + WARN"""
    rc = cli.main(
        [str(patched_env), "--analyze-only", "--target-fps", "48.5", "--target-fps-preset", "60"]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "--target-fps=48.5 overrides --target-fps-preset=60" in err


def test_parser_accepts_keep_resolution():
    """--target-resolution keep が choices に含まれパースできる"""
    args = cli._build_parser().parse_args(["in.mp4", "--target-resolution", "keep"])
    assert args.target_resolution == "keep"


def test_keep_resolution_disables_upscale(monkeypatch, patched_env, capsys):
    """--target-resolution keep → 入力幅未満でも upscale 非推奨 (解像度維持)"""
    # patched_env の FakeMeta は 1920 幅なので、1280 幅に差し替えて auto との差を出す
    from video_enhance_lite import analyzer

    @dataclass
    class _Meta(FakeMeta):
        width: int = 1280
        height: int = 720

    monkeypatch.setattr(analyzer, "analyze", lambda **kw: (_Meta(), FakeStats()))
    rc = cli.main([str(patched_env), "--analyze-only", "--target-resolution", "keep"])
    assert rc == 0
    out = capsys.readouterr().out
    # decision table の upscale 行: REC=no + reason に "upscale 不要"
    upscale_line = next(ln for ln in out.splitlines() if ln.startswith("upscale"))
    assert " no " in f" {upscale_line} "
    assert "upscale 不要" in upscale_line


def test_keep_resolution_warns_when_target_width_also_given(monkeypatch, patched_env, capsys):
    """keep + 明示 --target-width 併用時は --target-width が勝ち、WARN に '(0)' を出さない"""
    rc = cli.main(
        [
            str(patched_env),
            "--analyze-only",
            "--target-resolution",
            "keep",
            "--target-width",
            "2560",
        ]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "overrides --target-resolution=keep" in err
    assert "keep(0)" not in err


def test_parser_accepts_keep_fps():
    """--target-fps-preset keep が choices に含まれパースできる"""
    args = cli._build_parser().parse_args(["in.mp4", "--target-fps-preset", "keep"])
    assert args.target_fps_preset == "keep"


def test_keep_fps_disables_interpolate(monkeypatch, patched_env, capsys):
    """--target-fps-preset keep → source<30 でも interpolate 非推奨 (fps 維持)"""
    from video_enhance_lite import analyzer

    @dataclass
    class _Meta(FakeMeta):
        fps: Fraction = Fraction(24, 1)

    monkeypatch.setattr(analyzer, "analyze", lambda **kw: (_Meta(), FakeStats()))
    rc = cli.main([str(patched_env), "--analyze-only", "--target-fps-preset", "keep"])
    assert rc == 0
    out = capsys.readouterr().out
    interp_line = next(ln for ln in out.splitlines() if ln.startswith("interpolate"))
    assert " no " in f" {interp_line} "
    assert "補間しません" in interp_line


def test_keep_fps_warns_when_target_fps_also_given(patched_env, capsys):
    """keep + 明示 --target-fps 併用時は --target-fps が勝ち、WARN に '(0.0)' を出さない"""
    rc = cli.main(
        [str(patched_env), "--analyze-only", "--target-fps-preset", "keep", "--target-fps", "60"]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "overrides --target-fps-preset=keep" in err
    assert "keep(0" not in err
