from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from video_enhance_lite import analyzer, cli, config
from video_enhance_lite import deps as ven_deps


@dataclass
class FakeMeta:
    width: int = 1280
    height: int = 720
    fps: Fraction = Fraction(30, 1)
    duration_sec: float = 60.0
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


def _patch_common(monkeypatch, tmp_path) -> Path:
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    input_path = tmp_path / "in.mp4"
    input_path.write_bytes(b"")
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
    return input_path


def test_analyze_only_prints_runtime_line(monkeypatch, tmp_path, capsys):
    input_path = _patch_common(monkeypatch, tmp_path)
    rc = cli.main([str(input_path), "--analyze-only"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "estimated runtime:" in out
    # accuracy_note の文言を流用していること
    assert "±50%" in out
    # 非 verbose では breakdown を出さない
    assert "breakdown:" not in out


def test_analyze_only_verbose_prints_breakdown(monkeypatch, tmp_path, capsys):
    input_path = _patch_common(monkeypatch, tmp_path)
    rc = cli.main([str(input_path), "--analyze-only", "-v"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "estimated runtime:" in out
    assert "breakdown:" in out
