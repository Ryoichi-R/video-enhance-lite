from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from video_enhance_lite import cli, config, stages
from tests.enhance.test_cli import FakeMeta, FakeStats


# ---------- rife model param propagation ----------


def test_build_rife_args_v3_with_ratio_not_2_raises(tmp_path):
    """v3 系モデル + ratio != 2 で ValueError (silent な品質劣化を避ける)。"""
    with pytest.raises(ValueError, match="x2 interpolation"):
        stages.build_rife_args(
            rife_bin=tmp_path / "rife.exe",
            in_dir=tmp_path / "in",
            out_dir=tmp_path / "out",
            source_fps=Fraction(30, 1),
            target_fps=Fraction(120, 1),  # x4 -> v3 系では非対応
            input_frame_count=10,
            model="rife-v3.0",
        )


def test_build_rife_args_v3_with_ratio_2_returns_no_n(tmp_path):
    """v3 系 + ratio==2 のときは -n を付けず通る。"""
    argv = stages.build_rife_args(
        rife_bin=tmp_path / "rife.exe",
        in_dir=tmp_path / "in",
        out_dir=tmp_path / "out",
        source_fps=Fraction(24, 1),
        target_fps=Fraction(48, 1),
        input_frame_count=10,
        model="rife-v3.0",
    )
    assert "-n" not in argv
    assert argv[argv.index("-m") + 1] == "rife-v3.0"


def test_build_rife_args_v4_supports_any_integer_ratio(tmp_path):
    """v4 系は -n でカスタム整数倍補間。"""
    argv = stages.build_rife_args(
        rife_bin=tmp_path / "rife.exe",
        in_dir=tmp_path / "in",
        out_dir=tmp_path / "out",
        source_fps=Fraction(30, 1),
        target_fps=Fraction(120, 1),  # x4
        input_frame_count=20,
        model="rife-v4.6",
    )
    assert "-n" in argv
    # 20 frames * 4 = 80
    assert argv[argv.index("-n") + 1] == "80"


# ---------- CLI parser ----------


def test_cli_parser_rife_model_choices():
    parser = cli._build_parser()
    args = parser.parse_args(["dummy.mp4", "--rife-model", "rife-v3.0"])
    assert args.rife_model == "rife-v3.0"


def test_cli_parser_rife_model_invalid_choice():
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["dummy.mp4", "--rife-model", "not-a-model"])


def test_cli_parser_realesrgan_model_accepts_general_x4v3():
    """argparse は静的 choices で全候補 (同梱有無に関わらず) 受け付ける。"""
    parser = cli._build_parser()
    args = parser.parse_args(["dummy.mp4", "--realesrgan-model", "realesr-general-x4v3"])
    assert args.realesrgan_model == "realesr-general-x4v3"


def test_cli_parser_realesrgan_model_invalid_choice():
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["dummy.mp4", "--realesrgan-model", "fake-model"])


def test_cli_parser_default_rife_model():
    parser = cli._build_parser()
    args = parser.parse_args(["dummy.mp4"])
    assert args.rife_model == config.DEFAULT_RIFE_MODEL


# ---------- main() x rife model validation (P0-1 回帰) ----------


def _setup_main_fakes(monkeypatch, tmp_path, *, rife_exe: Path):
    """main() の外部依存 (deps / analyzer / pipeline) をフェイクする共通セットアップ。

    interpolate が推奨される素材 (fps=24 < 30) を返す。pipeline.run_pipeline は
    kwargs を捕捉して返す。
    """
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    input_path = tmp_path / "in.mp4"
    input_path.write_bytes(b"")

    from video_enhance_lite import deps as ven_deps

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
    monkeypatch.setattr(
        ven_deps,
        "check_rife",
        lambda: ven_deps.ResolvedBinary(path=rife_exe, version=None, sha256=None),
    )

    from video_enhance_lite import analyzer

    monkeypatch.setattr(
        analyzer,
        "analyze",
        lambda **kw: (FakeMeta(width=2000, fps=Fraction(24, 1)), FakeStats()),
    )

    captured: dict = {}
    from video_enhance_lite import pipeline as ven_pipeline

    monkeypatch.setattr(ven_pipeline, "run_pipeline", lambda **kw: captured.update(kw))
    return input_path, captured


def test_main_interpolate_passes_with_official_rife_layout(monkeypatch, tmp_path):
    """P0-1 回帰: 公式 20221029 配布構造 (exe 同階層にモデル dir、models/ なし) で
    interpolate ON の依存チェックが通過する。"""
    rife_dist = tmp_path / "rife-dist"
    model_dir = rife_dist / config.DEFAULT_RIFE_MODEL
    model_dir.mkdir(parents=True)
    (model_dir / "flownet.param").write_text("dummy")
    rife_exe = rife_dist / "rife-ncnn-vulkan.exe"
    rife_exe.write_bytes(b"")

    input_path, captured = _setup_main_fakes(monkeypatch, tmp_path, rife_exe=rife_exe)
    rc = cli.main([str(input_path), "--auto-all", "--no-realesrgan"])
    assert rc == 0
    assert captured["rife"] == rife_exe


def test_main_interpolate_fails_when_model_missing(monkeypatch, tmp_path, capsys):
    """モデル dir が exe 同階層にも models/ にも無ければ exit 2 で両候補を提示する。"""
    rife_dist = tmp_path / "rife-dist"
    rife_dist.mkdir()
    rife_exe = rife_dist / "rife-ncnn-vulkan.exe"
    rife_exe.write_bytes(b"")

    input_path, captured = _setup_main_fakes(monkeypatch, tmp_path, rife_exe=rife_exe)
    rc = cli.main([str(input_path), "--auto-all", "--no-realesrgan"])
    assert rc == 2
    assert not captured  # pipeline は呼ばれない
    err = capsys.readouterr().err
    assert "rife model dir not found" in err
