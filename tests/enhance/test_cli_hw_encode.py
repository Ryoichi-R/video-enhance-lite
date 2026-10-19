from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from video_enhance_lite import cli, deps


def test_check_hw_encoder_detects_nvenc(monkeypatch):
    def _fake_run(*args, **kwargs):
        return SimpleNamespace(
            stdout=" V..... libx264              x264\n V..... h264_nvenc           NVIDIA NVENC H.264 encoder\n",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(deps.subprocess, "run", _fake_run)
    assert deps.check_hw_encoder(Path(r"C:\bin\ffmpeg.exe"), "h264_nvenc") is True
    assert deps.check_hw_encoder(Path(r"C:\bin\ffmpeg.exe"), "h264_qsv") is False


def test_check_hw_encoder_handles_oserror(monkeypatch):
    def _raises(*args, **kwargs):
        raise OSError("subprocess failed")

    monkeypatch.setattr(deps.subprocess, "run", _raises)
    assert deps.check_hw_encoder(Path(r"C:\bin\ffmpeg.exe"), "h264_nvenc") is False


def test_cli_parser_hw_encode_default_none():
    parser = cli._build_parser()
    args = parser.parse_args(["dummy.mp4"])
    assert args.hw_encode == "none"


def test_cli_parser_hw_encode_choices():
    parser = cli._build_parser()
    for v in ("none", "nvenc", "qsv", "amf"):
        args = parser.parse_args(["dummy.mp4", "--hw-encode", v])
        assert args.hw_encode == v


def test_cli_parser_hw_encode_invalid():
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["dummy.mp4", "--hw-encode", "vaapi"])
