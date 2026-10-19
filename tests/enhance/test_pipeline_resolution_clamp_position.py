from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from video_enhance_lite import config, pipeline
from video_enhance_lite.decisions import StageDecision

# workspace / fake_run fixture は conftest.py 由来 (import 不要)。


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


def _make_decisions(
    *, upscale_enabled: bool, scale: int, target_width: int, interpolate_enabled: bool = False
) -> list[StageDecision]:
    out: list[StageDecision] = []
    for k in config.STAGE_KEYS:
        enabled = False
        params: dict = {}
        if k == "upscale":
            enabled = upscale_enabled
            params = {
                "scale": scale,
                "model": config.DEFAULT_REALESRGAN_MODEL,
                "target_width": target_width,
            }
        elif k == "interpolate":
            enabled = interpolate_enabled
            params = {"target_fps_num": 60, "target_fps_den": 1}
        elif k == "encode":
            enabled = True
            params = {"crf": config.DEFAULT_ENCODE_CRF, "preset": config.DEFAULT_ENCODE_PRESET}
        out.append(
            StageDecision(
                key=k,
                label_ja=config.STAGE_LABELS_JA[k],
                recommended=enabled,
                confidence="high",
                reason="test",
                enabled=enabled,
                params=params,
            )
        )
    return out


def _clamp_calls(calls):
    """`-vf scale=...:-2:flags=lanczos` を含む ffmpeg 呼び出しを抽出"""
    out = []
    for c in calls:
        if "ffmpeg" not in c[0].lower():
            continue
        if "-vf" not in c:
            continue
        idx = c.index("-vf")
        if idx + 1 < len(c) and "flags=lanczos" in c[idx + 1] and "scale=" in c[idx + 1]:
            out.append(c)
    return out


def _rife_calls(calls):
    return [c for c in calls if "rife" in c[0].lower()]


def test_clamp_runs_when_intermediate_differs_from_target(workspace, fake_run):
    """1080p x scale=2 = 3840 中間幅, target=2560 → clamp 実行 → RIFE は frames_clamp が入力"""
    decs = _make_decisions(
        upscale_enabled=True,
        scale=2,
        target_width=2560,
        interpolate_enabled=True,
    )
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=decs,
        meta=FakeMeta(width=1920, height=1080, duration_sec=10.0),
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        realesrgan=Path(r"C:\bin\realesrgan-ncnn-vulkan.exe"),
        rife=Path(r"C:\bin\rife-ncnn-vulkan.exe"),
    )
    clamp = _clamp_calls(fake_run.calls)
    assert len(clamp) == 1
    # scale=2560:-2:flags=lanczos を含む
    assert any("scale=2560:-2:flags=lanczos" in tok for tok in clamp[0])
    # RIFE は frames_clamp ディレクトリを入力にしている
    rife = _rife_calls(fake_run.calls)
    assert len(rife) == 1
    in_dir_idx = rife[0].index("-i") + 1
    assert "frames_clamp" in rife[0][in_dir_idx]


def test_clamp_flags_bicubic_propagates_to_ffmpeg(workspace, fake_run):
    """PR-G2: clamp_flags='bicubic' が clamp の ffmpeg -vf に伝播する。"""
    decs = _make_decisions(
        upscale_enabled=True,
        scale=2,
        target_width=2560,
        interpolate_enabled=True,
    )
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=decs,
        meta=FakeMeta(width=1920, height=1080, duration_sec=10.0),
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        realesrgan=Path(r"C:\bin\realesrgan-ncnn-vulkan.exe"),
        rife=Path(r"C:\bin\rife-ncnn-vulkan.exe"),
        clamp_flags="bicubic",
    )
    vf_calls = [
        c
        for c in fake_run.calls
        if "ffmpeg" in c[0].lower()
        and "-vf" in c
        and "scale=2560:-2:flags=bicubic" in c[c.index("-vf") + 1]
    ]
    assert len(vf_calls) == 1
    # 既定 lanczos は出ていない
    assert not _clamp_calls(fake_run.calls)


def test_clamp_skipped_when_intermediate_equals_target(workspace, fake_run):
    """1080p x scale=2 = 3840 中間, target=3840 (ぴったり) → clamp 不要 → RIFE は frames_up 直接"""
    decs = _make_decisions(
        upscale_enabled=True,
        scale=2,
        target_width=3840,
        interpolate_enabled=True,
    )
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=decs,
        meta=FakeMeta(width=1920, height=1080, duration_sec=10.0),
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        realesrgan=Path(r"C:\bin\realesrgan-ncnn-vulkan.exe"),
        rife=Path(r"C:\bin\rife-ncnn-vulkan.exe"),
    )
    clamp = _clamp_calls(fake_run.calls)
    assert len(clamp) == 0
    rife = _rife_calls(fake_run.calls)
    assert len(rife) == 1
    in_dir_idx = rife[0].index("-i") + 1
    assert "frames_up" in rife[0][in_dir_idx]
    assert "frames_clamp" not in rife[0][in_dir_idx]
