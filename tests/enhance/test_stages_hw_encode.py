from __future__ import annotations

import pytest

from video_enhance_lite import config, stages
from video_enhance_lite.decisions import StageDecision


def _dec(params: dict | None = None) -> StageDecision:
    return StageDecision(
        key="encode",
        label_ja=config.STAGE_LABELS_JA["encode"],
        recommended=True,
        confidence="high",
        reason="test",
        enabled=True,
        params=dict(params or {}),
    )


def test_build_encode_args_default_omitted_matches_legacy():
    """hw_backend を省略すると旧シグネチャと完全に同じ argv (既定不変)。"""
    args = stages.build_encode_args(_dec())
    assert args == [
        "-c:v",
        "libx264",
        "-preset",
        config.DEFAULT_ENCODE_PRESET,
        "-crf",
        str(config.DEFAULT_ENCODE_CRF),
        "-pix_fmt",
        "yuv420p",
    ]


def test_build_encode_args_none_explicit_matches_default():
    """`hw_backend="none"` 明示も同じ結果。"""
    args1 = stages.build_encode_args(_dec())
    args2 = stages.build_encode_args(_dec(), hw_backend="none")
    assert args1 == args2


def test_build_encode_args_nvenc():
    args = stages.build_encode_args(_dec({"crf": 18, "preset": "slow"}), hw_backend="nvenc")
    assert "h264_nvenc" in args
    assert "-cq" in args
    assert args[args.index("-cq") + 1] == "18"
    # libx264 slow -> nvenc p5
    assert args[args.index("-preset") + 1] == "p5"
    assert args[-2:] == ["-pix_fmt", "yuv420p"]


def test_build_encode_args_qsv():
    args = stages.build_encode_args(_dec({"crf": 22, "preset": "medium"}), hw_backend="qsv")
    assert "h264_qsv" in args
    assert "-global_quality" in args
    assert args[args.index("-global_quality") + 1] == "22"


def test_build_encode_args_amf():
    args = stages.build_encode_args(_dec({"crf": 20, "preset": "slow"}), hw_backend="amf")
    assert "h264_amf" in args
    assert "-quality" in args
    assert args[args.index("-quality") + 1] == "quality"  # libx264 slow -> amf quality
    assert "-qp_i" in args
    assert args[args.index("-qp_i") + 1] == "20"
    assert args[args.index("-rc") + 1] == "cqp"
    for option in ("-qp_i", "-qp_p", "-qp_b"):
        assert args[args.index(option) + 1] == "20"


def test_build_encode_args_unknown_hw_backend_raises():
    with pytest.raises(ValueError, match="unknown hw_backend"):
        stages.build_encode_args(_dec(), hw_backend="vaapi")


def test_unknown_preset_is_rejected_for_hw_backend():
    with pytest.raises(ValueError, match="unknown encode preset"):
        stages.build_encode_args(_dec({"preset": "unknown-preset"}), hw_backend="nvenc")


@pytest.mark.parametrize("backend", ["nvenc", "qsv", "amf"])
@pytest.mark.parametrize(
    "preset",
    [
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    ],
)
def test_all_cli_presets_are_mapped_for_hw_backends(backend, preset):
    args = stages.build_encode_args(_dec({"preset": preset}), hw_backend=backend)
    assert "-preset" in args or backend == "amf"


def test_hw_encode_backends_constant():
    assert stages.HW_ENCODE_BACKENDS == ("none", "nvenc", "qsv", "amf")
