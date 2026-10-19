from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from video_enhance_lite import config, stages
from video_enhance_lite.decisions import StageDecision


def _dec(key: str, *, enabled: bool, params: dict | None = None) -> StageDecision:
    return StageDecision(
        key=key,
        label_ja=config.STAGE_LABELS_JA[key],
        recommended=enabled,
        confidence="medium",
        reason="test",
        enabled=enabled,
        params=dict(params or {}),
    )


def _all(decision_map: dict) -> list[StageDecision]:
    """Build a full decision list for STAGE_KEYS, defaulting disabled where missing."""
    decs = []
    for k in config.STAGE_KEYS:
        if k in decision_map:
            decs.append(decision_map[k])
        else:
            decs.append(_dec(k, enabled=False))
    return decs


# ---------- escape_filter_path ----------


def test_escape_filter_path_windows_path_with_spaces():
    p = Path(r"C:\Users\foo bar\transforms.trf")
    s = stages.escape_filter_path(p)
    # コロンとバックスラッシュがエスケープされる
    assert "\\:" in s
    assert "\\\\" in s
    # スペースはそのまま (ffmpeg は ' で囲んだ中ならスペース許容)
    assert "foo bar" in s


def test_escape_filter_path_with_single_quote_and_brackets():
    p = "/tmp/test'name[1].trf"
    s = stages.escape_filter_path(p)
    assert "\\'" in s
    assert "\\[" in s
    assert "\\]" in s


def test_escape_filter_path_round_trip_in_filter_string():
    p = Path(r"C:\Users\foo bar\in.mp4")
    snippet = stages.build_stabilize_transform_filter(p)
    # rgb24 segfault 回避のため format=yuv420p が前置され、その後に vidstabtransform が続く
    prefix = f"format={stages.STABILIZE_PIXFMT},vidstabtransform=input='"
    assert snippet.startswith(prefix)
    assert snippet.endswith(":smoothing=10")
    # Windows path のコロンが filter 内部のセパレータ ':' と衝突しないことを確認
    # 「input='...'」の中に escape された \: が現れる
    inside = snippet[len(prefix) : -len("':smoothing=10")]
    assert "\\:" in inside


# ---------- join_filters ----------


def test_join_filters_skips_empty():
    assert stages.join_filters(["a", "", "b", "", "c"]) == "a,b,c"
    assert stages.join_filters([]) == ""
    assert stages.join_filters(["", ""]) == ""


# ---------- pre-filter chain ----------


def test_pre_filter_chain_all_disabled_returns_empty():
    decs = _all({})
    assert stages.build_pre_filter_chain(decs) == ""


def test_pre_filter_chain_deinterlace_only():
    decs = _all({"deinterlace": _dec("deinterlace", enabled=True)})
    chain = stages.build_pre_filter_chain(decs)
    assert chain == "yadif=0"


def test_pre_filter_chain_deinterlace_never_doubles_frames():
    """P0-2 回帰: yadif=1 (send_field) はフレーム数を倍化させ、固定 framerate の mux で
    2 倍スロー + 音ズレを起こす (実機確認済み)。send_frame (yadif=0) 固定であること。"""
    decs = _all({"deinterlace": _dec("deinterlace", enabled=True)})
    chain = stages.build_pre_filter_chain(decs)
    assert "yadif=1" not in chain
    assert "yadif=0" in chain


def test_pre_filter_chain_full_order():
    decs = _all(
        {
            "deinterlace": _dec("deinterlace", enabled=True),
            "stabilize": _dec("stabilize", enabled=True),
            "deflicker": _dec("deflicker", enabled=True),
            "denoise": _dec("denoise", enabled=True),
            "debanding": _dec("debanding", enabled=True),
        }
    )
    chain = stages.build_pre_filter_chain(decs, transforms_path=Path(r"C:\tmp\t.trf"))
    parts = chain.split(",")
    # 順序: deinterlace -> stabilize(format=yuv420p + vidstabtransform) -> deflicker -> denoise -> debanding
    # stabilize は rgb24 segfault 回避のため format=yuv420p を前置するので 2 要素になる
    assert parts[0] == "yadif=0"
    assert parts[1] == f"format={stages.STABILIZE_PIXFMT}"
    assert parts[2].startswith("vidstabtransform=")
    assert parts[3].startswith("deflicker=")
    assert parts[4].startswith("hqdn3d=")
    assert parts[5].startswith("deband=")


def test_pre_filter_chain_stabilize_without_transforms_path_omits():
    decs = _all({"stabilize": _dec("stabilize", enabled=True)})
    chain = stages.build_pre_filter_chain(decs, transforms_path=None)
    assert chain == ""


# ---------- post-filter chain ----------


def test_post_filter_chain_all_disabled():
    assert stages.build_post_filter_chain(_all({})) == ""


def test_post_filter_chain_full_order():
    decs = _all(
        {
            "color_correction": _dec(
                "color_correction",
                enabled=True,
                params={
                    "brightness": 0.1,
                    "contrast": 1.1,
                    "saturation": 1.2,
                },
            ),
            "sharpen": _dec("sharpen", enabled=True, params={"amount": 0.6}),
            "grain": _dec("grain", enabled=True, params={"amount": 4}),
        }
    )
    chain = stages.build_post_filter_chain(decs)
    parts = chain.split(",")
    assert parts[0] == "eq=brightness=0.1:contrast=1.1:saturation=1.2"
    assert parts[1].startswith("unsharp=")
    assert parts[2] == "noise=alls=4:allf=t"


# ---------- encode ----------


def test_build_encode_args_defaults():
    dec = _dec("encode", enabled=True, params={})
    args = stages.build_encode_args(dec)
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


def test_build_encode_args_override():
    dec = _dec("encode", enabled=True, params={"crf": 18, "preset": "medium"})
    args = stages.build_encode_args(dec)
    assert args == [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
    ]


# ---------- RIFE ----------


def test_build_rife_args_integer_multiple_ok(tmp_path):
    args = stages.build_rife_args(
        rife_bin=tmp_path / "rife.exe",
        in_dir=tmp_path / "in",
        out_dir=tmp_path / "out",
        source_fps=Fraction(24, 1),
        target_fps=Fraction(48, 1),
        input_frame_count=100,
    )
    # -n は 100 * 2 = 200
    assert "-n" in args
    assert args[args.index("-n") + 1] == "200"
    assert "-f" in args
    assert args[args.index("-f") + 1] == "png"
    # `-n` は v4 系モデルでしか効かないため、`-m` 必須 (regression: see config.DEFAULT_RIFE_MODEL)
    assert "-m" in args
    assert args[args.index("-m") + 1] == "rife-v4.6"
    assert args.count("-v") == 1


@pytest.mark.parametrize(
    ("model", "target_fps", "has_custom_n"),
    [
        ("rife-v4.6", Fraction(48, 1), True),
        ("rife-v4", Fraction(72, 1), True),
        ("rife-v2.3", Fraction(48, 1), False),
        ("rife-v3.1", Fraction(48, 1), False),
    ],
)
def test_build_rife_args_always_enables_single_verbose_liveness(
    tmp_path, model, target_fps, has_custom_n
):
    args = stages.build_rife_args(
        rife_bin=tmp_path / "rife.exe",
        in_dir=tmp_path / "in",
        out_dir=tmp_path / "out",
        source_fps=Fraction(24, 1),
        target_fps=target_fps,
        input_frame_count=10,
        model=model,
    )
    assert args.count("-v") == 1
    assert ("-n" in args) is has_custom_n


def test_build_rife_args_fractional_source_fps_exact_double(tmp_path):
    # 24000/1001 -> 48000/1001 はちょうど 2 倍
    src = Fraction(24000, 1001)
    tgt = src * 2
    args = stages.build_rife_args(
        rife_bin=tmp_path / "rife.exe",
        in_dir=tmp_path / "in",
        out_dir=tmp_path / "out",
        source_fps=src,
        target_fps=tgt,
        input_frame_count=50,
    )
    assert args[args.index("-n") + 1] == "100"


def test_build_rife_args_explicit_model_overrides_default(tmp_path):
    """`model` 引数を明示すると `-m <model>` に反映される (config 既定をオーバーライド)。"""
    args = stages.build_rife_args(
        rife_bin=tmp_path / "rife.exe",
        in_dir=tmp_path / "in",
        out_dir=tmp_path / "out",
        source_fps=Fraction(24, 1),
        target_fps=Fraction(48, 1),
        input_frame_count=10,
        model="rife-v4",
    )
    assert args[args.index("-m") + 1] == "rife-v4"


def test_build_rife_args_none_model_falls_back_to_default(tmp_path):
    """`model=None` / 未指定なら config.DEFAULT_RIFE_MODEL が使われる。"""
    args = stages.build_rife_args(
        rife_bin=tmp_path / "rife.exe",
        in_dir=tmp_path / "in",
        out_dir=tmp_path / "out",
        source_fps=Fraction(24, 1),
        target_fps=Fraction(48, 1),
        input_frame_count=10,
        model=None,
    )
    from video_enhance_lite import config

    assert args[args.index("-m") + 1] == config.DEFAULT_RIFE_MODEL


def test_build_rife_args_non_integer_ratio_raises(tmp_path):
    with pytest.raises(ValueError, match="integer multiple"):
        stages.build_rife_args(
            rife_bin=tmp_path / "rife.exe",
            in_dir=tmp_path / "in",
            out_dir=tmp_path / "out",
            source_fps=Fraction(24, 1),
            target_fps=Fraction(50, 1),  # 50/24 = 25/12 → non integer
            input_frame_count=10,
        )


def test_build_rife_args_invalid_inputs(tmp_path):
    with pytest.raises(ValueError):
        stages.build_rife_args(
            rife_bin=tmp_path / "rife.exe",
            in_dir=tmp_path / "in",
            out_dir=tmp_path / "out",
            source_fps=Fraction(0, 1),
            target_fps=Fraction(48, 1),
            input_frame_count=10,
        )
    with pytest.raises(ValueError):
        stages.build_rife_args(
            rife_bin=tmp_path / "rife.exe",
            in_dir=tmp_path / "in",
            out_dir=tmp_path / "out",
            source_fps=Fraction(24, 1),
            target_fps=Fraction(48, 1),
            input_frame_count=0,
        )


# ---------- stabilize detect args ----------


def test_build_stabilize_detect_args_uses_null_muxer(tmp_path):
    argv = stages.build_stabilize_detect_args(
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        input_pattern=tmp_path / "frames_in" / "%08d.png",
        transforms_path=tmp_path / "transforms.trf",
        fps=Fraction(30, 1),
    )
    # 1st pass は -f null
    assert argv[-2:] == ["-f", "null"] or argv[-2:] == ["null", "-"] or "null" in argv
    assert any("vidstabdetect" in a for a in argv)
    # vidstabdetect の result 引数に transforms path が含まれる
    vf_idx = argv.index("-vf")
    assert "vidstabdetect" in argv[vf_idx + 1]
    assert "result=" in argv[vf_idx + 1]
    # rgb24 segfault 回避: vidstabdetect の前段に format=yuv420p が前置される (V1-1)
    assert argv[vf_idx + 1].startswith(f"format={stages.STABILIZE_PIXFMT},vidstabdetect=")


def test_stabilize_filters_force_yuv_pixfmt_regression():
    """V1-1 回帰: ffmpeg N-120725+ で vidstabtransform が rgb24 入力時に segfault するため、
    detect / transform の双方で vidstab フィルタ直前に format=yuv420p を強制する。"""
    assert stages.STABILIZE_PIXFMT == "yuv420p"
    # transform snippet
    trf = stages.build_stabilize_transform_filter(Path(r"C:\tmp\t.trf"))
    assert trf.startswith(f"format={stages.STABILIZE_PIXFMT},vidstabtransform=")
    # detect args の -vf
    det = stages.build_stabilize_detect_args(
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        input_pattern=Path(r"C:\tmp\frames_in\%08d.png"),
        transforms_path=Path(r"C:\tmp\t.trf"),
        fps=Fraction(24, 1),
    )
    det_vf = det[det.index("-vf") + 1]
    assert det_vf.startswith(f"format={stages.STABILIZE_PIXFMT},vidstabdetect=")
    # pre_filter chain でも format が vidstabtransform の直前に来る
    decs = _all({"stabilize": _dec("stabilize", enabled=True)})
    chain = stages.build_pre_filter_chain(decs, transforms_path=Path(r"C:\tmp\t.trf"))
    assert chain.startswith(f"format={stages.STABILIZE_PIXFMT},vidstabtransform=")


# ---------- extract / mux helpers ----------


def test_build_extract_args_layout(tmp_path):
    argv = stages.build_extract_args(
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        input_path=tmp_path / "in.mp4",
        start_sec=1.5,
        dur_sec=10.0,
        frames_out=tmp_path / "frames_in",
    )
    assert "-ss" in argv and "-t" in argv
    assert argv[argv.index("-ss") + 1] == "1.5"
    assert argv[argv.index("-t") + 1] == "10.0"
    # output is %08d.png in frames_in
    assert argv[-1].endswith("%08d.png")
    # accurate input seek: -ss は -i の前、-t は output duration として -i の後
    assert argv.index("-ss") < argv.index("-i") < argv.index("-t")


def test_build_extract_args_with_filter_chain(tmp_path):
    argv = stages.build_extract_args(
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        input_path=tmp_path / "in.mp4",
        start_sec=0.0,
        dur_sec=1.0,
        frames_out=tmp_path / "frames_pre",
        vf_chain="format=rgb24,hqdn3d=1.5:1.5:6:6",
    )
    assert argv[argv.index("-vf") + 1].startswith("format=rgb24,")
    assert Path(argv[-1]).parent.name == "frames_pre"
    assert Path(argv[-1]).name == "%08d.png"


def test_build_extract_args_normalizes_vfr_before_existing_filter_chain(tmp_path):
    argv = stages.build_extract_args(
        ffmpeg=Path("ffmpeg"),
        input_path=tmp_path / "vfr.mp4",
        start_sec=10.0,
        dur_sec=2.0,
        frames_out=tmp_path / "frames",
        vf_chain="format=rgb24",
        normalization_fps=Fraction(30000, 1001),
        expected_frame_count=60,
    )

    assert argv[argv.index("-vf") + 1] == "fps=30000/1001,format=rgb24"
    assert argv[argv.index("-fps_mode") + 1] == "cfr"
    assert argv[argv.index("-frames:v") + 1] == "60"


def test_build_extract_args_cfr_layout_remains_unchanged(tmp_path):
    argv = stages.build_extract_args(
        ffmpeg=Path("ffmpeg"),
        input_path=tmp_path / "cfr.mp4",
        start_sec=0.0,
        dur_sec=1.0,
        frames_out=tmp_path / "frames",
    )

    assert argv[argv.index("-fps_mode") + 1] == "passthrough"
    assert "-vf" not in argv
    assert "-frames:v" not in argv


def test_build_extract_args_keeps_output_seek_for_transport_stream(tmp_path):
    argv = stages.build_extract_args(
        ffmpeg=Path("ffmpeg"),
        input_path=tmp_path / "input.ts",
        start_sec=3.7,
        dur_sec=1.3,
        frames_out=tmp_path / "frames",
    )
    assert argv.index("-i") < argv.index("-ss") < argv.index("-t")
    assert argv[argv.index("-ss") + 1] == "3.7"


def test_build_mux_args_uses_provided_encode_args(tmp_path):
    encode = ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p"]
    argv = stages.build_mux_args(
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        frames_dir=tmp_path / "frames_out",
        fps=Fraction(30, 1),
        chunk_path=tmp_path / "chunks" / "chunk_000.mp4",
        encode_args=encode,
    )
    for token in encode:
        assert token in argv
    assert argv[-1].endswith("chunk_000.mp4")
    # framerate は分数形式
    assert argv[argv.index("-framerate") + 1] == "30/1"
