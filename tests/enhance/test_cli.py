from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest

from video_enhance_lite import analyzer, cli, config, decisions as decisions_mod
from video_enhance_lite.decisions import StageDecision


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


def _full_decs() -> list[StageDecision]:
    return decisions_mod.recommend(FakeMeta(field_order="tt", width=640), FakeStats())


# ---------- parser ----------


def test_parser_default_modes():
    p = cli._build_parser()
    args = p.parse_args(["in.mp4"])
    assert args.interactive is False
    assert args.auto is False
    assert args.auto_all is False
    assert args.chunk_seconds == config.DEFAULT_CHUNK_SECONDS
    assert args.encode_crf == 22
    assert args.encode_preset == "slow"
    assert args.keep_temp_on_failure is False


def test_parser_keep_temp_on_failure_flag():
    """V2-9: --keep-temp-on-failure フラグ。"""
    p = cli._build_parser()
    assert p.parse_args(["in.mp4"]).keep_temp_on_failure is False
    assert p.parse_args(["in.mp4", "--keep-temp-on-failure"]).keep_temp_on_failure is True


def test_parser_enable_disable_csv():
    p = cli._build_parser()
    args = p.parse_args(["in.mp4", "--enable", "denoise,sharpen", "--disable", "stabilize"])
    assert args.enable == ["denoise", "sharpen"]
    assert args.disable == ["stabilize"]


def test_parser_face_restore_options():
    p = cli._build_parser()
    assert p.parse_args(["in.mp4"]).face_restore_idle_timeout == 3000
    args = p.parse_args(
        [
            "in.mp4",
            "--enable",
            "face_restore",
            "--face-restore-start",
            "30",
            "--face-restore-end",
            "40",
            "--face-restore-fidelity",
            "0.8",
            "--face-restore-idle-timeout",
            "600",
            "--face-restore-blend",
            "0.3",
            "--face-restore-mask",
            "diff",
            "--face-restore-feather",
            "32",
            "--face-restore-mask-dilate",
            "12",
            "--face-restore-mask-threshold",
            "10",
            "--face-restore-match-grain",
            "0.2",
            "--face-restore-soften",
            "0.15",
            "--face-restore-global-smooth",
            "0.1",
        ]
    )
    assert args.enable == ["face_restore"]
    assert args.face_restore_backend == "codeformer-local"
    assert args.face_restore_start == 30.0
    assert args.face_restore_end == 40.0
    assert args.face_restore_fidelity == 0.8
    assert args.face_restore_idle_timeout == 600
    assert args.face_restore_blend == 0.3
    assert args.face_restore_mask == "diff"
    assert args.face_restore_feather == 32
    assert args.face_restore_mask_dilate == 12
    assert args.face_restore_mask_threshold == 10
    assert args.face_restore_match_grain == 0.2
    assert args.face_restore_soften == 0.15
    assert args.face_restore_global_smooth == 0.1


def test_parser_face_restore_fidelity_rejects_out_of_range():
    p = cli._build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["in.mp4", "--face-restore-fidelity", "1.1"])


@pytest.mark.parametrize(
    "option",
    [
        "--face-restore-blend",
        "--face-restore-match-grain",
        "--face-restore-soften",
        "--face-restore-global-smooth",
    ],
)
@pytest.mark.parametrize("bad", ["-0.1", "1.1", "nan"])
def test_parser_face_restore_naturalize_rejects_invalid_unit_values(option, bad):
    p = cli._build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["in.mp4", option, bad])


def test_parser_face_restore_mask_threshold_rejects_out_of_range():
    p = cli._build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["in.mp4", "--face-restore-mask-threshold", "256"])


@pytest.mark.parametrize(
    "option",
    [
        "--face-restore-feather",
        "--face-restore-mask-dilate",
    ],
)
def test_parser_face_restore_mask_radius_rejects_out_of_range(option):
    p = cli._build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["in.mp4", option, "257"])


def test_parser_mutually_exclusive_modes():
    p = cli._build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["in.mp4", "--auto", "--auto-all"])


def test_parser_scale_policy_and_clamp_flags_defaults():
    """PR-G2: 既定は cover / lanczos (従来挙動・不変)。"""
    p = cli._build_parser()
    args = p.parse_args(["in.mp4"])
    assert args.upscale_scale_policy == "cover"
    assert args.clamp_flags == "lanczos"


def test_parser_scale_policy_nearest():
    p = cli._build_parser()
    args = p.parse_args(["in.mp4", "--upscale-scale-policy", "nearest"])
    assert args.upscale_scale_policy == "nearest"


def test_parser_clamp_flags_bicubic():
    p = cli._build_parser()
    args = p.parse_args(["in.mp4", "--clamp-flags", "bicubic"])
    assert args.clamp_flags == "bicubic"


def test_parser_clamp_flags_rejects_invalid():
    p = cli._build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["in.mp4", "--clamp-flags", "nearest"])


# ---------- PR-G3: ncnn-vulkan 実行フラグ ----------


def test_parser_gpu_flags_defaults_none():
    p = cli._build_parser()
    args = p.parse_args(["in.mp4"])
    assert args.gpu_tilesize is None
    assert args.gpu_threads is None
    assert args.gpu_id is None
    assert args.rife_gpu_id is None


def test_parser_gpu_tilesize_valid():
    p = cli._build_parser()
    assert p.parse_args(["in.mp4", "--gpu-tilesize", "0"]).gpu_tilesize == 0  # auto
    assert p.parse_args(["in.mp4", "--gpu-tilesize", "512"]).gpu_tilesize == 512


def test_parser_gpu_tilesize_rejects_below_32():
    p = cli._build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["in.mp4", "--gpu-tilesize", "16"])


def test_parser_gpu_threads_valid():
    p = cli._build_parser()
    assert p.parse_args(["in.mp4", "--gpu-threads", "2:4:4"]).gpu_threads == "2:4:4"


@pytest.mark.parametrize("bad", ["a:b:c", "1:2", "1:2:3:4", "1:2,2,2:2"])
def test_parser_gpu_threads_rejects_invalid(bad):
    p = cli._build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["in.mp4", "--gpu-threads", bad])


def test_parser_gpu_id_rejects_negative():
    p = cli._build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["in.mp4", "--gpu-id", "-1"])
    assert p.parse_args(["in.mp4", "--gpu-id", "0"]).gpu_id == 0


def test_parser_rife_gpu_id_allows_minus1():
    p = cli._build_parser()
    assert p.parse_args(["in.mp4", "--rife-gpu-id", "-1"]).rife_gpu_id == -1
    assert p.parse_args(["in.mp4", "--rife-gpu-id", "0"]).rife_gpu_id == 0


def test_parser_rife_gpu_id_rejects_below_minus1_and_multigpu():
    p = cli._build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["in.mp4", "--rife-gpu-id", "-2"])
    with pytest.raises(SystemExit):
        p.parse_args(["in.mp4", "--rife-gpu-id", "0,1"])


# ---------- argument range validation (P3) ----------


@pytest.mark.parametrize(
    "flag,bad",
    [
        ("--encode-crf", "-1"),
        ("--encode-crf", "52"),
        ("--encode-preset", "bogus"),
        ("--scale", "1"),
        ("--scale", "5"),
        ("--target-width", "0"),
        ("--target-width", "15"),
        ("--target-fps", "0"),
        ("--target-fps", "-30"),
        ("--max-temp-gib", "0"),
        ("--max-temp-gib", "-5"),
        ("--max-width", "0"),
        ("--max-duration-sec", "0"),
    ],
)
def test_parser_rejects_out_of_range_values(flag, bad):
    p = cli._build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["in.mp4", flag, bad])


@pytest.mark.parametrize(
    "flag,good,expected",
    [
        ("--encode-crf", "0", 0),
        ("--encode-crf", "51", 51),
        ("--encode-preset", "slower", "slower"),
        ("--scale", "2", 2),
        ("--target-width", "16", 16),
        ("--target-fps", "23.976", 23.976),
        ("--max-temp-gib", "0.5", 0.5),
    ],
)
def test_parser_accepts_boundary_values(flag, good, expected):
    p = cli._build_parser()
    args = p.parse_args(["in.mp4", flag, good])
    dest = flag.lstrip("-").replace("-", "_")
    assert getattr(args, dest) == expected


def test_version_matches_pyproject():
    """__version__ と installed metadata (pyproject [project].version) の同期。"""
    import importlib.metadata
    import video_enhance_lite

    assert video_enhance_lite.__version__ == importlib.metadata.version("video-enhance-lite")


# ---------- apply_enable_disable ----------


def test_apply_enable_disable_overrides():
    decs = _full_decs()
    by = decisions_mod.decisions_by_key(decs)
    # decisions の初期 enabled をリセットしておく
    for d in decs:
        d.enabled = False
    decs, warns = cli._apply_enable_disable(
        decs,
        enable=["denoise"],
        disable=["upscale"],
        no_realesrgan=False,
        no_rife=False,
        encode_crf=20,
        encode_preset="medium",
        upscale_scale=2,
        upscale_model=config.DEFAULT_REALESRGAN_MODEL,
    )
    assert by["denoise"].enabled is True
    assert by["upscale"].enabled is False
    # encode は常時 ON 化される + params 上書き
    assert by["encode"].enabled is True
    assert by["encode"].params["crf"] == 20
    assert by["encode"].params["preset"] == "medium"
    assert warns == []


def test_apply_enable_disable_face_restore_params():
    decs = _full_decs()
    decs, warns = cli._apply_enable_disable(
        decs,
        enable=["face_restore"],
        disable=[],
        no_realesrgan=False,
        no_rife=False,
        encode_crf=20,
        encode_preset="medium",
        upscale_scale=None,
        upscale_model=config.DEFAULT_REALESRGAN_MODEL,
        face_restore_start=30.0,
        face_restore_end=40.0,
        face_restore_fidelity=0.9,
        face_restore_blend=0.3,
        face_restore_mask="diff",
        face_restore_feather=32,
        face_restore_mask_dilate=12,
        face_restore_mask_threshold=10,
        face_restore_match_grain=0.2,
        face_restore_soften=0.15,
        face_restore_global_smooth=0.1,
    )
    by = decisions_mod.decisions_by_key(decs)
    assert by["face_restore"].enabled is True
    assert by["face_restore"].params["backend"] == "codeformer-local"
    assert by["face_restore"].params["start_sec"] == 30.0
    assert by["face_restore"].params["end_sec"] == 40.0
    assert by["face_restore"].params["fidelity"] == 0.9
    assert by["face_restore"].params["blend"] == 0.3
    assert by["face_restore"].params["mask"] == "diff"
    assert by["face_restore"].params["feather"] == 32
    assert by["face_restore"].params["mask_dilate"] == 12
    assert by["face_restore"].params["mask_threshold"] == 10
    assert by["face_restore"].params["match_grain"] == 0.2
    assert by["face_restore"].params["soften"] == 0.15
    assert by["face_restore"].params["global_smooth"] == 0.1
    assert warns == []


def test_apply_enable_disable_unknown_key_warns():
    decs = _full_decs()
    _, warns = cli._apply_enable_disable(
        decs,
        enable=["bogus"],
        disable=[],
        no_realesrgan=False,
        no_rife=False,
        encode_crf=18,
        encode_preset="slow",
        upscale_scale=2,
        upscale_model=config.DEFAULT_REALESRGAN_MODEL,
    )
    assert any("bogus" in w for w in warns)


def test_apply_enable_disable_no_realesrgan_forces_off():
    decs = _full_decs()
    by = decisions_mod.decisions_by_key(decs)
    by["upscale"].enabled = True
    decs, _ = cli._apply_enable_disable(
        decs,
        enable=[],
        disable=[],
        no_realesrgan=True,
        no_rife=False,
        encode_crf=18,
        encode_preset="slow",
        upscale_scale=2,
        upscale_model=config.DEFAULT_REALESRGAN_MODEL,
    )
    assert by["upscale"].enabled is False


def test_apply_enable_disable_fixed_model_adjusts_auto_scale():
    """P1-3 回帰: --scale 未指定 + fixed kind モデルは許容 scale へ自動調整 + WARN。"""
    decs = decisions_mod.recommend(FakeMeta(width=640), FakeStats())  # 自動算出 scale=3
    by = decisions_mod.decisions_by_key(decs)
    assert by["upscale"].enabled is True
    assert by["upscale"].params["scale"] == 3
    decs, warns = cli._apply_enable_disable(
        decs,
        enable=[],
        disable=[],
        no_realesrgan=False,
        no_rife=False,
        encode_crf=22,
        encode_preset="slow",
        upscale_scale=None,  # --scale 未指定
        upscale_model="realesrgan-x4plus",  # fixed kind (scale=4 のみ)
    )
    assert by["upscale"].params["scale"] == 4
    assert any("scale=4 固定" in w for w in warns)


def test_apply_enable_disable_fixed_model_keeps_explicit_scale():
    """--scale 明示時は調整しない (validate で不一致を弾く従来契約を維持)。"""
    decs = decisions_mod.recommend(FakeMeta(width=640), FakeStats())
    by = decisions_mod.decisions_by_key(decs)
    decs, warns = cli._apply_enable_disable(
        decs,
        enable=[],
        disable=[],
        no_realesrgan=False,
        no_rife=False,
        encode_crf=22,
        encode_preset="slow",
        upscale_scale=2,  # 明示指定 (x4plus では validate エラーになるが黙って書き換えない)
        upscale_model="realesrgan-x4plus",
    )
    assert by["upscale"].params["scale"] == 2
    assert not any("固定" in w for w in warns)


def test_apply_enable_disable_scaled_model_unchanged():
    """既定モデル (scaled kind) では自動算出 scale を変更しない (回帰なし)。"""
    decs = decisions_mod.recommend(FakeMeta(width=640), FakeStats())
    by = decisions_mod.decisions_by_key(decs)
    auto_scale = by["upscale"].params["scale"]
    decs, warns = cli._apply_enable_disable(
        decs,
        enable=[],
        disable=[],
        no_realesrgan=False,
        no_rife=False,
        encode_crf=22,
        encode_preset="slow",
        upscale_scale=None,
        upscale_model=config.DEFAULT_REALESRGAN_MODEL,
    )
    assert by["upscale"].params["scale"] == auto_scale
    assert warns == []


# ---------- interactive ----------


def test_interactive_accepts_recommended_with_blank_input():
    decs = _full_decs()
    responses = iter([""] * len(decs))
    decs, cancelled = cli._interactive_confirm(
        decs,
        input_fn=lambda _p: next(responses, ""),
        output_fn=lambda _m: None,
    )
    assert cancelled is False
    by = decisions_mod.decisions_by_key(decs)
    # interactive で「Enter」した stage は recommended に従う
    for k, d in by.items():
        if k == "encode":
            assert d.enabled is True
        else:
            assert d.enabled == d.recommended


def test_interactive_invalid_input_reprompts_then_accepts():
    """無効入力は再プロンプトされ、有効入力で確定する (黙って既定値にしない)。"""
    decs = _full_decs()
    prompts: list[str] = []
    messages: list[str] = []
    # 最初のステージだけ 2 回無効入力 -> 3 回目に "y"。以降は Enter。
    answers = iter(["x", "zz", "y"] + [""] * (len(decs) * 3))
    decs, cancelled = cli._interactive_confirm(
        decs,
        input_fn=lambda p: prompts.append(p) or next(answers),
        output_fn=messages.append,
    )
    assert cancelled is False
    first_key = next(d.key for d in decs if d.key != "encode")
    by = decisions_mod.decisions_by_key(decs)
    assert by[first_key].enabled is True  # "y" が効いている
    assert any("いずれかを入力してください" in m for m in messages)


def test_interactive_invalid_input_falls_back_after_3_attempts():
    """無効入力が 3 回続いたら推奨値を使い、その旨を通知する。"""
    decs = _full_decs()
    messages: list[str] = []
    answers = iter(["x"] * 3 + [""] * (len(decs) * 3))
    decs, cancelled = cli._interactive_confirm(
        decs,
        input_fn=lambda _p: next(answers),
        output_fn=messages.append,
    )
    assert cancelled is False
    first_key = next(d.key for d in decs if d.key != "encode")
    by = decisions_mod.decisions_by_key(decs)
    assert by[first_key].enabled == by[first_key].recommended
    assert any("推奨値" in m for m in messages)


def test_interactive_quit_aborts():
    decs = _full_decs()
    decs, cancelled = cli._interactive_confirm(
        decs,
        input_fn=lambda _p: "q",
        output_fn=lambda _m: None,
    )
    assert cancelled is True


def test_interactive_all_accepts_remaining():
    decs = _full_decs()
    # 最初の non-encode で "a" を返し、以降は (呼ばれないはずだが) "n"
    answers = []
    encountered_keys = []

    def _input(prompt: str) -> str:
        # encode はスキップされるので prompt に encode は出ない
        for k in (d.key for d in decs):
            if f"[{k}]" in prompt:
                encountered_keys.append(k)
                break
        if not answers:
            answers.append("a")
            return "a"
        return "n"

    decs, cancelled = cli._interactive_confirm(
        decs,
        input_fn=_input,
        output_fn=lambda _: None,
    )
    assert cancelled is False
    # all_accept 後は推奨に従う
    for d in decs:
        if d.key == "encode":
            assert d.enabled is True
        elif d.key == decs[0].key and d != decs[0]:
            pass
        else:
            assert d.enabled == d.recommended


# ---------- auto mode ----------


def test_auto_mode_defers_quality_altering_stages():
    """推奨に denoise(medium) などが含まれる場合 deferred に積まれる."""
    decs = decisions_mod.recommend(
        FakeMeta(width=640, field_order="tt"),
        FakeStats(highfreq_energy=config.T_NOISE + 5.0),  # denoise on
    )
    decs, deferred = cli._apply_auto_mode(decs)
    by = decisions_mod.decisions_by_key(decs)
    # high-confidence low-risk のみ ON
    assert by["upscale"].enabled is True  # high
    assert by["deinterlace"].enabled is True  # high
    assert by["encode"].enabled is True
    # denoise は deferred
    assert "denoise" in deferred
    assert by["denoise"].enabled is False


def test_auto_mode_progressive_no_deferred_when_nothing_quality_altering():
    decs = decisions_mod.recommend(FakeMeta(width=640), FakeStats())
    # FakeStats は閾値超えなしなので低リスクの upscale (high) のみ推奨
    decs, deferred = cli._apply_auto_mode(decs)
    by = decisions_mod.decisions_by_key(decs)
    assert by["upscale"].enabled is True
    assert deferred == []


def test_auto_mode_explicit_enable_is_not_deferred():
    """P1-2 回帰: --enable で明示同意したステージは deferred に積まず ON にする。"""
    decs = decisions_mod.recommend(
        FakeMeta(width=640, field_order="tt"),
        FakeStats(highfreq_energy=config.T_NOISE + 5.0),  # denoise 推奨
    )
    decs, deferred = cli._apply_auto_mode(decs, explicit_enable={"denoise"})
    by = decisions_mod.decisions_by_key(decs)
    assert by["denoise"].enabled is True
    assert "denoise" not in deferred


def test_auto_mode_explicit_enable_partial_still_defers_others():
    """--enable で同意していない推奨ステージは従来どおり保留される。"""
    decs = decisions_mod.recommend(
        FakeMeta(width=640, field_order="tt"),
        FakeStats(
            highfreq_energy=config.T_NOISE + 5.0,  # denoise 推奨
            interframe_diff_std=config.T_SHAKE + 5.0,  # stabilize 推奨
        ),
    )
    decs, deferred = cli._apply_auto_mode(decs, explicit_enable={"denoise"})
    by = decisions_mod.decisions_by_key(decs)
    assert by["denoise"].enabled is True
    assert "stabilize" in deferred
    assert by["stabilize"].enabled is False


def test_auto_mode_explicit_enable_turns_on_non_recommended():
    """--enable は推奨されていないステージも ON にできる (force-enable 契約)。"""
    decs = decisions_mod.recommend(FakeMeta(width=1920), FakeStats())
    by = decisions_mod.decisions_by_key(decs)
    assert by["sharpen"].recommended is False
    decs, deferred = cli._apply_auto_mode(decs, explicit_enable={"sharpen"})
    assert by["sharpen"].enabled is True
    assert "sharpen" not in deferred


def test_main_auto_with_enable_exits_zero(monkeypatch, tmp_path, capsys):
    """P1-2 回帰 (main 経路): --auto --enable <保留ステージ全部> で exit 1 にならず
    pipeline まで到達する。"""
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

    from video_enhance_lite import analyzer

    monkeypatch.setattr(
        analyzer,
        "analyze",
        lambda **kw: (
            FakeMeta(width=1920),
            FakeStats(highfreq_energy=config.T_NOISE + 5.0),
        ),  # denoise+grain 推奨
    )

    captured: dict = {}
    from video_enhance_lite import pipeline as ven_pipeline

    monkeypatch.setattr(ven_pipeline, "run_pipeline", lambda **kw: captured.update(kw))

    rc = cli.main([str(input_path), "--auto", "--enable", "denoise,grain", "--no-rife"])
    assert rc == 0
    by = decisions_mod.decisions_by_key(captured["decisions"])
    assert by["denoise"].enabled is True
    assert by["grain"].enabled is True


def test_main_unknown_framerate_allows_analysis_but_fails_before_pipeline(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    input_path = tmp_path / "unknown-rate.mp4"
    input_path.write_bytes(b"")

    from video_enhance_lite import analyzer as ven_analyzer
    from video_enhance_lite import deps as ven_deps
    from video_enhance_lite import pipeline as ven_pipeline

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
    meta = FakeMeta()
    meta.framerate_status = "unknown"
    meta.avg_fps = None
    monkeypatch.setattr(ven_analyzer, "analyze", lambda **_kwargs: (meta, FakeStats()))
    pipeline_called = False

    def _unexpected_pipeline(**_kwargs):
        nonlocal pipeline_called
        pipeline_called = True

    monkeypatch.setattr(ven_pipeline, "run_pipeline", _unexpected_pipeline)

    analyze_rc = cli.main([str(input_path), "--analyze-only"])
    analyze_output = capsys.readouterr()
    assert analyze_rc == 0
    assert "analyze-onlyのため診断表示は継続" in analyze_output.err

    rc = cli.main([str(input_path), "--auto", "--no-rife", "--no-realesrgan"])

    assert rc == 2
    assert "framerate_preflight" in capsys.readouterr().err
    assert pipeline_called is False


def test_auto_all_accepts_recommended():
    decs = decisions_mod.recommend(
        FakeMeta(width=640, field_order="tt"),
        FakeStats(highfreq_energy=config.T_NOISE + 5.0),
    )
    decs = cli._apply_auto_all_mode(decs)
    by = decisions_mod.decisions_by_key(decs)
    assert by["denoise"].enabled is True
    assert by["upscale"].enabled is True
    assert by["encode"].enabled is True


# ---------- forced overrides (明示フラグはモードより後で勝つ) ----------


def test_forced_overrides_no_realesrgan_no_rife_win_over_auto_all():
    """--auto-all で upscale/interpolate が ON になっても、--no-realesrgan/--no-rife が勝つ."""
    decs = decisions_mod.recommend(
        FakeMeta(width=640, fps=Fraction(24, 1)),
        FakeStats(highfreq_energy=config.T_NOISE + 5.0),
    )
    decs = cli._apply_auto_all_mode(decs)
    by = decisions_mod.decisions_by_key(decs)
    # 前提: auto-all が一旦両方 ON にしている
    assert by["upscale"].enabled is True
    assert by["interpolate"].enabled is True

    decs = cli._apply_forced_overrides(
        decs,
        enable=[],
        disable=[],
        no_realesrgan=True,
        no_rife=True,
    )
    assert by["upscale"].enabled is False
    assert by["interpolate"].enabled is False
    # 他の推奨段とエンコードは維持される
    assert by["denoise"].enabled is True
    assert by["encode"].enabled is True


def test_forced_overrides_disable_wins_over_auto_all():
    decs = decisions_mod.recommend(
        FakeMeta(width=640),
        FakeStats(highfreq_energy=config.T_NOISE + 5.0),
    )
    decs = cli._apply_auto_all_mode(decs)
    by = decisions_mod.decisions_by_key(decs)
    assert by["denoise"].enabled is True
    decs = cli._apply_forced_overrides(
        decs,
        enable=[],
        disable=["denoise"],
        no_realesrgan=False,
        no_rife=False,
    )
    assert by["denoise"].enabled is False


def test_forced_overrides_enable_wins_over_auto():
    """--auto は denoise 等を保留 (off) するが、--enable は最終的に ON にする."""
    decs = decisions_mod.recommend(FakeMeta(width=1920), FakeStats())
    decs, _deferred = cli._apply_auto_mode(decs)
    by = decisions_mod.decisions_by_key(decs)
    assert by["sharpen"].enabled is False
    decs = cli._apply_forced_overrides(
        decs,
        enable=["sharpen"],
        disable=[],
        no_realesrgan=False,
        no_rife=False,
    )
    assert by["sharpen"].enabled is True


def test_main_auto_all_no_realesrgan_keeps_resolution(monkeypatch, tmp_path, capsys):
    """main() 経由: --auto-all --no-realesrgan --no-rife で upscale/interpolate が OFF のまま
    pipeline に渡る (= 解像度・fps 据え置き)."""
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

    from video_enhance_lite import analyzer

    monkeypatch.setattr(
        analyzer,
        "analyze",
        lambda **kw: (
            FakeMeta(width=640, fps=Fraction(24, 1)),
            FakeStats(highfreq_energy=config.T_NOISE + 5.0),
        ),
    )

    captured: dict = {}
    from video_enhance_lite import pipeline as ven_pipeline

    def _fake_run(**kwargs):
        captured["decisions"] = kwargs["decisions"]

    monkeypatch.setattr(ven_pipeline, "run_pipeline", _fake_run)

    rc = cli.main([str(input_path), "--auto-all", "--no-realesrgan", "--no-rife"])
    assert rc == 0
    by = decisions_mod.decisions_by_key(captured["decisions"])
    assert by["upscale"].enabled is False
    assert by["interpolate"].enabled is False
    # データ量に影響しない補正は通常どおり採用 (回帰でないこと)
    assert by["denoise"].enabled is True
    assert by["encode"].enabled is True


# ---------- stabilize auto backend resolution (P2-4) ----------


def _setup_stabilize_main_fakes(monkeypatch, tmp_path, *, vidstab_available: bool):
    """stabilize が推奨される素材で main() を回すフェイク一式。"""
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

    def _vidstab():
        if vidstab_available:
            return ven_deps.ResolvedBinary(
                path=Path(r"C:\bin\ffmpeg.exe"), version="n", sha256=None
            )
        raise ven_deps.MissingDependencyError("ffmpeg(vidstab)", "no vidstab in build")

    monkeypatch.setattr(ven_deps, "check_ffmpeg_with_vidstab", _vidstab)

    from video_enhance_lite import analyzer

    monkeypatch.setattr(
        analyzer,
        "analyze",
        lambda **kw: (
            FakeMeta(width=1920),
            FakeStats(interframe_diff_std=config.T_SHAKE + 5.0),
        ),  # stabilize 推奨
    )

    captured: dict = {}
    from video_enhance_lite import pipeline as ven_pipeline

    monkeypatch.setattr(ven_pipeline, "run_pipeline", lambda **kw: captured.update(kw))
    return input_path, captured


def test_main_stabilize_auto_resolves_to_vidstab_when_available(monkeypatch, tmp_path):
    input_path, captured = _setup_stabilize_main_fakes(
        monkeypatch, tmp_path, vidstab_available=True
    )
    rc = cli.main([str(input_path), "--auto-all", "--no-rife", "--no-realesrgan"])
    assert rc == 0
    assert captured["stabilize_backend"] == "vidstab"


def test_main_stabilize_auto_falls_back_to_opencv(monkeypatch, tmp_path, capsys):
    """P2-4 回帰: vidstab 非搭載 ffmpeg + auto (既定) では opencv に解決して続行する
    (旧実装は run_pipeline が vidstab を仮定し、実行時 ffmpeg エラーになっていた)。"""
    input_path, captured = _setup_stabilize_main_fakes(
        monkeypatch, tmp_path, vidstab_available=False
    )
    rc = cli.main([str(input_path), "--auto-all", "--no-rife", "--no-realesrgan"])
    assert rc == 0
    assert captured["stabilize_backend"] == "opencv"
    assert "OpenCV" in capsys.readouterr().err


def test_main_stabilize_explicit_vidstab_still_errors_when_missing(monkeypatch, tmp_path, capsys):
    """--stabilize-backend vidstab 明示時は従来どおり exit 2 (フォールバックしない)。"""
    input_path, captured = _setup_stabilize_main_fakes(
        monkeypatch, tmp_path, vidstab_available=False
    )
    rc = cli.main(
        [
            str(input_path),
            "--auto-all",
            "--no-rife",
            "--no-realesrgan",
            "--stabilize-backend",
            "vidstab",
        ]
    )
    assert rc == 2
    assert not captured


# ---------- format_decision_table ----------


def test_format_decision_table_lists_all_stages():
    decs = decisions_mod.recommend(FakeMeta(), FakeStats())
    table = cli._format_decision_table(decs)
    for k in config.STAGE_KEYS:
        assert k in table


def test_format_face_restore_candidates_summary():
    stats = FakeStats()
    candidate = analyzer.FaceRestoreCandidateRange(
        start_sec=12.0,
        end_sec=18.5,
        sample_count=1,
        face_sample_count=1,
        max_face_ratio=0.03,
        confidence="medium",
        note="test",
    )
    stats.face_restore_candidates = (candidate,)
    decs = decisions_mod.recommend(FakeMeta(), stats)
    assert cli._format_face_restore_candidate_summary(decs) == (
        "[face_restore candidates] 12.0s-18.5s medium"
    )


def test_interactive_confirm_prints_face_restore_candidates_before_prompt():
    stats = FakeStats()
    stats.face_restore_candidates = (
        analyzer.FaceRestoreCandidateRange(
            start_sec=12.0,
            end_sec=18.5,
            sample_count=1,
            face_sample_count=1,
            max_face_ratio=0.03,
            confidence="medium",
            note="test",
        ),
    )
    decs = decisions_mod.recommend(FakeMeta(), stats)
    lines: list[str] = []

    _out, cancelled = cli._interactive_confirm(
        decs,
        input_fn=lambda _prompt: "q",
        output_fn=lines.append,
    )

    assert cancelled is True
    assert lines[0] == "[face_restore candidates] 12.0s-18.5s medium"
    assert "CodeFormer" in lines[1]


# ---------- main (analyze-only) ----------


def test_main_analyze_only(monkeypatch, tmp_path, capsys):
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

    fake_meta = FakeMeta()
    fake_stats = FakeStats()
    from video_enhance_lite import analyzer

    monkeypatch.setattr(analyzer, "analyze", lambda **kw: (fake_meta, fake_stats))

    rc = cli.main([str(input_path), "--analyze-only"])
    assert rc == 0
    captured = capsys.readouterr()
    # 表の全 stage が出力されている
    for k in config.STAGE_KEYS:
        assert k in captured.out
    assert "[face_restore candidates]" not in captured.out


def test_main_analyze_only_prints_face_restore_candidates(monkeypatch, tmp_path, capsys):
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

    fake_stats = FakeStats()
    fake_stats.face_restore_candidates = (
        analyzer.FaceRestoreCandidateRange(
            start_sec=12.0,
            end_sec=18.5,
            sample_count=1,
            face_sample_count=1,
            max_face_ratio=0.03,
            confidence="medium",
            note="test",
        ),
    )
    monkeypatch.setattr(analyzer, "analyze", lambda **kw: (FakeMeta(), fake_stats))

    rc = cli.main([str(input_path), "--analyze-only"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[face_restore candidates] 12.0s-18.5s medium" in out
    assert "may differ from CodeFormer detection" in out


def test_main_input_not_found(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    rc = cli.main([str(tmp_path / "missing.mp4")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err


def test_main_output_outside_workspace_is_not_rejected_by_policy(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    input_path = tmp_path / "in.mp4"
    input_path.write_bytes(b"")
    other = tmp_path.parent / "outside" / "out.mp4"
    rc = cli.main([str(input_path), "-o", str(other)])
    # 出力先ポリシーでは止まらず、未スタブの依存確認へ進む。
    assert rc == 2
    err = capsys.readouterr().err
    assert "workspace policy" not in err
    assert "must be under" not in err


def test_main_analyze_failed_returns_2(monkeypatch, tmp_path, capsys):
    """V2-13: analyzer.analyze が AnalyzerError を投げると exit code 2 で失敗する。"""
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

    from video_enhance_lite import analyzer

    def _boom(**kwargs):
        raise analyzer.AnalyzerError("fake analyze failure")

    monkeypatch.setattr(analyzer, "analyze", _boom)

    rc = cli.main([str(input_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "analyze failed" in err
