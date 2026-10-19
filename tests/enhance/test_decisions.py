from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import pytest

from video_enhance_lite import analyzer, config, decisions


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


def _by_key(decs):
    return {d.key: d for d in decs}


# ---------- structure ----------


def test_recommend_returns_all_stage_keys():
    decs = decisions.recommend(FakeMeta(), FakeStats())
    keys = [d.key for d in decs]
    assert keys == list(config.STAGE_KEYS)


def test_recommend_enabled_defaults_to_recommended():
    decs = decisions.recommend(FakeMeta(), FakeStats())
    for d in decs:
        assert d.enabled == d.recommended


def test_recommend_encode_always_on():
    decs = decisions.recommend(FakeMeta(), FakeStats())
    enc = _by_key(decs)["encode"]
    assert enc.recommended is True
    assert enc.confidence == "high"
    assert enc.params["crf"] == config.DEFAULT_ENCODE_CRF
    assert enc.params["preset"] == config.DEFAULT_ENCODE_PRESET


def test_face_restore_is_between_upscale_and_interpolate_and_default_off():
    decs = decisions.recommend(FakeMeta(), FakeStats())
    keys = [d.key for d in decs]
    assert keys.index("upscale") < keys.index("face_restore") < keys.index("interpolate")
    face = _by_key(decs)["face_restore"]
    assert face.recommended is False
    assert face.enabled is False
    assert face.confidence == "unknown"
    assert face.params["candidate_ranges"] == ()


def test_face_restore_candidate_ranges_are_copied_without_enabling():
    stats = FakeStats()
    candidate = analyzer.FaceRestoreCandidateRange(
        start_sec=12.0,
        end_sec=18.5,
        sample_count=2,
        face_sample_count=2,
        max_face_ratio=0.04,
        confidence="high",
        note="test",
    )
    stats.face_restore_candidates = (candidate,)
    decs = decisions.recommend(FakeMeta(), stats)
    face = _by_key(decs)["face_restore"]
    assert face.params["candidate_ranges"] == (candidate,)
    assert face.recommended is False
    assert face.enabled is False


# ---------- deinterlace ----------


def test_deinterlace_progressive_off_high():
    decs = decisions.recommend(FakeMeta(field_order="progressive"), FakeStats())
    d = _by_key(decs)["deinterlace"]
    assert d.recommended is False
    assert d.confidence == "high"


def test_deinterlace_interlaced_tt_on_high():
    decs = decisions.recommend(FakeMeta(field_order="tt"), FakeStats())
    d = _by_key(decs)["deinterlace"]
    assert d.recommended is True
    assert d.confidence == "high"


def test_deinterlace_unknown_off_unknown_confidence():
    decs = decisions.recommend(FakeMeta(field_order="unknown"), FakeStats())
    d = _by_key(decs)["deinterlace"]
    assert d.recommended is False
    assert d.confidence == "unknown"


# ---------- stabilize ----------


def test_stabilize_above_threshold_recommended_on_with_opencv_backend():
    """PR2 (TODO V1-1 解消): stabilize_backend=opencv で vidstabtransform を
    回避できるため、手ぶれ検出時は recommended=True に戻す。
    """
    stats = FakeStats(interframe_diff_std=config.T_SHAKE + 0.01)
    decs = decisions.recommend(FakeMeta(), stats)
    d = _by_key(decs)["stabilize"]
    assert d.recommended is True
    assert d.enabled is True
    assert "手ぶれ検出" in d.reason
    assert "OpenCV" in d.reason


def test_stabilize_at_threshold_off():
    # 境界値 (<=) でも当然 OFF (手ぶれなし)
    stats = FakeStats(interframe_diff_std=config.T_SHAKE)
    decs = decisions.recommend(FakeMeta(), stats)
    d = _by_key(decs)["stabilize"]
    assert d.recommended is False
    assert d.enabled is False
    assert "手ぶれ検出" not in d.reason


# ---------- deflicker ----------


def test_deflicker_threshold_boundary():
    on = decisions.recommend(FakeMeta(), FakeStats(luma_std_over_time=config.T_FLICKER + 0.1))
    off = decisions.recommend(FakeMeta(), FakeStats(luma_std_over_time=config.T_FLICKER))
    assert _by_key(on)["deflicker"].recommended is True
    assert _by_key(off)["deflicker"].recommended is False


# ---------- denoise / grain ----------


def test_denoise_on_via_highfreq():
    decs = decisions.recommend(FakeMeta(), FakeStats(highfreq_energy=config.T_NOISE + 1.0))
    d = _by_key(decs)
    assert d["denoise"].recommended is True
    # grain は denoise ON のとき推奨
    assert d["grain"].recommended is True


def test_denoise_on_via_low_bitrate():
    decs = decisions.recommend(FakeMeta(bitrate_bps=1_500_000), FakeStats())
    d = _by_key(decs)
    assert d["denoise"].recommended is True
    assert "bitrate" in d["denoise"].reason


def test_denoise_off_grain_off():
    decs = decisions.recommend(FakeMeta(bitrate_bps=5_000_000), FakeStats(highfreq_energy=1.0))
    d = _by_key(decs)
    assert d["denoise"].recommended is False
    assert d["grain"].recommended is False


# ---------- debanding ----------


def test_debanding_on_above_threshold():
    decs = decisions.recommend(FakeMeta(), FakeStats(histogram_flat_ratio=config.T_BANDING + 0.01))
    assert _by_key(decs)["debanding"].recommended is True


# ---------- color_correction ----------


@pytest.mark.parametrize(
    "luma,expected",
    [
        (config.T_LUMA_DARK - 1.0, True),
        (config.T_LUMA_BRIGHT + 1.0, True),
        (120.0, False),
    ],
)
def test_color_correction_luma(luma, expected):
    decs = decisions.recommend(FakeMeta(), FakeStats(luma_mean=luma))
    assert _by_key(decs)["color_correction"].recommended is expected


def test_color_correction_clipping_triggers():
    decs = decisions.recommend(
        FakeMeta(), FakeStats(luma_clipping_ratio=config.T_LUMA_CLIPPING + 0.001)
    )
    assert _by_key(decs)["color_correction"].recommended is True


def test_color_correction_low_saturation_triggers():
    decs = decisions.recommend(FakeMeta(), FakeStats(saturation_mean=config.T_SATURATION_LOW - 1.0))
    assert _by_key(decs)["color_correction"].recommended is True


# ---------- sharpen ----------


def test_sharpen_on_when_blurry():
    decs = decisions.recommend(FakeMeta(), FakeStats(laplacian_var_mean=config.T_BLUR - 1.0))
    assert _by_key(decs)["sharpen"].recommended is True


def test_sharpen_off_when_sharp():
    decs = decisions.recommend(FakeMeta(), FakeStats(laplacian_var_mean=config.T_BLUR + 1.0))
    assert _by_key(decs)["sharpen"].recommended is False


# ---------- upscale ----------


def test_upscale_on_when_below_target():
    decs = decisions.recommend(FakeMeta(width=1280), FakeStats(), defaults={"target_width": 1920})
    d = _by_key(decs)["upscale"]
    assert d.recommended is True
    assert d.confidence == "high"


def test_upscale_off_when_at_target():
    decs = decisions.recommend(FakeMeta(width=1920), FakeStats(), defaults={"target_width": 1920})
    assert _by_key(decs)["upscale"].recommended is False


# ---------- interpolate ----------


def test_interpolate_low_fps_on_doubles():
    decs = decisions.recommend(FakeMeta(fps=Fraction(24, 1)), FakeStats())
    d = _by_key(decs)["interpolate"]
    assert d.recommended is True
    # target_fps が source の 2 倍 = 48
    f = Fraction(d.params["target_fps_num"], d.params["target_fps_den"])
    assert f == Fraction(48, 1)


def test_interpolate_high_fps_off():
    decs = decisions.recommend(FakeMeta(fps=Fraction(60, 1)), FakeStats())
    assert _by_key(decs)["interpolate"].recommended is False


def test_interpolate_explicit_target_fps_rounding():
    # source=24 / target=50 (≒ 2.08倍) -> 2倍 = 48 に丸める
    decs = decisions.recommend(
        FakeMeta(fps=Fraction(24, 1)),
        FakeStats(),
        defaults={"target_fps": Fraction(50, 1)},
    )
    d = _by_key(decs)["interpolate"]
    f = Fraction(d.params["target_fps_num"], d.params["target_fps_den"])
    assert f == Fraction(48, 1)
    assert "warn" in d.reason.lower() or "warn" in d.reason


def test_interpolate_fractional_source_fps():
    # 24000/1001 のような非整数 fps でも整数倍判定が壊れない
    src = Fraction(24000, 1001)
    decs = decisions.recommend(FakeMeta(fps=src), FakeStats(), defaults={"target_fps": src * 2})
    d = _by_key(decs)["interpolate"]
    f = Fraction(d.params["target_fps_num"], d.params["target_fps_den"])
    assert f == src * 2
    # 整数倍ぴったりなので warn なし
    assert "warn" not in d.reason.lower()


# ---------- normalize_target_fps ----------


def test_normalize_target_fps_exact_multiple():
    src = Fraction(30, 1)
    normalized, warn = decisions.normalize_target_fps(src, Fraction(60, 1))
    assert normalized == Fraction(60, 1)
    assert warn is None


def test_normalize_target_fps_rounds_up():
    src = Fraction(24, 1)
    # 50 / 24 ≒ 2.083 -> 2 倍
    normalized, warn = decisions.normalize_target_fps(src, Fraction(50, 1))
    assert normalized == Fraction(48, 1)
    assert warn is not None


def test_normalize_target_fps_below_source_floored_to_source():
    src = Fraction(30, 1)
    # 20 / 30 = 0.67 -> 1 倍 (=source)
    normalized, warn = decisions.normalize_target_fps(src, Fraction(20, 1))
    assert normalized == src
    assert warn is not None


def test_normalize_target_fps_invalid_input():
    with pytest.raises(ValueError):
        decisions.normalize_target_fps(Fraction(0, 1), Fraction(60, 1))
    with pytest.raises(ValueError):
        decisions.normalize_target_fps(Fraction(30, 1), Fraction(0, 1))


# ---------- helpers ----------


def test_decisions_by_key_and_filter_enabled():
    decs = decisions.recommend(FakeMeta(), FakeStats())
    by = decisions.decisions_by_key(decs)
    assert set(by.keys()) == set(config.STAGE_KEYS)
    # 全部 disable -> filter_enabled が空
    for d in decs:
        d.enabled = False
    assert decisions.filter_enabled(decs) == []
    # 一部のみ enable
    decs[0].enabled = True
    assert [d.key for d in decisions.filter_enabled(decs)] == [decs[0].key]


# ---------- compute_output_summary ----------


def test_output_summary_auto_upscale_and_interp():
    """source 1280x720@29.97 + auto preset:
    - upscale ON: scale=2 → 中間 2560px → clamp 不要 (target 1920? いいえ 2560!= 1920 だから clamp)
      待って: needed = ceil(1920/1280)=2 → scale=2 → 中間 2560px > target 1920px → 下方 clamp
    - interpolate ON (29.97 < 30): ×2 → 59.94 fps
    """
    meta = FakeMeta(width=1280, height=720, fps=Fraction(30000, 1001))
    stats = FakeStats()
    decs = decisions.recommend(meta, stats)
    s = decisions.compute_output_summary(meta, decs)
    assert s.src_width == 1280 and s.src_height == 720
    assert s.upscale_enabled is True
    assert s.upscale_scale == 2
    assert s.intermediate_width == 2560
    assert s.out_width == 1920
    # 高さは AR 保持 + 2 の倍数: 720 * 1920 / 1280 = 1080
    assert s.out_height == 1080
    assert s.resolution_clamp_applied is True
    assert s.resolution_clamp_upscale is False  # 2560 > 1920 → 下方 lanczos
    assert s.interpolate_enabled is True
    assert s.interpolate_multiplier == 2
    assert s.out_fps == Fraction(60000, 1001)
    # encode + upscale + interpolate は最低限含まれる
    assert "encode" in s.active_stage_keys
    assert "upscale" in s.active_stage_keys
    assert "interpolate" in s.active_stage_keys


def test_output_summary_4k_preset_triggers_upper_lanczos():
    """source 1280x720@30 + target_width=3840 (4K) → needed=3 → scale=3 → 中間 3840px
    → clamp 不要 (intermediate == target)。
    """
    meta = FakeMeta(width=1280, height=720, fps=Fraction(30, 1))
    stats = FakeStats()
    defs = {"target_width": 3840, "target_width_explicit": True}
    decs = decisions.recommend(meta, stats, defs)
    s = decisions.compute_output_summary(meta, decs)
    assert s.upscale_enabled is True
    assert s.upscale_scale == 3
    assert s.intermediate_width == 3840
    assert s.out_width == 3840
    assert s.out_height == 2160
    assert s.resolution_clamp_applied is False
    assert s.resolution_clamp_upscale is False


def test_output_summary_8k_preset_caps_at_scale4_then_upper_lanczos():
    """source 1280x720 + target_width=7680 → needed=6 → scale=4 で頭打ち → 中間 5120px
    → target 7680px へ上方 lanczos clamp。
    """
    meta = FakeMeta(width=1280, height=720, fps=Fraction(30, 1))
    stats = FakeStats()
    defs = {"target_width": 7680, "target_width_explicit": True}
    decs = decisions.recommend(meta, stats, defs)
    s = decisions.compute_output_summary(meta, decs)
    assert s.upscale_scale == 4
    assert s.intermediate_width == 5120
    assert s.out_width == 7680
    assert s.resolution_clamp_applied is True
    assert s.resolution_clamp_upscale is True  # 7680 > 5120


def test_output_summary_upscale_off_keeps_source_size():
    """source 1920x1080 >= target_width=1920 → upscale OFF → 解像度維持."""
    meta = FakeMeta(width=1920, height=1080, fps=Fraction(60, 1))
    stats = FakeStats()
    decs = decisions.recommend(meta, stats)
    s = decisions.compute_output_summary(meta, decs)
    assert s.upscale_enabled is False
    assert s.upscale_scale is None
    assert s.intermediate_width is None
    assert s.out_width == 1920
    assert s.out_height == 1080
    assert s.resolution_clamp_applied is False


def test_output_summary_interpolate_off_keeps_source_fps():
    """source 60fps >= T_FPS_LOW=30 + Auto preset → 補間 OFF → fps 維持."""
    meta = FakeMeta(width=1920, height=1080, fps=Fraction(60, 1))
    stats = FakeStats()
    decs = decisions.recommend(meta, stats)
    s = decisions.compute_output_summary(meta, decs)
    assert s.interpolate_enabled is False
    assert s.interpolate_multiplier is None
    assert s.out_fps == Fraction(60, 1)


def test_output_summary_active_stages_reflects_enabled_only():
    """enabled な stage のみが active_stage_keys に並ぶ。順序は decisions の順を保持."""
    meta = FakeMeta(width=1280, height=720, fps=Fraction(30, 1), field_order="tt")
    decs = decisions.recommend(meta, FakeStats())
    # 手動で全部 OFF にしてから denoise だけ ON にする (encode は常時 ON 扱い)
    for d in decs:
        d.enabled = d.key == "encode"
    by = decisions.decisions_by_key(decs)
    by["denoise"].enabled = True
    s = decisions.compute_output_summary(meta, decs)
    assert s.active_stage_keys == ("denoise", "encode")
    # ラベルが日本語で返ること
    assert "ノイズ" in s.active_stage_labels[0]


def test_output_summary_height_aligned_to_even():
    """src height 721 → 偶数アライン。1920/1281 ≒ 1.499 → 1081 → 1080 へ偶数アライン。"""
    meta = FakeMeta(width=1281, height=721, fps=Fraction(30, 1))
    decs = decisions.recommend(meta, FakeStats())
    s = decisions.compute_output_summary(meta, decs)
    assert s.out_height % 2 == 0


# ---------- interpolate model param (rife-v4 必須) ----------


def test_interpolate_decision_always_carries_rife_v4_model():
    """interpolate decision の params には常に `model` キーが入り、v4 系が選ばれる。

    rife-ncnn-vulkan の `-n` (custom numframe) は v4 系モデルでしか動かないため
    (`only rife-v4 model support custom numframe and timestep` エラー)、
    decisions が常に v4 系モデル名を返すことを担保する。
    """
    # Auto + low fps → interpolate ON
    meta_auto_on = FakeMeta(fps=Fraction(24, 1))
    decs = decisions.recommend(meta_auto_on, FakeStats())
    d = _by_key(decs)["interpolate"]
    assert "model" in d.params
    assert d.params["model"].startswith("rife-v4")

    # Auto + high fps → interpolate OFF (params にも model 入っている)
    meta_auto_off = FakeMeta(fps=Fraction(60, 1))
    decs = decisions.recommend(meta_auto_off, FakeStats())
    d = _by_key(decs)["interpolate"]
    assert "model" in d.params
    assert d.params["model"].startswith("rife-v4")

    # explicit preset (src < target) → interpolate ON 明示同意
    decs = decisions.recommend(
        FakeMeta(fps=Fraction(24, 1)),
        FakeStats(),
        defaults={"target_fps": Fraction(48, 1), "target_fps_explicit": True},
    )
    d = _by_key(decs)["interpolate"]
    assert "model" in d.params
    assert d.params["model"].startswith("rife-v4")

    # explicit preset (src >= target) → interpolate OFF
    decs = decisions.recommend(
        FakeMeta(fps=Fraction(60, 1)),
        FakeStats(),
        defaults={"target_fps": Fraction(48, 1), "target_fps_explicit": True},
    )
    d = _by_key(decs)["interpolate"]
    assert "model" in d.params
    assert d.params["model"].startswith("rife-v4")


# ---------- format_output_spec_suffix (出力スペック命名) ----------


def _spec_decs(
    *, upscale=None, interpolate=None, face_restore=False
) -> list[decisions.StageDecision]:
    """encode 常時 ON。upscale=(scale, target_width) / interpolate=(num, den) を任意で ON にする。"""
    decs = [
        decisions.StageDecision(
            key="encode",
            label_ja="最終エンコード",
            recommended=True,
            confidence="high",
            reason="",
            enabled=True,
            params={},
        )
    ]
    if upscale is not None:
        scale, target_w = upscale
        decs.append(
            decisions.StageDecision(
                key="upscale",
                label_ja="アップスケール",
                recommended=True,
                confidence="high",
                reason="",
                enabled=True,
                params={"scale": scale, "target_width": target_w},
            )
        )
    if interpolate is not None:
        num, den = interpolate
        decs.append(
            decisions.StageDecision(
                key="interpolate",
                label_ja="フレーム補間",
                recommended=True,
                confidence="high",
                reason="",
                enabled=True,
                params={"target_fps_num": num, "target_fps_den": den},
            )
        )
    if face_restore:
        decs.append(
            decisions.StageDecision(
                key="face_restore",
                label_ja="顔復元",
                recommended=False,
                confidence="unknown",
                reason="",
                enabled=True,
                params={},
            )
        )
    return decs


@pytest.mark.parametrize(
    "fps,expected",
    [
        (Fraction(30000, 1001), "30"),  # 29.97 -> 30
        (Fraction(24000, 1001), "24"),  # 23.976 -> 24
        (Fraction(60000, 1001), "60"),  # 59.94 -> 60
        (Fraction(60, 1), "60"),
        (Fraction(24, 1), "24"),
    ],
)
def test_format_output_spec_suffix_rounds_fps(fps, expected):
    summary = decisions.compute_output_summary(
        FakeMeta(width=1920, height=1080, fps=fps), _spec_decs()
    )
    assert decisions.format_output_spec_suffix(summary) == f"_1920x1080-{expected}p"


def test_format_output_spec_suffix_passthrough_when_no_stages():
    # upscale / interpolate OFF → 入力解像度・入力 fps がそのまま載る
    summary = decisions.compute_output_summary(
        FakeMeta(width=1280, height=720, fps=Fraction(30, 1)), _spec_decs()
    )
    assert decisions.format_output_spec_suffix(summary) == "_1280x720-30p"


def test_format_output_spec_suffix_upscale_on_interpolate_off():
    # upscale ON (scale=4, target_width=3840), AR 保持で高さ 2160、fps は据え置き
    summary = decisions.compute_output_summary(
        FakeMeta(width=1920, height=1080, fps=Fraction(30, 1)), _spec_decs(upscale=(4, 3840))
    )
    assert decisions.format_output_spec_suffix(summary) == "_3840x2160-30p"


def test_format_output_spec_suffix_upscale_off_interpolate_on():
    # interpolate ON (60fps 化), upscale OFF → 解像度据え置き・fps だけ上がる
    summary = decisions.compute_output_summary(
        FakeMeta(width=1280, height=720, fps=Fraction(30, 1)), _spec_decs(interpolate=(60, 1))
    )
    assert decisions.format_output_spec_suffix(summary) == "_1280x720-60p"


def test_format_output_spec_suffix_upscale_and_interpolate_on():
    summary = decisions.compute_output_summary(
        FakeMeta(width=1920, height=1080, fps=Fraction(30, 1)),
        _spec_decs(upscale=(2, 3840), interpolate=(60, 1)),
    )
    assert decisions.format_output_spec_suffix(summary) == "_3840x2160-60p"


def test_format_output_spec_suffix_marks_face_restored():
    summary = decisions.compute_output_summary(
        FakeMeta(width=1920, height=1080, fps=Fraction(30, 1)), _spec_decs(face_restore=True)
    )
    assert decisions.format_output_spec_suffix(summary) == "_1920x1080-30p_face-restored"
