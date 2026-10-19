from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal

from . import config

__all__ = [
    "StageDecision",
    "STAGE_KEYS",
    "recommend",
    "Confidence",
    "filter_enabled",
    "decisions_by_key",
    "normalize_target_fps",
    "OutputSummary",
    "compute_output_summary",
    "format_output_spec_suffix",
]

Confidence = Literal["high", "medium", "low", "unknown"]

STAGE_KEYS = config.STAGE_KEYS


@dataclass
class StageDecision:
    key: str
    label_ja: str
    recommended: bool
    confidence: Confidence
    reason: str
    enabled: bool
    params: dict[str, Any] = field(default_factory=dict)


def _new(
    key: str,
    *,
    recommended: bool,
    confidence: Confidence,
    reason: str,
    params: dict[str, Any] | None = None,
) -> StageDecision:
    return StageDecision(
        key=key,
        label_ja=config.STAGE_LABELS_JA[key],
        recommended=recommended,
        confidence=confidence,
        reason=reason,
        enabled=recommended,
        params=dict(params or {}),
    )


def normalize_target_fps(source_fps: Fraction, target_fps: Fraction) -> tuple[Fraction, str | None]:
    """Round target_fps to the nearest integer multiple of source_fps.

    Returns (normalized, warning_or_None).
    `target_fps` is the *requested* target (Fraction). Output is guaranteed to satisfy
    `(normalized / source_fps).denominator == 1`.
    """
    if source_fps <= 0:
        raise ValueError(f"source_fps must be > 0, got {source_fps}")
    src = Fraction(source_fps)
    tgt = Fraction(target_fps)
    if tgt <= 0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")
    raw = tgt / src
    # 最も近い正の整数倍 (>=1)
    n = max(1, round(float(raw)))
    normalized = src * n
    if normalized != tgt:
        warning = (
            f"target_fps {float(tgt):.3f} は source_fps {float(src):.3f} の整数倍ではないため、"
            f"x{n} 倍 = {float(normalized):.3f} fps に丸めました。"
        )
        return normalized, warning
    return normalized, None


def recommend(
    meta,
    stats,
    defaults: dict[str, Any] | None = None,
) -> list[StageDecision]:
    """Build StageDecision list from VideoMetadata + FrameStats.

    `meta` / `stats` use duck typing (analyzer.VideoMetadata / FrameStats).
    `defaults` allows tests to override thresholds and config values.
    """
    d = defaults or {}

    t_shake = d.get("T_shake", config.T_SHAKE)
    t_flicker = d.get("T_flicker", config.T_FLICKER)
    t_noise = d.get("T_noise", config.T_NOISE)
    t_banding = d.get("T_banding", config.T_BANDING)
    t_blur = d.get("T_blur", config.T_BLUR)
    t_luma_dark = d.get("T_luma_dark", config.T_LUMA_DARK)
    t_luma_bright = d.get("T_luma_bright", config.T_LUMA_BRIGHT)
    t_luma_clip = d.get("T_luma_clipping", config.T_LUMA_CLIPPING)
    t_sat_low = d.get("T_saturation_low", config.T_SATURATION_LOW)
    t_fps_low = d.get("T_fps_low", config.T_FPS_LOW)
    target_width_raw = d.get("target_width", None)
    if target_width_raw is None:
        target_width = config.DEFAULT_TARGET_WIDTH
    elif target_width_raw == config.RESOLUTION_KEEP_SOURCE:
        # 「現状維持」: 入力解像度をそのまま target にする → upscale は常に非推奨。
        target_width = meta.width
    else:
        target_width = target_width_raw
    # NOTE: defaults["target_width_explicit"] は CLI/GUI が設定するが、現状 upscale 判定
    # ロジックでは未参照 (target_fps_explicit と異なり未配線)。読み捨てを避けるため
    # ここでは取得しない。配線が必要になったら target_fps_explicit と同様に追加する。
    target_fps_default = d.get("target_fps", None)  # None -> source_fps * 2
    target_fps_explicit = bool(d.get("target_fps_explicit", False))
    # PR-G2: scale 選択ポリシー (既定 "cover" = 従来の over-scale 方向)。
    scale_policy = d.get("upscale_scale_policy", config.DEFAULT_UPSCALE_SCALE_POLICY)

    decisions: list[StageDecision] = []

    # 1. deinterlace
    fo = (getattr(meta, "field_order", "") or "").lower()
    if fo and fo != "progressive" and fo != "unknown":
        decisions.append(
            _new(
                "deinterlace",
                recommended=True,
                confidence="high",
                reason=f"field_order={fo} (インターレース素材)",
            )
        )
    elif fo == "progressive":
        decisions.append(
            _new(
                "deinterlace",
                recommended=False,
                confidence="high",
                reason="field_order=progressive",
            )
        )
    else:
        decisions.append(
            _new(
                "deinterlace",
                recommended=False,
                confidence="unknown",
                reason="field_order 不明（必要なら手動で ON）",
            )
        )

    # 2. stabilize
    # TODO.md V1-1 解消: stabilize_backend=opencv (PR2) で vidstabtransform を回避できる
    # ようになったため、shake_detected を素直に recommended に反映する。
    # ffmpeg N-120725+ + auto backend では opencv 経路、それ以外では vidstab 経路に分岐。
    shake_detected = stats.interframe_diff_std > t_shake
    if shake_detected:
        # NOTE: reason は CLI 経由で cp932 コンソールに print されるため、
        # cp932 非収録文字 (U+2014 em-dash 等) を使わないこと (P1-1)。
        stabilize_reason = (
            f"interframe_diff_std={stats.interframe_diff_std:.2f} > {t_shake} (手ぶれ検出) / "
            f"OpenCV バックエンドで補正可 (--stabilize-backend opencv または auto)"
        )
        decisions.append(
            _new(
                "stabilize",
                recommended=True,
                confidence="medium",
                reason=stabilize_reason,
            )
        )
    else:
        stabilize_reason = f"interframe_diff_std={stats.interframe_diff_std:.2f} <= {t_shake}"
        decisions.append(
            _new(
                "stabilize",
                recommended=False,
                confidence="low",
                reason=stabilize_reason,
            )
        )

    # 3. deflicker
    if stats.luma_std_over_time > t_flicker:
        decisions.append(
            _new(
                "deflicker",
                recommended=True,
                confidence="medium",
                reason=f"luma_std_over_time={stats.luma_std_over_time:.2f} > {t_flicker}",
            )
        )
    else:
        decisions.append(
            _new(
                "deflicker",
                recommended=False,
                confidence="low",
                reason=f"luma_std_over_time={stats.luma_std_over_time:.2f} <= {t_flicker}",
            )
        )

    # 4. denoise (highfreq_energy か bitrate < 2 Mbps)
    bitrate_low = meta.bitrate_bps is not None and meta.bitrate_bps < 2_000_000
    if stats.highfreq_energy > t_noise or bitrate_low:
        reason_parts = []
        if stats.highfreq_energy > t_noise:
            reason_parts.append(f"highfreq_energy={stats.highfreq_energy:.2f} > {t_noise}")
        if bitrate_low:
            reason_parts.append(f"bitrate={meta.bitrate_bps} < 2Mbps")
        decisions.append(
            _new(
                "denoise",
                recommended=True,
                confidence="medium",
                reason="; ".join(reason_parts),
                params={"strength": config.DEFAULT_DENOISE_STRENGTH},
            )
        )
    else:
        decisions.append(
            _new(
                "denoise",
                recommended=False,
                confidence="low",
                reason=f"highfreq_energy={stats.highfreq_energy:.2f} <= {t_noise}",
                params={"strength": config.DEFAULT_DENOISE_STRENGTH},
            )
        )

    # 5. debanding
    if stats.histogram_flat_ratio > t_banding:
        decisions.append(
            _new(
                "debanding",
                recommended=True,
                confidence="medium",
                reason=f"histogram_flat_ratio={stats.histogram_flat_ratio:.3f} > {t_banding}",
                params={"range": config.DEFAULT_DEBAND_RANGE},
            )
        )
    else:
        decisions.append(
            _new(
                "debanding",
                recommended=False,
                confidence="low",
                reason=f"histogram_flat_ratio={stats.histogram_flat_ratio:.3f} <= {t_banding}",
                params={"range": config.DEFAULT_DEBAND_RANGE},
            )
        )

    # 6. color_correction
    color_reasons: list[str] = []
    if stats.luma_mean < t_luma_dark:
        color_reasons.append(f"luma_mean={stats.luma_mean:.1f} < {t_luma_dark} (暗め)")
    if stats.luma_mean > t_luma_bright:
        color_reasons.append(f"luma_mean={stats.luma_mean:.1f} > {t_luma_bright} (明るすぎ)")
    if stats.luma_clipping_ratio > t_luma_clip:
        color_reasons.append(
            f"luma_clipping_ratio={stats.luma_clipping_ratio:.3f} > {t_luma_clip} (潰れ)"
        )
    if stats.saturation_mean < t_sat_low:
        color_reasons.append(f"saturation_mean={stats.saturation_mean:.1f} < {t_sat_low} (低彩度)")
    if color_reasons:
        decisions.append(
            _new(
                "color_correction",
                recommended=True,
                confidence="medium",
                reason="; ".join(color_reasons),
                params={
                    "brightness": config.DEFAULT_COLOR_BRIGHTNESS,
                    "contrast": config.DEFAULT_COLOR_CONTRAST,
                    "saturation": config.DEFAULT_COLOR_SATURATION,
                },
            )
        )
    else:
        decisions.append(
            _new(
                "color_correction",
                recommended=False,
                confidence="low",
                reason="luma/clipping/saturation すべて正常範囲",
                params={
                    "brightness": config.DEFAULT_COLOR_BRIGHTNESS,
                    "contrast": config.DEFAULT_COLOR_CONTRAST,
                    "saturation": config.DEFAULT_COLOR_SATURATION,
                },
            )
        )

    # 7. sharpen (laplacian_var が低い = ボケ)
    if stats.laplacian_var_mean < t_blur:
        decisions.append(
            _new(
                "sharpen",
                recommended=True,
                confidence="medium",
                reason=f"laplacian_var={stats.laplacian_var_mean:.1f} < {t_blur} (ボケ気味)",
                params={"amount": config.DEFAULT_SHARPEN_AMOUNT},
            )
        )
    else:
        decisions.append(
            _new(
                "sharpen",
                recommended=False,
                confidence="low",
                reason=f"laplacian_var={stats.laplacian_var_mean:.1f} >= {t_blur}",
                params={"amount": config.DEFAULT_SHARPEN_AMOUNT},
            )
        )

    # 8. upscale
    # Real-ESRGAN-ncnn-vulkan は -s 2/3/4 のみ。
    #   - "cover" (既定): needed = ceil(target_w/src_w) を [2,4] にクランプ。needed>=5 は scale=4 で
    #     頭打ちにし、足りない分は後段の resolution_clamp (lanczos) で上方 scale する (over-scale 方向)。
    #   - "nearest" (PR-G2 opt-in): target_width に最も近い中間幅になる scale を選ぶ。target を下回れば
    #     後段 clamp が上方 lanczos になる (精細感とのトレードオフ)。
    if meta.width < target_width:
        if scale_policy == "nearest":
            # DEFAULT_REALESRGAN_MODEL がサポートする scale から target に最も近いものを選ぶ。
            candidates = config.REALESRGAN_MODEL_KIND.get(
                config.DEFAULT_REALESRGAN_MODEL, ("scaled", (2, 3, 4))
            )[1]
            scale = min(candidates, key=lambda s: abs(meta.width * s - target_width))
            intermediate_w = meta.width * scale
            if intermediate_w < target_width:
                direction = "上方 lanczos"
            elif intermediate_w > target_width:
                direction = "下方 lanczos"
            else:
                direction = "clamp 不要"
            reason = (
                f"width={meta.width} < target_width={target_width} "
                f"-> scale={scale} (nearest policy, 中間 {intermediate_w}px, {direction})"
            )
        else:
            needed = math.ceil(target_width / meta.width)
            if needed >= 5:
                scale = 4
                intermediate_w = meta.width * scale
                reason = (
                    f"width={meta.width} < target_width={target_width} (needed={needed}>=5) "
                    f"-> scale=4 でクランプ; 中間 {intermediate_w}px から lanczos で上方スケール"
                )
            else:
                scale = needed
                intermediate_w = meta.width * scale
                reason = (
                    f"width={meta.width} < target_width={target_width} "
                    f"-> scale={scale} (中間 {intermediate_w}px)"
                )
        decisions.append(
            _new(
                "upscale",
                recommended=True,
                confidence="high",
                reason=reason,
                params={
                    "scale": scale,
                    "model": config.DEFAULT_REALESRGAN_MODEL,
                    "target_width": target_width,
                },
            )
        )
    else:
        decisions.append(
            _new(
                "upscale",
                recommended=False,
                confidence="high",
                reason=f"width={meta.width} >= target_width={target_width} (upscale 不要)",
                params={
                    "scale": config.DEFAULT_REALESRGAN_SCALE,
                    "model": config.DEFAULT_REALESRGAN_MODEL,
                    "target_width": target_width,
                },
            )
        )

    # 9. face_restore (AI 顔復元。生成を伴うため標準では推奨しない)
    face_restore_candidates = tuple(getattr(stats, "face_restore_candidates", ()))
    decisions.append(
        _new(
            "face_restore",
            recommended=False,
            confidence="unknown",
            reason="AI顔復元は顔の細部を生成し得るため、必要な範囲だけ手動で ON",
            params={
                "backend": config.DEFAULT_FACE_RESTORE_BACKEND,
                "fidelity": config.DEFAULT_FACE_RESTORE_FIDELITY,
                "start_sec": None,
                "end_sec": None,
                "idle_timeout_sec": config.DEFAULT_FACE_RESTORE_IDLE_TIMEOUT_SEC,
                "detection_model": config.DEFAULT_FACE_RESTORE_DETECTION_MODEL,
                "blend": config.DEFAULT_FACE_RESTORE_BLEND,
                "mask": config.DEFAULT_FACE_RESTORE_MASK,
                "feather": config.DEFAULT_FACE_RESTORE_FEATHER,
                "mask_dilate": config.DEFAULT_FACE_RESTORE_MASK_DILATE,
                "mask_threshold": config.DEFAULT_FACE_RESTORE_MASK_THRESHOLD,
                "match_grain": config.DEFAULT_FACE_RESTORE_MATCH_GRAIN,
                "soften": config.DEFAULT_FACE_RESTORE_SOFTEN,
                "global_smooth": config.DEFAULT_FACE_RESTORE_GLOBAL_SMOOTH,
                "candidate_ranges": face_restore_candidates,
            },
        )
    )

    # 10. interpolate
    # プリセット明示時 (target_fps_explicit=True かつ target_fps_default is not None):
    #   - source_fps >= target_fps  -> 補間 OFF (no-op RIFE は起動しない)
    #   - source_fps  < target_fps  -> 整数倍丸めの上で補間 ON (enabled=True; プリセット指定=明示同意)
    # それ以外 (Auto) は現行ロジック維持: source<30 -> ×2 推奨, それ以外は OFF。
    src_fps = Fraction(meta.fps)
    if target_fps_explicit and target_fps_default is not None:
        req = Fraction(target_fps_default)
        if req == config.FPS_KEEP_SOURCE:
            # 「現状維持」: 入力 fps をそのまま target にする → interpolate は常に非推奨。
            req = src_fps
        if src_fps >= req:
            # 等値 (==) も OFF 側 (§2.4 / 利用者承認 #3)
            fallback_fps = src_fps * 2
            decisions.append(
                _new(
                    "interpolate",
                    recommended=False,
                    confidence="high",
                    reason=(
                        f"source_fps={float(src_fps):.3f} >= target_fps={float(req):.3f} (preset) "
                        f"のため補間しません"
                    ),
                    params={
                        "target_fps_num": fallback_fps.numerator,
                        "target_fps_den": fallback_fps.denominator,
                        "model": config.DEFAULT_RIFE_MODEL,
                    },
                )
            )
        else:
            normalized, target_warning = normalize_target_fps(src_fps, req)
            reason = (
                f"target_fps={float(req):.3f} (preset) -> 補間後 {float(normalized):.3f} fps "
                f"(source_fps={float(src_fps):.3f})"
            )
            if target_warning:
                reason += f" (warn: {target_warning})"
            dec = _new(
                "interpolate",
                recommended=True,
                confidence="high",
                reason=reason,
                params={
                    "target_fps_num": normalized.numerator,
                    "target_fps_den": normalized.denominator,
                    "model": config.DEFAULT_RIFE_MODEL,
                },
            )
            # プリセット明示指定は「明示同意」として enabled=True で運用 (§3.3)
            dec.enabled = True
            decisions.append(dec)
    else:
        # Auto: 現行ロジック維持
        if target_fps_default is None:
            target_fps = src_fps * 2
            target_warning = None
        else:
            target_fps, target_warning = normalize_target_fps(src_fps, Fraction(target_fps_default))
        if float(src_fps) < t_fps_low:
            reason = (
                f"fps={float(src_fps):.3f} < {t_fps_low} -> 補間後 {float(target_fps):.3f} "
                f"(target_fps=auto)"
            )
            if target_warning:
                reason += f" (warn: {target_warning})"
            decisions.append(
                _new(
                    "interpolate",
                    recommended=True,
                    confidence="medium",
                    reason=reason,
                    params={
                        "target_fps_num": target_fps.numerator,
                        "target_fps_den": target_fps.denominator,
                        "model": config.DEFAULT_RIFE_MODEL,
                    },
                )
            )
        else:
            decisions.append(
                _new(
                    "interpolate",
                    recommended=False,
                    confidence="low",
                    reason=f"fps={float(src_fps):.3f} >= {t_fps_low} (target_fps=auto)",
                    params={
                        "target_fps_num": target_fps.numerator,
                        "target_fps_den": target_fps.denominator,
                        "model": config.DEFAULT_RIFE_MODEL,
                    },
                )
            )

    # 11. grain (denoise ON 時のみ medium 推奨)
    denoise_dec = next((d for d in decisions if d.key == "denoise"), None)
    if denoise_dec is not None and denoise_dec.recommended:
        decisions.append(
            _new(
                "grain",
                recommended=True,
                confidence="medium",
                reason="denoise を有効化したため、喪失質感の補填を推奨",
                params={"amount": config.DEFAULT_GRAIN_AMOUNT},
            )
        )
    else:
        decisions.append(
            _new(
                "grain",
                recommended=False,
                confidence="low",
                reason="denoise OFF のため不要",
                params={"amount": config.DEFAULT_GRAIN_AMOUNT},
            )
        )

    # 12. encode (常時 ON、パラメータのみ確認)
    decisions.append(
        _new(
            "encode",
            recommended=True,
            confidence="high",
            reason="最終 H.264 エンコード (CRF/preset 確認)",
            params={
                "crf": config.DEFAULT_ENCODE_CRF,
                "preset": config.DEFAULT_ENCODE_PRESET,
            },
        )
    )

    return decisions


def decisions_by_key(decisions: list[StageDecision]) -> dict[str, StageDecision]:
    return {d.key: d for d in decisions}


def filter_enabled(decisions: list[StageDecision]) -> list[StageDecision]:
    return [d for d in decisions if d.enabled]


@dataclass(frozen=True)
class OutputSummary:
    """analyze + decisions から推定される最終的な出力スペックの要約。

    GUI の事前分析ダイアログで「自動」プリセットが実行後にどう確定するかを
    可視化するために使う。pipeline 側のロジック (upscale → resolution_clamp →
    interpolate) と一致するよう計算する。
    """

    src_width: int
    src_height: int
    src_fps: Fraction
    out_width: int
    out_height: int
    out_fps: Fraction
    upscale_enabled: bool
    upscale_scale: int | None  # 採用された Real-ESRGAN scale (1..4) / OFF なら None
    intermediate_width: int | None  # upscale 中間幅 (clamp 前)。OFF なら None
    resolution_clamp_applied: bool  # intermediate_width != out_width なら True
    resolution_clamp_upscale: bool  # clamp が上方 lanczos (warn 対象) なら True
    interpolate_enabled: bool
    interpolate_multiplier: int | None  # out_fps / src_fps が整数なら N、それ以外は None
    active_stage_keys: tuple[str, ...]
    active_stage_labels: tuple[str, ...]


def _align_even(value: float) -> int:
    """ffmpeg の `scale=W:-2` (2 の倍数アライン) と同じ丸め規則。"""
    n = int(round(value))
    if n % 2 != 0:
        n += 1 if n > 0 else -1
    return max(n, 2)


def compute_output_summary(meta, decisions: list[StageDecision]) -> OutputSummary:
    """`meta` (VideoMetadata 互換) と `decisions` から最終出力スペックを推定する。

    - upscale ON: out_width = params["target_width"], out_height は AR 保持 + 2 の倍数。
      intermediate_width = src_width * scale。両者が異なれば resolution_clamp 適用。
    - upscale OFF: 解像度は維持。
    - interpolate ON: out_fps = Fraction(target_fps_num, target_fps_den)。
    - interpolate OFF: out_fps = src_fps。
    - active_stage_* は `enabled=True` のステージのみ、`decisions` の順序を保持。
    """
    by = decisions_by_key(decisions)
    upscale = by.get("upscale")
    interpolate = by.get("interpolate")

    src_w = int(meta.width)
    src_h = int(meta.height)
    src_fps = Fraction(meta.fps)

    if upscale is not None and upscale.enabled:
        scale = int(upscale.params.get("scale", config.DEFAULT_REALESRGAN_SCALE))
        target_w = int(upscale.params.get("target_width", config.DEFAULT_TARGET_WIDTH))
        intermediate_w = src_w * scale
        out_w = target_w
        clamp_applied = intermediate_w != target_w
        clamp_upscale = clamp_applied and target_w > intermediate_w
        upscale_enabled = True
        upscale_scale: int | None = scale
        intermediate: int | None = intermediate_w
    else:
        out_w = src_w
        clamp_applied = False
        clamp_upscale = False
        upscale_enabled = False
        upscale_scale = None
        intermediate = None

    if src_w > 0:
        out_h = _align_even(src_h * out_w / src_w)
    else:
        out_h = src_h

    if interpolate is not None and interpolate.enabled:
        num = int(interpolate.params.get("target_fps_num", src_fps.numerator * 2))
        den = int(interpolate.params.get("target_fps_den", src_fps.denominator))
        out_fps = Fraction(num, den)
        interp_enabled = True
        if src_fps > 0:
            ratio = out_fps / src_fps
            interp_mult: int | None = ratio.numerator if ratio.denominator == 1 else None
        else:
            interp_mult = None
    else:
        out_fps = src_fps
        interp_enabled = False
        interp_mult = None

    active = [d for d in decisions if d.enabled]
    return OutputSummary(
        src_width=src_w,
        src_height=src_h,
        src_fps=src_fps,
        out_width=out_w,
        out_height=out_h,
        out_fps=out_fps,
        upscale_enabled=upscale_enabled,
        upscale_scale=upscale_scale,
        intermediate_width=intermediate,
        resolution_clamp_applied=clamp_applied,
        resolution_clamp_upscale=clamp_upscale,
        interpolate_enabled=interp_enabled,
        interpolate_multiplier=interp_mult,
        active_stage_keys=tuple(d.key for d in active),
        active_stage_labels=tuple(d.label_ja for d in active),
    )


def format_output_spec_suffix(summary: OutputSummary) -> str:
    """出力ファイル名末尾に付与する「実行スペック」suffix を生成する。

    形式は `_{幅}x{高}-{fps}p`。fps は整数に丸める (例: 60p / 30p / 24p)。
    `compute_output_summary` が upscale / interpolate OFF でも `out_*` を入力
    スペックへ解決するため、どのステージ構成でも常に有効なファイル名片を返す
    (例: アップスケール無しなら入力解像度が、補間無しなら入力 fps がそのまま載る)。

    例: 3840x2160 / 60fps -> "_3840x2160-60p"

    旧既定の固定 suffix `_enhanced` を置き換えるためのもの。`config.derive_output_path`
    の `suffix` 引数や GUI の自動命名に渡して使う。
    """
    # 解像度・fps は物理的に正の値だが、異常入力でも空文字や 0p を出さないよう下限を 1 に丸める。
    width = max(1, int(summary.out_width))
    height = max(1, int(summary.out_height))
    fps_int = max(1, int(round(float(summary.out_fps))))
    suffix = f"_{width}x{height}-{fps_int}p"
    if "face_restore" in summary.active_stage_keys:
        suffix += "_face-restored"
    return suffix
