from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

from . import config, stages
from .decisions import StageDecision, compute_output_summary, decisions_by_key

__all__ = [
    "RuntimeEstimate",
    "estimate_runtime",
    "estimate_peak_for_decisions",
    "format_eta_hms",
]


# CLI / GUI で共通表示する正規メッセージ (文言ズレ防止)
ACCURACY_NOTE = "GPU 性能で ±50% 程度ずれます (baseline: GTX 1660 / RTX 3060 級)"

_SCALE_MIN = 0.1
_SCALE_MAX = 10.0
_FACE_RESTORE_MIN_ETA_FACTOR = 0.25


@dataclass(frozen=True)
class RuntimeEstimate:
    total_seconds: float  # ETA 秒（チャンク粒度）
    per_stage_seconds: dict[str, float]  # 内訳: {"extract": 1.23, "upscale_scale2": 45.6, ...}
    assumed_frames_in: int
    assumed_frames_out: int
    assumed_scale_factor: float  # 利用者環境を未調整時は 1.0
    confidence: Literal["high", "medium", "low"]  # 出荷時 default は "low" (実測キャリブ未済)
    accuracy_note: str  # CLI/GUI で同文言を表示する正規メッセージ
    warnings: list[str] = field(default_factory=list)  # 環境変数パース失敗・clamp 等の警告のみ


def _resolve_scale_factor(scale_factor: float | None, warnings: list[str]) -> float:
    """`scale_factor` が None なら環境変数 `VIDEO_ENHANCE_RUNTIME_SCALE` を読む。

    - float に変換できない / 空文字 -> 1.0 にフォールバック (warnings に記録)
    - 値域 [0.1, 10.0] の外 -> clamp し warnings に追記
    """
    if scale_factor is not None:
        val = float(scale_factor)
    else:
        raw = os.environ.get(config.ENV_RUNTIME_SCALE)
        if raw is None:
            return 1.0
        raw = raw.strip()
        try:
            val = float(raw)
        except ValueError:
            warnings.append(
                f"{config.ENV_RUNTIME_SCALE}={raw!r} を float に変換できません。1.0 を使用"
            )
            return 1.0
    if val < _SCALE_MIN or val > _SCALE_MAX:
        clamped = min(_SCALE_MAX, max(_SCALE_MIN, val))
        warnings.append(f"{val} は範囲外。{clamped} を使用")
        return clamped
    return val


def _face_restore_eta_factor(
    *,
    total_duration: float,
    start_sec: float | None,
    end_sec: float | None,
) -> tuple[float, bool]:
    """Return (ETA factor, range_applied).

    容量ガードとは分離し、ユーザー向け ETA の face_restore 部分にだけ使う。
    不正・空範囲・duration 不明では全区間見積もりへフォールバックする。
    """
    if start_sec is None and end_sec is None:
        return 1.0, False
    if not math.isfinite(total_duration) or total_duration <= 0:
        return 1.0, False
    start = 0.0 if start_sec is None else float(start_sec)
    end = float(total_duration) if end_sec is None else float(end_sec)
    if not math.isfinite(start) or not math.isfinite(end):
        return 1.0, False
    start = min(float(total_duration), max(0.0, start))
    end = min(float(total_duration), max(0.0, end))
    selected = end - start
    if selected <= 0:
        return 1.0, False
    ratio = min(1.0, max(0.0, selected / float(total_duration)))
    return max(_FACE_RESTORE_MIN_ETA_FACTOR, ratio), True


def estimate_runtime(
    meta,  # analyzer.VideoMetadata
    decisions: list[StageDecision],
    *,
    baseline: dict[str, float] | None = None,
    scale_factor: float | None = None,  # None なら env 変数を読む
) -> RuntimeEstimate:
    """`meta` + `decisions` から推定実行時間とステージ別内訳を返す純関数。

    numpy / cv2 / subprocess を一切呼ばない。仮想ステージ判定は
    `stages.PRE_FILTER_ORDER` / `stages.POST_FILTER_ORDER` を直接参照し、
    実装側のリネームに自動追従する (ハードコードしない)。
    """
    base = baseline if baseline is not None else config.STAGE_BASELINE_FRAME_COST_SEC
    baseline_px = config.STAGE_BASELINE_RESOLUTION_PX
    warnings: list[str] = []
    sf = _resolve_scale_factor(scale_factor, warnings)

    by = decisions_by_key(decisions)
    summary = compute_output_summary(meta, decisions)

    def _enabled(key: str) -> bool:
        d = by.get(key)
        return d is not None and d.enabled

    # ----- 仮想ステージ判定 (§1.2.0 マップ) -----
    stabilize_on = _enabled("stabilize")
    pre_filter_on = any(_enabled(k) for k in stages.PRE_FILTER_ORDER)
    upscale_on = _enabled("upscale")
    post_filter_on = any(_enabled(k) for k in stages.POST_FILTER_ORDER)
    clamp_on = summary.resolution_clamp_applied
    face_restore_on = _enabled("face_restore")
    interpolate_on = _enabled("interpolate")
    enc = by.get("encode")
    mux_on = enc is None or enc.enabled

    # ----- フレーム数 / ピクセル比 -----
    src_fps = float(meta.fps)
    f_in = max(1, math.ceil(float(meta.duration_sec) * src_fps))
    if interpolate_on and summary.interpolate_multiplier:
        f_out = f_in * int(summary.interpolate_multiplier)
    else:
        f_out = f_in

    src_w = int(meta.width)
    src_h = int(meta.height)
    src_px = src_w * src_h
    if summary.upscale_enabled and summary.intermediate_width and src_w > 0:
        iw = summary.intermediate_width
        ih = iw * src_h / src_w  # upscale は AR 維持
        intermediate_px = iw * ih
    else:
        intermediate_px = src_px
    out_px = summary.out_width * summary.out_height

    src_ratio = (src_px / baseline_px) if baseline_px > 0 else 0.0
    inter_ratio = (intermediate_px / baseline_px) if baseline_px > 0 else 0.0
    out_ratio = (out_px / baseline_px) if baseline_px > 0 else 0.0

    per_stage: dict[str, float] = {}

    # extract は常時実行 (src ピクセル基準)
    per_stage["extract"] = f_in * base["extract"] * src_ratio
    if stabilize_on:
        per_stage["stabilize_detect"] = f_in * base["stabilize_detect"] * src_ratio
    if pre_filter_on:
        per_stage["pre_filter"] = f_in * base["pre_filter"] * src_ratio
    if upscale_on:
        scale = int(by["upscale"].params.get("scale", config.DEFAULT_REALESRGAN_SCALE))
        key = f"upscale_scale{scale}"
        if key in base:
            per_stage[key] = f_in * base[key] * src_ratio
    if clamp_on:
        # clamp 入力ピクセル基準 (intermediate_px)。
        # PR-G2 NOTE: --clamp-flags bicubic/bilinear は lanczos より速いが、見積もりは当面
        # lanczos baseline を保守的に据え置く (filter 別 baseline の実測が安定するまで、
        # 推定が楽観側に倒れて temp/ETA を過小評価するのを避けるため)。
        per_stage["resolution_clamp"] = f_in * base["resolution_clamp"] * inter_ratio
    if face_restore_on:
        face = by["face_restore"]
        eta_factor, range_applied = _face_restore_eta_factor(
            total_duration=float(meta.duration_sec),
            start_sec=face.params.get("start_sec"),
            end_sec=face.params.get("end_sec"),
        )
        per_stage["face_restore"] = f_in * base["face_restore"] * out_ratio * eta_factor
        if range_applied:
            warnings.append(
                "face_restore ETA は範囲指定を反映済みです。"
                "ただし低信頼 baseline のため最低係数 0.25 を適用し、"
                "容量ガードは安全側に全フレーム見積もりを維持します。"
            )
    if interpolate_on:
        # interpolate 後の総フレーム × intermediate ピクセル基準
        per_stage["interpolate"] = f_out * base["interpolate"] * inter_ratio
    if post_filter_on:
        per_stage["post_filter"] = f_out * base["post_filter"] * out_ratio
    if mux_on:
        per_stage["mux"] = f_out * base["mux"] * out_ratio

    # scale_factor を per_stage / total 双方に同一係数で適用
    per_stage = {k: v * sf for k, v in per_stage.items()}
    total = sum(per_stage.values())

    return RuntimeEstimate(
        total_seconds=total,
        per_stage_seconds=per_stage,
        assumed_frames_in=f_in,
        assumed_frames_out=f_out,
        assumed_scale_factor=sf,
        confidence="low",
        accuracy_note=ACCURACY_NOTE,
        warnings=warnings,
    )


def estimate_peak_for_decisions(
    meta,  # analyzer.VideoMetadata
    decisions: list[StageDecision],
    chunk_seconds: int,
    *,
    png_compression_ratio: float
    | None = None,  # None なら _estimate_compression_ratio(stats) / default
    worker_count: int = 1,  # PR3 並列実行時のチャンク別 tmp 倍率を反映
    stats=None,  # analyzer.FrameStats (png_compression_ratio=None のときのみ参照)
) -> tuple[float, int | None, Fraction | None]:
    """`(peak_gib, intermediate_width, effective_fps)` を返す。

    `--max-temp-gib` ガードと UI 表示はこの helper を共有し、decisions 変更後の
    再計算でズレないことを保証する。`intermediate_width` / `effective_fps` 算出と
    `pipeline.estimate_peak_temp_bytes` 呼び出しをここに一本化する。
    """
    # pipeline は重い import (deps 経由) になるため遅延 import (循環参照回避)
    from . import pipeline

    by = decisions_by_key(decisions)
    up_dec = by.get("upscale")
    interp_dec = by.get("interpolate")

    intermediate_width: int | None = None
    if up_dec is not None and up_dec.enabled:
        scale = int(up_dec.params.get("scale", config.DEFAULT_REALESRGAN_SCALE))
        intermediate_width = int(meta.width) * scale

    effective_fps: Fraction | None = None
    if interp_dec is not None and interp_dec.enabled:
        num = int(interp_dec.params.get("target_fps_num", meta.fps.numerator * 2))
        den = int(interp_dec.params.get("target_fps_den", meta.fps.denominator))
        effective_fps = Fraction(num, den)

    if png_compression_ratio is None:
        if stats is not None:
            png_compression_ratio = pipeline.estimate_compression_ratio(stats)
        else:
            png_compression_ratio = pipeline.DEFAULT_PNG_COMPRESSION_RATIO

    peak_bytes = pipeline.estimate_peak_temp_bytes(
        meta.width,
        meta.height,
        meta.fps,
        chunk_seconds,
        intermediate_width=intermediate_width,
        effective_fps=effective_fps,
        png_compression_ratio=float(png_compression_ratio),
        worker_count=max(1, int(worker_count)),
    )
    peak_gib = peak_bytes / (1024**3)
    return peak_gib, intermediate_width, effective_fps


def format_eta_hms(seconds: float) -> str:
    """例: 9213 -> '2h33m', 73 -> '1m13s', 4 -> '4s'."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        total = 0
    else:
        total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        m, s = divmod(total, 60)
        return f"{m}m{s}s"
    h, rem = divmod(total, 3600)
    m = rem // 60
    return f"{h}h{m}m"
