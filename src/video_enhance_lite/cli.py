from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Callable

from . import (
    analyzer,
    config,
    decisions as decisions_mod,
    deps,
    pipeline,
    runtime_estimator,
    stages,
    temp_workspace,
)
from .decisions import StageDecision, decisions_by_key
from .deps import MissingDependencyError
from .runtime_estimator import estimate_peak_for_decisions, format_eta_hms


__all__ = ["main", "_build_parser"]


def _positive_int(s: str) -> int:
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be an integer, got {s!r}")
    if v < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {v}")
    return v


def _comma_keys(s: str) -> list[str]:
    return [t.strip() for t in s.split(",") if t.strip()]


def _nonneg_int(s: str) -> int:
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be an integer, got {s!r}")
    if v < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {v}")
    return v


def _positive_float(s: str) -> float:
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a number, got {s!r}")
    if not math.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {s!r}")
    return v


def _nonnegative_float(s: str) -> float:
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a number, got {s!r}")
    if not math.isfinite(v) or v < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {s!r}")
    return v


def _fidelity_value(s: str) -> float:
    return _unit_float(s, label="fidelity")


def _unit_float(s: str, *, label: str) -> float:
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a number, got {s!r}")
    if not math.isfinite(v) or not (0.0 <= v <= 1.0):
        raise argparse.ArgumentTypeError(f"{label} must be in 0..1, got {s!r}")
    return v


def _face_restore_unit_float(s: str) -> float:
    return _unit_float(s, label="face_restore value")


def _byte_value(s: str) -> int:
    v = _nonneg_int(s)
    if v > 255:
        raise argparse.ArgumentTypeError(f"must be in 0..255, got {v}")
    return v


def _face_restore_mask_radius(s: str) -> int:
    v = _nonneg_int(s)
    if v > 256:
        raise argparse.ArgumentTypeError(f"must be in 0..256, got {v}")
    return v


def _crf_value(s: str) -> int:
    """libx264 CRF (0..51)。範囲外は ffmpeg 実行時ではなく argparse で早期に弾く。"""
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be an integer, got {s!r}")
    if not (0 <= v <= 51):
        raise argparse.ArgumentTypeError(f"crf must be in 0..51, got {v}")
    return v


def _target_width_value(s: str) -> int:
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be an integer, got {s!r}")
    if v < 16:
        raise argparse.ArgumentTypeError(f"target width must be >= 16, got {v}")
    return v


# libx264 の有効 preset 名 (man x264)。GUI の Combobox より広い完全列挙。
_X264_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
)


def _gpu_tilesize(s: str) -> int:
    """realesrgan -t tile-size。0=auto、それ以外は >=32 を要求 (公式 README 準拠)。"""
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be an integer, got {s!r}")
    if v != 0 and v < 32:
        raise argparse.ArgumentTypeError(f"tile-size must be 0 (auto) or >= 32, got {v}")
    return v


def _gpu_threads(s: str) -> str:
    """ncnn-vulkan -j load:proc:save。単一 GPU 形式 (3 個の正整数) のみ許可。

    マルチ GPU 形式 (`1:2,2,2:2` のようにカンマを含む) は本環境 (単一 GPU) で未検証のため
    当面弾く。将来 multi-GPU 対応時に解禁する。
    """
    parts = s.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"must be 'load:proc:save' (3 integers), got {s!r}")
    for p in parts:
        if "," in p:
            raise argparse.ArgumentTypeError(f"multi-GPU form (comma) is not supported yet: {s!r}")
        try:
            n = int(p)
        except ValueError:
            raise argparse.ArgumentTypeError(f"each thread count must be an integer: {s!r}")
        if n < 1:
            raise argparse.ArgumentTypeError(f"each thread count must be >= 1: {s!r}")
    return s


def _rife_gpu_id(s: str) -> int:
    """rife -g gpu-id。-1=cpu / >=0=gpu index。multi-GPU (カンマ) は当面非対応。"""
    if "," in s:
        raise argparse.ArgumentTypeError(f"multi-GPU form (comma) is not supported yet: {s!r}")
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be an integer, got {s!r}")
    if v < -1:
        raise argparse.ArgumentTypeError(f"must be -1 (cpu) or >= 0, got {v}")
    return v


def _format_resolution_preset(name: str, width: int | None) -> str:
    """WARN メッセージ用の解像度プリセット表示。

    数値幅を持つプリセット (fullhd/wqhd/4k) は `name(width)`、
    auto (None) / keep (RESOLUTION_KEEP_SOURCE センチネル) は数値を出さず name のみ。
    """
    if width is None or width == config.RESOLUTION_KEEP_SOURCE:
        return name
    return f"{name}({width})"


def _format_fps_preset(name: str, fps: float | None) -> str:
    """WARN メッセージ用の FPS プリセット表示。

    数値 fps を持つプリセット (48/60/120) は `name(fps)`、
    auto (None) / keep (FPS_KEEP_SOURCE センチネル) は数値を出さず name のみ。
    """
    if fps is None or fps == config.FPS_KEEP_SOURCE:
        return name
    return f"{name}({fps})"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video-enhance-lite",
        description=(
            f"Pre-analyze a video and run the {stages.PUBLIC_STAGE_COUNT_LABEL} "
            "enhancement pipeline "
            "(deinterlace / stabilize / deflicker / denoise / debanding / "
            "color / sharpen / upscale / interpolate / grain / encode). "
            "Decisions are presented to the user before execution."
        ),
    )
    p.add_argument("input", type=Path, nargs="?", help="input video file")
    p.add_argument(
        "--self-check",
        action="store_true",
        help="show capability and user-data diagnostics without requiring an input video",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output file path or existing output directory "
        "(default: <stem>_<WxH>-<fps>p<ext> next to input, "
        "auto-named from the resulting spec, e.g. clip_3840x2160-60p.mp4)",
    )
    p.add_argument(
        "--chunk-seconds",
        type=_positive_int,
        default=config.DEFAULT_CHUNK_SECONDS,
        help=f"chunk duration in seconds (default: {config.DEFAULT_CHUNK_SECONDS})",
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--interactive",
        action="store_true",
        help="prompt for each stage (default if no other mode given)",
    )
    mode.add_argument(
        "--auto",
        action="store_true",
        help="accept only encode + high-confidence low-risk stages; "
        "any other recommended quality-altering stage causes exit 1",
    )
    mode.add_argument(
        "--auto-all",
        action="store_true",
        help="accept all recommended stages (explicit consent for quality changes)",
    )

    p.add_argument(
        "--analyze-only",
        action="store_true",
        help="run analyze and print the decision table, then exit",
    )

    p.add_argument(
        "--enable",
        type=_comma_keys,
        default=[],
        help="comma-separated stage keys to force-enable (overrides recommendation)",
    )
    p.add_argument(
        "--disable",
        type=_comma_keys,
        default=[],
        help="comma-separated stage keys to force-disable",
    )

    p.add_argument(
        "--target-fps",
        type=_positive_float,
        default=None,
        help="target fps for interpolate; rounded to integer multiple of source fps. "
        "明示指定時はプリセットを上書きする",
    )
    p.add_argument(
        "--target-fps-preset",
        choices=tuple(config.FPS_PRESETS.keys()),
        default="auto",
        help="fps preset: auto / keep / 48 / 60 / 120 (default: auto). "
        "keep = 入力 fps のまま (フレーム補間しない); "
        "auto は source < 30fps なら ×2 補間を推奨する点に注意",
    )
    p.add_argument(
        "--target-width",
        type=_target_width_value,
        default=None,
        help="target width used by upscale recommendation. "
        "明示指定時はプリセットを上書きする (default: None -> プリセット解釈に従う)",
    )
    p.add_argument(
        "--target-resolution",
        choices=tuple(config.RESOLUTION_PRESETS.keys()),
        default="auto",
        help="resolution preset: auto / keep / fullhd / wqhd / 4k (default: auto). "
        "keep = 入力解像度のまま (upscale しない); "
        "auto は入力が 1920 未満なら 1920 へ拡大する点に注意",
    )
    p.add_argument(
        "--realesrgan-model",
        default=config.DEFAULT_REALESRGAN_MODEL,
        choices=tuple(config.REALESRGAN_MODEL_KIND.keys()),
        help=(
            "realesrgan model. 同梱: realesr-animevideov3 (default), "
            "realesrgan-x4plus, realesrgan-x4plus-anime. "
            "OPTIONAL: realesr-general-x4v3 (自前で param/bin を models/ に配置した場合のみ)"
        ),
    )
    p.add_argument(
        "--rife-model",
        default=config.DEFAULT_RIFE_MODEL,
        choices=config.ALLOWED_RIFE_MODELS,
        help=(
            f"rife model (default: {config.DEFAULT_RIFE_MODEL}). "
            f"rife-v4 系のみ任意整数倍補間。v2/v3 系は x2 のみ対応。"
        ),
    )
    p.add_argument(
        "--scale",
        type=int,
        choices=(2, 3, 4),
        default=None,
        help="realesrgan scale factor (default: None -> プリセットから自動算出)。"
        "realesrgan-ncnn-vulkan は -s 2/3/4 のみ対応",
    )
    p.add_argument(
        "--upscale-scale-policy",
        choices=config.UPSCALE_SCALE_POLICIES,
        default=config.DEFAULT_UPSCALE_SCALE_POLICY,
        help=(
            "Real-ESRGAN scale の選び方 (opt-in)。"
            "cover (既定): 必要倍率以上の最小 scale (over-scale 方向・従来挙動)。"
            "nearest: target_width に最も近い中間幅になる scale を選ぶ "
            "(upscale が軽くなり得るが、target を下回ると上方 lanczos で精細感が落ちる)。"
        ),
    )
    p.add_argument(
        "--clamp-flags",
        choices=config.CLAMP_FLAGS,
        default=config.DEFAULT_CLAMP_FLAGS,
        help=(
            "resolution_clamp の rescale アルゴリズム (opt-in)。"
            "lanczos (既定・従来挙動): 精細感重視。bicubic/bilinear: 高速だが精細感が落ちる。"
            "NOTE: runtime 見積もりは当面 lanczos baseline を保守的に据え置く。"
        ),
    )
    p.add_argument(
        "--no-realesrgan",
        action="store_true",
        help="force-disable upscale stage regardless of recommendation",
    )
    p.add_argument(
        "--no-rife",
        action="store_true",
        help="force-disable interpolate stage regardless of recommendation",
    )
    p.add_argument(
        "--face-restore-backend",
        choices=config.FACE_RESTORE_BACKENDS,
        default=config.DEFAULT_FACE_RESTORE_BACKEND,
        help="AI 顔復元 backend (default: codeformer-local)。クラウド API は未対応",
    )
    p.add_argument(
        "--face-restore-start",
        type=_nonnegative_float,
        default=None,
        help="AI 顔復元の開始秒。--enable face_restore と併用する",
    )
    p.add_argument(
        "--face-restore-end",
        type=_nonnegative_float,
        default=None,
        help="AI 顔復元の終了秒 (半開区間 [start,end))",
    )
    p.add_argument(
        "--face-restore-fidelity",
        type=_fidelity_value,
        default=config.DEFAULT_FACE_RESTORE_FIDELITY,
        help=(
            f"CodeFormer fidelity weight 0..1 (default: {config.DEFAULT_FACE_RESTORE_FIDELITY})。"
            "0 は品質寄り、1 は入力忠実度寄り"
        ),
    )
    p.add_argument(
        "--face-restore-idle-timeout",
        type=_positive_int,
        default=config.DEFAULT_FACE_RESTORE_IDLE_TIMEOUT_SEC,
        help=(
            "CodeFormer subprocess 専用 idle timeout 秒 "
            f"(default: {config.DEFAULT_FACE_RESTORE_IDLE_TIMEOUT_SEC})"
        ),
    )
    p.add_argument(
        "--face-restore-blend",
        type=_face_restore_unit_float,
        default=config.DEFAULT_FACE_RESTORE_BLEND,
        help=(
            "CodeFormer 出力のブレンド率 0..1 "
            f"(default: {config.DEFAULT_FACE_RESTORE_BLEND}; 1.0 は従来の完全置換)"
        ),
    )
    p.add_argument(
        "--face-restore-mask",
        choices=config.FACE_RESTORE_MASK_MODES,
        default=config.DEFAULT_FACE_RESTORE_MASK,
        help="自然化合成マスク。none=全体、diff=元フレームとの差分領域",
    )
    p.add_argument(
        "--face-restore-feather",
        type=_face_restore_mask_radius,
        default=config.DEFAULT_FACE_RESTORE_FEATHER,
        help=f"diff mask のぼかし半径 px (default: {config.DEFAULT_FACE_RESTORE_FEATHER})",
    )
    p.add_argument(
        "--face-restore-mask-dilate",
        type=_face_restore_mask_radius,
        default=config.DEFAULT_FACE_RESTORE_MASK_DILATE,
        help=f"diff mask の膨張半径 px (default: {config.DEFAULT_FACE_RESTORE_MASK_DILATE})",
    )
    p.add_argument(
        "--face-restore-mask-threshold",
        type=_byte_value,
        default=config.DEFAULT_FACE_RESTORE_MASK_THRESHOLD,
        help=f"diff mask の差分閾値 0..255 (default: {config.DEFAULT_FACE_RESTORE_MASK_THRESHOLD})",
    )
    p.add_argument(
        "--face-restore-match-grain",
        type=_face_restore_unit_float,
        default=config.DEFAULT_FACE_RESTORE_MATCH_GRAIN,
        help="元フレーム由来の高周波成分を戻す量 0..1",
    )
    p.add_argument(
        "--face-restore-soften",
        type=_face_restore_unit_float,
        default=config.DEFAULT_FACE_RESTORE_SOFTEN,
        help="CodeFormer 出力側を軽くぼかす量 0..1",
    )
    p.add_argument(
        "--face-restore-global-smooth",
        type=_face_restore_unit_float,
        default=config.DEFAULT_FACE_RESTORE_GLOBAL_SMOOTH,
        help="合成後フレーム全体の弱い平滑化 0..1",
    )

    p.add_argument(
        "--encode-crf",
        type=_crf_value,
        default=config.DEFAULT_ENCODE_CRF,
        help=f"libx264 CRF 0..51 (default: {config.DEFAULT_ENCODE_CRF})",
    )
    p.add_argument(
        "--encode-preset",
        choices=_X264_PRESETS,
        default=config.DEFAULT_ENCODE_PRESET,
        help=f"libx264 preset (default: {config.DEFAULT_ENCODE_PRESET})",
    )

    p.add_argument("--max-width", type=_positive_int, default=config.DEFAULT_MAX_WIDTH)
    p.add_argument(
        "--max-duration-sec", type=_positive_int, default=config.DEFAULT_MAX_DURATION_SEC
    )
    p.add_argument("--max-temp-gib", type=_positive_float, default=config.DEFAULT_MAX_TEMP_GIB)
    p.add_argument(
        "--temp-root",
        type=Path,
        default=None,
        help=(
            "一時ファイルのnamespace root。未指定時は "
            f"{config.ENV_TEMP_ROOT}、次に {config.DEFAULT_TEMP_ROOT} を使用"
        ),
    )
    p.add_argument(
        "--png-compression-ratio",
        type=float,
        default=None,  # None -> analyze 結果による動的補正 / 明示時は固定
        help=(
            "PNG 圧縮率係数 (peak temp 見積もり用)。"
            f"default: analyze 結果から推定 (アニメ系 0.65 / 通常 {pipeline.DEFAULT_PNG_COMPRESSION_RATIO})。"
            f"明示指定すると analyze 補正は無効化される。"
        ),
    )

    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--keep-temp", action="store_true")
    p.add_argument(
        "--keep-temp-on-failure",
        action="store_true",
        help="失敗・キャンセル等の異常終了時に中間生成物 (tmp) を削除せず診断用に残す",
    )
    p.add_argument(
        "--io-idle-timeout-sec",
        type=_positive_int,
        default=pipeline.DEFAULT_IO_IDLE_TIMEOUT_SEC,
        help=(
            "subprocess (ffmpeg/realesrgan/rife) の I/O idle timeout 秒。"
            "stdout/stderr に何も出力されない時間が N 秒を超えると kill する "
            f"(default: {pipeline.DEFAULT_IO_IDLE_TIMEOUT_SEC})"
        ),
    )
    p.add_argument(
        "--stabilize-backend",
        choices=config.STABILIZE_BACKENDS,
        default=config.DEFAULT_STABILIZE_BACKEND,
        help=(
            "手ぶれ補正バックエンド: vidstab (ffmpeg vidstabtransform) / "
            "opencv (stabilize_cv.py, cv2 ベース) / "
            "auto (ffmpeg ビルドの vidstab 有無で自動分岐, default)"
        ),
    )
    p.add_argument(
        "--worker-count",
        type=_positive_int,
        default=None,
        help=(
            "並列ワーカー数 (default: auto = min(cpu_count()//2, len(chunks), 4))。"
            "CPU 段 (extract/pre_filter/post_filter/mux) のみ並列、"
            "GPU 段 (upscale/interpolate) は Semaphore(1) で直列。"
            "worker > 1 はチャンク別 tmp ディレクトリを使うため peak temp が約 worker 倍。"
            "1 を明示すると逐次。"
        ),
    )
    p.add_argument(
        "--gpu-concurrency",
        type=_positive_int,
        default=1,
        help=(
            "GPU 段 (upscale/interpolate) の同時実行数。既定 1 (直列・従来挙動)。"
            "VRAM に余裕がある高性能 GPU で 2 以上にすると GPU 区間を並列化して高速化できるが、"
            "VRAM 使用量と peak temp が増える。worker-count を超える値は worker-count に丸める。"
            "WARN: VRAM 超過時の自動回復は ncnn-vulkan で保証されていない (失敗し得る実験的 opt-in)。"
        ),
    )
    p.add_argument(
        "--gpu-tilesize",
        type=_gpu_tilesize,
        default=None,
        help=(
            "realesrgan のタイルサイズ (-t)。0=auto / それ以外は >=32。未指定でバイナリ default。"
            "VRAM に余裕があれば大きくして高速化、不足時は小さくして VRAM を節約。"
            "RIFE は -t 非対応のため渡さない。"
        ),
    )
    p.add_argument(
        "--gpu-threads",
        type=_gpu_threads,
        default=None,
        help=(
            "ncnn-vulkan の load:proc:save スレッド数 (-j)。未指定でバイナリ default (1:2:2)。"
            "realesrgan / rife 両方に渡す。proc を増やすと GPU 充填率が上がるが VRAM 使用量も増える。"
            "単一 GPU 形式 (3 個の整数) のみ。マルチ GPU 形式 (カンマ) は当面非対応。"
        ),
    )
    p.add_argument(
        "--gpu-id",
        type=_nonneg_int,
        default=None,
        help=(
            "使用する GPU の index (-g, >=0)。未指定でバイナリ auto。realesrgan / rife 両方に渡す。"
            "CPU 実行は realesrgan 非対応のため、RIFE のみ --rife-gpu-id -1 を使うこと。"
        ),
    )
    p.add_argument(
        "--rife-gpu-id",
        type=_rife_gpu_id,
        default=None,
        help=(
            "RIFE 専用の GPU index (-g)。-1=cpu / >=0=gpu。指定時は RIFE のみ --gpu-id を上書きする "
            "(realesrgan には渡さない)。realesrgan は -1 非対応のため共通 --gpu-id とは分離。"
        ),
    )
    p.add_argument(
        "--debug-split-postfilter",
        action="store_true",
        help=argparse.SUPPRESS,  # 隠しフラグ: post_filter + mux 融合を無効化 (旧 2 段パス)
    )
    p.add_argument(
        "--hw-encode",
        choices=("none", "nvenc", "qsv", "amf"),
        default="none",
        help=(
            "HW エンコードバックエンド (default: none = libx264 安定パス)。"
            "nvenc/qsv/amf 指定時は H.264 系 HW encoder へ差し替え (opt-in、速度優先)。"
            "WARN: 同じ CRF/CQ 値でも libx264 とは画質・サイズが一致せず、再生互換性が下がる場合あり。"
        ),
    )
    return p


# ---------- helpers ----------


def _apply_enable_disable(
    decs: list[StageDecision],
    enable: list[str],
    disable: list[str],
    *,
    no_realesrgan: bool,
    no_rife: bool,
    encode_crf: int,
    encode_preset: str,
    upscale_scale: int | None,
    upscale_model: str,
    target_width: int | None = None,
    rife_model: str | None = None,
    face_restore_backend: str = config.DEFAULT_FACE_RESTORE_BACKEND,
    face_restore_start: float | None = None,
    face_restore_end: float | None = None,
    face_restore_fidelity: float = config.DEFAULT_FACE_RESTORE_FIDELITY,
    face_restore_idle_timeout: int = config.DEFAULT_FACE_RESTORE_IDLE_TIMEOUT_SEC,
    face_restore_blend: float = config.DEFAULT_FACE_RESTORE_BLEND,
    face_restore_mask: str = config.DEFAULT_FACE_RESTORE_MASK,
    face_restore_feather: int = config.DEFAULT_FACE_RESTORE_FEATHER,
    face_restore_mask_dilate: int = config.DEFAULT_FACE_RESTORE_MASK_DILATE,
    face_restore_mask_threshold: int = config.DEFAULT_FACE_RESTORE_MASK_THRESHOLD,
    face_restore_match_grain: float = config.DEFAULT_FACE_RESTORE_MATCH_GRAIN,
    face_restore_soften: float = config.DEFAULT_FACE_RESTORE_SOFTEN,
    face_restore_global_smooth: float = config.DEFAULT_FACE_RESTORE_GLOBAL_SMOOTH,
) -> tuple[list[StageDecision], list[str]]:
    """`upscale_scale` / `target_width` は None なら decision の値を保持する。

    プラン §3.4: 明示指定 (`--scale` / `--target-width`) のときだけ上書き。
    """
    by = decisions_by_key(decs)
    warnings: list[str] = []
    unknown = [k for k in (enable + disable) if k not in by]
    for k in unknown:
        warnings.append(f"unknown stage key: {k!r} (ignored)")
    for k in enable:
        if k in by:
            by[k].enabled = True
    for k in disable:
        if k in by:
            by[k].enabled = False
    if no_realesrgan and "upscale" in by:
        by["upscale"].enabled = False
    if no_rife and "interpolate" in by:
        by["interpolate"].enabled = False
    # encode/upscale params override
    if "encode" in by:
        by["encode"].params["crf"] = encode_crf
        by["encode"].params["preset"] = encode_preset
        by["encode"].enabled = True  # encode は常時 ON
    if "upscale" in by and by["upscale"].enabled:
        if upscale_scale is not None:
            by["upscale"].params["scale"] = upscale_scale
        else:
            # P1-3: --scale 未指定で fixed kind モデル (x4plus 等) を選んだ場合、
            # 自動算出 scale は既定モデル (scaled kind) 前提のため許容外になり、
            # 後段の validate_realesrgan_model で必ず exit 2 してしまう。
            # モデルの許容 scale に自動調整して WARN を出す (--scale 明示時は調整しない)。
            kind, allowed = config.REALESRGAN_MODEL_KIND.get(upscale_model, ("scaled", (2, 3, 4)))
            current_scale = int(by["upscale"].params.get("scale", config.DEFAULT_REALESRGAN_SCALE))
            if kind == "fixed" and current_scale not in allowed:
                adjusted = allowed[0]
                by["upscale"].params["scale"] = adjusted
                warnings.append(
                    f"--realesrgan-model={upscale_model} は scale={adjusted} 固定のため、"
                    f"自動算出 scale={current_scale} を {adjusted} に変更しました "
                    f"(中間解像度が変わります。明示指定するには --scale {adjusted})"
                )
        by["upscale"].params["model"] = upscale_model
        if target_width is not None:
            by["upscale"].params["target_width"] = target_width
    if rife_model is not None and "interpolate" in by:
        # interpolate enabled / disabled に関わらず params に反映 (enable 時に build_rife_args が使う)
        by["interpolate"].params["model"] = rife_model
    if "face_restore" in by:
        by["face_restore"].params["backend"] = face_restore_backend
        by["face_restore"].params["start_sec"] = face_restore_start
        by["face_restore"].params["end_sec"] = face_restore_end
        by["face_restore"].params["fidelity"] = face_restore_fidelity
        by["face_restore"].params["idle_timeout_sec"] = face_restore_idle_timeout
        by["face_restore"].params["blend"] = face_restore_blend
        by["face_restore"].params["mask"] = face_restore_mask
        by["face_restore"].params["feather"] = face_restore_feather
        by["face_restore"].params["mask_dilate"] = face_restore_mask_dilate
        by["face_restore"].params["mask_threshold"] = face_restore_mask_threshold
        by["face_restore"].params["match_grain"] = face_restore_match_grain
        by["face_restore"].params["soften"] = face_restore_soften
        by["face_restore"].params["global_smooth"] = face_restore_global_smooth
    return decs, warnings


def _format_decision_table(decs: list[StageDecision]) -> str:
    lines = []
    header = f"{'KEY':<18} {'LABEL':<18} {'REC':<4} {'CONF':<8} {'EN':<3} REASON"
    lines.append(header)
    lines.append("-" * max(80, len(header)))
    for d in decs:
        lines.append(
            f"{d.key:<18} {d.label_ja:<18} "
            f"{'YES' if d.recommended else 'no':<4} {d.confidence:<8} "
            f"{'ON' if d.enabled else 'off':<3} {d.reason}"
        )
    return "\n".join(lines)


def _format_seconds_for_candidate(value: object) -> str:
    return f"{float(value):.1f}s"


def _format_face_restore_candidate_summary(
    decs: list[StageDecision],
    *,
    limit: int = 5,
) -> str | None:
    face = decisions_by_key(decs).get("face_restore")
    if face is None:
        return None
    candidates = tuple(face.params.get("candidate_ranges") or ())
    if not candidates:
        return None

    parts: list[str] = []
    for c in candidates[:limit]:
        start = _format_seconds_for_candidate(getattr(c, "start_sec"))
        end = _format_seconds_for_candidate(getattr(c, "end_sec"))
        conf = str(getattr(c, "confidence", "unknown"))
        parts.append(f"{start}-{end} {conf}")
    if len(candidates) > limit:
        parts.append(f"... (+{len(candidates) - limit})")
    return "[face_restore candidates] " + "; ".join(parts)


def _format_face_restore_candidate_note() -> str:
    return (
        "[note] candidates are based on pre-analysis face detection "
        "and may differ from CodeFormer detection."
    )


def _interactive_confirm(
    decs: list[StageDecision],
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> tuple[list[StageDecision], bool]:
    """Prompt user per stage. Returns (decisions, cancelled)."""
    candidate_line = _format_face_restore_candidate_summary(decs)
    if candidate_line:
        output_fn(candidate_line)
        output_fn(_format_face_restore_candidate_note())
    output_fn("[interactive] 各ステージに y/n/a(全採用)/q(中止) を入力してください")
    all_accept = False
    for d in decs:
        if d.key == "encode":
            # encode は常時 ON
            d.enabled = True
            continue
        default = "Y/n" if d.recommended else "y/N"
        if all_accept:
            d.enabled = d.recommended
            continue
        prompt = (
            f"  [{d.key}] {d.label_ja} (推奨={d.recommended}, conf={d.confidence}) [{default}]: "
        )
        ans = ""
        # 無効入力は黙って既定値にせず再プロンプトする (最大 3 回 -> 推奨値 + 通知)
        for _attempt in range(3):
            try:
                ans = (input_fn(prompt) or "").strip().lower()
            except EOFError:
                ans = ""
            if ans in ("", "y", "yes", "n", "no", "a", "all", "q", "quit"):
                break
            output_fn(
                "  y / n / a(全採用) / q(中止) / Enter(推奨に従う) のいずれかを入力してください"
            )
        else:
            output_fn(f"  無効な入力が続いたため推奨値 (recommended={d.recommended}) を使用します")
            ans = ""
        if ans == "":
            d.enabled = d.recommended
        elif ans in ("y", "yes"):
            d.enabled = True
        elif ans in ("n", "no"):
            d.enabled = False
        elif ans in ("a", "all"):
            all_accept = True
            d.enabled = d.recommended
        elif ans in ("q", "quit"):
            return decs, True
    return decs, False


# encode 以外はすべて画質変化を伴う想定だが、`--auto` で許可するのは:
# - encode (常時 ON)
# - upscale (情報を破壊せず追加する; recommended かつ confidence=high のとき)
# - deinterlace (recommended かつ confidence=high; field_order != progressive のとき)
_AUTO_LOW_RISK_KEYS = ("encode", "upscale", "deinterlace")


def _apply_auto_mode(
    decs: list[StageDecision],
    explicit_enable: set[str] | frozenset[str] = frozenset(),
) -> tuple[list[StageDecision], list[str]]:
    """Returns (decisions, deferred_keys).

    `deferred_keys` is the list of recommended stages that were NOT accepted by `--auto`.
    Caller should print these and exit 1 if non-empty.

    `explicit_enable` (`--enable` のキー集合) は「明示同意済み」として enabled=True にし、
    deferred に積まない (P1-2)。これが無いと「--enable で個別採用してください」という
    エラー案内に従っても同じステージが保留され続け、抜けられない。
    """
    deferred: list[str] = []
    for d in decs:
        if d.key == "encode":
            d.enabled = True
            continue
        if d.key in explicit_enable:
            d.enabled = True
            continue
        if d.recommended and d.key in _AUTO_LOW_RISK_KEYS and d.confidence == "high":
            d.enabled = True
        else:
            if d.recommended:
                deferred.append(d.key)
            d.enabled = False
    return decs, deferred


def _apply_auto_all_mode(decs: list[StageDecision]) -> list[StageDecision]:
    for d in decs:
        d.enabled = d.recommended or d.key == "encode"
    return decs


def _apply_forced_overrides(
    decs: list[StageDecision],
    enable: list[str],
    disable: list[str],
    *,
    no_realesrgan: bool,
    no_rife: bool,
) -> list[StageDecision]:
    """モード (auto / auto-all / interactive) 確定後に、明示フラグを最終再適用する。

    `--enable` / `--disable` / `--no-realesrgan` / `--no-rife` は
    「推奨やモードに関わらず最終結果を確定させる」契約 (help の "force-*" /
    "regardless of recommendation")。mode dispatch が `enabled` を `recommended` で
    塗り直すため、その後に再適用しないと強制 OFF/ON が無視される
    (例: `--auto-all --no-realesrgan` で upscale が復活してしまう)。

    `_apply_enable_disable` と同じ優先順 (enable → disable → no_*) を保ち、
    encode は常時 ON を維持する。
    """
    by = decisions_by_key(decs)
    for k in enable:
        if k in by:
            by[k].enabled = True
    for k in disable:
        if k in by:
            by[k].enabled = False
    if no_realesrgan and "upscale" in by:
        by["upscale"].enabled = False
    if no_rife and "interpolate" in by:
        by["interpolate"].enabled = False
    if "encode" in by:
        by["encode"].enabled = True
    return decs


def _emit_final_summary_and_guard(
    meta,
    decs: list[StageDecision],
    args,
    *,
    png_ratio: float,
    worker_count: int,
) -> int | None:
    """確定 decisions で peak / runtime を再計算し、`--max-temp-gib` ガード + [summary] を出す。

    Returns exit code (2 = 超過) or None (続行可)。`--max-temp-gib` ガードと UI 表示は
    `estimate_peak_for_decisions` helper を共有し、対話で stage を変えてもズレない。
    """
    peak_gib, iw, eff_fps = estimate_peak_for_decisions(
        meta,
        decs,
        args.chunk_seconds,
        png_compression_ratio=png_ratio,
        worker_count=worker_count,
    )
    if peak_gib > args.max_temp_gib:
        print(
            f"[ERROR] estimated peak temp {peak_gib:.1f} GiB > "
            f"--max-temp-gib {args.max_temp_gib} "
            f"(intermediate_width={iw}, effective_fps={eff_fps})",
            file=sys.stderr,
        )
        return 2
    est = runtime_estimator.estimate_runtime(meta, decs)
    print(
        f"[summary] estimated runtime: {format_eta_hms(est.total_seconds)} "
        f"/ peak temp: {peak_gib:.2f} GiB",
        file=sys.stderr,
    )
    return None


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    # P1-1: cp932 等のレガシーコンソールでは判定表の文字がエンコード不能になり
    # UnicodeEncodeError でクラッシュし得る。encoding は変えず (既定 cp932 のまま)、
    # エンコード不能文字のみ '?' に縮退させて出力を続行する。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, OSError, ValueError):
            pass  # reconfigure 非対応のストリーム (テストの差し替え等) は無視

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.self_check:
        print(deps.format_capability_report())
        return 0
    if args.input is None:
        parser.error("the following arguments are required: input (or use --self-check)")
    if (
        args.face_restore_start is not None
        and args.face_restore_end is not None
        and args.face_restore_start >= args.face_restore_end
    ):
        parser.error("--face-restore-start must be smaller than --face-restore-end")

    # 1) output path normalization
    try:
        output_path = config.derive_output_path(args.input, args.output)
        output_path = config.validate_output_path(output_path)
        output_path = config.make_unique_output_path(output_path)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    # 2) input existence
    if not args.input.exists():
        print(f"[ERROR] input file not found: {args.input}", file=sys.stderr)
        return 2

    # 3) ffmpeg / ffprobe (vidstab はステージ ON のときだけ厳格チェック)
    try:
        ffprobe_rb = deps.check_ffprobe()
        # ffmpeg は vidstab 入りかをまだ確定できない (stabilize 採否前) → 通常 ffmpeg 確認のみ。
        # stabilize ON のときだけ後段で check_ffmpeg_with_vidstab を呼んで差し替える。
        ffmpeg_rb = deps.check_ffmpeg()
    except MissingDependencyError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2
    print(deps.format_resolved_binary("ffprobe", ffprobe_rb), file=sys.stderr)
    print(deps.format_resolved_binary("ffmpeg", ffmpeg_rb), file=sys.stderr)

    try:
        temp_run = temp_workspace.create_temp_run_context(args.input, args.temp_root)
    except temp_workspace.TempWorkspaceError as e:
        print(f"[ERROR] temp workspace failed: {e}", file=sys.stderr)
        return 2

    def _finish(code: int, *, failed: bool = False) -> int:
        preserve = bool(args.keep_temp or (failed and args.keep_temp_on_failure))
        if preserve:
            print(f"[keep-temp] temporary files preserved at: {temp_run.run_leaf}", file=sys.stderr)
            return code
        try:
            temp_workspace.cleanup_temp_run(temp_run)
        except temp_workspace.TempWorkspaceError as exc:
            print(
                f"[WARN] cleanup failed; temporary files remain at {temp_run.run_leaf}: {exc}",
                file=sys.stderr,
            )
        return code

    # 4) analyze
    try:
        print("[analyze] probing + sampling frames...", file=sys.stderr)
        meta, stats = analyzer.analyze(
            input_path=args.input,
            ffprobe=ffprobe_rb.path,
            ffmpeg=ffmpeg_rb.path,
            sample_count=config.DEFAULT_SAMPLE_COUNT,
            sample_dir=temp_run.run_leaf / "analyze",
            keep_samples=bool(args.keep_temp or args.keep_temp_on_failure),
            warning=lambda message: print(f"[WARN] {message}", file=sys.stderr),
        )
    except (analyzer.AnalyzerError, MissingDependencyError) as e:
        print(f"[ERROR] analyze failed: {e}", file=sys.stderr)
        return _finish(2, failed=True)

    try:
        pipeline.resolve_framerate_normalization(
            meta,
            log=lambda message: print(message, file=sys.stderr),
        )
    except pipeline.PipelineError as e:
        if args.analyze_only:
            print(f"[WARN] {e} analyze-onlyのため診断表示は継続します", file=sys.stderr)
        else:
            print(f"[ERROR] {e}", file=sys.stderr)
            return _finish(2, failed=True)

    # 5) 入力上限ガード
    if meta.width > args.max_width:
        print(f"[ERROR] input width {meta.width} > --max-width {args.max_width}", file=sys.stderr)
        return _finish(2, failed=True)
    if meta.duration_sec > args.max_duration_sec:
        print(
            f"[ERROR] input duration {meta.duration_sec:.1f}s > "
            f"--max-duration-sec {args.max_duration_sec}",
            file=sys.stderr,
        )
        return _finish(2, failed=True)

    # 6) preset 解釈: 明示指定 > プリセット > デフォルト
    preset_width = config.RESOLUTION_PRESETS.get(args.target_resolution)
    preset_fps = config.FPS_PRESETS.get(args.target_fps_preset)
    effective_target_width: int | None
    target_width_explicit = False
    if args.target_width is not None:
        effective_target_width = args.target_width
        target_width_explicit = True
        if preset_width is not None and preset_width != args.target_width:
            print(
                f"[WARN] --target-width={args.target_width} overrides "
                f"--target-resolution="
                f"{_format_resolution_preset(args.target_resolution, preset_width)}",
                file=sys.stderr,
            )
    elif preset_width is not None:
        effective_target_width = preset_width
        target_width_explicit = True
    else:
        effective_target_width = None  # Auto -> decisions 側で DEFAULT_TARGET_WIDTH

    effective_target_fps: float | None
    target_fps_explicit = False
    if args.target_fps is not None:
        effective_target_fps = float(args.target_fps)
        target_fps_explicit = True
        if preset_fps is not None and preset_fps != args.target_fps:
            print(
                f"[WARN] --target-fps={args.target_fps} overrides "
                f"--target-fps-preset="
                f"{_format_fps_preset(args.target_fps_preset, preset_fps)}",
                file=sys.stderr,
            )
    elif preset_fps is not None:
        effective_target_fps = preset_fps
        target_fps_explicit = True
    else:
        effective_target_fps = None

    # 7) decisions
    rec_defaults: dict = {}
    # PR-G2: scale 選択ポリシー (既定 cover = 従来挙動)
    rec_defaults["upscale_scale_policy"] = args.upscale_scale_policy
    if effective_target_width is not None:
        rec_defaults["target_width"] = effective_target_width
        rec_defaults["target_width_explicit"] = target_width_explicit
    if effective_target_fps is not None:
        rec_defaults["target_fps"] = Fraction(effective_target_fps).limit_denominator(10000)
        rec_defaults["target_fps_explicit"] = target_fps_explicit
    decs = decisions_mod.recommend(meta, stats, rec_defaults)

    # _apply_enable_disable で --scale が上書きする前に、自動算出された scale を保持
    auto_scale_before_override: int | None = None
    if args.scale is not None:
        up_pre = decisions_by_key(decs).get("upscale")
        if up_pre is not None and up_pre.enabled:
            auto_scale_before_override = int(
                up_pre.params.get("scale", config.DEFAULT_REALESRGAN_SCALE)
            )

    decs, warns = _apply_enable_disable(
        decs,
        args.enable,
        args.disable,
        no_realesrgan=args.no_realesrgan,
        no_rife=args.no_rife,
        encode_crf=args.encode_crf,
        encode_preset=args.encode_preset,
        upscale_scale=args.scale,
        upscale_model=args.realesrgan_model,
        target_width=args.target_width,
        rife_model=args.rife_model,
        face_restore_backend=args.face_restore_backend,
        face_restore_start=args.face_restore_start,
        face_restore_end=args.face_restore_end,
        face_restore_fidelity=args.face_restore_fidelity,
        face_restore_idle_timeout=args.face_restore_idle_timeout,
        face_restore_blend=args.face_restore_blend,
        face_restore_mask=args.face_restore_mask,
        face_restore_feather=args.face_restore_feather,
        face_restore_mask_dilate=args.face_restore_mask_dilate,
        face_restore_mask_threshold=args.face_restore_mask_threshold,
        face_restore_match_grain=args.face_restore_match_grain,
        face_restore_soften=args.face_restore_soften,
        face_restore_global_smooth=args.face_restore_global_smooth,
    )
    for w in warns:
        print(f"[WARN] {w}", file=sys.stderr)
    if auto_scale_before_override is not None and auto_scale_before_override != args.scale:
        print(
            f"[WARN] --scale={args.scale} overrides preset-derived scale={auto_scale_before_override}",
            file=sys.stderr,
        )

    # 8) 見積もりパラメータ確定 (decision 確定後)
    # PR4: PNG 圧縮率係数の決定
    if args.png_compression_ratio is not None:
        png_ratio = float(args.png_compression_ratio)
    else:
        png_ratio = pipeline.estimate_compression_ratio(stats)

    # PR3 (修正): --worker-count auto = min(cpu_count()//2, len(chunks), 4)
    estimated_chunks = max(1, math.ceil(float(meta.duration_sec) / float(args.chunk_seconds)))
    if args.worker_count is None:
        cpu = max(1, (os.cpu_count() or 1) // 2)
        effective_worker_count = min(cpu, estimated_chunks, 4)
    else:
        effective_worker_count = int(args.worker_count)

    # 9) analyze-only: print table + 見積もり and exit (ガードは掛けない -- 表示専用)
    if args.analyze_only:
        peak_gib_pre, _, _ = estimate_peak_for_decisions(
            meta,
            decs,
            args.chunk_seconds,
            png_compression_ratio=png_ratio,
            worker_count=effective_worker_count,
        )
        est = runtime_estimator.estimate_runtime(meta, decs)
        print(_format_decision_table(decs))
        candidate_line = _format_face_restore_candidate_summary(decs)
        if candidate_line:
            print(candidate_line)
            print(_format_face_restore_candidate_note())
        print()
        print(
            f"input: {meta.width}x{meta.height} @ {float(meta.fps):.3f}fps, "
            f"duration={meta.duration_sec:.1f}s, audio={meta.has_audio}"
        )
        print(f"estimated peak temp: {peak_gib_pre:.2f} GiB (chunk_seconds={args.chunk_seconds})")
        print(
            f"estimated runtime: {format_eta_hms(est.total_seconds)}   "
            f"(≒ {est.total_seconds:.0f} sec, {est.accuracy_note})"
        )
        if args.verbose:
            parts = ", ".join(f"{k}={v:.1f}s" for k, v in sorted(est.per_stage_seconds.items()))
            print(f"breakdown: {parts}")
        try:
            free = shutil.disk_usage(temp_run.temp_root).free
            print(f"temp root: {temp_run.temp_root} (free={free / (1024**3):.2f} GiB)")
        except OSError:
            print(f"temp root: {temp_run.temp_root} (free=unavailable)")
        return _finish(0)

    # 8) mode dispatch
    if args.auto:
        decs, deferred = _apply_auto_mode(decs, explicit_enable=set(args.enable))
        if deferred:
            print(
                "[--auto] 以下のステージは画質変化を伴うため保留しました。明示同意が必要です:",
                file=sys.stderr,
            )
            for k in deferred:
                print(f"  - {k}", file=sys.stderr)
            print("--auto-all で全採用するか、--enable で個別採用してください。", file=sys.stderr)
            return _finish(1, failed=True)
    elif args.auto_all:
        decs = _apply_auto_all_mode(decs)
    else:
        # default = interactive
        decs, cancelled = _interactive_confirm(decs)
        if cancelled:
            print("[cancelled] user quit at prompt", file=sys.stderr)
            return _finish(1, failed=True)

    # 8a) 明示フラグの最終再適用 (mode dispatch が enabled を recommended で塗り直すため、
    #     --enable / --disable / --no-realesrgan / --no-rife を必ずモードより後で勝たせる)
    decs = _apply_forced_overrides(
        decs,
        args.enable,
        args.disable,
        no_realesrgan=args.no_realesrgan,
        no_rife=args.no_rife,
    )
    by_after_forced = decisions_by_key(decs)
    if by_after_forced.get("face_restore") is not None and by_after_forced["face_restore"].enabled:
        if args.face_restore_start is None and args.face_restore_end is None:
            print(
                "[WARN] face_restore が全区間に適用されます。"
                "必要なら --face-restore-start/--face-restore-end で範囲指定してください。",
                file=sys.stderr,
            )

    # 8a-2) 出力名を「実行スペック」から確定する。
    #   --output が既存フォルダなら、その直下に実行スペック名で保存する。
    #   decisions がここで最終確定し out_width/out_height/out_fps が定まるため、既定出力名を
    #   `<stem>_<WxH>-<fps>p<ext>` に置き換える (旧 `_enhanced` 固定を廃止)。
    if args.output is None or config.is_output_directory(args.output):
        summary = decisions_mod.compute_output_summary(meta, decs)
        spec_suffix = decisions_mod.format_output_spec_suffix(summary)
        try:
            output_path = config.validate_output_path(
                config.derive_output_path(args.input, args.output, suffix=spec_suffix)
            )
            output_path = config.make_unique_output_path(output_path)
        except ValueError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            return _finish(2, failed=True)

    # 8b) 全streamをinventoryに基づきfail-closed preflightする。
    # run_pipelineも同じadapterを再実行するため、direct callerにも防御が残る。
    try:
        pipeline.resolve_stream_preservation(meta)
    except pipeline.PipelineError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return _finish(2, failed=True)

    # 8c) 確定 decisions で peak / runtime 再計算 + --max-temp-gib ガード + [summary]
    guard_rc = _emit_final_summary_and_guard(
        meta,
        decs,
        args,
        png_ratio=png_ratio,
        worker_count=effective_worker_count,
    )
    if guard_rc is not None:
        return _finish(guard_rc, failed=True)

    peak_gib_final, _, _ = estimate_peak_for_decisions(
        meta,
        decs,
        args.chunk_seconds,
        png_compression_ratio=png_ratio,
        worker_count=effective_worker_count,
    )
    estimated_peak_bytes = int(math.ceil(peak_gib_final * (1024**3)))
    try:
        capacity_warning = temp_workspace.check_temp_capacity(temp_run, estimated_peak_bytes)
    except temp_workspace.TempWorkspaceError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return _finish(2, failed=True)
    if capacity_warning:
        print(f"[WARN] {capacity_warning}", file=sys.stderr)

    # 9) stabilize backend の解決 + vidstab 再チェック
    by = decisions_by_key(decs)
    stab_on = by.get("stabilize") is not None and by["stabilize"].enabled
    resolved_stabilize_backend = args.stabilize_backend
    if stab_on and args.stabilize_backend == "vidstab":
        # vidstab 明示要求: 無ければエラー (従来挙動)
        try:
            ffmpeg_rb = deps.check_ffmpeg_with_vidstab()
        except MissingDependencyError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            return _finish(2, failed=True)
        print(deps.format_resolved_binary("ffmpeg(vidstab)", ffmpeg_rb), file=sys.stderr)
    elif stab_on and args.stabilize_backend == "auto":
        # P2-4: run_pipeline は ffmpeg を渡されると auto を vidstab と仮定するため、
        # CLI 側で vidstab の有無を確認して backend を確定させる
        # (vidstab 非搭載ビルドでは OpenCV バックエンドへフォールバック)。
        try:
            ffmpeg_rb = deps.check_ffmpeg_with_vidstab()
            resolved_stabilize_backend = "vidstab"
            print(deps.format_resolved_binary("ffmpeg(vidstab)", ffmpeg_rb), file=sys.stderr)
        except MissingDependencyError:
            resolved_stabilize_backend = "opencv"
            print(
                "[info] ffmpeg が vidstab 非搭載のため OpenCV 手ぶれ補正バックエンドを使用します",
                file=sys.stderr,
            )

    # 10) realesrgan / rife
    realesrgan_path = None
    rife_path = None
    codeformer_local = None
    try:
        if by.get("upscale") is not None and by["upscale"].enabled:
            rea_rb = deps.check_realesrgan()
            print(deps.format_resolved_binary("realesrgan", rea_rb), file=sys.stderr)
            # PR5: model 実存チェック (バイナリ同階層の models/ を見る)
            rea_models_dir = rea_rb.path.parent / "models"
            scale_for_validate = int(
                by["upscale"].params.get("scale", config.DEFAULT_REALESRGAN_SCALE)
            )
            try:
                config.validate_realesrgan_model(
                    rea_models_dir, args.realesrgan_model, scale_for_validate
                )
            except (FileNotFoundError, ValueError) as ve:
                raise MissingDependencyError("realesrgan", str(ve)) from ve
            realesrgan_path = rea_rb.path
        if by.get("interpolate") is not None and by["interpolate"].enabled:
            rife_rb = deps.check_rife()
            print(deps.format_resolved_binary("rife", rife_rb), file=sys.stderr)
            verification = deps.verify_rife_sha256(rife_rb)
            if verification.level == "warn":
                print(f"[WARN] {verification.message}", file=sys.stderr)
            else:
                print(f"[info] {verification.message}", file=sys.stderr)
            # PR5: rife model 実存チェック (公式配布は exe 同階層にモデル dir が並ぶ。
            # resolve_rife_model_dir が旧想定 models/ も含めて解決する)
            try:
                rife_models_dir = config.resolve_rife_model_dir(
                    rife_rb.path.parent, args.rife_model
                )
                config.validate_rife_model(rife_models_dir, args.rife_model)
            except (FileNotFoundError, ValueError) as ve:
                raise MissingDependencyError("rife", str(ve)) from ve
            rife_path = rife_rb.path
        if by.get("face_restore") is not None and by["face_restore"].enabled:
            codeformer_local = deps.check_codeformer_local()
            print(
                f"[deps] codeformer-local: repo={codeformer_local.repo_root} "
                f"python={codeformer_local.python} script={codeformer_local.script}",
                file=sys.stderr,
            )
            for model_key, model_path in codeformer_local.models.items():
                sha = codeformer_local.model_sha256.get(model_key) or "sha256=not-configured"
                if not sha.startswith("sha256="):
                    sha = f"sha256={sha}"
                print(f"[deps] codeformer {model_key}: path={model_path} {sha}", file=sys.stderr)
    except MissingDependencyError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return _finish(2, failed=True)

    # 10b) PR7: HW encode opt-in WARN + 実存確認
    if args.hw_encode != "none":
        print(
            f"[WARN] --hw-encode={args.hw_encode} を使用しています。"
            f"HW encoder は速度優先のオプションです。\n"
            f"       - 同じ CRF/CQ 値でも libx264 とは画質・サイズが一致しません\n"
            f"       - GPU ドライバ / SDK バージョンに依存し、再生互換性が下がるケースがあります\n"
            f"       - 保存品質を重視するなら --hw-encode none (既定) を推奨します",
            file=sys.stderr,
        )
        encoder_name = {"nvenc": "h264_nvenc", "qsv": "h264_qsv", "amf": "h264_amf"}[args.hw_encode]
        if not deps.check_hw_encoder(ffmpeg_rb.path, encoder_name):
            print(
                f"[ERROR] ffmpeg ビルドに {encoder_name} が含まれていません。"
                f"ffmpeg -hide_banner -encoders で対応 encoder を確認してください。",
                file=sys.stderr,
            )
            return _finish(2, failed=True)

    # 10c) PR-G1: GPU 並列 opt-in WARN + 中間解像度ベースの粗い VRAM 概算
    if args.gpu_concurrency > 1:
        effective_gpu = min(int(args.gpu_concurrency), effective_worker_count)
        summary = decisions_mod.compute_output_summary(meta, decs)
        vram_note = ""
        if summary.upscale_enabled and summary.intermediate_width:
            inter_w = int(summary.intermediate_width)
            # 中間高さは src のアスペクト比を維持 (intermediate_w / src_w 倍)
            inter_h = int(round(meta.height * inter_w / max(1, meta.width)))
            # 粗い概算: 中間フレーム RGBA (4ch) × 安全係数 8 (モデル作業領域 + パイプライン段)
            #   × 同時実行チャンク数。実 VRAM は tile/モデル依存のため上振れ目安。
            est_bytes = inter_w * inter_h * 4 * 8 * effective_gpu
            est_gib = est_bytes / (1024**3)
            vram_note = (
                f"\n       - GPU 中間フレーム概算: {inter_w}x{inter_h} × {effective_gpu} 並列 "
                f"≒ {est_gib:.1f} GiB 前後 (tile/モデル依存の粗い目安)"
            )
        print(
            f"[WARN] --gpu-concurrency={args.gpu_concurrency} (有効 {effective_gpu}) を使用しています。"
            f"GPU 段を並列化する実験的オプションです。\n"
            f"       - VRAM 超過時の自動回復は ncnn-vulkan で保証されていません "
            f"(device lost / crash の可能性)\n"
            f"       - VRAM が不足する場合は --gpu-concurrency 1 に戻してください"
            f"{vram_note}\n"
            f"       - 追加対策として --gpu-tilesize で 1 プロセスあたりの VRAM を抑えられます",
            file=sys.stderr,
        )

    # 11) run pipeline
    # PR-D: チャンク粒度 ETA。process_chunks で起点をセットし、chunk_done で残り時間を出す
    _run_started: list[float | None] = [None]
    _chunk_done_count = [0]

    def _on_progress(ev: pipeline.ProgressEvent) -> None:
        if ev.stage == "process_chunks" and _run_started[0] is None:
            _run_started[0] = time.monotonic()
        if ev.stage == "chunk_done":
            _chunk_done_count[0] += 1
            started = _run_started[0]
            if started is None:
                started = _run_started[0] = time.monotonic()
            elapsed = time.monotonic() - started
            done = _chunk_done_count[0]
            total = ev.total_chunks or 1
            eta_sec = elapsed / done * (total - done) if done >= 3 else None
            if args.verbose or eta_sec is not None:
                tail = f", ETA {format_eta_hms(eta_sec)}" if eta_sec is not None else ""
                print(f"[progress] {done}/{total} chunks{tail}", file=sys.stderr)

    # P2-2: 並列実行時のエラー伝播 / Ctrl+C 中断のため CLI でも cancel_event を配線する
    cancel_event = threading.Event()
    try:
        pipeline.run_pipeline(
            input_path=args.input,
            output_path=output_path,
            decisions=decs,
            meta=meta,
            chunk_seconds=args.chunk_seconds,
            keep_temp=args.keep_temp,
            keep_temp_on_failure=args.keep_temp_on_failure,
            log=print if args.verbose else (lambda m: None),
            cancel_event=cancel_event,
            ffmpeg=ffmpeg_rb.path,
            ffprobe=ffprobe_rb.path,
            realesrgan=realesrgan_path,
            rife=rife_path,
            codeformer=codeformer_local,
            io_idle_timeout_sec=args.io_idle_timeout_sec,
            face_restore_options=pipeline.FaceRestoreOptions(
                start_sec=args.face_restore_start,
                end_sec=args.face_restore_end,
                fidelity=args.face_restore_fidelity,
                idle_timeout_sec=args.face_restore_idle_timeout,
                blend=args.face_restore_blend,
                mask=args.face_restore_mask,
                feather=args.face_restore_feather,
                mask_dilate=args.face_restore_mask_dilate,
                mask_threshold=args.face_restore_mask_threshold,
                match_grain=args.face_restore_match_grain,
                soften=args.face_restore_soften,
                global_smooth=args.face_restore_global_smooth,
            ),
            stabilize_backend=resolved_stabilize_backend,
            worker_count=effective_worker_count,
            gpu_concurrency=args.gpu_concurrency,
            clamp_flags=args.clamp_flags,
            gpu_tilesize=args.gpu_tilesize,
            gpu_threads=args.gpu_threads,
            gpu_id=args.gpu_id,
            rife_gpu_id=args.rife_gpu_id,
            debug_split_postfilter=args.debug_split_postfilter,
            hw_encode=args.hw_encode,
            temp_run=temp_run,
            estimated_peak_temp_bytes=estimated_peak_bytes,
            progress=_on_progress,
        )
    except KeyboardInterrupt:
        # Ctrl+C: ワーカーへ cancel を伝播し、生 traceback を出さずに終了する
        cancel_event.set()
        print("[CANCELLED] Ctrl+C により中止しました", file=sys.stderr)
        if args.keep_temp_on_failure:
            print(f"[keep-temp-on-failure] 中間生成物: {temp_run.run_leaf}", file=sys.stderr)
        return _finish(130, failed=True)
    except pipeline.PipelineCancelled as e:
        print(f"[CANCELLED] {e}", file=sys.stderr)
        if args.keep_temp_on_failure:
            print(f"[keep-temp-on-failure] 中間生成物: {temp_run.run_leaf}", file=sys.stderr)
        return _finish(1, failed=True)
    except pipeline.PipelineError as e:
        print(f"[FAIL] stage={e.stage}: {e.message}", file=sys.stderr)
        if args.keep_temp_on_failure:
            print(f"[keep-temp-on-failure] 中間生成物: {temp_run.run_leaf}", file=sys.stderr)
        return _finish(1, failed=True)
    except MissingDependencyError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return _finish(2, failed=True)

    print(f"[OK] saved: {output_path}")
    return _finish(0)


if __name__ == "__main__":
    sys.exit(main())
