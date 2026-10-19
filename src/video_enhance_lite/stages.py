from __future__ import annotations

import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

from . import config
from .decisions import StageDecision, decisions_by_key

__all__ = [
    "StageSpec",
    "STAGE_REGISTRY",
    "PUBLIC_BUILTIN_STAGE_COUNT",
    "PUBLIC_STAGE_COUNT_LABEL",
    "build_pre_filter_chain",
    "build_post_filter_chain",
    "build_resolution_clamp_filter",
    "build_stabilize_detect_args",
    "build_stabilize_transform_filter",
    "build_encode_args",
    "build_rife_args",
    "build_codeformer_args",
    "build_codeformer_env",
    "validate_face_restore_fidelity",
    "build_extract_args",
    "build_mux_args",
    "join_filters",
    "escape_filter_path",
    "HW_ENCODE_BACKENDS",
    "STABILIZE_PIXFMT",
]


@dataclass(frozen=True)
class StageSpec:
    key: str
    label_ja: str
    capability: str | None = None
    optional_external: bool = False
    public_count: bool = True


# Public-facing stage contract.  Face restoration is an optional external
# CodeFormer capability, so documentation can distinguish the built-in count
# from optional engines without pretending that every installation has 12
# runnable stages.
STAGE_REGISTRY: tuple[StageSpec, ...] = (
    StageSpec("deinterlace", config.STAGE_LABELS_JA["deinterlace"]),
    StageSpec("stabilize", config.STAGE_LABELS_JA["stabilize"], "Stabilize"),
    StageSpec("deflicker", config.STAGE_LABELS_JA["deflicker"]),
    StageSpec("denoise", config.STAGE_LABELS_JA["denoise"]),
    StageSpec("debanding", config.STAGE_LABELS_JA["debanding"]),
    StageSpec("color_correction", config.STAGE_LABELS_JA["color_correction"]),
    StageSpec("sharpen", config.STAGE_LABELS_JA["sharpen"]),
    StageSpec("upscale", config.STAGE_LABELS_JA["upscale"], "Upscale", True),
    StageSpec("face_restore", config.STAGE_LABELS_JA["face_restore"], "Face restore", True, False),
    StageSpec("interpolate", config.STAGE_LABELS_JA["interpolate"], "Interpolate", True),
    StageSpec("grain", config.STAGE_LABELS_JA["grain"]),
    StageSpec("encode", config.STAGE_LABELS_JA["encode"], "Encode/Decode"),
)
PUBLIC_BUILTIN_STAGE_COUNT = sum(spec.public_count for spec in STAGE_REGISTRY)
PUBLIC_STAGE_COUNT_LABEL = f"{PUBLIC_BUILTIN_STAGE_COUNT}組込み段階 + 任意の顔復元1段階"

PRE_FILTER_ORDER = ("deinterlace", "stabilize", "deflicker", "denoise", "debanding")
POST_FILTER_ORDER = ("color_correction", "sharpen", "grain")

# vidstab (vidstabdetect / vidstabtransform) に渡す前に強制するピクセルフォーマット。
# ffmpeg N-120725+ (libavfilter 11.x) では vidstabtransform に rgb24 (PNG 入力) を
# 渡すと segfault する既知バグ (TODO.md V1-1) があるため、直前に yuv420p へ変換して回避する。
# チャンク抽出 PNG は元動画 (通常 yuv420p) 由来なので、yuv420p への変換で追加の画質劣化は生じない。
# detect / transform の双方に同一フォーマットを適用し、推定する動きベクトルの基準を揃える。
STABILIZE_PIXFMT = "yuv420p"


def escape_filter_path(p: Path | str) -> str:
    """Escape a path for ffmpeg filter parameters (e.g. -vf vidstabtransform=input=PATH).

    ffmpeg filter syntax interprets `:` and `\\` as separators / escapes,
    and the whole filter chain is comma/semicolon delimited.
    We follow the documented rule:
      1. Backslash-escape `\\`, `:`, `'`, `,`, `;`, `=`, `[`, `]`
      2. The result MUST be wrapped in single quotes by the caller if the filter
         string will appear inside `-vf` chain.
    Reference: https://ffmpeg.org/ffmpeg-filters.html#Notes-on-filtergraph-escaping
    """
    s = str(p)
    # 第1段階: バックスラッシュ自身は \\ にする
    out = []
    for ch in s:
        if ch in ("\\", ":", "'", ",", ";", "=", "[", "]"):
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def join_filters(parts: Iterable[str]) -> str:
    """Comma-join non-empty filter snippets."""
    return ",".join(p for p in parts if p)


# ---------- pre-filter snippets ----------


def _snippet_deinterlace(d: StageDecision) -> str:
    if not d.enabled:
        return ""
    # `yadif=0` (send_frame): フレーム毎に 1 枚出力し、フレーム数・実効 fps を不変に保つ。
    # `yadif=1` (send_field) はフィールド毎 1 枚 = 枚数が倍化するため、固定 framerate で
    # mux する本パイプラインでは使用不可 (チャンクが 2 倍スロー + 音ズレになる。2026-06 修正)。
    return "yadif=0"


def _snippet_stabilize_transform(d: StageDecision, transforms_path: Path | None) -> str:
    """vidstabtransform は 2 パス目で `input=transforms.trf` を必要とする."""
    if not d.enabled:
        return ""
    if transforms_path is None:
        # transforms_path がない場合は filter を発行しない（呼び出し側のミス）
        return ""
    escaped = escape_filter_path(transforms_path)
    # smoothing は既定 10、unsharp 後段に任せる
    # rgb24 入力での vidstabtransform segfault 回避のため format=yuv420p を前置する (STABILIZE_PIXFMT 参照)
    return f"format={STABILIZE_PIXFMT},vidstabtransform=input='{escaped}':smoothing=10"


def _snippet_deflicker(d: StageDecision) -> str:
    if not d.enabled:
        return ""
    size = int(d.params.get("size", config.DEFAULT_FLICKER_SIZE))
    return f"deflicker=size={size}:mode=pm"


def _snippet_denoise(d: StageDecision) -> str:
    if not d.enabled:
        return ""
    expr = d.params.get("strength") or config.DEFAULT_DENOISE_STRENGTH
    return str(expr)


def _snippet_debanding(d: StageDecision) -> str:
    if not d.enabled:
        return ""
    r = int(d.params.get("range", config.DEFAULT_DEBAND_RANGE))
    return f"deband=range={r}:direction=0"


def build_pre_filter_chain(
    decisions: list[StageDecision],
    *,
    transforms_path: Path | None = None,
    exclude_stabilize: bool = False,
) -> str:
    """Build pre_filter `-vf` chain.

    `exclude_stabilize=True` で stabilize スニペット (vidstabtransform) を強制除外する。
    opencv backend のときに pipeline 側から指定する (後段で独立に処理するため)。
    """
    by = decisions_by_key(decisions)
    parts: list[str] = []
    for key in PRE_FILTER_ORDER:
        d = by.get(key)
        if d is None or not d.enabled:
            continue
        if key == "deinterlace":
            parts.append(_snippet_deinterlace(d))
        elif key == "stabilize":
            if exclude_stabilize:
                continue
            parts.append(_snippet_stabilize_transform(d, transforms_path))
        elif key == "deflicker":
            parts.append(_snippet_deflicker(d))
        elif key == "denoise":
            parts.append(_snippet_denoise(d))
        elif key == "debanding":
            parts.append(_snippet_debanding(d))
    return join_filters(parts)


# ---------- post-filter snippets ----------


def _snippet_color_correction(d: StageDecision) -> str:
    if not d.enabled:
        return ""
    b = float(d.params.get("brightness", config.DEFAULT_COLOR_BRIGHTNESS))
    c = float(d.params.get("contrast", config.DEFAULT_COLOR_CONTRAST))
    s = float(d.params.get("saturation", config.DEFAULT_COLOR_SATURATION))
    return f"eq=brightness={b}:contrast={c}:saturation={s}"


def _snippet_sharpen(d: StageDecision) -> str:
    if not d.enabled:
        return ""
    amount = float(d.params.get("amount", config.DEFAULT_SHARPEN_AMOUNT))
    # unsharp: luma_msize_x:luma_msize_y:luma_amount
    return f"unsharp=5:5:{amount:.3f}:5:5:0.0"


def _snippet_grain(d: StageDecision) -> str:
    if not d.enabled:
        return ""
    amount = int(d.params.get("amount", config.DEFAULT_GRAIN_AMOUNT))
    return f"noise=alls={amount}:allf=t"


def build_resolution_clamp_filter(
    target_width: int,
    intermediate_width: int,
    flags: str = config.DEFAULT_CLAMP_FLAGS,
) -> str:
    """upscale 直後 / interpolate 直前に挟む独立 ffmpeg ステップの -vf 値を返す。

    intermediate_width == target_width のときは "" (clamp スキップ)。
    そうでなければ `scale=<target_width>:-2:flags=<flags>` を返す (高さは AR 維持 + 2 の倍数アライン)。

    `flags` は rescale アルゴリズム (既定 "lanczos" = 従来挙動・不変)。PR-G2 で bicubic/bilinear を
    opt-in 露出。target_width > intermediate_width の場合は「上方 lanczos」になる
    (Real-ESRGAN scale=4 で頭打ちした場合)。呼び出し側は警告ログを出すこと。
    """
    if target_width <= 0:
        raise ValueError(f"target_width must be > 0, got {target_width}")
    if intermediate_width <= 0:
        raise ValueError(f"intermediate_width must be > 0, got {intermediate_width}")
    if flags not in config.CLAMP_FLAGS:
        raise ValueError(f"flags must be one of {config.CLAMP_FLAGS}, got {flags!r}")
    if intermediate_width == target_width:
        return ""
    return f"scale={target_width}:-2:flags={flags}"


def build_post_filter_chain(decisions: list[StageDecision]) -> str:
    by = decisions_by_key(decisions)
    parts: list[str] = []
    for key in POST_FILTER_ORDER:
        d = by.get(key)
        if d is None or not d.enabled:
            continue
        if key == "color_correction":
            parts.append(_snippet_color_correction(d))
        elif key == "sharpen":
            parts.append(_snippet_sharpen(d))
        elif key == "grain":
            parts.append(_snippet_grain(d))
    return join_filters(parts)


# ---------- stabilize detect (pass 1) ----------


def build_stabilize_detect_args(
    ffmpeg: Path,
    input_pattern: Path,
    transforms_path: Path,
    *,
    fps: Fraction,
) -> list[str]:
    """vidstabdetect は 2 パス必須の 1 パス目。output 動画は捨てる (null muxer).

    `input_pattern` は frames_in/%08d.png を想定。
    """
    escaped = escape_filter_path(transforms_path)
    fps_str = f"{fps.numerator}/{fps.denominator}"
    return [
        str(ffmpeg),
        "-y",
        "-framerate",
        fps_str,
        "-start_number",
        "0",
        "-i",
        str(input_pattern),
        # rgb24 入力での vidstab segfault 回避のため format=yuv420p を前置する (STABILIZE_PIXFMT 参照)
        "-vf",
        f"format={STABILIZE_PIXFMT},vidstabdetect=result='{escaped}':shakiness=5:accuracy=15",
        "-f",
        "null",
        "-",
    ]


def build_stabilize_transform_filter(transforms_path: Path) -> str:
    """単独で transform filter snippet を取りたいケース (テスト用)."""
    escaped = escape_filter_path(transforms_path)
    # rgb24 入力での vidstabtransform segfault 回避のため format=yuv420p を前置する (STABILIZE_PIXFMT 参照)
    return f"format={STABILIZE_PIXFMT},vidstabtransform=input='{escaped}':smoothing=10"


# ---------- encode args ----------


HW_ENCODE_BACKENDS = ("none", "nvenc", "qsv", "amf")

# 各 HW backend の libx264 -> backend-specific preset 名マッピング
_HW_PRESET_MAP: dict[str, dict[str, str]] = {
    "nvenc": {
        "ultrafast": "p1",
        "superfast": "p1",
        "veryfast": "p2",
        "faster": "p2",
        "fast": "p3",
        "medium": "p4",
        "slow": "p5",
        "slower": "p6",
        "veryslow": "p7",
    },
    "qsv": {
        "ultrafast": "veryfast",
        "superfast": "veryfast",
        "veryfast": "faster",
        "faster": "faster",
        "fast": "fast",
        "medium": "medium",
        "slow": "slow",
        "slower": "slower",
        "veryslow": "veryslow",
    },
    "amf": {
        "ultrafast": "speed",
        "superfast": "speed",
        "veryfast": "speed",
        "faster": "balanced",
        "fast": "balanced",
        "medium": "balanced",
        "slow": "quality",
        "slower": "quality",
        "veryslow": "quality",
    },
}

_HW_ENCODER_NAMES: dict[str, str] = {
    "nvenc": "h264_nvenc",
    "qsv": "h264_qsv",
    "amf": "h264_amf",
}


def _map_preset_for_backend(preset: str, hw_backend: str) -> str:
    """libx264 preset -> hw backend-specific preset."""
    table = _HW_PRESET_MAP.get(hw_backend)
    if table is None:
        return preset
    if preset not in table:
        raise ValueError(f"unknown encode preset for {hw_backend}: {preset!r}")
    return table[preset]


def build_encode_args(decision: StageDecision, hw_backend: str = "none") -> list[str]:
    """encode 引数を返す。

    `hw_backend == "none"` (default) のときは旧シグネチャ・旧戻り値と完全一致
    (`-c:v libx264 -preset X -crf Y -pix_fmt yuv420p`)。既定挙動の不変性は
    test_stages_hw_encode.py で固定されている。

    `hw_backend` が nvenc/qsv/amf のときは H.264 系 HW encoder へ差し替える:
      - nvenc: `-c:v h264_nvenc -preset pN -cq Y -pix_fmt yuv420p`
      - qsv:   `-c:v h264_qsv -preset XX -global_quality Y -pix_fmt yuv420p`
      - amf:   `-c:v h264_amf -quality XX -rc cqp -qp_i/p/b Y -pix_fmt yuv420p`
    HW encoder は速度優先用途の opt-in。同じ CRF/CQ 値でも libx264 とは画質が一致しない。
    """
    crf = int(decision.params.get("crf", config.DEFAULT_ENCODE_CRF))
    preset = str(decision.params.get("preset", config.DEFAULT_ENCODE_PRESET))
    if hw_backend == "none":
        return [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
        ]
    if hw_backend not in _HW_ENCODER_NAMES:
        raise ValueError(f"unknown hw_backend: {hw_backend!r}")
    codec = _HW_ENCODER_NAMES[hw_backend]
    mapped_preset = _map_preset_for_backend(preset, hw_backend)
    if hw_backend == "nvenc":
        return [
            "-c:v",
            codec,
            "-preset",
            mapped_preset,
            "-cq",
            str(crf),
            "-pix_fmt",
            "yuv420p",
        ]
    if hw_backend == "qsv":
        return [
            "-c:v",
            codec,
            "-preset",
            mapped_preset,
            "-global_quality",
            str(crf),
            "-pix_fmt",
            "yuv420p",
        ]
    # amf
    return [
        "-c:v",
        codec,
        "-quality",
        mapped_preset,
        "-rc",
        "cqp",
        "-qp_i",
        str(crf),
        "-qp_p",
        str(crf),
        "-qp_b",
        str(crf),
        "-pix_fmt",
        "yuv420p",
    ]


# ---------- RIFE ----------


def build_rife_args(
    rife_bin: Path,
    in_dir: Path,
    out_dir: Path,
    source_fps: Fraction,
    target_fps: Fraction,
    input_frame_count: int,
    model: str | None = None,
    threads: str | None = None,
    gpu_id: int | None = None,
) -> list[str]:
    """RIFE-ncnn-vulkan の引数を組み立てる.

    target_fps は source_fps の整数倍であることを Fraction で検証する。
    24000/1001 のような非整数 fps でも比率判定が壊れないよう、
    `int()` での丸めではなく `Fraction` のまま `denominator == 1` を確認する。

    `model` は rife-ncnn-vulkan の `-m <model>` に渡すモデル名 (バイナリと同階層の
    サブディレクトリ名)。`-n` (custom numframe) は v4 系モデルでしか動かないため、
    既定は `config.DEFAULT_RIFE_MODEL` (= rife-v4.6)。v2.3 系を指定すると runtime に
    rife から "only rife-v4 model support custom numframe and timestep" エラーが返る。

    PR-G3: `threads` (-j load:proc:save) / `gpu_id` (-g, -1=cpu 可) は opt-in。
    RIFE は `-t` tilesize 非対応のため tilesize は受け取らない。idle watchdog の
    生存信号として `-v` は常時付与する。
    """
    if source_fps <= 0:
        raise ValueError(f"source_fps must be > 0, got {source_fps}")
    if target_fps <= 0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")
    if input_frame_count < 1:
        raise ValueError(f"input_frame_count must be >= 1, got {input_frame_count}")
    src = Fraction(source_fps)
    tgt = Fraction(target_fps)
    ratio = tgt / src
    if ratio.denominator != 1:
        raise ValueError(
            f"target_fps {target_fps} must be an integer multiple of source_fps {source_fps}; "
            f"ratio={ratio}"
        )
    n_out = input_frame_count * int(ratio)
    chosen_model = model if model else config.DEFAULT_RIFE_MODEL
    # RIFE は長時間処理中の既定出力が乏しいため、保存完了行を idle watchdog の
    # 生存信号として常時利用する。位置は外部契約にせず、両 return 共通で1個だけ付与する。
    extra: list[str] = ["-v"]
    # PR-G3: opt-in 実行フラグ (両 return 共通)。
    if threads is not None:
        extra += ["-j", str(threads)]
    if gpu_id is not None:
        extra += ["-g", str(gpu_id)]
    # v2/v3 系モデルは `-n` (custom numframe) 非対応。ratio==2 のみ許容、
    # それ以外は ValueError (silent な品質劣化を避ける)。
    if chosen_model not in config.RIFE_MODELS_WITH_CUSTOM_N:
        if int(ratio) != 2:
            raise ValueError(
                f"model {chosen_model} only supports x2 interpolation "
                f"(no `-n` custom numframe), but ratio is x{int(ratio)}. "
                f"Use rife-v4 series (e.g. rife-v4.6) for arbitrary integer ratios."
            )
        # v2/v3: -n を付けず、rife 側固定 ×2 動作にフォールバック
        return [
            str(rife_bin),
            "-m",
            chosen_model,
            "-i",
            str(in_dir),
            "-o",
            str(out_dir),
            "-f",
            "png",
        ] + extra
    return [
        str(rife_bin),
        "-m",
        chosen_model,
        "-i",
        str(in_dir),
        "-o",
        str(out_dir),
        "-n",
        str(n_out),
        "-f",
        "png",
    ] + extra


# ---------- CodeFormer local backend ----------


def validate_face_restore_fidelity(value: float) -> float:
    v = float(value)
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"face_restore fidelity must be in 0..1, got {value!r}")
    return v


def build_codeformer_args(
    *,
    python: Path,
    script: Path,
    input_dir: Path,
    output_dir: Path,
    fidelity: float = config.DEFAULT_FACE_RESTORE_FIDELITY,
    detection_model: str = config.DEFAULT_FACE_RESTORE_DETECTION_MODEL,
) -> list[str]:
    """Build CodeFormer stock CLI argv for local-only face restoration.

    The initial contract intentionally uses stock `final_results/` output, with
    `--upscale 1` and no background upsampler to avoid extra RealESRGAN weights.
    """
    w = validate_face_restore_fidelity(fidelity)
    return [
        str(python),
        str(script),
        "--input_path",
        str(input_dir),
        "--output_path",
        str(output_dir),
        "-w",
        f"{w:.3f}",
        "--upscale",
        "1",
        "--bg_upsampler",
        "None",
        "--detection_model",
        str(detection_model),
    ]


def _infer_python_prefix(python: Path) -> Path:
    if python.name.lower() == "python.exe" and python.parent.name.lower() == "scripts":
        return python.parent.parent
    return python.parent


def _prepend_path_entries(current_path: str, entries: Iterable[Path]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for raw in [*(str(p) for p in entries if p.is_dir()), *current_path.split(os.pathsep)]:
        if not raw:
            continue
        key = raw.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return os.pathsep.join(out)


def build_codeformer_env(
    base_env: dict[str, str] | None = None,
    *,
    python: Path | None = None,
) -> dict[str, str]:
    """Return a small subprocess env for CodeFormer.

    We do not dump or pass the entire process environment. Windows/Python still
    needs a few OS variables and sometimes PATH for DLL resolution.
    """
    src = os.environ if base_env is None else base_env
    keep = ("SystemRoot", "WINDIR", "TEMP", "TMP", "PATH")
    env = {k: src[k] for k in keep if k in src and src[k]}
    if python is not None:
        prefix = _infer_python_prefix(python)
        if prefix.is_dir():
            path_entries = (
                prefix / "bin",
                prefix / "Library" / "bin",
                prefix / "Scripts",
                prefix,
            )
            env["PATH"] = _prepend_path_entries(env.get("PATH", ""), path_entries)
            env["CONDA_PREFIX"] = str(prefix)
            cuda_path = src.get("CUDA_PATH")
            env["CUDA_PATH"] = cuda_path if cuda_path else str(prefix)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


# ---------- frame extract / mux helpers ----------


def build_extract_args(
    ffmpeg: Path,
    input_path: Path,
    start_sec: float,
    dur_sec: float,
    frames_out: Path,
    *,
    vf_chain: str = "",
    normalization_fps: Fraction | None = None,
    expected_frame_count: int | None = None,
) -> list[str]:
    """Chunk extract command for the standard chunked pipeline.

    Indexed containers use fast accurate input seeking. MPEG-TS family inputs can
    seek to a later GOP on current FFmpeg builds and silently lose boundary frames,
    so they retain the correctness-first output-side seek path.
    """
    transport_stream = input_path.suffix.lower() in {".ts", ".mts", ".m2ts"}
    argv = [str(ffmpeg), "-y"]
    if not transport_stream:
        argv += ["-ss", f"{start_sec}"]
    argv += ["-i", str(input_path)]
    if transport_stream:
        argv += ["-ss", f"{start_sec}"]
    if normalization_fps is not None:
        normalization_fps = Fraction(normalization_fps)
        if normalization_fps <= 0:
            raise ValueError(f"normalization_fps must be > 0, got {normalization_fps}")
        if expected_frame_count is not None and expected_frame_count < 1:
            raise ValueError(f"expected_frame_count must be >= 1, got {expected_frame_count}")
    elif expected_frame_count is not None:
        raise ValueError("expected_frame_count requires normalization_fps")

    argv += [
        "-t",
        f"{dur_sec}",
        "-fps_mode",
        "cfr" if normalization_fps is not None else "passthrough",
        "-start_number",
        "0",
    ]
    filter_parts: list[str] = []
    if normalization_fps is not None:
        filter_parts.append(f"fps={normalization_fps.numerator}/{normalization_fps.denominator}")
    if vf_chain:
        filter_parts.append(vf_chain)
    if filter_parts:
        argv += ["-vf", ",".join(filter_parts)]
    if expected_frame_count is not None:
        argv += ["-frames:v", str(expected_frame_count)]
    argv += [str(frames_out / "%08d.png")]
    return argv


def build_mux_args(
    ffmpeg: Path,
    frames_dir: Path,
    fps: Fraction,
    chunk_path: Path,
    encode_args: list[str],
    *,
    vf_chain: str = "",
) -> list[str]:
    """Build mux argv. `vf_chain` を渡せば PNG 入力 + フィルタ + encode を 1 段で行える
    (PR6: post_filter + mux 融合)。空文字なら従来通り encode のみ。
    """
    fps_str = f"{fps.numerator}/{fps.denominator}"
    argv: list[str] = [
        str(ffmpeg),
        "-y",
        "-framerate",
        fps_str,
        "-start_number",
        "0",
        "-i",
        str(frames_dir / "%08d.png"),
        "-map",
        "0:v:0",
    ]
    if vf_chain:
        argv += ["-vf", vf_chain]
    argv += [*encode_args, "-r", fps_str, str(chunk_path)]
    return argv
