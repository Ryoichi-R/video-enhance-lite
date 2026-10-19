from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

# Public runtime identity.  The source directory is intentionally not used as
# an application-data root: a checkout may be read-only and a wheel has no
# stable project root beside ``site-packages``.
APP_ID = "video-enhance-lite"
ENV_APP_DATA_ROOT = "VIDEO_ENHANCE_APP_DATA_ROOT"
ENV_CONFIG_ROOT = "VIDEO_ENHANCE_CONFIG_ROOT"
ENV_OUTPUT_ROOT = "VIDEO_ENHANCE_OUTPUT_ROOT"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Kept as a compatibility/test injection seam only.  Production defaults do
# not derive settings, temp files, binaries, or output from this value.
WORKSPACE_ROOT = PROJECT_ROOT.parent
_LEGACY_WORKSPACE_ROOT = WORKSPACE_ROOT


def _user_local_root() -> Path:
    explicit = os.environ.get(ENV_APP_DATA_ROOT, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            return Path(local_app_data)
        return Path.home() / "AppData" / "Local"
    xdg_state = os.environ.get("XDG_STATE_HOME", "").strip()
    if xdg_state:
        return Path(xdg_state).expanduser()
    return Path.home() / ".local" / "state"


def get_app_data_root() -> Path:
    """Return the user-owned root for settings, caches, and diagnostics."""
    return _user_local_root() / APP_ID


def get_config_root() -> Path:
    """Return the user-owned configuration directory."""
    explicit = os.environ.get(ENV_CONFIG_ROOT, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return get_app_data_root() / "config"


def get_binary_config_path() -> Path:
    return get_config_root() / "enhance-binaries.json"


def get_default_temp_root() -> Path:
    """Return the user-owned default temp/cache directory."""
    if os.name == "nt":
        return get_app_data_root() / "temp"
    return Path(tempfile.gettempdir()) / APP_ID


def get_default_output_root() -> Path:
    """Return the default output directory without touching the source tree."""
    explicit = os.environ.get(ENV_OUTPUT_ROOT, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    if WORKSPACE_ROOT != _LEGACY_WORKSPACE_ROOT:
        # Backward-compatible test injection.  A normal installation never
        # changes WORKSPACE_ROOT, so it cannot redirect user output.
        return WORKSPACE_ROOT
    return Path.home() / "Videos" / APP_ID


def validate_chunk_seconds(n: int) -> int:
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError(f"chunk_seconds must be int, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"chunk_seconds must be >= 1, got {n!r}")
    return n


def validate_output_path(output_path: Path) -> Path:
    try:
        return output_path.resolve()
    except OSError as e:
        raise ValueError(f"cannot resolve output path: {output_path}") from e


__all__ = [
    "PROJECT_ROOT",
    "WORKSPACE_ROOT",
    "APP_ID",
    "ENV_APP_DATA_ROOT",
    "ENV_CONFIG_ROOT",
    "ENV_OUTPUT_ROOT",
    "get_app_data_root",
    "get_config_root",
    "get_binary_config_path",
    "get_default_temp_root",
    "get_default_output_root",
    "validate_chunk_seconds",
    "validate_output_path",
    "is_output_directory",
    "make_unique_output_path",
    "DEFAULT_CHUNK_SECONDS",
    "DEFAULT_MAX_WIDTH",
    "DEFAULT_MAX_DURATION_SEC",
    "DEFAULT_MAX_TEMP_GIB",
    "DEFAULT_TEMP_ROOT",
    "ENV_TEMP_ROOT",
    "DEFAULT_ENCODE_CRF",
    "DEFAULT_ENCODE_PRESET",
    "DEFAULT_TARGET_WIDTH",
    "DEFAULT_SAMPLE_COUNT",
    "DEFAULT_REALESRGAN_MODEL",
    "DEFAULT_REALESRGAN_SCALE",
    "DEFAULT_RIFE_MODEL",
    "FACE_RESTORE_BACKENDS",
    "DEFAULT_FACE_RESTORE_BACKEND",
    "DEFAULT_FACE_RESTORE_FIDELITY",
    "DEFAULT_FACE_RESTORE_IDLE_TIMEOUT_SEC",
    "DEFAULT_FACE_RESTORE_DETECTION_MODEL",
    "FACE_RESTORE_MASK_MODES",
    "DEFAULT_FACE_RESTORE_BLEND",
    "DEFAULT_FACE_RESTORE_MASK",
    "DEFAULT_FACE_RESTORE_FEATHER",
    "DEFAULT_FACE_RESTORE_MASK_DILATE",
    "DEFAULT_FACE_RESTORE_MASK_THRESHOLD",
    "DEFAULT_FACE_RESTORE_MATCH_GRAIN",
    "DEFAULT_FACE_RESTORE_SOFTEN",
    "DEFAULT_FACE_RESTORE_GLOBAL_SMOOTH",
    "CODEFORMER_REQUIRED_MODEL_KEYS",
    "CODEFORMER_STOCK_MODEL_FILENAMES",
    "REALESRGAN_MODEL_KIND",
    "OPTIONAL_REALESRGAN_MODELS",
    "ALLOWED_RIFE_MODELS",
    "RIFE_MODELS_WITH_CUSTOM_N",
    "realesrgan_param_files",
    "validate_realesrgan_model",
    "resolve_rife_model_dir",
    "validate_rife_model",
    "STABILIZE_BACKENDS",
    "DEFAULT_STABILIZE_BACKEND",
    "RESOLUTION_PRESETS",
    "RESOLUTION_KEEP_SOURCE",
    "UPSCALE_SCALE_POLICIES",
    "DEFAULT_UPSCALE_SCALE_POLICY",
    "CLAMP_FLAGS",
    "DEFAULT_CLAMP_FLAGS",
    "FPS_PRESETS",
    "FPS_KEEP_SOURCE",
    "DEFAULT_SHARPEN_AMOUNT",
    "DEFAULT_GRAIN_AMOUNT",
    "DEFAULT_DENOISE_STRENGTH",
    "DEFAULT_DEBAND_RANGE",
    "DEFAULT_FLICKER_SIZE",
    "DEFAULT_COLOR_BRIGHTNESS",
    "DEFAULT_COLOR_CONTRAST",
    "DEFAULT_COLOR_SATURATION",
    "T_SHAKE",
    "T_FLICKER",
    "T_NOISE",
    "T_BANDING",
    "T_BLUR",
    "T_LUMA_DARK",
    "T_LUMA_BRIGHT",
    "T_LUMA_CLIPPING",
    "T_SATURATION_LOW",
    "T_FPS_LOW",
    "RIFE_RECOMMENDED_VERSION",
    "RIFE_KNOWN_SHA256",
    "STAGE_KEYS",
    "STAGE_LABELS_JA",
    "STAGE_BASELINE_FRAME_COST_SEC",
    "STAGE_BASELINE_RESOLUTION_PX",
    "ENV_RUNTIME_SCALE",
    "BINARY_NAMES",
    "DEFAULT_ALLOWED_BINARY_ROOTS",
    "LOCAL_BINARY_CONFIG_PATH",
    "ENV_BINARY_ROOTS",
    "ENV_BINARY_EXPECTED_SHA256",
    "derive_output_path",
    "derive_temp_dir",
    "compute_sha256",
    "load_binary_policy",
    "BinaryPolicy",
]

DEFAULT_CHUNK_SECONDS = 10

DEFAULT_MAX_WIDTH = 7680
DEFAULT_MAX_DURATION_SEC = 7200
DEFAULT_MAX_TEMP_GIB = 100
DEFAULT_TEMP_ROOT = get_default_temp_root()
ENV_TEMP_ROOT = "VIDEO_ENHANCE_TEMP_ROOT"

DEFAULT_ENCODE_CRF = 22
DEFAULT_ENCODE_PRESET = "slow"
DEFAULT_TARGET_WIDTH = 1920
DEFAULT_SAMPLE_COUNT = 24

DEFAULT_REALESRGAN_MODEL = "realesr-animevideov3"
DEFAULT_REALESRGAN_SCALE = 2

# Real-ESRGAN-ncnn-vulkan 20220424 同梱モデル + OPTIONAL (利用者自前配置):
#   "scaled": -s {scale} の値に応じて `<name>-x{scale}` 接尾辞のファイルを参照
#   "fixed":  scale 固定 (基本 4)。`<name>` を直接参照
REALESRGAN_MODEL_KIND: dict[str, tuple[str, tuple[int, ...]]] = {
    "realesr-animevideov3": ("scaled", (2, 3, 4)),
    "realesrgan-x4plus": ("fixed", (4,)),
    "realesrgan-x4plus-anime": ("fixed", (4,)),
    # OPTIONAL: Real-ESRGAN 本体 (PyTorch) の汎用実写軽量モデル。
    # ncnn-vulkan 20220424 同梱には含まれないため、利用者が自前で配置する必要あり。
    "realesr-general-x4v3": ("fixed", (4,)),
}
OPTIONAL_REALESRGAN_MODELS = ("realesr-general-x4v3",)

# RIFE-ncnn-vulkan: `-n` (custom numframe) は v4 系モデルでしか使えないため、
# v4 系を明示的に既定とする。20221029 リリースには rife-v4 / rife-v4.6 が同梱。
# 既定は品質の高い rife-v4.6。
DEFAULT_RIFE_MODEL = "rife-v4.6"

# AI 顔復元は optional / default OFF。video-enhance-lite 本体には PyTorch 依存を入れず、
# 利用者が別途構築した CodeFormer 環境だけを明示設定で呼び出す。
FACE_RESTORE_BACKENDS = ("codeformer-local",)
DEFAULT_FACE_RESTORE_BACKEND = "codeformer-local"
DEFAULT_FACE_RESTORE_FIDELITY = 0.8
DEFAULT_FACE_RESTORE_IDLE_TIMEOUT_SEC = 3000
DEFAULT_FACE_RESTORE_DETECTION_MODEL = "retinaface_resnet50"
FACE_RESTORE_MASK_MODES = ("none", "diff")
DEFAULT_FACE_RESTORE_BLEND = 1.0
DEFAULT_FACE_RESTORE_MASK = "none"
DEFAULT_FACE_RESTORE_FEATHER = 24
DEFAULT_FACE_RESTORE_MASK_DILATE = 16
DEFAULT_FACE_RESTORE_MASK_THRESHOLD = 8
DEFAULT_FACE_RESTORE_MATCH_GRAIN = 0.0
DEFAULT_FACE_RESTORE_SOFTEN = 0.0
DEFAULT_FACE_RESTORE_GLOBAL_SMOOTH = 0.0
CODEFORMER_REQUIRED_MODEL_KEYS = (
    "codeformer",
    "facelib_detection",
    "facelib_parsing",
)
CODEFORMER_STOCK_MODEL_FILENAMES = {
    "codeformer": "codeformer.pth",
    "facelib_detection": "detection_Resnet50_Final.pth",
    "facelib_parsing": "parsing_parsenet.pth",
}

# 公式 20221029 リリース同梱モデル + 一部第三者ビルドで追加されるもの
ALLOWED_RIFE_MODELS = (
    "rife-v2.3",
    "rife-v2.4",
    "rife-v3.0",
    "rife-v3.1",
    "rife-v4",
    "rife-v4.6",
)

# v4 系のみ -n (custom numframe) と任意整数倍補間対応。
# v2/v3 系を指定したら ratio==2 のみ許可する。
RIFE_MODELS_WITH_CUSTOM_N = ("rife-v4", "rife-v4.6")

# Stabilize backend 選択 (TODO.md V1-1):
#   - "vidstab": ffmpeg vidstabdetect/vidstabtransform (旧 ffmpeg ビルドで動く)
#   - "opencv":  stabilize_cv.py の cv2 ベース実装 (ffmpeg N-120725+ で必要)
#   - "auto":    ffmpeg -filters の vidstab 有無で自動分岐 (default)
STABILIZE_BACKENDS = ("auto", "vidstab", "opencv")
DEFAULT_STABILIZE_BACKEND = "auto"

# 「現状維持」センチネル: 入力解像度をそのまま target_width に使う (= upscale を行わない)。
# `recommend()` 内で meta.width に解決される。auto (None) と混同しないための専用値。
# 幅 0 は物理的にあり得ないため安全なセンチネルとして利用する。
RESOLUTION_KEEP_SOURCE = 0

# 解像度プリセット: target_width のみ (高さは入力 AR を保持)。
#   - auto (None): 現行ロジック (DEFAULT_TARGET_WIDTH=1920 を基準)。入力が 1920 未満なら拡大する。
#   - keep (RESOLUTION_KEEP_SOURCE): 入力解像度のまま。upscale は常に非推奨 (データ量増を抑える)。
RESOLUTION_PRESETS: dict[str, int | None] = {
    "auto": None,
    "keep": RESOLUTION_KEEP_SOURCE,
    "fullhd": 1920,
    "wqhd": 2560,
    "4k": 3840,
}

# PR-G2: Real-ESRGAN の scale 選択ポリシー (opt-in)。
#   - "cover"   (既定): 必要倍率以上の最小 scale を選ぶ (over-scale 方向)。従来挙動・不変。
#   - "nearest": target_width に最も近い中間幅になる scale を選ぶ (上方/下方 lanczos を許容)。
#                upscale 自体が軽くなり得るが、target を下回ると上方 lanczos で精細感が落ちる。
UPSCALE_SCALE_POLICIES = ("cover", "nearest")
DEFAULT_UPSCALE_SCALE_POLICY = "cover"

# PR-G2: resolution_clamp の rescale アルゴリズム (opt-in)。
#   - "lanczos"  (既定): 従来挙動・不変 (精細感重視)。
#   - "bicubic" / "bilinear": 高速だが精細感が落ちる。runtime_estimator は当面 lanczos
#                baseline を保守的に据え置く (楽観側への倒れを避ける)。
CLAMP_FLAGS = ("lanczos", "bicubic", "bilinear")
DEFAULT_CLAMP_FLAGS = "lanczos"

# 「現状維持」センチネル: 入力 fps をそのまま target に使う (= フレーム補間しない)。
# `recommend()` 内で source_fps に解決される。auto (None) と混同しないための専用値。
# fps 0 は物理的にあり得ないため安全なセンチネルとして利用する。
FPS_KEEP_SOURCE = 0.0

# FPS プリセット: target_fps (float)。
#   - auto (None): 現行ロジック (source_fps < 30 なら ×2 推奨)。
#   - keep (FPS_KEEP_SOURCE): 入力 fps のまま。interpolate は常に非推奨 (フレーム数を増やさない)。
FPS_PRESETS: dict[str, float | None] = {
    "auto": None,
    "keep": FPS_KEEP_SOURCE,
    "48": 48.0,
    "60": 60.0,
    "120": 120.0,
}

DEFAULT_SHARPEN_AMOUNT = 0.6
DEFAULT_GRAIN_AMOUNT = 4
DEFAULT_DENOISE_STRENGTH = "hqdn3d=1.5:1.5:6:6"
DEFAULT_DEBAND_RANGE = 16
DEFAULT_FLICKER_SIZE = 5
DEFAULT_COLOR_BRIGHTNESS = 0.0
DEFAULT_COLOR_CONTRAST = 1.0
DEFAULT_COLOR_SATURATION = 1.0

# Heuristic thresholds (素材依存。利用者は判定不能でも手動 ON 可能)
T_SHAKE = 8.0
T_FLICKER = 6.0
T_NOISE = 18.0
T_BANDING = 0.20
T_BLUR = 80.0
T_LUMA_DARK = 60.0
T_LUMA_BRIGHT = 200.0
T_LUMA_CLIPPING = 0.02
T_SATURATION_LOW = 30.0
T_FPS_LOW = 30.0

# 2026 年現在の最新公式リリース
RIFE_RECOMMENDED_VERSION = "20221029"

# 配布物名 → 期待 SHA-256 (小文字 hex)。
#
# 注意: 本辞書は後方互換のため残しているが、ハードニング後は
# `load_binary_policy().expected_sha256["rife-ncnn-vulkan"]` を
# 優先して参照すること。`verify_rife_sha256` は両方を見る。
RIFE_KNOWN_SHA256: dict[str, str] = {
    # "rife-ncnn-vulkan-20221029-windows.zip": "<your-computed-sha256-here>",
}

STAGE_KEYS = (
    "deinterlace",
    "stabilize",
    "deflicker",
    "denoise",
    "debanding",
    "color_correction",
    "sharpen",
    "upscale",
    "face_restore",
    "interpolate",
    "grain",
    "encode",
)

STAGE_LABELS_JA = {
    "deinterlace": "デインターレース",
    "stabilize": "手ぶれ補正",
    "deflicker": "フリッカー除去",
    "denoise": "ノイズ・圧縮ノイズ除去",
    "debanding": "デバンディング",
    "color_correction": "色補正・露出補正",
    "sharpen": "ブラー軽減・シャープ化",
    "upscale": "アップスケール",
    "face_restore": "AI顔復元",
    "interpolate": "フレーム補間",
    "grain": "粒状感の調整",
    "encode": "最終エンコード",
}


# ============================================================
# 推定実行時間 (runtime_estimator) 用の baseline (UI 表示専用)
# ============================================================
#
# 基準: 1920x1080 (Full HD) の 1 frame を処理する所要秒数 (デスクトップ GPU 想定)
# 出荷時 default は GTX 1660 / RTX 3060 級の中央値で、利用者環境による誤差は ±50% を許容する
# (UI 表示の目安であって、--max-temp-gib のような hard gate ではない)
#
# 注: 以下の baseline は perf-and-feature プラン PR1-PR7 適用後の実態とは ±50% ズレる前提:
#   - PR2 (--stabilize-backend opencv): stabilize が pre_filter chain から外れ OpenCV 単体実行になるため、
#     pre_filter baseline は過大計上気味。stabilize_detect は呼ばれない
#   - PR6 (mux 融合): post_filter コストが mux 1 段に統合済みのため、post_filter + mux の二重計上気味 (過大方向)
#   - PR7 (--hw-encode {nvenc,qsv,amf}): mux baseline 0.0120s/frame は libx264 想定で、HW encoder では 10x 以上速い
#     ため過大推定
# いずれも PR-D の 3 チャンク以降の実測線形外挿で自動補正されるため UI 表示としては許容範囲。
# 厳密な補正は将来のキャリブレーション機能 (PR-E) で扱う。
STAGE_BASELINE_FRAME_COST_SEC: dict[str, float] = {
    "extract": 0.0020,  # PNG decode (ffmpeg image2)
    "stabilize_detect": 0.0050,  # vidstabdetect (libavfilter)
    "pre_filter": 0.0060,  # deinterlace + stabilize(vidstabtransform) + deflicker + denoise + debanding
    "upscale_scale2": 0.30,  # Real-ESRGAN ncnn-vulkan scale=2
    "upscale_scale3": 0.55,
    "upscale_scale4": 0.95,
    "resolution_clamp": 0.0040,  # lanczos rescale (ffmpeg vf)
    "face_restore": 0.85,  # CodeFormer PyTorch local backend (低信頼 baseline)
    "interpolate": 0.18,  # RIFE ncnn-vulkan / 中間 1 枚生成
    "post_filter": 0.0050,  # color_correction + sharpen + grain
    "mux": 0.0120,  # libx264 preset=slow / crf=22
}

STAGE_BASELINE_RESOLUTION_PX: int = 1920 * 1080  # ピクセル比補正の分母

# 環境変数で出荷時 baseline を一括スケールできる (=利用者の体感補正)
ENV_RUNTIME_SCALE = "VIDEO_ENHANCE_RUNTIME_SCALE"  # 例: "1.4" -> 全体 1.4 倍


# ============================================================
# 外部バイナリ allowlist / expected hash 設定 (deps.py 検証層用)
# ============================================================
#
# 解決済み実行ファイルが信頼できる場所にあり、かつ既知ハッシュと
# 一致することを `deps.py` の各 `check_*` で確認する。
# 出荷時の許可ディレクトリは意図的に最小限。マシン固有の追加は
# 以下のいずれかで行う:
#   1. `get_binary_config_path()` (user-local configuration)
#   2. `VIDEO_ENHANCE_*` 環境変数 (override が local config より優先)
#
# 設定ファイル例 (`%LOCALAPPDATA%\video-enhance-lite\config\enhance-binaries.json`):
#   {
#     "allowed_binary_roots": {
#       "ffmpeg": ["C:\\ffmpeg\\bin"],
#       "ffprobe": ["C:\\ffmpeg\\bin"],
#       "realesrgan-ncnn-vulkan": [
#         "C:\\tools\\realesrgan-ncnn-vulkan-20220424"
#       ],
#       "rife-ncnn-vulkan": ["C:\\tools\\rife-ncnn-vulkan-20221029"]
#     },
#     "expected_sha256": {
#       "realesrgan-ncnn-vulkan": ["<sha256-hex>"],
#       "rife-ncnn-vulkan": [],
#       "ffmpeg": [],
#       "ffprobe": []
#     }
#   }

BINARY_NAMES: tuple[str, ...] = (
    "ffmpeg",
    "ffprobe",
    "realesrgan-ncnn-vulkan",
    "rife-ncnn-vulkan",
)

# 出荷時 default の許可ディレクトリ。空集合のキーは
# `local config + env override` のみで補う。ソースツリー隣接の外部
# バイナリは、wheel／source ZIPの境界を曖昧にするため暗黙採用しない。
DEFAULT_ALLOWED_BINARY_ROOTS: dict[str, tuple[Path, ...]] = {
    "ffmpeg": (),
    "ffprobe": (),
    "realesrgan-ncnn-vulkan": (),
    "rife-ncnn-vulkan": (),
}

LOCAL_BINARY_CONFIG_PATH: Path = get_binary_config_path()

# 環境変数の対応表 (allowlist root 用)。値はパス区切り (Windows: ';')。
ENV_BINARY_ROOTS: dict[str, str] = {
    "ffmpeg": "VIDEO_ENHANCE_FFMPEG_ROOTS",
    "ffprobe": "VIDEO_ENHANCE_FFPROBE_ROOTS",
    "realesrgan-ncnn-vulkan": "VIDEO_ENHANCE_REALESRGAN_ROOTS",
    "rife-ncnn-vulkan": "VIDEO_ENHANCE_RIFE_ROOTS",
}

# 期待 SHA-256 をまとめて override する環境変数 (JSON)。
# 例: VIDEO_ENHANCE_EXPECTED_SHA256_JSON='{"rife-ncnn-vulkan":["abc..."]}'
ENV_BINARY_EXPECTED_SHA256: str = "VIDEO_ENHANCE_EXPECTED_SHA256_JSON"


class BinaryPolicy:
    """Resolved binary allowlist + expected SHA-256 set.

    優先順:
        1. 環境変数 (VIDEO_ENHANCE_*)
        2. user-local config (`get_binary_config_path()`)
        3. `DEFAULT_ALLOWED_BINARY_ROOTS` / `RIFE_KNOWN_SHA256`

    上位ソースは下位ソースを「マージ」するのではなく**完全置換**する。
    （allowlist は意図せず広がると検証が骨抜きになるため）
    expected_sha256 についても同様。
    """

    __slots__ = ("allowed_binary_roots", "expected_sha256")

    def __init__(
        self,
        allowed_binary_roots: dict[str, tuple[Path, ...]],
        expected_sha256: dict[str, tuple[str, ...]],
    ) -> None:
        self.allowed_binary_roots = allowed_binary_roots
        self.expected_sha256 = expected_sha256

    def roots_for(self, name: str) -> tuple[Path, ...]:
        return self.allowed_binary_roots.get(name, ())

    def expected_for(self, name: str) -> tuple[str, ...]:
        return self.expected_sha256.get(name, ())


def _split_path_list(raw: str) -> list[str]:
    """Split a PATH-like environment variable value.

    Windows uses ';'. POSIX uses ':' (but on Windows we keep ';' to avoid
    breaking drive-letter paths like 'C:\\tools').
    """
    # A semicolon is unambiguous for Windows paths (including drive letters)
    # and is also accepted in fixtures running on another host.
    sep = ";" if ";" in raw else os.pathsep
    return [s.strip() for s in raw.split(sep) if s.strip()]


def _coerce_paths(raw: object) -> tuple[Path, ...]:
    """Convert a JSON value (string list / single string / None) into a tuple of Paths."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (Path(raw),) if raw else ()
    if isinstance(raw, (list, tuple)):
        out: list[Path] = []
        for item in raw:
            if isinstance(item, str) and item:
                out.append(Path(item))
        return tuple(out)
    return ()


def _coerce_hash_list(raw: object) -> tuple[str, ...]:
    """Convert a JSON value into a tuple of lowercase hex strings."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw.lower(),) if raw else ()
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str) and item:
                out.append(item.lower())
        return tuple(out)
    if isinstance(raw, dict):
        # legacy RIFE_KNOWN_SHA256 形式 {"<asset>": "<hash>"} 互換
        out = []
        for v in raw.values():
            if isinstance(v, str) and v:
                out.append(v.lower())
        return tuple(out)
    return ()


def _load_local_config(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _parse_expected_sha256_json(raw: str) -> dict[str, tuple[str, ...]] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    out: dict[str, tuple[str, ...]] = {}
    for name, value in parsed.items():
        if not isinstance(name, str):
            continue
        out[name] = _coerce_hash_list(value)
    return out


def load_binary_policy(
    *,
    local_config_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> BinaryPolicy:
    """Build a `BinaryPolicy` from defaults + local config + env overrides.

    Parameters are injectable for testing. Production callers should pass nothing.
    """
    if local_config_path is None:
        local_config_path = get_binary_config_path()
    if env is None:
        env = dict(os.environ)

    # 1. defaults
    allowed_roots: dict[str, tuple[Path, ...]] = {
        name: tuple(DEFAULT_ALLOWED_BINARY_ROOTS.get(name, ())) for name in BINARY_NAMES
    }
    expected: dict[str, tuple[str, ...]] = {name: () for name in BINARY_NAMES}
    # 後方互換: RIFE_KNOWN_SHA256 (dict) の value を rife の expected として吸収
    expected["rife-ncnn-vulkan"] = _coerce_hash_list(RIFE_KNOWN_SHA256)

    # 2. local config (上位ソースが override する。マージはしない)
    local = _load_local_config(local_config_path)
    if local is not None:
        roots_section = local.get("allowed_binary_roots")
        if isinstance(roots_section, dict):
            for name in BINARY_NAMES:
                if name in roots_section:
                    allowed_roots[name] = _coerce_paths(roots_section[name])
        sha_section = local.get("expected_sha256")
        if isinstance(sha_section, dict):
            for name in BINARY_NAMES:
                if name in sha_section:
                    expected[name] = _coerce_hash_list(sha_section[name])

    # 3. env overrides (最優先)
    for name, env_key in ENV_BINARY_ROOTS.items():
        raw = env.get(env_key)
        if raw is None:
            continue
        if not raw.strip():
            # 明示的に空指定 -> 許可ディレクトリ無し
            allowed_roots[name] = ()
        else:
            allowed_roots[name] = tuple(Path(p) for p in _split_path_list(raw))

    env_expected = _parse_expected_sha256_json(env.get(ENV_BINARY_EXPECTED_SHA256, ""))
    if env_expected is not None:
        for name, hashes in env_expected.items():
            if name in expected:
                expected[name] = hashes

    return BinaryPolicy(allowed_binary_roots=allowed_roots, expected_sha256=expected)


def is_output_directory(path: Path) -> bool:
    try:
        return path.exists() and path.is_dir()
    except OSError:
        return False


def make_unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}~({index}){suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def derive_output_path(input_path: Path, explicit: Path | None, suffix: str = "_enhanced") -> Path:
    if explicit is not None:
        if is_output_directory(explicit):
            return explicit / f"{input_path.stem}{suffix}{input_path.suffix}"
        return explicit
    return get_default_output_root() / f"{input_path.stem}{suffix}{input_path.suffix}"


def _hash_input_identity(input_path: Path) -> str:
    """Short stable hash of the input path (not the file content) for tmp dir uniqueness."""
    h = hashlib.sha256()
    try:
        h.update(str(input_path.resolve()).encode("utf-8", errors="replace"))
    except OSError:
        h.update(str(input_path).encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]


def derive_temp_dir(input_path: Path) -> Path:
    """User-owned temp parent (`DEFAULT_TEMP_ROOT/<hash>`)."""
    return DEFAULT_TEMP_ROOT / _hash_input_identity(input_path)


def realesrgan_param_files(name: str, scale: int) -> tuple[str, str]:
    """Return (param_filename, bin_filename) for Real-ESRGAN ncnn-vulkan model.

    scaled モデル ("realesr-animevideov3") は `<name>-x{scale}.param/bin`、
    fixed モデルは `<name>.param/bin` を返す。
    """
    if name not in REALESRGAN_MODEL_KIND:
        raise ValueError(f"unknown realesrgan model: {name!r}")
    kind, _ = REALESRGAN_MODEL_KIND[name]
    base = f"{name}-x{scale}" if kind == "scaled" else name
    return (f"{base}.param", f"{base}.bin")


def validate_realesrgan_model(models_dir: Path, name: str, scale: int) -> None:
    """upscale enabled の場合に限り、依存解決後に呼び出す。

    実存チェック (argparse の choices ではなく動的 validate)。エラーは呼び出し側で
    `MissingDependencyError` に変換する想定。
    """
    if name not in REALESRGAN_MODEL_KIND:
        raise ValueError(f"unknown realesrgan model: {name!r}")
    _, allowed = REALESRGAN_MODEL_KIND[name]
    if scale not in allowed:
        raise ValueError(f"realesrgan model {name!r} supports scale={allowed}, got {scale}")
    pf, bf = realesrgan_param_files(name, scale)
    missing = [f for f in (pf, bf) if not (models_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"realesrgan model files not found in {models_dir}: {missing}. "
            f"自前で配置するか --realesrgan-model を変更してください"
        )


def resolve_rife_model_dir(exe_dir: Path, name: str) -> Path:
    """RIFE モデル親ディレクトリを解決する。

    公式 20221029 配布は exe と同階層に `rife-v4.6/` 等のモデルディレクトリが
    **直接**並ぶ構造で、`models/` サブディレクトリは存在しない (実機確認済み)。
    旧想定の `<exe親>/models/<name>/` 配置も互換のため受け付ける。

    Returns: `validate_rife_model` に渡す models_dir (モデル親ディレクトリ)。
    Raises: FileNotFoundError (両候補に見つからない場合。候補パスを列挙)。
    """
    official = exe_dir / name / "flownet.param"
    if official.is_file():
        return exe_dir
    legacy = exe_dir / "models" / name / "flownet.param"
    if legacy.is_file():
        return exe_dir / "models"
    raise FileNotFoundError(
        f"rife model dir not found: {exe_dir / name} (公式配布構造) / "
        f"{exe_dir / 'models' / name} (旧想定) のどちらにも flownet.param がありません。"
        f"公式配布は exe と同階層にモデルディレクトリ (rife-v4.6 等) が並ぶ構造です。"
        f"利用可能なモデル名を確認してください"
    )


def validate_rife_model(models_dir: Path, name: str) -> None:
    """RIFE モデルの実存チェック (`<models_dir>/<name>/flownet.param`)。

    公式 20221029 配布は exe と同階層に `<name>/` ディレクトリが直接並ぶため、
    `models_dir` には `resolve_rife_model_dir(exe_dir, name)` の戻り値を渡すこと
    (exe 同階層 / 旧想定 `models/` の両対応)。interpolate enabled の場合に限り呼び出す。
    """
    model_dir = models_dir / name
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"rife model dir not found: {model_dir}. "
            f"利用可能なモデル名 (= models/ 配下のサブディレクトリ名) を確認してください"
        )
    if not (model_dir / "flownet.param").is_file():
        raise FileNotFoundError(f"rife model is missing flownet.param: {model_dir}")


def compute_sha256(path: Path, *, max_bytes: int | None = None) -> str | None:
    """ResolvedBinary.sha256 用。大きい binary はスキップ可能。失敗時は None."""
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if max_bytes is not None and size > max_bytes:
        return None
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()
