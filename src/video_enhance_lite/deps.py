from __future__ import annotations

import os
import json
import importlib.util
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class MissingDependencyError(Exception):
    def __init__(
        self,
        name: str,
        hint: str,
        *,
        capability: str | None = None,
        detected_path: Path | None = None,
        status: str = "unavailable",
        config_path: Path | None = None,
        remediation: tuple[str, ...] = (),
        official_url: str | None = None,
    ):
        super().__init__(f"missing dependency: {name}\n{hint}")
        self.name = name
        self.hint = hint
        self.capability = capability or name
        self.detected_path = detected_path
        self.status = status
        self.config_path = config_path
        self.remediation = tuple(remediation)
        self.official_url = official_url

    def as_dict(self) -> dict[str, object]:
        return {
            "dependency_id": self.name,
            "capability": self.capability,
            "status": self.status,
            "detected_path": str(self.detected_path) if self.detected_path else None,
            "configuration_location": str(self.config_path) if self.config_path else None,
            "remediation": list(self.remediation),
            "official_download_url": self.official_url,
            "detail": self.hint,
        }


def find_executable(name: str, extra_dirs: Iterable[Path] = ()) -> Path | None:
    candidates = [name]
    if sys.platform == "win32" and not name.lower().endswith(".exe"):
        candidates.append(name + ".exe")
    for d in extra_dirs:
        for c in candidates:
            p = Path(d) / c
            if p.exists() and os.access(p, os.X_OK if sys.platform != "win32" else os.F_OK):
                return p
    found = shutil.which(name)
    if found:
        return Path(found)
    return None


from . import config

_HIDE_WINDOW_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

__all__ = [
    "MissingDependencyError",
    "ResolvedBinary",
    "ResolvedCodeFormerLocal",
    "CapabilityDiagnostic",
    "diagnose_capabilities",
    "format_capability_report",
    "RifeVerification",
    "check_ffmpeg",
    "check_ffmpeg_with_vidstab",
    "check_ffprobe",
    "check_realesrgan",
    "check_rife",
    "check_codeformer_local",
    "check_hw_encoder",
    "format_resolved_binary",
    "verify_rife_sha256",
]


_FFMPEG_VIDSTAB_HINT = (
    "ffmpeg が見つからない、または libvidstab を含まないビルドです。\n"
    "Gyan FFmpeg essentials / full のどちらも libvidstab を含むため、\n"
    "`ffmpeg -filters` 出力に `vidstabdetect` を含むビルドを利用してください。\n"
    "(古いビルドや配布元によっては未収録の場合があります)\n"
    "推奨インストール: winget install --id=Gyan.FFmpeg -e"
)

_FFPROBE_HINT = (
    "ffprobe が PATH 上に見つかりません。\n推奨: winget install --id=Gyan.FFmpeg -e (ffmpeg に同梱)"
)

_REALESRGAN_HINT = (
    "realesrgan-ncnn-vulkan が PATH 上に見つかりません。\n"
    "ダウンロード: https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases\n"
    "展開後、ディレクトリを PATH に追加するか、PATH 上のディレクトリに\n"
    "`realesrgan-ncnn-vulkan.exe` を配置してください。"
)

_RIFE_HINT = (
    f"rife-ncnn-vulkan が PATH 上に見つかりません。\n"
    f"推奨バージョン: {config.RIFE_RECOMMENDED_VERSION}\n"
    f"ダウンロード: https://github.com/nihui/rife-ncnn-vulkan/releases\n"
    f"展開後、ディレクトリを PATH に追加するか、PATH 上のディレクトリに\n"
    f"`rife-ncnn-vulkan.exe` を配置してください。"
)

_CODEFORMER_HINT = (
    "CodeFormer local backend が未設定または不完全です。\n"
    f"{config.get_binary_config_path()} (ローカル専用) に roots.codeformer-local、"
    "executables.codeformer-python、models.codeformer/facelib_detection/facelib_parsing "
    "を明示してください。クラウド API や自動ダウンロードは使用しません。"
)


@dataclass(frozen=True)
class ResolvedBinary:
    path: Path
    version: str | None
    sha256: str | None


@dataclass(frozen=True)
class ResolvedCodeFormerLocal:
    repo_root: Path
    python: Path
    script: Path
    models: dict[str, Path]
    model_sha256: dict[str, str | None]


@dataclass(frozen=True)
class CapabilityDiagnostic:
    capability: str
    dependency_id: str
    status: str
    detected_path: Path | None = None
    version: str | None = None
    detail: str = ""
    configuration_location: Path | None = None
    official_download_url: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "capability": self.capability,
            "dependency_id": self.dependency_id,
            "status": self.status,
            "detected_path": str(self.detected_path) if self.detected_path else None,
            "version": self.version,
            "detail": self.detail,
            "configuration_location": (
                str(self.configuration_location) if self.configuration_location else None
            ),
            "official_download_url": self.official_download_url,
        }


_VERSION_HASH_MAX_BYTES = 200 * 1024 * 1024  # 200 MiB 以下のみ sha256 取得


def _run_capture(argv: list[str]) -> str | None:
    """subprocess.run wrapper. shell=False fixed, returns combined output first line or None."""
    try:
        proc = subprocess.run(  # noqa: S603 - shell=False, list argv
            argv,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            creationflags=_HIDE_WINDOW_FLAGS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = (proc.stdout or "") + (proc.stderr or "")
    if not out:
        return None
    return out


def _first_line(text: str | None) -> str | None:
    if not text:
        return None
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return None


# ncnn-vulkan 系バイナリは `-version` 非対応 (usage を出力するだけ) のため、
# version 取得をスキップする (旧実装は usage の断片 "-x enable spatial tta mode" 等を
# version として表示していた)。
_NO_VERSION_FLAG_MARKERS = ("realesrgan", "rife")


def _get_version(path: Path) -> str | None:
    name = path.name.lower()
    if any(m in name for m in _NO_VERSION_FLAG_MARKERS):
        return None
    out = _run_capture([str(path), "-version"])
    return _first_line(out)


def _get_filters_output(path: Path) -> str | None:
    return _run_capture([str(path), "-hide_banner", "-filters"])


def _has_vidstab_filters(filters_text: str | None) -> bool:
    if not filters_text:
        return False
    has_detect = re.search(r"\bvidstabdetect\b", filters_text) is not None
    has_transform = re.search(r"\bvidstabtransform\b", filters_text) is not None
    return has_detect and has_transform


def _build_resolved(path: Path) -> ResolvedBinary:
    return ResolvedBinary(
        path=path,
        version=_get_version(path),
        sha256=config.compute_sha256(path, max_bytes=_VERSION_HASH_MAX_BYTES),
    )


def _resolve_safely(path: Path) -> Path:
    """Best-effort resolve. Falls back to the original path on OSError."""
    try:
        return path.resolve()
    except OSError:
        return path


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attrs = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attrs & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT


def _has_reparse_component(path: Path) -> Path | None:
    """Return the first symlink/junction component on the binary path."""
    current = path
    while current != current.parent:
        if _is_reparse_point(current):
            return current
        current = current.parent
    return None


def _check_binary_shape(name: str, path: Path) -> None:
    if not path.is_file():
        raise MissingDependencyError(
            name,
            f"解決先が通常ファイルではありません: {path}",
            status="misconfigured",
            detected_path=path,
            config_path=config.get_binary_config_path(),
        )
    reparse = _has_reparse_component(path)
    if reparse is not None:
        raise MissingDependencyError(
            name,
            f"実行ファイルの経路に reparse point が含まれています: {reparse}",
            status="misconfigured",
            detected_path=path,
            config_path=config.get_binary_config_path(),
        )
    expected_names = {name.lower(), f"{name.lower()}.exe"}
    if path.name.lower() not in expected_names:
        expected = " / ".join(sorted(expected_names))
        raise MissingDependencyError(
            name,
            f"実行ファイル名が期待値と異なります: {path.name} (期待: {expected})",
            status="misconfigured",
            detected_path=path,
            config_path=config.get_binary_config_path(),
        )


def _is_under_any(path: Path, roots: tuple[Path, ...]) -> bool:
    """True if `path` (already resolved) is inside any of the resolved `roots`."""
    if not roots:
        return False
    for root in roots:
        resolved_root = _resolve_safely(root)
        try:
            path.relative_to(resolved_root)
            return True
        except ValueError:
            continue
    return False


def _check_allowed_root(
    name: str,
    path: Path,
    policy: "config.BinaryPolicy",
) -> None:
    """Raise MissingDependencyError if `path` is not under any allowed root."""
    roots = policy.roots_for(name)
    resolved = _resolve_safely(path)
    if not roots:
        raise MissingDependencyError(
            name,
            (
                f"{name} の許可ディレクトリが未設定です。解決先: {resolved}\n"
                f"許可ディレクトリを {config.get_binary_config_path()} (ローカル専用) または\n"
                f"環境変数 {config.ENV_BINARY_ROOTS.get(name, '<未定義>')} で明示してください。"
            ),
            status="misconfigured",
            detected_path=resolved,
            config_path=config.get_binary_config_path(),
        )
    if _is_under_any(resolved, roots):
        return
    root_list = "\n    - ".join(str(r) for r in roots)
    raise MissingDependencyError(
        name,
        (
            f"{name} の解決先 {resolved} は許可ディレクトリ外です。\n"
            f"許可ディレクトリ:\n    - {root_list}\n"
            f"許可ディレクトリは {config.get_binary_config_path()} (ローカル専用) または\n"
            f"環境変数 {config.ENV_BINARY_ROOTS.get(name, '<未定義>')} で上書きできます。"
        ),
        status="misconfigured",
        detected_path=resolved,
        config_path=config.get_binary_config_path(),
    )


def _check_expected_hash(
    name: str,
    sha256: str | None,
    policy: "config.BinaryPolicy",
) -> None:
    """Raise MissingDependencyError on hash mismatch.

    Behavior:
      - policy has no expected hashes for `name`         -> skip (informational only)
      - policy has expected hashes and `sha256` is None  -> raise (fail-close; P2-5)
      - sha256 matches one of the expected entries       -> pass
      - sha256 mismatch                                  -> raise

    P2-5: 期待ハッシュが設定されているのに実測できない (200MiB 超 / 読込失敗) 場合、
    旧実装は黙ってスキップしていた (fail-open)。これでは巨大な改ざんバイナリが
    ハッシュ検証を素通りするため、検証不能は明示エラーで停止する。
    """
    expected = policy.expected_for(name)
    if not expected:
        return
    if sha256 is None:
        raise MissingDependencyError(
            name,
            (
                f"{name} の期待 SHA-256 が設定されていますが、実測できませんでした\n"
                f"(ファイルが {_VERSION_HASH_MAX_BYTES // (1024 * 1024)} MiB 超または読込失敗)。\n"
                f"真正性を確認できないため停止します。ファイルを確認のうえ、必要なら\n"
                f"{config.get_binary_config_path()} の expected_sha256.{name} を見直してください。"
            ),
        )
    sha_lower = sha256.lower()
    for exp in expected:
        if exp.lower() == sha_lower:
            return
    expected_list = "\n    - ".join(expected)
    raise MissingDependencyError(
        name,
        (
            f"{name} の SHA-256 が期待値と一致しません。\n"
            f"  実測 sha256: {sha_lower}\n"
            f"  期待値:\n    - {expected_list}\n"
            f"配布物の真正性を確認し、必要なら {config.get_binary_config_path()} の\n"
            f"`expected_sha256.{name}` または環境変数 {config.ENV_BINARY_EXPECTED_SHA256} を更新してください。"
        ),
    )


def _validate_resolved(
    name: str,
    path: Path,
    *,
    policy: "config.BinaryPolicy | None" = None,
) -> ResolvedBinary:
    """Resolve + allowlist + expected hash checks. Returns a `ResolvedBinary`."""
    if policy is None:
        policy = config.load_binary_policy()
    # Inspect the original path before resolving it.  Resolving first would
    # erase a symlink/junction component and make the boundary check observe
    # only the target path.
    _check_binary_shape(name, path)
    resolved = _resolve_safely(path)
    if resolved != path:
        _check_binary_shape(name, resolved)
    _check_allowed_root(name, resolved, policy)
    rb = _build_resolved(resolved)
    _check_expected_hash(name, rb.sha256, policy)
    return rb


def _load_json_config(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        raise MissingDependencyError("codeformer-local", _CODEFORMER_HINT) from e
    except json.JSONDecodeError as e:
        raise MissingDependencyError(
            "codeformer-local",
            f"{path} の JSON を読めません: {e}",
        ) from e
    if not isinstance(data, dict):
        raise MissingDependencyError("codeformer-local", f"{path} は JSON object にしてください。")
    return data


def _first_path_from_section(data: dict, section: str, key: str) -> Path | None:
    raw_section = data.get(section)
    if not isinstance(raw_section, dict):
        return None
    raw = raw_section.get(key)
    if isinstance(raw, str) and raw:
        return Path(raw)
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item:
                return Path(item)
    return None


def _model_entry_path_and_hash(data: dict, key: str) -> tuple[Path | None, str | None]:
    models = data.get("models")
    if not isinstance(models, dict):
        return None, None
    raw = models.get(key)
    if isinstance(raw, str) and raw:
        return Path(raw), None
    if isinstance(raw, dict):
        p = raw.get("path")
        sha = raw.get("sha256")
        return (
            Path(p) if isinstance(p, str) and p else None,
            sha.lower() if isinstance(sha, str) and sha else None,
        )
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item:
                return Path(item), None
            if isinstance(item, dict):
                p = item.get("path")
                sha = item.get("sha256")
                if isinstance(p, str) and p:
                    return Path(p), sha.lower() if isinstance(sha, str) and sha else None
    return None, None


def _require_file(name: str, path: Path | None) -> Path:
    if path is None:
        raise MissingDependencyError(
            "codeformer-local", f"{name} が設定されていません。\n{_CODEFORMER_HINT}"
        )
    resolved = _resolve_safely(path)
    if not resolved.is_file():
        raise MissingDependencyError("codeformer-local", f"{name} が見つかりません: {resolved}")
    return resolved


def _require_dir(name: str, path: Path | None) -> Path:
    if path is None:
        raise MissingDependencyError(
            "codeformer-local", f"{name} が設定されていません。\n{_CODEFORMER_HINT}"
        )
    resolved = _resolve_safely(path)
    if not resolved.is_dir():
        raise MissingDependencyError("codeformer-local", f"{name} が見つかりません: {resolved}")
    return resolved


def _check_model_hash(key: str, path: Path, expected_sha256: str | None) -> str | None:
    if not expected_sha256:
        return None
    actual = config.compute_sha256(path)
    if actual is None:
        raise MissingDependencyError(
            "codeformer-local",
            f"{key} の SHA-256 を計算できません: {path}",
        )
    if actual.lower() != expected_sha256.lower():
        raise MissingDependencyError(
            "codeformer-local",
            (
                f"{key} の SHA-256 が期待値と一致しません。\n"
                f"  実測 sha256: {actual.lower()}\n"
                f"  期待 sha256: {expected_sha256.lower()}"
            ),
        )
    return actual.lower()


def _check_codeformer_stock_cli_model_path(repo_root: Path, key: str, path: Path) -> None:
    """Fail fast when the configured weight is not usable by stock inference_codeformer.py."""
    try:
        path.relative_to(repo_root)
    except ValueError as e:
        raise MissingDependencyError(
            "codeformer-local",
            (
                f"models.{key} は CodeFormer repo 配下に配置してください: {path}\n"
                "stock inference_codeformer.py は任意の重みパスを argv で受け取らないため、"
                "repo 外の重みは初期 backend では使用できません。"
            ),
        ) from e
    expected_name = config.CODEFORMER_STOCK_MODEL_FILENAMES.get(key)
    if expected_name and path.name != expected_name:
        raise MissingDependencyError(
            "codeformer-local",
            (
                f"models.{key} のファイル名が stock CLI 想定と異なります: {path.name}\n"
                f"期待: {expected_name}"
            ),
        )


def check_codeformer_local(
    *,
    local_config_path: Path | None = None,
) -> ResolvedCodeFormerLocal:
    """Resolve a local CodeFormer backend from explicit local config only.

    No PATH search, cloud API, token, .env, or model auto-download is used.
    """
    if local_config_path is None:
        local_config_path = config.get_binary_config_path()
    data = _load_json_config(local_config_path)

    repo_root = _require_dir(
        "roots.codeformer-local",
        _first_path_from_section(data, "roots", "codeformer-local")
        or _first_path_from_section(data, "allowed_binary_roots", "codeformer-local"),
    )
    script = _require_file("CodeFormer inference script", repo_root / "inference_codeformer.py")
    python = _require_file(
        "executables.codeformer-python",
        _first_path_from_section(data, "executables", "codeformer-python"),
    )

    models: dict[str, Path] = {}
    model_sha256: dict[str, str | None] = {}
    for key in config.CODEFORMER_REQUIRED_MODEL_KEYS:
        model_path, expected_sha = _model_entry_path_and_hash(data, key)
        resolved_model = _require_file(f"models.{key}", model_path)
        _check_codeformer_stock_cli_model_path(repo_root, key, resolved_model)
        models[key] = resolved_model
        model_sha256[key] = _check_model_hash(key, resolved_model, expected_sha)

    return ResolvedCodeFormerLocal(
        repo_root=repo_root,
        python=python,
        script=script,
        models=models,
        model_sha256=model_sha256,
    )


_BINARY_DIAGNOSTICS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "ffmpeg": (
        "Encode/Decode",
        "https://ffmpeg.org/download.html",
        ("Install FFmpeg from the official source.", "Add its directory to the allowlist or PATH."),
    ),
    "ffprobe": (
        "Analyze",
        "https://ffmpeg.org/download.html",
        (
            "Install FFmpeg; ffprobe is distributed with it.",
            "Add its directory to the allowlist or PATH.",
        ),
    ),
    "realesrgan-ncnn-vulkan": (
        "Upscale",
        "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases",
        (
            "Download the official release without adding it to this source tree.",
            "Configure its directory explicitly.",
        ),
    ),
    "rife-ncnn-vulkan": (
        "Interpolate",
        "https://github.com/nihui/rife-ncnn-vulkan/releases",
        (
            "Download the official release without adding it to this source tree.",
            "Configure its directory explicitly.",
        ),
    ),
}


def _resolve_binary(name: str, hint: str) -> ResolvedBinary:
    policy = config.load_binary_policy()
    p = find_executable(name, policy.roots_for(name))
    capability, official_url, remediation = _BINARY_DIAGNOSTICS.get(name, (name, None, ()))
    if p is None:
        raise MissingDependencyError(
            name,
            hint,
            capability=capability,
            status="unavailable",
            config_path=config.get_binary_config_path(),
            remediation=remediation,
            official_url=official_url,
        )
    try:
        return _validate_resolved(name, p, policy=policy)
    except MissingDependencyError as exc:
        if exc.capability == exc.name:
            exc.capability = capability
        if exc.official_url is None:
            exc.official_url = official_url
        if not exc.remediation:
            exc.remediation = remediation
        raise


def check_ffmpeg() -> ResolvedBinary:
    """Resolve ffmpeg without vid.stab requirement (for stages that don't need it)."""
    return _resolve_binary("ffmpeg", _FFMPEG_VIDSTAB_HINT)


def check_ffmpeg_with_vidstab() -> ResolvedBinary:
    rb = _resolve_binary("ffmpeg", _FFMPEG_VIDSTAB_HINT)
    filters = _get_filters_output(rb.path)
    if not _has_vidstab_filters(filters):
        raise MissingDependencyError(
            "ffmpeg(vidstab)",
            _FFMPEG_VIDSTAB_HINT,
            capability="Stabilize",
            detected_path=rb.path,
            status="unsupported",
            config_path=config.get_binary_config_path(),
            remediation=("Use an FFmpeg build containing vidstabdetect and vidstabtransform.",),
            official_url="https://ffmpeg.org/download.html",
        )
    return rb


def format_resolved_binary(label: str, rb: ResolvedBinary) -> str:
    """Render a 1-line INFO log entry for a resolved binary.

    Includes resolved path / version / sha256 per plan's security requirement #3.
    Long sha256 is preserved in full (16-char prefix is too easy to forge).
    """
    version = rb.version if rb.version is not None else "unknown"
    if rb.sha256 is None:
        sha = "sha256=unavailable"
    else:
        sha = f"sha256={rb.sha256}"
    return f"[deps] {label}: path={rb.path} version={version!r} {sha}"


def check_ffprobe() -> ResolvedBinary:
    return _resolve_binary("ffprobe", _FFPROBE_HINT)


def check_realesrgan() -> ResolvedBinary:
    return _resolve_binary("realesrgan-ncnn-vulkan", _REALESRGAN_HINT)


def check_rife() -> ResolvedBinary:
    return _resolve_binary("rife-ncnn-vulkan", _RIFE_HINT)


def _diagnostic_from_error(
    capability: str,
    dependency_id: str,
    error: MissingDependencyError,
) -> CapabilityDiagnostic:
    return CapabilityDiagnostic(
        capability=capability,
        dependency_id=dependency_id,
        status=error.status,
        detected_path=error.detected_path,
        detail=error.hint,
        configuration_location=error.config_path or config.get_binary_config_path(),
        official_download_url=error.official_url,
    )


def diagnose_capabilities() -> tuple[CapabilityDiagnostic, ...]:
    """Probe optional capabilities without starting a video processing run."""
    diagnostics: list[CapabilityDiagnostic] = []
    checks = (
        ("Analyze", "ffprobe", check_ffprobe),
        ("Encode/Decode", "ffmpeg", check_ffmpeg),
        ("Upscale", "realesrgan-ncnn-vulkan", check_realesrgan),
        ("Interpolate", "rife-ncnn-vulkan", check_rife),
    )
    for capability, dependency_id, check in checks:
        try:
            resolved = check()
        except MissingDependencyError as exc:
            diagnostics.append(_diagnostic_from_error(capability, dependency_id, exc))
        else:
            diagnostics.append(
                CapabilityDiagnostic(
                    capability=capability,
                    dependency_id=dependency_id,
                    status="available",
                    detected_path=resolved.path,
                    version=resolved.version,
                    detail="binary resolved and allowlist/hash checks passed",
                    configuration_location=config.get_binary_config_path(),
                    official_download_url=_BINARY_DIAGNOSTICS[dependency_id][1],
                )
            )

    try:
        resolved = check_codeformer_local()
    except MissingDependencyError as exc:
        diagnostics.append(_diagnostic_from_error("Face restore", "codeformer-local", exc))
    else:
        diagnostics.append(
            CapabilityDiagnostic(
                capability="Face restore",
                dependency_id="codeformer-local",
                status="available",
                detected_path=resolved.script,
                detail="explicit local configuration and model checks passed",
                configuration_location=config.get_binary_config_path(),
                official_download_url="https://github.com/sczhou/CodeFormer",
            )
        )

    try:
        resolved = check_ffmpeg_with_vidstab()
    except MissingDependencyError as strict_error:
        try:
            ffmpeg = check_ffmpeg()
        except MissingDependencyError as ffmpeg_error:
            diagnostics.append(_diagnostic_from_error("Stabilize", "ffmpeg", ffmpeg_error))
        else:
            if importlib.util.find_spec("cv2") is not None:
                diagnostics.append(
                    CapabilityDiagnostic(
                        capability="Stabilize",
                        dependency_id="ffmpeg/opencv",
                        status="available",
                        detected_path=ffmpeg.path,
                        version=ffmpeg.version,
                        detail="FFmpeg vidstab is unavailable; OpenCV fallback is available",
                        configuration_location=config.get_binary_config_path(),
                        official_download_url="https://ffmpeg.org/download.html",
                    )
                )
            else:
                diagnostics.append(_diagnostic_from_error("Stabilize", "ffmpeg", strict_error))
    else:
        diagnostics.append(
            CapabilityDiagnostic(
                capability="Stabilize",
                dependency_id="ffmpeg(vidstab)",
                status="available",
                detected_path=resolved.path,
                version=resolved.version,
                detail="FFmpeg vidstabdetect/vidstabtransform are available",
                configuration_location=config.get_binary_config_path(),
                official_download_url="https://ffmpeg.org/download.html",
            )
        )
    return tuple(diagnostics)


def format_capability_report(diagnostics: Iterable[CapabilityDiagnostic] | None = None) -> str:
    rows = diagnostics if diagnostics is not None else diagnose_capabilities()
    lines = [
        "video-enhance-lite environment diagnostics",
        f"configuration: {config.get_binary_config_path()}",
        f"temp root: {config.DEFAULT_TEMP_ROOT}",
        f"output root: {config.get_default_output_root()}",
    ]
    for item in rows:
        location = f" path={item.detected_path}" if item.detected_path else ""
        version = f" version={item.version}" if item.version else ""
        lines.append(f"- {item.capability}: {item.status}{location}{version}")
        if item.detail:
            first_line = item.detail.splitlines()[0]
            lines.append(f"  {first_line}")
        if item.official_download_url:
            lines.append(f"  official: {item.official_download_url}")
    return "\n".join(lines)


def check_hw_encoder(ffmpeg: Path, encoder_name: str) -> bool:
    """`<ffmpeg> -hide_banner -encoders` の出力に `encoder_name` が含まれるかを返す。

    Note: ffmpeg のパスは依存解決後の ResolvedBinary.path を渡すこと。
    `-encoders` の表形式 ` V..... h264_nvenc           NVIDIA NVENC H.264 encoder` を想定。
    """
    try:
        proc = subprocess.run(  # noqa: S603
            [str(ffmpeg), "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
            shell=False,
            creationflags=_HIDE_WINDOW_FLAGS,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    out = proc.stdout or ""
    return f" {encoder_name} " in out


@dataclass(frozen=True)
class RifeVerification:
    """Result of comparing rb.sha256 with RIFE expected hashes.

    `level` follows logging severity convention:
      - "info" : matched a known hash OR table is empty (informational only)
      - "warn" : table is non-empty but rb.sha256 didn't match any known value
                 (note: when allowlist/hash is enforced via BinaryPolicy,
                 mismatch is already raised in check_rife; this branch only
                 fires for legacy RIFE_KNOWN_SHA256 entries that don't make it
                 into policy.expected_for("rife-ncnn-vulkan"))
    `matched_name` is the dict key (distribution filename) when matched, else None.
    """

    level: str
    message: str
    matched_name: str | None = None


def verify_rife_sha256(
    rb: ResolvedBinary,
    *,
    policy: "config.BinaryPolicy | None" = None,
) -> RifeVerification:
    """Compare rb.sha256 against config.RIFE_KNOWN_SHA256 (legacy info channel).

    The strict allowlist/hash check lives in `check_rife` via `BinaryPolicy`.
    This function remains for the GUI log channel: it returns a structured
    INFO/WARN message based on `config.RIFE_KNOWN_SHA256` (the user-extensible
    asset → hash table). It does NOT raise: enforcement is done elsewhere.

    `policy` はテスト注入用。None なら `load_binary_policy()` (local config / env) を読む。
    """
    if rb.sha256 is None:
        return RifeVerification(
            level="info",
            message="rife-ncnn-vulkan: sha256 unavailable (binary too large or unreadable)",
        )
    # BinaryPolicy (local config / env) に期待値が設定されている場合、check_rife の
    # _check_expected_hash で既に検証済み (不一致なら到達しない)。legacy テーブルが
    # 空でも「known-hash table empty」と誤解させるメッセージは出さない。
    if policy is None:
        policy = config.load_binary_policy()
    policy_expected = policy.expected_for("rife-ncnn-vulkan")
    if policy_expected:
        return RifeVerification(
            level="info",
            message=(
                f"rife-ncnn-vulkan sha256={rb.sha256} verified via BinaryPolicy "
                f"(local config / env)"
            ),
        )
    if not config.RIFE_KNOWN_SHA256:
        return RifeVerification(
            level="info",
            message=(
                f"rife-ncnn-vulkan sha256={rb.sha256} (known-hash table empty; "
                f"add expected value to config.RIFE_KNOWN_SHA256 or "
                f"{config.get_binary_config_path()} for verification)"
            ),
        )
    for name, expected in config.RIFE_KNOWN_SHA256.items():
        if expected.lower() == rb.sha256.lower():
            return RifeVerification(
                level="info",
                message=f"rife-ncnn-vulkan sha256 verified against known distribution: {name}",
                matched_name=name,
            )
    return RifeVerification(
        level="warn",
        message=(
            f"rife-ncnn-vulkan sha256={rb.sha256} does not match any entry in "
            f"config.RIFE_KNOWN_SHA256 ({list(config.RIFE_KNOWN_SHA256)}). "
            f"Distribution may be unofficial or tampered; processing continues."
        ),
    )
