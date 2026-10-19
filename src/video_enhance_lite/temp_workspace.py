from __future__ import annotations

import json
import os
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import config

_MARKER_NAME = ".video-enhance-lite-run.json"
_MARKER_TYPE = "video-enhance-lite-run"
_MARKER_VERSION = 1
_MARKER_KEYS = {"type", "schema_version", "run_id", "input_identity"}


class TempWorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class TempRunContext:
    temp_root_requested: Path
    temp_root: Path
    input_identity_dir: Path
    run_leaf: Path
    run_id: str

    @property
    def marker_path(self) -> Path:
        return self.run_leaf / _MARKER_NAME


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attrs = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    except OSError as exc:
        raise TempWorkspaceError(f"cannot inspect temp path: {path}: {exc}") from exc
    reparse_flag = getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attrs & reparse_flag)


def _assert_owned_descendants_not_reparse(context: TempRunContext) -> None:
    current = context.input_identity_dir
    while True:
        if _is_reparse_point(current):
            raise TempWorkspaceError(f"owned temp path is a reparse point: {current}")
        if current == context.run_leaf:
            break
        try:
            relative = context.run_leaf.relative_to(current)
        except ValueError as exc:
            raise TempWorkspaceError("run leaf is outside input identity") from exc
        next_name = relative.parts[0]
        current = current / next_name


def _assert_run_tree_not_reparse(context: TempRunContext) -> None:
    """Reject any reparse point below the owned run leaf before recursive cleanup."""
    pending = [context.run_leaf]
    while pending:
        current = pending.pop()
        if _is_reparse_point(current):
            raise TempWorkspaceError(f"owned temp path is a reparse point: {current}")
        try:
            if current.is_dir():
                pending.extend(current.iterdir())
        except OSError as exc:
            raise TempWorkspaceError(f"cannot inspect owned temp tree: {current}: {exc}") from exc


def _atomic_write_marker(context: TempRunContext) -> None:
    payload = {
        "type": _MARKER_TYPE,
        "schema_version": _MARKER_VERSION,
        "run_id": context.run_id,
        "input_identity": context.input_identity_dir.name,
    }
    temporary = context.run_leaf / f".{_MARKER_NAME}.{secrets.token_hex(8)}.tmp"
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, context.marker_path)


def create_temp_run_context(
    input_path: Path,
    temp_root: Path | None = None,
) -> TempRunContext:
    env_value = os.environ.get(config.ENV_TEMP_ROOT, "").strip()
    requested = (
        Path(temp_root)
        if temp_root is not None
        else (Path(env_value) if env_value else config.DEFAULT_TEMP_ROOT)
    )
    try:
        requested.mkdir(parents=True, exist_ok=True)
        resolved_root = requested.resolve(strict=True)
        identity_name = config.derive_temp_dir(input_path).name
        input_identity = resolved_root / identity_name
        runs_dir = input_identity / "runs"
        run_id = secrets.token_hex(16)
        run_leaf = runs_dir / run_id
        run_leaf.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise TempWorkspaceError(f"cannot create temp workspace under {requested}: {exc}") from exc
    context = TempRunContext(
        temp_root_requested=requested,
        temp_root=resolved_root,
        input_identity_dir=input_identity,
        run_leaf=run_leaf,
        run_id=run_id,
    )
    try:
        _assert_owned_descendants_not_reparse(context)
        _atomic_write_marker(context)
    except Exception:
        try:
            shutil.rmtree(run_leaf)
        except OSError:
            pass
        raise
    return context


def validate_temp_relationships(
    context: TempRunContext,
    input_path: Path,
    output_path: Path,
) -> None:
    input_resolved = input_path.resolve(strict=False)
    output_resolved = output_path.resolve(strict=False)
    leaf = context.run_leaf.resolve(strict=False)
    if input_resolved == output_resolved:
        raise TempWorkspaceError("input and output paths must differ")
    for label, path in (("input", input_resolved), ("output", output_resolved)):
        if _is_relative_to(path, leaf) or _is_relative_to(leaf, path):
            raise TempWorkspaceError(f"temp run leaf overlaps {label} path: {path}")


def check_temp_capacity(context: TempRunContext, estimated_peak_bytes: int) -> str | None:
    if estimated_peak_bytes < 0:
        raise ValueError("estimated_peak_bytes must be non-negative")
    try:
        free = int(shutil.disk_usage(context.temp_root).free)
    except OSError as exc:
        return f"could not query temp free space for {context.temp_root}: {exc}"
    if free < estimated_peak_bytes:
        raise TempWorkspaceError(
            f"insufficient temp space: required={estimated_peak_bytes} bytes, free={free} bytes"
        )
    reserve_target = (estimated_peak_bytes * 110 + 99) // 100
    if free < reserve_target:
        return (
            "temp free-space headroom is below 10%: "
            f"required={estimated_peak_bytes} bytes, free={free} bytes"
        )
    return None


def _load_and_validate_marker(context: TempRunContext) -> None:
    try:
        payload = json.loads(context.marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TempWorkspaceError(f"invalid or missing temp ownership marker: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _MARKER_KEYS:
        raise TempWorkspaceError("temp ownership marker has unknown or missing fields")
    if payload != {
        "type": _MARKER_TYPE,
        "schema_version": _MARKER_VERSION,
        "run_id": context.run_id,
        "input_identity": context.input_identity_dir.name,
    }:
        raise TempWorkspaceError("temp ownership marker does not match run context")


def cleanup_temp_run(context: TempRunContext) -> None:
    try:
        resolved_root_now = context.temp_root_requested.resolve(strict=True)
    except OSError as exc:
        raise TempWorkspaceError(f"cannot re-resolve temp root: {exc}") from exc
    if resolved_root_now != context.temp_root:
        raise TempWorkspaceError("temp root target changed after run creation")
    expected_leaf = context.temp_root / context.input_identity_dir.name / "runs" / context.run_id
    if context.run_leaf != expected_leaf or context.run_leaf.name != context.run_id:
        raise TempWorkspaceError("run leaf layout does not match context")
    if not _is_relative_to(context.run_leaf, context.temp_root):
        raise TempWorkspaceError("run leaf is outside resolved temp root")
    _assert_owned_descendants_not_reparse(context)
    _assert_run_tree_not_reparse(context)
    _load_and_validate_marker(context)
    try:
        shutil.rmtree(context.run_leaf)
    except OSError as exc:
        raise TempWorkspaceError(
            f"failed to clean temp run leaf {context.run_leaf}: {exc}"
        ) from exc
    for parent in (context.run_leaf.parent, context.input_identity_dir):
        try:
            parent.rmdir()
        except OSError:
            break
