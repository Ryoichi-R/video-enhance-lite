from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from video_enhance_lite import config, temp_workspace


def test_temp_root_precedence(monkeypatch, tmp_path):
    env_root = tmp_path / "env-root"
    cli_root = tmp_path / "cli-root"
    monkeypatch.setenv(config.ENV_TEMP_ROOT, str(env_root))
    context = temp_workspace.create_temp_run_context(tmp_path / "input.mp4", cli_root)
    try:
        assert context.temp_root == cli_root.resolve()
        assert context.run_leaf.parent.name == "runs"
        assert len(context.run_id) == 32
    finally:
        temp_workspace.cleanup_temp_run(context)


def test_empty_env_uses_default_root(monkeypatch, tmp_path):
    monkeypatch.setenv(config.ENV_TEMP_ROOT, "   ")
    monkeypatch.setattr(config, "DEFAULT_TEMP_ROOT", tmp_path / "default")
    context = temp_workspace.create_temp_run_context(tmp_path / "input.mp4")
    try:
        assert context.temp_root == (tmp_path / "default").resolve()
    finally:
        temp_workspace.cleanup_temp_run(context)


def test_same_input_has_distinct_run_leaves(tmp_path):
    root = tmp_path / "root"
    first = temp_workspace.create_temp_run_context(tmp_path / "input.mp4", root)
    second = temp_workspace.create_temp_run_context(tmp_path / "input.mp4", root)
    try:
        assert first.input_identity_dir == second.input_identity_dir
        assert first.run_leaf != second.run_leaf
    finally:
        temp_workspace.cleanup_temp_run(first)
        temp_workspace.cleanup_temp_run(second)


def test_cleanup_rejects_marker_mismatch(tmp_path):
    context = temp_workspace.create_temp_run_context(tmp_path / "input.mp4", tmp_path / "root")
    payload = json.loads(context.marker_path.read_text(encoding="utf-8"))
    payload["run_id"] = "0" * 32
    context.marker_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(temp_workspace.TempWorkspaceError, match="does not match"):
        temp_workspace.cleanup_temp_run(context)
    assert context.run_leaf.exists()


def test_cleanup_rejects_missing_marker(tmp_path):
    context = temp_workspace.create_temp_run_context(tmp_path / "input.mp4", tmp_path / "root")
    marker_text = context.marker_path.read_text(encoding="utf-8")
    context.marker_path.unlink()
    try:
        with pytest.raises(temp_workspace.TempWorkspaceError, match="invalid or missing"):
            temp_workspace.cleanup_temp_run(context)
        assert context.run_leaf.exists()
    finally:
        context.marker_path.write_text(marker_text, encoding="utf-8")
        temp_workspace.cleanup_temp_run(context)


def test_cleanup_rejects_retargeted_requested_root(tmp_path):
    context = temp_workspace.create_temp_run_context(tmp_path / "input.mp4", tmp_path / "root")
    other_root = tmp_path / "other-root"
    other_root.mkdir()
    changed = replace(context, temp_root_requested=other_root)
    try:
        with pytest.raises(temp_workspace.TempWorkspaceError, match="target changed"):
            temp_workspace.cleanup_temp_run(changed)
        assert context.run_leaf.exists()
    finally:
        temp_workspace.cleanup_temp_run(context)


def test_cleanup_rejects_reparse_point_below_run_leaf(monkeypatch, tmp_path):
    context = temp_workspace.create_temp_run_context(tmp_path / "input.mp4", tmp_path / "root")
    child = context.run_leaf / "junction"
    child.mkdir()
    original = temp_workspace._is_reparse_point
    monkeypatch.setattr(
        temp_workspace,
        "_is_reparse_point",
        lambda path: path == child or original(path),
    )
    with pytest.raises(temp_workspace.TempWorkspaceError, match="reparse point"):
        temp_workspace.cleanup_temp_run(context)
    assert context.run_leaf.exists()
    monkeypatch.undo()
    temp_workspace.cleanup_temp_run(context)


def test_cleanup_failure_preserves_run_leaf(monkeypatch, tmp_path):
    context = temp_workspace.create_temp_run_context(tmp_path / "input.mp4", tmp_path / "root")

    def _fail(_path):
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(temp_workspace.shutil, "rmtree", _fail)
    with pytest.raises(temp_workspace.TempWorkspaceError, match="failed to clean"):
        temp_workspace.cleanup_temp_run(context)
    assert context.run_leaf.exists()
    monkeypatch.undo()
    temp_workspace.cleanup_temp_run(context)


def test_capacity_boundaries(monkeypatch, tmp_path):
    context = temp_workspace.create_temp_run_context(tmp_path / "input.mp4", tmp_path / "root")
    try:
        monkeypatch.setattr(
            temp_workspace.shutil, "disk_usage", lambda path: SimpleNamespace(free=99)
        )
        with pytest.raises(temp_workspace.TempWorkspaceError, match="insufficient"):
            temp_workspace.check_temp_capacity(context, 100)
        monkeypatch.setattr(
            temp_workspace.shutil, "disk_usage", lambda path: SimpleNamespace(free=105)
        )
        assert "below 10%" in temp_workspace.check_temp_capacity(context, 100)
        monkeypatch.setattr(
            temp_workspace.shutil, "disk_usage", lambda path: SimpleNamespace(free=110)
        )
        assert temp_workspace.check_temp_capacity(context, 100) is None
    finally:
        monkeypatch.undo()
        temp_workspace.cleanup_temp_run(context)


def test_output_inside_run_leaf_is_rejected(tmp_path):
    context = temp_workspace.create_temp_run_context(tmp_path / "input.mp4", tmp_path / "root")
    try:
        with pytest.raises(temp_workspace.TempWorkspaceError, match="overlaps output"):
            temp_workspace.validate_temp_relationships(
                context,
                tmp_path / "input.mp4",
                context.run_leaf / "output.mp4",
            )
    finally:
        temp_workspace.cleanup_temp_run(context)


def test_run_leaf_inside_output_ancestor_is_rejected(tmp_path):
    context = temp_workspace.create_temp_run_context(tmp_path / "input.mp4", tmp_path / "root")
    try:
        with pytest.raises(temp_workspace.TempWorkspaceError, match="overlaps output"):
            temp_workspace.validate_temp_relationships(
                context,
                tmp_path / "input.mp4",
                context.input_identity_dir,
            )
    finally:
        temp_workspace.cleanup_temp_run(context)
