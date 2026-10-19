"""Tests for video_enhance_lite.deps allowlist + SHA-256 validation layer.

ハードニング後の deps.check_* は以下を検証する:
  - 解決済みパスが BinaryPolicy.allowed_binary_roots 内にあるか
  - 期待 SHA-256 (BinaryPolicy.expected_sha256) と一致するか

local config (`.local/enhance-binaries.json`) と環境変数 (VIDEO_ENHANCE_*) で
allowlist/expected を上書きできる。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from video_enhance_lite import config, deps
from video_enhance_lite.deps import MissingDependencyError


# ---------- BinaryPolicy loading ----------


def test_load_binary_policy_defaults_only(tmp_path, monkeypatch):
    """No local config, no env override -> falls back to DEFAULT_ALLOWED_BINARY_ROOTS."""
    monkeypatch.setattr(os, "environ", {})
    pol = config.load_binary_policy(
        local_config_path=tmp_path / "missing.json",
        env={},
    )
    # default の rife/ffmpeg/ffprobe は空集合
    assert pol.roots_for("ffmpeg") == ()
    assert pol.roots_for("ffprobe") == ()
    assert pol.roots_for("rife-ncnn-vulkan") == ()
    # ソースツリー隣接の外部バイナリは公開候補へ暗黙採用しない
    assert pol.roots_for("realesrgan-ncnn-vulkan") == ()
    # expected_sha256 はすべて空 (RIFE_KNOWN_SHA256 が空なので)
    for name in config.BINARY_NAMES:
        assert pol.expected_for(name) == ()


def test_load_binary_policy_local_config(tmp_path):
    """local config が allowed_binary_roots を完全に上書きする (マージしない)."""
    cfg_path = tmp_path / "enhance-binaries.json"
    cfg_path.write_text(
        json.dumps(
            {
                "allowed_binary_roots": {
                    "ffmpeg": [r"C:\custom\ffmpeg\bin"],
                    "rife-ncnn-vulkan": [r"C:\tools\rife-ncnn-vulkan-20221029"],
                },
                "expected_sha256": {
                    "rife-ncnn-vulkan": ["abc123" + "0" * 58],
                },
            }
        ),
        encoding="utf-8",
    )

    pol = config.load_binary_policy(local_config_path=cfg_path, env={})

    assert pol.roots_for("ffmpeg") == (Path(r"C:\custom\ffmpeg\bin"),)
    assert pol.roots_for("rife-ncnn-vulkan") == (Path(r"C:\tools\rife-ncnn-vulkan-20221029"),)
    # local config に書かれていない realesrgan も暗黙採用されない
    assert pol.roots_for("realesrgan-ncnn-vulkan") == ()
    assert pol.expected_for("rife-ncnn-vulkan") == ("abc123" + "0" * 58,)


def test_load_binary_policy_env_overrides_local_config(tmp_path):
    """環境変数の値が local config より優先される."""
    cfg_path = tmp_path / "enhance-binaries.json"
    cfg_path.write_text(
        json.dumps(
            {
                "allowed_binary_roots": {
                    "ffmpeg": [r"C:\local\ffmpeg"],
                },
            }
        ),
        encoding="utf-8",
    )
    env = {"VIDEO_ENHANCE_FFMPEG_ROOTS": r"C:\env\ffmpeg;C:\other\ffmpeg"}

    pol = config.load_binary_policy(local_config_path=cfg_path, env=env)

    # local の C:\local\ffmpeg ではなく env の 2 つが採用される
    assert pol.roots_for("ffmpeg") == (
        Path(r"C:\env\ffmpeg"),
        Path(r"C:\other\ffmpeg"),
    )


def test_load_binary_policy_env_expected_sha256_overrides(tmp_path):
    """VIDEO_ENHANCE_EXPECTED_SHA256_JSON が expected_sha256 を上書きする."""
    cfg_path = tmp_path / "enhance-binaries.json"
    cfg_path.write_text(
        json.dumps(
            {
                "expected_sha256": {
                    "rife-ncnn-vulkan": ["from-local"],
                },
            }
        ),
        encoding="utf-8",
    )
    env = {
        "VIDEO_ENHANCE_EXPECTED_SHA256_JSON": json.dumps(
            {
                "rife-ncnn-vulkan": ["FROM-ENV"],
                "realesrgan-ncnn-vulkan": ["abc"],
            }
        ),
    }

    pol = config.load_binary_policy(local_config_path=cfg_path, env=env)

    # env の値が採用される (小文字化される)
    assert pol.expected_for("rife-ncnn-vulkan") == ("from-env",)
    assert pol.expected_for("realesrgan-ncnn-vulkan") == ("abc",)


def test_load_binary_policy_empty_env_means_no_roots(tmp_path):
    """環境変数を空文字で明示すると allowed_binary_roots が空集合になる."""
    env = {"VIDEO_ENHANCE_REALESRGAN_ROOTS": ""}
    pol = config.load_binary_policy(local_config_path=tmp_path / "x.json", env=env)
    assert pol.roots_for("realesrgan-ncnn-vulkan") == ()


def test_load_binary_policy_invalid_json_silently_falls_back(tmp_path):
    cfg_path = tmp_path / "enhance-binaries.json"
    cfg_path.write_text("{ not valid json", encoding="utf-8")
    pol = config.load_binary_policy(local_config_path=cfg_path, env={})
    # default にフォールバック
    assert pol.roots_for("ffmpeg") == ()


def test_check_codeformer_local_plan_style_config(tmp_path):
    repo = tmp_path / "CodeFormer"
    repo.mkdir()
    script = repo / "inference_codeformer.py"
    script.write_text("# fake\n", encoding="utf-8")
    python = tmp_path / "env" / "python.exe"
    python.parent.mkdir()
    python.write_bytes(b"fake python")
    codeformer = repo / "weights" / "CodeFormer" / "codeformer.pth"
    detection = repo / "weights" / "facelib" / "detection_Resnet50_Final.pth"
    parsing = repo / "weights" / "facelib" / "parsing_parsenet.pth"
    for p in (codeformer, detection, parsing):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"weight")

    cfg_path = tmp_path / "enhance-binaries.json"
    cfg_path.write_text(
        json.dumps(
            {
                "roots": {"codeformer-local": [str(repo)]},
                "executables": {"codeformer-python": str(python)},
                "models": {
                    "codeformer": [{"path": str(codeformer)}],
                    "facelib_detection": [{"path": str(detection)}],
                    "facelib_parsing": [{"path": str(parsing)}],
                },
            }
        ),
        encoding="utf-8",
    )

    resolved = deps.check_codeformer_local(local_config_path=cfg_path)

    assert resolved.repo_root == repo.resolve()
    assert resolved.python == python.resolve()
    assert resolved.script == script.resolve()
    assert resolved.models["codeformer"] == codeformer.resolve()
    assert resolved.models["facelib_detection"] == detection.resolve()
    assert resolved.models["facelib_parsing"] == parsing.resolve()


def test_check_codeformer_local_missing_config_raises(tmp_path):
    with pytest.raises(MissingDependencyError) as ei:
        deps.check_codeformer_local(local_config_path=tmp_path / "missing.json")
    assert ei.value.name == "codeformer-local"


def test_codeformer_optional_hash_has_no_size_cap(tmp_path, monkeypatch):
    calls = []

    def fake_compute_sha256(path: Path, *, max_bytes: int | None = None) -> str:
        calls.append((path, max_bytes))
        return "a" * 64

    monkeypatch.setattr(config, "compute_sha256", fake_compute_sha256)

    assert (
        deps._check_model_hash(
            "codeformer",
            tmp_path / "codeformer.pth",
            "a" * 64,
        )
        == "a" * 64
    )
    assert calls == [(tmp_path / "codeformer.pth", None)]


# ---------- _check_allowed_root ----------


def test_check_allowed_root_raises_when_no_roots():
    """policy に許可ディレクトリ未設定なら停止する."""
    pol = config.BinaryPolicy(
        allowed_binary_roots={"ffmpeg": ()},
        expected_sha256={"ffmpeg": ()},
    )
    with pytest.raises(MissingDependencyError) as ei:
        deps._check_allowed_root("ffmpeg", Path(r"C:\anywhere\ffmpeg.exe"), pol)
    assert "許可ディレクトリが未設定" in ei.value.hint


def test_check_allowed_root_inside_allowed(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "ffmpeg.exe"
    exe.write_bytes(b"")
    pol = config.BinaryPolicy(
        allowed_binary_roots={"ffmpeg": (bin_dir,)},
        expected_sha256={"ffmpeg": ()},
    )
    deps._check_allowed_root("ffmpeg", exe.resolve(), pol)


def test_check_allowed_root_outside_raises(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    exe = other / "ffmpeg.exe"
    exe.write_bytes(b"")
    pol = config.BinaryPolicy(
        allowed_binary_roots={"ffmpeg": (allowed,)},
        expected_sha256={"ffmpeg": ()},
    )
    with pytest.raises(MissingDependencyError) as ei:
        deps._check_allowed_root("ffmpeg", exe.resolve(), pol)
    assert ei.value.name == "ffmpeg"
    assert "許可ディレクトリ" in ei.value.hint


def test_check_binary_shape_rejects_reparse_component(tmp_path, monkeypatch):
    exe = tmp_path / "ffmpeg.exe"
    exe.write_bytes(b"fake")
    junction = tmp_path / "junction"
    monkeypatch.setattr(deps, "_has_reparse_component", lambda _path: junction)

    with pytest.raises(MissingDependencyError) as ei:
        deps._check_binary_shape("ffmpeg", exe)

    assert ei.value.status == "misconfigured"
    assert "reparse point" in ei.value.hint


def test_validate_resolved_checks_reparse_before_resolve(tmp_path, monkeypatch):
    exe = tmp_path / "ffmpeg.exe"
    exe.write_bytes(b"fake")
    policy = config.BinaryPolicy(
        allowed_binary_roots={"ffmpeg": (tmp_path,)},
        expected_sha256={"ffmpeg": ()},
    )
    monkeypatch.setattr(deps, "_has_reparse_component", lambda _path: tmp_path / "junction")

    with pytest.raises(MissingDependencyError, match="reparse point"):
        deps._validate_resolved("ffmpeg", exe, policy=policy)


# ---------- _check_expected_hash ----------


def test_check_expected_hash_skipped_when_empty():
    pol = config.BinaryPolicy(
        allowed_binary_roots={"ffmpeg": ()},
        expected_sha256={"ffmpeg": ()},
    )
    deps._check_expected_hash("ffmpeg", "abc123", pol)  # no raise


def test_check_expected_hash_matches():
    pol = config.BinaryPolicy(
        allowed_binary_roots={"ffmpeg": ()},
        expected_sha256={"ffmpeg": ("abc" + "0" * 61,)},
    )
    deps._check_expected_hash("ffmpeg", "ABC" + "0" * 61, pol)  # case-insensitive


def test_check_expected_hash_mismatch_raises():
    pol = config.BinaryPolicy(
        allowed_binary_roots={"ffmpeg": ()},
        expected_sha256={"ffmpeg": ("expected" + "0" * 56,)},
    )
    with pytest.raises(MissingDependencyError) as ei:
        deps._check_expected_hash("ffmpeg", "wrong" + "0" * 59, pol)
    assert "一致しません" in ei.value.hint


def test_check_expected_hash_none_fails_close_when_expected_set():
    """P2-5: 期待ハッシュ設定済み + sha256 実測不能 (None) は fail-close で停止する
    (旧実装は黙ってスキップし、200MiB 超のバイナリが検証を素通りしていた)。"""
    pol = config.BinaryPolicy(
        allowed_binary_roots={"ffmpeg": ()},
        expected_sha256={"ffmpeg": ("abc",)},
    )
    with pytest.raises(MissingDependencyError) as ei:
        deps._check_expected_hash("ffmpeg", None, pol)
    assert "実測できませんでした" in ei.value.hint


def test_check_expected_hash_none_skipped_when_no_expected():
    """期待ハッシュ未設定なら sha256=None でも従来どおり通す (informational only)。"""
    pol = config.BinaryPolicy(
        allowed_binary_roots={"ffmpeg": ()},
        expected_sha256={"ffmpeg": ()},
    )
    deps._check_expected_hash("ffmpeg", None, pol)  # no raise


# ---------- end-to-end via check_* ----------


def _write_fake_binary(path: Path, body: bytes = b"fake binary\n") -> str:
    """Write a fake exe and return its SHA-256 (hex)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def test_check_ffmpeg_allowlist_blocks_outside(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    fake = tmp_path / "evil" / "ffmpeg.exe"
    _write_fake_binary(fake)  # fake バイナリのファイル生成が目的。返り値 sha は未使用

    monkeypatch.setattr(deps.shutil, "which", lambda name: str(fake))
    monkeypatch.setattr(
        config,
        "load_binary_policy",
        lambda: config.BinaryPolicy(
            allowed_binary_roots={"ffmpeg": (allowed,)},
            expected_sha256={"ffmpeg": ()},
        ),
    )

    with pytest.raises(MissingDependencyError) as ei:
        deps.check_ffmpeg()
    assert ei.value.name == "ffmpeg"
    assert "許可ディレクトリ外" in ei.value.hint


def test_check_ffmpeg_allowlist_permits_inside(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    fake = allowed / "ffmpeg.exe"
    _write_fake_binary(fake)

    monkeypatch.setattr(deps.shutil, "which", lambda name: str(fake))
    monkeypatch.setattr(
        config,
        "load_binary_policy",
        lambda: config.BinaryPolicy(
            allowed_binary_roots={"ffmpeg": (allowed,)},
            expected_sha256={"ffmpeg": ()},
        ),
    )
    # _build_resolved は _get_version で subprocess を呼ぶ。fake は exe ではないので
    # version は None になる程度で OK (raise はしない)。
    monkeypatch.setattr(deps, "_get_version", lambda p: None)

    rb = deps.check_ffmpeg()
    assert rb.path == fake.resolve()


def test_check_realesrgan_hash_mismatch_raises(tmp_path, monkeypatch):
    """期待 SHA-256 と一致しない場合 check_realesrgan で停止する."""
    allowed = tmp_path / "tools"
    fake = allowed / "realesrgan-ncnn-vulkan.exe"
    actual_sha = _write_fake_binary(fake, b"REAL")
    wrong_sha = "f" * 64

    monkeypatch.setattr(deps.shutil, "which", lambda name: str(fake))
    monkeypatch.setattr(
        config,
        "load_binary_policy",
        lambda: config.BinaryPolicy(
            allowed_binary_roots={"realesrgan-ncnn-vulkan": (allowed,)},
            expected_sha256={"realesrgan-ncnn-vulkan": (wrong_sha,)},
        ),
    )
    monkeypatch.setattr(deps, "_get_version", lambda p: None)

    with pytest.raises(MissingDependencyError) as ei:
        deps.check_realesrgan()
    assert "SHA-256" in ei.value.hint
    # actual sha が出力に含まれている
    assert actual_sha in ei.value.hint


def test_check_realesrgan_hash_match_passes(tmp_path, monkeypatch):
    """期待 SHA-256 と一致する場合は ResolvedBinary を返す."""
    allowed = tmp_path / "tools"
    fake = allowed / "realesrgan-ncnn-vulkan.exe"
    sha = _write_fake_binary(fake, b"REAL\n")

    monkeypatch.setattr(deps.shutil, "which", lambda name: str(fake))
    monkeypatch.setattr(
        config,
        "load_binary_policy",
        lambda: config.BinaryPolicy(
            allowed_binary_roots={"realesrgan-ncnn-vulkan": (allowed,)},
            expected_sha256={"realesrgan-ncnn-vulkan": (sha,)},
        ),
    )
    monkeypatch.setattr(deps, "_get_version", lambda p: None)

    rb = deps.check_realesrgan()
    assert rb.sha256 == sha


def test_check_rife_missing_raises(monkeypatch):
    monkeypatch.setattr(deps.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        config,
        "load_binary_policy",
        lambda: config.BinaryPolicy(
            allowed_binary_roots={"rife-ncnn-vulkan": ()},
            expected_sha256={"rife-ncnn-vulkan": ()},
        ),
    )
    with pytest.raises(MissingDependencyError) as ei:
        deps.check_rife()
    assert ei.value.name == "rife-ncnn-vulkan"


def test_check_ffmpeg_no_allowlist_no_hash_raises(tmp_path, monkeypatch):
    """policy が空 (allowlist 無し) なら PATH 解決先を実行許可しない."""
    fake = tmp_path / "anywhere" / "ffmpeg.exe"
    _write_fake_binary(fake)

    monkeypatch.setattr(deps.shutil, "which", lambda name: str(fake))
    monkeypatch.setattr(
        config,
        "load_binary_policy",
        lambda: config.BinaryPolicy(
            allowed_binary_roots={"ffmpeg": ()},
            expected_sha256={"ffmpeg": ()},
        ),
    )
    monkeypatch.setattr(deps, "_get_version", lambda p: None)

    with pytest.raises(MissingDependencyError) as ei:
        deps.check_ffmpeg()
    assert "許可ディレクトリが未設定" in ei.value.hint


# ---------- verify_rife_sha256 (legacy info channel) ----------


def _empty_policy() -> config.BinaryPolicy:
    """local config / env に依存しない空ポリシー (テスト分離用)。"""
    return config.BinaryPolicy(
        allowed_binary_roots={name: () for name in config.BINARY_NAMES},
        expected_sha256={name: () for name in config.BINARY_NAMES},
    )


def test_verify_rife_sha256_empty_table_returns_info(monkeypatch):
    monkeypatch.setattr(config, "RIFE_KNOWN_SHA256", {})
    rb = deps.ResolvedBinary(path=Path(r"C:\x\rife.exe"), version=None, sha256="abc")
    res = deps.verify_rife_sha256(rb, policy=_empty_policy())
    assert res.level == "info"
    assert "known-hash table empty" in res.message


def test_verify_rife_sha256_matching_entry(monkeypatch):
    monkeypatch.setattr(config, "RIFE_KNOWN_SHA256", {"rife-20221029.zip": "ABC"})
    rb = deps.ResolvedBinary(path=Path(r"C:\x\rife.exe"), version=None, sha256="abc")
    res = deps.verify_rife_sha256(rb, policy=_empty_policy())
    assert res.level == "info"
    assert res.matched_name == "rife-20221029.zip"


def test_verify_rife_sha256_mismatch_returns_warn(monkeypatch):
    monkeypatch.setattr(config, "RIFE_KNOWN_SHA256", {"rife-20221029.zip": "abc"})
    rb = deps.ResolvedBinary(path=Path(r"C:\x\rife.exe"), version=None, sha256="xyz")
    res = deps.verify_rife_sha256(rb, policy=_empty_policy())
    assert res.level == "warn"
    assert "does not match" in res.message


def test_verify_rife_sha256_policy_expected_reports_verified(monkeypatch):
    """P3 回帰: BinaryPolicy に期待値があれば legacy テーブルが空でも
    「known-hash table empty」ではなく検証済みメッセージを返す。"""
    monkeypatch.setattr(config, "RIFE_KNOWN_SHA256", {})
    pol = config.BinaryPolicy(
        allowed_binary_roots={name: () for name in config.BINARY_NAMES},
        expected_sha256={
            **{name: () for name in config.BINARY_NAMES},
            "rife-ncnn-vulkan": ("abc",),
        },
    )
    rb = deps.ResolvedBinary(path=Path(r"C:\x\rife.exe"), version=None, sha256="abc")
    res = deps.verify_rife_sha256(rb, policy=pol)
    assert res.level == "info"
    assert "BinaryPolicy" in res.message
    assert "known-hash table empty" not in res.message
