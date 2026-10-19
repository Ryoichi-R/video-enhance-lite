from __future__ import annotations


import pytest

from video_enhance_lite import config


# ---------- realesrgan_param_files ----------


def test_realesrgan_param_files_scaled_animevideov3():
    assert config.realesrgan_param_files("realesr-animevideov3", 2) == (
        "realesr-animevideov3-x2.param",
        "realesr-animevideov3-x2.bin",
    )
    assert config.realesrgan_param_files("realesr-animevideov3", 3) == (
        "realesr-animevideov3-x3.param",
        "realesr-animevideov3-x3.bin",
    )
    assert config.realesrgan_param_files("realesr-animevideov3", 4) == (
        "realesr-animevideov3-x4.param",
        "realesr-animevideov3-x4.bin",
    )


def test_realesrgan_param_files_fixed_x4plus():
    assert config.realesrgan_param_files("realesrgan-x4plus", 4) == (
        "realesrgan-x4plus.param",
        "realesrgan-x4plus.bin",
    )


def test_realesrgan_param_files_unknown_model_raises():
    with pytest.raises(ValueError, match="unknown realesrgan model"):
        config.realesrgan_param_files("does-not-exist", 4)


# ---------- validate_realesrgan_model ----------


def test_validate_realesrgan_missing_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        config.validate_realesrgan_model(tmp_path, "realesr-animevideov3", 2)


def test_validate_realesrgan_scale_mismatch_raises(tmp_path):
    # x4plus は scale=4 のみ
    with pytest.raises(ValueError, match="supports scale"):
        config.validate_realesrgan_model(tmp_path, "realesrgan-x4plus", 2)


def test_validate_realesrgan_ok_when_files_present(tmp_path):
    (tmp_path / "realesr-animevideov3-x2.param").write_text("dummy")
    (tmp_path / "realesr-animevideov3-x2.bin").write_text("dummy")
    # 例外が出なければ OK
    config.validate_realesrgan_model(tmp_path, "realesr-animevideov3", 2)


def test_validate_realesrgan_general_x4v3_requires_files(tmp_path):
    """OPTIONAL モデル (同梱なし) は param/bin が無ければ validate に失敗。"""
    with pytest.raises(FileNotFoundError):
        config.validate_realesrgan_model(tmp_path, "realesr-general-x4v3", 4)


# ---------- validate_rife_model ----------


def test_validate_rife_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="rife model dir not found"):
        config.validate_rife_model(tmp_path, "rife-v4.6")


def test_validate_rife_missing_flownet_raises(tmp_path):
    (tmp_path / "rife-v4.6").mkdir()
    with pytest.raises(FileNotFoundError, match="flownet.param"):
        config.validate_rife_model(tmp_path, "rife-v4.6")


def test_validate_rife_ok_when_present(tmp_path):
    model_dir = tmp_path / "rife-v4.6"
    model_dir.mkdir()
    (model_dir / "flownet.param").write_text("dummy")
    config.validate_rife_model(tmp_path, "rife-v4.6")


# ---------- resolve_rife_model_dir ----------


def test_resolve_rife_model_dir_official_layout(tmp_path):
    """公式 20221029 配布構造: exe と同階層に <model>/ が直接並ぶ (models/ なし)。"""
    model_dir = tmp_path / "rife-v4.6"
    model_dir.mkdir()
    (model_dir / "flownet.param").write_text("dummy")
    resolved = config.resolve_rife_model_dir(tmp_path, "rife-v4.6")
    assert resolved == tmp_path
    # resolve の戻り値をそのまま validate に渡せる
    config.validate_rife_model(resolved, "rife-v4.6")


def test_resolve_rife_model_dir_legacy_models_layout(tmp_path):
    """旧想定の <exe親>/models/<model>/ 配置も互換のため解決できる。"""
    model_dir = tmp_path / "models" / "rife-v4.6"
    model_dir.mkdir(parents=True)
    (model_dir / "flownet.param").write_text("dummy")
    resolved = config.resolve_rife_model_dir(tmp_path, "rife-v4.6")
    assert resolved == tmp_path / "models"
    config.validate_rife_model(resolved, "rife-v4.6")


def test_resolve_rife_model_dir_missing_lists_both_candidates(tmp_path):
    """両候補に無い場合は FileNotFoundError で候補パスを両方提示する。"""
    with pytest.raises(FileNotFoundError) as ei:
        config.resolve_rife_model_dir(tmp_path, "rife-v4.6")
    msg = str(ei.value)
    assert str(tmp_path / "rife-v4.6") in msg
    assert str(tmp_path / "models" / "rife-v4.6") in msg


def test_resolve_rife_model_dir_prefers_official_over_legacy(tmp_path):
    """両方ある場合は公式構造 (exe 同階層) を優先する。"""
    for base in (tmp_path, tmp_path / "models"):
        d = base / "rife-v4.6"
        d.mkdir(parents=True)
        (d / "flownet.param").write_text("dummy")
    assert config.resolve_rife_model_dir(tmp_path, "rife-v4.6") == tmp_path


# ---------- model kind tables ----------


def test_realesrgan_model_kind_contains_optional():
    assert "realesr-general-x4v3" in config.REALESRGAN_MODEL_KIND
    assert "realesr-general-x4v3" in config.OPTIONAL_REALESRGAN_MODELS


def test_allowed_rife_models_includes_v4_and_legacy():
    assert "rife-v4.6" in config.ALLOWED_RIFE_MODELS
    assert "rife-v4" in config.ALLOWED_RIFE_MODELS
    assert "rife-v2.3" in config.ALLOWED_RIFE_MODELS
    assert config.DEFAULT_RIFE_MODEL in config.ALLOWED_RIFE_MODELS
