from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

from video_enhance_lite import cli, config, deps, stages
from video_enhance_lite.deps import MissingDependencyError


pytestmark = pytest.mark.release
PROJECT_ROOT = Path(__file__).resolve().parents[2]
windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="asserts Windows-specific path and executable resolution"
)


@windows_only
def test_runtime_roots_are_independent_from_project_root(monkeypatch, tmp_path):
    app_root = tmp_path / "ユーザー data" / "video-enhance-lite"
    config_root = app_root / "config"
    output_root = tmp_path / "Videos" / "video-enhance-lite"
    monkeypatch.setenv(config.ENV_APP_DATA_ROOT, str(app_root.parent))
    monkeypatch.setenv(config.ENV_CONFIG_ROOT, str(config_root))
    monkeypatch.setenv(config.ENV_OUTPUT_ROOT, str(output_root))

    assert config.get_config_root() == config_root
    assert config.get_binary_config_path() == config_root / "enhance-binaries.json"
    assert config.get_default_temp_root() == app_root / "temp"
    assert config.get_default_output_root() == output_root
    assert config.get_default_output_root() != config.PROJECT_ROOT


def test_external_binaries_are_never_discovered_beside_package():
    assert all(not roots for roots in config.DEFAULT_ALLOWED_BINARY_ROOTS.values())
    assert config.get_binary_config_path().parent != config.PROJECT_ROOT / ".local"


def test_stage_registry_is_the_public_count_source():
    assert tuple(spec.key for spec in stages.STAGE_REGISTRY) == config.STAGE_KEYS
    assert stages.PUBLIC_BUILTIN_STAGE_COUNT == 11
    assert stages.PUBLIC_STAGE_COUNT_LABEL == "11組込み段階 + 任意の顔復元1段階"
    face = next(spec for spec in stages.STAGE_REGISTRY if spec.key == "face_restore")
    assert face.optional_external is True
    assert face.public_count is False


def test_cli_self_check_does_not_require_an_input(monkeypatch, capsys):
    monkeypatch.setattr(cli.deps, "format_capability_report", lambda: "diagnostic output")

    assert cli.main(["--self-check"]) == 0
    assert capsys.readouterr().out == "diagnostic output\n"


def test_dependency_error_has_structured_remediation(tmp_path):
    error = MissingDependencyError(
        "realesrgan-ncnn-vulkan",
        "outside allowlist",
        capability="Upscale",
        detected_path=tmp_path / "realesrgan-ncnn-vulkan.exe",
        status="misconfigured",
        config_path=tmp_path / "enhance-binaries.json",
        remediation=("configure the binary root",),
        official_url="https://example.invalid/release",
    )

    payload = error.as_dict()
    assert payload["capability"] == "Upscale"
    assert payload["status"] == "misconfigured"
    assert payload["configuration_location"] == str(tmp_path / "enhance-binaries.json")
    assert payload["remediation"] == ["configure the binary root"]


@windows_only
def test_find_executable_prefers_explicit_allowed_root(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    binary = explicit / "ffmpeg.exe"
    binary.write_bytes(b"fake")
    monkeypatch.setattr(deps.shutil, "which", lambda _name: str(tmp_path / "path" / "ffmpeg.exe"))

    assert deps.find_executable("ffmpeg", (explicit,)) == binary


def test_package_metadata_and_public_documents_are_coherent():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    metadata = pyproject["project"]

    assert metadata["name"] == "video-enhance-lite"
    assert metadata["version"] == "0.2.0"
    assert metadata["readme"] == "README.md"
    assert metadata["requires-python"] == ">=3.11,<3.14"
    assert "enhance" in metadata["optional-dependencies"]
    assert "# video-enhance-lite" in (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "12段" not in (PROJECT_ROOT / "src/video_enhance_lite/README.md").read_text(
        encoding="utf-8"
    )


def test_short_video_analyzer_warning_is_documented():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    runbook = (PROJECT_ROOT / "RUN-enhance.md").read_text(encoding="utf-8")
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert "last decodable frame" in readme
    assert "final analyzer sample was beyond the last decodable frame" in runbook
    assert "final analyzer sample is beyond the last decodable" in changelog
    assert "中間sample" in readme
    assert "FFmpeg失敗" in runbook
    assert "Missing intermediate samples and FFmpeg failures remain fatal" in changelog


def test_launchers_are_checkout_independent():
    for filename in ("video-enhance-lite.bat", "install-enhance.bat"):
        text = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
        assert "%~dp0." in text
        assert "VIDEO_ENHANCE_VENV_DIR" in text
        assert "install-root.txt" in text
        assert "C:\\drive\\coding" not in text
        assert "C:\\coding" not in text
        if filename == "video-enhance-lite.bat":
            assert "where python" not in text.lower()
        else:
            assert "where python" in text.lower()


def test_installer_does_not_use_a_same_block_app_root_expansion_for_venv():
    text = (PROJECT_ROOT / "install-enhance.bat").read_text(encoding="utf-8")

    assert 'set "VENV_DIR=%APP_ROOT%\\venv"' not in text
    assert 'set "VENV_DIR=%LOCALAPPDATA%\\video-enhance-lite\\venv"' in text
    assert 'set "VENV_DIR=%USERPROFILE%\\AppData\\Local\\video-enhance-lite\\venv"' in text


def test_installer_rejects_unvalidated_python_versions_before_creating_the_venv():
    text = (PROJECT_ROOT / "install-enhance.bat").read_text(encoding="utf-8")

    assert "(3,11) <= sys.version_info < (3,14)" in text
    assert "Python 3.11 through 3.13 is required" in text


def test_installer_uses_the_hash_locked_windows_dependency_contract():
    installer = (PROJECT_ROOT / "install-enhance.bat").read_text(encoding="utf-8")
    lock_path = PROJECT_ROOT / "requirements-lock-win-x64.txt"

    assert lock_path.is_file()
    assert "requirements-lock-win-x64.txt" in installer
    assert "--only-binary :all:" in installer
    assert "--require-hashes" in installer
    assert "VIDEO_ENHANCE_WHEELHOUSE" in installer
    assert "--no-index --find-links" in installer
    assert '--no-build-isolation --no-deps -e "%PROJECT_ROOT%"' in installer
    assert "%PROJECT_ROOT%[dev,enhance]" not in installer


def test_windows_dependency_lock_is_fully_pinned_and_hashed():
    text = (PROJECT_ROOT / "requirements-lock-win-x64.txt").read_text(encoding="utf-8")
    logical_lines = text.replace("\\\n", " ").splitlines()
    requirements = [
        line.strip() for line in logical_lines if line.strip() and not line.startswith("#")
    ]

    assert len(requirements) == 14
    assert all(re.match(r"^[a-z0-9-]+==[^ ]+", line) for line in requirements)
    assert all("--hash=sha256:" in line for line in requirements)
    assert 'numpy==2.4.6 ; python_version == "3.11"' in text
    assert 'numpy==2.5.1 ; python_version >= "3.12" and python_version < "3.14"' in text


def test_gui_close_waits_for_worker_cleanup_and_is_documented():
    gui_source = (PROJECT_ROOT / "src/video_enhance_lite/gui.py").read_text(encoding="utf-8")
    runbook = (PROJECT_ROOT / "RUN-enhance.md").read_text(encoding="utf-8")
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert 'protocol("WM_DELETE_WINDOW", self._on_close)' in gui_source
    assert "self._close_requested = True" in gui_source
    assert "workerの停止完了後" in gui_source
    assert "worker停止と一時ファイルの後処理" in runbook
    assert "wait for the worker and external-process cleanup" in changelog


def test_external_engine_example_covers_all_supported_capabilities():
    example = json.loads(
        (PROJECT_ROOT / ".local/enhance-binaries.example.json").read_text(encoding="utf-8")
    )

    assert set(example["allowed_binary_roots"]) == {
        "ffmpeg",
        "ffprobe",
        "realesrgan-ncnn-vulkan",
        "rife-ncnn-vulkan",
    }
    assert set(example["expected_sha256"]) == set(example["allowed_binary_roots"])
    assert example["roots"]["codeformer-local"] == "<codeformer-root>"
    assert "codeformer-python" in example["executables"]
    assert set(example["models"]) == {
        "codeformer",
        "facelib_detection",
        "facelib_parsing",
    }
