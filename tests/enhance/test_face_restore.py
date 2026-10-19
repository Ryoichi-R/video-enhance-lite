from __future__ import annotations

import hashlib
import errno
import json
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest

from video_enhance_lite import cli, config, decisions, deps, pipeline, stages
from video_enhance_lite.decisions import StageDecision

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_link_or_copy_atomic_uses_hardlink_when_supported(tmp_path):
    src = tmp_path / "src.png"
    dst = tmp_path / "dst.png"
    src.write_bytes(b"source")
    result = pipeline._link_or_copy_atomic(src, dst)
    assert result == "linked"
    assert os.path.samefile(src, dst)


def test_link_or_copy_atomic_falls_back_for_cross_device(monkeypatch, tmp_path):
    src = tmp_path / "src.png"
    dst = tmp_path / "dst.png"
    src.write_bytes(b"source")

    def _cross_device(_src, _dst):
        raise OSError(errno.EXDEV, "cross-device")

    monkeypatch.setattr(os, "link", _cross_device)
    result = pipeline._link_or_copy_atomic(src, dst)
    assert result == "copied"
    assert dst.read_bytes() == b"source"
    assert not os.path.samefile(src, dst)


def test_atomic_replace_does_not_modify_existing_hardlink_source(tmp_path):
    original = tmp_path / "original.png"
    replacement = tmp_path / "replacement.png"
    dst = tmp_path / "dst.png"
    original.write_bytes(b"original")
    replacement.write_bytes(b"replacement")
    os.link(original, dst)
    original_hash = _sha256(original)

    pipeline._atomic_replace_from_file(replacement, dst)

    assert _sha256(original) == original_hash
    assert dst.read_bytes() == b"replacement"


def test_link_or_copy_atomic_replace_failure_preserves_destination(monkeypatch, tmp_path):
    src = tmp_path / "src.png"
    dst = tmp_path / "dst.png"
    src.write_bytes(b"new")
    dst.write_bytes(b"old")

    def _fail_replace(_src, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", _fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        pipeline._link_or_copy_atomic(src, dst)
    assert dst.read_bytes() == b"old"
    assert src.read_bytes() == b"new"
    assert not list(tmp_path.glob(".*.tmp"))


def test_non_ascii_path_safe_naturalize_composite(tmp_path):
    root = tmp_path / "日本語 temp root"
    root.mkdir()
    original = root / "元画像.png"
    restored = root / "復元画像.png"
    output = root / "合成結果.png"
    pipeline._write_png_path_safe_atomic(
        original,
        np.full((4, 4, 3), 10, dtype=np.uint8),
    )
    pipeline._write_png_path_safe_atomic(
        restored,
        np.full((4, 4, 3), 110, dtype=np.uint8),
    )

    result = pipeline._compose_face_restore_frame(
        original_path=original,
        restored_path=restored,
        output_path=output,
        options=pipeline.FaceRestoreOptions(blend=0.5),
    )

    assert result.wrote is True
    assert np.all(pipeline._read_image_path_safe(output, cv2.IMREAD_COLOR) == 60)


@dataclass
class FakeMeta:
    width: int = 1280
    height: int = 720
    fps: Fraction = Fraction(30, 1)
    duration_sec: float = 10.0
    codec: str = "h264"
    bitrate_bps: int | None = 5_000_000
    pix_fmt: str = "yuv420p"
    color_range: str = "tv"
    color_space: str = "bt709"
    field_order: str = "progressive"
    has_audio: bool = True


@dataclass
class FakeStats:
    laplacian_var_mean: float = 200.0
    luma_std_over_time: float = 1.0
    interframe_diff_mean: float = 1.0
    interframe_diff_std: float = 1.0
    histogram_flat_ratio: float = 0.05
    luma_mean: float = 120.0
    luma_clipping_ratio: float = 0.001
    saturation_mean: float = 80.0
    highfreq_energy: float = 5.0
    sample_count: int = 24


def _decisions(*, face_restore: bool, interpolate: bool = False) -> list[StageDecision]:
    out: list[StageDecision] = []
    for key in config.STAGE_KEYS:
        params: dict = {}
        enabled = False
        if key == "face_restore":
            enabled = face_restore
            params = {
                "backend": config.DEFAULT_FACE_RESTORE_BACKEND,
                "fidelity": config.DEFAULT_FACE_RESTORE_FIDELITY,
            }
        elif key == "interpolate":
            enabled = interpolate
            params = {
                "target_fps_num": 60,
                "target_fps_den": 1,
                "model": config.DEFAULT_RIFE_MODEL,
            }
        elif key == "encode":
            enabled = True
            params = {"crf": config.DEFAULT_ENCODE_CRF, "preset": config.DEFAULT_ENCODE_PRESET}
        out.append(
            StageDecision(
                key=key,
                label_ja=config.STAGE_LABELS_JA[key],
                recommended=enabled,
                confidence="high",
                reason="test",
                enabled=enabled,
                params=params,
            )
        )
    return out


def _fake_codeformer(tmp_path: Path) -> deps.ResolvedCodeFormerLocal:
    repo = tmp_path / "CodeFormer"
    repo.mkdir()
    script = repo / "inference_codeformer.py"
    script.write_text("print('fake')\n", encoding="utf-8")
    py_env = tmp_path / "codeformer-env"
    for rel in ("bin", "Library/bin", "Scripts"):
        (py_env / rel).mkdir(parents=True, exist_ok=True)
    py = py_env / "python.exe"
    py.write_bytes(b"py")
    models = {
        "codeformer": repo / "weights" / "CodeFormer" / "codeformer.pth",
        "facelib_detection": repo / "weights" / "facelib" / "detection_Resnet50_Final.pth",
        "facelib_parsing": repo / "weights" / "facelib" / "parsing_parsenet.pth",
    }
    for key in config.CODEFORMER_REQUIRED_MODEL_KEYS:
        p = models[key]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(key.encode("ascii"))
    return deps.ResolvedCodeFormerLocal(
        repo_root=repo,
        python=py,
        script=script,
        models=models,
        model_sha256={key: None for key in config.CODEFORMER_REQUIRED_MODEL_KEYS},
    )


def test_face_restore_decision_default_off_and_order():
    decs = decisions.recommend(FakeMeta(), FakeStats())
    keys = [d.key for d in decs]
    assert keys == list(config.STAGE_KEYS)
    assert keys.index("upscale") < keys.index("face_restore") < keys.index("interpolate")
    face = decisions.decisions_by_key(decs)["face_restore"]
    assert face.recommended is False
    assert face.enabled is False
    assert face.params["backend"] == "codeformer-local"
    face.reason.encode("cp932", errors="strict")


def test_face_restore_suffix_when_enabled():
    decs = _decisions(face_restore=True)
    summary = decisions.compute_output_summary(FakeMeta(), decs)
    assert decisions.format_output_spec_suffix(summary).endswith("_face-restored")


def test_parser_face_restore_options():
    p = cli._build_parser()
    args = p.parse_args(
        [
            "in.mp4",
            "--enable",
            "face_restore",
            "--face-restore-start",
            "30",
            "--face-restore-end",
            "40",
            "--face-restore-fidelity",
            "0.8",
            "--face-restore-idle-timeout",
            "600",
            "--face-restore-blend",
            "0.25",
            "--face-restore-mask",
            "diff",
            "--face-restore-match-grain",
            "0.2",
            "--face-restore-soften",
            "0.1",
            "--face-restore-global-smooth",
            "0.05",
        ]
    )
    assert args.enable == ["face_restore"]
    assert args.face_restore_start == 30.0
    assert args.face_restore_end == 40.0
    assert args.face_restore_fidelity == 0.8
    assert args.face_restore_idle_timeout == 600
    assert args.face_restore_blend == 0.25
    assert args.face_restore_mask == "diff"
    assert args.face_restore_match_grain == 0.2
    assert args.face_restore_soften == 0.1
    assert args.face_restore_global_smooth == 0.05


@pytest.mark.parametrize("bad", ["-0.1", "1.1", "nan"])
def test_parser_face_restore_fidelity_rejects_invalid(bad):
    p = cli._build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["in.mp4", "--face-restore-fidelity", bad])


def _write_png(path: Path, value: int | list[list[list[int]]]) -> None:
    if isinstance(value, int):
        img = np.full((4, 4, 3), value, dtype=np.uint8)
    else:
        img = np.array(value, dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def _read_png(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None
    return img


def test_face_restore_composite_full_frame_blend_midpoint(tmp_path):
    orig = tmp_path / "orig.png"
    rest = tmp_path / "rest.png"
    out = tmp_path / "out.png"
    _write_png(orig, 10)
    _write_png(rest, 110)

    result = pipeline._compose_face_restore_frame(
        original_path=orig,
        restored_path=rest,
        output_path=out,
        options=pipeline.FaceRestoreOptions(blend=0.5),
    )

    assert result.wrote is True
    assert result.resized is False
    assert np.all(_read_png(out) == 60)


def test_face_restore_composite_blend_zero_keeps_original(tmp_path):
    orig = tmp_path / "orig.png"
    rest = tmp_path / "rest.png"
    out = tmp_path / "out.png"
    _write_png(orig, 10)
    _write_png(rest, 250)

    pipeline._compose_face_restore_frame(
        original_path=orig,
        restored_path=rest,
        output_path=out,
        options=pipeline.FaceRestoreOptions(blend=0.0),
    )

    assert np.all(_read_png(out) == 10)


def test_face_restore_composite_diff_mask_only_changes_different_region(tmp_path):
    orig = tmp_path / "orig.png"
    rest = tmp_path / "rest.png"
    out = tmp_path / "out.png"
    original = np.full((4, 4, 3), 10, dtype=np.uint8)
    restored = original.copy()
    restored[1:3, 1:3] = 110
    assert cv2.imwrite(str(orig), original)
    assert cv2.imwrite(str(rest), restored)

    pipeline._compose_face_restore_frame(
        original_path=orig,
        restored_path=rest,
        output_path=out,
        options=pipeline.FaceRestoreOptions(
            blend=1.0,
            mask="diff",
            feather=0,
            mask_dilate=0,
            mask_threshold=8,
        ),
    )

    composed = _read_png(out)
    assert np.all(composed[0, 0] == 10)
    assert np.all(composed[1, 1] == 110)


def test_face_restore_composite_empty_diff_mask_skips_write(tmp_path):
    orig = tmp_path / "orig.png"
    rest = tmp_path / "rest.png"
    out = tmp_path / "out.png"
    _write_png(orig, 10)
    _write_png(rest, 10)

    result = pipeline._compose_face_restore_frame(
        original_path=orig,
        restored_path=rest,
        output_path=out,
        options=pipeline.FaceRestoreOptions(mask="diff"),
    )

    assert result.wrote is False
    assert result.empty_mask is True
    assert not out.exists()


def test_face_restore_composite_resizes_restored_to_original(tmp_path):
    orig = tmp_path / "orig.png"
    rest = tmp_path / "rest.png"
    out = tmp_path / "out.png"
    assert cv2.imwrite(str(orig), np.full((4, 4, 3), 10, dtype=np.uint8))
    assert cv2.imwrite(str(rest), np.full((2, 2, 3), 110, dtype=np.uint8))

    result = pipeline._compose_face_restore_frame(
        original_path=orig,
        restored_path=rest,
        output_path=out,
        options=pipeline.FaceRestoreOptions(blend=0.5),
    )

    assert result.resized is True
    assert _read_png(out).shape[:2] == (4, 4)


def test_face_restore_options_from_params_validates_values():
    opts = pipeline.FaceRestoreOptions.from_params(
        {
            "blend": 0.3,
            "mask": "diff",
            "feather": 12,
            "mask_dilate": 8,
            "mask_threshold": 9,
            "match_grain": 0.2,
            "soften": 0.1,
            "global_smooth": 0.05,
        }
    )

    assert opts.blend == 0.3
    assert opts.mask == "diff"
    with pytest.raises(ValueError):
        pipeline.FaceRestoreOptions(blend=float("nan")).validate()
    with pytest.raises(ValueError):
        pipeline.FaceRestoreOptions(feather=257).validate()


class _MismatchedCodeFormerRunner:
    def run_with_progress(self, argv, **_kwargs) -> int:
        out_dir = Path(argv[argv.index("--output_path") + 1]) / "final_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_png(out_dir / "99999999.png", 250)
        return 0


def test_face_restore_naturalize_requires_name_match_and_skips_position_fallback(tmp_path):
    codeformer = _fake_codeformer(tmp_path)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    selected_dir = tmp_path / "selected"
    raw_dir = tmp_path / "raw"
    for path in (input_dir, output_dir, selected_dir, raw_dir):
        path.mkdir()
    _write_png(input_dir / "00000000.png", 10)
    logs: list[str] = []

    count = pipeline._run_face_restore_stage(
        runner=_MismatchedCodeFormerRunner(),
        codeformer=codeformer,
        input_dir=input_dir,
        output_dir=output_dir,
        selected_input_dir=selected_dir,
        raw_output_dir=raw_dir,
        fps=Fraction(1, 1),
        chunk_idx=0,
        start_sec=0.0,
        total_chunks=1,
        progress=None,
        cancel_event=None,
        gpu_gate=None,
        log=logs.append,
        tag="[chunk 1/1]",
        options=pipeline.FaceRestoreOptions(blend=0.5),
    )

    assert count == 1
    assert np.all(_read_png(output_dir / "00000000.png") == 10)
    assert "naturalize unmatched frames=1" in "\n".join(logs)


def test_build_codeformer_args_and_env(tmp_path):
    argv = stages.build_codeformer_args(
        python=tmp_path / "python.exe",
        script=tmp_path / "CodeFormer" / "inference_codeformer.py",
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        fidelity=0.8,
    )
    assert argv[0].endswith("python.exe")
    assert "--input_path" in argv
    assert "--output_path" in argv
    assert argv[argv.index("-w") + 1] == "0.800"
    assert argv[argv.index("--upscale") + 1] == "1"
    assert argv[argv.index("--bg_upsampler") + 1] == "None"
    env = stages.build_codeformer_env({"PATH": "x", "SECRET_TOKEN": "do-not-pass"})
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert "SECRET_TOKEN" not in env


def test_build_codeformer_env_adds_python_prefix_paths(tmp_path):
    env_root = tmp_path / "env"
    for rel in ("bin", "Library/bin", "Scripts"):
        (env_root / rel).mkdir(parents=True)
    py = env_root / "python.exe"
    py.write_bytes(b"py")
    base_path = os.pathsep.join(["x", str(env_root / "bin")])

    env = stages.build_codeformer_env(
        {"PATH": base_path, "SECRET_TOKEN": "do-not-pass", "CUDA_PATH": "C:\\cuda-custom"},
        python=py,
    )

    path_parts = env["PATH"].split(os.pathsep)
    expected_prefix_parts = [
        str(env_root / "bin"),
        str(env_root / "Library" / "bin"),
        str(env_root / "Scripts"),
        str(env_root),
    ]
    assert path_parts[:4] == expected_prefix_parts
    assert path_parts.count(str(env_root / "bin")) == 1
    assert "x" in path_parts
    assert env["CONDA_PREFIX"] == str(env_root)
    assert env["CUDA_PATH"] == "C:\\cuda-custom"
    assert "SECRET_TOKEN" not in env


def test_build_codeformer_env_sets_cuda_path_when_missing_and_handles_scripts_python(tmp_path):
    env_root = tmp_path / "env"
    scripts = env_root / "Scripts"
    scripts.mkdir(parents=True)
    py = scripts / "python.exe"
    py.write_bytes(b"py")

    env = stages.build_codeformer_env({"PATH": ""}, python=py)

    assert env["CONDA_PREFIX"] == str(env_root)
    assert env["CUDA_PATH"] == str(env_root)
    assert env["PATH"].split(os.pathsep)[0] == str(scripts)


def test_check_codeformer_local_validates_required_files(tmp_path):
    repo = tmp_path / "CodeFormer"
    repo.mkdir()
    (repo / "inference_codeformer.py").write_text("print('ok')\n", encoding="utf-8")
    py = tmp_path / "env" / "python.exe"
    py.parent.mkdir()
    py.write_bytes(b"py")
    model_paths = {
        "codeformer": repo / "weights" / "CodeFormer" / "codeformer.pth",
        "facelib_detection": repo / "weights" / "facelib" / "detection_Resnet50_Final.pth",
        "facelib_parsing": repo / "weights" / "facelib" / "parsing_parsenet.pth",
    }
    model_entries = {}
    for key, p in model_paths.items():
        body = key.encode("ascii")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
        model_entries[key] = [{"path": str(p), "sha256": hashlib.sha256(body).hexdigest()}]
    cfg = tmp_path / "enhance-binaries.json"
    cfg.write_text(
        json.dumps(
            {
                "roots": {"codeformer-local": [str(repo)]},
                "executables": {"codeformer-python": str(py)},
                "models": model_entries,
            }
        ),
        encoding="utf-8",
    )

    resolved = deps.check_codeformer_local(local_config_path=cfg)

    assert resolved.repo_root == repo.resolve()
    assert resolved.python == py.resolve()
    assert resolved.script == (repo / "inference_codeformer.py").resolve()
    assert set(resolved.models) == set(config.CODEFORMER_REQUIRED_MODEL_KEYS)


def test_check_codeformer_local_missing_weight_fails(tmp_path):
    repo = tmp_path / "CodeFormer"
    repo.mkdir()
    (repo / "inference_codeformer.py").write_text("print('ok')\n", encoding="utf-8")
    py = tmp_path / "python.exe"
    py.write_bytes(b"py")
    cfg = tmp_path / "enhance-binaries.json"
    cfg.write_text(
        json.dumps(
            {
                "roots": {"codeformer-local": [str(repo)]},
                "executables": {"codeformer-python": str(py)},
                "models": {"codeformer": [{"path": str(tmp_path / "missing.pth")}]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(deps.MissingDependencyError):
        deps.check_codeformer_local(local_config_path=cfg)


def test_pipeline_face_restore_runs_before_interpolate(workspace, fake_run, tmp_path):
    codeformer = _fake_codeformer(tmp_path)
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=_decisions(face_restore=True, interpolate=True),
        meta=FakeMeta(duration_sec=10.0),
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        rife=Path(r"C:\bin\rife-ncnn-vulkan.exe"),
        codeformer=codeformer,
        face_restore_start_sec=0,
        face_restore_end_sec=10,
    )
    face_idx = next(
        i for i, c in enumerate(fake_run.calls) if "inference_codeformer.py" in " ".join(c)
    )
    rife_idx = next(i for i, c in enumerate(fake_run.calls) if "rife" in c[0].lower())
    assert face_idx < rife_idx
    rife_call = fake_run.calls[rife_idx]
    assert "frames_face" in rife_call[rife_call.index("-i") + 1]
    face_env = fake_run.envs[face_idx]
    assert face_env is not None
    codeformer_prefix = codeformer.python.parent
    assert face_env["CONDA_PREFIX"] == str(codeformer_prefix)
    assert face_env["CUDA_PATH"] == str(codeformer_prefix)
    face_path_parts = face_env["PATH"].split(os.pathsep)
    assert face_path_parts[:4] == [
        str(codeformer_prefix / "bin"),
        str(codeformer_prefix / "Library" / "bin"),
        str(codeformer_prefix / "Scripts"),
        str(codeformer_prefix),
    ]
    rife_env = fake_run.envs[rife_idx]
    assert rife_env is None or str(codeformer_prefix) not in rife_env.get("PATH", "")


def test_pipeline_face_restore_skips_chunk_outside_range(workspace, fake_run, tmp_path):
    codeformer = _fake_codeformer(tmp_path)
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=_decisions(face_restore=True, interpolate=False),
        meta=FakeMeta(duration_sec=10.0),
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        codeformer=codeformer,
        face_restore_start_sec=30,
        face_restore_end_sec=40,
    )
    assert not any("inference_codeformer.py" in " ".join(c) for c in fake_run.calls)


def test_pipeline_face_restore_logs_selected_and_restored_counts(workspace, fake_run, tmp_path):
    codeformer = _fake_codeformer(tmp_path)
    logs: list[str] = []
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=_decisions(face_restore=True, interpolate=False),
        meta=FakeMeta(duration_sec=10.0),
        chunk_seconds=10,
        log=logs.append,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        codeformer=codeformer,
        face_restore_start_sec=0,
        face_restore_end_sec=10,
    )

    joined = "\n".join(logs)
    assert "face_restore selected" in joined
    assert "face_restore running CodeFormer" in joined
    assert "face_restore restored" in joined


def test_frame_in_face_restore_range_one_sided_bounds():
    fps = Fraction(10, 1)
    assert pipeline._frame_in_face_restore_range(
        frame_index=20,
        fps=fps,
        chunk_start_sec=0,
        start_sec=2.0,
        end_sec=None,
    )
    assert not pipeline._frame_in_face_restore_range(
        frame_index=19,
        fps=fps,
        chunk_start_sec=0,
        start_sec=2.0,
        end_sec=None,
    )
    assert pipeline._frame_in_face_restore_range(
        frame_index=19,
        fps=fps,
        chunk_start_sec=0,
        start_sec=None,
        end_sec=2.0,
    )
    assert not pipeline._frame_in_face_restore_range(
        frame_index=20,
        fps=fps,
        chunk_start_sec=0,
        start_sec=None,
        end_sec=2.0,
    )


def test_runtime_estimator_includes_face_restore():
    decs = _decisions(face_restore=True)
    est = __import__(
        "video_enhance_lite.runtime_estimator",
        fromlist=["estimate_runtime"],
    ).estimate_runtime(FakeMeta(), decs)
    assert "face_restore" in est.per_stage_seconds
