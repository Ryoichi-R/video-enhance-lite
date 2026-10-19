from __future__ import annotations

from pathlib import Path


from video_enhance_lite import config, pipeline, stages
from video_enhance_lite.decisions import StageDecision

# 共通 helper を再利用 (workspace / fake_run fixture は conftest.py 由来で import 不要)
from tests.enhance.test_pipeline import FakeMeta, _make_decisions  # noqa: F401
from tests.support.process_runner import FakeProcessRunner


def test_pre_filter_chain_excludes_stabilize_when_flag_set(tmp_path):
    """build_pre_filter_chain(exclude_stabilize=True) で vidstabtransform が除外される。"""
    decs = []
    for k in config.STAGE_KEYS:
        params = {}
        if k == "encode":
            params = {"crf": config.DEFAULT_ENCODE_CRF, "preset": config.DEFAULT_ENCODE_PRESET}
        decs.append(
            StageDecision(
                key=k,
                label_ja=config.STAGE_LABELS_JA[k],
                recommended=(k == "stabilize"),
                confidence="medium",
                reason="test",
                enabled=(k == "stabilize"),
                params=params,
            )
        )
    # exclude_stabilize=False (default): vidstabtransform が含まれる
    chain_default = stages.build_pre_filter_chain(decs, transforms_path=tmp_path / "t.trf")
    assert "vidstabtransform" in chain_default

    # exclude_stabilize=True: スニペット除外
    chain_excluded = stages.build_pre_filter_chain(
        decs,
        transforms_path=tmp_path / "t.trf",
        exclude_stabilize=True,
    )
    assert "vidstabtransform" not in chain_excluded


def test_run_pipeline_stabilize_backend_opencv_skips_vidstab(workspace, fake_run, monkeypatch):
    """stabilize_backend=opencv なら vidstabdetect / vidstabtransform を ffmpeg に渡さない。"""
    decs = _make_decisions(stabilize=True)

    # opencv backend で stabilize_cv が呼ばれないように差し替え (出力ディレクトリを作って fake)
    def _fake_stab(frames_in, frames_out, **kwargs):
        Path(frames_out).mkdir(parents=True, exist_ok=True)
        # 入力をコピーするだけ
        for p in Path(frames_in).iterdir():
            (Path(frames_out) / p.name).write_bytes(p.read_bytes())

    monkeypatch.setattr(pipeline.stabilize_cv, "stabilize_chunk", _fake_stab)

    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=decs,
        meta=FakeMeta(duration_sec=10.0),
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        stabilize_backend="opencv",
    )
    # vidstab 関連が一切呼ばれていないこと
    for c in fake_run.calls:
        joined = " ".join(c)
        assert "vidstabdetect" not in joined
        assert "vidstabtransform" not in joined


def test_run_pipeline_stabilize_backend_vidstab_keeps_filter(workspace, fake_run):
    decs = _make_decisions(stabilize=True)
    pipeline.run_pipeline(
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=decs,
        meta=FakeMeta(duration_sec=10.0),
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        stabilize_backend="vidstab",
        runner=fake_run,
    )
    # vidstabdetect / vidstabtransform が含まれている
    det = [c for c in fake_run.calls if any("vidstabdetect" in t for t in c)]
    trf = [c for c in fake_run.calls if any("vidstabtransform" in t for t in c)]
    assert len(det) == 1
    assert len(trf) == 1


class _SingleFrameExtractRunner(FakeProcessRunner):
    """extract 出力を 1 枚に削り、端数チャンク (フレーム < 2) を再現する。"""

    def run_with_progress(self, argv, **kwargs):
        rc = super().run_with_progress(argv, **kwargs)
        last = Path(list(argv)[-1])
        if "%08d.png" in str(last) and "frames_in" in str(last):
            extra = last.parent / "00000001.png"
            if extra.exists():
                extra.unlink()
        return rc


def test_run_pipeline_stabilize_opencv_skips_single_frame_chunk(workspace, monkeypatch):
    """P2-3 回帰: フレーム 1 枚のチャンクでは stabilize_cv を呼ばず完走する
    (旧実装は ValueError -> PipelineError で全体失敗していた)。"""

    def _fail_stab(frames_in, frames_out, **kwargs):
        raise AssertionError("stabilize_chunk must not be called for a single-frame chunk")

    monkeypatch.setattr(pipeline.stabilize_cv, "stabilize_chunk", _fail_stab)

    logs: list[str] = []
    pipeline.run_pipeline(
        runner=_SingleFrameExtractRunner(),
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=_make_decisions(stabilize=True),
        meta=FakeMeta(duration_sec=10.0),
        chunk_seconds=10,
        log=logs.append,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        stabilize_backend="opencv",
    )
    assert any("stabilize skipped" in m for m in logs)


def test_run_pipeline_stabilize_vidstab_skips_single_frame_chunk(workspace):
    """vidstab backend でも同条件でスキップする (detect / transform とも呼ばれない)。"""
    runner = _SingleFrameExtractRunner()
    pipeline.run_pipeline(
        runner=runner,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=_make_decisions(stabilize=True),
        meta=FakeMeta(duration_sec=10.0),
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        stabilize_backend="vidstab",
    )
    for c in runner.calls:
        joined = " ".join(c)
        assert "vidstabdetect" not in joined
        assert "vidstabtransform" not in joined


def test_run_pipeline_interpolate_skips_single_frame_chunk(workspace):
    """P2-3: フレーム 1 枚のチャンクでは RIFE を呼ばず pass-through で完走する
    (補間ペアが成立しないため。スキップはログで可視化される)。"""
    runner = _SingleFrameExtractRunner()
    logs: list[str] = []
    pipeline.run_pipeline(
        runner=runner,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=_make_decisions(interpolate=True),
        meta=FakeMeta(duration_sec=10.0),
        chunk_seconds=10,
        log=logs.append,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        rife=Path(r"C:\bin\rife-ncnn-vulkan.exe"),
    )
    assert any("interpolate skipped" in m for m in logs)
    for c in runner.calls:
        assert "rife" not in Path(c[0]).name.lower()


def test_run_pipeline_interpolate_runs_for_multi_frame_chunk(workspace):
    """回帰: 通常チャンク (フレーム >= 2) では従来どおり RIFE が呼ばれる。"""
    runner = FakeProcessRunner()
    pipeline.run_pipeline(
        runner=runner,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=_make_decisions(interpolate=True),
        meta=FakeMeta(duration_sec=10.0),
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        rife=Path(r"C:\bin\rife-ncnn-vulkan.exe"),
    )
    rife_calls = [c for c in runner.calls if "rife" in Path(c[0]).name.lower()]
    assert len(rife_calls) == 1


def test_run_pipeline_stabilize_off_does_not_require_vidstab(workspace, monkeypatch):
    """stabilize OFF + ffmpeg=None でも check_ffmpeg_with_vidstab を呼ばない (vidstab 不要)。"""
    from video_enhance_lite import deps

    called = {"with_vidstab": 0, "plain": 0}

    def _fake_check_ffmpeg_with_vidstab():
        called["with_vidstab"] += 1
        raise deps.MissingDependencyError("ffmpeg(vidstab)", "should not be called")

    def _fake_check_ffmpeg():
        called["plain"] += 1
        return deps.ResolvedBinary(path=Path(r"C:\bin\ffmpeg.exe"), version="x", sha256=None)

    def _fake_check_ffprobe():
        return deps.ResolvedBinary(path=Path(r"C:\bin\ffprobe.exe"), version="x", sha256=None)

    monkeypatch.setattr(pipeline.deps, "check_ffmpeg_with_vidstab", _fake_check_ffmpeg_with_vidstab)
    monkeypatch.setattr(pipeline.deps, "check_ffmpeg", _fake_check_ffmpeg)
    monkeypatch.setattr(pipeline.deps, "check_ffprobe", _fake_check_ffprobe)

    decs = _make_decisions()  # stabilize OFF (default)
    pipeline.run_pipeline(
        runner=FakeProcessRunner(),
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=decs,
        meta=FakeMeta(duration_sec=10.0),
        chunk_seconds=10,
        log=lambda _: None,
        stabilize_backend="auto",
        # ffmpeg/ffprobe を渡さず deps 解決を強制
    )
    # plain check が使われる (vidstab は呼ばれない)
    assert called["plain"] >= 1
    assert called["with_vidstab"] == 0
