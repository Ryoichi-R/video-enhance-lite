from __future__ import annotations

from pathlib import Path


from video_enhance_lite import pipeline

# 共通 helper を再利用 (workspace / fake_run fixture は conftest.py 由来で import 不要)
from tests.enhance.test_pipeline import FakeMeta, _make_decisions  # noqa: F401


_REALESRGAN = Path(r"C:\bin\realesrgan-ncnn-vulkan.exe")
_RIFE = Path(r"C:\bin\rife-ncnn-vulkan.exe")


def _run(workspace, fake_run, **kw):
    decs = _make_decisions(upscale=True, interpolate=True)
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=decs,
        meta=FakeMeta(duration_sec=10.0),
        chunk_seconds=10,
        log=lambda _m: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        realesrgan=_REALESRGAN,
        rife=_RIFE,
        **kw,
    )


def _realesrgan_call(calls):
    return next(c for c in calls if "realesrgan" in c[0].lower())


def _rife_call(calls):
    return next(c for c in calls if "rife" in c[0].lower())


def test_no_gpu_flags_argv_unchanged(workspace, fake_run):
    """フラグ未指定で realesrgan/rife argv に -t/-j/-g が一切入らない (回帰)。"""
    _run(workspace, fake_run)
    up = _realesrgan_call(fake_run.calls)
    rife = _rife_call(fake_run.calls)
    for flag in ("-t", "-j", "-g"):
        assert flag not in up
        assert flag not in rife


def test_gpu_tilesize_only_on_realesrgan(workspace, fake_run):
    """--gpu-tilesize は realesrgan の -t にのみ入り、RIFE には入らない。"""
    _run(workspace, fake_run, gpu_tilesize=512)
    up = _realesrgan_call(fake_run.calls)
    rife = _rife_call(fake_run.calls)
    assert up[up.index("-t") + 1] == "512"
    assert "-t" not in rife


def test_gpu_threads_propagates_to_both(workspace, fake_run):
    """--gpu-threads は realesrgan / rife 両方の -j に伝播。"""
    _run(workspace, fake_run, gpu_threads="2:4:4")
    up = _realesrgan_call(fake_run.calls)
    rife = _rife_call(fake_run.calls)
    assert up[up.index("-j") + 1] == "2:4:4"
    assert rife[rife.index("-j") + 1] == "2:4:4"


def test_gpu_id_propagates_to_both(workspace, fake_run):
    """共通 --gpu-id (>=0) は realesrgan / rife 両方の -g に入る。"""
    _run(workspace, fake_run, gpu_id=1)
    up = _realesrgan_call(fake_run.calls)
    rife = _rife_call(fake_run.calls)
    assert up[up.index("-g") + 1] == "1"
    assert rife[rife.index("-g") + 1] == "1"


def test_rife_gpu_id_overrides_only_rife(workspace, fake_run):
    """--rife-gpu-id は RIFE の -g のみ上書きし、realesrgan には -1 を渡さない。"""
    _run(workspace, fake_run, gpu_id=0, rife_gpu_id=-1)
    up = _realesrgan_call(fake_run.calls)
    rife = _rife_call(fake_run.calls)
    # realesrgan は共通 gpu_id=0 のまま (-1 は渡さない)
    assert up[up.index("-g") + 1] == "0"
    # RIFE は -1 (cpu) に上書き
    assert rife[rife.index("-g") + 1] == "-1"


def test_rife_gpu_id_alone_does_not_touch_realesrgan(workspace, fake_run):
    """--rife-gpu-id のみ指定 (gpu_id 未指定) では realesrgan に -g が入らない。"""
    _run(workspace, fake_run, rife_gpu_id=0)
    up = _realesrgan_call(fake_run.calls)
    rife = _rife_call(fake_run.calls)
    assert "-g" not in up
    assert rife[rife.index("-g") + 1] == "0"
