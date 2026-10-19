from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from video_enhance_lite import cli, pipeline

# 共通 helper を再利用 (workspace fixture は conftest.py 由来で import 不要)
from tests.enhance.test_pipeline import FakeMeta, _make_decisions  # noqa: F401


# realesrgan / rife を渡しても dep 解決が走らないようにダミーパスを使う
_REALESRGAN = Path(r"C:\bin\realesrgan-ncnn-vulkan.exe")
_RIFE = Path(r"C:\bin\rife-ncnn-vulkan.exe")


class GpuGateFakeRunner:
    """GPU 段 (realesrgan / rife) の同時実行数を観測する fake (ProcessRunner 準拠)。

    GPU 段に入るたびに active カウンタを増減し、観測した最大同時実行数 (`max_gpu`)
    を記録する。CPU 段 (ffmpeg) はカウントしない。各 GPU 段で短時間 sleep し、
    並列ワーカーが GPU 段にオーバーラップして到達できる窓を作る。

    V2-8 DI: `run_pipeline(..., runner=gpu_fake)` で注入する。`_run_gpu_stage` が
    `gpu_gate` セマフォを保持したまま `runner.run_with_progress(...)` を呼ぶため、
    同時実行観測は従来どおり成立する。
    """

    def __init__(self, *, gpu_sleep: float = 0.08) -> None:
        self.calls: list[list[str]] = []
        self._lock = threading.Lock()
        self._active = 0
        self.max_gpu = 0
        self._gpu_sleep = gpu_sleep

    def run_with_progress(
        self,
        argv,
        *,
        cwd=None,
        idle_timeout=None,
        cancel_event=None,
        on_progress=None,
        stage="ffmpeg",
        chunk_idx=None,
    ) -> int:
        with self._lock:
            self.calls.append(list(argv))
        bin_name = argv[0].lower()
        is_gpu = ("realesrgan" in bin_name) or ("rife" in bin_name)
        if is_gpu:
            with self._lock:
                self._active += 1
                if self._active > self.max_gpu:
                    self.max_gpu = self._active
            time.sleep(self._gpu_sleep)
            with self._lock:
                self._active -= 1
            # 出力 PNG を書いて _count_png() > 0 を満たす
            if "-o" in argv:
                out_dir = Path(argv[argv.index("-o") + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "00000000.png").write_bytes(b"")
        elif "ffmpeg" in bin_name:
            last = Path(argv[-1])
            if last.suffix.lower() == ".mp4":
                last.parent.mkdir(parents=True, exist_ok=True)
                last.write_bytes(b"")
            elif "%08d.png" in str(last):
                last.parent.mkdir(parents=True, exist_ok=True)
                (last.parent / "00000000.png").write_bytes(b"")
        return 0


@pytest.fixture
def gpu_fake():
    return GpuGateFakeRunner()


def _run(
    workspace,
    runner,
    *,
    worker_count,
    gpu_concurrency,
    duration_sec=40.0,
    chunk_seconds=10,
    log=None,
):
    decs = _make_decisions(upscale=True, interpolate=True)
    pipeline.run_pipeline(
        runner=runner,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=decs,
        meta=FakeMeta(duration_sec=duration_sec),
        chunk_seconds=chunk_seconds,
        log=log or (lambda _m: None),
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        realesrgan=_REALESRGAN,
        rife=_RIFE,
        worker_count=worker_count,
        gpu_concurrency=gpu_concurrency,
    )


def test_gpu_concurrency_2_allows_two_overlap(workspace, gpu_fake):
    """gpu_concurrency=2, worker_count=4 で GPU 段の同時実行が最大 2 になる。"""
    _run(workspace, gpu_fake, worker_count=4, gpu_concurrency=2, duration_sec=40.0)
    assert gpu_fake.max_gpu == 2


def test_gpu_concurrency_1_serializes_gpu(workspace, gpu_fake):
    """gpu_concurrency=1 (既定) は worker>1 でも GPU 段がオーバーラップしない (既存挙動)。"""
    _run(workspace, gpu_fake, worker_count=4, gpu_concurrency=1, duration_sec=40.0)
    assert gpu_fake.max_gpu == 1


def test_gpu_concurrency_never_exceeds_cap(workspace, gpu_fake):
    """gpu_concurrency=3, worker_count=5 でも GPU 同時実行は 3 を超えない。"""
    _run(workspace, gpu_fake, worker_count=5, gpu_concurrency=3, duration_sec=50.0)
    assert gpu_fake.max_gpu <= 3


def test_gpu_concurrency_clamped_to_worker_count_with_warn(workspace, gpu_fake):
    """gpu_concurrency=8, worker_count=2 -> effective_gpu=2 に clamp + WARN ログ。"""
    logs: list[str] = []
    _run(workspace, gpu_fake, worker_count=2, gpu_concurrency=8, duration_sec=20.0, log=logs.append)
    # GPU 同時実行は worker 数 (2) に丸められ、それを超えない
    assert gpu_fake.max_gpu <= 2
    assert any("丸めます" in m and "gpu-concurrency=8" in m for m in logs)


def test_gpu_concurrency_1_no_warn(workspace, gpu_fake):
    """既定 gpu_concurrency=1 では clamp WARN を出さない。"""
    logs: list[str] = []
    _run(workspace, gpu_fake, worker_count=2, gpu_concurrency=1, duration_sec=20.0, log=logs.append)
    assert not any("gpu-concurrency" in m for m in logs)


# ---------- CLI フラグ ----------


def test_cli_gpu_concurrency_default_is_1():
    parser = cli._build_parser()
    args = parser.parse_args(["dummy.mp4"])
    assert args.gpu_concurrency == 1


def test_cli_gpu_concurrency_explicit_value():
    parser = cli._build_parser()
    args = parser.parse_args(["dummy.mp4", "--gpu-concurrency", "2"])
    assert args.gpu_concurrency == 2


def test_cli_gpu_concurrency_rejects_zero():
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["dummy.mp4", "--gpu-concurrency", "0"])
