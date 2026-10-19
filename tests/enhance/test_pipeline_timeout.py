from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from video_enhance_lite import pipeline
from tests.enhance.test_pipeline import FakeMeta, _make_decisions
from tests.support.process_runner import FakeProcessRunner


def _python_exe() -> str:
    return sys.executable


# ---------- _run_short ----------


def test_run_short_success():
    cp = pipeline._run_short(
        [_python_exe(), "-c", "print('hello')"],
        timeout=10,
        stage="probe",
    )
    assert cp.returncode == 0
    assert "hello" in cp.stdout


def test_run_short_timeout_raises_pipeline_error():
    # 5s sleep, but timeout=1
    with pytest.raises(pipeline.PipelineError) as exc:
        pipeline._run_short(
            [_python_exe(), "-c", "import time; time.sleep(5)"],
            timeout=1,
            stage="probe",
        )
    assert exc.value.stage == "probe"
    assert "timeout" in exc.value.message.lower()


def test_run_short_nonzero_exit_raises():
    with pytest.raises(subprocess.CalledProcessError):
        # _run_short は subprocess.run(check=True) なので非 0 終了で CalledProcessError
        pipeline._run_short(
            [_python_exe(), "-c", "import sys; sys.exit(2)"],
            timeout=10,
            stage="probe",
        )


# ---------- _run_with_progress ----------


def test_run_with_progress_success_quick():
    """short-lived command で _run_with_progress が正常完了する。"""
    rc = pipeline.RealProcessRunner().run_with_progress(
        [_python_exe(), "-c", "print('ok')"],
        idle_timeout=10,
        stage="ffmpeg",
    )
    assert rc == 0


def test_run_with_progress_idle_timeout_kills():
    """sleep が idle_timeout より長いと kill + PipelineError。"""
    start = time.monotonic()
    with pytest.raises(pipeline.PipelineError) as exc:
        pipeline.RealProcessRunner().run_with_progress(
            # 出力なしで sleep する
            [_python_exe(), "-c", "import time; time.sleep(10)"],
            idle_timeout=2,
            stage="ffmpeg",
            chunk_idx=3,
        )
    elapsed = time.monotonic() - start
    # idle_timeout=2 で kill されるので 10 秒は待たない (許容 5 秒)
    assert elapsed < 5.5
    assert exc.value.stage == "ffmpeg"
    assert exc.value.chunk_idx == 3
    assert "idle timeout" in exc.value.message
    assert "I/O無出力タイムアウト（秒）" in exc.value.message
    assert "--io-idle-timeout-sec" in exc.value.message


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
def test_run_with_progress_output_keeps_long_process_alive(stream_name):
    """総実行時間がtimeoutを超えても、0.5秒間隔の出力があれば完走する。"""
    target = "sys.stdout" if stream_name == "stdout" else "sys.stderr"
    script = (
        "import sys, time\n"
        f"stream = {target}\n"
        "for i in range(9):\n"
        "    print(f'alive {i}', file=stream, flush=True)\n"
        "    time.sleep(0.5)\n"
    )
    start = time.monotonic()
    rc = pipeline.RealProcessRunner().run_with_progress(
        [_python_exe(), "-c", script],
        idle_timeout=3,
        stage="interpolate",
    )
    elapsed = time.monotonic() - start
    assert rc == 0
    assert elapsed >= 4.0


def test_run_with_progress_heartbeat_fires_during_dense_output():
    """queue.Emptyにならない高頻度出力中も表示用heartbeatを発火する。"""
    script = (
        "import time\n"
        "for i in range(30):\n"
        "    print(f'alive {i}', flush=True)\n"
        "    time.sleep(0.02)\n"
    )
    heartbeats: list[tuple[float, float]] = []
    rc = pipeline.RealProcessRunner().run_with_progress(
        [_python_exe(), "-c", script],
        idle_timeout=2,
        on_heartbeat=lambda elapsed, idle: heartbeats.append((elapsed, idle)),
        heartbeat_interval_sec=0.1,
        stage="interpolate",
    )
    assert rc == 0
    assert heartbeats
    assert all(idle < 1.0 for _elapsed, idle in heartbeats)


def test_run_with_progress_nonzero_preserves_terminal_error_after_verbose_noise():
    """長いRIFE verbose相当行の後でも、終了直前の実エラーがtailに残る。"""
    marker = "vkAllocateMemory failed: diagnostic-marker"
    script = (
        "import sys\n"
        "line = 'C:/' + ('very-long-frame-path/' * 16) + ' done\\n'\n"
        "for _ in range(80):\n"
        "    sys.stderr.write(line)\n"
        f"sys.stderr.write('{marker}\\n')\n"
        "sys.stderr.write('cleanup after failure\\n')\n"
        "sys.stderr.flush()\n"
        "sys.exit(7)\n"
    )
    with pytest.raises(pipeline.PipelineError) as exc:
        pipeline.RealProcessRunner().run_with_progress(
            [_python_exe(), "-c", script],
            idle_timeout=10,
            stage="interpolate",
        )
    assert marker in exc.value.message
    assert len(exc.value.message) <= 8192


def test_run_with_progress_cancel_event_kills():
    """cancel_event.set() で速やかに PipelineCancelled。"""
    ev = threading.Event()

    def _set_after():
        time.sleep(0.5)
        ev.set()

    t = threading.Thread(target=_set_after, daemon=True)
    t.start()

    start = time.monotonic()
    with pytest.raises(pipeline.PipelineCancelled) as exc:
        pipeline.RealProcessRunner().run_with_progress(
            [_python_exe(), "-c", "import time; time.sleep(10)"],
            idle_timeout=30,
            cancel_event=ev,
            stage="upscale_frames",
            chunk_idx=2,
        )
    elapsed = time.monotonic() - start
    assert elapsed < 5.0
    assert exc.value.stage == "upscale_frames"
    assert exc.value.chunk_idx == 2


def test_run_with_progress_nonzero_exit_raises_pipeline_error():
    with pytest.raises(pipeline.PipelineError) as exc:
        pipeline.RealProcessRunner().run_with_progress(
            [_python_exe(), "-c", "import sys; sys.exit(3)"],
            idle_timeout=10,
            stage="pre_filter",
        )
    assert exc.value.stage == "pre_filter"


def test_run_with_progress_emits_carriage_return_output_events():
    """CodeFormer/tqdm 形式の '\\r' 更新も output event として拾える。"""
    events: list[tuple[str, str]] = []
    script = (
        "import sys, time\n"
        "sys.stderr.write('restore 1/2\\r'); sys.stderr.flush()\n"
        "time.sleep(0.1)\n"
        "sys.stderr.write('restore 2/2\\r'); sys.stderr.flush()\n"
    )

    rc = pipeline.RealProcessRunner().run_with_progress(
        [_python_exe(), "-c", script],
        idle_timeout=10,
        stage="face_restore",
        on_output_event=lambda tag, text: events.append((tag, text)),
    )

    assert rc == 0
    assert any(tag == "err" and "restore 1/2" in text for tag, text in events)
    assert any(tag == "err" and "restore 2/2" in text for tag, text in events)


def test_run_with_progress_heartbeat_does_not_extend_idle_timeout():
    """ハートビートは UI 表示専用で、無出力プロセスの idle timeout を延命しない。"""
    heartbeats: list[tuple[float, float]] = []
    start = time.monotonic()

    with pytest.raises(pipeline.PipelineError) as exc:
        pipeline.RealProcessRunner().run_with_progress(
            [_python_exe(), "-c", "import time; time.sleep(10)"],
            idle_timeout=2,
            stage="face_restore",
            on_heartbeat=lambda elapsed, idle: heartbeats.append((elapsed, idle)),
            heartbeat_interval_sec=0.25,
        )

    elapsed = time.monotonic() - start
    assert elapsed < 5.5
    assert heartbeats
    assert exc.value.stage == "face_restore"
    assert "idle timeout" in exc.value.message


def test_run_gpu_stage_interpolate_wires_low_frequency_heartbeat_only():
    captured: dict = {}
    logs: list[str] = []

    class _CapturingRunner:
        def run_with_progress(self, argv, **kwargs):
            captured.update(kwargs)
            kwargs["on_heartbeat"](17.0, 2.0)
            return 0

    pipeline._run_gpu_stage(
        ["rife.exe", "-v"],
        runner=_CapturingRunner(),
        stage="interpolate",
        chunk_idx=0,
        cancel_event=None,
        io_idle_timeout_sec=120,
        gpu_gate=None,
        log=logs.append,
    )
    assert callable(captured["on_heartbeat"])
    assert captured["heartbeat_interval_sec"] == 15.0
    assert "on_output_event" not in captured
    assert logs == ["       interpolate running... elapsed=17s idle=2s"]


def test_run_pipeline_interpolate_forwards_log_to_heartbeat(workspace):
    """本番の chunk→RIFE 呼び出しでも heartbeat がログへ届く。"""
    logs: list[str] = []

    class _HeartbeatRunner(FakeProcessRunner):
        def run_with_progress(self, argv, **kwargs):
            rc = super().run_with_progress(argv, **kwargs)
            if "rife" in str(argv[0]).lower():
                kwargs["on_heartbeat"](17.0, 2.0)
            return rc

    pipeline.run_pipeline(
        runner=_HeartbeatRunner(),
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=_make_decisions(interpolate=True),
        meta=FakeMeta(duration_sec=10.0),
        chunk_seconds=10,
        log=logs.append,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        rife=Path(r"C:\bin\rife-ncnn-vulkan.exe"),
        worker_count=1,
    )

    assert "       interpolate running... elapsed=17s idle=2s" in logs


# ---------- PR-R1: Popen 起動失敗の包装 ----------


def test_run_with_progress_launch_failure_wrapped_as_pipeline_error():
    """存在しない実行ファイルでの Popen 起動失敗が PipelineError に包まれる。

    逐次実行経路では CLI 側 except が PipelineError 系しか拾わないため、生
    FileNotFoundError/OSError がスタックトレースのまま伝播しないことを保証する。
    `run_pipeline` 経由ではなく `RealProcessRunner().run_with_progress` を直呼びする
    (FakeProcessRunner は実 Popen を起動しないため起動失敗を検証できない)。
    """
    missing = "this_executable_should_not_exist_vel_12345"
    with pytest.raises(pipeline.PipelineError) as exc:
        pipeline.RealProcessRunner().run_with_progress(
            [missing, "--frobnicate"],
            idle_timeout=10,
            stage="extract_frames",
            chunk_idx=0,
        )
    assert exc.value.stage == "extract_frames"
    assert exc.value.chunk_idx == 0
    assert "failed to launch process" in exc.value.message
    assert missing in exc.value.message


# ---------- PR-R2: 密出力プロセス下での cancel 応答性 ----------


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_run_with_progress_cancel_responsive_under_dense_output(stream):
    """出力を密に出し続けるプロセスでも cancel_event で速やかに停止する (PR-R2)。

    出力が途切れない (= queue.Empty 分岐に入らない) ため、旧実装では cancel が
    立っても次の出力途切れまで kill されず、GUI のキャンセル体感が悪化していた。
    queue.get 成功側でも cancel を確認することで、密出力下でも数秒以内に
    PipelineCancelled になることを保証する。`run_pipeline` 経由ではなく
    `RealProcessRunner().run_with_progress` を直呼びして実監視ループを直接検証する。
    """
    target = "sys.stdout" if stream == "stdout" else "sys.stderr"
    # 20 秒間、5ms 間隔で出力し続ける。cancel が効かなければテストは ~20 秒かかる。
    script = (
        "import sys, time\n"
        "end = time.monotonic() + 20\n"
        "while time.monotonic() < end:\n"
        f"    {target}.write('noise line\\n'); {target}.flush()\n"
        "    time.sleep(0.005)\n"
    )
    ev = threading.Event()

    def _set_after():
        time.sleep(0.5)
        ev.set()

    t = threading.Thread(target=_set_after, daemon=True)
    t.start()

    start = time.monotonic()
    with pytest.raises(pipeline.PipelineCancelled) as exc:
        pipeline.RealProcessRunner().run_with_progress(
            [_python_exe(), "-c", script],
            idle_timeout=30,
            cancel_event=ev,
            stage="upscale_frames",
            chunk_idx=5,
        )
    elapsed = time.monotonic() - start
    # cancel は 0.5s 後に set。密出力でも数秒以内に停止すること
    # (terminate->wait(1s)->kill->wait(1s) の Windows マージンを考慮しても 4s 以内)。
    assert elapsed < 4.0, f"cancel response too slow: {elapsed:.2f}s"
    assert exc.value.stage == "upscale_frames"
    assert exc.value.chunk_idx == 5
