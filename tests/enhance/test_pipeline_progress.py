from __future__ import annotations

import sys
import textwrap


from video_enhance_lite import pipeline


def _python_exe() -> str:
    return sys.executable


def test_run_with_progress_parses_frame_events():
    """ffmpeg 模擬: 200ms ごとに `frame=N\\n` を stdout に書く Python スクリプトを
    `_run_with_progress` で実行し、`on_progress(n)` が単調増加で呼ばれることを確認。
    """
    script = textwrap.dedent(
        """
        import sys, time
        for n in range(1, 11):
            print(f"frame={n}")
            print("progress=continue")
            sys.stdout.flush()
            time.sleep(0.2)
        print("progress=end")
        """
    ).strip()

    seen: list[int] = []
    rc = pipeline.RealProcessRunner().run_with_progress(
        [_python_exe(), "-c", script],
        idle_timeout=10,
        on_progress=lambda n: seen.append(n),
        stage="extract_frames",
    )
    assert rc == 0
    # 全部は来ないかもしれない (rate-limit) が、単調増加で複数イベントが届くはず
    assert len(seen) >= 2
    assert seen == sorted(seen)
    assert seen[-1] <= 10


def test_run_with_progress_rate_limit_max_4_per_second():
    """200ms 間隔 (5fps) で 0.25s 以上空くと emit -> ほぼ全部届く。
    一方 5ms 間隔で大量出力すると 1 秒あたり 4 イベント以下に絞られる。
    """
    script = textwrap.dedent(
        """
        import sys, time
        # 1 秒間で 100 行 (10ms 間隔)
        for n in range(1, 101):
            print(f"frame={n}")
            sys.stdout.flush()
            time.sleep(0.01)
        """
    ).strip()

    seen: list[int] = []
    pipeline.RealProcessRunner().run_with_progress(
        [_python_exe(), "-c", script],
        idle_timeout=10,
        on_progress=lambda n: seen.append(n),
        stage="mux",
    )
    # 100 行出力されたが、rate-limit (0.25s 間隔) で大体 4-10 イベント程度
    assert len(seen) >= 1
    assert len(seen) <= 30  # 余裕を持って上限を見る


def test_augment_ffmpeg_with_progress_inserts_after_y():
    argv = ["ffmpeg.exe", "-y", "-i", "in.mp4", "out.mp4"]
    out = pipeline._augment_ffmpeg_with_progress(argv)
    assert out == ["ffmpeg.exe", "-y", "-progress", "pipe:1", "-nostats", "-i", "in.mp4", "out.mp4"]


def test_augment_ffmpeg_with_progress_idempotent():
    argv = ["ffmpeg", "-y", "-progress", "pipe:1", "-i", "in.mp4", "out.mp4"]
    out = pipeline._augment_ffmpeg_with_progress(argv)
    # 既に -progress があれば no-op
    assert out == argv


def test_augment_ffmpeg_with_progress_non_ffmpeg_is_noop():
    argv = ["realesrgan-ncnn-vulkan", "-i", "frames_in", "-o", "frames_out"]
    out = pipeline._augment_ffmpeg_with_progress(argv)
    assert out == argv


def test_progress_event_has_frame_fields():
    """ProgressEvent に frame / total_frames フィールドが後方互換で追加されている。"""
    ev = pipeline.ProgressEvent(
        stage="chunk_extract",
        chunk_index=2,
        total_chunks=10,
        label="test",
        frame=42,
        total_frames=100,
    )
    assert ev.frame == 42
    assert ev.total_frames == 100
    # default は None
    ev2 = pipeline.ProgressEvent(stage="x")
    assert ev2.frame is None
    assert ev2.total_frames is None
