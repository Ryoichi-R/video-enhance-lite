from __future__ import annotations

import threading
from pathlib import Path

import pytest

from video_enhance_lite import pipeline

# 共通 helper を再利用 (workspace / fake_run fixture は conftest.py 由来で import 不要)
from tests.enhance.test_pipeline import FakeMeta, _make_decisions  # noqa: F401
from tests.support.process_runner import FakeProcessRunner


def test_worker_count_1_keeps_legacy_path(workspace, fake_run):
    """worker_count=1 (default) はシングルワーカー path で旧挙動と同じ。"""
    decs = _make_decisions()
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=decs,
        meta=FakeMeta(duration_sec=30.0),
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        worker_count=1,
    )
    # 3 chunks 処理されている
    mux = [
        c
        for c in fake_run.calls
        if "ffmpeg" in c[0].lower() and c[-1].endswith(".mp4") and "chunks" in c[-1]
    ]
    assert len(mux) == 3


def test_worker_count_2_processes_all_chunks(workspace, fake_run):
    """worker_count=2 で 4 chunk 全て処理される (順序保証 OK)。"""
    decs = _make_decisions()
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=decs,
        meta=FakeMeta(duration_sec=40.0),
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        worker_count=2,
    )
    mux = [
        c
        for c in fake_run.calls
        if "ffmpeg" in c[0].lower() and c[-1].endswith(".mp4") and "chunks" in c[-1]
    ]
    assert len(mux) == 4
    # 出力ファイル名は順序保証 (chunk_000, chunk_001, chunk_002, chunk_003)
    chunk_names = sorted({Path(c[-1]).name for c in mux})
    assert chunk_names == ["chunk_000.mp4", "chunk_001.mp4", "chunk_002.mp4", "chunk_003.mp4"]


def test_worker_count_2_creates_per_chunk_dirs(workspace, fake_run):
    """worker_count=2 のとき、tmp 内に chunk_<idx>/ サブディレクトリが作られる。"""
    decs = _make_decisions()
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=decs,
        meta=FakeMeta(duration_sec=20.0),
        chunk_seconds=10,
        log=lambda _: None,
        keep_temp=True,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        worker_count=2,
    )
    from video_enhance_lite import config

    identity_root = config.derive_temp_dir(workspace / "in.mp4")
    run_leaves = list((identity_root / "runs").iterdir())
    assert len(run_leaves) == 1
    tmp_root = run_leaves[0]
    # ※ keep_temp=True で tmp が残る
    children = {p.name for p in tmp_root.iterdir() if p.is_dir()}
    assert "chunks" in children
    assert "chunk_000" in children
    assert "chunk_001" in children


def test_make_chunk_tmp_dirs_returns_per_chunk_paths(tmp_path):
    dirs = pipeline._make_chunk_tmp_dirs(tmp_path, 3, src_width=1280)
    for k in ("frames_in", "frames_pre", "frames_up", "frames_clamp", "frames_rife", "frames_out"):
        assert dirs[k].parent.name == "chunk_003"
    assert dirs["_src_width"] == 1280


def test_cli_worker_count_default_is_none():
    """CLI default は None (= main() で動的算出)。"""
    from video_enhance_lite import cli

    parser = cli._build_parser()
    args = parser.parse_args(["dummy.mp4"])
    assert args.worker_count is None


def test_cli_worker_count_explicit_value():
    from video_enhance_lite import cli

    parser = cli._build_parser()
    args = parser.parse_args(["dummy.mp4", "--worker-count", "2"])
    assert args.worker_count == 2


def test_worker_count_failure_cancels_others(workspace):
    """1 chunk が失敗したら他チャンクも cancel_event 経由で停止。"""
    decs = _make_decisions()
    failer = FakeProcessRunner(fail_on_argv_marker="frames_in")
    with pytest.raises(pipeline.PipelineError):
        pipeline.run_pipeline(
            runner=failer,
            input_path=workspace / "in.mp4",
            output_path=workspace / "out.mp4",
            decisions=decs,
            meta=FakeMeta(duration_sec=30.0),
            chunk_seconds=10,
            log=lambda _: None,
            ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
            ffprobe=Path(r"C:\bin\ffprobe.exe"),
            worker_count=3,
        )


def test_worker_failure_without_external_event_cancels_others(workspace):
    """P2-2 回帰: cancel_event を渡さなくても (CLI 旧配線相当)、1 チャンク失敗後に
    残チャンクが run_pipeline 内部の Event 経由で停止する (失敗後に完走しない)。

    旧実装は cancel_event=None だと set() されず、他チャンクは最後まで走っていた。
    タイミング依存を避けるため、他チャンクは「chunk_0 の失敗発生後、cancel が
    伝播する (または猶予切れ) まで待ってから」cancel を判定する。
    """
    import time

    decs = _make_decisions()

    class FailFirstChunkRunner(FakeProcessRunner):
        """chunk_000 の extract を失敗させ、他チャンクは cancel 伝播を待って判定する。"""

        def __init__(self):
            super().__init__()
            self.failure_raised = threading.Event()

        def run_with_progress(
            self,
            argv,
            *,
            cwd=None,
            idle_timeout=pipeline.DEFAULT_IO_IDLE_TIMEOUT_SEC,
            cancel_event=None,
            on_progress=None,
            stage="ffmpeg",
            chunk_idx=None,
        ):
            if stage == "extract_frames" and chunk_idx == 0:
                self.failure_raised.set()
                raise pipeline.PipelineError(stage, "boom", chunk_idx=chunk_idx)
            if chunk_idx is not None:
                # chunk_0 の失敗 -> main thread が観測して cancel を set するまで待つ
                self.failure_raised.wait(timeout=5)
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    if cancel_event is not None and cancel_event.is_set():
                        # RealProcessRunner と同じ契約: cancel 観測で PipelineCancelled
                        raise pipeline.PipelineCancelled(stage, chunk_idx=chunk_idx)
                    time.sleep(0.01)
                # 猶予内に cancel が来ない = 旧挙動 (伝播なし) -> 完走させて検出する
            return super().run_with_progress(
                argv,
                cwd=cwd,
                idle_timeout=idle_timeout,
                cancel_event=cancel_event,
                on_progress=on_progress,
                stage=stage,
                chunk_idx=chunk_idx,
            )

    runner = FailFirstChunkRunner()
    with pytest.raises((pipeline.PipelineError, pipeline.PipelineCancelled)):
        pipeline.run_pipeline(
            runner=runner,
            input_path=workspace / "in.mp4",
            output_path=workspace / "out.mp4",
            decisions=decs,
            meta=FakeMeta(duration_sec=30.0),  # 3 chunks
            chunk_seconds=10,
            log=lambda _: None,
            # cancel_event は渡さない (旧 CLI 相当) -> 内部 Event が機能すること
            ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
            ffprobe=Path(r"C:\bin\ffprobe.exe"),
            worker_count=2,
        )
    # cancel が伝播していれば、chunk_1/chunk_2 は extract で Cancelled となり
    # mux (チャンク完走の証拠) は 1 つも作られない
    mux = [
        c
        for c in runner.calls
        if "ffmpeg" in c[0].lower() and c[-1].endswith(".mp4") and "chunks" in c[-1]
    ]
    assert mux == []


def test_worker_count_external_cancel_stops(workspace, fake_run):
    """V2-13: worker>1 実行中に外部 cancel_event を set すると PipelineCancelled で停止する。

    内部失敗起因の cancel (test_worker_count_failure_cancels_others) と異なり、
    GUI スレッド相当の外部から cancel_event を set するケースを検証する。
    最初のチャンク処理開始 (chunk_start) を合図に cancel し、並列でも停止すること。
    """
    ev = threading.Event()

    def progress(ep):
        # 最初のチャンク処理が始まった時点で外部 (GUI) が cancel した状況を模す
        if ep.stage == "chunk_start":
            ev.set()

    with pytest.raises(pipeline.PipelineCancelled):
        pipeline.run_pipeline(
            runner=fake_run,
            input_path=workspace / "in.mp4",
            output_path=workspace / "out.mp4",
            decisions=_make_decisions(),
            meta=FakeMeta(duration_sec=30.0),
            chunk_seconds=10,
            log=lambda _: None,
            cancel_event=ev,
            progress=progress,
            ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
            ffprobe=Path(r"C:\bin\ffprobe.exe"),
            worker_count=3,
        )
