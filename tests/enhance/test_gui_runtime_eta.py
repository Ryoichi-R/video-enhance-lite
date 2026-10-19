from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.gui

from video_enhance_lite import config, gui, pipeline


# _tk_session_root / tk_root は conftest.py の共有 fixture を使う
# (ファイルごとの tk.Tk() 多重生成による init.tcl ロード失敗を避ける)。


@pytest.fixture
def app(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    return gui.App(tk_root)


def test_chunk_done_shows_remaining_label_linear(app):
    # process_chunks で起点をセット → chunk_done を 3 回 → 実測線形外挿で「残り」表示
    app._run_started_monotonic = time.monotonic() - 10.0
    total = 10
    for i in range(3):
        app._handle_progress(
            pipeline.ProgressEvent(stage="chunk_done", chunk_index=i, total_chunks=total)
        )
    label = app.var_progress_label.get()
    assert "残り" in label
    assert "3 / 10" in label


def test_chunk_done_uses_prior_estimate_under_3_chunks(app, monkeypatch):
    # 先験値を差し込み、1 チャンク完了時に先験値ブレンド分岐が走ることを確認
    app._runtime_estimate_sec = 100.0
    app._run_started_monotonic = time.monotonic()
    app._chunk_done_times.clear()
    total = 10
    app._handle_progress(
        pipeline.ProgressEvent(stage="chunk_done", chunk_index=0, total_chunks=total)
    )
    label = app.var_progress_label.get()
    assert "残り" in label
    assert "1 / 10" in label


def test_process_chunks_sets_start_time(app):
    app._run_started_monotonic = None
    app._handle_progress(pipeline.ProgressEvent(stage="process_chunks", total_chunks=5))
    assert app._run_started_monotonic is not None
    assert app._chunk_done_times == []
    assert "0 / 5" in app.var_progress_label.get()


def test_chunk_done_fallback_initializes_start(app):
    # process_chunks を受信せずに chunk_done が来ても異常終了しない (fallback)
    app._run_started_monotonic = None
    app._chunk_done_times.clear()
    app._handle_progress(pipeline.ProgressEvent(stage="chunk_done", chunk_index=0, total_chunks=4))
    assert app._run_started_monotonic is not None
    assert "残り" in app.var_progress_label.get()
