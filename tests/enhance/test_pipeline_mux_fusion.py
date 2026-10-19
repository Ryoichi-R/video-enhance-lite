from __future__ import annotations

from fractions import Fraction
from pathlib import Path


from video_enhance_lite import config, pipeline, stages

# 共通 helper を再利用 (workspace / fake_run fixture は conftest.py 由来で import 不要)
from tests.enhance.test_pipeline import FakeMeta, _make_decisions  # noqa: F401


def test_build_mux_args_without_vf():
    encode = ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p"]
    argv = stages.build_mux_args(
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        frames_dir=Path(r"C:\tmp\frames_out"),
        fps=Fraction(30, 1),
        chunk_path=Path(r"C:\tmp\chunk_000.mp4"),
        encode_args=encode,
    )
    assert "-vf" not in argv


def test_build_mux_args_with_vf_includes_post_chain():
    encode = ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p"]
    argv = stages.build_mux_args(
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        frames_dir=Path(r"C:\tmp\frames_rife"),
        fps=Fraction(30, 1),
        chunk_path=Path(r"C:\tmp\chunk_000.mp4"),
        encode_args=encode,
        vf_chain="unsharp=5:5:0.6:5:5:0.0,noise=alls=4:allf=t",
    )
    assert "-vf" in argv
    vf_idx = argv.index("-vf")
    assert "unsharp" in argv[vf_idx + 1]
    assert "noise" in argv[vf_idx + 1]
    # -c:v / libx264 が含まれる
    assert "libx264" in argv


def test_run_pipeline_fused_mux_includes_postfilter(workspace, fake_run):
    """PR6: post_filter ON で frames_out への独立 ffmpeg 呼び出しが**ない**。
    かわりに mux 段で -vf として post_chain が渡る。
    """
    decs = _make_decisions(sharpen=True)
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
    )
    # frames_out/%08d.png を出力する ffmpeg 呼び出しは無いはず (融合 mux のみ)
    post_only = [
        c
        for c in fake_run.calls
        if "ffmpeg" in c[0].lower() and any("frames_out" in t and "%08d.png" in t for t in c)
    ]
    assert len(post_only) == 0
    # mux 出力 (chunk_000.mp4) は -vf を含む
    mux = [
        c
        for c in fake_run.calls
        if "ffmpeg" in c[0].lower() and c[-1].endswith(".mp4") and "chunks" in c[-1]
    ]
    assert len(mux) == 1
    assert "-vf" in mux[0]
    vf_idx = mux[0].index("-vf")
    assert "unsharp" in mux[0][vf_idx + 1]


def test_run_pipeline_fused_mux_no_postfilter(workspace, fake_run):
    """post_filter OFF でも融合パスを通り、frames_out への ffmpeg 呼び出しはない。"""
    decs = _make_decisions()
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
    )
    mux = [
        c
        for c in fake_run.calls
        if "ffmpeg" in c[0].lower() and c[-1].endswith(".mp4") and "chunks" in c[-1]
    ]
    assert len(mux) == 1
    # libx264 が含まれる (encode 引数)
    assert "libx264" in mux[0]
    # post_filter OFF なので -vf は付かない
    assert "-vf" not in mux[0]


def test_keep_temp_preserves_chunk_intermediate_frames(workspace, fake_run):
    """--keep-temp で chunk 別の中間 PNG (frames_rife/) が残ること (PR3+PR6 plan-gap 修正)。"""
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
        worker_count=2,  # PR3 のチャンク別 tmp パスに乗せる
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
    )
    identity_root = config.derive_temp_dir(workspace / "in.mp4")
    run_leaves = list((identity_root / "runs").iterdir())
    assert len(run_leaves) == 1
    tmp_root = run_leaves[0]
    # チャンク別ディレクトリが残り、その中の frames_rife/ も残る
    for idx in (0, 1):
        chunk_dir = tmp_root / f"chunk_{idx:03d}"
        assert chunk_dir.exists()
        frames_rife = chunk_dir / "frames_rife"
        assert frames_rife.exists()
        # FakeProcessRunner が realesrgan / 直接 rename 経由で書き出した PNG
        # (rife/upscale 未使用なので frames_in -> ... -> frames_rife へ rename されている)
        png_count = len(list(frames_rife.glob("*.png")))
        assert png_count >= 1


def test_no_keep_temp_cleans_intermediate_frames(workspace, fake_run):
    """--keep-temp 無し (default) では中間 PNG が消える。"""
    decs = _make_decisions()
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=decs,
        meta=FakeMeta(duration_sec=10.0),
        chunk_seconds=10,
        log=lambda _: None,
        keep_temp=False,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
    )
    tmp_root = config.derive_temp_dir(workspace / "in.mp4")
    # tmp 全体が削除されている
    assert not tmp_root.exists()


def test_run_pipeline_debug_split_postfilter_uses_legacy_2stage(workspace, fake_run):
    """--debug-split-postfilter 指定時のみ旧 2 段パスが走り frames_out への
    ffmpeg 呼び出しが残る。"""
    decs = _make_decisions(sharpen=True)
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
        debug_split_postfilter=True,
    )
    # debug 経路: post_filter の出力 (frames_out/%08d.png) と mux の入力 (frames_out/%08d.png) の 2 件
    frames_out_uses = [
        c
        for c in fake_run.calls
        if "ffmpeg" in c[0].lower() and any("frames_out" in t and "%08d.png" in t for t in c)
    ]
    assert len(frames_out_uses) == 2
    # うち 1 件は post_filter の出力 (最後の引数が frames_out/%08d.png)
    post_outputs = [c for c in frames_out_uses if c[-1].endswith("%08d.png")]
    assert len(post_outputs) == 1
