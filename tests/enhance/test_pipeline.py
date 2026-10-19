from __future__ import annotations

import threading
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest

from video_enhance_lite import config, pipeline
from video_enhance_lite.decisions import StageDecision
from video_enhance_lite.media_streams import MediaStreamInfo

# V2-8 DI: 単一 FakeProcessRunner を共有定義から import (旧来の二重 FakeRunner を統合)。
from tests.support.process_runner import FakeProcessRunner


@dataclass
class FakeMeta:
    width: int = 1280
    height: int = 720
    fps: Fraction = Fraction(30, 1)
    duration_sec: float = 30.0
    codec: str = "h264"
    bitrate_bps: int | None = 5_000_000
    pix_fmt: str = "yuv420p"
    color_range: str = "tv"
    color_space: str = "bt709"
    field_order: str = "progressive"
    has_audio: bool = True
    framerate_status: str = "cfr"
    avg_fps: Fraction | None = Fraction(30, 1)
    primary_video_index: int = 0
    streams: tuple[MediaStreamInfo, ...] | None = None


def _make_decisions(**overrides_enabled) -> list[StageDecision]:
    """Build full decisions list. overrides_enabled[<key>]=True/False sets enabled.
    encode is always enabled.
    """
    out: list[StageDecision] = []
    for k in config.STAGE_KEYS:
        enabled = overrides_enabled.get(k, False)
        if k == "encode":
            enabled = True
        params: dict = {}
        if k == "upscale":
            params = {"scale": 2, "model": config.DEFAULT_REALESRGAN_MODEL}
        elif k == "face_restore":
            params = {
                "backend": config.DEFAULT_FACE_RESTORE_BACKEND,
                "start_sec": None,
                "end_sec": None,
                "fidelity": config.DEFAULT_FACE_RESTORE_FIDELITY,
            }
        elif k == "interpolate":
            params = {"target_fps_num": 60, "target_fps_den": 1}
        elif k == "encode":
            params = {"crf": config.DEFAULT_ENCODE_CRF, "preset": config.DEFAULT_ENCODE_PRESET}
        out.append(
            StageDecision(
                key=k,
                label_ja=config.STAGE_LABELS_JA[k],
                recommended=enabled,
                confidence="high",
                reason="test",
                enabled=enabled,
                params=params,
            )
        )
    return out


# NOTE: `workspace` / `fake_run` fixture は tests/enhance/conftest.py に集約済み。
# `FakeProcessRunner` (旧 `FakeRunner`) は tests/support/process_runner.py に一本化。


def _ffmpeg_calls(calls):
    return [c for c in calls if "ffmpeg" in c[0].lower()]


def _filter_calls_with(calls, needle: str):
    return [c for c in calls if "-vf" in c and any(needle in t for t in c)]


# ---------- chunk ranges ----------


def test_compute_chunk_ranges_basic():
    r = pipeline._compute_chunk_ranges(30.0, 10)
    assert r == [(0.0, 10.0, 0), (10.0, 10.0, 1), (20.0, 10.0, 2)]


def test_compute_chunk_ranges_with_remainder():
    r = pipeline._compute_chunk_ranges(25.0, 10)
    assert r[-1] == (20.0, 5.0, 2)


def test_compute_chunk_ranges_invalid():
    with pytest.raises(ValueError):
        pipeline._compute_chunk_ranges(0.0, 10)
    with pytest.raises(ValueError):
        pipeline._compute_chunk_ranges(10.0, 0)


def test_normalized_frame_counts_share_global_rounding_boundaries():
    fps = Fraction(30000, 1001)
    ranges = pipeline._compute_chunk_ranges(25.0, 10)
    counts = [
        pipeline._normalized_frame_count(start, duration, fps) for start, duration, _ in ranges
    ]

    assert sum(counts) == round(Fraction(25, 1) * fps)


# ---------- estimate ----------


def test_estimate_peak_temp_bytes_monotonic():
    a = pipeline.estimate_peak_temp_bytes(1920, 1080, Fraction(30, 1), 10)
    b = pipeline.estimate_peak_temp_bytes(1920, 1080, Fraction(60, 1), 10)
    assert b > a


# ---------- run_pipeline structure ----------


def test_run_pipeline_encode_only(workspace, fake_run):
    """All stages OFF except encode -> per-chunk: extract -> mux."""
    out = workspace / "out.mp4"
    decs = _make_decisions()  # only encode enabled
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=out,
        decisions=decs,
        meta=FakeMeta(duration_sec=20.0),
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
    )
    # 2 chunks: extract, mux (no pre/post, no upscale/rife)
    # extract = ffmpeg + %08d.png 出力
    extract = [
        c
        for c in fake_run.calls
        if "ffmpeg" in c[0].lower() and "-ss" in c and any("frames_in" in t for t in c)
    ]
    assert len(extract) == 2
    # mux = ffmpeg + chunks/chunk_NNN.mp4 出力
    mux = [
        c
        for c in fake_run.calls
        if "ffmpeg" in c[0].lower() and c[-1].endswith(".mp4") and "chunks" in c[-1]
    ]
    assert len(mux) == 2
    # concat
    concat = [c for c in fake_run.calls if "-f" in c and c[c.index("-f") + 1] == "concat"]
    assert len(concat) == 1


def test_run_pipeline_normalizes_every_vfr_chunk_to_one_global_frame_budget(
    workspace,
    fake_run,
):
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=_make_decisions(),
        meta=FakeMeta(
            duration_sec=25.0,
            framerate_status="vfr_suspected",
            avg_fps=Fraction(24, 1),
        ),
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
    )

    extracts = [
        call
        for call in fake_run.calls
        if "-ss" in call and any("frames_in" in token for token in call)
    ]
    assert len(extracts) == 3
    assert [call[call.index("-vf") + 1] for call in extracts] == ["fps=30/1"] * 3
    counts = [int(call[call.index("-frames:v") + 1]) for call in extracts]
    assert counts == [300, 300, 150]
    assert sum(counts) == round(25.0 * 30.0)


def test_run_pipeline_rejects_unknown_framerate_before_temp_or_process(
    workspace,
    fake_run,
    monkeypatch,
):
    temp_called = False

    def _unexpected_temp(*_args, **_kwargs):
        nonlocal temp_called
        temp_called = True
        raise AssertionError("temp workspace must not be created")

    monkeypatch.setattr(pipeline.temp_workspace, "create_temp_run_context", _unexpected_temp)

    with pytest.raises(pipeline.PipelineError, match="framerate_preflight.*status=unknown"):
        pipeline.run_pipeline(
            runner=fake_run,
            input_path=workspace / "in.mp4",
            output_path=workspace / "out.mp4",
            decisions=_make_decisions(),
            meta=FakeMeta(framerate_status="unknown", avg_fps=None),
            chunk_seconds=10,
            log=lambda _: None,
            ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
            ffprobe=Path(r"C:\bin\ffprobe.exe"),
        )

    assert temp_called is False
    assert fake_run.calls == []


def test_run_pipeline_maps_all_supported_streams_and_metadata(workspace, fake_run):
    meta = FakeMeta(
        duration_sec=10.0,
        streams=(
            MediaStreamInfo(0, "video", "h264"),
            MediaStreamInfo(1, "audio", "aac", language="eng"),
            MediaStreamInfo(2, "audio", "aac", language="jpn"),
            MediaStreamInfo(3, "subtitle", "mov_text", language="eng"),
        ),
    )

    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=_make_decisions(),
        meta=meta,
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
    )

    concat = next(
        call for call in fake_run.calls if "-f" in call and call[call.index("-f") + 1] == "concat"
    )
    maps = [concat[index + 1] for index, value in enumerate(concat) if value == "-map"]
    assert maps == ["0:v:0", "1:1", "1:2", "1:3"]
    assert concat[concat.index("-map_metadata") + 1] == "1"
    assert "-map_metadata:s:v:0" in concat
    assert concat[concat.index("-t") + 1] == "10.000000"
    assert "-shortest" not in concat
    assert concat[concat.index("-c:a") + 1] == "copy"
    assert concat[concat.index("-c:s") + 1] == "copy"


def test_run_pipeline_rejects_unsupported_stream_before_temp_or_process(
    workspace,
    fake_run,
    monkeypatch,
):
    meta = FakeMeta(
        streams=(
            MediaStreamInfo(0, "video", "h264"),
            MediaStreamInfo(1, "subtitle", "hdmv_pgs_subtitle"),
        ),
    )
    temp_called = False

    def _unexpected_temp(*_args, **_kwargs):
        nonlocal temp_called
        temp_called = True
        raise AssertionError("temp workspace must not be created")

    monkeypatch.setattr(pipeline.temp_workspace, "create_temp_run_context", _unexpected_temp)

    with pytest.raises(pipeline.PipelineError, match="stream_preflight.*index=1"):
        pipeline.run_pipeline(
            runner=fake_run,
            input_path=workspace / "in.mp4",
            output_path=workspace / "out.mp4",
            decisions=_make_decisions(),
            meta=meta,
            chunk_seconds=10,
            log=lambda _: None,
            ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
            ffprobe=Path(r"C:\bin\ffprobe.exe"),
        )

    assert temp_called is False
    assert fake_run.calls == []


def test_run_pipeline_preserves_subtitle_when_audio_is_absent(workspace, fake_run):
    meta = FakeMeta(
        has_audio=False,
        duration_sec=10.0,
        streams=(
            MediaStreamInfo(0, "video", "h264"),
            MediaStreamInfo(1, "subtitle", "mov_text", language="jpn"),
        ),
    )

    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=_make_decisions(),
        meta=meta,
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
    )

    concat = next(
        call for call in fake_run.calls if "-f" in call and call[call.index("-f") + 1] == "concat"
    )
    maps = [concat[index + 1] for index, value in enumerate(concat) if value == "-map"]
    assert maps == ["0:v:0", "1:1"]
    assert "-c:a" not in concat
    assert concat[concat.index("-c:s") + 1] == "copy"


def test_run_pipeline_with_pre_post_filters(workspace, fake_run):
    decs = _make_decisions(deinterlace=True, denoise=True, sharpen=True)
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
    # pre_filter ffmpeg call exists with yadif=0 + hqdn3d
    pre = [
        c for c in fake_run.calls if "-vf" in c and any("yadif=0" in t and "hqdn3d" in t for t in c)
    ]
    assert len(pre) == 1
    # post_filter ffmpeg call exists with unsharp
    post = _filter_calls_with(fake_run.calls, "unsharp")
    assert len(post) == 1


def test_cfr_default_keeps_split_extract_and_pre_filter_until_go_gate(workspace, fake_run):
    decs = _make_decisions(deinterlace=True, denoise=True)
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=decs,
        meta=FakeMeta(duration_sec=10.0, framerate_status="cfr"),
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
    )
    split = [
        call
        for call in fake_run.calls
        if "-vf" in call
        and any("yadif=0" in token and "format=rgb24" not in token for token in call)
    ]
    assert len(split) == 1
    assert Path(split[0][-1]).parent.name == "frames_pre"


def test_experimental_cfr_fuses_extract_and_pre_filter(workspace, fake_run):
    decs = _make_decisions(deinterlace=True, denoise=True)
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=decs,
        meta=FakeMeta(duration_sec=10.0, framerate_status="cfr"),
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        experimental_fuse_extract_prefilter=True,
    )
    fused = [
        call
        for call in fake_run.calls
        if "-vf" in call and any("format=rgb24,yadif=0" in token for token in call)
    ]
    assert len(fused) == 1
    assert Path(fused[0][-1]).parent.name == "frames_pre"


def test_debug_option_keeps_split_extract_and_pre_filter(workspace, fake_run):
    decs = _make_decisions(deinterlace=True, denoise=True)
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=decs,
        meta=FakeMeta(duration_sec=10.0, framerate_status="cfr"),
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
        debug_split_extract_prefilter=True,
        experimental_fuse_extract_prefilter=True,
    )
    pre = [
        call
        for call in fake_run.calls
        if "-vf" in call
        and any("yadif=0" in token and "format=rgb24" not in token for token in call)
    ]
    assert len(pre) == 1
    assert Path(pre[0][-1]).parent.name == "frames_pre"


def test_run_pipeline_with_upscale(workspace, fake_run):
    decs = _make_decisions(upscale=True)
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
        realesrgan=Path(r"C:\bin\realesrgan-ncnn-vulkan.exe"),
    )
    # realesrgan が 1 chunk 分呼ばれている
    rea = [c for c in fake_run.calls if "realesrgan" in c[0].lower()]
    assert len(rea) == 1


def test_run_pipeline_with_interpolate(workspace, fake_run):
    decs = _make_decisions(interpolate=True)
    # source 30 → target 60 (整数倍)
    decs_by = {d.key: d for d in decs}
    decs_by["interpolate"].params = {"target_fps_num": 60, "target_fps_den": 1}
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
        rife=Path(r"C:\bin\rife-ncnn-vulkan.exe"),
    )
    rife = [c for c in fake_run.calls if "rife" in c[0].lower()]
    assert len(rife) == 1


def test_intermediate_png_stages_use_compression_level(workspace, fake_run):
    """V2-6: ffmpeg の中間 PNG 出力ステージ (extract/pre_filter/resolution_clamp/post) には
    共通ヘルパー _run_ffmpeg_png_stage 経由で -compression_level 1 が付与され、
    realesrgan/rife の ncnn 出力には付与されないこと (PNG 出力は全て同ヘルパーを通る)。"""
    decs = _make_decisions(upscale=True, interpolate=True)
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
        realesrgan=Path(r"C:\bin\realesrgan-ncnn-vulkan.exe"),
        rife=Path(r"C:\bin\rife-ncnn-vulkan.exe"),
    )

    def _has_pair(call, opt, val):
        return any(call[i] == opt and call[i + 1] == val for i in range(len(call) - 1))

    png_ffmpeg = [
        c
        for c in fake_run.calls
        if "ffmpeg" in c[0].lower() and str(c[-1]).lower().endswith(".png")
    ]
    assert png_ffmpeg, "PNG 出力 ffmpeg ステージが実行されていない"
    for c in png_ffmpeg:
        assert _has_pair(c, "-compression_level", "1"), f"compression_level 欠落: {c}"
        # 出力パスは末尾のまま (オプションは直前に挿入される)
        assert str(c[-1]).lower().endswith(".png"), f"出力が末尾でない: {c}"

    ncnn = [c for c in fake_run.calls if "realesrgan" in c[0].lower() or "rife" in c[0].lower()]
    assert ncnn, "ncnn ステージが実行されていない"
    for c in ncnn:
        assert "-compression_level" not in c, f"ncnn に compression_level 混入: {c}"


def test_has_png_existence_check(tmp_path):
    """V2-7: _has_png は PNG の有無を最初の 1 件で早期判定する (全件カウント回避)。"""
    assert pipeline._has_png(tmp_path) is False  # 空ディレクトリ
    (tmp_path / "note.txt").write_bytes(b"x")
    assert pipeline._has_png(tmp_path) is False  # PNG 以外のみ
    (tmp_path / "00000000.png").write_bytes(b"")
    assert pipeline._has_png(tmp_path) is True  # PNG あり
    assert pipeline._has_png(tmp_path / "missing") is False  # 非存在ディレクトリ


def test_run_pipeline_stabilize_runs_detect_pass(workspace, fake_run):
    decs = _make_decisions(stabilize=True)
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
    # vidstabdetect 1st pass call
    det = [c for c in fake_run.calls if "-vf" in c and any("vidstabdetect" in t for t in c)]
    assert len(det) == 1
    # pre_filter で vidstabtransform が含まれる
    trf = [c for c in fake_run.calls if "-vf" in c and any("vidstabtransform" in t for t in c)]
    assert len(trf) == 1


# ---------- cancellation ----------


def test_run_pipeline_cancels_at_start(workspace, fake_run):
    ev = threading.Event()
    ev.set()  # pre-cancel
    with pytest.raises(pipeline.PipelineCancelled):
        pipeline.run_pipeline(
            runner=fake_run,
            input_path=workspace / "in.mp4",
            output_path=workspace / "out.mp4",
            decisions=_make_decisions(),
            meta=FakeMeta(),
            chunk_seconds=10,
            log=lambda _: None,
            cancel_event=ev,
            ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
            ffprobe=Path(r"C:\bin\ffprobe.exe"),
        )


def test_run_pipeline_cancel_between_chunks(workspace, fake_run):
    ev = threading.Event()
    chunks_started: list[int] = []

    def progress(ev_p):
        if ev_p.stage == "chunk_start":
            chunks_started.append(ev_p.chunk_index)
            if ev_p.chunk_index == 1:
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
        )


# ---------- keep-temp-on-failure (V2-9) ----------


def test_keep_temp_on_failure_preserves_tmp(workspace):
    """V2-9: 失敗時に keep_temp_on_failure=True なら tmp を削除せず保持する。"""
    failer = FakeProcessRunner(fail_on_argv_marker="frames_in")
    tmp_root = config.derive_temp_dir(workspace / "in.mp4")
    with pytest.raises(pipeline.PipelineError):
        pipeline.run_pipeline(
            runner=failer,
            input_path=workspace / "in.mp4",
            output_path=workspace / "out.mp4",
            decisions=_make_decisions(),
            meta=FakeMeta(duration_sec=10.0),
            chunk_seconds=10,
            log=lambda _: None,
            keep_temp_on_failure=True,
            ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
            ffprobe=Path(r"C:\bin\ffprobe.exe"),
        )
    assert tmp_root.exists(), "失敗時に keep_temp_on_failure=True なのに tmp が削除された"


def test_failure_default_removes_tmp(workspace):
    """V2-9: 既定 (keep_temp_on_failure=False) では失敗時も tmp を削除する (回帰防止)。"""
    failer = FakeProcessRunner(fail_on_argv_marker="frames_in")
    tmp_root = config.derive_temp_dir(workspace / "in.mp4")
    with pytest.raises(pipeline.PipelineError):
        pipeline.run_pipeline(
            runner=failer,
            input_path=workspace / "in.mp4",
            output_path=workspace / "out.mp4",
            decisions=_make_decisions(),
            meta=FakeMeta(duration_sec=10.0),
            chunk_seconds=10,
            log=lambda _: None,
            ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
            ffprobe=Path(r"C:\bin\ffprobe.exe"),
        )
    assert not tmp_root.exists()


def test_keep_temp_on_failure_success_still_cleans(workspace, fake_run):
    """V2-9: 成功時は keep_temp_on_failure=True でも tmp を削除する (保全は異常終了時のみ)。"""
    tmp_root = config.derive_temp_dir(workspace / "in.mp4")
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=_make_decisions(),
        meta=FakeMeta(duration_sec=10.0),
        chunk_seconds=10,
        log=lambda _: None,
        keep_temp_on_failure=True,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
    )
    assert not tmp_root.exists()


# ---------- progress events ----------


def test_run_pipeline_emits_progress_events(workspace, fake_run):
    events: list[pipeline.ProgressEvent] = []
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=_make_decisions(),
        meta=FakeMeta(duration_sec=10.0),
        chunk_seconds=10,
        log=lambda _: None,
        progress=lambda e: events.append(e),
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
    )
    stages_seen = [e.stage for e in events]
    assert "process_chunks" in stages_seen
    assert "chunk_start" in stages_seen
    assert "chunk_done" in stages_seen
    assert "concat" in stages_seen
    assert "done" in stages_seen


# ---------- error propagation ----------


def test_run_pipeline_extract_failure_wraps_in_pipeline_error(workspace):
    failer = FakeProcessRunner(fail_on_argv_marker="frames_in")
    with pytest.raises(pipeline.PipelineError) as exc:
        pipeline.run_pipeline(
            runner=failer,
            input_path=workspace / "in.mp4",
            output_path=workspace / "out.mp4",
            decisions=_make_decisions(),
            meta=FakeMeta(duration_sec=10.0),
            chunk_seconds=10,
            log=lambda _: None,
            ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
            ffprobe=Path(r"C:\bin\ffprobe.exe"),
        )
    assert exc.value.stage == "extract_frames"
    assert exc.value.chunk_idx == 0


def test_run_pipeline_missing_encode_decision_raises(workspace):
    # encode を除外したリスト
    decs = [d for d in _make_decisions() if d.key != "encode"]
    with pytest.raises(ValueError, match="encode"):
        pipeline.run_pipeline(
            input_path=workspace / "in.mp4",
            output_path=workspace / "out.mp4",
            decisions=decs,
            meta=FakeMeta(),
            chunk_seconds=10,
            log=lambda _: None,
            ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
            ffprobe=Path(r"C:\bin\ffprobe.exe"),
        )


# ---------- keep_temp ----------


def test_run_pipeline_cleanup_when_not_keep_temp(workspace, fake_run):
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=_make_decisions(),
        meta=FakeMeta(duration_sec=10.0),
        chunk_seconds=10,
        keep_temp=False,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
    )
    tmp_root = config.derive_temp_dir(workspace / "in.mp4")
    assert not tmp_root.exists()


def test_run_pipeline_allows_output_outside_workspace(workspace, fake_run, tmp_path):
    """final output can be written outside WORKSPACE_ROOT."""
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside = outside_dir / "out.mp4"
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=outside,
        decisions=_make_decisions(),
        meta=FakeMeta(),
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
    )


def test_run_pipeline_keep_temp_preserves(workspace, fake_run):
    pipeline.run_pipeline(
        runner=fake_run,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=_make_decisions(),
        meta=FakeMeta(duration_sec=10.0),
        chunk_seconds=10,
        keep_temp=True,
        log=lambda _: None,
        ffmpeg=Path(r"C:\bin\ffmpeg.exe"),
        ffprobe=Path(r"C:\bin\ffprobe.exe"),
    )
    tmp_root = config.derive_temp_dir(workspace / "in.mp4")
    assert tmp_root.exists()
