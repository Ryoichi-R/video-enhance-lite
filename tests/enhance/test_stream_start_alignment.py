"""Regressions for stream start offsets and safe publication of the final output."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_enhance_lite import analyzer, pipeline, stages
from video_enhance_lite.pipeline import PipelineError
from tests.enhance.test_analyzer import _fake_ffprobe_factory
from tests.enhance.test_pipeline import FakeMeta, _make_decisions
from tests.support.process_runner import FakeProcessRunner


def _alignment_meta(video, preserved, rebase, container=4.5):
    return SimpleNamespace(
        video_start_offset_sec=video,
        preserved_streams_start_offset_sec=preserved,
        per_stream_timestamp_rebase=rebase,
        container_duration_sec=container,
    )


@pytest.mark.parametrize(
    "meta,expected",
    [
        # MP4/MKV: input 1 is rebased on the container start.
        (_alignment_meta(0.5, 0.0, False), (0.5, 0.0, 4.5)),
        (_alignment_meta(0.0, 0.5, False, container=4.5), (0.0, 0.0, 4.5)),
        # MPEG-TS/PS: input 1 is rebased on its earliest used (audio) stream.
        (_alignment_meta(0.5, 0.0, True), (0.5, 0.0, 4.5)),
        (_alignment_meta(0.0, 0.5, True, container=4.5), (0.0, 0.5, 4.5)),
        (_alignment_meta(0.2, 0.5, True, container=4.5), (0.0, 0.3, 4.3)),
        # No audio/subtitle stream to align against.
        (_alignment_meta(0.0, None, True, container=4.0), (0.0, 0.0, 4.0)),
    ],
)
def test_mux_alignment_restores_source_stream_offsets(meta, expected):
    video_shift, source_shift, duration = pipeline._mux_alignment(meta, 4.0)
    assert video_shift == pytest.approx(expected[0])
    assert source_shift == pytest.approx(expected[1])
    assert duration == pytest.approx(expected[2])


def test_extract_seeks_container_timeline_for_indexed_inputs():
    argv = stages.build_extract_args(
        Path("ffmpeg"), Path("in.mp4"), 10.0, 5.0, Path("frames"), video_start_offset_sec=0.5
    )
    assert argv[argv.index("-ss") + 1] == "10.5"
    assert argv.index("-ss") < argv.index("-i")


def test_extract_keeps_video_timeline_for_transport_streams():
    argv = stages.build_extract_args(
        Path("ffmpeg"), Path("in.ts"), 10.0, 5.0, Path("frames"), video_start_offset_sec=0.5
    )
    assert argv[argv.index("-ss") + 1] == "10.0"
    assert argv.index("-ss") > argv.index("-i")


def test_probe_records_stream_start_offsets(tmp_path, monkeypatch):
    video = {
        "streams": [
            {
                "index": 0,
                "width": 320,
                "height": 240,
                "r_frame_rate": "30/1",
                "avg_frame_rate": "30/1",
                "duration": "4.0",
                "start_time": "1.923",
                "codec_name": "h264",
            }
        ],
        "format": {"duration": "4.5", "start_time": "1.400", "format_name": "mpegts"},
    }
    inventory = {
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264", "start_time": "1.923"},
            {"index": 1, "codec_type": "audio", "codec_name": "aac", "start_time": "1.400"},
        ]
    }
    monkeypatch.setattr(analyzer, "_run", _fake_ffprobe_factory(video, inventory))

    meta = analyzer.probe_metadata(Path("ffprobe"), tmp_path / "in.ts")

    assert meta.video_start_offset_sec == pytest.approx(0.523)
    assert meta.preserved_streams_start_offset_sec == pytest.approx(0.0)
    assert meta.per_stream_timestamp_rebase is True
    assert meta.container_duration_sec == pytest.approx(4.5)


def test_probe_defaults_to_no_offset_without_start_times(tmp_path, monkeypatch):
    video = {
        "streams": [
            {"index": 0, "width": 320, "height": 240, "r_frame_rate": "30/1", "duration": "4.0"}
        ],
        "format": {"duration": "4.0", "format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
    }
    inventory = {"streams": [{"index": 0, "codec_type": "video", "codec_name": "h264"}]}
    monkeypatch.setattr(analyzer, "_run", _fake_ffprobe_factory(video, inventory))

    meta = analyzer.probe_metadata(Path("ffprobe"), tmp_path / "in.mp4")

    assert meta.video_start_offset_sec == 0.0
    assert meta.preserved_streams_start_offset_sec is None
    assert meta.per_stream_timestamp_rebase is False


def _run_encode_only(tmp_path, runner, output):
    return pipeline.run_pipeline(
        runner=runner,
        input_path=tmp_path / "in.mp4",
        output_path=output,
        decisions=_make_decisions(),
        meta=FakeMeta(duration_sec=4.0, fps=Fraction(30, 1)),
        chunk_seconds=10,
        log=lambda _message: None,
        ffmpeg=Path("ffmpeg"),
        ffprobe=Path("ffprobe"),
    )


def _is_concat(argv):
    return "-f" in argv and argv[argv.index("-f") + 1] == "concat"


class _OutputAppearsDuringConcat(FakeProcessRunner):
    def __init__(self, output: Path) -> None:
        super().__init__()
        self.output = output

    def run_with_progress(self, argv, **kwargs):
        if _is_concat(argv):
            self.output.write_bytes(b"created by someone else")
        result = super().run_with_progress(argv, **kwargs)
        if _is_concat(argv):
            Path(argv[-1]).write_bytes(b"finished output")
        return result


def test_final_output_never_replaces_a_file_created_during_processing(workspace, tmp_path):
    output = workspace / "out.mp4"
    runner = _OutputAppearsDuringConcat(output)

    saved = _run_encode_only(tmp_path, runner, output)

    assert saved != output
    assert output.read_bytes() == b"created by someone else"
    assert saved.read_bytes() == b"finished output"
    assert not list(workspace.glob("*.partial-*"))


class _ConcatFailsAfterWriting(FakeProcessRunner):
    def run_with_progress(self, argv, **kwargs):
        result = super().run_with_progress(argv, **kwargs)
        if _is_concat(argv):
            Path(argv[-1]).write_bytes(b"truncated")
            raise PipelineError("concat", "simulated failure")
        return result


def test_failed_final_mux_leaves_no_file_under_the_output_name(workspace, tmp_path):
    output = workspace / "out.mp4"

    with pytest.raises(PipelineError):
        _run_encode_only(tmp_path, _ConcatFailsAfterWriting(), output)

    assert not output.exists()
    assert not list(workspace.glob("*.partial-*"))


def test_concat_writes_partial_file_and_returns_final_path(workspace, tmp_path):
    output = workspace / "out.mp4"
    runner = FakeProcessRunner()

    saved = _run_encode_only(tmp_path, runner, output)

    concat = next(argv for argv in runner.calls if _is_concat(argv))
    assert Path(concat[-1]).parent == output.parent
    assert ".partial-" in Path(concat[-1]).name
    assert Path(concat[-1]).suffix == ".mp4"
    assert saved == output
    assert output.exists()


def test_publish_falls_back_to_rename_without_hard_links(tmp_path, monkeypatch):
    partial = tmp_path / "out.partial-1234.mp4"
    partial.write_bytes(b"finished")
    output = tmp_path / "out.mp4"

    def no_links(*_args, **_kwargs):
        raise OSError(22, "Incorrect function")  # e.g. exFAT on Windows

    monkeypatch.setattr(pipeline.os, "link", no_links)

    assert pipeline._publish_output_without_overwrite(partial, output) == output
    assert output.read_bytes() == b"finished"
    assert not partial.exists()


def test_finished_output_is_kept_when_it_cannot_be_moved(workspace, tmp_path, monkeypatch):
    output = workspace / "out.mp4"

    def cannot_publish(_temporary, _output):
        raise PermissionError(13, "denied")

    monkeypatch.setattr(pipeline, "_publish_output_without_overwrite", cannot_publish)

    with pytest.raises(PipelineError, match="was kept at"):
        _run_encode_only(tmp_path, FakeProcessRunner(), output)

    assert not output.exists()
    assert len(list(workspace.glob("out.partial-*.mp4"))) == 1
