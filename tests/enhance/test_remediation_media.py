"""Real encode/decode regressions for the 0.3 media pipeline."""

import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from video_enhance_lite import analyzer, config, gui, pipeline, stages, temp_workspace
from video_enhance_lite.decisions import compute_output_summary
from tests.enhance.test_media_correctness_e2e import _FFMPEG, _FFPROBE, _run_ffmpeg
from tests.enhance.test_pipeline import _make_decisions

pytestmark = pytest.mark.e2e


def probe(path):
    return json.loads(
        subprocess.check_output(
            [
                _FFPROBE,
                "-v",
                "error",
                "-count_frames",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                path,
            ]
        )
    )


def raw(path, vf, fmt="gray"):
    return subprocess.check_output(
        [
            _FFMPEG,
            "-v",
            "error",
            "-i",
            path,
            "-vf",
            vf,
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            fmt,
            "-f",
            "rawvideo",
            "-",
        ]
    )


def process(source, output, *, workers=1, seconds=10, decisions=None, **opts):
    meta = analyzer.probe_metadata(Path(_FFPROBE), source)
    context = temp_workspace.create_temp_run_context(source, output.parent / "temp")
    try:
        pipeline.run_pipeline(
            input_path=source,
            output_path=output,
            decisions=decisions or _make_decisions(),
            meta=meta,
            ffmpeg=Path(_FFMPEG),
            ffprobe=Path(_FFPROBE),
            temp_run=context,
            chunk_seconds=seconds,
            worker_count=workers,
            log=lambda _: None,
            **opts,
        )
    finally:
        temp_workspace.cleanup_temp_run(context)
    return meta


def test_vfr_held_frame_across_boundaries(tmp_path):
    source = tmp_path / "held.mp4"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "color=black:s=32x24:r=30:d=21",
            "-vf",
            "drawbox=color=white:t=fill:enable='gte(n,331)',select='lte(n,259)+gte(n,331)',"
            "settb=1/90000,setpts='if(eq(N,260),993600,PTS)'",
            "-enc_time_base",
            "1/90000",
            "-fps_mode",
            "vfr",
            "-video_track_timescale",
            "90000",
            "-g",
            "30",
            "-c:v",
            "libx264",
            str(source),
        ]
    )
    ref = np.frombuffer(raw(source, "setpts=PTS-STARTPTS,fps=30,scale=1:1"), np.uint8) > 128
    assert len(ref) == 630 and ref.any() and not ref.all()
    for workers, seconds in ((1, 10), (3, 10), (2, 1)):
        out = tmp_path / f"out-{workers}.mp4"
        process(source, out, workers=workers, seconds=seconds)
        actual = np.frombuffer(raw(out, "scale=1:1"), np.uint8) > 128
        np.testing.assert_array_equal(actual, ref)


@pytest.mark.parametrize(
    "rate,count,workers",
    [
        ("30000/1001", 299, 1),
        ("30000/1001", 300, 2),
        ("30000/1001", 301, 2),
        ("24000/1001", 241, 2),
        ("1/2", 12, 1),
    ],
)
def test_ntsc_tail_matches_unpartitioned_frames(tmp_path, rate, count, workers):
    source, out = tmp_path / "ntsc.mp4", tmp_path / "out.mp4"
    # Distinct luma identifies each frame (including duplicates/drops at the boundary).
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"nullsrc=s=32x24:r={rate}",
            "-vf",
            "geq=lum='32+mod(N*17,184)':cb=128:cr=128",
            "-frames:v",
            str(count),
            "-g",
            "250",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            str(source),
        ]
    )
    process(source, out, workers=workers)
    assert int(probe(out)["streams"][0]["nb_read_frames"]) == count
    ref = np.frombuffer(raw(source, "scale=1:1"), np.uint8)
    actual = np.frombuffer(raw(out, "scale=1:1"), np.uint8)
    np.testing.assert_allclose(actual, ref, atol=4, rtol=0)


@pytest.mark.parametrize("transfer", ["smpte2084", "arib-std-b67"])
def test_hdr_ramp_preflight_and_output(tmp_path, transfer):
    source, out = tmp_path / "hdr.mp4", tmp_path / "709.mp4"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "nullsrc=s=96x32:r=4:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=d=2",
            "-vf",
            f"format=yuv420p10le,geq=lum='64+floor(X/12)*117':cb=512:cr=512,"
            f"setparams=range=limited:colorspace=bt2020nc:color_primaries=bt2020:color_trc={transfer}",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            transfer,
            "-colorspace",
            "bt2020nc",
            "-color_range",
            "tv",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-c:a",
            "aac",
            "-metadata",
            "camera_id=試験01",
            "-metadata",
            "title=映像の検証",
            "-movflags",
            "use_metadata_tags",
            str(source),
        ]
    )
    with pytest.raises(pipeline.PipelineError, match="hdr-to-sdr"):
        process(source, out)
    assert not out.exists()
    meta = process(source, out, workers=2, seconds=1, hdr_to_sdr=True)
    assert meta.bit_depth == 10
    info = probe(out)
    v, a = info["streams"]
    assert (
        v["pix_fmt"],
        v["color_range"],
        v["color_space"],
        v["color_transfer"],
        v["color_primaries"],
    ) == ("yuv420p", "tv", "bt709", "bt709", "bt709")
    assert abs(float(a["duration"]) - float(v["duration"])) < 0.05
    assert info["format"]["tags"]["camera_id"] == "試験01"
    assert info["format"]["tags"]["title"] == "映像の検証"
    rows = np.frombuffer(raw(out, "null"), np.uint8).reshape(-1, 32, 96)[:, 16, 6::12].astype(int)
    assert (np.diff(rows, axis=1) > 0).all()
    np.testing.assert_allclose(rows, np.tile(rows[0], (len(rows), 1)), rtol=0, atol=2)
    # Independent single-pass graph, encoding directly from linear-light conversion.
    reference = raw(
        source,
        "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
        f"tonemap=hable:peak={100 if transfer == 'smpte2084' else 10},zscale=t=bt709:m=bt709:r=limited,format=yuv420p",
    )
    expected = np.frombuffer(reference, np.uint8).reshape(-1, 32, 96)[:, 16, 6::12].astype(int)
    np.testing.assert_allclose(rows, expected, rtol=0, atol=4)


def test_webm_odd_edges_and_sar(tmp_path):
    source = tmp_path / "input.webm"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=s=95x63:r=5:d=1",
            "-vf",
            "setsar=4/3",
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuv444p",
            str(source),
        ]
    )
    meta = analyzer.probe_metadata(Path(_FFPROBE), source)
    decisions = _make_decisions()
    summary = compute_output_summary(meta, decisions)
    out = gui.derive_output_path_for_gui(source, summary, tmp_path)
    assert out == config.derive_output_path(source, tmp_path, "_96x64-5p")
    process(source, out)
    info = probe(out)["streams"][0]
    assert (info["width"], info["height"], info["codec_name"]) == (96, 64, "h264")
    # Matroska may approximate the display width in integer pixels.
    assert Fraction(info["sample_aspect_ratio"].replace(":", "/")) == meta.sample_aspect_ratio
    before = np.frombuffer(raw(source, "format=rgb24", "rgb24"), np.uint8).reshape(-1, 63, 95, 3)
    padded = np.frombuffer(
        raw(source, "format=rgb24," + stages.build_canvas_filter(meta, None), "rgb24"), np.uint8
    ).reshape(-1, 64, 96, 3)
    np.testing.assert_array_equal(
        padded, np.pad(before, ((0, 0), (0, 1), (0, 1), (0, 0)), mode="edge")
    )


@pytest.mark.parametrize("model", [config.DEFAULT_RIFE_MODEL, "rife-v2.3"])
def test_real_rife_one_frame_tail_and_model_layout(tmp_path, model):
    from shutil import copytree, copy2

    location = os.environ.get("VIDEO_ENHANCE_E2E_RIFE_DIR")
    if not location:
        pytest.skip("requires installed RIFE")
    binary_dir = Path(location)
    binary = next(
        p
        for p in (binary_dir / "rife-ncnn-vulkan.exe", binary_dir / "rife-ncnn-vulkan")
        if p.is_file()
    )
    # Use a legacy model parent whose name contains a different generation string.
    parent = tmp_path / "rife-v2" / "models"
    copytree(binary_dir / model, parent / model)
    for dependency in binary_dir.iterdir():
        if dependency.is_file() and (
            dependency == binary or dependency.suffix in (".native", ".dll", ".dylib")
        ):
            copy2(dependency, parent.parent / dependency.name)
    binary = parent.parent / binary.name
    source, out = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=64x48:r=30",
            "-f",
            "lavfi",
            "-i",
            "sine=d=1.033333",
            "-frames:v",
            "31",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(source),
        ]
    )
    decisions = _make_decisions(interpolate=True)
    next(d for d in decisions if d.key == "interpolate").params["model"] = model
    process(
        source,
        out,
        seconds=1,
        rife=binary,
        rife_model_parent=parent,
        decisions=decisions,
    )
    v, a = probe(out)["streams"]
    assert (int(v["nb_read_frames"]), Fraction(v["avg_frame_rate"])) == (62, 60)
    assert float(v["duration"]) == pytest.approx(62 / 60, abs=0.0001)
    original_audio = probe(source)["streams"][1]
    assert float(a["duration"]) == pytest.approx(float(original_audio["duration"]), abs=0.025)


@pytest.mark.parametrize(
    "matrix,level", [("bt709", "tv"), ("bt709", "pc"), ("smpte170m", "tv"), ("smpte170m", "pc")]
)
def test_sdr_colour_chart_round_trip(tmp_path, matrix, level):
    source, out, reference = [
        tmp_path / name for name in ("chart.mkv", "output.mp4", "reference.mp4")
    ]
    labels = (
        f"setparams=range={level}:colorspace={matrix}:color_primaries={matrix}:color_trc={matrix}"
    )
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "smptebars=s=96x64:r=10:d=1",
            "-vf",
            f"scale=in_color_matrix=smpte170m:out_color_matrix={matrix}:out_range={level},format=yuv444p10le,{labels}",
            "-color_range",
            level,
            "-c:v",
            "ffv1",
            str(source),
        ]
    )
    meta = process(source, out, hdr_to_sdr=True)
    assert meta.bit_depth == 10
    v = probe(out)["streams"][0]
    assert (v["color_space"], v["color_range"], v["color_primaries"], v["color_transfer"]) == (
        matrix,
        level,
        matrix,
        matrix,
    )
    _run_ffmpeg(
        [
            "-i",
            str(source),
            "-vf",
            f"scale=in_color_matrix={matrix}:in_range={level}:out_range=full,format=rgb24,"
            f"scale=in_range=full:out_color_matrix={matrix}:out_range={level},format=yuv420p,{labels}",
            "-color_range",
            level,
            "-c:v",
            "libx264",
            "-crf",
            str(config.DEFAULT_ENCODE_CRF),
            "-preset",
            config.DEFAULT_ENCODE_PRESET,
            str(reference),
        ]
    )
    actual = np.frombuffer(raw(out, "null", "rgb24"), np.uint8).astype(int)
    ref = np.frombuffer(raw(reference, "null", "rgb24"), np.uint8).astype(int)
    np.testing.assert_allclose(actual, ref, rtol=0, atol=3)


@pytest.mark.parametrize("order", ["tff", "bff"])
def test_yadif_matches_native_yuv_at_each_boundary(tmp_path, order):
    source = tmp_path / "interlaced.mov"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=96x64:r=20:d=2",
            "-vf",
            "tinterlace=mode=" + ("interleave_top" if order == "tff" else "interleave_bottom"),
            "-c:v",
            "libx264",
            "-x264-params",
            order + "=1",
            "-field_order",
            order[0] * 2,
            str(source),
        ]
    )
    meta = analyzer.probe_metadata(Path(_FFPROBE), source)
    out = []
    for start in (0, 1):
        dest = tmp_path / f"f{start}"
        dest.mkdir()
        args = stages.build_extract_args(
            Path(_FFMPEG),
            source,
            start,
            1,
            dest,
            source_fps=meta.fps,
            expected_frame_count=10,
            first_frame=10 * start,
            field_order=meta.field_order,
        )
        subprocess.run(args, check=True, capture_output=True)
        out.append(
            subprocess.check_output(
                [
                    _FFMPEG,
                    "-v",
                    "error",
                    "-i",
                    str(dest / "%08d.png"),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-",
                ]
            )
        )
    assert b"".join(out) == raw(source, f"yadif=0:parity={order}", "rgb24")


def test_opencv_stabilization_keeps_anamorphic_display(tmp_path):
    source, out = tmp_path / "source.mp4", tmp_path / "out.mp4"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=96x64:r=10:d=1",
            "-vf",
            "setsar=4/3",
            "-c:v",
            "libx264",
            str(source),
        ]
    )
    process(source, out, decisions=_make_decisions(stabilize=True), stabilize_backend="opencv")
    video = probe(out)["streams"][0]
    assert video["sample_aspect_ratio"] == "4:3" and video["display_aspect_ratio"] == "2:1"


@pytest.mark.parametrize("width,height", [(95, 63), (97, 65), (95, 64), (96, 63), (96, 64)])
def test_active_pixels_and_duplicated_edges(tmp_path, width, height):
    from types import SimpleNamespace

    meta = SimpleNamespace(width=width, height=height, fps=Fraction(30))
    filt = stages.build_canvas_filter(meta, None) or "null"
    source = np.random.default_rng(7).integers(0, 256, (height, width, 3), dtype=np.uint8)
    data = subprocess.check_output(
        [
            _FFMPEG,
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-i",
            "-",
            "-vf",
            filt,
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ],
        input=source.tobytes(),
    )
    expected = np.pad(source, ((0, height % 2), (0, width % 2), (0, 0)), mode="edge")
    assert data == expected.tobytes()
    summary = compute_output_summary(meta, _make_decisions())
    assert (summary.out_height, summary.out_width) == expected.shape[:2]


@pytest.mark.parametrize("transfer", ["smpte2084", "arib-std-b67"])
def test_hdr_coloured_patches_and_mastering_metadata(tmp_path, transfer):
    source, output = tmp_path / "hdr.mkv", tmp_path / "mapped.mp4"
    if "libx265" not in _run_ffmpeg(["-encoders"]).stdout:
        pytest.skip("libx265 is needed to generate HDR10 side data")
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "nullsrc=s=96x48:r=4:d=2",
            "-vf",
            "format=yuv420p10le,geq=lum='320+floor(X/24)*140':cb='450+floor(X/12)*32':cr='560-floor(X/12)*32',"
            f"setparams=range=limited:colorspace=bt2020nc:color_primaries=bt2020:color_trc={transfer}",
            "-color_range",
            "tv",
            "-color_primaries",
            "bt2020",
            "-colorspace",
            "bt2020nc",
            "-color_trc",
            transfer,
            "-c:v",
            "libx265",
            "-x265-params",
            f"colorprim=9:colormatrix=9:range=limited:transfer={16 if transfer == 'smpte2084' else 18}:"
            "pools=1:frame-threads=1:lossless=1:max-cll=1200,400:"
            "master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(40000000,50)",
            "-metadata",
            "MaxCLL=1200",
            "-metadata",
            "camera_id=試験01",
            str(source),
        ]
    )
    meta = process(source, output, hdr_to_sdr=True, seconds=1, workers=2)
    assert meta.peak_nits == 1200
    info = probe(output)
    assert {k.lower(): v for k, v in info["format"]["tags"].items()}["camera_id"] == "試験01"
    assert "maxcll" not in {key.lower() for key in info["format"]["tags"]}
    frames = json.loads(
        subprocess.check_output(
            [
                _FFPROBE,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_entries",
                "frame_side_data=side_data_type",
                "-of",
                "json",
                output,
            ]
        )
    )["frames"]
    assert not any(
        "mastering" in entry["side_data_type"].lower()
        or "light level" in entry["side_data_type"].lower()
        for frame in frames
        for entry in frame.get("side_data_list", [])
    )
    graph = (
        "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
        "tonemap=hable:peak=12,zscale=t=bt709:m=bt709:r=limited,format=yuv420p"
    )

    def patches(data):
        return np.frombuffer(data, np.uint8).reshape(-1, 48, 96, 3)[:, 24, 12::24, :].astype(int)

    actual = patches(raw(output, "null", "rgb24"))
    expected = patches(raw(source, graph, "rgb24"))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=6)


def test_delayed_vfr_with_audio_tail(tmp_path):
    source, out = tmp_path / "late.mp4", tmp_path / "out.mp4"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=32x24:r=10:d=6",
            "-f",
            "lavfi",
            "-i",
            "sine=d=9",
            "-vf",
            "select='lt(n,5)+gte(n,36)',setpts=PTS+1/TB",
            "-fps_mode",
            "vfr",
            "-g",
            "4",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(source),
        ]
    )
    meta = process(source, out, seconds=1, workers=2)
    assert meta.framerate_status == "vfr_suspected"
    expected = np.frombuffer(raw(source, "setpts=PTS-STARTPTS,fps=10,scale=1:1"), np.uint8)
    actual = np.frombuffer(raw(out, "scale=1:1"), np.uint8)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=4)
    v, a = probe(out)["streams"]
    assert float(v["start_time"]) == pytest.approx(1, abs=0.02)
    assert float(a["duration"]) == pytest.approx(9, abs=0.05)


def test_rotated_odd_anamorphic_canvas(tmp_path):
    source, rotated, out = (tmp_path / name for name in ("raw.mp4", "rotated.mp4", "out.mp4"))
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=s=95x63:r=5:d=1",
            "-vf",
            "setsar=4/3",
            "-pix_fmt",
            "yuv444p",
            "-c:v",
            "libx264",
            str(source),
        ]
    )
    _run_ffmpeg(["-display_rotation", "90", "-i", str(source), "-c", "copy", str(rotated)])
    meta = process(rotated, out)
    assert (meta.width, meta.height, meta.sample_aspect_ratio) == (63, 95, Fraction(3, 4))
    v = probe(out)["streams"][0]
    assert (v["width"], v["height"], v["sample_aspect_ratio"]) == (64, 96, "3:4")
    assert not any(entry.get("rotation") for entry in v.get("side_data_list", []))


def test_missing_hdr_filters_fail_before_chunk_processing(tmp_path, monkeypatch):
    from types import SimpleNamespace

    source, out = tmp_path / "input.mp4", tmp_path / "output.mp4"
    meta = SimpleNamespace(
        width=64,
        height=48,
        fps=Fraction(30),
        duration_sec=1,
        has_audio=False,
        color_transfer="smpte2084",
        color_space="bt2020nc",
        color_primaries="bt2020",
        color_range="tv",
    )

    def unavailable(args, *, stage):
        assert stage == "color_preflight"
        raise subprocess.CalledProcessError(1, args, stderr="No such filter: zscale")

    monkeypatch.setattr(pipeline, "_run_short", unavailable)
    monkeypatch.setattr(
        temp_workspace, "create_temp_run_context", lambda *a, **kw: pytest.fail("started chunks")
    )
    with pytest.raises(pipeline.PipelineError, match="No such filter: zscale"):
        pipeline.run_pipeline(
            input_path=source,
            output_path=out,
            meta=meta,
            decisions=_make_decisions(),
            ffmpeg=Path(_FFMPEG),
            ffprobe=Path(_FFPROBE),
            hdr_to_sdr=True,
        )
    assert not out.exists()


def test_native_title_and_custom_tags_preserve_chapters(tmp_path):
    from video_enhance_lite.mp4_metadata import write_native_title

    output = tmp_path / "partial.mp4"
    chapters = tmp_path / "chapters.txt"
    chapters.write_text(
        ";FFMETADATA1\ncamera_id=cam-12\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=1000\ntitle=第一章\n",
        encoding="utf-8",
    )
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=64x48:r=10:d=1",
            "-f",
            "ffmetadata",
            "-i",
            str(chapters),
            "-map",
            "0:v",
            "-map_metadata",
            "1",
            "-map_chapters",
            "1",
            "-movflags",
            "use_metadata_tags",
            "-c:v",
            "libx264",
            str(output),
        ]
    )
    write_native_title(output, "撮影記録")
    info = json.loads(
        subprocess.check_output(
            [_FFPROBE, "-v", "error", "-show_format", "-show_chapters", "-of", "json", output]
        )
    )
    assert info["format"]["tags"]["title"] == "撮影記録"
    assert info["format"]["tags"]["camera_id"] == "cam-12"
    assert info["chapters"][0]["tags"]["title"] == "第一章"
    assert b"\xa9nam" in output.read_bytes()
