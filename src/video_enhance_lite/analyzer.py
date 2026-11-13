from __future__ import annotations

import collections
import json
import math
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Literal

from . import config
from .deps import MissingDependencyError
from .media_streams import MediaStreamInfo

_HIDE_WINDOW_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

__all__ = [
    "VideoMetadata",
    "FramerateStatus",
    "FrameStats",
    "MediaStreamInfo",
    "analyze",
    "AnalyzerError",
    "compute_frame_stats",
    "FaceDetectionSample",
    "FaceRestoreCandidateRange",
]


class AnalyzerError(Exception):
    pass


FramerateStatus = Literal["cfr", "vfr_suspected", "unknown"]

_FPS_MISMATCH_RATIO = 0.001
_FRAME_INTERVAL_REL_TOLERANCE = 0.10
_FRAME_INTERVAL_ABS_TOLERANCE_SEC = 0.001
_FRAME_INTERVAL_MIN_DELTAS_PER_WINDOW = 4
_FRAME_INTERVAL_SAMPLE_LIMIT = 32
_FRAME_INTERVAL_PACKET_LIMIT = 64
_TRANSPORT_STREAM_SUFFIXES = frozenset({".ts", ".mts", ".m2ts"})
# FFmpeg rebases these demuxers (AVFMT_TS_DISCONT) on the earliest *used*
# stream instead of the container start. Measured with FFmpeg 9.0: mpegts and
# MPEG-PS ("mpeg") do; mov/mp4, matroska/webm, flv, and avi do not.
_PER_STREAM_REBASE_FORMATS = frozenset({"mpegts", "mpegtsraw", "mpeg"})


@dataclass(frozen=True)
class VideoMetadata:
    # Dimensions after FFmpeg autorotation; duration_sec is video-only.
    width: int
    height: int
    fps: Fraction
    duration_sec: float
    codec: str
    bitrate_bps: int | None
    pix_fmt: str
    color_range: str
    color_space: str
    field_order: str  # "progressive" / "tt" / "bb" / "unknown"
    has_audio: bool
    avg_fps: Fraction | None = None
    framerate_status: FramerateStatus = "unknown"
    primary_video_index: int = 0
    streams: tuple[MediaStreamInfo, ...] | None = None
    container_duration_sec: float | None = None
    # Video start relative to the container start that FFmpeg uses for input
    # seeking. Audio/subtitles may begin earlier; processing must not drop it.
    video_start_offset_sec: float = 0.0
    # Earliest audio/subtitle start relative to the container start, if any.
    preserved_streams_start_offset_sec: float | None = None
    # True when FFmpeg rebases timestamps on the streams a command uses.
    per_stream_timestamp_rebase: bool = False


FaceCandidateConfidence = Literal["high", "medium", "low", "unknown"]


@dataclass(frozen=True)
class FaceDetectionSample:
    """事前分析用の顔候補検出結果。

    confidence は stage recommendation ではなく、OpenCV Haar cascade による
    参考候補検出の信頼度を表す。
    """

    timestamp_sec: float
    face_count: int
    max_face_ratio: float
    confidence: FaceCandidateConfidence


@dataclass(frozen=True)
class FaceRestoreCandidateRange:
    start_sec: float
    end_sec: float
    sample_count: int
    face_sample_count: int
    max_face_ratio: float
    confidence: FaceCandidateConfidence
    note: str


@dataclass(frozen=True)
class FrameStats:
    laplacian_var_mean: float
    luma_std_over_time: float
    interframe_diff_mean: float
    interframe_diff_std: float
    histogram_flat_ratio: float
    luma_mean: float
    luma_clipping_ratio: float
    saturation_mean: float
    highfreq_energy: float
    sample_count: int
    face_restore_candidates: tuple[FaceRestoreCandidateRange, ...] = field(default_factory=tuple)


_FACE_CASCADE_LOCK = threading.Lock()
_FACE_CASCADE: Any | None = None
_FACE_CASCADE_LOAD_ATTEMPTED = False


def _require_numpy_cv2() -> tuple[Any, Any]:
    try:
        import numpy as np  # noqa: F401
        import cv2  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised in lazy import test
        raise MissingDependencyError(
            "numpy/opencv-python-headless",
            "事前分析には numpy と opencv-python-headless が必要です。\n"
            'インストール: pip install -e ".[enhance]"',
        ) from e
    import numpy as _np
    import cv2 as _cv2

    return _np, _cv2


def _read_image_path_safe(path: Path, np: Any, cv2: Any) -> Any | None:
    try:
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


_CANCEL_POLL_SEC = 0.2
# ffprobe -show_packets reads the whole file without decoding; allow for a slow
# (network or USB) source at this throughput before declaring it stuck.
_PACKET_SCAN_MIN_BYTES_PER_SEC = 10 * 1024 * 1024


def _run(
    argv: list[str],
    *,
    timeout: float = config.DEFAULT_ANALYZE_PROCESS_TIMEOUT_SEC,
    cancel_event: threading.Event | None = None,
) -> subprocess.CompletedProcess:
    """Run ffprobe/ffmpeg; stop it on timeout (total wall time) or cancellation.

    A damaged file or an unresponsive network share must not hang the CLI/GUI.
    Non-zero exit raises CalledProcessError as subprocess.run(check=True) does.
    """
    proc = subprocess.Popen(  # noqa: S603 - shell=False, list argv
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        creationflags=_HIDE_WINDOW_FLAGS,
    )
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=_CANCEL_POLL_SEC)
                break
            except subprocess.TimeoutExpired:
                cancelled = cancel_event is not None and cancel_event.is_set()
                if not cancelled and time.monotonic() < deadline:
                    continue
                proc.kill()
                proc.communicate()
                name = Path(str(argv[0])).name
                if cancelled:
                    raise AnalyzerError(f"cancelled while running {name}") from None
                raise AnalyzerError(
                    f"{name} did not finish within {timeout:.0f}s; the input may be damaged "
                    "or on a slow or unresponsive drive"
                ) from None
    except BaseException:
        # Ctrl+C or any other interruption: never leave the child running.
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
        raise
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, argv, stdout, stderr)
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)


def _parse_rate(rate: str | None) -> Fraction | None:
    if not rate or rate == "0/0":
        return None
    try:
        return Fraction(rate)
    except (ValueError, ZeroDivisionError):
        return None


def _parse_float(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        if not s or s.upper() == "N/A":
            return None
        try:
            return float(s)
        except ValueError:
            return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _parse_int(val: Any) -> int | None:
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        if not s or s.upper() == "N/A":
            return None
        try:
            return int(s)
        except ValueError:
            return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _probe_frame_timestamp_windows(
    ffprobe: Path,
    input_path: Path,
    duration_sec: float,
    cancel_event: threading.Event | None = None,
    video_start_sec: float = 0.0,
    nominal_fps: Fraction | None = None,
) -> list[list[float]] | None:
    """Sample presentation timestamps, retrying failed seeks by decoding from the start.

    ffprobe read_intervals uses absolute stream timestamps. A packet-limited
    seek can miss every keyframe in a long GOP; never classify an earlier
    window as evidence for the requested one.
    """
    window_span = min(2.0, max(0.25, duration_sec / 4.0))
    # Observe enough frames even at low fps, while retaining the requirement
    # for four positive deltas. Short inputs use the whole video, not a tiny
    # quarter-window that makes otherwise usable CFR clips unclassifiable.
    frame_interval = 1.0 / float(nominal_fps) if nominal_fps and nominal_fps > 0 else 0.0
    window_span = min(
        duration_sec, max(window_span, (_FRAME_INTERVAL_MIN_DELTAS_PER_WINDOW + 2) * frame_interval)
    )
    starts = (
        0.0,
        max(0.0, duration_sec / 2.0 - window_span / 2.0),
        max(0.0, duration_sec - window_span),
    )
    windows: list[list[float]] = []
    for relative_start in starts:
        start = video_start_sec + relative_start
        end = start + window_span
        argv = [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-read_intervals",
            f"{start:.6f}%+#{_FRAME_INTERVAL_PACKET_LIMIT}",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(input_path),
        ]

        def read_window(command: list[str]) -> list[float]:
            proc = _run(command, cancel_event=cancel_event)
            payload = json.loads(proc.stdout or "{}")
            values = []
            for frame in payload.get("frames") or []:
                timestamp = _parse_float(frame.get("best_effort_timestamp_time"))
                if (
                    timestamp is not None
                    and math.isfinite(timestamp)
                    and start - 1e-6 <= timestamp < end
                ):
                    values.append(timestamp)
            return values[:_FRAME_INTERVAL_SAMPLE_LIMIT]

        try:
            values = read_window(argv)
            margin = 5.0
            while len(values) <= _FRAME_INTERVAL_MIN_DELTAS_PER_WINDOW:
                # Retry from progressively earlier points, decoding through the
                # requested window rather than stopping after 64 packets. This
                # avoids a full-file decode for an ordinary GOP in a long video.
                # At the video start, omit the seek entirely to include its
                # first keyframe. Timeout/cancellation remain active on retries.
                coarse = max(video_start_sec, start - margin)
                seek = f"{coarse:.6f}" if coarse > video_start_sec else ""
                # Read past the observed region so reordered B frames are
                # emitted; timestamps outside the observation window stay out
                # of classification. Reaching EOF flushes the decoder as well.
                decode_end = end + _FRAME_INTERVAL_SAMPLE_LIMIT * frame_interval
                argv[argv.index("-read_intervals") + 1] = f"{seek}%{decode_end:.6f}"
                values = read_window(argv)
                if not seek:
                    break
                margin *= 2
        except (subprocess.CalledProcessError, json.JSONDecodeError, OSError, ValueError):
            return None
        windows.append(values)
    return windows


def _classify_framerate(
    ffprobe: Path,
    input_path: Path,
    duration_sec: float,
    r_rate: Fraction,
    avg_rate: Fraction | None,
    cancel_event: threading.Event | None = None,
    video_start_sec: float = 0.0,
) -> FramerateStatus:
    # avg_frame_rate が欠落・無効な入力は、安定した短時間サンプルだけを根拠に
    # CFR と断定しない。PTS を失う pipeline では安全側の unknown へ倒す。
    if avg_rate is None or avg_rate <= 0:
        return "unknown"
    mismatch = abs(float(r_rate - avg_rate)) / max(float(r_rate), float(avg_rate))
    if mismatch > _FPS_MISMATCH_RATIO:
        return "vfr_suspected"

    nominal_rate = avg_rate
    if nominal_rate <= 0:
        return "unknown"
    windows = _probe_frame_timestamp_windows(
        ffprobe, input_path, duration_sec, cancel_event, video_start_sec, nominal_rate
    )
    if windows is None:
        return "unknown"

    deltas: list[float] = []
    for timestamps in windows:
        window_deltas: list[float] = []
        for previous, current in zip(timestamps, timestamps[1:]):
            delta = current - previous
            if delta <= 0:
                return "unknown"
            window_deltas.append(delta)
        if len(window_deltas) < _FRAME_INTERVAL_MIN_DELTAS_PER_WINDOW:
            return "unknown"
        deltas.extend(window_deltas)

    nominal_interval = 1.0 / float(nominal_rate)
    tolerance = max(
        nominal_interval * _FRAME_INTERVAL_REL_TOLERANCE,
        _FRAME_INTERVAL_ABS_TOLERANCE_SEC,
    )
    epsilon = 1e-9
    if any(abs(delta - nominal_interval) > tolerance + epsilon for delta in deltas):
        return "vfr_suspected"
    if max(deltas) - min(deltas) > tolerance + epsilon:
        return "vfr_suspected"
    return "cfr"


def _probe_video_packet_duration(
    ffprobe: Path,
    input_path: Path,
    fps: Fraction,
    cancel_event: threading.Event | None = None,
) -> float:
    """Containers such as Matroska may omit stream.duration.

    Read video packet timestamps (no frame decoding), never substitute the
    container duration: another stream may continue after the video ends.
    One short `key=value` line per packet keeps memory use low on long files.
    """
    argv = [
        str(ffprobe),
        "-v",
        "error",
        # Reconstruct omitted PTS from DTS when possible (notably the final
        # MPEG-PS packet); otherwise its last frame would be excluded.
        "-fflags",
        "+genpts",
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_entries",
        "packet=pts_time,duration_time",
        "-of",
        "compact=p=0:nk=0",
        str(input_path),
    ]
    try:
        size = input_path.stat().st_size
    except OSError:
        size = 0
    timeout = max(
        float(config.DEFAULT_ANALYZE_PROCESS_TIMEOUT_SEC), size / _PACKET_SCAN_MIN_BYTES_PER_SEC
    )
    try:
        output = _run(argv, timeout=timeout, cancel_event=cancel_event).stdout or ""
    except (subprocess.CalledProcessError, OSError) as exc:
        raise AnalyzerError(f"ffprobe(video duration) failed: {exc}") from exc
    first_pts = math.inf
    last_end = -math.inf
    for line in output.splitlines():
        packet = dict(item.split("=", 1) for item in line.strip().split("|") if "=" in item)
        pts = _parse_float(packet.get("pts_time"))
        if pts is None or not math.isfinite(pts):
            continue
        packet_duration = _parse_float(packet.get("duration_time"))
        if packet_duration is None or not math.isfinite(packet_duration) or packet_duration <= 0:
            packet_duration = 1.0 / float(fps)
        first_pts = min(first_pts, pts)
        last_end = max(last_end, pts + packet_duration)
    duration = last_end - first_pts
    if not math.isfinite(duration) or duration <= 0:
        raise AnalyzerError("invalid duration: no usable video packet timestamps")
    return duration


def probe_metadata(
    ffprobe: Path,
    input_path: Path,
    *,
    cancel_event: threading.Event | None = None,
) -> VideoMetadata:
    """Public for tests. Runs ffprobe for video/format and the complete stream inventory."""
    v_argv = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=index,width,height,r_frame_rate,avg_frame_rate,duration,start_time,codec_name,"
        "bit_rate,pix_fmt,color_range,color_space,field_order:"
        "format=duration,start_time,bit_rate,format_name:"
        "stream_side_data=rotation:stream_tags=rotate",
        "-of",
        "json",
        str(input_path),
    ]
    try:
        v_proc = _run(v_argv, cancel_event=cancel_event)
    except subprocess.CalledProcessError as e:
        raise AnalyzerError(f"ffprobe(video stream) failed: {e.stderr or e}") from e
    try:
        v_data = json.loads(v_proc.stdout or "{}")
    except json.JSONDecodeError as e:
        raise AnalyzerError(f"ffprobe(video) JSON decode failed: {e}") from e

    streams = v_data.get("streams") or []
    if not streams:
        raise AnalyzerError("no video stream found")
    s = streams[0]
    fmt = v_data.get("format") or {}
    format_names = {name.strip() for name in str(fmt.get("format_name") or "").split(",")}

    width = _parse_int(s.get("width")) or 0
    height = _parse_int(s.get("height")) or 0
    if width <= 0 or height <= 0:
        raise AnalyzerError(f"invalid resolution: {width}x{height}")

    rotation = _parse_float((s.get("tags") or {}).get("rotate")) or 0.0
    for side_data in s.get("side_data_list") or []:
        if "rotation" in side_data:
            rotation = _parse_float(side_data["rotation"])
            break
    if rotation is None or not math.isfinite(rotation):
        raise AnalyzerError("invalid video rotation")
    if any(math.isclose(rotation % 360, angle, abs_tol=0.001) for angle in (90, 270)):
        width, height = height, width

    r_rate = _parse_rate(s.get("r_frame_rate"))
    if r_rate is None or r_rate <= 0:
        raise AnalyzerError(f"could not parse r_frame_rate: {s.get('r_frame_rate')!r}")

    stream_duration = _parse_float(s.get("duration"))
    format_duration = _parse_float(fmt.get("duration"))
    # MPEG-PS stream.duration can omit the final frame duration. Packet ends
    # are authoritative here, as for containers without a stream duration.
    if (
        "mpeg" not in format_names
        and stream_duration is not None
        and math.isfinite(stream_duration)
        and stream_duration > 0
    ):
        duration = stream_duration
    else:
        duration = _probe_video_packet_duration(ffprobe, input_path, r_rate, cancel_event)
    video_start = _parse_float(s.get("start_time"))
    container_start = _parse_float(fmt.get("start_time"))
    video_start_offset = 0.0
    if (
        video_start is not None
        and container_start is not None
        and math.isfinite(video_start)
        and math.isfinite(container_start)
    ):
        # FFmpeg input seeking is relative to the container start (the earliest
        # stream). A later video start must be added to every video seek.
        video_start_offset = max(0.0, video_start - container_start)
    if format_duration is None or not math.isfinite(format_duration) or format_duration <= 0:
        format_duration = video_start_offset + duration

    codec = str(s.get("codec_name") or "unknown")
    bitrate = _parse_int(s.get("bit_rate"))
    if bitrate is None:
        bitrate = _parse_int(fmt.get("bit_rate"))
    pix_fmt = str(s.get("pix_fmt") or "unknown")
    color_range = str(s.get("color_range") or "unknown")
    color_space = str(s.get("color_space") or "unknown")
    field_order = str(s.get("field_order") or "unknown")
    avg_rate = _parse_rate(s.get("avg_frame_rate"))
    primary_video_index = _parse_int(s.get("index"))
    if primary_video_index is None:
        # Older ffprobe-shaped test doubles omitted index. Real ffprobe includes it.
        primary_video_index = 0
    if primary_video_index < 0:
        raise AnalyzerError(f"invalid primary video stream index: {s.get('index')!r}")

    inventory_argv = [
        str(ffprobe),
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type,codec_name,start_time:stream_tags=language:stream_disposition",
        "-of",
        "json",
        str(input_path),
    ]
    try:
        inventory_proc = _run(inventory_argv, cancel_event=cancel_event)
    except subprocess.CalledProcessError as e:
        raise AnalyzerError(f"ffprobe(stream inventory) failed: {e.stderr or e}") from e
    try:
        inventory_data = json.loads(inventory_proc.stdout or "{}")
    except json.JSONDecodeError as e:
        raise AnalyzerError(f"ffprobe(stream inventory) JSON decode failed: {e}") from e

    inventory: list[MediaStreamInfo] = []
    for raw_stream in inventory_data.get("streams") or []:
        index = _parse_int(raw_stream.get("index"))
        if index is None or index < 0:
            raise AnalyzerError(f"invalid stream index in inventory: {raw_stream.get('index')!r}")
        codec_type = str(raw_stream.get("codec_type") or "unknown").strip().lower()
        codec_name = str(raw_stream.get("codec_name") or "unknown").strip().lower()
        tags = raw_stream.get("tags") or {}
        language_value = tags.get("language")
        language = str(language_value).strip() if language_value else None
        disposition_data = raw_stream.get("disposition") or {}
        disposition = tuple(
            sorted(
                str(name).strip().lower() for name, enabled in disposition_data.items() if enabled
            )
        )
        inventory.append(
            MediaStreamInfo(
                index=index,
                codec_type=codec_type,
                codec_name=codec_name,
                language=language,
                disposition=disposition,
            )
        )
    has_audio = any(stream.codec_type == "audio" for stream in inventory)
    preserved_starts = [
        start - container_start
        for raw_stream in inventory_data.get("streams") or []
        if str(raw_stream.get("codec_type") or "").strip().lower() in {"audio", "subtitle"}
        for start in [_parse_float(raw_stream.get("start_time"))]
        if start is not None
        and container_start is not None
        and math.isfinite(start)
        and math.isfinite(container_start)
    ]
    preserved_streams_start_offset = max(0.0, min(preserved_starts)) if preserved_starts else None

    framerate_status = _classify_framerate(
        ffprobe,
        input_path,
        float(duration),
        r_rate,
        avg_rate,
        cancel_event=cancel_event,
        video_start_sec=video_start
        if video_start is not None and math.isfinite(video_start)
        else 0.0,
    )

    return VideoMetadata(
        width=width,
        height=height,
        fps=r_rate,
        duration_sec=float(duration),
        codec=codec,
        bitrate_bps=bitrate,
        pix_fmt=pix_fmt,
        color_range=color_range,
        color_space=color_space,
        field_order=field_order,
        has_audio=has_audio,
        avg_fps=avg_rate,
        framerate_status=framerate_status,
        primary_video_index=primary_video_index,
        streams=tuple(inventory),
        container_duration_sec=float(format_duration),
        video_start_offset_sec=float(video_start_offset),
        preserved_streams_start_offset_sec=preserved_streams_start_offset,
        per_stream_timestamp_rebase=bool(format_names & _PER_STREAM_REBASE_FORMATS),
    )


def _run_with_frame_progress(
    argv: list[str],
    *,
    cancel_event: threading.Event | None = None,
    idle_timeout_sec: float | None = None,
    on_progress: Callable[[], None] | None = None,
) -> None:
    """Run a decode pass with an advancing-frame idle limit, not a wall-time limit.

    The first output must process every video frame and FFmpeg must emit
    ``-progress pipe:1``. Repeated counters/log chatter do not extend the limit.
    Reader threads drain both pipes on Windows as well as POSIX; diagnostic
    storage is bounded and cancellation always kills and reaps the child.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise AnalyzerError("cancelled during sampling")
    timeout = (
        float(config.DEFAULT_ANALYZE_PROCESS_TIMEOUT_SEC)
        if idle_timeout_sec is None
        else idle_timeout_sec
    )
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
            creationflags=_HIDE_WINDOW_FLAGS,
        )
    except OSError as exc:
        raise AnalyzerError(f"could not start sample extraction: {exc}") from exc
    lock = threading.Lock()
    last_frame = -1
    last_advance = time.monotonic()
    errors: collections.deque[str] = collections.deque(maxlen=64)

    def read_progress() -> None:
        nonlocal last_frame, last_advance
        for line in proc.stdout:
            key, separator, value = line.strip().partition("=")
            if separator and key == "frame":
                frame = _parse_int(value)
                with lock:
                    if frame is not None and frame > last_frame:
                        last_frame = frame
                        last_advance = time.monotonic()

    def read_errors() -> None:
        for line in proc.stderr:
            errors.append(line[-4096:])

    readers = [
        threading.Thread(target=target, daemon=True) for target in (read_progress, read_errors)
    ]
    try:
        for reader in readers:
            reader.start()
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise AnalyzerError("cancelled during sampling")
            try:
                proc.wait(timeout=_CANCEL_POLL_SEC)
                break
            except subprocess.TimeoutExpired:
                pass
            if on_progress is not None:
                on_progress()
            with lock:
                idle = time.monotonic() - last_advance
            if idle >= timeout:
                raise AnalyzerError(f"sample extraction made no decoding progress for {timeout:g}s")
        for reader in readers:
            reader.join()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, argv, stderr="".join(errors))
        if on_progress is not None:
            on_progress()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for reader in readers:
            if reader.ident is not None:
                reader.join()
        proc.stdout.close()
        proc.stderr.close()


def _extract_samples_one_pass(
    ffmpeg: Path,
    input_path: Path,
    duration_sec: float,
    paths: list[Path],
    cancel_event: threading.Event | None,
    progress: Callable[[int, int], None] | None,
) -> None:
    # Share one decode across sample branches. Each branch emits a seed at
    # frame zero to initialize its encoder immediately, then its actual sample.
    # Without seeds, FFmpeg may defer progress until every distant sample has
    # initialized its output. Independent branches also preserve duplicates
    # when multiple sample times land in the same frame interval.
    count = len(paths)
    graph = (
        f"[0:v:0]setpts=PTS-STARTPTS,split={count + 1}[scan]"
        + "".join(f"[s{i}]" for i in range(count))
        + ";"
        + ";".join(
            f"[s{i}]select='eq(n,0)+gte(t,{duration_sec * (i + 0.5) / count:.6f})',"
            f"trim=end_frame=2,setpts=PTS-STARTPTS[o{i}]"
            for i in range(count)
        )
    )
    argv = [
        str(ffmpeg),
        "-y",
        "-nostdin",
        "-v",
        "error",
        "-nostats",
        "-progress",
        "pipe:1",
        "-stats_period",
        "0.5",
        "-i",
        str(input_path),
        "-filter_complex",
        graph,
        # The first output counts every decoded frame, including the stretches
        # between samples. The null muxer writes no media data.
        "-map",
        "[scan]",
        "-f",
        "null",
        "-",
    ]
    with tempfile.TemporaryDirectory(prefix=".sampling-", dir=paths[0].parent) as scratch:
        selected = []
        for i, path in enumerate(paths):
            prefix = Path(scratch) / f"sample_{i:04d}"
            selected.append(Path(f"{prefix}_1.png"))
            argv += [
                "-map",
                f"[o{i}]",
                "-frames:v",
                "2",
                "-fps_mode",
                "passthrough",
                "-threads:v",
                "1",
                "-start_number",
                "0",
                f"{prefix}_%d.png",
            ]
        reported = 0

        def report_completed() -> None:
            nonlocal reported
            completed = sum(path.is_file() for path in selected)
            if progress is not None and completed > reported:
                reported = completed
                progress(completed, count)

        _run_with_frame_progress(argv, cancel_event=cancel_event, on_progress=report_completed)
        for source, destination in zip(selected, paths):
            if source.is_file():
                source.replace(destination)


def _extract_samples(
    ffmpeg: Path,
    input_path: Path,
    duration_sec: float,
    sample_count: int,
    out_dir: Path,
    cancel_event: threading.Event | None,
    progress: Callable[[int, int], None] | None,
    warning: Callable[[str], None] | None = None,
    start_offset_sec: float = 0.0,
    per_stream_timestamp_rebase: bool = False,
) -> list[Path]:
    if sample_count < 2:
        raise ValueError(f"sample_count must be >= 2, got {sample_count}")
    out_dir.mkdir(parents=True, exist_ok=True)
    decode_seek = (
        per_stream_timestamp_rebase or input_path.suffix.lower() in _TRANSPORT_STREAM_SUFFIXES
    )
    paths = [out_dir / f"sample_{i:04d}.png" for i in range(sample_count)]
    if decode_seek:
        try:
            _extract_samples_one_pass(
                ffmpeg, input_path, duration_sec, paths, cancel_event, progress
            )
        except subprocess.CalledProcessError as exc:
            raise AnalyzerError(f"ffmpeg sample extract failed: {exc.stderr or exc}") from exc
    samples: list[Path] = []
    for i in range(sample_count):
        if cancel_event is not None and cancel_event.is_set():
            raise AnalyzerError("cancelled during sampling")
        # 等間隔で sample_count 枚: t_i = duration * (i + 0.5) / sample_count
        t = duration_sec * (i + 0.5) / sample_count
        out_path = paths[i]
        if not decode_seek:
            target = start_offset_sec + t
            argv = [
                str(ffmpeg),
                "-y",
                "-ss",
                f"{target:.6f}",
                "-i",
                str(input_path),
                "-frames:v",
                "1",
                "-f",
                "image2",
                str(out_path),
            ]
            try:
                _run(argv, cancel_event=cancel_event)
            except subprocess.CalledProcessError as exc:
                raise AnalyzerError(
                    f"ffmpeg sample extract failed at t={t:.3f}: {exc.stderr or exc}"
                ) from exc
        if not out_path.exists() and i == sample_count - 1 and samples:
            if warning is not None:
                warning(
                    "final analyzer sample was beyond the last decodable frame "
                    f"at t={t:.3f}; continuing with {len(samples)} samples"
                )
            if progress is not None:
                progress(i + 1, sample_count)
            continue
        if not out_path.exists():
            raise AnalyzerError(f"ffmpeg sample extract produced no file at t={t:.3f}")
        samples.append(out_path)
        if progress is not None and not decode_seek:
            progress(i + 1, sample_count)
    return samples


def _sample_timestamp(duration_sec: float, index: int, sample_count: int) -> float:
    if sample_count < 2:
        raise ValueError(f"sample_count must be >= 2, got {sample_count}")
    return float(duration_sec) * (int(index) + 0.5) / int(sample_count)


def _face_sample_confidence(face_count: int, max_face_ratio: float) -> FaceCandidateConfidence:
    if face_count <= 0:
        return "unknown"
    if face_count >= 2 or max_face_ratio >= 0.08:
        return "high"
    if max_face_ratio >= 0.02:
        return "medium"
    return "low"


def _load_face_cascade(cv2: Any) -> Any | None:
    global _FACE_CASCADE, _FACE_CASCADE_LOAD_ATTEMPTED
    with _FACE_CASCADE_LOCK:
        if _FACE_CASCADE_LOAD_ATTEMPTED:
            return _FACE_CASCADE
        _FACE_CASCADE_LOAD_ATTEMPTED = True
        try:
            base = getattr(getattr(cv2, "data", None), "haarcascades", "")
            if not base:
                return None
            cascade_path = Path(base) / "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(str(cascade_path))
            if cascade.empty():
                return None
            _FACE_CASCADE = cascade
            return _FACE_CASCADE
        except Exception:
            return None


def _detect_face_samples(
    timestamped_frames_bgr: list[tuple[float, Any]],
    cancel_event: threading.Event | None = None,
) -> tuple[FaceDetectionSample, ...]:
    """OpenCV Haar cascade による軽量な参考候補検出。

    CodeFormer 本番の RetinaFace とは別検出器なので、ここで得た結果は
    face_restore の自動 ON 判定には使わない。
    """
    if cancel_event is not None and cancel_event.is_set():
        return ()
    _, cv2 = _require_numpy_cv2()
    cascade = _load_face_cascade(cv2)
    if cascade is None:
        return ()

    samples: list[FaceDetectionSample] = []
    for timestamp_sec, frame in timestamped_frames_bgr:
        if cancel_event is not None and cancel_event.is_set():
            return ()
        if frame is None or getattr(frame, "size", 0) == 0:
            continue
        try:
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(24, 24),
            )
        except Exception:
            continue
        face_count = int(len(faces))
        if face_count <= 0:
            continue
        frame_area = max(1, int(gray.shape[0]) * int(gray.shape[1]))
        max_face_ratio = max(float(w) * float(h) / frame_area for (_x, _y, w, h) in faces)
        samples.append(
            FaceDetectionSample(
                timestamp_sec=float(timestamp_sec),
                face_count=face_count,
                max_face_ratio=max_face_ratio,
                confidence=_face_sample_confidence(face_count, max_face_ratio),
            )
        )
    return tuple(samples)


def _range_confidence(face_sample_count: int, max_face_ratio: float) -> FaceCandidateConfidence:
    if face_sample_count >= 2 or max_face_ratio >= 0.08:
        return "high"
    if max_face_ratio >= 0.02:
        return "medium"
    return "low"


def _clamp_candidate_window(
    timestamp_sec: float,
    *,
    duration_sec: float,
    half_window_sec: float,
    min_width_sec: float,
) -> tuple[float, float]:
    start = max(0.0, float(timestamp_sec) - half_window_sec)
    end = min(float(duration_sec), float(timestamp_sec) + half_window_sec)
    if end - start < min_width_sec:
        extra = (min_width_sec - (end - start)) / 2.0
        start = max(0.0, start - extra)
        end = min(float(duration_sec), end + extra)
        if end - start < min_width_sec:
            if start <= 0.0:
                end = min(float(duration_sec), start + min_width_sec)
            elif end >= float(duration_sec):
                start = max(0.0, end - min_width_sec)
    if end <= start:
        end = min(float(duration_sec), start + max(0.001, min_width_sec))
    return start, end


def _build_face_restore_candidate_ranges(
    detections: list[FaceDetectionSample] | tuple[FaceDetectionSample, ...],
    *,
    duration_sec: float,
    sample_count: int,
) -> tuple[FaceRestoreCandidateRange, ...]:
    duration = float(duration_sec)
    if duration <= 0 or sample_count < 2:
        return ()
    face_samples = sorted(
        (d for d in detections if int(d.face_count) > 0),
        key=lambda d: float(d.timestamp_sec),
    )
    if not face_samples:
        return ()

    sample_interval = duration / max(2, int(sample_count))
    half_window = min(sample_interval / 2.0, 5.0)
    min_width = min(2.0, sample_interval, max(0.001, half_window * 2.0))
    merge_gap = sample_interval * 1.5
    coarse_note = (
        " サンプル間隔が粗いため短い出現は取り逃す場合があります。"
        if sample_interval > 10.0
        else ""
    )

    groups: list[tuple[float, float, list[FaceDetectionSample]]] = []
    for det in face_samples:
        start, end = _clamp_candidate_window(
            float(det.timestamp_sec),
            duration_sec=duration,
            half_window_sec=half_window,
            min_width_sec=min_width,
        )
        if groups and start - groups[-1][1] <= merge_gap:
            prev_start, prev_end, prev_dets = groups[-1]
            groups[-1] = (prev_start, max(prev_end, end), [*prev_dets, det])
        else:
            groups.append((start, end, [det]))

    ranges: list[FaceRestoreCandidateRange] = []
    for start, end, group in groups:
        max_ratio = max(float(d.max_face_ratio) for d in group)
        confidence = _range_confidence(len(group), max_ratio)
        ranges.append(
            FaceRestoreCandidateRange(
                start_sec=max(0.0, min(duration, start)),
                end_sec=max(0.0, min(duration, end)),
                sample_count=len(group),
                face_sample_count=len(group),
                max_face_ratio=max_ratio,
                confidence=confidence,
                note="OpenCV Haar cascade による事前分析の参考候補です。"
                "CodeFormer 本番検出とは一致しない場合があります。" + coarse_note,
            )
        )
    return tuple(r for r in ranges if r.end_sec > r.start_sec)


def compute_frame_stats(frames_bgr: list[Any]) -> FrameStats:
    """Pure function. Takes a list of BGR ndarrays (HxWx3 uint8) and returns FrameStats.

    Separated for tests: callers can pass synthetic numpy arrays.
    """
    np, cv2 = _require_numpy_cv2()
    if not frames_bgr:
        raise ValueError("frames_bgr must be non-empty")

    luma_means: list[float] = []
    laplacian_vars: list[float] = []
    saturations: list[float] = []
    highfreq_energies: list[float] = []
    histogram_flat_ratios: list[float] = []
    clipping_ratios: list[float] = []

    grays: list[Any] = []
    for img in frames_bgr:
        if img is None or img.size == 0:
            continue
        # cv2.imread already returns BGR uint8
        if img.ndim == 2:
            gray = img
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            bgr = img
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        grays.append(gray)

        luma_means.append(float(np.mean(gray)))
        # Laplacian variance: ブラー指標（低いほどボケ）
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_vars.append(float(lap.var()))
        # 彩度（HSV）
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        saturations.append(float(np.mean(hsv[..., 1])))
        # high-pass energy: gaussian blur との差分 RMS
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        diff = gray.astype(np.float32) - blurred.astype(np.float32)
        highfreq_energies.append(float(np.sqrt(np.mean(diff * diff))))
        # ヒストグラムのゼロ bin 比率 (バンディング指標の近似)。連続区間は見ていない。
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        zero_bins = int(np.sum(hist == 0))
        histogram_flat_ratios.append(zero_bins / 256.0)
        # クリッピング: luma が 0..2 or 253..255
        total = gray.size
        clipped = int(np.sum(gray <= 2)) + int(np.sum(gray >= 253))
        clipping_ratios.append(clipped / max(total, 1))

    if not grays:
        raise ValueError("no usable frames in frames_bgr")

    # 時間方向 luma 標準偏差: フリッカー指標
    luma_std_over_time = float(np.std(luma_means)) if len(luma_means) > 1 else 0.0

    # 隣接サンプル absdiff の平均/標準偏差: 手ブレ指標
    diffs: list[float] = []
    for prev_g, cur_g in zip(grays[:-1], grays[1:]):
        if prev_g.shape != cur_g.shape:
            continue
        d = cv2.absdiff(prev_g, cur_g)
        diffs.append(float(np.mean(d)))
    interframe_diff_mean = float(np.mean(diffs)) if diffs else 0.0
    interframe_diff_std = float(np.std(diffs)) if len(diffs) > 1 else 0.0

    return FrameStats(
        laplacian_var_mean=float(np.mean(laplacian_vars)) if laplacian_vars else 0.0,
        luma_std_over_time=luma_std_over_time,
        interframe_diff_mean=interframe_diff_mean,
        interframe_diff_std=interframe_diff_std,
        histogram_flat_ratio=float(np.mean(histogram_flat_ratios))
        if histogram_flat_ratios
        else 0.0,
        luma_mean=float(np.mean(luma_means)) if luma_means else 0.0,
        luma_clipping_ratio=float(np.mean(clipping_ratios)) if clipping_ratios else 0.0,
        saturation_mean=float(np.mean(saturations)) if saturations else 0.0,
        highfreq_energy=float(np.mean(highfreq_energies)) if highfreq_energies else 0.0,
        sample_count=len(grays),
    )


def analyze(
    input_path: Path,
    ffprobe: Path,
    ffmpeg: Path,
    sample_count: int = config.DEFAULT_SAMPLE_COUNT,
    cancel_event: threading.Event | None = None,
    progress: Callable[[int, int], None] | None = None,
    *,
    sample_dir: Path | None = None,
    keep_samples: bool = False,
    warning: Callable[[str], None] | None = None,
) -> tuple[VideoMetadata, FrameStats]:
    """Probe metadata then sample frames and compute FrameStats.

    `sample_dir` defaults to `derive_temp_dir(input_path) / "analyze"`.
    """
    np, cv2 = _require_numpy_cv2()

    meta = probe_metadata(ffprobe, input_path, cancel_event=cancel_event)
    if cancel_event is not None and cancel_event.is_set():
        raise AnalyzerError("cancelled after probe")

    sd = sample_dir if sample_dir is not None else (config.derive_temp_dir(input_path) / "analyze")
    sd.mkdir(parents=True, exist_ok=True)
    try:
        # Do not request more timestamps than the nominal number of frames.
        # On a low-fps/short clip, several targets could otherwise fall after
        # the last frame and turn a usable video into a missing-middle-sample
        # error. Retain the minimum of two and the final-sample warning policy.
        nominal_frames = max(2, int(meta.duration_sec * float(meta.avg_fps or meta.fps) + 1e-6))
        effective_sample_count = min(max(2, int(sample_count)), nominal_frames)
        sample_paths = _extract_samples(
            ffmpeg=ffmpeg,
            input_path=input_path,
            duration_sec=meta.duration_sec,
            sample_count=effective_sample_count,
            out_dir=sd,
            cancel_event=cancel_event,
            progress=progress,
            warning=warning,
            start_offset_sec=meta.video_start_offset_sec,
            per_stream_timestamp_rebase=meta.per_stream_timestamp_rebase,
        )
        frames: list[Any] = []
        timestamped_frames: list[tuple[float, Any]] = []
        for index, p in enumerate(sample_paths):
            timestamp = _sample_timestamp(meta.duration_sec, index, effective_sample_count)
            img = _read_image_path_safe(p, np, cv2)
            if img is None:
                continue
            frames.append(img)
            timestamped_frames.append((timestamp, img))
        if not frames:
            raise AnalyzerError("could not read any sample frame")
        stats = compute_frame_stats(frames)
        face_samples = _detect_face_samples(timestamped_frames, cancel_event=cancel_event)
        face_candidates = _build_face_restore_candidate_ranges(
            face_samples,
            duration_sec=meta.duration_sec,
            sample_count=effective_sample_count,
        )
        stats = replace(stats, face_restore_candidates=face_candidates)
    finally:
        if not keep_samples:
            shutil.rmtree(sd, ignore_errors=True)

    return meta, stats
