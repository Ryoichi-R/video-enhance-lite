from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import threading
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


@dataclass(frozen=True)
class VideoMetadata:
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


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 - shell=False, list argv
        argv,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        creationflags=_HIDE_WINDOW_FLAGS,
    )


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
) -> list[list[float]] | None:
    """Decode three bounded presentation-order frame windows with ffprobe."""
    window_span = min(2.0, max(0.25, duration_sec / 4.0))
    starts = (
        0.0,
        max(0.0, duration_sec / 2.0 - window_span / 2.0),
        max(0.0, duration_sec - window_span),
    )
    windows: list[list[float]] = []
    for start in starts:
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
        try:
            proc = _run(argv)
            payload = json.loads(proc.stdout or "{}")
        except (subprocess.CalledProcessError, json.JSONDecodeError, OSError, ValueError):
            return None
        values: list[float] = []
        for frame in payload.get("frames") or []:
            timestamp = _parse_float(frame.get("best_effort_timestamp_time"))
            if timestamp is not None and math.isfinite(timestamp):
                values.append(timestamp)
        # Packet-limited probing can end before delayed B frames are emitted. Ask for
        # extra packets, then classify only a bounded presentation-order prefix.
        windows.append(values[:_FRAME_INTERVAL_SAMPLE_LIMIT])
    return windows


def _classify_framerate(
    ffprobe: Path,
    input_path: Path,
    duration_sec: float,
    r_rate: Fraction,
    avg_rate: Fraction | None,
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
    windows = _probe_frame_timestamp_windows(ffprobe, input_path, duration_sec)
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


def probe_metadata(ffprobe: Path, input_path: Path) -> VideoMetadata:
    """Public for tests. Runs ffprobe for video/format and the complete stream inventory."""
    v_argv = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=index,width,height,r_frame_rate,avg_frame_rate,duration,codec_name,"
        "bit_rate,pix_fmt,color_range,color_space,field_order:format=duration,bit_rate",
        "-of",
        "json",
        str(input_path),
    ]
    try:
        v_proc = _run(v_argv)
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

    width = _parse_int(s.get("width")) or 0
    height = _parse_int(s.get("height")) or 0
    if width <= 0 or height <= 0:
        raise AnalyzerError(f"invalid resolution: {width}x{height}")

    r_rate = _parse_rate(s.get("r_frame_rate"))
    if r_rate is None:
        raise AnalyzerError(f"could not parse r_frame_rate: {s.get('r_frame_rate')!r}")

    stream_duration = _parse_float(s.get("duration"))
    format_duration = _parse_float(fmt.get("duration"))
    duration = format_duration if format_duration is not None else stream_duration
    if duration is None or not math.isfinite(duration) or duration <= 0:
        raise AnalyzerError(f"invalid duration: {fmt.get('duration')!r}/{s.get('duration')!r}")

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
        "stream=index,codec_type,codec_name:stream_tags=language:stream_disposition",
        "-of",
        "json",
        str(input_path),
    ]
    try:
        inventory_proc = _run(inventory_argv)
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

    framerate_status = _classify_framerate(
        ffprobe,
        input_path,
        float(duration),
        r_rate,
        avg_rate,
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
    )


def _extract_samples(
    ffmpeg: Path,
    input_path: Path,
    duration_sec: float,
    sample_count: int,
    out_dir: Path,
    cancel_event: threading.Event | None,
    progress: Callable[[int, int], None] | None,
    warning: Callable[[str], None] | None = None,
) -> list[Path]:
    if sample_count < 2:
        raise ValueError(f"sample_count must be >= 2, got {sample_count}")
    out_dir.mkdir(parents=True, exist_ok=True)
    samples: list[Path] = []
    for i in range(sample_count):
        if cancel_event is not None and cancel_event.is_set():
            raise AnalyzerError("cancelled during sampling")
        # 等間隔で sample_count 枚: t_i = duration * (i + 0.5) / sample_count
        t = duration_sec * (i + 0.5) / sample_count
        out_path = out_dir / f"sample_{i:04d}.png"
        argv = [
            str(ffmpeg),
            "-y",
            "-ss",
            f"{t:.6f}",
            "-i",
            str(input_path),
            "-frames:v",
            "1",
            "-f",
            "image2",
            str(out_path),
        ]
        try:
            _run(argv)
        except subprocess.CalledProcessError as e:
            raise AnalyzerError(
                f"ffmpeg sample extract failed at t={t:.3f}: {e.stderr or e}"
            ) from e
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
        if progress is not None:
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

    meta = probe_metadata(ffprobe, input_path)
    if cancel_event is not None and cancel_event.is_set():
        raise AnalyzerError("cancelled after probe")

    sd = sample_dir if sample_dir is not None else (config.derive_temp_dir(input_path) / "analyze")
    sd.mkdir(parents=True, exist_ok=True)
    try:
        effective_sample_count = max(2, int(sample_count))
        sample_paths = _extract_samples(
            ffmpeg=ffmpeg,
            input_path=input_path,
            duration_sec=meta.duration_sec,
            sample_count=effective_sample_count,
            out_dir=sd,
            cancel_event=cancel_event,
            progress=progress,
            warning=warning,
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
