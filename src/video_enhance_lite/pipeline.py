from __future__ import annotations

import collections
import concurrent.futures
import errno
import math
import os
import queue
import secrets
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Protocol

from . import config, deps, framerate, media_streams, stages, temp_workspace
from .decisions import StageDecision, decisions_by_key

try:
    from . import stabilize_cv  # type: ignore

    _STABILIZE_CV_AVAILABLE = True
except ImportError:
    stabilize_cv = None  # type: ignore
    _STABILIZE_CV_AVAILABLE = False

_HIDE_WINDOW_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

# Default timeouts (override via CLI flags / kwargs)
DEFAULT_SHORT_TIMEOUT_SEC = 60
DEFAULT_IO_IDLE_TIMEOUT_SEC = 120

__all__ = [
    "PipelineError",
    "PipelineCancelled",
    "ProgressEvent",
    "FaceRestoreOptions",
    "ProcessRunner",
    "RealProcessRunner",
    "STAGES",
    "DEFAULT_SHORT_TIMEOUT_SEC",
    "DEFAULT_IO_IDLE_TIMEOUT_SEC",
    "estimate_compression_ratio",
    "estimate_peak_temp_bytes",
    "resolve_framerate_normalization",
    "resolve_stream_preservation",
    "run_pipeline",
]

# 6 トップレベルステージ + 内部サブステージは ProgressEvent.label で表現
STAGES = (
    "probe",
    "analyze",
    "user_confirm",
    "process_chunks",
    "concat",
    "cleanup",
)


class PipelineError(Exception):
    def __init__(self, stage: str, message: str, chunk_idx: int | None = None):
        if chunk_idx is not None:
            message = f"[chunk_{chunk_idx}] {message}"
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.message = message
        self.chunk_idx = chunk_idx


class PipelineCancelled(Exception):
    def __init__(self, stage: str, chunk_idx: int | None = None):
        if chunk_idx is not None:
            msg = f"[{stage}] [chunk_{chunk_idx}] cancelled"
        else:
            msg = f"[{stage}] cancelled"
        super().__init__(msg)
        self.stage = stage
        self.chunk_idx = chunk_idx


def resolve_stream_preservation(
    meta,
    *,
    log: Callable[[str], None] | None = None,
) -> media_streams.StreamPreservationPlan:
    """Shared CLI/GUI/pipeline adapter for the fail-closed MP4 stream policy."""
    stream_inventory = getattr(meta, "streams", None)
    if stream_inventory is None:
        if log is not None:
            log(
                "[WARN] stream inventoryのないlegacy metadataです。"
                "字幕・複数音声の保持可否は保証されません"
            )
        return media_streams.legacy_stream_preservation_plan(
            bool(getattr(meta, "has_audio", False)),
        )
    try:
        return media_streams.resolve_mp4_stream_preservation(
            tuple(stream_inventory),
            primary_video_index=int(getattr(meta, "primary_video_index", 0)),
        )
    except media_streams.UnsupportedMediaStreamsError as exc:
        raise PipelineError("stream_preflight", str(exc)) from exc


def resolve_framerate_normalization(
    meta,
    *,
    log: Callable[[str], None] | None = None,
) -> framerate.FramerateNormalizationPlan:
    """Shared CLI/GUI/pipeline adapter for the fail-closed D11 policy."""
    status_missing = not hasattr(meta, "framerate_status")
    status = "cfr" if status_missing else str(getattr(meta, "framerate_status"))
    try:
        plan = framerate.resolve_framerate_normalization(
            status,
            getattr(meta, "fps", None),
            legacy_status_missing=status_missing,
        )
    except framerate.UnsupportedFramerateError as exc:
        avg_fps = getattr(meta, "avg_fps", None)
        raise PipelineError(
            "framerate_preflight",
            f"{exc} (status={status}, r={getattr(meta, 'fps', None)}, avg={avg_fps})",
        ) from exc
    if log is not None:
        if plan.requires_normalization:
            log(
                "[WARN] VFRの可能性があるため、PNG抽出前に"
                f"{plan.nominal_fps} fpsへ明示CFR normalizationします。"
                "frame duplication/dropが発生する場合があります"
            )
    return plan


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    chunk_index: int | None = None
    total_chunks: int | None = None
    label: str = ""
    # PR1: 後方互換でフレーム単位の進捗を追加 (ffmpeg `-progress pipe:1` の `frame=` から取得)
    frame: int | None = None
    total_frames: int | None = None


@dataclass(frozen=True)
class GpuRunOptions:
    """ncnn-vulkan (realesrgan / rife) 実行フラグの束 (PR-R3c)。

    `_process_chunk` の引数数を減らすため GPU 実行フラグをまとめた内部 dataclass。
    `run_pipeline` の公開シグネチャは従来どおり個別 kwarg のままで、内部で本型に束ねる。
    - tilesize: realesrgan `-t` (RIFE 非対応)
    - threads:  `-j` (realesrgan / RIFE 共通)
    - gpu_id:   `-g` 共通 GPU id (realesrgan は -1 非対応)
    - rife_gpu_id: RIFE 用 `-g` (rife_gpu_id を優先, -1=cpu 可。無ければ gpu_id)
    """

    tilesize: int | None = None
    threads: str | None = None
    gpu_id: int | None = None
    rife_gpu_id: int | None = None


@dataclass(frozen=True)
class FaceRestoreOptions:
    """CodeFormer face_restore と自然化合成の実行オプション。"""

    start_sec: float | None = None
    end_sec: float | None = None
    fidelity: float = config.DEFAULT_FACE_RESTORE_FIDELITY
    idle_timeout_sec: int = config.DEFAULT_FACE_RESTORE_IDLE_TIMEOUT_SEC
    detection_model: str = config.DEFAULT_FACE_RESTORE_DETECTION_MODEL
    blend: float = config.DEFAULT_FACE_RESTORE_BLEND
    mask: str = config.DEFAULT_FACE_RESTORE_MASK
    feather: int = config.DEFAULT_FACE_RESTORE_FEATHER
    mask_dilate: int = config.DEFAULT_FACE_RESTORE_MASK_DILATE
    mask_threshold: int = config.DEFAULT_FACE_RESTORE_MASK_THRESHOLD
    match_grain: float = config.DEFAULT_FACE_RESTORE_MATCH_GRAIN
    soften: float = config.DEFAULT_FACE_RESTORE_SOFTEN
    global_smooth: float = config.DEFAULT_FACE_RESTORE_GLOBAL_SMOOTH

    @classmethod
    def from_params(
        cls,
        params: dict | None,
        *,
        fallback: "FaceRestoreOptions | None" = None,
    ) -> "FaceRestoreOptions":
        base = fallback or cls()
        params = params or {}
        return cls(
            start_sec=params.get("start_sec", base.start_sec),
            end_sec=params.get("end_sec", base.end_sec),
            fidelity=float(params.get("fidelity", base.fidelity)),
            idle_timeout_sec=int(params.get("idle_timeout_sec", base.idle_timeout_sec)),
            detection_model=str(params.get("detection_model", base.detection_model)),
            blend=float(params.get("blend", base.blend)),
            mask=str(params.get("mask", base.mask)),
            feather=int(params.get("feather", base.feather)),
            mask_dilate=int(params.get("mask_dilate", base.mask_dilate)),
            mask_threshold=int(params.get("mask_threshold", base.mask_threshold)),
            match_grain=float(params.get("match_grain", base.match_grain)),
            soften=float(params.get("soften", base.soften)),
            global_smooth=float(params.get("global_smooth", base.global_smooth)),
        ).validate()

    def validate(self) -> "FaceRestoreOptions":
        for name in ("fidelity", "blend", "match_grain", "soften", "global_smooth"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not (0.0 <= value <= 1.0):
                raise ValueError(f"face_restore {name} must be in 0..1, got {value!r}")
        if self.mask not in config.FACE_RESTORE_MASK_MODES:
            raise ValueError(f"face_restore mask must be one of {config.FACE_RESTORE_MASK_MODES!r}")
        for name in ("feather", "mask_dilate", "mask_threshold", "idle_timeout_sec"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"face_restore {name} must be >= 0, got {value!r}")
        for name in ("feather", "mask_dilate"):
            value = int(getattr(self, name))
            if value > 256:
                raise ValueError(f"face_restore {name} must be in 0..256, got {value!r}")
        if self.idle_timeout_sec < 1:
            raise ValueError("face_restore idle_timeout_sec must be >= 1")
        if self.mask_threshold > 255:
            raise ValueError("face_restore mask_threshold must be in 0..255")
        if self.start_sec is not None and (
            not math.isfinite(float(self.start_sec)) or float(self.start_sec) < 0
        ):
            raise ValueError("face_restore start_sec must be >= 0")
        if self.end_sec is not None and (
            not math.isfinite(float(self.end_sec)) or float(self.end_sec) < 0
        ):
            raise ValueError("face_restore end_sec must be >= 0")
        return self


def _check_cancel(
    cancel_event: threading.Event | None, stage: str, chunk_idx: int | None = None
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise PipelineCancelled(stage, chunk_idx)


def _emit_progress(
    progress: Callable[[ProgressEvent], None] | None,
    stage: str,
    *,
    chunk_index: int | None = None,
    total_chunks: int | None = None,
    label: str = "",
    frame: int | None = None,
    total_frames: int | None = None,
) -> None:
    if progress is None:
        return
    progress(
        ProgressEvent(
            stage=stage,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            label=label,
            frame=frame,
            total_frames=total_frames,
        )
    )


def _run_short(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = DEFAULT_SHORT_TIMEOUT_SEC,
    stage: str = "subprocess",
    chunk_idx: int | None = None,
) -> subprocess.CompletedProcess:
    """Short-lived subprocess runner (ffprobe / concat / stabilize_detect / rename).

    `timeout` 秒で `subprocess.TimeoutExpired` -> `PipelineError`。
    """
    try:
        return subprocess.run(  # noqa: S603 - shell=False, list argv
            argv,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(cwd) if cwd is not None else None,
            shell=False,
            timeout=timeout,
            creationflags=_HIDE_WINDOW_FLAGS,
        )
    except subprocess.TimeoutExpired as e:
        raise PipelineError(
            stage,
            f"short subprocess timeout {timeout}s: {' '.join(argv[:2])}",
            chunk_idx=chunk_idx,
        ) from e


_STREAM_STDOUT = "out"
_STREAM_STDERR = "err"
_STREAM_EOF = "eof"


def _terminate_process(proc: "subprocess.Popen") -> None:
    """`terminate -> wait(1s) -> kill -> wait(1s)` でプロセスを確実に停止する。

    cancel / idle / finally の各経路で同じ手順を重複記述しないための内部ヘルパー。
    既に停止済みなら何もしない。terminate で素直に終わればそこで戻る。
    """
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=1.0)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass


class ProcessRunner(Protocol):
    """長時間プロセス実行の抽象 (V2-8 DI seam)。

    `run_with_progress` のみを契約とする。**失敗時は必ず `PipelineError` または
    `PipelineCancelled` を送出する** (生の `subprocess.CalledProcessError` 等を
    呼び出し側へ漏らしてはならない)。本番実装は `RealProcessRunner`、テストは
    構造的部分型 (Protocol) として `FakeProcessRunner` を差し込む。

    短命プロセス実行 (`_run_short`) は本番の長時間経路から呼ばれないため Protocol
    には含めない (全 fake へ不要な実装を強制しないため)。
    """

    def run_with_progress(
        self,
        argv: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        idle_timeout: int = DEFAULT_IO_IDLE_TIMEOUT_SEC,
        cancel_event: threading.Event | None = None,
        on_progress: Callable[[int], None] | None = None,
        on_output_event: Callable[[str, str], None] | None = None,
        on_heartbeat: Callable[[float, float], None] | None = None,
        heartbeat_interval_sec: float | None = None,
        stage: str = "ffmpeg",
        chunk_idx: int | None = None,
    ) -> int: ...


class RealProcessRunner:
    """`ProcessRunner` の本番実装 (ステートレス)。

    `subprocess.Popen` 監視ループ本体。stdout / stderr を別 thread で queue.Queue に
    流し、main thread は `queue.get(timeout=1.0)` で 1 秒粒度の cancel / idle /
    プロセス終了監視を回す (Windows の anonymous pipe でも selectors 非対応問題を回避)。

    `idle_timeout` は「最後の stdout/stderr 出力からの経過時間」で測る (chunk が長く
    ても誤殺しない)。非 0 終了 / idle / cancel / 起動失敗はすべて `PipelineError` /
    `PipelineCancelled` で送出する (例外契約を満たす)。
    """

    def run_with_progress(
        self,
        argv: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        idle_timeout: int = DEFAULT_IO_IDLE_TIMEOUT_SEC,
        cancel_event: threading.Event | None = None,
        on_progress: Callable[[int], None] | None = None,
        on_output_event: Callable[[str, str], None] | None = None,
        on_heartbeat: Callable[[float, float], None] | None = None,
        heartbeat_interval_sec: float | None = None,
        stage: str = "ffmpeg",
        chunk_idx: int | None = None,
    ) -> int:
        """Long-running subprocess runner with idle-timeout + cancel + progress monitoring.

        主な用途: ffmpeg (`-progress pipe:1` 付きでフレーム単位の `frame=N` を吸い上げる),
        realesrgan-ncnn-vulkan, rife-ncnn-vulkan のような長時間 GPU/CPU bound 処理。

        Returns: 終了コード (= 0)。Non-zero / idle / cancel ですべて例外を投げる。
        """
        # PR-R1: Popen 起動失敗 (実行ファイル不在・権限不足など) は try/finally の外で
        # 発生し、生 OSError として伝播してしまう。逐次実行時は CLI 側 except が
        # PipelineError 系しか拾わずスタックトレースのまま落ちるため、ここで包む。
        try:
            proc = subprocess.Popen(  # noqa: S603 - shell=False, list argv
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                creationflags=_HIDE_WINDOW_FLAGS,
            )
        except (FileNotFoundError, OSError) as e:
            launched = argv[0] if argv else "<empty argv>"
            raise PipelineError(
                stage,
                f"failed to launch process: {launched} ({e})",
                chunk_idx=chunk_idx,
            ) from e
        q: "queue.Queue[tuple[str, str]]" = queue.Queue(maxsize=4096)
        stderr_tail: collections.deque[str] = collections.deque(maxlen=800)  # ~64KB
        eofs_remaining = 2

        def _reader(stream, tag: str) -> None:
            try:
                if stream is None:
                    return
                buf: list[str] = []
                while True:
                    ch = stream.read(1)
                    if ch == "":
                        if buf:
                            try:
                                q.put((tag, "".join(buf)), timeout=1.0)
                            except queue.Full:
                                pass
                        break
                    # PR-R2: cancel 検知時は reader ループを早期終了し、密出力でも
                    # consumer 側の停止を妨げない。
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    if ch in ("\n", "\r"):
                        if buf:
                            try:
                                q.put((tag, "".join(buf) + ch), timeout=1.0)
                            except queue.Full:
                                # コンシューマが詰まったときは行を捨てる (詰まり継続を避ける)
                                pass
                            buf.clear()
                        continue
                    buf.append(ch)
            finally:
                try:
                    q.put((_STREAM_EOF, tag), timeout=1.0)
                except queue.Full:
                    pass

        t_out = threading.Thread(target=_reader, args=(proc.stdout, _STREAM_STDOUT), daemon=True)
        t_err = threading.Thread(target=_reader, args=(proc.stderr, _STREAM_STDERR), daemon=True)
        t_out.start()
        t_err.start()

        last_event = time.monotonic()
        started_at = last_event
        last_heartbeat = last_event
        last_frame = -1
        last_emit = 0.0

        def _emit_heartbeat_if_due(now: float) -> None:
            nonlocal last_heartbeat
            if (
                on_heartbeat is None
                or heartbeat_interval_sec is None
                or heartbeat_interval_sec <= 0
                or now - last_heartbeat < heartbeat_interval_sec
            ):
                return
            last_heartbeat = now
            try:
                on_heartbeat(now - started_at, now - last_event)
            except Exception:
                pass

        try:
            while True:
                try:
                    tag, line = q.get(timeout=1.0)
                except queue.Empty:
                    if cancel_event is not None and cancel_event.is_set():
                        _terminate_process(proc)
                        raise PipelineCancelled(stage, chunk_idx=chunk_idx)
                    if time.monotonic() - last_event > idle_timeout:
                        _terminate_process(proc)
                        raise PipelineError(
                            stage,
                            f"idle timeout {idle_timeout}s; no stdout/stderr was received "
                            f"for {idle_timeout}s. "
                            "If the process is healthy but unusually slow, adjust the GUI "
                            "'I/O無出力タイムアウト（秒）' setting or CLI "
                            "--io-idle-timeout-sec. last stderr tail: "
                            + ("".join(stderr_tail)[-2048:] or "<no stderr>"),
                            chunk_idx=chunk_idx,
                        )
                    now = time.monotonic()
                    _emit_heartbeat_if_due(now)
                    # プロセスが死んでいて queue も空なら短時間で抜ける
                    if proc.poll() is not None and q.empty():
                        # reader thread の EOF を待ち続ける
                        pass
                    continue

                # PR-R2: stdout/stderr が密に出続けると queue.Empty 分岐に入らず、
                # 旧実装では cancel が立っても次の出力途切れまで kill されなかった。
                # 出力受信側 (= 出力が途切れていない経路) でも行処理前に cancel を確認する。
                if cancel_event is not None and cancel_event.is_set():
                    _terminate_process(proc)
                    raise PipelineCancelled(stage, chunk_idx=chunk_idx)

                last_event = time.monotonic()

                if tag == _STREAM_EOF:
                    eofs_remaining -= 1
                    if eofs_remaining == 0:
                        break
                    continue

                # 高頻度出力で queue.Empty にならない場合も、表示用 heartbeat は
                # 定期発火させる。last_event は出力イベントだけで更新済みであり、
                # callback 自体が idle watchdog を延命することはない。
                _emit_heartbeat_if_due(last_event)

                if on_output_event is not None:
                    try:
                        on_output_event(tag, line)
                    except Exception:
                        pass

                if tag == _STREAM_STDERR:
                    stderr_tail.append(line)
                    continue

                # _STREAM_STDOUT: ffmpeg `-progress pipe:1` 行
                stripped = line.strip()
                if stripped.startswith("frame=") and on_progress is not None:
                    try:
                        n = int(stripped.split("=", 1)[1].strip())
                    except (ValueError, IndexError):
                        continue
                    now = time.monotonic()
                    if n != last_frame and (now - last_emit) >= 0.25:
                        last_frame = n
                        last_emit = now
                        try:
                            on_progress(n)
                        except Exception:
                            # progress コールバック例外は処理を止めない
                            pass

            rc = proc.wait(timeout=10)
            if rc != 0:
                raise PipelineError(
                    stage,
                    ("".join(stderr_tail)[-8192:] or f"exit code {rc}"),
                    chunk_idx=chunk_idx,
                )
            return rc
        finally:
            _terminate_process(proc)
            t_out.join(timeout=5)
            t_err.join(timeout=5)
            try:
                if proc.stdout is not None:
                    proc.stdout.close()
                if proc.stderr is not None:
                    proc.stderr.close()
            except Exception:
                pass


def _make_chunk_tmp_dirs(
    tmp_root: Path, chunk_idx: int, *, src_width: int
) -> dict[str, Path | int]:
    """Per-chunk tmp ディレクトリ。PR3 並列実行で衝突回避のため。"""
    base = tmp_root / f"chunk_{chunk_idx:03d}"
    return {
        "frames_in": base / "frames_in",
        "frames_pre": base / "frames_pre",
        "frames_up": base / "frames_up",
        "frames_clamp": base / "frames_clamp",
        "frames_face": base / "frames_face",
        "frames_face_input": base / "frames_face_input",
        "frames_face_raw": base / "frames_face_raw",
        "frames_rife": base / "frames_rife",
        "frames_out": base / "frames_out",
        "_src_width": int(src_width),
    }


def _augment_ffmpeg_with_progress(argv: list[str]) -> list[str]:
    """ffmpeg argv に `-progress pipe:1 -nostats` を挿入する (既に有れば no-op)。

    argv[0] が ffmpeg であることが前提。`-y` の直後に挿入する (`-y` がなければ argv[1] 直後)。
    """
    if not argv:
        return argv
    bin_lower = str(argv[0]).lower()
    if "ffmpeg" not in bin_lower:
        return argv
    if any(a == "-progress" for a in argv):
        return argv  # 既に含まれる
    # `-y` の直後に挿入 (典型的な layout: ffmpeg.exe -y -i ...)
    new_argv = list(argv)
    insert_at = 1
    if len(new_argv) >= 2 and new_argv[1] == "-y":
        insert_at = 2
    new_argv[insert_at:insert_at] = ["-progress", "pipe:1", "-nostats"]
    return new_argv


def _make_frame_emitter(
    progress: Callable[[ProgressEvent], None] | None,
    stage: str,
    chunk_idx: int,
    total_chunks: int,
    total_frames: int | None = None,
) -> Callable[[int], None] | None:
    """`runner.run_with_progress` の `on_progress` 用に ProgressEvent emitter を組み立てる。"""
    if progress is None:
        return None

    def _emit(n: int) -> None:
        _emit_progress(
            progress,
            stage,
            chunk_index=chunk_idx,
            total_chunks=total_chunks,
            label=f"chunk {chunk_idx + 1}/{total_chunks} {stage} frame={n}",
            frame=n,
            total_frames=total_frames,
        )

    return _emit


def _compute_chunk_ranges(
    duration_sec: float, chunk_seconds: int
) -> list[tuple[float, float, int]]:
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be > 0, got {duration_sec!r}")
    if chunk_seconds < 1:
        raise ValueError(f"chunk_seconds must be >= 1, got {chunk_seconds!r}")
    ranges: list[tuple[float, float, int]] = []
    idx = 0
    start = 0.0
    while start < duration_sec:
        end = min(start + float(chunk_seconds), duration_sec)
        dur = end - start
        if dur <= 0:
            break
        ranges.append((start, dur, idx))
        idx += 1
        start = end
    return ranges


def _normalized_frame_count(start_sec: float, dur_sec: float, fps: Fraction) -> int:
    start_frame = round(Fraction(str(start_sec)) * fps)
    end_frame = round(Fraction(str(start_sec + dur_sec)) * fps)
    count = end_frame - start_frame
    if count < 1:
        raise ValueError(
            "normalized chunk must contain at least one frame: "
            f"start={start_sec}, duration={dur_sec}, fps={fps}"
        )
    return count


def _reset_dirs(*dirs: Path) -> None:
    for d in dirs:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)


def _count_png(dir_: Path) -> int:
    if not dir_.exists():
        return 0
    return sum(1 for p in dir_.iterdir() if p.suffix.lower() == ".png")


def _has_png(dir_: Path) -> bool:
    """ディレクトリに PNG が 1 枚以上あるかを最初の 1 件で早期判定する (V2-7)。

    枚数 (magnitude) が不要で 0 枚判定しか使わない検証用。`_count_png` の全件走査を
    避け、`glob` ジェネレータの先頭 1 件で短絡する。出力は常に `%08d.png` (小文字) のため
    case-sensitive な glob でも取りこぼさない。
    """
    if not dir_.exists():
        return False
    return next(dir_.glob("*.png"), None) is not None


def _rename_dir_contents(src: Path, dst: Path) -> None:
    """Move all files from src to dst (dst must exist). Used when a stage is skipped."""
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        target = dst / p.name
        if target.exists():
            target.unlink()
        shutil.move(str(p), str(target))


def _copy_pngs(src: Path, dst: Path) -> list[Path]:
    dst.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for p in sorted(src.glob("*.png")):
        target = dst / p.name
        _link_or_copy_atomic(p, target)
        copied.append(target)
    return copied


_LINK_FALLBACK_ERRNOS = {
    errno.EXDEV,
    errno.EACCES,
    errno.EPERM,
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}


def _link_or_copy_atomic(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.parent / f".{dst.name}.{secrets.token_hex(8)}.tmp"
    try:
        try:
            os.link(src, temporary)
            result = "linked"
        except OSError as exc:
            if exc.errno not in _LINK_FALLBACK_ERRNOS:
                raise
            shutil.copy2(src, temporary)
            result = "copied"
        os.replace(temporary, dst)
        return result
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _atomic_replace_from_file(src: Path, dst: Path) -> None:
    _link_or_copy_atomic(src, dst)


def _read_image_path_safe(path: Path, flags: int):
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(data, flags)
    except OSError as exc:
        raise PipelineError("face_restore", f"failed to read image bytes: {path}: {exc}") from exc
    if image is None:
        raise PipelineError("face_restore", f"failed to decode image: {path}")
    return image


def _write_png_path_safe_atomic(path: Path, image) -> None:
    try:
        import cv2  # type: ignore

        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise PipelineError("face_restore", f"failed to encode composed frame: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.stem}.{secrets.token_hex(8)}.png"
        try:
            temporary.write_bytes(encoded.tobytes())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    except OSError as exc:
        raise PipelineError(
            "face_restore", f"failed to write composed frame: {path}: {exc}"
        ) from exc


def _frame_in_face_restore_range(
    *,
    frame_index: int,
    fps: Fraction,
    chunk_start_sec: float,
    start_sec: float | None,
    end_sec: float | None,
) -> bool:
    if fps <= 0:
        return False
    global_sec = float(chunk_start_sec) + (float(frame_index) / float(fps))
    if start_sec is not None and global_sec < float(start_sec):
        return False
    if end_sec is not None and global_sec >= float(end_sec):
        return False
    return True


def _find_codeformer_final_results(raw_dir: Path) -> Path | None:
    direct = raw_dir / "final_results"
    if direct.is_dir():
        return direct
    candidates = [p for p in raw_dir.rglob("final_results") if p.is_dir()]
    if candidates:
        return sorted(candidates)[0]
    return None


@dataclass(frozen=True)
class FaceRestoreCompositeResult:
    wrote: bool
    resized: bool = False
    empty_mask: bool = False


def _face_restore_needs_composite(options: FaceRestoreOptions) -> bool:
    return not (
        options.blend == config.DEFAULT_FACE_RESTORE_BLEND
        and options.mask == config.DEFAULT_FACE_RESTORE_MASK
        and options.match_grain == config.DEFAULT_FACE_RESTORE_MATCH_GRAIN
        and options.soften == config.DEFAULT_FACE_RESTORE_SOFTEN
        and options.global_smooth == config.DEFAULT_FACE_RESTORE_GLOBAL_SMOOTH
    )


def _gaussian_kernel_for_sigma(sigma: float) -> int:
    if sigma <= 0:
        return 0
    return max(3, int(math.ceil(sigma * 6.0)) | 1)


def _gaussian_kernel_for_radius(radius: int) -> int:
    if radius <= 0:
        return 0
    return max(3, (int(radius) * 2 + 1) | 1)


def _compose_face_restore_frame(
    *,
    original_path: Path,
    restored_path: Path,
    output_path: Path,
    options: FaceRestoreOptions,
) -> FaceRestoreCompositeResult:
    """Blend a CodeFormer result back into its source frame using float32 math."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as e:
        raise PipelineError(
            "face_restore",
            "naturalize composite requires numpy and opencv-python-headless",
        ) from e

    orig = _read_image_path_safe(original_path, cv2.IMREAD_COLOR)
    restored = _read_image_path_safe(restored_path, cv2.IMREAD_COLOR)

    resized = False
    if orig.shape[:2] != restored.shape[:2]:
        restored = cv2.resize(
            restored, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        resized = True

    orig_f = orig.astype(np.float32)
    restored_f = restored.astype(np.float32)

    if options.soften > 0:
        sigma = 0.3 + 1.2 * float(options.soften)
        ksize = _gaussian_kernel_for_sigma(sigma)
        restored_f = cv2.GaussianBlur(restored_f, (ksize, ksize), sigmaX=sigma)

    if options.mask == "none":
        mask = np.ones((orig.shape[0], orig.shape[1], 1), dtype=np.float32)
    elif options.mask == "diff":
        diff = cv2.absdiff(restored, orig)
        gray = np.max(diff, axis=2)
        binary = (gray > int(options.mask_threshold)).astype(np.uint8) * 255
        if not np.any(binary):
            return FaceRestoreCompositeResult(wrote=False, resized=resized, empty_mask=True)
        if options.mask_dilate > 0:
            ksize = _gaussian_kernel_for_radius(int(options.mask_dilate))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            binary = cv2.dilate(binary, kernel, iterations=1)
        if options.feather > 0:
            ksize = _gaussian_kernel_for_radius(int(options.feather))
            binary_f = cv2.GaussianBlur(binary.astype(np.float32), (ksize, ksize), sigmaX=0)
        else:
            binary_f = binary.astype(np.float32)
        mask = (binary_f / 255.0)[:, :, None]
    else:
        raise PipelineError("face_restore", f"unsupported face_restore mask: {options.mask!r}")

    effective_mask = np.clip(mask * float(options.blend), 0.0, 1.0)
    base = orig_f * (1.0 - effective_mask) + restored_f * effective_mask

    if options.match_grain > 0:
        orig_blur = cv2.GaussianBlur(orig_f, (0, 0), sigmaX=1.0)
        grain = orig_f - orig_blur
        base = base + grain * float(options.match_grain) * effective_mask

    if options.global_smooth > 0:
        sigma = 0.4 + 1.6 * float(options.global_smooth)
        ksize = _gaussian_kernel_for_sigma(sigma)
        smooth = cv2.GaussianBlur(base, (ksize, ksize), sigmaX=sigma)
        amount = float(options.global_smooth)
        base = base * (1.0 - amount) + smooth * amount

    out = np.clip(base, 0, 255).astype(np.uint8)
    _write_png_path_safe_atomic(output_path, out)
    return FaceRestoreCompositeResult(wrote=True, resized=resized)


def _run_face_restore_stage(
    *,
    runner: ProcessRunner,
    codeformer: deps.ResolvedCodeFormerLocal,
    input_dir: Path,
    output_dir: Path,
    selected_input_dir: Path,
    raw_output_dir: Path,
    fps: Fraction,
    chunk_idx: int,
    start_sec: float,
    total_chunks: int,
    progress: Callable[[ProgressEvent], None] | None,
    cancel_event: threading.Event | None,
    gpu_gate: threading.Semaphore | None,
    log: Callable[[str], None],
    tag: str,
    options: FaceRestoreOptions,
) -> int:
    """Apply CodeFormer to selected frames, leaving pass-through frames in place."""
    input_frames = sorted(input_dir.glob("*.png"))
    if not input_frames:
        raise PipelineError("face_restore", "no input frames", chunk_idx=chunk_idx)
    if (
        options.start_sec is not None
        and options.end_sec is not None
        and options.start_sec >= options.end_sec
    ):
        raise PipelineError(
            "face_restore",
            "--face-restore-start must be smaller than --face-restore-end",
            chunk_idx=chunk_idx,
        )

    # 先に全フレームを pass-through として置く。CodeFormer 出力が欠けたフレームだけ残る。
    _copy_pngs(input_dir, output_dir)

    selected: list[Path] = []
    for idx, frame in enumerate(input_frames):
        if _frame_in_face_restore_range(
            frame_index=idx,
            fps=fps,
            chunk_start_sec=start_sec,
            start_sec=options.start_sec,
            end_sec=options.end_sec,
        ):
            target = selected_input_dir / frame.name
            _link_or_copy_atomic(frame, target)
            selected.append(target)

    if not selected:
        log(f"       {tag} face_restore skipped (no frames in range)")
        return len(input_frames)

    needs_composite = _face_restore_needs_composite(options)
    natural_suffix = ""
    if needs_composite:
        natural_suffix = (
            f" blend={options.blend:.3f} mask={options.mask}"
            f" grain={options.match_grain:.3f} soften={options.soften:.3f}"
            f" smooth={options.global_smooth:.3f}"
        )
    log(
        f"       {tag} face_restore selected {len(selected)}/{len(input_frames)} frames "
        f"w={options.fidelity:.3f}{natural_suffix}"
    )
    _emit_progress(
        progress,
        "chunk_face_restore",
        chunk_index=chunk_idx,
        total_chunks=total_chunks,
        label=f"chunk {chunk_idx + 1}/{total_chunks} face_restore",
    )
    argv = stages.build_codeformer_args(
        python=codeformer.python,
        script=codeformer.script,
        input_dir=selected_input_dir,
        output_dir=raw_output_dir,
        fidelity=options.fidelity,
        detection_model=options.detection_model,
    )
    log(f"       {tag} face_restore running CodeFormer...")
    _run_gpu_stage(
        argv,
        runner=runner,
        stage="face_restore",
        chunk_idx=chunk_idx,
        cancel_event=cancel_event,
        io_idle_timeout_sec=options.idle_timeout_sec,
        gpu_gate=gpu_gate,
        cwd=codeformer.repo_root,
        env=stages.build_codeformer_env(python=codeformer.python),
        log=log,
    )

    final_results = _find_codeformer_final_results(raw_output_dir)
    if final_results is None:
        log(f"       {tag} [WARN] face_restore final_results not found; pass-through")
        return len(input_frames)

    restored = 0
    unmatched = 0
    resized = 0
    empty_mask = 0
    output_by_name = {p.name: p for p in sorted(final_results.glob("*.png"))}
    fallback_outputs = sorted(final_results.glob("*.png"))
    for i, selected_frame in enumerate(selected):
        out = output_by_name.get(selected_frame.name)
        if out is None and not needs_composite and i < len(fallback_outputs):
            out = fallback_outputs[i]
        if out is None or not out.is_file():
            if needs_composite:
                unmatched += 1
            continue
        if needs_composite:
            result = _compose_face_restore_frame(
                original_path=selected_frame,
                restored_path=out,
                output_path=output_dir / selected_frame.name,
                options=options,
            )
            if result.resized:
                resized += 1
            if result.empty_mask:
                empty_mask += 1
                continue
            if result.wrote:
                restored += 1
        else:
            _atomic_replace_from_file(out, output_dir / selected_frame.name)
            restored += 1
    missing = len(selected) - restored
    log(f"       {tag} face_restore restored {restored}/{len(selected)} frames")
    if unmatched:
        log(f"       {tag} [WARN] face_restore naturalize unmatched frames={unmatched}")
    if resized:
        log(f"       {tag} [WARN] face_restore naturalize resized frames={resized}")
    if empty_mask:
        log(f"       {tag} [WARN] face_restore naturalize empty mask frames={empty_mask}")
    if missing:
        log(f"       {tag} [WARN] face_restore pass-through frames={missing}")
    return len(input_frames)


DEFAULT_PNG_COMPRESSION_RATIO = 0.80


def estimate_compression_ratio(stats) -> float:
    """analyze 結果からモデルごとの圧縮率係数を推定する。

    histogram_flat_ratio が高い -> アニメ・スライド寄り -> 圧縮率高い -> 係数小さい。
    安全側に倒すため、明らかな平坦素材以外は default (0.80) を返す。
    cli / runtime_estimator から参照される公開 API (旧名 `_estimate_compression_ratio`)。
    """
    flat = float(getattr(stats, "histogram_flat_ratio", 0.0))
    if flat > 0.40:
        return 0.65
    return DEFAULT_PNG_COMPRESSION_RATIO


# 後方互換 alias (旧 private 名)。新規コードは estimate_compression_ratio を使うこと。
_estimate_compression_ratio = estimate_compression_ratio


def estimate_peak_temp_bytes(
    width: int,
    height: int,
    fps: Fraction,
    chunk_seconds: int,
    *,
    intermediate_width: int | None = None,
    effective_fps: Fraction | None = None,
    concurrent_dirs: int = 2,
    bytes_per_pixel: int = 3,
    png_compression_ratio: float = DEFAULT_PNG_COMPRESSION_RATIO,
    worker_count: int = 1,
) -> int:
    """Rough peak temp 見積もり。

    `intermediate_width` を渡すと upscale 中間幅で再計算 (例: 1080p × scale=2 = 2160p)。
    `effective_fps` を渡すと補間後 fps を反映 (例: 30 -> 60)。
    `png_compression_ratio` で PNG 圧縮率を加味する (default = 0.80 安全側)。
        - 1.0 を明示すれば旧シグネチャ相当の戻り値 (legacy 完全一致)
        - シグネチャ自体は後方互換 (位置引数 4 つの呼び出しが TypeError にならない)
    `worker_count` で並列実行時の中間ピーク倍率を反映 (PR3 のチャンク別 tmp 用)。
    """
    if width <= 0:
        return 0
    w = intermediate_width if intermediate_width is not None else width
    h_ratio = height / width
    h = int(w * h_ratio)
    f = effective_fps if effective_fps is not None else fps
    frames_per_chunk = float(f) * float(chunk_seconds)
    raw = w * h * bytes_per_pixel * frames_per_chunk * concurrent_dirs
    return int(raw * float(png_compression_ratio) * max(1, int(worker_count)))


def _run_ffmpeg_png_stage(
    argv: list[str],
    *,
    runner: ProcessRunner,
    stage: str,
    progress_label: str,
    chunk_idx: int,
    total_chunks: int,
    progress: Callable[[ProgressEvent], None] | None,
    cancel_event: threading.Event | None,
    io_idle_timeout_sec: int,
    out_dir: Path | None = None,
    empty_error: str | None = None,
    require_count: bool = True,
) -> int | None:
    """ffmpeg PNG 出力ステージの共通処理 (PR-R3a)。

    `_augment_ffmpeg_with_progress -> runner.run_with_progress(frame emitter) -> _count_png`
    の同型パターン (extract / pre_filter / resolution_clamp / debug post_filter) を集約する。
    `argv` は raw な ffmpeg argv を渡す (progress 付与は本ヘルパー内で行う)。

    `out_dir` を渡すと出力 PNG 枚数を検証し、0 枚なら
    `PipelineError(stage, empty_error, chunk_idx)` を送出して枚数を返す。
    `out_dir=None` の場合は枚数検証を行わず None を返す (pre_filter のように
    枚数検証を呼び出し側で skip パスと共有するケース用)。
    """
    argv = _augment_ffmpeg_with_progress(argv)
    # V2-6: 中間 PNG (extract / pre_filter / resolution_clamp / debug post_filter) は
    # zlib 圧縮レベルを 1 に下げて書込 CPU を削減する。PNG は compression_level に依らず
    # ロスレスのため画素データは不変 (ファイルサイズだけ増える)。realesrgan / rife の ncnn
    # 出力は本ヘルパーを通らないため影響しない。出力パス (argv[-1]) の直前に出力オプション
    # として挿入する (_augment は出力を末尾に保つ)。
    if argv and str(argv[-1]).lower().endswith(".png") and "-compression_level" not in argv:
        argv = argv[:-1] + ["-compression_level", "1", argv[-1]]
    runner.run_with_progress(
        argv,
        idle_timeout=io_idle_timeout_sec,
        cancel_event=cancel_event,
        on_progress=_make_frame_emitter(progress, progress_label, chunk_idx, total_chunks),
        stage=stage,
        chunk_idx=chunk_idx,
    )
    if out_dir is None:
        return None
    # V2-7: 後段が枚数 (magnitude) を使う場合だけ全件カウントする。extract のように 0 枚
    # 判定しか使わない箇所は require_count=False で存在判定に切り替え、巨大な frames
    # ディレクトリの全走査 (os.listdir + Path 生成) を避ける。
    if not require_count:
        if not _has_png(out_dir):
            raise PipelineError(stage, empty_error or "no frames produced", chunk_idx=chunk_idx)
        return None
    count = _count_png(out_dir)
    if count == 0:
        raise PipelineError(stage, empty_error or "no frames produced", chunk_idx=chunk_idx)
    return count


def _run_gpu_stage(
    argv: list[str],
    *,
    runner: ProcessRunner,
    stage: str,
    chunk_idx: int,
    cancel_event: threading.Event | None,
    io_idle_timeout_sec: int,
    gpu_gate: threading.Semaphore | None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log: Callable[[str], None] | None = None,
    output_log_interval_sec: float = 5.0,
    heartbeat_interval_sec: float = 15.0,
) -> None:
    """GPU (ncnn-vulkan) ステージの薄いラッパ (PR-R3b)。

    `on_progress=None` は GUI 向け frame progress へ解析しないという意味で固定する。
    Real-ESRGAN のタイル `%` と RIFE の verbose 保存完了行は、いずれも runner 内で
    idle watchdog の生存信号として扱われる。
    `gpu_gate` (並列実行時の GPU 同時実行数セマフォ) が渡されたら取得してから実行する。
    argv 構築は呼び出し側に残す (差分最小化)。
    """
    run_kwargs = {
        "idle_timeout": io_idle_timeout_sec,
        "cancel_event": cancel_event,
        "on_progress": None,
        "stage": stage,
        "chunk_idx": chunk_idx,
    }
    if stage == "face_restore" and log is not None:
        last_output_log = {"t": 0.0}

        def _on_output_event(stream_tag: str, text: str) -> None:
            stripped = text.strip()
            if not stripped:
                return
            now = time.monotonic()
            if now - last_output_log["t"] < output_log_interval_sec:
                return
            last_output_log["t"] = now
            stream_name = "stderr" if stream_tag == _STREAM_STDERR else "stdout"
            if len(stripped) > 240:
                stripped = stripped[:237] + "..."
            log(f"       face_restore output[{stream_name}]: {stripped}")

        def _on_heartbeat(elapsed: float, idle: float) -> None:
            log(f"       face_restore running... elapsed={elapsed:.0f}s idle={idle:.0f}s")

        run_kwargs["on_output_event"] = _on_output_event
        run_kwargs["on_heartbeat"] = _on_heartbeat
        run_kwargs["heartbeat_interval_sec"] = heartbeat_interval_sec
    elif stage == "interpolate" and log is not None:

        def _on_interpolate_heartbeat(elapsed: float, idle: float) -> None:
            log(f"       interpolate running... elapsed={elapsed:.0f}s idle={idle:.0f}s")

        # UI 可視化専用。runner の last_event は更新しないため、真の無出力ハングを延命しない。
        run_kwargs["on_heartbeat"] = _on_interpolate_heartbeat
        run_kwargs["heartbeat_interval_sec"] = heartbeat_interval_sec
    if cwd is not None:
        run_kwargs["cwd"] = cwd
    if env is not None:
        run_kwargs["env"] = env
    if gpu_gate is not None:
        with gpu_gate:
            runner.run_with_progress(argv, **run_kwargs)
    else:
        runner.run_with_progress(argv, **run_kwargs)


def _process_chunk(
    *,
    runner: ProcessRunner,
    ffmpeg: Path,
    realesrgan: Path | None,
    rife: Path | None,
    codeformer: deps.ResolvedCodeFormerLocal | None,
    input_path: Path,
    fps: Fraction,
    normalization_fps: Fraction | None,
    expected_normalized_frames: int | None,
    chunk_idx: int,
    start_sec: float,
    dur_sec: float,
    total_chunks: int,
    tmp_dirs: dict[str, Path | int],
    decisions: list[StageDecision],
    encode_args: list[str],
    log: Callable[[str], None],
    cancel_event: threading.Event | None,
    progress: Callable[[ProgressEvent], None] | None,
    io_idle_timeout_sec: int = DEFAULT_IO_IDLE_TIMEOUT_SEC,
    face_restore_options: FaceRestoreOptions = FaceRestoreOptions(),
    stabilize_backend: str = config.DEFAULT_STABILIZE_BACKEND,
    gpu_gate: threading.Semaphore | None = None,
    clamp_flags: str = config.DEFAULT_CLAMP_FLAGS,
    gpu_opts: GpuRunOptions = GpuRunOptions(),
    debug_split_postfilter: bool = False,
    fuse_extract_prefilter: bool = False,
    keep_temp: bool = False,
) -> Path:
    tag = f"[chunk {chunk_idx + 1}/{total_chunks}]"
    by = decisions_by_key(decisions)

    frames_in = tmp_dirs["frames_in"]
    frames_pre = tmp_dirs["frames_pre"]
    frames_up = tmp_dirs["frames_up"]
    frames_clamp = tmp_dirs["frames_clamp"]
    frames_face = tmp_dirs["frames_face"]
    frames_face_input = tmp_dirs["frames_face_input"]
    frames_face_raw = tmp_dirs["frames_face_raw"]
    frames_rife = tmp_dirs["frames_rife"]
    frames_out = tmp_dirs["frames_out"]
    chunks_dir = tmp_dirs["chunks"]
    transforms_path = tmp_dirs["transforms_root"] / f"chunk_{chunk_idx:03d}.trf"

    _reset_dirs(
        frames_in,
        frames_pre,
        frames_up,
        frames_clamp,
        frames_face,
        frames_face_input,
        frames_face_raw,
        frames_rife,
        frames_out,
    )

    # ---- resolve extract/pre-filter layout ----
    stabilize_dec = by.get("stabilize")
    stabilize_enabled = stabilize_dec is not None and stabilize_dec.enabled
    use_vidstab = stabilize_enabled and stabilize_backend == "vidstab"
    use_opencv_stabilize = stabilize_enabled and stabilize_backend == "opencv"
    preliminary_pre_chain = stages.build_pre_filter_chain(
        decisions,
        transforms_path=None,
        exclude_stabilize=use_opencv_stabilize,
    )
    fused_extract = bool(fuse_extract_prefilter and not use_vidstab and preliminary_pre_chain)

    _check_cancel(cancel_event, "process_chunks", chunk_idx=chunk_idx)
    if fused_extract:
        extract_target = frames_pre
        fused_chain = f"format=rgb24,{preliminary_pre_chain}"
        log(f"       {tag} extract+pre_filter start={start_sec:.2f}s dur={dur_sec:.2f}s")
        progress_stage = "chunk_extract_pre_filter"
        failure_stage = "extract_pre_filter"
    else:
        extract_target = frames_in
        fused_chain = ""
        log(f"       {tag} extract  start={start_sec:.2f}s dur={dur_sec:.2f}s")
        progress_stage = "chunk_extract"
        failure_stage = "extract_frames"
    _emit_progress(
        progress,
        progress_stage,
        chunk_index=chunk_idx,
        total_chunks=total_chunks,
        label=f"chunk {chunk_idx + 1}/{total_chunks} {progress_stage.removeprefix('chunk_')}",
    )
    extract_argv = stages.build_extract_args(
        ffmpeg,
        input_path,
        start_sec,
        dur_sec,
        extract_target,
        vf_chain=fused_chain,
        normalization_fps=normalization_fps,
        expected_frame_count=expected_normalized_frames,
    )
    _run_ffmpeg_png_stage(
        extract_argv,
        runner=runner,
        stage=failure_stage,
        progress_label=progress_stage,
        chunk_idx=chunk_idx,
        total_chunks=total_chunks,
        progress=progress,
        cancel_event=cancel_event,
        io_idle_timeout_sec=io_idle_timeout_sec,
        out_dir=extract_target,
        empty_error="no frames extracted",
        require_count=False,
    )

    # ---- stabilize detect (only if stabilize ON, vidstab backend) ----
    # P2-3: フレーム 2 枚未満の端数チャンク (duration の余りが 1 フレーム分未満の
    # 最終チャンク等) では軌跡推定が成立しない (stabilize_cv は ValueError で全体失敗、
    # vidstab も平滑化が無意味)。backend を問わずこのチャンクのみスキップする。
    extracted_frame_dir = frames_pre if fused_extract else frames_in
    if stabilize_enabled and _count_png(extracted_frame_dir) < 2:
        log(f"       {tag} stabilize skipped (frames < 2)")
        stabilize_enabled = False
    use_vidstab = stabilize_enabled and stabilize_backend == "vidstab"
    use_opencv_stabilize = stabilize_enabled and stabilize_backend == "opencv"
    if use_vidstab:
        _check_cancel(cancel_event, "process_chunks", chunk_idx=chunk_idx)
        log(f"       {tag} stabilize-detect -> {transforms_path.name}")
        _emit_progress(
            progress,
            "chunk_stabilize_detect",
            chunk_index=chunk_idx,
            total_chunks=total_chunks,
            label=f"chunk {chunk_idx + 1}/{total_chunks} stabilize-detect",
        )
        det_argv = stages.build_stabilize_detect_args(
            ffmpeg=ffmpeg,
            input_pattern=frames_in / "%08d.png",
            transforms_path=transforms_path,
            fps=fps,
        )
        det_argv = _augment_ffmpeg_with_progress(det_argv)
        runner.run_with_progress(
            det_argv,
            idle_timeout=io_idle_timeout_sec,
            cancel_event=cancel_event,
            on_progress=_make_frame_emitter(
                progress, "chunk_stabilize_detect", chunk_idx, total_chunks
            ),
            stage="stabilize_detect",
            chunk_idx=chunk_idx,
        )
        transforms_for_filter: Path | None = transforms_path
    else:
        transforms_for_filter = None

    # ---- pre filter ----
    _check_cancel(cancel_event, "process_chunks", chunk_idx=chunk_idx)
    # opencv backend では stabilize スニペットを pre_filter チェーンから外す
    # (後段の独立な stabilize_cv.stabilize_chunk で処理する)
    pre_chain = (
        preliminary_pre_chain
        if fused_extract
        else stages.build_pre_filter_chain(
            decisions,
            transforms_path=transforms_for_filter,
            exclude_stabilize=use_opencv_stabilize,
        )
    )
    if fused_extract:
        pass
    elif pre_chain:
        log(f"       {tag} pre_filter  vf={pre_chain}")
        _emit_progress(
            progress,
            "chunk_pre_filter",
            chunk_index=chunk_idx,
            total_chunks=total_chunks,
            label=f"chunk {chunk_idx + 1}/{total_chunks} pre_filter",
        )
        pre_argv = [
            str(ffmpeg),
            "-y",
            "-framerate",
            f"{fps.numerator}/{fps.denominator}",
            "-start_number",
            "0",
            "-i",
            str(frames_in / "%08d.png"),
            "-vf",
            pre_chain,
            "-start_number",
            "0",
            str(frames_pre / "%08d.png"),
        ]
        # out_dir=None: 枚数検証は skip パス (move) と共有するため呼び出し側で行う。
        _run_ffmpeg_png_stage(
            pre_argv,
            runner=runner,
            stage="pre_filter",
            progress_label="chunk_pre_filter",
            chunk_idx=chunk_idx,
            total_chunks=total_chunks,
            progress=progress,
            cancel_event=cancel_event,
            io_idle_timeout_sec=io_idle_timeout_sec,
        )
    else:
        # Skip: move frames_in -> frames_pre
        _rename_dir_contents(frames_in, frames_pre)

    pre_count = _count_png(frames_pre)
    if pre_count == 0:
        raise PipelineError("pre_filter", "no frames after pre_filter", chunk_idx=chunk_idx)

    # ---- stabilize (opencv backend, pre_filter 後に挟む) ----
    if use_opencv_stabilize:
        if not _STABILIZE_CV_AVAILABLE:
            raise PipelineError(
                "stabilize_cv",
                "stabilize_cv (cv2) is not available; install opencv-python-headless",
                chunk_idx=chunk_idx,
            )
        _check_cancel(cancel_event, "process_chunks", chunk_idx=chunk_idx)
        # チャンク別の中間出力 (frames_pre_stab) を使い、frames_pre と入れ替える
        frames_pre_stab = tmp_dirs["frames_pre"].parent / f"frames_pre_stab_{chunk_idx}"
        _reset_dirs(frames_pre_stab)
        log(f"       {tag} stabilize (opencv) {pre_count} frames")
        _emit_progress(
            progress,
            "chunk_stabilize_cv",
            chunk_index=chunk_idx,
            total_chunks=total_chunks,
            label=f"chunk {chunk_idx + 1}/{total_chunks} stabilize-cv",
        )

        def _stab_progress(n: int, total: int) -> None:
            _emit_progress(
                progress,
                "chunk_stabilize_cv",
                chunk_index=chunk_idx,
                total_chunks=total_chunks,
                label=f"chunk {chunk_idx + 1}/{total_chunks} stabilize-cv frame={n}",
                frame=n,
                total_frames=total,
            )

        try:
            stabilize_cv.stabilize_chunk(  # type: ignore[union-attr]
                frames_pre,
                frames_pre_stab,
                cancel_event=cancel_event,
                on_progress=_stab_progress if progress is not None else None,
            )
        except InterruptedError as e:
            raise PipelineCancelled("stabilize_cv", chunk_idx=chunk_idx) from e
        except (ValueError, RuntimeError) as e:
            raise PipelineError("stabilize_cv", str(e), chunk_idx=chunk_idx) from e
        # frames_pre を frames_pre_stab で置き換える: 後段がそのまま frames_pre を入力するように
        _reset_dirs(frames_pre)
        _rename_dir_contents(frames_pre_stab, frames_pre)
        try:
            frames_pre_stab.rmdir()
        except OSError:
            pass

    # ---- upscale ----
    _check_cancel(cancel_event, "process_chunks", chunk_idx=chunk_idx)
    upscale_dec = by.get("upscale")
    if upscale_dec is not None and upscale_dec.enabled:
        if realesrgan is None:
            raise PipelineError(
                "upscale_frames", "realesrgan binary not available", chunk_idx=chunk_idx
            )
        scale = int(upscale_dec.params.get("scale", config.DEFAULT_REALESRGAN_SCALE))
        model = str(upscale_dec.params.get("model", config.DEFAULT_REALESRGAN_MODEL))
        log(f"       {tag} upscale  scale={scale} model={model}")
        _emit_progress(
            progress,
            "chunk_upscale",
            chunk_index=chunk_idx,
            total_chunks=total_chunks,
            label=f"chunk {chunk_idx + 1}/{total_chunks} upscale",
        )
        up_argv = [
            str(realesrgan),
            "-i",
            str(frames_pre),
            "-o",
            str(frames_up),
            "-s",
            str(scale),
            "-n",
            model,
            "-f",
            "png",
        ]
        # PR-G3: Real-ESRGAN tuningフラグ (すべて未指定なら従来と完全に同一 argv)。
        #   -t (tilesize) は realesrgan 専用 (RIFE 非対応)。-j / -g は両者に渡す。
        #   -g は共通 gpu_id (>=0) のみ。realesrgan は -1 非対応のため rife_gpu_id は渡さない。
        if gpu_opts.tilesize is not None:
            up_argv += ["-t", str(gpu_opts.tilesize)]
        if gpu_opts.threads is not None:
            up_argv += ["-j", str(gpu_opts.threads)]
        if gpu_opts.gpu_id is not None:
            up_argv += ["-g", str(gpu_opts.gpu_id)]
        # Real-ESRGAN のタイル `%` は idle 生存信号として使うが、GUI frame進捗には解析しない。
        _run_gpu_stage(
            up_argv,
            runner=runner,
            stage="upscale_frames",
            chunk_idx=chunk_idx,
            cancel_event=cancel_event,
            io_idle_timeout_sec=io_idle_timeout_sec,
            gpu_gate=gpu_gate,
        )
    else:
        _rename_dir_contents(frames_pre, frames_up)

    up_count = _count_png(frames_up)
    if up_count == 0:
        raise PipelineError("upscale_frames", "no frames after upscale", chunk_idx=chunk_idx)

    # ---- resolution_clamp (upscale 中間幅 != target_width のとき独立 ffmpeg 起動) ----
    # `meta.width` (= src_width) は呼び出し側 (run_pipeline) しか知らないため、tmp_dirs に
    # "_src_width" として埋めて渡してもらう。`upscale_dec.params` に保存された
    # `scale` と `target_width` から intermediate_width = src_width * scale を算出する。
    _check_cancel(cancel_event, "process_chunks", chunk_idx=chunk_idx)
    rife_input_dir = frames_up
    clamp_input_fps = fps  # extract 直後の素 fps (補間はまだ走っていない)
    upscale_dec_local = by.get("upscale")
    clamp_filter = ""
    if upscale_dec_local is not None and upscale_dec_local.enabled:
        target_w = int(upscale_dec_local.params.get("target_width", 0))
        scale_done = int(upscale_dec_local.params.get("scale", config.DEFAULT_REALESRGAN_SCALE))
        src_w = int(tmp_dirs.get("_src_width", 0))
        if target_w > 0 and scale_done > 0 and src_w > 0:
            intermediate_w = src_w * scale_done
            clamp_filter = stages.build_resolution_clamp_filter(
                target_w, intermediate_w, flags=clamp_flags
            )
            if clamp_filter and intermediate_w < target_w:
                log(
                    f"       {tag} [WARN] resolution_clamp 上方 lanczos "
                    f"({intermediate_w}px -> {target_w}px); 品質低下の可能性"
                )

    if clamp_filter:
        log(f"       {tag} resolution_clamp vf={clamp_filter}")
        _emit_progress(
            progress,
            "chunk_resolution_clamp",
            chunk_index=chunk_idx,
            total_chunks=total_chunks,
            label=f"chunk {chunk_idx + 1}/{total_chunks} resolution_clamp",
        )
        clamp_argv = [
            str(ffmpeg),
            "-y",
            "-framerate",
            f"{clamp_input_fps.numerator}/{clamp_input_fps.denominator}",
            "-start_number",
            "0",
            "-i",
            str(frames_up / "%08d.png"),
            "-vf",
            clamp_filter,
            "-start_number",
            "0",
            str(frames_clamp / "%08d.png"),
        ]
        clamp_count = _run_ffmpeg_png_stage(
            clamp_argv,
            runner=runner,
            stage="resolution_clamp",
            progress_label="chunk_resolution_clamp",
            chunk_idx=chunk_idx,
            total_chunks=total_chunks,
            progress=progress,
            cancel_event=cancel_event,
            io_idle_timeout_sec=io_idle_timeout_sec,
            out_dir=frames_clamp,
            empty_error="no frames after resolution_clamp",
        )
        rife_input_dir = frames_clamp
        rife_input_count = clamp_count
    else:
        rife_input_count = up_count

    # ---- face_restore (CodeFormer local, resolution_clamp 後 / interpolate 前) ----
    face_dec = by.get("face_restore")
    face_enabled = face_dec is not None and face_dec.enabled
    if face_enabled:
        if codeformer is None:
            raise PipelineError(
                "face_restore", "CodeFormer local backend not available", chunk_idx=chunk_idx
            )
        _check_cancel(cancel_event, "process_chunks", chunk_idx=chunk_idx)
        rife_input_count = _run_face_restore_stage(
            runner=runner,
            codeformer=codeformer,
            input_dir=rife_input_dir,
            output_dir=frames_face,
            selected_input_dir=frames_face_input,
            raw_output_dir=frames_face_raw,
            fps=fps,
            chunk_idx=chunk_idx,
            start_sec=start_sec,
            total_chunks=total_chunks,
            progress=progress,
            cancel_event=cancel_event,
            gpu_gate=gpu_gate,
            log=log,
            tag=tag,
            options=face_restore_options,
        )
        rife_input_dir = frames_face

    # ---- interpolate (RIFE) ----
    _check_cancel(cancel_event, "process_chunks", chunk_idx=chunk_idx)
    interp_dec = by.get("interpolate")
    interp_enabled = interp_dec is not None and interp_dec.enabled
    # P2-3: フレーム 2 枚未満の端数チャンクでは補間ペアが成立しない (rife-ncnn-vulkan
    # は連続 2 フレーム間の中間フレームを生成する)。stabilize と同様にこのチャンク
    # のみスキップして pass-through する (発生は端数 0.0x 秒の最終チャンクのみ)。
    if interp_enabled and rife_input_count < 2:
        log(f"       {tag} interpolate skipped (frames < 2)")
        interp_enabled = False
    if interp_enabled:
        if rife is None:
            raise PipelineError("interpolate", "rife binary not available", chunk_idx=chunk_idx)
        tgt_num = int(interp_dec.params.get("target_fps_num", fps.numerator * 2))
        tgt_den = int(interp_dec.params.get("target_fps_den", fps.denominator))
        target_fps = Fraction(tgt_num, tgt_den)
        log(f"       {tag} interpolate  {float(fps):.3f} -> {float(target_fps):.3f} fps")
        _emit_progress(
            progress,
            "chunk_interpolate",
            chunk_index=chunk_idx,
            total_chunks=total_chunks,
            label=f"chunk {chunk_idx + 1}/{total_chunks} interpolate",
        )
        # PR-G3: RIFE の -g は rife_gpu_id を優先 (-1=cpu 可)、無ければ共通 gpu_id (>=0)。
        #   -t は RIFE 非対応のため渡さない。-j は両者共通。
        rife_resolved_gpu = (
            gpu_opts.rife_gpu_id if gpu_opts.rife_gpu_id is not None else gpu_opts.gpu_id
        )
        rife_argv = stages.build_rife_args(
            rife_bin=rife,
            in_dir=rife_input_dir,
            out_dir=frames_rife,
            source_fps=fps,
            target_fps=target_fps,
            input_frame_count=rife_input_count,
            model=interp_dec.params.get("model"),
            threads=gpu_opts.threads,
            gpu_id=rife_resolved_gpu,
        )
        # RIFE verbose は idle 生存信号として使い、GUIには低頻度 heartbeat だけを流す。
        _run_gpu_stage(
            rife_argv,
            runner=runner,
            stage="interpolate",
            chunk_idx=chunk_idx,
            cancel_event=cancel_event,
            io_idle_timeout_sec=io_idle_timeout_sec,
            gpu_gate=gpu_gate,
            log=log,
        )
        mux_fps = target_fps
    else:
        _rename_dir_contents(rife_input_dir, frames_rife)
        mux_fps = fps

    # V2-7: rife 出力枚数 (magnitude) は後段で使わないため存在判定で十分。
    if not _has_png(frames_rife):
        raise PipelineError("interpolate", "no frames after interpolate", chunk_idx=chunk_idx)

    # ---- post filter + mux (PR6: 融合) ----
    _check_cancel(cancel_event, "process_chunks", chunk_idx=chunk_idx)
    post_chain = stages.build_post_filter_chain(decisions)
    chunk_path = chunks_dir / f"chunk_{chunk_idx:03d}.mp4"

    if debug_split_postfilter and post_chain:
        # 旧 2 段パス (debug 用): post_filter -> frames_out -> mux
        log(f"       {tag} post_filter vf={post_chain} (debug split path)")
        _emit_progress(
            progress,
            "chunk_post_filter",
            chunk_index=chunk_idx,
            total_chunks=total_chunks,
            label=f"chunk {chunk_idx + 1}/{total_chunks} post_filter",
        )
        post_argv = [
            str(ffmpeg),
            "-y",
            "-framerate",
            f"{mux_fps.numerator}/{mux_fps.denominator}",
            "-start_number",
            "0",
            "-i",
            str(frames_rife / "%08d.png"),
            "-vf",
            post_chain,
            "-start_number",
            "0",
            str(frames_out / "%08d.png"),
        ]
        _run_ffmpeg_png_stage(
            post_argv,
            runner=runner,
            stage="post_filter",
            progress_label="chunk_post_filter",
            chunk_idx=chunk_idx,
            total_chunks=total_chunks,
            progress=progress,
            cancel_event=cancel_event,
            io_idle_timeout_sec=io_idle_timeout_sec,
            out_dir=frames_out,
            empty_error="no frames after post_filter",
        )
        log(f"       {tag} mux      -> {chunk_path.name}")
        _emit_progress(
            progress,
            "chunk_mux",
            chunk_index=chunk_idx,
            total_chunks=total_chunks,
            label=f"chunk {chunk_idx + 1}/{total_chunks} mux",
        )
        mux_argv = stages.build_mux_args(
            ffmpeg=ffmpeg,
            frames_dir=frames_out,
            fps=mux_fps,
            chunk_path=chunk_path,
            encode_args=encode_args,
        )
        mux_argv = _augment_ffmpeg_with_progress(mux_argv)
        runner.run_with_progress(
            mux_argv,
            idle_timeout=io_idle_timeout_sec,
            cancel_event=cancel_event,
            on_progress=_make_frame_emitter(progress, "chunk_mux", chunk_idx, total_chunks),
            stage="reassemble",
            chunk_idx=chunk_idx,
        )
    else:
        # PR6 融合パス: post_filter があれば mux に -vf として渡す。中間 frames_out を生成しない。
        # post_filter なしでもこのパスを通る (1 ffmpeg コマンドで完結)。
        log(f"       {tag} mux+post vf={post_chain or '(none)'} -> {chunk_path.name}")
        _emit_progress(
            progress,
            "chunk_mux",
            chunk_index=chunk_idx,
            total_chunks=total_chunks,
            label=f"chunk {chunk_idx + 1}/{total_chunks} mux",
        )
        mux_argv = stages.build_mux_args(
            ffmpeg=ffmpeg,
            frames_dir=frames_rife,
            fps=mux_fps,
            chunk_path=chunk_path,
            encode_args=encode_args,
            vf_chain=post_chain,
        )
        mux_argv = _augment_ffmpeg_with_progress(mux_argv)
        runner.run_with_progress(
            mux_argv,
            idle_timeout=io_idle_timeout_sec,
            cancel_event=cancel_event,
            on_progress=_make_frame_emitter(progress, "chunk_mux", chunk_idx, total_chunks),
            stage="reassemble",
            chunk_idx=chunk_idx,
        )

    # ---- cleanup chunk dirs ----
    # `keep_temp=True` のときは中間 PNG を残す (PR6/PR3 で `--keep-temp` を
    # デバッグ目的に使うため)。PR3 のチャンク別 tmp ディレクトリと組み合わせると、
    # `tmp_root/chunk_<idx>/frames_rife/` がチャンク数分残る。
    if not keep_temp:
        log(f"       {tag} cleanup")
        _reset_dirs(
            frames_in,
            frames_pre,
            frames_up,
            frames_clamp,
            frames_face,
            frames_face_input,
            frames_face_raw,
            frames_rife,
            frames_out,
        )
        if transforms_path.exists():
            try:
                transforms_path.unlink()
            except OSError:
                pass
    else:
        log(f"       {tag} cleanup skipped (--keep-temp)")

    return chunk_path


def _concat_chunks(
    *,
    runner: ProcessRunner,
    ffmpeg: Path,
    chunks_dir: Path,
    chunk_paths: list[Path],
    input_path: Path,
    stream_plan: media_streams.StreamPreservationPlan,
    duration_sec: float,
    output_path: Path,
    log: Callable[[str], None],
    cancel_event: threading.Event | None = None,
    io_idle_timeout_sec: int = DEFAULT_IO_IDLE_TIMEOUT_SEC,
) -> None:
    list_path = chunks_dir / "concat_list.txt"
    lines = [f"file '{p.name}'\n" for p in chunk_paths]
    list_path.write_text("".join(lines), encoding="utf-8")

    log(f"[concat] ({len(chunk_paths)} chunks) -> {output_path}")
    argv: list[str] = [
        str(ffmpeg),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path.name,
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
    ]
    if stream_plan.inventory_complete:
        for stream in stream_plan.audio_streams:
            argv += ["-map", f"1:{stream.index}"]
        for stream in stream_plan.subtitle_streams:
            argv += ["-map", f"1:{stream.index}"]
    elif stream_plan.has_audio:
        # Older direct callers may provide only has_audio. Production metadata always
        # uses the complete inventory and therefore never takes this compatibility path.
        argv += ["-map", "1:a:0?"]
    argv += [
        "-map_metadata",
        "1",
        "-map_metadata:s:v:0",
        "1:s:v:0",
        "-c:v",
        "copy",
    ]
    if stream_plan.has_audio:
        argv += ["-c:a", "copy"]
    if stream_plan.has_subtitle:
        argv += ["-c:s", "copy"]
    if stream_plan.inventory_complete:
        for output_index, stream in enumerate(stream_plan.audio_streams):
            disposition = "+".join(stream.disposition) if stream.disposition else "0"
            argv += [
                f"-map_metadata:s:a:{output_index}",
                f"1:s:{stream.index}",
                f"-disposition:a:{output_index}",
                disposition,
            ]
        for output_index, stream in enumerate(stream_plan.subtitle_streams):
            disposition = "+".join(stream.disposition) if stream.disposition else "0"
            argv += [
                f"-map_metadata:s:s:{output_index}",
                f"1:s:{stream.index}",
                f"-disposition:s:{output_index}",
                disposition,
            ]
    argv += ["-t", f"{duration_sec:.6f}"]
    argv += [str(output_path)]
    argv = _augment_ffmpeg_with_progress(argv)
    runner.run_with_progress(
        argv,
        cwd=chunks_dir,
        idle_timeout=io_idle_timeout_sec,
        cancel_event=cancel_event,
        on_progress=None,
        stage="concat",
    )


def run_pipeline(
    *,
    runner: ProcessRunner | None = None,
    input_path: Path,
    output_path: Path,
    decisions: list[StageDecision],
    meta,  # analyzer.VideoMetadata
    chunk_seconds: int = config.DEFAULT_CHUNK_SECONDS,
    keep_temp: bool = False,
    keep_temp_on_failure: bool = False,
    log: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    progress: Callable[[ProgressEvent], None] | None = None,
    ffmpeg: Path | None = None,
    ffprobe: Path | None = None,
    realesrgan: Path | None = None,
    rife: Path | None = None,
    codeformer: deps.ResolvedCodeFormerLocal | None = None,
    io_idle_timeout_sec: int = DEFAULT_IO_IDLE_TIMEOUT_SEC,
    face_restore_idle_timeout_sec: int = config.DEFAULT_FACE_RESTORE_IDLE_TIMEOUT_SEC,
    face_restore_start_sec: float | None = None,
    face_restore_end_sec: float | None = None,
    face_restore_fidelity: float = config.DEFAULT_FACE_RESTORE_FIDELITY,
    face_restore_detection_model: str = config.DEFAULT_FACE_RESTORE_DETECTION_MODEL,
    face_restore_options: FaceRestoreOptions | None = None,
    stabilize_backend: str = config.DEFAULT_STABILIZE_BACKEND,
    worker_count: int = 1,
    gpu_concurrency: int = 1,
    clamp_flags: str = config.DEFAULT_CLAMP_FLAGS,
    gpu_tilesize: int | None = None,
    gpu_threads: str | None = None,
    gpu_id: int | None = None,
    rife_gpu_id: int | None = None,
    debug_split_postfilter: bool = False,
    debug_split_extract_prefilter: bool = False,
    experimental_fuse_extract_prefilter: bool = False,
    hw_encode: str = "none",
    temp_run: temp_workspace.TempRunContext | None = None,
    estimated_peak_temp_bytes: int | None = None,
) -> None:
    """Run the hybrid 4-pass enhance pipeline.

    `decisions` は事前に確定済み (analyze + user confirm 後)。
    `meta` は probe 結果。fps/duration の再取得を避けるため呼び出し側が渡す。

    Defense-in-depth: `output_path` を再解決する。CLI/GUI 側でも検証しているが、
    本関数を直接呼ぶ他フロントエンド・テストでも解決不能パスを早期に弾く。
    """
    log = log or print
    # DI seam (V2-8): 既定は本番実装 RealProcessRunner。`is None` で明示判定する
    # (`runner or ...` の truthy 判定だと将来 `__bool__` を持つ runner を渡したとき
    # 意図せず差し替わるため避ける)。
    if runner is None:
        runner = RealProcessRunner()
    # P2-2: cancel_event が無くても内部で必ず Event を持つ。これにより並列実行時の
    # 「1 チャンク失敗 -> 残チャンクへの cancel 伝播」が呼び出し側の配線に依存しなくなる
    # (旧実装は cancel_event=None だと set() されず、失敗後も全チャンクが完走していた)。
    if cancel_event is None:
        cancel_event = threading.Event()
    config.validate_chunk_seconds(chunk_seconds)
    # Resolve once before passing the path to ffmpeg.
    output_path = config.validate_output_path(Path(output_path))

    stream_plan = resolve_stream_preservation(meta, log=log)
    framerate_plan = resolve_framerate_normalization(meta, log=log)

    by = decisions_by_key(decisions)
    encode_dec = by.get("encode")
    if encode_dec is None:
        raise ValueError("decisions must include 'encode' stage")
    encode_args = stages.build_encode_args(encode_dec, hw_backend=hw_encode)
    face_dec = by.get("face_restore")
    if face_restore_options is None:
        legacy_face_restore_options = FaceRestoreOptions(
            start_sec=face_restore_start_sec,
            end_sec=face_restore_end_sec,
            fidelity=face_restore_fidelity,
            idle_timeout_sec=face_restore_idle_timeout_sec,
            detection_model=face_restore_detection_model,
        ).validate()
        resolved_face_restore_options = (
            FaceRestoreOptions.from_params(face_dec.params, fallback=legacy_face_restore_options)
            if face_dec is not None
            else legacy_face_restore_options
        )
    else:
        resolved_face_restore_options = face_restore_options.validate()

    # stabilize_backend を解決 (auto -> opencv/vidstab を ffmpeg 機能で決定)
    stabilize_dec = by.get("stabilize")
    stabilize_on = stabilize_dec is not None and stabilize_dec.enabled
    resolved_stabilize_backend = stabilize_backend
    if stabilize_backend == "auto":
        # auto:
        #   - stabilize OFF: opencv 扱い (副作用なし)
        #   - stabilize ON + ffmpeg 未指定: vidstab 検出を試行 (成功 -> vidstab, 失敗 -> opencv)
        #   - stabilize ON + ffmpeg 指定済: 旧挙動互換で vidstab を仮定
        # NOTE (P2-4): CLI / GUI は自前で vidstab 有無を確認し、解決済みの
        # "vidstab" / "opencv" を渡してくる。この分岐は run_pipeline を直接呼ぶ
        # 他フロントエンド / テスト向けのフォールバックとして残している。
        if not stabilize_on:
            resolved_stabilize_backend = "opencv"  # 影響なし
        elif ffmpeg is not None:
            resolved_stabilize_backend = "vidstab"
        else:
            try:
                rb = deps.check_ffmpeg_with_vidstab()
                resolved_stabilize_backend = "vidstab"
                ffmpeg = rb.path
            except deps.MissingDependencyError:
                resolved_stabilize_backend = "opencv"

    # 依存解決: stabilize 必要時のみ vidstab build を要求
    needs_vidstab = stabilize_on and resolved_stabilize_backend == "vidstab"
    if ffmpeg is None:
        if needs_vidstab:
            ffmpeg = deps.check_ffmpeg_with_vidstab().path
        else:
            ffmpeg = deps.check_ffmpeg().path
    if ffprobe is None:
        ffprobe = deps.check_ffprobe().path
    upscale_dec = by.get("upscale")
    if realesrgan is None and upscale_dec is not None and upscale_dec.enabled:
        realesrgan = deps.check_realesrgan().path
    interp_dec = by.get("interpolate")
    if rife is None and interp_dec is not None and interp_dec.enabled:
        rife = deps.check_rife().path
    if codeformer is None and face_dec is not None and face_dec.enabled:
        codeformer = deps.check_codeformer_local()

    _check_cancel(cancel_event, "probe")

    fps = Fraction(meta.fps)
    has_audio = stream_plan.has_audio
    duration = float(meta.duration_sec)

    ranges = _compute_chunk_ranges(duration, chunk_seconds)
    log(
        f"[pipeline] fps={fps} audio={'yes' if has_audio else 'no'} "
        f"duration={duration:.2f}s chunks={len(ranges)} (chunk_seconds={chunk_seconds})"
    )
    _emit_progress(
        progress, "process_chunks", total_chunks=len(ranges), label=f"{len(ranges)} chunks"
    )

    owns_temp_run = temp_run is None
    try:
        if temp_run is None:
            temp_run = temp_workspace.create_temp_run_context(input_path)
        temp_workspace.validate_temp_relationships(temp_run, input_path, output_path)
        if estimated_peak_temp_bytes is not None:
            capacity_warning = temp_workspace.check_temp_capacity(
                temp_run,
                int(estimated_peak_temp_bytes),
            )
            if capacity_warning:
                log(f"[WARN] {capacity_warning}")
    except temp_workspace.TempWorkspaceError as exc:
        if owns_temp_run and temp_run is not None:
            try:
                temp_workspace.cleanup_temp_run(temp_run)
            except temp_workspace.TempWorkspaceError as cleanup_exc:
                log(
                    f"[WARN] cleanup failed after temp preflight error; "
                    f"temporary files remain at {temp_run.run_leaf}: {cleanup_exc}"
                )
        raise PipelineError("temp_workspace", str(exc)) from exc
    tmp_root = temp_run.run_leaf
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dirs: dict[str, Path | int] = {
        "frames_in": tmp_root / "frames_in",
        "frames_pre": tmp_root / "frames_pre",
        "frames_up": tmp_root / "frames_up",
        "frames_clamp": tmp_root / "frames_clamp",
        "frames_face": tmp_root / "frames_face",
        "frames_face_input": tmp_root / "frames_face_input",
        "frames_face_raw": tmp_root / "frames_face_raw",
        "frames_rife": tmp_root / "frames_rife",
        "frames_out": tmp_root / "frames_out",
        "chunks": tmp_root / "chunks",
        "transforms_root": tmp_root / "transforms",
        # resolution_clamp ステップが intermediate_width を計算するためのヒント。
        # Path 以外を入れる例外的キーは "_" prefix で示す。
        "_src_width": int(meta.width),
    }
    for k, d in tmp_dirs.items():
        if k.startswith("_"):
            continue
        assert isinstance(d, Path)
        d.mkdir(parents=True, exist_ok=True)

    # PR3: worker_count > 1 で並列実行を有効化。GPU 段は Semaphore(N) で同時実行数を制限。
    effective_workers = max(1, min(int(worker_count), len(ranges)))
    # PR-G1: GPU 段 (upscale/interpolate) の同時実行数。既定 1 で従来と完全等価。
    # gpu_concurrency は worker 数を超えても無意味 (同時に走るチャンク数が物理上限) なので clamp。
    effective_gpu = max(1, min(int(gpu_concurrency), effective_workers))
    if int(gpu_concurrency) > effective_workers:
        log(
            f"[WARN] --gpu-concurrency={gpu_concurrency} は worker 数 "
            f"{effective_workers} に丸めます"
        )
    gpu_gate = threading.Semaphore(effective_gpu)

    def _process_one(args_tuple) -> tuple[int, Path]:
        start_sec, dur_sec, idx = args_tuple
        _check_cancel(cancel_event, "process_chunks", chunk_idx=idx)
        _emit_progress(
            progress,
            "chunk_start",
            chunk_index=idx,
            total_chunks=len(ranges),
            label=f"chunk {idx + 1}/{len(ranges)}",
        )
        # チャンク別 tmp ディレクトリ (worker_count > 1 のときは衝突回避必須)
        if effective_workers > 1:
            chunk_tmp_dirs = _make_chunk_tmp_dirs(tmp_root, idx, src_width=int(meta.width))
            for k, d in chunk_tmp_dirs.items():
                if k.startswith("_"):
                    continue
                d.mkdir(parents=True, exist_ok=True)
            # chunks / transforms_root は共有 (atomic な書き込み)
            chunk_tmp_dirs["chunks"] = tmp_dirs["chunks"]
            chunk_tmp_dirs["transforms_root"] = tmp_dirs["transforms_root"]
            local_dirs = chunk_tmp_dirs
        else:
            local_dirs = tmp_dirs
        cp = _process_chunk(
            runner=runner,
            ffmpeg=ffmpeg,
            realesrgan=realesrgan,
            rife=rife,
            codeformer=codeformer,
            input_path=input_path,
            fps=fps,
            normalization_fps=framerate_plan.normalization_fps,
            expected_normalized_frames=(
                _normalized_frame_count(start_sec, dur_sec, fps)
                if framerate_plan.requires_normalization
                else None
            ),
            chunk_idx=idx,
            start_sec=start_sec,
            dur_sec=dur_sec,
            total_chunks=len(ranges),
            tmp_dirs=local_dirs,
            decisions=decisions,
            encode_args=encode_args,
            log=log,
            cancel_event=cancel_event,
            progress=progress,
            io_idle_timeout_sec=io_idle_timeout_sec,
            face_restore_options=resolved_face_restore_options,
            stabilize_backend=resolved_stabilize_backend,
            gpu_gate=gpu_gate if effective_workers > 1 else None,
            clamp_flags=clamp_flags,
            gpu_opts=GpuRunOptions(
                tilesize=gpu_tilesize,
                threads=gpu_threads,
                gpu_id=gpu_id,
                rife_gpu_id=rife_gpu_id,
            ),
            debug_split_postfilter=debug_split_postfilter,
            fuse_extract_prefilter=(
                experimental_fuse_extract_prefilter
                and not debug_split_extract_prefilter
                and getattr(meta, "framerate_status", "unknown") == "cfr"
            ),
            keep_temp=keep_temp,
        )
        _emit_progress(
            progress,
            "chunk_done",
            chunk_index=idx,
            total_chunks=len(ranges),
            label=f"chunk {idx + 1}/{len(ranges)} done",
        )
        return (idx, cp)

    succeeded = False
    try:
        chunk_paths_map: dict[int, Path] = {}
        if effective_workers > 1:
            errors: list[BaseException] = []
            errors_lock = threading.Lock()
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as pool:
                    futures = [pool.submit(_process_one, r) for r in ranges]
                    for f in concurrent.futures.as_completed(futures):
                        try:
                            idx, cp = f.result()
                            chunk_paths_map[idx] = cp
                        except BaseException as e:
                            with errors_lock:
                                errors.append(e)
                            cancel_event.set()
            except BaseException:
                # KeyboardInterrupt 等が as_completed の待機中に発生した場合も、
                # ワーカーへ cancel を伝えてから抜ける。これが無いと with __exit__ の
                # shutdown(wait=True) が走行中ワーカーの完走を待ってブロックする。
                cancel_event.set()
                raise
            if errors:
                raise errors[0]
            chunk_paths = [chunk_paths_map[i] for i in range(len(ranges))]
        else:
            chunk_paths = []
            for r in ranges:
                idx, cp = _process_one(r)
                chunk_paths.append(cp)

        _check_cancel(cancel_event, "concat")
        _emit_progress(progress, "concat", total_chunks=len(ranges), label="concatenating")
        _concat_chunks(
            runner=runner,
            ffmpeg=ffmpeg,
            chunks_dir=tmp_dirs["chunks"],
            chunk_paths=chunk_paths,
            input_path=input_path,
            stream_plan=stream_plan,
            duration_sec=duration,
            output_path=output_path,
            log=log,
            cancel_event=cancel_event,
            io_idle_timeout_sec=io_idle_timeout_sec,
        )
        log("[pipeline] done")
        _emit_progress(progress, "done", total_chunks=len(ranges), label="done")
        succeeded = True
    finally:
        if keep_temp:
            log(f"[keep-temp] temporary files preserved at: {tmp_root}")
        elif keep_temp_on_failure and not succeeded:
            # V2-9: 失敗・キャンセル等の異常終了時のみ中間生成物を残し、途中成功チャンクの
            # 再利用・診断を可能にする。成功時は keep_temp_on_failure でも削除する。
            log(f"[keep-temp-on-failure] 異常終了のため中間生成物を保持: {tmp_root}")
        elif owns_temp_run:
            try:
                temp_workspace.cleanup_temp_run(temp_run)
            except temp_workspace.TempWorkspaceError as exc:
                log(f"[WARN] cleanup failed; temporary files remain at {tmp_root}: {exc}")
