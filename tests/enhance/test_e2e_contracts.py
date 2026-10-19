"""実バイナリ契約の E2E スモークテスト (PR12)。

既存テストは FakeProcessRunner / モック前提のため「外部バイナリとの契約」を検証できず、
P0-1 (RIFE 配布構造) / P0-2 (yadif の枚数) / P1-1 (cp932 出力) はすべてこのギャップ
から漏れた。本モジュールは実 ffmpeg 等で契約だけを薄く検証する。

gate:
  - ffmpeg が PATH に無ければ module ごと skip (CI の最小環境で安全)
  - RIFE 実配布レイアウトの検証は環境変数 VIDEO_ENHANCE_E2E_RIFE_DIR 指定時のみ
  - 本リポジトリの既定 WORKSPACE_ROOT 前提のテストは
    config.WORKSPACE_ROOT が実在するときのみ実行
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from video_enhance_lite import analyzer, config, stages

pytestmark = pytest.mark.e2e

_FFMPEG = shutil.which("ffmpeg")
if _FFMPEG is None:
    pytest.skip("ffmpeg not on PATH; e2e contract tests skipped", allow_module_level=True)


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=True,
        shell=False,
    )


def _gen_pngs(out_dir: Path, count: int) -> None:
    """testsrc から count 枚の PNG 連番 (00000000.png 始まり) を生成する。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=10",
            "-frames:v",
            str(count),
            "-start_number",
            "0",
            str(out_dir / "%08d.png"),
        ]
    )


def _count_png(d: Path) -> int:
    return sum(1 for p in d.iterdir() if p.suffix == ".png")


def _png_bytes(directory: Path) -> list[bytes]:
    return [path.read_bytes() for path in sorted(directory.glob("*.png"))]


@pytest.mark.parametrize("suffix", ["mp4", "ts"])
def test_contract_input_seek_matches_legacy_output_seek_at_non_keyframe(tmp_path, suffix):
    """Input-side accurate seek preserves the legacy boundary for MP4 and non-zero-PTS TS."""
    video = tmp_path / f"source.{suffix}"
    fast = tmp_path / f"fast-{suffix}"
    legacy = tmp_path / f"legacy-{suffix}"
    fast.mkdir()
    legacy.mkdir()
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=10",
            "-t",
            "8",
            "-c:v",
            "libx264",
            "-g",
            "47",
            "-bf",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ]
    )
    start_sec = 3.7
    duration_sec = 1.3
    subprocess.run(
        stages.build_extract_args(
            Path(_FFMPEG),
            video,
            start_sec,
            duration_sec,
            fast,
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=True,
        shell=False,
    )
    _run_ffmpeg(
        [
            "-i",
            str(video),
            "-ss",
            str(start_sec),
            "-t",
            str(duration_sec),
            "-fps_mode",
            "passthrough",
            "-start_number",
            "0",
            str(legacy / "%08d.png"),
        ]
    )

    assert _png_bytes(fast) == _png_bytes(legacy)
    assert len(_png_bytes(fast)) == 13


def test_contract_b_frames_and_millisecond_timescale_are_cfr(tmp_path):
    """decoded-frame timestamps avoid B-frame packet order and tolerate 33/34 ms cadence."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        pytest.skip("ffprobe not on PATH")
    video = tmp_path / "bframes-timescale1000.mp4"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=30",
            "-t",
            "3",
            "-c:v",
            "libx264",
            "-bf",
            "2",
            "-video_track_timescale",
            "1000",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ]
    )
    meta = analyzer.probe_metadata(Path(ffprobe), video)
    windows = analyzer._probe_frame_timestamp_windows(Path(ffprobe), video, meta.duration_sec)
    deltas = [b - a for window in windows or [] for a, b in zip(window, window[1:])]
    assert meta.framerate_status == "cfr", (meta.fps, meta.avg_fps, min(deltas), max(deltas))


@pytest.mark.parametrize(
    "filter_chain",
    ["hqdn3d=1.5:1.5:6:6", "yadif=0", "deflicker=size=5:mode=pm"],
)
def test_contract_fused_extract_prefilter_matches_split(tmp_path, filter_chain):
    """The explicit rgb24 conversion keeps fused and PNG-split pixels identical."""
    video = tmp_path / "source.mp4"
    split_in = tmp_path / "split-in"
    split_out = tmp_path / "split-out"
    fused_out = tmp_path / "fused-out"
    for directory in (split_in, split_out, fused_out):
        directory.mkdir()
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=10",
            "-t",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ]
    )
    _run_ffmpeg(["-i", str(video), str(split_in / "%08d.png")])
    _run_ffmpeg(
        [
            "-framerate",
            "10",
            "-i",
            str(split_in / "%08d.png"),
            "-vf",
            filter_chain,
            str(split_out / "%08d.png"),
        ]
    )
    _run_ffmpeg(
        [
            "-i",
            str(video),
            "-vf",
            f"format=rgb24,{filter_chain}",
            str(fused_out / "%08d.png"),
        ]
    )
    assert _png_bytes(split_out) == _png_bytes(fused_out)


# ---------- 契約 1: yadif とフレーム枚数 (P0-2) ----------


def test_contract_yadif0_preserves_frame_count(tmp_path):
    """P0-2 契約: yadif=0 (send_frame) はフレーム枚数を変えない。"""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _gen_pngs(in_dir, 10)
    assert _count_png(in_dir) == 10
    _run_ffmpeg(
        [
            "-framerate",
            "10",
            "-start_number",
            "0",
            "-i",
            str(in_dir / "%08d.png"),
            "-vf",
            "yadif=0",
            "-start_number",
            "0",
            str(out_dir / "%08d.png"),
        ]
    )
    assert _count_png(out_dir) == 10


def test_contract_yadif1_doubles_frame_count(tmp_path):
    """対照実験: yadif=1 (send_field) は枚数を倍化する = 固定 framerate mux の
    本パイプラインで採用してはならない理由。退行時にどちら向きに壊れたか判別する。"""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _gen_pngs(in_dir, 10)
    _run_ffmpeg(
        [
            "-framerate",
            "10",
            "-start_number",
            "0",
            "-i",
            str(in_dir / "%08d.png"),
            "-vf",
            "yadif=1",
            "-start_number",
            "0",
            str(out_dir / "%08d.png"),
        ]
    )
    assert _count_png(out_dir) == 20


# ---------- 契約 2: RIFE 公式配布レイアウト (P0-1) ----------


def test_contract_resolve_rife_model_dir_against_real_distribution():
    """P0-1 契約: 実 RIFE 配布ディレクトリ (env 指定時のみ) で
    resolve_rife_model_dir が成功し、validate も通ること。"""
    rife_dir = os.environ.get("VIDEO_ENHANCE_E2E_RIFE_DIR")
    if not rife_dir:
        pytest.skip("VIDEO_ENHANCE_E2E_RIFE_DIR not set")
    exe_dir = Path(rife_dir)
    if not exe_dir.is_dir():
        pytest.skip(f"VIDEO_ENHANCE_E2E_RIFE_DIR is not a directory: {exe_dir}")
    models_dir = config.resolve_rife_model_dir(exe_dir, config.DEFAULT_RIFE_MODEL)
    config.validate_rife_model(models_dir, config.DEFAULT_RIFE_MODEL)


def test_contract_rife_verbose_emits_liveness_against_real_distribution(tmp_path):
    """実配布版で build_rife_args の -v が保存完了 `done` をstderrへ出す。"""
    rife_dir = os.environ.get("VIDEO_ENHANCE_E2E_RIFE_DIR")
    if not rife_dir:
        pytest.skip("VIDEO_ENHANCE_E2E_RIFE_DIR not set")
    exe_dir = Path(rife_dir)
    exe_candidates = [
        exe_dir / "rife-ncnn-vulkan.exe",
        exe_dir / "rife-ncnn-vulkan",
    ]
    exe = next((p for p in exe_candidates if p.is_file()), None)
    if exe is None:
        pytest.skip(f"rife executable not found in: {exe_dir}")

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _gen_pngs(in_dir, 2)
    argv = stages.build_rife_args(
        rife_bin=exe,
        in_dir=in_dir,
        out_dir=out_dir,
        source_fps=Fraction(24, 1),
        target_fps=Fraction(48, 1),
        input_frame_count=2,
        model=config.DEFAULT_RIFE_MODEL,
    )
    proc = subprocess.run(
        argv,
        cwd=exe_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
        shell=False,
    )
    assert proc.returncode == 0, proc.stderr[-1000:]
    assert _count_png(out_dir) == 4
    assert "done" in proc.stderr


# ---------- 契約 3: image2 の 1 始まり連番フォールバック (mux) ----------


def test_contract_image2_reads_one_based_sequence_with_start_number_0(tmp_path):
    """mux 連番契約: RIFE の出力は 00000001.png (1 始まり) だが、mux は
    `-start_number 0` で読む。image2 demuxer の探索フォールバック
    (start_number_range) で全枚数読めることを ffmpeg ビルドに対して固定する。"""
    src = tmp_path / "src"
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _gen_pngs(src, 4)
    for i, p in enumerate(sorted(src.glob("*.png")), start=1):
        shutil.copy(p, in_dir / f"{i:08d}.png")
    proc = subprocess.run(
        [
            _FFMPEG,
            "-y",
            "-hide_banner",
            "-framerate",
            "10",
            "-start_number",
            "0",
            "-i",
            str(in_dir / "%08d.png"),
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=True,
        shell=False,
    )
    # 最終行の frame= カウンタが全 4 枚を示すこと
    assert re.search(r"frame=\s*4\b", proc.stderr), proc.stderr[-500:]


# ---------- 契約 4: cp932 コンソールでの --analyze-only (P1-1) ----------


def test_contract_analyze_only_survives_cp932_console(tmp_path):
    """P1-1 契約: PYTHONIOENCODING=cp932 の子プロセスで --analyze-only が exit 0。

    実行環境で出力先を固定するため `-o` で workspace 内を明示する。
    allowlist は環境変数 override (完全置換) で自己完結させる。
    """
    if not config.WORKSPACE_ROOT.is_dir():
        pytest.skip(f"workspace root not present: {config.WORKSPACE_ROOT}")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        pytest.skip("ffprobe not on PATH")

    # NOTE: rate / 長さは analyze の 24 サンプル目 (t = duration * 23.5 / 24) が
    # 必ず実フレーム上に乗るよう余裕を持たせる (短すぎると最終サンプルの -ss が
    # 末尾フレーム PTS を超えて "produced no file" になる)。
    video = tmp_path / "in.mp4"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=30",
            "-t",
            "5",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ]
    )

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp932"
    ffmpeg_dir = str(Path(_FFMPEG).parent)
    env["VIDEO_ENHANCE_FFMPEG_ROOTS"] = ffmpeg_dir
    env["VIDEO_ENHANCE_FFPROBE_ROOTS"] = str(Path(ffprobe).parent)
    # 期待ハッシュは local config の値が PATH 上の実バイナリと一致しない環境でも
    # 自己完結するよう空 override する (検証スキップ = informational)
    env[config.ENV_BINARY_EXPECTED_SHA256] = "{}"

    out_path = config.WORKSPACE_ROOT / ".tmp" / "e2e_cp932_out.mp4"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "video_enhance_lite",
            str(video),
            "-o",
            str(out_path),
            "--analyze-only",
        ],
        capture_output=True,
        text=True,
        encoding="cp932",
        errors="replace",
        timeout=300,
        env=env,
        shell=False,
    )
    assert proc.returncode == 0, f"stderr tail: {proc.stderr[-1000:]}"
    for key in config.STAGE_KEYS:
        assert key in proc.stdout
