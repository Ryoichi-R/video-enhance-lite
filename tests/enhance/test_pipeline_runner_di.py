"""V2-8 DI 固有の観点を検証する新規テスト。

実監視ループそのもの (frame 進捗 / idle / cancel / 起動失敗 / 非0終了) は
test_pipeline_progress.py / test_pipeline_timeout.py が `RealProcessRunner` 直呼びで
網羅する。本ファイルは「runner を注入する DI 配線」に固有の 4 観点 + fake 契約を扱う:

  1. 注入した runner が全ステージ (extract/pre/clamp/post/upscale/interpolate/
     stabilize_detect) と concat で使われること。
  2. fake の失敗が PipelineError 契約を満たす (CalledProcessError が逐次/並列経路へ
     漏れないこと)。
  3. 同一 runner を worker>1 で共有しても記録に race が無いこと。
  4. stage / chunk_idx / cwd が runner へ欠落なく伝播すること。
  5. FakeProcessRunner が呼び出しを記録し fail_on_argv_marker で失敗を再現すること。
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest

from video_enhance_lite import config, pipeline
from video_enhance_lite.decisions import StageDecision

from tests.enhance.test_pipeline import FakeMeta, _make_decisions  # noqa: F401
from tests.support.process_runner import FakeProcessRunner

_FFMPEG = Path(r"C:\bin\ffmpeg.exe")
_FFPROBE = Path(r"C:\bin\ffprobe.exe")
_REALESRGAN = Path(r"C:\bin\realesrgan-ncnn-vulkan.exe")
_RIFE = Path(r"C:\bin\rife-ncnn-vulkan.exe")


class RecordingRunner(FakeProcessRunner):
    """FakeProcessRunner を継承し、stage / chunk_idx / cwd も構造化記録する。

    出力 (PNG/mp4) 書き出しと例外契約は基底クラスに委譲する。
    """

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.events: list[dict] = []
        self._evlock = threading.Lock()

    def run_with_progress(self, argv, *, stage="ffmpeg", chunk_idx=None, cwd=None, **kw):
        with self._evlock:
            self.events.append(
                {
                    "argv": list(argv),
                    "stage": stage,
                    "chunk_idx": chunk_idx,
                    "cwd": str(cwd) if cwd is not None else None,
                }
            )
        return super().run_with_progress(argv, stage=stage, chunk_idx=chunk_idx, cwd=cwd, **kw)


def _full_decisions(*, target_width: int) -> list[StageDecision]:
    """全ステージ ON の decisions (extract 以外の全 runner ステージを発火させる)。"""
    enabled = {"deinterlace", "denoise", "sharpen", "stabilize", "upscale", "interpolate"}
    out: list[StageDecision] = []
    for k in config.STAGE_KEYS:
        on = (k in enabled) or (k == "encode")
        params: dict = {}
        if k == "upscale":
            params = {
                "scale": 2,
                "model": config.DEFAULT_REALESRGAN_MODEL,
                "target_width": target_width,
            }
        elif k == "interpolate":
            params = {"target_fps_num": 60, "target_fps_den": 1}
        elif k == "encode":
            params = {"crf": config.DEFAULT_ENCODE_CRF, "preset": config.DEFAULT_ENCODE_PRESET}
        out.append(
            StageDecision(
                key=k,
                label_ja=config.STAGE_LABELS_JA[k],
                recommended=on,
                confidence="high",
                reason="test",
                enabled=on,
                params=params,
            )
        )
    return out


# ---------- 観点 1: 全ステージ + concat で runner が使われる ----------


def test_injected_runner_used_across_all_stages_and_concat(workspace):
    """注入 runner が extract/stabilize_detect/pre/upscale/clamp/interpolate/post/mux/concat
    の全 runner 呼び出し点で使われること。

    fused mux では post_filter は mux に融合されるため、post_filter を独立ステージとして
    観測するには debug_split_postfilter=True を使う。
    """
    runner = RecordingRunner()
    # 1920px x scale2 = 3840 中間, target=2560 → resolution_clamp 発火。
    decs = _full_decisions(target_width=2560)
    pipeline.run_pipeline(
        runner=runner,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=decs,
        meta=FakeMeta(width=1920, height=1080, duration_sec=10.0),
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=_FFMPEG,
        ffprobe=_FFPROBE,
        realesrgan=_REALESRGAN,
        rife=_RIFE,
        stabilize_backend="vidstab",
        debug_split_postfilter=True,
    )
    seen = {e["stage"] for e in runner.events}
    expected = {
        "extract_frames",
        "stabilize_detect",
        "pre_filter",
        "upscale_frames",
        "resolution_clamp",
        "interpolate",
        "post_filter",
        "reassemble",  # mux
        "concat",
    }
    missing = expected - seen
    assert not missing, f"runner が呼ばれていないステージ: {missing} (seen={seen})"


# ---------- 観点 2: PipelineError 契約 (CalledProcessError が漏れない) ----------


@pytest.mark.parametrize("worker_count", [1, 3])
def test_runner_failure_raises_pipeline_error_not_called_process_error(workspace, worker_count):
    """fake の失敗が逐次 (worker=1) / 並列 (worker=3) いずれも PipelineError で送出され、
    生 subprocess.CalledProcessError が呼び出し側へ漏れないこと。"""
    failer = FakeProcessRunner(fail_on_argv_marker="frames_in")
    with pytest.raises(pipeline.PipelineError) as exc:
        pipeline.run_pipeline(
            runner=failer,
            input_path=workspace / "in.mp4",
            output_path=workspace / "out.mp4",
            decisions=_make_decisions(),
            meta=FakeMeta(duration_sec=30.0),
            chunk_seconds=10,
            log=lambda _: None,
            ffmpeg=_FFMPEG,
            ffprobe=_FFPROBE,
            worker_count=worker_count,
        )
    assert not isinstance(exc.value, subprocess.CalledProcessError)
    assert exc.value.stage == "extract_frames"


# ---------- 観点 3: worker>1 共有時の記録 race が無い ----------


def test_shared_runner_records_no_race_under_parallel(workspace):
    """同一 runner を worker=3 で共有しても記録欠落/破損が無い (Lock 保護)。

    encode-only: chunk あたり extract + mux = 2 回、最後に concat 1 回。
    3 chunks → 3*2 + 1 = 7 回の決定的な呼び出し回数になる。
    """
    runner = FakeProcessRunner()
    pipeline.run_pipeline(
        runner=runner,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=_make_decisions(),
        meta=FakeMeta(duration_sec=30.0),
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=_FFMPEG,
        ffprobe=_FFPROBE,
        worker_count=3,
    )
    calls = runner.calls
    assert len(calls) == 7, f"記録欠落/重複 (race) の疑い: {len(calls)} != 7"
    # 各記録が torn write されず list[str] のまま
    assert all(isinstance(c, list) and all(isinstance(t, str) for t in c) for c in calls)
    # calls / cwds の長さが一致 (単一記録 list からの導出)
    assert len(runner.calls) == len(runner.cwds)


# ---------- 観点 4: stage / chunk_idx / cwd の伝播 ----------


def test_stage_chunk_idx_cwd_propagated_to_runner(workspace):
    """stage / chunk_idx / cwd が欠落なく runner へ伝播すること。"""
    runner = RecordingRunner()
    pipeline.run_pipeline(
        runner=runner,
        input_path=workspace / "in.mp4",
        output_path=workspace / "out.mp4",
        decisions=_make_decisions(),
        meta=FakeMeta(duration_sec=20.0),  # 2 chunks
        chunk_seconds=10,
        log=lambda _: None,
        ffmpeg=_FFMPEG,
        ffprobe=_FFPROBE,
    )
    extract = [e for e in runner.events if e["stage"] == "extract_frames"]
    mux = [e for e in runner.events if e["stage"] == "reassemble"]
    concat = [e for e in runner.events if e["stage"] == "concat"]

    # extract / mux は各 2 チャンク分、chunk_idx が 0/1 で伝播
    assert {e["chunk_idx"] for e in extract} == {0, 1}
    assert {e["chunk_idx"] for e in mux} == {0, 1}
    # 全 chunk ステージは chunk_idx 非 None
    for e in extract + mux:
        assert e["chunk_idx"] is not None
        assert e["cwd"] is None  # chunk ステージは cwd を渡さない

    # concat は 1 回、chunk_idx は None、cwd は chunks ディレクトリ
    assert len(concat) == 1
    assert concat[0]["chunk_idx"] is None
    assert concat[0]["cwd"] is not None
    assert concat[0]["cwd"].endswith("chunks")


# ---------- 観点 5: FakeProcessRunner の記録 + fail marker ----------


def test_fake_process_runner_records_and_fails(tmp_path):
    """FakeProcessRunner が呼び出しを記録し、fail_on_argv_marker で PipelineError を送出する。"""
    runner = FakeProcessRunner()
    out_mp4 = tmp_path / "chunks" / "chunk_000.mp4"
    rc = runner.run_with_progress(
        [r"C:\bin\ffmpeg.exe", "-y", "-i", "in", str(out_mp4)],
        cwd=tmp_path,
        stage="reassemble",
        chunk_idx=2,
    )
    assert rc == 0
    assert out_mp4.exists()  # fake 成果物が書かれる
    assert runner.calls == [[r"C:\bin\ffmpeg.exe", "-y", "-i", "in", str(out_mp4)]]
    assert runner.cwds == [str(tmp_path)]

    # fail marker 一致時は PipelineError を直接送出 (CalledProcessError は投げない)
    failer = FakeProcessRunner(fail_on_argv_marker="frames_in")
    with pytest.raises(pipeline.PipelineError) as exc:
        failer.run_with_progress(
            [r"C:\bin\ffmpeg.exe", "-i", str(tmp_path / "frames_in" / "%08d.png"), "o"],
            stage="extract_frames",
            chunk_idx=1,
        )
    assert not isinstance(exc.value, subprocess.CalledProcessError)
    assert exc.value.stage == "extract_frames"
    assert exc.value.chunk_idx == 1
