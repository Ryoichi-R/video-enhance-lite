from __future__ import annotations

import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from video_enhance_lite import analyzer, cli, config, decisions as decisions_mod, pipeline
from video_enhance_lite import deps as ven_deps
from video_enhance_lite.decisions import StageDecision


@dataclass
class FakeMeta:
    width: int = 1280
    height: int = 720
    fps: Fraction = Fraction(30, 1)
    duration_sec: float = 60.0
    codec: str = "h264"
    bitrate_bps: int | None = 5_000_000
    pix_fmt: str = "yuv420p"
    color_range: str = "tv"
    color_space: str = "bt709"
    field_order: str = "progressive"
    has_audio: bool = True


@dataclass
class FakeStats:
    laplacian_var_mean: float = 200.0
    luma_std_over_time: float = 1.0
    interframe_diff_mean: float = 1.0
    interframe_diff_std: float = 1.0
    histogram_flat_ratio: float = 0.05
    luma_mean: float = 120.0
    luma_clipping_ratio: float = 0.001
    saturation_mean: float = 80.0
    highfreq_energy: float = 5.0
    sample_count: int = 24


def _encode_only_decs() -> list[StageDecision]:
    decs: list[StageDecision] = []
    for key in config.STAGE_KEYS:
        rec = key == "encode"
        decs.append(
            StageDecision(
                key=key,
                label_ja=config.STAGE_LABELS_JA.get(key, key),
                recommended=rec,
                confidence="high",
                reason="test",
                enabled=rec,
                params={"crf": 22, "preset": "slow"} if rec else {},
            )
        )
    return decs


def test_cli_progress_emits_eta(monkeypatch, tmp_path, capsys):
    decs = _encode_only_decs()
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    input_path = tmp_path / "in.mp4"
    input_path.write_bytes(b"")
    monkeypatch.setattr(
        ven_deps,
        "check_ffmpeg",
        lambda: ven_deps.ResolvedBinary(path=Path(r"C:\bin\ffmpeg.exe"), version="n", sha256=None),
    )
    monkeypatch.setattr(
        ven_deps,
        "check_ffprobe",
        lambda: ven_deps.ResolvedBinary(path=Path(r"C:\bin\ffprobe.exe"), version="n", sha256=None),
    )
    monkeypatch.setattr(analyzer, "analyze", lambda **kw: (FakeMeta(), FakeStats()))
    monkeypatch.setattr(decisions_mod, "recommend", lambda meta, stats, defaults: decs)

    def _fake_run_pipeline(**kw):
        prog = kw.get("progress")
        assert prog is not None
        total = 12
        prog(pipeline.ProgressEvent(stage="process_chunks", total_chunks=total))
        for i in range(5):
            # elapsed を確実に正にするため 3 チャンク目以降で僅かに待つ
            time.sleep(0.01)
            prog(pipeline.ProgressEvent(stage="chunk_done", chunk_index=i, total_chunks=total))

    monkeypatch.setattr(pipeline, "run_pipeline", _fake_run_pipeline)

    rc = cli.main([str(input_path), "--auto-all", "--max-temp-gib", "1000"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "[progress]" in err
    assert "ETA" in err
