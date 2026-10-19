from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from video_enhance_lite import analyzer, cli, config, decisions as decisions_mod, pipeline
from video_enhance_lite import deps as ven_deps
from video_enhance_lite.decisions import StageDecision
from video_enhance_lite.runtime_estimator import estimate_peak_for_decisions


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


def _make_decs(*, upscale: bool = False, scale: int = 4) -> list[StageDecision]:
    """テスト制御用 decisions。encode は常時 ON、upscale は引数で切替。"""
    decs: list[StageDecision] = []
    for key in config.STAGE_KEYS:
        rec = False
        en = False
        params: dict = {}
        if key == "encode":
            rec = True
            en = True
            params = {"crf": 22, "preset": "slow"}
        elif key == "upscale" and upscale:
            rec = True
            en = True
            params = {"scale": scale}
        decs.append(
            StageDecision(
                key=key,
                label_ja=config.STAGE_LABELS_JA.get(key, key),
                recommended=rec,
                confidence="high",
                reason="test",
                enabled=en,
                params=params,
            )
        )
    return decs


def _patch_common(monkeypatch, tmp_path, decs: list[StageDecision]) -> Path:
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
    return input_path


def test_auto_all_emits_summary(monkeypatch, tmp_path, capsys):
    decs = _make_decs(upscale=False)  # encode のみ → 追加 deps 不要
    input_path = _patch_common(monkeypatch, tmp_path, decs)
    monkeypatch.setattr(pipeline, "run_pipeline", lambda **kw: None)

    rc = cli.main([str(input_path), "--auto-all", "--max-temp-gib", "1000"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "[summary]" in err
    assert "estimated runtime:" in err
    assert "peak temp:" in err


def test_auto_all_over_max_temp_exits_2(monkeypatch, tmp_path, capsys):
    decs = _make_decs(upscale=True, scale=4)  # 大きい中間解像度で peak 超過させる
    input_path = _patch_common(monkeypatch, tmp_path, decs)
    monkeypatch.setattr(pipeline, "run_pipeline", lambda **kw: None)

    rc = cli.main([str(input_path), "--auto-all", "--max-temp-gib", "0.01"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "[ERROR]" in err
    assert "max-temp-gib" in err


def test_interactive_upscale_off_lowers_peak(monkeypatch, tmp_path, capsys):
    # upscale 推奨 ON の decisions を返すが、対話で upscale を切ったと仮定する。
    decs = _make_decs(upscale=True, scale=4)
    input_path = _patch_common(monkeypatch, tmp_path, decs)
    monkeypatch.setattr(pipeline, "run_pipeline", lambda **kw: None)

    # 対話確認をスタブ化: upscale を OFF にした decisions を返す
    def _fake_confirm(d, **kw):
        for x in d:
            if x.key == "upscale":
                x.enabled = False
        return d, False

    monkeypatch.setattr(cli, "_interactive_confirm", _fake_confirm)

    rc = cli.main([str(input_path), "--max-temp-gib", "1000"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "[summary]" in err

    # [summary] に出た peak が upscale OFF の見積もりと一致し、ON より小さい
    m = re.search(r"peak temp: ([\d.]+) GiB", err)
    assert m is not None
    printed_peak = float(m.group(1))

    # worker_count による倍率はあるが、upscale OFF の peak 下限 ≤ 表示値 < upscale ON 上限
    peak_off, _, _ = estimate_peak_for_decisions(
        FakeMeta(),
        _make_decs(upscale=False),
        config.DEFAULT_CHUNK_SECONDS,
    )
    peak_on, _, _ = estimate_peak_for_decisions(
        FakeMeta(),
        _make_decs(upscale=True, scale=4),
        config.DEFAULT_CHUNK_SECONDS,
    )
    assert peak_off < peak_on
    assert printed_peak < peak_on
