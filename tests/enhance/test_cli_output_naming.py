from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from video_enhance_lite import analyzer, cli, config, decisions as decisions_mod, pipeline
from video_enhance_lite import deps as ven_deps
from video_enhance_lite.decisions import StageDecision


@dataclass
class FakeMeta:
    width: int = 1920
    height: int = 1080
    fps: Fraction = Fraction(60, 1)
    duration_sec: float = 10.0
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
    """encode のみ ON (追加 deps 不要)。出力 = 入力スペックそのまま。"""
    decs: list[StageDecision] = []
    for key in config.STAGE_KEYS:
        on = key == "encode"
        params = {"crf": 22, "preset": "slow"} if on else {}
        decs.append(
            StageDecision(
                key=key,
                label_ja=config.STAGE_LABELS_JA.get(key, key),
                recommended=on,
                confidence="high",
                reason="test",
                enabled=on,
                params=params,
            )
        )
    return decs


def _patch(monkeypatch, tmp_path, meta: FakeMeta):
    """analyzer / deps / recommend / run_pipeline をスタブ化し、run_pipeline が受けた kwargs を捕捉する。"""
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    input_path = tmp_path / "clip.mp4"
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
    monkeypatch.setattr(analyzer, "analyze", lambda **kw: (meta, FakeStats()))
    monkeypatch.setattr(
        decisions_mod, "recommend", lambda meta, stats, defaults: _encode_only_decs()
    )
    captured: dict = {}
    monkeypatch.setattr(pipeline, "run_pipeline", lambda **kw: captured.update(kw))
    return input_path, captured


def test_default_output_name_uses_exec_spec(monkeypatch, tmp_path):
    # -o 未指定 → 既定出力名は実行スペック `_{幅}x{高}-{fps}p`
    meta = FakeMeta(width=1920, height=1080, fps=Fraction(60, 1))
    input_path, captured = _patch(monkeypatch, tmp_path, meta)
    rc = cli.main([str(input_path), "--auto-all", "--max-temp-gib", "100000"])
    assert rc == 0
    assert captured["output_path"] == tmp_path / "clip_1920x1080-60p.mp4"


def test_default_output_name_rounds_fps(monkeypatch, tmp_path):
    # 29.97fps → 30p に丸める
    meta = FakeMeta(width=1280, height=720, fps=Fraction(30000, 1001))
    input_path, captured = _patch(monkeypatch, tmp_path, meta)
    rc = cli.main([str(input_path), "--auto-all", "--max-temp-gib", "100000"])
    assert rc == 0
    assert captured["output_path"] == tmp_path / "clip_1280x720-30p.mp4"


def test_default_output_name_avoids_existing_file(monkeypatch, tmp_path):
    meta = FakeMeta(width=1920, height=1080, fps=Fraction(60, 1))
    input_path, captured = _patch(monkeypatch, tmp_path, meta)
    (tmp_path / "clip_1920x1080-60p.mp4").write_bytes(b"existing")
    rc = cli.main([str(input_path), "--auto-all", "--max-temp-gib", "100000"])
    assert rc == 0
    assert captured["output_path"] == tmp_path / "clip_1920x1080-60p~(1).mp4"


def test_default_output_name_uses_next_available_suffix(monkeypatch, tmp_path):
    meta = FakeMeta(width=1920, height=1080, fps=Fraction(60, 1))
    input_path, captured = _patch(monkeypatch, tmp_path, meta)
    (tmp_path / "clip_1920x1080-60p.mp4").write_bytes(b"existing")
    (tmp_path / "clip_1920x1080-60p~(1).mp4").write_bytes(b"existing")
    rc = cli.main([str(input_path), "--auto-all", "--max-temp-gib", "100000"])
    assert rc == 0
    assert captured["output_path"] == tmp_path / "clip_1920x1080-60p~(2).mp4"


def test_explicit_output_is_preserved(monkeypatch, tmp_path):
    # -o 明示 → スペック suffix を付けず指定パスをそのまま使う
    meta = FakeMeta(width=1920, height=1080, fps=Fraction(60, 1))
    input_path, captured = _patch(monkeypatch, tmp_path, meta)
    explicit = tmp_path / "my_custom_name.mp4"
    rc = cli.main(
        [
            str(input_path),
            "-o",
            str(explicit),
            "--auto-all",
            "--max-temp-gib",
            "100000",
        ]
    )
    assert rc == 0
    assert captured["output_path"] == explicit


def test_explicit_output_avoids_existing_file(monkeypatch, tmp_path):
    meta = FakeMeta(width=1920, height=1080, fps=Fraction(60, 1))
    input_path, captured = _patch(monkeypatch, tmp_path, meta)
    explicit = tmp_path / "my_custom_name.mp4"
    explicit.write_bytes(b"existing")
    rc = cli.main(
        [
            str(input_path),
            "-o",
            str(explicit),
            "--auto-all",
            "--max-temp-gib",
            "100000",
        ]
    )
    assert rc == 0
    assert captured["output_path"] == tmp_path / "my_custom_name~(1).mp4"


def test_explicit_output_directory_uses_exec_spec_name(monkeypatch, tmp_path):
    # -o に既存フォルダを渡す → その直下に実行スペック名で保存する
    meta = FakeMeta(width=1920, height=1080, fps=Fraction(60, 1))
    input_path, captured = _patch(monkeypatch, tmp_path, meta)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rc = cli.main(
        [
            str(input_path),
            "-o",
            str(out_dir),
            "--auto-all",
            "--max-temp-gib",
            "100000",
        ]
    )
    assert rc == 0
    assert captured["output_path"] == out_dir / "clip_1920x1080-60p.mp4"


def test_explicit_output_directory_avoids_existing_file(monkeypatch, tmp_path):
    meta = FakeMeta(width=1920, height=1080, fps=Fraction(60, 1))
    input_path, captured = _patch(monkeypatch, tmp_path, meta)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "clip_1920x1080-60p.mp4").write_bytes(b"existing")
    rc = cli.main(
        [
            str(input_path),
            "-o",
            str(out_dir),
            "--auto-all",
            "--max-temp-gib",
            "100000",
        ]
    )
    assert rc == 0
    assert captured["output_path"] == out_dir / "clip_1920x1080-60p~(1).mp4"
