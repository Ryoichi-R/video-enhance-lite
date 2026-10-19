from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

import pytest

pytestmark = pytest.mark.gui

from video_enhance_lite import config, decisions as decisions_mod, gui
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


# _tk_session_root / tk_root は conftest.py の共有 fixture を使う
# (ファイルごとの tk.Tk() 多重生成による init.tcl ロード失敗を避ける)。


def _all_decs() -> list[StageDecision]:
    return decisions_mod.recommend(FakeMeta(field_order="tt", width=640), FakeStats())


def test_summary_runtime_and_capacity_populated(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    dlg = gui.EnhanceDecisionDialog(
        tk_root,
        decs,
        meta=FakeMeta(),
        chunk_seconds=10,
        max_temp_gib=100.0,
    )
    assert dlg._summary_runtime_var.get() not in ("", "※分析未実施")
    assert dlg._summary_capacity_var.get() not in ("", "※分析未実施")
    assert "約" in dlg._summary_runtime_var.get()
    assert "GiB" in dlg._summary_capacity_var.get()


def test_summary_capacity_scales_with_worker_count(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    one = gui.EnhanceDecisionDialog(
        tk_root,
        decs,
        meta=FakeMeta(),
        chunk_seconds=10,
        worker_count=1,
    )
    four = gui.EnhanceDecisionDialog(
        tk_root,
        decs,
        meta=FakeMeta(),
        chunk_seconds=10,
        worker_count=4,
    )

    def _capacity(dialog) -> float:
        match = re.search(r"([\d.]+) GiB", dialog._summary_capacity_var.get())
        assert match is not None
        return float(match.group(1))

    assert _capacity(four) == pytest.approx(_capacity(one) * 4, abs=0.02)
    assert "worker_count=4" in four._summary_capacity_var.get()


def test_summary_updates_when_upscale_toggled(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    dlg = gui.EnhanceDecisionDialog(
        tk_root,
        decs,
        meta=FakeMeta(),
        chunk_seconds=10,
        max_temp_gib=100.0,
    )

    def _cap_gib() -> float:
        m = re.search(r"([\d.]+) GiB", dlg._summary_capacity_var.get())
        assert m is not None
        return float(m.group(1))

    before = _cap_gib()
    # upscale を切ると推定容量が下がる (trace 経由で _refresh_summary が走る)
    if "upscale" in dlg.vars:
        dlg.vars["upscale"]["enabled"].set(False)
        after = _cap_gib()
        assert after <= before


def test_capacity_warning_color_red_when_over_limit(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    # 極端に小さい上限で赤字判定を強制
    dlg = gui.EnhanceDecisionDialog(
        tk_root,
        decs,
        meta=FakeMeta(),
        chunk_seconds=10,
        max_temp_gib=0.001,
    )
    color = str(dlg._summary_capacity_label.cget("foreground"))
    assert color == "#c33"


def test_dialog_without_chunk_seconds_still_builds(tk_root, monkeypatch, tmp_path):
    # 後方互換: chunk_seconds / max_temp_gib 省略でも構築できる
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    dlg = gui.EnhanceDecisionDialog(tk_root, decs, meta=FakeMeta())
    assert dlg._chunk_seconds == config.DEFAULT_CHUNK_SECONDS
    assert dlg._summary_runtime_var.get() not in ("", "※分析未実施")
