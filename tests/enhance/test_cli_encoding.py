"""P1-1 回帰: cp932 コンソールでの UnicodeEncodeError クラッシュ防止。

2 層の防御を検証する:
  1. 利用者向け文言 (decision reason / 判定表) が cp932 エンコード可能であること
  2. 万一エンコード不能文字が混入しても main() の reconfigure(errors="replace") で
     クラッシュしないこと
"""

from __future__ import annotations

import io
import sys
from fractions import Fraction
from pathlib import Path

from video_enhance_lite import cli, config, decisions as decisions_mod

from tests.enhance.test_cli import FakeMeta, FakeStats


def _decisions_all_branches() -> list[list]:
    """全ステージの推奨/非推奨の両分岐を踏む decisions セットを生成する。"""
    all_on = decisions_mod.recommend(
        # interlaced + 低解像度 + 低 fps -> deinterlace/upscale/interpolate 推奨
        FakeMeta(width=640, fps=Fraction(24, 1), field_order="tt", bitrate_bps=1_000_000),
        # 全閾値超え -> stabilize/deflicker/denoise/debanding/color/sharpen/grain 推奨
        FakeStats(
            laplacian_var_mean=config.T_BLUR - 10.0,
            luma_std_over_time=config.T_FLICKER + 5.0,
            interframe_diff_std=config.T_SHAKE + 5.0,
            histogram_flat_ratio=config.T_BANDING + 0.2,
            luma_mean=config.T_LUMA_DARK - 20.0,
            luma_clipping_ratio=config.T_LUMA_CLIPPING + 0.05,
            saturation_mean=config.T_SATURATION_LOW - 10.0,
            highfreq_energy=config.T_NOISE + 5.0,
        ),
    )
    all_off = decisions_mod.recommend(FakeMeta(width=1920), FakeStats())
    unknown_fo = decisions_mod.recommend(FakeMeta(width=1920, field_order=""), FakeStats())
    return [all_on, all_off, unknown_fo]


def test_all_decision_reasons_are_cp932_encodable():
    """全ステージ・全分岐の reason / label が cp932 でエンコードできること。"""
    for decs in _decisions_all_branches():
        for d in decs:
            d.reason.encode("cp932")  # raises UnicodeEncodeError on regression
            d.label_ja.encode("cp932")


def test_decision_table_is_cp932_encodable():
    for decs in _decisions_all_branches():
        cli._format_decision_table(decs).encode("cp932")


def test_main_analyze_only_survives_cp932_stdout(monkeypatch, tmp_path):
    """main() は cp932 (errors=strict) の stdout でもクラッシュせず exit 0 すること。

    実コンソール相当の TextIOWrapper(encoding="cp932") を差し込み、
    main() 冒頭の reconfigure(errors="replace") が効くことを検証する。
    """
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    input_path = tmp_path / "in.mp4"
    input_path.write_bytes(b"")

    from video_enhance_lite import deps as ven_deps

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

    from video_enhance_lite import analyzer

    # 手ぶれ検出 (= 旧 U+2014 入り reason の分岐) を必ず踏む
    monkeypatch.setattr(
        analyzer,
        "analyze",
        lambda **kw: (FakeMeta(width=640), FakeStats(interframe_diff_std=config.T_SHAKE + 5.0)),
    )

    out_buf = io.BytesIO()
    err_buf = io.BytesIO()
    cp932_out = io.TextIOWrapper(out_buf, encoding="cp932", errors="strict")
    cp932_err = io.TextIOWrapper(err_buf, encoding="cp932", errors="strict")
    monkeypatch.setattr(sys, "stdout", cp932_out)
    monkeypatch.setattr(sys, "stderr", cp932_err)

    rc = cli.main([str(input_path), "--analyze-only"])
    cp932_out.flush()
    cp932_err.flush()
    assert rc == 0
    table = out_buf.getvalue().decode("cp932")
    for k in config.STAGE_KEYS:
        assert k in table
