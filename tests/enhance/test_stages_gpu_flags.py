from __future__ import annotations

from fractions import Fraction


from video_enhance_lite import stages


def _rife(tmp_path, **kw):
    return stages.build_rife_args(
        rife_bin=tmp_path / "rife.exe",
        in_dir=tmp_path / "in",
        out_dir=tmp_path / "out",
        source_fps=Fraction(24, 1),
        target_fps=Fraction(48, 1),
        input_frame_count=10,
        **kw,
    )


# ---------- 既定: tuning flag 未指定でも必須の -v 生存信号だけは付与 ----------


def test_rife_default_uses_only_required_liveness_flag(tmp_path):
    base = _rife(tmp_path)
    assert base.count("-v") == 1
    assert "-j" not in base
    assert "-g" not in base
    assert "-t" not in base  # RIFE は -t を一切付けない


def test_rife_threads_and_gpu_id(tmp_path):
    args = _rife(tmp_path, threads="2:4:4", gpu_id=0)
    assert args.count("-v") == 1
    assert args[args.index("-j") + 1] == "2:4:4"
    assert args[args.index("-g") + 1] == "0"
    # -t は RIFE には付かない
    assert "-t" not in args


def test_rife_gpu_id_minus1_cpu(tmp_path):
    args = _rife(tmp_path, gpu_id=-1)
    assert args[args.index("-g") + 1] == "-1"


def test_rife_v2_path_also_gets_flags(tmp_path):
    """v2/v3 系 (-n なし return path) でも -j/-g が反映される。"""
    args = _rife(tmp_path, model="rife-v2.3", threads="1:2:2", gpu_id=1)
    assert args.count("-v") == 1
    assert "-n" not in args  # v2 系は -n 非対応
    assert args[args.index("-j") + 1] == "1:2:2"
    assert args[args.index("-g") + 1] == "1"


def test_clamp_filter_unaffected_by_gpu_flags():
    """無関係な関数: build_resolution_clamp_filter は GPU フラグの影響を受けない (sanity)。"""
    assert stages.build_resolution_clamp_filter(2560, 3840) == "scale=2560:-2:flags=lanczos"
