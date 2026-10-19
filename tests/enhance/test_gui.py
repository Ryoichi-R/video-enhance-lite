from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import threading

import pytest

pytestmark = pytest.mark.gui

from video_enhance_lite import (
    analyzer,
    config,
    decisions as decisions_mod,
    deps,
    gui,
    pipeline,
    runtime_estimator,
)
from video_enhance_lite.decisions import StageDecision, compute_output_summary


@dataclass
class _FakeMeta:
    width: int = 1280
    height: int = 720
    fps: Fraction = Fraction(30, 1)
    duration_sec: float = 10.0
    has_audio: bool = False


def _encode_only_decisions() -> list[StageDecision]:
    """upscale / interpolate OFF の最小 decisions (出力 = 入力スペック)。"""
    return [
        StageDecision(
            key="encode",
            label_ja="最終エンコード",
            recommended=True,
            confidence="high",
            reason="",
            enabled=True,
            params={},
        )
    ]


# ---------- pure helpers ----------


@pytest.mark.parametrize("good", ["1", "10", " 30 "])
def test_parse_chunk_seconds_accepts(good):
    assert gui._parse_chunk_seconds(good) >= 1


@pytest.mark.parametrize("bad", ["", "  ", "abc", "1.5", "0", "-1"])
def test_parse_chunk_seconds_rejects(bad):
    with pytest.raises(ValueError):
        gui._parse_chunk_seconds(bad)


@pytest.mark.parametrize("good", ["1", "120", " 1800 "])
def test_parse_positive_int_accepts(good):
    assert gui._parse_positive_int(good, label="timeout") >= 1


@pytest.mark.parametrize("bad", ["", "  ", "abc", "1.5", "0", "-1"])
def test_parse_positive_int_rejects(bad):
    with pytest.raises(ValueError):
        gui._parse_positive_int(bad, label="timeout")


def test_format_framerate_warning_uses_user_facing_status():
    suspected = _FakeMeta()
    suspected.framerate_status = "vfr_suspected"
    suspected.avg_fps = Fraction(24, 1)
    suspected_warning = gui._format_framerate_warning(suspected)
    assert "VFRの可能性" in suspected_warning
    assert "明示CFR normalization" in suspected_warning

    unknown = _FakeMeta()
    unknown.framerate_status = "unknown"
    unknown.avg_fps = None
    warning = gui._format_framerate_warning(unknown)
    assert warning is not None
    assert "判定できませんでした" in warning
    assert "処理は開始できません" in warning


def test_temp_guard_passes_effective_worker_count(app, monkeypatch):
    app._meta = _FakeMeta()
    captured = {}

    def _estimate(_meta, _decs, _chunk_seconds, *, worker_count=1, **_kwargs):
        captured["worker_count"] = worker_count
        return 0.1, None, None

    monkeypatch.setattr(gui.runtime_estimator, "estimate_peak_for_decisions", _estimate)
    assert app._estimate_temp_and_abort_if_exceed([], 10, 100.0, 4) is True
    assert captured["worker_count"] == 4


def test_derive_output_path_for_gui_uses_workspace(monkeypatch, tmp_path):
    # 出力フォルダ未指定時は WORKSPACE_ROOT 直下に置かれ、実行スペックで命名される。
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    inp = Path(r"D:\elsewhere\my video.mp4")
    summary = compute_output_summary(
        _FakeMeta(width=1280, height=720, fps=Fraction(30, 1)),
        _encode_only_decisions(),
    )
    out = gui.derive_output_path_for_gui(inp, summary)
    assert out == tmp_path / "my video_1280x720-30p.mp4"
    assert out.parent == tmp_path


def test_derive_output_path_for_gui_with_summary_uses_spec(monkeypatch, tmp_path):
    # summary 付き = 実行スペック命名 `_{幅}x{高}-{fps}p`
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    inp = Path(r"D:\elsewhere\my video.mp4")
    summary = compute_output_summary(
        _FakeMeta(width=1920, height=1080, fps=Fraction(60, 1)),
        _encode_only_decisions(),
    )
    out = gui.derive_output_path_for_gui(inp, summary)
    assert out == tmp_path / "my video_1920x1080-60p.mp4"


def test_derive_output_path_for_gui_uses_selected_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    selected = tmp_path / "out"
    selected.mkdir()
    inp = Path(r"D:\elsewhere\my video.mp4")
    summary = compute_output_summary(
        _FakeMeta(width=1920, height=1080, fps=Fraction(60, 1)),
        _encode_only_decisions(),
    )
    out = gui.derive_output_path_for_gui(inp, summary, selected)
    assert out == selected / "my video_1920x1080-60p.mp4"


def test_derive_output_path_for_gui_avoids_existing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    inp = Path(r"D:\elsewhere\my video.mp4")
    summary = compute_output_summary(
        _FakeMeta(width=1920, height=1080, fps=Fraction(60, 1)),
        _encode_only_decisions(),
    )
    (tmp_path / "my video_1920x1080-60p.mp4").write_bytes(b"existing")
    (tmp_path / "my video_1920x1080-60p~(1).mp4").write_bytes(b"existing")
    out = gui.derive_output_path_for_gui(inp, summary)
    assert out == tmp_path / "my video_1920x1080-60p~(2).mp4"


def test_validate_output_dir_for_gui_accepts_outside_workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path / "workspace")
    config.WORKSPACE_ROOT.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    assert gui._validate_output_dir_for_gui(outside) == outside.resolve()


def test_validate_output_dir_for_gui_accepts_workspace_root(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    assert gui._validate_output_dir_for_gui(tmp_path) == tmp_path.resolve()


# ---------- Tk fixtures ----------
# _tk_session_root / tk_root は conftest.py の共有 fixture を使う
# (ファイルごとの tk.Tk() 多重生成による init.tcl ロード失敗を避ける)。


@pytest.fixture
def app(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    return gui.App(tk_root)


class _AliveWorker:
    @staticmethod
    def is_alive() -> bool:
        return True


def test_window_close_without_worker_destroys_immediately(app, monkeypatch):
    destroyed = []
    monkeypatch.setattr(app.root, "destroy", lambda: destroyed.append(True))

    app._on_close()

    assert destroyed == [True]
    assert app._close_requested is False


def test_window_close_during_worker_can_be_declined(app, monkeypatch):
    destroyed = []
    app._worker_thread = _AliveWorker()
    app._cancel_event = threading.Event()
    monkeypatch.setattr(gui.messagebox, "askyesno", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(app.root, "destroy", lambda: destroyed.append(True))

    app._on_close()

    assert app._cancel_event.is_set() is False
    assert app._close_requested is False
    assert destroyed == []


def test_window_close_during_worker_waits_for_cancelled_finish(app, monkeypatch):
    destroyed = []
    app._worker_thread = _AliveWorker()
    app._cancel_event = threading.Event()
    app._current_worker_kind = "run"
    monkeypatch.setattr(gui.messagebox, "askyesno", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(app.root, "destroy", lambda: destroyed.append(True))

    app._on_close()

    assert app._cancel_event.is_set() is True
    assert app._close_requested is True
    assert destroyed == []
    assert app.var_status.get() == "終了待機中…"

    app._handle_finish(gui._FinishEvent("cancelled"))

    assert destroyed == [True]


# ---------- App tests ----------


def test_io_idle_timeout_default_comes_from_pipeline_constant(app):
    assert app.var_io_idle_timeout_sec.get() == str(pipeline.DEFAULT_IO_IDLE_TIMEOUT_SEC)


@pytest.mark.parametrize("bad", ["", "0", "-1", "abc"])
def test_on_run_rejects_invalid_io_idle_timeout_before_worker(app, monkeypatch, tmp_path, bad):
    input_path = tmp_path / "in.mp4"
    input_path.write_bytes(b"")
    app.var_input.set(str(input_path))
    app.var_io_idle_timeout_sec.set(bad)
    started: list[object] = []
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(app, "_start_analyze_worker", lambda *args: started.append(args))
    monkeypatch.setattr(
        gui.messagebox, "showerror", lambda title, message: errors.append((title, message))
    )

    app._on_run()

    assert not started
    assert errors
    assert "I/O無出力タイムアウト" in errors[0][1]


def test_run_passes_io_idle_timeout_without_changing_worker_settings(app, monkeypatch, tmp_path):
    input_path = tmp_path / "in.mp4"
    input_path.write_bytes(b"")
    app.var_input.set(str(input_path))
    app.var_output.set(str(tmp_path))
    app.var_io_idle_timeout_sec.set("1800")
    app.var_chunk.set("17")
    app.var_worker_count.set("2")
    app.var_gpu_concurrency.set("1")
    app._meta = FakeMeta(duration_sec=10.0)
    decs = _encode_only_decisions()
    app._decisions = decs
    monkeypatch.setattr(app, "_open_decision_dialog", lambda: decs)
    monkeypatch.setattr(app, "_estimate_temp_and_abort_if_exceed", lambda *args: True)
    monkeypatch.setattr(runtime_estimator, "estimate_runtime", lambda *args: None)
    # FakeMeta intentionally omits framerate_status and takes the explicit
    # legacy-CFR compatibility path without opening a modal dialog.
    monkeypatch.setattr(gui.messagebox, "askyesno", lambda *_args, **_kwargs: True)

    resolved = deps.ResolvedBinary(path=tmp_path / "tool.exe", version=None, sha256=None)
    monkeypatch.setattr(deps, "check_ffprobe", lambda: resolved)
    monkeypatch.setattr(deps, "check_ffmpeg", lambda: resolved)
    monkeypatch.setattr(deps, "format_resolved_binary", lambda *args: "[deps] test")
    captured: dict = {}
    monkeypatch.setattr(pipeline, "run_pipeline", lambda **kwargs: captured.update(kwargs))

    class _ImmediateThread:
        def __init__(self, *, target, daemon):
            self._target = target
            self._alive = False

        def start(self):
            self._alive = True
            try:
                self._target()
            finally:
                self._alive = False

        def is_alive(self):
            return self._alive

    monkeypatch.setattr(gui.threading, "Thread", _ImmediateThread)

    app._proceed_to_dialog_and_run()

    assert captured["io_idle_timeout_sec"] == 1800
    assert captured["chunk_seconds"] == 17
    assert captured["worker_count"] == 2
    assert captured["gpu_concurrency"] == 1


def test_run_rejects_unknown_framerate_before_dependencies_or_worker(
    app,
    monkeypatch,
    tmp_path,
):
    input_path = tmp_path / "unknown-rate.mp4"
    input_path.write_bytes(b"")
    app.var_input.set(str(input_path))
    app.var_output.set(str(tmp_path))
    meta = FakeMeta(duration_sec=10.0)
    meta.framerate_status = "unknown"
    meta.avg_fps = None
    app._meta = meta
    decs = _encode_only_decisions()
    app._decisions = decs
    monkeypatch.setattr(app, "_open_decision_dialog", lambda: decs)
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        gui.messagebox,
        "showerror",
        lambda title, message: errors.append((title, message)),
    )
    dependency_called = False

    def _unexpected_dependency():
        nonlocal dependency_called
        dependency_called = True
        raise AssertionError("dependencies must not be resolved")

    monkeypatch.setattr(deps, "check_ffmpeg", _unexpected_dependency)

    app._proceed_to_dialog_and_run()

    assert errors
    assert errors[0][0] == "フレームレート非対応"
    assert "framerate_preflight" in errors[0][1]
    assert dependency_called is False
    assert app._worker_thread is None


def test_auto_output_empty_before_analyze(app, tmp_path):
    # 未分析 (meta/decisions 未確定) では自動命名せず空欄を維持する
    app.var_input.set(str(Path(r"D:\elsewhere\clip.mp4")))
    app._maybe_refresh_auto_output()
    assert app.var_output.get() == ""


def test_auto_output_keeps_folder_field_empty_after_analyze(app, tmp_path):
    # 出力欄は保存先フォルダなので、分析後もファイル名では自動入力しない
    app.var_input.set(str(Path(r"D:\elsewhere\clip.mp4")))
    app._meta = _FakeMeta(width=1280, height=720, fps=Fraction(30, 1))
    app._decisions = _encode_only_decisions()
    app._maybe_refresh_auto_output()
    assert app.var_output.get() == ""


def test_selected_output_folder_is_not_overridden(app, tmp_path):
    app.var_input.set(str(Path(r"D:\elsewhere\clip.mp4")))
    app._meta = _FakeMeta(width=1280, height=720, fps=Fraction(30, 1))
    app._decisions = _encode_only_decisions()
    app._maybe_refresh_auto_output()
    selected = tmp_path / "selected"
    selected.mkdir()
    app.var_output.set(str(selected))
    app._maybe_refresh_auto_output()
    assert app.var_output.get() == str(selected)


def test_pick_output_accepts_outside_workspace(app, monkeypatch, tmp_path):
    outside = tmp_path.parent / "outside-output"
    outside.mkdir(exist_ok=True)
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(gui.filedialog, "askdirectory", lambda **_kw: str(outside))
    monkeypatch.setattr(
        gui.messagebox,
        "showerror",
        lambda title, message, **_kw: errors.append((title, message)),
    )

    app._pick_output()

    assert app.var_output.get() == str(outside.resolve())
    assert errors == []


def test_finalize_output_path_reflects_dialog_changes(app, tmp_path):
    # 出力フォルダ直下に、ダイアログで確定した実行スペック名で保存される
    app.var_input.set(str(Path(r"D:\elsewhere\clip.mp4")))
    app._meta = _FakeMeta(width=1920, height=1080, fps=Fraction(30, 1))
    app._decisions = _encode_only_decisions()
    selected = tmp_path / "selected"
    selected.mkdir()
    app.var_output.set(str(selected))

    # ダイアログで upscale(scale=4, target_width=3840) を ON にしたと仮定した dec_result
    dec_result = _encode_only_decisions() + [
        StageDecision(
            key="upscale",
            label_ja="アップスケール",
            recommended=True,
            confidence="high",
            reason="",
            enabled=True,
            params={"scale": 4, "target_width": 3840},
        )
    ]
    out = app._finalize_output_path(Path(r"D:\elsewhere\clip.mp4"), dec_result)
    assert out == selected / "clip_3840x2160-30p.mp4"
    assert app.var_output.get() == str(selected)


def test_finalize_output_path_uses_default_workspace_when_folder_empty(app, tmp_path):
    # 出力フォルダ空欄なら WORKSPACE_ROOT 直下に自動命名する
    app.var_input.set(str(Path(r"D:\elsewhere\clip.mp4")))
    app._meta = _FakeMeta(width=1920, height=1080, fps=Fraction(30, 1))
    app._decisions = _encode_only_decisions()
    out = app._finalize_output_path(Path(r"D:\elsewhere\clip.mp4"), _encode_only_decisions())
    assert out == tmp_path / "clip_1920x1080-30p.mp4"
    assert app.var_output.get() == ""


def test_invalidate_analysis_keeps_output_folder(app, tmp_path):
    # 入力/プリセット変更で分析キャッシュを破棄しても、保存先フォルダは保持する
    app.var_input.set(str(Path(r"D:\elsewhere\clip.mp4")))
    app._meta = _FakeMeta(width=1280, height=720, fps=Fraction(30, 1))
    app._decisions = _encode_only_decisions()
    selected = tmp_path / "selected"
    selected.mkdir()
    app.var_output.set(str(selected))

    app._invalidate_analysis()
    assert app.var_output.get() == str(selected)


def test_invalidate_analysis_keeps_manual_output(app, tmp_path):
    # 手入力した保存先フォルダは分析キャッシュ破棄でも消さない
    app.var_input.set(str(Path(r"D:\elsewhere\clip.mp4")))
    app._meta = _FakeMeta(width=1280, height=720, fps=Fraction(30, 1))
    app._decisions = _encode_only_decisions()
    manual = tmp_path / "keep"
    manual.mkdir()
    app.var_output.set(str(manual))
    assert app._auto_output_value is None

    app._invalidate_analysis()
    assert app.var_output.get() == str(manual)


# ---------- EnhanceDecisionDialog ----------


@dataclass
class FakeMeta:
    width: int = 1280
    height: int = 720
    fps: Fraction = Fraction(30, 1)
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


def _all_decs() -> list[StageDecision]:
    return decisions_mod.recommend(FakeMeta(field_order="tt", width=640), FakeStats())


def test_decision_dialog_builds_and_finalizes_with_recommended(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    dlg = gui.EnhanceDecisionDialog(tk_root, decs)
    # 推奨に応じて初期 enabled が反映されている
    for d in decs:
        if d.key == "encode":
            assert dlg.vars[d.key]["enabled"].get() is True
        else:
            assert dlg.vars[d.key]["enabled"].get() == d.recommended
    # OK 押下 → result = decisions
    dlg._on_ok()
    assert dlg.result is decs
    # encode は常時 ON にされる
    by = {d.key: d for d in dlg.result}
    assert by["encode"].enabled is True


def test_decision_dialog_cancel_returns_none(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    dlg = gui.EnhanceDecisionDialog(tk_root, decs)
    dlg._on_cancel()
    assert dlg.result is None


def test_decision_dialog_deselect_all_keeps_encode(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    dlg = gui.EnhanceDecisionDialog(tk_root, decs)
    dlg._deselect_all()
    for d in decs:
        if d.key == "encode":
            assert dlg.vars[d.key]["enabled"].get() is True
        else:
            assert dlg.vars[d.key]["enabled"].get() is False


def test_decision_dialog_select_recommended_restores(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    dlg = gui.EnhanceDecisionDialog(tk_root, decs)
    dlg._deselect_all()
    dlg._select_recommended()
    for d in decs:
        expected = d.recommended or d.key == "encode"
        assert dlg.vars[d.key]["enabled"].get() == expected


def test_decision_dialog_user_toggle_propagates_to_finalize(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    dlg = gui.EnhanceDecisionDialog(tk_root, decs)
    # 全部 OFF にしてから denoise だけ ON にして finalize
    dlg._deselect_all()
    dlg.vars["denoise"]["enabled"].set(True)
    out = dlg._finalize()
    by = {d.key: d for d in out}
    assert by["denoise"].enabled is True
    assert by["encode"].enabled is True
    # 他の stage は OFF
    for k in (
        "stabilize",
        "deflicker",
        "debanding",
        "color_correction",
        "sharpen",
        "upscale",
        "face_restore",
        "interpolate",
        "grain",
    ):
        assert by[k].enabled is False


def test_decision_dialog_encode_param_override(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    dlg = gui.EnhanceDecisionDialog(tk_root, decs)
    dlg.vars["encode"]["params"]["crf"].set(22)
    dlg.vars["encode"]["params"]["preset"].set("medium")
    out = dlg._finalize()
    by = {d.key: d for d in out}
    assert by["encode"].params["crf"] == 22
    assert by["encode"].params["preset"] == "medium"


def test_decision_dialog_face_restore_range_defaults_to_none(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    dlg = gui.EnhanceDecisionDialog(tk_root, decs, meta=FakeMeta(duration_sec=60.0))
    dlg.vars["face_restore"]["enabled"].set(True)
    dlg.vars["face_restore"]["params"]["_range_limited"].set(False)

    out = dlg._finalize()

    face = {d.key: d for d in out}["face_restore"]
    assert face.enabled is True
    assert face.params["start_sec"] is None
    assert face.params["end_sec"] is None


def test_decision_dialog_face_restore_range_values_finalize(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    dlg = gui.EnhanceDecisionDialog(tk_root, decs, meta=FakeMeta(duration_sec=60.0))
    dlg.vars["face_restore"]["enabled"].set(True)
    params = dlg.vars["face_restore"]["params"]
    params["_range_limited"].set(True)
    params["start_sec"].set("30")
    params["end_sec"].set("40.5")

    out = dlg._finalize()

    face = {d.key: d for d in out}["face_restore"]
    assert face.params["start_sec"] == 30.0
    assert face.params["end_sec"] == 40.5


def test_decision_dialog_face_restore_naturalize_values_finalize(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    dlg = gui.EnhanceDecisionDialog(tk_root, decs, meta=FakeMeta(duration_sec=60.0))
    dlg.vars["face_restore"]["enabled"].set(True)
    params = dlg.vars["face_restore"]["params"]
    params["blend"].set(0.3)
    params["mask"].set("diff")
    params["match_grain"].set(0.2)
    params["soften"].set(0.15)
    params["global_smooth"].set(0.1)

    out = dlg._finalize()

    face = {d.key: d for d in out}["face_restore"]
    assert face.params["blend"] == 0.3
    assert face.params["mask"] == "diff"
    assert face.params["match_grain"] == 0.2
    assert face.params["soften"] == 0.15
    assert face.params["global_smooth"] == 0.1
    assert face.params["feather"] == config.DEFAULT_FACE_RESTORE_FEATHER


def test_decision_dialog_face_restore_naturalize_descriptions_are_visible(
    tk_root,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    dlg = gui.EnhanceDecisionDialog(tk_root, decs, meta=FakeMeta(duration_sec=60.0))

    def _label_texts(widget):
        texts = []
        for child in widget.winfo_children():
            try:
                text = child.cget("text")
            except Exception:
                text = ""
            if text:
                texts.append(str(text))
            texts.extend(_label_texts(child))
        return texts

    labels = "\n".join(_label_texts(dlg.top))
    assert "CodeFormer 出力の混合率" in labels
    assert "diff は変化した顔周辺だけを合成" in labels
    assert "元映像の粒状感を戻し" in labels
    assert "復元顔を軽くぼかし" in labels
    assert "顔と背景の質感差を減らす" in labels


def test_decision_dialog_face_restore_rejects_invalid_naturalize_value(
    tk_root, monkeypatch, tmp_path
):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    dlg = gui.EnhanceDecisionDialog(tk_root, decs, meta=FakeMeta(duration_sec=60.0))
    params = dlg.vars["face_restore"]["params"]
    params["blend"].set(1.5)

    with pytest.raises(ValueError):
        dlg._finalize()


def test_decision_dialog_face_restore_rejects_invalid_range(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    dlg = gui.EnhanceDecisionDialog(tk_root, decs, meta=FakeMeta(duration_sec=60.0))
    params = dlg.vars["face_restore"]["params"]
    params["_range_limited"].set(True)
    params["start_sec"].set("40")
    params["end_sec"].set("30")

    with pytest.raises(ValueError):
        dlg._finalize()


def test_decision_dialog_face_restore_rejects_duration_overflow(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    dlg = gui.EnhanceDecisionDialog(tk_root, decs, meta=FakeMeta(duration_sec=60.0))
    params = dlg.vars["face_restore"]["params"]
    params["_range_limited"].set(True)
    params["start_sec"].set("10")
    params["end_sec"].set("61")

    with pytest.raises(ValueError):
        dlg._finalize()


# ---------- summary preview ----------


def test_decision_dialog_summary_shows_placeholder_when_meta_none(tk_root, monkeypatch, tmp_path):
    """meta 未指定 (テスト / 旧呼び出し互換) ならサマリーはプレースホルダのまま."""
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    dlg = gui.EnhanceDecisionDialog(tk_root, decs)  # meta=None
    assert dlg._summary_resolution_var is not None
    assert dlg._summary_resolution_var.get() == "※分析未実施"
    assert dlg._summary_fps_var.get() == "※分析未実施"


def test_decision_dialog_summary_initial_render_with_meta(tk_root, monkeypatch, tmp_path):
    """meta を渡すと初期サマリーが描画される (推奨どおりの enabled に基づく)."""
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    meta = FakeMeta(width=1280, height=720, fps=Fraction(30, 1))
    decs = decisions_mod.recommend(meta, FakeStats())
    dlg = gui.EnhanceDecisionDialog(tk_root, decs, meta=meta)
    res = dlg._summary_resolution_var.get()
    fps = dlg._summary_fps_var.get()
    stages = dlg._summary_stages_var.get()
    # 1280x720 < 1920 (DEFAULT_TARGET_WIDTH) → upscale ON → 1920x1080 が出る
    assert "1280x720" in res
    assert "1920x1080" in res
    assert "Real-ESRGAN" in res
    # 30fps == T_FPS_LOW=30 なので Auto は補間 OFF → fps は変更なし表示
    assert "30.000fps" in fps
    assert "OFF" in fps or "変更なし" in fps
    # encode は必ず active
    assert "encode" in stages


def test_decision_dialog_summary_updates_when_toggling_upscale(tk_root, monkeypatch, tmp_path):
    """upscale チェックを OFF にするとサマリーの解像度が source 維持に切り替わる."""
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    meta = FakeMeta(width=1280, height=720, fps=Fraction(30, 1))
    decs = decisions_mod.recommend(meta, FakeStats())
    dlg = gui.EnhanceDecisionDialog(tk_root, decs, meta=meta)
    # 初期は upscale ON → 1920x1080
    assert "1920x1080" in dlg._summary_resolution_var.get()
    # upscale チェックを OFF にする → trace でサマリー更新
    dlg.vars["upscale"]["enabled"].set(False)
    res = dlg._summary_resolution_var.get()
    assert "1280x720" in res
    assert "upscale OFF" in res or "変更なし" in res
    # 元の decisions は破壊されていない (OK 押下前)
    by = {d.key: d for d in dlg.decisions}
    assert by["upscale"].enabled is True


def test_decision_dialog_summary_updates_when_toggling_interpolate(tk_root, monkeypatch, tmp_path):
    """interpolate を手動で ON にすると fps プレビューが補間後の値に変わる."""
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    # 60fps → Auto では interpolate OFF
    meta = FakeMeta(width=1920, height=1080, fps=Fraction(60, 1))
    decs = decisions_mod.recommend(meta, FakeStats())
    dlg = gui.EnhanceDecisionDialog(tk_root, decs, meta=meta)
    # 初期: interpolate OFF
    assert "OFF" in dlg._summary_fps_var.get() or "変更なし" in dlg._summary_fps_var.get()
    # 手動で interpolate ON
    dlg.vars["interpolate"]["enabled"].set(True)
    fps = dlg._summary_fps_var.get()
    # 60 → 120 (×2)
    assert "120.000fps" in fps
    assert "×2" in fps or "RIFE" in fps


def test_decision_dialog_summary_stage_list_reflects_toggles(tk_root, monkeypatch, tmp_path):
    """全部 OFF にすると active stages は encode のみになる."""
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    meta = FakeMeta(width=1280, height=720, fps=Fraction(30, 1), field_order="tt")
    decs = decisions_mod.recommend(meta, FakeStats())
    dlg = gui.EnhanceDecisionDialog(tk_root, decs, meta=meta)
    dlg._deselect_all()
    stages = dlg._summary_stages_var.get()
    assert "encode" in stages
    # deinterlace は recommend で ON だったが OFF にしたので消える
    assert "deinterlace" not in stages


def test_decision_dialog_summary_shows_face_restore_range(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    meta = FakeMeta(width=1280, height=720, fps=Fraction(30, 1), duration_sec=60.0)
    decs = decisions_mod.recommend(meta, FakeStats())
    dlg = gui.EnhanceDecisionDialog(tk_root, decs, meta=meta)
    dlg.vars["face_restore"]["enabled"].set(True)
    params = dlg.vars["face_restore"]["params"]
    params["_range_limited"].set(True)
    params["start_sec"].set("30")
    params["end_sec"].set("40")
    dlg._refresh_summary()

    stages = dlg._summary_stages_var.get()
    assert "face_restore: 30s-40s" in stages


def test_decision_dialog_summary_shows_face_restore_naturalize(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    meta = FakeMeta(width=1280, height=720, fps=Fraction(30, 1), duration_sec=60.0)
    decs = decisions_mod.recommend(meta, FakeStats())
    dlg = gui.EnhanceDecisionDialog(tk_root, decs, meta=meta)
    dlg.vars["face_restore"]["enabled"].set(True)
    params = dlg.vars["face_restore"]["params"]
    params["blend"].set(0.3)
    params["mask"].set("diff")
    params["match_grain"].set(0.2)
    dlg._refresh_summary()

    stages = dlg._summary_stages_var.get()
    assert "blend 0.3" in stages
    assert "mask diff" in stages
    assert "grain 0.2" in stages


def test_decision_dialog_face_restore_candidate_applies_selected_range_without_enabling(
    tk_root,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    meta = FakeMeta(width=1280, height=720, fps=Fraction(30, 1), duration_sec=60.0)
    stats = FakeStats()
    first = analyzer.FaceRestoreCandidateRange(
        start_sec=12.0,
        end_sec=18.5,
        sample_count=1,
        face_sample_count=1,
        max_face_ratio=0.03,
        confidence="medium",
        note="test",
    )
    second = analyzer.FaceRestoreCandidateRange(
        start_sec=42.0,
        end_sec=48.0,
        sample_count=1,
        face_sample_count=1,
        max_face_ratio=0.01,
        confidence="low",
        note="test",
    )
    stats.face_restore_candidates = (first, second)
    decs = decisions_mod.recommend(meta, stats)
    dlg = gui.EnhanceDecisionDialog(tk_root, decs, meta=meta)

    assert dlg.vars["face_restore"]["enabled"].get() is False
    assert dlg._face_candidate_choices["face_restore"] == (first, second)
    dlg._face_candidate_choice_var["face_restore"].set(
        gui.EnhanceDecisionDialog._format_face_restore_candidate_choice(second)
    )
    dlg._apply_face_restore_candidate("face_restore")

    params = dlg.vars["face_restore"]["params"]
    assert params["_range_limited"].get() is True
    assert params["start_sec"].get() == "42"
    assert params["end_sec"].get() == "48"
    assert dlg.vars["face_restore"]["enabled"].get() is False


def test_decision_dialog_select_recommended_keeps_face_restore_off(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    decs = _all_decs()
    dlg = gui.EnhanceDecisionDialog(tk_root, decs)
    dlg.vars["face_restore"]["enabled"].set(True)
    dlg._select_recommended()
    assert dlg.vars["face_restore"]["enabled"].get() is False


# ---------- analysis cache invalidation (P1-4 / P2-1) ----------


def test_input_change_invalidates_analysis_cache(app, tmp_path):
    """P1-4 回帰: 入力ファイルを変更すると古い meta/decisions が破棄される。"""
    app._meta = FakeMeta()
    app._decisions = _all_decs()
    app._runtime_estimate_sec = 123.0
    app.var_input.set(str(tmp_path / "another.mp4"))
    assert app._meta is None
    assert app._decisions is None
    assert app._runtime_estimate_sec is None
    assert app.var_fps_preview.get() == "※分析後に表示"


def test_preset_change_invalidates_analysis_cache(app):
    """P2-1 回帰: プリセット変更でも分析キャッシュが破棄される。"""
    app._meta = FakeMeta()
    app._decisions = _all_decs()
    app._on_preset_changed()
    assert app._meta is None
    assert app._decisions is None


def test_invalidation_bumps_generation(app, tmp_path):
    """世代カウンタが進む (分析ワーカーの破棄判定に使われる)。"""
    g0 = app._analysis_generation
    app.var_input.set(str(tmp_path / "a.mp4"))
    assert app._analysis_generation == g0 + 1
    app._on_preset_changed()
    assert app._analysis_generation == g0 + 2


def test_run_after_invalidation_restarts_analyze(app, monkeypatch, tmp_path):
    """破棄後の「実行」は analyze からやり直すフローに乗る。"""
    input_path = tmp_path / "in.mp4"
    input_path.write_bytes(b"")
    # 事前分析済み状態を作ってから入力を変更 (キャッシュ破棄)
    app._meta = FakeMeta()
    app._decisions = _all_decs()
    app.var_input.set(str(input_path))
    app._maybe_refresh_auto_output()

    called: dict = {"proceed": 0, "start_analyze": 0}
    monkeypatch.setattr(
        app,
        "_proceed_to_dialog_and_run",
        lambda: called.__setitem__("proceed", called["proceed"] + 1),
    )
    monkeypatch.setattr(
        app,
        "_start_analyze_worker",
        lambda _inp, _d: called.__setitem__("start_analyze", called["start_analyze"] + 1),
    )

    app._on_run()
    assert called["start_analyze"] == 1
    assert called["proceed"] == 0
    assert app._pending_run_after_analyze is True


# ---------- _on_run analyze-then-run flow ----------


def test_on_run_without_prior_analyze_sets_pending_and_starts_analyze(app, monkeypatch, tmp_path):
    """plan: 「実行」直接押下 → 内部で analyze → dialog → 実行 を連続実行."""
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    input_path = tmp_path / "in.mp4"
    input_path.write_bytes(b"")
    app.var_input.set(str(input_path))
    app._maybe_refresh_auto_output()

    started: dict = {}

    def _fake_start(inp, defaults):
        started["input"] = inp
        started["defaults"] = defaults

    monkeypatch.setattr(app, "_start_analyze_worker", _fake_start)
    # decisions/meta は未設定 (事前分析未実施)
    assert app._decisions is None
    assert app._meta is None

    app._on_run()

    # analyze worker が起動され、pending フラグが立っている
    assert started.get("input") == input_path
    assert app._pending_run_after_analyze is True


def test_on_run_with_prior_analyze_skips_analyze_and_calls_dialog(app, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    input_path = tmp_path / "in.mp4"
    input_path.write_bytes(b"")
    app.var_input.set(str(input_path))
    app._maybe_refresh_auto_output()

    # 事前分析済みをシミュレート
    app._meta = FakeMeta()
    app._decisions = _all_decs()

    called: dict = {"proceed": 0, "start_analyze": 0}

    def _fake_proceed():
        called["proceed"] += 1

    def _fake_start(_inp, _defaults):
        called["start_analyze"] += 1

    monkeypatch.setattr(app, "_proceed_to_dialog_and_run", _fake_proceed)
    monkeypatch.setattr(app, "_start_analyze_worker", _fake_start)

    app._on_run()

    assert called["proceed"] == 1
    assert called["start_analyze"] == 0
    assert app._pending_run_after_analyze is False


def test_handle_finish_chains_to_dialog_when_pending(app, monkeypatch):
    """analyze worker が success で終わり、pending_run_after_analyze=True なら
    _proceed_to_dialog_and_run が呼ばれる."""
    proceed_called: dict = {"count": 0}

    def _fake_proceed():
        proceed_called["count"] += 1

    monkeypatch.setattr(app, "_proceed_to_dialog_and_run", _fake_proceed)

    # after() は同期的に実行されるよう差し替え
    def _immediate_after(_ms, fn=None, *args):
        if fn is not None:
            fn(*args)

    monkeypatch.setattr(app.root, "after", _immediate_after)

    app._current_worker_kind = "analyze"
    app._pending_run_after_analyze = True

    app._handle_finish(gui._FinishEvent("success"))

    assert proceed_called["count"] == 1
    assert app._pending_run_after_analyze is False


def test_handle_finish_no_chain_when_analyze_failed(app, monkeypatch):
    proceed_called: dict = {"count": 0}
    monkeypatch.setattr(
        app,
        "_proceed_to_dialog_and_run",
        lambda: proceed_called.__setitem__("count", proceed_called["count"] + 1),
    )
    app._current_worker_kind = "analyze"
    app._pending_run_after_analyze = True
    app._handle_finish(gui._FinishEvent("failed", message="boom"))
    assert proceed_called["count"] == 0
    # 失敗時は pending フラグもクリアされる
    assert app._pending_run_after_analyze is False


def test_handle_finish_no_chain_when_not_pending(app, monkeypatch):
    proceed_called: dict = {"count": 0}
    monkeypatch.setattr(
        app,
        "_proceed_to_dialog_and_run",
        lambda: proceed_called.__setitem__("count", proceed_called["count"] + 1),
    )
    app._current_worker_kind = "analyze"
    app._pending_run_after_analyze = False  # 通常の「事前分析」ボタン経路
    app._handle_finish(gui._FinishEvent("success"))
    assert proceed_called["count"] == 0
