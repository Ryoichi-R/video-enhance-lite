from __future__ import annotations

import math
import os
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Literal

from . import (
    analyzer,
    config,
    decisions as decisions_mod,
    deps,
    pipeline,
    runtime_estimator,
    temp_workspace,
)
from .decisions import (
    OutputSummary,
    StageDecision,
    compute_output_summary,
    decisions_by_key,
    format_output_spec_suffix,
)
from .deps import MissingDependencyError
from .runtime_estimator import format_eta_hms


_VIDEO_FILETYPES = [
    ("Video files", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v"),
    ("All files", "*.*"),
]

# 解像度プリセット (表示ラベル -> target_width)。
#   None は Auto (入力 < 1920 なら 1920 へ拡大)。
#   RESOLUTION_KEEP_SOURCE は「現状維持」(入力解像度のまま; upscale しない)。
_RESOLUTION_PRESET_LABELS: dict[str, int | None] = {
    "自動": None,
    "現状維持": config.RESOLUTION_KEEP_SOURCE,
    "Full HD (1920)": 1920,
    "WQHD (2560)": 2560,
    "4K (3840)": 3840,
}
# FPS プリセット (表示ラベル -> target_fps)。
#   None は Auto (source < 30fps なら ×2 補間を推奨)。
#   FPS_KEEP_SOURCE は「現状維持」(入力 fps のまま; フレーム補間しない)。
_FPS_PRESET_LABELS: dict[str, float | None] = {
    "自動": None,
    "現状維持": config.FPS_KEEP_SOURCE,
    "48 fps": 48.0,
    "60 fps": 60.0,
    "120 fps": 120.0,
}


@dataclass(frozen=True)
class _FinishEvent:
    status: Literal["success", "failed", "cancelled"]
    message: str | None = None


def _parse_chunk_seconds(raw: str) -> int:
    s = (raw or "").strip()
    if not s:
        raise ValueError("分割秒数を入力してください。")
    try:
        n = int(s)
    except ValueError:
        raise ValueError(f"分割秒数は整数で入力してください: {raw!r}")
    if n < 1:
        raise ValueError(f"分割秒数は 1 以上にしてください: {n}")
    return n


def _parse_optional_positive_int(raw: str, *, label: str) -> int | None:
    """空欄なら None (= auto)、それ以外は 1 以上の整数を返す。"""
    s = (raw or "").strip()
    if not s:
        return None
    try:
        n = int(s)
    except ValueError:
        raise ValueError(f"{label}は整数で入力してください: {raw!r}")
    if n < 1:
        raise ValueError(f"{label}は 1 以上にしてください: {n}")
    return n


def _parse_positive_int(raw: str, *, label: str) -> int:
    """空欄を許可せず、1以上の整数を返す。"""
    value = _parse_optional_positive_int(raw, label=label)
    if value is None:
        raise ValueError(f"{label}を入力してください。")
    return value


def _format_framerate_warning(meta: object) -> str | None:
    status = getattr(meta, "framerate_status", "unknown")
    if status == "cfr":
        return None
    avg_fps = getattr(meta, "avg_fps", None)
    rate_detail = f"r={getattr(meta, 'fps', 'unknown')}, avg={avg_fps or 'unknown'}"
    if status == "vfr_suspected":
        return (
            f"VFRの可能性があります ({rate_detail})。PNG抽出前にnominal r_frame_rateへ"
            "明示CFR normalizationします（frame duplication/dropを伴う場合があります）"
        )
    return (
        f"フレームレートを判定できませんでした ({rate_detail})。"
        "安全にCFR化できないため処理は開始できません"
    )


def _parse_optional_nonnegative_float(
    raw: str,
    *,
    label: str,
    max_value: float | None = None,
) -> float | None:
    """空欄なら None、それ以外は 0 以上の有限小数を返す。"""
    s = (raw or "").strip()
    if not s:
        return None
    try:
        value = float(s)
    except ValueError:
        raise ValueError(f"{label}は秒数を数値で入力してください: {raw!r}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label}は 0 以上の有限値にしてください: {raw!r}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{label}は動画長 {max_value:.3f} 秒以内にしてください: {value:g}")
    return value


def _parse_unit_float(raw: str, *, label: str) -> float:
    s = (raw or "").strip()
    try:
        value = float(s)
    except ValueError:
        raise ValueError(f"{label}は 0..1 の数値で入力してください: {raw!r}")
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        raise ValueError(f"{label}は 0..1 の有限値にしてください: {raw!r}")
    return value


def _attach_tooltip(widget: tk.Widget, text: str) -> None:
    """ホバーで text を表示する軽量ツールチップ (依存追加なしの最小実装)。"""
    state: dict[str, Any] = {"tip": None}

    def _show(_event: object = None) -> None:
        if state["tip"] is not None:
            return
        try:
            x = widget.winfo_rootx() + 12
            y = widget.winfo_rooty() + widget.winfo_height() + 4
        except tk.TclError:
            return
        tip = tk.Toplevel(widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        ttk.Label(
            tip,
            text=text,
            justify="left",
            relief="solid",
            borderwidth=1,
            background="#ffffe0",
            foreground="#222",
            padding=(6, 3),
        ).pack()
        state["tip"] = tip

    def _hide(_event: object = None) -> None:
        tip = state["tip"]
        if tip is not None:
            tip.destroy()
            state["tip"] = None

    widget.bind("<Enter>", _show)
    widget.bind("<Leave>", _hide)
    widget.bind("<Destroy>", _hide)


def derive_output_path_for_gui(
    input_path: Path,
    summary: OutputSummary,
    output_dir: Path | None = None,
) -> Path:
    """GUI の最終出力パスを組み立てる。

    末尾は「実行スペック」`_{幅}x{高}-{fps}p` で命名する (例: `clip_1920x1080-60p.mp4`)。
    `output_dir` 未指定時はユーザーの Videos 配下へ保存する。
    """
    suffix = format_output_spec_suffix(summary)
    name = f"{input_path.stem}{suffix}{input_path.suffix}"
    return config.make_unique_output_path((output_dir or config.get_default_output_root()) / name)


def _validate_output_dir_for_gui(path: Path) -> Path:
    output_dir = config.validate_output_path(path)
    if not output_dir.is_dir():
        raise ValueError(f"出力フォルダが見つかりません: {output_dir}")
    return output_dir


def _show_output_dir_error(message: str) -> None:
    messagebox.showerror(
        "設定エラー",
        f"{message}\n\n出力フォルダには既存のフォルダを指定してください。",
    )


class _EventQueue:
    def __init__(self) -> None:
        self._q: queue.Queue[str | pipeline.ProgressEvent | _FinishEvent] = queue.Queue()

    def write(self, msg: str) -> None:
        self._q.put(str(msg))

    def progress(self, ev: pipeline.ProgressEvent) -> None:
        self._q.put(ev)

    def finish(self, ev: _FinishEvent) -> None:
        self._q.put(ev)

    def drain(self) -> list[str | pipeline.ProgressEvent | _FinishEvent]:
        items: list[str | pipeline.ProgressEvent | _FinishEvent] = []
        while True:
            try:
                items.append(self._q.get_nowait())
            except queue.Empty:
                break
        return items


class EnhanceDecisionDialog:
    """Modal-ish Toplevel that lets the user toggle each StageDecision.

    Returns the (mutated) decisions list on OK, or None on Cancel.
    For testability, `result` and `_finalize`/`_on_cancel` are public-ish.
    `meta` (analyzer.VideoMetadata) を渡すとヘッダ直下に「最終解像度/FPS/有効ステージ」の
    サマリーが表示され、チェックボックス変更で自動的に更新される。
    """

    def __init__(
        self,
        parent: tk.Misc,
        decisions: list[StageDecision],
        *,
        meta: "analyzer.VideoMetadata | None" = None,
        chunk_seconds: int | None = None,
        max_temp_gib: float = config.DEFAULT_MAX_TEMP_GIB,
        worker_count: int = 1,
    ):
        self.parent = parent
        self.decisions = decisions
        self.meta = meta
        self._chunk_seconds = chunk_seconds or config.DEFAULT_CHUNK_SECONDS
        self._max_temp_gib = float(max_temp_gib)
        self._worker_count = max(1, int(worker_count))
        self.result: list[StageDecision] | None = None
        self.top = tk.Toplevel(parent)
        self.top.title("事前分析結果 — ステージ選択")
        self.top.transient(parent)
        self.top.geometry("780x620")
        # サマリー表示用 (meta が None なら _build_summary 内で「※分析未実施」表示にフォールバック)
        self._summary_resolution_var: tk.StringVar | None = None
        self._summary_fps_var: tk.StringVar | None = None
        self._summary_stages_var: tk.StringVar | None = None
        self._summary_runtime_var: tk.StringVar | None = None
        self._summary_capacity_var: tk.StringVar | None = None
        self._summary_capacity_label: ttk.Label | None = None
        self._face_candidate_choices: dict[str, tuple[Any, ...]] = {}
        self._face_candidate_choice_var: dict[str, tk.StringVar] = {}
        self._build()

    def _face_restore_max_sec(self) -> float | None:
        if self.meta is None:
            return None
        duration = float(getattr(self.meta, "duration_sec", 0.0) or 0.0)
        if not math.isfinite(duration) or duration <= 0:
            return None
        return duration

    @staticmethod
    def _format_candidate_seconds(value: object) -> str:
        return f"{float(value):.1f}s"

    @classmethod
    def _format_face_restore_candidate_choice(cls, candidate: Any) -> str:
        start = cls._format_candidate_seconds(getattr(candidate, "start_sec"))
        end = cls._format_candidate_seconds(getattr(candidate, "end_sec"))
        conf = str(getattr(candidate, "confidence", "unknown"))
        return f"{start}-{end} ({conf})"

    @classmethod
    def _format_face_restore_candidate_summary(
        cls,
        candidates: tuple[Any, ...],
        *,
        limit: int = 5,
    ) -> str:
        parts = [cls._format_face_restore_candidate_choice(c) for c in candidates[:limit]]
        if len(candidates) > limit:
            parts.append(f"... (+{len(candidates) - limit})")
        return ", ".join(parts)

    def _selected_face_restore_candidate(self, stage_key: str) -> Any | None:
        candidates = self._face_candidate_choices.get(stage_key, ())
        if not candidates:
            return None
        choice_var = self._face_candidate_choice_var.get(stage_key)
        if choice_var is None:
            return candidates[0]
        selected = choice_var.get()
        for candidate in candidates:
            if self._format_face_restore_candidate_choice(candidate) == selected:
                return candidate
        return candidates[0]

    def _apply_face_restore_candidate(self, stage_key: str = "face_restore") -> None:
        candidate = self._selected_face_restore_candidate(stage_key)
        if candidate is None:
            return
        values = self.vars.get(stage_key, {}).get("params", {})
        if not values:
            return
        values["_range_limited"].set(True)
        values["start_sec"].set(
            f"{float(getattr(candidate, 'start_sec')):.3f}".rstrip("0").rstrip(".")
        )
        values["end_sec"].set(f"{float(getattr(candidate, 'end_sec')):.3f}".rstrip("0").rstrip("."))
        self._refresh_summary()

    def _build(self) -> None:
        frm = ttk.Frame(self.top)
        frm.pack(fill="both", expand=True, padx=10, pady=8)

        header = ttk.Label(
            frm,
            text="チェック ON のステージを処理します。推奨/不要/判定不能 と理由を確認してください。",
            foreground="#444",
        )
        header.pack(anchor="w", pady=(0, 6))

        # サマリーフレーム (チェックボックス変更で自動更新)
        self._build_summary(frm)

        canvas = tk.Canvas(frm, highlightthickness=0)
        sb = ttk.Scrollbar(frm, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)

        # 責務を分離: inner の <Configure> はスクロール領域更新のみ。
        # canvas の <Configure> で inner_id の幅を canvas の実幅 (e.width) に追従させる。
        # こうしないと inner.<Configure> 内で canvas.winfo_width() を読んだ際、
        # canvas がまだ geometry manager に引き伸ばされておらず 1 を返すレースが起き、
        # inner が幅 1 px に潰れて中身が見えなくなる。
        def _on_inner_configure(_e):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(e):
            canvas.itemconfigure(inner_id, width=e.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.vars: dict[str, dict[str, Any]] = {}
        for d in self.decisions:
            row = ttk.Frame(inner)
            row.pack(fill="x", pady=2)
            var_enabled = tk.BooleanVar(value=d.enabled)
            var_enabled.trace_add("write", lambda *_a: self._refresh_summary())
            # encode は常時 ON
            state = "disabled" if d.key == "encode" else "normal"
            cb = ttk.Checkbutton(
                row, text=f"{d.label_ja} ({d.key})", variable=var_enabled, state=state
            )
            cb.grid(row=0, column=0, sticky="w", padx=(0, 8))
            badge_text = self._badge(d)
            ttk.Label(row, text=badge_text, foreground=self._badge_color(d)).grid(
                row=0, column=1, sticky="w", padx=(0, 8)
            )
            ttk.Label(row, text=d.reason, foreground="#666", wraplength=520).grid(
                row=1, column=0, columnspan=3, sticky="w", padx=(20, 0)
            )

            params_vars: dict[str, tk.Variable] = {}
            # 一部 stage は param を編集可能に
            if d.key == "encode":
                ttk.Label(row, text="CRF").grid(row=0, column=2, sticky="e")
                v_crf = tk.IntVar(value=int(d.params.get("crf", config.DEFAULT_ENCODE_CRF)))
                ttk.Spinbox(row, from_=0, to=51, textvariable=v_crf, width=5).grid(
                    row=0, column=3, sticky="w", padx=(2, 8)
                )
                ttk.Label(row, text="preset").grid(row=0, column=4, sticky="e")
                v_pre = tk.StringVar(
                    value=str(d.params.get("preset", config.DEFAULT_ENCODE_PRESET))
                )
                ttk.Combobox(
                    row,
                    textvariable=v_pre,
                    width=10,
                    state="readonly",
                    values=["ultrafast", "veryfast", "fast", "medium", "slow", "veryslow"],
                ).grid(row=0, column=5, sticky="w")
                params_vars["crf"] = v_crf
                params_vars["preset"] = v_pre
            elif d.key == "sharpen":
                v = tk.DoubleVar(value=float(d.params.get("amount", config.DEFAULT_SHARPEN_AMOUNT)))
                ttk.Label(row, text="amount").grid(row=0, column=2, sticky="e")
                ttk.Spinbox(row, from_=0.0, to=2.0, increment=0.1, textvariable=v, width=6).grid(
                    row=0, column=3, sticky="w"
                )
                params_vars["amount"] = v
            elif d.key == "grain":
                v = tk.IntVar(value=int(d.params.get("amount", config.DEFAULT_GRAIN_AMOUNT)))
                ttk.Label(row, text="amount").grid(row=0, column=2, sticky="e")
                ttk.Spinbox(row, from_=0, to=64, textvariable=v, width=5).grid(
                    row=0, column=3, sticky="w"
                )
                params_vars["amount"] = v
            elif d.key == "face_restore":
                start_default = d.params.get("start_sec")
                end_default = d.params.get("end_sec")
                limit_default = start_default is not None or end_default is not None
                v_limit = tk.BooleanVar(value=limit_default)
                v_start = tk.StringVar(value="" if start_default is None else str(start_default))
                v_end = tk.StringVar(value="" if end_default is None else str(end_default))
                max_sec = self._face_restore_max_sec()
                spin_to = max_sec if max_sec is not None else 86400.0

                limit_cb = ttk.Checkbutton(row, text="範囲を限定", variable=v_limit)
                limit_cb.grid(row=2, column=0, sticky="w", padx=(20, 8), pady=(3, 0))
                ttk.Label(row, text="start").grid(row=2, column=1, sticky="e", pady=(3, 0))
                start_spin = ttk.Spinbox(
                    row,
                    from_=0.0,
                    to=spin_to,
                    increment=0.1,
                    textvariable=v_start,
                    width=8,
                )
                start_spin.grid(row=2, column=2, sticky="w", padx=(2, 8), pady=(3, 0))
                ttk.Label(row, text="end").grid(row=2, column=3, sticky="e", pady=(3, 0))
                end_spin = ttk.Spinbox(
                    row,
                    from_=0.0,
                    to=spin_to,
                    increment=0.1,
                    textvariable=v_end,
                    width=8,
                )
                end_spin.grid(row=2, column=4, sticky="w", padx=(2, 8), pady=(3, 0))
                ttk.Label(row, text="秒").grid(row=2, column=5, sticky="w", pady=(3, 0))

                v_blend = tk.DoubleVar(
                    value=float(d.params.get("blend", config.DEFAULT_FACE_RESTORE_BLEND))
                )
                v_mask = tk.StringVar(
                    value=str(d.params.get("mask", config.DEFAULT_FACE_RESTORE_MASK))
                )
                v_grain = tk.DoubleVar(
                    value=float(
                        d.params.get("match_grain", config.DEFAULT_FACE_RESTORE_MATCH_GRAIN)
                    )
                )
                v_soften = tk.DoubleVar(
                    value=float(d.params.get("soften", config.DEFAULT_FACE_RESTORE_SOFTEN))
                )
                v_smooth = tk.DoubleVar(
                    value=float(
                        d.params.get("global_smooth", config.DEFAULT_FACE_RESTORE_GLOBAL_SMOOTH)
                    )
                )

                def _naturalize_row(
                    row_index: int,
                    label: str,
                    widget: tk.Widget,
                    description: str,
                ) -> None:
                    ttk.Label(row, text=label).grid(
                        row=row_index,
                        column=0,
                        sticky="e",
                        padx=(42, 6),
                        pady=(3, 0),
                    )
                    widget.grid(row=row_index, column=1, sticky="w", padx=(2, 8), pady=(3, 0))
                    ttk.Label(
                        row,
                        text=description,
                        foreground="#666",
                        wraplength=720,
                        justify="left",
                    ).grid(row=row_index, column=2, columnspan=8, sticky="w", pady=(3, 0))

                _naturalize_row(
                    3,
                    "blend",
                    ttk.Spinbox(
                        row,
                        from_=0.0,
                        to=1.0,
                        increment=0.05,
                        textvariable=v_blend,
                        width=5,
                    ),
                    "CodeFormer 出力の混合率。低いほど元映像寄り。",
                )
                _naturalize_row(
                    4,
                    "mask",
                    ttk.Combobox(
                        row,
                        textvariable=v_mask,
                        width=6,
                        state="readonly",
                        values=list(config.FACE_RESTORE_MASK_MODES),
                    ),
                    "none は全体を混ぜる。diff は変化した顔周辺だけを合成。",
                )
                _naturalize_row(
                    5,
                    "grain",
                    ttk.Spinbox(
                        row,
                        from_=0.0,
                        to=1.0,
                        increment=0.05,
                        textvariable=v_grain,
                        width=5,
                    ),
                    "元映像の粒状感を戻し、顔だけ滑らかになるのを抑える。",
                )
                _naturalize_row(
                    6,
                    "soften",
                    ttk.Spinbox(
                        row,
                        from_=0.0,
                        to=1.0,
                        increment=0.05,
                        textvariable=v_soften,
                        width=5,
                    ),
                    "復元顔を軽くぼかし、古い映像や遠景になじませる。",
                )
                _naturalize_row(
                    7,
                    "smooth",
                    ttk.Spinbox(
                        row,
                        from_=0.0,
                        to=1.0,
                        increment=0.05,
                        textvariable=v_smooth,
                        width=5,
                    ),
                    "全体を弱く平滑化し、顔と背景の質感差を減らす。上げすぎ注意。",
                )

                candidates = tuple(d.params.get("candidate_ranges") or ())
                if candidates:
                    self._face_candidate_choices[d.key] = candidates
                    candidate_frame = ttk.Frame(row)
                    candidate_frame.grid(
                        row=8,
                        column=0,
                        columnspan=10,
                        sticky="we",
                        padx=(20, 0),
                        pady=(4, 0),
                    )
                    summary = self._format_face_restore_candidate_summary(candidates)
                    ttk.Label(
                        candidate_frame,
                        text=(
                            "参考候補: "
                            + summary
                            + " / 本番 CodeFormer とは別検出器による参考値です。"
                        ),
                        foreground="#666",
                        wraplength=640,
                        justify="left",
                    ).grid(row=0, column=0, columnspan=3, sticky="w")
                    labels = [self._format_face_restore_candidate_choice(c) for c in candidates[:5]]
                    v_candidate = tk.StringVar(value=labels[0])
                    self._face_candidate_choice_var[d.key] = v_candidate
                    if len(labels) > 1:
                        ttk.Combobox(
                            candidate_frame,
                            textvariable=v_candidate,
                            values=labels,
                            width=24,
                            state="readonly",
                        ).grid(row=1, column=0, sticky="w", pady=(3, 0))
                    ttk.Button(
                        candidate_frame,
                        text="候補を範囲へ反映",
                        command=lambda key=d.key: self._apply_face_restore_candidate(key),
                    ).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(3, 0))
                    candidate_frame.columnconfigure(2, weight=1)

                def _sync_range_state(*_a: object) -> None:
                    state_range = "normal" if v_limit.get() else "disabled"
                    try:
                        start_spin.configure(state=state_range)
                        end_spin.configure(state=state_range)
                    except tk.TclError:
                        pass
                    self._refresh_summary()

                v_limit.trace_add("write", _sync_range_state)
                params_vars["_range_limited"] = v_limit
                params_vars["start_sec"] = v_start
                params_vars["end_sec"] = v_end
                params_vars["blend"] = v_blend
                params_vars["mask"] = v_mask
                params_vars["match_grain"] = v_grain
                params_vars["soften"] = v_soften
                params_vars["global_smooth"] = v_smooth
                _sync_range_state()

            # 各 param 変更でもサマリーが追従するように trace を仕掛ける
            for pv in params_vars.values():
                pv.trace_add("write", lambda *_a: self._refresh_summary())

            self.vars[d.key] = {"enabled": var_enabled, "params": params_vars}

        # 初期サマリーを描画 (var trace は build 完了後に意味を持つ)
        self._refresh_summary()

        # buttons
        btn_row = ttk.Frame(self.top)
        btn_row.pack(fill="x", padx=10, pady=8)
        ttk.Button(btn_row, text="すべて推奨採用", command=self._select_recommended).pack(
            side="left"
        )
        ttk.Button(btn_row, text="すべて解除", command=self._deselect_all).pack(side="left", padx=6)
        ttk.Button(btn_row, text="OK", command=self._on_ok).pack(side="right")
        ttk.Button(btn_row, text="キャンセル", command=self._on_cancel).pack(side="right", padx=6)

    def _build_summary(self, parent: tk.Misc) -> None:
        """サマリーフレーム (解像度 / FPS / 有効ステージ一覧) を構築する。

        meta が None (テスト/異常系) の場合は「※分析未実施」プレースホルダを表示。
        """
        sum_frm = ttk.LabelFrame(parent, text="実行後の出力 (プレビュー)")
        sum_frm.pack(fill="x", pady=(0, 8))
        self._summary_resolution_var = tk.StringVar(value="※分析未実施")
        self._summary_fps_var = tk.StringVar(value="※分析未実施")
        self._summary_stages_var = tk.StringVar(value="※分析未実施")
        ttk.Label(sum_frm, text="解像度:").grid(row=0, column=0, sticky="nw", padx=(8, 4), pady=2)
        ttk.Label(
            sum_frm,
            textvariable=self._summary_resolution_var,
            foreground="#06c",
            wraplength=680,
            justify="left",
        ).grid(row=0, column=1, sticky="w", padx=(0, 8), pady=2)
        ttk.Label(sum_frm, text="FPS:").grid(row=1, column=0, sticky="nw", padx=(8, 4), pady=2)
        ttk.Label(
            sum_frm,
            textvariable=self._summary_fps_var,
            foreground="#06c",
            wraplength=680,
            justify="left",
        ).grid(row=1, column=1, sticky="w", padx=(0, 8), pady=2)
        ttk.Label(sum_frm, text="有効ステージ:").grid(
            row=2, column=0, sticky="nw", padx=(8, 4), pady=2
        )
        ttk.Label(
            sum_frm,
            textvariable=self._summary_stages_var,
            foreground="#444",
            wraplength=680,
            justify="left",
        ).grid(row=2, column=1, sticky="w", padx=(0, 8), pady=2)
        self._summary_runtime_var = tk.StringVar(value="※分析未実施")
        self._summary_capacity_var = tk.StringVar(value="※分析未実施")
        ttk.Label(sum_frm, text="推定実行時間:").grid(
            row=3, column=0, sticky="nw", padx=(8, 4), pady=2
        )
        ttk.Label(
            sum_frm,
            textvariable=self._summary_runtime_var,
            foreground="#444",
            wraplength=680,
            justify="left",
        ).grid(row=3, column=1, sticky="w", padx=(0, 8), pady=2)
        ttk.Label(sum_frm, text="推定容量:").grid(row=4, column=0, sticky="nw", padx=(8, 4), pady=2)
        self._summary_capacity_label = ttk.Label(
            sum_frm,
            textvariable=self._summary_capacity_var,
            foreground="#444",
            wraplength=680,
            justify="left",
        )
        self._summary_capacity_label.grid(row=4, column=1, sticky="w", padx=(0, 8), pady=2)
        sum_frm.columnconfigure(1, weight=1)

    def _snapshot_decisions(self) -> list[StageDecision]:
        """現在の checkbox / params の値で `decisions` の非破壊スナップショットを作る。

        サマリー再計算のためだけに使う (self.decisions は OK 時まで触らない)。
        """
        snap: list[StageDecision] = []
        for d in self.decisions:
            v = self.vars.get(d.key)
            if v is None:
                snap.append(d)
                continue
            enabled = bool(v["enabled"].get()) or d.key == "encode"
            params = dict(d.params)
            try:
                params = self._read_params_for_decision(d, validate=False)
            except ValueError:
                for pk, pv in v["params"].items():
                    try:
                        if not pk.startswith("_"):
                            params[pk] = pv.get()
                    except tk.TclError:
                        pass
            snap.append(
                StageDecision(
                    key=d.key,
                    label_ja=d.label_ja,
                    recommended=d.recommended,
                    confidence=d.confidence,
                    reason=d.reason,
                    enabled=enabled,
                    params=params,
                )
            )
        return snap

    def _refresh_summary(self, *_args) -> None:
        """サマリーラベルを再計算して書き換える。

        - meta=None なら全ラベル「※分析未実施」のまま。
        - 例外発生時はサマリー欄に簡易エラーを出すが、ダイアログ操作は妨げない。
        """
        if self._summary_resolution_var is None:
            return
        if self.meta is None:
            return
        try:
            snap = self._snapshot_decisions()
            summary = compute_output_summary(self.meta, snap)
        except Exception as e:  # noqa: BLE001 — サマリーは best-effort
            self._summary_resolution_var.set(f"(サマリー計算失敗: {type(e).__name__}: {e})")
            self._summary_fps_var.set("-")
            self._summary_stages_var.set("-")
            return
        self._summary_resolution_var.set(self._format_resolution_line(summary))
        self._summary_fps_var.set(self._format_fps_line(summary))
        self._summary_stages_var.set(self._format_stages_line(summary, snap))
        self._refresh_estimates(snap)

    def _refresh_estimates(self, snap: list[StageDecision]) -> None:
        """推定実行時間 / 推定容量を再計算してラベルへ反映 (best-effort)。"""
        if self._summary_runtime_var is None or self._summary_capacity_var is None:
            return
        try:
            estimate = runtime_estimator.estimate_runtime(self.meta, snap)
            peak_gib, _iw, _fps = runtime_estimator.estimate_peak_for_decisions(
                self.meta,
                snap,
                self._chunk_seconds,
                worker_count=self._worker_count,
            )
        except Exception as e:  # noqa: BLE001 — 推定は best-effort
            self._summary_runtime_var.set(f"(推定失敗: {type(e).__name__}: {e})")
            self._summary_capacity_var.set("-")
            return
        self._summary_runtime_var.set(
            f"約 {format_eta_hms(estimate.total_seconds)} ({estimate.accuracy_note})"
        )
        self._summary_capacity_var.set(
            f"{peak_gib:.2f} GiB / 上限 {self._max_temp_gib:.0f} GiB "
            f"(chunk_seconds={self._chunk_seconds}, worker_count={self._worker_count})"
        )
        self._update_capacity_warning_color(peak_gib)

    def _update_capacity_warning_color(self, peak_gib: float) -> None:
        """容量上限超過なら赤字 (OK 押下後の hard gate と同じ閾値で判定)。"""
        if self._summary_capacity_label is None:
            return
        color = "#c33" if peak_gib > self._max_temp_gib else "#444"
        try:
            self._summary_capacity_label.configure(foreground=color)
        except tk.TclError:
            pass

    @staticmethod
    def _format_resolution_line(s: OutputSummary) -> str:
        src = f"{s.src_width}x{s.src_height}"
        out = f"{s.out_width}x{s.out_height}"
        if s.src_width == s.out_width and s.src_height == s.out_height:
            return f"{src} (変更なし — upscale OFF)"
        if s.upscale_enabled and s.upscale_scale is not None and s.intermediate_width is not None:
            detail = f"Real-ESRGAN scale=x{s.upscale_scale} → 中間 {s.intermediate_width}px"
            if s.resolution_clamp_applied:
                clamp_kind = "上方 lanczos" if s.resolution_clamp_upscale else "下方 lanczos"
                detail += f" → {clamp_kind}で {s.out_width}px へクランプ"
            return f"{src} → {out} ({detail})"
        return f"{src} → {out}"

    @staticmethod
    def _format_fps_line(s: OutputSummary) -> str:
        src = f"{float(s.src_fps):.3f}fps"
        out = f"{float(s.out_fps):.3f}fps"
        if not s.interpolate_enabled:
            return f"{src} (補間 OFF — 変更なし)"
        if s.interpolate_multiplier is not None:
            return f"{src} → {out} (RIFE ×{s.interpolate_multiplier} 補間)"
        return f"{src} → {out} (RIFE 補間)"

    @staticmethod
    def _format_face_restore_range(decisions: list[StageDecision]) -> str | None:
        by = decisions_by_key(decisions)
        face = by.get("face_restore")
        if face is None or not face.enabled:
            return None
        start = face.params.get("start_sec")
        end = face.params.get("end_sec")
        if start is None and end is None:
            return "全区間"
        left = "先頭" if start is None else f"{float(start):g}s"
        right = "末尾" if end is None else f"{float(end):g}s"
        return f"{left}-{right}"

    @staticmethod
    def _format_face_restore_naturalize(decisions: list[StageDecision]) -> str | None:
        by = decisions_by_key(decisions)
        face = by.get("face_restore")
        if face is None or not face.enabled:
            return None
        blend = float(face.params.get("blend", config.DEFAULT_FACE_RESTORE_BLEND))
        mask = str(face.params.get("mask", config.DEFAULT_FACE_RESTORE_MASK))
        grain = float(face.params.get("match_grain", config.DEFAULT_FACE_RESTORE_MATCH_GRAIN))
        soften = float(face.params.get("soften", config.DEFAULT_FACE_RESTORE_SOFTEN))
        smooth = float(face.params.get("global_smooth", config.DEFAULT_FACE_RESTORE_GLOBAL_SMOOTH))
        if (
            blend == config.DEFAULT_FACE_RESTORE_BLEND
            and mask == config.DEFAULT_FACE_RESTORE_MASK
            and grain == config.DEFAULT_FACE_RESTORE_MATCH_GRAIN
            and soften == config.DEFAULT_FACE_RESTORE_SOFTEN
            and smooth == config.DEFAULT_FACE_RESTORE_GLOBAL_SMOOTH
        ):
            return None
        parts = [f"blend {blend:g}", f"mask {mask}", f"grain {grain:g}"]
        if soften:
            parts.append(f"soften {soften:g}")
        if smooth:
            parts.append(f"smooth {smooth:g}")
        return " / ".join(parts)

    @staticmethod
    def _format_stages_line(s: OutputSummary, decisions: list[StageDecision] | None = None) -> str:
        if not s.active_stage_labels:
            return "(なし)"
        # 「ラベル(key)」を「, 」で連結
        face_range = (
            EnhanceDecisionDialog._format_face_restore_range(decisions)
            if decisions is not None
            else None
        )
        face_natural = (
            EnhanceDecisionDialog._format_face_restore_naturalize(decisions)
            if decisions is not None
            else None
        )
        parts = []
        for label, key in zip(s.active_stage_labels, s.active_stage_keys):
            if key == "face_restore" and (face_range or face_natural):
                detail = " / ".join(p for p in (face_range, face_natural) if p)
                parts.append(f"{label}({key}: {detail})")
            else:
                parts.append(f"{label}({key})")
        return ", ".join(parts)

    @staticmethod
    def _badge(d: StageDecision) -> str:
        if d.recommended:
            return f"[推奨/{d.confidence}]"
        if d.confidence == "unknown":
            return "[判定不能]"
        return "[不要]"

    @staticmethod
    def _badge_color(d: StageDecision) -> str:
        if d.recommended:
            return "#0a7" if d.confidence == "high" else "#06c"
        if d.confidence == "unknown":
            return "#a60"
        return "#888"

    def _select_recommended(self) -> None:
        for d in self.decisions:
            self.vars[d.key]["enabled"].set(bool(d.recommended) or d.key == "encode")

    def _deselect_all(self) -> None:
        for d in self.decisions:
            # encode は外せない
            self.vars[d.key]["enabled"].set(d.key == "encode")

    def _read_params_for_decision(
        self,
        d: StageDecision,
        *,
        validate: bool,
    ) -> dict[str, Any]:
        params = dict(d.params)
        values = self.vars[d.key]["params"]
        if d.key != "face_restore":
            for pk, pv in values.items():
                if pk.startswith("_"):
                    continue
                params[pk] = pv.get()
            return params

        limited = bool(values["_range_limited"].get())
        max_sec = self._face_restore_max_sec()
        if limited:
            start = _parse_optional_nonnegative_float(
                values["start_sec"].get(),
                label="顔復元 start",
                max_value=max_sec,
            )
            end = _parse_optional_nonnegative_float(
                values["end_sec"].get(),
                label="顔復元 end",
                max_value=max_sec,
            )
            if validate and start is not None and end is not None and start >= end:
                raise ValueError("顔復元の start は end より小さくしてください。")
            params["start_sec"] = start
            params["end_sec"] = end
        else:
            params["start_sec"] = None
            params["end_sec"] = None
        params["blend"] = _parse_unit_float(str(values["blend"].get()), label="顔復元 blend")
        mask = str(values["mask"].get())
        if mask not in config.FACE_RESTORE_MASK_MODES:
            raise ValueError(
                f"顔復元 mask は {config.FACE_RESTORE_MASK_MODES} から選んでください。"
            )
        params["mask"] = mask
        params["match_grain"] = _parse_unit_float(
            str(values["match_grain"].get()), label="顔復元 grain"
        )
        params["soften"] = _parse_unit_float(str(values["soften"].get()), label="顔復元 soften")
        params["global_smooth"] = _parse_unit_float(
            str(values["global_smooth"].get()), label="顔復元 smooth"
        )
        params.setdefault("feather", config.DEFAULT_FACE_RESTORE_FEATHER)
        params.setdefault("mask_dilate", config.DEFAULT_FACE_RESTORE_MASK_DILATE)
        params.setdefault("mask_threshold", config.DEFAULT_FACE_RESTORE_MASK_THRESHOLD)
        return params

    def _finalize(self) -> list[StageDecision]:
        for d in self.decisions:
            d.enabled = bool(self.vars[d.key]["enabled"].get()) or d.key == "encode"
            d.params = self._read_params_for_decision(d, validate=True)
        return self.decisions

    def _on_ok(self) -> None:
        try:
            self.result = self._finalize()
        except ValueError as e:
            messagebox.showerror("入力エラー", str(e), parent=self.top)
            return
        self.top.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.top.destroy()


class App:
    def __init__(self, root: tk.Misc) -> None:
        self.root = root
        try:
            self.root.title("video-enhance-lite")
        except tk.TclError:
            pass
        try:
            self.root.geometry("860x720")
        except tk.TclError:
            pass

        self._worker_thread: threading.Thread | None = None
        self._cancel_event: threading.Event | None = None
        self._close_requested = False
        self._eventq = _EventQueue()
        self._meta: analyzer.VideoMetadata | None = None
        self._decisions: list[StageDecision] | None = None
        # P1-4: 入力/プリセット変更で分析キャッシュを破棄した世代カウンタ。
        # 分析ワーカーは開始時の世代を捕捉し、完了時に不一致なら結果を捨てる
        # (分析中に入力が変わった場合の競合対策)。
        self._analysis_generation = 0
        # "analyze" / "run" — what the currently running worker is doing
        self._current_worker_kind: str | None = None
        # When True, a successful analyze worker should immediately proceed to dialog + run
        self._pending_run_after_analyze: bool = False
        self._temp_run: temp_workspace.TempRunContext | None = None

        self._auto_output_value: str | None = None
        self._setting_auto_output: bool = False

        # PR-D: チャンク粒度 ETA 用の状態
        self._run_started_monotonic: float | None = None
        self._chunk_done_times: list[float] = []  # chunk_done 受信ごとの monotonic 時刻
        self._runtime_estimate_sec: float | None = None  # PR-A の事前推定値 (先験値)

        self.var_input = tk.StringVar()
        self.var_output = tk.StringVar()
        self.var_chunk = tk.StringVar(value=str(config.DEFAULT_CHUNK_SECONDS))
        # ワーカー数 (空欄 = auto = min(cpu//2, chunks, 4))。CLI --worker-count 相当。
        self.var_worker_count = tk.StringVar(value="")
        # GPU 同時実行数 (既定 1 = 直列・従来挙動)。CLI --gpu-concurrency 相当。
        self.var_gpu_concurrency = tk.StringVar(value="1")
        # subprocess の総実行時間ではなく、最後の stdout/stderr からの無出力時間。
        self.var_io_idle_timeout_sec = tk.StringVar(value=str(pipeline.DEFAULT_IO_IDLE_TIMEOUT_SEC))
        self.var_keep_temp = tk.BooleanVar(value=False)
        self.var_status = tk.StringVar(value="待機中")
        self.var_progress_label = tk.StringVar(value="")
        # 残り時間 (ETA) 専用ラベル。chunk_* イベントで上書きされる var_progress_label とは別管理。
        # 並列実行でも chunk_done でのみ更新し、残り時間が常時見えるようにする。
        self.var_eta_label = tk.StringVar(value="")
        # 内部状態 (プリセット解釈後の値)。後方互換のため保持。
        self.var_target_width = tk.StringVar(value=str(config.DEFAULT_TARGET_WIDTH))
        self.var_target_fps = tk.StringVar(value="")
        # 解像度 / FPS プリセット (Combobox 表示用)
        self.var_target_resolution = tk.StringVar(value="自動")
        self.var_target_fps_preset = tk.StringVar(value="自動")
        # FPS 丸めプレビュー (analyze 完了後に更新)
        self.var_fps_preview = tk.StringVar(value="※分析後に表示")

        self._status_label: ttk.Label | None = None

        self._build_ui()
        self._set_status("待機中", "idle")
        self.var_output.trace_add("write", self._on_output_changed)
        # P1-4: 入力ファイル変更で古い分析結果のまま実行されるのを防ぐ
        self.var_input.trace_add("write", self._invalidate_analysis)
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        except (AttributeError, tk.TclError):
            pass
        try:
            self.root.after(100, self._poll_events)
        except tk.TclError:
            pass

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}

        frm_io = ttk.LabelFrame(self.root, text="入出力")
        frm_io.pack(fill="x", **pad)
        ttk.Label(frm_io, text="入力動画").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(frm_io, textvariable=self.var_input, width=70).grid(
            row=0, column=1, sticky="we", padx=6
        )
        ttk.Button(frm_io, text="参照…", command=self._pick_input).grid(row=0, column=2, padx=6)
        ttk.Label(frm_io, text="出力フォルダ").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(frm_io, textvariable=self.var_output, width=70).grid(
            row=1, column=1, sticky="we", padx=6
        )
        ttk.Button(frm_io, text="フォルダ…", command=self._pick_output).grid(
            row=1, column=2, padx=6
        )
        frm_io.columnconfigure(1, weight=1)
        ttk.Label(
            frm_io,
            text=(
                f"※ 空欄なら {config.get_default_output_root()} に自動命名で保存。"
                "指定時はそのフォルダ直下に保存"
            ),
            foreground="#666",
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=6)
        ttk.Button(
            frm_io,
            text="環境診断",
            command=self._show_environment_diagnostics,
        ).grid(row=3, column=2, sticky="e", padx=6, pady=(2, 0))

        frm_opt = ttk.LabelFrame(self.root, text="オプション")
        frm_opt.pack(fill="x", **pad)
        ttk.Label(frm_opt, text="分割秒数").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Spinbox(frm_opt, from_=1, to=99999, textvariable=self.var_chunk, width=10).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Label(frm_opt, text="解像度プリセット").grid(row=0, column=2, sticky="w", padx=6)
        cmb_res = ttk.Combobox(
            frm_opt,
            textvariable=self.var_target_resolution,
            width=14,
            state="readonly",
            values=tuple(_RESOLUTION_PRESET_LABELS.keys()),
        )
        cmb_res.grid(row=0, column=3, sticky="w")
        ttk.Label(frm_opt, text="FPSプリセット").grid(row=0, column=4, sticky="w", padx=6)
        cmb_fps = ttk.Combobox(
            frm_opt,
            textvariable=self.var_target_fps_preset,
            width=10,
            state="readonly",
            values=tuple(_FPS_PRESET_LABELS.keys()),
        )
        cmb_fps.grid(row=0, column=5, sticky="w")
        ttk.Label(frm_opt, textvariable=self.var_fps_preview, foreground="#666").grid(
            row=2, column=2, columnspan=4, sticky="w", padx=6, pady=(0, 4)
        )
        cmb_res.bind("<<ComboboxSelected>>", lambda _e: self._on_preset_changed())
        cmb_fps.bind("<<ComboboxSelected>>", lambda _e: self._on_preset_changed())
        ttk.Checkbutton(frm_opt, text="中間ファイルを残す", variable=self.var_keep_temp).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=6, pady=4
        )
        ttk.Label(frm_opt, text="ワーカー数").grid(row=1, column=2, sticky="w", padx=6)
        sp_worker = ttk.Spinbox(
            frm_opt, from_=1, to=64, textvariable=self.var_worker_count, width=6
        )
        sp_worker.grid(row=1, column=3, sticky="w")
        _attach_tooltip(
            sp_worker,
            "並列ワーカー数。空欄 = 自動 (min(CPU//2, チャンク数, 4))。\n"
            "CPU 段のみ並列。worker > 1 は peak temp が約 worker 倍になります。",
        )
        ttk.Label(frm_opt, text="GPU同時実行数").grid(row=1, column=4, sticky="w", padx=6)
        sp_gpu = ttk.Spinbox(
            frm_opt, from_=1, to=16, textvariable=self.var_gpu_concurrency, width=6
        )
        sp_gpu.grid(row=1, column=5, sticky="w")
        _attach_tooltip(
            sp_gpu,
            "GPU 段 (upscale/interpolate) の同時実行数。既定 1 (従来挙動)。\n"
            "上級者向け: VRAM に余裕がある高性能 GPU でのみ 2 以上を推奨。\n"
            "VRAM 超過時は device lost / crash の可能性あり。worker 数を超える値は丸められます。",
        )
        ttk.Label(frm_opt, text="I/O無出力タイムアウト（秒）").grid(
            row=2, column=0, sticky="w", padx=6, pady=4
        )
        sp_idle = ttk.Spinbox(
            frm_opt,
            from_=1,
            to=99999,
            textvariable=self.var_io_idle_timeout_sec,
            width=10,
        )
        sp_idle.grid(row=2, column=1, sticky="w")
        _attach_tooltip(
            sp_idle,
            "処理全体の制限ではなく、最後の stdout/stderr から何も出ない時間です。\n"
            f"既定 {pipeline.DEFAULT_IO_IDLE_TIMEOUT_SEC} 秒。通常は変更不要です。",
        )

        frm_act = ttk.Frame(self.root)
        frm_act.pack(fill="x", **pad)
        self.btn_analyze = ttk.Button(frm_act, text="事前分析", command=self._on_analyze)
        self.btn_analyze.pack(side="left", padx=6)
        self.btn_run = ttk.Button(frm_act, text="実行", command=self._on_run)
        self.btn_run.pack(side="left", padx=6)
        self.btn_cancel = ttk.Button(
            frm_act, text="キャンセル", command=self._on_cancel, state="disabled"
        )
        self.btn_cancel.pack(side="left", padx=6)
        ttk.Button(frm_act, text="ログクリア", command=self._clear_log).pack(side="left", padx=6)
        self._status_label = ttk.Label(frm_act, textvariable=self.var_status)
        self._status_label.pack(side="right", padx=6)

        frm_prog = ttk.Frame(self.root)
        frm_prog.pack(fill="x", **pad)
        self.progress = ttk.Progressbar(frm_prog, mode="determinate", length=100)
        self.progress.pack(side="left", fill="x", expand=True, padx=6)
        # 残り時間 (ETA) を常時表示する専用ラベル (chunk_* で上書きされない)
        ttk.Label(frm_prog, textvariable=self.var_eta_label, foreground="#06c").pack(
            side="right", padx=6
        )
        ttk.Label(frm_prog, textvariable=self.var_progress_label, foreground="#444").pack(
            side="right", padx=6
        )

        frm_log = ttk.LabelFrame(self.root, text="ログ")
        frm_log.pack(fill="both", expand=True, **pad)
        self.txt = tk.Text(frm_log, height=20, wrap="none", font=("Consolas", 10))
        self.txt.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(frm_log, command=self.txt.yview)
        sb.pack(side="right", fill="y")
        self.txt.configure(yscrollcommand=sb.set)

    # ---------- output auto-fill ----------

    def _set_output_auto(self, value: str) -> None:
        self._setting_auto_output = True
        try:
            self.var_output.set(value)
        finally:
            self._setting_auto_output = False
        self._auto_output_value = value

    def _on_output_changed(self, *_args) -> None:
        if self._setting_auto_output:
            return
        current = self.var_output.get()
        if self._auto_output_value is None or current != self._auto_output_value:
            self._auto_output_value = None

    def _maybe_refresh_auto_output(self) -> None:
        # 出力欄はファイル名ではなく「保存先フォルダ」を保持する。
        # 最終ファイル名は decisions 確定後に _finalize_output_path で組み立てる。
        return

    # ---------- analysis cache invalidation ----------

    def _invalidate_analysis(self, *_args) -> None:
        """入力ファイル / プリセット変更時に事前分析キャッシュを破棄する (P1-4 / P2-1)。

        古い meta/decisions のまま実行すると、別ファイル・旧プリセット前提のステージ選択や
        見積もりが適用されてしまう。破棄後の「実行」は `_on_run` -> (`_decisions is None`)
        -> analyze からやり直す既存フローに乗る。実行中ワーカーは meta/decisions を
        ローカル変数に capture 済みのため、ここでの破棄は実行中ジョブに影響しない。
        """
        self._analysis_generation += 1
        self._meta = None
        self._decisions = None
        self._runtime_estimate_sec = None
        # 出力欄は保存先フォルダなので、分析キャッシュ破棄では変更しない。
        try:
            self.var_fps_preview.set("※分析後に表示")
        except tk.TclError:
            pass

    def _on_preset_changed(self) -> None:
        """解像度 / FPS プリセット変更時: 分析キャッシュを破棄し、プレビューを更新する。"""
        self._invalidate_analysis()
        self._update_fps_preview()

    # ---------- pickers ----------

    def _pick_input(self) -> None:
        p = filedialog.askopenfilename(title="入力動画を選択", filetypes=_VIDEO_FILETYPES)
        if p:
            self.var_input.set(p)
            self._maybe_refresh_auto_output()

    def _pick_output(self) -> None:
        current = self.var_output.get().strip()
        initialdir = current if current else str(config.get_default_output_root())
        p = filedialog.askdirectory(
            title="出力フォルダを選択",
            initialdir=initialdir,
            mustexist=True,
        )
        if p:
            try:
                output_dir = _validate_output_dir_for_gui(Path(p))
            except ValueError as e:
                _show_output_dir_error(str(e))
                return
            self._auto_output_value = None
            self.var_output.set(str(output_dir))

    def _show_environment_diagnostics(self) -> None:
        try:
            report = deps.format_capability_report()
        except Exception as exc:  # noqa: BLE001 - diagnostics must never crash the GUI
            report = f"環境診断に失敗しました: {type(exc).__name__}: {exc}"
        messagebox.showinfo("環境診断", report, parent=self.root)

    # ---------- log / status ----------

    def _clear_log(self) -> None:
        self.txt.delete("1.0", "end")

    def _append_log(self, msg: str) -> None:
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")

    def _set_status(
        self, text: str, kind: Literal["idle", "running", "success", "failed", "cancelled"]
    ) -> None:
        color = {
            "idle": "#444",
            "running": "#06c",
            "success": "#0a7",
            "failed": "#c33",
            "cancelled": "#a60",
        }.get(kind, "#444")
        self.var_status.set(text)
        if self._status_label is not None:
            self._status_label.configure(foreground=color)

    # ---------- analyze / run ----------

    def _validate_common(self) -> tuple[Path, Path | None, int, dict] | None:
        input_str = self.var_input.get().strip()
        if not input_str:
            messagebox.showerror("入力エラー", "入力動画を選択してください。")
            return None
        input_path = Path(input_str)
        if not input_path.exists():
            messagebox.showerror("入力エラー", f"入力ファイルが見つかりません: {input_path}")
            return None
        try:
            chunk_seconds = _parse_chunk_seconds(self.var_chunk.get())
            _parse_positive_int(
                self.var_io_idle_timeout_sec.get(),
                label="I/O無出力タイムアウト（秒）",
            )
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("設定エラー", str(e))
            return None
        out_str = self.var_output.get().strip()
        try:
            if out_str:
                # 出力欄は保存先フォルダ。ファイル名は decisions 確定後に自動命名する。
                output_path: Path | None = _validate_output_dir_for_gui(Path(out_str))
            else:
                # 空欄 = ユーザーの既定 Videos 配下へ自動命名。
                output_path = None
        except ValueError as e:
            _show_output_dir_error(str(e))
            return None

        defaults: dict = {}
        # 解像度プリセット (Combobox)
        res_label = self.var_target_resolution.get()
        preset_w = _RESOLUTION_PRESET_LABELS.get(res_label, None)
        if preset_w is not None:
            defaults["target_width"] = preset_w
            defaults["target_width_explicit"] = True
        # FPS プリセット (Combobox)
        fps_label = self.var_target_fps_preset.get()
        preset_fps = _FPS_PRESET_LABELS.get(fps_label, None)
        if preset_fps is not None:
            try:
                defaults["target_fps"] = Fraction(float(preset_fps)).limit_denominator(10000)
                defaults["target_fps_explicit"] = True
            except (ValueError, ZeroDivisionError):
                messagebox.showerror(
                    "設定エラー", f"FPS プリセットの解釈に失敗しました: {fps_label!r}"
                )
                return None
        return input_path, output_path, chunk_seconds, defaults

    def _update_fps_preview(self) -> None:
        """fps プルダウン変更時 / analyze 完了時に丸めプレビュー Label を更新する。

        analyze 未実施なら「※分析後に表示」。
        プリセット=自動 (target_fps_default=None) なら「自動（source 依存）」。
        source >= target なら「補間しません（…）」。
        source < target なら整数倍丸めの結果を表示。
        """
        if self._meta is None:
            self.var_fps_preview.set("※分析後に表示")
            return
        fps_label = self.var_target_fps_preset.get()
        preset_fps = _FPS_PRESET_LABELS.get(fps_label, None)
        src_fps = Fraction(self._meta.fps)
        if preset_fps is None:
            self.var_fps_preview.set(
                f"自動（source_fps={float(src_fps):.3f} を基準にロジック判定）"
            )
            return
        if preset_fps == config.FPS_KEEP_SOURCE:
            self.var_fps_preview.set(
                f"現状維持（補間しません / source {float(src_fps):.3f}fps のまま）"
            )
            return
        req = Fraction(float(preset_fps)).limit_denominator(10000)
        if src_fps >= req:
            self.var_fps_preview.set(
                f"補間しません（source {float(src_fps):.3f}fps >= target {float(req):.3f}fps）"
            )
            return
        normalized, warn = decisions_mod.normalize_target_fps(src_fps, req)
        msg = (
            f"-> {float(normalized):.3f} fps "
            f"(×{int(normalized / src_fps)}, source {float(src_fps):.3f})"
        )
        if warn:
            msg += " (整数倍に丸めました)"
        self.var_fps_preview.set(msg)

    def _estimate_temp_and_abort_if_exceed(
        self,
        decs: list[StageDecision],
        chunk_seconds: int,
        max_temp_gib: float,
        worker_count: int,
    ) -> bool:
        """decisions 確定後に peak temp を見積もり、`--max-temp-gib` 超過なら messagebox で停止通知。

        Returns True if OK to proceed, False if aborted (caller must NOT start pipeline).
        """
        if self._meta is None:
            return True
        peak_gib, intermediate_width, effective_fps_peak = (
            runtime_estimator.estimate_peak_for_decisions(
                self._meta,
                decs,
                chunk_seconds,
                worker_count=worker_count,
            )
        )
        if peak_gib > max_temp_gib:
            messagebox.showerror(
                "temp 容量上限超過",
                (
                    f"見積もり peak temp = {peak_gib:.1f} GiB が上限 {max_temp_gib} GiB を超えました。\n"
                    f"intermediate_width={intermediate_width}, effective_fps={effective_fps_peak}\n\n"
                    "実行を中止します。プリセットを下げるか、--max-temp-gib を変更してください。"
                ),
            )
            self._append_log(
                f"[ERROR] estimated peak temp {peak_gib:.1f} GiB > --max-temp-gib {max_temp_gib} "
                f"(intermediate_width={intermediate_width}, effective_fps={effective_fps_peak})"
            )
            self._set_status("temp 上限超過 — 中止", "failed")
            return False
        return True

    def _on_analyze(self) -> None:
        """「事前分析」ボタンハンドラ。pending_run_after_analyze は False のまま。"""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            messagebox.showwarning("実行中", "既に処理が実行中です。")
            return
        validated = self._validate_common()
        if validated is None:
            return
        input_path, _output, _cs, defaults = validated
        self._pending_run_after_analyze = False
        self._start_analyze_worker(input_path, defaults)

    def _start_analyze_worker(self, input_path: Path, defaults: dict) -> None:
        self._clear_log()
        self._append_log(f"input  : {input_path}")
        self._append_log("[analyze] 起動中…")
        self._set_status("分析中…", "running")
        self.progress.configure(mode="indeterminate", maximum=100, value=0)
        try:
            self.progress.start(80)
        except tk.TclError:
            pass
        self.var_progress_label.set("分析中…")
        self.btn_analyze.configure(state="disabled")
        self.btn_run.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self._cancel_event = threading.Event()
        cancel_event = self._cancel_event
        self._current_worker_kind = "analyze"
        gen = self._analysis_generation

        def _worker() -> None:
            try:
                temp_run = temp_workspace.create_temp_run_context(input_path)
                self._temp_run = temp_run
                ffprobe_rb = deps.check_ffprobe()
                ffmpeg_rb = deps.check_ffmpeg()
                self._eventq.write(deps.format_resolved_binary("ffprobe", ffprobe_rb))
                self._eventq.write(deps.format_resolved_binary("ffmpeg", ffmpeg_rb))
                meta, stats = analyzer.analyze(
                    input_path=input_path,
                    ffprobe=ffprobe_rb.path,
                    ffmpeg=ffmpeg_rb.path,
                    sample_count=config.DEFAULT_SAMPLE_COUNT,
                    sample_dir=temp_run.run_leaf / "analyze",
                    keep_samples=True,
                    cancel_event=cancel_event,
                    warning=lambda message: self._eventq.write(f"[WARN] {message}"),
                )
                decs = decisions_mod.recommend(meta, stats, defaults)
                if gen != self._analysis_generation:
                    # 分析中に入力/プリセットが変更された -> 古い入力の結果を採用しない (P1-4)
                    self._eventq.write(
                        "[info] 入力/プリセットが変更されたため、この分析結果を破棄しました"
                    )
                    self._eventq.finish(_FinishEvent("cancelled"))
                    return
                self._meta = meta
                self._decisions = decs
                self._eventq.write(
                    f"[analyze] {meta.width}x{meta.height} @ {float(meta.fps):.3f}fps "
                    f"duration={meta.duration_sec:.1f}s audio={meta.has_audio}"
                )
                framerate_warning = _format_framerate_warning(meta)
                if framerate_warning:
                    self._eventq.write(f"[WARN] {framerate_warning}")
                self._eventq.finish(_FinishEvent("success"))
            except (analyzer.AnalyzerError, MissingDependencyError) as e:
                self._eventq.write(f"[ERROR] {e}")
                self._eventq.finish(_FinishEvent("failed", message=str(e)))
            except Exception as e:  # noqa: BLE001
                self._eventq.write(f"[ERROR] {type(e).__name__}: {e}")
                self._eventq.finish(_FinishEvent("failed", message=str(e)))

        self._worker_thread = threading.Thread(target=_worker, daemon=True)
        self._worker_thread.start()

    def _open_decision_dialog(self) -> list[StageDecision] | None:
        if self._decisions is None:
            return None
        try:
            cs = int(self.var_chunk.get())
        except (ValueError, tk.TclError):
            cs = config.DEFAULT_CHUNK_SECONDS
        try:
            worker_in = _parse_optional_positive_int(
                self.var_worker_count.get(), label="ワーカー数"
            )
        except (ValueError, tk.TclError):
            worker_in = 1
        estimated_chunks = (
            max(1, math.ceil(float(self._meta.duration_sec) / float(cs)))
            if self._meta is not None
            else 1
        )
        if worker_in is None:
            cpu = max(1, (os.cpu_count() or 1) // 2)
            dialog_worker_count = min(cpu, estimated_chunks, 4)
        else:
            dialog_worker_count = worker_in
        dlg = EnhanceDecisionDialog(
            self.root,
            self._decisions,
            meta=self._meta,
            chunk_seconds=cs,
            max_temp_gib=float(config.DEFAULT_MAX_TEMP_GIB),
            worker_count=dialog_worker_count,
        )
        # 非モーダル: wait_window で完了待ち
        try:
            self.root.wait_window(dlg.top)
        except tk.TclError:
            pass
        return dlg.result

    def _on_run(self) -> None:
        """「実行」ボタンハンドラ。事前分析が未実施なら analyze → dialog → run を連続実行."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            messagebox.showwarning("実行中", "既に処理が実行中です。")
            return
        validated = self._validate_common()
        if validated is None:
            return
        input_path, _output, _cs, defaults = validated

        # 事前分析がまだなら analyze を起動して、完了時に自動的に dialog → run へ続ける
        if self._decisions is None or self._meta is None:
            self._pending_run_after_analyze = True
            self._start_analyze_worker(input_path, defaults)
            return

        self._proceed_to_dialog_and_run()

    def _finalize_output_path(
        self, input_path: Path, dec_result: list[StageDecision]
    ) -> Path | None:
        """ダイアログ確定後の最終出力パスを決める。

        出力欄は保存先フォルダとして扱う。最終ファイル名は、ダイアログで確定した
        `dec_result` から実行スペック名を再計算して `<出力フォルダ>/<入力名>_<spec>.<ext>` にする。
        """
        out_str = self.var_output.get().strip()
        try:
            output_dir = (
                _validate_output_dir_for_gui(Path(out_str))
                if out_str
                else config.get_default_output_root()
            )
            summary = compute_output_summary(self._meta, dec_result)
            output_path = derive_output_path_for_gui(input_path, summary, output_dir)
            output_path = config.validate_output_path(output_path)
        except ValueError as e:
            _show_output_dir_error(str(e))
            return None
        return output_path

    def _proceed_to_dialog_and_run(self) -> None:
        """analyze 完了後（または事前分析済み）のステージ選択ダイアログ表示 → run 起動."""
        validated = self._validate_common()
        if validated is None:
            return
        input_path, _pre_output, chunk_seconds, _ = validated

        try:
            framerate_plan = pipeline.resolve_framerate_normalization(self._meta)
        except pipeline.PipelineError as e:
            messagebox.showerror("フレームレート非対応", str(e))
            self._append_log(f"[FAIL] stage={e.stage}: {e.message}")
            self._set_status("実行不可", "failed")
            self._cleanup_temp_run()
            return

        framerate_warning = _format_framerate_warning(self._meta)
        if framerate_plan.requires_normalization and framerate_warning:
            self._append_log(f"[WARN] {framerate_warning}")
            if not messagebox.askyesno(
                "フレームレート警告",
                f"{framerate_warning}。\n続行しますか?",
            ):
                self._cleanup_temp_run()
                return

        dec_result = self._open_decision_dialog()
        if dec_result is None:
            self._append_log("[cancelled] ステージ選択ダイアログでキャンセルされました")
            self._set_status("キャンセル", "cancelled")
            self._cleanup_temp_run()
            return

        face = decisions_by_key(dec_result).get("face_restore")
        if (
            face is not None
            and face.enabled
            and face.params.get("start_sec") is None
            and face.params.get("end_sec") is None
        ):
            ok = messagebox.askyesno(
                "顔復元の全区間適用",
                "face_restore が全区間に適用されます。\n\n"
                "CodeFormer は処理時間が長く、顔の細部を AI 生成し得ます。続行しますか?",
            )
            if not ok:
                self._cleanup_temp_run()
                return

        # 出力名の最終確定: 保存先フォルダ直下に、実行スペックを反映したファイル名で保存する。
        # 上書き確認・peak 見積もりより前に確定し、以降は最終パスで検証する。
        output_path = self._finalize_output_path(input_path, dec_result)
        if output_path is None:
            self._cleanup_temp_run()
            return

        try:
            pipeline.resolve_stream_preservation(self._meta)
        except pipeline.PipelineError as e:
            messagebox.showerror("ストリーム非対応", str(e))
            self._append_log(f"[FAIL] stage={e.stage}: {e.message}")
            self._set_status("実行不可", "failed")
            self._cleanup_temp_run()
            return

        # temp 容量見積もり (CLI と同じ閾値で即停止; 確認ダイアログ無し — 利用者承認 #5)
        # PR-G1: ワーカー数 / GPU 同時実行数を解決 (worker 空欄 = CLI と同じ auto)
        try:
            worker_in = _parse_optional_positive_int(
                self.var_worker_count.get(), label="ワーカー数"
            )
            gpu_concurrency = (
                _parse_optional_positive_int(self.var_gpu_concurrency.get(), label="GPU同時実行数")
                or 1
            )
            io_idle_timeout_sec = _parse_positive_int(
                self.var_io_idle_timeout_sec.get(),
                label="I/O無出力タイムアウト（秒）",
            )
        except ValueError as e:
            messagebox.showerror("入力エラー", str(e))
            self._cleanup_temp_run()
            return
        estimated_chunks = max(1, math.ceil(float(self._meta.duration_sec) / float(chunk_seconds)))
        if worker_in is None:
            cpu = max(1, (os.cpu_count() or 1) // 2)
            effective_worker_count = min(cpu, estimated_chunks, 4)
        else:
            effective_worker_count = worker_in
        if not self._estimate_temp_and_abort_if_exceed(
            dec_result,
            chunk_seconds,
            float(config.DEFAULT_MAX_TEMP_GIB),
            effective_worker_count,
        ):
            self._cleanup_temp_run()
            return
        if gpu_concurrency > 1:
            eff_gpu = min(gpu_concurrency, effective_worker_count)
            self._append_log(
                f"[WARN] GPU同時実行数={gpu_concurrency} (有効 {eff_gpu}): "
                f"VRAM 超過時は device lost/crash の可能性があります。"
                f"不足する場合は 1 に戻してください。"
            )

        # PR-D: ETA の先験値だけ確定し、起点時刻は process_chunks 受信時にセットする
        try:
            self._runtime_estimate_sec = runtime_estimator.estimate_runtime(
                self._meta, dec_result
            ).total_seconds
        except Exception:  # noqa: BLE001 — 推定は best-effort
            self._runtime_estimate_sec = None
        self._chunk_done_times.clear()
        self._run_started_monotonic = None

        try:
            ffprobe_rb = deps.check_ffprobe()
            by = decisions_by_key(dec_result)
            # P2-4: CLI と同じ auto 解決。vidstab 入り ffmpeg なら vidstab、
            # 非搭載なら OpenCV バックエンドへフォールバックする。
            stabilize_backend = config.DEFAULT_STABILIZE_BACKEND
            if by.get("stabilize") is not None and by["stabilize"].enabled:
                try:
                    ffmpeg_rb = deps.check_ffmpeg_with_vidstab()
                    stabilize_backend = "vidstab"
                except MissingDependencyError:
                    ffmpeg_rb = deps.check_ffmpeg()
                    stabilize_backend = "opencv"
                    self._append_log(
                        "[info] ffmpeg が vidstab 非搭載のため OpenCV 手ぶれ補正バックエンドを使用します"
                    )
            else:
                ffmpeg_rb = deps.check_ffmpeg()
            self._append_log(deps.format_resolved_binary("ffprobe", ffprobe_rb))
            self._append_log(deps.format_resolved_binary("ffmpeg", ffmpeg_rb))
            realesrgan_path = None
            rife_path = None
            codeformer_local = None
            if by.get("upscale") is not None and by["upscale"].enabled:
                rea_rb = deps.check_realesrgan()
                self._append_log(deps.format_resolved_binary("realesrgan", rea_rb))
                # CLI と同じモデル実存チェック (実行時の PipelineError より手前で検出する)
                rea_scale = int(by["upscale"].params.get("scale", config.DEFAULT_REALESRGAN_SCALE))
                rea_model = str(by["upscale"].params.get("model", config.DEFAULT_REALESRGAN_MODEL))
                try:
                    config.validate_realesrgan_model(
                        rea_rb.path.parent / "models", rea_model, rea_scale
                    )
                except (FileNotFoundError, ValueError) as ve:
                    raise MissingDependencyError("realesrgan", str(ve)) from ve
                realesrgan_path = rea_rb.path
            if by.get("interpolate") is not None and by["interpolate"].enabled:
                rife_rb = deps.check_rife()
                self._append_log(deps.format_resolved_binary("rife", rife_rb))
                verification = deps.verify_rife_sha256(rife_rb)
                if verification.level == "warn":
                    self._append_log(f"[WARN] {verification.message}")
                else:
                    self._append_log(f"[info] {verification.message}")
                # CLI と同じモデル実存チェック (公式配布 = exe 同階層 / 旧想定 models/ 両対応)
                rife_model = str(by["interpolate"].params.get("model", config.DEFAULT_RIFE_MODEL))
                try:
                    rife_models_dir = config.resolve_rife_model_dir(rife_rb.path.parent, rife_model)
                    config.validate_rife_model(rife_models_dir, rife_model)
                except (FileNotFoundError, ValueError) as ve:
                    raise MissingDependencyError("rife", str(ve)) from ve
                rife_path = rife_rb.path
            if by.get("face_restore") is not None and by["face_restore"].enabled:
                codeformer_local = deps.check_codeformer_local()
                self._append_log(
                    f"[deps] codeformer-local: repo={codeformer_local.repo_root} "
                    f"python={codeformer_local.python}"
                )
        except MissingDependencyError as e:
            messagebox.showerror("依存関係エラー", str(e))
            self._cleanup_temp_run()
            return

        try:
            if self._temp_run is None:
                self._temp_run = temp_workspace.create_temp_run_context(input_path)
            peak_gib, _, _ = runtime_estimator.estimate_peak_for_decisions(
                self._meta,
                dec_result,
                chunk_seconds,
                worker_count=effective_worker_count,
            )
            capacity_warning = temp_workspace.check_temp_capacity(
                self._temp_run, math.ceil(peak_gib * (1024**3))
            )
            if capacity_warning:
                self._append_log(f"[WARN] {capacity_warning}")
        except temp_workspace.TempWorkspaceError as e:
            messagebox.showerror("temp workspace エラー", str(e))
            self._cleanup_temp_run()
            return

        self._append_log(f"input  : {input_path}")
        self._append_log(f"output : {output_path}")
        self._append_log("-" * 60)
        self.progress.configure(mode="determinate", maximum=100, value=0)
        self.var_progress_label.set("準備中…")
        self.btn_run.configure(state="disabled")
        self.btn_analyze.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self._set_status("実行中…", "running")

        self._cancel_event = threading.Event()
        cancel_event = self._cancel_event
        meta = self._meta
        decs_local = dec_result
        self._current_worker_kind = "run"

        def _log_cb(msg: str) -> None:
            self._eventq.write(msg)

        def _progress_cb(ev: pipeline.ProgressEvent) -> None:
            self._eventq.progress(ev)

        def _worker() -> None:
            try:
                face_dec_local = decisions_by_key(decs_local).get("face_restore")
                face_restore_options = (
                    pipeline.FaceRestoreOptions.from_params(face_dec_local.params)
                    if face_dec_local is not None
                    else pipeline.FaceRestoreOptions()
                )
                pipeline.run_pipeline(
                    input_path=input_path,
                    output_path=output_path,
                    decisions=decs_local,
                    meta=meta,
                    chunk_seconds=chunk_seconds,
                    keep_temp=bool(self.var_keep_temp.get()),
                    log=_log_cb,
                    cancel_event=cancel_event,
                    progress=_progress_cb,
                    ffmpeg=ffmpeg_rb.path,
                    ffprobe=ffprobe_rb.path,
                    realesrgan=realesrgan_path,
                    rife=rife_path,
                    codeformer=codeformer_local,
                    face_restore_options=face_restore_options,
                    stabilize_backend=stabilize_backend,
                    worker_count=effective_worker_count,
                    gpu_concurrency=gpu_concurrency,
                    io_idle_timeout_sec=io_idle_timeout_sec,
                    temp_run=self._temp_run,
                    estimated_peak_temp_bytes=math.ceil(peak_gib * (1024**3)),
                )
                self._eventq.write(f"[OK] saved: {output_path}")
                self._eventq.finish(_FinishEvent("success"))
            except pipeline.PipelineCancelled as e:
                self._eventq.write(f"[CANCELLED] {e}")
                self._eventq.finish(_FinishEvent("cancelled"))
            except pipeline.PipelineError as e:
                self._eventq.write(f"[FAIL] stage={e.stage}: {e.message}")
                self._eventq.finish(_FinishEvent("failed", message=str(e)))
            except MissingDependencyError as e:
                self._eventq.write(f"[ERROR] {e}")
                self._eventq.finish(_FinishEvent("failed", message=str(e)))
            except Exception as e:  # noqa: BLE001
                self._eventq.write(f"[ERROR] {type(e).__name__}: {e}")
                self._eventq.finish(_FinishEvent("failed", message=str(e)))
            finally:
                if not bool(self.var_keep_temp.get()):
                    self._cleanup_temp_run(log_to_eventq=True)

        self._worker_thread = threading.Thread(target=_worker, daemon=True)
        self._worker_thread.start()

    def _on_cancel(self) -> None:
        if self._cancel_event is None or self._worker_thread is None:
            return
        if not self._worker_thread.is_alive():
            return
        self._cancel_event.set()
        self.btn_cancel.configure(state="disabled")
        self._set_status("中止要求中…", "cancelled")
        self._append_log("[INFO] キャンセル要求受付。次のステージ境界で停止します。")

    def _on_close(self) -> None:
        worker_running = self._worker_thread is not None and self._worker_thread.is_alive()
        if not worker_running:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            return

        if self._close_requested:
            return
        try:
            should_close = messagebox.askyesno(
                "処理中",
                "動画処理が実行中です。処理を中止し、安全に終了しますか?\n\n"
                "外部プロセスの停止と一時ファイルの後処理が終わるまで画面を閉じません。",
                parent=self.root,
            )
        except tk.TclError:
            return
        if not should_close:
            return

        self._close_requested = True
        self._pending_run_after_analyze = False
        self._on_cancel()
        self._set_status("終了待機中…", "cancelled")
        self._append_log("[INFO] 終了要求受付。workerの停止完了後にウィンドウを閉じます。")

    # ---------- event polling ----------

    def _handle_progress(self, ev: pipeline.ProgressEvent) -> None:
        if ev.stage == "process_chunks":
            # ETA の起点を pipeline 実行開始 (process_chunks emit) に合わせる
            if self._run_started_monotonic is None:
                self._run_started_monotonic = time.monotonic()
            self._chunk_done_times.clear()
            try:
                self.progress.stop()
            except tk.TclError:
                pass
            total = ev.total_chunks or 1
            self.progress.configure(mode="determinate", maximum=total, value=0)
            self.var_progress_label.set(f"0 / {total} チャンク")
            self.var_eta_label.set("残り 計測中…")
            return
        if ev.stage == "chunk_done":
            now = time.monotonic()
            # fallback: process_chunks を取り逃した場合 (テスト等) にここで初期化
            if self._run_started_monotonic is None:
                self._run_started_monotonic = now
            self._chunk_done_times.append(now)
            elapsed = now - self._run_started_monotonic
            done = len(self._chunk_done_times)
            total = ev.total_chunks or 1
            if done >= 3:
                eta_sec = elapsed / done * (total - done)  # 実測線形外挿
            else:
                # 1〜2 チャンクは外挿が不安定。先験値で残り割合を見積もる
                prior = self._runtime_estimate_sec or elapsed * total / max(done, 1)
                eta_sec = max(0.0, prior * (1 - done / total))
            self.progress.configure(value=done)
            eta_text = format_eta_hms(eta_sec)
            self.var_progress_label.set(f"{done} / {total} チャンク (残り {eta_text})")
            # 専用ラベルは chunk_* イベントで上書きされないため、残り時間が常時見える
            self.var_eta_label.set(f"残り {eta_text}")
            # ログ欄にも残り時間を出す (CLI の [progress] 相当。ログ派ユーザー向け)
            self._append_log(f"[progress] {done}/{total} チャンク完了 残り {eta_text}")
            return
        if ev.stage == "chunk_start":
            total = ev.total_chunks or 1
            cur = (ev.chunk_index or 0) + 1
            self.var_progress_label.set(f"{cur - 1} / {total} チャンク（処理中: {cur}）")
            return
        if ev.stage in (
            "chunk_extract",
            "chunk_pre_filter",
            "chunk_upscale",
            "chunk_resolution_clamp",
            "chunk_face_restore",
            "chunk_interpolate",
            "chunk_post_filter",
            "chunk_mux",
            "chunk_stabilize_detect",
        ):
            self.var_progress_label.set(ev.label or ev.stage)
            return
        if ev.stage == "concat":
            total = ev.total_chunks or 1
            self.progress.configure(value=total)
            self.var_progress_label.set("結合中…")
            return
        if ev.stage == "done":
            total = ev.total_chunks or 1
            self.progress.configure(value=total)
            self.var_progress_label.set("完了")
            return

    def _cleanup_temp_run(self, *, log_to_eventq: bool = False) -> None:
        temp_run = self._temp_run
        if temp_run is None:
            return
        try:
            temp_workspace.cleanup_temp_run(temp_run)
            self._temp_run = None
        except temp_workspace.TempWorkspaceError as exc:
            message = f"[WARN] cleanup failed; temporary files remain at {temp_run.run_leaf}: {exc}"
            if log_to_eventq:
                self._eventq.write(message)
            else:
                self._append_log(message)

    def _handle_finish(self, ev: _FinishEvent) -> None:
        self.btn_run.configure(state="normal")
        self.btn_analyze.configure(state="normal")
        self.btn_cancel.configure(state="disabled")
        try:
            self.progress.stop()
        except tk.TclError:
            pass
        finished_kind = self._current_worker_kind
        self._current_worker_kind = None
        self._cancel_event = None

        if self._close_requested:
            self._pending_run_after_analyze = False
            if finished_kind == "analyze":
                self._cleanup_temp_run()
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            return

        # analyze 成功時は fps プレビューを更新 (meta が確定したため)
        if finished_kind == "analyze" and ev.status == "success":
            try:
                self._update_fps_preview()
            except tk.TclError:
                pass
            # 出力欄は保存先フォルダだけを保持する。最終ファイル名は実行直前に確定する。
            self._maybe_refresh_auto_output()

        # analyze ステージが成功し、かつ pending_run_after_analyze が立っているなら
        # そのまま dialog → run のフローへ進む（プラン「実行押下で analyze→dialog→run 連続実行」要件）
        if (
            finished_kind == "analyze"
            and ev.status == "success"
            and self._pending_run_after_analyze
        ):
            self._pending_run_after_analyze = False
            self._set_status("分析完了 — ステージ選択中", "running")
            self.var_progress_label.set("ステージ選択中…")
            # メインスレッドの mainloop に戻ってからダイアログ表示
            try:
                self.root.after(0, self._proceed_to_dialog_and_run)
            except tk.TclError:
                self._proceed_to_dialog_and_run()
            return

        if finished_kind == "analyze":
            self._cleanup_temp_run()

        # それ以外は通常の完了処理（pending フラグはこの時点でクリアしておく）
        self._pending_run_after_analyze = False
        if ev.status == "success":
            self._set_status("完了", "success")
            self.var_progress_label.set("完了")
        elif ev.status == "cancelled":
            self._set_status("中止しました", "cancelled")
            self.var_progress_label.set("中止")
        else:
            self._set_status("失敗", "failed")
            self.var_progress_label.set("失敗")

    def _poll_events(self) -> None:
        for item in self._eventq.drain():
            if isinstance(item, _FinishEvent):
                self._handle_finish(item)
            elif isinstance(item, pipeline.ProgressEvent):
                self._handle_progress(item)
            else:
                self._append_log(item)
        try:
            self.root.after(100, self._poll_events)
        except tk.TclError:
            pass


def main(argv: list[str] | None = None) -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
