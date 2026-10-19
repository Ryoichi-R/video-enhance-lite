"""テスト用の単一 `FakeProcessRunner` (V2-8 DI)。

`video_enhance_lite.pipeline.ProcessRunner` 構造的部分型 (Protocol) として
`run_pipeline(..., runner=FakeProcessRunner())` に注入する。実 `subprocess.Popen`
を起動せず、呼び出し引数 (argv / cwd) を記録し、後続の `_count_png` / `_has_png`
が成功するよう出力ディレクトリへ fake な PNG / mp4 を書き出す。

旧来 `tests/enhance/test_pipeline.py` と
`tests/enhance/test_pipeline_resolution_clamp_position.py` に二重定義されていた
`FakeRunner` を本クラスに一本化した。

例外契約 (ProcessRunner): 失敗時は **必ず** `PipelineError` を直接送出する
(`subprocess.CalledProcessError` を投げて production 側が包む旧 seam 経路は再現
しない)。`RealProcessRunner` も非 0 終了を `PipelineError` で投げるため、本 fake も
契約を揃える。

スレッドセーフ: 並列実行 (worker>1) で同一インスタンスを共有してもズレないよう、
単一の呼び出し記録 list を `threading.Lock` 内で append する。`calls` / `cwds` / `envs` は
そこから導出する read-only プロパティとして公開する (別 list への二重 append による
race を避ける)。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from video_enhance_lite.pipeline import DEFAULT_IO_IDLE_TIMEOUT_SEC, PipelineError


class FakeProcessRunner:
    """`ProcessRunner` 準拠の fake。呼び出しを記録し fake 成果物を書き出す。"""

    def __init__(self, *, fail_on_argv_marker: str | None = None) -> None:
        self.fail_on_argv_marker = fail_on_argv_marker
        self._lock = threading.Lock()
        # 単一の記録 list (argv, cwd, env) を Lock 内で append する。
        self._records: list[tuple[list[str], str | None, dict[str, str] | None]] = []

    # ---- ProcessRunner protocol ----
    def run_with_progress(
        self,
        argv: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        idle_timeout: int = DEFAULT_IO_IDLE_TIMEOUT_SEC,
        cancel_event: "threading.Event | None" = None,
        on_progress: Callable[[int], None] | None = None,
        on_output_event: Callable[[str, str], None] | None = None,
        on_heartbeat: Callable[[float, float], None] | None = None,
        heartbeat_interval_sec: float | None = None,
        stage: str = "ffmpeg",
        chunk_idx: int | None = None,
    ) -> int:
        argv_list = list(argv)
        env_copy = dict(env) if env is not None else None
        with self._lock:
            self._records.append((argv_list, str(cwd) if cwd is not None else None, env_copy))

        if self.fail_on_argv_marker is not None and any(
            self.fail_on_argv_marker in tok for tok in argv_list
        ):
            # 例外契約: 失敗時は必ず PipelineError を直接送出する。
            raise PipelineError(
                stage,
                f"fake fail: marker={self.fail_on_argv_marker!r}",
                chunk_idx=chunk_idx,
            )

        # 後続の _count_png() / _has_png() > 0 を満たすよう fake 成果物を書く。
        # ffmpeg: 末尾 arg = 出力パターン / realesrgan / rife: -o <dir>
        # PNG は 2 枚書く: 実チャンクは常に複数フレームであり、1 枚だと
        # 「フレーム < 2 で stabilize をスキップする」端数チャンク特例 (P2-3) を
        # 誤って踏んでしまうため。
        bin_name = argv_list[0].lower()
        if "inference_codeformer.py" in " ".join(argv_list):
            if "--output_path" in argv_list:
                out_dir = Path(argv_list[argv_list.index("--output_path") + 1]) / "final_results"
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "00000000.png").write_bytes(b"")
                (out_dir / "00000001.png").write_bytes(b"")
        elif "realesrgan" in bin_name or "rife" in bin_name:
            if "-o" in argv_list:
                out_dir = Path(argv_list[argv_list.index("-o") + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "00000000.png").write_bytes(b"")
                (out_dir / "00000001.png").write_bytes(b"")
        elif "ffmpeg" in bin_name:
            # concat 出力 / mux 出力 (mp4) / pattern 出力 (%08d.png)
            last = Path(argv_list[-1])
            if last.suffix.lower() == ".mp4":
                last.parent.mkdir(parents=True, exist_ok=True)
                last.write_bytes(b"")
            elif "%08d.png" in str(last):
                last.parent.mkdir(parents=True, exist_ok=True)
                (last.parent / "00000000.png").write_bytes(b"")
                (last.parent / "00000001.png").write_bytes(b"")
            # `-f null -` (vidstabdetect 1st pass) は何も生成しないが OK

        return 0

    # ---- 旧 FakeRunner 互換の read-only ビュー ----
    @property
    def calls(self) -> list[list[str]]:
        with self._lock:
            return [argv for (argv, _cwd, _env) in self._records]

    @property
    def cwds(self) -> list[str | None]:
        with self._lock:
            return [cwd for (_argv, cwd, _env) in self._records]

    @property
    def envs(self) -> list[dict[str, str] | None]:
        with self._lock:
            return [env for (_argv, _cwd, env) in self._records]
