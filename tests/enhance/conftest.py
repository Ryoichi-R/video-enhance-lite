from __future__ import annotations

import pytest

# enhance extras (numpy / opencv-python-headless) が無い環境では tests/enhance を丸ごと skip。
# pyproject.toml の `[dev]` だけインストールした人が落ちないようにする保険。
np = pytest.importorskip(
    "numpy", reason='numpy is required for video_enhance_lite tests (pip install -e ".[enhance]")'
)
cv2 = pytest.importorskip(
    "cv2",
    reason='opencv-python-headless is required for video_enhance_lite tests (pip install -e ".[enhance]")',
)

# importorskip より後に import する (numpy/cv2 未導入環境では上の importorskip で
# collection が止まり、以下の pipeline 依存 fixture を評価しないため)。
from video_enhance_lite import config  # noqa: E402
from tests.support.process_runner import FakeProcessRunner  # noqa: E402


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """WORKSPACE_ROOT を tmp_path に差し替えた作業ディレクトリ (出力ポリシー検証用)。"""
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def fake_run():
    """V2-8 DI: `run_pipeline(..., runner=fake_run)` へ注入する単一 FakeProcessRunner。

    旧来の monkeypatch による pipeline モジュール属性差し替え方式は廃止。テストは
    `runner=fake_run` で注入し、`fake_run.calls` を読むだけにする。
    """
    return FakeProcessRunner()


@pytest.fixture(scope="session")
def _tk_session_root():
    """プロセスで 1 度だけ生成する共有 Tk root (GUI テスト用)。

    旧来は test_gui*.py の各ファイルが独自の session fixture で `tk.Tk()` を生成して
    いた。同一プロセスで 2 個目以降の `Tk()` 生成は、先行テストの `tk.Variable` GC
    (`__del__` の Tcl コマンド削除) と Tcl 初期化が競合して `init.tcl` ロード失敗 →
    skip になることがあるため、conftest に一本化して `Tk()` を 1 回に固定する。
    tkinter import は遅延させ、headless 環境では skip にする。
    """
    tkinter = pytest.importorskip("tkinter")
    try:
        root = tkinter.Tk()
    except tkinter.TclError as e:
        pytest.skip(f"Tk not available: {e}")
    root.withdraw()
    yield root
    try:
        root.destroy()
    except tkinter.TclError:
        pass


@pytest.fixture
def tk_root(_tk_session_root):
    import tkinter

    top = tkinter.Toplevel(_tk_session_root)
    top.withdraw()
    yield top
    try:
        top.destroy()
    except tkinter.TclError:
        pass
