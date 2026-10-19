from __future__ import annotations

from fractions import Fraction


from video_enhance_lite import pipeline


def test_default_signature_equivalent_to_old():
    """intermediate_width=None / effective_fps=None なら旧シグネチャと同等"""
    baseline_new = pipeline.estimate_peak_temp_bytes(1920, 1080, Fraction(30, 1), 10)
    baseline_keyword = pipeline.estimate_peak_temp_bytes(
        1920,
        1080,
        Fraction(30, 1),
        10,
        intermediate_width=None,
        effective_fps=None,
    )
    assert baseline_new == baseline_keyword


def test_intermediate_width_4x_factor():
    """中間幅 3840 (= 1920×2) は 1920×1080 baseline の 4倍 (面積 4倍)"""
    baseline = pipeline.estimate_peak_temp_bytes(1920, 1080, Fraction(30, 1), 10)
    big = pipeline.estimate_peak_temp_bytes(
        1920,
        1080,
        Fraction(30, 1),
        10,
        intermediate_width=3840,
    )
    # AR 維持 -> 3840×2160 ≈ baseline × 4
    assert big >= baseline * 4 - 1
    assert big <= baseline * 4 + 1


def test_effective_fps_doubles_peak():
    """effective_fps 60 は source 30 の 2倍 peak"""
    base = pipeline.estimate_peak_temp_bytes(1920, 1080, Fraction(30, 1), 10)
    doubled = pipeline.estimate_peak_temp_bytes(
        1920,
        1080,
        Fraction(30, 1),
        10,
        effective_fps=Fraction(60, 1),
    )
    assert doubled >= base * 2 - 1
    assert doubled <= base * 2 + 1


def test_intermediate_and_effective_fps_compound():
    """1080p 30fps + intermediate 3840(=4K) + effective 60fps -> baseline の 8倍 (面積4倍 × fps2倍)"""
    base = pipeline.estimate_peak_temp_bytes(1920, 1080, Fraction(30, 1), 10)
    compound = pipeline.estimate_peak_temp_bytes(
        1920,
        1080,
        Fraction(30, 1),
        10,
        intermediate_width=3840,
        effective_fps=Fraction(60, 1),
    )
    assert compound > base * 4  # 少なくとも 4倍は超える (実際は ~8倍)
    assert compound >= base * 8 - 8
    assert compound <= base * 8 + 8


def test_zero_width_returns_zero():
    """ゼロ除算を防ぐ"""
    assert pipeline.estimate_peak_temp_bytes(0, 1080, Fraction(30, 1), 10) == 0


# ---------- PR4: png_compression_ratio / worker_count ----------
#
# プラン §4.3 の test 要件:
#   - `png_compression_ratio=1.0` 明示指定で旧式の戻り値と完全一致
#   - 既定 (未指定) で旧値の 0.80 倍
#   - `png_compression_ratio=0.5` で旧値の 0.50 倍


def _legacy_value(width=1920, height=1080, fps=Fraction(30, 1), chunk_seconds=10) -> int:
    """旧シグネチャ相当 (= 圧縮係数 1.0) の値。"""
    return pipeline.estimate_peak_temp_bytes(
        width,
        height,
        fps,
        chunk_seconds,
        png_compression_ratio=1.0,
    )


def test_png_compression_ratio_1_0_equivalent_to_legacy():
    """`png_compression_ratio=1.0` 明示が旧式の戻り値と完全一致。"""
    legacy = _legacy_value()
    explicit = pipeline.estimate_peak_temp_bytes(
        1920,
        1080,
        Fraction(30, 1),
        10,
        png_compression_ratio=1.0,
    )
    assert explicit == legacy


def test_default_now_applies_80_percent():
    """関数 default が 0.80 になった: 未指定で旧値の 0.80 倍 (PR4 plan-gap 修正)。"""
    legacy = _legacy_value()
    default_call = pipeline.estimate_peak_temp_bytes(1920, 1080, Fraction(30, 1), 10)
    assert abs(default_call - int(legacy * 0.80)) < 2


def test_png_compression_ratio_0_80_returns_80_percent():
    legacy = _legacy_value()
    ratio = pipeline.estimate_peak_temp_bytes(
        1920,
        1080,
        Fraction(30, 1),
        10,
        png_compression_ratio=0.80,
    )
    assert abs(ratio - int(legacy * 0.80)) < 2


def test_png_compression_ratio_0_50_halves_peak():
    legacy = _legacy_value()
    half = pipeline.estimate_peak_temp_bytes(
        1920,
        1080,
        Fraction(30, 1),
        10,
        png_compression_ratio=0.50,
    )
    assert abs(half - int(legacy * 0.50)) < 2


def test_worker_count_multiplies_peak():
    """同じ係数のもとで worker_count=4 は 4 倍 (default 0.80 で比較)。"""
    single = pipeline.estimate_peak_temp_bytes(1920, 1080, Fraction(30, 1), 10)
    quad = pipeline.estimate_peak_temp_bytes(
        1920,
        1080,
        Fraction(30, 1),
        10,
        worker_count=4,
    )
    assert quad == single * 4


def test_estimate_compression_ratio_anime_low_065():
    """histogram_flat_ratio > 0.40 でアニメ判定 -> 0.65"""
    from types import SimpleNamespace

    s = SimpleNamespace(histogram_flat_ratio=0.50)
    assert pipeline._estimate_compression_ratio(s) == 0.65


def test_estimate_compression_ratio_default_080():
    from types import SimpleNamespace

    s = SimpleNamespace(histogram_flat_ratio=0.20)
    assert pipeline._estimate_compression_ratio(s) == pipeline.DEFAULT_PNG_COMPRESSION_RATIO
    assert pipeline._estimate_compression_ratio(s) == 0.80


def test_estimate_compression_ratio_boundary_at_threshold():
    """境界値 (0.40) でも default 側 (0.80)"""
    from types import SimpleNamespace

    s = SimpleNamespace(histogram_flat_ratio=0.40)
    assert pipeline._estimate_compression_ratio(s) == 0.80
