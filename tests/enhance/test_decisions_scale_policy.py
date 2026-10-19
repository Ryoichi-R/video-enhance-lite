from __future__ import annotations

from video_enhance_lite import config, decisions

# FakeMeta / FakeStats を再利用
from tests.enhance.test_decisions_resolution_preset import FakeMeta, FakeStats, _upscale


def _rec(width, target_width, policy=None):
    defaults = {"target_width": target_width, "target_width_explicit": True}
    if policy is not None:
        defaults["upscale_scale_policy"] = policy
    return _upscale(
        decisions.recommend(FakeMeta(width=width, height=480), FakeStats(), defaults=defaults)
    )


# ---------- cover (既定) は従来挙動 ----------


def test_848_to_1920_cover_default_is_scale_3():
    """既定 (cover): 848->1920 は needed=ceil(2.264)=3 -> scale=3 (従来挙動・不変)。"""
    up = _rec(848, 1920)  # policy 未指定 = cover
    assert up.params["scale"] == 3


def test_848_to_1920_cover_explicit_is_scale_3():
    up = _rec(848, 1920, policy="cover")
    assert up.params["scale"] == 3


# ---------- nearest は target に最も近い scale ----------


def test_848_to_1920_nearest_is_scale_2():
    """nearest: 848->1920 は scale=2(1696, |224|) が scale=3(2544, |624|) より近い -> scale=2。"""
    up = _rec(848, 1920, policy="nearest")
    assert up.params["scale"] == 2
    assert "nearest policy" in up.reason
    assert "上方 lanczos" in up.reason  # 1696 < 1920


def test_nearest_picks_smaller_when_equidistant_prefers_lower_compute():
    """等距離なら小さい scale (計算量が軽い) を選ぶ。"""
    # 384*2=768, 384*4=1536 は target 1152 から等距離 (384)、384*3=1152 が完全一致。
    up = _rec(384, 1152, policy="nearest")
    assert up.params["scale"] == 3  # ぴったり一致


def test_nearest_downscale_direction_reported():
    """中間幅が target を上回るケースは「下方 lanczos」と報告。"""
    # 1000*2=2000 > 1920 (|80|) < 1000*1 不可。2 が最近。下方 lanczos。
    up = _rec(1000, 1920, policy="nearest")
    assert up.params["scale"] == 2
    assert "下方 lanczos" in up.reason


def test_nearest_clamps_to_4_when_far_above():
    """target が 4x をはるかに超える場合は scale=4 (cover と一致)。"""
    up = _rec(400, 3840, policy="nearest")  # 400*4=1600 が最近
    assert up.params["scale"] == 4


# ---------- 既定値は config ----------


def test_default_policy_constant():
    assert config.DEFAULT_UPSCALE_SCALE_POLICY == "cover"
    assert set(config.UPSCALE_SCALE_POLICIES) == {"cover", "nearest"}
