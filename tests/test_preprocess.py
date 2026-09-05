"""前処理(位相折りたたみ・ビン・ビュー)の検査.

期待値の出所は **解析解**である。実装の出力を写して固定した値は 1 つも無い。
AstroNet の定義(SPEC §5)から手で導ける量だけを期待値にしている。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "training"))

from tl import preprocess as pp  # noqa: E402

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------- 位相折りたたみ


def test_phase_fold_returns_offset_from_transit_centre():
    """T-002 / F-02 —— 周期の整数倍だけ離れた時刻は、同じ位相に畳まれる.

    期待値の出所: 解析解。t = t0 + kP + δ を畳めば δ になる(|δ| < P/2)。
    """
    period, t0 = 3.5, 131.25
    offsets = np.array([-1.2, -0.4, 0.0, 0.4, 1.2])
    for k in (-7, 0, 1, 25):
        folded = pp.phase_fold(t0 + k * period + offsets, period, t0)
        np.testing.assert_allclose(folded, offsets, atol=1e-9)


def test_phase_fold_range_is_half_open_around_zero():
    """T-002 —— 畳んだ位相は [−P/2, P/2) に入る(解析解)."""
    period, t0 = 4.0, 0.0
    rng = np.random.default_rng(20260905)
    t = rng.uniform(-500, 500, 5000)
    folded = pp.phase_fold(t, period, t0)
    assert folded.min() >= -period / 2 - 1e-12
    assert folded.max() < period / 2


# ---------------------------------------------------------------------- ビン


def test_bin_geometry_matches_astronet_definition():
    """T-003 / F-02 —— 最初のビンは x_min から、最後のビンは x_max で終わる.

    期待値の出所: SPEC §5 P-4/P-5 の定義(AstroNet の median_filter と同じ刻み)。
    y = x を入れると、各ビンの中央値はそのビンの中心に一致する(標本が密なら
    刻み幅より小さい誤差で)。
    """
    x_min, x_max, num_bins, width = -4.0, 4.0, 201, 0.16 * 2.0
    step = 1e-3
    x = np.arange(x_min, x_max + step / 2, step)
    out, empty = pp._binned_median(x, x.copy(), num_bins, width, x_min, x_max)

    assert empty == 0
    spacing = (x_max - x_min - width) / (num_bins - 1)
    centres = x_min + spacing * np.arange(num_bins) + width / 2
    np.testing.assert_allclose(out, centres, atol=step)

    # 端の点はそれぞれ最初/最後のビンに入る
    assert out[0] == pytest.approx(x_min + width / 2, abs=step)
    assert out[-1] == pytest.approx(x_max - width / 2, abs=step)


def test_empty_bins_are_reported_not_silently_filled():
    """T-003 —— 点の無いビンは NaN で返り、個数が数えられる(G-02 の土台).

    期待値の出所: 解析解。[0,1] にしか点が無く、窓が [0,2] で 4 ビンなら
    後ろ 2 ビンは必ず空になる。
    """
    # 刻みは (2 − 0 − 0.5)/3 = 0.5 なのでビンは [0,.5] [.5,1] [1,1.5] [1.5,2]。
    # 点を [0, 0.99] にしか置かなければ、後ろ 2 ビンは必ず空になる
    x = np.linspace(0.0, 0.99, 500)
    out, empty = pp._binned_median(x, np.ones_like(x), 4, 0.5, 0.0, 2.0)
    assert empty == 2
    assert np.isnan(out[-2:]).all()
    assert np.isfinite(out[:2]).all()


# ------------------------------------------------------------------ 穴埋め・正規化


def test_fill_gaps_is_exact_linear_interpolation():
    """T-006 / F-02 —— 空ビンは前後から線形に埋まる(解析解)."""
    view = np.array([0.0, np.nan, np.nan, 3.0, np.nan, 5.0])
    filled = pp._fill_gaps(view)
    np.testing.assert_allclose(filled, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


def test_fill_gaps_extends_flat_at_the_edges():
    """T-006 —— 端の空きは端の値で埋める(np.interp の定義どおり)."""
    filled = pp._fill_gaps(np.array([np.nan, 2.0, 4.0, np.nan]))
    np.testing.assert_allclose(filled, [2.0, 2.0, 4.0, 4.0])


def test_normalise_puts_median_at_zero_and_minimum_at_minus_one():
    """T-007 / F-02 —— SPEC §5 P-6 の定義そのもの."""
    rng = np.random.default_rng(1)
    view = rng.normal(1.0, 0.1, 401)
    view[200] = 0.2  # 明確な最小値
    out = pp._normalize(view)
    assert np.median(out) == pytest.approx(0.0, abs=1e-12)
    assert out.min() == pytest.approx(-1.0, abs=1e-12)


# --------------------------------------------------------------------- ビュー


def _box_transit(
    period: float,
    t0: float,
    duration: float,
    depth: float,
    span_days: float = 1400.0,
    cadence: float = 29.4 / 60 / 24,
) -> tuple[np.ndarray, np.ndarray]:
    """解析的な箱形トランジット(雑音なし)を作る."""
    t = np.arange(0.0, span_days, cadence)
    folded = pp.phase_fold(t, period, t0)
    flux = np.ones_like(t)
    flux[np.abs(folded) <= duration / 2] -= depth
    return t, flux


def test_global_view_places_the_transit_at_the_centre_bin():
    """T-004 / F-02 —— 位相 0 は 2001 ビンの中央(添字 1000)に来る.

    期待値の出所: 解析解。ビンは [−P/2, P/2] を等分するので、位相 0 は中央ビン。
    """
    period, t0, duration, depth = 8.0, 133.0, 0.25, 0.01
    t, flux = _box_transit(period, t0, duration, depth)
    folded = pp.phase_fold(t, period, t0)
    view = pp.global_view(folded, flux, period)

    assert view.shape == (pp.GLOBAL_BINS,)
    assert view.min() == pytest.approx(-1.0, abs=1e-12)
    assert np.median(view) == pytest.approx(0.0, abs=1e-12)
    # 箱形なので通過中のビンはすべて同じ深さで並ぶ(argmin は先頭で止まるので使わない)。
    # 最深ビンの並びの中心が、位相 0 のビン(添字 1000)に来ることを見る
    deep = np.flatnonzero(view <= view.min() + 1e-9)
    assert deep.mean() == pytest.approx(pp.GLOBAL_BINS // 2, abs=1.0)
    assert view[pp.GLOBAL_BINS // 2] == pytest.approx(-1.0, abs=1e-12)


def test_local_view_window_is_four_durations_each_side():
    """T-005 / F-02 —— local ビューの窓は ±4 継続時間、201 ビン.

    期待値の出所: SPEC §5 P-5。通過の幅は窓全体の 1/8 なので、
    「−1 に近い」ビンの割合は 1/8 前後になるはずである(解析解)。
    箱形なのでビンの重なり(幅 0.16d、刻み 0.0392d)ぶんだけ縁が滲む。
    """
    period, t0, duration, depth = 20.0, 200.0, 0.4, 0.02
    t, flux = _box_transit(period, t0, duration, depth)
    folded = pp.phase_fold(t, period, t0)
    view = pp.local_view(folded, flux, period, duration)

    assert view.shape == (pp.LOCAL_BINS,)
    assert view.min() == pytest.approx(-1.0, abs=1e-12)
    deep = np.mean(view < -0.5)
    assert abs(deep - 1 / 8) < 0.02, f"深いビンの割合 {deep:.3f}(解析値 0.125)"
    centre = np.flatnonzero(view < -0.5).mean()
    assert centre == pytest.approx(pp.LOCAL_BINS // 2, abs=1.5)


def test_local_view_window_is_clipped_by_half_the_period():
    """T-005 —— 周期が短いと窓は ±P/2 で切られる(SPEC §5 P-5 の min/max)."""
    period, t0, duration, depth = 1.0, 10.0, 0.2, 0.01  # 4d = 0.8 > P/2 = 0.5
    t, flux = _box_transit(period, t0, duration, depth)
    folded = pp.phase_fold(t, period, t0)
    view = pp.local_view(folded, flux, period, duration)
    # 窓が ±0.5 に切られるので、通過(幅 0.2)は窓の 1/5 を占める
    deep = np.mean(view < -0.5)
    assert abs(deep - 1 / 5) < 0.03, f"深いビンの割合 {deep:.3f}(解析値 0.2)"


def test_view_error_when_too_many_bins_are_empty():
    """T-008 / G-02 —— 空ビンが 25% を超えたら黙って埋めずに落とす.

    期待値の出所: SPEC §5「空ビンの扱い」。点を位相の半分にしか置かなければ
    global ビューの空ビンは 50% になる。
    """
    period = 10.0
    folded = np.linspace(-period / 2, 0.0, 4000)  # 位相の前半にしか点が無い
    flux = np.ones_like(folded)
    flux[2000] = 0.99
    with pytest.raises(pp.ViewError):
        pp.global_view(folded, flux, period)


def test_transit_mask_covers_the_transit_and_little_else():
    """T-009 / F-02 —— 覆う範囲は通過中心 ±(継続時間).

    期待値の出所: 解析解。覆われる時間の割合は 2·duration/period。
    """
    period, t0, duration = 12.0, 3.0, 0.3
    t = np.arange(0.0, 1200.0, 29.4 / 60 / 24)
    mask = pp.transit_mask(t, period, t0, duration)
    assert mask.mean() == pytest.approx(2 * duration / period, rel=0.02)
    # 通過中心はすべて覆われている
    centres = t0 + period * np.arange(1, 90)
    centres = centres[centres < t.max()]
    idx = np.searchsorted(t, centres)
    assert mask[idx].all()
