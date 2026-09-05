"""スプライン除去の検査.

**この検査の主張**: 通過を覆ってスプラインを当てると、低周波の変動だけが落ち、
通過の深さは保たれる。覆わないと深さが浅く出る。
後者は「覆う意味がある」ことの陽性対照であり、覆う実装を弱めたときに落ちる。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "training"))

from tl import preprocess as pp  # noqa: E402
from tl import spline as sp  # noqa: E402

pytestmark = pytest.mark.unit

CADENCE = 29.4 / 60 / 24  # Kepler 長時間ケイデンス(日)


def _synthetic(
    span: float = 400.0,
    trend_period: float = 18.0,
    trend_amp: float = 0.02,
    period: float = 6.0,
    t0: float = 2.0,
    duration: float = 0.22,
    depth: float = 0.004,
    noise: float = 2e-4,
    seed: int = 20260905,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """低周波変動 × 箱形トランジット + ガウス雑音。真の変動も返す."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, span, CADENCE)
    trend = 1.0 + trend_amp * np.sin(2 * np.pi * t / trend_period)
    folded = pp.phase_fold(t, period, t0)
    transit = np.where(np.abs(folded) <= duration / 2, 1.0 - depth, 1.0)
    flux = trend * transit + rng.normal(0.0, noise, t.size)
    return t, flux, trend, folded


def test_split_segments_breaks_on_gaps():
    """T-010 —— 0.75 日以上の空きで区間が割れる(解析解)."""
    t = np.concatenate([np.arange(0, 5, 0.02), np.arange(9, 12, 0.02)])
    segs = sp.split_segments(t)
    assert len(segs) == 2
    assert t[segs[0]].max() < 5.0
    assert t[segs[1]].min() >= 9.0


def test_spline_recovers_the_slow_trend():
    """T-011 / F-02 —— 当てたスプラインは真の低周波変動に一致する.

    期待値の出所: 合成データの真値。許容は雑音水準 2e-4 の 3 倍。
    """
    t, flux, trend, folded = _synthetic()
    mask = np.abs(folded) <= 0.22
    fit = sp.fit_spline(t, flux, mask=mask)
    ok = np.isfinite(fit) & ~mask
    assert ok.mean() > 0.9
    assert np.max(np.abs(fit[ok] - trend[ok])) < 6e-4


def _measured_depth(flux: np.ndarray, fit: np.ndarray, in_transit: np.ndarray) -> float:
    ok = np.isfinite(fit) & (fit != 0)
    rel = flux[ok] / fit[ok]
    return float(1.0 - np.median(rel[in_transit[ok]]))


def test_deep_transits_survive_even_without_masking():
    """T-012 / F-02 —— 深い通過は 3σ クリップだけで落ちるので、覆っても覆わなくても同じ.

    期待値の出所: 合成データの真値(深さ 0.004 = 雑音 2e-4 の 20 倍)。
    これは AstroNet が覆わずに済ませている理由の実測である。
    """
    depth = 0.004
    t, flux, _, folded = _synthetic(depth=depth)
    in_transit = np.abs(folded) <= 0.22 / 2 * 0.6  # 底の中央 60% だけを見る

    d_masked = _measured_depth(flux, sp.fit_spline(t, flux, mask=np.abs(folded) <= 0.22), in_transit)
    d_unmasked = _measured_depth(flux, sp.fit_spline(t, flux, mask=None), in_transit)

    assert d_masked == pytest.approx(depth, rel=0.06), f"覆った場合 {d_masked:.5f}"
    assert d_unmasked == pytest.approx(depth, rel=0.06), f"覆わない場合 {d_unmasked:.5f}"


def test_shallow_long_transits_are_absorbed_unless_masked():
    """T-012b / F-02 / SPEC §5 P-2 —— 覆う実装の存在理由を測る.

    浅くて長い通過(1 点あたり 1.3σ・継続時間 0.6 日)は 3σ クリップに掛からず、
    変動の速い星ではスプラインが通過を追いかけて深さを食う。
    **これが「覆う」実装の陽性対照**である。差が出なければ SPEC §5 P-2 の
    逸脱には根拠が無いことになる。

    期待値の出所: 合成データの真値と、覆う/覆わないの相対比較。
    2026-09-05 に 4 通りの組み合わせで実測した(真の深さ 4e-4・雑音 3e-4):

    | 変動の周期 | 継続時間 | 覆った | 覆わない |
    |---|---|---|---|
    | 2.5 日 | 0.60 日 | +52% | **−103%(通過が消える)** |
    | 8.0 日 | 0.60 日 | +4% | −34% |
    | 2.5 日 | 0.15 日 | −21% | −57% |
    | 18.0 日 | 0.25 日 | −2% | −11% |

    4 通りすべてで覆った方が真値に近い。本ケースは 2 行目(8.0 日 / 0.60 日)を使う。
    """
    depth, noise, duration, period = 4e-4, 3e-4, 0.6, 40.0
    t, flux, _, folded = _synthetic(
        span=400.0,
        trend_period=8.0,
        trend_amp=0.01,
        period=period,
        t0=5.0,
        duration=duration,
        depth=depth,
        noise=noise,
    )
    in_transit = np.abs(folded) <= duration / 2 * 0.6

    d_masked = _measured_depth(
        flux, sp.fit_spline(t, flux, mask=np.abs(folded) <= duration), in_transit
    )
    d_unmasked = _measured_depth(flux, sp.fit_spline(t, flux, mask=None), in_transit)

    assert d_masked == pytest.approx(depth, rel=0.25), f"覆った場合 {d_masked:.6f}"
    assert d_unmasked < d_masked * 0.9, (
        f"陽性対照が働いていない: 覆わない {d_unmasked:.6f} / 覆った {d_masked:.6f}"
    )


def test_spline_returns_nan_where_it_cannot_fit():
    """T-013 / G-02 —— 当てられない区間は NaN を返す(黙って値を作らない).

    期待値の出所: 実装契約(SPEC §5 P-2)。点が 8 個未満の区間は当てられない。
    """
    t = np.concatenate([np.arange(0, 30, CADENCE), np.array([100.0, 100.02, 100.04])])
    flux = np.ones_like(t)
    flux += np.linspace(0, 0.01, t.size)
    fit = sp.fit_spline(t, flux)
    assert np.isnan(fit[-3:]).all()
    assert np.isfinite(fit[:-3]).all()
