"""トランジットの物理(`src/lib/physics.ts` と同じ式).

窪みの形は一様輝度円盤の掩蔽解(Mandel & Agol 2002, eq. 1)から出す。
惑星のように小さい相手なら底が平らに、伴星のように大きい相手なら V 字になる ——
**同じ式の両端**であって、作り分けではない。

TypeScript 側との一致は `tests/physics.test.ts` が
`tests/fixtures/physics.json`(このモジュールが権威)で確かめる。
"""

from __future__ import annotations

import numpy as np

R_SUN_IN_EARTH = 6.957e8 / 6.3781e6
AU_IN_R_SUN = 1.495978707e11 / 6.957e8
CADENCE_DAYS = 29.4244 / 60 / 24
BASELINE_DAYS = 1459.5
KEPLER_MES_THRESHOLD = 7.1


def radius_ratio(planet_earth_radii: float, star_solar_radii: float) -> float:
    return planet_earth_radii / (R_SUN_IN_EARTH * star_solar_radii)


def transit_depth(planet_earth_radii: float, star_solar_radii: float) -> float:
    k = radius_ratio(planet_earth_radii, star_solar_radii)
    return k * k


def scaled_semi_major_axis(
    period_days: float, star_solar_radii: float, star_solar_masses: float = 1.0
) -> float:
    a_au = np.cbrt(star_solar_masses * (period_days / 365.25) ** 2)
    return a_au * AU_IN_R_SUN / star_solar_radii


def transit_duration(
    period_days: float,
    star_solar_radii: float,
    planet_earth_radii: float,
    impact: float = 0.0,
    star_solar_masses: float = 1.0,
) -> float:
    k = radius_ratio(planet_earth_radii, star_solar_radii)
    a_over_r = scaled_semi_major_axis(period_days, star_solar_radii, star_solar_masses)
    inner = np.sqrt(max(0.0, (1 + k) ** 2 - impact**2)) / a_over_r
    if inner >= 1:
        return period_days / 2
    return (period_days / np.pi) * np.arcsin(inner)


def occulted_fraction(z: np.ndarray | float, k: float) -> np.ndarray:
    """一様光源の減光率。``z`` は恒星半径を単位とする投影距離."""
    z = np.abs(np.asarray(z, dtype=np.float64))
    out = np.zeros_like(z)

    full = z <= 1 - k
    out[full] = k * k
    if k >= 1:
        out[z <= k - 1] = 1.0

    edge = (~full) & (z < 1 + k) & (z > np.abs(1 - k))
    if np.any(edge):
        ze = z[edge]
        k0 = np.arccos(np.clip((k * k + ze**2 - 1) / (2 * k * ze), -1, 1))
        k1 = np.arccos(np.clip((1 - k * k + ze**2) / (2 * ze), -1, 1))
        area = np.sqrt(np.maximum(0.0, 4 * ze**2 - (1 + ze**2 - k * k) ** 2)) / 2
        out[edge] = (k * k * k0 + k1 - area) / np.pi
    return out


def flux_at_time(
    t_days: np.ndarray | float,
    period_days: float,
    k: float,
    a_over_r: float,
    impact: float = 0.0,
) -> np.ndarray:
    """通過中心からの時間における相対フラックス(1 が通過外)."""
    phase = 2 * np.pi * np.asarray(t_days, dtype=np.float64) / period_days
    x = a_over_r * np.sin(phase)
    y = impact * np.cos(phase)
    z = np.hypot(x, y)
    flux = 1.0 - occulted_fraction(z, k)
    # 恒星の裏側は二次食。一次通過だけを見る
    return np.where(np.cos(phase) < 0, 1.0, flux)


def transit_count(period_days: float, baseline_days: float = BASELINE_DAYS) -> int:
    return max(1, int(baseline_days // period_days))


def in_transit_points(
    period_days: float, duration_days: float, baseline_days: float = BASELINE_DAYS
) -> float:
    return transit_count(period_days, baseline_days) * max(2.0, duration_days / CADENCE_DAYS)


def transit_snr(
    depth: float,
    sigma_per_point: float,
    period_days: float,
    duration_days: float,
    baseline_days: float = BASELINE_DAYS,
) -> float:
    n = in_transit_points(period_days, duration_days, baseline_days)
    return depth / (sigma_per_point / np.sqrt(n))
