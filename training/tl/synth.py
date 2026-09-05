"""合成トランジットの注入(目玉 1 の測定に使う).

画面のつまみ(`src/lib/synth.ts`)と**同じ物理・同じ前処理**で曲線を作る。
乱数だけは処理系ごとの都合で違う(こちらは numpy、あちらは mulberry32)——
測定に使うのは分布であって特定の系列ではないので、これは問題にならない。
乱数を除いた部分の一致は `tests/physics.test.ts` が確かめる。
"""

from __future__ import annotations

import numpy as np

from . import preprocess as pp
from .physics import (
    BASELINE_DAYS,
    CADENCE_DAYS,
    flux_at_time,
    radius_ratio,
    scaled_semi_major_axis,
    transit_depth,
    transit_duration,
    transit_snr,
)


def synthesize(
    planet_earth_radii: float,
    period_days: float,
    sigma_per_point: float,
    star_solar_radii: float = 1.0,
    impact: float = 0.0,
    seed: int = 20260905,
    dropout: float = 0.08,
) -> dict:
    """つまみの値から曲線とビューを作る."""
    rng = np.random.default_rng(seed)
    k = radius_ratio(planet_earth_radii, star_solar_radii)
    a_over_r = scaled_semi_major_axis(period_days, star_solar_radii)
    depth = transit_depth(planet_earth_radii, star_solar_radii)
    duration = transit_duration(period_days, star_solar_radii, planet_earth_radii, impact)

    n = int(BASELINE_DAYS / CADENCE_DAYS)
    t = np.arange(n) * CADENCE_DAYS
    t = t[rng.random(n) >= dropout]
    folded = pp.phase_fold(t, period_days, 0.0)
    flux = flux_at_time(folded, period_days, k, a_over_r, impact) + rng.normal(
        0.0, sigma_per_point, t.size
    )

    return {
        "depth": depth,
        "duration": duration,
        "snr": transit_snr(depth, sigma_per_point, period_days, duration),
        "global": pp.global_view(folded, flux, period_days),
        "local": pp.local_view(folded, flux, period_days, duration),
    }
