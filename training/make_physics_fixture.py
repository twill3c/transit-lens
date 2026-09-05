"""物理側の二実装照合フィクスチャ(Python が権威).

乱数を含まない量 —— 深さ・継続時間・掩蔽の減光率・MES —— だけを並べる。
乱数の系列は処理系ごとに違ってよい(測定に使うのは分布であって系列ではない)。

出力: ``tests/fixtures/physics.json``
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tl import physics as ph  # noqa: E402

CASES = [
    # 地球・太陽。継続時間 13 時間ほど、深さ 84 ppm という教科書の値が出るはず
    dict(rp=1.0, rs=1.0, period=365.25, sigma=100e-6),
    dict(rp=1.0, rs=1.0, period=1.0, sigma=100e-6),
    dict(rp=11.2, rs=1.0, period=3.5, sigma=200e-6),  # 木星ほど
    dict(rp=2.4, rs=0.47, period=129.9, sigma=700e-6),  # 暗い星をまわる小さい惑星
    dict(rp=60.0, rs=1.0, period=2.41, sigma=200e-6),  # 半径比 0.55 = 伴星に近い
]

Z_GRID = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0, 1.05, 1.2, 1.5, 2.0]
K_GRID = [0.01, 0.05, 0.1, 0.3, 0.55, 0.9, 1.2]


def main() -> int:
    cases = []
    for c in CASES:
        k = ph.radius_ratio(c["rp"], c["rs"])
        a_over_r = ph.scaled_semi_major_axis(c["period"], c["rs"])
        duration = ph.transit_duration(c["period"], c["rs"], c["rp"])
        depth = ph.transit_depth(c["rp"], c["rs"])
        times = list(np.linspace(-duration, duration, 21))
        cases.append(
            {
                **c,
                "k": k,
                "a_over_r": a_over_r,
                "depth": depth,
                "duration_days": duration,
                "snr": ph.transit_snr(depth, c["sigma"], c["period"], duration),
                "transit_count": ph.transit_count(c["period"]),
                "times": times,
                "flux": [
                    float(v) for v in ph.flux_at_time(np.array(times), c["period"], k, a_over_r)
                ],
            }
        )

    occult = [
        {"k": k, "z": Z_GRID, "f": [float(v) for v in ph.occulted_fraction(np.array(Z_GRID), k)]}
        for k in K_GRID
    ]

    out = pathlib.Path("tests/fixtures/physics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "note": "Python 側(training/tl/physics.py)が権威。乱数を含まない量だけを並べてある。",
                "constants": {
                    "R_SUN_IN_EARTH": ph.R_SUN_IN_EARTH,
                    "AU_IN_R_SUN": ph.AU_IN_R_SUN,
                    "CADENCE_DAYS": ph.CADENCE_DAYS,
                    "BASELINE_DAYS": ph.BASELINE_DAYS,
                },
                "cases": cases,
                "occultation": occult,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"→ {out} ({out.stat().st_size:,} バイト)")
    for c in cases:
        print(
            f"   Rp {c['rp']:5.1f} R⊕ / P {c['period']:7.2f} 日 → "
            f"深さ {c['depth'] * 1e6:8.1f} ppm / 継続 {c['duration_days'] * 24:5.2f} 時間 / "
            f"MES {c['snr']:8.1f}σ"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
