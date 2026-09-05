"""二実装照合のフィクスチャを作る(Python 側が権威).

ブラウザ側(`src/lib/views.ts`)は、つまみで作った曲線を**実データとまったく同じ手続き**で
ビューにしなければならない。同じでなければ、合成の曲線と実データが別の物差しで
モデルに入ることになる。

ここで作るのは「入力(時刻・フラックス)」と「Python が出したビュー」の対である。
配列は base64 の float64 で書く —— 十進で丸めて書くと、丸めのぶんだけ差が出て
**照合が緩む**(何桁まで一致すべきかが分からなくなる)。

出力: ``tests/fixtures/views.json``
"""

from __future__ import annotations

import base64
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tl import preprocess as pp  # noqa: E402

CADENCE = 29.4244 / 60 / 24


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).decode()


def make_case(
    name: str,
    period: float,
    duration: float,
    depth: float,
    noise: float,
    n_points: int,
    seed: int,
    dropout: float = 0.08,
) -> dict:
    """箱形でない、縁の丸い窪みを持つ合成曲線(形は照合の本題ではないので簡素でよい)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_points) * CADENCE
    keep = rng.random(n_points) >= dropout
    t = t[keep]
    folded = pp.phase_fold(t, period, 0.0)
    x = np.abs(folded) / (duration / 2)
    shape = np.where(x >= 1.0, 0.0, np.sqrt(np.clip(1.0 - x**4, 0.0, None)))
    flux = 1.0 - depth * shape + rng.normal(0.0, noise, t.size)

    gv = pp.global_view(folded, flux, period)
    lv = pp.local_view(folded, flux, period, duration)
    return {
        "name": name,
        "period": period,
        "duration": duration,
        "depth": depth,
        "n": int(t.size),
        "time": b64(t),
        "flux": b64(flux),
        "global": b64(gv),
        "local": b64(lv),
    }


CASES = [
    # 周期が短く、窓が ±P/2 で切られる場合(local ビューの min/max 分岐)
    dict(name="short-period", period=0.84, duration=0.08, depth=1.5e-4, noise=1.6e-4, n_points=8000, seed=11),
    # ふつうの場合
    dict(name="typical", period=9.6, duration=0.21, depth=8.0e-4, noise=2.4e-4, n_points=8000, seed=23),
    # 周期が長く、global ビューに空ビンが多く出る場合(穴埋めの経路)
    dict(name="long-period", period=88.0, duration=0.55, depth=2.6e-3, noise=3.0e-4, n_points=8000, seed=31),
    # 窪みが雑音に埋もれる場合(最小値が雑音の尖りになり、正規化の意味が変わる)
    dict(name="buried", period=14.2, duration=0.18, depth=3.0e-5, noise=6.0e-4, n_points=8000, seed=47),
]


def main() -> int:
    out = pathlib.Path("tests/fixtures/views.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": (
            "Python 側(training/tl/preprocess.py)が権威。配列は base64 の float64 リトルエンディアン。"
            "十進で丸めて書くと照合が緩むので、丸めていない。"
        ),
        "generated_by": "training/make_fixtures.py",
        "cases": [make_case(**c) for c in CASES],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"→ {out} ({out.stat().st_size:,} バイト) 事例 {len(CASES)} 件")
    for c in payload["cases"]:
        print(f"   {c['name']:14s} 点 {c['n']:5d}  周期 {c['period']:7.2f} 日")
    return 0


if __name__ == "__main__":
    sys.exit(main())
