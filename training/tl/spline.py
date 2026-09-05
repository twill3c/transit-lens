"""基底スプラインによる低周波変動の除去.

Vanderburg & Johnson (2014) / Shallue & Vanderburg (2018) の手続きに合わせる:

* 光度曲線を連続区間に切る(欠測で分ける)
* 各区間に等間隔ノットの三次 B スプラインを当てる
* 3 シグマ外れ値を落として当て直す(収束するまで)
* ノット間隔の候補を複数試し、BIC が最小のものを採る

本実装は参照実装から一点だけ外れる —— **当該 TCE の通過部分を覆ってから当てる**。
参照実装は覆わずシグマクリップだけに任せるが、深い通過はスプラインを引き込み、
通過の深さを浅く見せる方向に働く。覆う方が素直なので覆う。
この差は SPEC §5 P-2 に明記してある。
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import LSQUnivariateSpline

# ノット間隔の候補(日). AstroNet の kepler_spline は 0.5〜20 日を対数で刻む
DEFAULT_SPACINGS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0)


def split_segments(time: np.ndarray, gap_width: float = 0.75) -> list[np.ndarray]:
    """欠測(既定 0.75 日以上の空き)で連続区間の添字に切り分ける."""
    if time.size == 0:
        return []
    breaks = np.where(np.diff(time) > gap_width)[0]
    bounds = [0, *(breaks + 1), time.size]
    return [np.arange(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _fit_one(
    time: np.ndarray,
    flux: np.ndarray,
    keep: np.ndarray,
    spacing: float,
    max_iter: int = 5,
    sigma: float = 3.0,
) -> tuple[np.ndarray, float, int] | None:
    """1 区間に 1 つのノット間隔でスプラインを当て、(値, 残差二乗和, 自由度) を返す.

    ``keep`` は当てはめに使う点(通過を覆った後の点)の真偽配列.
    当てはめ自体は ``keep`` の点だけで行い、値は全点で評価する.
    """
    span = time[-1] - time[0]
    if span <= spacing:
        return None

    for _ in range(max_iter):
        if keep.sum() < 8:
            return None
        # ノットは区間の内側にのみ置く。両端のノットは splrep が自動で持つ
        n_knots = int(span / spacing)
        if n_knots < 1:
            return None
        knots = np.linspace(time[0], time[-1], n_knots + 2)[1:-1]
        # 使う点が無い区間にノットを置くと LSQUnivariateSpline が落ちる。
        # ノットの間に必ず 1 点以上あることを確かめてから落とす
        t_keep = time[keep]
        if t_keep.size < 4:
            return None
        counts, _ = np.histogram(t_keep, bins=np.concatenate([[time[0]], knots, [time[-1]]]))
        knots = knots[(counts[:-1] > 0) & (counts[1:] > 0)]
        if knots.size == 0:
            # ノットを 1 本も置けない = 直線当てはめに相当
            knots = np.array([])

        try:
            spline = LSQUnivariateSpline(t_keep, flux[keep], knots, k=3, ext="const")
        except (ValueError, TypeError):
            return None

        model_keep = spline(t_keep)
        resid = flux[keep] - model_keep
        scatter = np.std(resid)
        if scatter == 0 or not np.isfinite(scatter):
            break
        new_keep = keep.copy()
        new_keep[keep] = np.abs(resid) < sigma * scatter
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    values = spline(time)
    resid = flux[keep] - spline(time[keep])
    sse = float(np.sum(resid**2))
    dof = int(knots.size + 4)
    return values, sse, dof


def fit_spline(
    time: np.ndarray,
    flux: np.ndarray,
    mask: np.ndarray | None = None,
    spacings: tuple[float, ...] = DEFAULT_SPACINGS,
) -> np.ndarray:
    """BIC でノット間隔を選び、全点で評価したスプラインの値を返す.

    ``mask`` が真の点は当てはめから除く(通過部分を覆う用途).
    どのノット間隔でも当てられなかった区間は NaN を返す —— 呼び出し側で捨てる.
    """
    out = np.full(time.size, np.nan, dtype=np.float64)
    if mask is None:
        mask = np.zeros(time.size, dtype=bool)

    for idx in split_segments(time):
        t_seg, f_seg = time[idx], flux[idx]
        keep0 = ~mask[idx]
        if keep0.sum() < 8:
            continue

        best = None
        best_bic = np.inf
        n = int(keep0.sum())
        for spacing in spacings:
            got = _fit_one(t_seg, f_seg, keep0.copy(), spacing)
            if got is None:
                continue
            values, sse, dof = got
            if sse <= 0:
                continue
            # 正規誤差を仮定した BIC。AstroNet の kepler_spline と同じ形
            bic = n * np.log(sse / n) + dof * np.log(n)
            if bic < best_bic:
                best_bic, best = bic, values
        if best is not None:
            out[idx] = best
    return out
