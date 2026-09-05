"""位相折りたたみとビュー生成(AstroNet §3 準拠).

用語:

* **global ビュー** —— 位相の全域を 2001 の等幅ビンに割り、各ビンの中央値を取る
* **local ビュー** —— 通過中心 ±(4 × 継続時間)を 201 ビンに割る。ビン幅は
  0.16 × 継続時間なので、隣り合うビンは重なる(平滑化の効果を持つ)

どちらも最後に「中央値を引いて最小値の絶対値で割る」= 中央値 0・最小値 −1 に揃える。
"""

from __future__ import annotations

import numpy as np

GLOBAL_BINS = 2001
LOCAL_BINS = 201
LOCAL_NUM_DURATIONS = 4
LOCAL_BIN_WIDTH_FACTOR = 0.16
MAX_EMPTY_FRACTION = 0.25


class ViewError(ValueError):
    """ビューを作れなかった(空ビンが多すぎる・点が足りない等)."""


def phase_fold(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """通過中心を 0 とする位相(単位は日、範囲は [-period/2, period/2))を返す."""
    half = period / 2.0
    return np.mod(time - t0 + half, period) - half


def _binned_median(
    x: np.ndarray,
    y: np.ndarray,
    num_bins: int,
    bin_width: float,
    x_min: float,
    x_max: float,
) -> tuple[np.ndarray, int]:
    """AstroNet の median_filter と同じ刻み方でビン中央値を返す.

    ビン i は ``[x_min + i*spacing, x_min + i*spacing + bin_width]`` を覆う。
    ``spacing = (x_max - x_min - bin_width) / (num_bins - 1)`` なので、
    最初のビンは x_min から、最後のビンは x_max で終わる。
    空ビンは NaN で返し、その個数を第二要素で返す。
    """
    order = np.argsort(x)
    xs, ys = x[order], y[order]

    spacing = (x_max - x_min - bin_width) / (num_bins - 1)
    starts = x_min + spacing * np.arange(num_bins)
    ends = starts + bin_width

    lo = np.searchsorted(xs, starts, side="left")
    hi = np.searchsorted(xs, ends, side="right")

    out = np.full(num_bins, np.nan, dtype=np.float64)
    empty = 0
    for i in range(num_bins):
        if hi[i] > lo[i]:
            out[i] = np.median(ys[lo[i] : hi[i]])
        else:
            empty += 1
    return out, empty


def _fill_gaps(view: np.ndarray) -> np.ndarray:
    """空ビン(NaN)を前後の非空ビンから線形補間で埋める.

    参照実装は全体の中央値で埋めるが、通過の近くに連続した空きがあると
    そこだけ平らな床ができて形が変わる。線形補間の方が素直なのでそうする
    (SPEC §5 の「空ビンの扱い」)。
    """
    filled = view.copy()
    bad = np.isnan(filled)
    if not bad.any():
        return filled
    if bad.all():
        raise ViewError("全ビンが空")
    idx = np.arange(filled.size)
    filled[bad] = np.interp(idx[bad], idx[~bad], filled[~bad])
    return filled


def _normalize(view: np.ndarray) -> np.ndarray:
    """中央値 0・最小値 −1 に揃える."""
    out = view - np.median(view)
    depth = np.abs(np.min(out))
    if depth == 0 or not np.isfinite(depth):
        raise ViewError("正規化できない(最小値が 0)")
    return out / depth


def global_view(folded: np.ndarray, flux: np.ndarray, period: float) -> np.ndarray:
    raw, empty = _binned_median(
        folded, flux, GLOBAL_BINS, period / GLOBAL_BINS, -period / 2, period / 2
    )
    if empty > MAX_EMPTY_FRACTION * GLOBAL_BINS:
        raise ViewError(f"global ビューの空ビンが {empty}/{GLOBAL_BINS}")
    return _normalize(_fill_gaps(raw))


def local_view(
    folded: np.ndarray, flux: np.ndarray, period: float, duration: float
) -> np.ndarray:
    t_max = min(period / 2, duration * LOCAL_NUM_DURATIONS)
    t_min = -t_max
    width = duration * LOCAL_BIN_WIDTH_FACTOR
    if t_max - t_min <= width:
        raise ViewError("local ビューの窓がビン幅より狭い")
    sel = (folded >= t_min) & (folded <= t_max)
    if sel.sum() < LOCAL_BINS // 4:
        raise ViewError(f"local ビューの窓に点が {int(sel.sum())} しかない")
    raw, empty = _binned_median(
        folded[sel], flux[sel], LOCAL_BINS, width, t_min, t_max
    )
    if empty > MAX_EMPTY_FRACTION * LOCAL_BINS:
        raise ViewError(f"local ビューの空ビンが {empty}/{LOCAL_BINS}")
    return _normalize(_fill_gaps(raw))


def transit_mask(
    time: np.ndarray, period: float, t0: float, duration: float, factor: float = 1.0
) -> np.ndarray:
    """通過中心 ±(factor × 継続時間/2 × 2) を覆う真偽配列を返す.

    ``factor=1.0`` で継続時間の 2 倍幅(前後にそれぞれ半分ずつの余白)を覆う。
    """
    folded = phase_fold(time, period, t0)
    return np.abs(folded) < duration * factor
