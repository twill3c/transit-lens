/**
 * 位相折りたたみとビュー生成の TypeScript 実装。
 *
 * `training/tl/preprocess.py` と**同じ手続き**でなければならない。
 * 同じでなければ、つまみで作った曲線(ブラウザ側で前処理)と
 * 実データ(Python 側で前処理)が別の物差しでモデルに入ることになる。
 * 二実装照合は `tests/views.test.ts` が Python 由来のフィクスチャで行う。
 */

export const GLOBAL_BINS = 2001;
export const LOCAL_BINS = 201;
export const LOCAL_NUM_DURATIONS = 4;
export const LOCAL_BIN_WIDTH_FACTOR = 0.16;
export const MAX_EMPTY_FRACTION = 0.25;

export class ViewError extends Error {}

/** 通過中心を 0 とする位相(単位は日、範囲は [-period/2, period/2))。 */
export function phaseFold(time: Float64Array, period: number, t0: number): Float64Array {
  const half = period / 2;
  const out = new Float64Array(time.length);
  for (let i = 0; i < time.length; i++) {
    const v = (time[i] - t0 + half) % period;
    out[i] = (v < 0 ? v + period : v) - half;
  }
  return out;
}

function medianOf(values: Float64Array, lo: number, hi: number): number {
  const n = hi - lo;
  const slice = values.slice(lo, hi);
  slice.sort();
  return n % 2 === 1 ? slice[(n - 1) / 2] : (slice[n / 2 - 1] + slice[n / 2]) / 2;
}

/**
 * AstroNet の median_filter と同じ刻み方でビン中央値を返す。
 * ビン i は [xMin + i*spacing, xMin + i*spacing + width] を覆う。
 * 空ビンは NaN。第二要素は空ビンの数。
 */
export function binnedMedian(
  x: Float64Array,
  y: Float64Array,
  numBins: number,
  binWidth: number,
  xMin: number,
  xMax: number,
): { view: Float64Array; empty: number } {
  const order = Array.from(x.keys()).sort((a, b) => x[a] - x[b]);
  const xs = new Float64Array(order.length);
  const ys = new Float64Array(order.length);
  for (let i = 0; i < order.length; i++) {
    xs[i] = x[order[i]];
    ys[i] = y[order[i]];
  }

  const spacing = (xMax - xMin - binWidth) / (numBins - 1);
  const view = new Float64Array(numBins).fill(Number.NaN);
  let empty = 0;
  let lo = 0;
  let hi = 0;
  for (let i = 0; i < numBins; i++) {
    const start = xMin + spacing * i;
    const end = start + binWidth;
    // xs は昇順、start / end も昇順なので走査位置は戻さなくてよい
    while (lo < xs.length && xs[lo] < start) lo++;
    if (hi < lo) hi = lo;
    while (hi < xs.length && xs[hi] <= end) hi++;
    if (hi > lo) view[i] = medianOf(ys, lo, hi);
    else empty++;
  }
  return { view, empty };
}

/** 空ビン(NaN)を前後の非空ビンから線形補間で埋める。端は端の値で伸ばす。 */
export function fillGaps(view: Float64Array): Float64Array {
  const known: number[] = [];
  for (let i = 0; i < view.length; i++) if (Number.isFinite(view[i])) known.push(i);
  if (known.length === 0) throw new ViewError("全ビンが空");
  const out = Float64Array.from(view);
  for (let i = 0; i < view.length; i++) {
    if (Number.isFinite(view[i])) continue;
    if (i < known[0]) {
      out[i] = view[known[0]];
      continue;
    }
    if (i > known[known.length - 1]) {
      out[i] = view[known[known.length - 1]];
      continue;
    }
    let k = 0;
    while (known[k + 1] < i) k++;
    const a = known[k];
    const b = known[k + 1];
    out[i] = view[a] + ((view[b] - view[a]) * (i - a)) / (b - a);
  }
  return out;
}

/** 中央値 0・最小値 −1 に揃える(SPEC §5 P-6)。 */
export function normalizeView(view: Float64Array): Float64Array {
  const sorted = Float64Array.from(view).sort();
  const n = sorted.length;
  const median = n % 2 === 1 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  const out = Float64Array.from(view, (v) => v - median);
  let min = Infinity;
  for (const v of out) if (v < min) min = v;
  const depth = Math.abs(min);
  if (!(depth > 0) || !Number.isFinite(depth)) throw new ViewError("正規化できない(最小値が 0)");
  return Float64Array.from(out, (v) => v / depth);
}

export function globalView(folded: Float64Array, flux: Float64Array, period: number): Float64Array {
  const { view, empty } = binnedMedian(
    folded,
    flux,
    GLOBAL_BINS,
    period / GLOBAL_BINS,
    -period / 2,
    period / 2,
  );
  if (empty > MAX_EMPTY_FRACTION * GLOBAL_BINS) {
    throw new ViewError(`global ビューの空ビンが ${empty}/${GLOBAL_BINS}`);
  }
  return normalizeView(fillGaps(view));
}

export function localView(
  folded: Float64Array,
  flux: Float64Array,
  period: number,
  duration: number,
): Float64Array {
  const tMax = Math.min(period / 2, duration * LOCAL_NUM_DURATIONS);
  const tMin = -tMax;
  const width = duration * LOCAL_BIN_WIDTH_FACTOR;
  if (tMax - tMin <= width) throw new ViewError("local ビューの窓がビン幅より狭い");

  const idx: number[] = [];
  for (let i = 0; i < folded.length; i++) {
    if (folded[i] >= tMin && folded[i] <= tMax) idx.push(i);
  }
  if (idx.length < LOCAL_BINS / 4) throw new ViewError(`local ビューの窓に点が ${idx.length} しかない`);
  const fx = Float64Array.from(idx, (i) => folded[i]);
  const fy = Float64Array.from(idx, (i) => flux[i]);
  const { view, empty } = binnedMedian(fx, fy, LOCAL_BINS, width, tMin, tMax);
  if (empty > MAX_EMPTY_FRACTION * LOCAL_BINS) {
    throw new ViewError(`local ビューの空ビンが ${empty}/${LOCAL_BINS}`);
  }
  return normalizeView(fillGaps(view));
}
