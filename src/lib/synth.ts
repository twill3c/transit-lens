/**
 * つまみから合成の光度曲線を作る(画面「自分で惑星を置いてみる」)。
 *
 * **合成であることを隠さない。** ただし作り方は本物の観測に寄せる:
 *
 * * 標本の刻みは Kepler の長時間ケイデンス(29.42 分)、期間は Q1–Q17 の 1459.5 日
 * * 窪みの形は一様光源の掩蔽解(`physics.ts`)。台形の作り物ではない
 * * 出来た曲線は、実データとまったく同じ手続き(`views.ts`)でビューにする
 *
 * したがって、つまみの判定は「別の式で作った近似」ではなく、
 * **実データと同じモデルに、同じ形の入力を与えた結果**である。
 */

import {
  BASELINE_DAYS,
  CADENCE_DAYS,
  fluxAtTime,
  radiusRatio,
  scaledSemiMajorAxis,
  transitDepth,
  transitDuration,
  transitSnr,
} from "./physics";
import { globalView, localView, phaseFold } from "./views";

/** 32 bit の決定論的乱数(mulberry32)。同じ種は同じ曲線を返す。 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 標準正規乱数(Box–Muller)。 */
export function gaussian(rand: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export type SynthParams = {
  /** 惑星半径(地球半径) */
  planetEarthRadii: number;
  /** 公転周期(日) */
  periodDays: number;
  /** 1 点あたりのばらつき(相対) */
  sigmaPerPoint: number;
  /** 恒星半径(太陽半径) */
  starSolarRadii?: number;
  /** 衝突係数 */
  impact?: number;
  seed?: number;
  /** 観測の欠測率。Kepler の実測稼働率に近い 0.08 を既定にする */
  dropout?: number;
};

export type SynthResult = {
  depth: number;
  durationDays: number;
  snr: number;
  transits: number;
  /** 折りたたんだ位相(日)と相対フラックス。散布図の描画に使う */
  folded: Float64Array;
  flux: Float64Array;
  globalView: Float64Array;
  localView: Float64Array;
};

/**
 * つまみの値から曲線とビューを作る。
 *
 * 点の数は 1459.5 / 0.020434 ≈ 71,400。折りたたみと中央値ビンはこの規模なら
 * ブラウザで 20 ms 前後で終わる(つまみは入力ごとに再計算してよい)。
 */
export function synthesize(params: SynthParams): SynthResult {
  const {
    planetEarthRadii,
    periodDays,
    sigmaPerPoint,
    starSolarRadii = 1,
    impact = 0,
    seed = 20260905,
    dropout = 0.08,
  } = params;

  const k = radiusRatio(planetEarthRadii, starSolarRadii);
  const aOverR = scaledSemiMajorAxis(periodDays, starSolarRadii);
  const depth = transitDepth(planetEarthRadii, starSolarRadii);
  const durationDays = transitDuration(periodDays, starSolarRadii, planetEarthRadii, impact);

  const rand = mulberry32(seed);
  const n = Math.floor(BASELINE_DAYS / CADENCE_DAYS);
  const times: number[] = [];
  const fluxes: number[] = [];
  for (let i = 0; i < n; i++) {
    if (rand() < dropout) continue;
    const t = i * CADENCE_DAYS;
    // 通過中心を t0 = 0 に置く。位相は折りたたみ側で合わせる
    let tt = ((t + periodDays / 2) % periodDays) - periodDays / 2;
    if (tt < -periodDays / 2) tt += periodDays;
    times.push(t);
    fluxes.push(fluxAtTime(tt, periodDays, k, aOverR, impact) + gaussian(rand) * sigmaPerPoint);
  }

  const time = Float64Array.from(times);
  const flux = Float64Array.from(fluxes);
  const folded = phaseFold(time, periodDays, 0);

  return {
    depth,
    durationDays,
    snr: transitSnr(depth, sigmaPerPoint, periodDays, durationDays),
    transits: Math.max(1, Math.floor(BASELINE_DAYS / periodDays)),
    folded,
    flux,
    globalView: globalView(folded, flux, periodDays),
    localView: localView(folded, flux, periodDays, durationDays),
  };
}
