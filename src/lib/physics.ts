/**
 * トランジットの物理。
 *
 * 形は**一つの式**から出す —— 一様輝度の円盤を円が覆うときの面積比(Mandel & Agol 2002 の
 * 一様光源解)。惑星のように小さい相手なら底が平らな窪みになり、伴星のように大きい相手なら
 * 隠れ切る前にすれ違って V 字になる。画面の「なぜ間違えるのか」で並べる二つの形は、
 * 別々の作り物ではなく**同じ式の両端**である。
 */

/** 太陽半径 ÷ 地球半径。IAU 公称値 R_sun = 6.957e8 m, R_earth = 6.3781e6 m より。 */
export const R_SUN_IN_EARTH = 6.957e8 / 6.3781e6; // = 109.077...
/** 1 天文単位 ÷ 太陽半径。 */
export const AU_IN_R_SUN = 1.495978707e11 / 6.957e8; // = 215.032...
/** Kepler 長時間ケイデンス(日)。29.4244 分。 */
export const CADENCE_DAYS = 29.4244 / 60 / 24;
/** Kepler の観測期間(Q1–Q17、日)。 */
export const BASELINE_DAYS = 1459.5;
/** Kepler ミッション自身の検出しきい値(MES)。 */
export const KEPLER_MES_THRESHOLD = 7.1;

/** 半径比 k = Rp/Rs。惑星半径は地球半径、恒星半径は太陽半径で与える。 */
export function radiusRatio(planetEarthRadii: number, starSolarRadii: number): number {
  return planetEarthRadii / (R_SUN_IN_EARTH * starSolarRadii);
}

/** 窪みの深さ(= k²)。 */
export function transitDepth(planetEarthRadii: number, starSolarRadii: number): number {
  const k = radiusRatio(planetEarthRadii, starSolarRadii);
  return k * k;
}

/** 軌道長半径 ÷ 恒星半径。ケプラーの第三法則から。 */
export function scaledSemiMajorAxis(
  periodDays: number,
  starSolarRadii: number,
  starSolarMasses = 1,
): number {
  const aAu = Math.cbrt(starSolarMasses * (periodDays / 365.25) ** 2);
  return (aAu * AU_IN_R_SUN) / starSolarRadii;
}

/**
 * 通過継続時間(日)。中心通過(衝突係数 b)での第一〜第四接触の間隔。
 * T = (P/π) · asin( sqrt((1+k)² − b²) / (a/Rs) )。
 */
export function transitDuration(
  periodDays: number,
  starSolarRadii: number,
  planetEarthRadii: number,
  impact = 0,
  starSolarMasses = 1,
): number {
  const k = radiusRatio(planetEarthRadii, starSolarRadii);
  const aOverR = scaledSemiMajorAxis(periodDays, starSolarRadii, starSolarMasses);
  const inner = Math.sqrt(Math.max(0, (1 + k) ** 2 - impact * impact)) / aOverR;
  if (inner >= 1) return periodDays / 2;
  return (periodDays / Math.PI) * Math.asin(inner);
}

/**
 * 一様光源を円が覆うときの減光率(Mandel & Agol 2002, eq. 1)。
 * `z` は恒星半径を単位とする投影距離、`k` は半径比。
 */
export function occultedFraction(z: number, k: number): number {
  const az = Math.abs(z);
  if (az >= 1 + k) return 0;
  if (k >= 1 && az <= k - 1) return 1;
  if (az <= 1 - k) return k * k;
  const k0 = Math.acos(clamp((k * k + az * az - 1) / (2 * k * az)));
  const k1 = Math.acos(clamp((1 - k * k + az * az) / (2 * az)));
  const area = Math.sqrt(Math.max(0, 4 * az * az - (1 + az * az - k * k) ** 2)) / 2;
  return (k * k * k0 + k1 - area) / Math.PI;
}

function clamp(v: number): number {
  return v < -1 ? -1 : v > 1 ? 1 : v;
}

/**
 * 通過中心からの時間(日)における相対フラックス(1 が通過外)。
 * 円軌道・一様光源。
 */
export function fluxAtTime(
  tDays: number,
  periodDays: number,
  k: number,
  aOverR: number,
  impact = 0,
): number {
  const phase = (2 * Math.PI * tDays) / periodDays;
  const x = aOverR * Math.sin(phase);
  const y = impact * Math.cos(phase);
  const z = Math.hypot(x, y);
  // 恒星の裏側(cos phase < 0)は二次食。ここでは一次通過だけを見る
  if (Math.cos(phase) < 0) return 1;
  return 1 - occultedFraction(z, k);
}

/** 4 年間の通過回数。 */
export function transitCount(periodDays: number, baselineDays = BASELINE_DAYS): number {
  return Math.max(1, Math.floor(baselineDays / periodDays));
}

/** 通過中に落ちる観測点の数。 */
export function inTransitPoints(
  periodDays: number,
  durationDays: number,
  baselineDays = BASELINE_DAYS,
): number {
  return transitCount(periodDays, baselineDays) * Math.max(2, durationDays / CADENCE_DAYS);
}

/**
 * 信号対雑音比 —— 深さ ÷(1 点あたりのばらつき ÷ √通過中の点数)。
 * Kepler の MES と同じ数え方(積み上げた通過の総和で測る)。
 */
export function transitSnr(
  depth: number,
  sigmaPerPoint: number,
  periodDays: number,
  durationDays: number,
  baselineDays = BASELINE_DAYS,
): number {
  const n = inTransitPoints(periodDays, durationDays, baselineDays);
  return depth / (sigmaPerPoint / Math.sqrt(n));
}
