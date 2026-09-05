/** 画面に出す数の書き方。専門語には必ず言い換えを添える(SPEC §7 の取り決め)。 */

/** 深さ。1% 以上ならパーセント、下なら ppm。 */
export function fmtDepth(fraction: number): string {
  if (!Number.isFinite(fraction)) return "—";
  return fraction >= 0.01
    ? `${(fraction * 100).toFixed(1)}%`
    : `${Math.round(fraction * 1e6).toLocaleString("ja-JP")} ppm`;
}

/** 深さの言い換え。「明るさが 0.0084% 下がる」 */
export function fmtDepthPlain(fraction: number): string {
  if (!Number.isFinite(fraction)) return "—";
  const pct = fraction * 100;
  const digits = pct >= 1 ? 2 : pct >= 0.01 ? 3 : 4;
  return `${pct.toFixed(digits)}%`;
}

/** 周期。1 日未満は時間で。 */
export function fmtPeriod(days: number): string {
  if (!Number.isFinite(days)) return "—";
  if (days < 1) return `${(days * 24).toFixed(1)} 時間`;
  return days < 10 ? `${days.toFixed(2)} 日` : `${days.toFixed(1)} 日`;
}

/** 継続時間(日 → 時間)。 */
export function fmtHours(days: number): string {
  if (!Number.isFinite(days)) return "—";
  return `${(days * 24).toFixed(1)} 時間`;
}

export function fmtPercent(x: number, digits = 0): string {
  if (!Number.isFinite(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}

/** 惑星の大きさの言い換え。 */
export function sizeWord(earthRadii: number): string {
  if (earthRadii < 1.25) return "地球ほど";
  if (earthRadii < 2.0) return "スーパーアース級";
  if (earthRadii < 3.6) return "ミニ海王星級";
  if (earthRadii < 8) return "海王星ほど";
  return "木星ほど";
}
