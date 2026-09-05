"use client";

/** 判定ゲージ。確率をそのまま円弧の長さにする。 */
export function Gauge({ prob }: { prob: number | null }) {
  const c = 2 * Math.PI * 37;
  const p = prob ?? 0;
  const colour = prob === null ? "var(--ink-3)" : p >= 0.5 ? "var(--yes)" : "var(--no)";
  return (
    <div className="verdict">
      <div className="gauge">
        <svg viewBox="0 0 88 88" aria-hidden="true">
          <circle cx="44" cy="44" r="37" fill="none" stroke="var(--line)" strokeWidth="7" />
          <circle
            cx="44"
            cy="44"
            r="37"
            fill="none"
            stroke={colour}
            strokeWidth="7"
            strokeLinecap="round"
            strokeDasharray={c.toFixed(1)}
            strokeDashoffset={(c * (1 - p)).toFixed(1)}
          />
        </svg>
        <span className="val" style={{ color: colour }}>
          {prob === null ? "—" : `${Math.round(p * 100)}%`}
        </span>
      </div>
      <div>
        <div className="lbl">モデルの判定</div>
        <div className="word" style={{ color: colour }}>
          {prob === null ? "計算中" : p >= 0.5 ? "惑星の可能性" : "惑星ではなさそう"}
        </div>
      </div>
    </div>
  );
}
