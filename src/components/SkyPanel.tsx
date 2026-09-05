"use client";

import { useMemo } from "react";
import { occultedFraction } from "@/lib/physics";

export type SkyPanelProps = {
  /** 通過の位相。半継続時間を単位とし、|phase| < 1 が通過中 */
  phase: number;
  /** 半径比 k = Rp/Rs。これが実際の比 */
  ratio: number;
  /** 見た目の種(背景の星の散らばりを固定するため) */
  seed: number;
  /** 相手が恒星(食連星)なら true。描画の比率を実寸に近づける */
  companionIsStar: boolean;
};

const VW = 420;
const VH = 244;
const CX = 210;
const CY = 118;
const STAR_R = 66;

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * 「起きていること」の模式図。
 *
 * **誇張は明記する(SPEC G-09)。** 地球サイズの窪みは実寸で描くと画面上で
 * 0.5 ピクセルにもならない。だから惑星だけは拡大して描き、その旨を図の中に書く。
 * 伴星(食連星)は拡大しない —— 実寸に近い比率で十分見えるからである。
 */
export function SkyPanel({ phase, ratio, seed, companionIsStar }: SkyPanelProps) {
  const field = useMemo(() => {
    const r = rng(seed * 7 + 3);
    return Array.from({ length: 34 }, () => ({
      cx: r() * VW,
      cy: r() * VH,
      r: 0.5 + r() * 1.1,
      o: 0.1 + r() * 0.22,
    }));
  }, [seed]);

  const drawR = companionIsStar
    ? STAR_R * Math.min(0.9, Math.max(0.25, ratio))
    : Math.max(7, STAR_R * Math.pow(Math.min(ratio, 0.5), 0.45) * 0.62);
  const px = CX + phase * (STAR_R + drawR);
  // 星の減光は実際の掩蔽解で出す。ただし見えるように誇張した半径比を使う
  const shownRatio = drawR / STAR_R;
  const z = Math.abs(phase) * (1 + shownRatio);
  const dim = occultedFraction(z, shownRatio);

  return (
    <svg viewBox={`0 0 ${VW} ${VH}`} role="img" aria-label="惑星が恒星の前を横切る様子">
      <defs>
        <radialGradient id="starG" cx="42%" cy="38%" r="68%">
          <stop offset="0%" stopColor="#fff6dd" />
          <stop offset="52%" stopColor="#f0b854" />
          <stop offset="100%" stopColor="#b4711c" />
        </radialGradient>
        <radialGradient id="glowG" cx="50%" cy="50%" r="50%">
          <stop offset="55%" stopColor="#e0a94a" stopOpacity=".30" />
          <stop offset="100%" stopColor="#e0a94a" stopOpacity="0" />
        </radialGradient>
        <clipPath id="frameClip">
          <rect x="0" y="0" width={VW} height={VH} />
        </clipPath>
      </defs>
      <g clipPath="url(#frameClip)">
        {field.map((s, i) => (
          <circle
            key={i}
            cx={s.cx.toFixed(1)}
            cy={s.cy.toFixed(1)}
            r={s.r.toFixed(2)}
            fill="#c6cddc"
            opacity={s.o.toFixed(2)}
          />
        ))}
        <circle
          cx={CX}
          cy={CY}
          r={STAR_R * 2.05}
          fill="url(#glowG)"
          opacity={Math.max(0, 1 - dim * 2.1).toFixed(3)}
        />
        <circle
          cx={CX}
          cy={CY}
          r={STAR_R}
          fill="url(#starG)"
          opacity={(1 - dim).toFixed(3)}
        />
        <circle
          cx={px.toFixed(1)}
          cy={(CY - drawR * 0.34).toFixed(1)}
          r={drawR.toFixed(1)}
          fill={companionIsStar ? "#c98a3c" : "#0c0f16"}
          stroke={companionIsStar ? "none" : "#1d2432"}
          strokeWidth="1"
        />
        <text x="14" y="232" fill="var(--sky-ink-2)" fontSize="10.5" fontFamily="var(--f-mono)">
          {companionIsStar ? "伴星は実寸に近い比率" : "惑星は見やすさのため拡大して表示"}
        </text>
      </g>
    </svg>
  );
}
