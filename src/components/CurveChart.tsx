"use client";

import { useMemo } from "react";

export type CurveMode = "local" | "global" | "raw";

export type CurveChartProps = {
  /** 縦軸の値の列(ビュー、または生の相対フラックス) */
  values: number[] | Float64Array;
  /** 横軸の値の列。省略時は等間隔 */
  x?: number[] | Float64Array;
  /** 横軸の目盛りに添える単位 */
  xUnit: string;
  /** 横軸の左端・右端の値 */
  xMin: number;
  xMax: number;
  /** モデルの視線(values と同じ長さに引き伸ばした CAM)。null なら重ねない */
  gaze: Float64Array | null;
  /** 点で描くか(生データ)、線で描くか(ビュー) */
  scatter?: boolean;
  yLabel: string;
  height?: number;
  /** 通過の位置を示す縦線を引く位置(横軸の値)。null なら引かない */
  markerX?: number | null;
};

const W = 620;
const L = 60;
const R = 14;
const T = 20;
const B = 42;

/**
 * 折りたたんだ曲線と「モデルの視線」を重ねる図。
 *
 * 視線は CAM(SPEC §6.1 H-A)。頭部が大域平均プーリング + 全結合 1 層なので、
 * ロジットは各位置の寄与の平均に**厳密に**分解できる。近似ではない。
 * ただし枝の出力長は入力より短いので、描くときには線形に引き伸ばしている
 * —— そのことは画面の注記に書く(G-09)。
 */
export function CurveChart({
  values,
  x,
  xUnit,
  xMin,
  xMax,
  gaze,
  scatter = false,
  yLabel,
  height = 236,
  markerX = null,
}: CurveChartProps) {
  const H = height;
  const geom = useMemo(() => {
    const n = values.length;
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!Number.isFinite(lo)) {
      lo = 0;
      hi = 1;
    }
    const pad = (hi - lo) * 0.12 || 0.1;
    lo -= pad;
    hi += pad;
    const xs = (i: number) => (x ? x[i] : xMin + ((xMax - xMin) * i) / (n - 1));
    const X = (v: number) => L + ((v - xMin) / (xMax - xMin)) * (W - L - R);
    const Y = (v: number) => T + ((hi - v) / (hi - lo)) * (H - T - B);
    return { n, lo, hi, xs, X, Y };
  }, [values, x, xMin, xMax, H]);

  const path = useMemo(() => {
    if (scatter) return "";
    let d = "";
    for (let i = 0; i < geom.n; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) continue;
      d += `${d ? "L" : "M"}${geom.X(geom.xs(i)).toFixed(1)} ${geom.Y(v).toFixed(1)}`;
    }
    return d;
  }, [values, geom, scatter]);

  const gazePath = useMemo(() => {
    if (!gaze || gaze.length === 0) return "";
    let max = 0;
    for (const v of gaze) if (v > max) max = v;
    if (!(max > 0)) return "";
    const base = H - B;
    const top = T + 4;
    let d = `M${L} ${base}`;
    for (let i = 0; i < gaze.length; i++) {
      const u = i / (gaze.length - 1);
      const px = L + u * (W - L - R);
      const strength = Math.max(0, gaze[i]) / max;
      d += `L${px.toFixed(1)} ${(base - strength * (base - top)).toFixed(1)}`;
    }
    d += `L${W - R} ${base}Z`;
    return d;
  }, [gaze, H]);

  const ticks = [xMin, (xMin + xMax) / 2, xMax];

  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="光度曲線">
      <defs>
        <linearGradient id="gazeG" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--gaze)" stopOpacity="0.30" />
          <stop offset="100%" stopColor="var(--gaze)" stopOpacity="0.03" />
        </linearGradient>
      </defs>

      {gazePath && <path d={gazePath} fill="url(#gazeG)" />}
      {gazePath && (
        <text
          x={L + 6}
          y={T + 11}
          fill="var(--gaze)"
          fontSize="10.5"
          fontFamily="var(--f-mono)"
          letterSpacing=".06em"
        >
          モデルの視線
        </text>
      )}

      <line x1={L} y1={geom.Y(0)} x2={W - R} y2={geom.Y(0)} stroke="var(--line)" strokeWidth="1" />
      <text
        x={L - 9}
        y={geom.Y(0) + 4}
        fill="var(--ink-3)"
        fontSize="10.5"
        textAnchor="end"
        fontFamily="var(--f-mono)"
      >
        0
      </text>
      <text
        x={L - 9}
        y={T - 6}
        fill="var(--ink-3)"
        fontSize="10"
        textAnchor="end"
        fontFamily="var(--f-mono)"
        letterSpacing=".06em"
      >
        {yLabel}
      </text>

      {markerX !== null && markerX >= xMin && markerX <= xMax && (
        <line
          x1={geom.X(markerX)}
          y1={T}
          x2={geom.X(markerX)}
          y2={H - B}
          stroke="var(--gaze)"
          strokeWidth="1.5"
          strokeDasharray="3 3"
          opacity="0.8"
        />
      )}

      {scatter
        ? Array.from({ length: geom.n }, (_, i) => {
            const v = values[i];
            if (!Number.isFinite(v)) return null;
            return (
              <circle
                key={i}
                cx={geom.X(geom.xs(i)).toFixed(1)}
                cy={geom.Y(v).toFixed(1)}
                r="1.1"
                fill="var(--star)"
                opacity="0.45"
              />
            );
          })
        : (
            <path
              d={path}
              fill="none"
              stroke="var(--star)"
              strokeWidth="1.7"
              strokeLinejoin="round"
            />
          )}

      {ticks.map((tv, i) => (
        <g key={i}>
          <line x1={geom.X(tv)} y1={H - B} x2={geom.X(tv)} y2={H - B + 4} stroke="var(--line)" />
          <text
            x={geom.X(tv)}
            y={H - B + 18}
            fill="var(--ink-3)"
            fontSize="10.5"
            textAnchor="middle"
            fontFamily="var(--f-mono)"
          >
            {formatTick(tv)}
          </text>
        </g>
      ))}
      <text
        x={(L + W - R) / 2}
        y={H - 6}
        fill="var(--ink-3)"
        fontSize="10.5"
        textAnchor="middle"
      >
        {xUnit}
      </text>
    </svg>
  );
}

function formatTick(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}
