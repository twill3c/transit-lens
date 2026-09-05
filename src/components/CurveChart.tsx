"use client";

import { useMemo } from "react";

export type CurveChartProps = {
  /** 縦軸の値の列(ビュー、または生の相対フラックス) */
  values: number[] | Float64Array;
  /** 横軸の値の列。省略時は等間隔 */
  x?: number[] | Float64Array;
  /** 横軸に添える説明 */
  xUnit: string;
  /** 横軸の左端・右端の値 */
  xMin: number;
  xMax: number;
  /** モデルの視線(CAM)。null なら重ねない */
  gaze: Float64Array | null;
  /** 点で描くか(生データ)、線で描くか(ビュー) */
  scatter?: boolean;
  yLabel: string;
  height?: number;
  /** 通過の位置を示す縦線。毎フレーム動く */
  markerX?: number | null;
};

const W = 620;
const L = 60;
const R = 14;
const T = 20;
const B = 42;

function formatTick(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

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

  // 図の中身は**印の位置に依存しない**。印は毎フレーム動くので、分けておかないと
  // 1,400 個の点を毎フレーム作り直すことになる
  const body = useMemo(() => {
    let path = "";
    if (!scatter) {
      for (let i = 0; i < geom.n; i++) {
        const v = values[i];
        if (!Number.isFinite(v)) continue;
        path += `${path ? "L" : "M"}${geom.X(geom.xs(i)).toFixed(1)} ${geom.Y(v).toFixed(1)}`;
      }
    }

    // 視線は **中央値からの隔たり**([-1,1] に正規化済み)。符号で色を分ける。
    // 上向き(藍)= 惑星である側へ、下向き(朱)= 惑星でない側へ動かした位置。
    // 正の寄与だけを描くと、local 枝の「減点で効く」働き方がまったく見えない
    let gazeUp = "";
    let gazeDown = "";
    if (gaze && gaze.length > 1) {
      const base = H - B;
      const span = (base - (T + 4)) * 0.62;
      let up = `M${L} ${base}`;
      let down = `M${L} ${base}`;
      for (let i = 0; i < gaze.length; i++) {
        const px = (L + (i / (gaze.length - 1)) * (W - L - R)).toFixed(1);
        up += `L${px} ${(base - Math.max(0, gaze[i]) * span).toFixed(1)}`;
        down += `L${px} ${(base + Math.max(0, -gaze[i]) * span * 0.42).toFixed(1)}`;
      }
      gazeUp = `${up}L${W - R} ${base}Z`;
      gazeDown = `${down}L${W - R} ${base}Z`;
    }
    const gazePath = gazeUp;

    const ticks = [xMin, (xMin + xMax) / 2, xMax];

    return (
      <>
        <defs>
          <linearGradient id="gazeG" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--gaze)" stopOpacity="0.30" />
            <stop offset="100%" stopColor="var(--gaze)" stopOpacity="0.03" />
          </linearGradient>
          <linearGradient id="gazeD" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--no)" stopOpacity="0.28" />
            <stop offset="100%" stopColor="var(--no)" stopOpacity="0.03" />
          </linearGradient>
        </defs>

        {gazeDown && <path d={gazeDown} fill="url(#gazeD)" />}
        {gazePath && <path d={gazePath} fill="url(#gazeG)" />}
        {gazePath && (
          <>
            <text
              x={L + 6}
              y={T + 11}
              fill="var(--gaze)"
              fontSize="10.5"
              fontFamily="var(--f-mono)"
              letterSpacing=".06em"
            >
              モデルの視線 ▲惑星の側へ
            </text>
            <text
              x={W - R - 6}
              y={T + 11}
              fill="var(--no)"
              fontSize="10.5"
              textAnchor="end"
              fontFamily="var(--f-mono)"
              letterSpacing=".06em"
            >
              ▼惑星でない側へ
            </text>
          </>
        )}

        <line
          x1={L}
          y1={geom.Y(0)}
          x2={W - R}
          y2={geom.Y(0)}
          stroke="var(--line)"
          strokeWidth="1"
        />
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

        {scatter ? (
          Array.from({ length: geom.n }, (_, i) => {
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
        ) : (
          <path d={path} fill="none" stroke="var(--star)" strokeWidth="1.7" strokeLinejoin="round" />
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
        <text x={(L + W - R) / 2} y={H - 6} fill="var(--ink-3)" fontSize="10.5" textAnchor="middle">
          {xUnit}
        </text>
      </>
    );
  }, [values, geom, gaze, scatter, yLabel, xUnit, xMin, xMax, H]);

  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="光度曲線">
      {body}
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
    </svg>
  );
}
