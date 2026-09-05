"use client";

import { useMemo } from "react";
import {
  fluxAtTime,
  radiusRatio,
  scaledSemiMajorAxis,
  transitDuration,
} from "@/lib/physics";
import { gaussian, mulberry32 } from "@/lib/synth";
import { fmtDepth } from "@/lib/format";

/**
 * 「なぜ間違えるのか」—— 惑星の平底と食連星の V 字を並べる。
 *
 * **二つの絵は同じ式から出ている。** 一様光源の掩蔽解に、半径比 k だけを変えて通す。
 * 惑星のように小さい相手なら完全に前へ入りきるので底が平らになり、
 * 伴星のように大きい相手なら隠れ切る前にすれ違うので底が尖る。
 * 別々に作った作り物を並べているのではない。
 */
type Shape = {
  title: string;
  why: React.ReactNode;
  planetEarthRadii: number;
  starSolarRadii: number;
  periodDays: number;
  colour: string;
  seed: number;
};

const SHAPES: Shape[] = [
  {
    title: "惑星が通ったとき",
    why: (
      <>
        惑星は星よりずっと小さいので、完全に前に入りきる。だから底が<b>平ら</b>で、
        入口と出口が同じ角度になる。
      </>
    ),
    planetEarthRadii: 10,
    starSolarRadii: 1,
    periodDays: 8,
    colour: "var(--star)",
    seed: 5,
  },
  {
    title: "星がもう一つ隠れたとき",
    why: (
      <>
        相手も星なので大きく、隠れ切る前にすれ違う。底が尖った<b>V 字</b>になり、
        深さも桁違いになる。
      </>
    ),
    planetEarthRadii: 60,
    starSolarRadii: 1,
    periodDays: 2.41,
    colour: "var(--no)",
    seed: 8,
  },
];

const W = 420;
const H = 168;
const L = 52;
const R = 12;
const T = 16;
const B = 34;

function build(shape: Shape) {
  const k = radiusRatio(shape.planetEarthRadii, shape.starSolarRadii);
  const aOverR = scaledSemiMajorAxis(shape.periodDays, shape.starSolarRadii);
  const duration = transitDuration(
    shape.periodDays,
    shape.starSolarRadii,
    shape.planetEarthRadii,
  );
  const span = duration * 1.9;
  const rand = mulberry32(shape.seed);
  const n = 220;
  const model: { x: number; y: number }[] = [];
  const points: { x: number; y: number }[] = [];
  const depth = k * k;
  for (let i = 0; i < n; i++) {
    const x = -span + (2 * span * i) / (n - 1);
    const y = fluxAtTime(x, shape.periodDays, k, aOverR);
    model.push({ x, y });
    points.push({ x, y: y + gaussian(rand) * depth * 0.05 });
  }
  return { model, points, depth, span, duration };
}

export function ShapePair() {
  const built = useMemo(() => SHAPES.map(build), []);

  return (
    <section className="step">
      <div className="step-head">
        <span className="step-n">出力 4</span>
        <h2>なぜ間違えるのかを見せる</h2>
        <span className="io">OUT: 並べるだけ</span>
      </div>
      <p className="note">
        誤検出のほとんどは食連星 —— 惑星ではなく、二つの星が互いを隠している。
        窪みの形が違う。文章で説明せず、隣に置く。
      </p>
      <div className="compare">
        {SHAPES.map((shape, i) => {
          const b = built[i];
          const lo = 1 - b.depth * 1.35;
          const hi = 1 + b.depth * 0.35;
          const X = (x: number) => L + ((x + b.span) / (2 * b.span)) * (W - L - R);
          const Y = (y: number) => T + ((hi - y) / (hi - lo)) * (H - T - B);
          let d = "";
          for (const p of b.model) d += `${d ? "L" : "M"}${X(p.x).toFixed(1)} ${Y(p.y).toFixed(1)}`;
          return (
            <div className="cmp" key={shape.title}>
              <h4>{shape.title}</h4>
              <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={shape.title}>
                <line x1={L} y1={Y(1)} x2={W - R} y2={Y(1)} stroke="var(--line)" />
                <text
                  x={L - 8}
                  y={Y(1) + 4}
                  fill="var(--ink-3)"
                  fontSize="10"
                  textAnchor="end"
                  fontFamily="var(--f-mono)"
                >
                  0
                </text>
                <text
                  x={L - 8}
                  y={Y(1 - b.depth) + 4}
                  fill="var(--ink-3)"
                  fontSize="10"
                  textAnchor="end"
                  fontFamily="var(--f-mono)"
                >
                  −{fmtDepth(b.depth)}
                </text>
                {b.points.map((p, j) => (
                  <circle
                    key={j}
                    cx={X(p.x).toFixed(1)}
                    cy={Y(p.y).toFixed(1)}
                    r="1.4"
                    fill={shape.colour}
                    opacity=".45"
                  />
                ))}
                <path
                  d={d}
                  fill="none"
                  stroke={shape.colour}
                  strokeWidth="2.2"
                  strokeLinejoin="round"
                />
              </svg>
              <p className="why">{shape.why}</p>
            </div>
          );
        })}
      </div>
      <p className="foot">
        この二つは同じ式(一様光源の掩蔽解)に半径比だけを変えて通したもので、
        形を作り分けてはいない。左は半径比 0.09、右は 0.55。
      </p>
    </section>
  );
}
