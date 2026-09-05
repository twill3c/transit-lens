"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Gauge } from "./Gauge";
import { fmtDepth } from "@/lib/format";
import { BASELINE_DAYS, CADENCE_DAYS } from "@/lib/physics";
import { predict, type Prediction } from "@/lib/model";
import { gaussian, mulberry32 } from "@/lib/synth";
import { globalView, localView, phaseFold } from "@/lib/views";

/**
 * 曲線を手で描く。
 *
 * 描いた形は**窪みの形そのもの**として扱い、そこから 4 年ぶんの光度曲線を組み立てる
 * (周期ごとに同じ形を置き、ばらつきを載せる)。そのうえで実データとまったく同じ手続きで
 * ビューを作ってモデルに通す。**描いた線を直接モデルに入れているのではない。**
 * V 字を描くと「惑星ではない」と言われる、という発見がこの画面の狙い。
 */

const N = 61; // 制御点の数
const SPAN = 1.5; // 継続時間を単位とする描画範囲(±)
const MAX_DEPTH = 0.03; // 画面の底 = 3% の減光
const PERIOD = 4.2; // 日。描く人には見せないが、曲線の組み立てに要る
const DURATION = 0.18; // 日(4.3 時間)
const SIGMA = 1.6e-4;

const W = 620;
const H = 230;
const L = 58;
const R = 14;
const T = 18;
const B = 40;

type Preset = { name: string; make: (i: number) => number };

const x_of = (i: number) => (-SPAN + (2 * SPAN * i) / (N - 1)) as number;

const PRESETS: Preset[] = [
  {
    name: "平底(惑星)",
    make: (i) => {
      const x = Math.abs(x_of(i));
      if (x >= 0.5) return 0;
      if (x <= 0.32) return 0.004;
      return (0.004 * (0.5 - x)) / 0.18;
    },
  },
  {
    name: "V 字(食連星)",
    make: (i) => {
      const x = Math.abs(x_of(i));
      return x >= 0.5 ? 0 : 0.026 * (1 - x / 0.5);
    },
  },
  {
    name: "まっすぐ(何も無い)",
    make: () => 0,
  },
];

function buildViews(profile: number[]) {
  const rand = mulberry32(4242);
  const n = Math.floor(BASELINE_DAYS / CADENCE_DAYS);
  const times: number[] = [];
  const fluxes: number[] = [];
  for (let i = 0; i < n; i++) {
    if (rand() < 0.08) continue;
    const t = i * CADENCE_DAYS;
    let tt = ((t + PERIOD / 2) % PERIOD) - PERIOD / 2;
    if (tt < -PERIOD / 2) tt += PERIOD;
    // 描いた形は継続時間を単位とする位置で定義されている
    const u = tt / DURATION;
    let dip = 0;
    if (Math.abs(u) < SPAN) {
      const p = ((u + SPAN) / (2 * SPAN)) * (N - 1);
      const a = Math.floor(p);
      const b = Math.min(N - 1, a + 1);
      dip = profile[a] + (profile[b] - profile[a]) * (p - a);
    }
    times.push(t);
    fluxes.push(1 - dip + gaussian(rand) * SIGMA);
  }
  const time = Float64Array.from(times);
  const flux = Float64Array.from(fluxes);
  const folded = phaseFold(time, PERIOD, 0);
  return {
    depth: Math.max(...profile),
    globalView: globalView(folded, flux, PERIOD),
    localView: localView(folded, flux, PERIOD, DURATION),
  };
}

export function DrawPad() {
  const [profile, setProfile] = useState<number[]>(() =>
    Array.from({ length: N }, (_, i) => PRESETS[0].make(i)),
  );
  const [pred, setPred] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const drawing = useRef(false);
  const lastIdx = useRef<number | null>(null);
  const seq = useRef(0);

  // なぞっている最中に 71,400 点を畳み直すと、線が指に付いてこない。
  // 描画は即時、組み立ては手を止めてから
  const [applied, setApplied] = useState<number[]>(profile);
  useEffect(() => {
    const t = setTimeout(() => setApplied(profile), 140);
    return () => clearTimeout(t);
  }, [profile]);

  const built = useMemo(() => {
    try {
      return buildViews(applied);
    } catch {
      return null;
    }
  }, [applied]);

  useEffect(() => {
    if (!built) return;
    const mine = ++seq.current;
    setBusy(true);
    const timer = setTimeout(() => {
      predict(Float32Array.from(built.globalView), Float32Array.from(built.localView))
        .then((p) => {
          if (seq.current === mine) {
            setPred(p);
            setBusy(false);
          }
        })
        .catch(() => seq.current === mine && setBusy(false));
    }, 150);
    return () => clearTimeout(timer);
  }, [built]);

  const X = (i: number) => L + (i / (N - 1)) * (W - L - R);
  const Y = (d: number) => T + (d / MAX_DEPTH) * (H - T - B);

  function apply(clientX: number, clientY: number) {
    const svg = svgRef.current;
    if (!svg) return;
    const box = svg.getBoundingClientRect();
    const px = ((clientX - box.left) / box.width) * W;
    const py = ((clientY - box.top) / box.height) * H;
    const i = Math.round(((px - L) / (W - L - R)) * (N - 1));
    if (i < 0 || i >= N) return;
    const d = Math.min(MAX_DEPTH, Math.max(0, ((py - T) / (H - T - B)) * MAX_DEPTH));
    setProfile((prev) => {
      const next = [...prev];
      const from = lastIdx.current ?? i;
      const lo = Math.min(from, i);
      const hi = Math.max(from, i);
      if (hi === lo) {
        next[i] = d;
      } else {
        const dFrom = next[from];
        for (let k = lo; k <= hi; k++) {
          const u = from === i ? 1 : (k - from) / (i - from);
          next[k] = dFrom + (d - dFrom) * u;
        }
      }
      return next;
    });
    lastIdx.current = i;
  }

  const path = profile.map((d, i) => `${i ? "L" : "M"}${X(i).toFixed(1)} ${Y(d).toFixed(1)}`).join("");
  const area = `M${X(0).toFixed(1)} ${Y(0).toFixed(1)}${path.slice(1)}L${X(N - 1).toFixed(1)} ${Y(0).toFixed(1)}Z`;

  return (
    <section className="step">
      <div className="step-head">
        <span className="step-n">入力 3</span>
        <h2>曲線を手描きする</h2>
        <span className="io">IN: なぞる → OUT: 同じモデルの判定</span>
      </div>
      <p className="note">
        窪みの形をなぞると、その形で 4 年ぶんの観測を組み立て直し、
        実データと同じ手続きでモデルに通す。<b>描いた線をそのままモデルに入れてはいない。</b>
        V 字を深く描いてみてほしい —— 信号は強いのに、モデルは惑星と呼ばなくなる。
      </p>

      <div className="lab">
        <div className="dials">
          <div className="dial">
            <label>形をえらぶ</label>
            <div className="toggles" style={{ marginTop: 6 }}>
              {PRESETS.map((p) => (
                <button
                  key={p.name}
                  type="button"
                  className="tg"
                  onClick={() => setProfile(Array.from({ length: N }, (_, i) => p.make(i)))}
                >
                  {p.name}
                </button>
              ))}
            </div>
            <div className="hint" style={{ marginTop: 10 }}>
              えらんでから、図の上をドラッグして形を変えられる。
              縦は減光の深さ(いちばん下で 3%)、横は通過の中心からの時間。
            </div>
          </div>
          <div className="dial">
            <label>
              いま描いてある深さ <span className="amt">{fmtDepth(built?.depth ?? 0)}</span>
            </label>
            <div className="hint">
              周期 {PERIOD} 日・通過の長さ {(DURATION * 24).toFixed(1)} 時間・
              ばらつき {(SIGMA * 1e6).toFixed(0)} ppm は固定してある。
            </div>
          </div>
        </div>

        <div className="lab-out">
          <div className="readout-top">
            <div className="rt-l">
              <span className="cap">なぞった窪みの形</span>
              <h3>手描きの曲線</h3>
            </div>
            <Gauge prob={busy ? null : (pred?.prob ?? null)} />
          </div>

          <svg
            ref={svgRef}
            className="chart"
            viewBox={`0 0 ${W} ${H}`}
            style={{ touchAction: "none", cursor: "crosshair" }}
            role="img"
            aria-label="窪みの形をなぞる図"
            onPointerDown={(e) => {
              drawing.current = true;
              lastIdx.current = null;
              (e.target as Element).setPointerCapture?.(e.pointerId);
              apply(e.clientX, e.clientY);
            }}
            onPointerMove={(e) => drawing.current && apply(e.clientX, e.clientY)}
            onPointerUp={() => {
              drawing.current = false;
              lastIdx.current = null;
            }}
            onPointerLeave={() => {
              drawing.current = false;
              lastIdx.current = null;
            }}
          >
            <rect x={L} y={T} width={W - L - R} height={H - T - B} fill="var(--panel-2)" />
            <path d={area} fill="var(--star-wash)" />
            <path d={path} fill="none" stroke="var(--star)" strokeWidth="2.2" strokeLinejoin="round" />
            <line x1={L} y1={Y(0)} x2={W - R} y2={Y(0)} stroke="var(--line)" />
            <text x={L - 9} y={Y(0) + 4} fill="var(--ink-3)" fontSize="10.5" textAnchor="end" fontFamily="var(--f-mono)">
              0
            </text>
            <text
              x={L - 9}
              y={Y(MAX_DEPTH) + 4}
              fill="var(--ink-3)"
              fontSize="10.5"
              textAnchor="end"
              fontFamily="var(--f-mono)"
            >
              −3%
            </text>
            {[-1, 0, 1].map((u) => (
              <text
                key={u}
                x={L + ((u + SPAN) / (2 * SPAN)) * (W - L - R)}
                y={H - B + 18}
                fill="var(--ink-3)"
                fontSize="10.5"
                textAnchor="middle"
                fontFamily="var(--f-mono)"
              >
                {u === 0 ? "0" : `${u > 0 ? "+" : "−"}${((DURATION * 24) / 2).toFixed(1)} h`}
              </text>
            ))}
            <text x={(L + W - R) / 2} y={H - 6} fill="var(--ink-3)" fontSize="10.5" textAnchor="middle">
              通過の中心からの時間
            </text>
          </svg>

          <p className="plain">
            {busy || !pred ? (
              "モデルに通しています…"
            ) : pred.prob >= 0.5 ? (
              <>
                <b>惑星と呼ばれた({Math.round(pred.prob * 100)}%)。</b>
                底が平らで、入口と出口が対称な形をしている。
              </>
            ) : (
              <>
                <b>惑星とは呼ばれなかった({Math.round(pred.prob * 100)}%)。</b>
                深さや尖り方が、惑星の通過として見た形から外れている。
              </>
            )}
          </p>
        </div>
      </div>
    </section>
  );
}
