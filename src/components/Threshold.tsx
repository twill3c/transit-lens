"use client";

import { useEffect, useMemo, useState } from "react";
import type { Scores } from "@/lib/data";
import { fmtPercent } from "@/lib/format";

const W = 620;
const H = 210;
const L = 44;
const R = 14;
const T = 38;
const B = 40;
const BINS = 26;

/**
 * 見逃しと空振りの取り引き。
 *
 * ここに出る分布は**保留集合の実スコア**である(学習にも検証にも使っていない TCE)。
 * 合成でも、学習集合の成績でもない。
 */
export function Threshold() {
  const [scores, setScores] = useState<Scores | null>(null);
  const [th, setTh] = useState(50);

  useEffect(() => {
    let alive = true;
    fetch("/tl/scores.json")
      .then((r) => r.json())
      .then((d: Scores) => alive && setScores(d))
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, []);

  const bins = useMemo(() => {
    if (!scores) return null;
    const pos = new Array(BINS).fill(0);
    const neg = new Array(BINS).fill(0);
    for (const row of scores.rows) {
      const i = Math.min(BINS - 1, Math.floor(row.s * BINS));
      if (row.y === 1) pos[i]++;
      else neg[i]++;
    }
    return { pos, neg, max: Math.max(...pos, ...neg) };
  }, [scores]);

  const cm = useMemo(() => {
    if (!scores) return null;
    const t = th / 100;
    let tp = 0;
    let fn = 0;
    let fp = 0;
    let tn = 0;
    for (const row of scores.rows) {
      const called = row.s >= t;
      if (row.y === 1) called ? tp++ : fn++;
      else called ? fp++ : tn++;
    }
    return { tp, fn, fp, tn };
  }, [scores, th]);

  return (
    <section className="step">
      <div className="step-head">
        <span className="step-n">出力 5</span>
        <h2>見逃しと空振りの取り引き</h2>
        <span className="io">IN: しきい値 → OUT: 成績</span>
      </div>
      <p className="note">
        確率のどこで線を引くかを自分で決められる。慎重にすれば空振りは減るが、本物を見逃す。
        分布は<b>保留集合の実スコア</b> —— 学習にも検証にも一度も使っていない星の TCE だけを集めてある。
      </p>

      {!scores || !bins || !cm ? (
        <p className="loading">成績を読み込んでいます…</p>
      ) : (
        <div className="scores">
          <div className="lab-out">
            <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="スコアの分布">
              {Array.from({ length: BINS }, (_, i) => {
                const bw = (W - L - R) / BINS;
                const x = L + i * bw;
                const Y = (v: number) => T + (1 - v / bins.max) * (H - T - B);
                return (
                  <g key={i}>
                    <rect
                      x={x + 0.6}
                      y={Y(bins.neg[i])}
                      width={bw * 0.46}
                      height={Math.max(0, H - B - Y(bins.neg[i]))}
                      fill="var(--no)"
                      opacity=".62"
                    />
                    <rect
                      x={x + bw * 0.5}
                      y={Y(bins.pos[i])}
                      width={bw * 0.46}
                      height={Math.max(0, H - B - Y(bins.pos[i]))}
                      fill="var(--yes)"
                      opacity=".62"
                    />
                  </g>
                );
              })}
              <line x1={L} y1={H - B} x2={W - R} y2={H - B} stroke="var(--line)" />
              <line
                x1={L + (th / 100) * (W - L - R)}
                y1={T - 2}
                x2={L + (th / 100) * (W - L - R)}
                y2={H - B}
                stroke="var(--gaze)"
                strokeWidth="2"
              />
              <text
                x={L + (th / 100) * (W - L - R)}
                y={T - 6}
                fill="var(--gaze)"
                fontSize="10.5"
                textAnchor="middle"
                fontFamily="var(--f-mono)"
              >
                しきい値
              </text>
              {[0, 0.5, 1].map((u) => (
                <text
                  key={u}
                  x={L + u * (W - L - R)}
                  y={H - B + 17}
                  fill="var(--ink-3)"
                  fontSize="10.5"
                  textAnchor="middle"
                  fontFamily="var(--f-mono)"
                >
                  {`${u * 100}%`}
                </text>
              ))}
              <text
                x={(L + W - R) / 2}
                y={H - 6}
                fill="var(--ink-3)"
                fontSize="10.5"
                textAnchor="middle"
              >
                モデルが出した「惑星である確率」
              </text>
              <text x={L} y={14} fill="var(--yes)" fontSize="10.5" fontFamily="var(--f-mono)">
                ■ 惑星候補(PC)
              </text>
              <text x={L + 120} y={14} fill="var(--no)" fontSize="10.5" fontFamily="var(--f-mono)">
                ■ 誤検出(AFP / NTP)
              </text>
            </svg>
            <div className="dial">
              <label htmlFor="dTh">
                この確率以上を「惑星」と呼ぶ <span className="amt">{th}%</span>
              </label>
              <input
                id="dTh"
                type="range"
                min={2}
                max={98}
                value={th}
                onChange={(e) => setTh(Number(e.target.value))}
              />
            </div>
          </div>
          <div>
            <div className="matrix">
              <div className="hd" />
              <div className="hd">モデル：惑星</div>
              <div className="hd">モデル：誤検出</div>
              <div className="hd">惑星候補</div>
              <div className="num ok">
                {cm.tp}
                <small>見つけた</small>
              </div>
              <div className="num bad">
                {cm.fn}
                <small>見逃した</small>
              </div>
              <div className="hd">誤検出</div>
              <div className="num bad">
                {cm.fp}
                <small>空振り</small>
              </div>
              <div className="num">
                {cm.tn}
                <small>正しく捨てた</small>
              </div>
            </div>
            <div className="snr" style={{ marginTop: 12 }}>
              <span className="pill">
                適合率 <b>{cm.tp + cm.fp ? fmtPercent(cm.tp / (cm.tp + cm.fp)) : "—"}</b>
              </span>
              <span className="pill">
                再現率 <b>{cm.tp + cm.fn ? fmtPercent(cm.tp / (cm.tp + cm.fn)) : "—"}</b>
              </span>
            </div>
            <p className="foot">{scores.note}</p>
          </div>
        </div>
      )}
    </section>
  );
}
