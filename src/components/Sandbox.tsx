"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { CurveChart } from "./CurveChart";
import { Gauge } from "./Gauge";
import { fmtDepth, fmtHours, fmtPeriod, sizeWord } from "@/lib/format";
import { KEPLER_MES_THRESHOLD } from "@/lib/physics";
import { gazeBand, predict, type Prediction } from "@/lib/model";
import { synthesize, type SynthResult } from "@/lib/synth";
import { LOCAL_NUM_DURATIONS } from "@/lib/views";

/** つまみの位置(0–100)を対数で実際の値へ移す。 */
const logMap = (v: number, a: number, b: number) => a * Math.pow(b / a, v / 100);

export function Sandbox() {
  const [rpDial, setRpDial] = useState(52);
  const [perDial, setPerDial] = useState(46);
  const [sigDial, setSigDial] = useState(38);
  const [pred, setPred] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const seq = useRef(0);

  // つまみの値と、実際に計算した値を分ける。曲線の生成は 71,400 点を畳むので
  // 数十ミリ秒かかる —— ドラッグ中の入力ごとに走らせると、つまみが引っかかる
  const [applied, setApplied] = useState({ rp: rpDial, per: perDial, sig: sigDial });
  useEffect(() => {
    const t = setTimeout(() => setApplied({ rp: rpDial, per: perDial, sig: sigDial }), 90);
    return () => clearTimeout(t);
  }, [rpDial, perDial, sigDial]);

  const rp = logMap(applied.rp, 0.5, 14);
  const period = logMap(applied.per, 0.4, 400);
  const sigma = logMap(applied.sig, 40e-6, 2200e-6);

  const synth = useMemo<SynthResult | null>(() => {
    try {
      return synthesize({ planetEarthRadii: rp, periodDays: period, sigmaPerPoint: sigma });
    } catch {
      return null;
    }
  }, [rp, period, sigma]);

  useEffect(() => {
    if (!synth) return;
    const mine = ++seq.current;
    setBusy(true);
    const timer = setTimeout(() => {
      predict(Float32Array.from(synth.globalView), Float32Array.from(synth.localView))
        .then((p) => {
          if (seq.current === mine) {
            setPred(p);
            setBusy(false);
          }
        })
        .catch((e: unknown) => {
          if (seq.current === mine) {
            setError(e instanceof Error ? e.message : String(e));
            setBusy(false);
          }
        });
    }, 120);
    return () => clearTimeout(timer);
  }, [synth]);

  const halfWindowHours = synth
    ? Math.min(period / 2, synth.durationDays * LOCAL_NUM_DURATIONS) * 24
    : 1;
  const gaze = pred && !busy ? gazeBand(pred.camLocal, synth?.localView.length ?? 201) : null;
  const detected = pred ? pred.prob >= 0.5 : null;

  return (
    <section className="step">
      <div className="step-head">
        <span className="step-n">入力 2</span>
        <h2>自分で惑星を置いてみる</h2>
        <span className="io">IN: つまみ 3 つ → OUT: 同じモデルの判定</span>
      </div>
      <p className="note">
        惑星を小さくしていくと、ある所でモデルが見失う。周期を短くすると、同じ大きさでも
        見つかるようになる。説明せずに、限界を手で触らせる。
        —— ここで判定しているのは<b>ギャラリーとまったく同じモデル</b>で、
        曲線も実データと<b>同じ手続き</b>(位相折りたたみ → 2001 点 / 201 点のビン)で
        作っている。窪みの形は台形の作り物ではなく、一様光源の掩蔽解から出している。
      </p>

      <div className="lab">
        <div className="dials">
          <div className="dial">
            <label htmlFor="dRp">
              惑星の大きさ <span className="amt">地球の {rp.toFixed(1)} 倍</span>
            </label>
            <input
              id="dRp"
              type="range"
              min={0}
              max={100}
              value={rpDial}
              onChange={(e) => setRpDial(Number(e.target.value))}
            />
            <div className="hint">
              {sizeWord(rp)}の大きさ。窪みの深さは {synth ? fmtDepth(synth.depth) : "—"}。
            </div>
          </div>
          <div className="dial">
            <label htmlFor="dP">
              公転周期 <span className="amt">{fmtPeriod(period)}</span>
            </label>
            <input
              id="dP"
              type="range"
              min={0}
              max={100}
              value={perDial}
              onChange={(e) => setPerDial(Number(e.target.value))}
            />
            <div className="hint">短いほど何度も通過するので、観測を重ねられる。</div>
          </div>
          <div className="dial">
            <label htmlFor="dN">
              観測のばらつき <span className="amt">{(sigma * 1e6).toFixed(0)} ppm</span>
            </label>
            <input
              id="dN"
              type="range"
              min={0}
              max={100}
              value={sigDial}
              onChange={(e) => setSigDial(Number(e.target.value))}
            />
            <div className="hint">星の明るさや望遠鏡の状態で決まる、測定のざらつき。</div>
          </div>
        </div>

        <div className="lab-out">
          <div className="readout-top">
            <div className="rt-l">
              <span className="cap">合成した曲線(位相折りたたみ後)</span>
              <h3>
                {sizeWord(rp)}・{fmtPeriod(period)}
              </h3>
            </div>
            <Gauge prob={busy ? null : (pred?.prob ?? null)} />
          </div>

          {synth && (
            <CurveChart
              values={synth.localView}
              xUnit="通過の中心からの時間(時間)"
              xMin={-halfWindowHours}
              xMax={halfWindowHours}
              gaze={gaze}
              yLabel="明るさの変化(正規化)"
              height={214}
            />
          )}

          <div className="snr">
            <span className="pill">
              4 年間の通過回数 <b>{synth?.transits ?? "—"} 回</b>
            </span>
            <span className="pill">
              通過の長さ <b>{synth ? fmtHours(synth.durationDays) : "—"}</b>
            </span>
            <span className="pill">
              窪みの深さ <b>{synth ? fmtDepth(synth.depth) : "—"}</b>
            </span>
            <span className={`pill${synth && synth.snr >= KEPLER_MES_THRESHOLD ? " hot" : ""}`}>
              信号の強さ <b>{synth ? `${synth.snr.toFixed(1)}σ` : "—"}</b>
            </span>
          </div>

          <p className="plain">
            {error ? (
              <>モデルを動かせなかった: {error}</>
            ) : detected === null || !synth ? (
              "モデルに通しています…"
            ) : detected ? (
              <>
                <b>見つかった。</b> 信号の強さは <b>{synth.snr.toFixed(1)}σ</b>。
                惑星を小さくするか、ばらつきを上げると、どこかで見えなくなる。
              </>
            ) : (
              <>
                <b>見失った。</b> 信号の強さは <b>{synth.snr.toFixed(1)}σ</b>。
                周期を短くしてみる —— 同じ大きさの惑星でも、通過回数が増えれば拾えるようになる。
              </>
            )}
          </p>
          <p className="foot" style={{ margin: 0, borderTop: "none", paddingTop: 0 }}>
            信号の強さ σ は Kepler の MES と同じ数え方(深さ ÷ ばらつき × √通過中の点数)。
            Kepler ミッション自身が候補として拾う線は {KEPLER_MES_THRESHOLD}σ にある。
            モデルはその線を教えられていない —— 学習で見た TCE の集合から、
            結果としてどのあたりに線を引いたのかは「この模型はどこまで本物か」で測ってある。
          </p>
        </div>
      </div>
    </section>
  );
}
