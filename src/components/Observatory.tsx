"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CurveChart } from "./CurveChart";
import { Gauge } from "./Gauge";
import { SkyPanel } from "./SkyPanel";
import type { DemoStar } from "@/lib/data";
import { fmtDepth, fmtDepthPlain, fmtHours, fmtPeriod } from "@/lib/format";
import { predict, upsample, type Prediction } from "@/lib/model";
import { LOCAL_NUM_DURATIONS } from "@/lib/views";

type View = "local" | "global" | "raw";

const SPLIT_WORD: Record<DemoStar["split"], string> = {
  train: "この TCE は学習に使った",
  val: "この TCE は検証に使った(学習には入れていない)",
  test: "この TCE は学習にも検証にも使っていない",
};

function sparkPath(lview: number[], w = 160, h = 34): string {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of lview) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const pad = (hi - lo) * 0.1 || 0.1;
  lo -= pad;
  hi += pad;
  let d = "";
  for (let i = 0; i < lview.length; i++) {
    const x = 3 + (i / (lview.length - 1)) * (w - 6);
    const y = h - 4 - ((lview[i] - lo) / (hi - lo)) * (h - 8);
    d += `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`;
  }
  return d;
}

export function Observatory() {
  const [stars, setStars] = useState<DemoStar[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [view, setView] = useState<View>("local");
  const [gazeOn, setGazeOn] = useState(true);
  const [pred, setPred] = useState<Prediction | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [phase, setPhase] = useState(-1.6);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    let alive = true;
    fetch("/tl/demo.json")
      .then((r) => r.json())
      .then((d: { stars: DemoStar[] }) => {
        if (!alive) return;
        setStars(d.stars);
        setSelected(d.stars[0]?.id ?? null);
      })
      .catch(() => setModelError("見本データを読み込めなかった"));
    return () => {
      alive = false;
    };
  }, []);

  const current = useMemo(
    () => stars?.find((s) => s.id === selected) ?? null,
    [stars, selected],
  );

  // 共有の時計 —— 惑星が星の前を通る瞬間と、曲線が凹む瞬間を一致させる
  useEffect(() => {
    const reduced =
      typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) {
      setPhase(0);
      return;
    }
    let last = 0;
    const tick = (ts: number) => {
      if (!last) last = ts;
      const dt = Math.min(48, ts - last);
      last = ts;
      setPhase((p) => (p > 1.6 ? -1.6 : p + (dt / 1000) * 0.62));
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // 選んだ星を、ブラウザの中の ONNX に通す
  useEffect(() => {
    if (!current) return;
    let alive = true;
    setPred(null);
    predict(Float32Array.from(current.gview), Float32Array.from(current.lview))
      .then((p) => {
        if (alive) setPred(p);
      })
      .catch((e: unknown) => {
        if (alive) setModelError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      alive = false;
    };
  }, [current]);

  const pick = useCallback((id: string) => setSelected(id), []);

  // ギャラリーは **共有の時計に依らない**。分けておかないと、
  // 位相が動くたびに 6 枚のカードとミニ曲線を作り直すことになる
  const cards = useMemo(
    () =>
      (stars ?? []).map((s) => (
        <button
          key={s.id}
          type="button"
          className="card"
          aria-pressed={s.id === selected}
          onClick={() => pick(s.id)}
        >
          <span className={`tag ${s.av === "PC" ? "p" : "f"}`}>
            {s.av === "PC" ? "惑星候補" : "誤検出"}
          </span>
          <span className="nm">{s.name}</span>
          <span className="sb">{s.sub}</span>
          <svg viewBox="0 0 160 34" preserveAspectRatio="none" aria-hidden="true">
            <path
              d={sparkPath(s.lview)}
              fill="none"
              stroke={s.av === "PC" ? "var(--star)" : "var(--no)"}
              strokeWidth="1.4"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      )),
    [stars, selected, pick],
  );

  if (!stars) {
    return <p className="loading">見本データを読み込んでいます…</p>;
  }

  const ratio = current ? Math.sqrt(current.depthPpm / 1e6) : 0.05;
  const companionIsStar = current ? current.depthPpm > 30000 : false;

  const gaze =
    gazeOn && pred
      ? view === "global"
        ? upsample(pred.camGlobal, current?.gview.length ?? 2001)
        : view === "local"
          ? upsample(pred.camLocal, current?.lview.length ?? 201)
          : null
      : null;

  let values: number[] | Float64Array = current?.lview ?? [];
  let xMin = -1;
  let xMax = 1;
  let xUnit = "";
  let scatter = false;
  let markerX: number | null = null;

  if (current) {
    if (view === "local") {
      const halfWindowHours = Math.min(current.period / 2, current.durationDays * LOCAL_NUM_DURATIONS) * 24;
      values = current.lview;
      xMin = -halfWindowHours;
      xMax = halfWindowHours;
      xUnit = "通過の中心からの時間(時間)";
      markerX = (phase * current.durationDays * 24) / 2;
    } else if (view === "global") {
      values = current.gview;
      xMin = -current.period / 2;
      xMax = current.period / 2;
      xUnit = "通過の中心からの時間(日)— 位相の全域";
      markerX = (phase * current.durationDays) / 2;
    } else {
      values = current.rawFlux;
      xMin = current.rawTime[0] ?? 0;
      xMax = current.rawTime[current.rawTime.length - 1] ?? 1;
      xUnit = "観測日(BKJD)— 折りたたむ前";
      scatter = true;
    }
  }

  const agree = current && pred ? (pred.prob >= 0.5) === (current.av === "PC") : null;

  return (
    <>
      <section className="step">
        <div className="step-head">
          <span className="step-n">入力 1</span>
          <h2>星を選ぶ</h2>
          <span className="io">IN: クリック 1 回</span>
        </div>
        <p className="note">
          KIC 番号を打たせない。カードの絵だけで「何が起きた星か」が伝わるようにする。
          ミニ曲線は<b>その星の実データ</b>を前処理したもので、選ぶ前から違いが目に入る。
          誤検出の例を最初から混ぜてある —— 全部当たるギャラリーは何も教えない。
        </p>
        <div className="gallery">{cards}</div>
        <div className="dice">
          <button
            type="button"
            className="btn"
            onClick={() => {
              const others = stars.filter((s) => s.id !== selected);
              if (others.length) pick(others[Math.floor(Math.random() * others.length)].id);
            }}
          >
            今夜の一天体をまかせる
          </button>
        </div>
      </section>

      <section className="step">
        <div className="step-head">
          <span className="step-n">出力 1–3</span>
          <h2>観測卓で見る</h2>
          <span className="io">OUT: 絵 → 曲線 → 確率</span>
        </div>
        <p className="note">
          左は起きていること、右は測れたこと。同じ時計で動くので、惑星が星の前を通る瞬間と
          曲線が凹む瞬間が一致する。判定はこのブラウザの中で走っている ——
          曲線を送り出しても、返ってくる数字を待ってもいない。
        </p>

        <div className="desk">
          <div className="sky">
            <span className="cap">起きていること(模式図)</span>
            <SkyPanel
              phase={phase}
              ratio={ratio}
              seed={current?.kepid ?? 1}
              companionIsStar={companionIsStar}
            />
            <div className="scalebar">
              <span className="txt">
                {current === null ? (
                  "—"
                ) : companionIsStar ? (
                  <>
                    相手は惑星ではなく<b>もう一つの恒星</b>。だから窪みが桁違いに深い。
                  </>
                ) : current.prad ? (
                  <>
                    半径は地球の <b>{current.prad.toFixed(2)} 倍</b>。この大きさだと、星の光は{" "}
                    <b>{fmtDepth(current.depthPpm / 1e6)}</b> だけ翳る。
                  </>
                ) : (
                  <>
                    窪みの深さは <b>{fmtDepth(current.depthPpm / 1e6)}</b>。
                  </>
                )}
              </span>
            </div>
          </div>

          <div className="readout">
            <div className="readout-top">
              <div className="rt-l">
                <span className="cap">
                  {view === "raw" ? "測れたこと(折りたたむ前)" : "測れたこと(位相折りたたみ後)"}
                </span>
                <h3>{current?.name ?? "—"}</h3>
              </div>
              <Gauge prob={pred?.prob ?? null} />
            </div>

            <CurveChart
              values={values}
              xUnit={xUnit}
              xMin={xMin}
              xMax={xMax}
              gaze={gaze}
              scatter={scatter}
              yLabel={view === "raw" ? "相対フラックス" : "明るさの変化(正規化)"}
              markerX={view === "raw" ? null : markerX}
            />

            <div className="toggles">
              <button
                type="button"
                className="tg"
                aria-pressed={view === "local"}
                onClick={() => setView("local")}
              >
                通過の拡大(モデルの入力 2)
              </button>
              <button
                type="button"
                className="tg"
                aria-pressed={view === "global"}
                onClick={() => setView("global")}
              >
                位相の全域(モデルの入力 1)
              </button>
              <button
                type="button"
                className="tg"
                aria-pressed={view === "raw"}
                onClick={() => setView("raw")}
              >
                折りたたむ前を見る
              </button>
              <button
                type="button"
                className="tg"
                aria-pressed={gazeOn}
                onClick={() => setGazeOn((v) => !v)}
              >
                モデルの視線を重ねる
              </button>
            </div>

            <p className="plain">
              {modelError ? (
                <>モデルを動かせなかった: {modelError}</>
              ) : current === null ? (
                "—"
              ) : (
                <>
                  <b>この星について:</b> {current.story}
                </>
              )}
            </p>
          </div>
        </div>

        <div className="answer">
          <div className="ans-i">
            <span className="k">モデル</span>
            <span className="v">
              {pred === null
                ? "計算中"
                : `${pred.prob >= 0.5 ? "惑星" : "誤検出"}(${Math.round(pred.prob * 100)}%)`}
            </span>
          </div>
          <div className="ans-i">
            <span className="k">NASA アーカイブの判定</span>
            <span className="v">{current?.truth ?? "—"}</span>
          </div>
          <div className="ans-i">
            <span className="k">窪みの深さ</span>
            <span className="v">
              {current ? `${fmtDepth(current.depthPpm / 1e6)}(明るさの ${fmtDepthPlain(current.depthPpm / 1e6)})` : "—"}
            </span>
          </div>
          <div className="ans-i">
            <span className="k">公転周期 / 通過の長さ</span>
            <span className="v">
              {current ? `${fmtPeriod(current.period)} / ${fmtHours(current.durationDays)}` : "—"}
            </span>
          </div>
          {agree !== null && (
            <span className={`stamp ${agree ? "ok" : "ng"}`}>{agree ? "一致" : "不一致"}</span>
          )}
        </div>
        {current && (
          <p className="foot" style={{ marginTop: 10, borderTop: "none", paddingTop: 0 }}>
            {SPLIT_WORD[current.split]}。KIC {current.kepid}・TCE {current.plnt}。
            {pred && (
              <>
                {" "}
                視線はモデルの畳み込み枝の出力(global 枝で {pred.camGlobal.length} 点、
                local 枝で {pred.camLocal.length} 点)を曲線の長さへ線形に引き伸ばして描いている。
                ロジットはこの視線の平均に厳密に分解できる(近似ではない)。
              </>
            )}
          </p>
        )}
      </section>
    </>
  );
}
