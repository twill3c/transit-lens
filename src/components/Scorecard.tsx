"use client";

import { useEffect, useState } from "react";
import type { Metrics } from "@/lib/data";
import { fmtPercent } from "@/lib/format";

/**
 * 「この模型はどこまで本物か」。
 *
 * 画面に出す数はすべて `public/tl/metrics.json` から来る。**手で書かない**(G-07)。
 * metrics.json は学習の報告(data/models/*.json)から作られる。
 */
export function Scorecard() {
  const [m, setM] = useState<Metrics | null>(null);

  useEffect(() => {
    let alive = true;
    fetch("/tl/metrics.json")
      .then((r) => r.json())
      .then((d: Metrics) => alive && setM(d))
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, []);

  if (!m) return <p className="loading">成績を読み込んでいます…</p>;

  // 未測定は null で来る。**「0.000」と描いてはならない** —— 測っていないことと
  // 測って 0 だったことは違う(metrics.json は NaN を出さず null にしている)
  const num = (v: number | null | undefined, digits = 4) =>
    typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "未測定";

  return (
    <div className="grades">
      <div className="grade">
        <div className="k">保留集合の AUC</div>
        <div className="v">{m.test.auc.toFixed(4)}</div>
        <div className="n">
          {m.reference.name} の {m.reference.auc.toFixed(3)} が比較先。{m.reference.note}
        </div>
      </div>
      <div className="grade">
        <div className="k">しきい値 50% での成績</div>
        <div className="v">{fmtPercent(m.test["precision@0.5"], 1)}</div>
        <div className="n">
          適合率。再現率は {fmtPercent(m.test["recall@0.5"], 1)}。
          保留集合 {m.test.n.toLocaleString("ja-JP")} 件のうち惑星候補は{" "}
          {m.test.positives.toLocaleString("ja-JP")} 件。
        </div>
      </div>
      <div className="grade">
        <div className="k">陰性対照</div>
        <div className="v">{num(m.control_shuffled_auc, 3)}</div>
        <div className="n">
          ラベルを無作為に入れ替えて同じ学習を回したときの AUC。
          帰無は 0.5 ではない —— 同じ初期値の<b>未学習</b>モデルが{" "}
          {num(m.untrained_baseline?.same_seed_as_runs, 4)}、初期値を 5 通り振ると{" "}
          {num(m.untrained_baseline?.min, 3)}〜{num(m.untrained_baseline?.max, 3)} に散る。
          対照はここに落ちるべきである。
        </div>
      </div>
      <div className="grade">
        <div className="k">二実装照合</div>
        <div className="v">
          {typeof m.onnx_max_abs_diff === "number" && Number.isFinite(m.onnx_max_abs_diff)
            ? m.onnx_max_abs_diff.toExponential(1)
            : "未測定"}
        </div>
        <div className="n">
          学習に使った PyTorch と、この画面が使う ONNX Runtime の出力の最大絶対差
          (保留集合の全件)。
        </div>
      </div>
      <div className="grade">
        <div className="k">頭部の選定</div>
        <div className="v">{m.heads.chosen === "cam" ? "CAM 型" : "AstroNet 型"}</div>
        <div className="n">
          視線が厳密に出る CAM 型で AUC {num(m.heads.cam)}、published と同じ
          AstroNet 型で {num(m.heads.astronet)}。差は {num(m.heads.delta)}。
        </div>
      </div>
      <div className="grade">
        <div className="k">学習に使った TCE</div>
        <div className="v">{m.counts.train.toLocaleString("ja-JP")}</div>
        <div className="n">
          検証 {m.counts.val.toLocaleString("ja-JP")} / 保留 {m.counts.test.toLocaleString("ja-JP")}。
          分割は<b>恒星単位</b> —— 同じ星の TCE が両側に現れないようにしてある。
        </div>
      </div>
    </div>
  );
}
