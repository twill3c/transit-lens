/**
 * 配布物の検査。
 *
 * **画面に出る数はすべて生成物から来ていなければならない**(G-07)。
 * ここでは「生成物にその数が入っていること」と「ギャラリーが保留集合だけで
 * できていること」を見る —— 手で書いた数字が画面に紛れ込む道を塞ぐため。
 *
 * 配布物が無い環境(学習前)では skip する。**skip は緑ではない。**
 */

import { existsSync, readFileSync, statSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { DemoStar, Metrics, Scores } from "@/lib/data";

const DEMO = "public/tl/demo.json";
const METRICS = "public/tl/metrics.json";
const SCORES = "public/tl/scores.json";
const MODEL = "public/tl/model.onnx";

const ready = [DEMO, METRICS, SCORES, MODEL].every((p) => existsSync(p));
const d = describe.skipIf(!ready);

d("T-107 配布データ(G-07)", () => {
  it("ギャラリーは全件が保留集合で、誤検出の例を含む", () => {
    const demo = JSON.parse(readFileSync(DEMO, "utf-8")) as { stars: DemoStar[] };
    expect(demo.stars.length).toBeGreaterThanOrEqual(4);
    for (const s of demo.stars) {
      // 学習に使った天体の判定を見せるのは、答えを知っている問題を解かせるのと同じ
      expect(s.split, `${s.name} が保留集合でない`).toBe("test");
      expect(s.gview.length).toBe(2001);
      expect(s.lview.length).toBe(201);
      expect(Number.isFinite(s.period) && s.period > 0).toBe(true);
      expect(Number.isFinite(s.durationDays) && s.durationDays > 0).toBe(true);
      // 正規化の定義(最小値 −1)が配布物でも保たれている
      expect(Math.min(...s.lview)).toBeCloseTo(-1, 3);
      expect(Math.min(...s.gview)).toBeCloseTo(-1, 3);
    }
    const kinds = new Set(demo.stars.map((s) => s.av));
    expect(kinds.has("PC"), "惑星候補が 1 件も無い").toBe(true);
    expect(kinds.size, "誤検出の例が混ざっていない").toBeGreaterThan(1);
  });

  it("しきい値の分布は保留集合の実スコアで、両クラスを含む", () => {
    const scores = JSON.parse(readFileSync(SCORES, "utf-8")) as Scores;
    expect(scores.rows.length).toBeGreaterThan(200);
    const pos = scores.rows.filter((r) => r.y === 1).length;
    expect(pos).toBeGreaterThan(0);
    expect(pos).toBeLessThan(scores.rows.length);
    for (const r of scores.rows) {
      expect(r.s).toBeGreaterThanOrEqual(0);
      expect(r.s).toBeLessThanOrEqual(1);
    }
  });

  it("成績はすべて metrics.json にあり、ゲートの値が埋まっている", () => {
    const m = JSON.parse(readFileSync(METRICS, "utf-8")) as Metrics;
    expect(m.test.auc).toBeGreaterThan(0.5);
    expect(m.test.auc).toBeLessThanOrEqual(1);
    expect(m.test.n).toBeGreaterThan(0);
    // G-04 陰性対照 —— ラベルを入れ替えたら 0.55 未満に落ちていること
    expect(m.control_shuffled_auc, "G-04 が未測定").toBeTypeOf("number");
    expect(m.control_shuffled_auc).toBeLessThan(0.55);
    // G-06 二実装照合
    expect(m.onnx_max_abs_diff, "G-06 が未測定").toBeTypeOf("number");
    expect(m.onnx_max_abs_diff).toBeLessThan(1e-5);
    // G-05 頭部の選定 —— 両方の AUC が入っていること
    expect(m.heads.cam).toBeGreaterThan(0.5);
    expect(m.heads.astronet).toBeGreaterThan(0.5);
    expect(["cam", "astronet"]).toContain(m.heads.chosen);
  });
});

d("T-108 配布物の重さ(N-02 / G-11)", () => {
  it("ONNX モデルは 5 MB 以下", () => {
    const bytes = statSync(MODEL).size;
    expect(bytes, `${(bytes / 1048576).toFixed(2)} MB`).toBeLessThan(5 * 1024 * 1024);
  });

  it("出荷する静的書き出しにサーバ実行物が無い", () => {
    if (!existsSync("out")) return;
    // next の static export はサーバ関数を作らない。作られたら構成が変わったということ
    expect(existsSync("out/api")).toBe(false);
    expect(existsSync(".next/server/app/api")).toBe(false);
  });
});

describe("T-109 誇張の明記(G-09)", () => {
  it("拡大表示と引き伸ばしを画面に書いてある", () => {
    // 文字列そのものを見る。**消したら落ちる**ようにしておく
    const sky = readFileSync("src/components/SkyPanel.tsx", "utf-8");
    expect(sky).toContain("拡大して表示");
    const obs = readFileSync("src/components/Observatory.tsx", "utf-8");
    expect(obs).toContain("引き伸ばして");
  });
});
