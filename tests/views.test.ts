/**
 * 二実装照合 —— ブラウザ側のビュー生成が Python 側と一致することを確かめる。
 *
 * 期待値の出所は **Python 側の実装**(`training/tl/preprocess.py`)。
 * フィクスチャは `training/make_fixtures.py` が作る。
 *
 * ここが緩むと、つまみで作った曲線と実データが**別の物差し**でモデルに入る。
 * その食い違いは画面上ではもっともらしい数字として出るので、検査でしか捕まらない。
 */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  GLOBAL_BINS,
  LOCAL_BINS,
  binnedMedian,
  fillGaps,
  globalView,
  localView,
  normalizeView,
  phaseFold,
} from "@/lib/views";

type Case = {
  name: string;
  period: number;
  duration: number;
  depth: number;
  n: number;
  time: string;
  flux: string;
  global: string;
  local: string;
};

function decode(b64: string): Float64Array {
  const buf = Buffer.from(b64, "base64");
  return new Float64Array(buf.buffer, buf.byteOffset, buf.byteLength / 8);
}

const fixture = JSON.parse(readFileSync("tests/fixtures/views.json", "utf-8")) as {
  cases: Case[];
};

describe("T-101 二実装照合(G-06) — ビュー生成が Python と一致する", () => {
  for (const c of fixture.cases) {
    it(`${c.name}: global / local の全ビンが一致する`, () => {
      const time = decode(c.time);
      const flux = decode(c.flux);
      const wantG = decode(c.global);
      const wantL = decode(c.local);

      const folded = phaseFold(time, c.period, 0);
      const gotG = globalView(folded, flux, c.period);
      const gotL = localView(folded, flux, c.period, c.duration);

      expect(gotG.length).toBe(GLOBAL_BINS);
      expect(gotL.length).toBe(LOCAL_BINS);
      expect(wantG.length).toBe(GLOBAL_BINS);
      expect(wantL.length).toBe(LOCAL_BINS);

      // 中央値・線形補間・除算しか通らないので、差は倍精度の丸めの範囲に収まるはず。
      // 一致の閾値は 1e-12 —— 手続きが違えばこの桁では必ず落ちる
      let maxG = 0;
      for (let i = 0; i < GLOBAL_BINS; i++) maxG = Math.max(maxG, Math.abs(gotG[i] - wantG[i]));
      let maxL = 0;
      for (let i = 0; i < LOCAL_BINS; i++) maxL = Math.max(maxL, Math.abs(gotL[i] - wantL[i]));

      expect(maxG, `global の最大差 ${maxG}`).toBeLessThan(1e-12);
      expect(maxL, `local の最大差 ${maxL}`).toBeLessThan(1e-12);
    });
  }
});

describe("T-103 照合の陽性対照 — わざと外した手続きは落ちる", () => {
  // 「全部一致した」が「照合が働いている」を意味しないため、
  // **落ちるはずの計算が実際に落ちること**をここで示す(G-06 の対照)。
  it("local ビューのビン幅を重なりの無い値にすると 1e-12 では合わない", () => {
    const c = fixture.cases.find((x) => x.name === "typical")!;
    const time = decode(c.time);
    const flux = decode(c.flux);
    const want = decode(c.local);
    const folded = phaseFold(time, c.period, 0);

    const tMax = Math.min(c.period / 2, c.duration * 4);
    const idx: number[] = [];
    for (let i = 0; i < folded.length; i++) {
      if (folded[i] >= -tMax && folded[i] <= tMax) idx.push(i);
    }
    const fx = Float64Array.from(idx, (i) => folded[i]);
    const fy = Float64Array.from(idx, (i) => flux[i]);
    // 正しい幅は 0.16 × 継続時間。ここでは重なりの無い幅(窓 ÷ ビン数)を使う
    const wrongWidth = (2 * tMax) / LOCAL_BINS;
    const { view } = binnedMedian(fx, fy, LOCAL_BINS, wrongWidth, -tMax, tMax);
    const got = normalizeView(fillGaps(view));

    let max = 0;
    for (let i = 0; i < LOCAL_BINS; i++) max = Math.max(max, Math.abs(got[i] - want[i]));
    expect(max, "ビン幅を外しても差が出ないなら、照合は何も見ていない").toBeGreaterThan(1e-6);
  });
});

describe("T-102 部品の性質(解析解)", () => {
  it("phaseFold: 周期の整数倍だけ離れた時刻は同じ位相に畳まれる", () => {
    const period = 3.5;
    const t0 = 131.25;
    const offsets = [-1.2, -0.4, 0, 0.4, 1.2];
    for (const k of [-7, 0, 1, 25]) {
      const t = Float64Array.from(offsets, (d) => t0 + k * period + d);
      const folded = phaseFold(t, period, t0);
      for (let i = 0; i < offsets.length; i++) {
        expect(folded[i]).toBeCloseTo(offsets[i], 9);
      }
    }
  });

  it("binnedMedian: 最初のビンは xMin から、最後のビンは xMax で終わる", () => {
    // y = x を入れると各ビンの中央値はビンの中心に一致する(標本が密なら刻み幅の中で)
    const step = 1e-3;
    const xMin = -4;
    const xMax = 4;
    const width = 0.32;
    const n = Math.round((xMax - xMin) / step) + 1;
    const x = Float64Array.from({ length: n }, (_, i) => xMin + i * step);
    const { view, empty } = binnedMedian(x, Float64Array.from(x), 201, width, xMin, xMax);
    expect(empty).toBe(0);
    const spacing = (xMax - xMin - width) / 200;
    for (let i = 0; i < 201; i++) {
      expect(view[i]).toBeCloseTo(xMin + spacing * i + width / 2, 2);
    }
  });

  it("fillGaps: 空ビンは前後から線形に埋まる", () => {
    const out = fillGaps(Float64Array.from([0, NaN, NaN, 3, NaN, 5]));
    expect(Array.from(out)).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it("normalizeView: 中央値 0・最小値 −1", () => {
    const v = Float64Array.from({ length: 401 }, (_, i) => 1 + Math.sin(i) * 0.01);
    v[200] = 0.2;
    const out = normalizeView(v);
    const sorted = Float64Array.from(out).sort();
    expect(sorted[200]).toBeCloseTo(0, 12);
    expect(Math.min(...out)).toBeCloseTo(-1, 12);
  });
});
