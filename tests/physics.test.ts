/**
 * 物理の二実装照合と、外部の値との突き合わせ。
 *
 * 期待値の出所は二つある:
 *
 * * **教科書の値** —— 地球が太陽の前を通るとき、深さは約 84 ppm、継続時間は約 13 時間。
 *   この 2 つは実装から独立に決まっているので、式を取り違えたらここで落ちる
 * * **Python 側の実装**(`training/tl/physics.py`)—— 乱数を含まない量だけを比べる
 */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  AU_IN_R_SUN,
  BASELINE_DAYS,
  CADENCE_DAYS,
  R_SUN_IN_EARTH,
  fluxAtTime,
  occultedFraction,
  radiusRatio,
  scaledSemiMajorAxis,
  transitCount,
  transitDepth,
  transitDuration,
  transitSnr,
} from "@/lib/physics";

type Case = {
  rp: number;
  rs: number;
  period: number;
  sigma: number;
  k: number;
  a_over_r: number;
  depth: number;
  duration_days: number;
  snr: number;
  transit_count: number;
  times: number[];
  flux: number[];
};

const fx = JSON.parse(readFileSync("tests/fixtures/physics.json", "utf-8")) as {
  constants: Record<string, number>;
  cases: Case[];
  occultation: { k: number; z: number[]; f: number[] }[];
};

describe("T-104 外部の値との突き合わせ", () => {
  it("地球が太陽の前を通ると、深さ約 84 ppm・継続時間約 13 時間になる", () => {
    // 出所: 教科書的な値(Winn 2010 "Transits and Occultations" §2 など)。
    // 実装から独立に決まっているので、式を取り違えればここで落ちる
    const depth = transitDepth(1, 1);
    expect(depth * 1e6).toBeGreaterThan(83);
    expect(depth * 1e6).toBeLessThan(85);

    const durationHours = transitDuration(365.25, 1, 1) * 24;
    expect(durationHours).toBeGreaterThan(12.5);
    expect(durationHours).toBeLessThan(13.5);
  });

  it("木星ほどの惑星なら深さは 1% 台になる", () => {
    // 出所: (R_jup/R_sun)^2 = (0.1005)^2 ≒ 1.01%
    const depth = transitDepth(11.2, 1);
    expect(depth).toBeGreaterThan(0.009);
    expect(depth).toBeLessThan(0.012);
  });
});

describe("T-105 二実装照合(物理)", () => {
  it("定数が一致する", () => {
    expect(R_SUN_IN_EARTH).toBeCloseTo(fx.constants.R_SUN_IN_EARTH, 10);
    expect(AU_IN_R_SUN).toBeCloseTo(fx.constants.AU_IN_R_SUN, 8);
    expect(CADENCE_DAYS).toBeCloseTo(fx.constants.CADENCE_DAYS, 12);
    expect(BASELINE_DAYS).toBeCloseTo(fx.constants.BASELINE_DAYS, 10);
  });

  for (const c of fx.cases) {
    it(`Rp=${c.rp} R⊕ / P=${c.period} 日 の導出量が一致する`, () => {
      expect(radiusRatio(c.rp, c.rs)).toBeCloseTo(c.k, 12);
      expect(scaledSemiMajorAxis(c.period, c.rs)).toBeCloseTo(c.a_over_r, 8);
      expect(transitDepth(c.rp, c.rs)).toBeCloseTo(c.depth, 12);
      expect(transitDuration(c.period, c.rs, c.rp)).toBeCloseTo(c.duration_days, 12);
      expect(transitCount(c.period)).toBe(c.transit_count);
      const snr = transitSnr(c.depth, c.sigma, c.period, c.duration_days);
      expect(Math.abs(snr / c.snr - 1)).toBeLessThan(1e-12);
    });

    it(`Rp=${c.rp} R⊕ / P=${c.period} 日 の窪みの形が一致する`, () => {
      for (let i = 0; i < c.times.length; i++) {
        const got = fluxAtTime(c.times[i], c.period, c.k, c.a_over_r);
        expect(Math.abs(got - c.flux[i])).toBeLessThan(1e-12);
      }
    });
  }

  for (const o of fx.occultation) {
    it(`掩蔽の減光率が一致する(k=${o.k})`, () => {
      for (let i = 0; i < o.z.length; i++) {
        expect(Math.abs(occultedFraction(o.z[i], o.k) - o.f[i])).toBeLessThan(1e-12);
      }
    });
  }
});

describe("T-106 掩蔽の形の性質(解析解)", () => {
  it("小さい相手は底が平ら、大きい相手は底が尖る", () => {
    // 「U 字と V 字は同じ式の両端」という画面の主張(ShapePair)を確かめる。
    // 底の平らさは、中心と中心から窪みの半分ずれた点の減光率の比で測る
    const flatness = (k: number) => {
      const total = occultedFraction(0, k);
      const half = occultedFraction((1 + k) / 2, k);
      return half / total;
    };
    expect(flatness(0.09)).toBeGreaterThan(0.99); // 惑星: ほぼ変わらない = 平ら
    expect(flatness(0.55)).toBeLessThan(0.85); // 伴星: 中心から離れると浅くなる = V 字
  });

  it("接触の外では減光しない", () => {
    for (const k of [0.02, 0.1, 0.5]) {
      expect(occultedFraction(1 + k + 1e-9, k)).toBe(0);
      expect(occultedFraction(0, k)).toBeCloseTo(k * k, 12);
    }
  });
});
