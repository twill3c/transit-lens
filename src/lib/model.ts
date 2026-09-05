/**
 * ONNX Runtime のセッション管理と推論。
 *
 * 手口は manazashi-lab と同じで、理由も同じ:
 *
 * * 入口は `onnxruntime-web/wasm` の束を**自オリジンから**取る。既定は CDN を見に行くので、
 *   閲覧のたびに外部へ通信することになる(SPEC N-01)
 * * ORT を module 直下で import しない。`"use client"` を付けても静的書き出しは
 *   一度サーバ側で prerender され、ORT が組み立てる `new URL(...)` が Node で落ちる。
 *   **ブラウザでしか動かないものは関数の中で取る**
 * * `webpackIgnore` でバンドラに触らせない。預けると wasm が束の側にも複製される
 */

import type * as OrtNS from "onnxruntime-web/wasm";

const BASE = "/tl";
/** 実行系の入口。束に文字列としてそのまま残るよう、組み立てずに書く */
const ORT_ENTRY = "/tl/ort/ort.wasm.bundle.min.mjs";
const MODEL_URL = "/tl/model.onnx";

export type Prediction = {
  /** 惑星である確率(シグモイド後) */
  prob: number;
  /** ロジット */
  logit: number;
  /** global 枝の視線(モデル内部の分解能。長さは枝の出力長) */
  camGlobal: Float32Array;
  /** local 枝の視線 */
  camLocal: Float32Array;
};

let ortPromise: Promise<typeof OrtNS> | null = null;

export function loadOrt(): Promise<typeof OrtNS> {
  if (!ortPromise) {
    ortPromise = (import(/* webpackIgnore: true */ ORT_ENTRY) as Promise<typeof OrtNS>).then(
      (ort) => {
        ort.env.wasm.wasmPaths = `${BASE}/ort/`;
        // 単スレッド固定。SharedArrayBuffer(COOP/COEP)を前提にしない。
        // 速さより「どこでも同じ答え」を採る(G-06)
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.simd = true;
        ort.env.logLevel = "error";
        return ort;
      },
    );
    ortPromise.catch(() => {
      ortPromise = null;
    });
  }
  return ortPromise;
}

let sessionPromise: Promise<OrtNS.InferenceSession> | null = null;

export function getSession(onProgress?: (frac: number) => void): Promise<OrtNS.InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await loadOrt();
      const res = await fetch(MODEL_URL);
      if (!res.ok) throw new Error(`モデルを取得できない (${res.status})`);
      const total = Number(res.headers.get("content-length") ?? 0);
      let buf: ArrayBuffer;
      if (!res.body || !total || !onProgress) {
        buf = await res.arrayBuffer();
      } else {
        const reader = res.body.getReader();
        const chunks: Uint8Array[] = [];
        let got = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          got += value.length;
          onProgress(Math.min(1, got / total));
        }
        const merged = new Uint8Array(got);
        let off = 0;
        for (const c of chunks) {
          merged.set(c, off);
          off += c.length;
        }
        buf = merged.buffer;
      }
      return ort.InferenceSession.create(buf, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
    })();
    sessionPromise.catch(() => {
      sessionPromise = null;
    });
  }
  return sessionPromise;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** global 2001 点 / local 201 点を与えて 1 件分の判定を得る。 */
export async function predict(
  globalView: Float32Array | Float64Array,
  localView: Float32Array | Float64Array,
): Promise<Prediction> {
  const ort = await loadOrt();
  const session = await getSession();
  const g = new ort.Tensor("float32", Float32Array.from(globalView), [1, 1, 2001]);
  const l = new ort.Tensor("float32", Float32Array.from(localView), [1, 1, 201]);
  const out = await session.run({ global_view: g, local_view: l });
  const logit = (out.logit.data as Float32Array)[0];
  return {
    logit,
    prob: sigmoid(logit),
    camGlobal: out.cam_global.data as Float32Array,
    camLocal: out.cam_local.data as Float32Array,
  };
}

/**
 * 視線(CAM)を表示用の長さへ伸ばす。
 *
 * CAM は枝の出力長(global で 59、local で 46)しかないので、
 * 曲線に重ねるには入力長へ引き伸ばす必要がある。線形補間で伸ばし、
 * **「引き伸ばした」ことを画面にも書く**(G-09)。
 */
export function upsample(cam: Float32Array, length: number): Float64Array {
  const out = new Float64Array(length);
  if (cam.length === 0) return out;
  if (cam.length === 1) return out.fill(cam[0]);
  for (let i = 0; i < length; i++) {
    const u = (i / (length - 1)) * (cam.length - 1);
    const a = Math.floor(u);
    const b = Math.min(cam.length - 1, a + 1);
    out[i] = cam[a] + (cam[b] - cam[a]) * (u - a);
  }
  return out;
}

/**
 * 表示用の視線 —— **中央値からの隔たり**を [-1, 1] に収めて返す。
 *
 * ロジットは `mean(cam) + …` なので、ある位置が判定を動かすのは
 * **その位置が枝の平均からどれだけ離れているか**による。平らな CAM は
 * 位置による差を持たない。だから描くべきは生の値ではなく中央値からの隔たりである。
 *
 * 符号は残す。実測(2026-09-06・保留集合の正例 335 件)では、
 * local 枝は「惑星である証拠」ではなく**「この窪みは怪しい」という減点**として働き、
 * 正例では減点が小さく(平均 −1.26)、負例では大きい(−8.01)。
 * 正の寄与だけを描くと、この働き方はまったく見えない ——
 * 実際、正の質量の 4.7% しか通過窓に落ちていなかった。
 * 大きさで測り直すと 76.0% が通過窓に落ちる。
 */
export function gazeBand(cam: Float32Array, length: number): Float64Array {
  const up = upsample(cam, length);
  if (up.length === 0) return up;
  const sorted = Float64Array.from(up).sort();
  const n = sorted.length;
  const median = n % 2 === 1 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  let max = 0;
  for (let i = 0; i < up.length; i++) {
    up[i] -= median;
    const a = Math.abs(up[i]);
    if (a > max) max = a;
  }
  if (max > 0) for (let i = 0; i < up.length; i++) up[i] /= max;
  return up;
}
