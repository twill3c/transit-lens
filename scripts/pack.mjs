// ONNX Runtime の実行系を public/tl/ort へ複製する。
//
// **複製であって変換ではない。** wasm 専用の版だけを置く。WebGPU 込みの jsep 版は
// 倍の大きさがあり、控えとして置くと寄せ替えが壊れたときに黙って倍が配られる。
// 置かなければ壊れたと分かる。
//
// 実行: node scripts/pack.mjs

import { copyFileSync, existsSync, mkdirSync, statSync } from "node:fs";
import { join } from "node:path";

const OUT = "public/tl/ort";
const SRC = "node_modules/onnxruntime-web/dist";
const FILES = [
  "ort.wasm.bundle.min.mjs",
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.mjs",
];

mkdirSync(OUT, { recursive: true });

let total = 0;
for (const f of FILES) {
  const src = join(SRC, f);
  if (!existsSync(src)) {
    throw new Error(`${src} が無い — onnxruntime-web の版でファイル名が変わった可能性がある`);
  }
  copyFileSync(src, join(OUT, f));
  const n = statSync(src).size;
  total += n;
  console.log(`  ort/${f}  ${(n / 1048576).toFixed(2)} MB`);
}
console.log(`計 ${(total / 1048576).toFixed(2)} MB → ${OUT}`);
