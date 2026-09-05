import type { NextConfig } from "next";

// 静的書き出しのみ。サーバ関数を一つも持たない(SPEC N-01 / G-11)。
// モデルと ONNX Runtime の wasm は public/tl/ から同一オリジンで配る。
// **next build は外部へ取りに行かない** — 行けば Vercel のビルドが外部依存になる。
const nextConfig: NextConfig = {
  output: "export",
  reactStrictMode: true,
  trailingSlash: true,
};

export default nextConfig;
