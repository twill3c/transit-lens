import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";

export default defineConfig({
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  test: {
    include: ["tests/**/*.test.ts"],
    environment: "node",
    // 二実装照合は 71,400 点の合成曲線をビンに落とすので既定の 5 秒では足りない
    testTimeout: 180_000,
  },
});
