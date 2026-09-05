// 実ブラウザでの検品(SPEC N-05 / T-201〜T-203)。
//
// **検査が全部緑でも動かない故障がある。** ONNX の読み込み経路、wasm の寄せ替え、
// SVG の描画、つまみの再計算 —— どれも vitest からは見えない。
// ここでは本番と同じ静的書き出し(out/)を配って、実際に開いて触る。
//
// 実行: npm run build && node scripts/verify-browser.mjs
// 出力: logs/browser/*.png と標準出力の判定

import { createServer } from "node:http";
import { existsSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { chromium } from "playwright";

const ROOT = "out";
const SHOT_DIR = "logs/browser";
const PORT = 4173;

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".onnx": "application/octet-stream",
  ".wasm": "application/wasm",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
};

if (!existsSync(ROOT)) {
  console.error(`${ROOT}/ が無い。先に npm run build を走らせること`);
  process.exit(2);
}
mkdirSync(SHOT_DIR, { recursive: true });

const server = createServer((req, res) => {
  let path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (path.endsWith("/")) path += "index.html";
  const file = join(ROOT, normalize(path).replace(/^(\.\.[/\\])+/, ""));
  if (!existsSync(file) || statSync(file).isDirectory()) {
    res.writeHead(404).end("not found");
    return;
  }
  const body = readFileSync(file);
  res.writeHead(200, {
    "content-type": MIME[extname(file)] ?? "application/octet-stream",
    "content-length": body.length,
  });
  res.end(body);
});

const problems = [];
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "  ok  " : "  NG  "} ${name}${detail ? " — " + detail : ""}`);
  if (!ok) problems.push(`${name}${detail ? ": " + detail : ""}`);
};

await new Promise((r) => server.listen(PORT, r));
const base = `http://127.0.0.1:${PORT}`;
console.log(`配信中: ${base}\n`);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 1000 } });

const requests = [];
page.on("request", (r) => requests.push(r.url()));
const consoleErrors = [];
page.on("console", (m) => m.type() === "error" && consoleErrors.push(m.text()));
page.on("pageerror", (e) => consoleErrors.push(String(e)));

await page.goto(`${base}/`, { waitUntil: "networkidle" });

// ---- T-201 ギャラリーと判定 -------------------------------------------------
// networkidle の時点ではまだ demo.json の fetch が始まっていないことがある。
// **数える前に出現を待つ**(待たずに数えて 0 枚で落ちた)
await page.locator(".card").first().waitFor({ timeout: 60_000 }).catch(() => {});
const cards = await page.locator(".card").count();
check("T-201 ギャラリーのカードが並ぶ", cards >= 4, `${cards} 枚`);

// waitForFunction の第 2 引数は **関数へ渡す値**であって options ではない。
// options は第 3 引数。ここを詰めると既定の 30 秒で切れる(実際に踏んだ)
await page.waitForFunction(
  () => {
    const v = document.querySelector(".gauge .val");
    return v && /\d+%/.test(v.textContent ?? "");
  },
  undefined,
  { timeout: 90_000 },
);
const firstVerdict = await page.locator(".gauge .val").first().textContent();
check("T-201 判定ゲージが数字を出す(ブラウザ内推論)", /\d+%/.test(firstVerdict ?? ""), firstVerdict ?? "");

const gazeDrawn = await page.locator("text=モデルの視線").count();
check("T-201 モデルの視線が描かれる", gazeDrawn > 0, `${gazeDrawn} 箇所`);

const stamp = await page.locator(".stamp").first().textContent();
check("T-201 答え合わせのスタンプが出る", stamp === "一致" || stamp === "不一致", stamp ?? "");

await page.screenshot({ path: join(SHOT_DIR, "01-observatory.png"), fullPage: false });

// 別の星を選ぶと判定が変わる(= 選択が効いている)
const before = await page.locator(".gauge .val").first().textContent();
await page.locator(".card").nth(cards - 1).click();
await page.waitForTimeout(2500);
const after = await page.locator(".gauge .val").first().textContent();
check("T-201 星を替えると判定が変わる", before !== after, `${before} → ${after}`);
await page.screenshot({ path: join(SHOT_DIR, "02-second-star.png"), fullPage: false });

// ---- T-202 つまみ -----------------------------------------------------------
await page.locator("#dRp").scrollIntoViewIfNeeded();
await page.waitForFunction(
  () => {
    const vals = [...document.querySelectorAll(".gauge .val")];
    return vals.length > 1 && /\d+%/.test(vals[1].textContent ?? "");
  },
  undefined,
  { timeout: 90_000 },
);
const bigVerdict = await page.locator(".gauge .val").nth(1).textContent();

// 惑星を最小まで小さくする
await page.locator("#dRp").fill("0");
await page.waitForTimeout(3000);
const smallVerdict = await page.locator(".gauge .val").nth(1).textContent();
const smallPct = Number((smallVerdict ?? "0").replace("%", ""));
check(
  "T-202 惑星を最小にすると判定が下がる",
  smallPct < Number((bigVerdict ?? "100").replace("%", "")),
  `${bigVerdict} → ${smallVerdict}`,
);
await page.screenshot({ path: join(SHOT_DIR, "03-sandbox.png"), fullPage: false });

// 周期を最短にすると拾い直せることがある(通過回数が増える)
await page.locator("#dP").fill("0");
await page.waitForTimeout(3000);
const shortVerdict = await page.locator(".gauge .val").nth(1).textContent();
console.log(`       参考: 周期を最短にすると ${smallVerdict} → ${shortVerdict}`);

// ---- しきい値 ---------------------------------------------------------------
await page.locator("#dTh").scrollIntoViewIfNeeded();
// セルの textContent には <small> の説明語が続く(「335見つけた」)。
// **数だけを取り出す** —— Number("335見つけた") は NaN なので、比較が黙って偽になる
const cell = async () => {
  const t = (await page.locator(".matrix .num").first().textContent()) ?? "";
  return Number((t.match(/^\s*(\d+)/) ?? [])[1] ?? Number.NaN);
};
const tpBefore = await cell();
await page.locator("#dTh").fill("95");
await page.waitForTimeout(400);
const tpAfter = await cell();
check(
  "T-201 しきい値を上げると見つけた数が減る",
  Number.isFinite(tpBefore) && Number.isFinite(tpAfter) && tpAfter < tpBefore,
  `${tpBefore} → ${tpAfter}`,
);
await page.screenshot({ path: join(SHOT_DIR, "04-threshold.png"), fullPage: false });

// ---- 成績表 -----------------------------------------------------------------
await page.locator(".grades").scrollIntoViewIfNeeded();
const gradeCount = await page.locator(".grade .v").count();
const grades = await page.locator(".grade .v").allTextContents();
check("T-201 成績表に実測値が並ぶ", gradeCount >= 5 && !grades.includes("—"), grades.join(" / "));
await page.screenshot({ path: join(SHOT_DIR, "05-scorecard.png"), fullPage: false });

await page.screenshot({ path: join(SHOT_DIR, "00-full.png"), fullPage: true });

// ---- T-203 通信先 -----------------------------------------------------------
const external = [...new Set(requests)].filter(
  (u) => !u.startsWith(base) && !u.startsWith("data:") && !/fonts\.(googleapis|gstatic)\.com/.test(u),
);
check("T-203 自オリジンとフォント以外へ出ない", external.length === 0, external.join(", "));

const ortHits = [...new Set(requests.filter((u) => u.includes("/tl/ort/")))].map((u) => u.split("/").pop());
check("T-203 ORT を自オリジンから取っている", ortHits.length > 0, ortHits.join(", "));

check("コンソールにエラーが出ていない", consoleErrors.length === 0, consoleErrors.slice(0, 3).join(" | "));

// ---- フッタ -----------------------------------------------------------------
const footer = (await page.locator(".site-footer__inner").innerText()).replace(/\s+/g, " ");
check(
  "フリート共通フッタが 6 項目そろっている",
  ["MIT License", "© 2026 坂田哲朗", "GitHub", "歩き方", "設計図", "App Menu"].every((t) =>
    footer.includes(t),
  ),
  footer,
);

await browser.close();
server.close();

console.log(`\nスクリーンショット → ${SHOT_DIR}/`);
if (problems.length) {
  console.error(`\n不合格 ${problems.length} 件:\n  ` + problems.join("\n  "));
  process.exit(1);
}
console.log("\nすべて合格");
