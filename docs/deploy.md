# デプロイ手順

静的書き出し(`output: "export"`)だけを配る。サーバ関数・cron・DB を持たない。

## 出荷前に通すもの

```bash
.venv/Scripts/python.exe -m pytest -q     # 前処理・台帳・分割・字種
npx vitest run                            # 二実装照合・配布物・誇張の明記
npm run typecheck
npm run build                             # out/ を作る
node scripts/verify-browser.mjs           # 実ブラウザ検品(N-05)
```

**`npm run build` が通っただけでは出荷判定にならない。** ONNX の読み込み経路と wasm の
寄せ替えはビルドからは見えない。`verify-browser.mjs` が実際に開いて、判定ゲージに
数字が出るところまで確かめる。

## 配布物の内訳

| 経路 | 中身 | だいたいの大きさ |
|---|---|---|
| `public/tl/model.onnx` | 学習済みモデル(CAM 型) | 数 MB(N-02 で 5 MB 以下) |
| `public/tl/ort/` | ONNX Runtime(wasm 専用版) | 13.41 MB |
| `public/tl/demo.json` | ギャラリー 6 天体のビューと折りたたむ前の曲線 | 数百 KB |
| `public/tl/scores.json` | 保留集合の実スコア | 数十 KB |
| `public/tl/metrics.json` | 成績と実測値 | 数 KB |

ORT の wasm は**リポジトリに入れる**(`.gitignore` に入れない)。Vercel のビルドが
外部へ取りに出ないようにするため。先例は manazashi-lab / kototoi-do。

## Vercel

```bash
gh repo create twill3c/transit-lens --public --source . --remote origin --push
vercel link --project transit-lens --yes
vercel --prod
```

**気をつけること**([[vercel-deploy-quirks]]):

- 無料枠には 1 日あたりのビルド数の上限がある。**失敗しても自動で再試行されない**ので、
  枠を使い切ったら翌日まで待つ。横断作業(フリート全体のフッタ統一など)と
  同じ日に重ねない
- `.vercelignore` は**ローカルの `next build` では読まれない**。ここを間違えても
  手元では気づけない(HC-048)。先頭の `/` を省くとどの階層にも当たるので、
  `data` と書くと `src/lib/data.ts` まで消える
- Free プランはファイル数に 24 時間 5,000 件の上限がある。`.vercelignore` は
  **最初のデプロイの前に**置くこと

## デプロイ後に確かめること

1. 本番 URL を開いて、判定ゲージに数字が出るか(= ブラウザ内推論が動いているか)
2. `curl -I <URL>/tl/model.onnx` が 200 で、`content-type` が `application/octet-stream`
3. `curl -I <URL>/tl/ort/ort-wasm-simd-threaded.wasm` が `application/wasm`
4. フッタの 6 項目のリンクがすべて開くか(**歩き方・設計図のアーティファクトは
   既定で非公開**なので、共有設定を開くまでリンク先が見えない)
5. app-menu(cat7 量子・計算・天体)にカードを足す
