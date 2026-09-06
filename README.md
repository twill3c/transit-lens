# transit-lens — Transit Lens 観測卓

NASA Kepler 宇宙望遠鏡が 4 年間測り続けた恒星の明るさから、1 次元 CNN が惑星の通過
(トランジット)を見つける。**その判断を、専門知識なしで最後まで追える観測卓**。

推論はすべてブラウザの中で走る(onnxruntime-web)。サーバ側の計算・cron・DB を持たない。

- 本番: **https://transit-lens-one.vercel.app/**
  (`transit-lens.vercel.app` は別人の既存プロジェクトが取得済みで、Vercel が
  `-one` 付きの別名を割り当てた。**`vercel alias ls` で実物を確かめてから書くこと**)
- 仕様: [SPEC.md](SPEC.md) / 検査: [TEST_SPEC.md](TEST_SPEC.md)

## この画面でできること

| 入口 | すること | 出るもの |
|---|---|---|
| 星を選ぶ | カードを 1 回押す | 判定ゲージ・模式アニメ・モデルの視線・NASA アーカイブとの答え合わせ |
| 自分で惑星を置く | つまみ 3 つ(大きさ・周期・ばらつき) | 合成した曲線を**同じモデル**に通した判定。惑星を小さくすると、ある所で見失う |
| しきい値を引く | スライダ | 保留集合の実スコアに対する混同行列(見逃しと空振りの取り引き) |

## データと権利

| 出典 | 内容 | 権利 |
|---|---|---|
| NASA Exoplanet Archive `q1_q17_dr24_tce` | TCE 台帳 20,367 件・教師ラベル `av_training_set` | NASA 制作物。パブリックドメイン |
| MAST | Kepler 長時間ケイデンス光度曲線(PDCSAP) | 同上 |

前処理は Shallue & Vanderburg (2018) *AJ* 155:94 §3 に合わせている(**同じラベル集合**を
使うので、published の AUC 0.988 と数字を並べて読める)。

**本アプリは既知の TCE に対する再現であり、新しい惑星の発見を主張しない。**

## 作り直し方

```bash
# 0) 環境
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe numpy scipy astropy scikit-learn onnx onnxruntime pytest
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
npm install

# 1) ラベル台帳(件数オラクル G-01 が同時に走る)
.venv/Scripts/python.exe training/fetch_labels.py

# 2) 光度曲線の取得と前処理(9,865 星・約 2.5 時間・FITS はディスクに残さない)
.venv/Scripts/python.exe training/build_dataset.py --out data/views --workers 14

# 3) 学習(出荷する CAM 型・比較用の AstroNet 型・陰性対照の 3 本)
.venv/Scripts/python.exe training/train.py --head cam
.venv/Scripts/python.exe training/train.py --head astronet
.venv/Scripts/python.exe training/train.py --head cam --shuffle-labels

# 4) ゲートと目玉を測る
.venv/Scripts/python.exe training/measure.py

# 5) 配布物を作る
.venv/Scripts/python.exe training/export_web.py
node scripts/pack.mjs

# 6) 検査
.venv/Scripts/python.exe -m pytest -q
npx vitest run
npm run build
```

`training/build_dataset.py` は `data/views/progress.jsonl` から再開できる。
途中で止めても、済んだ星は二度取りに行かない。

## 設計上の判断(いずれも SPEC に根拠がある)

- **頭部を CAM 型にした。** 各枝を大域平均プーリングして全結合 1 層にすると、
  ロジットが各位置の寄与の平均に**厳密に**分解できる。「モデルの視線」が近似ではなくなる。
  表現力を捨てているので、published の頭部との AUC 差を測ってから決めた(SPEC §6.1 / G-05)
- **分割は恒星単位。** 同じ星の TCE は光度曲線を共有するので、TCE 単位で切ると漏れる(G-03)
- **スプラインを当てるとき、その TCE の通過を覆う。** 参照実装は覆わないが、
  浅く長い通過は 3σ クリップに掛からず深さを食われる。4 通りの合成で測って決めた
- **窪みの形は一様光源の掩蔽解ひとつから出す。** 惑星の平底と食連星の V 字は
  同じ式の両端であって、作り分けではない

## ライセンス

MIT License © 2026 坂田哲朗
