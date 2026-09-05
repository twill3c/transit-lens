/** 配布データ(public/tl/*.json)の型と読み込み。 */

export type Disposition = "PC" | "AFP" | "NTP";

export type DemoStar = {
  id: string;
  kepid: number;
  plnt: number;
  /** 表示名。通称のある星は通称、無ければ KIC 番号 */
  name: string;
  sub: string;
  story: string;
  av: Disposition;
  /** 「惑星候補」/「誤検出」 */
  truth: string;
  /** この TCE がモデルの学習に使われたかどうか。**画面に出す** */
  split: "train" | "val" | "test";
  period: number;
  durationDays: number;
  depthPpm: number;
  prad: number | null;
  snr: number;
  mes: number;
  ntrans: number;
  /** 前処理済みビュー(Python 側で作ったもの) */
  gview: number[];
  lview: number[];
  /** 折りたたむ前の曲線(間引き済み)。位相ではなく時刻(BKJD) */
  rawTime: number[];
  rawFlux: number[];
};

export type Metrics = {
  generated_at: string;
  model: {
    head: string;
    width: number;
    params: number;
    epochs: number;
    best_epoch: number;
    onnx_bytes: number;
  };
  counts: { train: number; val: number; test: number; stars: number; tce: number };
  test: {
    auc: number;
    ap: number;
    "precision@0.5": number;
    "recall@0.5": number;
    n: number;
    positives: number;
  };
  /** 頭部の二択の実測(SPEC §6.1 / G-05) */
  heads: { cam: number; astronet: number; delta: number; chosen: string };
  /** 陰性対照(G-04) */
  control_shuffled_auc: number;
  /** 二実装照合(G-06) */
  onnx_max_abs_diff: number;
  reference: { name: string; auc: number; note: string };
};

export type ScoreRow = { s: number; y: 0 | 1 };

export type Scores = {
  note: string;
  rows: ScoreRow[];
};
