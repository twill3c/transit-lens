"""SPEC §9 に貼る実測値をまとめて出す.

**数を手で書き写さないため**の道具。走らせた出力をそのまま SPEC と
アーティファクトへ移す(HC-152: 実測値は母集団と一緒に書く)。
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from train import load_shards, make_splits  # noqa: E402


def main() -> int:
    views = pathlib.Path("data/views")
    models = pathlib.Path("data/models")

    statuses: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    quarters: list[int] = []
    tce_seen = 0
    progress = views / "progress.jsonl"
    if progress.exists():
        with progress.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                statuses[r["status"]] += 1
                tce_seen += r.get("n_tce", 0)
                if r["status"] == "ok":
                    quarters.append(r.get("quarters", 0))
                for f in r.get("failed", []):
                    key = f["reason"].split(":")[1].strip().split(" が")[0].split(" に")[0]
                    reasons[key] += 1

    print("## 取得と前処理")
    print(f"  星: {sum(statuses.values()):,} 件を走査 — {dict(statuses)}")
    if quarters:
        print(f"  四半期数: 中央値 {int(np.median(quarters))} / 最小 {min(quarters)} / 最大 {max(quarters)}")
    print(f"  台帳の教師つき TCE: {tce_seen:,}")

    data = load_shards(views)
    n = int(data["label"].size)
    print(f"  ビューを作れた TCE: {n:,} / {tce_seen:,}  ({n / max(1, tce_seen) * 100:.2f}%)")
    print(f"  捨てた TCE: {tce_seen - n:,} — 理由の内訳 {dict(reasons)}")
    av = collections.Counter(data["av"].tolist())
    print(f"  内訳: {dict(av)}")
    print(f"  星の数: {len(set(data['kepid'].tolist())):,}")
    print(f"  非有限値: gview {int((~np.isfinite(data['gview'])).sum())} / lview {int((~np.isfinite(data['lview'])).sum())}")

    splits = make_splits(data)
    print("\n## 分割(恒星単位・種 20260905)")
    for name in ("train", "val", "test"):
        idx = splits[name]
        pos = int((data["label"][idx] == 1).sum())
        print(
            f"  {name:5s}: TCE {idx.size:6,} ({idx.size / n * 100:5.2f}%) / "
            f"星 {len(set(data['kepid'][idx].tolist())):5,} / 正例 {pos:5,} ({pos / max(1, idx.size) * 100:.1f}%)"
        )

    print("\n## 学習")
    for tag in ("cam", "astronet", "cam_shuffled"):
        p = models / f"{tag}.json"
        if not p.exists():
            print(f"  {tag}: 未実施")
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        print(
            f"  {tag:12s}: test AUC {r['test']['auc']:.4f} / AP {r['test']['ap']:.4f} / "
            f"適合率 {r['test']['precision@0.5']:.3f} / 再現率 {r['test']['recall@0.5']:.3f} / "
            f"最良 epoch {r['best_epoch']}/{r['epochs']} / パラメータ {r['params']:,}"
        )

    mpath = models / "measurements.json"
    if mpath.exists():
        m = json.loads(mpath.read_text(encoding="utf-8"))
        print("\n## ゲートと目玉")
        if "heads" in m:
            h = m["heads"]
            print(f"  G-05 頭部: cam {h['cam']:.4f} / astronet {h['astronet']:.4f} / 差 {h['delta']:+.4f} → {h['chosen']}")
        if "control_shuffled_auc" in m:
            print(f"  G-04 陰性対照 AUC: {m['control_shuffled_auc']:.4f}(基準 0.55 未満)")
        if "onnx_max_abs_diff" in m:
            print(f"  G-06 二実装照合 最大絶対差: {m['onnx_max_abs_diff']:.3e}(基準 1e-5 未満)")
        eye = m.get("eyeballs", {})
        if "detection_limit" in eye:
            d = eye["detection_limit"]
            print(f"  目玉 1 検出限界: 50% 交差 中央値 {d['median_crossing_mes']} σ / 参照 {d['threshold_reference']} σ")
            for c in d["configs"]:
                print(f"      周期 {c['period']:5.1f} 日 / ばらつき {c['sigma'] * 1e6:4.0f} ppm → {c['crossing_mes']}")
        if "gaze" in eye:
            g = eye["gaze"]
            print(
                f"  目玉 2 視線: 通過窓内の比率 平均 {g['mean_in_window_positive']} / "
                f"対照 {g['mean_in_window_control']} / 無情報の基準 {g['uninformative_baseline']:.3f} / "
                f"対照を上回った割合 {g['fraction_above_control']}(n={g['n_positive_called']})"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
