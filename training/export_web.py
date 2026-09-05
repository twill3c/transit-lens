"""配布データを作る —— public/tl/{demo,metrics,scores}.json と model.onnx.

**画面に出る数はすべてここで作る。** 手で書いた数字を画面に置かない(G-07)。

ギャラリーの 6 天体は**保留集合**(学習にも検証にも使っていない星)から選ぶ。
学習に使った天体の判定を見せるのは、答えを知っている問題を解かせるのと同じである。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import pathlib
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tl import kepler_io, preprocess, spline  # noqa: E402
from tl.model import TransitNet  # noqa: E402
from train import load_shards, make_splits, predict  # noqa: E402

RAW_POINTS = 1400
REFERENCE = {
    "name": "AstroNet(Shallue & Vanderburg 2018)",
    "auc": 0.988,
    "note": "同じ Q1–Q17 DR24 の教師つき TCE を使った published の値。10 個のモデルの平均。",
}


def load_koi_names(path: pathlib.Path) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                key = (int(row["kepid"]), int(float(row["koi_tce_plnt_num"])))
            except (ValueError, KeyError):
                continue
            out[key] = row
    return out


def story_for(rec: dict) -> str:
    """その天体が「モデルにとってどう際どいか」をデータから書く.

    人が知っている逸話は書かない —— 書けば、確かめていないことを断定することになる。
    """
    bits: list[str] = []
    period = rec["period"]
    depth = rec["depth_ppm"] / 1e6
    ntrans = rec["ntrans"]
    mes = rec["mes"]

    if depth >= 0.02:
        bits.append(
            f"窪みの深さ {depth * 100:.1f}% は、惑星の通過としては桁違いに深い。"
            "相手が惑星ではなく、もう一つの星である形をしている"
        )
    if period < 1.5:
        bits.append(
            f"{period * 24:.1f} 時間で一周する。通過回数が多いぶん、弱い信号でも積み上げられる"
        )
    elif np.isfinite(ntrans) and 1 <= ntrans <= 8:
        bits.append(
            f"4 年の観測で {int(ntrans)} 回しか通らない。回数が足りないと、確信は上がりにくい"
        )
    if np.isfinite(mes) and mes < 12:
        bits.append(f"信号の強さは {mes:.1f}σ —— Kepler の検出線 7.1σ のすぐ上にある、際どい一件")
    elif np.isfinite(mes) and mes > 200:
        bits.append(f"信号の強さは {mes:.0f}σ と圧倒的で、迷いようがない")
    if depth < 2e-4:
        bits.append(f"明るさの変化は {depth * 100:.4f}%。1 万分の 2 に満たない翳り")
    elif np.isfinite(ntrans) and ntrans > 300:
        bits.append(f"4 年で {int(ntrans)} 回も通るので、弱い窪みでも重ねれば形が出てくる")

    if not bits:
        bits.append(
            f"周期 {period:.2f} 日、深さ {depth * 1e6:.0f} ppm。"
            "教科書どおりの一件で、モデルも迷わない"
        )
    return "。".join(bits[:2]) + "。"


def fetch_raw(kepid: int, period: float, t0: float, duration: float) -> tuple[list[float], list[float]]:
    """折りたたむ前の曲線を取り直して間引く(ギャラリーの数天体だけ)."""
    names = kepler_io.list_quarters(kepid)
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        blobs = dict(
            zip(names, ex.map(lambda n: kepler_io.fetch_quarter(kepid, n), names))
        )
    lc = kepler_io.load_light_curve(kepid, blobs)
    mask = preprocess.transit_mask(lc.time, period, t0, duration)
    trend = spline.fit_spline(lc.time, lc.flux, mask=mask)
    ok = np.isfinite(trend) & (trend != 0)
    t, f = lc.time[ok], lc.flux[ok] / trend[ok]
    step = max(1, t.size // RAW_POINTS)
    return (
        [round(float(v), 4) for v in t[::step]],
        [round(float(v), 6) for v in f[::step]],
    )


def choose_gallery(
    data: dict[str, np.ndarray],
    test_idx: np.ndarray,
    scores: np.ndarray,
    koi: dict[tuple[int, int], dict],
) -> list[int]:
    """保留集合から、絵として違いが見える 6 件を選ぶ.

    * 惑星候補 4 件 —— 通称のある(= 確認された)ものを優先し、周期を散らす
    * 誤検出 2 件 —— 深い食連星らしいものと、信号の弱いもの
    """
    av = data["av"][test_idx]
    period = data["period"][test_idx]
    depth = data["depth"][test_idx]
    mes = data["mes"][test_idx]
    named = np.array(
        [
            (int(k), int(p)) in koi and bool(koi[(int(k), int(p))]["kepler_name"])
            for k, p in zip(data["kepid"][test_idx], data["plnt"][test_idx])
        ]
    )

    ntrans = data["ntrans"][test_idx]
    picks: list[int] = []

    # 台帳には物理的にありえない値が混ざる(深さ 232%・通過回数 0 など)。
    # **ギャラリーは人が見る絵なので、ここで弾く。**弾いた理由は台帳の値であって
    # モデルの出力ではないので、選び方に循環は入らない
    sane = (
        np.isfinite(depth)
        & (depth > 5)
        & (depth < 250000)
        & np.isfinite(ntrans)
        & (ntrans >= 3)
        & np.isfinite(mes)
    )

    # 惑星候補: 通称つきを、周期の帯ごとに 1 件ずつ。
    # **帯ごとに選び方を変える** —— 全部を「信号 30σ の教科書的な一件」にすると、
    # 6 枚並べても同じ絵が 4 枚できるだけで、何も教えない
    pc = np.flatnonzero((av == "PC") & named & sane)
    bands: list[tuple[float, float, str]] = [
        (0, 3, "strong"),  # 短周期・強い信号 = 迷いようのない一件
        (3, 20, "median"),  # ふつうの一件
        (20, 80, "marginal"),  # 検出線のすぐ上 = 際どい一件
        (80, 10000, "fewest"),  # 通過回数が足りない一件
    ]
    for lo, hi, how in bands:
        cand = pc[(period[pc] >= lo) & (period[pc] < hi)]
        if cand.size == 0:
            continue
        if how == "strong":
            pick = cand[np.argmax(mes[cand])]
        elif how == "median":
            pick = cand[np.argsort(mes[cand])[cand.size // 2]]
        elif how == "marginal":
            eligible = cand[mes[cand] >= 7.1]
            pick = (eligible if eligible.size else cand)[
                np.argmin(mes[eligible if eligible.size else cand])
            ]
        else:
            pick = cand[np.argmin(ntrans[cand])]
        picks.append(int(pick))

    # 誤検出 1: 食連星らしい深い窪み(天体由来 = AFP に限る)
    fp_deep = np.flatnonzero((av == "AFP") & sane & (depth > 20000) & (ntrans >= 5))
    if fp_deep.size:
        picks.append(int(fp_deep[np.argmax(depth[fp_deep])]))

    # 誤検出 2: 信号が弱く、モデルも低い点を付けたもの(器材由来 = NTP)
    fp_weak = np.flatnonzero((av == "NTP") & sane & (mes >= 7) & (mes < 12) & (ntrans >= 20))
    if fp_weak.size:
        picks.append(int(fp_weak[np.argmin(scores[fp_weak])]))

    seen: set[int] = set()
    out: list[int] = []
    for p in picks:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:6]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--views", default="data/views")
    ap.add_argument("--models", default="data/models")
    ap.add_argument("--tag", default="cam")
    ap.add_argument("--koi", default="data/labels/koi_cumulative.csv")
    ap.add_argument("--out", default="public/tl")
    ap.add_argument("--skip-raw", action="store_true", help="折りたたむ前の曲線を取り直さない")
    args = ap.parse_args()

    models = pathlib.Path(args.models)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = json.loads((models / f"{args.tag}.json").read_text(encoding="utf-8"))
    data = load_shards(pathlib.Path(args.views))
    splits = make_splits(data)
    test_idx = splits["test"]

    model = TransitNet(head=report["head"], width=report["width"])
    model.load_state_dict(torch.load(models / f"{args.tag}.pt", weights_only=True))
    model.eval()

    g_te = torch.from_numpy(data["gview"][test_idx]).unsqueeze(1)
    l_te = torch.from_numpy(data["lview"][test_idx]).unsqueeze(1)
    scores = predict(model, g_te, l_te)

    koi = load_koi_names(pathlib.Path(args.koi))
    chosen = choose_gallery(data, test_idx, scores, koi)
    print(f"ギャラリー {len(chosen)} 件を保留集合 {test_idx.size} 件から選んだ")

    stars = []
    for local_i in chosen:
        i = int(test_idx[local_i])
        kepid = int(data["kepid"][i])
        plnt = int(data["plnt"][i])
        info = koi.get((kepid, plnt), {})
        name = info.get("kepler_name") or info.get("kepoi_name") or f"KIC {kepid}"
        av = str(data["av"][i])
        rec = {
            "id": f"{kepid}-{plnt}",
            "kepid": kepid,
            "plnt": plnt,
            "name": name.strip(),
            "av": av,
            "truth": "惑星候補" if av == "PC" else "誤検出",
            "split": "test",
            "period": round(float(data["period"][i]), 6),
            "durationDays": round(float(data["duration"][i]), 6),
            "depth_ppm": float(data["depth"][i]),
            "depthPpm": round(float(data["depth"][i]), 1),
            "prad": (
                round(float(data["prad"][i]), 2) if np.isfinite(data["prad"][i]) else None
            ),
            "snr": round(float(data["snr"][i]), 2),
            "mes": round(float(data["mes"][i]), 2),
            "ntrans": float(data["ntrans"][i]),
            "gview": [round(float(v), 5) for v in data["gview"][i]],
            "lview": [round(float(v), 5) for v in data["lview"][i]],
        }
        rec["sub"] = (
            {"PC": "惑星候補", "AFP": "天体由来の誤検出", "NTP": "器材由来の誤検出"}[av]
            + f"・{'周期 ' + format(rec['period'], '.2f') + ' 日'}"
        )
        rec["story"] = story_for(rec)
        del rec["depth_ppm"]
        if args.skip_raw:
            rec["rawTime"], rec["rawFlux"] = [], []
        else:
            print(f"  KIC {kepid} の折りたたむ前の曲線を取得中…", flush=True)
            rec["rawTime"], rec["rawFlux"] = fetch_raw(
                kepid,
                float(data["period"][i]),
                float(data["t0"][i]) if "t0" in data else 0.0,
                float(data["duration"][i]),
            )
        stars.append(rec)

    (out / "demo.json").write_text(
        json.dumps({"stars": stars}, ensure_ascii=False), encoding="utf-8"
    )

    rows = [
        {"s": round(float(s), 4), "y": int(y)}
        for s, y in zip(scores, data["label"][test_idx])
    ]
    (out / "scores.json").write_text(
        json.dumps(
            {
                "note": (
                    f"保留集合 {len(rows):,} 件(学習にも検証にも使っていない星の TCE)の実スコア。"
                    "合成でも学習集合の成績でもない。"
                ),
                "rows": rows,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    onnx_src = models / f"{args.tag}.onnx"
    shutil.copyfile(onnx_src, out / "model.onnx")

    extra_path = models / "measurements.json"
    extra = json.loads(extra_path.read_text(encoding="utf-8")) if extra_path.exists() else {}

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "head": report["head"],
            "width": report["width"],
            "params": report["params"],
            "epochs": report["epochs"],
            "best_epoch": report["best_epoch"],
            "onnx_bytes": (out / "model.onnx").stat().st_size,
        },
        "counts": {
            **report["counts"],
            "stars": len(set(data["kepid"].tolist())),
            "tce": int(data["label"].size),
        },
        "test": report["test"],
        "heads": extra.get("heads", {"cam": report["test"]["auc"], "astronet": float("nan"), "delta": float("nan"), "chosen": report["head"]}),
        "control_shuffled_auc": extra.get("control_shuffled_auc", float("nan")),
        "onnx_max_abs_diff": extra.get("onnx_max_abs_diff", float("nan")),
        "eyeballs": extra.get("eyeballs", {}),
        "reference": REFERENCE,
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for f in ("demo.json", "scores.json", "metrics.json", "model.onnx"):
        print(f"  {f}  {(out / f).stat().st_size:,} バイト")
    return 0


if __name__ == "__main__":
    sys.exit(main())
