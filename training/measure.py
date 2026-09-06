"""ゲートと目玉を測って `data/models/measurements.json` にまとめる.

* **G-05 頭部の選定** —— cam / astronet 両方の報告を読み、SPEC §6.1 の規則で選ぶ
* **G-04 陰性対照** —— ラベルを入れ替えた学習の AUC
* **G-06 二実装照合** —— PyTorch と onnxruntime の出力を保留集合の全件で比べる
* **目玉 1 検出限界** —— 合成トランジットの注入回収で、判定が 50% を横切る MES を求め、
  Kepler 自身の検出線 7.1σ と比べる
* **目玉 2 視線** —— local ビューの視線の質量が通過窓にどれだけ落ちるかを、
  巡回ずらしの対照と比べる

**測る前に SPEC §7 に予測を書いてある。** 外れたら外れたと書く。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tl import synth  # noqa: E402
from tl.physics import (  # noqa: E402
    KEPLER_MES_THRESHOLD,
    R_SUN_IN_EARTH,
    transit_duration,
    transit_snr,
)
from tl.model import TransitNet  # noqa: E402
from train import load_shards, make_splits  # noqa: E402

HEAD_DELTA_LIMIT = 0.010  # SPEC §6.1 の判定規則


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def run_model(
    model: TransitNet, gview: np.ndarray, lview: np.ndarray, batch: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """(確率, local の視線を 201 点へ引き伸ばしたもの) を返す."""
    model.eval()
    probs: list[np.ndarray] = []
    cams: list[np.ndarray] = []
    g = torch.from_numpy(np.ascontiguousarray(gview, dtype=np.float32)).unsqueeze(1)
    l = torch.from_numpy(np.ascontiguousarray(lview, dtype=np.float32)).unsqueeze(1)
    with torch.no_grad():
        for i in range(0, g.shape[0], batch):
            logit, _, cam_l = model(g[i : i + batch], l[i : i + batch])
            probs.append(torch.sigmoid(logit).numpy())
            cams.append(cam_l.numpy())
    cam = np.concatenate(cams)
    # 画面と同じ引き伸ばし方(線形)で 201 点に合わせる
    src = np.linspace(0, 1, cam.shape[1])
    dst = np.linspace(0, 1, 201)
    up = np.stack([np.interp(dst, src, row) for row in cam])
    return np.concatenate(probs), up


# ----------------------------------------------------------------- G-06


def check_onnx(model_dir: pathlib.Path, tag: str, gview: np.ndarray, lview: np.ndarray) -> float:
    import onnxruntime as ort

    report = json.loads((model_dir / f"{tag}.json").read_text(encoding="utf-8"))
    model = TransitNet(head=report["head"], width=report["width"])
    model.load_state_dict(torch.load(model_dir / f"{tag}.pt", weights_only=True))
    torch_p, _ = run_model(model, gview, lview)

    sess = ort.InferenceSession(str(model_dir / f"{tag}.onnx"), providers=["CPUExecutionProvider"])
    outs: list[np.ndarray] = []
    for i in range(0, gview.shape[0], 256):
        res = sess.run(
            ["logit"],
            {
                "global_view": np.ascontiguousarray(gview[i : i + 256, None, :], dtype=np.float32),
                "local_view": np.ascontiguousarray(lview[i : i + 256, None, :], dtype=np.float32),
            },
        )
        outs.append(res[0])
    onnx_p = _sigmoid(np.concatenate(outs))
    return float(np.max(np.abs(torch_p - onnx_p)))


# ------------------------------------------------------ G-04 の物差しの作り直し


def class_separation(model_dir: pathlib.Path) -> dict:
    """陰性対照と本モデルの「クラスの離れ具合」を、出力そのもので測る.

    **AUC は退化した予測器に対して意味を持たない。** 実測(2026-09-06)では、
    ラベルを入れ替えて学習したモデルの出力は**ほぼ定数**になり
    (平均 0.2197 = 学習集合の正例率 0.2256 とほぼ同じ、標準偏差 0.00177)、
    その微小な揺らぎが MES と順位相関 +0.60 を持つ。DR24 では MES の高い側は
    深い食連星が占めるので、AUC は 0.117 まで下がる —— **ラベルを一切学んでいないのに、
    AUC は 0.5 から大きく離れる。**

    だから G-04 は AUC ではなく、**正例と負例の出力平均の差を出力の標準偏差で割った量**で
    見る。これは定数予測器に対して素直に 0 に近づく。
    """
    out: dict = {}
    for key, tag in (("real", "cam"), ("shuffled", "cam_shuffled")):
        path = model_dir / f"{tag}_scores.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as z:
            s, y = z["score"].astype(np.float64), z["label"]
        sd = float(s.std())
        mp, mn = float(s[y == 1].mean()), float(s[y == 0].mean())
        out[key] = {
            "mean_positive": mp,
            "mean_negative": mn,
            "score_sd": sd,
            "separation_sd": (mp - mn) / sd if sd > 0 else 0.0,
            "n": int(s.size),
        }
    return out


# --------------------------------------------------------------- 目玉 1


def detection_limit(model: TransitNet, seeds: tuple[int, ...] = (1, 2, 3)) -> dict:
    """注入回収。判定確率が 0.5 を横切る MES を、線形補間で求める."""
    configs = [
        {"period": 2.0, "sigma": 120e-6},
        {"period": 2.0, "sigma": 500e-6},
        {"period": 12.0, "sigma": 120e-6},
        {"period": 12.0, "sigma": 500e-6},
        {"period": 45.0, "sigma": 200e-6},
    ]
    rows = []
    for cfg in configs:
        period, sigma = cfg["period"], cfg["sigma"]
        # 目標 MES から必要な惑星半径を逆算し、1σ〜60σ を対数で刻む
        targets = np.logspace(np.log10(1.0), np.log10(60.0), 16)
        curve = []
        for target in targets:
            rp = _radius_for_mes(target, period, sigma)
            gviews, lviews, snrs = [], [], []
            for seed in seeds:
                try:
                    s = synth.synthesize(rp, period, sigma, seed=seed)
                except Exception:  # noqa: BLE001 — 作れない設定は飛ばす
                    continue
                gviews.append(s["global"])
                lviews.append(s["local"])
                snrs.append(s["snr"])
            if not gviews:
                continue
            probs, _ = run_model(model, np.stack(gviews), np.stack(lviews))
            curve.append(
                {
                    "target_mes": float(target),
                    "mes": float(np.mean(snrs)),
                    "planet_radii": float(rp),
                    "prob": float(np.mean(probs)),
                }
            )
        crossing = _crossing(curve)
        rows.append({**cfg, "curve": curve, "crossing_mes": crossing})
        print(
            f"  周期 {period:5.1f} 日 / ばらつき {sigma * 1e6:4.0f} ppm → "
            f"50% 交差 {crossing if crossing is None else round(crossing, 2)}σ",
            flush=True,
        )

    values = [r["crossing_mes"] for r in rows if r["crossing_mes"] is not None]
    return {
        "threshold_reference": KEPLER_MES_THRESHOLD,
        "configs": rows,
        "median_crossing_mes": float(np.median(values)) if values else None,
        "n_configs_with_crossing": len(values),
    }


def _radius_for_mes(target_mes: float, period: float, sigma: float) -> float:
    """目標の MES を与える惑星半径(地球半径)を求める.

    深さは半径の 2 乗、MES は深さに比例するので、半径を仮置きして 1 度だけ
    スケールし直せば十分な精度で当たる(継続時間の半径依存は極めて弱い)。
    """
    rp = 1.0
    for _ in range(3):
        duration = transit_duration(period, 1.0, rp)
        depth = (rp / R_SUN_IN_EARTH) ** 2
        mes = transit_snr(depth, sigma, period, duration)
        if mes <= 0:
            break
        rp *= float(np.sqrt(target_mes / mes))
    return rp


def _crossing(curve: list[dict]) -> float | None:
    """確率が 0.5 を下から上へ横切る MES を線形補間で返す."""
    for a, b in zip(curve, curve[1:]):
        if a["prob"] < 0.5 <= b["prob"]:
            span = b["prob"] - a["prob"]
            if span <= 0:
                continue
            u = (0.5 - a["prob"]) / span
            return float(a["mes"] + u * (b["mes"] - a["mes"]))
    return None


# --------------------------------------------------------------- 目玉 2


def _window_share(
    weight: np.ndarray, lo: int, hi: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """行ごとに (窓内の割合, 巡回ずらし対照, 有効行) を返す."""
    total = weight.sum(axis=1)
    ok = total > 0
    frac = np.zeros(weight.shape[0])
    frac[ok] = weight[ok, lo:hi].sum(axis=1) / total[ok]
    shifts = rng.integers(1, weight.shape[1], size=weight.shape[0])
    rolled = np.stack([np.roll(w, int(s)) for w, s in zip(weight, shifts)])
    ctrl = np.zeros(weight.shape[0])
    ctrl[ok] = rolled[ok, lo:hi].sum(axis=1) / total[ok]
    return frac, ctrl, ok


def gaze_mass(
    model: TransitNet,
    gview: np.ndarray,
    lview: np.ndarray,
    label: np.ndarray,
    rng_seed: int = 20260905,
) -> dict:
    """視線が通過窓(中心 ±1 継続時間)にどれだけ寄っているか.

    local ビューの窓は ±4 継続時間なので、201 ビンのうち通過窓は 1/4。
    **情報を持たない視線なら 0.25 になる。**
    対照は巡回ずらし —— 視線の形はそのままに、位置だけを壊す。

    物差しは 2 つ出す。**順序が意味を持つので、両方を残す。**

    * ``registered`` —— SPEC §7 に**測る前に**書いた物差し。正の寄与の質量が
      窓に落ちる割合。「窪みの壁を見て惑星だと判断している」という予測に対応する
    * ``posthoc`` —— 上が落ちた**後に**、実測(local 枝は正例 −1.26 / 負例 −8.01 と、
      符号を問わず大きさで効いていた)を見てから足した物差し。
      中央値からの隔たりの大きさ |cam − median| が窓に落ちる割合。
      **結果を見てから足した物差しなので、これを合格の根拠にはしない**
    """
    probs, cam = run_model(model, gview, lview)
    half = 201 / 8.0  # 1 継続時間 = 201/8 ビン
    lo = int(round(100 - half))
    hi = int(round(100 + half)) + 1
    called = (probs >= 0.5) & (label == 1)

    out: dict = {
        "window_bins": [lo, hi - 1],
        "uninformative_baseline": (hi - lo) / 201,
        "n_positive_called": int(called.sum()),
    }

    for key, weight in (
        ("registered", np.maximum(cam, 0.0)),
        ("posthoc", np.abs(cam - np.median(cam, axis=1, keepdims=True))),
    ):
        rng = np.random.default_rng(rng_seed)
        frac, ctrl, ok = _window_share(weight, lo, hi, rng)
        sel = called & ok
        out[key] = {
            "n": int(sel.sum()),
            "mean_in_window": float(frac[sel].mean()) if sel.any() else None,
            "median_in_window": float(np.median(frac[sel])) if sel.any() else None,
            "mean_control": float(ctrl[sel].mean()) if sel.any() else None,
            "fraction_above_control": (
                float((frac[sel] > ctrl[sel]).mean()) if sel.any() else None
            ),
        }

    # 枝ごとの寄与(どちらが判別を担っているか)。画面の説明の根拠になる
    _, cam_g_raw = run_model(model, gview, lview)  # noqa: F841 — local だけを使う
    with torch.no_grad():
        g = torch.from_numpy(np.ascontiguousarray(gview, dtype=np.float32)).unsqueeze(1)
        l = torch.from_numpy(np.ascontiguousarray(lview, dtype=np.float32)).unsqueeze(1)
        parts_g, parts_l = [], []
        for i in range(0, g.shape[0], 256):
            _, cg, cl = model(g[i : i + 256], l[i : i + 256])
            parts_g.append(cg.mean(dim=1).numpy())
            parts_l.append(cl.mean(dim=1).numpy())
    contrib_g = np.concatenate(parts_g)
    contrib_l = np.concatenate(parts_l)
    neg = (probs < 0.5) & (label == 0)
    out["branch_contribution"] = {
        "positive_called": {
            "global": float(contrib_g[called].mean()) if called.any() else None,
            "local": float(contrib_l[called].mean()) if called.any() else None,
        },
        "negative_called": {
            "global": float(contrib_g[neg].mean()) if neg.any() else None,
            "local": float(contrib_l[neg].mean()) if neg.any() else None,
        },
        "note": (
            "local 枝は「惑星である証拠」ではなく「この窪みは怪しい」という減点として働く。"
            "正例では減点が小さく、負例では大きい。符号ではなく大きさが判別を担っている"
        ),
    }
    return out


# ---------------------------------------------------------------- 本体


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--views", default="data/views")
    ap.add_argument("--models", default="data/models")
    ap.add_argument("--tag", default="cam")
    ap.add_argument("--skip-eyeballs", action="store_true")
    args = ap.parse_args()

    models = pathlib.Path(args.models)
    data = load_shards(pathlib.Path(args.views))
    splits = make_splits(data)
    te = splits["test"]
    gview, lview, label = data["gview"][te], data["lview"][te], data["label"][te]

    out: dict = {}

    # G-05 頭部の選定
    reports = {}
    for tag in ("cam", "astronet"):
        p = models / f"{tag}.json"
        if p.exists():
            reports[tag] = json.loads(p.read_text(encoding="utf-8"))
    if "cam" in reports and "astronet" in reports:
        a = reports["cam"]["test"]["auc"]
        b = reports["astronet"]["test"]["auc"]
        chosen = "cam" if (b - a) <= HEAD_DELTA_LIMIT else "astronet"
        out["heads"] = {"cam": a, "astronet": b, "delta": b - a, "chosen": chosen}
        print(f"G-05 頭部: cam {a:.4f} / astronet {b:.4f} / 差 {b - a:+.4f} → {chosen}")
    elif "cam" in reports:
        print("G-05: astronet の報告がまだ無い(cam のみ)")

    # G-04 陰性対照
    shuffled = models / "cam_shuffled.json"
    if shuffled.exists():
        out["control_shuffled_auc"] = json.loads(shuffled.read_text(encoding="utf-8"))["test"]["auc"]
        print(f"G-04 陰性対照 AUC: {out['control_shuffled_auc']:.4f}")
        out["control_separation"] = class_separation(models)
        cs = out["control_separation"]
        print(
            "G-04 クラス分離: 対照 {:.3f} SD(正例 {:.4f} / 負例 {:.4f}・出力の標準偏差 {:.5f})"
            " / 本モデル {:.3f} SD".format(
                cs["shuffled"]["separation_sd"],
                cs["shuffled"]["mean_positive"],
                cs["shuffled"]["mean_negative"],
                cs["shuffled"]["score_sd"],
                cs["real"]["separation_sd"],
            )
        )

    # G-06 二実装照合
    if (models / f"{args.tag}.onnx").exists():
        out["onnx_max_abs_diff"] = check_onnx(models, args.tag, gview, lview)
        print(f"G-06 二実装照合 最大絶対差: {out['onnx_max_abs_diff']:.3e}")

    if not args.skip_eyeballs:
        report = json.loads((models / f"{args.tag}.json").read_text(encoding="utf-8"))
        model = TransitNet(head=report["head"], width=report["width"])
        model.load_state_dict(torch.load(models / f"{args.tag}.pt", weights_only=True))
        print("目玉 1(検出限界)を測る…", flush=True)
        eye1 = detection_limit(model)
        print("目玉 2(視線)を測る…", flush=True)
        eye2 = gaze_mass(model, gview, lview, label)
        out["eyeballs"] = {"detection_limit": eye1, "gaze": eye2}
        print(json.dumps(eye2, ensure_ascii=False, indent=2))

    path = models / "measurements.json"
    prev = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    prev.update(out)
    path.write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
