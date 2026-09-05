"""二枝 1D CNN を学習し、保留集合で測り、ONNX に書き出す.

分割は **恒星単位**(SPEC §6.2 / G-03)。同じ KIC の TCE は同じ光度曲線から
作られているので、TCE 単位で切ると学習集合の情報が保留集合に漏れる。

``--shuffle-labels`` は陰性対照(G-04)。ラベルを無作為に入れ替えて学習し、
試験 AUC が 0.5 付近に落ちることを確かめる。落ちなければ学習経路に漏れがある。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tl.model import TransitNet  # noqa: E402

SPLIT_SEED = "20260905"
SPLIT_FRACTIONS = (0.80, 0.10)  # train, val(残りが test)


def star_split(kepid: int) -> str:
    """KIC から決定論的に train / val / test を決める.

    ハッシュを使うので、星の集合が増えても既存の星の帰属は変わらない。
    """
    digest = hashlib.sha1(f"{SPLIT_SEED}:{kepid}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    if u < SPLIT_FRACTIONS[0]:
        return "train"
    if u < SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]:
        return "val"
    return "test"


def load_shards(view_dir: pathlib.Path) -> dict[str, np.ndarray]:
    shards = sorted(view_dir.glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"shard が無い: {view_dir}")
    keys = [
        "kepid", "plnt", "label", "av", "period", "t0", "duration",
        "depth", "prad", "snr", "mes", "ntrans", "gview", "lview",
    ]
    out: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for path in shards:
        with np.load(path, allow_pickle=False) as z:
            for k in keys:
                out[k].append(z[k])
    return {k: np.concatenate(v) for k, v in out.items()}


def make_splits(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    assign = np.array([star_split(int(k)) for k in data["kepid"]])
    splits = {name: np.flatnonzero(assign == name) for name in ("train", "val", "test")}
    # G-03: 集合間に共通の KIC が 1 件でもあれば落とす
    stars = {name: set(data["kepid"][idx].tolist()) for name, idx in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = stars[a] & stars[b]
        if shared:
            raise SystemExit(f"G-03 不通過: {a} と {b} が KIC を {len(shared)} 件共有している")
    return splits


def evaluate(
    model: nn.Module, g: torch.Tensor, l: torch.Tensor, y: np.ndarray, batch: int = 256
) -> dict[str, float]:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, g.shape[0], batch):
            logit, _, _ = model(g[i : i + batch], l[i : i + batch])
            scores.append(torch.sigmoid(logit).numpy())
    p = np.concatenate(scores)
    return {
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "precision@0.5": float(((p >= 0.5) & (y == 1)).sum() / max(1, (p >= 0.5).sum())),
        "recall@0.5": float(((p >= 0.5) & (y == 1)).sum() / max(1, (y == 1).sum())),
        "n": int(y.size),
        "positives": int((y == 1).sum()),
    }


def predict(model: nn.Module, g: torch.Tensor, l: torch.Tensor, batch: int = 256) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, g.shape[0], batch):
            logit, _, _ = model(g[i : i + batch], l[i : i + batch])
            out.append(torch.sigmoid(logit).numpy())
    return np.concatenate(out)


def export_onnx(model: nn.Module, path: pathlib.Path) -> None:
    model.eval()
    dummy = (torch.zeros(1, 1, 2001), torch.zeros(1, 1, 201))
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["global_view", "local_view"],
        output_names=["logit", "cam_global", "cam_local"],
        dynamic_axes={
            "global_view": {0: "batch"},
            "local_view": {0: "batch"},
            "logit": {0: "batch"},
            "cam_global": {0: "batch"},
            "cam_local": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--views", default="data/views")
    ap.add_argument("--out", default="data/models")
    ap.add_argument("--head", choices=["cam", "astronet"], default="cam")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    # 学習率は実測で決めた(2026-09-05・部分データ 12,594 TCE・幅 0.5・3 epoch):
    #   1e-4 → 検証 AUC 0.729 / 3e-4 → 0.821
    # 参照実装は 1e-5 で 25,000 step 回すが、ここでは時間の都合で step 数を減らすので、
    # 勾配クリップ(参照実装と同じ 1.0)を効かせたうえで学習率を上げている
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--shuffle-labels", action="store_true", help="陰性対照(G-04)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_shards(pathlib.Path(args.views))
    splits = make_splits(data)
    print(
        "TCE {} 件 / 星 {} 個 — train {} / val {} / test {}".format(
            data["label"].size,
            len(set(data["kepid"].tolist())),
            *(splits[k].size for k in ("train", "val", "test")),
        ),
        flush=True,
    )

    y_all = data["label"].astype(np.float32)
    if args.shuffle_labels:
        rng = np.random.default_rng(args.seed)
        # 学習集合の中だけで入れ替える。保留集合の正解は動かさない
        tr = splits["train"]
        y_all = y_all.copy()
        y_all[tr] = rng.permutation(y_all[tr])

    tensors = {}
    for name, idx in splits.items():
        tensors[name] = (
            torch.from_numpy(data["gview"][idx]).unsqueeze(1),
            torch.from_numpy(data["lview"][idx]).unsqueeze(1),
            y_all[idx],
        )

    model = TransitNet(head=args.head, width=args.width)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"頭部 {args.head} / 幅 {args.width} / パラメータ {n_params:,}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    g_tr, l_tr, y_tr = tensors["train"]
    y_tr_t = torch.from_numpy(y_tr)
    g_va, l_va, y_va = tensors["val"]

    best = {"auc": -1.0, "epoch": -1}
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (args.head + ("_shuffled" if args.shuffle_labels else ""))
    ckpt = out_dir / f"{tag}.pt"
    history = []

    n = g_tr.shape[0]
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        t0 = time.time()
        total = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            opt.zero_grad(set_to_none=True)
            logit, _, _ = model(g_tr[idx], l_tr[idx])
            loss = loss_fn(logit, y_tr_t[idx])
            loss.backward()
            # AstroNet の設定にある勾配クリップ(clip_gradient_norm: 1.0)。
            # 正規化層を持たない深い畳み込みなので、これが無いと学習率を上げられない
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total += float(loss) * idx.numel()
        val = evaluate(model, g_va, l_va, y_va)
        history.append({"epoch": epoch, "loss": total / n, **val, "seconds": round(time.time() - t0, 1)})
        print(
            f"  epoch {epoch:3d}  loss {total / n:.4f}  val AUC {val['auc']:.4f}  "
            f"({history[-1]['seconds']:.0f}s)",
            flush=True,
        )
        if val["auc"] > best["auc"]:
            best = {"auc": val["auc"], "epoch": epoch}
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    g_te, l_te, y_te = tensors["test"]
    report = {
        "tag": tag,
        "head": args.head,
        "width": args.width,
        "params": n_params,
        "epochs": args.epochs,
        "lr": args.lr,
        "clip": args.clip,
        "batch": args.batch,
        "seed": args.seed,
        "shuffled_labels": args.shuffle_labels,
        "best_epoch": best["epoch"],
        "split_seed": SPLIT_SEED,
        "counts": {k: int(v.size) for k, v in splits.items()},
        "val": evaluate(model, g_va, l_va, y_va),
        "test": evaluate(model, g_te, l_te, y_te),
        "history": history,
    }
    (out_dir / f"{tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        out_dir / f"{tag}_scores.npz",
        kepid=data["kepid"][splits["test"]],
        plnt=data["plnt"][splits["test"]],
        av=data["av"][splits["test"]],
        label=data["label"][splits["test"]],
        score=predict(model, g_te, l_te),
    )
    if not args.shuffle_labels:
        export_onnx(model, out_dir / f"{tag}.onnx")

    print(json.dumps({k: report[k] for k in ("val", "test", "best_epoch")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
