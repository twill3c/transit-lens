"""光度曲線を取得し、TCE ごとの global / local ビューへ前処理する.

星ごとに: MAST から全四半期を取る → PDCSAP を連結 → その星の各 TCE について
通過を覆ってスプラインで割る → 位相折りたたみ → ビュー生成。

FITS はメモリ上でだけ扱い、ディスクに残さない(全体で数十 GB になるため)。
途中で止めても ``data/views/progress.jsonl`` から再開できる。

**404 は正常系** —— 公開ツリーに無い星は `absent` として記録して先へ進む。
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import json
import multiprocessing
import pathlib
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tl import kepler_io, preprocess, spline  # noqa: E402

LABELS = {"PC": 1, "AFP": 0, "NTP": 0}
QUARTER_THREADS = 4


def load_tces(path: pathlib.Path) -> dict[int, list[dict]]:
    """教師ラベル付きの TCE を星ごとにまとめる."""
    by_star: dict[int, list[dict]] = collections.defaultdict(list)
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["av_training_set"] not in LABELS:
                continue
            try:
                rec = {
                    "kepid": int(row["kepid"]),
                    "plnt": int(row["tce_plnt_num"]),
                    "period": float(row["tce_period"]),
                    "t0": float(row["tce_time0bk"]),
                    "duration": float(row["tce_duration"]) / 24.0,  # 時間 → 日
                    "depth": float(row["tce_depth"] or "nan"),
                    "prad": float(row["tce_prad"] or "nan"),
                    "snr": float(row["tce_model_snr"] or "nan"),
                    "mes": float(row["tce_max_mult_ev"] or "nan"),
                    "ntrans": float(row["tce_num_transits"] or "nan"),
                    "av": row["av_training_set"],
                }
            except ValueError:
                continue
            if not (rec["period"] > 0 and rec["duration"] > 0):
                continue
            by_star[rec["kepid"]].append(rec)
    return dict(by_star)


def process_star(job: tuple[int, list[dict]]) -> dict:
    """1 星ぶんを取得して前処理する。子プロセスで走る."""
    kepid, tces = job
    t_start = time.time()
    result: dict = {"kepid": kepid, "n_tce": len(tces), "views": [], "failed": []}
    try:
        names = kepler_io.list_quarters(kepid)
    except kepler_io.NotInArchive:
        result["status"] = "absent"
        return result
    except Exception as exc:  # noqa: BLE001
        result["status"] = "list_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    if not names:
        result["status"] = "no_quarters"
        return result

    blobs: dict[str, bytes] = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(QUARTER_THREADS) as ex:
            for name, blob in zip(names, ex.map(lambda n: kepler_io.fetch_quarter(kepid, n), names)):
                blobs[name] = blob
    except Exception as exc:  # noqa: BLE001
        result["status"] = "download_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    try:
        lc = kepler_io.load_light_curve(kepid, blobs)
    except Exception as exc:  # noqa: BLE001
        result["status"] = "read_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        blobs.clear()

    result["status"] = "ok"
    result["quarters"] = lc.quarters
    result["points"] = int(lc.time.size)

    for tce in tces:
        try:
            mask = preprocess.transit_mask(lc.time, tce["period"], tce["t0"], tce["duration"])
            trend = spline.fit_spline(lc.time, lc.flux, mask=mask)
            ok = np.isfinite(trend) & (trend != 0)
            if ok.sum() < 500:
                raise preprocess.ViewError(f"スプラインを当てられた点が {int(ok.sum())} しかない")
            t, f = lc.time[ok], lc.flux[ok] / trend[ok]
            folded = preprocess.phase_fold(t, tce["period"], tce["t0"])
            gv = preprocess.global_view(folded, f, tce["period"])
            lv = preprocess.local_view(folded, f, tce["period"], tce["duration"])
            if not (np.isfinite(gv).all() and np.isfinite(lv).all()):
                raise preprocess.ViewError("ビューに非有限値がある")
            result["views"].append({**tce, "global": gv.astype(np.float32), "local": lv.astype(np.float32)})
        except Exception as exc:  # noqa: BLE001
            result["failed"].append(
                {"plnt": tce["plnt"], "reason": f"{type(exc).__name__}: {exc}"}
            )
    result["seconds"] = round(time.time() - t_start, 2)
    return result


def write_shard(path: pathlib.Path, views: list[dict]) -> None:
    np.savez_compressed(
        path,
        kepid=np.array([v["kepid"] for v in views], dtype=np.int32),
        plnt=np.array([v["plnt"] for v in views], dtype=np.int8),
        label=np.array([LABELS[v["av"]] for v in views], dtype=np.int8),
        av=np.array([v["av"] for v in views]),
        period=np.array([v["period"] for v in views], dtype=np.float64),
        t0=np.array([v["t0"] for v in views], dtype=np.float64),
        duration=np.array([v["duration"] for v in views], dtype=np.float64),
        depth=np.array([v["depth"] for v in views], dtype=np.float64),
        prad=np.array([v["prad"] for v in views], dtype=np.float64),
        snr=np.array([v["snr"] for v in views], dtype=np.float64),
        mes=np.array([v["mes"] for v in views], dtype=np.float64),
        ntrans=np.array([v["ntrans"] for v in views], dtype=np.float64),
        gview=np.stack([v["global"] for v in views]),
        lview=np.stack([v["local"] for v in views]),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="data/labels/dr24_tce.csv")
    ap.add_argument("--out", default="data/views")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--shard-stars", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 星だけ処理する(試走用)")
    ap.add_argument("--only", type=int, nargs="*", help="この KIC だけ処理する")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"

    by_star = load_tces(pathlib.Path(args.labels))
    done: set[int] = set()
    if progress_path.exists():
        with progress_path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["kepid"])
                except (json.JSONDecodeError, KeyError):
                    continue

    stars = sorted(by_star)
    if args.only:
        stars = [k for k in stars if k in set(args.only)]
    stars = [k for k in stars if k not in done]
    if args.limit:
        stars = stars[: args.limit]

    total_tce = sum(len(by_star[k]) for k in stars)
    print(f"未処理 {len(stars)} 星 / TCE {total_tce} 件(済 {len(done)} 星)", flush=True)
    if not stars:
        return 0

    shard_no = len(list(out.glob("shard_*.npz")))
    buffer: list[dict] = []
    stars_in_buffer = 0
    counts = collections.Counter()
    t0 = time.time()

    jobs = [(k, by_star[k]) for k in stars]
    ctx = multiprocessing.get_context("spawn")
    with progress_path.open("a", encoding="utf-8") as plog, ctx.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_star, jobs, chunksize=1), 1):
            counts[res["status"]] += 1
            counts["tce_ok"] += len(res["views"])
            counts["tce_failed"] += len(res["failed"])
            buffer.extend(res["views"])
            stars_in_buffer += 1
            plog.write(
                json.dumps(
                    {k: v for k, v in res.items() if k != "views"}
                    | {"n_views": len(res["views"])},
                    ensure_ascii=False,
                )
                + "\n"
            )
            plog.flush()

            if stars_in_buffer >= args.shard_stars and buffer:
                write_shard(out / f"shard_{shard_no:04d}.npz", buffer)
                shard_no += 1
                buffer, stars_in_buffer = [], 0

            if i % 50 == 0:
                rate = i / (time.time() - t0)
                left = (len(jobs) - i) / rate / 60
                print(
                    f"  {i}/{len(jobs)} 星  {rate:.2f} 星/秒  残り {left:.0f} 分  "
                    f"TCE 成功 {counts['tce_ok']} 失敗 {counts['tce_failed']}",
                    flush=True,
                )

    if buffer:
        write_shard(out / f"shard_{shard_no:04d}.npz", buffer)

    print(json.dumps(dict(counts), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        traceback.print_exc()
        sys.exit(130)
