"""NASA Exoplanet Archive(TAP)から DR24 TCE 台帳を取り、件数オラクルと照合する.

出力: ``data/labels/dr24_tce.csv``

件数オラクルは Shallue & Vanderburg (2018) の公表値:

* 全 TCE 20,367 件
* 教師ラベル付き 15,737 件(PC 3,600 / AFP 9,596 / NTP 2,541)

食い違ったら **黙って進まず落とす**。台帳が更新されたのなら、更新されたと分かる形で
落ちてほしい(G-01)。
"""

from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import sys
import urllib.parse
import urllib.request

TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
COLUMNS = [
    "kepid",
    "tce_plnt_num",
    "tce_period",
    "tce_time0bk",
    "tce_duration",
    "tce_depth",
    "tce_prad",
    "tce_model_snr",
    "tce_max_mult_ev",
    "tce_num_transits",
    "tce_impact",
    "tce_steff",
    "tce_sradius",
    "av_training_set",
    "av_pred_class",
]

# 公表値(Shallue & Vanderburg 2018, AJ 155:94, Table 1 と §2)
ORACLE_TOTAL = 20367
ORACLE_COUNTS = {"PC": 3600, "AFP": 9596, "NTP": 2541}


def fetch(timeout: float = 600.0) -> bytes:
    query = f"select {','.join(COLUMNS)} from q1_q17_dr24_tce"
    url = f"{TAP}?{urllib.parse.urlencode({'query': query, 'format': 'csv'})}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def check(path: pathlib.Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    counts = collections.Counter(r["av_training_set"] for r in rows)
    problems = []
    if len(rows) != ORACLE_TOTAL:
        problems.append(f"全 TCE 数 {len(rows)} ≠ 公表値 {ORACLE_TOTAL}")
    for label, want in ORACLE_COUNTS.items():
        if counts[label] != want:
            problems.append(f"{label} {counts[label]} 件 ≠ 公表値 {want} 件")
    if problems:
        raise SystemExit("G-01 不通過:\n  " + "\n  ".join(problems))
    return dict(counts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/labels/dr24_tce.csv")
    ap.add_argument("--offline", action="store_true", help="取得せず既存ファイルを照合するだけ")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    if not args.offline:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(fetch())

    counts = check(out)
    labeled = sum(counts[k] for k in ORACLE_COUNTS)
    print(f"G-01 通過: 全 {ORACLE_TOTAL} 件 / 教師ラベル付き {labeled} 件 {counts}")
    print(f"→ {out} ({out.stat().st_size:,} バイト)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
