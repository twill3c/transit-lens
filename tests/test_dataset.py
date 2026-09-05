"""台帳・分割・生成物の検査(品質ゲート G-01 / G-02 / G-03 / G-10).

生成物(shard)を要する検査は、shard が無ければ skip する ——
`training/build_dataset.py` の走行には 2.5 時間ほどかかるため。
**skip は緑ではない。** 出荷前には必ず走らせる。
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from fetch_labels import ORACLE_COUNTS, ORACLE_TOTAL, check  # noqa: E402
from train import make_splits, star_split  # noqa: E402

LABELS = ROOT / "data/labels/dr24_tce.csv"
VIEWS = ROOT / "data/views"


@pytest.mark.validation
def test_label_counts_match_the_published_values():
    """T-014 / G-01 —— 台帳の件数が published 値と一致する.

    期待値の出所: Shallue & Vanderburg (2018) AJ 155:94。
    全 TCE 20,367 件、教師ラベル付き 15,737 件(PC 3,600 / AFP 9,596 / NTP 2,541)。
    **ここは定数で書く。** 台帳が更新されたのなら、更新されたと分かる形で落ちてほしい。
    """
    if not LABELS.exists():
        pytest.skip("data/labels/dr24_tce.csv が無い(training/fetch_labels.py を走らせる)")
    counts = check(LABELS)
    assert sum(counts[k] for k in ORACLE_COUNTS) == 15737
    assert ORACLE_TOTAL == 20367


@pytest.mark.unit
def test_star_split_is_deterministic_and_roughly_balanced():
    """T-015 / G-03 —— 分割は KIC から決まり、星が増えても既存の帰属が動かない.

    期待値の出所: SPEC §6.2(80/10/10・種 20260905)。
    割合は厳密には 80/10/10 にならない(ハッシュの粒度)ので、
    **件数ではなく「偏りが 2 ポイント以内」という不変量**で書く。
    """
    kepids = list(range(1000000, 1050000, 3))
    assign = [star_split(k) for k in kepids]
    # 決定論性: 二度呼んでも同じ
    assert assign == [star_split(k) for k in kepids]
    n = len(kepids)
    for name, want in (("train", 0.80), ("val", 0.10), ("test", 0.10)):
        got = assign.count(name) / n
        assert abs(got - want) < 0.02, f"{name} が {got:.3f}(目標 {want})"


def _load_views():
    shards = sorted(VIEWS.glob("shard_*.npz"))
    if not shards:
        pytest.skip("data/views/shard_*.npz が無い(training/build_dataset.py を走らせる)")
    keys = ["kepid", "plnt", "label", "av", "gview", "lview", "period", "duration"]
    out: dict[str, list] = {k: [] for k in keys}
    for path in shards:
        with np.load(path, allow_pickle=False) as z:
            for k in keys:
                out[k].append(z[k])
    return {k: np.concatenate(v) for k, v in out.items()}


@pytest.mark.validation
def test_generated_views_have_no_non_finite_values():
    """T-016 / G-02 —— 生成したビューに NaN / Inf が 1 つも無い.

    期待値の出所: SPEC §5(空ビンは補間、埋められないものは捨てる)。
    件数ではなく**不変量**で書く —— 取得できる星はアーカイブの都合で動く。
    """
    data = _load_views()
    for key, width in (("gview", 2001), ("lview", 201)):
        arr = data[key]
        assert arr.shape[1] == width
        assert np.isfinite(arr).all(), f"{key} に非有限値がある"
        # 正規化の定義(中央値 0・最小値 −1)が全件で成り立っている
        assert np.allclose(np.median(arr, axis=1), 0.0, atol=1e-5), f"{key} の中央値が 0 でない"
        assert np.allclose(arr.min(axis=1), -1.0, atol=1e-5), f"{key} の最小値が −1 でない"


@pytest.mark.validation
def test_splits_share_no_star():
    """T-015 / G-03 —— 実際の生成物でも、集合間に共通 KIC が 0 件."""
    data = _load_views()
    splits = make_splits(data)  # 共有があれば SystemExit で落ちる
    stars = {k: set(data["kepid"][v].tolist()) for k, v in splits.items()}
    assert stars["train"] & stars["test"] == set()
    assert stars["train"] & stars["val"] == set()
    assert stars["val"] & stars["test"] == set()
    assert sum(len(s) for s in stars.values()) == len(set(data["kepid"].tolist()))


@pytest.mark.validation
def test_every_tce_in_the_views_is_a_labelled_one():
    """T-016 / G-02 —— 生成物に UNK が混ざっていない(取りこぼしの不在)."""
    data = _load_views()
    assert set(np.unique(data["av"]).tolist()) <= {"PC", "AFP", "NTP"}
    # ラベルの符号化が av と食い違っていない
    assert ((data["av"] == "PC") == (data["label"] == 1)).all()


@pytest.mark.validation
def test_no_cyrillic_in_japanese_text():
    """T-017 / G-10 —— 日本語本文にキリル文字が混入していない.

    字形が近く目視では気づけないので、機械に見張らせる([[fleet-cyrillic-leak]])。
    """
    proc = subprocess.run(
        [sys.executable, str(ROOT / "harness/text_hygiene.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
