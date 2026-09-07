"""中断と再開の検査.

**この機では背景の学習がセッション境界で死ぬ**(loop_002 で二度実測)。
そこで毎 epoch 再開点を書いているが、この仕掛けには黙って壊れる道が二つある:

* 再開したのに epoch や最良値が引き継がれない → **短い学習を長い学習と取り違える**
* 完走しても再開点が残る → 次に同じタグで回したとき**黙って前回の続きから始まる**
  (条件を変えたつもりが変わっていない)

どちらもログを読まなければ気づけないので、機械に見張らせる。
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"

pytestmark = pytest.mark.integration


def _write_toy_shard(views: pathlib.Path, n_stars: int = 240, seed: int = 7) -> None:
    """分割が成立する程度の小さな合成 shard を書く(星ごとに 1 TCE)."""
    rng = np.random.default_rng(seed)
    views.mkdir(parents=True, exist_ok=True)
    label = (rng.random(n_stars) < 0.25).astype(np.int8)
    # 正例だけ中心に窪みを置く。学習が動くことが分かればよいので形は簡素でよい
    gview = rng.normal(0, 0.05, (n_stars, 2001)).astype(np.float32)
    lview = rng.normal(0, 0.05, (n_stars, 201)).astype(np.float32)
    gview[label == 1, 995:1006] -= 1.0
    lview[label == 1, 95:106] -= 1.0
    gview -= np.median(gview, axis=1, keepdims=True)
    lview -= np.median(lview, axis=1, keepdims=True)
    gview /= np.abs(gview.min(axis=1, keepdims=True))
    lview /= np.abs(lview.min(axis=1, keepdims=True))
    np.savez_compressed(
        views / "shard_0000.npz",
        kepid=np.arange(1_000_000, 1_000_000 + n_stars, dtype=np.int32),
        plnt=np.ones(n_stars, dtype=np.int8),
        label=label,
        av=np.where(label == 1, "PC", "AFP"),
        period=np.full(n_stars, 5.0),
        t0=np.zeros(n_stars),
        duration=np.full(n_stars, 0.2),
        depth=np.full(n_stars, 500.0),
        prad=np.full(n_stars, 2.0),
        snr=np.full(n_stars, 20.0),
        mes=np.full(n_stars, 20.0),
        ntrans=np.full(n_stars, 100.0),
        gview=gview,
        lview=lview,
    )


def _train(views: pathlib.Path, out: pathlib.Path, epochs: int, resume: bool) -> str:
    cmd = [
        str(PY), str(ROOT / "training" / "train.py"),
        "--views", str(views), "--out", str(out),
        "--head", "cam", "--width", "0.35",
        "--epochs", str(epochs), "--batch", "64", "--threads", "2",
        "--tag", "toy",
    ]
    if resume:
        cmd.append("--resume")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


@pytest.mark.skipif(not PY.exists(), reason=".venv が無い")
def test_resume_continues_and_is_cleaned_up(tmp_path: pathlib.Path):
    """T-022 —— 中断した学習が続きから再開し、完走したら再開点が消える.

    期待値の出所: 実装契約(SPEC の「長い学習は中断される」)。
    件数ではなく**不変量**で書く —— 再開後の history の長さが総 epoch 数に一致すること、
    完走後に再開点が残っていないこと。
    """
    views = tmp_path / "views"
    out = tmp_path / "models"
    _write_toy_shard(views)

    # 1 回目: 2 epoch だけ回す(= 途中で止まった状態の代役)
    _train(views, out, epochs=2, resume=False)
    resume_path = out / "toy_resume.pt"
    report = json.loads((out / "toy.json").read_text(encoding="utf-8"))
    assert len(report["history"]) == 2
    # **完走したら再開点は残らない**
    assert not resume_path.exists(), "完走後に再開点が残っている"

    # 再開点が無い状態で --resume を付けても、最初から回るだけで落ちない
    stdout = _train(views, out, epochs=3, resume=True)
    assert "から再開" not in stdout, "再開点が無いのに再開したと言っている"
    report = json.loads((out / "toy.json").read_text(encoding="utf-8"))
    assert len(report["history"]) == 3
    assert report["history"][0]["epoch"] == 1


@pytest.mark.skipif(not PY.exists(), reason=".venv が無い")
def test_resume_picks_up_where_it_stopped(tmp_path: pathlib.Path):
    """T-023 —— 再開点があるときは、その続きの epoch から始まる.

    期待値の出所: 実装契約。再開点は epoch を持っているので、
    再開後の history は「残りの epoch ぶん」だけ増える。
    """
    views = tmp_path / "views"
    out = tmp_path / "models"
    _write_toy_shard(views)

    # 3 epoch 回して完走 → 再開点は消える。そこで**完走前の状態**を自前で作る
    _train(views, out, epochs=3, resume=False)

    # 手で 3 epoch ぶんの再開点を書く(中断を模す)
    import torch

    sys.path.insert(0, str(ROOT / "training"))
    from tl.model import TransitNet  # noqa: E402

    model = TransitNet(head="cam", width=0.35)
    model.load_state_dict(torch.load(out / "toy.pt", weights_only=True))
    opt = torch.optim.Adam(model.parameters(), 3e-4)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "history": [{"epoch": i, "auc": 0.5} for i in range(1, 4)],
            "best": {"auc": 0.5, "epoch": 1},
            "epoch": 3,
        },
        out / "toy_resume.pt",
    )

    stdout = _train(views, out, epochs=5, resume=True)
    assert "epoch 4 から" in stdout, stdout
    report = json.loads((out / "toy.json").read_text(encoding="utf-8"))
    # 引き継いだ 3 件 + 新しい 2 件
    assert len(report["history"]) == 5
    assert [h["epoch"] for h in report["history"]] == [1, 2, 3, 4, 5]
    assert not (out / "toy_resume.pt").exists(), "完走後に再開点が残っている"
