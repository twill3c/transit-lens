"""モデルの構造が持つべき性質の検査.

**この画面の主張の土台**は「ロジットは視線の平均に厳密に分解できる」である。
近似ではないと言い切るなら、それは検査で押さえておかなければならない。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from tl.model import TransitNet, _tower_out_len  # noqa: E402

pytestmark = pytest.mark.unit


def test_tower_output_lengths_are_what_the_screen_claims():
    """T-018 / F-06 —— 枝の出力長は解析的に決まる.

    期待値の出所: maxpool の定義 floor((L − k)/s) + 1 を手で回した値。
    画面はこの長さを `pred` から読んで表示するので、ここがずれても画面は嘘をつかない
    —— が、ずれたこと自体には気づきたい。
    """
    # global: 2001 → 999 → 498 → 247 → 122 → 59(kernel 5 / stride 2 を 5 回)
    assert _tower_out_len(2001, 5, 5, 2) == 59
    # local: 201 → 98 → 46(kernel 7 / stride 2 を 2 回)
    assert _tower_out_len(201, 2, 7, 2) == 46

    m = TransitNet(head="cam", width=1.0)
    assert m.global_out_len == 59
    assert m.local_out_len == 46


def test_logit_is_exactly_the_mean_of_the_gaze():
    """T-019 / F-06 / SPEC §6.1 —— CAM 型の頭部は厳密な分解を持つ.

    期待値の出所: 数式。頭部が大域平均プーリング + 全結合 1 層なら
        logit = Σ_c w_c · mean_t f_c[t] + b = mean_t (Σ_c w_c f_c[t]) + b
    が恒等式として成り立つ。**近似ではないので、許容は浮動小数の丸めだけ。**
    """
    torch.manual_seed(0)
    m = TransitNet(head="cam", width=0.5)
    m.eval()
    g = torch.randn(4, 1, 2001)
    l = torch.randn(4, 1, 201)
    with torch.no_grad():
        logit, cam_g, cam_l = m(g, l)
        rebuilt = cam_g.mean(dim=1) + cam_l.mean(dim=1) + m.classifier.bias[0]
    assert torch.max(torch.abs(logit - rebuilt)).item() < 1e-5


def test_astronet_head_returns_zero_gaze_rather_than_a_fake_one():
    """T-020 / SPEC §6.1 —— 出せないものは 0 で返す(黙って近似値を作らない).

    期待値の出所: 実装契約。AstroNet 型の頭部では前向き計算だけで視線は出ない。
    ここで適当な値を返すと、画面は「視線」として表示してしまう。
    """
    m = TransitNet(head="astronet", width=0.35)
    m.eval()
    with torch.no_grad():
        logit, cam_g, cam_l = m(torch.randn(2, 1, 2001), torch.randn(2, 1, 201))
    assert logit.shape == (2,)
    assert torch.count_nonzero(cam_g).item() == 0
    assert torch.count_nonzero(cam_l).item() == 0


def test_onnx_export_matches_pytorch():
    """T-021 / G-06 —— 書き出した ONNX が PyTorch と同じ数を返す.

    期待値の出所: PyTorch 側の出力。**基準は SPEC G-06 の 1e-5。**
    ここは学習済みモデルでも measure.py が全件で測り直すが、
    構造が変わったときに真っ先に落ちるのはこの検査である。
    """
    onnxruntime = pytest.importorskip("onnxruntime")
    from train import export_onnx

    torch.manual_seed(1)
    m = TransitNet(head="cam", width=0.35)
    m.eval()
    tmp = ROOT / "data" / "models"
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / "_test_export.onnx"
    export_onnx(m, path)
    try:
        g = np.random.default_rng(3).standard_normal((3, 1, 2001)).astype(np.float32)
        l = np.random.default_rng(4).standard_normal((3, 1, 201)).astype(np.float32)
        with torch.no_grad():
            want, want_g, want_l = m(torch.from_numpy(g), torch.from_numpy(l))
        sess = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"global_view": g, "local_view": l})
        assert np.max(np.abs(got[0] - want.numpy())) < 1e-5
        assert got[1].shape == (3, 59)
        assert got[2].shape == (3, 46)
        assert np.max(np.abs(got[1] - want_g.numpy())) < 1e-4
    finally:
        path.unlink(missing_ok=True)
