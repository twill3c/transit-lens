"""二枝 1D CNN(AstroNet の local_global 構成を土台にする).

頭部は二択で、SPEC §6.1 の規則で選ぶ:

* **H-A(CAM 型)** —— 各枝を大域平均プーリングして連結し、全結合 1 層でロジットにする。
  このとき ``logit = mean_t cam_g[t] + mean_t cam_l[t] + b`` が**厳密に**成り立つ。
  ``cam[t] = w · f[:, t]`` は前向き計算だけで出るので、ブラウザで勾配を求めずに
  「モデルがどこを見たか」を厳密に出せる
* **H-B(AstroNet 型)** —— 平坦化して全結合 512 を 4 段。published 構成に一致するが、
  視線は前向きだけでは出ない(遮蔽感度に頼ることになる)

``width`` は各層のフィルタ数の倍率。1.0 が published 構成。
"""

from __future__ import annotations

import torch
from torch import nn

GLOBAL_LEN = 2001
LOCAL_LEN = 201


def _filters(base: list[int], width: float) -> list[int]:
    return [max(4, int(round(c * width))) for c in base]


class ConvTower(nn.Module):
    """conv×2 → maxpool を繰り返す枝."""

    def __init__(self, channels: list[int], kernel: int, pool: int, stride: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_ch, out_ch, kernel, padding=kernel // 2),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(pool, stride=stride),
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _tower_out_len(length: int, n_blocks: int, pool: int, stride: int) -> int:
    for _ in range(n_blocks):
        length = (length - pool) // stride + 1
    return length


class TransitNet(nn.Module):
    """global 枝 + local 枝 + 頭部."""

    GLOBAL_FILTERS = [16, 32, 64, 128, 256]
    LOCAL_FILTERS = [16, 32]

    def __init__(self, head: str = "cam", width: float = 1.0, fc_units: int = 512):
        super().__init__()
        if head not in ("cam", "astronet"):
            raise ValueError(f"head は cam / astronet のいずれか: {head!r}")
        self.head_kind = head
        gf = _filters(self.GLOBAL_FILTERS, width)
        lf = _filters(self.LOCAL_FILTERS, width)
        self.global_tower = ConvTower(gf, kernel=5, pool=5, stride=2)
        self.local_tower = ConvTower(lf, kernel=5, pool=7, stride=2)

        self.global_out_len = _tower_out_len(GLOBAL_LEN, len(gf), 5, 2)
        self.local_out_len = _tower_out_len(LOCAL_LEN, len(lf), 7, 2)

        if head == "cam":
            self.classifier = nn.Linear(gf[-1] + lf[-1], 1)
        else:
            flat = gf[-1] * self.global_out_len + lf[-1] * self.local_out_len
            fc: list[nn.Module] = []
            in_dim = flat
            for _ in range(4):
                fc += [nn.Linear(in_dim, fc_units), nn.ReLU(inplace=True)]
                in_dim = fc_units
            fc.append(nn.Linear(in_dim, 1))
            self.classifier = nn.Sequential(*fc)

    def forward(
        self, gview: torch.Tensor, lview: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(ロジット, global の視線, local の視線) を返す.

        ``head='astronet'`` のときの視線は 0 埋め —— 前向きだけでは出ないので、
        「出せない」ことを値で表す(黙って近似値を返さない)。
        """
        fg = self.global_tower(gview)
        fl = self.local_tower(lview)

        if self.head_kind == "cam":
            w = self.classifier.weight[0]
            cg = fg.shape[1]
            wg = w[:cg].view(1, cg, 1)
            wl = w[cg:].view(1, fl.shape[1], 1)
            cam_g = (fg * wg).sum(dim=1)
            cam_l = (fl * wl).sum(dim=1)
            logit = cam_g.mean(dim=1) + cam_l.mean(dim=1) + self.classifier.bias[0]
            return logit, cam_g, cam_l

        flat = torch.cat([fg.flatten(1), fl.flatten(1)], dim=1)
        logit = self.classifier(flat).squeeze(1)
        zero_g = torch.zeros(fg.shape[0], fg.shape[2], device=fg.device, dtype=fg.dtype)
        zero_l = torch.zeros(fl.shape[0], fl.shape[2], device=fl.device, dtype=fl.dtype)
        return logit, zero_g, zero_l
