"""MAST から Kepler 長時間ケイデンス光度曲線を取り、PDCSAP を連結する.

MAST の公開ツリー::

    https://archive.stsci.edu/pub/kepler/lightcurves/<KIC の上 4 桁>/<KIC 9 桁>/

このディレクトリが 404 を返す星がある(短時間ケイデンスのみ等)。**404 は異常ではない**
—— 取得できなかった星として記録し、先へ進む。
"""

from __future__ import annotations

import io
import re
import time as _time
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np

BASE = "https://archive.stsci.edu/pub/kepler/lightcurves"
# 同時接続はプロセス数 × QUARTER_THREADS になる。公開アーカイブに対して
# 100 本を超える同時接続を張らないよう、既定を 4 に抑えている
UA = "transit-lens/0.1 (educational, non-commercial; contact via github)"
_FILE_RE = re.compile(r"kplr(\d{9})-(\d+)_llc\.fits")


class NotInArchive(LookupError):
    """公開ツリーにその星のディレクトリが無い."""


@dataclass(frozen=True)
class LightCurve:
    kepid: int
    time: np.ndarray  # BKJD(= BJD − 2454833)
    flux: np.ndarray  # 四半期ごとに中央値で割った相対フラックス
    quarters: int


def _dir_url(kepid: int) -> str:
    s = f"{kepid:09d}"
    return f"{BASE}/{s[:4]}/{s}/"


def _get(url: str, retries: int = 4, timeout: float = 120.0) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise NotInArchive(url) from exc
            last = exc
        except Exception as exc:  # noqa: BLE001 — ネットワークは何でも飛んでくる
            last = exc
        _time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"取得できなかった: {url}") from last


def list_quarters(kepid: int) -> list[str]:
    """その星の長時間ケイデンス FITS のファイル名を、四半期の時刻順で返す."""
    html = _get(_dir_url(kepid)).decode("utf-8", "replace")
    names = sorted({m.group(0) for m in _FILE_RE.finditer(html)})
    return names


def fetch_quarter(kepid: int, name: str) -> bytes:
    return _get(_dir_url(kepid) + name)


def read_pdcsap(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    """FITS のバイト列から (時刻, PDCSAP フラックス) を返す.

    ``SAP_QUALITY != 0`` の点と、時刻またはフラックスが非有限の点を落とす。
    フラックスはこの四半期の中央値で割って規格化する。
    """
    from astropy.io import fits

    with fits.open(io.BytesIO(blob), memmap=False) as hdul:
        data = hdul[1].data
        t = np.asarray(data["TIME"], dtype=np.float64)
        f = np.asarray(data["PDCSAP_FLUX"], dtype=np.float64)
        q = np.asarray(data["SAP_QUALITY"], dtype=np.int64)

    good = (q == 0) & np.isfinite(t) & np.isfinite(f)
    t, f = t[good], f[good]
    if t.size == 0:
        return t, f
    med = np.median(f)
    if med == 0 or not np.isfinite(med):
        return t[:0], f[:0]
    return t, f / med


def load_light_curve(kepid: int, blobs: dict[str, bytes]) -> LightCurve:
    """四半期ごとの FITS バイト列をつないで 1 本の光度曲線にする."""
    times: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    used = 0
    for name in sorted(blobs):
        t, f = read_pdcsap(blobs[name])
        if t.size == 0:
            continue
        times.append(t)
        fluxes.append(f)
        used += 1
    if not times:
        raise ValueError(f"KIC {kepid}: 使える点が 1 つも無い")
    time = np.concatenate(times)
    flux = np.concatenate(fluxes)
    order = np.argsort(time)
    return LightCurve(kepid=kepid, time=time[order], flux=flux[order], quarters=used)
