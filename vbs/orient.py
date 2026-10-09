"""紙面の向きの判定.

動画の回転情報（表示行列）とは別に、本そのものが画面の中で寝ていることがある
（スマートフォンを縦に構えて見開きを横向きに撮る等）。写っている文字から
正しい向きを推定し、np.rot90 の k（反時計回り90°の回数）で返す。
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from vbs.imgio import imwrite
from vbs.ocr import _NO_WINDOW, OcrError, REPO_TESSDATA, find_tesseract

OSD_MAX_SIDE = 1800
MIN_CONF = 1.5


def detect_osd(img_bgr: np.ndarray) -> tuple[int, float] | None:
    """Tesseract OSD で向きを推定する。(k, 信頼度) か、判定できなければ None."""
    try:
        exe = find_tesseract()
    except OcrError:
        return None
    tessdata = Path(os.environ.get("VBS_TESSDATA") or REPO_TESSDATA)
    if not (tessdata / "osd.traineddata").exists():
        return None
    h, w = img_bgr.shape[:2]
    s = min(1.0, OSD_MAX_SIDE / max(h, w))
    small = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img_bgr
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "osd.png"
        imwrite(p, small)
        r = subprocess.run([str(exe), str(p), "-", "--psm", "0", "--tessdata-dir", str(tessdata)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=dict(os.environ, OMP_THREAD_LIMIT="1"), creationflags=_NO_WINDOW)
    out = r.stdout + r.stderr
    m = re.search(r"Rotate:\s*(\d+)", out)
    c = re.search(r"Orientation confidence:\s*([\d.]+)", out)
    if not m or not c:
        return None
    rotate_cw = int(m.group(1))  # 時計回りにこの角度回すと正立する
    k = ((360 - rotate_cw) // 90) % 4
    return k, float(c.group(1))


def vote(images: list[np.ndarray]) -> dict:
    """複数の画像の判定を信頼度で重み付けして多数決する."""
    results = []
    for img in images:
        r = detect_osd(img)
        if r:
            results.append({"k": r[0], "conf": round(r[1], 2)})
    score: Counter[int] = Counter()
    for r in results:
        if r["conf"] >= MIN_CONF:
            score[r["k"]] += r["conf"]
    if not score:
        return {"k": 0, "decided": False, "votes": results}
    k, best = score.most_common(1)[0]
    total = sum(score.values())
    return {"k": int(k), "decided": best >= 0.6 * total, "votes": results}
