"""レシートの金額の行を、数字だけに絞って読み直す.

日本語のモデルは、レシートの細い数字を別の文字や別の数字に読み違えることがある（「¥865」→「oc」など）。
「合計」「対象」「消費税」の行だけ、見出しの右側を切り出して英数字のモデルで数字だけを読み直し、
その結果を金額の読み取りに使う。
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vbs.imgio import imread, imwrite
from vbs.ocr import _NO_WINDOW, REPO_TESSDATA, OcrError, find_tesseract

AMOUNT_KEYS = ("合計", "対象", "消費税", "税額", "領収金額", "お買上", "ご利用金額", "お支払", "総額")


def _lines(tsv: Path) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("level") == "5" and (r.get("text") or "").strip():
            groups[(r["block_num"], r["par_num"], r["line_num"])].append(r)
    out = []
    for ws in groups.values():
        ws.sort(key=lambda w: int(w["left"]))
        out.append({"words": ws, "text": "".join(w["text"] for w in ws),
                    "top": min(int(w["top"]) for w in ws),
                    "bottom": max(int(w["top"]) + int(w["height"]) for w in ws)})
    out.sort(key=lambda l: l["top"])
    return out


def _right_group(strip: np.ndarray) -> np.ndarray | None:
    """行の帯から、右端の文字のかたまり（右寄せの金額）を切り出す."""
    g = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip
    ink = (g < min(160, int(np.median(g)) - 40)).any(axis=0)
    cols = np.where(ink)[0]
    if cols.size == 0:
        return None
    h = strip.shape[0]
    gap = max(8, int(0.9 * h))  # 文字の間より広い空き
    right = cols[-1]
    left = right
    for c in cols[::-1]:
        if left - c > gap:
            break
        left = c
    pad = max(4, h // 4)
    return strip[:, max(0, left - pad):min(strip.shape[1], right + pad)]


def _read_digits(crop: np.ndarray, exe: Path, tessdata: Path) -> str:
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h = g.shape[0]
    if h < 48:  # 数字の高さを確保する
        s = 48 / max(1, h)
        g = cv2.resize(g, (int(g.shape[1] * s), 48), interpolation=cv2.INTER_CUBIC)
    g = cv2.copyMakeBorder(g, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "amount.png"
        imwrite(p, g)
        # 円記号は Y や \ として読ませて捨てる（数字だけに絞ると、円記号が数字に化ける）
        r = subprocess.run([str(exe), str(p), "-", "-l", "eng", "--psm", "7", "--tessdata-dir", str(tessdata),
                            "-c", "tessedit_char_whitelist=0123456789,Y\\()-"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=dict(os.environ, OMP_THREAD_LIMIT="1"), creationflags=_NO_WINDOW, timeout=60)
    t = r.stdout.strip()
    m = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)\)?\s*$", t)
    return m.group(1) if m else ""


def reread_amounts(image_path: Path, tsv_path: Path, scale: float) -> dict[int, str]:
    """金額の行の、見出しより右の部分を数字だけで読む。{行の順番: 数字} を返す."""
    try:
        exe = find_tesseract()
    except OcrError:
        return {}
    tessdata = Path(os.environ.get("VBS_TESSDATA") or REPO_TESSDATA)
    if not (tessdata / "eng.traineddata").exists():
        return {}
    img = imread(image_path)
    H, W = img.shape[:2]
    out = {}
    for i, line in enumerate(_lines(tsv_path)):
        text = line["text"]
        k = next((key for key in AMOUNT_KEYS if key in text), None)
        if not k:
            continue
        y0 = int(line["top"] / scale)
        y1 = int(line["bottom"] / scale)
        pad = max(4, (y1 - y0) // 4)
        strip = img[max(0, y0 - pad):min(H, y1 + pad), W // 3:W]  # 金額は右寄せ
        crop = _right_group(strip) if strip.size else None
        if crop is None or crop.shape[1] < 10:
            continue
        digits = _read_digits(crop, exe, tessdata)
        if re.fullmatch(r"\d{1,3}(,\d{3})+|\d+", digits or ""):
            out[i] = digits
    return out


def augmented_text(image_path: Path, tsv_path: Path, scale: float) -> tuple[str, dict[str, str]]:
    """OCRの行のうち金額の行を「見出し ¥数字」に置き換えた文字列と、置き換えの記録を返す."""
    lines = _lines(tsv_path)
    digits = reread_amounts(image_path, tsv_path, scale)
    notes = {}
    out = []
    for i, line in enumerate(lines):
        t = line["text"]
        if i in digits:
            k = next(key for key in AMOUNT_KEYS if key in t)
            head = t[:t.find(k) + len(k)]
            # 「(10%対象」のような見出しはそのまま残し、金額だけ差し替える
            new = f"{head} ¥{digits[i]}"
            if new != t:
                notes[new] = t
            t = new
        out.append(t)
    return "\n".join(out), notes


def reread_line(image_path: Path, tsv_path: Path, scale: float, contains: str, lang: str = "jpn") -> str | None:
    """指定の文字列を含む行を、拡大して1行として読み直す（日付の確かめ用）."""
    try:
        exe = find_tesseract()
    except OcrError:
        return None
    tessdata = Path(os.environ.get("VBS_TESSDATA") or REPO_TESSDATA)
    key = contains.replace(" ", "")
    line = next((l for l in _lines(tsv_path) if key and key[:8] in l["text"].replace(" ", "")), None)
    if line is None:
        return None
    img = imread(image_path)
    H, W = img.shape[:2]
    y0, y1 = int(line["top"] / scale), int(line["bottom"] / scale)
    pad = max(4, (y1 - y0) // 3)
    crop = img[max(0, y0 - pad):min(H, y1 + pad), :]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    s = max(1.0, 64 / max(1, g.shape[0]))
    g = cv2.resize(g, (int(g.shape[1] * s), int(g.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    g = cv2.copyMakeBorder(g, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "line.png"
        imwrite(p, g)
        r = subprocess.run([str(exe), str(p), "-", "-l", lang, "--psm", "7", "--tessdata-dir", str(tessdata)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=dict(os.environ, OMP_THREAD_LIMIT="1"), creationflags=_NO_WINDOW, timeout=60)
    return r.stdout.strip() or None
