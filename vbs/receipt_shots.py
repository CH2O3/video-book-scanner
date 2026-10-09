"""レシートの動画から、レシートごとの良いフレームを選ぶ.

本は「めくって止める」ので静止区間で切り出せるが、レシートは手で持ったまま・手持ちのカメラで
撮ることが多く、画面全体が止まる時間がほとんどない。そこで、
1. 各フレームで「白い長方形の紙（レシート）」が画面の端に掛からずに写っているかを調べ、
2. 続けて写っている間を、同じレシートとしてまとめ（紙の中身の見た目で入れ替わりを見分ける）、
3. まとまりごとに、鮮明で、手の重なりが少なく、紙が欠けていないフレームを候補にする。
戻り値は、静止区間の抽出と同じ形（区間と候補フレームの番号）なので、後の処理はそのまま使える。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

WIDTH = 360            # 判定に使う縮小幅
MIN_AREA = 0.05        # 画面に対する紙の面積の下限
MIN_FILL = 0.72        # 紙の外接矩形のうち紙が占める割合の下限（手で大きく隠れていない）
GAP_SEC = 0.6          # これより長く写っていなければ、別のまとまり
SAME_CORR = 0.55       # 中身の見た目の相関がこれ未満なら、別のレシートに入れ替わった
MIN_GROUP_SEC = 0.25   # これより短く写っただけのまとまりは捨てる（通り過ぎただけ）
CAND_SPACING_SEC = 0.15
HAND_MAX = 0.08        # 紙の範囲に肌色がこれより多い（手で押さえたまま）なら採用しない
WEAK_RATIO = 0.3       # 一番良いフレームの鮮明さが、全体の中央値のこの割合未満なら採用しない


def _paper(img: np.ndarray) -> dict[str, Any] | None:
    """白い紙の一番大きいかたまりを探す."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V, S = hsv[..., 2], hsv[..., 1]
    # 紙は明るく色が薄い。露出に合わせて、明るい側の分布から閾値を決める
    v_thr = max(185, min(215, int(np.percentile(V, 97)) - 25))
    m = ((V > v_thr) & (S < 45)).astype(np.uint8)
    k = max(3, img.shape[1] // 60) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[i, cv2.CC_STAT_AREA])
    H, W = m.shape
    if area < MIN_AREA * H * W:
        return None
    comp = (lab == i).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    (cx, cy), (rw, rh), ang = rect
    fill = area / max(1.0, rw * rh)
    x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    e = 2
    touches = int(x <= e) + int(y <= e) + int(x + w >= W - e) + int(y + h >= H - e)
    return {"rect": rect, "fill": fill, "area": area / (H * W), "touches": touches, "bbox": (x, y, x + w, y + h)}


def _upright(gray: np.ndarray, rect) -> np.ndarray:
    """紙を長い辺が縦になるように切り出して、小さくそろえる（見分け用）."""
    (cx, cy), (rw, rh), ang = rect
    if rw > rh:
        rw, rh, ang = rh, rw, ang - 90
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rot = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]), borderValue=255)
    x0, y0 = int(cx - rw / 2), int(cy - rh / 2)
    crop = rot[max(0, y0):max(0, y0) + int(rh), max(0, x0):max(0, x0) + int(rw)]
    if crop.size == 0:
        return np.zeros((96, 32), np.float32)
    small = cv2.resize(crop, (32, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
    small = cv2.GaussianBlur(small, (3, 3), 0)
    return (small - small.mean()) / (small.std() + 1e-6)


def _skin(img: np.ndarray, bbox) -> float:
    x0, y0, x1, y1 = bbox
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[..., 1], ycc[..., 2]
    return float(((cr > 138) & (cr < 180) & (cb > 80) & (cb < 130)).mean())


def plan_receipt_shots(video_path: Path, a, progress: Callable[[float, str], None] | None = None
                       ) -> list[tuple[dict[str, Any], list[int]]]:
    """レシートごとのまとまりと、その候補フレーム（Analysis の番号）を返す."""
    from vbs.video import iter_frames

    index = {int(p): i for i, p in enumerate(a.pts)}
    frames: list[dict[str, Any]] = []
    total = max(1, len(a.pts))
    for n, (t, pts, img) in enumerate(iter_frames(Path(video_path), width=WIDTH, fmt="bgr24")):
        i = index.get(int(pts))
        if i is None:
            continue
        info = _paper(img)
        if info is None or info["touches"] or info["fill"] < MIN_FILL:
            frames.append({"i": i, "t": float(t), "ok": False})
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x0, y0, x1, y1 = info["bbox"]
            roi = gray[y0:y1, x0:x1]
            sharp = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0
            skin = _skin(img, info["bbox"])
            frames.append({"i": i, "t": float(t), "ok": True, "sig": _upright(gray, info["rect"]),
                           "score": sharp * info["fill"] ** 2 * max(0.05, 1.0 - 3.0 * skin),
                           "skin": skin, "fill": info["fill"]})
        if progress and n % 30 == 0:
            progress(min(0.99, n / total), f"レシートを探しています {t:.0f}秒")
    groups: list[list[dict]] = []
    cur: list[dict] = []
    last_t = -1e9
    for f in frames:
        if not f["ok"]:
            if cur and f["t"] - last_t > GAP_SEC:
                groups.append(cur)
                cur = []
            continue
        if cur:
            ref = max(cur, key=lambda x: x["score"])["sig"]
            same = float((ref * f["sig"]).mean()) >= SAME_CORR
            if not same or f["t"] - last_t > GAP_SEC:
                groups.append(cur)
                cur = []
        cur.append(f)
        last_t = f["t"]
    if cur:
        groups.append(cur)
    plan = []
    bests = []
    for g in groups:
        dur = g[-1]["t"] - g[0]["t"]
        if dur < MIN_GROUP_SEC:
            continue
        chosen: list[dict] = []
        for f in sorted(g, key=lambda x: -x["score"]):
            if all(abs(f["t"] - c["t"]) >= CAND_SPACING_SEC for c in chosen):
                chosen.append(f)
            if len(chosen) >= 3:
                break
        r = {"i0": g[0]["i"], "i1": g[-1]["i"], "start": g[0]["t"], "end": g[-1]["t"],
             "duration": float(dur), "short": False,
             "receipt": {"frames": len(g), "best_skin": round(chosen[0]["skin"], 3),
                         "best_fill": round(chosen[0]["fill"], 3)}}
        plan.append((r, [c["i"] for c in chosen]))
        bests.append(chosen[0]["score"])
    # 一番良いフレームでも、手が大きく重なる・ぶれているまとまりは採用しない（未採用として残す）
    ref = float(np.median(bests)) if bests else 0.0
    for (r, _), b in zip(plan, bests):
        if r["receipt"]["best_skin"] > HAND_MAX:
            r["exclude_reason"] = "receipt_hand"
        elif ref > 0 and b < WEAK_RATIO * ref:
            r["exclude_reason"] = "receipt_blur"
    return plan
