"""行の湾曲補正（横書き）.

綴じ目に向かって紙が反ると、本文の行が弓なりに曲がる。各行の中心線を検出して
多項式で近似し、行がまっすぐになるように縦方向へ画素を移す。

- 文字を作り出したり消したりはしない（画素を上下に動かすだけ）
- 行が十分に見つからない・変形が大きすぎる場合は補正しない
- 綴じ目付近の横方向の縮み（奥行きによる圧縮）は扱わない
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

WORK_W = 1200          # 解析の作業幅
MIN_LINES = 6          # これ未満の行しか見つからなければ補正しない
MAX_SHIFT_FRAC = 0.04  # 補正量の上限（ページ高さ比）。超えたら誤検出とみなす
MIN_BEND_PX = 2.5      # 行の平均の曲がりがこれ未満なら、ほぼ平らとみなして補正しない
DEG_X, DEG_Y = 4, 2    # 変位面の多項式の次数


def _text_mask(gray: np.ndarray) -> np.ndarray:
    block = max(15, (min(gray.shape) // 40) | 1)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 15)


def detect_lines(gray: np.ndarray) -> list[np.ndarray]:
    """本文の行ごとに中心線の点列 (x, y) を返す（作業座標）."""
    m = _text_mask(gray)
    h, w = m.shape
    # 文字をつないで行の帯にする（横に長く、縦は行間を越えない）
    kx = max(9, w // 40)
    band = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)))
    band = cv2.morphologyEx(band, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kx // 2, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(band)
    lines = []
    step = max(4, w // 150)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < 0.25 * w or bh > 0.06 * h or bh < 4:
            continue  # 短い・太すぎる（図や複数行の塊）ものは使わない
        pts = []
        comp = labels[y:y + bh, x:x + bw] == i
        txt = m[y:y + bh, x:x + bw] > 0
        for cx in range(0, bw, step):
            col = comp[:, cx:cx + step] & txt[:, cx:cx + step]
            ys = np.nonzero(col)[0]
            if ys.size >= 3:
                pts.append((x + cx + step / 2, y + float(np.median(ys))))
        if len(pts) >= 8:
            lines.append(np.array(pts, np.float64))
    return lines


def _design(x: np.ndarray, y: np.ndarray, w: float, h: float) -> np.ndarray:
    xn, yn = x / w * 2 - 1, y / h * 2 - 1
    cols = [xn ** i * yn ** j for i in range(DEG_X + 1) for j in range(DEG_Y + 1)]
    return np.stack(cols, axis=1)


def estimate(gray: np.ndarray) -> dict[str, Any] | None:
    """縦方向の変位面 d(x, y) の係数を推定する。補正しない場合は None."""
    g = gray
    s = WORK_W / g.shape[1]
    if s < 1:
        g = cv2.resize(g, (WORK_W, int(round(g.shape[0] * s))), interpolation=cv2.INTER_AREA)
    else:
        s = 1.0
    h, w = g.shape
    lines = detect_lines(g)
    if len(lines) < MIN_LINES:
        return {"applied": False, "reason": f"行が少ない（{len(lines)}）", "lines": len(lines)}
    X, Y, D = [], [], []
    for pts in lines:
        # 各行を3次で近似し、行の中央値の高さとのずれを変位とする
        c = np.polyfit(pts[:, 0], pts[:, 1], 3)
        fy = np.polyval(c, pts[:, 0])
        resid = np.abs(pts[:, 1] - fy)
        ok = resid < max(2.0, 3 * np.median(resid) + 1)
        target = float(np.median(fy[ok]))
        X.append(pts[ok, 0])
        Y.append(np.full(ok.sum(), target))
        D.append(fy[ok] - target)
    X, Y, D = np.concatenate(X), np.concatenate(Y), np.concatenate(D)
    A = _design(X, Y, w, h)
    coef, *_ = np.linalg.lstsq(A, D, rcond=None)
    # 外れ値を除いてもう一度
    r = np.abs(A @ coef - D)
    keep = r < max(1.5, 3 * np.median(r))
    coef, *_ = np.linalg.lstsq(A[keep], D[keep], rcond=None)
    fit_err = float(np.median(np.abs(A[keep] @ coef - D[keep])))
    # 行が見つかった範囲の外では多項式が暴れるので、範囲の端の値で止める
    bounds = [float(X.min()), float(Y.min()), float(X.max()), float(Y.max())]
    gx, gy = _grid(w, h, bounds, 40)
    dd = (_design(gx.ravel(), gy.ravel(), w, h) @ coef)
    max_shift = float(np.abs(dd).max())
    if max_shift > MAX_SHIFT_FRAC * h:
        return {"applied": False, "reason": f"補正量が大きすぎる（{max_shift / h:.3f}）", "lines": len(lines)}
    curvature = float(np.mean(np.abs(D)))
    if curvature / s < MIN_BEND_PX:
        return {"applied": False, "reason": "ほぼ平ら", "lines": len(lines), "mean_bend_px": round(curvature / s, 2)}
    return {"applied": True, "coef": coef.tolist(), "work_w": w, "work_h": h, "scale": s, "bounds": bounds,
            "lines": len(lines), "max_shift_px": round(max_shift / s, 1),
            "mean_bend_px": round(curvature / s, 2), "fit_err_px": round(fit_err / s, 2)}


def _grid(w: float, h: float, bounds: list[float], n: int) -> tuple[np.ndarray, np.ndarray]:
    """画像全体の格子点を、変位を評価する範囲（行のある範囲）に押し込めたもの."""
    gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, n))
    x0, y0, x1, y1 = bounds
    return np.clip(gx, x0, x1), np.clip(gy, y0, y1)


def apply(img: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """推定した変位面で画像を補正する（行をまっすぐにする）."""
    if not model or not model.get("applied"):
        return img
    H, W = img.shape[:2]
    coef = np.asarray(model["coef"])
    s = model["scale"]
    w, h = model["work_w"], model["work_h"]
    # 粗い格子で変位を計算して補間（全画素で多項式を評価しない）
    gx, gy = _grid(w, h, model["bounds"], 64)
    d = (_design(gx.ravel(), gy.ravel(), w, h) @ coef).reshape(gy.shape).astype(np.float32)
    d_full = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC) / s
    map_x, map_y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    map_y = map_y + d_full
    border = (255, 255, 255) if img.ndim == 3 else 255
    return cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def dewarp(img: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    model = estimate(gray) or {"applied": False}
    return apply(img, model), model
