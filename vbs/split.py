"""紙面検出・左右分割・補正（仕様8章の簡易版）.

ScanTailor統合までの暫定実装。湾曲補正はまだ行わず、
紙面範囲の切り出し・綴じ目での分割・傾き補正・照明ムラ補正を行う。
元画像は変更せず、ページ画像を別に生成する。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

WORK_W = 1000  # 検出処理の作業幅
WHITE_LEVEL = 235.0  # 照明補正後にこれ以上を白にする（裏写り対策）
BLACK_LEVEL = 10.0


def _scale_to(img: np.ndarray, w: int) -> tuple[np.ndarray, float]:
    s = w / img.shape[1]
    if s >= 1:
        return img, 1.0
    h = max(2, int(round(img.shape[0] * s)))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), s


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def text_mask(gray: np.ndarray) -> np.ndarray:
    """文字らしい暗い画素（局所的に暗い所）を1にする."""
    block = max(15, (min(gray.shape) // 40) | 1)
    m = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 15)
    return m


def detect_page_region(img: np.ndarray) -> tuple[tuple[int, int, int, int], list[str]]:
    """紙面（明るい大きな領域）の外接矩形を元画像座標で返す."""
    small, s = _scale_to(img, WORK_W)
    if small.ndim == 3:
        # 紙は明るく無彩色、明るい木目の机や手は色みがある → 明るさから彩度を引いた値で分ける
        g = paper_score(small)
    else:
        g = small
    H, W = g.shape
    blur = cv2.GaussianBlur(g, (7, 7), 0)
    bw = ((blur > paper_threshold(blur)) * 255).astype(np.uint8)
    k = max(5, W // 60)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k // 2 + 1, k // 2 + 1)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    warnings: list[str] = []
    if n <= 1:
        return (0, 0, img.shape[1], img.shape[0]), ["page_region_uncertain"]
    # 綴じ目の影で左右のページが分かれることがあるので、大きな領域はまとめて紙面とする
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = [1 + j for j in np.flatnonzero(areas >= 0.25 * areas.max())]
    x = int(min(stats[j, cv2.CC_STAT_LEFT] for j in big))
    y = int(min(stats[j, cv2.CC_STAT_TOP] for j in big))
    w = int(max(stats[j, cv2.CC_STAT_LEFT] + stats[j, cv2.CC_STAT_WIDTH] for j in big)) - x
    h = int(max(stats[j, cv2.CC_STAT_TOP] + stats[j, cv2.CC_STAT_HEIGHT] for j in big)) - y
    area = int(sum(areas[j - 1] for j in big))
    if area < 0.12 * W * H or (w * h) > 0.985 * W * H:
        # 紙面が小さすぎる / 画面全体が明るく背景と分離できない
        warnings.append("page_region_uncertain")
        if area < 0.12 * W * H:
            return (0, 0, img.shape[1], img.shape[0]), warnings
    edge = 2
    if x <= edge or y <= edge or x + w >= W - edge or y + h >= H - edge:
        warnings.append("page_touches_frame")  # 紙面の一部が画面外かもしれない
    m = int(round(0.006 * W))
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(W, x + w + m), min(H, y + h + m)
    inv = 1.0 / s
    return (int(x0 * inv), int(y0 * inv), int(round(x1 * inv)), int(round(y1 * inv))), warnings


def detect_gutter(img: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, float, list[str]]:
    """綴じ目のx座標（元画像座標）を返す。画像中央に固定しない.

    文字のない帯（内側余白）を探し、その中で最も暗い列（綴じ目の影）を選ぶ。
    """
    x0, y0, x1, y1 = bbox
    crop = img[y0:y1, x0:x1]
    g, s = _scale_to(_gray(crop), WORK_W)
    H, W = g.shape
    lo, hi = int(W * 0.30), int(W * 0.70)
    if hi - lo < 10:
        return (x0 + x1) // 2, 0.0, ["gutter_uncertain"]
    # 上下の端は手や影の影響を受けやすいので中央80%を使う
    band = g[int(H * 0.1): int(H * 0.9)]
    # 列ごとの「紙の明るさ」。文字が密でも影響されにくいよう上位10%の値を使う
    med = np.percentile(band, 90, axis=0).astype(np.float32)
    win = max(3, W // 100) | 1
    med_s = cv2.blur(med.reshape(1, -1), (win, 1)).ravel()
    tm = text_mask(band)
    dens = tm.mean(axis=0) / 255.0
    dens_s = cv2.blur(dens.reshape(1, -1).astype(np.float32), (win * 3, 1)).ravel()

    ref_dens = float(np.median(dens_s[int(W * 0.1): int(W * 0.9)])) + 1e-6
    empty = dens_s[lo:hi] < 0.25 * ref_dens
    dark = med_s[lo:hi]
    paper = float(np.percentile(med_s[int(W * 0.1): int(W * 0.9)], 75))
    warnings: list[str] = []
    conf = 0.0
    if empty.any():
        # 中央に近い空白帯を選ぶ
        idx = np.flatnonzero(empty)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        center = (hi - lo) / 2
        best = min(runs, key=lambda r: abs((r[0] + r[-1]) / 2 - center) - 0.5 * len(r))
        sub = dark[best[0]: best[-1] + 1]
        gx = lo + best[0] + int(np.argmin(sub))
        depth = (paper - float(sub.min())) / max(paper, 1.0)
        conf = min(1.0, 0.5 + depth * 5 + len(best) / (0.1 * W))
    else:
        gx = lo + int(np.argmin(dark))
        depth = (paper - float(dark.min())) / max(paper, 1.0)
        conf = min(1.0, depth * 5)
    if conf < 0.5:
        warnings.append("gutter_uncertain")
    return x0 + int(round(gx / s)), round(conf, 3), warnings


def estimate_skew(gray: np.ndarray, max_deg: float = 5.0, step: float = 0.25) -> float:
    """横書き行の投影プロファイルが最も鋭くなる角度を返す（度、反時計回り正）."""
    g, _ = _scale_to(gray, 800)
    m = text_mask(g)
    if m.mean() < 255 * 0.005:
        return 0.0
    h, w = m.shape
    c = (w / 2, h / 2)
    best, best_score = 0.0, -1.0
    for a in np.arange(-max_deg, max_deg + 1e-9, step):
        M = cv2.getRotationMatrix2D(c, float(a), 1.0)
        r = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST)
        prof = r.sum(axis=1, dtype=np.float64)
        score = float(np.var(np.diff(prof)))
        if score > best_score:
            best, best_score = float(a), score
    return best


def estimate_line_pitch(gray: np.ndarray) -> float | None:
    """横書きの行間隔(px, 元画像座標)を行方向の投影の自己相関から推定する."""
    m = text_mask(gray)
    prof = m.mean(axis=1).astype(np.float64)
    if prof.mean() < 255 * 0.005:
        return None
    prof -= prof.mean()
    n = len(prof)
    ac = np.correlate(prof, prof, mode="full")[n - 1:]
    if ac[0] <= 0:
        return None
    ac /= ac[0]
    lo = max(4, int(gray.shape[0] * 0.006))
    hi = min(n - 1, int(gray.shape[0] * 0.08))
    if hi <= lo + 2:
        return None
    seg = ac[lo:hi]
    peaks = [i for i in range(1, len(seg) - 1)
             if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.05]
    if not peaks:
        return None
    # 倍数（2行・3行分）の方が強く出ることがあるので、最大ピークの4割以上ある最短周期を採る
    top = max(seg[i] for i in peaks)
    for i in peaks:
        if seg[i] >= 0.4 * top:
            return float(lo + i)
    return None


def rotate_keep(img: np.ndarray, deg: float) -> np.ndarray:
    """回転して欠けないよう外側を広げる（余白は白）."""
    if abs(deg) < 0.05:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos + 0.5), int(h * cos + w * sin + 0.5)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    border = (255, 255, 255) if img.ndim == 3 else 255
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=border)


def normalize_illumination(img: np.ndarray) -> np.ndarray:
    """照明ムラを除き、紙を白に近づける。文字の細線は残す（二値化しない）."""
    small, s = _scale_to(img, 400)
    k = max(3, small.shape[1] // 30) | 1
    bg = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg = cv2.medianBlur(bg, k if k <= 255 else 255)
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    bg = cv2.resize(bg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    out = img.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0) * 245.0
    # 紙(≈245)より少し暗い裏写り・紙の地合いを白へ飛ばす。本文の文字はずっと暗いので残る
    out = (out - BLACK_LEVEL) * (255.0 / (WHITE_LEVEL - BLACK_LEVEL))
    return np.clip(out, 0, 255).astype(np.uint8)


def page_checks(page: np.ndarray, gutter_side: str | None) -> list[str]:
    """露出・端の文字切れの疑いを調べる（IMG-06）."""
    warnings = []
    g = _gray(page)
    mean = float(g.mean())
    clip = float((g >= 252).mean())
    if mean < 80:
        warnings.append("exposure_dark")
    if clip > 0.05:
        warnings.append("exposure_bright")
    small, _ = _scale_to(g, 800)
    m = text_mask(small) > 0
    total = m.mean()
    if total < 0.002:
        warnings.append("blank_page")
        return warnings
    # 外周は背景との境界が写るので、画面端での欠けは page_touches_frame で扱う。
    # ここでは背景を含まない綴じ目側だけを見て、分割位置が文字に掛かっていないか調べる。
    if gutter_side:
        h, w = m.shape
        b = max(2, int(0.012 * w))
        strip = m[int(h * 0.05): int(h * 0.95), :b] if gutter_side == "left" else m[int(h * 0.05): int(h * 0.95), -b:]
        if strip.mean() > 0.8 * total and strip.mean() > 0.02:
            warnings.append(f"edge_content_{gutter_side}")
    return warnings


FRAME_EDGE = 0.004     # 画面端に接しているとみなす距離（画像幅比）
CUT_STRIP = 0.03       # 画面端側で文字の有無を調べる帯の幅（ページ寸法比）


def touching_sides(shape: tuple[int, ...], bbox: list[int] | tuple[int, ...]) -> set[str]:
    """紙面範囲が画面の端に接している辺."""
    H, W = shape[:2]
    e = max(2, int(FRAME_EDGE * max(H, W)))
    x0, y0, x1, y1 = bbox
    out = set()
    if x0 <= e:
        out.add("left")
    if y0 <= e:
        out.add("top")
    if x1 >= W - e:
        out.add("right")
    if y1 >= H - e:
        out.add("bottom")
    return out


def text_near_edges(page: np.ndarray, sides: set[str], exclude: np.ndarray | None = None) -> list[str]:
    """指定した辺の近くに文字があるか（画面外へ紙面がはみ出し、文字が欠けている疑い）.

    exclude（白く塗った紙の外）の近くは紙の縁の線が出るので数えない。
    """
    if not sides:
        return []
    small, _ = _scale_to(_gray(page), 800)
    m = text_mask(small) > 0
    if exclude is not None:
        ex = cv2.resize(exclude.astype(np.uint8), (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
        r = max(3, int(0.015 * max(m.shape)))
        ex = cv2.dilate(ex, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        m &= ex == 0
    total = m.mean()
    if total < 0.002:
        return []
    h, w = m.shape
    by, bx = max(2, int(CUT_STRIP * h)), max(2, int(CUT_STRIP * w))
    strips = {"top": m[:by], "bottom": m[-by:], "left": m[:, :bx], "right": m[:, -bx:]}
    return [side for side in sorted(sides) if strips[side].mean() > max(0.5 * total, 0.01)]


PAPER_REF_TOP = 0.05   # 画面で最も明るいこの割合の画素の色を「紙の色」の基準にする
PAPER_CAST_MIN = 8.0   # 紙の色が無彩色からこれ以上離れているときだけ、色かぶりとして補正する


def paper_reference(lab: np.ndarray) -> tuple[float, float]:
    """紙の色（Lab の a, b）を推定する。画面の最も明るい部分（たいてい紙）の色の中央値.

    照明や白の合わせ方で画面全体が色かぶりすると、紙も色づく（SmartDoc の映像で、
    紙の彩度 27.7 が背景の 20.9 より高かった）。無彩色からの距離で測ると紙を見失うので、
    紙そのものの色からの距離で測る。白い紙が写った普通の映像では基準はほぼ無彩色になる。
    """
    L = lab[..., 0]
    cut = np.percentile(L, 100 * (1 - PAPER_REF_TOP))
    sel = L >= cut
    a, b = float(np.median(lab[..., 1][sel])), float(np.median(lab[..., 2][sel]))
    # 色かぶりがはっきりしないときは無彩色を基準にする。わずかなずれ（128 と 129 など）でも
    # 紙の範囲が数画素動き、OCRの結果が変わることを実写の回帰試験で確認したため
    if np.hypot(a - 128, b - 128) < PAPER_CAST_MIN:
        return 128.0, 128.0
    return a, b


def paper_score(img: np.ndarray) -> np.ndarray:
    """紙らしさ（明るく、紙の色に近いほど高い）。明るい木目の机や手は色みの違いで下がる."""
    if img.ndim == 2:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    a0, b0 = paper_reference(lab)
    chroma = np.hypot(lab[..., 1] - a0, lab[..., 2] - b0)
    return np.clip(lab[..., 0] - 10.0 * chroma, 0, 255).astype(np.uint8)


def paper_threshold(score: np.ndarray) -> float:
    """紙とそれ以外を分ける閾値。照明で白っぽく見える机を紙に含めないよう、
    Otsu より厳しく「紙の典型値（90パーセンタイル）から35下」以上を紙とする."""
    otsu, _ = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(max(otsu, np.percentile(score, 90) - 35))


def whiten_outside_paper(page: np.ndarray, gutter_side: str | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """外周から入り込んだ紙以外（机・手）を白にする。(結果, 白にした範囲のマスク) を返す.

    紙以外の領域のうち画像の外周に接するものだけを対象にするので、
    ページの中の図や写真は残る。
    """
    small, _ = _scale_to(page, 600)
    sc = cv2.GaussianBlur(paper_score(small), (5, 5), 0)
    paper = (sc > paper_threshold(sc)).astype(np.uint8) * 255
    k = max(3, small.shape[1] // 50) | 1
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    non = (paper == 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(non)
    edges = {"top": labels[0], "bottom": labels[-1], "left": labels[:, 0], "right": labels[:, -1]}
    border = np.unique(np.concatenate([v for k, v in edges.items() if k != gutter_side]))
    if gutter_side:
        # 綴じ目側から続く暗い部分は影と文字なので消さない
        border = np.setdiff1d(border, np.unique(edges[gutter_side]))
    border = border[border > 0]
    outside = np.isin(labels, border).astype(np.uint8)
    frac = float(outside.mean())
    if frac == 0 or frac > 0.5:  # 紙と背景を分けられていない
        return page, None
    outside = cv2.dilate(outside, np.ones((3, 3), np.uint8))
    mask = cv2.resize(outside, (page.shape[1], page.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    out = page.copy()
    out[mask] = 255
    return out, mask


CONTENT_MARGIN = 0.02  # 内容の範囲の外側に残す余白（ページ寸法比）


def content_boxes(page: np.ndarray, outside: np.ndarray | None = None
                  ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    """(内容の範囲, 本文の中心部) を返す.

    内容の範囲は本文・図・ページ番号などすべてを含む。画像の端に接するもの（影・紙の縁）と、
    紙の外（机・手）にかかるもの（手のしわ・木目）は除く。
    本文の中心部は外れ値（余白のページ番号など）を除いた範囲。
    """
    small, s = _scale_to(_gray(page), 1000)
    m = text_mask(small)
    h, w = m.shape
    out_small = None
    if outside is not None:
        r = max(3, int(0.01 * max(h, w)))
        out_small = cv2.resize(outside.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        out_small = cv2.dilate(out_small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    e = max(2, int(0.01 * max(h, w)))
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 6:
            continue  # 点状のノイズ
        if x <= e or y <= e or x + bw >= w - e or y + bh >= h - e:
            continue
        if bw > 0.9 * w or bh > 0.9 * h:
            continue
        if out_small is not None and out_small[y:y + bh, x:x + bw].mean() > 0.2:
            continue  # 手のしわ・木目など
        boxes.append((x, y, x + bw, y + bh, area))
    if len(boxes) < 3:
        return None
    b = np.array(boxes, np.float64)
    inv = 1.0 / s
    full = (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())
    # 面積で重み付けした分位点で、少数の外れた要素を除く
    def q(vals: np.ndarray, frac: float) -> float:
        order = np.argsort(vals)
        cw = np.cumsum(b[order, 4]) / b[:, 4].sum()
        return float(vals[order][min(len(vals) - 1, np.searchsorted(cw, frac))])
    body = (q(b[:, 0], 0.03), q(b[:, 1], 0.03), q(b[:, 2], 0.97), q(b[:, 3], 0.97))
    to_px = lambda t: (int(t[0] * inv), int(t[1] * inv), int(np.ceil(t[2] * inv)), int(np.ceil(t[3] * inv)))
    return to_px(full), to_px(body)


def clean_margins(page: np.ndarray, outside: np.ndarray | None) -> tuple[np.ndarray, list[int] | None, bool]:
    """余白を白に統一する（ページは長方形のまま）.

    - 内容の範囲の外は白にする
    - 範囲の中でも、本文の中心部にかからない手・机は白にする
    - 本文の中心部にかかる手は消さずに残し、確認へ回す（消すと隠れた文字の場所が白紙に見える）
    """
    boxes = content_boxes(page, outside)
    if boxes is None:
        return page, None, False
    (fx0, fy0, fx1, fy1), (bx0, by0, bx1, by1) = boxes
    H, W = page.shape[:2]
    mx, my = int(CONTENT_MARGIN * W), int(CONTENT_MARGIN * H)
    x0, y0 = max(0, fx0 - mx), max(0, fy0 - my)
    x1, y1 = min(W, fx1 + mx), min(H, fy1 + my)
    out = np.full_like(page, 255)
    out[y0:y1, x0:x1] = page[y0:y1, x0:x1]
    hand = False
    if outside is not None and outside.any():
        body = np.zeros(outside.shape, bool)
        body[by0:by1, bx0:bx1] = True
        hand_in_body = outside & body
        hand = bool(hand_in_body.mean() > 0.0005 and hand_in_body[by0:by1, bx0:bx1].mean() > 0.002)
        out[outside & ~body] = 255
    return out, [x0, y0, x1, y1], hand


def refine_outer(crop: np.ndarray, gutter_side: str | None) -> tuple[np.ndarray, list[int]]:
    """ページ単位で紙の範囲を探し、綴じ目側以外の外側（机・手）を切り詰める."""
    (x0, y0, x1, y1), warn = detect_page_region(crop)
    H, W = crop.shape[:2]
    if "page_region_uncertain" in warn or (x1 - x0) < 0.5 * W or (y1 - y0) < 0.5 * H:
        return crop, [0, 0, W, H]
    if gutter_side == "left":
        x0 = 0
    elif gutter_side == "right":
        x1 = W
    return crop[y0:y1, x0:x1], [int(x0), int(y0), int(x1), int(y1)]


def split_spread(
    img: np.ndarray,
    layout: str = "spread",
    direction: str = "ltr",
    enhance: str = "normalize",
    geometry: dict[str, Any] | None = None,
    dewarp_mode: str = "auto",
    margins: str = "content_box",
) -> tuple[dict[str, Any], list[tuple[str, np.ndarray, list[str], float]]]:
    """見開き画像からページ画像を作る.

    geometry を与えると（手修正）それを使い、自動検出しない。
    戻り値: (使用した geometry, [(side, page_img, warnings, skew_deg), ...]) 。
    行間隔の推定は呼び出し側で estimate_line_pitch を使う。
    リストは読み順（PDFの順）に並ぶ。
    """
    geo = dict(geometry or {})
    manual = "bbox" in geo
    geo_warn: list[str] = []
    if not manual:
        bbox, w = detect_page_region(img)
        geo["bbox"] = list(bbox)
        # 画面端への接触は文字の有無を見てページごとに判定する
        geo_warn += [x for x in w if x != "page_touches_frame"]
    x0, y0, x1, y1 = geo["bbox"]
    touch = touching_sides(img.shape, geo["bbox"])
    if layout == "spread":
        if "gutter_x" not in geo:
            gx, conf, w = detect_gutter(img, (x0, y0, x1, y1))
            geo["gutter_x"] = gx
            geo["gutter_conf"] = conf
            geo_warn += w
        gx = int(geo["gutter_x"])
        parts = [("L", img[y0:y1, x0:gx], "right", touch - {"right"}),
                 ("R", img[y0:y1, gx:x1], "left", touch - {"left"})]
        if direction == "rtl":
            parts.reverse()
    else:
        parts = [("S", img[y0:y1, x0:x1], None, touch)]
    geo["warnings"] = geo_warn
    geo["touching"] = sorted(touch)

    pages = []
    for side, crop, gutter_side, sides in parts:
        warnings = list(geo_warn)
        if crop.size == 0 or crop.shape[1] < 20 or crop.shape[0] < 20:
            pages.append((side, crop, warnings + ["empty_crop"], 0.0))
            continue
        if not manual:
            full_h, full_w = crop.shape[:2]
            crop, inner = refine_outer(crop, gutter_side)
            geo.setdefault("page_crops", {})[side] = inner
            # ページ単位で切り詰めた辺は、もう画面の端には接していない
            e = max(2, int(FRAME_EDGE * max(img.shape[:2])))
            trimmed = {"left": inner[0] > e, "top": inner[1] > e,
                       "right": inner[2] < full_w - e, "bottom": inner[3] < full_h - e}
            sides = {x for x in sides if not trimmed[x]}
        skew = float(geo.get("skew", {}).get(side, estimate_skew(_gray(crop))))
        page = rotate_keep(crop, skew)
        if dewarp_mode == "auto":
            from vbs.dewarp import dewarp

            page, model = dewarp(page)
            geo.setdefault("dewarp", {})[side] = {k: v for k, v in model.items() if k != "coef"}
        # 紙の外（机・端から入る手）の範囲は、端の文字切れの検査にだけ使う
        papered, outside = whiten_outside_paper(page, gutter_side)
        if text_near_edges(papered, sides, outside):
            warnings.append("page_cut")
        elif sides:
            warnings.append("page_touches_frame")
        if enhance == "normalize":
            page = normalize_illumination(page)
        if margins == "content_box":
            page, box, hand = clean_margins(page, outside)
            geo.setdefault("content_box", {})[side] = box
            if hand:
                warnings.append("hand_in_content")
        warnings += page_checks(crop, gutter_side)
        pages.append((side, page, warnings, skew))
    return geo, pages
