"""検証用の合成ページめくり動画を作る.

実写の代わりにはならないが、時刻対応・区間検出・分割・OCR・PDFまでの
配線を自動テストするために使う。正解（区間の時刻とページの語）をJSONで出す。

使い方: python tools/make_test_video.py out.mp4 [--spreads 5] [--vfr] [--bump]
"""

from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
PAGE_W, PAGE_H = 720, 980
FONT_PATHS = ["C:/Windows/Fonts/msgothic.ttc", "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/YuGothM.ttc"]

SENTENCES = [
    "光合成は植物が光のエネルギーを使って二酸化炭素と水から糖をつくる反応である。",
    "細胞膜は脂質の二重層からなり、物質の出入りを選択的に調節している。",
    "酵素は特定の基質にだけ作用し、反応の速さを大きく変える。",
    "地層の重なり方を調べると、その地域の過去の環境を推定できる。",
    "電流の大きさは電圧に比例し、抵抗に反比例する。これをオームの法則という。",
    "江戸時代の農村では、村ごとに年貢をまとめて納める仕組みがとられた。",
    "需要が増えて供給が変わらなければ、一般に価格は上昇する。",
    "三角形の内角の和は一八〇度であり、この性質は多くの証明で使われる。",
    "水溶液の性質は、溶けている物質の種類と濃度によって決まる。",
    "気温が下がると空気中の水蒸気が凝結し、雲や霧が生じる。",
]
SUBJECTS = ["植物の細胞", "地層の重なり", "電流の大きさ", "江戸時代の村", "市場の価格", "三角形の内角", "水溶液の濃度",
            "空気中の水蒸気", "酵素の働き", "気圧の配置", "物体の運動", "国際的な条約", "人口の分布", "光の屈折"]
OBJECTS = ["環境の変化", "温度と圧力", "需要と供給", "生物の多様性", "社会の仕組み", "資源の配分", "地域の特色",
           "実験の結果", "観察の記録", "歴史的な背景", "法則の適用", "数量の関係"]
VERBS = ["によって大きく変わる。", "と深く関係している。", "を調べると理解が深まる。", "から推定することができる。",
         "について多くの研究がある。", "を比べると違いがわかる。", "の影響を受けやすい。", "として説明されることが多い。"]


def make_sentence(rng: random.Random) -> str:
    return f"{rng.choice(SUBJECTS)}は{rng.choice(OBJECTS)}{rng.choice(VERBS)}"


KEYWORDS = ["葉緑体", "浸透圧", "活性部位", "堆積岩", "抵抗値", "検地帳", "均衡価格", "外角定理",
            "中和反応", "露点温度", "相同染色体", "等圧線", "比熱容量", "条約改正", "需要曲線",
            "素因数", "電磁誘導", "光屈性", "化学平衡", "季節風"]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    raise SystemExit("日本語フォントが見つかりません")


PAGE_TEXT: dict[int, str] = {}


def render_page(num: int, keyword: str, rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (PAGE_W, PAGE_H), (246, 243, 235))
    d = ImageDraw.Draw(img)
    f_head = load_font(30)
    f_body = load_font(21)
    f_foot = load_font(18)
    head = f"第{(num + 1) // 2}章　理科と社会の基礎"
    d.text((60, 50), head, font=f_head, fill=(20, 20, 20))
    # 実際の本のように段落の改行・字下げ・小見出しを入れ、ページごとに行の位置をずらす。
    # （文字が全ページ同じ格子に並ぶと、別のページでも画像の特徴が一致してしまう）
    y = 118 + rng.randint(0, 12)
    x0 = 56 + rng.randint(0, 8)
    chars = 28
    body = []
    first = True
    while y < PAGE_H - 90:
        if not first and rng.random() < 0.15 and y < PAGE_H - 160:
            heading = f"({rng.randint(1, 9)}) {rng.choice(SUBJECTS)}"
            y += 10
            d.text((x0, y), heading, font=f_body, fill=(15, 15, 15))
            body.append(heading)
            y += 36
        para = "".join(make_sentence(rng) for _ in range(rng.randint(2, 6)))
        if first:
            para = f"重要語は{keyword}である。" + para
            first = False
        para = "　" + para
        for k in range(0, len(para), chars):
            if y >= PAGE_H - 90:
                break
            line = para[k:k + chars]
            d.text((x0, y), line, font=f_body, fill=(25, 25, 25))
            body.append(line.lstrip("　"))
            y += 26
    d.text((PAGE_W // 2 - 10, PAGE_H - 50), str(num), font=f_foot, fill=(40, 40, 40))
    PAGE_TEXT[num] = "\n".join([head] + body + [str(num)])
    return img


def make_spread(left: Image.Image, right: Image.Image) -> np.ndarray:
    sp = np.concatenate([np.asarray(left, np.float32), np.asarray(right, np.float32)], axis=1)
    # 綴じ目の影
    x = np.arange(sp.shape[1], dtype=np.float32)
    c = PAGE_W
    shade = 1.0 - 0.35 * np.exp(-((x - c) / 25.0) ** 2)
    sp *= shade[None, :, None]
    return sp


def background(rng: np.random.Generator) -> np.ndarray:
    bg = np.zeros((H, W, 3), np.float32)
    bg[:] = (55, 45, 38)
    grain = rng.normal(0, 6, (H // 8, W // 8)).astype(np.float32)
    grain = np.kron(grain, np.ones((8, 8), np.float32))
    bg += grain[..., None]
    return bg


def compose(bg: np.ndarray, spread: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    out = bg.copy()
    sh, sw = spread.shape[:2]
    x0, y0 = (W - sw) // 2 + dx, (H - sh) // 2 + dy
    out[y0:y0 + sh, x0:x0 + sw] = spread
    return out


def turning_frame(bg, prev_spread, next_spread, t: float) -> np.ndarray:
    """右ページが綴じ目を軸に左へめくられる様子を近似する（t=0..1）."""
    base = compose(bg, np.concatenate([prev_spread[:, :PAGE_W], next_spread[:, PAGE_W:]], axis=1))
    sh = PAGE_H
    x_g = (W - 2 * PAGE_W) // 2 + PAGE_W
    y0 = (H - sh) // 2
    # めくれているページの見かけの幅
    wv = int(abs(np.cos(np.pi * t)) * PAGE_W)
    if wv > 2:
        src = prev_spread[:, PAGE_W:] if t < 0.5 else next_spread[:, :PAGE_W]
        img = np.asarray(Image.fromarray(src.astype(np.uint8)).resize((wv, sh)), np.float32)
        img *= 0.85 + 0.15 * abs(np.cos(np.pi * t))
        if t < 0.5:
            base[y0:y0 + sh, x_g:x_g + wv] = img
        else:
            base[y0:y0 + sh, x_g - wv:x_g] = img
    # 手
    hx = int(x_g + np.cos(np.pi * t) * PAGE_W * 0.9)
    hy = y0 + int(sh * 0.75)
    yy, xx = np.ogrid[:H, :W]
    mask = ((xx - hx) / 110.0) ** 2 + ((yy - hy) / 170.0) ** 2 <= 1
    base[mask] = (190, 140, 110)
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--spreads", type=int, default=5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hold", type=float, default=1.5)
    ap.add_argument("--turn", type=float, default=0.7)
    ap.add_argument("--vfr", action="store_true", help="可変フレームレート")
    ap.add_argument("--bump", action="store_true", help="2番目の見開きで手が当たって少しずれる")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    bg = background(nrng)
    pages = [render_page(i + 1, KEYWORDS[i % len(KEYWORDS)], rng) for i in range(args.spreads * 2)]
    spreads = [make_spread(pages[2 * i], pages[2 * i + 1]) for i in range(args.spreads)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tb = Fraction(1, 90000)
    container = av.open(str(out), "w")
    stream = container.add_stream("libx264", rate=args.fps)
    stream.width, stream.height = W, H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "veryfast"}
    stream.codec_context.time_base = tb
    stream.time_base = tb

    t = 0.0
    truth = {"spreads": [], "pages": [], "fps": args.fps, "vfr": args.vfr}

    def emit(img: np.ndarray) -> None:
        nonlocal t
        noisy = img + nrng.normal(0, 2.0, img.shape).astype(np.float32)
        frame = av.VideoFrame.from_ndarray(np.clip(noisy, 0, 255).astype(np.uint8), format="rgb24")
        frame.pts = int(round(t / tb))
        frame.time_base = tb
        for pkt in stream.encode(frame):
            container.mux(pkt)
        dt = 1.0 / args.fps
        if args.vfr:
            dt = rng.choice([1 / 30, 1 / 24, 1 / 60, 1 / 30])
        t += dt

    for i, sp in enumerate(spreads):
        start = t
        n_hold = int(args.hold * args.fps)
        bump = args.bump and i == 1
        for k in range(n_hold):
            emit(compose(bg, sp))
        if bump:
            # 手が当たって本が少しずれ、また静止する
            for k in range(6):
                emit(compose(bg, sp, dx=3 * k, dy=k))
            for k in range(n_hold):
                emit(compose(bg, sp, dx=15, dy=5))
        truth["spreads"].append({"index": i, "start": start, "end": t, "pages": [2 * i + 1, 2 * i + 2],
                                 "bump": bump})
        if i + 1 < len(spreads):
            n_turn = int(args.turn * args.fps)
            for k in range(n_turn):
                emit(turning_frame(bg, sp, spreads[i + 1], (k + 1) / (n_turn + 1)))
    for i in range(len(pages)):
        truth["pages"].append({"number": i + 1, "keyword": KEYWORDS[i % len(KEYWORDS)],
                               "text": PAGE_TEXT[i + 1]})
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    out.with_suffix(".truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} ({t:.1f}s)")


if __name__ == "__main__":
    main()
