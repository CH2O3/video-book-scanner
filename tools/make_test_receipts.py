"""検証用の合成レシートを作る（写真、または1枚ずつ置き換える動画）.

店名・日付・金額などは乱数で作った架空のもの。正解を <出力>.truth.json に書く。

  python tools/make_test_receipts.py work/r --n 4            # 写真 r/receipt_01.jpg ...
  python tools/make_test_receipts.py work/r.mp4 --n 4 --video
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_test_video import load_font  # noqa: E402

STORES = ["みどり商店 駅前店", "株式会社サンプル文具", "あおば書店 本店", "ひかりドラッグ 中央店", "カフェ ことり"]
ITEMS = [("ノート", 220), ("ボールペン", 165), ("コーヒー", 480), ("サンドイッチ", 398), ("付箋", 330),
         ("クリアファイル", 110), ("お茶", 151), ("電池", 528), ("封筒", 275), ("消しゴム", 99)]
W = 576  # 80mm 幅の感熱紙を約183dpiで描く（縮小して置くので、撮影後の解像度はこれより下がる）


def render(rng: random.Random, n: int) -> tuple[Image.Image, dict]:
    store = rng.choice(STORES)
    day = date(2024, 1, 1) + timedelta(days=rng.randrange(0, 600))
    body = [rng.randrange(10) for _ in range(12)]
    check = 9 - sum(d * (1 if i % 2 == 0 else 2) for i, d in enumerate(body[::-1])) % 9
    reg = "T" + str(check) + "".join(map(str, body))  # 法人番号と同じ検査用数字を付けた架空の番号
    items = rng.sample(ITEMS, rng.randint(2, 4))
    lines: list[tuple[str, str, int]] = []  # (左, 右, 大きさ)
    lines += [(store, "", 34), ("東京都千代田区架空町1-2-3", "", 22), ("TEL 03-0000-0000", "", 22),
              (f"登録番号 {reg}", "", 22), ("", "", 14), ("領 収 書", "", 30),
              (f"{day.year}年{day.month:02d}月{day.day:02d}日 {rng.randint(9, 20)}:{rng.randint(0, 59):02d}", "", 24),
              (f"レジ{rng.randint(1, 4)} 責No.{rng.randint(100, 999)}", "", 22), ("", "", 14)]
    base10 = base8 = 0
    for name, price in items:
        light = name in ("コーヒー", "サンドイッチ", "お茶")  # 飲食料品は軽減税率
        if light:
            base8 += price
        else:
            base10 += price
        lines.append((name + ("※" if light else ""), f"¥{price:,}", 24))
    tax10 = base10 - round(base10 / 1.1)
    tax8 = base8 - round(base8 / 1.08)
    total = base10 + base8
    paid = (total // 1000 + 1) * 1000
    lines += [("", "", 14), ("小計", f"¥{total:,}", 24), ("合計", f"¥{total:,}", 34),
              ("(10%対象", f"¥{base10:,})", 22), ("(8%対象", f"¥{base8:,})", 22),
              ("(内消費税等", f"¥{tax10 + tax8:,})", 22), ("お預り", f"¥{paid:,}", 24),
              ("お釣り", f"¥{paid - total:,}", 24), ("", "", 14), ("※印は軽減税率対象商品です", "", 20),
              ("ありがとうございました", "", 22)]
    H = 60 + sum(int(s * 1.55) for _, _, s in lines)
    img = Image.new("RGB", (W, H), (250, 250, 247))
    d = ImageDraw.Draw(img)
    y = 30
    for left, right, size in lines:
        f = load_font(size)
        if left:
            d.text((30, y), left, font=f, fill=(25, 25, 30))
        if right:
            tw = d.textlength(right, font=f)
            d.text((W - 30 - tw, y), right, font=f, fill=(25, 25, 30))
        y += int(size * 1.55)
    truth = {"n": n, "date": day.isoformat(), "payee": store, "total": total, "base10": base10 or None,
             "base8": base8 or None, "tax": tax10 + tax8, "registration_no": reg}
    return img, truth


def place(receipt: Image.Image, rng: random.Random, size=(1920, 1080), scale=0.95, angle=None) -> np.ndarray:
    """暗い机の上に、少し傾けて置く."""
    bg = np.zeros((size[1], size[0], 3), np.uint8)
    bg[:] = (52, 44, 40)
    noise = np.random.default_rng(rng.randrange(1 << 30)).integers(0, 10, bg.shape, dtype=np.uint8)
    bg = (bg + noise).astype(np.uint8)
    s = min(scale * size[1] / receipt.height, 0.7 * size[0] / receipt.width)
    r = receipt.resize((int(receipt.width * s), int(receipt.height * s)), Image.LANCZOS)
    r = r.rotate(rng.uniform(-3, 3) if angle is None else angle, expand=True, fillcolor=(52, 44, 40))
    canvas = Image.fromarray(bg)
    canvas.paste(r, ((size[0] - r.width) // 2 + rng.randint(-60, 60), (size[1] - r.height) // 2))
    return np.asarray(canvas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--video", action="store_true", help="1枚ずつ置き換える動画にする")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--size", default="1440x2560", help="写真の大きさ（幅x高さ）")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    receipts = [render(rng, i + 1) for i in range(a.n)]
    truth = [t for _, t in receipts]
    if a.video:
        out.parent.mkdir(parents=True, exist_ok=True)
        fps = 30
        with av.open(str(out), "w") as c:
            st = c.add_stream("libx264", rate=fps)
            st.width, st.height, st.pix_fmt = 1080, 1920, "yuv420p"  # 縦持ち
            st.options = {"crf": "18"}
            empty = place(Image.new("RGB", (10, 10), (52, 44, 40)), rng, size=(1080, 1920), scale=0.01)
            frames = []
            for img, _ in receipts:
                still = place(img, rng, size=(1080, 1920), scale=0.92)
                frames += [empty] * int(0.6 * fps)            # 置き換えの間（何もない）
                frames += [still] * int(1.6 * fps)            # 置いて止める
            frames += [empty] * fps
            for fr in frames:
                for pkt in st.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                    c.mux(pkt)
            for pkt in st.encode():
                c.mux(pkt)
        tpath = out.with_suffix(".truth.json")
    else:
        out.mkdir(parents=True, exist_ok=True)
        w, h = (int(x) for x in a.size.split("x"))
        for i, (img, _) in enumerate(receipts):
            Image.fromarray(place(img, rng, size=(w, h), scale=0.9)).save(out / f"receipt_{i + 1:02d}.jpg", quality=93)
        tpath = out / "truth.json"
    tpath.write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    print(tpath)


if __name__ == "__main__":
    main()
