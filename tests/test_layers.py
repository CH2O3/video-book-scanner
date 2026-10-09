from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vbs.dewarp import detect_lines, dewarp
from vbs.textlayer import build_content


def _tsv(path: Path, words: list[tuple[str, int, int, int, int, int]]) -> None:
    head = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    rows = ["1\t1\t0\t0\t0\t0\t0\t0\t1000\t1400\t-1\t"]
    for i, (t, line, x, y, w, h) in enumerate(words):
        rows.append(f"5\t1\t1\t1\t{line}\t{i}\t{x}\t{y}\t{w}\t{h}\t90\t{t}")
    path.write_text(head + "\n".join(rows) + "\n", encoding="utf-8")


def test_text_layer_has_no_spaces_between_japanese_words(tmp_path: Path):
    p = tmp_path / "a.tsv"
    _tsv(p, [("反則", 1, 100, 100, 60, 30), ("の", 1, 165, 100, 30, 30), ("違反", 1, 200, 100, 60, 30),
             ("VAR", 2, 100, 200, 60, 30), ("check", 2, 170, 200, 90, 30), ("判定", 2, 270, 200, 60, 30)])
    data = build_content(p, 500.0, 700.0, "f-0-0").decode()
    hexes = [part.split(">")[0] for part in data.split("<")[1:]]
    texts = [bytes.fromhex(h).decode("utf-16-be") for h in hexes]
    assert texts == ["反則", "の", "違反", "VAR ", "check", "判定"]  # 英字どうしの間だけ空白
    assert "3 Tr" in data  # 不可視


def test_dewarp_straightens_curved_lines():
    h, w = 1400, 1000
    img = np.full((h, w), 245, np.uint8)
    for k in range(20):
        y0 = 150 + k * 55
        for x in range(80, 920, 18):
            bend = 40 * ((x - 80) / 840) ** 2  # 綴じ目側（右）へ反る
            y = int(y0 - bend)
            cv2.rectangle(img, (x, y), (x + 12, y + 14), 30, -1)
    out, model = dewarp(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    assert model["applied"], model
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    before = np.mean([np.ptp(l[:, 1]) for l in detect_lines(img)])
    after = np.mean([np.ptp(l[:, 1]) for l in detect_lines(g)])
    assert after < before * 0.35, (before, after)


def test_dewarp_leaves_flat_page_alone():
    img = np.full((1400, 1000, 3), 245, np.uint8)
    for k in range(20):
        for x in range(80, 920, 18):
            cv2.rectangle(img, (x, 150 + k * 55), (x + 12, 164 + k * 55), (30, 30, 30), -1)
    out, model = dewarp(img)
    assert not model["applied"] and np.array_equal(out, img)


def test_unreliable_page_keeps_only_confident_words(tmp_path: Path):
    from vbs.textlayer import line_texts, page_min_conf

    p = tmp_path / "blank.tsv"
    head = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    rows = ["1\t1\t0\t0\t0\t0\t0\t0\t1000\t1400\t-1\t"]
    for i, (t, c) in enumerate([("ささ", 81), ("いき", 5), ("男", 0), ("区", 21), ("ルン", 12)]):
        rows.append(f"5\t1\t1\t1\t1\t{i}\t{100 + i * 40}\t100\t30\t30\t{c}\t{t}")
    p.write_text(head + "\n".join(rows) + "\n", encoding="utf-8")
    th = page_min_conf(p)
    assert th >= 60 and line_texts(p, th) == ["ささ"]
    # 通常のページ（平均が高い）では語ごとの足切りをしない
    q = tmp_path / "ok.tsv"
    rows = ["1\t1\t0\t0\t0\t0\t0\t0\t1000\t1400\t-1\t"]
    for i, (t, c) in enumerate([("経済法", 25), ("独禁法", 92), ("違反要件", 90)]):
        rows.append(f"5\t1\t1\t1\t1\t{i}\t{100 + i * 90}\t100\t80\t30\t{c}\t{t}")
    q.write_text(head + "\n".join(rows) + "\n", encoding="utf-8")
    assert page_min_conf(q) == 0 and line_texts(q, 0) == ["経済法独禁法違反要件"]
