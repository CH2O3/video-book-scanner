from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from pypdf import PdfReader

from vbs.extract import Analysis, _features, _inliers, auto_threshold, find_still_runs, pick_candidates
from vbs.manifest import Project, ProjectError
from vbs.pdf import build_pdf, compose_page, jpeg_page_pdf
from vbs.split import detect_gutter, detect_page_region, estimate_line_pitch, split_spread


def _analysis(motion: list[float], fps: float = 30.0) -> Analysis:
    n = len(motion)
    t = np.arange(n) / fps
    return Analysis(times=t, pts=np.arange(n), motion=np.asarray(motion, np.float32),
                    sharp=np.ones(n, np.float32), bright=np.full(n, 128, np.float32),
                    sigs=np.zeros((n, 18, 32), np.uint8), threshold=1.0)


def test_still_runs_basic_and_edges():
    # 静止1.0秒 → 動き0.5秒 → 静止0.9秒（末尾で録画終了）
    m = [0.2] * 30 + [10.0] * 15 + [0.2] * 27
    runs = find_still_runs(_analysis(m), min_still_sec=0.5)
    assert len(runs) == 2
    assert runs[0]["start"] == 0.0  # 冒頭も候補に含める（EXT-03）
    assert runs[1]["i1"] == len(m) - 1  # 末尾も含める
    assert not runs[0]["short"] and not runs[1]["short"]


def test_short_still_is_kept_but_flagged():
    m = [10.0] * 10 + [0.2] * 10 + [10.0] * 10  # 約0.3秒の静止
    runs = find_still_runs(_analysis(m), min_still_sec=0.5)
    assert len(runs) == 1 and runs[0]["short"]


def test_single_frame_glitch_is_merged():
    m = [0.2] * 20 + [5.0] + [0.2] * 20
    runs = find_still_runs(_analysis(m), min_still_sec=0.5)
    assert len(runs) == 1


def test_seconds_based_independent_of_fps():
    for fps in (24.0, 30.0, 60.0):
        n = int(fps)
        m = [0.2] * n + [10.0] * n + [0.2] * n
        assert len(find_still_runs(_analysis(m, fps), 0.5)) == 2


def test_candidates_spaced_and_sharpest_first():
    a = _analysis([0.2] * 30)
    a.sharp = np.linspace(1, 2, 30).astype(np.float32)
    ks = pick_candidates(a, 0, 29, 3)
    assert ks[0] == 29
    assert all(abs(a.times[i] - a.times[j]) >= 0.15 for i in ks for j in ks if i != j)


def test_auto_threshold_bounds():
    assert 1.0 <= auto_threshold(np.full(100, 0.05, np.float32)) <= 5.0
    assert auto_threshold(np.full(100, 10.0, np.float32)) == 5.0


def _spread(gutter_x: int = 520, w: int = 1000, h: int = 700, seed: int | None = None) -> np.ndarray:
    img = np.full((h + 200, w + 300, 3), (40, 50, 60), np.uint8)
    x0, y0 = 150, 100
    rng = np.random.default_rng(seed)
    img[y0:y0 + h, x0:x0 + w] = (240, 240, 235)
    for y in range(y0 + 60, y0 + h - 60, 22):  # 行
        for side in ((x0 + 50, x0 + gutter_x - 50), (x0 + gutter_x + 50, x0 + w - 50)):
            for x in range(side[0], side[1], 16):
                if seed is not None and rng.random() < 0.3:
                    continue  # 文字ごとに形を変えて「別の本文」を作る
                cv2.rectangle(img, (x, y), (x + 11, y + 12), (30, 30, 30), -1)
                if seed is not None:
                    cv2.line(img, (x + int(rng.integers(0, 11)), y), (x + int(rng.integers(0, 11)), y + 12), (240, 240, 235), 2)
    xs = np.arange(img.shape[1])
    shade = 1 - 0.3 * np.exp(-((xs - (x0 + gutter_x)) / 12.0) ** 2)
    img[y0:y0 + h] = (img[y0:y0 + h] * shade[None, :, None]).astype(np.uint8)
    return img


def test_page_region_and_gutter_not_fixed_to_center():
    img = _spread(gutter_x=560)
    bbox, warn = detect_page_region(img)
    x0, y0, x1, y1 = bbox
    assert abs(x0 - 150) < 20 and abs(x1 - 1150) < 20 and abs(y0 - 100) < 20
    gx, conf, _ = detect_gutter(img, bbox)
    assert abs(gx - (150 + 560)) < 12  # 中央(650)ではなく実際の綴じ目
    assert conf >= 0.5


def test_split_order_ltr_rtl_and_manual_geometry():
    img = _spread()
    geo, pages = split_spread(img, "spread", "ltr")
    assert [p[0] for p in pages] == ["L", "R"]
    _, pages = split_spread(img, "spread", "rtl")
    assert [p[0] for p in pages] == ["R", "L"]
    manual = {"bbox": [150, 100, 1150, 800], "gutter_x": 700}
    geo2, pages = split_spread(img, "spread", "ltr", geometry=manual)
    assert geo2["gutter_x"] == 700
    _, pages = split_spread(img, "single", "ltr")
    assert [p[0] for p in pages] == ["S"]


def test_line_pitch():
    img = _spread()
    g = cv2.cvtColor(img[100:800, 200:650], cv2.COLOR_BGR2GRAY)
    assert estimate_line_pitch(g) == pytest.approx(22, abs=2)


def test_duplicate_matching_tolerates_shift_and_occlusion():
    """同じ見開きは位置ずれ・手の映り込みがあっても対応点が多く、別の見開きは少ない."""
    a = _spread(seed=0)
    b = np.roll(a, (12, 25), axis=(0, 1)).copy()
    cv2.ellipse(b, (200, 700), (90, 160), 0, 0, 360, (110, 140, 190), -1)  # 手
    c = _spread(seed=1)
    fa, fb, fc = (_features(cv2.cvtColor(i, cv2.COLOR_BGR2RGB)) for i in (a, b, c))
    same, _ = _inliers(fa, fb)
    diff, _ = _inliers(fa, fc)
    assert same >= 120 and same > 3 * diff


def test_paper_vs_light_wooden_table():
    """明るい木目の机（彩度あり）を紙に含めない."""
    from vbs.split import detect_page_region
    img = np.zeros((900, 1600, 3), np.uint8)
    img[:] = (150, 185, 215)            # 明るい木目（BGR, 黄褐色）
    img[150:850, 300:1300] = (228, 230, 232)  # 紙
    (x0, y0, x1, y1), _ = detect_page_region(img)
    assert abs(x0 - 300) < 20 and abs(x1 - 1300) < 20 and abs(y0 - 150) < 20


def test_orientation_vote_logic(monkeypatch):
    import vbs.orient as O
    answers = iter([(1, 10.0), (1, 4.0), (3, 0.5), None, (2, 2.0)])
    monkeypatch.setattr(O, "detect_osd", lambda img: next(answers))
    res = O.vote([None] * 5)
    assert res["k"] == 1 and res["decided"]


def test_manifest_roundtrip_and_lock(tmp_path: Path):
    p = Project.create(tmp_path / "proj", {"direction": "rtl"})
    assert p.settings["direction"] == "rtl" and p.data["schema_version"] == 1
    assert p.new_id("segment") == "s0001" and p.new_id("segment") == "s0002"
    p.save()
    q = Project.load(tmp_path / "proj")
    assert q.data["next_ids"]["segment"] == 3
    with q.lock():
        with pytest.raises(ProjectError):
            with Project.load(tmp_path / "proj").lock():
                pass
    with pytest.raises(ProjectError):
        Project.create(tmp_path / "proj")


def test_unsupported_schema_is_refused(tmp_path: Path):
    p = Project.create(tmp_path / "proj")
    p.data["schema_version"] = 999
    (tmp_path / "proj" / "project.json").write_text(json.dumps(p.data), encoding="utf-8")
    with pytest.raises(ProjectError):
        Project.load(tmp_path / "proj")


def test_jpeg_passthrough_pdf(tmp_path: Path):
    img = np.full((300, 200, 3), 200, np.uint8)
    path = tmp_path / "日本語 ページ.jpg"
    ok, enc = cv2.imencode(".jpg", img)
    path.write_bytes(enc.tobytes())
    data = jpeg_page_pdf(path, dpi=100)
    assert enc.tobytes() in data  # 再圧縮していない
    page = compose_page(path, 100, None)
    assert float(page.mediabox.width) == pytest.approx(144.0)
    out = tmp_path / "out.pdf"
    n = build_pdf([("a", None, path, 100), ("b", None, path, 100)], out, title="t")
    assert n == 2 and len(PdfReader(str(out)).pages) == 2


def test_auto_threshold_ignores_long_dead_still():
    fps = 30
    hand = np.full(60 * fps, 2.3, np.float32)       # 手持ちの見開き（静止中でもこの程度動く）
    dead = np.full(300 * fps, 0.05, np.float32)     # 手を離して5分録画
    m = np.concatenate([hand, dead, hand])
    t = np.arange(len(m)) / fps
    assert auto_threshold(m, t) == auto_threshold(np.concatenate([hand, hand]))
    short = np.concatenate([hand, np.full(10 * fps, 0.05, np.float32), hand])  # 10秒なら外さない
    assert auto_threshold(short, np.arange(len(short)) / fps) == auto_threshold(short)
