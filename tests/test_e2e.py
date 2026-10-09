"""合成動画で動画→PDFまでを通す（Tesseractが必要）."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pypdf import PdfReader

from vbs import pipeline
from vbs.manifest import Project
from vbs.ocr import OcrError, find_tessdata, find_tesseract

ROOT = Path(__file__).resolve().parent.parent


def _have_ocr() -> bool:
    try:
        find_tesseract()
        find_tessdata("jpn+eng")
        return True
    except OcrError:
        return False


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> tuple[Path, dict]:
    d = tmp_path_factory.mktemp("vid")
    out = d / "テスト 動画.mp4"  # 日本語・空白を含むパス（IN-05）
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_test_video.py"), str(out),
                    "--spreads", "3", "--vfr", "--bump", "--seed", "3"], check=True, capture_output=True)
    return out, json.loads(out.with_suffix(".truth.json").read_text(encoding="utf-8"))


def test_extract_matches_truth(video, tmp_path: Path):
    path, truth = video
    p = Project.create(tmp_path / "p")
    pipeline.add_videos(p, [path])
    pipeline.analyze(p)
    inc = [s for s in p.data["segments"] if s["include"]]
    assert len(inc) == len(truth["spreads"])
    for sp in truth["spreads"]:
        hits = [s for s in inc if s["start_sec"] < sp["end"] and s["end_sec"] > sp["start"]]
        assert len(hits) == 1, sp
    # 位置ずれした同じ見開きは重複として残る（消さない）
    assert any(s["duplicate_of"] for s in p.data["segments"])
    # 採用フレームの時刻は正解区間内（可変fpsでもPTSで対応, AC-07）
    for s in inc:
        f = p.data["frames"][s["chosen"]]
        assert any(sp["start"] <= f["time_sec"] <= sp["end"] for sp in truth["spreads"])
        assert "frame_mismatch" not in f["warnings"]


@pytest.mark.skipif(not _have_ocr(), reason="Tesseract/言語データなし")
def test_full_pipeline_and_partial_reprocess(video, tmp_path: Path):
    path, truth = video
    p = Project.create(tmp_path / "p", {"auto_export": True})
    pipeline.add_videos(p, [path])
    res = pipeline.run_all(p)
    assert res["export"], res["review"]
    # 出力したPDFそのものを開き直した検証記録（ハッシュ・ページ数）が残る
    vs = res["export"]["verify_summary"]
    assert vs["page_count_ok"] and len(res["export"]["sha256"]) == 64
    assert Path(res["export"]["verify"]).exists()
    pdf = PdfReader(res["export"]["path"])
    assert len(pdf.pages) == 2 * len(truth["spreads"])
    text = "".join((pg.extract_text() or "") for pg in pdf.pages)
    assert "理科と社会の基礎" in text.replace(" ", "")

    # 一つの区間の境界だけ変える → その区間のページだけOCRし直す（仕様13章）
    seg = next(s for s in p.data["segments"] if s["include"])
    before = {pid: pg["ocr"]["key"] for pid, pg in p.data["pages"].items()}
    gx = seg["geometry"]["gutter_x"]
    pipeline.set_geometry(p, seg["id"], gutter_ratio=(gx + 6) / p.data["frames"][seg["chosen"]]["width"])
    pipeline.run_all(p)
    after = {pid: pg["ocr"]["key"] for pid, pg in p.data["pages"].items()}
    changed = {pid for pid in before if before[pid] != after[pid]}
    assert changed == {f"{seg['id']}-L", f"{seg['id']}-R"}

    # 候補の差し替えでは手修正の境界を引き継がない
    if len(seg["candidates"]) > 1:
        other = next(f for f in seg["candidates"] if f != seg["chosen"])
        seg["chosen"] = other
        assert pipeline._revision(p, seg) != p.data["pages"][f"{seg['id']}-L"]["revision"]
