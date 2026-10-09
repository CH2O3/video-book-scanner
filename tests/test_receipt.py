"""レシートの読み取り・一覧・並べ替え."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vbs import pipeline, receipt
from vbs.manifest import Project, with_preset
from vbs.ocr import OcrError, find_tessdata, find_tesseract

ROOT = Path(__file__).resolve().parent.parent

SAMPLE = """みどり商店 駅前店
東京都千代田区架空町1-2-3
TEL 03-0000-0000
登録番号 T7000012050002
領 収 書
2024年06月03日 17:27
ノート ¥220
コーヒー※ ¥480
小計 ¥700
合計 ¥700
(10%対象 ¥220)
(8%対象 ¥480)
(内消費税等 ¥55)
お預り ¥1,000
お釣り ¥300
"""


def test_parse_sample():
    r = receipt.parse(SAMPLE)["fields"]
    assert r["date"] == "2024-06-03"
    assert r["total"] == 700          # 小計・お預り・お釣りではなく合計
    assert r["base10"] == 220 and r["base8"] == 480 and r["tax"] == 55
    assert r["registration_no"] == "T7000012050002"
    assert r["payee"] == "みどり商店 駅前店"
    assert receipt.missing(r) == []


def test_dates_and_ocr_variants():
    for text, want in [("令和6年1月2日", "2024-01-02"), ("R6.1.2 12:00", "2024-01-02"), ("2024/1/2", "2024-01-02"),
                       ("24/01/02", "2024-01-02"), ("令和元年5月1日", "2019-05-01")]:
        assert receipt.find_date([text])[0] == want, text
    # 円記号が「\\」や「Y」に化けても金額として読む
    assert receipt.find_total(["合計\\1,234"])[0] == 1234
    assert receipt.find_total(["合計 Y980"])[0] == 980
    assert receipt.find_total(["小計 ¥500", "お預り ¥1,000"])[0] is None


def test_registration_check_digit():
    # 公表されている法人番号（国税庁、トヨタ自動車）で検算する
    assert receipt.registration_check_ok("T7000012050002")
    assert receipt.registration_check_ok("T1180301018771")
    assert receipt.registration_check_ok("T7000012050003") is False
    # T が化けて数字が1つ多い行からは、検査に通る13桁を選ぶ
    no, _ = receipt.find_registration_no(["登録番号「17000012050002"])
    assert no == "T7000012050002"
    assert "receipt_regno_suspect" in receipt.missing({"date": "2024-01-01", "total": 1, "payee": "x",
                                                       "registration_no": "T7000012050003"})


def test_coerce_and_merge():
    assert receipt.coerce("total", "¥1,234円") == 1234
    assert receipt.coerce("date", "2024/6/3") == "2024-06-03"
    assert receipt.coerce("registration_no", "T7000-0120-50002") == "T7000012050002"
    with pytest.raises(ValueError):
        receipt.coerce("registration_no", "T123")
    rec = {"auto": {"fields": {"total": 700, "payee": "A"}}, "manual": {"payee": "B"}}
    assert receipt.merged(rec)["payee"] == "B" and receipt.merged(rec)["total"] == 700


def test_receipt_preset_keeps_appearance():
    s = with_preset({"document": "receipt", "margins": "content_box", "dewarp": "auto"})
    # 証憑として見た目を変えない（明示されても上書きしない）
    assert s["layout"] == "single" and s["enhance"] == "none" and s["dewarp"] == "off"
    assert s["margins"] == "detect" and s["dedupe"] == "flag"
    assert with_preset({})["document"] == "book" and with_preset({})["margins"] == "content_box"


def _have_ocr() -> bool:
    try:
        find_tesseract()
        find_tessdata("jpn")
        return True
    except OcrError:
        return False


@pytest.mark.skipif(not _have_ocr(), reason="Tesseract/言語データなし")
def test_receipt_photos_end_to_end(tmp_path: Path):
    photos = tmp_path / "photos"
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_test_receipts.py"), str(photos), "--n", "3",
                    "--seed", "11"], check=True, capture_output=True)
    truth = json.loads((photos / "truth.json").read_text(encoding="utf-8"))
    p = Project.create(tmp_path / "p", {"document": "receipt", "auto_export": False})
    p.data["title"] = "テスト"
    for f in sorted(photos.glob("*.jpg")):
        pipeline.add_photo(p, f, None)
    pipeline.update_order(p)
    pipeline.run_all(p, export=False)
    assert len(p.data["order"]) == 3
    right = 0
    for pid, t in zip(p.data["order"], truth):
        pg = p.data["pages"][pid]
        f = pipeline.receipt_fields(p, pg)
        ok = f["date"] == t["date"] and f["total"] == t["total"]
        right += ok
        # 読み誤ったら、必ず要確認に回っている（黙って違う値を出さない）
        assert ok or pipeline.receipt_warnings(p, pg), (pid, f, t)
        assert pg["dpi"] >= pipeline.RECEIPT_MIN_DPI and "low_resolution" not in pg["warnings"]
    assert right >= 2
    # 日付順（直した値で並ぶ。ここでは正解の日付を手で入れてから並べる）
    for pid, t in zip(p.data["order"], truth):
        pipeline.set_receipt_fields(p, pid, {"date": t["date"]})
    pipeline.sort_receipts(p, "date")
    dates = [pipeline.receipt_fields(p, p.data["pages"][pid])["date"] for pid in p.data["order"]]
    assert dates == sorted(t["date"] for t in truth)
    # 出力と、一覧の読み込み
    rec = pipeline.export_pdf(p, force=True)
    files = rec["receipt_files"]
    for k in ("csv", "json", "images", "ai_request"):
        assert Path(files[k]).exists()
    from vbs.receipt_export import import_csv

    csv_text = Path(files["csv"]).read_text(encoding="utf-8-sig")
    first_id = p.data["order"][0]
    edited = csv_text.replace(f",{first_id},", f",{first_id},", 1)
    lines = edited.splitlines()
    cols = lines[1].split(",")
    cols[9] = "手書きの但し書きあり"  # メモ
    lines[1] = ",".join(cols)
    res = import_csv(p, "\n".join(lines))
    assert res["changed"] == 1 and not res["errors"]
    assert pipeline.receipt_fields(p, p.data["pages"][first_id])["note"] == "手書きの但し書きあり"
