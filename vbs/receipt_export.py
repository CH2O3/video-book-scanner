"""レシートの一覧（CSV・JSON）、画像、AIへの確認依頼文を書き出す。直した一覧を読み込む.

PDFと同じ output フォルダに置く。AIに頼むときは、このフォルダ（またはPDFと一覧）を渡せばよい。
"""

from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path
from typing import Any

from vbs import receipt
from vbs.manifest import Project, ProjectError, atomic_write_text, now_iso

COLUMNS = ["No", "ID"] + [receipt.FIELD_LABELS[k] for k in receipt.FIELDS] + ["状態", "画像", "要確認"]
STATE_LABELS = {None: "自動", "manual": "手修正", "import": "一覧から修正", "ai": "AIが修正"}


def _name_part(s: Any, n: int = 16) -> str:
    bad = '<>:"/\\|?*'
    t = "".join("_" if c in bad or ord(c) < 32 else c for c in str(s or "")).strip()
    return t[:n] or "不明"


def rows(project: Project, order: list[str]) -> list[dict[str, Any]]:
    from vbs import pipeline

    out = []
    review = {it["id"]: it for it in pipeline.review_items(project)}
    for n, pid in enumerate(order, 1):
        p = project.data["pages"][pid]
        rec = p.get("receipt") or {}
        f = receipt.merged(rec)
        out.append({"no": n, "id": pid, "fields": f, "page": p, "rec": rec,
                    "state": STATE_LABELS.get(rec.get("manual_source"), "手修正") if rec.get("manual") else "自動",
                    "review": review.get(pid, {}).get("labels", [])})
    return out


def write_package(project: Project, order: list[str], pdf_path: Path) -> dict[str, str]:
    """PDFと一緒に、一覧・画像・AIへの確認依頼文を書き出す."""
    out_dir = pdf_path.parent
    stem = pdf_path.stem
    img_dir = out_dir / f"{stem}_画像"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)
    rs = rows(project, order)
    items = []
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(COLUMNS)
    for r in rs:
        f, p = r["fields"], r["page"]
        name = (f"{r['no']:03d}_{f.get('date') or '日付不明'}_"
                f"{(str(f['total']) + '円') if f.get('total') else '金額不明'}_{_name_part(f.get('payee'))}.jpg")
        shutil.copyfile(project.abs(p["image"]), img_dir / name)
        rel_img = f"{img_dir.name}/{name}"
        w.writerow([r["no"], r["id"]] + [("" if f.get(k) is None else f.get(k)) for k in receipt.FIELDS]
                   + [r["state"], rel_img, "、".join(r["review"])])
        ocr_txt = ""
        if (p.get("ocr") or {}).get("txt"):
            try:
                ocr_txt = project.abs(p["ocr"]["txt"]).read_text(encoding="utf-8")
            except OSError:
                pass
        auto = (r["rec"].get("auto") or {})
        items.append({
            "no": r["no"], "id": r["id"], "pdf_page": r["no"], "image": rel_img,
            "fields": f, "state": r["state"], "needs_review": r["review"],
            "auto_fields": auto.get("fields"), "evidence_lines": auto.get("evidence"),
            "payee_candidates": (auto.get("candidates") or {}).get("payee"),
            "checks": receipt.checks(f), "ocr_text": ocr_txt,
            "resolution_dpi": p.get("dpi"),
        })
    csv_path = out_dir / f"{stem}_一覧.csv"
    # Excelで文字化けしないよう、BOM付きUTF-8
    atomic_write_text(csv_path, "﻿" + buf.getvalue())
    json_path = out_dir / f"{stem}_一覧.json"
    atomic_write_text(json_path, json.dumps({
        "title": project.data.get("title"), "created": now_iso(), "pdf": pdf_path.name,
        "columns": {k: receipt.FIELD_LABELS[k] for k in receipt.FIELDS},
        "receipt_width_mm": project.settings.get("receipt_width_mm"),
        "receipts": items}, ensure_ascii=False, indent=2))
    md_path = out_dir / f"{stem}_AIに確認を頼む.md"
    atomic_write_text(md_path, ai_request(stem, len(items)))
    return {"csv": str(csv_path), "json": str(json_path), "images": str(img_dir), "ai_request": str(md_path)}


def ai_request(stem: str, n: int) -> str:
    cols = "、".join(COLUMNS)
    return f"""# レシートの読み取り結果の確認のお願い

このフォルダには、レシート {n} 枚を撮影して文字を読み取った結果があります。
読み取りは機械によるもので、誤りや抜けがあります。画像と見比べて確認し、直してください。

## ファイル

- `{stem}.pdf`：レシート1枚が1ページ（{n}ページ）。順番は一覧の「No」と同じ
- `{stem}_画像/`：同じレシートの画像。ファイル名の先頭3桁が「No」
- `{stem}_一覧.csv`：読み取った項目の一覧（列：{cols}）
- `{stem}_一覧.json`：同じ内容に、読み取りの根拠になった行（evidence_lines）と、文字認識の全文（ocr_text）を加えたもの

## 確認してほしい項目

1レコード（1枚）ずつ、画像を見て次を確かめてください。

| 列 | 内容 |
|---|---|
| 取引日 | レシートに印字された日付。`2024-01-31` の形 |
| 取引先 | 店名・会社名（支店名があれば含める） |
| 金額（税込） | 支払った合計額。お預かり・お釣りではない。数字だけ（カンマ・円は不要） |
| 10%対象・8%対象（軽減） | 税率ごとの対象額。印字がなければ空欄 |
| 消費税 | 消費税額（内税なら「内消費税」の額）。印字がなければ空欄 |
| 登録番号 | 適格請求書発行事業者の登録番号。`T` と13桁。印字がなければ空欄 |
| メモ | 気づいたこと（例：品目の概要、手書きの但し書き、読めない箇所） |

## 守ってほしいこと

- **読めない文字を推測で埋めないでください。** 画像で読めない項目は空欄にし、「メモ」に「読めない」と書いてください。
- **「No」と「ID」は変えないでください。** 行を足したり消したりしないでください。
- 同じレシートが2回写っていると思ったら、「メモ」に「No.○と同じ」と書いてください（行は消さない）。
- 直した行は「状態」を `AIが修正` に、確認して正しかった行は `確認済み` にしてください。
- レシートにはカード番号の一部や住所などが含まれることがあります。確認以外の目的に使ったり、外部に送ったりしないでください。

## 返してほしいもの

1. 直した一覧を、**同じ列・同じ順番のCSV**で（`{stem}_一覧_確認済み.csv` として保存できる形）
2. 直した箇所の一覧（No・列・直す前・直した後・理由）
3. 自信がない箇所と、読めなかった箇所

直したCSVは、Video Book Scanner の確認画面の「一覧を読み込む」で取り込めます。
"""


def import_csv(project: Project, text: str) -> dict[str, Any]:
    """直した一覧（CSV）を取り込む。IDで対応させ、値が変わった項目だけ手修正として記録する."""
    from vbs import pipeline

    text = text.lstrip("﻿")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or "ID" not in reader.fieldnames:
        raise ProjectError("一覧に「ID」の列がありません。書き出した一覧と同じ列のCSVを読み込んでください。")
    by_label = {v: k for k, v in receipt.FIELD_LABELS.items()}
    changed, unchanged, errors = 0, 0, []
    for line_no, row in enumerate(reader, 2):
        pid = (row.get("ID") or "").strip()
        if pid not in project.data["pages"]:
            errors.append(f"{line_no}行目：ID {pid or '（空）'} のレシートがありません")
            continue
        p = project.data["pages"][pid]
        current = receipt.merged(p.get("receipt"))
        values = {}
        for col, val in row.items():
            k = by_label.get((col or "").strip())
            if not k:
                continue
            try:
                v = receipt.coerce(k, val)
            except ValueError as e:
                errors.append(f"{line_no}行目（{pid}）：{e}")
                continue
            if v != current.get(k):
                values[k] = val
        state = (row.get("状態") or "").strip()
        source = "ai" if "AI" in state else "import"
        if values:
            pipeline.set_receipt_fields(project, pid, values, source=source)
            changed += 1
        else:
            unchanged += 1
        f = receipt.merged(p.get("receipt"))
        if "確認済み" in state or "AI" in state:
            # 画像と見比べた行は、検査用の数字・対象額の食い違いの疑いを確認済みにする
            p["acknowledged"] = sorted(set(p.get("acknowledged", []))
                                       | {"receipt_regno_suspect", "receipt_amount_mismatch",
                                          "receipt_total_disagree", "receipt_date_disagree"})
        if "読めない" in (f.get("note") or ""):
            # 画像でも読めないと確かめた項目は、空欄のまま確認済みにする（推測で埋めない）
            p["acknowledged"] = sorted(set(p.get("acknowledged", [])) | set(receipt.missing(f)))
    project.save()
    return {"changed": changed, "unchanged": unchanged, "errors": errors}
