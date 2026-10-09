"""出力したPDFそのものの検証.

中間ファイルではなく、利用者に渡す最終PDFを開いて調べ、ファイルのハッシュと一緒に記録する。
ブラウザ（Chrome・Edge）と同じ PDFium で検索するので、実際の検索結果に近い。

- ページ数、各ページの文字層の文字数
- 重要語の表（{ページID: [語, ...]}）があれば、各語がそのページで検索できるか
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from vbs.manifest import atomic_write_json, now_iso


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(4 << 20):
            h.update(b)
    return h.hexdigest()


def _search(page_text, word: str) -> bool:
    s = page_text.search(word, match_case=False, match_whole_word=False)
    try:
        return s.get_next() is not None
    finally:
        s.close()


def verify_pdf(pdf_path: Path, order: list[str], terms: dict[str, list[str]] | None = None) -> dict[str, Any]:
    import pypdfium2 as pdfium

    pdf_path = Path(pdf_path)
    rec: dict[str, Any] = {
        "time": now_iso(),
        "pdf": str(pdf_path.resolve()),
        "sha256": sha256(pdf_path),
        "size": pdf_path.stat().st_size,
        "engine": f"pypdfium2 {pdfium.PYPDFIUM_INFO} / PDFium {pdfium.PDFIUM_INFO}",
    }
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        rec["pages"] = len(doc)
        rec["page_count_ok"] = len(doc) == len(order)
        per_page = []
        for i in range(len(doc)):
            tp = doc[i].get_textpage()
            text = tp.get_text_range()
            entry: dict[str, Any] = {"pdf_page": i + 1, "page_id": order[i] if i < len(order) else None,
                                     "text_chars": len("".join(text.split()))}
            pid = entry["page_id"]
            if terms and pid in terms:
                found = [w for w in terms[pid] if _search(tp, w)]
                entry["terms_found"] = found
                entry["terms_missing"] = [w for w in terms[pid] if w not in found]
            tp.close()
            per_page.append(entry)
        rec["per_page"] = per_page
    finally:
        doc.close()
    if terms:
        checked = [p for p in rec["per_page"] if "terms_found" in p]
        n_found = sum(len(p["terms_found"]) for p in checked)
        n_all = sum(len(p["terms_found"]) + len(p["terms_missing"]) for p in checked)
        rec["terms_summary"] = {"found": n_found, "total": n_all,
                                "missing": {p["page_id"]: p["terms_missing"] for p in checked if p["terms_missing"]}}
    return rec


def load_terms(path: Path | None) -> dict[str, list[str]] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def write_report(rec: dict[str, Any], pdf_path: Path) -> Path:
    out = Path(pdf_path).with_suffix(".verify.json")
    atomic_write_json(out, rec)
    return out
