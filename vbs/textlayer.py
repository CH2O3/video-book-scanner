"""透明テキスト層の組み立て.

Tesseract の PDF 出力は、日本語でも語ごとに空白文字（U+0020）を付けて配置するため、
コピーすると「反則 の 違反 要件」のように空白が混じり、ビューアによっては検索にも響く。
ここでは Tesseract の TSV（語ごとの位置）から行単位で配置し直す。

- フォントは Tesseract が埋め込む GlyphLessFont（CID = UTF-16 符号、幅 500）を流用する
- 同じ行の語は次の語の左端まで隙間なく伸ばすので、ビューアが空白を補わない
- 空白は英数字の語どうしの間にだけ入れる
- 描画モード 3（不可視）。画像の上に重ねても見た目は変わらない
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, StreamObject

GLYPH_W = 0.5  # GlyphLessFont の /DW 500
_LATIN_END = re.compile(r"[A-Za-z0-9]$")
_LATIN_START = re.compile(r"^[A-Za-z0-9]")


UNRELIABLE_PAGE_CONF = 35.0  # ページ平均の信頼度がこれ未満なら「信頼できないページ」
UNRELIABLE_WORD_MIN = 60.0   # 信頼できないページでは、この信頼度以上の語だけを文字層に入れる


def page_min_conf(tsv: Path) -> float:
    """文字層に入れる語の信頼度の下限を決める.

    語ごとに一律で足切りすると、目次などで信頼度の低い本物の語を落とす（実写で確認）。
    ほぼ白紙のページで背景を読んだゴミだけを止めるため、ページ全体が低いときだけ厳しくする。
    """
    _, lines = _read_tsv(tsv)
    words = [w for ws in lines for w in ws]
    n = sum(len(w["text"]) for w in words)
    if not n:
        return 0.0
    mean = sum(w["conf"] * len(w["text"]) for w in words) / n
    return UNRELIABLE_WORD_MIN if mean < UNRELIABLE_PAGE_CONF else 0.0


def line_texts(tsv: Path, min_conf: float = 0.0) -> list[str]:
    """文字層と同じ規則（日本語の語の間に空白を入れない）で行ごとの文字列を作る."""
    _, lines = _read_tsv(tsv)
    out = []
    for words in lines:
        words = [w for w in words if w["conf"] >= min_conf]
        buf = ""
        for i, w in enumerate(words):
            buf += w["text"]
            nxt = words[i + 1] if i + 1 < len(words) else None
            if nxt and _LATIN_END.search(w["text"]) and _LATIN_START.search(nxt["text"]):
                buf += " "
        if buf:
            out.append(buf)
    return out


def _read_tsv(tsv: Path) -> tuple[tuple[int, int], list[list[dict]]]:
    page = (0, 0)
    lines: dict[tuple[int, int, int], list[dict]] = {}
    with open(tsv, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            level = int(row["level"])
            if level == 1:
                page = (int(row["width"]), int(row["height"]))
            if level != 5:
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            key = (int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
            lines.setdefault(key, []).append({
                "text": text, "conf": float(row.get("conf") or -1),
                "left": int(row["left"]), "top": int(row["top"]),
                "width": int(row["width"]), "height": int(row["height"]),
            })
    ordered = [sorted(ws, key=lambda w: w["left"]) for _, ws in sorted(lines.items())]
    return page, ordered


def _hex(text: str) -> str:
    return text.encode("utf-16-be").hex().upper()


def build_content(tsv: Path, page_w_pt: float, page_h_pt: float, font: str, min_conf: float = 0.0) -> bytes:
    (pw, ph), lines = _read_tsv(tsv)
    lines = [ws for ws in ([w for w in words if w["conf"] >= min_conf] for words in lines) if ws]
    if not pw or not ph:
        return b""
    sx, sy = page_w_pt / pw, page_h_pt / ph
    out = ["BT", "3 Tr"]
    for words in lines:
        top = min(w["top"] for w in words)
        bottom = max(w["top"] + w["height"] for w in words)
        size = max(1.0, (bottom - top) * sy)
        base_y = page_h_pt - bottom * sy + 0.12 * size  # 文字の下端から少し上を基線にする
        out.append(f"/{font} {size:.2f} Tf")
        # 語ごとに位置を指定し、次の語の左端まで横に伸ばして隙間をなくす。
        # （1行を1つの TJ にまとめる方式は、PDFium で行の区切りが失われたので採らない）
        for i, w in enumerate(words):
            text = w["text"]
            nxt = words[i + 1] if i + 1 < len(words) else None
            if nxt and _LATIN_END.search(text) and _LATIN_START.search(nxt["text"]):
                text += " "
            x = w["left"] * sx
            end = (nxt["left"] if nxt else w["left"] + w["width"]) * sx
            natural = len(text.encode("utf-16-be")) // 2 * GLYPH_W * size
            tz = max(1.0, (end - x) / natural * 100) if natural > 0 else 100.0
            out.append(f"{tz:.2f} Tz 1 0 0 1 {x:.3f} {base_y:.3f} Tm <{_hex(text)}> Tj")
    out.append("ET")
    return ("\n".join(out) + "\n").encode("ascii")


def rebuild_text_pdf(tesseract_pdf: Path, tsv: Path, out: Path, min_conf: float = 0.0) -> None:
    """Tesseract のテキストのみPDFの本文を、行単位の配置に置き換えて保存する."""
    reader = PdfReader(str(tesseract_pdf))
    page = reader.pages[0]
    fonts = page["/Resources"]["/Font"]
    font = str(list(fonts.keys())[0]).lstrip("/")
    data = build_content(tsv, float(page.mediabox.width), float(page.mediabox.height), font, min_conf)
    stream = StreamObject()
    stream.set_data(data)
    writer = PdfWriter()
    writer.add_page(page)
    wpage = writer.pages[0]
    wpage[NameObject("/Contents")] = writer._add_object(stream)
    with open(out, "wb") as f:
        writer.write(f)
