"""PDFの組立て.

ページ画像（JPEG）は再圧縮せずにそのまま埋め込み、OCRの透明テキスト層を重ねる。
結合は確定したページID順で行い、一時ファイルで検証してから置換する（PDF-03, PDF-07）。
"""

from __future__ import annotations

import io
import os
from pathlib import Path

from PIL import Image

from vbs import faults
from pypdf import PageObject, PdfReader, PdfWriter, Transformation

_CS = {"L": "/DeviceGray", "RGB": "/DeviceRGB", "CMYK": "/DeviceCMYK"}


def jpeg_page_pdf(image: Path, dpi: int) -> bytes:
    """JPEGを1枚だけ含むPDFを作る（DCTDecodeでそのまま格納、再圧縮しない）."""
    data = Path(image).read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        if im.format != "JPEG" or im.mode not in _CS:
            # JPEG以外は可逆にJPEG化せず、品質95で変換する
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=95, subsampling=0)
            data = buf.getvalue()
            w, h, mode = im.width, im.height, "RGB"
        else:
            w, h, mode = im.width, im.height, im.mode
    wpt, hpt = w * 72.0 / dpi, h * 72.0 / dpi
    content = f"q {wpt:.4f} 0 0 {hpt:.4f} 0 0 cm /Im0 Do Q".encode()
    decode = " /Decode [1 0 1 0 1 0 1 0]" if mode == "CMYK" else ""
    objs: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {wpt:.4f} {hpt:.4f}] "
         f"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>").encode(),
        (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace {_CS[mode]} "
         f"/BitsPerComponent 8 /Filter /DCTDecode{decode} /Length {len(data)} >>\nstream\n").encode()
        + data + b"\nendstream",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def image_page(image: Path, dpi: int) -> PageObject:
    return PdfReader(io.BytesIO(jpeg_page_pdf(image, dpi))).pages[0]


def compose_page(image: Path, dpi: int, text_pdf: Path | None) -> PageObject:
    """画像ページに透明テキスト層を重ねる。テキスト層の寸法差は拡大縮小で合わせる."""
    page = image_page(image, dpi)
    if text_pdf is None:
        return page
    text = PdfReader(str(text_pdf)).pages[0]
    pw, ph = float(page.mediabox.width), float(page.mediabox.height)
    tw, th = float(text.mediabox.width), float(text.mediabox.height)
    if abs(tw - pw) > 0.01 or abs(th - ph) > 0.01:
        page.merge_transformed_page(text, Transformation().scale(pw / tw, ph / th))
    else:
        page.merge_page(text)
    return page


def write_single(page: PageObject, out: Path) -> None:
    w = PdfWriter()
    w.add_page(page)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        w.write(f)
    os.replace(tmp, out)


def build_pdf(entries: list[tuple[str, Path | None, Path, int]], out: Path, title: str | None = None) -> int:
    """entries: (page_id, 合成済みページPDF or None, 画像, dpi) の出力順リスト.

    ページPDFがない（OCRなしで残すと決めた）ページは画像だけで作る。
    """
    writer = PdfWriter()
    for page_id, page_pdf, image, dpi in entries:
        if page_pdf:
            reader = PdfReader(str(page_pdf))
            if len(reader.pages) != 1:
                raise RuntimeError(f"ページPDFのページ数が1ではありません: {page_id}")
            writer.add_page(reader.pages[0])
        else:
            writer.add_page(image_page(image, dpi))
    meta = {"/Producer": "video-book-scanner"}
    if title:
        meta["/Title"] = title
    writer.add_metadata(meta)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    faults.point("write", str(tmp))
    with open(tmp, "wb") as f:
        writer.write(f)
        faults.point("export.writing")
        f.flush()
        os.fsync(f.fileno())
    n = len(PdfReader(str(tmp)).pages)
    if n != len(entries):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"書き出したPDFのページ数が一致しません: {n} != {len(entries)}")
    os.replace(tmp, out)
    return n
