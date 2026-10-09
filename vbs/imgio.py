"""日本語パスでも動く画像入出力（cv2.imread/imwrite はWindowsで非ASCIIパスに弱い）."""

from __future__ import annotations

import io
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from vbs import faults


def imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise OSError(f"画像を読み込めません: {path}")
    return img


def imwrite(path: Path, img: np.ndarray, quality: int = 95, dpi: int | None = None) -> None:
    """一時ファイル経由で書き込む。dpi を指定するとJPEG/PNGに解像度を記録する."""
    path = Path(path)
    faults.point("write", str(path))
    ext = path.suffix.lower()
    tmp = path.with_name(path.name + ".tmp")
    if dpi:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
        buf = io.BytesIO()
        fmt = "JPEG" if ext in (".jpg", ".jpeg") else "PNG"
        kw = {"quality": quality, "subsampling": 0} if fmt == "JPEG" else {"compress_level": 3}
        Image.fromarray(rgb).save(buf, format=fmt, dpi=(dpi, dpi), **kw)
        data = buf.getvalue()
    else:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else [cv2.IMWRITE_PNG_COMPRESSION, 3]
        ok, enc = cv2.imencode(ext, img, params)
        if not ok:
            raise OSError(f"画像をエンコードできません: {path}")
        data = enc.tobytes()
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
