"""Tesseractによるページ単位のOCR（透明テキスト付きPDF・テキスト・TSV）."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from vbs import faults

REPO_TESSDATA = Path(__file__).resolve().parent.parent / "tessdata"


class OcrError(RuntimeError):
    pass


def find_tesseract() -> Path:
    env = os.environ.get("VBS_TESSERACT")
    if env and Path(env).exists():
        return Path(env)
    found = shutil.which("tesseract")
    if found:
        return Path(found)
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
                 os.environ.get("LOCALAPPDATA")):
        if base:
            p = Path(base) / "Tesseract-OCR" / "tesseract.exe"
            if p.exists():
                return p
    raise OcrError("Tesseract が見つかりません。インストールするか VBS_TESSERACT を設定してください。")


def find_tessdata(langs: str) -> Path:
    cands = [os.environ.get("VBS_TESSDATA"), str(REPO_TESSDATA)]
    for c in cands:
        if c and all((Path(c) / f"{l}.traineddata").exists() for l in langs.split("+")):
            return Path(c)
    raise OcrError(f"言語データが見つかりません: {langs}（{REPO_TESSDATA} に配置してください）")


def tesseract_version() -> str:
    exe = find_tesseract()
    r = subprocess.run([str(exe), "--version"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", creationflags=_NO_WINDOW)
    return (r.stdout or r.stderr).splitlines()[0].strip()


_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def awake_seconds() -> float:
    """PCが起きていた時間（秒）。スリープ・スタンバイ中は進まない.

    壁時計で時間切れを測ると、蓋を閉じてスタンバイしている間も数えてしまい、
    正常なページを失敗扱いにする（実機で発生）。Windows では QueryUnbiasedInterruptTime を使う。
    """
    if sys.platform == "win32":
        import ctypes

        t = ctypes.c_ulonglong()
        if ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(t)):
            return t.value / 1e7
    import time

    return time.monotonic()


def _run_awake_timeout(args: list[str], env: dict, timeout: float | None) -> subprocess.CompletedProcess:
    """起きていた時間で時間切れを判定して外部コマンドを実行する."""
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                            errors="replace", env=env, creationflags=_NO_WINDOW)
    start = awake_seconds()
    while True:
        try:
            out, err = proc.communicate(timeout=1.0)
            return subprocess.CompletedProcess(args, proc.returncode, out, err)
        except subprocess.TimeoutExpired:
            if timeout is not None and awake_seconds() - start > timeout:
                proc.kill()
                proc.communicate()
                raise


TARGET_CHAR_PX = 48.0  # OCR時の目標の文字高さ（実写の教科書で1.2〜1.5倍が最良だった）


def auto_scale(line_pitch_px: float | None) -> float:
    """行の間隔から、文字高さが目標に近づく拡大率を決める（1〜2倍）."""
    if not line_pitch_px:
        return 1.0
    char = 0.8 * line_pitch_px
    return float(min(2.0, max(1.0, TARGET_CHAR_PX / char)))


def run_page(image: Path, out_base: Path, langs: str, dpi: int, psm: int = 6,
             scale: float = 1.0, timeout: float | None = None) -> dict[str, Any]:
    """1ページをOCRし、out_base.{pdf,txt,tsv} を作る.

    scale > 1 のときはOCR用に拡大した画像を認識させる。PDFには元画像をそのまま入れ、
    透明テキスト層だけを重ねる（拡大してもPDFは重くならない）。
    一時名で出力してから置換するので、失敗時に既存の結果を壊さない。
    """
    from vbs.imgio import imread, imwrite
    from vbs.pdf import compose_page, write_single
    from vbs.textlayer import line_texts, page_min_conf, rebuild_text_pdf

    exe = find_tesseract()
    tessdata = find_tessdata(langs)
    tmp_base = out_base.with_name(out_base.name + ".tmp")
    src = image
    scaled = None
    if scale > 1.01:
        import cv2

        img = imread(image)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        scaled = tmp_base.with_name(tmp_base.name + ".png")
        imwrite(scaled, img)
        src = scaled
    args = [
        str(exe), str(src), str(tmp_base),
        "--tessdata-dir", str(tessdata),
        "-l", langs,
        "--psm", str(psm),
        "--dpi", str(int(round(dpi * scale))),
        "-c", "preserve_interword_spaces=1",
        # configs/ に依存しないよう出力形式を直接指定する
        "-c", "tessedit_create_pdf=1",
        "-c", "textonly_pdf=1",
        "-c", "tessedit_create_txt=1",
        "-c", "tessedit_create_tsv=1",
    ]
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    outs = {ext: tmp_base.with_name(tmp_base.name + f".{ext}") for ext in ("pdf", "txt", "tsv")}
    try:
        try:
            faults.point("ocr.page", str(image))
            r = _run_awake_timeout(args, env, timeout)
        except (subprocess.TimeoutExpired, faults.InjectedTimeout) as e:
            raise OcrError(f"Tesseractが時間切れになりました（{timeout}秒）: {image.name}") from e
        if r.returncode != 0 or not all(p.exists() and p.stat().st_size > 0 for p in (outs["pdf"], outs["tsv"])):
            raise OcrError(f"Tesseract失敗 (code={r.returncode}): {r.stderr.strip()[-500:]}")
        page_tmp = tmp_base.with_name(tmp_base.name + ".page.pdf")
        # 日本語の語ごとに入る空白を除いた透明テキスト層に作り直してから重ねる
        text_tmp = tmp_base.with_name(tmp_base.name + ".text.pdf")
        min_conf = page_min_conf(outs["tsv"])
        rebuild_text_pdf(outs["pdf"], outs["tsv"], text_tmp, min_conf)
        # 確認画面やコピーで見る文字と同じになるよう、txt も文字層の規則で作り直す
        outs["txt"].write_text("\n".join(line_texts(outs["tsv"], min_conf)) + "\n", encoding="utf-8")
        try:
            write_single(compose_page(image, dpi, text_tmp), page_tmp)
        finally:
            text_tmp.unlink(missing_ok=True)
        final = {}
        dst = out_base.with_name(out_base.name + ".pdf")
        os.replace(page_tmp, dst)
        final["pdf"] = dst
        for ext in ("txt", "tsv"):
            dst = out_base.with_name(out_base.name + f".{ext}")
            if outs[ext].exists():
                os.replace(outs[ext], dst)
            final[ext] = dst
    finally:
        for p in list(outs.values()) + ([scaled] if scaled else []):
            p.unlink(missing_ok=True)
    stats = tsv_stats(final["tsv"])
    stats.update(final)
    stats["text_min_conf"] = min_conf
    return stats


def tsv_stats(tsv: Path) -> dict[str, Any]:
    """語ごとの信頼度から平均と文字数を求める（CERの代用にはしない）."""
    confs, chars = [], 0
    with open(tsv, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            text = (row.get("text") or "").strip()
            try:
                conf = float(row.get("conf") or -1)
            except ValueError:
                conf = -1
            if text and conf >= 0:
                confs.append((conf, len(text)))
                chars += len(text)
    mean = sum(c * n for c, n in confs) / chars if chars else None
    sure = sum(n for c, n in confs if c >= SURE_CONF)
    return {"chars": chars, "mean_conf": round(mean, 1) if mean is not None else None, "sure_chars": sure}


SURE_CONF = 70.0          # この信頼度以上の語の文字を「確かな文字」として数える
RETRY_BELOW_CONF = 70.0   # 平均信頼度がこれ未満なら別の条件で読み直す
RETRY_PARAMS = [(1.0, 6), (1.0, 3), (0.6, 3), (0.6, 6)]


def run_page_best(image: Path, out_base: Path, langs: str, dpi: int, psm: int, scale: float,
                  timeout: float | None = None) -> dict[str, Any]:
    """まず指定の条件で読み、信頼度が低ければ拡大率と区切り方を変えて読み直す.

    大きな見出し文字を拡大しすぎると崩れる、段組みでは区切り方の自動判定が効く、
    といった差を拾う。比べるのは「確かな文字」の数（平均値だと読み落としが有利になる）。
    """
    first = run_page(image, out_base, langs, dpi, psm=psm, scale=scale, timeout=timeout)
    first.update(scale=scale, psm=psm, tried=1)
    if first["mean_conf"] is None or first["mean_conf"] >= RETRY_BELOW_CONF:
        return first
    best = first
    tried = 1
    for i, (sc, ps) in enumerate(RETRY_PARAMS):
        if abs(sc - scale) < 1e-6 and ps == psm:
            continue
        alt_base = out_base.with_name(out_base.name + f".try{i}")
        try:
            st = run_page(image, alt_base, langs, dpi, psm=ps, scale=sc, timeout=timeout)
        except OcrError:
            continue
        tried += 1
        st.update(scale=sc, psm=ps)
        if st["sure_chars"] > best["sure_chars"]:
            if best is not first:
                for ext in ("pdf", "txt", "tsv"):
                    Path(best[ext]).unlink(missing_ok=True)
            best = st
        else:
            for ext in ("pdf", "txt", "tsv"):
                Path(st[ext]).unlink(missing_ok=True)
    if best is not first:
        for ext in ("pdf", "txt", "tsv"):
            dst = out_base.with_name(out_base.name + f".{ext}")
            os.replace(best[ext], dst)
            best[ext] = dst
    best["tried"] = tried
    return best
