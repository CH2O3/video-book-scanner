"""全工程の進行管理：動画追加 → 抽出 → 分割・補正 → OCR → 確認判定 → PDF出力.

各ページは入力（採用フレーム・設定・手修正）から作った revision を持ち、
revision が変わったページだけを作り直す（仕様13章の部分再処理）。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import cv2

from vbs import __version__, faults
from vbs.extract import clear_checkpoints, extract_video
from vbs.imgio import imread, imwrite
from vbs.manifest import Project, ProjectError, now_iso
from vbs.ocr import OcrError, auto_scale, run_page_best, tesseract_version
from vbs.pdf import build_pdf
from vbs.split import estimate_line_pitch, split_spread
from vbs.video import VideoError, probe, sha256_file

ProgressFn = Callable[[str, float, str], None]  # (工程, 進捗0-1, メッセージ)

# 処理方式を変えたら上げる。既存ページ（OCR）は作り直しになる。
SPLIT_VERSION = 17
OCR_VERSION = 7

# 確認理由: code -> (表示名, 出力前に確認が必要か)
WARNINGS: dict[str, tuple[str, bool]] = {
    "blur": ("ぼけ", True),
    "exposure_dark": ("露出不足", True),
    "exposure_bright": ("白飛び", True),
    "page_cut": ("紙面の欠け（文字が画面端に掛かる）", True),
    "page_touches_frame": ("紙面が画面端に接触（文字は範囲内）", False),
    "page_region_uncertain": ("紙面の範囲を判定できない", True),
    "gutter_uncertain": ("綴じ目の位置が不確か", True),
    "edge_content_top": ("上端で文字切れの疑い", True),
    "edge_content_bottom": ("下端で文字切れの疑い", True),
    "edge_content_left": ("左端で文字切れの疑い", True),
    "edge_content_right": ("右端で文字切れの疑い", True),
    "empty_crop": ("ページ画像が空", True),
    "frame_mismatch": ("取り出した画像が解析時と一致しない", True),
    "frame_missing": ("候補フレームを取り出せない", True),
    "no_candidate": ("候補なし", True),
    "short_still": ("静止が短い", True),
    "duplicate_suspect": ("重複の疑い（同じものを続けて撮った、または白紙など）", True),
    "unreadable": ("解析できない区間", True),
    "ocr_failed": ("OCR失敗", True),
    "ocr_low_conf": ("OCRの信頼度が低い", True),
    "ocr_no_text": ("文字が認識されない", True),
    "page_number_gap": ("ページ番号の飛び", True),
    "orientation_uncertain": ("紙面の向きを判定できない", True),
    "hand_in_content": ("本文の範囲に手が重なっている疑い", True),
    "hdr_pq": ("PQ方式のHDR（色変換未検証）", True),
    "hdr_hlg": ("HLG方式のHDR（SDRとして処理）", False),
    "duplicate": ("重複候補として統合", False),
    "blank_page": ("白紙ページ", False),
    "no_paper": ("紙が写っていない（何も置いていない場面）", False),
    "receipt_hand": ("手で押さえたまま（よいフレームがない）", False),
    "receipt_blur": ("ぶれている（よいフレームがない）", False),
    "low_resolution": ("解像度が200dpi相当に届かない（電子帳簿保存法の目安）", True),
    "receipt_no_date": ("取引日を読み取れない", True),
    "receipt_no_total": ("金額を読み取れない", True),
    "receipt_no_payee": ("取引先を読み取れない", True),
    "receipt_regno_suspect": ("登録番号の検査用の数字が合わない（読み誤りの疑い）", True),
    "receipt_total_disagree": ("金額の読み直しで結果が分かれた", True),
    "receipt_date_disagree": ("取引日の読み直しで結果が分かれた", True),
    "receipt_amount_mismatch": ("税率ごとの対象額の合計が金額と合わない（読み誤りの疑い）", True),
}


def label(code: str) -> str:
    return WARNINGS.get(code, (code, True))[0]


def is_blocking(code: str) -> bool:
    return WARNINGS.get(code, (code, True))[1]


def _noop(stage: str, frac: float, msg: str) -> None:
    pass


# ---------------------------------------------------------------- 動画の追加
def add_videos(project: Project, paths: list[Path], copy: bool = True, allow_hdr: bool = False,
               progress: ProgressFn = _noop, names: list[str] | None = None) -> list[dict[str, Any]]:
    """動画を追加する。プロジェクトの videos/ 内に置かれた（アップロード済みの）ファイルは移動だけ行う."""
    added = []
    for n_src, src in enumerate(paths):
        src = Path(src).resolve()
        original_name = names[n_src] if names else src.name
        inside = project.path("videos").resolve() in src.parents
        if not src.exists():
            raise ProjectError(f"ファイルがありません: {src}")
        info = probe(src)
        if info["hdr"] and not allow_hdr:
            raise VideoError(
                f"PQ方式のHDR動画です（{info['color_trc']}）。この版はPQの色変換を検証していません。"
                "スマートフォンのHDR撮影をオフにして撮り直すか、--allow-hdr で警告付きで処理してください。"
            )
        progress("import", 0.0, f"ハッシュ計算中: {src.name}")
        digest = sha256_file(src)
        for v in project.data["videos"]:
            if v["sha256"] == digest:
                if inside:
                    src.unlink(missing_ok=True)
                raise ProjectError(f"同じ動画が既に追加されています: {original_name}（{v['id']}）")
        vid = project.new_id("video")
        if inside:
            dst = project.path("videos", f"{vid}{src.suffix.lower()}")
            os.replace(src, dst)
            path = project.rel(dst)
            copy = True
        elif copy:
            size = src.stat().st_size
            free = shutil.disk_usage(project.root).free
            # コピー＋候補画像・ページ画像などの作業分を見込む
            need = int(size * 1.1) + 2 * 1024**3
            if free < need:
                raise ProjectError(
                    f"空き容量が不足しています（必要の目安 {need / 1024**3:.1f}GB、空き {free / 1024**3:.1f}GB）"
                )
            dst = project.path("videos", f"{vid}{src.suffix.lower()}")
            _copy_with_progress(src, dst, lambda f: progress("import", f, f"コピー中: {src.name}"))
            if sha256_file(dst) != digest:
                dst.unlink(missing_ok=True)
                raise ProjectError(f"コピーした動画が元と一致しません: {src.name}")
            path = project.rel(dst)
        else:
            path = str(src)
        v = {
            "id": vid,
            "original_name": original_name,
            "path": path,
            "copied": copy,
            "sha256": digest,
            "size": project.abs(path).stat().st_size,
            **{k: info[k] for k in ("container", "codec", "width", "height", "pix_fmt", "time_base",
                                    "avg_rate", "duration_sec", "rotation", "color_trc",
                                    "color_primaries", "colorspace", "hdr", "hdr_kind", "high_bit_depth")},
            "warnings": [w for w in (["hdr_pq"] if info["hdr"] else []) + (["hdr_hlg"] if info["hdr_kind"] == "hlg" else [])],
            "analysis": None,
        }
        project.data["videos"].append(v)
        project.save()
        added.append(v)
    return added


def _copy_with_progress(src: Path, dst: Path, cb: Callable[[float], None]) -> None:
    total = max(1, src.stat().st_size)
    tmp = dst.with_name(dst.name + ".tmp")
    done = 0
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        while chunk := fi.read(8 << 20):
            fo.write(chunk)
            done += len(chunk)
            cb(done / total)
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp, dst)


def check_video_source(project: Project, v: dict[str, Any]) -> Path:
    """元動画が存在し、内容が一致するか確かめる（IN-07）."""
    p = project.abs(v["path"])
    if not p.exists():
        raise ProjectError(f"動画が見つかりません: {p}（vbs relink で再指定してください）")
    if p.stat().st_size != v["size"]:
        raise ProjectError(f"動画の内容が変わっています: {p}")
    return p


# ---------------------------------------------------------------- 抽出
def analyze(project: Project, progress: ProgressFn = _noop, force: bool = False) -> None:
    for v in project.data["videos"]:
        if v.get("analysis") and not force:
            clear_checkpoints(project, v["id"])  # 確定後に消し損ねた途中状態
            continue
        if v.get("analysis") and force:
            if any(s["manual"] for s in project.data["segments"] if s["video_id"] == v["id"]):
                raise ProjectError(f"{v['id']} には手動の判断があります。再解析すると失われるため中止しました。")
            project.data["segments"] = [s for s in project.data["segments"] if s["video_id"] != v["id"]]
            clear_checkpoints(project, v["id"])
        check_video_source(project, v)
        ensure_free_space(project)
        t0 = time.perf_counter()
        res = extract_video(project, v, progress=lambda f, m: progress("extract", f, m))
        v["analysis"]["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        project.save()
        clear_checkpoints(project, v["id"])  # manifest に確定してから途中状態を消す
        progress("extract", 1.0, f"{v['original_name']}: 区間 {res['segments']}、候補 {res['frames']} 枚")


# ---------------------------------------------------------------- 分割・補正
def _revision(project: Project, seg: dict[str, Any]) -> str:
    st = project.settings
    manual = seg.get("manual_geometry")
    if manual and manual.get("frame_id") != seg["chosen"]:
        manual = None
    key = {
        "frame": seg["chosen"],
        "layout": st["layout"],
        "direction": st["direction"],
        "enhance": st["enhance"],
        "dewarp": page_option(project, seg, "dewarp"),
        "margins": page_option(project, seg, "margins"),
        "page_height_mm": st["page_height_mm"],
        "geometry": manual,
        "rotation_rev": seg.get("rotation_rev", 0),
        "split_version": SPLIT_VERSION,
    }
    if st.get("document") == "receipt":  # 本の既存プロジェクトの revision は変えない
        key["receipt_width_mm"] = st.get("receipt_width_mm")
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


PAGE_OPTION_DEFAULTS = {"dewarp": "auto", "margins": "content_box"}


def page_option(project: Project, seg: dict[str, Any], key: str) -> str:
    """見開きごとの補正の指定（なければプロジェクトの設定）."""
    return (seg.get("page_options") or {}).get(key) or project.settings.get(key, PAGE_OPTION_DEFAULTS[key])


def is_receipt(project: Project) -> bool:
    return project.settings.get("document") == "receipt"


RECEIPT_MIN_DPI = 200  # 電子帳簿保存法のスキャナ保存で示されている解像度の目安


def _page_dpi(project: Project, height_px: int, width_px: int | None = None) -> int:
    """PDF上の寸法を決める解像度。本は紙の高さ、レシートは幅（58mm・80mmなど）から求める."""
    if is_receipt(project) and width_px:
        mm = float(project.settings.get("receipt_width_mm") or 80.0)
        return max(72, int(round(width_px / (mm / 25.4))))
    mm = float(project.settings["page_height_mm"])
    return max(72, int(round(height_px / (mm / 25.4))))


def process_pages(project: Project, progress: ProgressFn = _noop) -> int:
    """採用区間のページ画像を作る。revisionが変わった区間だけ処理する."""
    pages = project.data["pages"]
    targets = [s for s in project.data["segments"] if s["include"] and s["chosen"]]
    done = 0
    for n, seg in enumerate(targets):
        rev = _revision(project, seg)
        existing = [p for p in pages.values() if p["segment_id"] == seg["id"] and p.get("active", True)]
        if existing and all(p["revision"] == rev and project.abs(p["image"]).exists() for p in existing):
            continue
        frame = project.data["frames"][seg["chosen"]]
        img = imread(project.abs(frame["path"]))
        manual = seg.get("manual_geometry")
        if manual and manual.get("frame_id") != seg["chosen"]:
            manual = None
        geo, out = split_spread(img, project.settings["layout"], project.settings["direction"],
                                project.settings["enhance"], geometry=manual,
                                dewarp_mode=page_option(project, seg, "dewarp"),
                                margins=page_option(project, seg, "margins"))
        seg["geometry"] = {"frame_id": seg["chosen"], **geo}
        sides = []
        for side, page_img, warnings, skew in out:
            pid = f"{seg['id']}-{side}"
            sides.append(pid)
            path = project.path("pages", f"{pid}.jpg")
            dpi = _page_dpi(project, page_img.shape[0], page_img.shape[1]) if page_img.size else 72
            if is_receipt(project) and page_img.size and dpi < RECEIPT_MIN_DPI:
                warnings = warnings + ["low_resolution"]
            pitch = None
            if page_img.size:
                imwrite(path, page_img, quality=92, dpi=dpi)
                g = page_img if page_img.ndim == 2 else cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
                pitch = estimate_line_pitch(g)
                tw = 240
                thumb = cv2.resize(page_img, (tw, max(2, int(round(page_img.shape[0] * tw / page_img.shape[1])))),
                                   interpolation=cv2.INTER_AREA)
                imwrite(project.path("thumbs", f"{pid}.jpg"), thumb, quality=80)
            old = pages.get(pid, {})
            pages[pid] = {
                "id": pid,
                "segment_id": seg["id"],
                "side": side,
                "frame_id": seg["chosen"],
                "image": project.rel(path),
                "thumb": project.rel(project.path("thumbs", f"{pid}.jpg")),
                "width": int(page_img.shape[1]) if page_img.size else 0,
                "height": int(page_img.shape[0]) if page_img.size else 0,
                "dpi": dpi,
                "skew_deg": round(skew, 2),
                "line_pitch_px": pitch,
                "revision": rev,
                "include": old.get("include", True),
                "active": True,
                "warnings": warnings,
                # 内容が変わったら確認済みの印は引き継がない
                "acknowledged": old.get("acknowledged", []) if old.get("revision") == rev else [],
                "image_only_ok": old.get("image_only_ok", False) if old.get("revision") == rev else False,
                "ocr": old.get("ocr") if old.get("revision") == rev else None,
                "printed_number": old.get("printed_number"),
                "updated": now_iso(),
            }
        # レイアウト変更などで不要になった側は非活性にする（削除しない）
        for p in existing:
            if p["id"] not in sides:
                p["active"] = False
        done += 1
        progress("split", (n + 1) / len(targets), f"ページ画像を作成 {n + 1}/{len(targets)}")
        project.save()
        faults.point("split.segment_done")
    # 除外された区間のページは非活性
    included = {s["id"] for s in targets}
    for p in pages.values():
        if p["segment_id"] not in included:
            p["active"] = False
    project.save()
    return done


# ---------------------------------------------------------------- OCR
def active_pages(project: Project) -> list[dict[str, Any]]:
    return [p for p in project.data["pages"].values() if p.get("active", True) and p["include"]]


def ocr_params(project: Project, p: dict[str, Any]) -> dict[str, Any]:
    st = project.settings
    mode = st.get("ocr_scale", "auto")
    scale = auto_scale(p.get("line_pitch_px")) if mode == "auto" else float(mode)
    return {"lang": st["ocr_lang"], "psm": int(st.get("ocr_psm", 6)), "scale": round(scale, 2)}


def ocr_key(project: Project, p: dict[str, Any]) -> str:
    """ページ内容とOCR設定の組。どちらかが変わればOCRをやり直す（PDF-04）."""
    key = {"revision": p["revision"], "ocr_version": OCR_VERSION, **ocr_params(project, p)}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


OCR_MAX_ATTEMPTS = 2  # 失敗したページを自動で試し直す上限（実行をまたいで数える）


def ocr_current(project: Project, p: dict[str, Any]) -> bool:
    """このページのOCRが今の内容・設定で済んでいるか（失敗は上限回数まで試し直す）."""
    o = p.get("ocr")
    if not o or o.get("key") != ocr_key(project, p):
        return False
    if o.get("error"):
        return int(o.get("attempts", 1)) >= OCR_MAX_ATTEMPTS
    return bool(o.get("pdf")) and project.abs(o["pdf"]).exists()


def _ocr_sidecar(project: Project, pid: str) -> Path:
    return project.path("ocr", f"{pid}.result.json")


def run_ocr(project: Project, progress: ProgressFn = _noop, workers: int | None = None) -> int:
    """必要なページだけOCRする.

    - 1ページ終わるごとに結果ファイル（ocr/<ページ>.result.json）を書き、manifest も保存する。
      強制終了しても、書き終えたページは次回に取り込んで再実行しない
    - 同時に走らせる数（ocr_workers）と待ち行列の長さ（ocr_queue）に上限を設ける
    - 1ページの失敗・時間切れはそのページに記録して他を続ける。書き込み失敗は全体を止める
    """
    from concurrent.futures import FIRST_COMPLETED, wait

    from vbs.manifest import atomic_write_json

    st = project.settings
    todo = [p for p in active_pages(project) if not ocr_current(project, p)]
    # 前回の実行で終わっていたページを取り込む
    adopted = 0
    for p in list(todo):
        sc = _ocr_sidecar(project, p["id"])
        if not sc.exists():
            continue
        try:
            res = json.loads(sc.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if res.get("key") != ocr_key(project, p):
            continue
        if not res.get("error") and not (res.get("pdf") and project.abs(res["pdf"]).exists()):
            continue
        p["ocr"] = res
        if ocr_current(project, p):
            todo.remove(p)
            adopted += 1
    if adopted:
        faults.event("resume", stage="ocr", adopted=adopted)
        project.save()
    if not todo:
        return 0
    project.data["processor_versions"]["tesseract"] = tesseract_version()
    workers = int(workers or st.get("ocr_workers") or max(1, min(8, (os.cpu_count() or 2) // 2)))
    queue_max = max(workers, int(st.get("ocr_queue") or workers * 2))
    timeout = float(st.get("ocr_timeout_sec") or 600)

    def job(p: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        t0 = time.perf_counter()
        base = project.path("ocr", p["id"])
        prm = ocr_params(project, p)
        key = ocr_key(project, p)
        old = p.get("ocr") or {}
        attempts = int(old.get("attempts", 1)) + 1 if old.get("key") == key and old.get("error") else 1
        common = {"key": key, "revision": p["revision"], "attempts": attempts, **prm}
        try:
            r = run_page_best(project.abs(p["image"]), base, prm["lang"], p["dpi"], psm=prm["psm"],
                              scale=prm["scale"], timeout=timeout)
            res = {**common, "pdf": project.rel(r["pdf"]), "txt": project.rel(r["txt"]),
                   "tsv": project.rel(r["tsv"]), "chars": r["chars"], "mean_conf": r["mean_conf"],
                   "sure_chars": r["sure_chars"], "used_scale": r["scale"], "used_psm": r["psm"],
                   "tried": r["tried"], "text_min_conf": r.get("text_min_conf", 0.0), "error": None}
        except OcrError as e:  # このページだけの失敗。ページは落とさず、文字層なしとして記録する
            res = {**common, "pdf": None, "txt": None, "tsv": None, "chars": 0,
                   "mean_conf": None, "error": str(e)}
        res["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        atomic_write_json(_ocr_sidecar(project, p["id"]), res)  # 完了の記録は成果物を置いた後
        faults.point("ocr.sidecar_written")
        return p["id"], res

    done = failed = 0
    pending = iter(todo)
    running: set = set()
    peak = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        try:
            while True:
                while len(running) < queue_max:
                    nxt = next(pending, None)
                    if nxt is None:
                        break
                    running.add(ex.submit(job, nxt))
                if not running:
                    break
                peak = max(peak, len(running))
                finished, running = wait(running, return_when=FIRST_COMPLETED)
                for f in finished:
                    pid, res = f.result()  # 書き込み失敗などはここで送出され、全体を止める
                    project.data["pages"][pid]["ocr"] = res
                    done += 1
                    failed += bool(res.get("error"))
                    project.save()
                    faults.event("ocr_done", page=pid, error=res.get("error"), queue=len(running),
                                 elapsed=res["elapsed_sec"])
                    faults.point("ocr.page_done")
                    progress("ocr", done / len(todo), f"OCR {done}/{len(todo)}（失敗 {failed}）")
        except BaseException:
            for f in running:
                f.cancel()
            raise
    project.data.setdefault("stats", {})["ocr_queue_peak"] = peak
    project.save()
    return done


def ocr_warnings(p: dict[str, Any]) -> list[str]:
    o = p.get("ocr")
    if not o:
        return []
    if o.get("error") or not o.get("pdf"):
        return ["ocr_failed"]
    if "blank_page" in p["warnings"]:
        return []
    if o["chars"] < 5:
        return ["ocr_no_text"]
    if o["mean_conf"] is not None and o["mean_conf"] < 60:
        return ["ocr_low_conf"]
    return []


# ---------------------------------------------------------------- レシート
def read_receipts(project: Project, progress: ProgressFn = _noop) -> int:
    """OCRの結果からレシートの項目を読み取る。OCRが変わったページだけ読み直す（手修正は残す）."""
    from vbs import receipt

    targets = [p for p in active_pages(project) if (p.get("ocr") or {}).get("txt")]
    done = 0
    for n, p in enumerate(targets):
        o = p["ocr"]
        rec = p.get("receipt") or {}
        if (rec.get("auto") or {}).get("ocr_key") == o.get("key"):
            continue
        try:
            text = project.abs(o["txt"]).read_text(encoding="utf-8")
        except OSError:
            continue
        res = receipt.parse(text)
        if o.get("tsv"):
            # 金額の行は、数字だけで読み直した結果を使う（日本語のモデルは細い数字を読み違えやすい）
            import subprocess

            from vbs.receipt_ocr import augmented_text

            try:
                aug, notes = augmented_text(project.abs(p["image"]), project.abs(o["tsv"]),
                                            float(o.get("used_scale") or 1.0))
            except (OSError, subprocess.SubprocessError):
                aug, notes = None, {}
            if aug is not None:
                res2 = receipt.parse(aug)
                first = res["fields"].get("total")
                for k in ("total", "base10", "base8", "tax"):
                    res["fields"][k] = res2["fields"][k]
                    res["evidence"][k] = res2["evidence"][k]
                res["checks"] = receipt.checks(res["fields"])
                res["reread"] = notes
                if first is not None and res["fields"]["total"] is not None and first != res["fields"]["total"]:
                    res["total_disagree"] = [first, res["fields"]["total"]]
            # 日付は、その行だけを拡大して読み直し、食い違えば確認へ回す
            if res["fields"].get("date") and res["evidence"].get("date"):
                from vbs.receipt_ocr import reread_line

                try:
                    again = reread_line(project.abs(p["image"]), project.abs(o["tsv"]),
                                        float(o.get("used_scale") or 1.0), res["evidence"]["date"])
                except (OSError, subprocess.SubprocessError):
                    again = None
                d2 = receipt.find_date([receipt.normalize(again)])[0] if again else None
                res["date_reread"] = again
                if d2 != res["fields"]["date"]:
                    res["date_disagree"] = [res["fields"]["date"], d2]
        rec["auto"] = {**res, "ocr_key": o.get("key"), "time": now_iso()}
        p["receipt"] = rec
        done += 1
        progress("ocr", (n + 1) / max(1, len(targets)), f"レシートの項目を読み取り {n + 1}/{len(targets)}")
    project.save()
    return done


def receipt_fields(project: Project, p: dict[str, Any]) -> dict[str, Any]:
    from vbs import receipt

    return receipt.merged(p.get("receipt"))


def receipt_warnings(project: Project, p: dict[str, Any]) -> list[str]:
    if not is_receipt(project) or not (p.get("receipt") or {}).get("auto"):
        return []
    from vbs import receipt

    codes = receipt.missing(receipt_fields(project, p))
    rec = p["receipt"]
    if rec["auto"].get("total_disagree") and "total" not in (rec.get("manual") or {}):
        codes.append("receipt_total_disagree")
    if rec["auto"].get("date_disagree") and "date" not in (rec.get("manual") or {}):
        codes.append("receipt_date_disagree")
    return codes


def set_receipt_fields(project: Project, page_id: str, values: dict[str, Any], source: str = "manual") -> None:
    """レシートの項目を手で直す。空にした項目は自動の値に戻す."""
    from vbs import receipt

    p = project.data["pages"][page_id]
    rec = p.setdefault("receipt", {})
    manual = dict(rec.get("manual") or {})
    for k, v in values.items():
        if k not in receipt.FIELDS:
            raise ProjectError(f"変更できない項目です: {k}")
        val = receipt.coerce(k, v)
        auto = ((rec.get("auto") or {}).get("fields") or {}).get(k)
        if val is None or val == auto:
            manual.pop(k, None)
        else:
            manual[k] = val
    rec["manual"] = manual
    rec["manual_source"] = source if manual else None
    rec["manual_time"] = now_iso()
    project.save()


def sort_receipts(project: Project, by: str) -> None:
    """レシートを取引日・取引先の順に並べ替える（いま有効な値＝手修正やAIの修正を反映した値で）.

    読み取れていない項目のレシートは最後に、撮影順のまま並べる。
    """
    if by not in ("date", "payee", "payee_date", "shot"):
        raise ProjectError(f"並べ方が不明です: {by}")
    if by == "shot":
        project.data["receipt_sort"] = "shot"
        reset_order(project)
        return
    order = natural_order(project)
    pages = project.data["pages"]
    pos = {pid: i for i, pid in enumerate(order)}

    def key(pid: str):
        f = receipt_fields(project, pages[pid])
        d, py = f.get("date"), (f.get("payee") or "").replace(" ", "")
        if by == "date":
            return (d is None, d or "", pos[pid])
        return (not py, py, d is None, d or "", pos[pid])

    set_order(project, sorted(order, key=key))
    project.data["receipt_sort"] = by


# ---------------------------------------------------------------- 順序と確認
def natural_order(project: Project) -> list[str]:
    vorder = {v["id"]: i for i, v in enumerate(project.data["videos"])}
    segs = sorted(
        (s for s in project.data["segments"] if s["include"] and s["chosen"]),
        key=lambda s: (vorder.get(s["video_id"], 1e9), s["start_sec"]),
    )
    pages = project.data["pages"]
    order = []
    rank = {"L": 0, "R": 1, "S": 0}
    if project.settings["direction"] == "rtl":
        rank = {"R": 0, "L": 1, "S": 0}
    for s in segs:
        side_ids = [p for p in pages.values()
                    if p["segment_id"] == s["id"] and p.get("active", True) and p["include"]]
        order += [p["id"] for p in sorted(side_ids, key=lambda p: rank[p["side"]])]
    return order


def update_order(project: Project) -> list[str]:
    nat = natural_order(project)
    if project.data.get("order_manual"):
        keep = [pid for pid in project.data["order"] if pid in set(nat)]
        new = [pid for pid in nat if pid not in set(keep)]
        project.data["order"] = keep + new
    else:
        project.data["order"] = nat
    return project.data["order"]


def review_items(project: Project) -> list[dict[str, Any]]:
    """出力前に確認が必要な項目（理由付き）の一覧."""
    items = []
    for v in project.data["videos"]:
        ack = set(v.get("acknowledged", []))
        codes = [w for w in v.get("warnings", []) if is_blocking(w) and w not in ack]
        if codes:
            items.append({"kind": "video", "id": v["id"], "codes": codes, "labels": [label(c) for c in codes]})
    for s in project.data["segments"]:
        if s.get("duplicate_of"):
            continue  # 統合済みの重複は区間タブで確認できる。出力を止める理由にはしない
        if "no_paper" in s["warnings"]:
            continue  # 何も置いていない場面（レシート）
        ack = set(s.get("acknowledged", []))
        codes = [w for w in s["warnings"] if is_blocking(w) and w not in ack]
        # 採用区間のぼけ等はページ側でも見るが、区間側の理由も残す
        if codes:
            items.append({"kind": "segment", "id": s["id"], "codes": codes,
                          "labels": [label(c) for c in codes], "time": s["start_sec"],
                          "video_id": s["video_id"]})
    for pid in project.data["order"]:
        p = project.data["pages"][pid]
        ack = set(p.get("acknowledged", []))
        codes = [w for w in p["warnings"] + ocr_warnings(p) + receipt_warnings(project, p)
                 if is_blocking(w) and w not in ack]
        # 区間単位の警告はページに重複表示しない
        codes = [c for c in dict.fromkeys(codes)]
        if codes:
            items.append({"kind": "page", "id": pid, "codes": codes, "labels": [label(c) for c in codes],
                          "segment_id": p["segment_id"]})
    exp = project.settings.get("expected_pages")
    if exp and len(project.data["order"]) != int(exp) and "page_count" not in project.data.get("acknowledged", []):
        items.append({"kind": "project", "id": "page_count", "codes": ["page_count_mismatch"],
                      "labels": [f"ページ数が予定と違う（{len(project.data['order'])} / 予定 {exp}）"]})
    return items


# ---------------------------------------------------------------- 出力
def export_pdf(project: Project, out: Path | None = None, force: bool = False,
               page_ids: list[str] | None = None) -> dict[str, Any]:
    order = page_ids or project.data["order"]
    if not order:
        raise ProjectError("出力できるページがありません。")
    items = review_items(project)
    if items and not force:
        raise ProjectError(f"要確認が {len(items)} 件あります。確認するか --force で出力してください。")
    entries = []
    image_only = []
    for pid in order:
        p = project.data["pages"][pid]
        o = p.get("ocr") or {}
        failed = bool(o.get("error")) and o.get("key") == ocr_key(project, p)
        if not failed and not ocr_current(project, p):
            raise ProjectError(f"{pid} のOCRが古い状態です。処理を再実行してください。")
        if o.get("pdf") and not failed:
            entries.append((pid, project.abs(o["pdf"]), project.abs(p["image"]), p["dpi"]))
        elif p.get("image_only_ok") or "ocr_failed" in p.get("acknowledged", []) or force:
            # OCRに失敗したページも落とさない。画像だけ（文字層なし）で入れ、記録に残す
            entries.append((pid, None, project.abs(p["image"]), p["dpi"]))
            image_only.append(pid)
        else:
            raise ProjectError(f"{pid} はOCRに失敗しています。画像だけで残すか、ページを修正してください。")
    ensure_free_space(project)
    name = project.data.get("title") or ("receipts" if is_receipt(project) else "book")
    out = Path(out or project.path("output", f"{_safe(name)}.pdf"))
    # 一時ファイルに作って検証してから置き換える。失敗しても既存の正常なPDFは残る
    staged = out.with_name(out.stem + ".new.pdf")
    n = build_pdf(entries, staged, title=project.data.get("title"))
    from vbs.verify import load_terms, verify_pdf, write_report

    terms_path = project.settings.get("verify_terms")
    ver = verify_pdf(staged, list(order), load_terms(project.abs(terms_path)) if terms_path else None)
    if not ver["page_count_ok"]:
        raise ProjectError(f"書き出したPDFのページ数が合いません（{ver['pages']}）。旧版は残しています: {staged}")
    faults.point("export.before_replace")
    try:
        os.replace(staged, out)
    except PermissionError as e:
        raise ProjectError(f"{out} を置き換えられません（他のアプリで開かれている可能性）。旧版はそのまま残っています。"
                           f"新しいPDFは {staged} にあります。") from e
    ver["pdf"] = str(out.resolve())
    ver_path = write_report(ver, out)
    receipt_files = None
    if is_receipt(project):
        from vbs.receipt_export import write_package

        receipt_files = write_package(project, list(order), Path(out).resolve())
    rec = {"time": now_iso(), "path": str(Path(out).resolve()), "pages": n, "forced": bool(items) and force,
           "image_only_pages": image_only,
           "unresolved": len(items), "revisions": {pid: project.data["pages"][pid]["revision"] for pid in order},
           "order": list(order), "sha256": ver["sha256"], "verify": str(ver_path.resolve()),
           "verify_summary": {"pages": ver["pages"], "page_count_ok": ver["page_count_ok"],
                              "terms": ver.get("terms_summary")},
           "receipt_files": receipt_files}
    project.data["exports"].append(rec)
    project.save()
    return rec


def _safe(name: str) -> str:
    bad = '<>:"/\\|?*'
    return "".join("_" if c in bad or ord(c) < 32 else c for c in name).strip() or "book"


# ---------------------------------------------------------------- 一括実行
def run_all(project: Project, progress: ProgressFn = _noop, export: bool | None = None,
            force: bool = False, ocr: bool = True, workers: int | None = None) -> dict[str, Any]:
    cleanup_stale(project)
    analyze(project, progress)
    process_pages(project, progress)
    if ocr:
        ensure_free_space(project)
        run_ocr(project, progress, workers=workers)
        if is_receipt(project):
            read_receipts(project, progress)
    update_order(project)
    project.save()
    items = review_items(project)
    result: dict[str, Any] = {"pages": len(project.data["order"]), "review": items, "export": None}
    auto = project.settings.get("auto_export", True) if export is None else export
    if ocr and auto and (not items or force):
        result["export"] = export_pdf(project, force=force)
    return result


def summary(project: Project) -> dict[str, Any]:
    segs = project.data["segments"]
    return {
        "videos": len(project.data["videos"]),
        "segments": len(segs),
        "included_segments": sum(1 for s in segs if s["include"]),
        "pages": len(project.data["order"]),
        "ocr_done": sum(1 for pid in project.data["order"]
                        if (project.data["pages"][pid].get("ocr") or {}).get("pdf")),
        "review": len(review_items(project)),
        "version": __version__,
    }


# ---------------------------------------------------------------- 手動の追加・修正
def preview_at(project: Project, video_id: str, t: float, width: int = 960):
    """動画の指定時刻に最も近いフレームを (時刻, pts, BGR画像) で返す."""
    from vbs.video import iter_frames

    v = project.video(video_id)
    path = check_video_source(project, v)
    best = None
    for ft, pts, img in iter_frames(path, width=width, fmt="bgr24", start_sec=max(0.0, t - 0.6), end_sec=t + 0.6):
        if best is None or abs(ft - t) < abs(best[0] - t):
            best = (ft, pts, img)
        elif ft > t:
            break  # 目標時刻を過ぎて離れ始めた
    if best is None:
        raise ProjectError(f"{t:.2f}秒付近のフレームを取得できません。")
    return best


def add_frame_at(project: Project, video_id: str, t: float, segment_id: str | None = None) -> dict[str, Any]:
    """動画の任意時刻のフレームを候補に加える（EXT-07）.

    segment_id があればその区間の候補に追加して採用する。なければ新しい区間を作る。
    """
    import cv2 as _cv2

    from vbs.video import grab_frames

    ft, pts, _ = preview_at(project, video_id, t, width=64)
    v = project.video(video_id)
    rgb = grab_frames(check_video_source(project, v), [pts]).get(pts)
    if rgb is None:
        raise ProjectError(f"{ft:.2f}秒のフレームを元解像度で取り出せません。")
    bgr = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2BGR)
    fid = _save_frame(project, bgr, {"video_id": video_id, "pts": pts, "time_sec": ft, "source": "manual"})
    if segment_id:
        seg = project.segment(segment_id)
        seg["candidates"].append(fid)
    else:
        seg = _new_segment(project, video_id, ft, ft)
    seg["chosen"] = fid
    seg["include"] = True
    seg["manual"] = True
    project.save()
    return seg


def add_photo(project: Project, image_path: Path, after_segment_id: str | None,
              original_name: str | None = None) -> dict[str, Any]:
    """写真（JPEG/PNG）で欠落ページを補う。指定区間の直後に並ぶ（P0: 欠落ページの補充）."""
    bgr = imread(image_path)
    rotation = None
    if project.settings.get("content_rotation", "auto") == "auto":
        from vbs.orient import MIN_CONF, detect_osd

        r = detect_osd(bgr)
        if r and r[1] >= MIN_CONF and r[0]:
            import numpy as _np

            bgr = _np.ascontiguousarray(_np.rot90(bgr, r[0]))
        rotation = {"k": r[0] if r and r[1] >= MIN_CONF else 0, "conf": r[1] if r else None}
    if after_segment_id:
        prev = project.segment(after_segment_id)
        video_id, t = prev["video_id"], prev["end_sec"] + 1e-3
    elif project.data["videos"]:
        video_id, t = project.data["videos"][-1]["id"], 1e9
    else:
        # 動画がなければ追加した順に並べる
        photos = [s["start_sec"] for s in project.data["segments"] if s.get("video_id") is None]
        video_id, t = None, (max(photos) + 1.0 if photos else 0.0)
    fid = _save_frame(project, bgr, {"video_id": None, "pts": None, "time_sec": None, "source": "photo",
                                     "original_name": original_name or Path(image_path).name,
                                     "content_rotation": rotation})
    seg = _new_segment(project, video_id, t, t)
    seg["candidates"] = [fid]
    seg["chosen"] = fid
    seg["include"] = True
    seg["manual"] = True
    seg["source"] = "photo"
    project.save()
    return seg


def _save_frame(project: Project, bgr, meta: dict[str, Any]) -> str:
    import cv2 as _cv2

    fid = project.new_id("frame")
    path = project.path("frames", f"{fid}.jpg")
    imwrite(path, bgr, quality=95)
    thumb = _cv2.resize(bgr, (320, max(2, int(round(bgr.shape[0] * 320 / bgr.shape[1])))),
                        interpolation=_cv2.INTER_AREA)
    tpath = project.path("thumbs", f"{fid}.jpg")
    imwrite(tpath, thumb, quality=80)
    project.data["frames"][fid] = {
        "id": fid, "path": project.rel(path), "thumb": project.rel(tpath),
        "width": int(bgr.shape[1]), "height": int(bgr.shape[0]), "rotation_applied": 0,
        "scores": {}, "warnings": [], **meta,
    }
    return fid


def _new_segment(project: Project, video_id: str | None, start: float, end: float) -> dict[str, Any]:
    seg = {
        "id": project.new_id("segment"), "video_id": video_id, "start_sec": start, "end_sec": end,
        "duration_sec": round(end - start, 3), "candidates": [], "chosen": None, "include": True,
        "manual": True, "duplicate_of": None, "warnings": [], "flags": ["manual_add"],
    }
    project.data["segments"].append(seg)
    project.data["segments"].sort(key=lambda s: (s["video_id"] or "", s["start_sec"]))
    return seg


def set_geometry(project: Project, segment_id: str, bbox: list[int] | None = None,
                 gutter_ratio: float | None = None) -> None:
    """紙面範囲・綴じ目を手で指定する。採用フレームが変われば無効になる（仕様13章）."""
    seg = project.segment(segment_id)
    if not seg["chosen"]:
        raise ProjectError("採用フレームがありません。")
    frame = project.data["frames"][seg["chosen"]]
    cur = seg.get("manual_geometry") if (seg.get("manual_geometry") or {}).get("frame_id") == seg["chosen"] else None
    auto = seg.get("geometry") if (seg.get("geometry") or {}).get("frame_id") == seg["chosen"] else None
    geo = {"frame_id": seg["chosen"]}
    base = cur or auto or {}
    geo["bbox"] = [int(v) for v in (bbox or base.get("bbox") or [0, 0, frame["width"], frame["height"]])]
    if project.settings["layout"] == "spread":
        if gutter_ratio is not None:
            geo["gutter_x"] = int(round(gutter_ratio * frame["width"]))
        elif "gutter_x" in base:
            geo["gutter_x"] = int(base["gutter_x"])
    x0, _, x1, _ = geo["bbox"]
    if "gutter_x" in geo and not (x0 < geo["gutter_x"] < x1):
        raise ProjectError("綴じ目が紙面範囲の外にあります。")
    seg["manual_geometry"] = geo
    seg["manual"] = True
    project.save()


def clear_geometry(project: Project, segment_id: str) -> None:
    seg = project.segment(segment_id)
    seg.pop("manual_geometry", None)
    project.save()


def set_order(project: Project, order: list[str]) -> None:
    valid = set(natural_order(project))
    if set(order) != valid or len(order) != len(valid):
        raise ProjectError("並び順のページ集合が現在のページと一致しません。再読み込みしてください。")
    project.data["order"] = list(order)
    project.data["order_manual"] = True
    project.save()


def reset_order(project: Project) -> None:
    project.data["order_manual"] = False
    update_order(project)
    project.save()


def move_video(project: Project, video_id: str, delta: int) -> None:
    vids = project.data["videos"]
    i = next(k for k, v in enumerate(vids) if v["id"] == video_id)
    j = max(0, min(len(vids) - 1, i + delta))
    vids.insert(j, vids.pop(i))
    update_order(project)
    project.save()


def rotate_video_frames(project: Project, video_id: str, k: int) -> int:
    """動画の候補フレームをまとめて反時計回りに k×90° 回す（向きの自動判定が外れたとき用）.

    回転後は境界の自動検出・手修正とも無効になり、該当ページは作り直しになる。
    """
    import cv2 as _cv2
    import numpy as _np

    k %= 4
    v = project.video(video_id)
    if k == 0:
        return 0
    n = 0
    for f in project.data["frames"].values():
        if f.get("video_id") != video_id:
            continue
        for key in ("path", "thumb"):
            path = project.abs(f[key])
            img = _np.ascontiguousarray(_np.rot90(imread(path), k))
            imwrite(path, img, quality=95 if key == "path" else 80)
        if k % 2:
            f["width"], f["height"] = f["height"], f["width"]
        f["content_rotation_k"] = (f.get("content_rotation_k", 0) + k) % 4
        n += 1
    for s in project.data["segments"]:
        if s["video_id"] == video_id:
            s.pop("geometry", None)
            s.pop("manual_geometry", None)
            s["rotation_rev"] = s.get("rotation_rev", 0) + 1
    cr = v.setdefault("content_rotation", {"k": 0})
    cr["k"] = (cr.get("k", 0) + k) % 4
    cr["mode"] = "manual"
    v["warnings"] = [w for w in v.get("warnings", []) if w != "orientation_uncertain"]
    project.save()
    return n


# ---------------------------------------------------------------- 後始末と安全確認
def ensure_free_space(project: Project) -> None:
    """空き容量が足りなければ新しい処理を始めない（既存の成果物は消さない）."""
    need = float(project.settings.get("min_free_gb") or 2.0) * 1024**3
    free = shutil.disk_usage(project.root).free
    if free < need:
        raise ProjectError(f"空き容量が不足しています（空き {free / 1024**3:.1f}GB、必要 {need / 1024**3:.1f}GB）。"
                           "処理を止めました。確定済みの成果物は残っています。")


def cleanup_stale(project: Project) -> dict[str, int]:
    """強制終了の跡を片付ける.

    - このプロジェクトの画像を処理中のまま残った Tesseract を終了する（無視して再実行しない）
    - 書きかけの一時ファイルを消す（確定ファイルと途中状態の保存は消さない）
    """
    killed = 0
    try:
        import psutil

        root = str(project.root.resolve()).lower()
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                name = (proc.info["name"] or "").lower()
                cmd = " ".join(proc.info["cmdline"] or []).lower()
            except (psutil.Error, OSError):
                continue
            if name.startswith("tesseract") and root in cmd:
                try:
                    proc.kill()
                    killed += 1
                except psutil.Error:
                    pass
    except ImportError:
        pass
    removed = 0
    for d in ("pages", "ocr", "output", "frames", "thumbs", "jobs"):
        base = project.path(d)
        for f in list(base.glob("*.tmp")) + list(base.glob("*.tmp.*")):
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    (project.root / "project.json.tmp").unlink(missing_ok=True)
    if killed or removed:
        faults.event("cleanup", killed=killed, removed=removed)
    return {"killed": killed, "removed": removed}


# ---------------------------------------------------------------- 確認画面のための表示
def step_frame(project: Project, video_id: str, t: float, step: int, width: int = 960):
    """t のフレームから前後に step フレーム動いたフレームを返す（(時刻, pts, 画像)）."""
    from vbs.video import iter_frames

    v = project.video(video_id)
    path = check_video_source(project, v)
    span = max(1.0, abs(step) / 20.0 + 0.5)
    frames = [(ft, pts) for ft, pts, _ in iter_frames(path, width=16, fmt="gray",
                                                       start_sec=max(0.0, t - span), end_sec=t + span)]
    if not frames:
        raise ProjectError(f"{t:.2f}秒付近のフレームを取得できません。")
    cur = min(range(len(frames)), key=lambda i: abs(frames[i][0] - t))
    target = frames[max(0, min(len(frames) - 1, cur + step))][0]
    return preview_at(project, video_id, target, width=width)


def render_page_view(project: Project, page_id: str, mode: str, width: int = 1400):
    """ページの見え方を返す（BGR）.

    frame: 元の見開き（紙面範囲と分割線を重ねる。このページ側を明るく）
    plain: 分割・切り抜きだけ（傾き・湾曲補正・余白整理・照明補正なし）
    final: 仕上がり（保存済みのページ画像）
    """
    import cv2 as _cv2
    import numpy as _np

    pg = project.data["pages"][page_id]
    if mode == "final":
        img = imread(project.abs(pg["image"]))
    else:
        seg = project.segment(pg["segment_id"])
        frame = imread(project.abs(project.data["frames"][pg["frame_id"]]["path"]))
        geo = seg.get("geometry") if (seg.get("geometry") or {}).get("frame_id") == pg["frame_id"] else None
        manual = seg.get("manual_geometry") if (seg.get("manual_geometry") or {}).get("frame_id") == pg["frame_id"] else None
        g = manual or geo or {}
        if mode == "frame":
            img = frame.copy()
            H, W = img.shape[:2]
            th = max(2, W // 400)
            if g.get("bbox"):
                x0, y0, x1, y1 = g["bbox"]
                gx = g.get("gutter_x")
                shade = _np.zeros_like(img)
                if project.settings["layout"] == "spread" and gx:
                    left = pg["side"] == "L"
                    sx0, sx1 = (x0, gx) if left else (gx, x1)
                else:
                    sx0, sx1 = x0, x1
                mask = _np.ones(img.shape[:2], bool)
                mask[y0:y1, sx0:sx1] = False
                img[mask] = (img[mask] * 0.45).astype(_np.uint8)
                _cv2.rectangle(img, (x0, y0), (x1, y1), (255, 140, 30), th)
                if gx and project.settings["layout"] == "spread":
                    _cv2.line(img, (gx, y0), (gx, y1), (60, 60, 255), th)
                del shade
        else:  # plain
            geometry = {k: g[k] for k in ("bbox", "gutter_x") if k in g} or None
            _, pages = split_spread(frame, project.settings["layout"], project.settings["direction"], "none",
                                    geometry=geometry, dewarp_mode="off", margins="off")
            img = next((p for side, p, _, _ in pages if side == pg["side"]), frame)
    if img.shape[1] > width:
        img = _cv2.resize(img, (width, int(round(img.shape[0] * width / img.shape[1]))),
                          interpolation=_cv2.INTER_AREA)
    return img


def set_page_options(project: Project, segment_id: str, values: dict[str, str]) -> None:
    seg = project.segment(segment_id)
    opts = dict(seg.get("page_options") or {})
    for k, v in values.items():
        if k not in PAGE_OPTION_DEFAULTS:
            raise ProjectError(f"変更できない補正です: {k}")
        if v in (None, "", "default"):
            opts.pop(k, None)
        else:
            opts[k] = v
    seg["page_options"] = opts
    seg["manual"] = True
    project.save()


def pending_pages(project: Project) -> list[str]:
    """変更が反映されていない（作り直しやOCRが必要な）ページ."""
    out = []
    for s in project.data["segments"]:
        if not (s["include"] and s["chosen"]):
            continue
        rev = _revision(project, s)
        sides = [p for p in project.data["pages"].values() if p["segment_id"] == s["id"] and p.get("active", True)]
        if not sides:
            out.append(s["id"])
            continue
        for p in sides:
            if p["include"] and (p["revision"] != rev or not ocr_current(project, p)):
                out.append(p["id"])
    return out
