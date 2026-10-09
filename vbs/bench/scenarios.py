"""中断・再開と局所的な失敗の試験.

どの試験も、同じ入力・同じ設定の「連続実行」を基準にして、強制終了→再開の結果を比べる。
比べるのは区間（時刻・採否）、ページ（ID・順序・採用元・画像・OCRテキスト）、最終PDFの検証結果。
処理数が一致するだけでは合格にしない（欠落・二重登録・余分を個別に調べる）。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from vbs.bench.env import sha256_file
from vbs.bench.run import (compare, copy_project, kill_parent_only_when_tesseract, run_vbs, snapshot, when_count,
                           when_file, when_progress, when_tesseract_after)


def set_settings(project: Path, values: dict[str, Any]) -> None:
    p = Path(project) / "project.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["settings"].update(values)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def new_video_project(video: Path, dst: Path, logdir: Path, settings: dict[str, Any] | None = None) -> None:
    r = run_vbs(["new", str(dst), str(video), "--link", "--allow-hdr"], logdir, f"new-{dst.name}")
    if r["returncode"] != 0:
        raise RuntimeError(f"プロジェクトを作れません: {r['log_tail']}")
    if settings:
        set_settings(dst, settings)


def _ok(cond: bool, msg: str, checks: list[dict]) -> None:
    checks.append({"check": msg, "ok": bool(cond)})


# ---------------------------------------------------------------- 動画の解析・ページ作成の中断
def video_resume(video: Path, out: Path, kill_prefix: str, frac: float, settings: dict[str, Any]) -> dict[str, Any]:
    logs = out / "logs"
    ref, test = out / "ref", out / "test"
    new_video_project(video, ref, logs, settings)
    r_ref = run_vbs(["run", str(ref), "--no-ocr"], logs, "ref", workdir=ref)
    new_video_project(video, test, logs, settings)
    r1 = run_vbs(["run", str(test), "--no-ocr"], logs, "test-1", workdir=test,
                 kill=when_progress(kill_prefix, frac))
    r2 = run_vbs(["run", str(test), "--no-ocr"], logs, "test-2", workdir=test)
    a, b = snapshot(ref), snapshot(test)
    diffs = compare(a, b, parts=("segments", "pages"))
    checks: list[dict] = []
    _ok(r_ref["returncode"] == 0, "連続実行が完了", checks)
    _ok(r1["killed"] is not None, "指定の地点で強制終了できた", checks)
    _ok(r2["returncode"] == 0, "再開して完了", checks)
    _ok(any(e["stage"].startswith("extract") for e in r2["events"]["resume"]) or "ページ" in kill_prefix,
        "途中状態から再開した（最初からやり直していない）", checks)
    _ok(not diffs, "区間・ページが連続実行と一致", checks)
    return {"runs": [r_ref, r1, r2], "diffs": diffs, "checks": checks,
            "segments": len(b["segments"]), "pages": len(b["pages"])}


# ---------------------------------------------------------------- OCR・PDF出力の中断と失敗
def prepare_pages(video: Path, out: Path, settings: dict[str, Any]) -> Path:
    """OCR前までを済ませたプロジェクト（OCR系の試験で複製して使う）."""
    prep = out / "prepared"
    if (prep / "project.json").exists():
        return prep
    logs = out / "logs"
    new_video_project(video, prep, logs, settings)
    r = run_vbs(["run", str(prep), "--no-ocr"], logs, "prepare", workdir=prep)
    if r["returncode"] != 0:
        raise RuntimeError(r["log_tail"])
    return prep


def reference_full(prep: Path, out: Path, workers: int) -> tuple[Path, dict, dict]:
    ref = out / "ref"
    if not (ref / "project.json").exists():
        copy_project(prep, ref)
        r = run_vbs(["run", str(ref), "--force", "--workers", str(workers)], out / "logs", "ref", workdir=ref)
    else:
        r = {"name": "ref", "reused": True, "returncode": 0}
    return ref, r, snapshot(ref)


def ocr_resume(prep: Path, out: Path, workers: int, k: int, mode: str) -> dict[str, Any]:
    """mode: after（k件完了後に強制終了）/ mid（k件完了後、次のページのTesseract実行中に強制終了）/
    sidecar（k件目の結果ファイルを書いた直後、manifest に反映する前にプロセスが落ちる）."""
    logs = out / "logs"
    ref, r_ref, a = reference_full(prep, out, workers)
    test = out / f"test-{mode}"
    copy_project(prep, test)
    fault = kill = None
    orphans_before = 0
    if mode == "mid":
        kill = when_tesseract_after(k)
    elif mode == "orphan":
        kill = kill_parent_only_when_tesseract(k)
    elif mode == "after":
        kill = when_count("ocr_done", k)
    else:
        fault = f"crash:ocr.sidecar_written:{k}"
    r1 = run_vbs(["run", str(test), "--force", "--workers", str(workers)], logs, test.name + "-1", workdir=test,
                 kill=kill, fault=fault)
    done_events = r1["events"]["ocr_done"]
    if mode == "orphan":
        import psutil

        root = str(test.resolve()).lower()
        orphans_before = sum(1 for p in psutil.process_iter(["name", "cmdline"])
                             if (p.info["name"] or "").lower().startswith("tesseract")
                             and root in " ".join(p.info["cmdline"] or []).lower())
    d1 = json.loads((test / "project.json").read_text(encoding="utf-8"))
    in_manifest = sum(1 for p in d1["pages"].values() if (p.get("ocr") or {}).get("pdf"))
    sidecars = len(list((test / "ocr").glob("*.result.json")))
    r2 = run_vbs(["run", str(test), "--force", "--workers", str(workers)], logs, test.name + "-2", workdir=test)
    b = snapshot(test)
    diffs = compare(a, b)
    total = len(b["pages"])
    adopted = sum(e.get("adopted", 0) for e in r2["events"]["resume"] if e["stage"] == "ocr")
    confirmed = in_manifest + adopted
    checks: list[dict] = []
    _ok(r1["killed"] is not None or r1["returncode"] == 99, "指定の地点で止まった", checks)
    _ok(r2["returncode"] == 0, "再開して完了", checks)
    _ok(confirmed >= max(done_events, sidecars),
        f"止まる前に終わっていたOCR（通知 {done_events}件・結果ファイル {sidecars}件）をすべて再利用した"
        f"（manifest {in_manifest}件＋取り込み {adopted}件）", checks)
    if mode == "sidecar":
        _ok(adopted >= 1, f"manifest に未反映の結果を結果ファイルから取り込んだ（{adopted}件）", checks)
    _ok(r2["events"]["ocr_done"] == total - confirmed,
        f"再実行は未完了分だけ（{r2['events']['ocr_done']} = {total} - {confirmed}）", checks)
    cleaned = sum(e.get("killed", 0) for e in r2["events"]["cleanup"])
    if mode == "mid":
        _ok(True, f"残っていたTesseractの後始末: {cleaned} 件（プロセスツリーごと終了したので残らない）", checks)
    if mode == "orphan":
        _ok(orphans_before >= 1, f"本体の終了後もTesseractが残っていた（{orphans_before}件）", checks)
        _ok(cleaned >= 1 or orphans_before == 0, f"再開時に残っていたTesseractを終了した（{cleaned}件）", checks)
    _ok(not diffs, "ページ・OCRテキスト・検索結果が連続実行と一致", checks)
    return {"runs": [r_ref, r1, r2], "diffs": diffs, "checks": checks, "adopted": adopted,
            "in_manifest": in_manifest, "sidecars": sidecars, "done_events": done_events, "pages": total}


def export_resume(prep: Path, out: Path, workers: int) -> dict[str, Any]:
    logs = out / "logs"
    ref, r_ref, a = reference_full(prep, out, workers)
    test = out / "test-export"
    copy_project(ref, test)  # 正常なPDFが既にある状態から始める
    pdf = Path(json.loads((test / "project.json").read_text(encoding="utf-8"))["exports"][-1]["path"])
    pdf = test / "output" / pdf.name  # 複製先のPDF
    shutil.copy2(ref / "output" / pdf.name, pdf)
    old_hash = sha256_file(pdf)
    r1 = run_vbs(["export", str(test), "--force", "-o", str(pdf)], logs, "export-1", workdir=test,
                 fault="slow:export.writing:30", kill=when_file(test / "output", "*.new.pdf.tmp"))
    kept = pdf.exists() and sha256_file(pdf) == old_hash
    from vbs.verify import verify_pdf

    order = [p["id"] for p in a["pages"]]
    old_ok = verify_pdf(pdf, order)["page_count_ok"] if pdf.exists() else False
    r2 = run_vbs(["export", str(test), "--force", "-o", str(pdf)], logs, "export-2", workdir=test)
    new_ver = verify_pdf(pdf, order)
    checks: list[dict] = []
    _ok(r1["killed"] is not None, "一時出力の書き込み中に強制終了できた", checks)
    _ok(kept and old_ok, "旧版のPDFがそのまま残り、正常に開ける", checks)
    _ok(r2["returncode"] == 0 and new_ver["page_count_ok"], "再実行で正常なPDFを作れた", checks)
    leftovers = [p.name for p in (test / "output").glob("*.tmp")]
    _ok(not leftovers, f"一時ファイルが残っていない（{leftovers}）", checks)
    return {"runs": [r_ref, r1, r2], "checks": checks, "old_sha256": old_hash, "new_sha256": sha256_file(pdf)}


def ocr_local_failure(prep: Path, out: Path, workers: int, page_id: str) -> dict[str, Any]:
    """1ページだけOCRを時間切れにする → 他は続行・ページは落とさない・試し直しは上限まで."""
    logs = out / "logs"
    ref, r_ref, a = reference_full(prep, out, workers)
    test = out / "test-fail"
    copy_project(prep, test)
    fault = f"timeout:ocr.page:{page_id}"
    runs = []
    for i in range(3):
        runs.append(run_vbs(["run", str(test), "--force", "--workers", str(workers)], logs, f"fail-{i + 1}",
                            workdir=test, fault=fault))
    d = json.loads((test / "project.json").read_text(encoding="utf-8"))
    o = d["pages"][page_id]["ocr"]
    exp = d["exports"][-1] if d["exports"] else {}
    b = snapshot(test)
    checks: list[dict] = []
    _ok(runs[0]["returncode"] == 0, "1回目は失敗ページがあっても最後まで処理した", checks)
    _ok(runs[0]["events"]["ocr_done"] == len(b["pages"]), "全ページのOCRを1回ずつ試した", checks)
    _ok(page_id in (exp.get("image_only_pages") or []), "失敗ページは画像だけで残り、PDFから消えていない", checks)
    _ok(len(b["pages"]) == len(a["pages"]) and b["export"]["pages"] == len(a["pages"]), "ページ数が変わらない", checks)
    _ok(runs[1]["events"]["ocr_done"] == 1, "2回目は失敗ページだけを試し直した", checks)
    _ok(runs[2]["events"]["ocr_done"] == 0, f"上限（{o.get('attempts')}回）に達した後は試し直さない", checks)
    _ok(bool(o.get("error")), f"失敗理由が記録されている: {o.get('error')}", checks)
    # 障害を取り除けば回復する
    d["pages"][page_id]["ocr"] = None
    (test / "project.json").write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    (test / "ocr" / f"{page_id}.result.json").unlink(missing_ok=True)
    r_fix = run_vbs(["run", str(test), "--force", "--workers", str(workers)], logs, "fail-fixed", workdir=test)
    diffs = compare(a, snapshot(test))
    _ok(r_fix["returncode"] == 0 and not diffs, "障害を除いて再実行すると連続実行と一致", checks)
    return {"runs": [r_ref, *runs, r_fix], "checks": checks, "diffs": diffs}


def write_failure(video: Path, out: Path, settings: dict[str, Any], target: str = "pages") -> dict[str, Any]:
    """ページ画像の書き込みで失敗させる → 成功と誤報せず止まり、再開できる."""
    logs = out / "logs"
    ref, test = out / "ref-w", out / "test-w"
    new_video_project(video, ref, logs, settings)
    r_ref = run_vbs(["run", str(ref), "--no-ocr"], logs, "ref-w", workdir=ref)
    new_video_project(video, test, logs, settings)
    r1 = run_vbs(["run", str(test), "--no-ocr"], logs, "write-1", workdir=test, fault=f"fail:write:{target}")
    loadable = True
    try:
        json.loads((test / "project.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        loadable = False
    r2 = run_vbs(["run", str(test), "--no-ocr"], logs, "write-2", workdir=test)
    diffs = compare(snapshot(ref), snapshot(test), parts=("segments", "pages"))
    checks: list[dict] = []
    _ok(r1["returncode"] not in (0, None), f"書き込み失敗で止まり、成功と報告しない（終了コード {r1['returncode']}）", checks)
    _ok("書き込み・読み込みに失敗" in r1["log_tail"], "失敗理由を表示した", checks)
    _ok(loadable, "project.json は壊れていない", checks)
    _ok(r2["returncode"] == 0 and not diffs, "障害を除いて再開すると連続実行と一致", checks)
    return {"runs": [r_ref, r1, r2], "checks": checks, "diffs": diffs}
