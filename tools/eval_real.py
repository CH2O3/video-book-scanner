"""実写サンプルの固定ページで、処理条件ごとの文字認識を比べる.

    python tools/eval_real.py PROJECT TERMS.json [--gt-dir DIR] [--variants base,nodewarp,noretry]

- 各ページを、採用フレームから分割・補正・OCR までやり直す（プロジェクトは変更しない）
- 重要語の検出率（空白を除いて部分一致）と、信頼度の低い語の文字数（ゴミの目安）を出す
- 信頼度の足切り（0/20/30/40）ごとの影響を TSV から再計算する
- GT_DIR/<page_id>.txt があれば CER（空白・改行を除く）も出す
結果は work/eval/report.json に保存する。
"""

from __future__ import annotations

import argparse

import cv2
import csv
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import edit_distance, norm  # noqa: E402

from vbs import ocr as O  # noqa: E402
from vbs.imgio import imread, imwrite  # noqa: E402
from vbs.manifest import Project  # noqa: E402
from vbs.pipeline import _page_dpi  # noqa: E402
from vbs.split import estimate_line_pitch, split_spread  # noqa: E402
from vbs.textlayer import line_texts, page_min_conf  # noqa: E402

VARIANTS = {
    "base": {"dewarp": "auto", "retry": True},
    "nodewarp": {"dewarp": "off", "retry": True},
    "noretry": {"dewarp": "auto", "retry": False},
}
THRESHOLDS = (0, 20, 30, 40)


def words_from_tsv(tsv: Path) -> list[tuple[str, float]]:
    out = []
    with open(tsv, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            t = (r.get("text") or "").strip()
            if r["level"] == "5" and t:
                out.append((t, float(r["conf"])))
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("terms")
    ap.add_argument("--gt-dir")
    ap.add_argument("--variants", default="base,nodewarp,noretry")
    args = ap.parse_args()
    project = Project.load(Path(args.project))
    terms = {k: v for k, v in json.loads(Path(args.terms).read_text(encoding="utf-8")).items() if not k.startswith("_")}
    st = project.settings
    work = Path("work/eval")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    # ページ画像を条件ごとに作る
    jobs = []
    for vname in args.variants.split(","):
        var = VARIANTS[vname]
        (work / vname).mkdir()
        for pid in terms:
            seg_id, side = pid.rsplit("-", 1)
            seg = project.segment(seg_id)
            img = imread(project.abs(project.data["frames"][seg["chosen"]]["path"]))
            _, pages = split_spread(img, st["layout"], st["direction"], st["enhance"],
                                    dewarp_mode=var["dewarp"], margins=st.get("margins", "content_box"))
            page = next(p for s, p, _, _ in pages if s == side)
            path = work / vname / f"{pid}.jpg"
            dpi = _page_dpi(project, page.shape[0])
            imwrite(path, page, quality=92, dpi=dpi)
            pitch = estimate_line_pitch(cv2.cvtColor(page, cv2.COLOR_BGR2GRAY) if page.ndim == 3 else page)
            jobs.append((vname, pid, path, dpi, round(O.auto_scale(pitch), 2), var["retry"]))  # 本処理と同じ丸め

    def run(j):
        vname, pid, path, dpi, scale, retry = j
        base = path.with_suffix("")
        if retry:
            r = O.run_page_best(path, base, "jpn", dpi, psm=6, scale=scale)
        else:
            r = O.run_page(path, base, "jpn", dpi, psm=6, scale=scale)
            r.update(scale=scale, psm=6, tried=1)
        return j, r

    with ThreadPoolExecutor(max(1, (os.cpu_count() or 2) // 2)) as ex:
        results = list(ex.map(run, jobs))

    report = {}
    for (vname, pid, *_), r in results:
        words = words_from_tsv(Path(r["tsv"]))
        row = {"scale": r["scale"], "psm": r["psm"], "tried": r["tried"], "mean_conf": r["mean_conf"]}
        gt = Path(args.gt_dir) / f"{pid}.txt" if args.gt_dir else None
        for th in THRESHOLDS:
            text = norm("".join(t for t, c in words if c >= th))
            hits = [t for t in terms[pid] if norm(t) in text]
            row[f"t{th}"] = {
                "found": len(hits), "of": len(terms[pid]), "missing": [t for t in terms[pid] if t not in hits],
                "chars": len(text), "low_conf_chars": sum(len(t) for t, c in words if th <= c < 50),
            }
            if gt and gt.exists():
                ref = norm(gt.read_text(encoding="utf-8"))
                row[f"t{th}"]["cer"] = round(edit_distance(ref, text) / max(1, len(ref)) * 100, 2)
        # 実際に文字層へ入る規則（ページ単位の足切り）での結果
        tsv = Path(r["tsv"])
        text = norm("".join(line_texts(tsv, page_min_conf(tsv))))
        hits = [t for t in terms[pid] if norm(t) in text]
        row["final"] = {"found": len(hits), "of": len(terms[pid]),
                        "missing": [t for t in terms[pid] if t not in hits], "chars": len(text)}
        if gt and gt.exists():
            ref = norm(gt.read_text(encoding="utf-8"))
            row["final"]["cer"] = round(edit_distance(ref, text) / max(1, len(ref)) * 100, 2)
        report.setdefault(vname, {})[pid] = row

    (work / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for vname, pages in report.items():
        print(f"== {vname}")
        for th in THRESHOLDS:
            f = sum(p[f"t{th}"]["found"] for p in pages.values())
            n = sum(p[f"t{th}"]["of"] for p in pages.values())
            junk = pages.get("s0001-L", {}).get(f"t{th}", {}).get("chars")
            cers = [p[f"t{th}"]["cer"] for p in pages.values() if "cer" in p[f"t{th}"]]
            extra = f" CER平均 {sum(cers) / len(cers):.2f}%" if cers else ""
            print(f"  足切り{th:2d}: 重要語 {f}/{n}  白紙ページの文字数 {junk}{extra}")
        f = sum(p["final"]["found"] for p in pages.values())
        n = sum(p["final"]["of"] for p in pages.values())
        cers = [p["final"]["cer"] for p in pages.values() if "cer" in p["final"]]
        extra = f" CER平均 {sum(cers) / len(cers):.2f}%" if cers else ""
        print(f"  採用規則: 重要語 {f}/{n}  白紙ページの文字数 {pages.get('s0001-L', {}).get('final', {}).get('chars')}{extra}")
        for pid, p in pages.items():
            print(f"  {pid}: {p['t0']['found']}/{p['t0']['of']} 未検出 {p['t0']['missing']} (x{p['scale']} psm{p['psm']} 試行{p['tried']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
