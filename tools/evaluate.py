"""合成動画の正解とプロジェクトの結果を照合する.

- 区間: 正解の見開きごとに、採用区間が1つだけ対応するか（見逃し・余分・重複）
- OCR: ページごとのCER（空白・改行を除き、数字と句読点は残す。仕様AC-05）
- PDF: 出力PDFのテキストから各ページのキーワードが検索できるか（AC-06）

使い方: python tools/evaluate.py work/t1.truth.json work/p1
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from pypdf import PdfReader


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def edit_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = np.arange(len(b) + 1, dtype=np.int32)
    bb = np.frombuffer(b.encode("utf-32-le"), dtype=np.uint32)
    for i, ca in enumerate(a, 1):
        cur = np.empty_like(prev)
        cur[0] = i
        sub = prev[:-1] + (bb != ord(ca))
        dele = prev[1:] + 1
        best = np.minimum(sub, dele)
        # 挿入は左からの累積なので逐次
        for j in range(1, len(b) + 1):
            v = best[j - 1]
            ins = cur[j - 1] + 1
            cur[j] = v if v < ins else ins
        prev = cur
    return int(prev[-1])


def main() -> int:
    truth = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    root = Path(sys.argv[2])
    d = json.loads((root / "project.json").read_text(encoding="utf-8"))
    segs = [s for s in d["segments"] if s["include"]]
    ok = True

    # ---- 区間
    matched = []
    for sp in truth["spreads"]:
        hits = [s["id"] for s in segs if s["start_sec"] < sp["end"] and s["end_sec"] > sp["start"]]
        matched.append(hits)
        status = "OK" if len(hits) == 1 else ("MISS" if not hits else "DUP")
        ok &= status == "OK"
        print(f"見開き{sp['index'] + 1} [{sp['start']:.2f}-{sp['end']:.2f}] -> {hits} {status}")
    extra = [s["id"] for s in segs if not any(s["id"] in h for h in matched)]
    if extra:
        ok = False
        print(f"余分な採用区間: {extra}")

    # ---- OCR（読み順で正解ページと対応付け）
    order = d["order"]
    total_err = total_len = 0
    for i, pid in enumerate(order):
        if i >= len(truth["pages"]):
            break
        ref = norm(truth["pages"][i]["text"])
        txt_rel = (d["pages"][pid].get("ocr") or {}).get("txt")
        hyp = norm((root / txt_rel).read_text(encoding="utf-8")) if txt_rel else ""
        e = edit_distance(ref, hyp)
        total_err += e
        total_len += len(ref)
        print(f"{pid}: CER {e / len(ref) * 100:5.2f}% ({e}/{len(ref)})")
    if total_len:
        print(f"全体CER: {total_err / total_len * 100:.2f}%")

    # ---- PDF検索
    exports = d.get("exports") or []
    if exports:
        pdf = PdfReader(exports[-1]["path"])
        found = 0
        for i, page in enumerate(pdf.pages):
            if i >= len(truth["pages"]):
                break
            kw = truth["pages"][i]["keyword"]
            text = norm(page.extract_text() or "")
            hit = kw in text
            found += hit
            if not hit:
                print(f"  PDF p{i + 1}: 「{kw}」が見つからない")
        print(f"PDF検索: {found}/{min(len(pdf.pages), len(truth['pages']))} ページでキーワード検出")
    print("区間判定:", "合格" if ok else "不合格")
    return 0 if ok else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
