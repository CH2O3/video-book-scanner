"""合成レシートで読み取りの正解率を測る.

  python tools/eval_receipts.py work/reval --n 12 --seed 21
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vbs import pipeline  # noqa: E402
from vbs.manifest import Project  # noqa: E402

KEYS = ("date", "payee", "total", "base10", "base8", "tax", "registration_no")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    photos = out / "photos"
    subprocess.run([sys.executable, str(Path(__file__).with_name("make_test_receipts.py")), str(photos),
                    "--n", str(a.n), "--seed", str(a.seed)], check=True, capture_output=True)
    truth = json.loads((photos / "truth.json").read_text(encoding="utf-8"))
    p = Project.create(out / "p", {"document": "receipt", "auto_export": False})
    for f in sorted(photos.glob("*.jpg")):
        pipeline.add_photo(p, f, None)
    pipeline.update_order(p)
    pipeline.run_all(p, export=False)
    right = {k: 0 for k in KEYS}
    wrong = {k: 0 for k in KEYS}   # 値を出したが違う（危ない誤り）
    empty = {k: 0 for k in KEYS}   # 読めずに空（確認に回る）
    flagged_wrong = 0
    for pid, t in zip(p.data["order"], truth):
        pg = p.data["pages"][pid]
        f = pipeline.receipt_fields(p, pg)
        codes = pipeline.receipt_warnings(p, pg)
        bad = []
        for k in KEYS:
            got, want = f.get(k), t.get(k)
            if k == "payee" and got:
                got = got.replace(" ", "")
                want = (want or "").replace(" ", "")
            if got == want:
                right[k] += 1
            elif got is None:
                empty[k] += 1
            else:
                wrong[k] += 1
                bad.append((k, got, want))
        if bad and codes:
            flagged_wrong += 1
        print(pid, "OK" if not bad else bad, codes)
    n = len(truth)
    print("\n項目     正解  誤り  空欄（確認へ）")
    for k in KEYS:
        print(f"{k:16s} {right[k]:3d}/{n}  {wrong[k]:3d}  {empty[k]:3d}")
    print("誤りのあるレシートのうち、要確認に回ったもの:", flagged_wrong)


if __name__ == "__main__":
    main()
