"""連結動画の出力を、元動画単独の結果と1件ずつ照合する.

再エンコードせずにつないでいるので、各繰り返し回の中の時刻・フレームは元動画と同じになる。
件数だけでなく、回ごとに「どの見開きを採用したか（回の中の時刻）」と「ページ画像のハッシュ」を
元動画の結果と比べ、毎回同じページを落とす・余分に採る不具合を見つける。

期待値:
- 1回目は元動画と同じ（冒頭の扉の短い静止も、動画の先頭なので採用）
- 2回目以降は、扉の見開きが「動画の途中の短い静止」になるので採用されない（仕様どおり）
- それ以外の見開きは、各回で元動画と同じ時刻・同じ画像
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vbs.bench.env import sha256_file


def _load(project: Path) -> dict[str, Any]:
    return json.loads((Path(project) / "project.json").read_text(encoding="utf-8"))


def _pages_by_local_time(d: dict, root: Path, period: float | None) -> list[dict[str, Any]]:
    out = []
    for pid in d["order"]:
        p = d["pages"][pid]
        f = d["frames"][p["frame_id"]]
        t = f["time_sec"]
        rep, local = (int(t // period), t - int(t // period) * period) if period else (0, t)
        out.append({"pid": pid, "side": p["side"], "rep": rep, "local": round(local, 3),
                    "sha256": sha256_file(root / p["image"]) if (root / p["image"]).exists() else None})
    return out


def concat_check(base_project: Path, concat_project: Path, source_json: Path) -> dict[str, Any]:
    src = json.loads(Path(source_json).read_text(encoding="utf-8"))
    reps = src["repetitions"]
    joins = [0.0] + src["joins_sec"]
    period = joins[1] if len(joins) > 1 else src["duration_sec"]
    base, conc = _load(base_project), _load(concat_project)
    bp = _pages_by_local_time(base, Path(base_project), None)
    cp = _pages_by_local_time(conc, Path(concat_project), period)
    base_keys = [(x["local"], x["side"]) for x in bp]
    first_key = min(base_keys)[0] if base_keys else None  # 冒頭の扉の時刻
    per_rep = []
    problems = []
    for r in range(reps):
        got = [x for x in cp if x["rep"] == r]
        exp = [x for x in bp if r == 0 or x["local"] != first_key]
        # 回の中の時刻は割り算の丸めで数ミリ秒ずれるので、同じ左右で 20ms 以内の最も近いものを対応させる
        pairs, used = [], set()
        for e in exp:
            cands = [(abs(g["local"] - e["local"]), gi) for gi, g in enumerate(got)
                     if g["side"] == e["side"] and gi not in used and abs(g["local"] - e["local"]) < 0.02]
            if cands:
                gi = min(cands)[1]
                used.add(gi)
                pairs.append((e, got[gi]))
        matched_e = {id(e) for e, _ in pairs}
        missing = sorted((round(e["local"], 3), e["side"]) for e in exp if id(e) not in matched_e)
        extra = sorted((round(g["local"], 3), g["side"]) for gi, g in enumerate(got) if gi not in used)
        img_diff = sorted((round(e["local"], 3), e["side"]) for e, g in pairs if e["sha256"] != g["sha256"])
        per_rep.append({"rep": r, "expected": len(exp), "got": len(got), "missing": missing, "extra": extra,
                        "image_differs": img_diff})
        if missing or extra or img_diff:
            problems.append(per_rep[-1])
    expected_total = len(bp) + (reps - 1) * (len(bp) - sum(1 for k in base_keys if k[0] == first_key))
    # 元動画のページ → 連結動画のページの対応（検索語の割り当てに使う）
    mapping = {}
    for x in cp:
        for b in bp:
            if abs(b["local"] - x["local"]) < 0.01 and b["side"] == x["side"]:
                mapping[x["pid"]] = b["pid"]
    return {"kind": "concat-check", "repetitions": reps, "period_sec": round(period, 3),
            "base_pages": len(bp), "expected_total": expected_total, "got_total": len(cp),
            "reps_ok": reps - len(problems), "problems": problems[:10], "mapping": mapping}


def map_terms(base_terms: dict[str, list[str]], mapping: dict[str, str]) -> dict[str, list[str]]:
    return {cpid: base_terms[bpid] for cpid, bpid in mapping.items() if bpid in base_terms}
