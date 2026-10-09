"""PUCIT Page Turn Dataset でのページ抽出の評価.

正解（<動画>-GT.txt）: 1行目が区間数、以降「開始-終了」（1始まり・表示順のフレーム番号）の
採用してよい区間。ページIDや「手」「ページ戻り」などの条件の印はないので、推測で付けない。

測るもの（本番の抽出処理を設定を変えずに使う）
- 拾えた区間: 採用した区間の代表フレームが正解区間に入っていれば成功（1区間で複数あっても成功は1件）
- 誤採用: 代表フレームがどの正解区間にも入らない採用（めくり途中など）
- 重複: 1つの正解区間に2つ以上の採用
- 取りこぼしの内訳: 静止を検出できず / 短い静止で不採用 / 重複として統合 / 候補が区間外 / その他
- 正解区間の長さ別の拾えた割合

改善用（dev）と最終確認用（test）は、動画名の順で3本ごとに1本を test にする固定の分け方。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("samples/datasets/pucit/extracted/VIDEOS-SMALL")


def load_gt(path: Path) -> list[tuple[int, int]]:
    lines = [l.strip() for l in Path(path).read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    n = int(lines[0])
    out = []
    for l in lines[1:]:
        a, b = l.split("-")
        out.append((int(a), int(b)))
    if len(out) != n:
        raise ValueError(f"区間数が1行目と合いません: {path} ({len(out)} != {n})")
    return out


def split(videos: list[Path]) -> dict[str, list[str]]:
    names = sorted(v.stem for v in videos)
    return {"dev": [n for i, n in enumerate(names) if i % 3 != 2], "test": [n for i, n in enumerate(names) if i % 3 == 2]}


def frame_index_map(video: Path) -> dict[int, int]:
    """pts → 表示順の番号（1始まり）。正解のフレーム番号と対応させる."""
    from vbs.video import iter_frames

    return {pts: i for i, (_, pts, _) in enumerate(iter_frames(video, width=32, fmt="gray"), start=1)}


def evaluate_video(video: Path, gt: list[tuple[int, int]], workdir: Path,
                   settings: dict[str, Any] | None = None) -> dict[str, Any]:
    from vbs import pipeline
    from vbs.manifest import Project

    t0 = time.perf_counter()
    if (workdir / video.stem / "project.json").exists():
        proj = Project.load(workdir / video.stem)  # 抽出済みなら集計だけやり直す
    else:
        proj = Project.create(workdir / video.stem, settings or {})
        pipeline.add_videos(proj, [video], copy=False, allow_hdr=True)
        pipeline.analyze(proj)
    elapsed = time.perf_counter() - t0
    idx = frame_index_map(video)
    frames = proj.data["frames"]
    segs = [s for s in proj.data["segments"] if s["candidates"] or "unreadable" in s["warnings"]]
    times = sorted(idx)  # pts の昇順 = 表示順

    def fidx(fid: str) -> int:
        return idx[frames[fid]["pts"]]

    # 区間の範囲（フレーム番号）: 開始・終了時刻に最も近いフレーム
    pts_by_time = {}
    import av

    with av.open(str(video)) as c:
        tb = c.streams.video[0].time_base
    t_of = {p: float(p * tb) for p in times}
    order = sorted(times, key=lambda p: t_of[p])

    def nearest(t: float) -> int:
        k = int(np.argmin([abs(t_of[p] - t) for p in order]))
        return idx[order[k]]

    included = []
    for s in segs:
        info = {"id": s["id"], "include": s["include"], "warnings": s["warnings"],
                "duplicate_of": s.get("duplicate_of"),
                "span": [nearest(s["start_sec"]), nearest(s["end_sec"])],
                "chosen": fidx(s["chosen"]) if s.get("chosen") else None}
        s["_info"] = info
        if s["include"] and s.get("chosen"):
            included.append(info)

    def which(i: int | None) -> int | None:
        if i is None:
            return None
        for k, (a, b) in enumerate(gt):
            if a <= i <= b:
                return k
        return None

    hits: dict[int, list[str]] = {k: [] for k in range(len(gt))}
    false = []
    for inc in included:
        k = which(inc["chosen"])
        if k is None:
            # 正解区間からの距離（めくり途中か、区間の端の外側か）
            dist = min(min(abs(inc["chosen"] - a), abs(inc["chosen"] - b)) for a, b in gt)
            false.append({**inc, "dist_frames": dist})
        else:
            hits[k].append(inc["id"])
    misses = []
    for k, (a, b) in enumerate(gt):
        if hits[k]:
            continue
        overl = [s["_info"] for s in segs if s["_info"]["span"][0] <= b and s["_info"]["span"][1] >= a]
        by_id = {s["id"]: s["_info"] for s in segs}
        if not overl:
            cause = "静止を検出できず"
        elif any(o["include"] for o in overl):
            cause = "候補が区間外"
        elif any(o["duplicate_of"] for o in overl):
            keeps = [by_id.get(o["duplicate_of"]) for o in overl if o["duplicate_of"]]
            if any(kp and which(kp["chosen"]) not in (None, k) for kp in keeps):
                cause = "別ページと統合（誤判定の疑い）"
            else:
                cause = "統合先の候補が区間外"
        elif any("short_still" in o["warnings"] for o in overl):
            cause = "短い静止で不採用"
        else:
            cause = "その他"
        misses.append({"gt": k, "range": [a, b], "len": b - a + 1, "cause": cause,
                       "overlapping": [{"id": o["id"], "span": o["span"], "include": o["include"],
                                        "chosen": o["chosen"], "warnings": o["warnings"]} for o in overl]})
    fps = len(idx) / max(1e-6, t_of[order[-1]] - t_of[order[0]])
    lens = [(b - a + 1) / fps for a, b in gt]
    return {
        "video": video.name, "frames": len(idx), "fps": round(fps, 2), "gt_intervals": len(gt),
        "included": len(included), "found": sum(1 for k in hits if hits[k]),
        "duplicates": sum(1 for k in hits if len(hits[k]) > 1),
        "false_adoptions": len(false), "false": false, "misses": misses,
        "gt_len_sec": [round(x, 2) for x in lens],
        "hit_by_interval": [bool(hits[k]) for k in range(len(gt))],
        "threshold": (proj.data["videos"][0].get("analysis") or {}).get("threshold"),
        "elapsed_sec": round(elapsed, 1),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gt = sum(r["gt_intervals"] for r in rows)
    found = sum(r["found"] for r in rows)
    inc = sum(r["included"] for r in rows)
    causes: dict[str, int] = {}
    for r in rows:
        for m in r["misses"]:
            causes[m["cause"]] = causes.get(m["cause"], 0) + 1
    buckets = {"<0.5s": [0, 0], "0.5-1s": [0, 0], "1-2s": [0, 0], ">=2s": [0, 0]}
    for r in rows:
        for L, h in zip(r["gt_len_sec"], r["hit_by_interval"]):
            k = "<0.5s" if L < 0.5 else "0.5-1s" if L < 1 else "1-2s" if L < 2 else ">=2s"
            buckets[k][0] += h
            buckets[k][1] += 1
    false = [f for r in rows for f in r["false"]]
    return {
        "videos": len(rows), "gt_intervals": gt, "found": found,
        "recall": round(found / gt, 4) if gt else None,
        "included": inc, "false_adoptions": len(false),
        "false_rate": round(len(false) / inc, 4) if inc else None,
        "false_near_boundary(<=15f)": sum(1 for f in false if f["dist_frames"] <= 15),
        "duplicates": sum(r["duplicates"] for r in rows),
        "miss_causes": causes,
        "recall_by_gt_length": {k: f"{v[0]}/{v[1]}" for k, v in buckets.items()},
    }


def run(out_base: Path, only: list[str] | None = None, limit: int | None = None,
        subset: str | None = None, settings: dict[str, Any] | None = None, label: str = "") -> dict[str, Any]:
    videos = sorted(ROOT.glob("*.mp4"))
    sp = split(videos)
    if only:
        videos = [v for v in videos if v.stem in only]
    if subset:
        videos = [v for v in videos if v.stem in sp[subset]]
    if limit:
        videos = videos[:limit]
    out = Path(out_base) / ("pucit-" + time.strftime("%Y%m%d-%H%M%S") + (f"-{label}" if label else ""))
    out.mkdir(parents=True)
    rows = []
    for v in videos:
        gt = load_gt(v.with_name(v.stem + "-GT.txt"))
        r = evaluate_video(v, gt, out / "projects", settings)
        r["set"] = "test" if v.stem in sp["test"] else "dev"
        rows.append(r)
        print(f"{v.stem} [{r['set']}] 正解 {r['gt_intervals']} / 拾えた {r['found']} / 採用 {r['included']} / "
              f"誤採用 {r['false_adoptions']} / 重複 {r['duplicates']} / 閾値 {r['threshold']}", flush=True)
    res = {"kind": "pucit-eval", "settings": settings or "既定（変更なし）", "split": sp,
           "all": summarize(rows),
           "dev": summarize([r for r in rows if r["set"] == "dev"]),
           "test": summarize([r for r in rows if r["set"] == "test"]),
           "per_video": rows}
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    res["dir"] = str(out)
    return res
