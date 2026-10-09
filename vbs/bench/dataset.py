"""公開データの取得・読込・評価.

SmartDoc 2015 Challenge 1（CC-BY-4.0, Zenodo record 1230218）
- 単一文書を撮った短い動画と、各フレームの文書四隅の正解座標
- ここでは「動画読込（フレーム番号との対応）」と「紙面範囲の検出」を評価する
- アプリの紙面検出は軸に平行な矩形しか出さない。四隅の座標は出していないので、
  四隅の評価は「未対応」とし、正解四角形と検出矩形の画像上のIoUで測る
  （公式評価の、文書座標系に変換したJaccard指数とは異なる）
- ページめくり検出や日本語OCRの評価には使わない

PUCIT Page Turn Dataset
- 論文記載の配布先へ短く接続を確かめるだけ（取得は利用者の許可が別に必要）
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vbs.bench.env import sha256_file

ZENODO = "https://zenodo.org/api/records/1230218"
SAMPLE = {"key": "sampleDataset.tar.gz", "md5": "1ee5b7c290d707bd51c59f0b1c1a36f5"}
PUCIT_URL = "http://faculty.pucit.edu.pk/nazarkhan/datasets/pucit_page_turns.zip"


def _catalog(dirpath: Path, entry: dict[str, Any]) -> None:
    cat = dirpath / "CATALOG.json"
    data = json.loads(cat.read_text(encoding="utf-8")) if cat.exists() else {"datasets": []}
    data["datasets"] = [d for d in data["datasets"] if d.get("name") != entry["name"]] + [entry]
    cat.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def smartdoc_fetch(dirpath: Path, sample_only: bool = True) -> dict[str, Any]:
    """サンプル（約21MB）を取得・検証・展開し、データ目録に記録する。本体は sample_only=False のときだけ."""
    import hashlib
    import tarfile

    base = Path(dirpath) / "smartdoc15"
    base.mkdir(parents=True, exist_ok=True)
    meta = json.loads(urllib.request.urlopen(ZENODO, timeout=30).read().decode("utf-8"))
    files = {f["key"]: f for f in meta["files"]}
    keys = ["sampleDataset.tar.gz"] + ([] if sample_only else ["testDataset.tar.gz"])
    got = []
    for key in keys:
        f = files[key]
        dst = base / key
        if not dst.exists() or dst.stat().st_size != f["size"]:
            urllib.request.urlretrieve(f["links"]["self"], dst)
        md5 = hashlib.md5(dst.read_bytes()).hexdigest()
        if "md5:" + md5 != f["checksum"]:
            raise RuntimeError(f"チェックサムが一致しません: {key}")
        out = base / key.replace(".tar.gz", "")
        if not out.exists():
            with tarfile.open(dst) as t:
                t.extractall(out, filter="data")  # パスの不正な項目を拒否する
        got.append({"file": key, "size": f["size"], "md5": md5, "sha256": sha256_file(dst), "extracted": str(out)})
    entry = {"name": "SmartDoc 2015 Challenge 1", "source": ZENODO, "license": meta["metadata"].get("license"),
             "retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": got,
             "use": "動画読込と紙面範囲検出の評価のみ。ページめくり・日本語OCRには使わない"}
    _catalog(Path(dirpath), entry)
    return entry


def _gt(xml_path: Path) -> dict[int, np.ndarray | None]:
    root = ET.parse(xml_path).getroot()
    out: dict[int, np.ndarray | None] = {}
    for fr in root.iter("frame"):
        idx = int(fr.get("index"))
        if fr.get("rejected") == "true":
            out[idx] = None
            continue
        pts = {p.get("name"): (float(p.get("x")), float(p.get("y"))) for p in fr.iter("point")}
        out[idx] = np.array([pts["tl"], pts["tr"], pts["br"], pts["bl"]], np.float32)
    return out


def _iou_quad_rect(quad: np.ndarray, rect: tuple[int, int, int, int], shape: tuple[int, int]) -> float:
    h, w = shape
    a = np.zeros((h, w), np.uint8)
    b = np.zeros((h, w), np.uint8)
    cv2.fillPoly(a, [np.round(quad).astype(np.int32)], 1)
    x0, y0, x1, y1 = rect
    b[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 1
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def smartdoc_eval(dirpath: Path, out_dir: Path, max_examples: int = 6) -> dict[str, Any]:
    from vbs.split import detect_page_region
    from vbs.video import iter_frames, probe

    root = Path(dirpath) / "smartdoc15"
    videos = sorted(root.glob("**/input_sample/**/*.avi")) + sorted(root.glob("**/testDataset/**/*.avi"))
    out = Path(out_dir) / ("smartdoc-" + time.strftime("%Y%m%d-%H%M%S"))
    (out / "examples").mkdir(parents=True)
    per_video = []
    all_iou: list[float] = []
    worst: list[tuple[float, str, int, np.ndarray, np.ndarray, tuple]] = []
    for v in videos:
        gt_path = next(iter(root.glob(f"**/{v.parent.name}_gt/{v.stem}.gt.xml")), None)
        if gt_path is None:
            per_video.append({"video": str(v), "status": "正解なし"})
            continue
        gt = _gt(gt_path)
        info = probe(v)
        n = 0
        ious = []
        fails = 0
        times = []
        for i, (t, pts, img) in enumerate(iter_frames(v, fmt="bgr24"), start=1):
            n = i
            times.append(t)
            quad = gt.get(i)
            if quad is None:
                continue
            rect, warn = detect_page_region(img)
            iou = _iou_quad_rect(quad, rect, img.shape[:2])
            if "page_region_uncertain" in warn:
                fails += 1
            ious.append(iou)
            if len(worst) < max_examples or iou < max(w[0] for w in worst):
                worst.append((iou, v.stem, i, img.copy(), quad, rect))
                worst = sorted(worst, key=lambda w: w[0])[:max_examples]
        arr = np.array(ious)
        dts = np.diff(times) if len(times) > 1 else np.array([0.0])
        per_video.append({
            "video": f"{v.parent.name}/{v.name}", "frames_decoded": n, "frames_gt": len(gt),
            "frame_count_match": n == len(gt), "fps_nominal": info["avg_rate"],
            "dt_ms_min_max": [round(float(dts.min()) * 1000, 2), round(float(dts.max()) * 1000, 2)],
            "evaluated": int(arr.size), "detect_uncertain": fails,
            "iou_mean": round(float(arr.mean()), 4) if arr.size else None,
            "iou_median": round(float(np.median(arr)), 4) if arr.size else None,
            "iou_ge_0_9": round(float((arr >= 0.9).mean()), 4) if arr.size else None,
            "iou_lt_0_5": int((arr < 0.5).sum()),
        })
        all_iou += ious
    for k, (iou, name, i, img, quad, rect) in enumerate(worst):
        vis = img.copy()
        cv2.polylines(vis, [np.round(quad).astype(np.int32)], True, (0, 200, 0), 3)
        cv2.rectangle(vis, rect[:2], rect[2:], (0, 0, 255), 3)
        cv2.putText(vis, f"{name} #{i} IoU {iou:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imwrite(str(out / "examples" / f"{k:02d}_{name}_{i:04d}.jpg"), vis)
    a = np.array(all_iou)
    rec = {
        "kind": "smartdoc-eval", "dataset": "SmartDoc 2015 Challenge 1",
        "scope": "動画読込（フレーム番号の対応）と紙面範囲（軸平行の矩形）のIoU。四隅座標の評価は未対応",
        "metric_note": "公式の評価（文書座標系のJaccard）とは異なる。検出失敗も集計に含める（IoUが低い値として）",
        "videos": len(per_video), "frames_evaluated": int(a.size),
        "iou_mean": round(float(a.mean()), 4) if a.size else None,
        "iou_median": round(float(np.median(a)), 4) if a.size else None,
        "iou_ge_0_9": round(float((a >= 0.9).mean()), 4) if a.size else None,
        "iou_ge_0_8": round(float((a >= 0.8).mean()), 4) if a.size else None,
        "iou_lt_0_5": int((a < 0.5).sum()) if a.size else None,
        "examples_dir": str(out / "examples"), "per_video": per_video,
    }
    (out / "result.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return rec


def pucit_check(dirpath: Path) -> dict[str, Any]:
    """配布先に短く接続を試す。取得はしない（HEAD のみ）."""
    rec: dict[str, Any] = {"name": "PUCIT Page Turn Dataset", "url": PUCIT_URL,
                           "checked": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        req = urllib.request.Request(PUCIT_URL, method="HEAD")
        with urllib.request.urlopen(req, timeout=15) as r:
            rec.update(status="reachable", http=r.status, length=r.headers.get("Content-Length"),
                       note="接続できた。取得は未実施（利用者の許可範囲外）")
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        rec.update(status="未取得", error=f"{type(e).__name__}: {e}")
    _catalog(Path(dirpath), {"name": rec["name"], "source": PUCIT_URL, "status": rec["status"],
                             "checked": rec["checked"], "error": rec.get("error")})
    return rec
