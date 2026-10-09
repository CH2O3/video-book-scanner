"""本番のCLIを子プロセスとして動かし、計測・強制終了・比較を行う.

試験用の処理を別に実装せず、`python -m vbs ...` をそのまま起動する。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import psutil

from vbs.bench.env import environment, sha256_file
from vbs.bench.monitor import Monitor

KillFn = Callable[[list[dict], subprocess.Popen], str | None]


# ---------------------------------------------------------------- 実行
class EventReader:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.pos = 0
        self.all: list[dict] = []

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, "rb") as f:  # バイト位置で追う（テキストモードは改行の変換で位置がずれる）
            f.seek(self.pos)
            chunk = f.read()
        # 書きかけの最終行は次回に回す
        chunk = chunk[: chunk.rfind(b"\n") + 1]
        self.pos += len(chunk)
        new = [json.loads(l) for l in chunk.decode("utf-8").splitlines() if l.strip()]
        self.all += new
        return new


def kill_tree(pid: int) -> int:
    """強制終了（子孫も含む）。後始末の機会を与えない終わり方を再現する."""
    n = 0
    try:
        root = psutil.Process(pid)
        procs = root.children(recursive=True) + [root]
    except psutil.Error:
        return 0
    for p in procs:
        try:
            p.kill()
            n += 1
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=10)
    return n


def run_vbs(args: list[str], logdir: Path, name: str, workdir: Path | None = None,
            fault: str | None = None, kill: KillFn | None = None, interval: float = 5.0) -> dict[str, Any]:
    logdir.mkdir(parents=True, exist_ok=True)
    events = logdir / f"{name}.events.jsonl"
    events.unlink(missing_ok=True)
    env = dict(os.environ, VBS_EVENTS=str(events), PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    env.pop("VBS_FAULT", None)
    if fault:
        env["VBS_FAULT"] = fault
    from vbs.ocr import awake_seconds

    cmd = [sys.executable, "-m", "vbs", *args]
    t0 = time.time()
    a0 = awake_seconds()
    with open(logdir / f"{name}.log", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        mon = Monitor(proc.pid, workdir, interval=interval).start()
        reader = EventReader(events)
        killed = None
        while proc.poll() is None:
            new = reader.read()
            if kill and killed is None:
                reason = kill(reader.all, proc)
                if reason:
                    n = 0 if "だけ終了" in reason else kill_tree(proc.pid)
                    killed = {"reason": reason, "after_sec": round(time.time() - t0, 1), "processes": n}
            time.sleep(0.3)
        mon.stop()
        reader.read()
    tail = (logdir / f"{name}.log").read_text(encoding="utf-8", errors="replace")[-2000:]
    wall, awake = time.time() - t0, awake_seconds() - a0
    return {"name": name, "cmd": cmd[2:], "returncode": proc.returncode, "elapsed_sec": round(wall, 1),
            "awake_sec": round(awake, 1),
            "standby_suspected": wall - awake > 30,  # 蓋を閉じた等でPCが止まっていた時間がある
            "killed": killed, "fault": fault, "monitor": mon.summary(), "memory_samples": mon.samples,
            "events": summarize_events(reader.all), "log_tail": tail}


def summarize_events(evs: list[dict]) -> dict[str, Any]:
    stages: dict[str, dict[str, float]] = {}
    for e in evs:
        if e["event"] == "progress":
            st = stages.setdefault(e["stage"], {"first": e["t"], "last": e["t"]})
            st["last"] = e["t"]
    return {
        "count": len(evs),
        "stage_sec": {k: round(v["last"] - v["first"], 1) for k, v in stages.items()},
        "ocr_done": sum(1 for e in evs if e["event"] == "ocr_done"),
        "ocr_failed": sum(1 for e in evs if e["event"] == "ocr_done" and e.get("error")),
        "ocr_queue_max": max((e.get("queue", 0) for e in evs if e["event"] == "ocr_done"), default=0),
        "resume": [e for e in evs if e["event"] == "resume"],
        "faults": [e for e in evs if e["event"] == "fault"],
        "cleanup": [e for e in evs if e["event"] == "cleanup"],
    }


# ---------------------------------------------------------------- 停止条件
def when_progress(prefix: str, frac: float) -> KillFn:
    def f(evs: list[dict], proc) -> str | None:
        for e in reversed(evs):
            if e["event"] == "progress" and e["msg"].startswith(prefix) and e["frac"] >= frac:
                return f"progress '{prefix}' >= {frac}"
        return None
    return f


def when_count(kind: str, n: int) -> KillFn:
    def f(evs: list[dict], proc) -> str | None:
        if sum(1 for e in evs if e["event"] == kind) >= n:
            return f"{kind} >= {n}"
        return None
    return f


def when_tesseract_after(n_done: int) -> KillFn:
    """n ページのOCRが終わった後、次のページの Tesseract が動いている最中に止める."""
    def f(evs: list[dict], proc) -> str | None:
        if sum(1 for e in evs if e["event"] == "ocr_done") < n_done:
            return None
        try:
            for c in psutil.Process(proc.pid).children(recursive=True):
                if c.name().lower().startswith("tesseract"):
                    return f"tesseract running after {n_done} pages"
        except psutil.Error:
            pass
        return None
    return f


def kill_parent_only_when_tesseract(n_done: int) -> KillFn:
    """n ページ完了後、Tesseract を動かしている本体の Python だけを終了し、Tesseract は残す.

    タスクマネージャーで本体だけを終了した場合など、子プロセスが取り残される状況を再現する。
    """
    def f(evs: list[dict], proc) -> str | None:
        if sum(1 for e in evs if e["event"] == "ocr_done") < n_done:
            return None
        try:
            for c in psutil.Process(proc.pid).children(recursive=True):
                if c.name().lower().startswith("tesseract"):
                    parent = c.parent()
                    orphans = [x.pid for x in parent.children() if x.name().lower().startswith("tesseract")]
                    parent.kill()
                    return f"本体(pid {parent.pid})だけ終了。Tesseract {orphans} を残した"
        except psutil.Error:
            pass
        return None
    return f


def when_file(pattern_dir: Path, pattern: str) -> KillFn:
    def f(evs: list[dict], proc) -> str | None:
        hits = list(Path(pattern_dir).glob(pattern))
        return f"file {hits[0].name} exists" if hits else None
    return f


# ---------------------------------------------------------------- 比較
def snapshot(project_dir: Path) -> dict[str, Any]:
    """連続実行と再開後を比べるための要約（PDFのバイト列は比べない）."""
    root = Path(project_dir)
    d = json.loads((root / "project.json").read_text(encoding="utf-8"))
    frames = d["frames"]
    segs = []
    for s in sorted(d["segments"], key=lambda s: (s["video_id"] or "", s["start_sec"])):
        segs.append({
            "id": s["id"], "start": round(s["start_sec"], 4), "end": round(s["end_sec"], 4),
            "include": s["include"], "duplicate_of": s.get("duplicate_of"),
            "chosen_time": frames[s["chosen"]]["time_sec"] if s.get("chosen") and frames[s["chosen"]].get("time_sec") is not None else None,
            "candidate_times": [frames[c].get("time_sec") for c in s["candidates"]],
            "warnings": sorted(s["warnings"]),
        })
    pages = []
    for pid in d["order"]:
        p = d["pages"][pid]
        o = p.get("ocr") or {}
        txt = root / o["txt"] if o.get("txt") else None
        pages.append({
            "id": pid, "frame": p["frame_id"],
            "frame_time": frames[p["frame_id"]].get("time_sec"),
            "image_sha256": sha256_file(root / p["image"]) if (root / p["image"]).exists() else None,
            "ocr_text_sha256": sha256_file(txt) if txt and txt.exists() else None,
            "ocr_error": o.get("error"),
        })
    exp = d["exports"][-1] if d.get("exports") else None
    ver = None
    if exp and exp.get("verify") and Path(exp["verify"]).exists():
        v = json.loads(Path(exp["verify"]).read_text(encoding="utf-8"))
        ver = {"pages": v["pages"], "page_count_ok": v["page_count_ok"], "terms": v.get("terms_summary"),
               "per_page_terms": [(p["page_id"], p.get("terms_found")) for p in v["per_page"]]}
    return {"segments": segs, "pages": pages, "frames": len(frames), "export": exp and {
        "pages": exp["pages"], "sha256": exp.get("sha256"), "image_only_pages": exp.get("image_only_pages")},
        "verify": ver}


def compare(ref: dict[str, Any], got: dict[str, Any], parts=("segments", "pages", "verify")) -> list[str]:
    diffs = []
    for part in parts:
        a, b = ref.get(part), got.get(part)
        if part in ("segments", "pages"):
            if len(a) != len(b):
                diffs.append(f"{part}: 件数 {len(a)} != {len(b)}")
            ids_a = [x["id"] for x in a]
            ids_b = [x["id"] for x in b]
            dup = sorted({i for i in ids_b if ids_b.count(i) > 1})
            if dup:
                diffs.append(f"{part}: 二重登録 {dup}")
            missing = sorted(set(ids_a) - set(ids_b))
            extra = sorted(set(ids_b) - set(ids_a))
            if missing:
                diffs.append(f"{part}: 欠落 {missing[:10]}")
            if extra:
                diffs.append(f"{part}: 余分 {extra[:10]}")
            for x, y in zip(a, b):
                if x != y:
                    keys = [k for k in x if x.get(k) != y.get(k)]
                    diffs.append(f"{part}: {x['id']} の {keys} が違う")
        elif a != b:
            diffs.append(f"{part}: 違う")
    return diffs


# ---------------------------------------------------------------- 結果の保存
def new_run_dir(base: Path, label: str) -> Path:
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + label + "-" + uuid.uuid4().hex[:4]
    d = Path(base) / rid
    d.mkdir(parents=True)
    return d


def save_result(run_dir: Path, result: dict[str, Any]) -> Path:
    result.setdefault("environment", environment())
    out = run_dir / "result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def copy_project(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".lock", "*.tmp"))
