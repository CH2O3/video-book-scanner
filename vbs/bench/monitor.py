"""プロセスツリーのメモリと作業ディレクトリ容量の計測.

メモリの定義（Windows, psutil）:
- rss: Working Set（物理メモリ上の使用量）の合計
- private: Private Bytes（そのプロセス専用にコミットされた量）の合計。メモリ上限の判定にはこちらを使う
対象は起動したプロセスとその子孫すべて（venv のランチャー、本体の Python、Tesseract を含む）。
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import psutil


def dir_size(path: Path) -> tuple[int, int]:
    total = files = 0
    stack = [str(path)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        else:
                            total += e.stat(follow_symlinks=False).st_size
                            files += 1
                    except OSError:
                        pass
        except OSError:
            pass
    return total, files


class Monitor:
    def __init__(self, pid: int, workdir: Path | None = None, interval: float = 5.0, dir_every: float = 30.0):
        self.pid = pid
        self.workdir = workdir
        self.interval = interval
        self.dir_every = dir_every
        self.samples: list[dict[str, Any]] = []
        self.peak = {"rss": 0, "private": 0, "procs": 0, "tesseract": 0}
        self._stop = threading.Event()
        self._t0 = time.time()
        self._last_dir = 0.0
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "Monitor":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        self.sample(final=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    def sample(self, final: bool = False) -> None:
        rss = private = n = tess = 0
        try:
            root = psutil.Process(self.pid)
            procs = [root] + root.children(recursive=True)
        except psutil.Error:
            procs = []
        for p in procs:
            try:
                mi = p.memory_info()
                rss += mi.rss
                private += getattr(mi, "private", mi.rss)
                n += 1
                if p.name().lower().startswith("tesseract"):
                    tess += 1
            except psutil.Error:
                pass
        now = time.time()
        rec: dict[str, Any] = {"t": round(now - self._t0, 1), "rss_mib": round(rss / 2**20, 1),
                               "private_mib": round(private / 2**20, 1), "procs": n, "tesseract": tess}
        if self.workdir and (final or now - self._last_dir >= self.dir_every):
            self._last_dir = now
            size, files = dir_size(self.workdir)
            rec["workdir_mib"] = round(size / 2**20, 1)
            rec["workdir_files"] = files
        if n:
            self.samples.append(rec)
            self.peak["rss"] = max(self.peak["rss"], rss)
            self.peak["private"] = max(self.peak["private"], private)
            self.peak["procs"] = max(self.peak["procs"], n)
            self.peak["tesseract"] = max(self.peak["tesseract"], tess)
        elif final and self.workdir:
            self.samples.append(rec)

    def summary(self) -> dict[str, Any]:
        dirs = [s for s in self.samples if "workdir_mib" in s]
        return {
            "definition": "psutil: rss=Working Set, private=Private Bytes; プロセスツリー（子孫を含む）の合計",
            "interval_sec": self.interval,
            "peak_rss_mib": round(self.peak["rss"] / 2**20, 1),
            "peak_private_mib": round(self.peak["private"] / 2**20, 1),
            "peak_procs": self.peak["procs"],
            "peak_tesseract": self.peak["tesseract"],
            "workdir_peak_mib": max((s["workdir_mib"] for s in dirs), default=None),
            "workdir_final_mib": dirs[-1]["workdir_mib"] if dirs else None,
            "samples": len(self.samples),
        }
