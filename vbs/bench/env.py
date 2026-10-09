"""実行環境・コード・依存・入力の記録."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace").stdout.strip()
        except OSError:
            return ""
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status),
            "changed_files": [l[3:] for l in status.splitlines()][:50]}


def environment() -> dict[str, Any]:
    import psutil

    from vbs.ocr import REPO_TESSDATA, tesseract_version

    pkgs = {}
    for name in ("av", "opencv-python-headless", "numpy", "pypdf", "pypdfium2", "pillow", "psutil"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    models = {}
    for f in sorted(Path(REPO_TESSDATA).glob("*.traineddata")):
        models[f.name] = {"size": f.stat().st_size, "sha256": sha256_file(f)}
    try:
        tver = tesseract_version()
    except Exception as e:  # noqa: BLE001
        tver = f"unavailable: {e}"
    vm = psutil.virtual_memory()
    return {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_count": psutil.cpu_count(),
        "ram_total_gib": round(vm.total / 2**30, 2),
        "ram_available_gib": round(vm.available / 2**30, 2),
        "packages": pkgs,
        "tesseract": tver,
        "tessdata": models,
    }
