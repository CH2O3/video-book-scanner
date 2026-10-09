"""プロジェクトのmanifest（project.json）と作業ディレクトリの管理.

仕様書11章のデータ単位（Project / Video / Segment / Frame / Page）を
一つのJSONで持つ。IDは並べ替えても変わらない。保存は一時ファイル経由で置換する。
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from vbs import __version__

SCHEMA_VERSION = 1
MANIFEST_NAME = "project.json"
LOCK_NAME = ".lock"

DIRS = {
    "videos": "videos",
    "frames": "frames",
    "thumbs": "thumbs",
    "pages": "pages",
    "ocr": "ocr",
    "output": "output",
    "jobs": "jobs",
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "layout": "spread",          # spread（見開き） | single（片ページ）
    "direction": "ltr",          # ltr（左→右） | rtl（右→左）
    "expected_pages": None,      # 予定ページ数（任意）
    "page_height_mm": 257.0,     # PDF上の紙面の高さ。B5=257, A5=210, A4=297
    "ocr_lang": "jpn",           # jpn+eng は日本語行を英字と誤読することがある
    "ocr_psm": 6,                # 6=単一ブロック。段組みの多い本は 3 を試す
    "ocr_scale": "auto",         # auto=行間から1〜2倍に拡大してOCR
    "auto_export": True,         # 要確認がなければPDFを自動出力
    "analysis_width": 160,       # 動き解析用の縮小幅(px)
    "motion_threshold": None,    # None なら自動
    "min_still_sec": 0.5,        # 静止とみなす最短時間
    "max_candidates": 3,
    "candidate_format": "jpg",   # jpg（品質95） | png
    "enhance": "normalize",      # normalize（照明ムラ補正） | none
    "dewarp": "auto",
    "margins": "content_box",    # 余白: content_box（内容の外を白に統一） | detect（手の検出だけ） | off
    "content_rotation": "auto",  # 紙面の向き: auto（文字から判定） | 0 | 90 | 180 | 270（反時計回り）
    "document": "book",          # book（本） | receipt（レシート）
    "dedupe": "merge",           # 同じ見開きの続けての静止: merge（まとめる） | flag（確認に回すだけ）
    "receipt_width_mm": 80.0,    # レシートの幅（PDFの寸法と解像度の確認に使う）。58 か 80 が多い
}

# レシートは見た目を変えずに保存する（証憑として、切り抜きと傾き補正だけ）。
# 同じ店のレシートは見た目が似ているので、重複は自動でまとめず確認へ回す
RECEIPT_PRESET: dict[str, Any] = {
    "layout": "single",
    "enhance": "none",
    "dewarp": "off",
    "margins": "detect",
    "dedupe": "flag",
    "content_rotation": "auto",
    "expected_pages": None,
}


def with_preset(settings: dict[str, Any] | None) -> dict[str, Any]:
    """種類（本・レシート）の既定値に、明示された設定を重ねる."""
    s = {k: v for k, v in (settings or {}).items() if v is not None}
    merged = dict(DEFAULT_SETTINGS)
    if s.get("document") == "receipt":
        merged.update(RECEIPT_PRESET)
        # レシートの見た目を変える設定は受け付けない
        for k in ("layout", "enhance", "dewarp", "margins", "dedupe"):
            s.pop(k, None)
    merged.update(s)
    return merged


class ProjectError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    from vbs import faults

    faults.point("write", str(path))
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


class Project:
    def __init__(self, root: Path, data: dict[str, Any]):
        # 絶対パスにそろえる（外部コマンドに渡すパスや、残ったプロセスの照合を安定させる）
        self.root = Path(root).resolve()
        self.data = data

    # ---- 生成・読込 ---------------------------------------------------
    @classmethod
    def create(cls, root: Path, settings: dict[str, Any] | None = None) -> "Project":
        root = Path(root)
        if (root / MANIFEST_NAME).exists():
            raise ProjectError(f"既にプロジェクトがあります: {root}")
        root.mkdir(parents=True, exist_ok=True)
        for d in DIRS.values():
            (root / d).mkdir(exist_ok=True)
        merged = with_preset(settings)
        data = {
            "schema_version": SCHEMA_VERSION,
            "project_id": uuid.uuid4().hex,
            "created": now_iso(),
            "updated": now_iso(),
            "processor_versions": {"vbs": __version__},
            "settings": merged,
            "videos": [],          # 取込順＝動画順
            "segments": [],        # 撮影順
            "frames": {},          # frame_id -> frame
            "pages": {},           # page_id -> page
            "order": [],           # 出力ページID順
            "exports": [],
            "next_ids": {"video": 1, "segment": 1, "frame": 1},
        }
        p = cls(root, data)
        p.save()
        return p

    @classmethod
    def load(cls, root: Path) -> "Project":
        root = Path(root)
        path = root / MANIFEST_NAME
        if not path.exists():
            raise ProjectError(f"project.json がありません: {root}")
        data = json.loads(path.read_text(encoding="utf-8"))
        ver = data.get("schema_version")
        if ver != SCHEMA_VERSION:
            raise ProjectError(
                f"未対応の schema_version です: {ver}（このアプリは {SCHEMA_VERSION}）"
            )
        for d in DIRS.values():
            (root / d).mkdir(exist_ok=True)
        return cls(root, data)

    def save(self) -> None:
        self.data["updated"] = now_iso()
        atomic_write_json(self.root / MANIFEST_NAME, self.data)

    # ---- ロック（二重起動の防止） ---------------------------------------
    @contextmanager
    def lock(self) -> Iterator[None]:
        path = self.root / LOCK_NAME
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            pid = path.read_text(encoding="utf-8", errors="replace").strip()
            if pid.isdigit() and not _pid_alive(int(pid)):
                path.unlink(missing_ok=True)
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            else:
                raise ProjectError(
                    f"このプロジェクトは別のプロセス(pid={pid})が使用中です。"
                    f"使用中でなければ {path} を削除してください。"
                )
        try:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            yield
        finally:
            path.unlink(missing_ok=True)

    # ---- ヘルパー -----------------------------------------------------
    @property
    def settings(self) -> dict[str, Any]:
        return self.data["settings"]

    def path(self, kind: str, name: str = "") -> Path:
        base = self.root / DIRS[kind]
        return base / name if name else base

    def rel(self, path: Path) -> str:
        return Path(os.path.relpath(path, self.root)).as_posix()

    def abs(self, rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        return p if p.is_absolute() else self.root / p

    def new_id(self, kind: str) -> str:
        n = self.data["next_ids"][kind]
        self.data["next_ids"][kind] = n + 1
        prefix = {"video": "v", "segment": "s", "frame": "f"}[kind]
        return f"{prefix}{n:04d}"

    def video(self, video_id: str) -> dict[str, Any]:
        for v in self.data["videos"]:
            if v["id"] == video_id:
                return v
        raise ProjectError(f"動画IDが見つかりません: {video_id}")

    def segment(self, segment_id: str) -> dict[str, Any]:
        for s in self.data["segments"]:
            if s["id"] == segment_id:
                return s
        raise ProjectError(f"区間IDが見つかりません: {segment_id}")


def _pid_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
