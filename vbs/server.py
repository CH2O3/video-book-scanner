"""画面（ローカルWebアプリ）.

127.0.0.1 だけで待ち受け、起動ごとのトークンがない要求は拒否する。
処理（取り込み・解析・OCR・PDF出力）は別スレッドで実行し、実行中の編集は受け付けない。
処理は途中で中断でき、「再開」で続きから進む（途中状態は各工程が保存している）。
"""

from __future__ import annotations

import json
import mimetypes
import os
import secrets
import shutil
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2

from vbs import __version__, pipeline, receipt
from vbs.manifest import DEFAULT_SETTINGS, Project, ProjectError, atomic_write_json
from vbs.ocr import OcrError
from vbs.video import VideoError

UI_DIR = Path(__file__).resolve().parent / "ui"
APP_DIR = Path.home() / ".vbs"
RECENT = APP_DIR / "recent.json"
EDITABLE_SETTINGS = {"layout", "direction", "expected_pages", "page_height_mm", "ocr_lang", "ocr_psm",
                     "ocr_scale", "auto_export", "min_still_sec", "motion_threshold", "enhance",
                     "content_rotation", "dewarp", "margins", "candidate_still_both_ways",
                     "document", "receipt_width_mm"}
USER_ERRORS = (ProjectError, VideoError, OcrError, ValueError, KeyError)
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv"}
PHOTO_EXT = {".jpg", ".jpeg", ".png"}


class JobCancelled(BaseException):
    """中断の要求。BaseException なので途中の `except Exception` に捕まらない."""


def _coerce(key: str, v: Any) -> Any:
    """画面からの値（文字列のことがある）を設定の型に合わせる."""
    if v is None or v == "":
        if key in ("expected_pages", "motion_threshold"):
            return None
        raise ValueError(f"{key} は空にできません。")
    if key in ("auto_export", "candidate_still_both_ways"):
        return v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes")
    if key in ("ocr_psm", "expected_pages"):
        return int(v)
    if key == "document" and v not in ("book", "receipt"):
        raise ValueError("document は book か receipt です。")
    if key in ("page_height_mm", "min_still_sec", "motion_threshold", "receipt_width_mm"):
        return float(v)
    if key == "content_rotation":
        return "auto" if v == "auto" else int(v) % 360
    if key == "layout" and v not in ("spread", "single"):
        raise ValueError("layout は spread か single です。")
    if key == "direction" and v not in ("ltr", "rtl"):
        raise ValueError("direction は ltr か rtl です。")
    return v


def next_actions(message: str) -> list[dict[str, str]]:
    """エラーの内容から、次にできる操作を選ぶ（画面にボタンとして出す）."""
    m = message
    acts: list[dict[str, str]] = []
    if "空き容量" in m:
        acts.append({"label": "空きを作ってから再開", "action": "resume"})
    if "HDR" in m:
        acts.append({"label": "HDRのまま処理する（色は検証外）", "action": "allow_hdr"})
    if "見つかりません" in m or "ファイルがありません" in m or "内容が変わって" in m:
        acts.append({"label": "動画を入れ直す", "action": "goto_input"})
    if "OCR" in m or "Tesseract" in m:
        acts.append({"label": "再開（失敗ページは2回まで試し直す）", "action": "resume"})
    if "要確認" in m:
        acts.append({"label": "確認画面へ", "action": "goto_review"})
    if "開かれている" in m:
        acts.append({"label": "PDFを閉じてから出力し直す", "action": "export"})
    if not acts:
        acts.append({"label": "再開する", "action": "resume"})
    return acts


def load_recent() -> list[dict[str, Any]]:
    try:
        items = json.loads(RECENT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for it in items:
        if (Path(it["path"]) / "project.json").exists():
            out.append(it)
    return out


def remember(project: Project) -> None:
    APP_DIR.mkdir(exist_ok=True)
    items = [it for it in load_recent() if Path(it["path"]) != project.root]
    items.insert(0, {"path": str(project.root), "title": project.data.get("title") or project.root.name,
                     "opened": time.strftime("%Y-%m-%d %H:%M")})
    atomic_write_json(RECENT, items[:20])


class App:
    def __init__(self) -> None:
        self.project: Project | None = None
        self.stack = ExitStack()
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(18)
        self.job: dict[str, Any] = {"running": False}
        self.cancel = threading.Event()
        self.snapshot = "null"
        self._last_snap = 0.0

    # ---- プロジェクト --------------------------------------------------
    def open(self, root: Path) -> None:
        with self.lock:
            if self.job.get("running"):
                raise ProjectError("処理中は切り替えできません。")
            project = Project.load(root)
            self.stack.close()
            self.stack = ExitStack()
            self.stack.enter_context(project.lock())
            self.project = project
            pipeline.update_order(project)
            remember(project)
            self.job = {"running": False}
            self.refresh()

    def create(self, root: Path, settings: dict[str, Any], title: str | None) -> None:
        root = Path(root).expanduser()
        if (root / "project.json").exists():
            raise ProjectError(f"保存先にはすでにプロジェクトがあります: {root}（「開く」から開いてください）")
        if root.exists() and any(root.iterdir()):
            raise ProjectError(f"保存先のフォルダが空ではありません: {root}")
        project = Project.create(root, settings)
        if title:
            project.data["title"] = title
            project.save()
        self.open(root)

    def close(self) -> None:
        self.stack.close()

    def refresh(self) -> None:
        p = self.project
        if p is None:
            self.snapshot = json.dumps({"project": None, "version": __version__})
            return
        state = {
            "version": __version__,
            "root": str(p.root),
            "project": p.data,
            "summary": pipeline.summary(p),
            "review": pipeline.review_items(p),
            "natural_order": pipeline.natural_order(p),
            "labels": {k: v[0] for k, v in pipeline.WARNINGS.items()},
            "blocking": [k for k, v in pipeline.WARNINGS.items() if v[1]],
            "ocr_current": {pid: pipeline.ocr_current(p, pg) for pid, pg in p.data["pages"].items()},
            "pending": pipeline.pending_pages(p),
            "page_options": {s["id"]: {k: pipeline.page_option(p, s, k) for k in pipeline.PAGE_OPTION_DEFAULTS}
                             for s in p.data["segments"]},
            "receipts": ({pid: pipeline.receipt_fields(p, pg) for pid, pg in p.data["pages"].items()}
                         if pipeline.is_receipt(p) else None),
            "receipt_labels": receipt.FIELD_LABELS,
        }
        self.snapshot = json.dumps(state, ensure_ascii=False)

    # ---- 処理ジョブ ----------------------------------------------------
    def start_job(self, kind: str, **kw: Any) -> None:
        with self.lock:
            if self.project is None:
                raise ProjectError("プロジェクトが開かれていません。")
            if self.job.get("running"):
                raise ProjectError("処理中です。")
            self.cancel.clear()
            self.job = {"running": True, "kind": kind, "stage": "", "frac": 0.0, "msg": "開始",
                        "error": None, "actions": [], "cancelled": False, "result": None,
                        "started": time.time()}
        threading.Thread(target=self._run_job, args=(kind, kw), daemon=True).start()

    def _progress(self, stage: str, frac: float, msg: str) -> None:
        if self.cancel.is_set():
            raise JobCancelled()
        self.job.update(stage=stage, frac=frac, msg=msg)
        now = time.time()
        if now - self._last_snap > 1.0:
            self._last_snap = now
            self.refresh()

    def _run_job(self, kind: str, kw: dict[str, Any]) -> None:
        p = self.project
        assert p is not None
        try:
            if kind in ("import", "start"):
                paths = [Path(x) for x in kw.get("paths") or []]
                videos = [x for x in paths if x.suffix.lower() in VIDEO_EXT]
                photos = [x for x in paths if x.suffix.lower() in PHOTO_EXT]
                for x in paths:
                    if not x.is_file():
                        raise ProjectError(f"ファイルが見つかりません: {x}")
                    if x.suffix.lower() not in VIDEO_EXT | PHOTO_EXT:
                        raise ProjectError(f"対応していない形式です: {x.name}（動画 {' '.join(sorted(VIDEO_EXT))}、"
                                           f"写真 {' '.join(sorted(PHOTO_EXT))}）")
                if videos:
                    pipeline.add_videos(p, videos, copy=True, allow_hdr=bool(kw.get("allow_hdr")),
                                        progress=self._progress)
                for ph in photos:
                    pipeline.add_photo(p, ph, None)
                pipeline.update_order(p)
                p.save()
            if kind in ("start", "run"):
                res = pipeline.run_all(p, self._progress, export=kw.get("export"), force=kw.get("force", False))
                self.job["result"] = {"export": res["export"], "review": len(res["review"])}
            elif kind == "export":
                pipeline.update_order(p)
                rec = pipeline.export_pdf(p, force=kw.get("force", False), page_ids=kw.get("page_ids"))
                self.job["result"] = {"export": rec}
        except JobCancelled:
            self.job["cancelled"] = True
            self.job["msg"] = "中断しました。ここまでの結果は保存されています。「再開」で続きから進みます。"
        except USER_ERRORS as e:
            self.job["error"] = str(e)
            self.job["actions"] = next_actions(str(e))
        except OSError as e:
            self.job["error"] = f"書き込み・読み込みに失敗しました（{e}）。ここまでの結果は保存されています。"
            self.job["actions"] = next_actions("空き容量" if getattr(e, "errno", None) == 28 else str(e))
        except Exception as e:  # 予期しない失敗も画面へ返す
            traceback.print_exc()
            self.job["error"] = f"予期しないエラー: {type(e).__name__}: {e}"
            self.job["actions"] = next_actions("")
        finally:
            try:
                p.save()
            except Exception:
                traceback.print_exc()
            self.job["running"] = False
            self.job["finished"] = time.time()
            self.refresh()

    # ---- 編集操作 ------------------------------------------------------
    def action(self, body: dict[str, Any]) -> None:
        with self.lock:
            p = self.project
            if p is None:
                raise ProjectError("プロジェクトが開かれていません。")
            if self.job.get("running"):
                raise ProjectError("処理中は編集できません。中断するか、完了までお待ちください。")
            op = body.get("op")
            if op == "choose":
                seg = p.segment(body["segment"])
                if body["frame"] not in seg["candidates"]:
                    raise ProjectError("その区間の候補ではありません。")
                seg["chosen"] = body["frame"]
                seg["include"] = True
                seg["manual"] = True
            elif op == "include":
                target = body["id"]
                obj = p.data["pages"].get(target) or p.segment(target)
                if obj is not p.data["pages"].get(target) and body["value"] and not obj["candidates"]:
                    raise ProjectError("候補画像がない区間は採用できません。動画から選ぶか写真を追加してください。")
                obj["include"] = bool(body["value"])
                if target not in p.data["pages"]:
                    obj["manual"] = True
                    if body["value"]:
                        # 手動で採用した区間は、重複・短い静止の理由で止めない
                        obj["acknowledged"] = sorted(set(obj.get("acknowledged", [])) | set(obj["warnings"]))
                        obj["duplicate_of"] = None
            elif op in ("ack", "unack"):
                target = body["id"]
                if target == "page_count":
                    acks = set(p.data.get("acknowledged", []))
                    acks = acks | {"page_count"} if op == "ack" else acks - {"page_count"}
                    p.data["acknowledged"] = sorted(acks)
                else:
                    obj = (p.data["pages"].get(target)
                           or next((v for v in p.data["videos"] if v["id"] == target), None)
                           or p.segment(target))
                    if op == "ack":
                        codes = body.get("codes") or list(obj["warnings"]) + (
                            pipeline.ocr_warnings(obj) if target in p.data["pages"] else [])
                        obj["acknowledged"] = sorted(set(obj.get("acknowledged", [])) | set(codes))
                    else:
                        obj["acknowledged"] = []
            elif op == "image_only":
                pg = p.data["pages"][body["page"]]
                pg["image_only_ok"] = bool(body["value"])
                acks = set(pg.get("acknowledged", []))
                pg["acknowledged"] = sorted(acks | {"ocr_failed"} if body["value"] else acks - {"ocr_failed"})
            elif op == "geometry":
                pipeline.set_geometry(p, body["segment"], bbox=body.get("bbox"), gutter_ratio=body.get("gutter_ratio"))
            elif op == "clear_geometry":
                pipeline.clear_geometry(p, body["segment"])
            elif op == "page_options":
                pipeline.set_page_options(p, body["segment"], body.get("values") or {})
            elif op == "order":
                pipeline.set_order(p, list(body["order"]))
            elif op == "reset_order":
                pipeline.reset_order(p)
            elif op == "move_video":
                pipeline.move_video(p, body["video"], int(body["delta"]))
            elif op == "add_frame":
                pipeline.add_frame_at(p, body["video"], float(body["t"]), body.get("segment"))
            elif op == "settings":
                for k, v in (body.get("values") or {}).items():
                    if k not in EDITABLE_SETTINGS:
                        raise ProjectError(f"変更できない設定です: {k}")
                    p.settings[k] = _coerce(k, v)
            elif op == "rotate_video":
                pipeline.rotate_video_frames(p, body["video"], int(body["k"]))
            elif op == "sort_receipts":
                pipeline.sort_receipts(p, str(body.get("by")))
            elif op == "receipt_fields":
                pipeline.set_receipt_fields(p, body["page"], body.get("values") or {})
            elif op == "title":
                p.data["title"] = str(body["title"]).strip() or None
            else:
                raise ProjectError(f"不明な操作です: {op}")
            pipeline.update_order(p)
            p.save()
            self.refresh()


def default_dir() -> Path:
    docs = Path.home() / "Documents"
    return (docs if docs.exists() else Path.home()) / "VideoBookScanner"


def make_handler(app: App, port_ref: list[int]):
    class Handler(BaseHTTPRequestHandler):
        server_version = "vbs/" + __version__

        def log_message(self, fmt: str, *args: Any) -> None:  # 静かにする
            if os.environ.get("VBS_HTTP_LOG"):
                super().log_message(fmt, *args)

        # ---- 共通 -----------------------------------------------------
        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").lower()
            return host in {f"127.0.0.1:{port_ref[0]}", f"localhost:{port_ref[0]}"}

        def _token_ok(self, qs: dict[str, list[str]]) -> bool:
            t = self.headers.get("X-Token") or (qs.get("t") or [""])[0]
            return secrets.compare_digest(t, app.token)

        def _send(self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _err(self, msg: str, code: int = 400) -> None:
            self._json({"error": msg, "actions": next_actions(msg)}, code)

        def _body_json(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 10 * 1024 * 1024:
                raise ValueError("要求が大きすぎます。")
            return json.loads(self.rfile.read(n) or b"{}")

        def _guard(self) -> tuple[str, dict[str, list[str]]] | None:
            u = urllib.parse.urlsplit(self.path)
            qs = urllib.parse.parse_qs(u.query)
            if not self._host_ok():
                self._err("forbidden host", 403)
                return None
            if u.path not in ("/", "/index.html") and not self._token_ok(qs):
                self._err("token required", 403)
                return None
            return u.path, qs

        def _jpeg(self, img, extra: dict[str, str] | None = None) -> None:
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            self._send(200, enc.tobytes(), "image/jpeg", extra)

        # ---- GET ------------------------------------------------------
        def do_GET(self) -> None:
            g = self._guard()
            if not g:
                return
            path, qs = g
            try:
                if path in ("/", "/index.html"):
                    body = (UI_DIR / "index.html").read_bytes()
                    self._send(200, body, "text/html; charset=utf-8",
                               {"Content-Security-Policy": "default-src 'self'; img-src 'self' data: blob:; "
                                                           "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"})
                elif path == "/api/state":
                    self._send(200, app.snapshot.encode("utf-8"), "application/json; charset=utf-8")
                elif path == "/api/job":
                    self._json(app.job)
                elif path == "/api/home":
                    self._json({"recent": load_recent(), "default_dir": str(default_dir()),
                                "settings": DEFAULT_SETTINGS})
                elif path == "/api/preview":
                    self._preview(qs)
                elif path == "/api/page_view":
                    self._page_view(qs)
                elif path.startswith("/files/"):
                    self._file(urllib.parse.unquote(path[len("/files/"):]))
                else:
                    self._err("not found", 404)
            except USER_ERRORS as e:
                self._err(str(e))
            except Exception as e:
                traceback.print_exc()
                self._err(f"{type(e).__name__}: {e}", 500)

        def _file(self, rel: str) -> None:
            p = app.project
            if p is None:
                return self._err("no project", 404)
            root = p.root.resolve()
            target = (root / rel).resolve()
            if root not in target.parents or not target.is_file():
                return self._err("not found", 404)
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self._send(200, target.read_bytes(), ctype)

        def _preview(self, qs: dict[str, list[str]]) -> None:
            p = app.project
            if p is None:
                return self._err("no project", 404)
            t = float(qs["time"][0])
            step = int((qs.get("step") or ["0"])[0])
            w = int(qs.get("w", ["960"])[0])
            with app.lock:
                if step:
                    ft, pts, img = pipeline.step_frame(p, qs["video"][0], t, step, width=w)
                else:
                    ft, pts, img = pipeline.preview_at(p, qs["video"][0], t, width=w)
            self._jpeg(img, {"X-Frame-Time": f"{ft:.4f}"})

        def _page_view(self, qs: dict[str, list[str]]) -> None:
            p = app.project
            if p is None:
                return self._err("no project", 404)
            pid = qs["pid"][0]
            mode = (qs.get("mode") or ["final"])[0]
            if mode not in ("frame", "plain", "final"):
                return self._err("mode は frame / plain / final です。")
            w = int(qs.get("w", ["1400"])[0])
            pg = p.data["pages"][pid]
            cache = p.path("thumbs") / "views" / f"{pid}_{pg['revision']}_{mode}_{w}.jpg"
            if cache.exists():
                return self._send(200, cache.read_bytes(), "image/jpeg")
            with app.lock:
                img = pipeline.render_page_view(p, pid, mode, width=w)
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(enc.tobytes())
            self._send(200, enc.tobytes(), "image/jpeg")

        # ---- POST -----------------------------------------------------
        def do_POST(self) -> None:
            g = self._guard()
            if not g:
                return
            path, qs = g
            if path in ("/", "/index.html"):
                return self._err("token required", 403)
            try:
                if path == "/api/upload":
                    return self._upload(qs)
                if path == "/api/receipts_import":
                    return self._receipts_import()
                body = self._body_json()
                if path == "/api/action":
                    app.action(body)
                elif path == "/api/run":
                    app.start_job("run", force=bool(body.get("force")), export=body.get("export"))
                elif path == "/api/start":
                    app.start_job("start", paths=body.get("paths") or [], allow_hdr=bool(body.get("allow_hdr")),
                                  export=body.get("export"))
                elif path == "/api/import":
                    app.start_job("import", paths=body.get("paths") or [], allow_hdr=bool(body.get("allow_hdr")))
                elif path == "/api/cancel":
                    if app.job.get("running"):
                        app.cancel.set()
                elif path == "/api/export":
                    app.start_job("export", force=bool(body.get("force")), page_ids=body.get("page_ids"))
                elif path == "/api/open":
                    app.open(Path(body["path"]).expanduser())
                elif path == "/api/create":
                    settings = {k: _coerce(k, v) for k, v in (body.get("settings") or {}).items()
                                if k in DEFAULT_SETTINGS and k in EDITABLE_SETTINGS}
                    app.create(Path(body["path"]), settings, body.get("title"))
                elif path == "/api/reveal":
                    self._reveal(body.get("what", "output"))
                else:
                    return self._err("not found", 404)
                self._send(200, app.snapshot.encode("utf-8"), "application/json; charset=utf-8")
            except USER_ERRORS as e:
                self._err(str(e))
            except Exception as e:
                traceback.print_exc()
                self._err(f"{type(e).__name__}: {e}", 500)

        def _upload(self, qs: dict[str, list[str]]) -> None:
            """ブラウザにドロップされたファイルを受け取る（場所が分からないので中身を送ってもらう）."""
            p = app.project
            if p is None:
                return self._err("プロジェクトが開かれていません。")
            if app.job.get("running"):
                return self._err("処理中は追加できません。")
            name = Path((qs.get("name") or ["upload"])[0]).name
            suffix = Path(name).suffix.lower()
            kind = "video" if suffix in VIDEO_EXT else "photo" if suffix in PHOTO_EXT else None
            if kind is None:
                return self._err(f"対応していない形式です: {suffix}（動画 {' '.join(sorted(VIDEO_EXT))}、"
                                 f"写真 {' '.join(sorted(PHOTO_EXT))}）")
            n = int(self.headers.get("Content-Length") or 0)
            dest_dir = p.path("videos") if kind == "video" else p.path("jobs")
            if shutil.disk_usage(p.root).free < n * 1.1 + 2 * 1024**3:
                return self._err("空き容量が不足しています。")
            tmp = dest_dir / f"upload_{secrets.token_hex(6)}{suffix}"
            left = n
            try:
                with open(tmp, "wb") as f:
                    while left > 0:
                        chunk = self.rfile.read(min(8 << 20, left))
                        if not chunk:
                            raise OSError("アップロードが途中で切れました。")
                        f.write(chunk)
                        left -= len(chunk)
                with app.lock:
                    if kind == "video":
                        allow_hdr = (qs.get("allow_hdr") or ["0"])[0] == "1"
                        pipeline.add_videos(p, [tmp], copy=True, allow_hdr=allow_hdr, names=[name])
                    else:
                        after = (qs.get("after") or [""])[0] or None
                        pipeline.add_photo(p, tmp, after, original_name=name)
                    pipeline.update_order(p)
                    p.save()
                    app.refresh()
            except USER_ERRORS as e:
                return self._err(str(e))
            finally:
                # 動画は成功時に videos/ 内で改名済み。失敗時と写真の一時ファイルはここで消す
                tmp.unlink(missing_ok=True)
            self._send(200, app.snapshot.encode("utf-8"), "application/json; charset=utf-8")

        def _receipts_import(self) -> None:
            """直したレシートの一覧（CSV）を取り込む."""
            from vbs.receipt_export import import_csv

            p = app.project
            if p is None:
                return self._err("プロジェクトが開かれていません。")
            n = int(self.headers.get("Content-Length") or 0)
            if n > 20 * 1024 * 1024:
                return self._err("一覧のファイルが大きすぎます。")
            raw = self.rfile.read(n)
            for enc in ("utf-8-sig", "cp932"):
                try:
                    text = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return self._err("一覧の文字コードを読めません（UTF-8 か Shift_JIS で保存してください）。")
            try:
                with app.lock:
                    if app.job.get("running"):
                        return self._err("処理中は読み込めません。")
                    res = import_csv(p, text)
                    app.refresh()
            except USER_ERRORS as e:
                return self._err(str(e))
            state = json.loads(app.snapshot)
            state["import_result"] = res
            self._json(state)

        def _reveal(self, what: str) -> None:
            p = app.project
            if p is None:
                raise ProjectError("プロジェクトが開かれていません。")
            if what == "pdf":
                exp = p.data["exports"][-1] if p.data["exports"] else None
                if not exp:
                    raise ProjectError("まだPDFを出力していません。")
                target = Path(exp["path"])
            else:
                target = p.path("output") if what == "output" else p.root
            if sys.platform == "win32":
                os.startfile(str(target))  # type: ignore[attr-defined]

    return Handler


def serve(root: Path | None, port: int = 0, open_browser: bool = True) -> None:
    app = App()
    if root:
        app.open(Path(root))
    else:
        app.refresh()
    port_ref = [0]
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, port_ref))
    port_ref[0] = httpd.server_address[1]
    url = f"http://127.0.0.1:{port_ref[0]}/#t={app.token}"
    print(f"画面: {url}\n（このウィンドウを閉じると終了します）", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        app.close()
