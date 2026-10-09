"""画面のAPIを、合成動画で「動画を入れる → 処理 → 直す → 反映 → PDF」まで通す."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from vbs import server
from vbs.ocr import OcrError, find_tessdata, find_tesseract

ROOT = Path(__file__).resolve().parent.parent


def _have_ocr() -> bool:
    try:
        find_tesseract()
        find_tessdata("jpn")
        return True
    except OcrError:
        return False


@pytest.fixture()
def ui(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(server, "APP_DIR", tmp_path / "appdir")
    monkeypatch.setattr(server, "RECENT", tmp_path / "appdir" / "recent.json")
    app = server.App()
    app.refresh()
    port = [0]
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app, port))
    port[0] = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port[0]}"

    def call(path: str, body=None, token=True, raw=False):
        req = urllib.request.Request(base + path, method="GET" if body is None else "POST",
                                     data=None if body is None else json.dumps(body).encode())
        if token:
            req.add_header("X-Token", app.token)
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
            return (data, dict(r.headers)) if raw else json.loads(data)

    def wait():
        for _ in range(1200):
            job = call("/api/job")
            if not job["running"]:
                return job
            time.sleep(0.5)
        raise TimeoutError

    yield call, wait
    httpd.shutdown()
    httpd.server_close()
    app.close()


def test_token_required(ui):
    call, _ = ui
    with pytest.raises(urllib.error.HTTPError) as e:
        call("/api/state", token=False)
    assert e.value.code == 403
    page, _ = call("/", token=False, raw=True)
    assert b"Video Book Scanner" in page


@pytest.mark.skipif(not _have_ocr(), reason="Tesseract/言語データなし")
def test_ui_flow(ui, tmp_path: Path):
    call, wait = ui
    video = tmp_path / "v.mp4"
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_test_video.py"), str(video),
                    "--spreads", "2", "--seed", "5"], check=True, capture_output=True)
    # 1. 動画を入れる（空でないフォルダには作らない）
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "x.txt").write_text("x")
    with pytest.raises(urllib.error.HTTPError):
        call("/api/create", {"path": str(tmp_path / "busy"), "settings": {}})
    call("/api/create", {"path": str(tmp_path / "book"), "title": "テスト本",
                         "settings": {"direction": "ltr", "layout": "spread", "auto_export": "true"}})
    call("/api/start", {"paths": [str(video)], "export": None})
    job = wait()
    assert job["error"] is None and not job["cancelled"]
    st = call("/api/state")
    assert st["summary"]["pages"] == 4 and st["pending"] == []
    assert job["result"]["export"] or st["review"]  # 要確認がなければ自動でPDFまで
    home = call("/api/home")
    assert home["recent"][0]["title"] == "テスト本"

    # 2. 前後のコマ（時刻は time、トークンの t とぶつからない）
    seg = next(s for s in st["project"]["segments"]
               if s["include"] and st["project"]["frames"][s["chosen"]]["time_sec"] > 0.5)
    t0 = st["project"]["frames"][seg["chosen"]]["time_sec"]
    _, h = call(f"/api/preview?video={seg['video_id']}&time={t0}&step=1&w=320", raw=True)
    assert float(h["X-Frame-Time"]) > t0
    _, h = call(f"/api/preview?video={seg['video_id']}&time={t0}&step=-1&w=320", raw=True)
    assert float(h["X-Frame-Time"]) < t0

    # 3. 補正前後の表示
    pid = st["project"]["order"][0]
    for mode in ("final", "plain", "frame"):
        img, h = call(f"/api/page_view?pid={pid}&mode={mode}&w=400", raw=True)
        assert h["Content-Type"] == "image/jpeg" and len(img) > 1000

    # 4. 見開きだけ補正を変える → その見開きだけ未反映 → 反映
    st = call("/api/action", {"op": "page_options", "segment": seg["id"], "values": {"dewarp": "off"}})
    assert st["page_options"][seg["id"]]["dewarp"] == "off"
    assert sorted(st["pending"]) == sorted(p for p in st["project"]["order"]
                                           if st["project"]["pages"][p]["segment_id"] == seg["id"])
    call("/api/run", {"export": False})
    job = wait()
    assert job["error"] is None
    assert call("/api/state")["pending"] == []

    # 5. PDF出力
    call("/api/export", {"force": True})
    job = wait()
    rec = job["result"]["export"]
    assert rec["pages"] == 4 and Path(rec["path"]).exists()
