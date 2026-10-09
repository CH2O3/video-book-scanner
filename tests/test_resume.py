"""途中保存からの再開が、連続実行と同じ結果になるか（小さな合成動画で）."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vbs.extract import analyze_motion

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("rv")
    out = d / "v.mp4"
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_test_video.py"), str(out), "--spreads", "3",
                    "--vfr", "--seed", "5"], check=True, capture_output=True)
    return out


def test_pass1_resume_matches_continuous(video: Path, tmp_path: Path, monkeypatch):
    import vbs.extract as E

    ref = analyze_motion(video)
    ck = tmp_path / "p1.npz"
    monkeypatch.setattr(E, "CHECKPOINT_EVERY_SEC", 1.0)
    # 途中で止まったことにする：途中保存の地点で例外を出して中断
    calls = {"n": 0}

    def stop_at_third():
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(E.faults, "point", lambda name, target="": stop_at_third() if name == "extract.pass1.checkpoint" else None)
    with pytest.raises(KeyboardInterrupt):
        analyze_motion(video, checkpoint=ck, checkpoint_key="k")
    assert ck.exists()
    monkeypatch.setattr(E.faults, "point", lambda name, target="": None)
    got = analyze_motion(video, checkpoint=ck, checkpoint_key="k")
    assert np.array_equal(ref.pts, got.pts)
    assert np.allclose(ref.motion, got.motion)
    assert np.allclose(ref.sharp, got.sharp)
    assert len(got.sigs) == len(ref.pts)
    k = len(ref.pts) // 2
    assert np.array_equal(ref.sigs[k], got.sigs[k])
    # 鍵（設定）が違えば途中状態は使わない
    again = analyze_motion(video, checkpoint=ck, checkpoint_key="other")
    assert np.array_equal(again.pts, ref.pts)


def test_fault_injection_points(tmp_path: Path, monkeypatch):
    import json

    from vbs import faults

    ev = tmp_path / "ev.jsonl"
    monkeypatch.setenv("VBS_EVENTS", str(ev))
    monkeypatch.setenv("VBS_FAULT", "fail:write:pages;timeout:ocr.page:s0003")
    faults.point("write", "frames/f0001.jpg")  # 対象外は何もしない
    with pytest.raises(OSError):
        faults.point("write", "pages/s0001-L.jpg")
    with pytest.raises(faults.InjectedTimeout):
        faults.point("ocr.page", "pages/s0003-R.jpg")
    recs = [json.loads(l) for l in ev.read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in recs] == ["fail", "timeout"]
    monkeypatch.delenv("VBS_FAULT")
    faults.point("write", "pages/s0001-L.jpg")  # 無効なら何もしない


def test_ocr_timeout_counts_awake_time_only():
    import subprocess as sp

    from vbs.ocr import _run_awake_timeout, awake_seconds

    a = awake_seconds()
    with pytest.raises(sp.TimeoutExpired):
        _run_awake_timeout([sys.executable, "-c", "import time; time.sleep(30)"], None, 1.5)
    assert awake_seconds() - a < 10
    assert _run_awake_timeout([sys.executable, "-c", "print('ok')"], None, 10).stdout.strip() == "ok"


def test_grab_frames_after_remux_concat(video: Path, tmp_path: Path):
    """再多重化でつないだ動画でも、指定したフレームをすべて取り出せる（シークの行き過ぎ対策）."""
    from vbs.bench.generate import concat_video
    from vbs.video import grab_frames, iter_frames

    out = tmp_path / "c.mp4"
    concat_video(video, out, minutes=0.5)
    pts = [p for _, p, _ in iter_frames(out, width=32, fmt="gray")]
    want = pts[::5]
    got = grab_frames(out, want)
    assert sorted(got) == sorted(want)
