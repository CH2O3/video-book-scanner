"""試験用の障害注入とイベント記録.

環境変数で有効にしたときだけ働く。通常の実行には影響しない。

VBS_FAULT（「;」区切りで複数可）
    crash:<地点>[:<回数>]       地点を<回数>回目に通ったとき、その場でプロセスを終了する（既定1回目）
    fail:<地点>[:<対象>]        地点で OSError を起こす（対象を指定するとその名前を含むときだけ）
    timeout:<地点>[:<対象>]     地点で時間切れを起こす
    slow:<地点>:<秒>            地点で指定秒数だけ待つ（外から止める試験の時間稼ぎ）
VBS_EVENTS
    進捗・イベントを1行1JSONで追記するファイル
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from typing import Any

_counts: Counter[str] = Counter()
_lock = threading.Lock()


class InjectedTimeout(Exception):
    pass


def _specs() -> list[list[str]]:
    raw = os.environ.get("VBS_FAULT", "")
    return [s.split(":") for s in raw.split(";") if s.strip()]


def point(name: str, target: str = "") -> None:
    """障害注入の地点。VBS_FAULT が設定されていなければ何もしない."""
    specs = _specs()
    if not specs:
        return
    with _lock:
        _counts[name] += 1
        n = _counts[name]
    for spec in specs:
        kind, where = spec[0], spec[1] if len(spec) > 1 else ""
        if where != name:
            continue
        arg = spec[2] if len(spec) > 2 else ""
        if kind == "crash" and n == int(arg or 1):
            event("fault", kind="crash", point=name, count=n)
            os._exit(99)
        if kind == "fail" and (not arg or arg in target):
            event("fault", kind="fail", point=name, target=target)
            raise OSError(28, f"注入された書込み失敗（{name} {target}）")
        if kind == "timeout" and (not arg or arg in target):
            event("fault", kind="timeout", point=name, target=target)
            raise InjectedTimeout(f"注入された時間切れ（{name} {target}）")
        if kind == "slow":
            time.sleep(float(arg or 1))


def event(_event: str, **data: Any) -> None:
    path = os.environ.get("VBS_EVENTS")
    if not path:
        return
    rec = {"t": round(time.time(), 3), "pid": os.getpid(), "event": _event, **data}
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
