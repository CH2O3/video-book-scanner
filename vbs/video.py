"""動画の検査とフレーム取得（PyAV）.

フレームの位置は表示時刻PTS（秒＝pts×time_base）で扱い、
フレーム番号÷fpsでは計算しない（可変フレームレート対応、仕様6.1）。
"""

from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import av
import numpy as np

# 伝達特性（ITU-T H.273 の番号）。PyAV が名前を返さない場合に使う
TRC_NAMES = {1: "bt709", 2: "unknown", 4: "gamma22", 5: "gamma28", 6: "smpte170m", 7: "smpte240m",
             8: "linear", 13: "iec61966-2-1", 14: "bt2020-10", 15: "bt2020-12",
             16: "smpte2084", 18: "arib-std-b67"}
PRIMARIES_NAMES = {1: "bt709", 5: "bt470bg", 6: "smpte170m", 9: "bt2020", 12: "smpte432"}


class VideoError(RuntimeError):
    pass


def sha256_file(path: Path, chunk: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _name(v: Any, table: dict[int, str] | None = None) -> str | None:
    if v is None:
        return None
    name = getattr(v, "name", None)
    if name:
        return str(name).lower()
    try:
        n = int(v)
    except (TypeError, ValueError):
        return str(v).lower()
    return (table or {}).get(n, str(n))


def probe(path: Path) -> dict[str, Any]:
    """映像ストリームを調べる。拡張子ではなく中身で判定する（仕様6.1）。"""
    try:
        container = av.open(str(path))
    except av.error.FFmpegError as e:  # type: ignore[attr-defined]
        raise VideoError(f"動画として開けません: {path} ({e})") from e
    with container:
        if not container.streams.video:
            raise VideoError(f"映像ストリームがありません: {path}")
        s = container.streams.video[0]
        cc = s.codec_context
        info: dict[str, Any] = {
            "container": container.format.name,
            "codec": cc.name,
            "width": cc.width,
            "height": cc.height,
            "pix_fmt": cc.pix_fmt,
            "time_base": str(s.time_base),
            "avg_rate": str(s.average_rate) if s.average_rate else None,
            "duration_sec": None,
            "rotation": 0,
            "color_trc": _name(getattr(cc, "color_trc", None), TRC_NAMES),
            "color_primaries": _name(getattr(cc, "color_primaries", None), PRIMARIES_NAMES),
            "colorspace": _name(getattr(cc, "colorspace", None)),
        }
        if s.duration is not None and s.time_base is not None:
            info["duration_sec"] = float(s.duration * s.time_base)
        elif container.duration is not None:
            info["duration_sec"] = container.duration / av.time_base
        # 回転とカラー特性は最初のフレームで確定させる
        try:
            for frame in container.decode(s):
                info["rotation"] = int(getattr(frame, "rotation", 0) or 0)
                trc = _name(getattr(frame, "color_trc", None), TRC_NAMES)
                if trc:
                    info["color_trc"] = trc
                break
            else:
                raise VideoError(f"デコードできるフレームがありません: {path}")
        except av.error.FFmpegError as e:  # type: ignore[attr-defined]
            raise VideoError(f"デコードに失敗しました: {path} ({e})") from e
    trc = (info["color_trc"] or "").lower()
    bit10 = any(t in (info["pix_fmt"] or "") for t in ("10le", "10be", "p010", "12le"))
    # PQ は輝度の絶対値で符号化され、そのまま8bitにすると紙面が暗く潰れる → 既定で拒否。
    # HLG はSDR表示との互換を前提にした方式で、紙と文字の判読に支障がないことを
    # 実写サンプル（iPhone, 2026-10-08）で確認したため、警告付きでそのまま扱う。
    info["hdr_kind"] = {"smpte2084": "pq", "arib-std-b67": "hlg"}.get(trc)
    info["hdr"] = info["hdr_kind"] == "pq"
    info["high_bit_depth"] = bit10
    return info


def rotate_k(rotation_deg: int) -> int:
    """表示行列の回転角(反時計回り)を np.rot90 の k に変換する."""
    return int(round(rotation_deg / 90.0)) % 4


def apply_rotation(img: np.ndarray, k: int) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(img, k)) if k else img


def iter_frames(
    path: Path,
    width: int | None = None,
    fmt: str = "gray",
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> Iterator[tuple[float, int, np.ndarray]]:
    """(時刻秒, pts, 画像) を順に返す。width を与えると縮小して返す。

    画像は動画の回転情報を適用済み（表示上の向き）。
    デコードできないパケットは飛ばし、例外は呼び出し側へ伝える。
    """
    with av.open(str(path)) as container:
        s = container.streams.video[0]
        s.thread_type = "AUTO"
        tb: Fraction = s.time_base
        if start_sec is not None and start_sec > 0:
            container.seek(int(start_sec / tb), stream=s, backward=True, any_frame=False)
        k = None
        for frame in container.decode(s):
            if frame.pts is None:
                continue
            t = float(frame.pts * tb)
            if start_sec is not None and t < start_sec - 1e-6:
                continue
            if end_sec is not None and t > end_sec + 1e-6:
                break
            if k is None:
                k = rotate_k(int(getattr(frame, "rotation", 0) or 0))
            if width:
                # 回転後の幅が width になるように縮小する
                src_w = frame.height if k % 2 else frame.width
                src_h = frame.width if k % 2 else frame.height
                h = max(2, int(round(src_h * width / src_w)))
                tw, th = (h, width) if k % 2 else (width, h)
                img = frame.reformat(width=tw, height=th, format=fmt).to_ndarray()
            else:
                img = frame.to_ndarray(format=fmt)
            yield t, int(frame.pts), apply_rotation(img, k)


SEEK_BACKOFF_SEC = (0.0, 1.0, 3.0, 10.0, None)  # None は先頭から読む


def grab_frames(path: Path, pts_list: list[int], fmt: str = "rgb24") -> dict[int, np.ndarray]:
    """指定PTSのフレームを元解像度で取り出す。

    PTS順に並べ、近いものはまとめて前方へデコードする。
    シークは目的より後ろのキーフレームに着くことがある（再多重化したMP4で確認）。
    最初に出てきたフレームが目的より後ろなら、もっと手前から読み直す。
    それでも見つからないPTSは結果に含めない（呼び出し側で欠落として扱う）。
    """
    want = sorted(set(pts_list))
    out: dict[int, np.ndarray] = {}
    if not want:
        return out
    with av.open(str(path)) as container:
        s = container.streams.video[0]
        s.thread_type = "AUTO"
        tb: Fraction = s.time_base
        rate = float(s.average_rate or 30)
        i = 0
        while i < len(want):
            target = want[i]
            start_i = i
            for back in SEEK_BACKOFF_SEC:
                i = start_i
                if back is None:
                    container.seek(0, stream=s, backward=True, any_frame=False)
                else:
                    container.seek(max(0, target - int(back / tb)), stream=s, backward=True, any_frame=False)
                k = None
                first = None
                for frame in container.decode(s):
                    if frame.pts is None:
                        continue
                    if first is None:
                        first = frame.pts
                        if first > target:
                            break  # 行き過ぎた。もっと手前から読み直す
                    if k is None:
                        k = rotate_k(int(getattr(frame, "rotation", 0) or 0))
                    while i < len(want) and frame.pts > want[i]:
                        i += 1  # 該当PTSのフレームが存在しなかった
                    if i >= len(want):
                        break
                    if frame.pts == want[i]:
                        out[want[i]] = apply_rotation(frame.to_ndarray(format=fmt), k)
                        i += 1
                        if i >= len(want):
                            break
                        # 次の目標が遠ければシークし直す（3秒以上先）
                        if float((want[i] - frame.pts) * tb) > 3.0 + 1.0 / rate:
                            break
                else:
                    if first is not None and first <= target:
                        i = len(want)  # 末尾まで読んだ
                if first is not None and first <= target:
                    break  # この読み直しで目的の位置を通過できた
            if i == start_i:
                i += 1  # どうしても見つからない
    return out
