"""見開きの抽出（仕様7章）.

1. 低解像度で全フレームを走査し、約0.2秒前との差分（動き量）と鮮明さを記録する。
2. 動きが閾値未満の連続区間を「静止区間」とし、秒単位の最短時間で選別する。
3. 各区間から鮮明な候補を最大3枚選び、元解像度で取り出してディスクへ保存する。
4. 取り出した画像を解析時の縮小画像と照合し、同じ場面かを確認する。
5. 隣り合う区間が同じ見開きかを比較し、重複候補としてまとめる（参照は保持）。

長い動画のために:
- 元解像度の候補は区間ごとに取り出して保存し、すぐ手放す（メモリが区間数に比例しない）
- 照合用の縮小画像はディスクへ追記する（メモリが動画の長さに比例しない）
- 1回目・2回目の走査とも途中状態を jobs/ に保存し、強制終了後に続きから再開する
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from vbs import faults
from vbs.imgio import imwrite
from vbs.video import grab_frames, iter_frames

ProgressFn = Callable[[float, str], None]

LAG_SEC = 0.2          # 動き量を測る時間差
SHARP_WIDTH = 480      # 鮮明さ評価の幅
SIG_WIDTH = 32         # 照合用シグネチャの幅
MERGE_GAP_SEC = 0.12   # この長さ以下の動きの瞬断は静止区間をつなぐ
SHORT_STILL_SEC = 0.25 # これ以上 min_still 未満は「短い静止」として残す
CAND_SPACING_SEC = 0.15
GAP_WARN_SEC = 1.0     # タイムスタンプがこれ以上飛んだら読めない区間として記録
DUP_INLIERS = 120      # 対応点がこれ以上なら同じ見開き
SUSPECT_INLIERS = 60   # この間は統合せず確認へ
MIN_KEYPOINTS = 60     # 特徴点がこれ未満（白紙など）なら判断しない
CHECKPOINT_EVERY_SEC = 20.0  # 1回目の走査の途中状態を保存する間隔（動画時間）
RESUME_OVERLAP_SEC = 2.0     # 再開時に保存時刻より手前から読み直す長さ
TIME_EPS = 1e-6              # 時刻を比べるときの余裕（秒）
EXTRACT_VERSION = 5


@dataclass
class Analysis:
    times: np.ndarray
    pts: np.ndarray
    motion: np.ndarray
    sharp: np.ndarray
    bright: np.ndarray
    sigs: Any  # np.ndarray または SigStore（ディスク上）
    threshold: float
    gaps: list[tuple[float, float]] = field(default_factory=list)
    error: str | None = None
    width: int = 0
    height: int = 0


def _sig(gray: np.ndarray) -> np.ndarray:
    h = max(2, int(round(gray.shape[0] * SIG_WIDTH / gray.shape[1])))
    return cv2.resize(gray, (SIG_WIDTH, h), interpolation=cv2.INTER_AREA)


class SigStore:
    """照合用の縮小画像をディスクへ追記して保持する."""

    def __init__(self, path: Path, shape: tuple[int, int] | None = None, count: int = 0):
        self.path = Path(path)
        self.shape = tuple(shape) if shape else None
        self.count = count
        if self.shape:
            # 再開時は確定済みの件数まで切り詰める（途中まで書いた分を捨てる）
            with open(self.path, "r+b") as f:
                f.truncate(self.count * self._size)
        else:
            self.path.write_bytes(b"")

    @property
    def _size(self) -> int:
        return int(self.shape[0] * self.shape[1])

    def append(self, sig: np.ndarray) -> None:
        if self.shape is None:
            self.shape = tuple(sig.shape)
        if tuple(sig.shape) != self.shape:
            sig = cv2.resize(sig, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)
        with open(self.path, "ab") as f:
            f.write(np.ascontiguousarray(sig, dtype=np.uint8).tobytes())
        self.count += 1

    def __getitem__(self, i: int) -> np.ndarray:
        with open(self.path, "rb") as f:
            f.seek(int(i) * self._size)
            buf = f.read(self._size)
        return np.frombuffer(buf, np.uint8).reshape(self.shape)

    def __len__(self) -> int:
        return self.count


def analyze_motion(
    video_path: Path,
    analysis_width: int = 160,
    threshold: float | None = None,
    duration_hint: float | None = None,
    progress: ProgressFn | None = None,
    checkpoint: Path | None = None,
    checkpoint_key: str = "",
) -> Analysis:
    """全フレームを逐次読み、動き量・鮮明さを記録する.

    checkpoint を渡すと一定間隔で状態を保存し、次回はその続きから再開する。
    再開時は保存時刻より手前から読み直して直前のフレームを復元し（重ね読み）、
    保存済みの時刻までのフレームは記録し直さない。
    """
    times: list[float] = []
    pts: list[int] = []
    motion: list[float] = []
    sharp: list[float] = []
    bright: list[float] = []
    gaps: list[tuple[float, float]] = []
    size: tuple[int, int] = (0, 0)
    if checkpoint:
        sig_path = Path(str(checkpoint) + ".sigs")
    else:  # 途中保存なし（試験など）。元動画の隣には書かない
        import tempfile

        fd, name = tempfile.mkstemp(suffix=".sigs")
        os.close(fd)
        sig_path = Path(name)
    sigs: SigStore | None = None
    resume_after = None
    if checkpoint and checkpoint.exists() and sig_path.exists():
        try:
            # with で閉じる（開いたままだと Windows では次の保存で置き換えられない）
            with np.load(checkpoint, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                arrays = {k: z[k].tolist() for k in ("times", "pts", "motion", "sharp", "bright", "gaps")}
            if meta.get("key") == checkpoint_key:
                times, pts = arrays["times"], arrays["pts"]
                motion, sharp, bright = arrays["motion"], arrays["sharp"], arrays["bright"]
                gaps = [tuple(g) for g in arrays["gaps"]]
                size = tuple(meta["size"])
                sigs = SigStore(sig_path, meta["sig_shape"] or None, len(times)) if times else None
                resume_after = times[-1] if times else None
                faults.event("resume", stage="extract.pass1", frames=len(times), at=resume_after)
        except (OSError, ValueError, KeyError):
            times, pts, motion, sharp, bright, gaps = [], [], [], [], [], []
            sigs, resume_after = None, None
    if sigs is None:
        times, pts, motion, sharp, bright, gaps, resume_after = [], [], [], [], [], [], None
        sigs = SigStore(sig_path)

    def save_checkpoint() -> None:
        if not checkpoint:
            return
        meta = {"key": checkpoint_key, "size": list(size), "sig_shape": list(sigs.shape or ())}
        tmp = checkpoint.with_name(checkpoint.name + ".tmp.npz")
        np.savez(tmp, times=np.asarray(times, np.float64), pts=np.asarray(pts, np.int64),
                 motion=np.asarray(motion, np.float32), sharp=np.asarray(sharp, np.float32),
                 bright=np.asarray(bright, np.float32), gaps=np.asarray(gaps, np.float64).reshape(-1, 2),
                 meta=np.asarray(json.dumps(meta)))
        os.replace(tmp, checkpoint)
        faults.point("extract.pass1.checkpoint")

    recent: deque[tuple[float, np.ndarray]] = deque()
    error = None
    last_report = -1.0
    last_ckpt = times[-1] if times else 0.0
    start = max(0.0, resume_after - RESUME_OVERLAP_SEC) if resume_after is not None else None
    try:
        for t, p, img in iter_frames(video_path, width=SHARP_WIDTH, fmt="gray", start_sec=start):
            if not size[0]:
                size = (img.shape[1], img.shape[0])
            small = cv2.resize(
                img,
                (analysis_width, max(2, int(round(img.shape[0] * analysis_width / img.shape[1])))),
                interpolation=cv2.INTER_AREA,
            )
            g = cv2.GaussianBlur(small, (5, 5), 0).astype(np.float32)
            recent.append((t, g))
            # 時刻の比較に小さな余裕を持たせる。ちょうど LAG_SEC 前のフレームがあるとき、
            # 浮動小数の誤差で「等しい」かどうかが動画内の位置によって変わり、
            # 同じ映像でも基準フレームが1つずれて結果が変わっていた（連結動画の照合で確認）
            while len(recent) > 1 and recent[1][0] <= t - LAG_SEC + TIME_EPS:
                recent.popleft()
            if resume_after is not None and t <= resume_after + 1e-9:
                continue  # 重ね読み：直前のフレームを復元するだけ（記録済み）
            if times and t - times[-1] > GAP_WARN_SEC:
                gaps.append((times[-1], t))
            ref = recent[0][1]
            motion.append(float(np.mean(np.abs(g - ref))))
            lap = cv2.Laplacian(img, cv2.CV_32F)
            sharp.append(float(lap.var()))
            bright.append(float(small.mean()))
            sigs.append(_sig(img))
            times.append(t)
            pts.append(p)
            if t - last_ckpt >= CHECKPOINT_EVERY_SEC:
                last_ckpt = t
                save_checkpoint()
            if progress and duration_hint and t - last_report >= 2.0:
                last_report = t
                progress(min(t / duration_hint, 1.0), f"動きを解析中 {t:.0f}/{duration_hint:.0f}秒")
    except Exception as e:  # 途中破損：読めた範囲までで結果を返す（IN-06）
        error = f"{type(e).__name__}: {e}"
        end = times[-1] if times else 0.0
        gaps.append((end, duration_hint or end))
    save_checkpoint()
    m = np.asarray(motion, dtype=np.float32)
    thr = threshold if threshold is not None else auto_threshold(m, np.asarray(times, dtype=np.float64))
    return Analysis(
        times=np.asarray(times, dtype=np.float64),
        pts=np.asarray(pts, dtype=np.int64),
        motion=m,
        sharp=np.asarray(sharp, dtype=np.float32),
        bright=np.asarray(bright, dtype=np.float32),
        sigs=sigs,
        threshold=float(thr),
        gaps=gaps,
        error=error,
        width=size[0],
        height=size[1],
    )


DEAD_STILL_MOTION = 0.5    # これ未満の動きが
DEAD_STILL_SEC = 60.0      # この長さ以上続く区間は、閾値の推定から外す


def auto_threshold(motion: np.ndarray, times: np.ndarray | None = None) -> float:
    """静止時のノイズ水準から閾値を決める。

    静止している時間が全体の3割以上ある前提で、下位30%の値をノイズとみなす。
    ただし、手を離して長く録画し続けたような「ほぼ動かない長い区間」があると、
    下位30%がその区間の値に引っ張られ、手持ちの見開きが全部「動いている」と判定される
    （長い静止の試験で確認）。そうした区間（60秒以上）は推定から外す。
    普通の動画（静止は数秒）では何も外れないので、結果は変わらない。
    """
    if motion.size == 0:
        return 1.5
    m = motion
    if times is not None and len(times) == len(motion):
        keep = np.ones(len(m), bool)
        dead = m < DEAD_STILL_MOTION
        i = 0
        while i < len(m):
            if dead[i]:
                j = i
                while j + 1 < len(m) and dead[j + 1]:
                    j += 1
                if times[j] - times[i] >= DEAD_STILL_SEC:
                    keep[i:j + 1] = False
                i = j + 1
            else:
                i += 1
        if keep.sum() >= 30:
            m = m[keep]
    noise = float(np.percentile(m, 30))
    return float(np.clip(3.0 * noise + 0.6, 1.0, 5.0))


def find_still_runs(a: Analysis, min_still_sec: float) -> list[dict[str, Any]]:
    """静止区間を返す。短い静止も short=True で返す（黙って消さない）。"""
    n = len(a.times)
    if n == 0:
        return []
    still = a.motion < a.threshold
    runs: list[list[int]] = []
    i = 0
    while i < n:
        if still[i]:
            j = i
            while j + 1 < n and still[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    # 短い瞬断をつなぐ
    merged: list[list[int]] = []
    for r in runs:
        if merged and a.times[r[0]] - a.times[merged[-1][1]] <= MERGE_GAP_SEC:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    out = []
    for i0, i1 in merged:
        dur = a.times[i1] - a.times[i0]
        # 末尾で録画が切れた静止はフレーム間隔ぶん長さを補う
        if i1 == n - 1 and n > 1:
            dur += a.times[-1] - a.times[-2]
        if dur < SHORT_STILL_SEC:
            continue
        out.append({"i0": i0, "i1": i1, "start": float(a.times[i0]), "end": float(a.times[i1]),
                    "duration": float(dur), "short": dur < min_still_sec})
    return out


def forward_motion(a: Analysis) -> np.ndarray:
    """各フレームの「約0.2秒後との差」。動画の終わりで後ろがないフレームは0とする."""
    j = np.searchsorted(a.times, a.times + LAG_SEC - TIME_EPS, side="left")
    fwd = np.zeros_like(a.motion)
    ok = j < len(a.times)
    fwd[ok] = a.motion[j[ok]]
    return fwd


def pick_candidates(a: Analysis, i0: int, i1: int, max_n: int, both_ways: bool = False,
                    fwd: np.ndarray | None = None) -> list[int]:
    """静止区間から候補を選ぶ.

    both_ways=True のときは「0.2秒前とも0.2秒後とも差がない」フレームだけを候補にする。
    動き出す直前のフレーム（手が入り始め、輪郭が増えて鮮明に見えやすい）を避ける。
    """
    idx = np.arange(i0, i1 + 1)
    keep = a.motion[idx] < a.threshold
    if both_ways:
        f = fwd if fwd is not None else forward_motion(a)
        both = keep & (f[idx] < a.threshold)
        if both.any():
            keep = both
    idx = idx[keep]
    if idx.size == 0:
        return []
    # 鮮明さ優先。同点付近は動きの少ない方
    order = idx[np.lexsort((a.motion[idx], -a.sharp[idx]))]
    chosen: list[int] = []
    for k in order:
        if all(abs(a.times[k] - a.times[c]) >= CAND_SPACING_SEC for c in chosen):
            chosen.append(int(k))
        if len(chosen) >= max_n:
            break
    return chosen


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def _ckpt_key(project, video: dict[str, Any]) -> str:
    st = project.settings
    key = {"v": EXTRACT_VERSION, "sha": video["sha256"],
           **{k: st.get(k) for k in ("analysis_width", "motion_threshold", "min_still_sec", "max_candidates",
                                     "candidate_format", "content_rotation", "candidate_still_both_ways")}}
    return json.dumps(key, sort_keys=True)


def checkpoint_paths(project, video_id: str) -> tuple[Path, Path]:
    return (project.path("jobs", f"extract_{video_id}.p1.npz"),
            project.path("jobs", f"extract_{video_id}.p2.json"))


def clear_checkpoints(project, video_id: str) -> None:
    p1, p2 = checkpoint_paths(project, video_id)
    for p in (p1, Path(str(p1) + ".sigs"), p2):
        p.unlink(missing_ok=True)


def extract_video(project, video: dict[str, Any], progress: ProgressFn | None = None) -> dict[str, Any]:
    """1本の動画を解析し、区間・候補フレームをmanifestへ追加する.

    manifest へは全区間がそろってから一度に反映する（途中の区間を確定扱いにしない）。
    区間・フレームのIDは決まった順に振るので、再開しても連続実行と同じになる。
    """
    from vbs.manifest import atomic_write_json, now_iso

    st = project.settings
    vpath = project.abs(video["path"])
    p1, p2 = checkpoint_paths(project, video["id"])
    key = _ckpt_key(project, video)
    a = analyze_motion(
        vpath,
        analysis_width=int(st["analysis_width"]),
        threshold=st.get("motion_threshold"),
        duration_hint=video.get("duration_sec"),
        progress=progress,
        checkpoint=p1,
        checkpoint_key=key,
    )
    runs = find_still_runs(a, float(st["min_still_sec"]))
    both_ways = bool(st.get("candidate_still_both_ways", False))
    fwd = forward_motion(a) if both_ways else None
    plan: list[tuple[dict, list[int]]] = [
        (r, pick_candidates(a, r["i0"], r["i1"], int(st["max_candidates"]), both_ways, fwd)) for r in runs]
    ext = "png" if st.get("candidate_format") == "png" else "jpg"
    n_frames = len(a.times)
    sharp_ref = float(np.median([a.sharp[ks[0]] for _, ks in plan if ks])) if plan else 0.0

    # 2回目（候補の保存）の途中状態
    state = None
    if p2.exists():
        try:
            state = json.loads(p2.read_text(encoding="utf-8"))
            if state.get("key") != key:
                state = None
        except (OSError, ValueError):
            state = None
    if state is None:
        k_content, rot_rec = _content_rotation(st, plan, vpath, a, progress)
        state = {"key": key, "k_content": k_content, "content_rotation": rot_rec,
                 "next_ids": dict(project.data["next_ids"]), "done": 0, "segments": [], "frames": {},
                 "prev": None}
        atomic_write_json(p2, state)
    else:
        faults.event("resume", stage="extract.pass2", done=state["done"])
    video["content_rotation"] = state["content_rotation"]
    if not state["content_rotation"].get("decided", True):
        warns = video.setdefault("warnings", [])
        if "orientation_uncertain" not in warns:
            warns.append("orientation_uncertain")
    project.data["next_ids"] = dict(state["next_ids"])
    k_content = int(state["k_content"])
    segs: list[dict] = state["segments"]
    frames: dict[str, dict] = state["frames"]
    prev = state["prev"]
    prev_feat = _features_of(project, frames, segs[prev]) if prev is not None else None

    total = len(plan)
    for idx in range(state["done"], total):
        r, ks = plan[idx]
        seg_id = project.new_id("segment")
        warnings: list[str] = []
        cand_ids = []
        grabbed = grab_frames(vpath, [int(a.pts[k]) for k in ks], fmt="rgb24") if ks else {}
        for k in ks:
            p = int(a.pts[k])
            rgb = grabbed.pop(p, None)
            if rgb is None:
                warnings.append("frame_missing")
                continue
            # 解析時と同じ場面か照合（AC-07）。紙面の向きを直す前に比べる
            sig_now = _sig(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
            mismatch = _diff(a.sigs[k], sig_now)
            frame_warn = ["frame_mismatch"] if mismatch > 12.0 else []
            if frame_warn:
                warnings.append("frame_mismatch")
            if k_content:
                rgb = np.ascontiguousarray(np.rot90(rgb, k_content))
            fid = project.new_id("frame")
            fpath = project.path("frames", f"{fid}.{ext}")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            imwrite(fpath, bgr, quality=95)
            thumb = cv2.resize(bgr, (320, max(2, int(round(bgr.shape[0] * 320 / bgr.shape[1])))),
                               interpolation=cv2.INTER_AREA)
            imwrite(project.path("thumbs", f"{fid}.jpg"), thumb, quality=80)
            frames[fid] = {
                "id": fid,
                "video_id": video["id"],
                "pts": p,
                "time_sec": float(a.times[k]),
                "path": project.rel(fpath),
                "thumb": project.rel(project.path("thumbs", f"{fid}.jpg")),
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                # 表示行列の回転（デコード時）と紙面の向きの補正（ここ）を分けて記録し、二重回転を防ぐ
                "rotation_applied": video.get("rotation", 0),
                "content_rotation_k": k_content,
                "scores": {
                    "sharpness": round(float(a.sharp[k]), 2),
                    "motion": round(float(a.motion[k]), 3),
                    "brightness": round(float(a.bright[k]), 1),
                    "signature_diff": round(mismatch, 2),
                },
                "warnings": frame_warn,
            }
            cand_ids.append(fid)
            del rgb, bgr, thumb
        grabbed.clear()
        at_edge = r["i0"] == 0 or r["i1"] == n_frames - 1
        if not cand_ids:
            warnings.append("no_candidate")
        if r["short"]:
            warnings.append("short_still")
        if cand_ids and sharp_ref > 0 and a.sharp[ks[0]] < 0.35 * sharp_ref:
            warnings.append("blur")
        flags = (["video_start"] if r["i0"] == 0 else []) + (["video_end"] if r["i1"] == n_frames - 1 else [])
        seg = {
            "id": seg_id,
            "video_id": video["id"],
            "start_sec": r["start"],
            "end_sec": r["end"],
            "duration_sec": round(r["duration"], 3),
            "candidates": cand_ids,
            "chosen": cand_ids[0] if cand_ids else None,
            # 途中の短い静止はめくりの途中のことが多いので採用しない（一覧には残す）。
            # 動画の冒頭・末尾は前後にめくりがないので、短くても確認付きで採用する（EXT-03）
            "include": bool(cand_ids) and (not r["short"] or at_edge),
            "manual": False,
            "duplicate_of": None,
            "warnings": warnings,
            "flags": flags,
        }
        segs.append(seg)
        # 重複判定は直前の区間とだけ比べるので、特徴量は1区間分だけ持つ
        feat = _features_of(project, frames, seg) if cand_ids else None
        if feat is not None:
            prev, prev_feat = _dedupe_step(segs, prev, prev_feat, len(segs) - 1, feat)
        state.update(done=idx + 1, next_ids=dict(project.data["next_ids"]), prev=prev)
        atomic_write_json(p2, state)
        faults.point("extract.pass2.segment")
        if progress:
            progress((idx + 1) / max(1, total), f"候補フレームを保存中 {idx + 1}/{total}")

    new_segments = list(segs)
    # 読めなかった時間帯も区間として残す（EXT-01, IN-06）
    for g0, g1 in a.gaps:
        seg_id = project.new_id("segment")
        new_segments.append({
            "id": seg_id, "video_id": video["id"], "start_sec": g0, "end_sec": g1,
            "duration_sec": round(g1 - g0, 3), "candidates": [], "chosen": None,
            "include": False, "manual": False, "duplicate_of": None,
            "warnings": ["unreadable"], "flags": [],
        })
    new_segments.sort(key=lambda s: s["start_sec"])

    project.data["frames"].update(frames)
    project.data["segments"].extend(new_segments)
    video["analysis"] = {
        "analyzed": now_iso(),
        "frames": int(n_frames),
        "threshold": round(a.threshold, 3),
        "motion_p30": round(float(np.percentile(a.motion, 30)), 3) if a.motion.size else None,
        "still_runs": len(runs),
        "error": a.error,
    }
    return {"segments": len(new_segments), "frames": len(frames), "error": a.error}


def _features_of(project, frames: dict[str, dict], seg: dict) -> tuple | None:
    """重複判定用の特徴量。連続実行と再開で同じになるよう、保存した画像から計算する."""
    if not seg["candidates"]:
        return None
    from vbs.imgio import imread

    bgr = imread(project.abs(frames[seg["candidates"][0]]["path"]))
    return _features(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _dedupe_step(segs: list[dict], prev: int | None, prev_feat, i: int, feat) -> tuple[int, Any]:
    """直前の区間と同じ見開きなら重複としてまとめる（EXT-05）。参照は duplicate_of で残す.

    特徴点の対応を射影変換で検証した数（インライア）で判定する。手の映り込みや
    本の位置ずれに強い。実写サンプルでは同じ見開き 485〜673、別の見開き 6〜18 だった。
    特徴点が少ない（白紙に近い）画像同士は判断せず、確認へ回す。
    戻り値は次に比べる区間とその特徴量。
    """
    seg = segs[i]
    if prev is not None and prev_feat is not None:
        n_in, n_kp = _inliers(prev_feat, feat)
        seg["match_prev"] = {"segment": segs[prev]["id"], "inliers": n_in, "keypoints": n_kp}
        if n_kp < MIN_KEYPOINTS:
            seg["warnings"].append("duplicate_suspect")
        elif n_in >= DUP_INLIERS:
            # 静止時間の長い方を採用する（短い方はめくり直前・直後の可能性が高い）
            keep, drop = (prev, i) if segs[prev]["duration_sec"] >= seg["duration_sec"] else (i, prev)
            if segs[keep]["include"] or segs[drop]["include"]:
                segs[keep]["include"] = bool(segs[keep]["candidates"])
            segs[drop]["include"] = False
            segs[drop]["duplicate_of"] = segs[keep]["id"]
            segs[drop]["warnings"].append("duplicate")
            return keep, (prev_feat if keep == prev else feat)
        elif n_in >= SUSPECT_INLIERS:
            seg["warnings"].append("duplicate_suspect")
    return i, feat


_ORB = None


def _features(rgb: np.ndarray, width: int = 1000) -> tuple:
    global _ORB
    if _ORB is None:
        _ORB = cv2.ORB_create(3000)
    from vbs.split import detect_gutter, detect_page_region

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = width / g.shape[1]
    g = cv2.resize(g, (width, max(2, int(round(g.shape[0] * s)))), interpolation=cv2.INTER_AREA)
    # 見開きが変わっても同じ位置に写るもの（机の模様・紙の縁・綴じ目の影・柱・ノンブル）からは
    # 特徴点を取らない。本文の領域だけを比べる
    (x0, y0, x1, y1), warn = detect_page_region(bgr)
    mask = None
    if "page_region_uncertain" not in warn:
        gx, _, _ = detect_gutter(bgr, (x0, y0, x1, y1))
        w, h = x1 - x0, y1 - y0
        mask = np.zeros(g.shape, np.uint8)
        mask[int((y0 + 0.12 * h) * s):int((y1 - 0.12 * h) * s), int((x0 + 0.04 * w) * s):int((x1 - 0.04 * w) * s)] = 255
        mask[:, max(0, int((gx - 0.05 * w) * s)):int((gx + 0.05 * w) * s)] = 0
    kp, des = _ORB.detectAndCompute(g, mask)
    return (np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)), des


def _inliers(fa: tuple, fb: tuple) -> tuple[int, int]:
    """(射影変換に整合する対応点の数, 少ない方の特徴点数)."""
    pa, da = fa
    pb, db = fb
    n_kp = min(len(pa), len(pb))
    if da is None or db is None or n_kp < 8:
        return 0, n_kp
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        return 0, n_kp
    src = np.float32([pa[m.queryIdx] for m in good])
    dst = np.float32([pb[m.trainIdx] for m in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    return (int(mask.sum()) if mask is not None else 0), n_kp


def _content_rotation(st: dict, plan: list, vpath: Path, a: Analysis,
                      progress: ProgressFn | None) -> tuple[int, dict]:
    """紙面の向き（np.rot90 の k）を決める。静止の長い区間から最大5枚だけ取り出して多数決する."""
    from vbs.orient import vote

    mode = st.get("content_rotation", "auto")
    if mode != "auto":
        k = (int(mode) // 90) % 4
        return k, {"k": k, "mode": "manual", "decided": True, "votes": []}
    runs = sorted((x for x in plan if x[1] and not x[0]["short"]), key=lambda x: -x[0]["duration"])[:5]
    want = [int(a.pts[ks[0]]) for _, ks in runs]
    if progress:
        progress(0.0, "紙面の向きを判定中")
    grabbed = grab_frames(vpath, want, fmt="rgb24") if want else {}
    imgs = [cv2.cvtColor(grabbed[p], cv2.COLOR_RGB2BGR) for p in want if p in grabbed]
    grabbed.clear()
    res = vote(imgs)
    rec = {"mode": "auto", **res}
    if not imgs:
        rec["decided"] = True  # 候補がない動画では判定の対象がない
    return int(res["k"]), rec
