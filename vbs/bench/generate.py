"""試験素材の生成（すべて逐次処理。全フレームや全ページをメモリに持たない）.

- concat_video: 元動画のパケットを再エンコードせずに繰り返しつなぐ（長時間のデコード・時刻管理の試験）
- long_still_video: 元動画の途中に長い静止を挟んで再エンコードする（長い静止の試験）
- synth_pages: 内容の異なる合成ページ画像と正解目録を作る（大量ページの試験）
"""

from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np

from vbs.bench.env import sha256_file


def concat_video(src: Path, out: Path, minutes: float) -> dict[str, Any]:
    """元動画をつないで約 minutes 分の動画を作る。つなぎ目の時刻を記録する."""
    src, out = Path(src), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp" + out.suffix)
    joins: list[float] = []
    with av.open(str(src)) as inp:
        s = inp.streams.video[0]
        one = float(s.duration * s.time_base) if s.duration else float(inp.duration / av.time_base)
        reps = max(1, int(round(minutes * 60 / one)))
        with av.open(str(tmp), "w") as outc:
            o = outc.add_stream_from_template(s)
            off = 0
            for rep in range(reps):
                inp.seek(0)
                last = 0
                for pkt in inp.demux(s):
                    if pkt.dts is None:
                        continue
                    if pkt.pts is not None:
                        pkt.pts += off
                    pkt.dts += off
                    last = max(last, pkt.pts or 0, pkt.dts)
                    pkt.stream = o
                    pkt.time_base = s.time_base
                    outc.mux(pkt)
                frame_dur = int(round(1 / float(s.average_rate or 30) / s.time_base))
                off = last + frame_dur
                joins.append(float(off * s.time_base))
    tmp.replace(out)
    rec = {"kind": "concat", "source": str(src), "source_sha256": sha256_file(src), "repetitions": reps,
           "source_duration_sec": round(one, 3), "duration_sec": round(joins[-1], 3),
           "joins_sec": [round(j, 3) for j in joins[:-1]], "output": str(out), "output_sha256": sha256_file(out),
           "method": "PyAV でパケットをそのまま再多重化（再エンコードなし）。つなぎ目は人工的な切り替わり"}
    out.with_suffix(".source.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return rec


def long_still_video(src: Path, out: Path, at_sec: float, still_minutes: float, width: int = 1080,
                     seed: int = 1) -> dict[str, Any]:
    """元動画の at_sec の画面で still_minutes 分静止させた動画を作る（H.264で再エンコード）.

    静止中も撮像ノイズに近い弱い揺らぎを加える（完全に同一のフレームにはしない）。
    """
    from vbs.video import iter_frames

    src, out = Path(src), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp" + out.suffix)
    rng = np.random.default_rng(seed)
    tb = Fraction(1, 90000)
    fps = 30
    with av.open(str(tmp), "w") as outc:
        st = None
        t_out = 0.0
        still_frame = None
        noise = None

        def emit(img: np.ndarray) -> None:
            nonlocal st, t_out
            if st is None:
                st = outc.add_stream("libx264", rate=fps)
                st.width, st.height = img.shape[1], img.shape[0]
                st.pix_fmt = "yuv420p"
                st.options = {"crf": "23", "preset": "ultrafast"}
                st.codec_context.time_base = tb
                st.time_base = tb
            fr = av.VideoFrame.from_ndarray(img, format="rgb24")
            fr.pts = int(round(t_out / tb))
            fr.time_base = tb
            for pkt in st.encode(fr):
                outc.mux(pkt)
            t_out += 1 / fps

        for t, _, img in iter_frames(src, width=width, fmt="rgb24"):
            if still_frame is None and t >= at_sec:
                still_frame = img.astype(np.int16)
                # 比較の間隔（約6フレーム）と周期が合うと差が厳密に0になるので、素数個を回す
                noise = [rng.normal(0, 1.5, img.shape).astype(np.int16) for _ in range(7)]
                still_start = t_out
                for i in range(int(still_minutes * 60 * fps)):
                    emit(np.clip(still_frame + noise[i % len(noise)], 0, 255).astype(np.uint8))
                still_end = t_out
            emit(img)
        for pkt in st.encode():
            outc.mux(pkt)
    tmp.replace(out)
    rec = {"kind": "long_still", "source": str(src), "source_sha256": sha256_file(src), "at_sec": at_sec,
           "still_minutes": still_minutes, "still_interval_sec": [round(still_start, 3), round(still_end, 3)],
           "duration_sec": round(t_out, 3), "width": width, "fps": fps, "codec": "libx264 crf23 ultrafast",
           "output": str(out), "output_sha256": sha256_file(out),
           "method": "元動画を表示向きでデコードし、at_sec の画面に弱いノイズを加えて静止させ再エンコード"}
    out.with_suffix(".source.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return rec


# ---------------------------------------------------------------- 合成ページ
_FONT_PATHS = ["C:/Windows/Fonts/msgothic.ttc", "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/YuGothM.ttc"]
_MINCHO = ["C:/Windows/Fonts/msmincho.ttc", "C:/Windows/Fonts/yumin.ttf"]
SUBJ = ["市場の構造", "企業の行動", "取引の条件", "価格の決定", "競争の効果", "規制の目的", "行為の要件", "因果の関係",
        "需要の変化", "供給の制約", "契約の解釈", "審査の手続", "排除の効果", "合意の存在", "情報の交換", "製品の品質"]
OBJ = ["消費者の利益", "事業者の自由", "新規の参入", "技術の革新", "取引の相手方", "地域の市場", "国際的な基準",
       "過去の判例", "行政の指針", "経済の分析", "当事者の主張", "具体的な事例"]
VERB = ["に大きな影響を与える。", "と密接に関連している。", "を検討する必要がある。", "から判断される。",
        "について議論がある。", "を比較すると違いが分かる。", "によって左右される。", "として理解されている。"]
KANJI_A = ["独占", "寡占", "協調", "排他", "垂直", "水平", "優越", "共同", "差別", "略奪", "抱合", "拘束", "再販", "入札",
           "談合", "提携", "統合", "分割", "移転", "許諾"]
KANJI_B = ["価格", "取引", "契約", "販売", "購入", "流通", "生産", "研究", "開発", "広告", "技術", "情報", "標準", "設備",
           "原料", "部品", "製品", "役務", "特許", "商標"]
KANJI_C = ["規制", "指針", "要件", "審査", "命令", "勧告", "制度", "効果", "責任", "基準"]


def _font(paths: list[str], size: int):
    from PIL import ImageFont

    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    raise RuntimeError("日本語フォントが見つかりません")


def synth_pages(out_dir: Path, n: int, seed: int = 20261008, width: int = 1400, height: int = 2000,
                border: int = 70) -> dict[str, Any]:
    """内容の異なる合成ページを n 枚作る。1枚ずつ書き出し、メモリに溜めない.

    各ページ：柱・見出し・段落（組み合わせを変える）・表・英数字・脚注・ノンブル。
    写真に近づけるため、周囲に暗い縁を付ける。
    正解目録には、ページごとの期待文字列（そのページにしかない複合語・コード）と画像ハッシュを記録する。
    """
    from PIL import Image, ImageDraw

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    used: set[str] = set()
    f_head = _font(_FONT_PATHS, 40)
    f_sub = _font(_FONT_PATHS, 30)
    f_body = _font(_MINCHO + _FONT_PATHS, 28)
    f_small = _font(_MINCHO + _FONT_PATHS, 22)
    pages = []
    for i in range(n):
        img = Image.new("RGB", (width + 2 * border, height + 2 * border), (45, 40, 36))
        paper = Image.new("RGB", (width, height), (247, 245, 240))
        d = ImageDraw.Draw(paper)
        # このページにしかない語（複合語2つ＋英数字コード）
        terms = []
        while len(terms) < 2:
            w = rng.choice(KANJI_A) + rng.choice(KANJI_B) + rng.choice(KANJI_C)
            if w not in used:
                used.add(w)
                terms.append(w)
        code = f"Case {rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{i + 1:04d}"
        terms.append(code)
        chapter = i // 20 + 1
        d.text((90, 60), f"第{chapter}章　{rng.choice(SUBJ)}と{rng.choice(OBJ)}", font=f_small, fill=(70, 70, 70))
        y = 140
        d.text((90, y), f"{(i % 9) + 1}　{terms[0]}の考え方", font=f_head, fill=(15, 15, 15))
        y += 80
        chars = 44
        lines_written = 0
        para_count = rng.randint(3, 5)
        table_after = rng.randint(1, para_count - 1)
        for k in range(para_count):
            sents = [f"{rng.choice(SUBJ)}は{rng.choice(OBJ)}{rng.choice(VERB)}" for _ in range(rng.randint(3, 6))]
            if k == 0:
                sents.insert(1, f"ここでは{terms[1]}が問題となる。")
            if k == para_count - 1:
                sents.append(f"詳しくは {code} を参照。")
            text = "　" + "".join(sents)
            for j in range(0, len(text), chars):
                if y > height - 260:
                    break
                d.text((90, y), text[j:j + chars], font=f_body, fill=(20, 20, 20))
                y += 44
                lines_written += 1
            y += 14
            if k == table_after and y < height - 520:
                # 表（罫線・数字・英字）
                cols = [90, 420, 760, 1100, width - 90]
                rows = 4
                d.text((90, y), f"表{i + 1}　{rng.choice(KANJI_B)}の比較", font=f_sub, fill=(15, 15, 15))
                y += 50
                for r in range(rows + 1):
                    d.line([(cols[0], y + r * 50), (cols[-1], y + r * 50)], fill=(40, 40, 40), width=2)
                for c in cols:
                    d.line([(c, y), (c, y + rows * 50)], fill=(40, 40, 40), width=2)
                for r in range(rows):
                    for c in range(4):
                        cell = (rng.choice(KANJI_B) if c == 0 else
                                f"{rng.randint(10, 999)}" if c < 3 else rng.choice(["Yes", "No", "N/A"]))
                        d.text((cols[c] + 14, y + r * 50 + 10), cell, font=f_small, fill=(20, 20, 20))
                y += rows * 50 + 40
        d.line([(90, height - 200), (500, height - 200)], fill=(80, 80, 80), width=1)
        d.text((90, height - 185), f"（注）{rng.choice(SUBJ)}については、{rng.choice(OBJ)}も参照。", font=f_small,
               fill=(40, 40, 40))
        d.text((width // 2 - 20, height - 90), str(i + 1), font=f_small, fill=(40, 40, 40))
        img.paste(paper, (border, border))
        path = out_dir / f"page_{i + 1:04d}.jpg"
        img.save(path, format="JPEG", quality=92)
        pages.append({"index": i + 1, "file": path.name, "sha256": sha256_file(path), "terms": terms,
                      "lines": lines_written})
    manifest = {"kind": "synth_pages", "seed": seed, "count": n, "size": [width + 2 * border, height + 2 * border],
                "paper": [width, height], "note": "合成ページ。処理件数と復旧の試験用で、実写の認識精度の根拠にはしない",
                "pages": pages}
    (out_dir / "truth.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
