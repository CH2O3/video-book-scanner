"""レシートのOCR文字から、取引日・取引先・金額・登録番号を読み取る.

読み取りは手がかりの文字（「合計」「年月日」「T+13桁」など）に頼るので、誤りや取りこぼしがある。
結果には根拠の行を付け、人やAIが画像と見比べて直せるようにする。読めないものは推測で埋めない。
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Any

FIELDS = ["date", "payee", "total", "base10", "base8", "tax", "registration_no", "note"]
FIELD_LABELS = {
    "date": "取引日",
    "payee": "取引先",
    "total": "金額（税込）",
    "base10": "10%対象",
    "base8": "8%対象（軽減）",
    "tax": "消費税",
    "registration_no": "登録番号",
    "note": "メモ",
}

# OCRで円記号が化けやすい形
_YEN = r"[¥\\Y￥羊半]"
_AMOUNT = re.compile(rf"{_YEN}?\s*(-?\d{{1,3}}(?:[,.]\d{{3}})+|-?\d+)\s*円?")
_TOTAL_WORDS = ["領収金額", "お買上合計", "お買上計", "お買い上げ合計", "ご利用金額", "ご請求額", "お支払金額",
                "お支払い金額", "総合計", "税込合計", "合計金額", "合計", "総額", "計"]
_NOT_TOTAL = ["小計", "対象", "消費税", "税額", "内税", "外税", "お預", "預り", "釣", "点数", "割引", "値引", "ポイント",
              "クーポン", "数量", "単価"]
_PAYEE_HINTS = ["株式会社", "(株)", "（株）", "有限会社", "合同会社", "店", "堂", "屋", "ストア", "マート", "薬局",
                "ドラッグ", "スーパー", "商店", "ホテル", "タクシー", "交通", "鉄道", "書店", "カフェ", "食堂"]
_NOT_PAYEE = ["領収", "レシート", "TEL", "電話", "〒", "登録番号", "毎度", "ありがとう", "いらっしゃいませ", "合計",
              "小計", "対象", "消費税", "お預", "釣", "年", "月", "日", "No", "NO", "レジ", "担当", "責"]


def normalize(text: str) -> str:
    """全角→半角などをそろえる（NFKC）。行の中の空白は詰める."""
    t = unicodedata.normalize("NFKC", text)
    return "\n".join(re.sub(r"[ \t　]+", " ", line).strip() for line in t.splitlines())


def _amount(s: str) -> int | None:
    s = s.replace(",", "").replace(".", "")
    try:
        v = int(s)
    except ValueError:
        return None
    return v if 0 < abs(v) < 10_000_000 else None


def _amounts_in(line: str) -> list[int]:
    out = []
    for m in _AMOUNT.finditer(line):
        raw = m.group(1)
        # 「10%」「8%」の数字や、日付・時刻の数字は金額にしない
        tail = line[m.end(1):m.end(1) + 1]
        if tail in ("%", ":", "/", "年", "月", "日", "点", "個"):
            continue
        v = _amount(raw)
        if v is not None:
            out.append(v)
    return out


def find_date(lines: list[str]) -> tuple[str | None, str | None]:
    pats = [
        (re.compile(r"(20\d{2})\s*[年/.\-]\s*(\d{1,2})\s*[月/.\-]\s*(\d{1,2})"), "y"),
        (re.compile(r"(?:令和|R)\s*(\d{1,2}|元)\s*[年/.\-]\s*(\d{1,2})\s*[月/.\-]\s*(\d{1,2})"), "r"),
        (re.compile(r"(?<!\d)(\d{2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{1,2})(?!\d)"), "yy"),
    ]
    for pat, kind in pats:
        for line in lines:
            m = pat.search(line)
            if not m:
                continue
            y, mo, d = m.groups()
            if kind == "r":
                y = 2018 + (1 if y == "元" else int(y))
            elif kind == "yy":
                y = 2000 + int(y)
            try:
                dt = date(int(y), int(mo), int(d))
            except ValueError:
                continue
            if 2000 <= dt.year <= date.today().year + 1:
                return dt.isoformat(), line
    return None, None


def find_total(lines: list[str]) -> tuple[int | None, str | None]:
    best: tuple[int, int, str] | None = None  # (手がかりの優先順位, 金額, 行)
    for i, line in enumerate(lines):
        compact = line.replace(" ", "")
        if any(w in compact for w in _NOT_TOTAL):
            continue
        for rank, w in enumerate(_TOTAL_WORDS):
            if w in compact:
                vals = _amounts_in(line[line.find(w[0]):] if w[0] in line else line)
                nxt = lines[i + 1].replace(" ", "") if i + 1 < len(lines) else ""
                if not vals and nxt and not any(x in nxt for x in _NOT_TOTAL) and not re.search(r"[぀-ヿ一-鿿]", nxt):
                    vals = _amounts_in(lines[i + 1])  # 金額だけが次の行に折り返している
                if vals:
                    cand = (rank, vals[-1], line)
                    if best is None or cand[0] < best[0]:
                        best = cand
                break
    if best:
        return best[1], best[2]
    return None, None


def find_tax(lines: list[str]) -> dict[str, tuple[int | None, str | None]]:
    out: dict[str, tuple[int | None, str | None]] = {"base10": (None, None), "base8": (None, None), "tax": (None, None)}
    for line in lines:
        c = line.replace(" ", "")
        m = re.search(r"(10|8)%(?:対象|課税|税率対象)", c)
        if m:
            key = "base10" if m.group(1) == "10" else "base8"
            vals = _amounts_in(c[m.end():])
            if vals and out[key][0] is None:
                out[key] = (vals[0], line)
            continue
        if ("消費税" in c or "税額" in c) and "対象" not in c and out["tax"][0] is None:
            vals = _amounts_in(re.sub(r"(10|8)%", "", c))
            if vals:
                out["tax"] = (vals[-1], line)
    return out


def find_registration_no(lines: list[str]) -> tuple[str | None, str | None]:
    """適格請求書発行事業者の登録番号（T＋13桁）."""
    for line in lines:
        c = line.replace(" ", "").replace("-", "")
        m = re.search(r"[TＴ丁][:：]?(\d{13})(?!\d)", c)
        if m:
            return "T" + m.group(1), line
    for line in lines:  # 「登録番号」の後ろの数字（T が別の文字に化けた、数字が1つ多い等）
        c = line.replace(" ", "").replace("-", "")
        if "登録" in c:
            run = max(re.findall(r"\d+", c), key=len, default="")
            if len(run) < 13:
                continue
            windows = ["T" + run[i:i + 13] for i in range(len(run) - 12)]
            # 検査用の数字が合うものを優先する。合うものがなければ先頭から13桁（確認へ回る）
            ok = [w for w in windows if registration_check_ok(w)]
            return (ok[0] if len(ok) == 1 else windows[0]), line
    return None, None


def registration_check_ok(no: str | None) -> bool | None:
    """登録番号（T＋13桁）の先頭の検査用数字を確かめる（法人番号と同じ計算）。

    下12桁を右から数えて、奇数番目は1倍・偶数番目は2倍して足し、9で割った余りを9から引いた数が先頭の数字。
    """
    if not no or not re.fullmatch(r"T\d{13}", no):
        return None
    digits = [int(c) for c in no[1:]]
    body = digits[1:][::-1]
    total = sum(d * (1 if i % 2 == 0 else 2) for i, d in enumerate(body))
    return digits[0] == 9 - total % 9


def find_payee(lines: list[str]) -> tuple[str | None, str | None, list[str]]:
    cands = []
    for i, line in enumerate(lines[:12]):
        c = line.strip()
        if len(re.sub(r"[^぀-ヿ一-鿿A-Za-z]", "", c)) < 2:
            continue
        if any(w in c for w in _NOT_PAYEE) or _amounts_in(c) and len(c) < 8:
            continue
        score = (3 if any(h in c for h in _PAYEE_HINTS) else 0) - i * 0.2
        cands.append((score, c))
    cands.sort(key=lambda x: -x[0])
    names = [c for _, c in cands[:3]]
    return (names[0] if names else None), (names[0] if names else None), names


def parse(text: str) -> dict[str, Any]:
    """OCR文字から項目を読み取る。値・根拠の行・候補を返す."""
    norm = normalize(text)
    lines = [l for l in norm.splitlines() if l.strip()]
    res: dict[str, Any] = {"fields": {}, "evidence": {}, "candidates": {}}
    d, dl = find_date(lines)
    t, tl = find_total(lines)
    p, pl, pc = find_payee(lines)
    r, rl = find_registration_no(lines)
    tax = find_tax(lines)
    for k, (v, src) in {"date": (d, dl), "total": (t, tl), "payee": (p, pl), "registration_no": (r, rl),
                        **tax}.items():
        res["fields"][k] = v
        res["evidence"][k] = src
    res["candidates"]["payee"] = pc
    res["checks"] = checks(res["fields"])
    return res


def checks(f: dict[str, Any]) -> list[str]:
    """読み取り結果の食い違い（参考）。確定を止める理由ではない."""
    out = []
    b10, b8, total = f.get("base10"), f.get("base8"), f.get("total")
    if total and (b10 or b8) and abs((b10 or 0) + (b8 or 0) - total) > 2:
        out.append("対象額の合計が金額と合わない（外税、または読み誤りの可能性）")
    return out


def missing(fields: dict[str, Any]) -> list[str]:
    """確認に回す理由コード（検索に必要な3項目：取引日・金額・取引先、と金額の食い違い）."""
    codes = []
    if not fields.get("date"):
        codes.append("receipt_no_date")
    if not fields.get("total"):
        codes.append("receipt_no_total")
    if not fields.get("payee"):
        codes.append("receipt_no_payee")
    if checks(fields):
        codes.append("receipt_amount_mismatch")
    if registration_check_ok(fields.get("registration_no")) is False:
        codes.append("receipt_regno_suspect")
    return codes


def merged(rec: dict[str, Any] | None) -> dict[str, Any]:
    """自動の読み取りに手修正を重ねた、いま有効な値."""
    rec = rec or {}
    out = {k: None for k in FIELDS}
    out.update({k: v for k, v in ((rec.get("auto") or {}).get("fields") or {}).items() if k in out})
    out.update({k: v for k, v in (rec.get("manual") or {}).items() if k in out})
    return out


def coerce(key: str, value: Any) -> Any:
    """画面・CSVから来た値を項目の型にそろえる。空は None."""
    if value is None:
        return None
    s = unicodedata.normalize("NFKC", str(value)).strip()
    if s == "":
        return None
    if key in ("total", "base10", "base8", "tax"):
        v = _amount(re.sub(r"[^\d\-,.]", "", s))
        if v is None:
            raise ValueError(f"{FIELD_LABELS[key]} は数字で入れてください: {value}")
        return v
    if key == "date":
        d, _ = find_date([s.replace("-", "/")])
        if not d:
            raise ValueError(f"取引日は 2024-01-31 の形で入れてください: {value}")
        return d
    if key == "registration_no":
        digits = re.sub(r"\D", "", s)
        if len(digits) != 13:
            raise ValueError(f"登録番号は T と13桁の数字です: {value}")
        return "T" + digits
    return s
