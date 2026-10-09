"""work/bench/*/result.json を集計して表示する.

    python tools/bench_report.py [work/bench] > work/bench/REPORT.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "work/bench")
    results = []
    for p in sorted(base.glob("*/result.json")):
        try:
            results.append((p.parent.name, json.loads(p.read_text(encoding="utf-8"))))
        except ValueError:
            continue
    env = next((r["environment"] for _, r in reversed(results) if "environment" in r), {})
    print("# ベンチマーク結果\n")
    if env:
        print(f"- コミット: `{env['git']['commit'][:10]}`（未コミット変更: {env['git']['dirty']}）")
        print(f"- PC: {env['cpu']} / {env['cpu_count']}スレッド / RAM {env['ram_total_gib']} GiB / {env['platform']}")
        print(f"- Python {env['python']}, Tesseract {env['tesseract']}, PyAV {env['packages'].get('av')}")
        print()
    print("## 耐久（bench run）\n")
    print("| 実行 | 入力 | 段階 | 所要(実/起) | 段階別秒 | 最大 Private | 最大 RSS | 作業容量 最大/最終 | 区間/ページ | OCR完了/失敗 | 待ち行列 | 結果 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, r in results:
        if r.get("kind") != "bench-run":
            continue
        s = r["summary"]
        inp = r["input"]
        what = (f"動画 {inp['source']['duration_sec']:.0f}秒" if inp.get("source") else
                f"ページ {inp.get('count')}枚" if inp.get("count") else Path(inp.get("video", "")).name)
        run = r["runs"][-1]
        mode = "OCRなし" if "--no-ocr" in run["cmd"] else "全段階"
        ok = s["returncode"] in (0, 2)
        extra = ""
        if "order_ok" in s:
            extra = f" 順序{'OK' if s['order_ok'] else 'NG'} 欠落{len(s['missing'])} 重複{len(s['duplicates'])}"
        ver = s.get("verify") or {}
        terms = (ver.get("terms") or {})
        if terms:
            extra += f" 期待語 {terms.get('found')}/{terms.get('total')}"
        print(f"| {name} | {what} | {mode} | {s['elapsed_sec']:.0f}/{run.get('awake_sec', 0):.0f}s | "
              f"{s['stage_sec']} | {s['peak_private_mib']:.0f} MiB | {s['peak_rss_mib']:.0f} MiB | "
              f"{s['workdir_peak_mib']}/{s['workdir_final_mib']} MiB | {s['segments']}/{s['pages']} | "
              f"{s['ocr_done']}/{s['ocr_failed']} | {s.get('ocr_queue_peak')} | {'完了' if ok else '失敗'}{extra} |")
    print("\n## 中断・再開・局所失敗（bench resume-test / regress）\n")
    for name, r in results:
        if r.get("kind") not in ("resume-test", "regress"):
            continue
        title = r.get("scenario", "regress")
        print(f"### {title}（{name}） → {'合格' if r.get('passed') else '不合格'}")
        for run in r.get("runs", []):
            if run.get("killed"):
                print(f"- 強制終了: {run['name']} {run['killed']['reason']}（開始から {run['killed']['after_sec']}秒）")
            if run.get("standby_suspected"):
                print(f"- 注意: {run['name']} の実行中にPCが停止していた疑い（実 {run['elapsed_sec']}s / 起 {run['awake_sec']}s）")
        for c in r.get("checks", []):
            print(f"- [{'OK' if c['ok'] else 'NG'}] {c['check']}")
        for d in r.get("diffs", [])[:10]:
            print(f"  - 差分: {d}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
