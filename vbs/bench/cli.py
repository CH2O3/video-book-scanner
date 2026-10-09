"""vbs bench サブコマンド."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_OUT = Path("work/bench")


def _print_checks(res: dict[str, Any]) -> bool:
    ok = True
    for c in res.get("checks", []):
        print(f"  [{'OK' if c['ok'] else 'NG'}] {c['check']}")
        ok &= c["ok"]
    for d in res.get("diffs", [])[:20]:
        print(f"    差分: {d}")
    return ok


def cmd_generate(args: argparse.Namespace) -> int:
    from vbs.bench import generate as G

    if args.kind == "video":
        rec = G.concat_video(Path(args.src), Path(args.out), args.minutes)
    elif args.kind == "still":
        rec = G.long_still_video(Path(args.src), Path(args.out), args.at, args.minutes)
    else:
        rec = G.synth_pages(Path(args.out), args.n, seed=args.seed)
        rec = {k: v for k, v in rec.items() if k != "pages"}
    print(json.dumps(rec, ensure_ascii=False, indent=2))
    return 0


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    st: dict[str, Any] = {}
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        try:
            st[k] = json.loads(v)
        except ValueError:
            st[k] = v
    return st


def cmd_run(args: argparse.Namespace) -> int:
    """素材を本番の処理に通して計測する（動画：抽出から / ページ：写真として追加して後段から）."""
    from vbs.bench.env import sha256_file
    from vbs.bench.run import new_run_dir, run_vbs, save_result, snapshot
    from vbs.bench.scenarios import new_video_project, set_settings

    out = new_run_dir(Path(args.out), "pages" if args.pages else "video")
    proj = out / "project"
    logs = out / "logs"
    st = _settings(args)
    runs = []
    result: dict[str, Any] = {"kind": "bench-run", "settings": st}
    if args.video:
        src = Path(args.video)
        meta = src.with_suffix(".source.json")
        result["input"] = {"video": str(src.resolve()), "size": src.stat().st_size,
                           "sha256": sha256_file(src), "source": json.loads(meta.read_text(encoding="utf-8"))
                           if meta.exists() else None}
        new_video_project(src, proj, logs, st)
    else:
        pages = Path(args.pages)
        truth = json.loads((pages / "truth.json").read_text(encoding="utf-8"))
        result["input"] = {"pages_dir": str(pages.resolve()), "count": truth["count"], "seed": truth["seed"],
                           "truth_sha256": sha256_file(pages / "truth.json")}
        r = run_vbs(["new", str(proj), "--single", "--title", "synthetic"], logs, "new")
        set_settings(proj, st)
        runs.append(run_vbs(["add-photos", str(proj), str(pages)], logs, "add", workdir=proj))
        # 正解目録の期待文字列を、追加されたページIDに対応付けて検証に使う
        d = json.loads((proj / "project.json").read_text(encoding="utf-8"))
        segs = [s for s in d["segments"] if s.get("source") == "photo"]
        terms = {f"{s['id']}-S": truth["pages"][i]["terms"] for i, s in enumerate(segs)}
        (out / "terms.json").write_text(json.dumps(terms, ensure_ascii=False, indent=2), encoding="utf-8")
        set_settings(proj, {"verify_terms": str((out / "terms.json").resolve())})
    run_args = ["run", str(proj), "--force"]
    if args.no_ocr:
        run_args.append("--no-ocr")
    if args.workers:
        run_args += ["--workers", str(args.workers)]
    runs.append(run_vbs(run_args, logs, "run", workdir=proj, interval=args.interval))
    snap = snapshot(proj)
    d = json.loads((proj / "project.json").read_text(encoding="utf-8"))
    main = runs[-1]
    result.update({
        "runs": runs,
        "summary": {
            "returncode": main["returncode"],
            "elapsed_sec": main["elapsed_sec"],
            "stage_sec": main["events"]["stage_sec"],
            "peak_private_mib": main["monitor"]["peak_private_mib"],
            "peak_rss_mib": main["monitor"]["peak_rss_mib"],
            "workdir_peak_mib": main["monitor"]["workdir_peak_mib"],
            "workdir_final_mib": main["monitor"]["workdir_final_mib"],
            "frames_analyzed": sum((v.get("analysis") or {}).get("frames", 0) for v in d["videos"]),
            "segments": len(snap["segments"]),
            "pages": len(snap["pages"]),
            "ocr_done": main["events"]["ocr_done"],
            "ocr_failed": main["events"]["ocr_failed"],
            "ocr_queue_peak": (d.get("stats") or {}).get("ocr_queue_peak"),
            "export": snap["export"],
            "verify": snap["verify"] and {k: snap["verify"][k] for k in ("pages", "page_count_ok", "terms")},
        },
    })
    if args.pages:
        exp_ids = [f"{s['id']}-S" for s in segs]
        got = [p["id"] for p in snap["pages"]]
        result["summary"]["order_ok"] = got == exp_ids
        result["summary"]["missing"] = sorted(set(exp_ids) - set(got))
        result["summary"]["duplicates"] = sorted({x for x in got if got.count(x) > 1})
    path = save_result(out, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"結果: {path}")
    return 0 if main["returncode"] in (0, 2) else 1


def cmd_resume(args: argparse.Namespace) -> int:
    from vbs.bench import scenarios as S
    from vbs.bench.run import new_run_dir, save_result

    out = new_run_dir(Path(args.out), f"resume-{args.scenario}")
    st = _settings(args)
    video = Path(args.video)
    sc = args.scenario
    if sc == "extract-pass1":
        res = S.video_resume(video, out, "動きを解析中", args.frac, st)
    elif sc == "extract-pass2":
        res = S.video_resume(video, out, "候補フレームを保存中", args.frac, st)
    elif sc == "split":
        res = S.video_resume(video, out, "ページ画像を作成", args.frac, st)
    elif sc == "write-fail":
        res = S.write_failure(video, out, st)
    else:
        prep = S.prepare_pages(video, Path(args.prepared_dir) if args.prepared_dir else out, st)
        if sc == "ocr-after":
            res = S.ocr_resume(prep, out, args.workers, args.k, "after")
        elif sc == "ocr-mid":
            res = S.ocr_resume(prep, out, args.workers, args.k, "mid")
        elif sc == "ocr-orphan":
            res = S.ocr_resume(prep, out, args.workers, args.k, "orphan")
        elif sc == "ocr-sidecar":
            res = S.ocr_resume(prep, out, args.workers, args.k, "sidecar")
        elif sc == "export":
            res = S.export_resume(prep, out, args.workers)
        elif sc == "ocr-fail":
            res = S.ocr_local_failure(prep, out, args.workers, args.page)
        else:
            raise SystemExit(f"不明な試験: {sc}")
    res.update(kind="resume-test", scenario=sc, input={"video": str(video.resolve())}, settings=st)
    ok = _print_checks(res)
    res["passed"] = ok
    path = save_result(out, res)
    print(f"{'合格' if ok else '不合格'}  結果: {path}")
    return 0 if ok else 1


def cmd_regress(args: argparse.Namespace) -> int:
    """現行の短い実写動画を同じ条件で処理し、基準（ページ順・採用元・60件の検索）と比べる."""
    from vbs.bench.run import compare, new_run_dir, run_vbs, save_result, snapshot
    from vbs.bench.scenarios import new_video_project

    out = new_run_dir(Path(args.out), "regress")
    base_proj = Path(args.baseline_project)
    st = json.loads((base_proj / "project.json").read_text(encoding="utf-8"))["settings"]
    keep = {k: st[k] for k in st if k not in ("verify_terms",)}
    keep["verify_terms"] = str(Path(args.terms).resolve())
    keep.update(_settings(args))
    proj = out / "project"
    new_video_project(Path(args.video), proj, out / "logs", keep)
    r = run_vbs(["run", str(proj), "--force", "--workers", str(args.workers)], out / "logs", "run", workdir=proj)
    a, b = snapshot(base_proj), snapshot(proj)
    diffs = compare(a, b, parts=("segments",))
    order_a = [(p["id"], p["frame_time"]) for p in a["pages"]]
    order_b = [(p["id"], p["frame_time"]) for p in b["pages"]]
    base_ver = json.loads(Path(args.baseline_verify).read_text(encoding="utf-8"))
    base_terms = {p["page_id"]: p.get("terms_found") for p in base_ver["per_page"] if "terms_found" in p}
    new_terms = {pid: found for pid, found in (b["verify"] or {}).get("per_page_terms", []) if found is not None}
    term_diffs = {pid: {"base": base_terms.get(pid), "new": new_terms.get(pid)} for pid in base_terms
                  if base_terms.get(pid) != new_terms.get(pid)}
    checks = [
        {"check": "処理が完了", "ok": r["returncode"] in (0, 2)},
        {"check": "採用ページと順序・採用元の時刻が基準と一致", "ok": order_a == order_b},
        {"check": "60件の検索成否が基準と一致", "ok": not term_diffs},
    ]
    res = {"kind": "regress", "runs": [r], "checks": checks, "diffs": diffs, "term_diffs": term_diffs,
           "baseline": {"project": str(base_proj.resolve()), "verify": base_ver.get("sha256"),
                        "terms": base_ver.get("terms_summary")},
           "new": {"terms": (b["verify"] or {}).get("terms"), "export": b["export"]}}
    ok = _print_checks(res)
    if term_diffs:
        print(f"    検索の差: {json.dumps(term_diffs, ensure_ascii=False)}")
    res["passed"] = ok
    print(f"結果: {save_result(out, res)}")
    return 0 if ok else 1


def cmd_concat_check(args: argparse.Namespace) -> int:
    """連結動画の結果を、元動画単独の結果と回ごとに照合し、最終PDFの検索結果も回ごとに比べる."""
    from vbs.bench.concat import concat_check, map_terms
    from vbs.bench.run import save_result
    from vbs.verify import load_terms, verify_pdf, write_report

    res = concat_check(Path(args.base), Path(args.project), Path(args.source))
    mapping = res.pop("mapping")
    checks = [
        {"check": f"総ページ数が期待値と一致（{res['got_total']} / 期待 {res['expected_total']}）",
         "ok": res["got_total"] == res["expected_total"]},
        {"check": f"全{res['repetitions']}回で、採用した見開き（回の中の時刻・左右）とページ画像が元動画と一致"
                  f"（一致 {res['reps_ok']} 回）", "ok": res["reps_ok"] == res["repetitions"]},
    ]
    if args.terms:
        base = json.loads((Path(args.base) / "project.json").read_text(encoding="utf-8"))
        bexp = base["exports"][-1]
        bver = json.loads(Path(bexp["verify"]).read_text(encoding="utf-8"))
        base_found = {p["page_id"]: sorted(p.get("terms_found") or []) for p in bver["per_page"] if "terms_found" in p}
        terms = map_terms(load_terms(Path(args.terms)), mapping)
        conc = json.loads((Path(args.project) / "project.json").read_text(encoding="utf-8"))
        exp = conc["exports"][-1]
        ver = verify_pdf(Path(exp["path"]), exp.get("order") or conc["order"], terms)
        out = Path(args.project) / "concat-terms.verify.json"
        out.write_text(json.dumps(ver, ensure_ascii=False, indent=2), encoding="utf-8")
        diffs = []
        for p in ver["per_page"]:
            if "terms_found" in p and sorted(p["terms_found"]) != base_found.get(mapping[p["page_id"]]):
                diffs.append({"page": p["page_id"], "pdf_page": p["pdf_page"], "base": mapping[p["page_id"]],
                              "found": p["terms_found"], "base_found": base_found.get(mapping[p["page_id"]])})
        res["terms"] = {"checked_pages": len(terms), "found": ver["terms_summary"]["found"],
                        "total": ver["terms_summary"]["total"], "pages_differ_from_base": diffs[:20],
                        "pdf_sha256": ver["sha256"], "pdf_pages": ver["pages"]}
        checks.append({"check": f"最終PDFの検索結果が、全回で元動画のPDFと同じ（違うページ {len(diffs)} 件）",
                       "ok": not diffs})
        checks.append({"check": f"最終PDFのページ数が一致（{ver['pages']}）", "ok": ver["page_count_ok"]})
    res["checks"] = checks
    ok = _print_checks(res)
    for p in res.get("problems", [])[:5]:
        print(f"    回 {p['rep']}: 欠落 {p['missing']} 余分 {p['extra']} 画像違い {p['image_differs']}")
    res.update(kind="concat-check", passed=ok, base=str(Path(args.base).resolve()), project=str(Path(args.project).resolve()))
    out_dir = Path(args.project).parent
    save_result(out_dir, {**json.loads((out_dir / "result.json").read_text(encoding="utf-8")), "concat_check": res}
                if (out_dir / "result.json").exists() else res)
    print("合格" if ok else "不合格")
    return 0 if ok else 1


def cmd_dataset(args: argparse.Namespace) -> int:
    from vbs.bench import dataset as D

    if args.action == "smartdoc-fetch":
        rec = D.smartdoc_fetch(Path(args.dir), sample_only=not args.full)
    elif args.action == "smartdoc-eval":
        rec = D.smartdoc_eval(Path(args.dir), Path(args.out))
    else:
        rec = D.pucit_check(Path(args.dir))
    print(json.dumps({k: v for k, v in rec.items() if k != "per_video"}, ensure_ascii=False, indent=2)[:4000])
    return 0


def add_bench_parser(sub) -> None:
    b = sub.add_parser("bench", help="長時間・大量ページ・中断復旧の試験")
    bs = b.add_subparsers(dest="bench_cmd", required=True)

    g = bs.add_parser("generate", help="試験素材を作る")
    g.add_argument("kind", choices=["video", "still", "pages"])
    g.add_argument("--src", help="元動画")
    g.add_argument("--out", required=True)
    g.add_argument("--minutes", type=float, default=10)
    g.add_argument("--at", type=float, default=11.0, help="静止させる時刻（still）")
    g.add_argument("--n", type=int, default=300)
    g.add_argument("--seed", type=int, default=20261008)
    g.set_defaults(func=cmd_generate)

    r = bs.add_parser("run", help="素材を本番の処理に通して計測する")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--pages", help="generate pages の出力フォルダ")
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--no-ocr", action="store_true")
    r.add_argument("--workers", type=int)
    r.add_argument("--interval", type=float, default=5.0, help="メモリ計測の間隔（秒）")
    r.add_argument("--set", nargs="*", help="設定 key=value（JSON値）")
    r.set_defaults(func=cmd_run)

    t = bs.add_parser("resume-test", help="強制終了→再開を連続実行と比べる")
    t.add_argument("scenario", choices=["extract-pass1", "extract-pass2", "split", "ocr-after", "ocr-mid",
                                        "export", "ocr-fail", "write-fail", "ocr-sidecar", "ocr-orphan"])
    t.add_argument("--video", required=True)
    t.add_argument("--out", default=str(DEFAULT_OUT))
    t.add_argument("--prepared-dir", help="OCR前のプロジェクトを置く/再利用する場所")
    t.add_argument("--frac", type=float, default=0.5)
    t.add_argument("--k", type=int, default=6, help="OCRが何ページ終わったら止めるか")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--page", default="s0013-R", help="ocr-fail で失敗させるページ")
    t.add_argument("--set", nargs="*")
    t.set_defaults(func=cmd_resume)

    rg = bs.add_parser("regress", help="現行の実写動画で基準と比べる")
    rg.add_argument("--video", required=True)
    rg.add_argument("--baseline-project", required=True)
    rg.add_argument("--baseline-verify", required=True)
    rg.add_argument("--terms", required=True)
    rg.add_argument("--workers", type=int, default=4)
    rg.add_argument("--out", default=str(DEFAULT_OUT))
    rg.add_argument("--set", nargs="*", help="比べる側の設定 key=value")
    rg.set_defaults(func=cmd_regress)

    cc = bs.add_parser("concat-check", help="連結動画の結果を元動画単独の結果と回ごとに照合")
    cc.add_argument("--base", required=True, help="元動画を単独で処理したプロジェクト")
    cc.add_argument("--project", required=True, help="連結動画を処理したプロジェクト")
    cc.add_argument("--source", required=True, help="連結動画の .source.json")
    cc.add_argument("--terms", help="元動画の重要語の表（最終PDFの検索を回ごとに比べる）")
    cc.set_defaults(func=cmd_concat_check)

    ds = bs.add_parser("dataset", help="公開データの取得・評価")
    ds.add_argument("action", choices=["smartdoc-fetch", "smartdoc-eval", "pucit-check"])
    ds.add_argument("--dir", default="samples/datasets")
    ds.add_argument("--out", default=str(DEFAULT_OUT))
    ds.add_argument("--full", action="store_true", help="本体（約1.5GB）も取得する")
    ds.set_defaults(func=cmd_dataset)
