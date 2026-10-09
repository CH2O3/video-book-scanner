"""コマンドライン."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vbs import faults, pipeline
from vbs.manifest import Project, ProjectError
from vbs.ocr import OcrError
from vbs.video import VideoError


def _progress(stage: str, frac: float, msg: str) -> None:
    faults.event("progress", stage=stage, frac=round(frac, 4), msg=msg)
    sys.stderr.write(f"\r[{stage:7}] {frac * 100:5.1f}% {msg[:70]:<70}")
    if frac >= 1.0:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _settings_from(args: argparse.Namespace) -> dict:
    return {
        "layout": "single" if getattr(args, "single", False) else "spread",
        "direction": "rtl" if getattr(args, "rtl", False) else "ltr",
        "expected_pages": getattr(args, "expected_pages", None),
        "page_height_mm": getattr(args, "page_height_mm", None),
        "auto_export": False if getattr(args, "always_review", False) else True,
    }


def _print_review(project: Project) -> None:
    items = pipeline.review_items(project)
    s = pipeline.summary(project)
    print(f"区間 {s['segments']}（採用 {s['included_segments']}）/ ページ {s['pages']} / "
          f"OCR済み {s['ocr_done']} / 要確認 {s['review']}")
    for it in items:
        where = it["id"]
        if it["kind"] == "segment":
            where += f" @{it['time']:.1f}s"
        print(f"  - {it['kind']:7} {where}: {'、'.join(it['labels'])}")


def cmd_new(args: argparse.Namespace) -> int:
    project = Project.create(Path(args.project), _settings_from(args))
    if args.title:
        project.data["title"] = args.title
        project.save()
    with project.lock():
        pipeline.add_videos(project, [Path(v) for v in args.videos], copy=not args.link,
                            allow_hdr=args.allow_hdr, progress=_progress)
    print(f"プロジェクトを作成しました: {project.root}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    project = Project.load(Path(args.project))
    with project.lock():
        pipeline.add_videos(project, [Path(v) for v in args.videos], copy=not args.link,
                            allow_hdr=args.allow_hdr, progress=_progress)
    return 0


def _run(project: Project, force: bool, export: bool | None, ocr: bool = True,
         workers: int | None = None) -> int:
    faults.event("stage", stage="run_start")
    res = pipeline.run_all(project, _progress, export=export, force=force, ocr=ocr, workers=workers)
    faults.event("stage", stage="run_end", export=(res["export"] or {}).get("path"))
    _print_review(project)
    if res["export"]:
        print(f"PDFを出力しました: {res['export']['path']}（{res['export']['pages']}ページ）")
        return 0
    if not ocr:
        return 0  # OCRなしの実行ではPDFを作らないのが正常
    if res["review"]:
        print("要確認があるためPDFは未出力です。`vbs ui` で確認するか、`vbs export --force` で出力できます。")
        return 2
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    project = Project.load(Path(args.project))
    with project.lock():
        return _run(project, args.force, False if args.no_export else None, ocr=not args.no_ocr,
                    workers=args.workers)


def cmd_scan(args: argparse.Namespace) -> int:
    """動画を渡すだけで、隣にプロジェクトを作って最後まで進める."""
    videos = [Path(v).resolve() for v in args.videos]
    for v in videos:
        if not v.is_file():
            raise ProjectError(f"ファイルがありません: {v}")
    root =Path(args.project) if args.project else videos[0].with_name(videos[0].stem + "_scan")
    if (root / "project.json").exists():
        project = Project.load(root)
    else:
        project = Project.create(root, _settings_from(args))
        project.data["title"] = args.title or videos[0].stem
        project.save()
    with project.lock():
        known = {v["original_name"] for v in project.data["videos"]}
        new = [v for v in videos if v.name not in known]
        if new:
            pipeline.add_videos(project, new, copy=not args.link, allow_hdr=args.allow_hdr, progress=_progress)
        code = _run(project, args.force, None)
    print(f"プロジェクト: {project.root}")
    if code == 2 and args.ui:
        from vbs.server import serve
        serve(project.root)
    elif code == 0 and args.ui and sys.platform == "win32" and project.data["exports"]:
        import os
        os.startfile(str(project.path("output")))  # type: ignore[attr-defined]
    return code


def cmd_add_photos(args: argparse.Namespace) -> int:
    """写真（JPEG/PNG）でページを足す。フォルダを渡すと名前順にすべて追加する."""
    project = Project.load(Path(args.project))
    files: list[Path] = []
    for a in args.paths:
        p = Path(a)
        if p.is_dir():
            files += sorted(x for x in p.iterdir() if x.suffix.lower() in (".jpg", ".jpeg", ".png"))
        elif p.is_file():
            files.append(p)
        else:
            raise ProjectError(f"ファイルがありません: {p}")
    with project.lock():
        after = args.after
        for i, f in enumerate(files, 1):
            seg = pipeline.add_photo(project, f, after)
            if after:
                after = seg["id"]  # 指定位置の後ろへ順に並べる
            if i % 50 == 0 or i == len(files):
                _progress("import", i / len(files), f"写真を追加 {i}/{len(files)}")
        pipeline.update_order(project)
        project.save()
    print(f"{len(files)} 枚を追加しました。")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    project = Project.load(Path(args.project))
    pipeline.update_order(project)
    if args.json:
        print(json.dumps({"summary": pipeline.summary(project), "review": pipeline.review_items(project)},
                         ensure_ascii=False, indent=2))
    else:
        _print_review(project)
    return 0


def _find(project: Project, target: str) -> tuple[str, dict]:
    if target in project.data["pages"]:
        return "page", project.data["pages"][target]
    return "segment", project.segment(target)


def cmd_choose(args: argparse.Namespace) -> int:
    project = Project.load(Path(args.project))
    with project.lock():
        seg = project.segment(args.segment)
        if args.frame not in seg["candidates"]:
            raise ProjectError(f"{args.frame} は {seg['id']} の候補ではありません: {seg['candidates']}")
        seg["chosen"] = args.frame
        seg["manual"] = True
        project.save()
    print(f"{seg['id']} の採用を {args.frame} にしました。`vbs run` で該当ページだけ再処理します。")
    return 0


def cmd_include(args: argparse.Namespace, value: bool) -> int:
    project = Project.load(Path(args.project))
    with project.lock():
        for t in args.ids:
            kind, obj = _find(project, t)
            if kind == "segment" and value and not obj["candidates"]:
                raise ProjectError(f"{t} には候補画像がありません。")
            obj["include"] = value
            if kind == "segment":
                obj["manual"] = True
        pipeline.update_order(project)
        project.save()
    return 0


def cmd_ack(args: argparse.Namespace) -> int:
    project = Project.load(Path(args.project))
    with project.lock():
        for t in args.ids:
            if t == "page_count":
                project.data.setdefault("acknowledged", []).append("page_count")
                continue
            kind, obj = _find(project, t)
            codes = args.codes or [w for w in obj["warnings"] + (pipeline.ocr_warnings(obj) if kind == "page" else [])]
            obj["acknowledged"] = sorted(set(obj.get("acknowledged", [])) | set(codes))
            if kind == "segment":
                obj["manual"] = True
        project.save()
    return 0


def cmd_gutter(args: argparse.Namespace) -> int:
    project = Project.load(Path(args.project))
    with project.lock():
        seg = project.segment(args.segment)
        geo = seg.get("geometry") or {}
        if geo.get("frame_id") != seg["chosen"]:
            raise ProjectError("先に `vbs run` で自動検出を行ってください。")
        frame = project.data["frames"][seg["chosen"]]
        x = int(round(args.x * frame["width"])) if args.x <= 1 else int(args.x)
        seg["manual_geometry"] = {"frame_id": seg["chosen"], "bbox": geo["bbox"], "gutter_x": x}
        seg["manual"] = True
        project.save()
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    project = Project.load(Path(args.project))
    with project.lock():
        pipeline.cleanup_stale(project)
        pipeline.update_order(project)
        rec = pipeline.export_pdf(project, Path(args.output) if args.output else None, force=args.force)
    print(f"PDFを出力しました: {rec['path']}（{rec['pages']}ページ）")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from vbs.verify import load_terms, verify_pdf, write_report

    project = Project.load(Path(args.project))
    if not project.data["exports"]:
        raise ProjectError("まだPDFを出力していません。")
    exp = project.data["exports"][-1]
    pdf = Path(args.pdf or exp["path"])
    order = exp.get("order") or project.data["order"]
    terms_path = args.terms or project.settings.get("verify_terms")
    rec = verify_pdf(pdf, order, load_terms(project.abs(terms_path)) if terms_path else None)
    out = write_report(rec, pdf)
    print(f"PDF: {rec['pdf']}")
    print(f"SHA-256: {rec['sha256']}  ({rec['size']:,} bytes, {rec['pages']}ページ, ページ数一致={rec['page_count_ok']})")
    print(f"検索エンジン: {rec['engine']}")
    empty = [p["pdf_page"] for p in rec["per_page"] if p["text_chars"] == 0]
    if empty:
        print(f"文字層が空のページ: {empty}")
    ts = rec.get("terms_summary")
    if ts:
        print(f"重要語: {ts['found']}/{ts['total']} が期待したページで検索できた")
        for pid, miss in ts["missing"].items():
            pno = next(p["pdf_page"] for p in rec["per_page"] if p["page_id"] == pid)
            print(f"  {pno}ページ目 ({pid}) 見つからない: {'、'.join(miss)}")
    print(f"記録: {out}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from vbs.server import serve
    serve(Path(args.project) if args.project else None, port=args.port, open_browser=not args.no_browser)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vbs", description="ページめくり動画から検索可能PDFを作る")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_new(p: argparse.ArgumentParser) -> None:
        p.add_argument("--single", action="store_true", help="片ページ撮影（左右分割しない）")
        p.add_argument("--rtl", action="store_true", help="右から左へ読む本")
        p.add_argument("--expected-pages", type=int, help="予定ページ数（任意）")
        p.add_argument("--page-height-mm", type=float, help="紙面の高さmm（既定257=B5）")
        p.add_argument("--title")
        p.add_argument("--link", action="store_true", help="動画をコピーせず参照する")
        p.add_argument("--allow-hdr", action="store_true")
        p.add_argument("--always-review", action="store_true", help="要確認がなくても自動出力しない")

    p = sub.add_parser("scan", help="動画からPDFまで一括（推奨）")
    p.add_argument("videos", nargs="+")
    p.add_argument("-p", "--project", help="プロジェクトの場所（既定: 動画名_scan）")
    p.add_argument("--force", action="store_true", help="要確認があっても出力")
    p.add_argument("--ui", action="store_true", help="要確認があれば確認画面を開く")
    common_new(p)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("new", help="プロジェクトを作成して動画を追加")
    p.add_argument("project")
    p.add_argument("videos", nargs="*")
    common_new(p)
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("add", help="動画を追加")
    p.add_argument("project")
    p.add_argument("videos", nargs="+")
    p.add_argument("--link", action="store_true")
    p.add_argument("--allow-hdr", action="store_true")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("run", help="未処理・変更分を処理")
    p.add_argument("project")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--no-ocr", action="store_true", help="抽出とページ画像の作成までで止める")
    p.add_argument("--workers", type=int, help="OCRの同時実行数")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("add-photos", help="写真でページを追加（フォルダ可）")
    p.add_argument("project")
    p.add_argument("paths", nargs="+")
    p.add_argument("--after", help="この区間IDの後ろに並べる")
    p.set_defaults(func=cmd_add_photos)

    from vbs.bench.cli import add_bench_parser
    add_bench_parser(sub)

    p = sub.add_parser("status", help="状態と要確認の一覧")
    p.add_argument("project")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("choose", help="区間の採用候補を変更")
    p.add_argument("project")
    p.add_argument("segment")
    p.add_argument("frame")
    p.set_defaults(func=cmd_choose)

    p = sub.add_parser("include", help="区間/ページを採用")
    p.add_argument("project")
    p.add_argument("ids", nargs="+")
    p.set_defaults(func=lambda a: cmd_include(a, True))

    p = sub.add_parser("exclude", help="区間/ページを除外")
    p.add_argument("project")
    p.add_argument("ids", nargs="+")
    p.set_defaults(func=lambda a: cmd_include(a, False))

    p = sub.add_parser("ack", help="確認済みにする（警告を承知で残す）")
    p.add_argument("project")
    p.add_argument("ids", nargs="+")
    p.add_argument("--codes", nargs="*", help="対象の理由コード（省略時は全部）")
    p.set_defaults(func=cmd_ack)

    p = sub.add_parser("gutter", help="綴じ目位置を手で指定（0-1の比率またはpx）")
    p.add_argument("project")
    p.add_argument("segment")
    p.add_argument("x", type=float)
    p.set_defaults(func=cmd_gutter)

    p = sub.add_parser("export", help="PDFを出力")
    p.add_argument("project")
    p.add_argument("-o", "--output")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("verify", help="出力したPDFを開き直して検索を確認（ハッシュ付きで記録）")
    p.add_argument("project")
    p.add_argument("--pdf", help="既定は最後に出力したPDF")
    p.add_argument("--terms", help="重要語の表 {ページID: [語,...]}（JSON）")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("ui", help="確認画面を開く")
    p.add_argument("project", nargs="?")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_ui)
    return ap


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ProjectError, VideoError, OcrError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n中断しました（確定済みの成果物は保持されています）", file=sys.stderr)
        return 130
    except OSError as e:
        print(f"エラー: 書き込み・読み込みに失敗したため止めました（{e}）。確定済みの成果物は保持されています。"
              "原因を取り除いてから同じコマンドで再開できます。", file=sys.stderr)
        return 3
