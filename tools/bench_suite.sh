#!/usr/bin/env bash
# 必須試験をまとめて順に実行する（長時間。手動で起動する）
#   bash tools/bench_suite.sh resume     中断・再開・局所失敗の試験
#   bash tools/bench_suite.sh endurance  30分入力・長い静止・300ページ
# 手元の実写動画は VIDEO=動画のパス で指定する（リポジトリには含めない）
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
IN=work/bench/inputs
OUT=work/bench
VIDEO=${VIDEO:-samples/sample.MOV}
SUMMARY=$OUT/suite-$1-$(date +%Y%m%d-%H%M%S).txt

run() {
  echo "=== $* ($(date +%H:%M:%S))" | tee -a "$SUMMARY"
  "$PY" -m vbs bench "$@" 2>&1 | grep -E "^\s+\[(OK|NG)\]|差分|合格|結果|\"(returncode|elapsed_sec|peak_private_mib|pages|segments|order_ok)\"" | tee -a "$SUMMARY"
}

if [ "$1" = "resume" ]; then
  run resume-test extract-pass1 --video "$VIDEO" --frac 0.5          # 元のVFR動画
  run resume-test extract-pass1 --video $IN/concat03.mov --frac 0.45
  run resume-test extract-pass2 --video $IN/concat03.mov --frac 0.5
  run resume-test split         --video $IN/concat03.mov --frac 0.5
  run resume-test write-fail    --video "$VIDEO"
  PREP=$OUT/prepared-$(basename "$VIDEO")
  run resume-test ocr-after --video "$VIDEO" --prepared-dir $PREP --k 6
  run resume-test ocr-mid   --video "$VIDEO" --prepared-dir $PREP --k 6
  run resume-test export    --video "$VIDEO" --prepared-dir $PREP
  run resume-test ocr-fail  --video "$VIDEO" --prepared-dir $PREP --page s0013-R
elif [ "$1" = "endurance" ]; then
  run run --video $IN/concat10.mov --no-ocr --interval 5
  run run --video $IN/concat30.mov --no-ocr --interval 5
  run run --video $IN/still05.mp4 --interval 5 --workers 4
  run run --pages $IN/pages300 --interval 5 --workers 4
fi
echo "summary: $SUMMARY"
