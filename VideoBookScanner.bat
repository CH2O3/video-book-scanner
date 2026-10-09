@echo off
chcp 65001 >nul
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo 先に setup.bat を実行してください。
  pause
  exit /b 1
)
if "%~1"=="" (
  rem ダブルクリック: 確認画面（プロジェクトを開く／作る）
  "%PY%" -m vbs ui
) else (
  rem 動画をドラッグ: 動画名_scan フォルダを作って最後まで処理。要確認があれば確認画面を開く
  "%PY%" -m vbs scan --ui %*
)
pause
