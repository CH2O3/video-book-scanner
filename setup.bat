@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo [1/3] Python仮想環境を準備します
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv || (echo Pythonが見つかりません。https://www.python.org/ から入れてください & pause & exit /b 1)
)
".venv\Scripts\python.exe" -m pip install -q --upgrade pip
".venv\Scripts\python.exe" -m pip install -q -e ".[dev]" || (echo パッケージの導入に失敗しました & pause & exit /b 1)

echo [2/3] Tesseract OCR を確認します
if not exist "%ProgramFiles%\Tesseract-OCR\tesseract.exe" (
  echo Tesseractを導入します（winget）
  winget install --id UB-Mannheim.TesseractOCR -e --accept-source-agreements --accept-package-agreements --silent
)
if not exist "%ProgramFiles%\Tesseract-OCR\tesseract.exe" (
  echo Tesseractを導入できませんでした。README の「困ったとき」を見て手動で入れてから、もう一度実行してください
  pause & exit /b 1
)

echo [3/3] 日本語の学習データを取得します（tessdata_best）
if not exist tessdata mkdir tessdata
for %%L in (jpn jpn_vert eng) do (
  if not exist "tessdata\%%L.traineddata" curl -sSLf -o "tessdata\%%L.traineddata" "https://github.com/tesseract-ocr/tessdata_best/raw/main/%%L.traineddata"
)
if not exist "tessdata\osd.traineddata" copy /y "%ProgramFiles%\Tesseract-OCR\tessdata\osd.traineddata" tessdata >nul
rem PDFの透明文字層に使う、Tesseract付属のフォント（Apache License 2.0）
if not exist "tessdata\pdf.ttf" copy /y "%ProgramFiles%\Tesseract-OCR\tessdata\pdf.ttf" tessdata >nul
if not exist "tessdata\jpn.traineddata" (
  echo 日本語の学習データを取得できませんでした。インターネット接続を確認して、もう一度実行してください
  pause & exit /b 1
)

echo.
echo 準備できました。VideoBookScanner.bat に動画をドラッグするか、ダブルクリックで起動します。
pause
