# video-enhance-lite

Windows 11 x64 でローカル動画を解析し、利用者が選択した補正だけを実行する
Python アプリです。入力動画や診断情報をクラウドへ送信しません。

## 概要

- `ffprobe` による入力解析と、利用可能な補正の提案
- 11 組込み段階: デインターレース、手ぶれ補正、フリッカー除去、ノイズ除去、
  デバンディング、色補正、シャープ化、アップスケール、フレーム補間、粒状感調整、
  最終エンコード
- 任意の外部能力: Real-ESRGAN、RIFE、CodeFormer 顔復元
- GUI と CLI の同じ能力診断、明示的な外部バイナリ許可ルート、SHA-256 検証

CodeFormer は外部環境でのみ動く任意の顔復元段階です。そのため公開表示は
「11 組込み段階 + 任意の顔復元1段階」とし、単純な「12段」とは表示しません。
正本は `src/video_enhance_lite/stages.py` の `STAGE_REGISTRY` です。

リポジトリ名と CLI 名は `video-enhance-lite`、Python package 名は
`video_enhance_lite` です。

## 動作要件

- Windows 11 x64
- Python 3.11〜3.13（package metadata: `>=3.11,<3.14`）
- GPU は必須ではありませんが、Real-ESRGAN/RIFE は Vulkan 対応環境が必要です

FFmpeg、Real-ESRGAN、RIFE、CodeFormer、モデル、サンプルメディアは配布物に
同梱しません。外部エンジンは各公式配布元から利用者が取得してください。

## インストール

ソースを任意の読み取り可能な場所へ展開し、プロジェクトルートで実行します。

```bat
cd <project-root>
install-enhance.bat
```

インストーラーは次の場所にユーザー専用 venv を作り、現在の checkout と install
receipt を結び付けます。任意の `pythonw.exe` へは fallback しません。
依存関係はWindows x64専用の `requirements-lock-win-x64.txt` に固定し、wheelの
SHA-256を検証します。source distributionへのfallbackとinstaller実行時の依存再解決は行いません。
事前取得した完全なwheel集合でオフライン検証する場合は、`VIDEO_ENHANCE_WHEELHOUSE`を
対象Python版のwheelhouseへ設定してからinstallerを実行します。

```text
%LOCALAPPDATA%\video-enhance-lite\venv\
%LOCALAPPDATA%\video-enhance-lite\install-root.txt
```

外部エンジンが未導入でも package の導入と GUI 起動は可能です。導入後に診断を
確認してください。

```bat
<project-root>\video-enhance-lite.bat
<venv-python> -m video_enhance_lite --self-check
```

## 設定

設定ファイルの既定場所は次です。

```text
%LOCALAPPDATA%\video-enhance-lite\config\enhance-binaries.json
```

サンプルは `.local/enhance-binaries.example.json` にあります。これは説明用の
プレースホルダーであり、そのまま実行できる設定ではありません。実設定は source
tree の `.local` ではなく、ユーザー領域へ置きます。

### 1. 設定ファイルを準備する

まだ実設定がない場合は、PowerShell 7でプロジェクトルートから次を実行します。
既に `enhance-binaries.json` がある場合はコピーせず、既存JSONへ必要な項目だけを
追加してください。

```powershell
$configDir = Join-Path $env:LOCALAPPDATA 'video-enhance-lite\config'
$configPath = Join-Path $configDir 'enhance-binaries.json'
New-Item -Path $configDir -ItemType Directory -Force | Out-Null
if (Test-Path -LiteralPath $configPath) {
    Write-Host "既存設定を編集してください: $configPath"
} else {
    Copy-Item -LiteralPath '.local\enhance-binaries.example.json' -Destination $configPath
}
```

コピー後、`<ffmpeg-root>` などのプレースホルダーを、各PCで実際に配置した
ディレクトリへ置換します。JSON内のWindowsパスでは `\` を `\\` と記述します。

### 2. 実行ファイルの許可ルートとSHA-256を設定する

`allowed_binary_roots` には実行ファイルを含むディレクトリを能力ごとに明示します。
`expected_sha256` を設定した能力はハッシュ不一致時に停止します。環境変数で一時的に
上書きする場合は `VIDEO_ENHANCE_FFMPEG_ROOTS`、`VIDEO_ENHANCE_FFPROBE_ROOTS`、
`VIDEO_ENHANCE_REALESRGAN_ROOTS`、`VIDEO_ENHANCE_RIFE_ROOTS`、
`VIDEO_ENHANCE_EXPECTED_SHA256_JSON` を使います。

解決順序は明示設定、許可ルート、PATH です。PATH で発見できても許可ルート外なら
拒否します。symlink/junction を含む経路、通常ファイルでない実体、期待名でない
実行ファイルも拒否します。

各実行ファイルのSHA-256はPowerShell 7で確認できます。

```powershell
Get-FileHash -LiteralPath '<ffmpeg-root>\ffmpeg.exe' -Algorithm SHA256
Get-FileHash -LiteralPath '<ffmpeg-root>\ffprobe.exe' -Algorithm SHA256
Get-FileHash -LiteralPath '<realesrgan-root>\realesrgan-ncnn-vulkan.exe' -Algorithm SHA256
Get-FileHash -LiteralPath '<rife-root>\rife-ncnn-vulkan.exe' -Algorithm SHA256
```

得られた値を小文字または大文字の16進数で `expected_sha256` の対応する配列へ記載
します。ローカルで計算したハッシュは以後の改変検出には使えますが、最初から正規
配布物だったことは証明しません。公式配布元、対象バージョン、公式チェックサムが
提供されている場合は、それらを先に照合してください。

### 3. CodeFormer顔復元を設定する

CodeFormerは `allowed_binary_roots` ではなく、repo、実行に使うPython、次の3モデルを
それぞれ設定します。統合形式は `.local/enhance-binaries.example.json`、詳細は
`MIGRATE-CODEFORMER.md` を参照してください。

- `roots.codeformer-local`: `inference_codeformer.py` があるCodeFormer repo
- `executables.codeformer-python`: CodeFormer依存を導入した環境の `python.exe`
- `models.codeformer`: `weights\CodeFormer\codeformer.pth`
- `models.facelib_detection`: `weights\facelib\detection_Resnet50_Final.pth`
- `models.facelib_parsing`: `weights\facelib\parsing_parsenet.pth`

GPUを使う場合は、設定するPython環境でCUDAが利用できることも確認します。

```powershell
& '<codeformer-python>' -c "import torch, cv2, basicsr, facelib; print(torch.cuda.is_available())"
```

### 4. 設定を検証する

設定後はself-checkを実行します。

```powershell
& "$env:LOCALAPPDATA\video-enhance-lite\venv\Scripts\python.exe" `
    -m video_enhance_lite --self-check
```

使用する能力が `available` になったことを確認してください。`misconfigured` の場合は
表示された解決先、許可ルート、ハッシュ、モデル配置を見直します。GUIを開いたまま
設定を変更した場合は、「環境診断」を再実行するかGUIを再起動します。

## 使用方法

```text
<venv-python> -m video_enhance_lite --self-check
<venv-python> -m video_enhance_lite <input-video> --analyze-only
<venv-python> -m video_enhance_lite <input-video> -o <output-dir>
<venv-python> -m video_enhance_lite <input-video> --auto --disable interpolate
```

出力先を省略した場合は `%USERPROFILE%\Videos\video-enhance-lite\` を使います。
一時ファイルは `%LOCALAPPDATA%\video-enhance-lite\temp\`、または
`VIDEO_ENHANCE_TEMP_ROOT`／`--temp-root` で明示した場所です。ソースフォルダの親へ
暗黙に出力しません。

入力ストリームは処理前に全件検査します。出力 MP4 には、処理済み video と、入力順の
全 audio（`aac` / `ac3` / `eac3` / `alac` / `mp3`）、全 `mov_text` subtitle、global・
stream metadata、language、disposition を無変換で保持します。attachment、data、追加
video・cover art、image subtitle、上記以外の audio/subtitle codec がある入力は、silent
drop や暗黙 transcode を行わず、stream index と codec を表示して処理開始前に停止します。

フレームレートは解析時にCFR、VFR suspected、unknownへ分類します。VFR suspectedは
nominal `r_frame_rate`へ明示CFR normalizationしてからPNGへ抽出し、全チャンクへ同じfpsと
global frame budgetを適用します。frame duplication/dropを伴う可能性はCLI/GUIで開始前に
表示します。unknownは推測で処理せず`framerate_preflight`で停止しますが、
`--analyze-only`による診断表示は利用できます。

短尺動画などで解析用の最終sample時刻がlast decodable frameを越えた場合は、最終sample
だけを警告付きで省き、それまでのsampleで解析を継続します。中間sampleの欠落、FFmpegの
失敗、全sampleを取得できない場合は、破損やmetadata不整合を隠さないため従来どおり停止します。

## GUI と診断

GUI の「環境診断」ボタンまたは `--self-check` は、Analyze、Encode/Decode、
Upscale、Interpolate、Face restore、Stabilize を個別に `available`、
`unavailable`、`misconfigured`、`unsupported` と表示します。不要な任意エンジンの
不足は、選択した処理の開始を妨げません。

## セキュリティとプライバシー

- 外部バイナリの自動ダウンロード、クラウド API、認証トークンを使用しません
- 実行ファイルは許可ルート、通常ファイル、reparse point、期待名、任意の SHA-256
  を確認してから実行します
- 設定、temp、出力、venv は公開候補から分離します
- ログを共有する前に、ユーザー名、入力動画名、ローカルパスを確認してください
- 脆弱性報告は `SECURITY.md` を参照してください

## アンインストール

自動アンインストーラーはありません。アプリを終了してから、展開したソース一式と
`%LOCALAPPDATA%\video-enhance-lite\venv\` を削除してください。設定、ログ、一時
ファイルも不要な場合は `%LOCALAPPDATA%\video-enhance-lite\` を削除できます。
生成済みの動画は出力先に残るため、必要に応じて別途管理してください。

## 既知の制限

- 正式サポートは Windows 11 x64 のみです
- 外部エンジンのバージョン、GPU、Vulkan ドライバー、モデルは利用者環境に依存します
- GUI の実表示テストは表示環境が必要です。headless 環境では marker により分離します
- `e2e` テストは実 FFmpeg と `VIDEO_ENHANCE_E2E_RIFE_DIR` がある環境でのみ実行します
- 出力 container は MP4 固定です。非対応ストリームを変換・削除する option は未実装です
- VFR normalizationはsourceの局所PTSを保存する方式ではなく、nominal fpsへの明示CFR化です

## 開発者向け

```text
python -m pip install -e ".[dev,enhance]"
python -m pytest tests/enhance -q -m "not gui and not requires_display and not e2e"
python -m pytest tests/release -q
python -m ruff check src tests
python -m build
```

外部バイナリ、モデル、入力・出力メディア、`.local` の実設定を source tree に追加
しないでください。詳細は `CONTRIBUTING.md` と `THIRD-PARTY-NOTICES.md` を参照します。

## ライセンス

第一者コードは [MIT License](LICENSE) の下で提供します。Copyright (c) 2026
Ryoichi-R and contributors. 第三者コンポーネントにはそれぞれのライセンスが
適用されます。依存関係の境界、ライセンス、公式URLは
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) を参照してください。
