# video-enhance-lite 実行手順

この runbook は公開利用者向けの通常起動、self-check、能力別トラブルシュートだけを
扱います。ローカルベンチマーク、修復バックアップ、開発者固有の実測記録は含めません。

## 1. 導入

```bat
cd <project-root>
install-enhance.bat
```

installerは `requirements-lock-win-x64.txt` の固定version・SHA-256とwheel-only制約を
適用します。ネットワークを使わずに再検証する場合は、対象Python版の完全なwheel集合を
用意し、`VIDEO_ENHANCE_WHEELHOUSE=<wheelhouse>` を設定してから実行してください。

launcher は `%LOCALAPPDATA%\video-enhance-lite\venv\Scripts\pythonw.exe` と
install receipt を使います。別 checkout の venv や PATH 上の任意 pythonw へは
fallback しません。別の場所へ展開した場合は、その場所で `install-enhance.bat` を
再実行してください。

## 2. 初回診断

```text
<venv-python> -m video_enhance_lite --self-check
```

診断は次の能力を個別に表示します。

| capability    | 依存                         | 未導入時の影響             |
| ------------- | ---------------------------- | -------------------------- |
| Analyze       | ffprobe                      | 解析を開始できない         |
| Encode/Decode | ffmpeg                       | 動画処理を開始できない     |
| Upscale       | Real-ESRGAN                  | アップスケールだけ選べない |
| Interpolate   | RIFE                         | フレーム補間だけ選べない   |
| Face restore  | CodeFormer local             | 顔復元だけ選べない         |
| Stabilize     | FFmpeg vidstab または OpenCV | 手ぶれ補正が制限される     |

## 3. 外部バイナリ設定

実設定は次です。

```text
%LOCALAPPDATA%\video-enhance-lite\config\enhance-binaries.json
```

例は `.local\enhance-binaries.example.json` にあります。実行ファイル、モデル、
ハッシュは例へ書き込まないでください。外部資産は source tree の外へ置き、公式配布元、
バージョン、SHA-256 を自分で記録してください。

初回の設定ファイル作成、既存JSONへの追記、SHA-256取得、CodeFormerのrepo／Python／
3モデルを含む統合設定、self-check、GUI再起動の手順は `README.md` の
「外部エンジンの設定」を参照してください。

環境変数は次の順で設定できます。

```text
VIDEO_ENHANCE_FFMPEG_ROOTS
VIDEO_ENHANCE_FFPROBE_ROOTS
VIDEO_ENHANCE_REALESRGAN_ROOTS
VIDEO_ENHANCE_RIFE_ROOTS
VIDEO_ENHANCE_EXPECTED_SHA256_JSON
```

解決できても許可ルート外なら拒否されます。symlink/junction 経由の境界越え、期待名
以外の exe、SHA-256 不一致は安全側で停止します。

## 4. 実行

分析または動画処理中にウィンドウを閉じると、処理を中止して終了するか確認します。
承認した場合は外部プロセスへキャンセルを伝播し、worker停止と一時ファイルの後処理が
完了してからウィンドウを閉じます。終了待機中に強制終了しないでください。

```text
<venv-python> -m video_enhance_lite <input-video> --analyze-only
<venv-python> -m video_enhance_lite <input-video> -o <output-dir>
<venv-python> -m video_enhance_lite <input-video> --auto
<venv-python> -m video_enhance_lite <input-video> --auto-all
```

GUI は `video-enhance-lite.bat` で起動します。空の出力フォルダ欄は
`%USERPROFILE%\Videos\video-enhance-lite\` を使います。temp は
`%LOCALAPPDATA%\video-enhance-lite\temp\`、または `--temp-root`／
`VIDEO_ENHANCE_TEMP_ROOT` で指定した場所です。

## 5. 段階数

公開表示は `src\video_enhance_lite\stages.py` の registry に従い、
「11 組込み段階 + 任意の顔復元1段階」です。CodeFormer がない環境でも他の処理を
利用できます。

## 6. トラブルシュート

### ffprobe / ffmpeg

公式 FFmpeg 配布物を導入し、`ffmpeg.exe` と `ffprobe.exe` のディレクトリを明示設定
または PATH に追加します。PATH を変更した直後は、既存の端末や GUI を閉じて再起動
してください。診断で `misconfigured` なら、発見先が許可ルート内か確認します。

実行前の `stream_preflight` で停止した場合は、表示された stream index、type、codec を
確認してください。MP4へ無変換で保持する対応範囲は、audio が `aac` / `ac3` / `eac3` /
`alac` / `mp3`、subtitle が `mov_text` です。全対応 audio/subtitle と metadata は保持し、
attachment、data、追加video・cover art、image subtitle、非対応codecは削除せず拒否します。
入力を別途変換する場合は、元ファイルを保持したうえで明示的に行ってください。

`vfr_suspected`と表示された入力は、PNG抽出前に表示されたnominal `r_frame_rate`へCFR
normalizationします。全チャンクのframe budgetはsource全体の時間軸から割り当て、出力
durationとaudio終端を維持します。duplication/dropは仕様上発生し得ます。`unknown`は
安全なtargetを決められないため実行を拒否します。原因確認には`--analyze-only --verbose`を
使い、`r_frame_rate`、`avg_frame_rate`、入力container/PTSを確認してください。

短尺動画の解析で`final analyzer sample was beyond the last decodable frame`と表示された場合、
最終sampleだけを省いて解析を継続した通知です。処理を中止するerrorではありません。最終以外の
sample欠落やFFmpeg失敗は入力破損・duration metadata不整合の可能性があるため停止します。

### Real-ESRGAN / RIFE

公式 release から取得した実行ファイルとモデルを source tree の外へ展開し、対応する
root を設定します。Vulkan、GPU driver、モデル配置を確認してください。これらがない
場合はアップスケール／補間を OFF にすれば解析や他の段階を使えます。

### CodeFormer

`MIGRATE-CODEFORMER.md` の専用設定を確認します。repo、python、3つのモデルが明示され、
stock CLI の想定ファイル名と一致する必要があります。

### Stabilize

FFmpeg の `vidstabdetect`／`vidstabtransform` が無い場合、OpenCV fallback が利用できる
ことがあります。診断で `unsupported` と表示される場合は、FFmpeg build または OpenCV
依存を確認してください。

### 出力・temp

既定先への書き込み権限がない場合は、`VIDEO_ENHANCE_OUTPUT_ROOT` または CLI の
`--output`／`--temp-root` で書き込み可能な場所を明示します。ソースフォルダや package
の親を手動で設定する必要はありません。

## 7. 安全なログ共有

ログには入力名、設定先、解決したバイナリパスが含まれる場合があります。共有前に
ユーザー名、個人フォルダ名、入力動画名、環境変数の値をマスクしてください。
