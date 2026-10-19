# CodeFormer の移行

この文書は、既存の CodeFormer local 設定を新しいユーザー設定へ移すための専用手順です。
一般設定の移行は `MIGRATE-CONFIG.md` を参照してください。

## 前提

- CodeFormer は任意のローカル機能であり、配布物へ同梱しません
- 自動ダウンロード、クラウド API、トークンは使用しません
- CodeFormer 本体とモデルのライセンス・入手元を利用者が確認してください

## 設定場所

新しい設定ファイル:

```text
%LOCALAPPDATA%\video-enhance-lite\config\enhance-binaries.json
```

旧 `.local\enhance-binaries.json` を見つけても自動削除・自動上書きはしません。
内容を確認し、バックアップを保持したまま、ユーザー設定へ必要な値だけを移します。

## 設定例

```json
{
  "roots": {
    "codeformer-local": "<codeformer-root>"
  },
  "executables": {
    "codeformer-python": "<codeformer-venv>\\Scripts\\python.exe"
  },
  "models": {
    "codeformer": {
      "path": "<codeformer-root>\\weights\\CodeFormer\\codeformer.pth"
    },
    "facelib_detection": {
      "path": "<codeformer-root>\\weights\\facelib\\detection_Resnet50_Final.pth"
    },
    "facelib_parsing": {
      "path": "<codeformer-root>\\weights\\facelib\\parsing_parsenet.pth"
    }
  }
}
```

stock `inference_codeformer.py` の契約に合わせ、3つのモデルは CodeFormer repo 配下に
あり、期待されるファイル名を使う必要があります。移行後は次で確認します。

```text
<venv-python> -m video_enhance_lite --self-check
```

Face restore が `available` になるまで、顔復元を選択した処理を開始しないでください。
