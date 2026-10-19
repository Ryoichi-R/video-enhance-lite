# video_enhance_lite package

この package は `video-enhance-lite` の実行本体です。

- 公開 stage の正本: `stages.STAGE_REGISTRY`
- 公開表示: 11 組込み段階 + 任意の CodeFormer 顔復元
- 設定: `%LOCALAPPDATA%\video-enhance-lite\config\`
- temp: `%LOCALAPPDATA%\video-enhance-lite\temp\`
- 既定出力: `%USERPROFILE%\Videos\video-enhance-lite\`

CLI の外部能力診断:

```text
python -m video_enhance_lite --self-check
```

ソース checkout の親、`site-packages`、package 隣接フォルダを実行時データや
外部バイナリの既定場所として扱いません。
