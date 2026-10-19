# 設定の移行

公開版は source tree の `.local`、checkout の親、package の親へ設定を書き込みません。
ユーザー設定は次に保存します。

```text
%LOCALAPPDATA%\video-enhance-lite\config\enhance-binaries.json
```

旧設定がある場合は、まずファイルを別名でバックアップし、内容を目視確認してから
必要な root と hash だけを新しい場所へコピーします。旧設定の自動削除、自動上書き、
旧実行ファイルの無条件採用は行いません。

一時領域は `%LOCALAPPDATA%\video-enhance-lite\temp\`、出力は
`%USERPROFILE%\Videos\video-enhance-lite\` が既定です。必要なら
`VIDEO_ENHANCE_CONFIG_ROOT`、`VIDEO_ENHANCE_TEMP_ROOT`、`VIDEO_ENHANCE_OUTPUT_ROOT`
で個別に変更できます。

移行後:

```text
<venv-python> -m video_enhance_lite --self-check
```

診断が `available` にならない能力は、表示された設定場所・拒否理由・公式 URL を確認
してください。
