# Third-party notices

初回公開は source-only です。以下の外部資産は配布物へ同梱せず、利用者が各公式配布元
から取得します。候補生成時には使用バージョン、取得元、SHA-256、ライセンス原文を
再確認します。

`requirements-lock-win-x64.txt` が固定する Python wheel は配布物へ同梱しませんが、
installer が利用者環境へ導入します。版とライセンス表記は取得済み wheel の
`.dist-info/METADATA` と同梱ライセンスを候補更新時に再確認します。

| Locked Python distribution | Version       | Use                     | Bundled | License metadata                              |
| -------------------------- | ------------- | ----------------------- | ------: | --------------------------------------------- |
| build                      | 1.5.0         | development wheel build |      No | MIT                                           |
| colorama                   | 0.4.6         | pytest dependency       |      No | BSD-3-Clause (bundled license file)           |
| iniconfig                  | 2.3.0         | pytest dependency       |      No | MIT                                           |
| NumPy                      | 2.4.6 / 2.5.1 | array operations        |      No | BSD-3-Clause plus bundled third-party notices |
| opencv-python-headless     | 5.0.0.93      | image operations        |      No | Apache-2.0                                    |
| packaging                  | 26.3          | build/test dependency   |      No | Apache-2.0 OR BSD-2-Clause                    |
| pluggy                     | 1.6.0         | pytest dependency       |      No | MIT                                           |
| Pygments                   | 2.20.0        | pytest dependency       |      No | BSD-2-Clause                                  |
| pyproject-hooks            | 1.2.0         | build dependency        |      No | MIT (bundled license file)                    |
| pytest                     | 9.1.1         | development/test        |      No | MIT                                           |
| Ruff                       | 0.16.2        | development lint        |      No | MIT                                           |
| setuptools                 | 83.0.0        | editable build backend  |      No | MIT                                           |
| wheel                      | 0.47.0        | editable build support  |      No | MIT                                           |

| Component               | Use                                     | Bundled | License / source                                                                      |
| ----------------------- | --------------------------------------- | ------: | ------------------------------------------------------------------------------------- |
| NumPy                   | optional array operations               |      No | BSD-3-Clause; https://github.com/numpy/numpy/blob/main/LICENSE.txt                    |
| OpenCV                  | optional stabilization/image operations |      No | Apache-2.0; https://github.com/opencv/opencv/blob/4.x/LICENSE                         |
| pytest                  | development/test only                   |      No | MIT; https://github.com/pytest-dev/pytest/blob/main/LICENSE                           |
| Ruff                    | development lint only                   |      No | MIT; https://github.com/astral-sh/ruff/blob/main/LICENSE.md                           |
| build                   | development wheel build only            |      No | MIT; https://github.com/pypa/build/blob/main/LICENSE                                  |
| FFmpeg / ffprobe        | analysis and encode/decode              |      No | Obtain and review the selected build license at https://ffmpeg.org/legal.html         |
| Real-ESRGAN             | optional upscale                        |      No | BSD-3-Clause; https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE              |
| Real-ESRGAN-ncnn-vulkan | optional upscale executable             |      No | MIT; https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/blob/master/LICENSE           |
| RIFE-ncnn-vulkan        | optional interpolation executable       |      No | Review the selected release license at https://github.com/nihui/rife-ncnn-vulkan      |
| CodeFormer              | optional local face restore             |      No | Review the selected repository/model licenses at https://github.com/sczhou/CodeFormer |

The project does not redistribute any listed binary, model, sample media, or CodeFormer
environment. A future binary distribution must be treated as a separate candidate with a new
license audit.
