from __future__ import annotations

import pytest

from video_enhance_lite.media_streams import (
    MediaStreamInfo,
    UnsupportedMediaStreamsError,
    resolve_mp4_stream_preservation,
)


def _stream(index: int, codec_type: str, codec_name: str, **kwargs) -> MediaStreamInfo:
    return MediaStreamInfo(index, codec_type, codec_name, **kwargs)


def test_resolve_mp4_stream_preservation_keeps_all_supported_streams_in_input_order():
    streams = (
        _stream(0, "video", "h264"),
        _stream(1, "audio", "aac", language="eng", disposition=("default",)),
        _stream(2, "audio", "ac3", language="jpn"),
        _stream(3, "subtitle", "mov_text", language="eng", disposition=("forced",)),
    )

    plan = resolve_mp4_stream_preservation(streams, primary_video_index=0)

    assert plan.audio_stream_indices == (1, 2)
    assert plan.subtitle_stream_indices == (3,)
    assert plan.has_audio is True
    assert plan.has_subtitle is True


@pytest.mark.parametrize(
    ("unsupported", "message"),
    [
        (_stream(1, "audio", "flac"), "MP4非互換audio codec"),
        (_stream(1, "subtitle", "hdmv_pgs_subtitle"), "image subtitle"),
        (_stream(1, "attachment", "ttf"), "attachment stream"),
        (_stream(1, "data", "bin_data"), "data stream"),
        (
            _stream(1, "video", "mjpeg", disposition=("attached_pic",)),
            "追加video/attachment",
        ),
    ],
)
def test_resolve_mp4_stream_preservation_rejects_silent_loss(
    unsupported: MediaStreamInfo,
    message: str,
):
    with pytest.raises(UnsupportedMediaStreamsError, match=message) as exc_info:
        resolve_mp4_stream_preservation(
            (_stream(0, "video", "h264"), unsupported),
            primary_video_index=0,
        )

    assert f"index={unsupported.index}" in str(exc_info.value)
    assert unsupported.codec_name in str(exc_info.value)


def test_resolve_mp4_stream_preservation_rejects_missing_primary_video():
    with pytest.raises(UnsupportedMediaStreamsError, match="primary video"):
        resolve_mp4_stream_preservation(
            (_stream(1, "audio", "aac"),),
            primary_video_index=0,
        )
