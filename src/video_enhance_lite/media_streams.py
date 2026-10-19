from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MediaStreamInfo",
    "StreamPreservationPlan",
    "UnsupportedMediaStreamsError",
    "legacy_stream_preservation_plan",
    "resolve_mp4_stream_preservation",
]


_MP4_COPY_AUDIO_CODECS = frozenset({"aac", "ac3", "alac", "eac3", "mp3"})
_MP4_COPY_SUBTITLE_CODECS = frozenset({"mov_text"})


@dataclass(frozen=True)
class MediaStreamInfo:
    index: int
    codec_type: str
    codec_name: str
    language: str | None = None
    disposition: tuple[str, ...] = ()

    def describe(self) -> str:
        language = f", language={self.language}" if self.language else ""
        disposition = f", disposition={','.join(self.disposition)}" if self.disposition else ""
        return (
            f"stream index={self.index}, type={self.codec_type}, "
            f"codec={self.codec_name or 'unknown'}{language}{disposition}"
        )


@dataclass(frozen=True)
class StreamPreservationPlan:
    primary_video_index: int
    audio_streams: tuple[MediaStreamInfo, ...]
    subtitle_streams: tuple[MediaStreamInfo, ...]
    inventory_complete: bool = True

    @property
    def audio_stream_indices(self) -> tuple[int, ...]:
        return tuple(stream.index for stream in self.audio_streams)

    @property
    def subtitle_stream_indices(self) -> tuple[int, ...]:
        return tuple(stream.index for stream in self.subtitle_streams)

    @property
    def has_audio(self) -> bool:
        return bool(self.audio_streams)

    @property
    def has_subtitle(self) -> bool:
        return bool(self.subtitle_streams)


class UnsupportedMediaStreamsError(ValueError):
    def __init__(self, issues: tuple[str, ...]):
        self.issues = issues
        detail = "; ".join(issues)
        super().__init__(
            f"MP4へ無変換で保持できないストリームがあります。処理は開始していません: {detail}"
        )


def resolve_mp4_stream_preservation(
    streams: tuple[MediaStreamInfo, ...],
    *,
    primary_video_index: int,
) -> StreamPreservationPlan:
    """Resolve the fail-closed MP4 stream-copy contract without side effects."""
    audio_streams: list[MediaStreamInfo] = []
    subtitle_streams: list[MediaStreamInfo] = []
    issues: list[str] = []
    seen_indices: set[int] = set()
    primary_found = False

    for stream in streams:
        codec_type = stream.codec_type.strip().lower()
        codec_name = stream.codec_name.strip().lower()
        dispositions = {item.strip().lower() for item in stream.disposition}

        if stream.index < 0:
            issues.append(f"invalid stream index: {stream.describe()}")
            continue
        if stream.index in seen_indices:
            issues.append(f"duplicate stream index: {stream.describe()}")
            continue
        seen_indices.add(stream.index)

        if codec_type == "video":
            if stream.index == primary_video_index and "attached_pic" not in dispositions:
                primary_found = True
            else:
                issues.append(f"追加video/attachmentは保持対象外: {stream.describe()}")
        elif codec_type == "audio":
            if codec_name in _MP4_COPY_AUDIO_CODECS:
                audio_streams.append(stream)
            else:
                issues.append(f"MP4非互換audio codec: {stream.describe()}")
        elif codec_type == "subtitle":
            if codec_name in _MP4_COPY_SUBTITLE_CODECS:
                subtitle_streams.append(stream)
            else:
                issues.append(f"MP4非互換またはimage subtitle: {stream.describe()}")
        elif codec_type in {"attachment", "data"}:
            issues.append(f"{codec_type} streamは保持対象外: {stream.describe()}")
        else:
            issues.append(f"未対応stream type: {stream.describe()}")

    if not primary_found:
        issues.append(f"primary video stream index={primary_video_index} がinventoryにありません")
    if issues:
        raise UnsupportedMediaStreamsError(tuple(issues))

    return StreamPreservationPlan(
        primary_video_index=primary_video_index,
        audio_streams=tuple(audio_streams),
        subtitle_streams=tuple(subtitle_streams),
    )


def legacy_stream_preservation_plan(has_audio: bool) -> StreamPreservationPlan:
    """Compatibility for older third-party/test metadata without stream inventory."""
    return StreamPreservationPlan(
        primary_video_index=0,
        audio_streams=(MediaStreamInfo(1, "audio", "unknown"),) if has_audio else (),
        subtitle_streams=(),
        inventory_complete=False,
    )
