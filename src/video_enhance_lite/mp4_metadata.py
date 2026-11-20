"""Store a native MP4 title alongside FFmpeg's arbitrary keyed metadata."""

import struct
from pathlib import Path


def _boxes(stream, start: int, end: int):
    while start < end:
        stream.seek(start)
        header = stream.read(8)
        if len(header) != 8:
            raise ValueError("incomplete MP4 box header")
        size, kind = struct.unpack(">I4s", header)
        header_size = 8
        if size == 1:
            size = int.from_bytes(stream.read(8), "big")
            header_size = 16
        elif size == 0:
            size = end - start
        if not header_size <= size <= end - start:
            raise ValueError("invalid MP4 box size")
        yield kind, start, start + header_size, start + size
        start += size


def _box(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I4s", len(data) + 8, kind) + data


def write_native_title(path: Path, title: str) -> None:
    """Update only the final moov of our unfragmented FFmpeg partial output."""
    data = _box(b"data", struct.pack(">II", 1, 0) + title.encode("utf-8"))
    items = _box(b"ilst", _box(b"\xa9nam", data))
    handler = _box(b"hdlr", bytes(8) + b"mdirappl" + bytes(9))
    standard = _box(b"meta", bytes(4) + handler + items)

    with path.open("r+b") as stream:
        end = stream.seek(0, 2)
        movie = next((b for b in _boxes(stream, 0, end) if b[0] == b"moov"), None)
        if movie is None or movie[3] != end:
            raise ValueError("expected final moov in FFmpeg output")
        _, movie_start, children_start, _ = movie
        user_data = next((b for b in _boxes(stream, children_start, end) if b[0] == b"udta"), None)
        keyed = b""
        other = bytearray()
        write_at = end
        if user_data:
            _, start, payload, stop = user_data
            for kind, offset, _, box_end in _boxes(stream, payload, stop):
                stream.seek(offset)
                contents = stream.read(box_end - offset)
                if kind == b"meta":
                    keyed = contents
                else:
                    other.extend(contents)  # e.g. the chapter list
            if stop == end:
                write_at = start
            else:
                stream.seek(start + 4)
                stream.write(b"free")
        # Keep the keyed meta at movie level and the conventional title under
        # udta/meta/ilst. This also preserves the byte offsets of all media.
        stream.seek(write_at)
        stream.write(keyed)
        stream.write(_box(b"udta", standard + other))
        movie_size = stream.tell() - movie_start
        stream.truncate()
        if children_start - movie_start == 16:
            stream.seek(movie_start + 8)
            stream.write(struct.pack(">Q", movie_size))
        else:
            stream.seek(movie_start)
            stream.write(struct.pack(">I", movie_size))
