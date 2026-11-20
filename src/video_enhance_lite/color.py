"""Colour conversion on the two sides of the 8-bit RGB processing pipeline."""

from dataclasses import dataclass


_HDR_NAMES = {"smpte2084": "PQ", "arib-std-b67": "HLG"}
_MATRICES = {"bt709", "smpte170m", "bt470bg", "bt2020nc", "fcc", "smpte240m"}
_PRIMARIES = {
    "bt709",
    "bt470m",
    "bt470bg",
    "smpte170m",
    "smpte240m",
    "film",
    "bt2020",
    "smpte428",
    "smpte431",
    "smpte432",
    "ebu3213",
    "jedec-p22",
}
_TRANSFERS = {
    "bt709",
    "gamma22",
    "gamma28",
    "smpte170m",
    "smpte240m",
    "linear",
    "log100",
    "log316",
    "iec61966-2-4",
    "bt1361e",
    "iec61966-2-1",
    "bt2020-10",
    "bt2020-12",
    "smpte428",
}
_RANGES = {"pc": "full", "tv": "limited"}


@dataclass(frozen=True)
class ColorConversion:
    extract: str = ""
    encode: str = ""
    tags: tuple[str, ...] = ()


def hdr_name(meta) -> str | None:
    return _HDR_NAMES.get(getattr(meta, "color_transfer", ""))


def conversion(meta, hdr_to_sdr: bool = False) -> ColorConversion:
    matrix = getattr(meta, "color_space", "unknown")
    primaries = getattr(meta, "color_primaries", "unknown")
    transfer = getattr(meta, "color_transfer", "unknown")
    level = getattr(meta, "color_range", "unknown")
    hdr = hdr_name(meta)
    if hdr:
        if not hdr_to_sdr:
            raise ValueError(f"{hdr} HDR input requires --hdr-to-sdr (BT.709 SDR)")
        if (
            matrix not in _MATRICES
            or primaries not in ("bt2020", "bt709", "smpte432")
            or level not in _RANGES
        ):
            raise ValueError(f"{hdr} conversion needs known matrix, primaries and range")
        peak = max(
            1.0, (getattr(meta, "peak_nits", None) or (10000.0 if hdr == "PQ" else 1000.0)) / 100
        )
        extract = (
            f"zscale=matrixin={matrix}:primariesin={primaries}:transferin={transfer}:"
            f"rangein={_RANGES[level]}:transfer=linear:npl=100,format=gbrpf32le,"
            f"zscale=primaries=bt709,tonemap=hable:peak={peak:.8g},"
            "zscale=transfer=bt709:matrix=gbr:range=full:dither=error_diffusion,"
            "format=gbrp,format=rgb24"
        )
        matrix, primaries, transfer, level = "bt709", "bt709", "bt709", "tv"
    else:
        inputs = []
        if matrix in _MATRICES:
            inputs.append(f"in_color_matrix={matrix}")
        if level in _RANGES:
            inputs.append(f"in_range={_RANGES[level]}")
        extract = (
            "scale=" + ":".join([*inputs, "out_range=full"]) + ",format=rgb24" if inputs else ""
        )
    outputs = []
    tags = []
    if matrix in _MATRICES:
        outputs.append(f"out_color_matrix={matrix}")
        tags.extend(["-colorspace", matrix])
    if level in _RANGES:
        outputs.append(f"out_range={_RANGES[level]}")
        tags.extend(["-color_range", level])
    if primaries in _PRIMARIES:
        tags.extend(["-color_primaries", primaries])
    if transfer in _TRANSFERS:
        tags.extend(["-color_trc", transfer])
    encode = (
        "format=rgb24,scale=in_range=full:" + ":".join(outputs) + ",format=yuv420p"
        if outputs
        else ""
    )
    # Carry the same labels on the filtered frames. Some encoders override
    # codec-context colour settings with values negotiated by the filter graph.
    labels = []
    if "-color_trc" in tags:
        labels.append(f"color_trc={transfer}")
    if "-color_primaries" in tags:
        labels.append(f"color_primaries={primaries}")
    if labels:
        encode = ",".join(filter(None, [encode, "setparams=" + ":".join(labels)]))
    return ColorConversion(extract, encode, tuple(tags))


def probe_args(ffmpeg, input_path, plan):
    return [
        str(ffmpeg),
        "-v",
        "error",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-vf",
        plan.extract,
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]
