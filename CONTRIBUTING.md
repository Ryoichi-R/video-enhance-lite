# Contributing

## Development setup

```text
python -m pip install -e ".[dev,enhance]"
python -m pytest tests/enhance -q --ignore=tests/enhance/test_gui.py
python -m pytest tests/release -q
python -m ruff check src tests
python -m build
```

`install-enhance.bat` installs the Windows x64 environment from
`requirements-lock-win-x64.txt` with exact versions, SHA-256 hashes, and wheel-only
resolution. To validate without network access, set `VIDEO_ENHANCE_WHEELHOUSE` to a directory
containing the complete wheel set for the selected CPython version before running the installer.

Update the lock as one candidate: download wheels from official indexes, review package metadata
and licenses, replace all affected hashes together, and run fresh installer, release, headless,
GUI-marker, Ruff, and build checks on CPython 3.11, 3.12, and 3.13. Do not update a single pin or
hash without rerunning that matrix.

GUI tests use the `gui` marker and require a working tkinter display. The `e2e` marker is
reserved for real external-binary contracts and must not be replaced with a second optional
engine smoke marker.

## Repository boundary

Do not add `.venv`, `.local` real settings, caches, `_internal` material, executables, DLLs,
models, sample media, or user input/output files. The `.gitignore` file is only an aid for a
future external repository; it is not the public-candidate policy. Candidate generation uses an
explicit default-deny allowlist.

Do not add external binary auto-downloaders. New dependencies require a reason, an official
source, license review, and an update/lock procedure.

## Naming and stage contract

Keep `video-enhance-lite` as the repository, distribution, and CLI name, and
`video_enhance_lite` as the Python package name. Update `stages.STAGE_REGISTRY` and the release
tests together when changing stage order or public counts.

## Privacy

Never commit credentials, private URLs, machine-specific absolute paths, or unredacted logs.

## License

By submitting a contribution, you agree that it is provided under this repository's
[MIT License](LICENSE). This does not relicense third-party components, which remain subject to
their respective terms documented in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
