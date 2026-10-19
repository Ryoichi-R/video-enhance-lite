# Changelog

## Unreleased

- Declare Python 3.11 through 3.13 as the supported range and reject unvalidated Python 3.14+
  before creating the user venv.
- Install Windows x64 dependencies from a fully pinned, SHA-256-checked wheel lock and bind the
  checkout without build isolation or dependency re-resolution.
- When the GUI is closed during analysis or processing, ask for confirmation, request cancellation,
  and wait for the worker and external-process cleanup before destroying the window.
- Normalize suspected variable-frame-rate inputs to an explicit constant frame rate before frame
  extraction; inputs whose timing cannot be classified now fail closed before processing.
- Preserve all MP4-compatible audio and subtitle streams together with global and stream metadata;
  unsupported stream codecs are rejected during preflight instead of being silently dropped.
- Continue short-video analysis when only the final analyzer sample is beyond the last decodable
  frame, with a warning. Missing intermediate samples and FFmpeg failures remain fatal.

## 0.2.0 — release-remediation baseline

- Separated user configuration, temp data, output, and user venv from the source checkout.
- Removed implicit discovery of external binaries beside the package/source tree.
- Added capability-level diagnostics for FFmpeg, Real-ESRGAN, RIFE, CodeFormer, and stabilize.
- Added source-only distribution guidance, migration documents, third-party notices, and release
  contract tests.
- Standardized public stage wording as 11 built-in stages plus optional CodeFormer face restore.

## Future releases

Record user-visible changes, compatibility impact, dependency changes, and migration notes here.
Do not describe a candidate as publicly releasable until the license, contact, candidate manifest,
clean install, and required quality gates are complete.
