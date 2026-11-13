# Changelog

## Unreleased

## 0.2.3 — 2026-11-13

- Size frame-rate observation windows by the nominal frame rate and use the full video when
  it is shorter. Read enough lookahead to flush reordered frames. Low-fps and short CFR clips
  are no longer rejected merely because a fixed observation window contains too few frames;
  genuinely insufficient or invalid timing still fails closed. Bound the requested sample count
  by the nominal number of video frames so low-fps clips do not request several samples beyond EOF.
- Extract all MPEG-PS/TS analysis samples in one decoding pass, retaining video-relative sample
  positions and duplicate selections where needed. Monitor advancing decoded-frame counts
  instead of applying a fixed total-runtime limit. Stalled decodes and cancellation still stop
  and reap the child process; repeated progress counters do not hide a stall.
- Add real-media low-fps, short-clip, sample-position, and throttled-decoding regressions,
  plus process tests for advancing progress, stalls, cancellation, and failed extraction.

## 0.2.2 — 2026-11-13

- Publish completed output with an atomic no-overwrite rename when hard links are unavailable.
  Concurrently created files are preserved; unsupported filesystems keep the completed partial
  output and report its location instead of risking replacement.
- Decode MPEG-PS/TS analyzer samples from the video start to avoid failed seeks near the final
  GOP, including video streams that start after audio and inputs with misleading extensions.
- Probe frame-rate windows using the video stream's absolute timestamps. If seeking yields too
  few frames, retry from earlier points through the requested window before classifying it.
  Unknown timing still fails closed and variable cadence still requires normalization.
- Add publication-race tests and real-FFmpeg analysis-to-output regressions for delayed MPEG-PS
  video and long-GOP transport streams.
- MPEG analysis may take longer because samples are decoded from the video start; existing
  subprocess timeouts and cancellation remain active. Include the final MPEG-PS frame when its
  presentation timestamp can be reconstructed from decoding timestamps.

## 0.2.1 — 2026-11-13

- Correct OpenCV stabilization to apply the trajectory correction without adding camera motion
  again; add a regression that measures the remaining displacement of a fixed scene.
- Create and check the output directory before video processing, including the first-use default
  destination. Resolve relative input paths before subprocesses change their working directory.
- Derive bounded exposure, contrast, and saturation corrections from analysis instead of enabling
  an identity filter. Keep normally exposed monochrome material neutral.
- Sample and process the video stream's own duration, using video packet timestamps when the
  container omits it. Preserve longer audio/subtitle tails in the final mux.
- Use autorotated dimensions for resolution summaries, upscale decisions, and resolution clamping.
- Add real-FFmpeg regressions for CLI output, relative inputs, unequal stream lengths, exposure,
  and rotated video; check package/runtime version agreement.
- Keep the source offset between video and audio/subtitle start times. Inputs whose video starts
  after the audio were previously muxed out of sync and lost their final video frames.
  MPEG-TS/PS follow FFmpeg's per-stream timestamp rebasing.
- Write the final mux to a `.partial-*` file beside the output and move it into place only on
  success. Failed or cancelled runs leave no file under the output name, and a file that appears
  under that name during processing is never overwritten (the result gets a `~(n)` suffix).
  If only the final move fails, the finished file is kept and its path is reported.
- `--auto` no longer defers stages that `--disable`, `--no-realesrgan`, or `--no-rife` turn off.
- Stop each analysis ffprobe/ffmpeg call after 120 seconds (the full-file video packet scan allows
  for 10 MiB/s of input) and terminate it promptly on GUI cancellation or Ctrl+C. Damaged inputs or
  unresponsive drives now fail with a message instead of hanging. Executable checks stop after
  30 seconds. The packet scan uses one short line per packet instead of a JSON document.
- OpenCV stabilization rotates about the image centre, so corrections that include rotation no
  longer add a translation error; frame write failures now stop the stage.

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
