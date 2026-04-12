# VideoUpgrader Requirements

## Status

This document is the current baseline specification for VideoUpgrader.

It replaces the inconsistent placeholder naming and prior carry-over text. It is written to match the current objective: a Windows-first desktop workstation for evaluating both video upscaling quality and frame-rate interpolation quality, with performance as a secondary concern.

## Product Summary

VideoUpgrader is a Windows-first desktop application for:

- Comparing multiple video upscaler models on the same source content.
- Interpolating existing video to higher frame rates such as 30 fps and 60 fps.
- Running interpolation after upscaling when the user wants final-resolution motion synthesis.
- Exporting full-length upscaled video while preserving or reattaching the original audio.
- Analyzing output quality at the frame, pixel, and temporal level.
- Running repeatable synthetic benchmark scenarios for objective regression testing.

## Primary Users

- Developers evaluating super-resolution model quality.
- Power users comparing model behavior on long-form video.
- Researchers validating spatial and temporal restoration quality.

## Core Product Goals

1. Maximize output quality first.
2. Measure performance second.
3. Preserve full video duration during upscale and export.
4. Keep the original audio synchronized in exported outputs.
5. Support reproducible model comparisons on both real and synthetic content.
6. Preserve exact audio-video sync through interpolation-only and post-upscale interpolation pipelines.

## MVP Scope

The first deliverable must support:

- Importing a local video file.
- Inspecting source metadata.
- Selecting the Real-ESRGAN x4 Plus model.
- Configuring interpolation-only or after-upscale frame-rate upgrades.
- Choosing one of the supported 4K framing modes.
- Preparing and running a first-end-to-end upscale job.
- Exporting an output video with original audio remuxed back in.
- Saving and restoring project state.
- Generating synthetic benchmark fixtures for automated comparison.

## Functional Requirements

### Source Handling

- The app must load local video files through native desktop file browsing.
- The app must inspect and display source metadata including resolution, frame rate, duration, audio presence, and aspect ratio.
- The app must support long-form video rather than only short clips.
- The app must support MP4 and WEBM and MKV input.
- The app should support MOV input

### Upscaling

- The app must support upscaling to 4K UHD output targets.
- The app must support native model-scale output where needed for analysis.
- The app must allow different models to be run against the same decoded source sequence.
- The app must cache intermediate work so repeated comparisons do not re-run unnecessary steps.

### Interpolation

- The app must support interpolation-only processing for existing videos.
- The app must support interpolation after upscaling in the main export pipeline.
- The app must support explicit interpolation targets of 30 fps and 60 fps.
- The app must warn per job when the selected interpolation target fps is not higher than the source fps.
- The app must preserve exact playback duration unless the user explicitly changes timing semantics in a future feature.

### Output Framing

- The app must support preserving aspect ratio within a 3840 x 2160 canvas.
- The app must support crop-to-fill behavior for a fixed 3840 x 2160 output.
- The app must preserve the internal native upscale result for analysis, even when the user exports a padded or cropped 4K frame.
- The app must support h.264 encoding
- The app should support h.265 encoding
- The compression ratio, and other parameters which affect quality must be adjustable.
- The parameters used to encode the output video that are not inherently stored there must be stored in the video as metadata.

### Comparison And Analysis

- The app must support side-by-side comparison of model outputs.
- The app must support blind comparison of multiple generated samples from the same captured source interval.
- The app must support a detached comparison workspace that keeps the source reference and generated samples time-aligned.
- The app must support zoomed pixel inspection.
- The app must support visual overlays such as absolute-difference heatmaps.
- The app must calculate spatial metrics where reference data exists.
- The app must calculate temporal stability metrics for video comparisons.

### Export

- The app must export the entire processed video rather than only cut segments.
- The exported result must preserve exact playback length unless the user explicitly changes timing.
- The app must reattach the original audio to the exported video when audio is present.
- The app must validate that output duration and audio duration remain in sync.
- The app must validate that interpolation pipelines keep audio sync within the configured tolerance across source, intermediate, and exported media.

### Project Persistence

- The app must persist workspace state locally.
- The app must restore recent files, selected settings, comparison state, and known outputs across sessions.
- The app must persist enough job metadata that historical runs can restore their recorded request as an editable template in the main workspace.
- The app must persist enough job metadata that cancelled, interrupted, or stale paused runs can be restarted from saved settings.
- The app must persist effective run metadata for completed and historical outputs so the user can inspect the resolved settings that produced an output file, including the selected quality preset and effective execution settings when available.

### Job Control

- The app must support pausing an active upscale pipeline job from the main status controls and the Jobs workspace.
- The app must support pausing an active source conversion job from the main status controls and the Jobs workspace.
- The app must support resuming a paused in-memory job without creating a new job ID while the current app session remains alive.
- The app must support stopping a running or paused job without deleting its recorded settings or managed artifacts metadata.
- The app must expose `queued`, `running`, `paused`, `succeeded`, `cancelled`, `failed`, and `interrupted` job-state semantics consistently across the live status card and the Jobs workspace.
- The app must expose a `Load Template` action for replayable historical runs that restores the recorded request into the editable form without automatically starting a new job.
- The app must expose a `Restart` action for stopped pipeline jobs when the run can be started again from the beginning using saved settings.
- The app must expose a `Resume` action only when the current in-memory job state is still resumable within the active desktop session.
- If the desktop app exits while a job is paused, the recovered historical record must be restartable from saved settings rather than treated as a live-resumable job.

### Look And Feel

- The app should be mouse-first rather than touch-first.
- The app should feel workspace-first rather than demo-first.
- Settings changes that affect framing or preview composition should be reflected immediately in the preview window where feasible.

### Demo And Verification Support

- The app must include a scripted demo mode for repeatable UI and synchronization verification.

## Desktop Requirements

- Windows is the primary supported platform.
- The desktop shell must run through a high-performance framework.
- The desktop workflow must support native file browsing and native file system access.

## Browser Requirements

- Browser support is optional.
- Browser execution may be used only if required by testing tools.

## Technical Requirements

### Frontend

- The frontend must support fluid interaction while background processing is active.
- The UI must support comparison-centric workflows rather than a single-output wizard flow.

### Desktop Backend

- The backend must coordinate media probing, decode, job scheduling, caching, export, and diagnostics.
- The backend may use GPU resources up to approximately 12 GB of VRAM.
- High-performance backends added to the repository must prefer CUDA-accelerated PyTorch or equivalent discrete-GPU execution on the detected NVIDIA workstation GPU when that path is available.
- High-performance backends added to the repository must propagate the selected GPU into their runtime layer and must surface an explicit fallback or setup error instead of silently dropping to a slower path.
- The backend must support memory-safe execution options such as tiling.
- The backend must expose progress telemetry for extracting, upscaling, interpolating, encoding, and remuxing stages.
- The backend must expose average fps, rolling fps, elapsed time, ETA, RAM usage, GPU memory usage, scratch growth, and output growth while jobs are running.
- The backend must implement pause and resume as first-class job-control commands for both pipeline and source-conversion jobs.
- The backend must suspend long-running subprocess work during pause so ffmpeg and other external stages do not continue processing in the background.
- The backend must preserve current progress counters when a job is paused and must surface a paused progress message without resetting stage progress.

### Model Integration

- Model artifacts must be sourced from supported upstream distributions such as GitHub releases or Hugging Face.
- Model artifacts must support local caching.
- Each integrated model must record source, version, hash, and license metadata.

## Test Requirements

- Testing must take place without human intervention.
- Unit tests must target at least 80 percent coverage in the maintained codebase.
- Integration tests must cover end-to-end flows including video loading, job preparation, interpolation planning, export, audio remux, and project restore.
- Automated GUI tests must verify the real desktop workflow for both upscaling and interpolation controls.
- Automated GUI tests must verify pause, resume, and stop controls for active jobs.
- Automated GUI tests must verify the distinction between `Resume`, `Restart`, and `Load Template` in the Jobs workspace.
- Automated GUI tests must verify detached blind-comparison synchronization between the full-length source reference and generated sample previews.
- Integration tests must include synthetic AV-sync fixtures that validate source-to-output sync across interpolation-only and post-upscale interpolation flows.
- Unit tests must cover interpolation request validation, target-fps mapping, frame-count planning, and progress telemetry helpers.
- Unit and integration tests must cover paused-state telemetry and the transition from running to paused to resumed to terminal job states.
- Integration and GUI tests must assert that interpolation progress, fps throughput, RAM usage, and GPU usage are surfaced correctly to the user.
- Synthetic benchmark tests must generate known reference content, degrade it deterministically, upscale it, and compare outputs against the original high-resolution master.

## Development And Build Requirements

- The repository must support a one-script bootstrap flow.
- The repository must support a one-script build flow.
- The repository must support optional test execution during build.
- The build must provide actionable feedback when local prerequisites or resources are insufficient.
- Dependent tools and model assets must be fetched from supported sources or documented user-provided locations.

## Acceptance Criteria For Initial Vertical Slice

The first implementation slice is acceptable when:

1. A Windows user can bootstrap the repository from documented scripts.
2. The desktop shell launches successfully.
3. A local video can be selected and inspected.
4. A Real-ESRGAN job configuration can be created with native x4 or 4K framing settings.
5. Synthetic benchmark fixtures can be generated from the repository.
6. Automated tests pass for the scaffolded math, benchmark generation, and job-configuration logic.

