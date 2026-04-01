# Upscaler Requirements

## Status

This document is the current baseline specification for Upscaler.

It replaces the inconsistent placeholder naming and prior carry-over text. It is written to match the current objective: a Windows-first desktop workstation for evaluating and comparing video upscaler quality, with performance as a secondary concern.

## Product Summary

Upscaler is a Windows-first desktop application for:

- Comparing multiple video upscaler models on the same source content.
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

## MVP Scope

The first deliverable must support:

- Importing a local video file.
- Inspecting source metadata.
- Selecting the Real-ESRGAN x4 Plus model.
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
- The app must support zoomed pixel inspection.
- The app must support visual overlays such as absolute-difference heatmaps.
- The app must calculate spatial metrics where reference data exists.
- The app must calculate temporal stability metrics for video comparisons.

### Export

- The app must export the entire processed video rather than only cut segments.
- The exported result must preserve exact playback length unless the user explicitly changes timing.
- The app must reattach the original audio to the exported video when audio is present.
- The app must validate that output duration and audio duration remain in sync.

### Project Persistence

- The app must persist workspace state locally.
- The app must restore recent files, selected settings, comparison state, and known outputs across sessions.

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
- The backend must support memory-safe execution options such as tiling.

### Model Integration

- Model artifacts must be sourced from supported upstream distributions such as GitHub releases or Hugging Face.
- Model artifacts must support local caching.
- Each integrated model must record source, version, hash, and license metadata.

## Test Requirements

- Testing must take place without human intervention.
- Unit tests must target at least 80 percent coverage in the maintained codebase.
- Integration tests must cover end-to-end flows including video loading, job preparation, export, and project restore.
- Automated GUI tests must verify the real desktop workflow.
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

