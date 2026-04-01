# Upscaler App Implementation Plan

## Objective

Build a Windows-first desktop application for comparing video upscaler models with the primary goal of maximizing output quality and the secondary goal of measuring performance. The app should ingest source video, upscale it up to 4K UHD, optionally preserve aspect ratio or crop to a fixed 4K frame, and produce analytical comparisons of competing models at the pixel level and over time.

This plan is based on the current baseline in [requirements.md](requirements.md) and the existing build/runbook direction in [build_upscalers.md](../build_upscalers.md).

## Key Requirements Interpreted

- Windows is the primary platform.
- The app is desktop-first, mouse-first, and workspace-first.
- Long videos must remain full-length, with original audio re-synced into the exported output.
- Settings changes should be reflected immediately in the preview window where feasible.
- Build and test flow must be hands-off from the repository.
- Automated testing is required, including GUI verification.
- Model artifacts may come from GitHub or Hugging Face and should be locally cached.
- GPU usage can target up to roughly 12 GB of VRAM.

## Product Recommendation

The product should be built as a quality-analysis workstation, not just an upscaling wrapper. That means the first release should optimize for:

1. Deterministic reproducibility.
2. Side-by-side model comparison.
3. Pixel-level and temporal analytics.
4. Full-job export with original audio retained.
5. Reliable execution on a single Windows GPU with tiling and resumable jobs.

## Recommended Architecture

### Desktop Stack

- Shell: Tauri 2
- Frontend: React + TypeScript + Vite
- Native orchestration: Rust
- Model execution layer: Python workers with PyTorch and CUDA
- Media pipeline: FFmpeg and FFprobe
- Persistence: SQLite plus project JSON snapshots

### Why This Stack

- Tauri keeps the desktop shell lightweight and performant on Windows.
- Rust is well suited for job orchestration, file I/O, process control, caching, and crash-safe resumability.
- Python remains the practical integration layer for state-of-the-art super-resolution models.
- FFmpeg is the right deterministic backbone for decode, frame extraction, encode, and audio remux.

### High-Level Components

1. Workspace manager
2. Source video inspector
3. Decode and frame cache pipeline
4. Model adapter system
5. Comparison and metrics engine
6. Preview compositor
7. Export pipeline with audio remux
8. Synthetic test generator
9. Automation and verification harness

## Model Investigation

The product should support both frame-based image SR models and temporal video SR models. They serve different quality goals and should not be treated as interchangeable.

### Frame-Based Models

These usually produce the strongest single-frame sharpness and are easier to package on Windows.

#### 1. Real-ESRGAN x4 Plus

- Role: default baseline for real-world content
- Strengths: mature ecosystem, practical inference, Windows-friendly options, tiling support, broad community use
- Weaknesses: temporal inconsistency when run frame-by-frame on video
- Licensing: BSD-3-Clause
- Recommendation: include in MVP

#### 2. SwinIR Real-World x4

- Role: high-fidelity reference model
- Strengths: strong restoration quality, good detail reconstruction, widely cited benchmark model family
- Weaknesses: slower and heavier than Real-ESRGAN for desktop inference
- Licensing: Apache-2.0
- Recommendation: include in MVP if VRAM tests are acceptable

#### 3. HAT Real_HAT_GAN_SRx4 and Real_HAT_GAN_SRx4_sharper

- Role: perceptual-quality leader for detail-focused comparisons
- Strengths: stronger perceptual sharpness than SwinIR in many SRx4 comparisons, explicit fidelity versus sharper variants
- Weaknesses: more artifact-prone on some content, heavier model family
- Licensing: Apache-2.0
- Recommendation: include at least one HAT variant in MVP, ideally both fidelity and sharper variants for comparison

### Temporal Video Models

These are the best fit for long-form video quality because they reduce flicker and exploit adjacent frames.

#### 4. BasicVSR++

- Role: main temporal-consistency benchmark
- Strengths: strong video SR results, better propagation and alignment than BasicVSR with similar computational intent
- Weaknesses: harder packaging and runtime orchestration than image models
- Licensing: use the OpenMMLab implementation and weights only after validating redistribution terms for the selected checkpoints
- Recommendation: include in phase 2 after the frame-based pipeline is stable

#### 5. RealBasicVSR

- Role: real-world degraded video benchmark
- Strengths: designed for realistic degradations, balances detail synthesis with artifact suppression through pre-cleaning and propagation
- Weaknesses: more complex inference flow and model dependencies
- Licensing: validate selected checkpoint terms before bundling
- Recommendation: include after BasicVSR++ if target content is noisy or compressed live-action video

### Research-Only Candidate

#### RVRT

- Role: research comparison target for high-end temporal restoration
- Strengths: strong balance of effectiveness, memory, and runtime for video restoration benchmarks
- Weaknesses: more complex deployment, heavier runtime, and repository licensing is not a good default for commercial redistribution
- Licensing: CC-BY-NC in the upstream repository, so do not plan to ship it in a commercial-capable MVP
- Recommendation: keep as an optional non-commercial research adapter only

## Recommended MVP Model Set

The first release should compare these four tracks:

1. Real-ESRGAN x4 Plus as the practical real-world baseline.
2. SwinIR Real-World x4 as the fidelity-oriented frame model.
3. HAT Real_HAT_GAN_SRx4 or Real_HAT_GAN_SRx4_sharper as the perceptual-detail model.
4. BasicVSR++ as the temporal video model.

This gives the app a meaningful quality spectrum instead of multiple near-identical models.

## Output Modes

The app should support explicit output framing modes rather than silently resizing.

### Mode A: Preserve Aspect Ratio Within 4K Canvas

- Target canvas: 3840 x 2160
- Output is letterboxed or pillarboxed as needed
- Best for analytical fairness and source-faithful exports

### Mode B: Crop To Fill 4K Canvas

- Target canvas: 3840 x 2160
- Scale to cover, then crop according to user-selected anchor
- Best for cases where exact UHD frame occupancy is required

### Mode C: Native x2, x3, x4 Output

- Preserve model-native scale result even if not exactly 4K
- Best for raw model evaluation before the final presentation resize

For analytic comparisons, native model output should be preserved internally even when the export view is padded or cropped to 4K.

## Comparison Engine Design

The app should compare models along both spatial quality and temporal stability.

### Spatial Metrics

- PSNR
- SSIM
- MS-SSIM
- LPIPS
- DISTS
- Edge preservation score
- Chroma error score

### Temporal Metrics

- Inter-frame flicker score
- Temporal warping error using optical-flow-aligned frame comparison
- Edge stability over time
- Temporal LPIPS or a similar perceptual frame-delta metric

### Visual Analytics

- Side-by-side synchronized preview
- Pixel inspector with zoomed crop
- Heatmap of absolute error
- Edge-map difference overlay
- Luma and chroma histogram deltas
- Temporal strip view across successive frames

## Synthetic Test Strategy

The test suite should generate known-good 4K masters, degrade them into lower-resolution inputs, then upscale them back and compare to the original masters.

### Synthetic Master Content

Generate 4K videos containing:

- Slanted edges
- Siemens stars
- Checkerboards and zone plates
- Fine text at multiple font sizes
- Chroma wedges and color bars
- High-frequency fabric and foliage textures
- Gradients and near-flat surfaces
- Moving diagonal lines
- Panning scenes with subpixel motion
- Repeating patterns that can reveal ringing and moire

### Source Variants

Generate known low-resolution inputs at:

- 426 x 240
- 640 x 360
- 854 x 480
- 1280 x 720
- 1920 x 1080

Apply controlled degradations:

- Bicubic downscale
- Lanczos downscale
- Blur-downsample
- Compression artifacts
- Sensor-like noise
- Mild ringing
- Motion blur

### Why This Matters

This approach gives exact reference frames for objective comparison and also lets the test suite catch failure modes like false detail, zippering, temporal shimmer, color bleeding, oversmoothing, and haloing.

## App Workflow

### Main Workspace

The main workspace should include:

- Source panel
- Model selection panel
- Output framing panel
- Metrics panel
- Timeline and frame browser
- Comparison viewport
- Export queue
- Diagnostics/log panel

### Typical User Flow

1. Import source video.
2. Inspect source metadata and choose evaluation presets.
3. Select one or more models.
4. Select output framing mode.
5. Run preview on a segment or full clip.
6. Review synchronized side-by-side output and metrics.
7. Export selected model outputs with original audio reattached.
8. Save project state for later comparison.

## Execution Pipeline

1. Probe source with FFprobe.
2. Decode frames to a managed cache.
3. Run selected model adapters against the same normalized frame stream.
4. Store per-frame outputs and job metadata.
5. Compute metrics against reference data where available.
6. Render preview and overlays from cached outputs.
7. Encode final video.
8. Remux original audio with frame-accurate duration checks.

## Caching And Reproducibility

Cache keys should include:

- Source file fingerprint
- Decode settings
- Model name and version
- Model weights hash
- Tiling settings
- Precision mode
- Output framing mode
- Final resize kernel

This is required to make repeated experiments trustworthy.

## Precision And VRAM Strategy

Because the target machine budget is about 12 GB of VRAM, the inference layer should support:

- FP16 by default where model-safe
- Configurable tiling
- Overlap padding to reduce seam artifacts
- Sequential clip execution for temporal models
- CPU fallback for metadata and analytics only, not as the default upscale path

The UI should expose memory-safe presets such as:

- Quality Max
- Quality Balanced
- VRAM Safe

## Test Plan

### Unit Tests

- Project persistence
- Cache key generation
- frame-to-time mapping
- output framing math
- crop and pad calculations
- metrics math on known fixtures
- audio remux validation helpers

### Integration Tests

- Import source video
- Decode short fixture clip
- Run at least one frame model end to end
- Run at least one temporal model end to end
- Compute and persist metrics
- Export final video with original audio
- Reopen saved project and restore state

### Automated GUI Tests

- Import fixture video
- Change framing mode and verify preview state changes
- Run comparison job
- Inspect metrics panel for populated values
- Export and verify output artifact exists
- Reload project and verify session restore

### Golden Tests

- Synthetic clip baseline outputs for deterministic regression detection
- Metric threshold assertions per model and preset
- Temporal flicker thresholds for the video model path

## Recommended Project Phases

### Phase 0: Specification Cleanup

- Normalize the requirements document to the actual product name and scope.
- Record supported licenses for each planned model.
- Define the first three benchmark clips and synthetic fixtures.

### Phase 1: Desktop Shell And Job System

- Scaffold Tauri app and workspace layout.
- Add project persistence.
- Add FFprobe and FFmpeg integration.
- Add diagnostics logging.

### Phase 2: Single-Model Vertical Slice

- Integrate Real-ESRGAN.
- Implement preview, export, audio remux, and cache.
- Implement native x4 and 4K canvas framing modes.
- Add automated tests for the first end-to-end path.

### Phase 3: Comparison Engine

- Add side-by-side synchronized viewport.
- Add pixel inspector and heatmaps.
- Add PSNR, SSIM, LPIPS, and edge metrics.
- Add saved comparison sessions.

### Phase 4: Higher-End Models

- Add SwinIR and HAT adapters.
- Establish per-model presets and VRAM guidance.
- Validate reproducibility across repeated runs.

### Phase 5: Temporal Video SR

- Add BasicVSR++ adapter.
- Add temporal metrics and clip-window execution.
- Compare frame-wise models versus temporal model on synthetic motion clips.

### Phase 6: Synthetic Benchmark Lab

- Build the synthetic video generator.
- Add golden baselines and metric threshold gates.
- Publish benchmark reports from CI.

## Risks And Mitigations

### Risk: Model packaging on Windows is fragile

Mitigation: isolate each model family behind a Python adapter contract and pin environments per adapter.

### Risk: Frame-based models look sharper but flicker badly on motion

Mitigation: report temporal metrics separately and present temporal previews, not just still frames.

### Risk: 4K exports exceed VRAM on large models

Mitigation: require tiled inference with overlap and expose memory-safe presets.

### Risk: Upstream licenses differ by model and checkpoint

Mitigation: maintain a model registry that records source URL, weights hash, license, and redistribution status.

### Risk: Synthetic metrics reward oversmoothing or penalize perceptual detail unfairly

Mitigation: use both distortion metrics and perceptual metrics, then inspect failures visually.

## Final Recommendation

Build the app as a Tauri desktop workstation with Rust orchestration, Python model adapters, and FFmpeg-based media handling. Start with Real-ESRGAN as the first vertical slice, then add SwinIR and HAT for image-quality comparison, and BasicVSR++ for temporal video quality comparison.

If the goal is best quality first and performance second, the most important architectural decision is to treat temporal video SR as a first-class comparison lane rather than assuming the best image model will also be the best video model.