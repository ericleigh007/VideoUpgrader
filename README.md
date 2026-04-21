# VideoUpgrader

![VideoUpgrader icon](docs/images/video-upgrader-icon-512.png)

VideoUpgrader is a Windows-first desktop app for evaluating video upscalers the way enthusiasts, researchers, and developers actually use them: by comparing outputs side by side, zooming into problem areas, checking framing behavior, boosting frame rate when needed, and then exporting a full processed video when a model and setting combination earns it.

The project combines a Tauri desktop shell, a React comparison-first UI, and a Python worker that handles probing, synthetic benchmark generation, model execution, interpolation, encode, remux, and diagnostic benchmarking.

## What's New

Updated April 12, 2026.

- The detached comparison window now keeps the Source reference and all blind-comparison players frame-synced in real desktop validation, so you can inspect the same logical frame instead of guessing whether offsets are coming from the tool.
- The comparison Source pane now uses a valid full-length playable reference instead of falling back to a short browser clip, which removes the black-source and wrong-timestamp failures that made blind comparisons untrustworthy.
- Comparison controls are now clearer during review: `Shift + wheel` resizes the comparison panes, `Ctrl + wheel` zooms the video content in sync across players, and drag pans the crop only after a real drag starts.
- The comparison workflow was validated with deterministic AV-sync fixtures and the desktop smoke harness, which matters because it proves the separate comparison window is doing the same job in the native app that it appears to do in the browser tests.

## What VideoUpgrader Does

- Load a local video and inspect source metadata.
- Run multiple upscale jobs against the same source material.
- Run automatic and reference-guided colorization experiments on grayscale footage.
- Run interpolation-only jobs on an existing source video.
- Run a combined pipeline that upscales first and then interpolates to a target frame rate.
- Compare outputs with a zoomed inspection workflow.
- Run blind sample comparisons before committing to a final export.
- Generate reproducible synthetic benchmark fixtures.
- Benchmark backend and runtime combinations directly from the worker.
- Preserve full-length video output and reattach original audio during export workflows.

The current UI already includes a comparison inspector with zoom, focus presets, blind-sample selection, and external opening for full-size playback.

## Current Focus

VideoUpgrader is currently centered on:

- Real-ESRGAN-family and PyTorch image super-resolution model evaluation.
- RIFE-based frame interpolation to 30 fps or 60 fps.
- 4K framing workflows.
- Comparison-centric desktop usage instead of a one-click demo flow.
- Repeatable benchmarking so quality and speed decisions are evidence-based.

The current product direction is quality first, performance second.

Colorization is now part of the workstation, but it is still experimental. The current goal is to make it practical to compare approaches, preserve what worked, and iteratively improve difficult grayscale footage rather than claim one-click historically accurate restoration.

## Current State

The current app is a real desktop pipeline, not just a frontend shell around isolated model demos.

- The Tauri desktop host owns file access, job orchestration, persistence, and native integration.
- The React UI owns the comparison-first workstation workflow, run configuration, jobs view, and context-library management.
- The Python worker owns media probing, frame extraction, optional colorization, optional upscaling, optional interpolation, encode, concat, audio remux, and benchmark tooling.
- The model catalog is task-aware and currently distinguishes `upscale` models from `colorize` models, then routes each model request to the correct backend/runtime path.

Today the app can run automatic grayscale-to-color workflows, reference-guided colorization experiments, frame-based image super-resolution, research video-SR through an external runner contract, and RIFE-based frame interpolation in one managed pipeline.

## Current Processing Modes

The app currently supports these processing combinations:

- Colorize only.
- Upscale only.
- Colorize before upscale.
- Interpolate only.
- Upscale first, then interpolate.
- Colorize before upscale, then interpolate.

Interpolation targets currently support 30 fps and 60 fps outputs through the Windows RIFE NCNN runtime.

The exact stages that run depend on the requested mode, but the worker always treats the job as one managed export pipeline with progress telemetry and final output assembly.

## Pipeline Stages

The current worker pipeline is built from these parts:

1. Probe the source and resolve output dimensions, framing rules, codec/container, and effective runtime settings.
2. Decode or extract the source into working frames, usually in short internal segments for long-form jobs.
3. Optionally colorize the extracted frames.
4. Optionally upscale the colorized or original frames.
5. Optionally interpolate frame rate after the upstream stage completes.
6. Encode segment outputs to video.
7. Concatenate segment outputs when the job ran in multiple parts.
8. Remux the original audio back onto the final video when the source contains audio.
9. Persist job metadata, stage timings, logs, and effective settings so the run can be inspected or replayed later.

In current worker telemetry, the major exported phases are `extracting`, `colorizing`, `upscaling`, `interpolating`, `encoding`, and `remuxing`.

## How Models Are Handled

Model execution is driven by the catalog in [config/model_catalog.json](config/model_catalog.json). Each model entry declares:

- its task, such as `upscale` or `colorize`
- the backend family it belongs to
- whether it is runnable now or only planned
- whether it is comparison-eligible in the current UI
- whether it accepts source-linked context input such as reference images
- any runtime asset metadata like checkpoint download source

The current backend families are:

- `realesrgan-ncnn`: portable NCNN Vulkan execution for bundled Real-ESRGAN-family models.
- `pytorch-image-sr`: PyTorch frame-by-frame image super-resolution.
- `pytorch-video-sr`: PyTorch research video-SR through an external runner contract.
- `pytorch-video-interpolation`: PyTorch or native interpolation support in the worker ecosystem.
- `pytorch-image-colorization`: PyTorch automatic and reference-guided colorization.

In practical terms, model handling works like this:

- NCNN models run through a portable executable runtime and are the most self-contained path.
- PyTorch image-SR models run frame batches through Python/Torch and support GPU selection, precision control, and tiling.
- Research video-SR models like RVRT use an external command contract rather than an in-repo native implementation.
- Automatic colorizers like DDColor and DeOldify run through the PyTorch image-colorization path without extra context input.
- Reference-guided colorizers like DeepRemaster and ColorMNet run through the same colorization pipeline but also consume source-linked reference images from the desktop context library.

The worker decides which runtime path to use from the selected model, not from ad hoc per-screen logic. That is what allows the same job pipeline to support colorize-only, before-upscale colorization, and multiple upscale backends without inventing a separate export path for each model.

## Source Context Libraries

Reference-guided colorization uses source-linked context libraries stored under `artifacts/context-libraries`.

- Each source clip gets its own managed reference library.
- Imported context images are deduplicated so the same content is not added repeatedly.
- Entries can be selected, cleared, or deleted from the desktop UI.
- DeepRemaster can use multiple source-linked reference images.
- ColorMNet is currently integrated as a single-exemplar workflow even though it uses the same library surface for selection.

This matters because the current app is not just passing a loose file picker result into the worker. It is maintaining source-specific reference state that can be reviewed and reused.

## Job Controls

Active upscale and source-conversion jobs now support in-session pause, resume, and stop controls.

- `Pause` must halt active processing without discarding the current live job record.
- `Resume` must continue the same in-memory job when the desktop app is still running.
- `Stop` must cancel the active job and keep its saved settings available for a later restart.
- `Load Template` must restore any replayable historical run into the form so the user can adjust settings before starting a new run.
- `Restart` must reload a stopped pipeline job from its saved settings and immediately queue a fresh run from the beginning.
- Paused jobs are session-local. If the app exits while a job is paused, that run should fall back to the existing historical restart path rather than pretending it can live-resume after restart.
- Live progress and the Jobs workspace must show `paused` as a first-class state.

In practice, the action model is:

- `Resume` = continue the same in-memory job.
- `Restart` = rerun the saved request immediately from the beginning.
- `Load Template` = restore the saved request into the editor without starting it.

For new high-performance backends, the repo policy is CUDA-first on the detected NVIDIA workstation GPU whenever that execution path exists. Backends must either honor the selected GPU explicitly or surface a clear setup/fallback message rather than quietly dropping to a weaker path.

The Python worker pause path depends on `psutil` for suspending active subprocess trees during long ffmpeg or NCNN stages.

## Available Models

### Runnable Now

- `realesrgan-x4plus`: Real-ESRGAN x4 Plus via NCNN Vulkan.
- `realesrnet-x4plus`: Real-ESRNet x4 Plus via PyTorch.
- `bsrgan-x4`: BSRGAN x4 via PyTorch.
- `swinir-realworld-x4`: SwinIR Real-World x4 via PyTorch.
- `rvrt-x4`: RVRT x4 via an external video-SR runner configured through `UPSCALER_RVRT_COMMAND`.
- `ddcolor-modelscope`: DDColor automatic colorization checkpoint via PyTorch.
- `ddcolor-paper`: higher-fidelity DDColor checkpoint via PyTorch.
- `deoldify-stable`: conservative DeOldify colorization checkpoint via PyTorch.
- `deoldify-video`: DeOldify variant tuned for video stability via PyTorch.
- `deepremaster`: reference-guided DeepRemaster video colorization via PyTorch.
- `colormnet`: exemplar-guided ColorMNet video colorization via PyTorch.
- `realesrgan-x4plus-anime`: compatibility model.
- `realesr-animevideov3-x4`: compatibility model.

### Cataloged But Not Yet Runnable In The Current App Build

- `hat-realhat-gan-x4`

### Backend Types

- `realesrgan-ncnn`: portable NCNN Vulkan backend.
- `pytorch-image-sr`: PyTorch frame-by-frame image SR backend.
- `pytorch-video-sr`: research video-SR backend driven by an external command contract.
- `pytorch-image-colorization`: PyTorch automatic and reference-guided colorization backend.

## Colorization Models

VideoUpgrader now includes several colorization backends with different strengths and different levels of user guidance.

- `ddcolor-modelscope`: fully automatic photo-realistic colorization for grayscale photos and live-action frames. This is a fast baseline when you want plausible color without managing references.
- `ddcolor-paper`: a higher-fidelity DDColor checkpoint for local comparisons when you want to test whether the paper-weighted model preserves better detail or tone.
- `deoldify-stable`: a more conservative DeOldify variant that tends to suit portraits and natural live-action material.
- `deoldify-video`: the DeOldify variant intended to be steadier across consecutive frames, making it a better default than the portrait-focused model for moving footage.
- `deepremaster`: a reference-guided video colorizer that can use source-associated context images. It is useful when you have stills, posters, or other look references and want the model to steer toward them.
- `colormnet`: an exemplar-propagation video colorizer built around one anchor reference image that closely matches a shot. It can produce strong matches when the exemplar is very close to the target shot, but it is not a general semantic recoloring system.

In practice, the automatic models are good for fast exploration, while the reference-guided models are the more promising path when you are willing to curate inputs and review shot by shot.

## Upscaling Models

The current upscale side of the app spans a few different model types.

- `realesrgan-x4plus`: the main NCNN Vulkan baseline for photographic footage and the most portable runnable path.
- `realesrnet-x4plus`: a more conservative PyTorch image-SR baseline.
- `bsrgan-x4`: a blind real-world SR model for degraded or compressed inputs.
- `swinir-realworld-x4`: a transformer-based frame SR model for higher-fidelity inspection.
- `rvrt-x4`: a research-tier video-native model routed through an external runner contract.

These are not treated as equivalent backends. Some are framewise image models, some are video-native research models, and some are packaged through NCNN rather than PyTorch. The catalog and worker are structured to keep those differences explicit.

## Experimental Shot-Based Colorization Workflow

The likely path to decent quality on difficult grayscale film is not a single end-to-end pass over the whole movie. The expected workflow is closer to a finishing pipeline:

1. Split the source video into shots or short visually coherent sections.
2. Grab a representative frame, often the first frame, from each shot.
3. Send that frame to an external AI image editor that supports prompt-guided image colorization or image editing.
4. Manually describe important colors and materials so the edited frame becomes a stronger target look.
5. Feed the edited frame back into the colorizer as the exemplar or shot reference.
6. Review the result shot by shot and accept each shot only when it is good enough.
7. Reassemble the approved shots into a full video.
8. Check shot boundaries for color continuity problems and rerender or retouch when transitions feel wrong.
9. Restore the original audio onto the final colorized output.

This is especially relevant for ColorMNet. In the current integration, ColorMNet uses one selected reference image for one run. That makes it a better fit for a shot or a short coherent sequence than for a whole edited reel with changing lighting, framing, and subject emphasis.

## Colorization Caveat

Colorization in VideoUpgrader should currently be treated as experimental.

- These models can generate visually convincing results without producing historically correct color.
- Reference-guided workflows can still drift when adjacent shots differ too much in lighting, exposure, costume visibility, or composition.
- Shot-to-shot blending and continuity review are part of the work, not an implementation detail the current app can guarantee away.
- The app is meant to help evaluate and iterate on colorization strategies, not to present current colorized output as authoritative restoration.

## Desktop Workflow

The intended workflow is:

1. Select a local source video.
2. Inspect source metadata and preview.
3. Choose whether the job is colorization, upscaling, interpolation, or a combined pipeline.
4. Pick the active model or models for the enabled stages.
5. Attach source-linked reference images when a reference-guided colorizer needs them.
6. Run one or more jobs.
7. Compare outputs with zoomed inspection and blind samples.
8. Export the winning result.

The app is designed to support model-vs-model and settings-vs-settings evaluation on the same source material, not just single-pass transcoding.

## Interface Tour

### Main Workspace

The top of the main workspace keeps the source preview, framing, live pipeline summary, and the first colorization controls in one place.

![Main page top](docs/images/main-page-top.png)

Blind comparison setup lives directly in the main page so you can capture a preview start offset, choose candidate models, and launch anonymized samples without leaving the run configuration flow.

![Main page blind compare](docs/images/main-page-blind-compare.png)

The blind-test box is where you set the one-second or multi-second sample workflow, capture the source position, and open the synchronized comparison workspace.

![Blind test box](docs/images/Blind-test-box.png)

### Model And Pipeline Controls

The focused colorization panel is where you turn the color stage on, choose the active colorizer, tune model-specific options such as DeepRemaster processing mode, and manage the source-linked reference library for that clip.

![Right-hand model selector](docs/images/right-hand-model-selector.png)

The full processing track keeps colorization, upscaling, interpolation, export settings, and the final run action in one continuous path so you can see the whole configured pipeline before launching it.

![Right-hand encoder and interpolation controls](docs/images/right-hand-w-encoder-and-interpolation.png)

### Jobs And Comparison

The Jobs page gives you a compact queue and history view for pipeline runs, source conversions, and replayable templates.

![Jobs page](docs/images/jobs-page.png)

The job details page exposes the stored request, progress state, and replay actions needed to restart or reload a run without rebuilding the settings by hand.

![Job details page](docs/images/jobs-details-page.png)

The detached comparison workspace is the core review surface for blind testing: the Source pane and every sample pane stay synchronized while you resize panes, zoom content, and pan around artifacts.

![Model comparison page](docs/images/Model-comparison-page.png)

## Output Showcase

These reference stills in `docs/images/showcase` show the kind of before-and-after material the app is meant to inspect and export.

Source frame example:

![Voyager source](docs/images/showcase/voyager-source-576x432.png)

Upscaled output example:

![Voyager upscaled](docs/images/showcase/voyager-upscaled-2304x1728.png)

Upscaled plus interpolated output example:

![Voyager upscaled and interpolated](docs/images/showcase/voyager-upscaled-interpolated-2304x1728-60fps.png)

Probe frame at `00:02:00.000`:

![Probe 00 02 00](docs/images/showcase/probe-00-02-00_000.png)

Probe frame at `00:02:30.000`:

![Probe 00 02 30](docs/images/showcase/probe-00-02-30_000.png)

Probe frame at `00:03:00.000`:

![Probe 00 03 00](docs/images/showcase/probe-00-03-00_000.png)

Probe frame at `00:03:30.000`:

![Probe 00 03 30](docs/images/showcase/probe-00-03-30_000.png)

## Pipeline Instructions

### Desktop App

Launch the desktop app:

```powershell
./scripts/run.ps1
```

Then use one of these workflows:

1. Colorize only

- Select a source video.
- Enable the Colorization step.
- Turn the Upscaler and Frame Rate Booster steps off.
- Choose a colorizer model.
- If the selected colorizer supports reference images, add them from the source context panel.
- Choose output codec, container, GPU, and output path.
- Click Run.

2. Upscale only

- Select a source video.
- In the Upscaler section, choose the upscale model you want.
- Set Frame Rate Booster to Off.
- Choose output sizing, codec, container, GPU, and quality settings.
- Click Run Upscale.

3. Colorize before upscale

- Select a source video.
- Enable Colorization and Upscaler.
- Choose a colorizer model and an upscale model.
- Add reference images if the colorizer uses source context.
- Leave Frame Rate Booster off if you only want colorization plus upscale.
- Click Run.

4. Interpolate an existing video without upscaling

- Select a source video.
- In Frame Rate Booster, choose Interpolate Existing Video.
- Choose the target frame rate: 30 fps or 60 fps.
- Set output codec, container, and GPU.
- Click Run Interpolation.

5. Run the combined VideoUpgrader pipeline

- Select a source video.
- Choose the upscale model and output sizing settings.
- Optionally enable Colorization first if you want grayscale footage colorized before upscale.
- In Frame Rate Booster, choose Interpolate After Upscale.
- Choose the target frame rate: 30 fps or 60 fps.
- Click Run Upscale + Interpolation.

Notes:

- `Colorize only` uses the selected colorizer as the active processing model for the run.
- `Colorize before upscale` runs colorization on extracted frames before the upscale stage begins.
- If the source is already at or above the selected target frame rate, the app warns before continuing.
- Interpolation keeps the original audio track attached to the final export.
- Full pipeline outputs keep the original audio by remuxing it back onto the final rendered video after processing.
- The result panel includes interpolation diagnostics in a collapsed details box for segment count, overlap, source fps, and output fps.

### Python Worker CLI

Set the worker path first:

```powershell
$env:UPSCALER_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
$env:PYTHONPATH='python'
```

1. Run colorize only

```powershell
& $env:UPSCALER_PYTHON python/upscaler_worker/cli.py run-realesrgan-pipeline --source input.mp4 --model-id colormnet --colorization-mode colorizeOnly --colorizer-model-id colormnet --color-reference-image path/to/reference.png --output-mode preserveAspect4k --preset qualityBalanced --interpolation-mode off --aspect-ratio-preset 16:9 --resolution-basis exact --target-width 3840 --target-height 2160 --output-path artifacts/output/colorized-only.mp4 --codec h264 --container mp4
```

2. Run upscale only

```powershell
& $env:UPSCALER_PYTHON python/upscaler_worker/cli.py run-realesrgan-pipeline --source input.mp4 --model-id realesrgan-x4plus --output-mode preserveAspect4k --preset qualityBalanced --interpolation-mode off --aspect-ratio-preset 16:9 --resolution-basis exact --target-width 3840 --target-height 2160 --output-path artifacts/output/upscaled-only.mp4 --codec h264 --container mp4
```

3. Run colorize before upscale

```powershell
& $env:UPSCALER_PYTHON python/upscaler_worker/cli.py run-realesrgan-pipeline --source input.mp4 --model-id realesrnet-x4plus --colorization-mode beforeUpscale --colorizer-model-id deepremaster --color-reference-image path/to/reference-a.png --color-reference-image path/to/reference-b.png --output-mode preserveAspect4k --preset qualityBalanced --interpolation-mode off --aspect-ratio-preset 16:9 --resolution-basis exact --target-width 3840 --target-height 2160 --output-path artifacts/output/colorized-and-upscaled.mp4 --codec h264 --container mp4
```

4. Run interpolation only

```powershell
& $env:UPSCALER_PYTHON python/upscaler_worker/cli.py run-realesrgan-pipeline --source input.mp4 --model-id realesrgan-x4plus --output-mode preserveAspect4k --preset qualityBalanced --interpolation-mode interpolateOnly --interpolation-target-fps 60 --aspect-ratio-preset 16:9 --resolution-basis exact --target-width 3840 --target-height 2160 --output-path artifacts/output/interpolated-only.mp4 --codec h264 --container mp4
```

5. Run upscale and interpolation together

```powershell
& $env:UPSCALER_PYTHON python/upscaler_worker/cli.py run-realesrgan-pipeline --source input.mp4 --model-id realesrgan-x4plus --output-mode preserveAspect4k --preset qualityBalanced --interpolation-mode afterUpscale --interpolation-target-fps 60 --aspect-ratio-preset 16:9 --resolution-basis exact --target-width 3840 --target-height 2160 --output-path artifacts/output/upscaled-and-interpolated.mp4 --codec h264 --container mp4
```

CLI notes:

- `--colorization-mode off` disables colorization.
- `--colorization-mode colorizeOnly` runs the selected colorizer as the main processing stage.
- `--colorization-mode beforeUpscale` colorizes frames before they enter the upscale stage.
- `--colorizer-model-id` selects the colorization model when colorization is enabled.
- `--color-reference-image` can be repeated for models that consume reference images.
- `--interpolation-mode off` means upscale only.
- `--interpolation-mode interpolateOnly` skips the upscale stage and runs interpolation on the source video.
- `--interpolation-mode afterUpscale` runs interpolation after the upscale stage completes.
- The worker lazily downloads the RIFE runtime the first time an interpolation job runs.
- Add `--gpu-id`, `--tile-size`, `--precision`, `--pytorch-runner`, or `--deepremaster-processing-mode` if you need to pin a device or tune runtime behavior.

## Prerequisites

- Windows
- `winget` available through App Installer on a current Windows install

`bootstrap.ps1` is now the intended soup-to-nuts setup path. It will install any missing local workstation dependencies it needs for this repo, including:

- Node.js LTS
- Python 3.10
- Rust and Cargo
- Microsoft C++ Build Tools with the Desktop C++ workload
- Microsoft Edge WebView2 Runtime

Preferred Python environment variable:

```powershell
$env:UPSCALER_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
```

If that variable is not set, the repository scripts prefer the repo-local `.venv` and only fall back to `python` when needed.

## One-Time Setup

```powershell
./scripts/bootstrap.ps1
```

`bootstrap.ps1` now performs the full local deploy path for a fresh clone:

- installs missing Windows toolchains and runtimes through `winget`
- creates the repo-local `.venv` when needed
- installs npm and Python dependencies
- installs the pinned CUDA PyTorch runtime
- builds the web and desktop host
- pre-downloads the current runtime packages and runnable model weights, including the built-in RVRT repo and Vimeo x4 checkpoint

After bootstrap completes, the first app launch should not need to stop for model or runtime downloads.

## Standard Commands

Run tests:

```powershell
./scripts/test.ps1
```

Build frontend and desktop host checks:

```powershell
./scripts/build.ps1
```

Build and run tests in the same step:

```powershell
./scripts/build.ps1 -RunTests
```

Launch the desktop app:

```powershell
./scripts/run.ps1
```

Generate synthetic benchmark fixtures:

```powershell
./scripts/generate-benchmarks.ps1
```

Run the automated desktop smoke test:

```powershell
./scripts/desktop-test.ps1
```

## Frontend / Desktop Dev Commands

These are also available directly through `npm`:

```powershell
npm run dev
npm run build:web
npm run test:web
npm run test:gui
npm run tauri:dev
npm run tauri:build
```

## Benchmarking And Worker Tools

The Python worker exposes direct benchmark and diagnostic entrypoints.

Examples:

Generate a small synthetic fixture:

```powershell
$env:PYTHONPATH='python'
& $env:UPSCALER_PYTHON python/upscaler_worker/cli.py generate-benchmark --output-dir artifacts/benchmarks --name quick_fixture --frames 4 --width 1280 --height 720 --downscale-width 320 --downscale-height 180
```

Compare precision quality on a small frame set:

```powershell
$env:PYTHONPATH='python'
& $env:UPSCALER_PYTHON python/upscaler_worker/cli.py compare-precision-quality --manifest-path artifacts/benchmarks/quick_fixture/manifest.json --model-id swinir-realworld-x4 --tile-size 128 --reference-precision fp32 --candidate-precision bf16 --max-frames 4
```

Install the optional TensorRT runner dependencies:

```powershell
& $env:UPSCALER_PYTHON -m pip install -r python/requirements-tensorrt.txt
```

Benchmark SwinIR with the TensorRT runner:

```powershell
$env:PYTHONPATH='python'
& $env:UPSCALER_PYTHON python/upscaler_worker/cli.py benchmark-upscaler --manifest-path artifacts/benchmarks/quick_fixture/manifest.json --model-id swinir-realworld-x4 --tile-sizes 128 --repeats 1 --precision fp32 --pytorch-runner tensorrt
```

Benchmark the PyTorch streaming path:

```powershell
$env:PYTHONPATH='python'
& $env:UPSCALER_PYTHON python/upscaler_worker/benchmark_pytorch_pipeline_paths.py --model-id swinir-realworld-x4 --execution-paths streaming --repeats 1 --duration-seconds 10 --width 1280 --height 720 --fps 24 --tile-size 128 --preset qualityBalanced --precision bf16 --output-mode preserveAspect4k --resolution-basis exact --target-width 3840 --target-height 2160
```

RVRT now runs through the same external runner contract in both the desktop pipeline and the worker benchmarks. If the official RVRT repo is available at `tmp/RVRT`, the app will default to the built-in `upscaler_worker.rvrt_external_runner` automatically. `UPSCALER_RVRT_COMMAND` remains available as an override and can point to any command template that reads an input PNG sequence and writes an output PNG sequence. The command may use placeholders like `{input_dir}`, `{output_dir}`, `{model_id}`, `{tile_size}`, and `{frame_count}`.

Example:

```powershell
$env:PYTHONPATH='python'
$env:UPSCALER_RVRT_COMMAND='python path/to/your_rvrt_runner.py --input {input_dir} --output {output_dir} --model {model_id}'
& $env:UPSCALER_PYTHON python/upscaler_worker/cli.py benchmark-upscaler --manifest-path artifacts/benchmarks/quick_fixture/manifest.json --model-id rvrt-x4 --tile-sizes 128 --repeats 1 --precision fp32
```

On this repo, the built-in default is active when `tmp/RVRT` exists, so the desktop app and worker CLI can run RVRT without setting `UPSCALER_RVRT_COMMAND` manually.

## Runtime Notes

The current measured PyTorch worker stack on the main workstation is:

- `torch 2.11.0+cu128`
- CUDA runtime `12.8`
- cuDNN `91900`
- Triton `3.6.0`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`

The worker now supports:

- Explicit precision selection: `fp32`, `fp16`, `bf16`
- Selectable PyTorch runner: `torch` or `tensorrt`
- `torch.compile`
- Selectable compile mode
- Optional cudagraphs path
- Adaptive micro-batching
- Overlapped streaming decode / upscale / encode execution

The worker also enables CUDA fast-paths for this hardware:

- cuDNN benchmark
- TF32 matmul
- high float32 matmul precision

## Benchmark Snapshot

The following numbers are verified benchmark artifacts from this repository, using the 10-second SwinIR streaming proof case:

- Model: `swinir-realworld-x4`
- Execution path: `streaming`
- Input: `1280x720`, `24 fps`, `10 seconds`
- Output: `3840x2160`
- Tile size: `128`
- Preset: `qualityBalanced`

| Configuration | Throughput (fps) | Wall Time (s) | Peak GPU Memory | Notes |
| --- | ---: | ---: | ---: | --- |
| fp32 baseline | 0.293727 | 817.10 | 5.28 GB | baseline streaming result |
| fp32 + torch.compile | 0.298449 | 804.17 | 22.84 GB | small gain, very large VRAM cost |
| bf16 | 0.412037 | 582.48 | 7.24 GB | major speedup over fp32 |
| bf16 + torch.compile | 0.478656 | 501.41 | 19.53 GB | faster than bf16, expensive in VRAM |
| bf16 + torch.compile + cudagraphs | 0.522266 | 459.55 | 9.10 GB | best measured PyTorch result so far |
| TensorRT fp32, cached engine, dedicated stream | 0.544173 | 441.15 | 8.30 GB | current best measured long-run SwinIR result |

### What These Results Mean

- `bf16` is the first big win for the PyTorch SwinIR path.
- `torch.compile` on its own helps, but its VRAM cost matters.
- `bf16 + torch.compile + cudagraphs` is the strongest measured PyTorch configuration so far on this GPU.
- Cached TensorRT fp32 is now slightly faster than the best PyTorch path on the same 10-second SwinIR streaming case.
- Compile startup overhead is still real, so short clips and smoke tests can understate steady-state gains.
- TensorRT still has a large first-run engine build cost, so cold runs remain much slower than cached runs.

## Quality Snapshot

We also ran a short 4-frame fp32-vs-bf16 comparison on a synthetic fixture.

- fp32 vs bf16 average MAE: `0.894882`
- fp32 vs bf16 average RMSE: `1.507988`
- fp32 vs bf16 average PSNR: `44.567502`
- fp32 vs bf16 average SSIM: `0.999676`

The practical takeaway is that bf16 is extremely close to fp32 on the tested SwinIR sample while being materially faster.

## Artifact Paths

Benchmark artifacts live under:

```text
artifacts/benchmarks
```

Relevant benchmark outputs generated in this workspace include:

- `pytorch-pipeline-swinir-streaming-t128-10s-720p-to-4k.json`
- `pytorch-pipeline-swinir-streaming-t128-10s-720p-to-4k-compile.json`
- `pytorch-pipeline-swinir-streaming-t128-10s-720p-to-4k-bf16.json`
- `pytorch-pipeline-swinir-streaming-t128-10s-720p-to-4k-bf16-compile.json`
- `pytorch-pipeline-swinir-streaming-t128-10s-720p-to-4k-bf16-compile-cudagraphs.json`
- `swinir-precision-quality-fp32-vs-bf16-4f.json`

## Repository Layout

- `src/`: React desktop UI
- `src-tauri/`: Tauri desktop host and Rust-side command surface
- `python/`: worker runtime, benchmarking, model integration, and media pipeline code
- `scripts/`: bootstrap, build, test, and run entrypoints
- `config/`: model catalog and app configuration
- `artifacts/`: generated benchmarks, outputs, jobs, and runtime assets
- `context/`: product requirements and planning context

## Project Status

This is an actively evolving workstation-oriented app, not a finished mass-market release.

Already real:

- desktop shell
- comparison-oriented UI
- runnable model catalog
- synthetic benchmark generation
- PyTorch and NCNN worker paths
- verified benchmark and precision-comparison tooling

Still expanding:

- broader model coverage
- deeper export polish
- richer settings-vs-settings comparison workflows
- video-native temporal SR backends
- more automatic visual diagnostics and comparison outputs

## Why This Project Exists

VideoUpgrader is for people who do not want to judge an upscaler from marketing shots, one cherry-picked crop, or a single render preset. It is for repeatable evaluation, careful comparison, evidence-backed choices, and now frame-rate upgrades that can be validated the same way.