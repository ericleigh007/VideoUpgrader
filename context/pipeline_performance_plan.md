# Pipeline Performance Plan

## Goal

Speed up VideoUpgrader pipelines by keeping the discrete GPU busy, using available VRAM deliberately, and avoiding unnecessary write/read cycles between stages.

This plan is based on the current worker state, the GPU validation run from May 2026, and existing benchmark artifacts.

## Current Baseline

- The workstation exposes different GPU ordinal spaces:
  - App/Vulkan runtime ids: Intel `0`, NVIDIA RTX PRO 6000 `1`.
  - PyTorch CUDA ids: NVIDIA `cuda:0`.
- Recent GPU routing validation confirmed:
  - DRUNet AI denoise can run on PyTorch `cuda:0`.
  - Real-ESRGAN NCNN can run with Vulkan `-g 1`.
  - RIFE NCNN can run with Vulkan `-g 1`.
  - H.265 can use `hevc_nvenc` when NVIDIA GPU `1` is selected.
- The existing PyTorch streaming path is implemented only for the no-denoise, no-interpolation PyTorch image-SR path.
- Denoise and interpolation still force directory-based frame transport.
- For the 60s 720p-to-4K PyTorch SR benchmark, the lower-scratch streaming path was faster than file-IO and reduced scratch use sharply, but model inference still dominated wall time.

## Primary Bottlenecks

1. **Stage boundaries materialize frames to disk.**
   Extract, denoise, colorize, upscale, interpolate, encode, and remux currently exchange most data through frame directories. This burns disk bandwidth, CPU time, and synchronization time.

2. **GPU work is bursty.**
   Each stage owns the GPU for a short period, then waits for file IO, subprocess startup, or downstream work. The goal should be a bounded pipeline where decode, GPU inference, and encode run concurrently enough that the GPU almost always has queued work.

3. **Model/runtime boundaries are inconsistent.**
   PyTorch models can share CUDA tensors and batching logic, but NCNN/RIFE external executables require directory or process-pipe boundaries today.

4. **VRAM is under-used by default.**
   The requirements allow roughly 24 GB VRAM. Recent DRUNet validation used only a few GB. We should prefer larger batches/chunks until telemetry shows real pressure.

5. **Cache semantics are frame-file oriented.**
   The app caches whole job outputs and frame directories, but does not yet have a reusable decoded-frame, tensor-batch, or stage-output cache policy.

## Strategy

Do not try to make every model fully streaming at once. Introduce a staged execution graph where each edge declares its transport:

- `memory-tensor`: CUDA or CPU tensor batches inside one Python process.
- `memory-array`: NumPy/RGB arrays through queues.
- `pipe-rawvideo`: FFmpeg raw video stdin/stdout.
- `directory-frames`: compatibility fallback for NCNN/external tools.
- `encoded-video`: final or intermediate container.

Then optimize the graph one edge at a time while preserving existing file-IO fallback.

## Phase 1: Instrument And Autotune

### 1. Add Per-Stage GPU Telemetry

Current progress samples GPU memory globally, but not stage-specific utilization. Add telemetry fields:

- `stageGpuUtilizationPeakPercent`
- `stageGpuMemoryPeakBytes`
- `stageGpuActiveSampleCount`
- `stageQueueDepths`
- `stageInputWaitSeconds`
- `stageOutputWaitSeconds`

For PyTorch stages, also record:

- `torchDevice`
- `torchPrecision`
- `torchBatchSize`
- `torchPeakAllocatedBytes`
- `torchPeakReservedBytes`

### 2. Add Benchmark Matrix

Extend `benchmark_pytorch_pipeline_paths.py` or add a new `benchmark_pipeline_matrix.py` to run the same fixture across:

- Denoiser: off, DRUNet, FastDVDnet.
- Upscaler: Real-ESRGAN NCNN, PyTorch RealESRNet.
- Interpolation: off, RIFE after upscale.
- Encoder: libx265, hevc_nvenc.
- Transport: file-IO, streaming where supported.
- Tile size: 0/default, 128, 192, 256, 384.
- Batch/chunk size: model-specific sweep.

Outputs should include wall time, per-stage time, scratch bytes, peak VRAM, peak GPU utilization, and output path.

### 3. Autotune Defaults From Measured Hardware

Add a small runtime policy layer:

- If NVIDIA GPU is selected and VRAM >= 24 GB, default AI denoise to BF16 and larger batches.
- For DRUNet, prefer full-frame batches until `torch.cuda.OutOfMemoryError`, then reduce batch size and/or enable tiling.
- For FastDVDnet, increase temporal chunk size while preserving radius overlap.
- For PyTorch SR, select batch size using available VRAM, tile size, and precision.

## Phase 2: Stream The All-PyTorch Path

Target graph:

```text
FFmpeg decode pipe -> RGB array queue -> PyTorch denoise/color/SR tensor batches -> FFmpeg/NVENC encode pipe -> audio remux
```

This is the highest-value clean path because denoise, colorization, and PyTorch SR can share one process and one CUDA runtime.

Implementation steps:

1. Generalize `_run_streaming_pytorch_pipeline` into a reusable `PipelineGraph` or `StreamingStageGraph`.
2. Let it compose optional PyTorch stages:
   - AI denoise framewise models.
   - PyTorch colorizers.
   - PyTorch image SR.
3. Keep decoded frames as arrays until the first PyTorch stage, then keep CUDA tensors between compatible PyTorch stages when shapes permit.
4. Use pinned host memory for CPU-to-GPU transfer.
5. Use non-blocking tensor transfers.
6. Use one or more CUDA streams only after basic batching is stable.
7. Keep final audio remux as-is initially.

Fallback: if any stage is external-only, route just that edge through directory frames until a pipe bridge exists.

## Phase 3: Add Pipe Bridges For External GPU Tools

NCNN tools are currently directory-first. Directory transport is safe but expensive. Investigate whether the selected external tools can support stdin/stdout or a frame-server pattern:

- Real-ESRGAN NCNN Vulkan: likely directory-oriented; if no pipe support, keep directory fallback.
- RIFE NCNN Vulkan: likely directory-oriented; if no pipe support, keep directory fallback.
- FFmpeg: already supports rawvideo pipe for decode/encode.

If NCNN tools cannot stream, reduce their damage:

- Use larger segment sizes to amortize subprocess startup.
- Keep denoise/upscale/interpolation directories on the fastest local scratch disk.
- Avoid PNG when lossless raw or faster image formats are safe for intermediate frames.
- Consider uncompressed raw frame chunks or memory-mapped frame blocks as an internal transport if external tools can be adapted.

## Phase 4: Keep The GPU Busy Across Stages

The current overlapped pipeline runs extract and downstream work concurrently, but GPU stages are largely serial. Improve utilization with bounded queues:

- Decode queue depth: 2-4 batches.
- Inference queue depth: enough to hide decode/encode latency, not enough to exceed VRAM budget.
- Encode queue depth: 1-2 batches.
- For temporal models, queue chunks with overlap metadata, not duplicated frame files.

For all-PyTorch stages, prefer fused stage execution per batch:

```text
batch -> denoise -> colorize -> upscale -> enqueue encoded pixels
```

This avoids writing denoised frames only to read them back for upscaling.

## Phase 5: Smarter Caching

Cache only when reuse is likely or recomputation is expensive:

### Keep

- Downloaded model assets and source repos.
- Final outputs.
- Benchmark manifests.
- Optional user-requested comparison artifacts.

### Conditional

- Decoded source frame cache: useful for repeated comparisons over the same interval.
- Denoised output cache: useful when comparing multiple upscalers with the same denoiser.
- Colorized output cache: useful when comparing multiple upscalers after colorization.

### Avoid By Default

- Writing every intermediate frame for one-off full exports.
- Keeping both pre- and post-denoise directories when no downstream reuse is planned.

Cache keys must include:

- Source path, source media hash/mtime/size, start/duration/fps.
- Model id, checkpoint hash/version, precision, tile, batch/chunk settings.
- GPU-affecting runtime settings only when they change output.
- Color references and denoise strength parameters.

## Phase 6: UI And Policy Changes

- Show selected GPU as both app/Vulkan id and resolved PyTorch CUDA device when a PyTorch stage is present.
- Show whether encoder is NVENC or software.
- Add a performance mode selector:
  - `Balanced`: current safe defaults.
  - `Max GPU`: larger batches/chunks, BF16 where appropriate, more VRAM use.
  - `VRAM Safe`: smaller batches/tiles and lower queue depths.
- Surface warnings when a stage falls back to CPU or integrated GPU.

## Priority Order

1. **Benchmark matrix and telemetry.** Needed to prevent blind tuning.
2. **Correct GPU id mapping everywhere.** Recently fixed for denoise; keep tests around it.
3. **Autotune AI denoise batches/chunks.** Fast payoff, low architecture risk.
4. **Generalize the streaming graph for PyTorch-only pipelines.** Biggest clean reduction in IO.
5. **Add denoise/colorize support to streaming PyTorch path.** Removes major intermediate writes.
6. **Investigate NCNN/RIFE pipe alternatives.** Valuable but riskier because external tool support may be limited.
7. **Cache reusable stage outputs intentionally.** Avoid turning scratch into a permanent landfill.

## Validation Gates

Each optimization must pass:

- Unit tests for routing, precision policy, and cache-key behavior.
- A small end-to-end smoke with DRUNet -> Real-ESRGAN -> RIFE -> NVENC.
- A PyTorch streaming-vs-file-IO benchmark.
- GPU activity detection with nonzero utilization or meaningful memory delta.
- Output frame count and AV-sync checks.
- Scratch-size comparison against the baseline.

## Near-Term Engineering Tasks

1. Add a `pipeline_matrix` benchmark command with GPU utilization sampling.
2. Add tests that app GPU `1` maps to PyTorch `cuda:0` but NCNN/RIFE receive `-g 1`.
3. Add DRUNet batch-size autotuning with OOM backoff.
4. Extend streaming path to allow AI denoise before PyTorch SR.
5. Add a cache-policy object that decides when to retain or discard intermediate stage outputs.
6. Record stage transport types in result JSON so the UI and benchmarks can show where file IO still exists.

## Current Benchmark Commands

Generate an execution-path benchmark with overall FPS, per-stage effective FPS, resource peaks, scratch usage, and GPU activity sampling:

```powershell
$env:PYTHONPATH='python'
.\.venv\Scripts\python.exe -m upscaler_worker.benchmark_pytorch_pipeline_paths --execution-paths file-io,streaming --repeats 1 --duration-seconds 4 --width 320 --height 180 --fps 12 --tile-size 128 --target-width 640 --target-height 360 --output artifacts/benchmarks/pytorch-pipeline-paths-stage-fps-smoke.json
```

Compare two execution paths inside one benchmark file as a before/after report:

```powershell
$env:PYTHONPATH='python'
.\.venv\Scripts\python.exe -m upscaler_worker.compare_pipeline_benchmarks --input artifacts/benchmarks/pytorch-pipeline-paths-stage-fps-smoke.json --before-execution-path file-io --after-execution-path streaming --output artifacts/benchmarks/pytorch-pipeline-paths-stage-fps-smoke-comparison.json
```

Compare two separately captured benchmark files around a change:

```powershell
$env:PYTHONPATH='python'
.\.venv\Scripts\python.exe -m upscaler_worker.compare_pipeline_benchmarks --before artifacts/benchmarks/before.json --after artifacts/benchmarks/after.json --before-execution-path file-io --after-execution-path streaming --output artifacts/benchmarks/before-after-comparison.json
```
