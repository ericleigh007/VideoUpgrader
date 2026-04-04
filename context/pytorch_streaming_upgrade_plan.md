# PyTorch Streaming Upgrade Plan

## Objective

Replace the current PyTorch frame-file pipeline with an in-memory buffered decode -> upscale -> encode path while preserving the current desktop contract, telemetry, and output behavior.

## Required Order

1. Measure the current file-io path and lock in a comparison harness.
2. Add a stable execution-path seam so file-io and streaming can coexist safely.
3. Plan and implement the streaming path behind that seam.

## Completed Baseline Work

- Direct FFmpeg launch-overhead benchmarks exist in:
  - `artifacts/benchmarks/ffmpeg-overhead.json`
  - `artifacts/benchmarks/ffmpeg-overhead-720p.json`
- PyTorch execution-path comparison harness exists in `python/upscaler_worker/benchmark_pytorch_pipeline_paths.py`.
- The pipeline now exposes an explicit PyTorch execution path selector with `file-io` as the default and `streaming` reserved for the future implementation.

## Safety And Comparison Strategy

- Keep the current file-io path as the default.
- Route only `pytorch-image-sr` models through the execution-path selector.
- Leave NCNN and other non-PyTorch backends unchanged.
- Benchmark both paths with the same fixture, tile size, model, and output settings.
- Keep the public Tauri command surface stable while the worker evolves behind it.

## Streaming Slice A

### Scope

- Decode video frames in memory.
- Batch decoded frames directly into PyTorch tensors.
- Encode upscaled frames from memory.
- Keep a final mux/remux step if full in-process audio handling is not ready.

### Modules

- `python/upscaler_worker/media_streaming.py`
  - PyAV-backed decode and encode helpers.
- `python/upscaler_worker/pipeline_pytorch_streaming.py`
  - In-memory PyTorch fast path orchestration.
- `python/upscaler_worker/models/pytorch_sr.py`
  - Shared inference helpers that accept arrays/tensors rather than only file paths.

### Data Flow

1. PyAV decoder yields RGB frames with timing metadata.
2. A bounded queue feeds micro-batches to the model.
3. PyTorch runs the same descriptor path already used in the file-io flow.
4. An encoder consumes the upscaled frames from memory.
5. Audio is preserved with a final mux step first, then upgraded later.

## Telemetry Requirements

- Preserve existing progress fields:
  - `segmentIndex`
  - `segmentCount`
  - `segmentProcessedFrames`
  - `segmentTotalFrames`
  - `batchIndex`
  - `batchCount`
- Preserve stage timings and throughput reporting.
- Add queue-depth telemetry only if it is cheap and stable.

## Fallback Requirements

- If the streaming path fails or is disabled, file-io remains the supported fallback.
- Benchmarks must be able to compare `file-io` and `streaming` under the same output contract.

## Benchmark Criteria

- Wall-clock time
- Average throughput FPS
- Stage timings
- Peak RAM
- Peak VRAM
- Scratch disk usage
- Output equivalence checks where feasible

## Implementation Order

1. Add in-memory frame batch helpers to the PyTorch model layer.
2. Add PyAV decoder with bounded buffering.
3. Add persistent encoder path.
4. Add streaming pipeline orchestration behind the execution-path selector.
5. Benchmark `streaming` against `file-io` using the comparison harness.
6. Only then consider deeper audio-path integration.