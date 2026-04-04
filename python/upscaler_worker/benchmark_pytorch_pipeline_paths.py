from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upscaler_worker.measure_ffmpeg_overhead import _build_fixture
from upscaler_worker.model_catalog import model_backend_id
from upscaler_worker.models.pytorch_sr import FRAME_BATCH_SIZE_OVERRIDE_ENV, resolve_frame_batch_size_override, resolve_pytorch_runner
from upscaler_worker.precision import resolve_precision_mode
from upscaler_worker.pipeline import (
    PYTORCH_EXECUTION_PATH_FILE_IO,
    PYTORCH_EXECUTION_PATH_STREAMING,
    SUPPORTED_PYTORCH_EXECUTION_PATHS,
    run_realesrgan_pipeline,
)
from upscaler_worker.runtime import ensure_runtime_assets


LIVE_PROGRESS_POLL_SECONDS = 1.0
LIVE_PROGRESS_HEARTBEAT_SECONDS = 5.0
GPU_ACTIVITY_UTILIZATION_THRESHOLD = 5
GPU_ACTIVITY_MEMORY_DELTA_BYTES = 256 * 1024 * 1024


@dataclass
class GpuSample:
    utilization_percent: int | None
    memory_used_bytes: int | None
    memory_total_bytes: int | None


@dataclass
class LiveRunMonitorState:
    sample_count: int = 0
    active_sample_count: int = 0
    max_utilization_percent: int | None = None
    first_memory_used_bytes: int | None = None
    peak_memory_used_bytes: int | None = None
    last_progress_signature: tuple[object, ...] | None = None
    last_reported_at: float = 0.0
    last_message: str | None = None


def _format_bytes_gib(raw_value: int | None) -> str:
    if raw_value is None:
        return "n/a"
    return f"{raw_value / (1024 ** 3):.2f} GiB"


def _sample_gpu_activity() -> GpuSample:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return GpuSample(None, None, None)

    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return GpuSample(None, None, None)

    if completed.returncode != 0:
        return GpuSample(None, None, None)

    utilization_values: list[int] = []
    used_values_mb: list[int] = []
    total_values_mb: list[int] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            utilization_text, used_text, total_text = [part.strip() for part in line.split(",", maxsplit=2)]
            utilization_values.append(int(float(utilization_text)))
            used_values_mb.append(int(float(used_text)))
            total_values_mb.append(int(float(total_text)))
        except ValueError:
            continue

    if not utilization_values or not total_values_mb:
        return GpuSample(None, None, None)

    return GpuSample(
        utilization_percent=max(utilization_values),
        memory_used_bytes=sum(used_values_mb) * 1024 * 1024,
        memory_total_bytes=sum(total_values_mb) * 1024 * 1024,
    )


def _assess_gpu_activity(state: LiveRunMonitorState) -> dict[str, object]:
    memory_delta = None
    if state.first_memory_used_bytes is not None and state.peak_memory_used_bytes is not None:
        memory_delta = max(0, state.peak_memory_used_bytes - state.first_memory_used_bytes)

    activity_detected = False
    if state.max_utilization_percent is not None and state.max_utilization_percent >= GPU_ACTIVITY_UTILIZATION_THRESHOLD:
        activity_detected = True
    if memory_delta is not None and memory_delta >= GPU_ACTIVITY_MEMORY_DELTA_BYTES:
        activity_detected = True

    warning = None
    if state.sample_count > 0 and not activity_detected:
        warning = (
            "No meaningful GPU activity was observed during this run. "
            "The benchmark may not have exercised the GPU as expected."
        )

    return {
        "sampleCount": state.sample_count,
        "activeSampleCount": state.active_sample_count,
        "maxUtilizationPercent": state.max_utilization_percent,
        "firstObservedMemoryUsedBytes": state.first_memory_used_bytes,
        "peakObservedMemoryUsedBytes": state.peak_memory_used_bytes,
        "observedMemoryDeltaBytes": memory_delta,
        "activityDetected": activity_detected,
        "warning": warning,
    }


def _emit_live_status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _monitor_run_progress(
    *,
    progress_path: Path,
    stop_event: threading.Event,
    state: LiveRunMonitorState,
    execution_path: str,
    repeat_index: int,
) -> None:
    started_at = time.time()
    while not stop_event.wait(LIVE_PROGRESS_POLL_SECONDS):
        now = time.time()
        gpu_sample = _sample_gpu_activity()
        if gpu_sample.utilization_percent is not None:
            state.sample_count += 1
            state.max_utilization_percent = gpu_sample.utilization_percent if state.max_utilization_percent is None else max(state.max_utilization_percent, gpu_sample.utilization_percent)
            if gpu_sample.utilization_percent >= GPU_ACTIVITY_UTILIZATION_THRESHOLD:
                state.active_sample_count += 1
            if gpu_sample.memory_used_bytes is not None:
                if state.first_memory_used_bytes is None:
                    state.first_memory_used_bytes = gpu_sample.memory_used_bytes
                state.peak_memory_used_bytes = gpu_sample.memory_used_bytes if state.peak_memory_used_bytes is None else max(state.peak_memory_used_bytes, gpu_sample.memory_used_bytes)

        payload = None
        if progress_path.exists():
            try:
                payload = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None

        if payload is not None:
            signature = (
                payload.get("phase"),
                payload.get("percent"),
                payload.get("processedFrames"),
                payload.get("message"),
            )
            if signature != state.last_progress_signature or (now - state.last_reported_at) >= LIVE_PROGRESS_HEARTBEAT_SECONDS:
                phase = payload.get("phase", "unknown")
                percent = payload.get("percent")
                processed_frames = payload.get("processedFrames")
                total_frames = payload.get("totalFrames")
                rolling_fps = payload.get("rollingFramesPerSecond")
                average_fps = payload.get("averageFramesPerSecond")
                gpu_text = "GPU util n/a"
                if gpu_sample.utilization_percent is not None:
                    gpu_text = f"GPU util {gpu_sample.utilization_percent}%"
                    if gpu_sample.memory_used_bytes is not None and gpu_sample.memory_total_bytes is not None:
                        gpu_text += f", mem {_format_bytes_gib(gpu_sample.memory_used_bytes)}/{_format_bytes_gib(gpu_sample.memory_total_bytes)}"

                parts = [f"[{execution_path} run {repeat_index}] {phase}"]
                if percent is not None:
                    parts.append(f"{percent}%")
                if processed_frames is not None and total_frames is not None:
                    parts.append(f"frames {processed_frames}/{total_frames}")
                if rolling_fps is not None:
                    parts.append(f"rolling {float(rolling_fps):.2f} fps")
                if average_fps is not None:
                    parts.append(f"avg {float(average_fps):.2f} fps")
                parts.append(gpu_text)
                if payload.get("message"):
                    parts.append(str(payload["message"]))

                message = " | ".join(parts)
                _emit_live_status(message)
                state.last_message = message
                state.last_progress_signature = signature
                state.last_reported_at = now
        elif (now - state.last_reported_at) >= LIVE_PROGRESS_HEARTBEAT_SECONDS:
            gpu_text = "GPU util n/a"
            if gpu_sample.utilization_percent is not None:
                gpu_text = f"GPU util {gpu_sample.utilization_percent}%"
                if gpu_sample.memory_used_bytes is not None and gpu_sample.memory_total_bytes is not None:
                    gpu_text += f", mem {_format_bytes_gib(gpu_sample.memory_used_bytes)}/{_format_bytes_gib(gpu_sample.memory_total_bytes)}"
            message = (
                f"[{execution_path} run {repeat_index}] initializing | "
                f"elapsed {now - started_at:.1f}s | {gpu_text} | waiting for pipeline progress"
            )
            _emit_live_status(message)
            state.last_message = message
            state.last_reported_at = now


def _parse_execution_paths(raw_value: str) -> list[str]:
    paths = [part.strip().lower() for part in raw_value.split(",") if part.strip()]
    if not paths:
        raise ValueError("At least one execution path must be provided")

    unique_paths = []
    for path in paths:
        if path not in SUPPORTED_PYTORCH_EXECUTION_PATHS:
            supported = ", ".join(sorted(SUPPORTED_PYTORCH_EXECUTION_PATHS))
            raise ValueError(f"Unsupported execution path '{path}'. Expected one of: {supported}")
        if path not in unique_paths:
            unique_paths.append(path)
    return unique_paths


def _summarize_completed_runs(runs: list[dict[str, object]]) -> dict[str, object] | None:
    completed = [run for run in runs if run["status"] == "completed"]
    if not completed:
        return None

    wall_seconds = [float(run["wallSeconds"]) for run in completed]
    throughput = [float(run["averageThroughputFps"]) for run in completed]
    peak_rss = [float(run["resourcePeaks"]["processRssBytes"]) for run in completed if run.get("resourcePeaks", {}).get("processRssBytes") is not None]
    peak_gpu = [float(run["resourcePeaks"]["gpuMemoryUsedBytes"]) for run in completed if run.get("resourcePeaks", {}).get("gpuMemoryUsedBytes") is not None]
    return {
        "completedRuns": len(completed),
        "medianWallSeconds": round(statistics.median(wall_seconds), 6),
        "averageWallSeconds": round(statistics.fmean(wall_seconds), 6),
        "medianThroughputFps": round(statistics.median(throughput), 6),
        "averageThroughputFps": round(statistics.fmean(throughput), 6),
        "medianPeakProcessRssBytes": round(statistics.median(peak_rss), 2) if peak_rss else None,
        "medianPeakGpuMemoryUsedBytes": round(statistics.median(peak_gpu), 2) if peak_gpu else None,
    }


def benchmark_pytorch_pipeline_paths(
    *,
    model_id: str,
    execution_paths: list[str],
    repeats: int,
    duration_seconds: float,
    width: int,
    height: int,
    fps: int,
    tile_size: int,
    output_mode: str,
    resolution_basis: str,
    target_width: int | None,
    target_height: int | None,
    preset: str,
    fp16: bool,
    torch_compile_enabled: bool,
    torch_compile_mode: str = "reduce-overhead",
    torch_compile_cudagraphs: bool = False,
    bf16: bool,
    precision: str | None = None,
    pytorch_runner: str = "torch",
    channels_last: bool = False,
    frame_batch_size_override: int | None = None,
    reuse_loaded_model: bool = True,
) -> dict[str, object]:
    precision_mode = resolve_precision_mode(fp16=fp16, bf16=bf16, precision=precision)
    resolved_runner = resolve_pytorch_runner(pytorch_runner)
    runtime = ensure_runtime_assets()
    should_reuse_loaded_model = reuse_loaded_model and model_backend_id(model_id) == "pytorch-image-sr"

    with tempfile.TemporaryDirectory(prefix="upscaler-pytorch-pipeline-bench-") as temp_dir:
        root = Path(temp_dir)
        source_path = root / "fixture.mp4"
        _emit_live_status(f"Building benchmark fixture at {width}x{height} {fps}fps for {duration_seconds:.1f}s")
        _build_fixture(str(runtime["ffmpegPath"]), source_path, duration_seconds, width, height, fps)

        results: list[dict[str, object]] = []
        previous_frame_batch_size_override = os.environ.get(FRAME_BATCH_SIZE_OVERRIDE_ENV)
        try:
            if frame_batch_size_override is None:
                os.environ.pop(FRAME_BATCH_SIZE_OVERRIDE_ENV, None)
            else:
                os.environ[FRAME_BATCH_SIZE_OVERRIDE_ENV] = str(frame_batch_size_override)

            preloaded_pytorch_model = None
            if should_reuse_loaded_model:
                from upscaler_worker.models.pytorch_sr import load_runtime_model

                preload_log: list[str] = []
                _emit_live_status(
                    f"Preloading model {model_id} with runner={resolved_runner}, precision={precision_mode}, tile={tile_size}"
                )
                preloaded_pytorch_model = load_runtime_model(
                    model_id,
                    None,
                    precision_mode == "fp16",
                    tile_size,
                    preload_log,
                    preset=preset,
                    torch_compile_enabled=torch_compile_enabled,
                    torch_compile_mode=torch_compile_mode,
                    torch_compile_cudagraphs=torch_compile_cudagraphs,
                    bf16=precision_mode == "bf16",
                    precision=precision_mode,
                    pytorch_runner=resolved_runner,
                    channels_last_enabled=channels_last,
                )

            for execution_path in execution_paths:
                path_runs: list[dict[str, object]] = []
                for repeat_index in range(repeats):
                    output_path = root / f"{execution_path}_{repeat_index:02d}.mp4"
                    progress_path = root / f"{execution_path}_{repeat_index:02d}_progress.json"
                    started = time.perf_counter()
                    monitor_stop_event = threading.Event()
                    monitor_state = LiveRunMonitorState()
                    monitor_thread = threading.Thread(
                        target=_monitor_run_progress,
                        kwargs={
                            "progress_path": progress_path,
                            "stop_event": monitor_stop_event,
                            "state": monitor_state,
                            "execution_path": execution_path,
                            "repeat_index": repeat_index + 1,
                        },
                        daemon=True,
                    )
                    _emit_live_status(f"Starting {execution_path} run {repeat_index + 1}/{repeats}")
                    monitor_thread.start()
                    try:
                        result = run_realesrgan_pipeline(
                            source_path=str(source_path),
                            model_id=model_id,
                            output_mode=output_mode,
                            preset=preset,
                            gpu_id=None,
                            aspect_ratio_preset="16:9",
                            custom_aspect_width=16,
                            custom_aspect_height=9,
                            resolution_basis=resolution_basis,
                            target_width=target_width,
                            target_height=target_height,
                            crop_left=None,
                            crop_top=None,
                            crop_width=None,
                            crop_height=None,
                            progress_path=str(progress_path),
                            cancel_path=None,
                            preview_mode=True,
                            preview_duration_seconds=duration_seconds,
                            segment_duration_seconds=None,
                            output_path=str(output_path),
                            codec="h264",
                            container="mp4",
                            tile_size=tile_size,
                            fp16=precision_mode == "fp16",
                            torch_compile_enabled=torch_compile_enabled,
                            torch_compile_mode=torch_compile_mode,
                            torch_compile_cudagraphs=torch_compile_cudagraphs,
                            pytorch_execution_path=execution_path,
                            pytorch_runner=resolved_runner,
                            crf=18,
                            bf16=precision_mode == "bf16",
                            precision=precision_mode,
                            channels_last=channels_last,
                            preloaded_pytorch_model=preloaded_pytorch_model,
                        )
                        monitor_stop_event.set()
                        monitor_thread.join(timeout=3)
                        gpu_activity = _assess_gpu_activity(monitor_state)
                        if gpu_activity["warning"]:
                            _emit_live_status(f"[{execution_path} run {repeat_index + 1}] WARNING | {gpu_activity['warning']}")
                        path_runs.append(
                            {
                                "repeat": repeat_index + 1,
                                "status": "completed",
                                "wallSeconds": round(time.perf_counter() - started, 6),
                                "executionPath": result["executionPath"],
                                "averageThroughputFps": round(float(result["averageThroughputFps"]), 6),
                                "segmentCount": int(result["segmentCount"]),
                                "segmentFrameLimit": int(result["segmentFrameLimit"]),
                                "stageTimings": result["stageTimings"],
                                "resourcePeaks": result.get("resourcePeaks", {}),
                                "modelRuntime": result.get("modelRuntime"),
                                "gpuActivity": gpu_activity,
                            }
                        )
                    except NotImplementedError as error:
                        monitor_stop_event.set()
                        monitor_thread.join(timeout=3)
                        path_runs.append(
                            {
                                "repeat": repeat_index + 1,
                                "status": "not-implemented",
                                "executionPath": execution_path,
                                "error": str(error),
                            }
                        )
                        break
                    except Exception:
                        monitor_stop_event.set()
                        monitor_thread.join(timeout=3)
                        raise

                results.append(
                    {
                        "executionPath": execution_path,
                        "runs": path_runs,
                        "summary": _summarize_completed_runs(path_runs),
                    }
                )
        finally:
            if previous_frame_batch_size_override is None:
                os.environ.pop(FRAME_BATCH_SIZE_OVERRIDE_ENV, None)
            else:
                os.environ[FRAME_BATCH_SIZE_OVERRIDE_ENV] = previous_frame_batch_size_override

    return {
        "fixture": {
            "durationSeconds": duration_seconds,
            "width": width,
            "height": height,
            "fps": fps,
        },
        "output": {
            "mode": output_mode,
            "resolutionBasis": resolution_basis,
            "targetWidth": target_width,
            "targetHeight": target_height,
        },
        "modelId": model_id,
        "preset": preset,
        "runner": resolved_runner,
        "fp16": precision_mode == "fp16",
        "bf16": precision_mode == "bf16",
        "precision": precision_mode,
        "channelsLast": channels_last,
        "torchCompileEnabled": torch_compile_enabled,
        "torchCompileMode": torch_compile_mode,
        "torchCompileCudagraphs": torch_compile_cudagraphs,
        "tileSize": tile_size,
        "frameBatchSizeOverride": frame_batch_size_override,
        "reuseLoadedModel": should_reuse_loaded_model,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch pipeline execution paths.")
    parser.add_argument("--model-id", default="realesrnet-x4plus")
    parser.add_argument("--execution-paths", default=f"{PYTORCH_EXECUTION_PATH_FILE_IO},{PYTORCH_EXECUTION_PATH_STREAMING}")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--duration-seconds", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--preset", default="qualityBalanced", choices=["qualityMax", "qualityBalanced", "vramSafe"])
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--torch-compile-mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    parser.add_argument("--torch-compile-cudagraphs", action="store_true")
    parser.add_argument("--pytorch-runner", default="torch", choices=["torch", "tensorrt"])
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--frame-batch-size-override", type=int)
    parser.add_argument("--reload-model-per-run", action="store_true")
    parser.add_argument("--output-mode", default="preserveAspect4k")
    parser.add_argument("--resolution-basis", default="exact", choices=["exact", "width", "height"])
    parser.add_argument("--target-width", type=int, default=640)
    parser.add_argument("--target-height", type=int, default=360)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark_pytorch_pipeline_paths(
        model_id=args.model_id,
        execution_paths=_parse_execution_paths(args.execution_paths),
        repeats=args.repeats,
        duration_seconds=args.duration_seconds,
        width=args.width,
        height=args.height,
        fps=args.fps,
        tile_size=args.tile_size,
        output_mode=args.output_mode,
        resolution_basis=args.resolution_basis,
        target_width=args.target_width,
        target_height=args.target_height,
        preset=args.preset,
        fp16=args.fp16,
        torch_compile_enabled=args.torch_compile,
        torch_compile_mode=args.torch_compile_mode,
        torch_compile_cudagraphs=args.torch_compile_cudagraphs,
        bf16=args.bf16,
        precision=args.precision,
        pytorch_runner=args.pytorch_runner,
        channels_last=args.channels_last,
        frame_batch_size_override=resolve_frame_batch_size_override(
            None if args.frame_batch_size_override is None else str(args.frame_batch_size_override)
        ),
        reuse_loaded_model=not args.reload_model_per_run,
    )
    payload = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())