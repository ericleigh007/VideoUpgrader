from __future__ import annotations

import json
import sys
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from upscaler_worker.model_catalog import ensure_runnable_model, model_backend_id, model_label
from upscaler_worker.models.pytorch_sr import resolve_precision_mode
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


def _emit_live_status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


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


def _track_gpu_activity(state: LiveRunMonitorState, gpu_sample: GpuSample) -> None:
    if gpu_sample.utilization_percent is None:
        return

    state.sample_count += 1
    state.max_utilization_percent = gpu_sample.utilization_percent if state.max_utilization_percent is None else max(state.max_utilization_percent, gpu_sample.utilization_percent)
    if gpu_sample.utilization_percent >= GPU_ACTIVITY_UTILIZATION_THRESHOLD:
        state.active_sample_count += 1
    if gpu_sample.memory_used_bytes is not None:
        if state.first_memory_used_bytes is None:
            state.first_memory_used_bytes = gpu_sample.memory_used_bytes
        state.peak_memory_used_bytes = gpu_sample.memory_used_bytes if state.peak_memory_used_bytes is None else max(state.peak_memory_used_bytes, gpu_sample.memory_used_bytes)


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


def _sample_and_render_gpu_status(*, state: LiveRunMonitorState, prefix: str) -> str:
    gpu_sample = _sample_gpu_activity()
    _track_gpu_activity(state, gpu_sample)
    if gpu_sample.utilization_percent is None:
        return prefix + " | GPU util n/a"
    message = prefix + f" | GPU util {gpu_sample.utilization_percent}%"
    if gpu_sample.memory_used_bytes is not None and gpu_sample.memory_total_bytes is not None:
        message += f", mem {_format_bytes_gib(gpu_sample.memory_used_bytes)}/{_format_bytes_gib(gpu_sample.memory_total_bytes)}"
    return message


def _run_ncnn_command_with_heartbeat(
    *,
    command: list[str],
    tile_size: int,
    repeat_index: int,
) -> tuple[subprocess.CompletedProcess[str], float, dict[str, object]]:
    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    monitor_state = LiveRunMonitorState()
    last_reported_at = 0.0

    try:
        while process.poll() is None:
            now = time.time()
            if (now - last_reported_at) >= LIVE_PROGRESS_POLL_SECONDS:
                elapsed = time.perf_counter() - started
                _emit_live_status(
                    _sample_and_render_gpu_status(
                        state=monitor_state,
                        prefix=f"[tile {tile_size} run {repeat_index}] running NCNN benchmark | elapsed {elapsed:.1f}s",
                    )
                )
                last_reported_at = now
            time.sleep(0.1)

        stdout, stderr = process.communicate()
    finally:
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()

    wall_seconds = time.perf_counter() - started
    _emit_live_status(
        _sample_and_render_gpu_status(
            state=monitor_state,
            prefix=f"[tile {tile_size} run {repeat_index}] completed NCNN run in {wall_seconds:.2f}s",
        )
    )
    return (
        subprocess.CompletedProcess(command, process.returncode or 0, stdout, stderr),
        wall_seconds,
        _assess_gpu_activity(monitor_state),
    )


def _load_manifest(manifest_path: Path) -> dict[str, object]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _parse_tile_sizes(tile_sizes: str) -> list[int]:
    parsed = sorted({int(part.strip()) for part in tile_sizes.split(",") if part.strip()})
    if not parsed:
        raise ValueError("At least one tile size is required")
    return parsed


def _safe_images_per_second(frame_count: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return frame_count / seconds


def _safe_megapixels_per_second(frame_count: int, width: int, height: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    megapixels = (frame_count * width * height) / 1_000_000
    return megapixels / seconds


def _load_rgb_pixels(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _compute_psnr(reference_pixels: np.ndarray, candidate_pixels: np.ndarray) -> float:
    reference = reference_pixels.astype(np.float32)
    candidate = candidate_pixels.astype(np.float32)
    mse = float(np.mean((reference - candidate) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


def _compute_ssim(reference_pixels: np.ndarray, candidate_pixels: np.ndarray) -> float:
    reference = reference_pixels.astype(np.float32) / 255.0
    candidate = candidate_pixels.astype(np.float32) / 255.0
    c1 = 0.01**2
    c2 = 0.03**2

    ref_channels = reference.reshape(-1, reference.shape[2])
    cand_channels = candidate.reshape(-1, candidate.shape[2])
    ref_mean = ref_channels.mean(axis=0)
    cand_mean = cand_channels.mean(axis=0)
    ref_var = ((ref_channels - ref_mean) ** 2).mean(axis=0)
    cand_var = ((cand_channels - cand_mean) ** 2).mean(axis=0)
    covariance = ((ref_channels - ref_mean) * (cand_channels - cand_mean)).mean(axis=0)

    numerator = (2.0 * ref_mean * cand_mean + c1) * (2.0 * covariance + c2)
    denominator = (ref_mean**2 + cand_mean**2 + c1) * (ref_var + cand_var + c2)
    return float(np.clip(np.mean(numerator / denominator), 0.0, 1.0))


def _compare_frame_pair(reference_path: Path, candidate_path: Path) -> dict[str, float | str]:
    reference_pixels = _load_rgb_pixels(reference_path)
    candidate_pixels = _load_rgb_pixels(candidate_path)
    if reference_pixels.shape != candidate_pixels.shape:
        raise ValueError(
            f"Cannot compare frames with different shapes: {reference_path.name} {reference_pixels.shape} vs {candidate_pixels.shape}"
        )

    diff = reference_pixels.astype(np.int16) - candidate_pixels.astype(np.int16)
    abs_diff = np.abs(diff)
    return {
        "frame": reference_path.name,
        "meanAbsoluteError": round(float(abs_diff.mean()), 6),
        "rootMeanSquaredError": round(float(np.sqrt(np.mean(diff.astype(np.float32) ** 2))), 6),
        "maxAbsoluteError": int(abs_diff.max()),
        "psnr": round(_compute_psnr(reference_pixels, candidate_pixels), 6),
        "ssim": round(_compute_ssim(reference_pixels, candidate_pixels), 6),
    }


def _summarize_frame_metrics(frame_metrics: list[dict[str, float | str]]) -> dict[str, float | int] | None:
    if not frame_metrics:
        return None

    mae = [float(metric["meanAbsoluteError"]) for metric in frame_metrics]
    rmse = [float(metric["rootMeanSquaredError"]) for metric in frame_metrics]
    psnr = [float(metric["psnr"]) for metric in frame_metrics if np.isfinite(float(metric["psnr"]))]
    ssim = [float(metric["ssim"]) for metric in frame_metrics]
    max_abs = [int(metric["maxAbsoluteError"]) for metric in frame_metrics]
    return {
        "frameCount": len(frame_metrics),
        "averageMeanAbsoluteError": round(float(np.mean(mae)), 6),
        "averageRootMeanSquaredError": round(float(np.mean(rmse)), 6),
        "averagePsnr": round(float(np.mean(psnr)), 6) if psnr else float("inf"),
        "averageSsim": round(float(np.mean(ssim)), 6),
        "maxAbsoluteError": max(max_abs),
    }


def _render_pytorch_fixture_outputs(
    *,
    degraded_paths: list[Path],
    output_dir: Path,
    model_id: str,
    tile_size: int,
    gpu_id: int | None,
    precision: str,
    pytorch_runner: str = "torch",
) -> dict[str, object]:
    import torch

    from upscaler_worker.models.pytorch_sr import (
        _load_frame_batch,
        _run_descriptor,
        _save_frame_array,
        _tensor_to_pixel_batch,
        load_runtime_model,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    loaded_model = load_runtime_model(
        model_id,
        gpu_id,
        precision == "fp16",
        tile_size,
        log,
        bf16=precision == "bf16",
        precision=precision,
        pytorch_runner=pytorch_runner,
    )
    if loaded_model.device.type == "cuda":
        torch.cuda.synchronize(loaded_model.device)

    output_paths = [output_dir / frame_path.name for frame_path in degraded_paths]
    for batch_start in range(0, len(degraded_paths), loaded_model.frame_batch_size):
        input_batch = degraded_paths[batch_start:batch_start + loaded_model.frame_batch_size]
        output_batch = output_paths[batch_start:batch_start + loaded_model.frame_batch_size]
        frame_tensor = _load_frame_batch(input_batch, loaded_model.device, loaded_model.dtype, loaded_model.non_blocking)
        upscaled_tensor = _run_descriptor(
            loaded_model.descriptor,
            frame_tensor,
            tile_size=tile_size,
            scale=loaded_model.scale,
            autocast_dtype=loaded_model.autocast_dtype,
        )
        for frame_path, pixels in zip(output_batch, _tensor_to_pixel_batch(upscaled_tensor), strict=True):
            _save_frame_array(pixels, frame_path)
        del frame_tensor
        del upscaled_tensor

    return {
        "precision": precision,
        "outputPaths": [str(path) for path in output_paths],
        "modelRuntime": {
            "runner": loaded_model.runner,
            "precision": loaded_model.precision_mode,
            "dtype": str(loaded_model.dtype).replace("torch.", ""),
            "autocastDtype": str(loaded_model.autocast_dtype).replace("torch.", "") if loaded_model.autocast_dtype is not None else None,
            "frameBatchSize": loaded_model.frame_batch_size,
        },
        "notes": log,
    }


def compare_precision_quality(
    *,
    manifest_path: Path,
    model_id: str,
    tile_size: int,
    gpu_id: int | None,
    reference_precision: str = "fp32",
    candidate_precision: str = "bf16",
    max_frames: int | None = None,
    pytorch_runner: str = "torch",
) -> dict[str, object]:
    reference_precision = resolve_precision_mode(precision=reference_precision)
    candidate_precision = resolve_precision_mode(precision=candidate_precision)
    if reference_precision == candidate_precision:
        raise ValueError("reference and candidate precision must be different")

    ensure_runnable_model(model_id)
    backend_id = model_backend_id(model_id)
    if backend_id != "pytorch-image-sr":
        raise RuntimeError("Precision quality comparison is currently only supported for pytorch-image-sr models")

    manifest = _load_manifest(manifest_path)
    entries = list(manifest.get("entries", []))
    if not entries:
        raise ValueError(f"Benchmark manifest at {manifest_path} does not contain any entries")
    if max_frames is not None:
        entries = entries[:max_frames]

    degraded_paths = [Path(str(entry["degraded"])) for entry in entries]
    master_paths = [Path(str(entry["master"])) for entry in entries if "master" in entry]

    with tempfile.TemporaryDirectory(prefix="upscaler-precision-quality-") as temp_dir:
        root = Path(temp_dir)
        reference_render = _render_pytorch_fixture_outputs(
            degraded_paths=degraded_paths,
            output_dir=root / reference_precision,
            model_id=model_id,
            tile_size=tile_size,
            gpu_id=gpu_id,
            precision=reference_precision,
            pytorch_runner=pytorch_runner,
        )
        candidate_render = _render_pytorch_fixture_outputs(
            degraded_paths=degraded_paths,
            output_dir=root / candidate_precision,
            model_id=model_id,
            tile_size=tile_size,
            gpu_id=gpu_id,
            precision=candidate_precision,
            pytorch_runner=pytorch_runner,
        )

        reference_paths = [Path(path) for path in reference_render["outputPaths"]]
        candidate_paths = [Path(path) for path in candidate_render["outputPaths"]]
        precision_diff = [
            _compare_frame_pair(reference_path, candidate_path)
            for reference_path, candidate_path in zip(reference_paths, candidate_paths, strict=True)
        ]

        reference_vs_master = None
        candidate_vs_master = None
        if len(master_paths) == len(reference_paths):
            reference_vs_master = [
                _compare_frame_pair(master_path, reference_path)
                for master_path, reference_path in zip(master_paths, reference_paths, strict=True)
            ]
            candidate_vs_master = [
                _compare_frame_pair(master_path, candidate_path)
                for master_path, candidate_path in zip(master_paths, candidate_paths, strict=True)
            ]

    return {
        "manifestPath": str(manifest_path),
        "fixtureName": manifest.get("name", manifest_path.stem),
        "modelId": model_id,
        "modelLabel": model_label(model_id),
        "tileSize": tile_size,
        "frameCount": len(degraded_paths),
        "runner": pytorch_runner,
        "referencePrecision": reference_precision,
        "candidatePrecision": candidate_precision,
        "referenceModelRuntime": reference_render["modelRuntime"],
        "candidateModelRuntime": candidate_render["modelRuntime"],
        "referenceNotes": reference_render["notes"],
        "candidateNotes": candidate_render["notes"],
        "candidateVsReference": {
            "frameMetrics": precision_diff,
            "summary": _summarize_frame_metrics(precision_diff),
        },
        "referenceVsMaster": {
            "frameMetrics": reference_vs_master,
            "summary": _summarize_frame_metrics(reference_vs_master or []),
        },
        "candidateVsMaster": {
            "frameMetrics": candidate_vs_master,
            "summary": _summarize_frame_metrics(candidate_vs_master or []),
        },
    }


def benchmark_fixture(
    *,
    manifest_path: Path,
    model_id: str,
    tile_sizes: list[int],
    repeats: int,
    gpu_id: int | None,
    fp16: bool,
    bf16: bool = False,
    precision: str | None = None,
    pytorch_runner: str = "torch",
) -> dict[str, object]:
    precision_mode = resolve_precision_mode(fp16=fp16, bf16=bf16, precision=precision)
    ensure_runnable_model(model_id)
    manifest = _load_manifest(manifest_path)
    entries = list(manifest.get("entries", []))
    if not entries:
        raise ValueError(f"Benchmark manifest at {manifest_path} does not contain any entries")

    degraded_paths = [Path(str(entry["degraded"])) for entry in entries]
    width = int(manifest["degradedResolution"]["width"])
    height = int(manifest["degradedResolution"]["height"])
    backend_id = model_backend_id(model_id)

    results: list[dict[str, object]] = []
    for tile_size in tile_sizes:
        _emit_live_status(f"Benchmarking {model_id} tile={tile_size} backend={backend_id} repeats={repeats}")
        if backend_id == "pytorch-image-sr":
            results.append(
                _benchmark_pytorch_fixture(
                    degraded_paths=degraded_paths,
                    width=width,
                    height=height,
                    model_id=model_id,
                    tile_size=tile_size,
                    repeats=repeats,
                    gpu_id=gpu_id,
                    fp16=precision_mode == "fp16",
                    bf16=precision_mode == "bf16",
                    precision=precision_mode,
                    pytorch_runner=pytorch_runner,
                )
            )
        elif backend_id == "realesrgan-ncnn":
            results.append(
                _benchmark_ncnn_fixture(
                    degraded_paths=degraded_paths,
                    width=width,
                    height=height,
                    model_id=model_id,
                    tile_size=tile_size,
                    repeats=repeats,
                    gpu_id=gpu_id,
                )
            )
        else:
            raise RuntimeError(f"Unsupported benchmark backend '{backend_id}' for model '{model_id}'")

    return {
        "manifestPath": str(manifest_path),
        "fixtureName": manifest.get("name", manifest_path.stem),
        "frameCount": len(degraded_paths),
        "inputResolution": {"width": width, "height": height},
        "modelId": model_id,
        "modelLabel": model_label(model_id),
        "backendId": backend_id,
        "runner": pytorch_runner if backend_id == "pytorch-image-sr" else "external-executable",
        "repeats": repeats,
        "precision": precision_mode,
        "results": results,
    }


def _benchmark_pytorch_fixture(
    *,
    degraded_paths: list[Path],
    width: int,
    height: int,
    model_id: str,
    tile_size: int,
    repeats: int,
    gpu_id: int | None,
    fp16: bool,
    bf16: bool,
    precision: str,
    pytorch_runner: str,
) -> dict[str, object]:
    import torch

    from upscaler_worker.models.pytorch_sr import (
        _load_frame_batch,
        _run_descriptor,
        _save_frame_array,
        _tensor_to_pixel_batch,
        load_runtime_model,
    )

    log: list[str] = []
    load_started = time.perf_counter()
    _emit_live_status(f"Loading PyTorch model {model_id} runner={pytorch_runner} precision={precision} tile={tile_size}")
    loaded_model = load_runtime_model(
        model_id,
        gpu_id,
        fp16,
        tile_size,
        log,
        bf16=bf16,
        precision=precision,
        pytorch_runner=pytorch_runner,
    )
    if loaded_model.device.type == "cuda":
        torch.cuda.synchronize(loaded_model.device)
    load_seconds = time.perf_counter() - load_started

    repeat_metrics: list[dict[str, object]] = []
    for repeat_index in range(repeats):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            output_paths = [output_dir / frame_path.name for frame_path in degraded_paths]

            load_batch_seconds = 0.0
            inference_seconds = 0.0
            save_seconds = 0.0
            wall_started = time.perf_counter()
            last_reported_at = 0.0
            monitor_state = LiveRunMonitorState()
            _emit_live_status(f"[tile {tile_size} run {repeat_index + 1}/{repeats}] starting PyTorch fixture benchmark")

            if loaded_model.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(loaded_model.device)

            for batch_start in range(0, len(degraded_paths), loaded_model.frame_batch_size):
                input_batch = degraded_paths[batch_start:batch_start + loaded_model.frame_batch_size]
                output_batch = output_paths[batch_start:batch_start + loaded_model.frame_batch_size]
                now = time.time()
                if batch_start == 0 or (now - last_reported_at) >= LIVE_PROGRESS_POLL_SECONDS:
                    processed = min(batch_start, len(degraded_paths))
                    progress_prefix = f"[tile {tile_size} run {repeat_index + 1}] batching | frames {processed}/{len(degraded_paths)}"
                    _emit_live_status(_sample_and_render_gpu_status(state=monitor_state, prefix=progress_prefix))
                    last_reported_at = now

                batch_load_started = time.perf_counter()
                frame_tensor = _load_frame_batch(input_batch, loaded_model.device, loaded_model.dtype, loaded_model.non_blocking)
                if loaded_model.device.type == "cuda":
                    torch.cuda.synchronize(loaded_model.device)
                load_batch_seconds += time.perf_counter() - batch_load_started

                infer_started = time.perf_counter()
                upscaled_tensor = _run_descriptor(
                    loaded_model.descriptor,
                    frame_tensor,
                    tile_size=tile_size,
                    scale=loaded_model.scale,
                    autocast_dtype=loaded_model.autocast_dtype,
                )
                if loaded_model.device.type == "cuda":
                    torch.cuda.synchronize(loaded_model.device)
                inference_seconds += time.perf_counter() - infer_started

                save_started = time.perf_counter()
                for frame_path, pixels in zip(output_batch, _tensor_to_pixel_batch(upscaled_tensor), strict=True):
                    _save_frame_array(pixels, frame_path)
                save_seconds += time.perf_counter() - save_started

                del frame_tensor
                del upscaled_tensor

                now = time.time()
                if (now - last_reported_at) >= LIVE_PROGRESS_HEARTBEAT_SECONDS or (batch_start + len(input_batch)) >= len(degraded_paths):
                    processed = min(batch_start + len(input_batch), len(degraded_paths))
                    progress_prefix = f"[tile {tile_size} run {repeat_index + 1}] completed batch | frames {processed}/{len(degraded_paths)}"
                    _emit_live_status(_sample_and_render_gpu_status(state=monitor_state, prefix=progress_prefix))
                    last_reported_at = now

            if loaded_model.device.type == "cuda":
                torch.cuda.synchronize(loaded_model.device)
                peak_allocated_mb = torch.cuda.max_memory_allocated(loaded_model.device) / (1024 * 1024)
                peak_reserved_mb = torch.cuda.max_memory_reserved(loaded_model.device) / (1024 * 1024)
            else:
                peak_allocated_mb = 0.0
                peak_reserved_mb = 0.0

            wall_seconds = time.perf_counter() - wall_started
            gpu_activity = _assess_gpu_activity(monitor_state)
            if gpu_activity["warning"]:
                _emit_live_status(f"[tile {tile_size} run {repeat_index + 1}] WARNING | {gpu_activity['warning']}")
            repeat_metrics.append(
                {
                    "repeat": repeat_index + 1,
                    "wallSeconds": wall_seconds,
                    "batchLoadSeconds": load_batch_seconds,
                    "inferenceSeconds": inference_seconds,
                    "saveSeconds": save_seconds,
                    "imagesPerSecond": _safe_images_per_second(len(degraded_paths), wall_seconds),
                    "megapixelsPerSecond": _safe_megapixels_per_second(len(degraded_paths), width, height, wall_seconds),
                    "peakAllocatedMb": round(peak_allocated_mb, 2),
                    "peakReservedMb": round(peak_reserved_mb, 2),
                    "gpuActivity": gpu_activity,
                }
            )

    best_wall = min(metric["wallSeconds"] for metric in repeat_metrics)
    avg_wall = sum(metric["wallSeconds"] for metric in repeat_metrics) / len(repeat_metrics)
    avg_infer = sum(metric["inferenceSeconds"] for metric in repeat_metrics) / len(repeat_metrics)
    avg_load = sum(metric["batchLoadSeconds"] for metric in repeat_metrics) / len(repeat_metrics)
    avg_save = sum(metric["saveSeconds"] for metric in repeat_metrics) / len(repeat_metrics)

    return {
        "tileSize": tile_size,
        "modelLoadSeconds": load_seconds,
        "runner": loaded_model.runner,
        "frameBatchSize": loaded_model.frame_batch_size,
        "dtype": str(loaded_model.dtype).replace("torch.", ""),
        "device": str(loaded_model.device),
        "repeatMetrics": repeat_metrics,
        "summary": {
            "bestWallSeconds": round(best_wall, 4),
            "averageWallSeconds": round(avg_wall, 4),
            "averageInferenceSeconds": round(avg_infer, 4),
            "averageBatchLoadSeconds": round(avg_load, 4),
            "averageSaveSeconds": round(avg_save, 4),
            "bestImagesPerSecond": round(_safe_images_per_second(len(degraded_paths), best_wall), 3),
            "bestMegapixelsPerSecond": round(_safe_megapixels_per_second(len(degraded_paths), width, height, best_wall), 3),
        },
        "notes": log,
    }


def _benchmark_ncnn_fixture(
    *,
    degraded_paths: list[Path],
    width: int,
    height: int,
    model_id: str,
    tile_size: int,
    repeats: int,
    gpu_id: int | None,
) -> dict[str, object]:
    runtime = ensure_runtime_assets()
    input_dir = degraded_paths[0].parent
    repeat_metrics: list[dict[str, object]] = []

    for repeat_index in range(repeats):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "out"
            output_dir.mkdir(parents=True, exist_ok=True)
            command = [
                str(runtime["realesrganPath"]),
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-m",
                str(runtime["modelDir"]),
                "-n",
                model_id,
                "-f",
                "png",
            ]
            if gpu_id is not None:
                command.extend(["-g", str(gpu_id)])
            if tile_size >= 0:
                command.extend(["-t", str(tile_size)])

            _emit_live_status(f"[tile {tile_size} run {repeat_index + 1}/{repeats}] starting NCNN fixture benchmark")
            completed, wall_seconds, gpu_activity = _run_ncnn_command_with_heartbeat(
                command=command,
                tile_size=tile_size,
                repeat_index=repeat_index + 1,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"NCNN benchmark failed for tile size {tile_size}. stdout: {completed.stdout} stderr: {completed.stderr}"
                )
            if gpu_activity["warning"]:
                _emit_live_status(f"[tile {tile_size} run {repeat_index + 1}] WARNING | {gpu_activity['warning']}")

            repeat_metrics.append(
                {
                    "repeat": repeat_index + 1,
                    "wallSeconds": wall_seconds,
                    "imagesPerSecond": _safe_images_per_second(len(degraded_paths), wall_seconds),
                    "megapixelsPerSecond": _safe_megapixels_per_second(len(degraded_paths), width, height, wall_seconds),
                    "gpuActivity": gpu_activity,
                }
            )

    best_wall = min(metric["wallSeconds"] for metric in repeat_metrics)
    avg_wall = sum(metric["wallSeconds"] for metric in repeat_metrics) / len(repeat_metrics)
    return {
        "tileSize": tile_size,
        "frameBatchSize": 1,
        "dtype": "ncnn",
        "device": f"gpu:{gpu_id}" if gpu_id is not None else "default",
        "repeatMetrics": repeat_metrics,
        "summary": {
            "bestWallSeconds": round(best_wall, 4),
            "averageWallSeconds": round(avg_wall, 4),
            "bestImagesPerSecond": round(_safe_images_per_second(len(degraded_paths), best_wall), 3),
            "bestMegapixelsPerSecond": round(_safe_megapixels_per_second(len(degraded_paths), width, height, best_wall), 3),
        },
        "notes": [],
    }


__all__ = ["benchmark_fixture", "_parse_tile_sizes"]