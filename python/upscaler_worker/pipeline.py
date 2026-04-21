from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from upscaler_worker.cancellation import JobCancelledError, cancellation_requested, ensure_not_cancelled, terminate_process, terminate_process_tree, wait_if_paused
from upscaler_worker.interpolation import (
    build_rife_command,
    resolve_output_fps,
    resolve_segment_output_frame_count,
    should_skip_interpolation,
    validate_interpolation_request,
)
from upscaler_worker.media import probe_video
from upscaler_worker.model_catalog import ensure_runnable_model, model_backend_id, model_label, model_task
from upscaler_worker.models.pytorch_video_sr import build_external_video_sr_command, validate_external_video_sr_outputs
from upscaler_worker.precision import resolve_precision_mode
from upscaler_worker.models.realesrgan import model_label
from upscaler_worker.runtime import ensure_rife_runtime, ensure_runtime_assets, repo_root
from upscaler_worker.video_encoding import VideoEncoderConfig, probe_video_encoder, resolve_video_encoder_config


BATCH_FRAME_COUNT = 12
PIPELINE_SEGMENT_FRAME_LIMIT = 48
PIPELINE_SEGMENT_TARGET_SECONDS = 10.0
PIPELINE_STAGE_QUEUE_DEPTH = 2
PIPELINE_INTERMEDIATE_CONTAINER = "mkv"
PYTORCH_EXECUTION_PATH_FILE_IO = "file-io"
PYTORCH_EXECUTION_PATH_STREAMING = "streaming"
SUPPORTED_PYTORCH_EXECUTION_PATHS = {
    PYTORCH_EXECUTION_PATH_FILE_IO,
    PYTORCH_EXECUTION_PATH_STREAMING,
}


@dataclass(frozen=True)
class PipelineSegment:
    index: int
    start_frame: int
    frame_count: int
    start_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class InterpolationSegmentPlan:
    index: int
    source_start_frame: int
    source_frame_count: int
    expanded_start_frame: int
    expanded_frame_count: int
    overlap_before_frames: int
    overlap_after_frames: int
    output_start_frame: int
    output_frame_count: int
    expanded_output_frame_count: int
    expanded_start_seconds: float
    expanded_duration_seconds: float


@dataclass(frozen=True)
class StreamingFrameBatch:
    batch_index: int
    frames: list[np.ndarray]


@dataclass(frozen=True)
class StreamingPixelBatch:
    batch_index: int
    pixels: list[np.ndarray]


@dataclass(frozen=True)
class InterpolationEncodeTask:
    segment_index: int
    output_dir: Path
    output_file: Path
    frame_start_number: int
    frame_limit: int
    segment_total_frames: int
    extracted_frames: int
    upscaled_frames: int
    interpolated_frames: int
    cleanup_paths: tuple[Path, ...]


@dataclass
class PipelineProgressState:
    extracted_frames: int = 0
    colorized_frames: int = 0
    upscaled_frames: int = 0
    interpolated_frames: int = 0
    encoded_frames: int = 0
    remuxed_frames: int = 0


def _should_publish_stage_progress(phase: str, progress_state: PipelineProgressState) -> bool:
    if phase != "extracting":
        return True

    return (
        progress_state.upscaled_frames <= 0
        and progress_state.interpolated_frames <= 0
        and progress_state.encoded_frames <= 0
        and progress_state.remuxed_frames <= 0
    )


@dataclass
class PipelineTelemetryResources:
    sampled_at: float = 0.0
    last_processed_frames: int = 0
    process_rss_bytes: int | None = None
    gpu_memory_used_bytes: int | None = None
    gpu_memory_total_bytes: int | None = None
    scratch_size_bytes: int = 0
    output_size_bytes: int = 0
    rolling_frames_per_second: float | None = None
    peak_process_rss_bytes: int | None = None
    peak_gpu_memory_used_bytes: int | None = None
    peak_scratch_size_bytes: int = 0
    peak_output_size_bytes: int = 0


@dataclass
class PipelineTelemetryState:
    started_at: float
    source_path: str
    scratch_path: Path
    output_path: Path
    job_id: str | None = None
    resources: PipelineTelemetryResources = field(default_factory=PipelineTelemetryResources)
    segment_index: int | None = None
    segment_count: int | None = None
    segment_processed_frames: int | None = None
    segment_total_frames: int | None = None
    batch_index: int | None = None
    batch_count: int | None = None
    extract_stage_seconds: float = 0.0
    colorize_stage_seconds: float = 0.0
    upscale_stage_seconds: float = 0.0
    interpolate_stage_seconds: float = 0.0
    encode_stage_seconds: float = 0.0
    remux_stage_seconds: float = 0.0


def _round_dimension(value: float) -> int:
    rounded = max(2, int(round(value)))
    return rounded if rounded % 2 == 0 else rounded + 1


def _probe_video_encoder(ffmpeg: str, config: VideoEncoderConfig) -> bool:
    return probe_video_encoder(ffmpeg, config)


def _resolve_video_encoder_config(
    *,
    ffmpeg: str,
    runtime: dict[str, object],
    gpu_id: int | None,
    codec: str,
    crf: int,
    log: list[str],
) -> VideoEncoderConfig:
    return resolve_video_encoder_config(
        ffmpeg=ffmpeg,
        runtime=runtime,
        gpu_id=gpu_id,
        codec=codec,
        crf=crf,
        log=log,
        probe_encoder=_probe_video_encoder,
    )


def _clamp_ratio_position(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp_crop_rect(
    crop_left: float | None,
    crop_top: float | None,
    crop_width: float | None,
    crop_height: float | None,
) -> tuple[float, float, float, float] | None:
    if crop_left is None or crop_top is None or crop_width is None or crop_height is None:
        return None

    width = _clamp_ratio_position(crop_width)
    height = _clamp_ratio_position(crop_height)
    left = min(max(0.0, crop_left), max(0.0, 1.0 - width))
    top = min(max(0.0, crop_top), max(0.0, 1.0 - height))
    return left, top, width, height


def _resolve_aspect_ratio(
    source_width: int,
    source_height: int,
    aspect_ratio_preset: str,
    custom_aspect_width: int | None,
    custom_aspect_height: int | None,
) -> float:
    if aspect_ratio_preset == "source":
        return source_width / source_height

    if aspect_ratio_preset == "custom":
        if custom_aspect_width and custom_aspect_height:
            return custom_aspect_width / custom_aspect_height
        return source_width / source_height

    try:
        width_text, height_text = aspect_ratio_preset.split(":", maxsplit=1)
        width = float(width_text)
        height = float(height_text)
    except ValueError:
        return source_width / source_height

    if width <= 0 or height <= 0:
        return source_width / source_height

    return width / height


def _resolve_output_dimensions(
    *,
    source_width: int,
    source_height: int,
    output_mode: str,
    aspect_ratio_preset: str,
    custom_aspect_width: int | None,
    custom_aspect_height: int | None,
    resolution_basis: str,
    target_width: int | None,
    target_height: int | None,
) -> tuple[int, int, float]:
    aspect_ratio = _resolve_aspect_ratio(
        source_width,
        source_height,
        aspect_ratio_preset,
        custom_aspect_width,
        custom_aspect_height,
    )

    if resolution_basis == "exact" and target_width and target_height:
        return _round_dimension(target_width), _round_dimension(target_height), aspect_ratio

    if resolution_basis == "width" and target_width:
        return _round_dimension(target_width), _round_dimension(target_width / aspect_ratio), aspect_ratio

    if resolution_basis == "height" and target_height:
        return _round_dimension(target_height * aspect_ratio), _round_dimension(target_height), aspect_ratio

    if target_width and not target_height:
        return _round_dimension(target_width), _round_dimension(target_width / aspect_ratio), aspect_ratio

    if target_height and not target_width:
        return _round_dimension(target_height * aspect_ratio), _round_dimension(target_height), aspect_ratio

    source_aspect = source_width / source_height
    if output_mode == "native4x" and abs(aspect_ratio - source_aspect) < 0.0001:
        return _round_dimension(source_width * 2), _round_dimension(source_height * 2), aspect_ratio

    if aspect_ratio >= 1:
        return 3840, _round_dimension(3840 / aspect_ratio), aspect_ratio

    return _round_dimension(2160 * aspect_ratio), 2160, aspect_ratio


def _run(
    command: list[str],
    log: list[str],
    cancel_path: str | None = None,
    pause_path: str | None = None,
    on_pause=None,
    on_resume=None,
    env: dict[str, str] | None = None,
) -> None:
    log.append("$ " + " ".join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
    output_lines: list[str] = []
    output_lock = threading.Lock()

    def drain_stream(stream: object) -> None:
        if stream is None:
            return
        try:
            for line in stream:
                text = line.rstrip()
                if text:
                    with output_lock:
                        output_lines.append(text)
        finally:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    stdout_thread = threading.Thread(target=drain_stream, args=(process.stdout,), daemon=True)
    stderr_thread = threading.Thread(target=drain_stream, args=(process.stderr,), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    try:
        while process.poll() is None:
            wait_if_paused(
                pause_path,
                cancel_path=cancel_path,
                process=process,
                on_pause=on_pause,
                on_resume=on_resume,
            )
            if cancellation_requested(cancel_path):
                terminate_process(process)
                raise JobCancelledError("Job cancelled by user")
            time.sleep(0.1)
    except BaseException:
        terminate_process_tree(process)
        raise
    finally:
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

    if output_lines:
        log.append("\n".join(output_lines))
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")


def _resolve_pytorch_execution_path(model_id: str, requested_path: str | None) -> str | None:
    if model_backend_id(model_id) != "pytorch-image-sr":
        return None

    configured_path = requested_path or os.environ.get("UPSCALER_PYTORCH_EXECUTION_PATH") or PYTORCH_EXECUTION_PATH_FILE_IO
    resolved_path = configured_path.strip().lower()
    if resolved_path == "auto":
        resolved_path = PYTORCH_EXECUTION_PATH_FILE_IO

    if resolved_path not in SUPPORTED_PYTORCH_EXECUTION_PATHS:
        supported = ", ".join(sorted(SUPPORTED_PYTORCH_EXECUTION_PATHS))
        raise ValueError(f"Unsupported PyTorch execution path '{configured_path}'. Expected one of: {supported}")

    return resolved_path


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0

    total = 0
    try:
        for root, _, files in os.walk(path):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


def _current_process_rss_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
            get_process_memory_info.restype = wintypes.BOOL
            success = get_process_memory_info(handle, ctypes.byref(counters), counters.cb)
            if success:
                return int(counters.WorkingSetSize)
        except Exception:  # noqa: BLE001
            return None
        return None

    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        if os.uname().sysname == "Darwin":
            return int(usage.ru_maxrss)
        return int(usage.ru_maxrss * 1024)
    except Exception:  # noqa: BLE001
        return None


def _sample_gpu_memory_bytes() -> tuple[int | None, int | None]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None, None

    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None

    if completed.returncode != 0:
        return None, None

    used_mb = 0
    total_mb = 0
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            used_text, total_text = [part.strip() for part in line.split(",", maxsplit=1)]
            used_mb += int(float(used_text))
            total_mb += int(float(total_text))
        except ValueError:
            continue

    if total_mb <= 0:
        return None, None
    return used_mb * 1024 * 1024, total_mb * 1024 * 1024


def _sample_progress_telemetry(
    telemetry_state: PipelineTelemetryState,
    *,
    processed_frames: int,
    total_frames: int,
) -> dict[str, float | int | None]:
    now = time.time()
    elapsed_seconds = max(0.0, now - telemetry_state.started_at)
    average_fps = processed_frames / elapsed_seconds if elapsed_seconds > 0 and processed_frames > 0 else None
    estimated_remaining_seconds = None
    if average_fps and average_fps > 0 and total_frames > processed_frames:
        estimated_remaining_seconds = (total_frames - processed_frames) / average_fps

    resources = telemetry_state.resources
    if now - resources.sampled_at >= 1.0 or resources.sampled_at <= 0:
        sample_window = now - resources.sampled_at if resources.sampled_at > 0 else 0.0
        if sample_window > 0:
            frame_delta = max(0, processed_frames - resources.last_processed_frames)
            resources.rolling_frames_per_second = frame_delta / sample_window if frame_delta > 0 else 0.0
        resources.sampled_at = now
        resources.last_processed_frames = processed_frames
        resources.process_rss_bytes = _current_process_rss_bytes()
        resources.gpu_memory_used_bytes, resources.gpu_memory_total_bytes = _sample_gpu_memory_bytes()
        resources.scratch_size_bytes = _path_size_bytes(telemetry_state.scratch_path)
        resources.output_size_bytes = _path_size_bytes(telemetry_state.output_path)
        if resources.process_rss_bytes is not None:
            if resources.peak_process_rss_bytes is None:
                resources.peak_process_rss_bytes = resources.process_rss_bytes
            else:
                resources.peak_process_rss_bytes = max(resources.peak_process_rss_bytes, resources.process_rss_bytes)
        if resources.gpu_memory_used_bytes is not None:
            if resources.peak_gpu_memory_used_bytes is None:
                resources.peak_gpu_memory_used_bytes = resources.gpu_memory_used_bytes
            else:
                resources.peak_gpu_memory_used_bytes = max(resources.peak_gpu_memory_used_bytes, resources.gpu_memory_used_bytes)
        resources.peak_scratch_size_bytes = max(resources.peak_scratch_size_bytes, resources.scratch_size_bytes)
        resources.peak_output_size_bytes = max(resources.peak_output_size_bytes, resources.output_size_bytes)

    return {
        "segmentIndex": telemetry_state.segment_index,
        "segmentCount": telemetry_state.segment_count,
        "segmentProcessedFrames": telemetry_state.segment_processed_frames,
        "segmentTotalFrames": telemetry_state.segment_total_frames,
        "batchIndex": telemetry_state.batch_index,
        "batchCount": telemetry_state.batch_count,
        "elapsedSeconds": elapsed_seconds,
        "averageFramesPerSecond": average_fps,
        "rollingFramesPerSecond": resources.rolling_frames_per_second,
        "estimatedRemainingSeconds": estimated_remaining_seconds,
        "processRssBytes": resources.process_rss_bytes,
        "gpuMemoryUsedBytes": resources.gpu_memory_used_bytes,
        "gpuMemoryTotalBytes": resources.gpu_memory_total_bytes,
        "scratchSizeBytes": resources.scratch_size_bytes,
        "outputSizeBytes": resources.output_size_bytes,
        "extractStageSeconds": telemetry_state.extract_stage_seconds,
        "colorizeStageSeconds": telemetry_state.colorize_stage_seconds,
        "upscaleStageSeconds": telemetry_state.upscale_stage_seconds,
        "interpolateStageSeconds": telemetry_state.interpolate_stage_seconds,
        "encodeStageSeconds": telemetry_state.encode_stage_seconds,
        "remuxStageSeconds": telemetry_state.remux_stage_seconds,
    }


def _record_stage_duration(telemetry_state: PipelineTelemetryState, stage: str, duration_seconds: float) -> None:
    if duration_seconds <= 0:
        return
    if stage == "extract":
        telemetry_state.extract_stage_seconds += duration_seconds
        return
    if stage == "colorize":
        telemetry_state.colorize_stage_seconds += duration_seconds
        return
    if stage == "upscale":
        telemetry_state.upscale_stage_seconds += duration_seconds
        return
    if stage == "interpolate":
        telemetry_state.interpolate_stage_seconds += duration_seconds
        return
    if stage == "encode":
        telemetry_state.encode_stage_seconds += duration_seconds
        return
    if stage == "remux":
        telemetry_state.remux_stage_seconds += duration_seconds


def _set_segment_progress(
    telemetry_state: PipelineTelemetryState,
    *,
    segment_index: int | None,
    segment_count: int | None,
    segment_processed_frames: int | None,
    segment_total_frames: int | None,
    batch_index: int | None = None,
    batch_count: int | None = None,
) -> None:
    telemetry_state.segment_index = segment_index
    telemetry_state.segment_count = segment_count
    telemetry_state.segment_processed_frames = segment_processed_frames
    telemetry_state.segment_total_frames = segment_total_frames
    telemetry_state.batch_index = batch_index
    telemetry_state.batch_count = batch_count


def _write_progress(
    progress_path: str | None,
    *,
    phase: str,
    percent: int,
    message: str,
    processed_frames: int,
    total_frames: int,
    extracted_frames: int = 0,
    colorized_frames: int = 0,
    upscaled_frames: int = 0,
    interpolated_frames: int = 0,
    encoded_frames: int = 0,
    remuxed_frames: int = 0,
    telemetry_state: PipelineTelemetryState | None = None,
) -> None:
    if not progress_path:
        return

    target = Path(progress_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase,
        "percent": percent,
        "message": message,
        "processedFrames": processed_frames,
        "totalFrames": total_frames,
        "extractedFrames": extracted_frames,
        "colorizedFrames": colorized_frames,
        "upscaledFrames": upscaled_frames,
        "interpolatedFrames": interpolated_frames,
        "encodedFrames": encoded_frames,
        "remuxedFrames": remuxed_frames,
    }
    if telemetry_state is not None:
        payload.update(
            {
                "jobId": telemetry_state.job_id,
                "sourcePath": telemetry_state.source_path,
                "scratchPath": str(telemetry_state.scratch_path),
                "outputPath": str(telemetry_state.output_path),
            }
        )
        payload.update(_sample_progress_telemetry(telemetry_state, processed_frames=processed_frames, total_frames=total_frames))
    target.write_text(json.dumps(payload), encoding="utf-8")


def _run_ffmpeg_with_frame_progress(
    command: list[str],
    log: list[str],
    progress_path: str | None,
    cancel_path: str | None,
    pause_path: str | None,
    *,
    phase: str,
    percent_base: int,
    percent_span: int,
    total_frames: int,
    message_prefix: str,
    extracted_frames: int,
    colorized_frames: int,
    upscaled_frames: int,
    interpolated_frames: int,
    encoded_frames: int,
    remuxed_frames: int,
    telemetry_state: PipelineTelemetryState | None,
    stage_frame_offset: int = 0,
) -> tuple[int, int]:
    progress_command = command[:-1] + ["-progress", "pipe:1", "-nostats", command[-1]]
    log.append("$ " + " ".join(progress_command))
    process = subprocess.Popen(
        progress_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    current_frame = 0
    output_lines: list[str] = []
    state = {"current_frame": 0}

    def reader() -> None:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            output_lines.append(line)
            if not line.startswith("frame="):
                continue
            try:
                frame_value = int(line.split("=", maxsplit=1)[1])
            except ValueError:
                continue

            state["current_frame"] = frame_value
            stage_progress_frames = min(total_frames, stage_frame_offset + frame_value)
            percent = percent_base if total_frames <= 0 else min(percent_base + percent_span, percent_base + int((stage_progress_frames / max(total_frames, 1)) * percent_span))
            _write_progress(
                progress_path,
                phase=phase,
                percent=percent,
                message=f"{message_prefix} ({stage_progress_frames}/{total_frames})",
                processed_frames=stage_progress_frames,
                total_frames=total_frames,
                extracted_frames=extracted_frames,
                colorized_frames=colorized_frames,
                upscaled_frames=upscaled_frames,
                interpolated_frames=interpolated_frames,
                encoded_frames=stage_progress_frames if phase == "encoding" else encoded_frames,
                remuxed_frames=stage_progress_frames if phase == "remuxing" else remuxed_frames,
                telemetry_state=telemetry_state,
            )

    reader_thread = threading.Thread(target=reader, name="ffmpeg-frame-progress-reader", daemon=True)
    reader_thread.start()

    try:
        while process.poll() is None:
            wait_if_paused(
                pause_path,
                cancel_path=cancel_path,
                process=process,
                on_pause=lambda: _write_progress(
                    progress_path,
                    phase="paused",
                    percent=percent_base if total_frames <= 0 else min(percent_base + percent_span, percent_base + int((min(total_frames, stage_frame_offset + state["current_frame"]) / max(total_frames, 1)) * percent_span)),
                    message=f"Paused: {message_prefix} ({min(total_frames, stage_frame_offset + state['current_frame'])}/{total_frames})",
                    processed_frames=min(total_frames, stage_frame_offset + state["current_frame"]),
                    total_frames=total_frames,
                    extracted_frames=extracted_frames,
                    colorized_frames=colorized_frames,
                    upscaled_frames=upscaled_frames,
                    interpolated_frames=interpolated_frames,
                    encoded_frames=min(total_frames, stage_frame_offset + state["current_frame"]) if phase == "encoding" else encoded_frames,
                    remuxed_frames=min(total_frames, stage_frame_offset + state["current_frame"]) if phase == "remuxing" else remuxed_frames,
                    telemetry_state=telemetry_state,
                ),
                on_resume=lambda: _write_progress(
                    progress_path,
                    phase=phase,
                    percent=percent_base if total_frames <= 0 else min(percent_base + percent_span, percent_base + int((min(total_frames, stage_frame_offset + state["current_frame"]) / max(total_frames, 1)) * percent_span)),
                    message=f"Resumed: {message_prefix} ({min(total_frames, stage_frame_offset + state['current_frame'])}/{total_frames})",
                    processed_frames=min(total_frames, stage_frame_offset + state["current_frame"]),
                    total_frames=total_frames,
                    extracted_frames=extracted_frames,
                    colorized_frames=colorized_frames,
                    upscaled_frames=upscaled_frames,
                    interpolated_frames=interpolated_frames,
                    encoded_frames=min(total_frames, stage_frame_offset + state["current_frame"]) if phase == "encoding" else encoded_frames,
                    remuxed_frames=min(total_frames, stage_frame_offset + state["current_frame"]) if phase == "remuxing" else remuxed_frames,
                    telemetry_state=telemetry_state,
                ),
            )
            if cancellation_requested(cancel_path):
                terminate_process(process)
                raise JobCancelledError("Job cancelled by user")
            reader_thread.join(timeout=0.1)
        reader_thread.join(timeout=1)
    except BaseException:
        terminate_process_tree(process)
        raise
    finally:
        if process.stdout:
            process.stdout.close()

    return_code = process.wait()
    if output_lines:
        log.append("\n".join(output_lines))
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(progress_command)}")
    current_frame = state["current_frame"]
    return current_frame, total_frames


def _run_realesrgan_batch(
    command: list[str],
    log: list[str],
    cancel_path: str | None = None,
    pause_path: str | None = None,
) -> None:
    _run(command, log, cancel_path, pause_path)


def _run_command_with_output_frame_progress(
    *,
    command: list[str],
    output_dir: Path,
    target_frame_count: int,
    log: list[str],
    cancel_path: str | None,
    pause_path: str | None,
    progress_callback=None,
    env: dict[str, str] | None = None,
) -> int:
    log.append("$ " + " ".join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
    output_lines: list[str] = []
    output_lock = threading.Lock()

    def drain_stream(stream: object) -> None:
        if stream is None:
            return
        try:
            for line in stream:
                text = line.rstrip()
                if text:
                    with output_lock:
                        output_lines.append(text)
        finally:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    stdout_thread = threading.Thread(target=drain_stream, args=(process.stdout,), daemon=True)
    stderr_thread = threading.Thread(target=drain_stream, args=(process.stderr,), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    last_reported_count = -1
    try:
        while process.poll() is None:
            wait_if_paused(pause_path, cancel_path=cancel_path, process=process)
            if cancellation_requested(cancel_path):
                terminate_process(process)
                raise JobCancelledError("Job cancelled by user")
            current_count = len(list(output_dir.glob("frame_*.png")))
            if progress_callback is not None and current_count != last_reported_count:
                progress_callback(current_count, target_frame_count)
                last_reported_count = current_count
            time.sleep(0.25)
    except BaseException:
        terminate_process_tree(process)
        raise
    finally:
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

    final_count = len(list(output_dir.glob("frame_*.png")))
    if progress_callback is not None and final_count != last_reported_count:
        progress_callback(final_count, target_frame_count)
    if output_lines:
        log.append("\n".join(output_lines))
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")
    return final_count


def _run_rife_segment(
    *,
    runtime: dict[str, object],
    input_dir: Path,
    output_dir: Path,
    target_frame_count: int,
    gpu_id: int | None,
    width: int,
    height: int,
    log: list[str],
    cancel_path: str | None,
    progress_callback=None,
) -> int:
    input_frames = sorted(input_dir.glob("frame_*.png"))
    if not input_frames:
        raise RuntimeError("No frames were available for interpolation.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rife_command = build_rife_command(
        executable_path=str(runtime["rifePath"]),
        model_root=str(runtime["rifeModelRoot"]),
        input_dir=input_dir,
        output_dir=output_dir,
        target_frame_count=target_frame_count,
        gpu_id=gpu_id,
        uhd_mode=width >= 1920 or height >= 1080,
    )
    log.append("$ " + " ".join(rife_command))
    process = subprocess.Popen(rife_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    output_lines: list[str] = []
    output_lock = threading.Lock()

    def drain_stream(stream: object) -> None:
        if stream is None:
            return
        try:
            for line in stream:
                text = line.rstrip()
                if text:
                    with output_lock:
                        output_lines.append(text)
        finally:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    stdout_thread = threading.Thread(target=drain_stream, args=(process.stdout,), daemon=True)
    stderr_thread = threading.Thread(target=drain_stream, args=(process.stderr,), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    last_reported_count = -1
    try:
        while process.poll() is None:
            if cancellation_requested(cancel_path):
                terminate_process(process)
                raise JobCancelledError("Job cancelled by user")
            current_count = len(list(output_dir.glob("frame_*.png")))
            if progress_callback is not None and current_count != last_reported_count:
                progress_callback(current_count, target_frame_count)
                last_reported_count = current_count
            time.sleep(0.25)
    finally:
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

    final_count = len(list(output_dir.glob("frame_*.png")))
    if progress_callback is not None and final_count != last_reported_count:
        progress_callback(final_count, target_frame_count)
    if output_lines:
        log.append("\n".join(output_lines))
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(rife_command)}")
    if final_count <= 0:
        raise RuntimeError("Interpolation completed without producing output frames")
    return final_count


def _read_exact_bytes(stream, expected_size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = expected_size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _append_process_stderr(log: list[str], process: subprocess.Popen[bytes], label: str) -> None:
    if process.stderr is None:
        return
    stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
    if stderr:
        log.append(f"[{label}] {stderr}")


def _write_raw_rgb_frame(stream, pixels: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(pixels, dtype=np.uint8)
    stream.write(contiguous.tobytes())


def _segment_frame_limit(fps: float, segment_duration_seconds: float | None = None) -> int:
    resolved_segment_seconds = (
        segment_duration_seconds
        if segment_duration_seconds is not None and segment_duration_seconds > 0
        else PIPELINE_SEGMENT_TARGET_SECONDS
    )
    if fps <= 0:
        return PIPELINE_SEGMENT_FRAME_LIMIT
    return max(PIPELINE_SEGMENT_FRAME_LIMIT, int(round(fps * resolved_segment_seconds)))


def _plan_pipeline_segments(
    total_frames: int,
    fps: float,
    *,
    force_single_segment: bool = False,
    segment_duration_seconds: float | None = None,
) -> list[PipelineSegment]:
    if total_frames <= 0:
        return []

    effective_fps = fps if fps > 0 else 1.0
    if force_single_segment:
        segment_frame_count = total_frames
    else:
        segment_frame_count = max(1, min(_segment_frame_limit(effective_fps, segment_duration_seconds), total_frames))
    segments: list[PipelineSegment] = []
    for index, start_frame in enumerate(range(0, total_frames, segment_frame_count)):
        frame_count = min(segment_frame_count, total_frames - start_frame)
        segments.append(
            PipelineSegment(
                index=index,
                start_frame=start_frame,
                frame_count=frame_count,
                start_seconds=start_frame / effective_fps,
                duration_seconds=frame_count / effective_fps,
            )
        )
    return segments


def _plan_interpolation_segments(
    source_segments: list[PipelineSegment],
    *,
    total_source_frames: int,
    source_fps: float,
    output_fps: float,
) -> list[InterpolationSegmentPlan]:
    if not source_segments:
        return []

    ratio = output_fps / source_fps
    plans: list[InterpolationSegmentPlan] = []
    for index, segment in enumerate(source_segments):
        overlap_before_frames = 1 if index > 0 else 0
        overlap_after_frames = 1 if index < len(source_segments) - 1 else 0
        expanded_start_frame = max(0, segment.start_frame - overlap_before_frames)
        expanded_end_frame = min(total_source_frames, segment.start_frame + segment.frame_count + overlap_after_frames)
        expanded_frame_count = max(1, expanded_end_frame - expanded_start_frame)
        expanded_output_frame_count = resolve_segment_output_frame_count(
            start_frame=expanded_start_frame,
            frame_count=expanded_frame_count,
            source_fps=source_fps,
            output_fps=output_fps,
        )
        output_start_frame = max(0, round(segment.start_frame * ratio) - round(expanded_start_frame * ratio))
        output_frame_count = resolve_segment_output_frame_count(
            start_frame=segment.start_frame,
            frame_count=segment.frame_count,
            source_fps=source_fps,
            output_fps=output_fps,
        )
        plans.append(
            InterpolationSegmentPlan(
                index=segment.index,
                source_start_frame=segment.start_frame,
                source_frame_count=segment.frame_count,
                expanded_start_frame=expanded_start_frame,
                expanded_frame_count=expanded_frame_count,
                overlap_before_frames=segment.start_frame - expanded_start_frame,
                overlap_after_frames=max(0, expanded_end_frame - (segment.start_frame + segment.frame_count)),
                output_start_frame=output_start_frame,
                output_frame_count=output_frame_count,
                expanded_output_frame_count=expanded_output_frame_count,
                expanded_start_seconds=expanded_start_frame / source_fps if source_fps > 0 else 0.0,
                expanded_duration_seconds=expanded_frame_count / source_fps if source_fps > 0 else 0.0,
            )
        )
    return plans


def _shrink_final_interpolation_segment_plan(
    segment_plan: InterpolationSegmentPlan,
    *,
    actual_expanded_frame_count: int,
    source_fps: float,
    output_fps: float,
) -> InterpolationSegmentPlan:
    if actual_expanded_frame_count >= segment_plan.expanded_frame_count:
        return segment_plan

    adjusted_source_frame_count = max(1, actual_expanded_frame_count - segment_plan.overlap_before_frames)
    adjusted_expanded_output_frame_count = resolve_segment_output_frame_count(
        start_frame=segment_plan.expanded_start_frame,
        frame_count=actual_expanded_frame_count,
        source_fps=source_fps,
        output_fps=output_fps,
    )
    adjusted_output_frame_count = resolve_segment_output_frame_count(
        start_frame=segment_plan.source_start_frame,
        frame_count=adjusted_source_frame_count,
        source_fps=source_fps,
        output_fps=output_fps,
    )
    ratio = output_fps / source_fps if source_fps > 0 else 1.0
    adjusted_output_start_frame = max(
        0,
        round(segment_plan.source_start_frame * ratio) - round(segment_plan.expanded_start_frame * ratio),
    )
    return InterpolationSegmentPlan(
        index=segment_plan.index,
        source_start_frame=segment_plan.source_start_frame,
        source_frame_count=adjusted_source_frame_count,
        expanded_start_frame=segment_plan.expanded_start_frame,
        expanded_frame_count=actual_expanded_frame_count,
        overlap_before_frames=min(segment_plan.overlap_before_frames, max(0, actual_expanded_frame_count - 1)),
        overlap_after_frames=0,
        output_start_frame=adjusted_output_start_frame,
        output_frame_count=adjusted_output_frame_count,
        expanded_output_frame_count=adjusted_expanded_output_frame_count,
        expanded_start_seconds=segment_plan.expanded_start_seconds,
        expanded_duration_seconds=actual_expanded_frame_count / source_fps if source_fps > 0 else 0.0,
    )


def _pipeline_ratio(processed_frames: int, total_frames: int) -> float:
    if total_frames <= 0:
        return 0.0
    return max(0.0, min(1.0, processed_frames / total_frames))


def _pipeline_percent(
    total_frames: int,
    progress_state: PipelineProgressState,
    *,
    source_total_frames: int | None = None,
    interpolation_mode: str = "off",
    colorization_mode: str = "off",
) -> int:
    if interpolation_mode == "interpolateOnly":
        extract_weight = 10
        colorize_weight = 0
        upscale_weight = 0
        interpolate_weight = 70
        encode_weight = 15
        remux_weight = 5
    elif colorization_mode == "colorizeOnly":
        extract_weight = 10
        colorize_weight = 70
        upscale_weight = 0
        interpolate_weight = 0
        encode_weight = 15
        remux_weight = 5
    elif interpolation_mode == "afterUpscale":
        extract_weight = 10
        colorize_weight = 15 if colorization_mode == "beforeUpscale" else 0
        upscale_weight = 20 if colorization_mode == "beforeUpscale" else 35
        interpolate_weight = 35
        encode_weight = 15
        remux_weight = 5
    else:
        extract_weight = 10
        colorize_weight = 20 if colorization_mode == "beforeUpscale" else 0
        upscale_weight = 50 if colorization_mode == "beforeUpscale" else 70
        interpolate_weight = 0
        encode_weight = 15
        remux_weight = 5

    effective_source_total_frames = source_total_frames if source_total_frames is not None and source_total_frames > 0 else total_frames
    if interpolation_mode == "off":
        effective_extracted_frames = progress_state.extracted_frames
        effective_colorized_frames = progress_state.colorized_frames
        effective_upscaled_frames = progress_state.upscaled_frames
    else:
        effective_extracted_frames = _scaled_pipeline_frames(progress_state.extracted_frames, effective_source_total_frames, total_frames)
        effective_colorized_frames = _scaled_pipeline_frames(progress_state.colorized_frames, effective_source_total_frames, total_frames)
        effective_upscaled_frames = _scaled_pipeline_frames(progress_state.upscaled_frames, effective_source_total_frames, total_frames)

    aggregate = (
        _pipeline_ratio(effective_extracted_frames, total_frames) * extract_weight
        + _pipeline_ratio(effective_colorized_frames, total_frames) * colorize_weight
        + _pipeline_ratio(effective_upscaled_frames, total_frames) * upscale_weight
        + _pipeline_ratio(progress_state.interpolated_frames, total_frames) * interpolate_weight
        + _pipeline_ratio(progress_state.encoded_frames, total_frames) * encode_weight
        + _pipeline_ratio(progress_state.remuxed_frames, total_frames) * remux_weight
    )
    return min(99, int(round(aggregate)))


def _emit_pipeline_progress(
    progress_path: str | None,
    *,
    phase: str,
    message: str,
    total_frames: int,
    progress_state: PipelineProgressState,
    source_total_frames: int | None = None,
    interpolation_mode: str = "off",
    colorization_mode: str = "off",
    telemetry_state: PipelineTelemetryState | None = None,
) -> None:
    effective_source_total_frames = source_total_frames if source_total_frames is not None and source_total_frames > 0 else total_frames
    if interpolation_mode == "off":
        effective_extracted_frames = progress_state.extracted_frames
        effective_colorized_frames = progress_state.colorized_frames
        effective_upscaled_frames = progress_state.upscaled_frames
    else:
        effective_extracted_frames = _scaled_pipeline_frames(progress_state.extracted_frames, effective_source_total_frames, total_frames)
        effective_colorized_frames = _scaled_pipeline_frames(progress_state.colorized_frames, effective_source_total_frames, total_frames)
        effective_upscaled_frames = _scaled_pipeline_frames(progress_state.upscaled_frames, effective_source_total_frames, total_frames)

    _write_progress(
        progress_path,
        phase=phase,
        percent=_pipeline_percent(
            total_frames,
            progress_state,
            source_total_frames=effective_source_total_frames,
            interpolation_mode=interpolation_mode,
            colorization_mode=colorization_mode,
        ),
        message=message,
        processed_frames=max(
            effective_extracted_frames,
            effective_colorized_frames,
            effective_upscaled_frames,
            progress_state.interpolated_frames,
            progress_state.encoded_frames,
            progress_state.remuxed_frames,
        ),
        total_frames=total_frames,
        extracted_frames=progress_state.extracted_frames,
        colorized_frames=progress_state.colorized_frames,
        upscaled_frames=progress_state.upscaled_frames,
        interpolated_frames=progress_state.interpolated_frames,
        encoded_frames=progress_state.encoded_frames,
        remuxed_frames=progress_state.remuxed_frames,
        telemetry_state=telemetry_state,
    )


def _extract_segment_frames(
    *,
    ffmpeg: str,
    source_path: str,
    segment: PipelineSegment,
    source_start_seconds: float,
    input_dir: Path,
    log: list[str],
    cancel_path: str | None,
    pause_path: str | None,
) -> int:
    input_dir.mkdir(parents=True, exist_ok=True)
    extract_command = [
        ffmpeg,
        "-y",
        "-i",
        source_path,
        "-ss",
        f"{source_start_seconds + segment.start_seconds:.6f}",
        "-map",
        "0:v:0",
        "-frames:v",
        str(segment.frame_count),
        str(input_dir / "frame_%08d.png"),
    ]
    _run(extract_command, log, cancel_path, pause_path)
    return len(list(input_dir.glob("frame_*.png")))


def _upscale_ncnn_segment(
    *,
    runtime: dict[str, object],
    input_dir: Path,
    output_dir: Path,
    model_id: str,
    gpu_id: int | None,
    effective_tile: int,
    log: list[str],
    cancel_path: str | None,
    pause_path: str | None,
    progress_callback=None,
) -> int:
    extracted_frames = sorted(input_dir.glob("frame_*.png"))
    if not extracted_frames:
        raise RuntimeError("No extracted frames were found for upscaling.")

    output_dir.mkdir(parents=True, exist_ok=True)
    realesrgan_command = [
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
        realesrgan_command.extend(["-g", str(gpu_id)])
    if effective_tile >= 0:
        realesrgan_command.extend(["-t", str(effective_tile)])
    output_count = _run_command_with_output_frame_progress(
        command=realesrgan_command,
        output_dir=output_dir,
        target_frame_count=len(extracted_frames),
        log=log,
        cancel_path=cancel_path,
        pause_path=pause_path,
        progress_callback=progress_callback,
    )
    if output_count <= 0:
        raise RuntimeError("NCNN upscaling completed without producing output frames")
    return output_count


def _upscale_pytorch_segment(
    *,
    loaded_model,
    input_dir: Path,
    output_dir: Path,
    effective_tile: int,
    cancel_path: str | None,
    pause_path: str | None,
    progress_callback=None,
) -> int:
    from upscaler_worker.models.pytorch_sr import upscale_frames

    extracted_frames = sorted(input_dir.glob("frame_*.png"))
    if not extracted_frames:
        raise RuntimeError("No extracted frames were found for upscaling.")

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_frames = 0
    batch_size = loaded_model.frame_batch_size
    batch_count = max(1, math.ceil(len(extracted_frames) / batch_size))
    for batch_index, batch_start in enumerate(range(0, len(extracted_frames), batch_size), start=1):
        ensure_not_cancelled(cancel_path)
        wait_if_paused(pause_path, cancel_path=cancel_path)
        frame_batch = extracted_frames[batch_start:batch_start + batch_size]
        output_batch = [output_dir / frame.name for frame in frame_batch]
        processed_frames += upscale_frames(
            loaded_model=loaded_model,
            input_frames=frame_batch,
            output_frames=output_batch,
            tile_size=effective_tile,
        )
        if progress_callback is not None:
            progress_callback(batch_index, batch_count, processed_frames, len(extracted_frames))
    return processed_frames


def _upscale_external_video_segment(
    *,
    input_dir: Path,
    output_dir: Path,
    model_id: str,
    effective_tile: int,
    gpu_id: int | None,
    precision_mode: str,
    log: list[str],
    cancel_path: str | None,
    pause_path: str | None,
    progress_callback=None,
) -> tuple[int, dict[str, object]]:
    command_info = build_external_video_sr_command(
        model_id=model_id,
        input_dir=input_dir,
        output_dir=output_dir,
        tile_size=effective_tile,
        gpu_id=gpu_id,
        precision=precision_mode,
    )
    log.append(f"Using external video SR runner via {command_info.command_env_var}")
    _run_command_with_output_frame_progress(
        command=command_info.command,
        output_dir=output_dir,
        target_frame_count=len(sorted(input_dir.glob("frame_*.png"))),
        log=log,
        cancel_path=cancel_path,
        pause_path=pause_path,
        progress_callback=progress_callback,
        env=command_info.environment,
    )
    output_count = validate_external_video_sr_outputs(input_dir=input_dir, output_dir=output_dir)
    return output_count, {
        "runner": "external-command",
        "commandEnvVar": command_info.command_env_var,
        "launchCommand": command_info.command,
        "gpuId": gpu_id,
        "precision": precision_mode,
    }


def _run_streaming_pytorch_pipeline(
    *,
    ffmpeg: str,
    source_path: str,
    source_width: int,
    source_height: int,
    fps: str,
    total_frames: int,
    effective_duration: float,
    source_start_seconds: float,
    silent_video: Path,
    codec: str,
    crf: int,
    video_encoder_config: VideoEncoderConfig,
    filter_chain: str | None,
    loaded_model,
    effective_tile: int,
    log: list[str],
    progress_path: str | None,
    cancel_path: str | None,
    pause_path: str | None,
    progress_state: PipelineProgressState,
    telemetry_state: PipelineTelemetryState,
) -> int:
    from upscaler_worker.models.pytorch_sr import upscale_arrays

    scale = loaded_model.scale
    input_frame_bytes = source_width * source_height * 3
    output_width = source_width * scale
    output_height = source_height * scale
    decode_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        source_path,
        *( ["-ss", f"{source_start_seconds:.6f}"] if source_start_seconds > 0 else [] ),
        *( ["-t", f"{effective_duration:.6f}"] if effective_duration > 0 else [] ),
        "-map",
        "0:v:0",
        "-frames:v",
        str(total_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]

    encode_command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{output_width}x{output_height}",
        "-framerate",
        fps,
        "-i",
        "pipe:0",
    ]
    if filter_chain is not None:
        encode_command.extend(["-vf", filter_chain])

    encode_command.extend(
        [
            "-c:v",
            video_encoder_config.encoder,
            *video_encoder_config.quality_args,
            "-pix_fmt",
            "yuv420p",
            str(silent_video),
        ]
    )

    log.append("$ " + " ".join(decode_command))
    log.append("$ " + " ".join(encode_command))
    decode_process = subprocess.Popen(decode_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encode_process = subprocess.Popen(encode_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    batch_size = loaded_model.frame_batch_size
    batch_count = max(1, math.ceil(total_frames / batch_size))
    progress_lock = threading.Lock()
    stage_errors: queue.Queue[BaseException] = queue.Queue()
    stop_event = threading.Event()
    decode_queue: queue.Queue[StreamingFrameBatch | object] = queue.Queue(maxsize=PIPELINE_STAGE_QUEUE_DEPTH)
    encode_queue: queue.Queue[StreamingPixelBatch | object] = queue.Queue(maxsize=PIPELINE_STAGE_QUEUE_DEPTH)
    sentinel = object()
    counters = {
        "decoded_frames": 0,
        "upscaled_frames": 0,
        "encoded_frames": 0,
    }

    def _record_error(error: BaseException) -> None:
        if stage_errors.empty():
            stage_errors.put(error)
        stop_event.set()
        terminate_process(decode_process)
        terminate_process(encode_process)

    def _queue_put(target_queue: queue.Queue[object], value: object) -> None:
        while not stop_event.is_set():
            try:
                target_queue.put(value, timeout=0.5)
                return
            except queue.Full:
                continue

    def _queue_get(target_queue: queue.Queue[object]) -> object:
        while True:
            if stop_event.is_set() and target_queue.empty():
                return sentinel
            try:
                return target_queue.get(timeout=0.5)
            except queue.Empty:
                if stop_event.is_set():
                    return sentinel

    def _publish_progress(phase: str, message: str) -> None:
        with progress_lock:
            _emit_pipeline_progress(
                progress_path,
                phase=phase,
                message=message,
                total_frames=total_frames,
                progress_state=progress_state,
                telemetry_state=telemetry_state,
            )

    def decoder_worker() -> None:
        try:
            batch_index = 0
            while counters["decoded_frames"] < total_frames and not stop_event.is_set():
                wait_if_paused(
                    pause_path,
                    cancel_path=cancel_path,
                    on_pause=lambda: _publish_progress("paused", f"Paused: streaming decode {counters['decoded_frames']}/{total_frames} frames"),
                    on_resume=lambda: _publish_progress("extracting", f"Resumed: streaming decode {counters['decoded_frames']}/{total_frames} frames"),
                )
                ensure_not_cancelled(cancel_path)
                batch_arrays: list[np.ndarray] = []
                batch_extract_started_at = time.time()
                for _ in range(batch_size):
                    if counters["decoded_frames"] >= total_frames:
                        break
                    if decode_process.stdout is None:
                        raise RuntimeError("Streaming decoder stdout is unavailable")
                    payload = _read_exact_bytes(decode_process.stdout, input_frame_bytes)
                    if not payload:
                        break
                    if len(payload) != input_frame_bytes:
                        raise RuntimeError("Streaming decoder returned a partial frame")
                    frame = np.frombuffer(payload, dtype=np.uint8).reshape((source_height, source_width, 3)).copy()
                    batch_arrays.append(frame.astype(np.float32) / 255.0)
                    counters["decoded_frames"] += 1

                if not batch_arrays:
                    break

                batch_index += 1
                with progress_lock:
                    _record_stage_duration(telemetry_state, "extract", time.time() - batch_extract_started_at)
                    progress_state.extracted_frames = counters["decoded_frames"]
                    _set_segment_progress(
                        telemetry_state,
                        segment_index=1,
                        segment_count=1,
                        segment_processed_frames=counters["decoded_frames"],
                        segment_total_frames=total_frames,
                        batch_index=batch_index,
                        batch_count=batch_count,
                    )
                _publish_progress(
                    "extracting",
                    f"Streaming decode {counters['decoded_frames']}/{total_frames} frames",
                )
                _queue_put(decode_queue, StreamingFrameBatch(batch_index=batch_index, frames=batch_arrays))

            if counters["decoded_frames"] < total_frames and not stop_event.is_set():
                raise RuntimeError(
                    f"Streaming decoder ended early after {counters['decoded_frames']} of {total_frames} frames"
                )
        except BaseException as error:  # noqa: BLE001
            _record_error(error)
        finally:
            _queue_put(decode_queue, sentinel)

    def upscaler_worker() -> None:
        try:
            while True:
                item = _queue_get(decode_queue)
                if item is sentinel:
                    break
                if not isinstance(item, StreamingFrameBatch):
                    continue

                wait_if_paused(
                    pause_path,
                    cancel_path=cancel_path,
                    on_pause=lambda: _publish_progress("paused", f"Paused: streaming upscale {progress_state.upscaled_frames}/{total_frames} frames"),
                    on_resume=lambda: _publish_progress("upscaling", f"Resumed: streaming upscale {progress_state.upscaled_frames}/{total_frames} frames"),
                )
                ensure_not_cancelled(cancel_path)
                batch_upscale_started_at = time.time()
                pixel_batch = upscale_arrays(
                    loaded_model=loaded_model,
                    input_arrays=item.frames,
                    tile_size=effective_tile,
                )
                with progress_lock:
                    _record_stage_duration(telemetry_state, "upscale", time.time() - batch_upscale_started_at)
                    counters["upscaled_frames"] += len(pixel_batch)
                    progress_state.upscaled_frames = min(total_frames, counters["upscaled_frames"])
                    _set_segment_progress(
                        telemetry_state,
                        segment_index=1,
                        segment_count=1,
                        segment_processed_frames=progress_state.upscaled_frames,
                        segment_total_frames=total_frames,
                        batch_index=item.batch_index,
                        batch_count=batch_count,
                    )
                _publish_progress(
                    "upscaling",
                    f"Streaming upscale {progress_state.upscaled_frames}/{total_frames} frames",
                )
                _queue_put(encode_queue, StreamingPixelBatch(batch_index=item.batch_index, pixels=pixel_batch))
        except BaseException as error:  # noqa: BLE001
            _record_error(error)
        finally:
            _queue_put(encode_queue, sentinel)

    def encoder_worker() -> None:
        try:
            while True:
                item = _queue_get(encode_queue)
                if item is sentinel:
                    break
                if not isinstance(item, StreamingPixelBatch):
                    continue

                wait_if_paused(
                    pause_path,
                    cancel_path=cancel_path,
                    on_pause=lambda: _publish_progress("paused", f"Paused: streaming encode {progress_state.encoded_frames}/{total_frames} frames"),
                    on_resume=lambda: _publish_progress("encoding", f"Resumed: streaming encode {progress_state.encoded_frames}/{total_frames} frames"),
                )
                ensure_not_cancelled(cancel_path)
                batch_encode_started_at = time.time()
                if encode_process.stdin is None:
                    raise RuntimeError("Streaming encoder stdin is unavailable")
                for pixels in item.pixels:
                    _write_raw_rgb_frame(encode_process.stdin, pixels)
                encode_process.stdin.flush()
                with progress_lock:
                    _record_stage_duration(telemetry_state, "encode", time.time() - batch_encode_started_at)
                    counters["encoded_frames"] += len(item.pixels)
                    progress_state.encoded_frames = min(total_frames, counters["encoded_frames"])
                    _set_segment_progress(
                        telemetry_state,
                        segment_index=1,
                        segment_count=1,
                        segment_processed_frames=progress_state.encoded_frames,
                        segment_total_frames=total_frames,
                        batch_index=item.batch_index,
                        batch_count=batch_count,
                    )
                _publish_progress(
                    "encoding",
                    f"Streaming encode {progress_state.encoded_frames}/{total_frames} frames",
                )
        except BaseException as error:  # noqa: BLE001
            _record_error(error)

    try:
        workers = [
            threading.Thread(target=decoder_worker, name="streaming-decoder", daemon=True),
            threading.Thread(target=upscaler_worker, name="streaming-upscaler", daemon=True),
            threading.Thread(target=encoder_worker, name="streaming-encoder", daemon=True),
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        if not stage_errors.empty():
            raise stage_errors.get()

        if encode_process.stdin is not None:
            encode_process.stdin.close()
        if decode_process.stdout is not None:
            decode_process.stdout.close()

        decode_returncode = decode_process.wait()
        encode_returncode = encode_process.wait()
        _append_process_stderr(log, decode_process, "ffmpeg-decode")
        _append_process_stderr(log, encode_process, "ffmpeg-encode")
        if decode_returncode != 0:
            raise RuntimeError(f"Streaming decoder failed with exit code {decode_returncode}")
        if encode_returncode != 0:
            raise RuntimeError(f"Streaming encoder failed with exit code {encode_returncode}")
    finally:
        if decode_process.stdout is not None:
            decode_process.stdout.close()
        if decode_process.stderr is not None:
            decode_process.stderr.close()
        if encode_process.stdin is not None and not encode_process.stdin.closed:
            encode_process.stdin.close()
        if encode_process.stderr is not None:
            encode_process.stderr.close()

    if counters["encoded_frames"] <= 0:
        raise RuntimeError("Streaming pipeline did not encode any frames")
    return counters["encoded_frames"]


def _encode_segment_video(
    *,
    ffmpeg: str,
    upscaled_dir: Path,
    output_file: Path,
    fps: str,
    codec: str,
    crf: int,
    video_encoder_config: VideoEncoderConfig,
    filter_chain: str | None,
    model_name: str,
    output_mode: str,
    aspect_ratio_preset: str,
    resolution_basis: str,
    resolved_width: int,
    resolved_height: int,
    crop_left: float | None,
    crop_top: float | None,
    crop_width: float | None,
    crop_height: float | None,
    container: str,
    input_start_number: int = 1,
    input_frame_limit: int | None = None,
    log: list[str],
    progress_path: str | None,
    cancel_path: str | None,
    pause_path: str | None,
    total_frames: int,
    extracted_frames: int,
    colorized_frames: int,
    upscaled_frames: int,
    interpolated_frames: int,
    encoded_frames_before_segment: int,
    telemetry_state: PipelineTelemetryState | None,
) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    encode_command = [
        ffmpeg,
        "-y",
        "-framerate",
        fps,
        "-start_number",
        str(max(1, input_start_number)),
        "-i",
        str(upscaled_dir / "frame_%08d.png"),
    ]
    if filter_chain is not None:
        encode_command.extend(["-vf", filter_chain])
    if input_frame_limit is not None:
        encode_command.extend(["-frames:v", str(max(1, input_frame_limit))])

    encode_command.extend(
        [
            "-c:v",
            video_encoder_config.encoder,
            *video_encoder_config.quality_args,
            "-pix_fmt",
            "yuv420p",
            "-metadata",
            f"upscaler_model={model_name}",
            "-metadata",
            f"upscaler_output_mode={output_mode}",
            "-metadata",
            f"upscaler_aspect_ratio={aspect_ratio_preset}",
            "-metadata",
            f"upscaler_resolution_basis={resolution_basis}",
            "-metadata",
            f"upscaler_target_width={resolved_width}",
            "-metadata",
            f"upscaler_target_height={resolved_height}",
            "-metadata",
            f"upscaler_crop_left={crop_left or 0:.4f}",
            "-metadata",
            f"upscaler_crop_top={crop_top or 0:.4f}",
            "-metadata",
            f"upscaler_crop_width={crop_width or 0:.4f}",
            "-metadata",
            f"upscaler_crop_height={crop_height or 0:.4f}",
            "-metadata",
            f"upscaler_codec={codec}",
            "-metadata",
            f"upscaler_container={container}",
            str(output_file),
        ]
    )
    encoded_frame_count, _ = _run_ffmpeg_with_frame_progress(
        encode_command,
        log,
        progress_path,
        cancel_path,
        pause_path,
        phase="encoding",
        percent_base=80,
        percent_span=15,
        total_frames=total_frames,
        message_prefix=f"Encoding segment video {output_file.name}",
        extracted_frames=extracted_frames,
        colorized_frames=colorized_frames,
        upscaled_frames=upscaled_frames,
        interpolated_frames=interpolated_frames,
        encoded_frames=encoded_frames_before_segment,
        remuxed_frames=0,
        telemetry_state=telemetry_state,
        stage_frame_offset=encoded_frames_before_segment,
    )
    return max(0, encoded_frame_count - encoded_frames_before_segment)


def _concat_segment_videos(
    *,
    ffmpeg: str,
    segment_files: list[Path],
    concat_manifest: Path,
    output_file: Path,
    progress_path: str | None,
    cancel_path: str | None,
    pause_path: str | None,
    total_frames: int,
    extracted_frames: int,
    colorized_frames: int,
    upscaled_frames: int,
    interpolated_frames: int,
    encoded_frames: int,
    log: list[str],
    telemetry_state: PipelineTelemetryState | None = None,
) -> int:
    concat_manifest.parent.mkdir(parents=True, exist_ok=True)
    concat_manifest.write_text(
        "\n".join(f"file '{segment_file.as_posix()}'" for segment_file in segment_files),
        encoding="utf-8",
    )
    concat_command = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_manifest),
        "-c",
        "copy",
        str(output_file),
    ]
    concatenated_frames, _ = _run_ffmpeg_with_frame_progress(
        concat_command,
        log,
        progress_path,
        cancel_path,
        pause_path,
        phase="remuxing",
        percent_base=95,
        percent_span=2,
        total_frames=total_frames,
        message_prefix="Concatenating encoded segments",
        extracted_frames=extracted_frames,
        colorized_frames=colorized_frames,
        upscaled_frames=upscaled_frames,
        interpolated_frames=interpolated_frames,
        encoded_frames=encoded_frames,
        remuxed_frames=0,
        telemetry_state=telemetry_state,
    )
    return concatenated_frames


def _fps_text(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _scaled_pipeline_frames(processed_frames: int, source_total_frames: int, pipeline_total_frames: int) -> int:
    if source_total_frames <= 0:
        return 0
    return min(pipeline_total_frames, max(0, int(round((processed_frames / source_total_frames) * pipeline_total_frames))))


def _upscale_frames_in_batches(
    *,
    realesrgan: str,
    input_frames: Path,
    output_frames: Path,
    model_dir: str,
    model_id: str,
    gpu_id: int | None,
    effective_tile: int,
    progress_path: str | None,
    total_frames: int,
    log: list[str],
    work_dir: Path,
) -> int:
    extracted_frames = sorted(input_frames.glob("frame_*.png"))
    if not extracted_frames:
        raise RuntimeError("No extracted frames were found for upscaling.")

    batch_root = work_dir / "batches"
    batch_root.mkdir(parents=True, exist_ok=True)
    processed_frames = 0
    batch_count = (len(extracted_frames) + BATCH_FRAME_COUNT - 1) // BATCH_FRAME_COUNT

    for batch_index, batch_start in enumerate(range(0, len(extracted_frames), BATCH_FRAME_COUNT), start=1):
        batch_frames = extracted_frames[batch_start:batch_start + BATCH_FRAME_COUNT]
        batch_in = batch_root / f"batch_{batch_index:03d}_in"
        batch_out = batch_root / f"batch_{batch_index:03d}_out"
        batch_in.mkdir(parents=True, exist_ok=True)
        batch_out.mkdir(parents=True, exist_ok=True)

        for frame in batch_frames:
            shutil.copy2(frame, batch_in / frame.name)

        _write_progress(
            progress_path,
            phase="upscaling",
            percent=15 if total_frames <= 0 else min(85, 15 + int((processed_frames / max(total_frames, 1)) * 70)),
            message=f"Upscaling batch {batch_index}/{batch_count} ({processed_frames}/{total_frames} frames completed)",
            processed_frames=processed_frames,
            total_frames=total_frames,
            extracted_frames=total_frames,
            upscaled_frames=processed_frames,
        )

        realesrgan_command = [
            realesrgan,
            "-i",
            str(batch_in),
            "-o",
            str(batch_out),
            "-m",
            model_dir,
            "-n",
            model_id,
            "-f",
            "png",
        ]
        if gpu_id is not None:
            realesrgan_command.extend(["-g", str(gpu_id)])
        if effective_tile >= 0:
            realesrgan_command.extend(["-t", str(effective_tile)])

        _run_realesrgan_batch(realesrgan_command, log)

        for frame in sorted(batch_out.glob("frame_*.png")):
            shutil.move(str(frame), output_frames / frame.name)

        processed_frames += len(batch_frames)
        _write_progress(
            progress_path,
            phase="upscaling",
            percent=15 if total_frames <= 0 else min(85, 15 + int((processed_frames / max(total_frames, 1)) * 70)),
            message=f"Upscaling frames with Real-ESRGAN ({processed_frames}/{total_frames})",
            processed_frames=processed_frames,
            total_frames=total_frames,
            extracted_frames=total_frames,
            upscaled_frames=processed_frames,
        )

        shutil.rmtree(batch_in, ignore_errors=True)
        shutil.rmtree(batch_out, ignore_errors=True)

    return processed_frames


def _upscale_frames_with_pytorch(
    *,
    input_frames: Path,
    output_frames: Path,
    model_id: str,
    gpu_id: int | None,
    effective_tile: int,
    progress_path: str | None,
    total_frames: int,
    log: list[str],
    fp16: bool,
    preset: str,
    torch_compile_enabled: bool,
    bf16: bool,
    pytorch_runner: str,
    channels_last: bool,
    preloaded_pytorch_model=None,
) -> int:
    from upscaler_worker.models.pytorch_sr import load_runtime_model, upscale_frames

    extracted_frames = sorted(input_frames.glob("frame_*.png"))
    if not extracted_frames:
        raise RuntimeError("No extracted frames were found for upscaling.")

    loaded_model = preloaded_pytorch_model
    if loaded_model is None:
        loaded_model = load_runtime_model(
            model_id,
            gpu_id,
            fp16,
            effective_tile,
            log,
            preset=preset,
            torch_compile_enabled=torch_compile_enabled,
            bf16=bf16,
            pytorch_runner=pytorch_runner,
            channels_last_enabled=channels_last,
        )
        log.append(f"Loaded PyTorch model checkpoint: {loaded_model.checkpoint_path}")

    processed_frames = 0
    batch_size = loaded_model.frame_batch_size
    for batch_start in range(0, len(extracted_frames), batch_size):
        frame_batch = extracted_frames[batch_start:batch_start + batch_size]
        output_batch = [output_frames / frame.name for frame in frame_batch]
        processed_frames += upscale_frames(
            loaded_model=loaded_model,
            input_frames=frame_batch,
            output_frames=output_batch,
            tile_size=effective_tile,
        )
        _write_progress(
            progress_path,
            phase="upscaling",
            percent=15 if total_frames <= 0 else min(85, 15 + int((processed_frames / max(total_frames, 1)) * 70)),
            message=f"Upscaling frames with {loaded_model.model_label} ({processed_frames}/{total_frames})",
            processed_frames=processed_frames,
            total_frames=total_frames,
            extracted_frames=total_frames,
            upscaled_frames=processed_frames,
        )

    if loaded_model.device.type == "cuda":
        import torch

        torch.cuda.synchronize(loaded_model.device)

    return processed_frames


def _upscale_frames(
    *,
    runtime: dict[str, object],
    input_frames: Path,
    output_frames: Path,
    model_id: str,
    gpu_id: int | None,
    effective_tile: int,
    progress_path: str | None,
    total_frames: int,
    log: list[str],
    work_dir: Path,
    fp16: bool,
    preset: str,
    torch_compile_enabled: bool,
    bf16: bool,
    channels_last: bool,
    preloaded_pytorch_model=None,
) -> int:
    backend_id = model_backend_id(model_id)
    if backend_id == "realesrgan-ncnn":
        return _upscale_frames_in_batches(
            realesrgan=str(runtime["realesrganPath"]),
            input_frames=input_frames,
            output_frames=output_frames,
            model_dir=str(runtime["modelDir"]),
            model_id=model_id,
            gpu_id=gpu_id,
            effective_tile=effective_tile,
            progress_path=progress_path,
            total_frames=total_frames,
            log=log,
            work_dir=work_dir,
        )
    if backend_id == "pytorch-image-sr":
        return _upscale_frames_with_pytorch(
            input_frames=input_frames,
            output_frames=output_frames,
            model_id=model_id,
            gpu_id=gpu_id,
            effective_tile=effective_tile,
            progress_path=progress_path,
            total_frames=total_frames,
            log=log,
            fp16=fp16,
            preset=preset,
            torch_compile_enabled=torch_compile_enabled,
            bf16=bf16,
            channels_last=channels_last,
            pytorch_runner="torch",
            preloaded_pytorch_model=preloaded_pytorch_model,
        )
    if backend_id == "pytorch-video-sr":
        output_count, _ = _upscale_external_video_segment(
            input_dir=input_frames,
            output_dir=output_frames,
            model_id=model_id,
            effective_tile=effective_tile,
            gpu_id=gpu_id,
            precision_mode="fp16" if fp16 else "bf16" if bf16 else "fp32",
            log=log,
            cancel_path=None,
            pause_path=None,
        )
        _write_progress(
            progress_path,
            phase="upscaling",
            percent=15 if total_frames <= 0 else min(85, 15 + int((output_count / max(total_frames, 1)) * 70)),
            message=f"Upscaling frames with external video SR ({output_count}/{total_frames})",
            processed_frames=output_count,
            total_frames=total_frames,
            extracted_frames=total_frames,
            upscaled_frames=output_count,
        )
        return output_count

    raise RuntimeError(f"Backend '{backend_id}' is cataloged but not runnable in the current app build")


def _output_filter(
    output_mode: str,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    crop_left: float | None,
    crop_top: float | None,
    crop_width: float | None,
    crop_height: float | None,
) -> str | None:
    if output_mode == "preserveAspect4k":
        return (
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black"
        )
    if output_mode == "cropTo4k":
        crop_rect = _clamp_crop_rect(crop_left, crop_top, crop_width, crop_height)
        if crop_rect is not None:
            left, top, width, height = crop_rect
            return (
                f"crop={_round_dimension(source_width * 4 * width)}:{_round_dimension(source_height * 4 * height)}:"
                f"{_round_dimension(source_width * 4 * left)}:{_round_dimension(source_height * 4 * top)},"
                f"scale={target_width}:{target_height}"
            )
        return (
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase,"
            f"crop={target_width}:{target_height}"
        )
    return None


def _effective_tile_size(model_id: str, preset: str, tile_size: int) -> int:
    if tile_size > 0:
        return tile_size
    backend_id = model_backend_id(model_id)
    if backend_id == "pytorch-image-sr":
        if preset == "qualityMax":
            return 512
        if preset == "qualityBalanced":
            return 384
        return 256
    if backend_id == "pytorch-video-sr":
        if preset == "qualityMax":
            return 256
        if preset == "qualityBalanced":
            return 192
        return 128
    if preset == "qualityMax":
        return 0
    if preset == "qualityBalanced":
        return 256
    return 128


def _has_explicit_precision_request(*, fp16: bool, bf16: bool, precision: str | None) -> bool:
    return bool(fp16 or bf16 or (precision is not None and precision.strip()))


def _default_precision_mode_for_backend(model_id: str, preset: str) -> tuple[str, str]:
    backend_id = model_backend_id(model_id)
    if backend_id in {"pytorch-image-sr", "pytorch-video-sr"}:
        if preset == "qualityMax":
            return "fp32", "preset-default"
        if preset == "qualityBalanced":
            return "bf16", "preset-default"
        return "fp16", "preset-default"
    return "fp32", "backend-fixed"


def _resolve_quality_policy(
    model_id: str,
    preset: str,
    tile_size: int,
    *,
    fp16: bool,
    bf16: bool,
    precision: str | None,
) -> dict[str, object]:
    explicit_precision = _has_explicit_precision_request(fp16=fp16, bf16=bf16, precision=precision)
    requested_precision_mode = resolve_precision_mode(fp16=fp16, bf16=bf16, precision=precision)
    if explicit_precision:
        selected_precision_mode = requested_precision_mode
        precision_source = "explicit-request"
    else:
        selected_precision_mode, precision_source = _default_precision_mode_for_backend(model_id, preset)
    return {
        "backendId": model_backend_id(model_id),
        "qualityPreset": preset,
        "requestedTileSize": int(tile_size),
        "effectiveTileSize": int(_effective_tile_size(model_id, preset, tile_size)),
        "requestedPrecision": requested_precision_mode if explicit_precision else None,
        "selectedPrecision": selected_precision_mode,
        "precisionSource": precision_source,
    }


def _resolve_preview_start_offset_seconds(
    requested_offset_seconds: float | None,
    *,
    source_duration_seconds: float,
    frame_rate: float,
) -> tuple[float, int]:
    if requested_offset_seconds is None or requested_offset_seconds <= 0 or frame_rate <= 0 or source_duration_seconds <= 0:
        return 0.0, 0

    total_source_frames = max(1, int(round(frame_rate * source_duration_seconds)))
    clamped_offset = max(0.0, min(float(requested_offset_seconds), source_duration_seconds))
    start_frame_index = min(total_source_frames - 1, max(0, int(math.ceil((clamped_offset * frame_rate) - 1e-9))))
    return start_frame_index / frame_rate, start_frame_index


def _resolve_preview_window(
    *,
    preview_mode: bool,
    preview_duration_seconds: float | None,
    preview_start_offset_seconds: float | None,
    source_duration_seconds: float,
    frame_rate: float,
) -> tuple[float, float, int]:
    if frame_rate <= 0 or source_duration_seconds <= 0:
        return 0.0, 0.0, 1

    if not preview_mode:
        total_frames = max(1, int(round(frame_rate * source_duration_seconds)))
        return 0.0, source_duration_seconds, total_frames

    start_offset_seconds, start_frame_index = _resolve_preview_start_offset_seconds(
        preview_start_offset_seconds,
        source_duration_seconds=source_duration_seconds,
        frame_rate=frame_rate,
    )
    total_source_frames = max(1, int(round(frame_rate * source_duration_seconds)))
    available_frames = max(1, total_source_frames - start_frame_index)
    requested_preview_duration = preview_duration_seconds if preview_duration_seconds and preview_duration_seconds > 0 else 8.0
    requested_frames = max(1, int(round(frame_rate * requested_preview_duration)))
    total_frames = min(available_frames, requested_frames)
    effective_duration = total_frames / frame_rate
    return start_offset_seconds, effective_duration, total_frames


def _build_pipeline_media_summary(
    *,
    width: int,
    height: int,
    frame_rate: float,
    duration_seconds: float,
    frame_count: int,
    has_audio: bool | None = None,
    container: str | None = None,
    video_codec: str | None = None,
) -> dict[str, object]:
    pixel_count = max(0, int(width) * int(height))
    aspect_ratio = (float(width) / float(height)) if height else 0.0
    return {
        "width": int(width),
        "height": int(height),
        "frameRate": float(frame_rate),
        "durationSeconds": float(duration_seconds),
        "frameCount": int(frame_count),
        "aspectRatio": aspect_ratio,
        "pixelCount": pixel_count,
        "hasAudio": has_audio,
        "container": container,
        "videoCodec": video_codec,
    }


def _build_pipeline_effective_settings(
    *,
    backend_id: str,
    quality_preset: str,
    requested_tile_size: int,
    effective_tile_size: int,
    requested_precision: str | None,
    selected_precision: str,
    effective_precision: str,
    precision_source: str,
    processed_duration_seconds: float,
    preview_start_offset_seconds: float | None,
    segment_frame_limit: int,
    preview_mode: bool,
    preview_duration_seconds: float | None,
    segment_duration_seconds: float | None,
) -> dict[str, object]:
    return {
        "backendId": backend_id,
        "qualityPreset": quality_preset,
        "requestedTileSize": int(requested_tile_size),
        "effectiveTileSize": int(effective_tile_size),
        "requestedPrecision": requested_precision,
        "selectedPrecision": selected_precision,
        "effectivePrecision": effective_precision,
        "precisionSource": precision_source,
        "processedDurationSeconds": float(processed_duration_seconds),
        "previewStartOffsetSeconds": preview_start_offset_seconds,
        "segmentFrameLimit": int(segment_frame_limit),
        "previewMode": bool(preview_mode),
        "previewDurationSeconds": preview_duration_seconds,
        "segmentDurationSeconds": segment_duration_seconds,
    }


def run_realesrgan_pipeline(
    *,
    source_path: str,
    model_id: str,
    colorization_mode: str = "off",
    colorizer_model_id: str | None = None,
    color_context_library_id: str | None = None,
    color_reference_images: list[str] | None = None,
    deepremaster_processing_mode: str = "standard",
    output_mode: str,
    preset: str,
    interpolation_mode: str = "off",
    interpolation_target_fps: int | None = None,
    gpu_id: int | None,
    aspect_ratio_preset: str,
    custom_aspect_width: int | None,
    custom_aspect_height: int | None,
    resolution_basis: str,
    target_width: int | None,
    target_height: int | None,
    crop_left: float | None,
    crop_top: float | None,
    crop_width: float | None,
    crop_height: float | None,
    job_id: str | None = None,
    progress_path: str | None,
    cancel_path: str | None,
    pause_path: str | None = None,
    preview_mode: bool,
    preview_duration_seconds: float | None,
    preview_start_offset_seconds: float | None = None,
    segment_duration_seconds: float | None,
    output_path: str,
    codec: str,
    container: str,
    tile_size: int,
    fp16: bool,
    torch_compile_enabled: bool,
    torch_compile_mode: str = "reduce-overhead",
    torch_compile_cudagraphs: bool = False,
    crf: int,
    pytorch_execution_path: str | None = None,
    pytorch_runner: str | None = None,
    bf16: bool = False,
    precision: str | None = None,
    channels_last: bool = False,
    preloaded_pytorch_model=None,
) -> dict[str, object]:
    ensure_not_cancelled(cancel_path)
    wait_if_paused(pause_path, cancel_path=cancel_path)
    colorization_enabled = colorization_mode != "off"
    selected_color_reference_images = [str(path) for path in (color_reference_images or [])]
    colorizer_backend_id: str | None = None
    if colorization_enabled:
        if not colorizer_model_id:
            raise ValueError("Colorization was requested but no colorizer model id was provided")
        ensure_runnable_model(colorizer_model_id)
        if model_task(colorizer_model_id) != "colorize":
            raise ValueError(f"Model '{colorizer_model_id}' is not cataloged as a colorizer")
        colorizer_backend_id = model_backend_id(colorizer_model_id)
        if colorizer_backend_id != "pytorch-image-colorization":
            raise NotImplementedError(
                f"Colorizer backend '{colorizer_backend_id}' is cataloged but not implemented in the worker pipeline"
            )
        if selected_color_reference_images:
            missing_reference_images = [path for path in selected_color_reference_images if not Path(path).is_file()]
            if missing_reference_images:
                raise ValueError(
                    "Selected color reference images do not exist: " + ", ".join(missing_reference_images)
                )

    active_model_id = colorizer_model_id if colorization_mode == "colorizeOnly" and colorizer_model_id else model_id
    if colorization_mode != "colorizeOnly":
        ensure_runnable_model(model_id)
    ensure_runnable_model(active_model_id)
    backend_id = model_backend_id(active_model_id)
    if backend_id == "pytorch-image-colorization" and colorization_mode != "colorizeOnly":
        colorization_mode = "colorizeOnly"
        colorization_enabled = True
        colorizer_model_id = colorizer_model_id or active_model_id
        colorizer_backend_id = backend_id
    quality_policy = _resolve_quality_policy(
        active_model_id,
        preset,
        tile_size,
        fp16=fp16,
        bf16=bf16,
        precision=precision,
    )
    requested_precision_mode = quality_policy["requestedPrecision"]
    selected_precision_mode = str(quality_policy["selectedPrecision"])
    precision_source = str(quality_policy["precisionSource"])
    effective_precision_mode = selected_precision_mode
    fp16_enabled = selected_precision_mode == "fp16"
    bf16_enabled = selected_precision_mode == "bf16"
    resolved_pytorch_execution_path = _resolve_pytorch_execution_path(active_model_id, pytorch_execution_path)
    runtime = ensure_runtime_assets()
    metadata = probe_video(source_path)
    validate_interpolation_request(interpolation_mode, interpolation_target_fps)
    if colorization_enabled and interpolation_mode != "off":
        raise NotImplementedError("Colorization with interpolation is not implemented yet. Run colorization with interpolation disabled for now.")
    requested_output = Path(output_path)
    if not requested_output.is_absolute():
        requested_output = repo_root() / requested_output
    normalized_output = requested_output.with_suffix(f".{container}").resolve()
    normalized_output.parent.mkdir(parents=True, exist_ok=True)

    jobs_root = repo_root() / "artifacts" / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)

    cache_key = job_id or hashlib.sha256(
        "|".join(
            [
                source_path,
                model_id,
                colorization_mode,
                colorizer_model_id or "",
                color_context_library_id or "",
                *selected_color_reference_images,
                deepremaster_processing_mode,
                output_mode,
                preset,
                interpolation_mode,
                str(interpolation_target_fps or 0),
                str(gpu_id if gpu_id is not None else -1),
                aspect_ratio_preset,
                str(custom_aspect_width or 0),
                str(custom_aspect_height or 0),
                resolution_basis,
                str(target_width or 0),
                str(target_height or 0),
                f"{crop_left or 0:.4f}",
                f"{crop_top or 0:.4f}",
                f"{crop_width or 0:.4f}",
                f"{crop_height or 0:.4f}",
                output_path,
                codec,
                container,
                str(tile_size),
                selected_precision_mode,
                str(segment_duration_seconds or 0),
                str(crf),
                resolved_pytorch_execution_path or "external-executable",
                pytorch_runner or "torch",
                str(int(channels_last)),
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]

    work_dir = jobs_root / f"job_{cache_key}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    segment_root = work_dir / "segments"
    encoded_dir = work_dir / "enc"
    segment_root.mkdir(parents=True, exist_ok=True)
    encoded_dir.mkdir(parents=True, exist_ok=True)

    output_file = normalized_output
    silent_video = encoded_dir / f"video_no_audio.{PIPELINE_INTERMEDIATE_CONTAINER}"
    model_name = model_label(active_model_id)

    log: list[str] = []
    if color_context_library_id:
        log.append(f"Color context library: {color_context_library_id}")
    if selected_color_reference_images:
        log.append(
            "Selected color references: " + ", ".join(Path(path).name for path in selected_color_reference_images)
        )
    model_runtime: dict[str, object] | None = None
    ffmpeg = runtime["ffmpegPath"]
    video_encoder_config = _resolve_video_encoder_config(
        ffmpeg=str(ffmpeg),
        runtime=runtime,
        gpu_id=gpu_id,
        codec=codec,
        crf=crf,
        log=log,
    )
    source_duration_seconds = float(metadata["durationSeconds"])
    source_frame_rate = float(metadata["frameRate"])
    fps = f"{source_frame_rate:.6f}".rstrip("0").rstrip(".")
    preview_window_start_seconds, effective_duration, total_frames = _resolve_preview_window(
        preview_mode=preview_mode,
        preview_duration_seconds=preview_duration_seconds,
        preview_start_offset_seconds=preview_start_offset_seconds,
        source_duration_seconds=source_duration_seconds,
        frame_rate=source_frame_rate,
    )
    force_single_stream_segment = backend_id == "pytorch-image-sr" and resolved_pytorch_execution_path == PYTORCH_EXECUTION_PATH_STREAMING
    segment_frame_limit = total_frames if preview_mode or force_single_stream_segment else _segment_frame_limit(source_frame_rate, segment_duration_seconds)
    segments = _plan_pipeline_segments(
        total_frames,
        source_frame_rate,
        force_single_segment=preview_mode or force_single_stream_segment,
        segment_duration_seconds=segment_duration_seconds,
    )
    resolved_width, resolved_height, resolved_aspect_ratio = _resolve_output_dimensions(
        source_width=int(metadata["width"]),
        source_height=int(metadata["height"]),
        output_mode=output_mode,
        aspect_ratio_preset=aspect_ratio_preset,
        custom_aspect_width=custom_aspect_width,
        custom_aspect_height=custom_aspect_height,
        resolution_basis=resolution_basis,
        target_width=target_width,
        target_height=target_height,
    )

    effective_tile = int(quality_policy["effectiveTileSize"])
    log.append(
        f"Quality preset policy: {preset} -> tile {effective_tile} and precision {selected_precision_mode} ({precision_source})."
    )
    source_media = _build_pipeline_media_summary(
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        frame_rate=float(metadata["frameRate"]),
        duration_seconds=float(metadata["durationSeconds"]),
        frame_count=total_frames,
        has_audio=bool(metadata["hasAudio"]),
        container=str(metadata.get("container") or ""),
        video_codec=str(metadata.get("videoCodec") or ""),
    )

    filter_chain = _output_filter(
        output_mode,
        int(metadata["width"]),
        int(metadata["height"]),
        resolved_width,
        resolved_height,
        crop_left,
        crop_top,
        crop_width,
        crop_height,
    )
    if colorization_mode == "colorizeOnly":
        resolved_width = int(metadata["width"])
        resolved_height = int(metadata["height"])
        resolved_aspect_ratio = float(resolved_width / max(1, resolved_height))
        filter_chain = None

    if interpolation_mode != "off":
        runtime.update(ensure_rife_runtime())
        source_fps = float(metadata["frameRate"])
        output_fps = resolve_output_fps(source_fps, interpolation_mode, interpolation_target_fps)
        if output_fps < source_fps - 0.01:
            raise ValueError(
                f"Interpolation target fps {output_fps:.2f} is lower than source fps {source_fps:.2f}. Choose a target that preserves playback duration."
            )

        total_output_frames = resolve_segment_output_frame_count(
            start_frame=0,
            frame_count=total_frames,
            source_fps=source_fps,
            output_fps=output_fps,
        )
        encode_fps = _fps_text(output_fps)
        encode_width = resolved_width
        encode_height = resolved_height
        encode_filter_chain = filter_chain
        if interpolation_mode == "interpolateOnly":
            encode_width = int(metadata["width"])
            encode_height = int(metadata["height"])
            encode_filter_chain = None

        interpolation_segments = _plan_interpolation_segments(
            segments,
            total_source_frames=total_frames,
            source_fps=source_fps,
            output_fps=output_fps,
        )
        if backend_id == "pytorch-image-sr" and resolved_pytorch_execution_path == PYTORCH_EXECUTION_PATH_STREAMING:
            raise NotImplementedError(
                "PyTorch streaming execution path is not supported when interpolation is enabled yet. Use file-io for now."
            )

        progress_state = PipelineProgressState()
        telemetry_state = PipelineTelemetryState(
            started_at=time.time(),
            source_path=source_path,
            scratch_path=work_dir,
            output_path=output_file,
            job_id=cache_key,
        )
        _write_progress(
            progress_path,
            phase="queued",
            percent=0,
            message="Job queued",
            processed_frames=0,
            total_frames=total_output_frames,
            telemetry_state=telemetry_state,
        )

        loaded_model = None
        segment_outputs: list[Path | None] = [None] * len(interpolation_segments)
        concat_manifest = encoded_dir / "segments.txt"
        progress_lock = threading.Lock()
        stage_errors: queue.Queue[BaseException] = queue.Queue()
        stop_event = threading.Event()
        encode_queue: queue.Queue[InterpolationEncodeTask | object] = queue.Queue(maxsize=PIPELINE_STAGE_QUEUE_DEPTH)
        sentinel = object()

        def _record_error(error: BaseException) -> None:
            if stage_errors.empty():
                stage_errors.put(error)
            stop_event.set()

        def _queue_put(target_queue: queue.Queue[object], value: object) -> None:
            while not stop_event.is_set():
                try:
                    target_queue.put(value, timeout=0.5)
                    return
                except queue.Full:
                    continue

        def _queue_get(target_queue: queue.Queue[object]) -> object:
            while True:
                if stop_event.is_set() and target_queue.empty():
                    return sentinel
                try:
                    return target_queue.get(timeout=0.5)
                except queue.Empty:
                    if stop_event.is_set():
                        return sentinel

        def _publish_progress(phase: str, message: str) -> None:
            with progress_lock:
                if not _should_publish_stage_progress(phase, progress_state):
                    return
                _emit_pipeline_progress(
                    progress_path,
                    phase=phase,
                    message=message,
                    total_frames=total_output_frames,
                    progress_state=progress_state,
                    source_total_frames=total_frames,
                    interpolation_mode=interpolation_mode,
                    telemetry_state=telemetry_state,
                )

        def preprocess_worker() -> None:
            nonlocal loaded_model, model_runtime
            try:
                if interpolation_mode == "afterUpscale" and backend_id == "pytorch-image-sr":
                    from upscaler_worker.models.pytorch_sr import load_runtime_model

                    loaded_model = load_runtime_model(
                        model_id,
                        gpu_id,
                        False,
                        effective_tile,
                        log,
                        preset=preset,
                        torch_compile_enabled=torch_compile_enabled,
                        torch_compile_mode=torch_compile_mode,
                        torch_compile_cudagraphs=torch_compile_cudagraphs,
                        bf16=False,
                        precision=selected_precision_mode,
                        pytorch_runner=pytorch_runner,
                        channels_last_enabled=channels_last,
                    )
                    log.append(f"Loaded PyTorch model checkpoint: {loaded_model.checkpoint_path}")
                    effective_precision_mode = loaded_model.precision_mode
                    model_runtime = {
                        "runner": loaded_model.runner,
                        "precision": loaded_model.precision_mode,
                        "dtype": str(loaded_model.dtype).replace("torch.", ""),
                        "frameBatchSize": loaded_model.frame_batch_size,
                        "channelsLast": loaded_model.channels_last,
                        "torchCompileRequested": loaded_model.torch_compile_requested,
                        "torchCompileEnabled": loaded_model.torch_compile_enabled,
                        "torchCompileMode": loaded_model.torch_compile_mode,
                        "torchCompileCudagraphs": loaded_model.torch_compile_cudagraphs,
                    }

                for segment_plan in interpolation_segments:
                    if stop_event.is_set():
                        break

                    active_segment_plan = segment_plan

                    expanded_segment = PipelineSegment(
                        index=active_segment_plan.index,
                        start_frame=active_segment_plan.expanded_start_frame,
                        frame_count=active_segment_plan.expanded_frame_count,
                        start_seconds=active_segment_plan.expanded_start_seconds,
                        duration_seconds=active_segment_plan.expanded_duration_seconds,
                    )
                    segment_dir = segment_root / f"segment_{active_segment_plan.index:04d}"
                    segment_input_dir = segment_dir / "in"
                    segment_upscaled_dir = segment_dir / "upscaled"
                    segment_interpolated_dir = segment_dir / "interpolated"
                    segment_file = encoded_dir / f"segment_{active_segment_plan.index:04d}.{PIPELINE_INTERMEDIATE_CONTAINER}"
                    frame_stage_output_dir = segment_input_dir

                    with progress_lock:
                        _set_segment_progress(
                            telemetry_state,
                            segment_index=active_segment_plan.index + 1,
                            segment_count=len(interpolation_segments),
                            segment_processed_frames=0,
                            segment_total_frames=active_segment_plan.source_frame_count,
                        )
                    _publish_progress(
                        "extracting",
                        f"Extracting segment {active_segment_plan.index + 1}/{len(interpolation_segments)} (0/{active_segment_plan.source_frame_count} frames)",
                    )

                    extract_stage_started_at = time.time()
                    extracted_count = _extract_segment_frames(
                        ffmpeg=str(ffmpeg),
                        source_path=source_path,
                        segment=expanded_segment,
                        source_start_seconds=preview_window_start_seconds,
                        input_dir=segment_input_dir,
                        log=log,
                        cancel_path=cancel_path,
                        pause_path=pause_path,
                    )
                    if extracted_count != active_segment_plan.expanded_frame_count:
                        if (
                            extracted_count < active_segment_plan.expanded_frame_count
                            and active_segment_plan.index == len(interpolation_segments) - 1
                            and extracted_count > active_segment_plan.overlap_before_frames
                        ):
                            active_segment_plan = _shrink_final_interpolation_segment_plan(
                                active_segment_plan,
                                actual_expanded_frame_count=extracted_count,
                                source_fps=source_fps,
                                output_fps=output_fps,
                            )
                            expanded_segment = PipelineSegment(
                                index=active_segment_plan.index,
                                start_frame=active_segment_plan.expanded_start_frame,
                                frame_count=active_segment_plan.expanded_frame_count,
                                start_seconds=active_segment_plan.expanded_start_seconds,
                                duration_seconds=active_segment_plan.expanded_duration_seconds,
                            )
                            log.append(
                                f"Final interpolation segment {active_segment_plan.index + 1} reached EOF early; adjusting expected extracted frames from {segment_plan.expanded_frame_count} to {extracted_count}."
                            )
                        else:
                            raise RuntimeError(
                                f"Expected {active_segment_plan.expanded_frame_count} extracted frames for segment {active_segment_plan.index + 1}, received {extracted_count}"
                            )
                    with progress_lock:
                        _record_stage_duration(telemetry_state, "extract", time.time() - extract_stage_started_at)
                        progress_state.extracted_frames = min(total_frames, progress_state.extracted_frames + active_segment_plan.source_frame_count)
                        extracted_snapshot = progress_state.extracted_frames
                        _set_segment_progress(
                            telemetry_state,
                            segment_index=active_segment_plan.index + 1,
                            segment_count=len(interpolation_segments),
                            segment_processed_frames=active_segment_plan.source_frame_count,
                            segment_total_frames=active_segment_plan.source_frame_count,
                        )
                    _publish_progress(
                        "extracting",
                        f"Extracted segment {active_segment_plan.index + 1}/{len(interpolation_segments)} ({extracted_snapshot}/{total_frames} source frames)",
                    )

                    if interpolation_mode == "afterUpscale":
                        with progress_lock:
                            upscaled_before_segment = progress_state.upscaled_frames

                        def report_upscale_progress(processed_in_segment: int, total_in_segment: int, batch_index: int | None = None, batch_count: int | None = None) -> None:
                            processed_without_overlap = max(0, processed_in_segment - active_segment_plan.overlap_before_frames)
                            unique_processed = min(active_segment_plan.source_frame_count, processed_without_overlap)
                            with progress_lock:
                                current_upscaled_frames = min(total_frames, upscaled_before_segment + unique_processed)
                                _set_segment_progress(
                                    telemetry_state,
                                    segment_index=active_segment_plan.index + 1,
                                    segment_count=len(interpolation_segments),
                                    segment_processed_frames=unique_processed,
                                    segment_total_frames=active_segment_plan.source_frame_count,
                                    batch_index=batch_index,
                                    batch_count=batch_count,
                                )
                            batch_label = f" batch {batch_index}/{batch_count}" if batch_index is not None and batch_count is not None else ""
                            _publish_progress(
                                "upscaling",
                                f"Upscaling segment {active_segment_plan.index + 1}/{len(interpolation_segments)}{batch_label} ({unique_processed}/{active_segment_plan.source_frame_count} frames)",
                            )

                        def report_upscale_batch(batch_index: int, batch_count: int, processed_in_segment: int, total_in_segment: int) -> None:
                            report_upscale_progress(processed_in_segment, total_in_segment, batch_index, batch_count)

                        upscale_stage_started_at = time.time()
                        if backend_id == "realesrgan-ncnn":
                            upscaled_count = _upscale_ncnn_segment(
                                runtime=runtime,
                                input_dir=segment_input_dir,
                                output_dir=segment_upscaled_dir,
                                model_id=model_id,
                                gpu_id=gpu_id,
                                effective_tile=effective_tile,
                                log=log,
                                cancel_path=cancel_path,
                                pause_path=pause_path,
                                progress_callback=report_upscale_progress,
                            )
                        elif backend_id == "pytorch-image-sr":
                            upscaled_count = _upscale_pytorch_segment(
                                loaded_model=loaded_model,
                                input_dir=segment_input_dir,
                                output_dir=segment_upscaled_dir,
                                effective_tile=effective_tile,
                                cancel_path=cancel_path,
                                pause_path=pause_path,
                                progress_callback=report_upscale_batch,
                            )
                        elif backend_id == "pytorch-video-sr":
                            upscaled_count, external_runtime = _upscale_external_video_segment(
                                input_dir=segment_input_dir,
                                output_dir=segment_upscaled_dir,
                                model_id=model_id,
                                effective_tile=effective_tile,
                                gpu_id=gpu_id,
                                precision_mode=selected_precision_mode,
                                log=log,
                                cancel_path=cancel_path,
                                pause_path=pause_path,
                                progress_callback=report_upscale_progress,
                            )
                            model_runtime = external_runtime
                            effective_precision_mode = str(external_runtime.get("precision", selected_precision_mode))
                        else:
                            raise RuntimeError(f"Backend '{backend_id}' is cataloged but not runnable in the current app build")
                        if upscaled_count != active_segment_plan.expanded_frame_count:
                            raise RuntimeError(
                                f"Expected {active_segment_plan.expanded_frame_count} upscaled frames for segment {active_segment_plan.index + 1}, received {upscaled_count}"
                            )
                        with progress_lock:
                            _record_stage_duration(telemetry_state, "upscale", time.time() - upscale_stage_started_at)
                            progress_state.upscaled_frames = min(total_frames, upscaled_before_segment + active_segment_plan.source_frame_count)
                            upscaled_snapshot = progress_state.upscaled_frames
                            _set_segment_progress(
                                telemetry_state,
                                segment_index=active_segment_plan.index + 1,
                                segment_count=len(interpolation_segments),
                                segment_processed_frames=active_segment_plan.source_frame_count,
                                segment_total_frames=active_segment_plan.source_frame_count,
                                batch_index=None,
                                batch_count=None,
                            )
                        _publish_progress(
                            "upscaling",
                            f"Upscaled segment {active_segment_plan.index + 1}/{len(interpolation_segments)} ({upscaled_snapshot}/{total_frames} source frames)",
                        )
                        frame_stage_output_dir = segment_upscaled_dir
                    else:
                        with progress_lock:
                            upscaled_snapshot = progress_state.upscaled_frames

                    if should_skip_interpolation(
                        input_frame_count=active_segment_plan.expanded_frame_count,
                        target_frame_count=active_segment_plan.expanded_output_frame_count,
                    ):
                        frame_start_number = active_segment_plan.overlap_before_frames + 1
                        frame_limit = active_segment_plan.source_frame_count
                        with progress_lock:
                            progress_state.interpolated_frames = min(total_output_frames, progress_state.interpolated_frames + active_segment_plan.output_frame_count)
                            interpolated_snapshot = progress_state.interpolated_frames
                            _set_segment_progress(
                                telemetry_state,
                                segment_index=active_segment_plan.index + 1,
                                segment_count=len(interpolation_segments),
                                segment_processed_frames=active_segment_plan.output_frame_count,
                                segment_total_frames=active_segment_plan.output_frame_count,
                            )
                        _publish_progress(
                            "interpolating",
                            f"Interpolation skipped for segment {active_segment_plan.index + 1}/{len(interpolation_segments)}; using {active_segment_plan.output_frame_count} source frames",
                        )
                    else:
                        with progress_lock:
                            interpolated_before_segment = progress_state.interpolated_frames

                        def report_interpolation_progress(processed_in_segment: int, _total_in_segment: int) -> None:
                            trimmed_processed = max(0, processed_in_segment - active_segment_plan.output_start_frame)
                            unique_processed = min(active_segment_plan.output_frame_count, trimmed_processed)
                            with progress_lock:
                                current_interpolated_frames = min(total_output_frames, interpolated_before_segment + unique_processed)
                                _set_segment_progress(
                                    telemetry_state,
                                    segment_index=active_segment_plan.index + 1,
                                    segment_count=len(interpolation_segments),
                                    segment_processed_frames=unique_processed,
                                    segment_total_frames=active_segment_plan.output_frame_count,
                                )
                            _publish_progress(
                                "interpolating",
                                f"Interpolating segment {active_segment_plan.index + 1}/{len(interpolation_segments)} ({unique_processed}/{active_segment_plan.output_frame_count} frames)",
                            )

                        interpolate_stage_started_at = time.time()
                        interpolated_count = _run_rife_segment(
                            runtime=runtime,
                            input_dir=frame_stage_output_dir,
                            output_dir=segment_interpolated_dir,
                            target_frame_count=active_segment_plan.expanded_output_frame_count,
                            gpu_id=gpu_id,
                            width=encode_width,
                            height=encode_height,
                            log=log,
                            cancel_path=cancel_path,
                            progress_callback=report_interpolation_progress,
                        )
                        if interpolated_count < active_segment_plan.output_start_frame + active_segment_plan.output_frame_count:
                            raise RuntimeError(
                                f"Interpolated segment {active_segment_plan.index + 1} produced {interpolated_count} frames, which is not enough to trim {active_segment_plan.output_frame_count} frames starting at {active_segment_plan.output_start_frame}."
                            )
                        with progress_lock:
                            _record_stage_duration(telemetry_state, "interpolate", time.time() - interpolate_stage_started_at)
                            progress_state.interpolated_frames = min(total_output_frames, interpolated_before_segment + active_segment_plan.output_frame_count)
                            interpolated_snapshot = progress_state.interpolated_frames
                        frame_stage_output_dir = segment_interpolated_dir
                        frame_start_number = active_segment_plan.output_start_frame + 1
                        frame_limit = active_segment_plan.output_frame_count

                    _queue_put(
                        encode_queue,
                        InterpolationEncodeTask(
                            segment_index=active_segment_plan.index,
                            output_dir=frame_stage_output_dir,
                            output_file=segment_file,
                            frame_start_number=frame_start_number,
                            frame_limit=frame_limit,
                            segment_total_frames=active_segment_plan.output_frame_count,
                            extracted_frames=extracted_snapshot,
                            upscaled_frames=upscaled_snapshot,
                            interpolated_frames=interpolated_snapshot,
                            cleanup_paths=(segment_input_dir, segment_upscaled_dir, segment_interpolated_dir),
                        ),
                    )
            except BaseException as error:  # noqa: BLE001
                _record_error(error)
            finally:
                if loaded_model is not None and getattr(loaded_model.device, "type", "cpu") == "cuda":
                    import torch

                    torch.cuda.synchronize(loaded_model.device)
                _queue_put(encode_queue, sentinel)

        def encoder_worker() -> None:
            try:
                while True:
                    item = _queue_get(encode_queue)
                    if item is sentinel:
                        break
                    if not isinstance(item, InterpolationEncodeTask):
                        continue

                    with progress_lock:
                        encoded_before_segment = progress_state.encoded_frames
                        _set_segment_progress(
                            telemetry_state,
                            segment_index=item.segment_index + 1,
                            segment_count=len(interpolation_segments),
                            segment_processed_frames=0,
                            segment_total_frames=item.segment_total_frames,
                        )
                    encode_stage_started_at = time.time()
                    encoded_count = _encode_segment_video(
                        ffmpeg=str(ffmpeg),
                        upscaled_dir=item.output_dir,
                        output_file=item.output_file,
                        fps=encode_fps,
                        codec=codec,
                        crf=crf,
                        video_encoder_config=video_encoder_config,
                        filter_chain=encode_filter_chain,
                        model_name=model_name,
                        output_mode=output_mode,
                        aspect_ratio_preset=aspect_ratio_preset,
                        resolution_basis=resolution_basis,
                        resolved_width=encode_width,
                        resolved_height=encode_height,
                        crop_left=crop_left,
                        crop_top=crop_top,
                        crop_width=crop_width,
                        crop_height=crop_height,
                        container=container,
                        input_start_number=item.frame_start_number,
                        input_frame_limit=item.frame_limit,
                        log=log,
                        progress_path=progress_path,
                        cancel_path=cancel_path,
                        pause_path=pause_path,
                        total_frames=total_output_frames,
                        extracted_frames=item.extracted_frames,
                        colorized_frames=0,
                        upscaled_frames=item.upscaled_frames,
                        interpolated_frames=item.interpolated_frames,
                        encoded_frames_before_segment=encoded_before_segment,
                        telemetry_state=telemetry_state,
                    )
                    for cleanup_path in item.cleanup_paths:
                        shutil.rmtree(cleanup_path, ignore_errors=True)
                    segment_outputs[item.segment_index] = item.output_file
                    with progress_lock:
                        _record_stage_duration(telemetry_state, "encode", time.time() - encode_stage_started_at)
                        progress_state.encoded_frames = min(total_output_frames, progress_state.encoded_frames + max(encoded_count, item.segment_total_frames))
                        encoded_snapshot = progress_state.encoded_frames
                        _set_segment_progress(
                            telemetry_state,
                            segment_index=item.segment_index + 1,
                            segment_count=len(interpolation_segments),
                            segment_processed_frames=max(encoded_count, item.segment_total_frames),
                            segment_total_frames=item.segment_total_frames,
                        )
                    _publish_progress(
                        "encoding",
                        f"Encoded segment {item.segment_index + 1}/{len(interpolation_segments)} ({encoded_snapshot}/{total_output_frames} frames)",
                    )
            except BaseException as error:  # noqa: BLE001
                _record_error(error)

        _publish_progress("extracting", "Starting overlapped extract/upscale/interpolate/encode pipeline")
        try:
            workers = [
                threading.Thread(target=preprocess_worker, name="interpolation-preprocess", daemon=True),
                threading.Thread(target=encoder_worker, name="interpolation-encoder", daemon=True),
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()

            if not stage_errors.empty():
                raise stage_errors.get()

            segment_files = [segment_file for segment_file in segment_outputs if segment_file is not None]
            if len(segment_files) != len(interpolation_segments):
                raise RuntimeError("Failed to encode one or more interpolation segments")

            concat_stage_started_at = time.time()
            concatenated_frame_count = _concat_segment_videos(
                ffmpeg=str(ffmpeg),
                segment_files=segment_files,
                concat_manifest=concat_manifest,
                output_file=silent_video,
                progress_path=progress_path,
                cancel_path=cancel_path,
                pause_path=pause_path,
                total_frames=total_output_frames,
                extracted_frames=progress_state.extracted_frames,
                colorized_frames=0,
                upscaled_frames=progress_state.upscaled_frames,
                interpolated_frames=progress_state.interpolated_frames,
                encoded_frames=progress_state.encoded_frames,
                log=log,
                telemetry_state=telemetry_state,
            )
            _record_stage_duration(telemetry_state, "remux", time.time() - concat_stage_started_at)
            progress_state.remuxed_frames = min(total_output_frames, concatenated_frame_count)
        finally:
            if loaded_model is not None and getattr(loaded_model.device, "type", "cpu") == "cuda":
                import torch

                torch.cuda.synchronize(loaded_model.device)

        if bool(metadata["hasAudio"]):
            remux_command = [
                ffmpeg,
                "-y",
                "-i",
                str(silent_video),
                "-i",
                source_path,
                *( ["-t", f"{effective_duration:.3f}"] if preview_mode else [] ),
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                "-metadata",
                "upscaler_audio_source=original",
                "-metadata",
                f"upscaler_model={active_model_id}",
                "-metadata",
                f"upscaler_codec={codec}",
                "-metadata",
                f"upscaler_container={container}",
                "-metadata",
                f"upscaler_interpolation_mode={interpolation_mode}",
                "-metadata",
                f"upscaler_interpolation_target_fps={int(round(output_fps))}",
                str(output_file),
            ]
            remux_stage_started_at = time.time()
            remuxed_frame_count, _ = _run_ffmpeg_with_frame_progress(
                remux_command,
                log,
                progress_path,
                cancel_path,
                pause_path,
                phase="remuxing",
                percent_base=97,
                percent_span=3,
                total_frames=total_output_frames,
                message_prefix="Remuxing original audio",
                extracted_frames=progress_state.extracted_frames,
                colorized_frames=progress_state.colorized_frames,
                upscaled_frames=progress_state.upscaled_frames,
                interpolated_frames=progress_state.interpolated_frames,
                encoded_frames=progress_state.encoded_frames,
                remuxed_frames=progress_state.remuxed_frames,
                telemetry_state=telemetry_state,
            )
            _record_stage_duration(telemetry_state, "remux", time.time() - remux_stage_started_at)
        else:
            finalize_command = [
                ffmpeg,
                "-y",
                "-i",
                str(silent_video),
                "-c",
                "copy",
                str(output_file),
            ]
            remux_stage_started_at = time.time()
            remuxed_frame_count, _ = _run_ffmpeg_with_frame_progress(
                finalize_command,
                log,
                progress_path,
                cancel_path,
                pause_path,
                phase="remuxing",
                percent_base=97,
                percent_span=3,
                total_frames=total_output_frames,
                message_prefix="Finalizing video output",
                extracted_frames=progress_state.extracted_frames,
                colorized_frames=progress_state.colorized_frames,
                upscaled_frames=progress_state.upscaled_frames,
                interpolated_frames=progress_state.interpolated_frames,
                encoded_frames=progress_state.encoded_frames,
                remuxed_frames=progress_state.remuxed_frames,
                telemetry_state=telemetry_state,
            )
            _record_stage_duration(telemetry_state, "remux", time.time() - remux_stage_started_at)

        progress_state.remuxed_frames = min(total_output_frames, remuxed_frame_count)
        _write_progress(
            progress_path,
            phase="completed",
            percent=100,
            message="Pipeline completed",
            processed_frames=total_output_frames,
            total_frames=total_output_frames,
            extracted_frames=progress_state.extracted_frames,
            colorized_frames=progress_state.colorized_frames,
            upscaled_frames=progress_state.upscaled_frames,
            interpolated_frames=progress_state.interpolated_frames,
            encoded_frames=progress_state.encoded_frames,
            remuxed_frames=progress_state.remuxed_frames,
            telemetry_state=telemetry_state,
        )

        silent_video.unlink(missing_ok=True)
        for segment_file in segment_files:
            segment_file.unlink(missing_ok=True)
        concat_manifest.unlink(missing_ok=True)
        total_elapsed_seconds = max(0.0, time.time() - telemetry_state.started_at)
        resolved_frame_count = max(
            1,
            progress_state.remuxed_frames or progress_state.encoded_frames or progress_state.interpolated_frames or total_output_frames,
        )
        average_throughput = resolved_frame_count / max(0.001, total_elapsed_seconds)
        output_media = _build_pipeline_media_summary(
            width=encode_width,
            height=encode_height,
            frame_rate=output_fps,
            duration_seconds=effective_duration,
            frame_count=resolved_frame_count,
            has_audio=bool(metadata["hasAudio"]),
            container=container,
            video_codec=codec,
        )
        effective_settings = _build_pipeline_effective_settings(
            backend_id=backend_id,
            quality_preset=preset,
            requested_tile_size=tile_size,
            effective_tile_size=effective_tile,
            requested_precision=requested_precision_mode,
            selected_precision=selected_precision_mode,
            effective_precision=effective_precision_mode,
            precision_source=precision_source,
            processed_duration_seconds=effective_duration,
            preview_start_offset_seconds=preview_window_start_seconds if preview_mode else None,
            segment_frame_limit=segment_frame_limit,
            preview_mode=preview_mode,
            preview_duration_seconds=preview_duration_seconds,
            segment_duration_seconds=segment_duration_seconds,
        )
        return {
            "outputPath": str(output_file),
            "workDir": str(work_dir),
            "executionPath": "rife-ncnn-vulkan",
            "videoEncoder": video_encoder_config.encoder,
            "videoEncoderLabel": video_encoder_config.label,
            "runner": str(model_runtime.get("runner")) if isinstance(model_runtime, dict) and model_runtime.get("runner") else ("ncnn-vulkan" if backend_id == "realesrgan-ncnn" else pytorch_runner or "torch"),
            "precision": effective_precision_mode,
            "torchCompileEnabled": torch_compile_enabled,
            "torchCompileMode": torch_compile_mode,
            "torchCompileCudagraphs": torch_compile_cudagraphs,
            "frameCount": resolved_frame_count,
            "hadAudio": bool(metadata["hasAudio"]),
            "codec": codec,
            "container": container,
            "sourceMedia": source_media,
            "outputMedia": output_media,
            "effectiveSettings": effective_settings,
            "interpolationDiagnostics": {
                "mode": interpolation_mode,
                "sourceFps": source_fps,
                "outputFps": output_fps,
                "sourceFrameCount": progress_state.extracted_frames or total_frames,
                "outputFrameCount": resolved_frame_count,
                "segmentCount": len(interpolation_segments),
                "segmentFrameLimit": segment_frame_limit,
                "segmentOverlapFrames": 1 if len(interpolation_segments) > 1 else 0,
            },
            "runtime": runtime,
            "stageTimings": {
                "extractSeconds": telemetry_state.extract_stage_seconds,
                "upscaleSeconds": telemetry_state.upscale_stage_seconds,
                "interpolateSeconds": telemetry_state.interpolate_stage_seconds,
                "encodeSeconds": telemetry_state.encode_stage_seconds,
                "remuxSeconds": telemetry_state.remux_stage_seconds,
            },
            "resourcePeaks": {
                "processRssBytes": telemetry_state.resources.peak_process_rss_bytes,
                "gpuMemoryUsedBytes": telemetry_state.resources.peak_gpu_memory_used_bytes,
                "gpuMemoryTotalBytes": telemetry_state.resources.gpu_memory_total_bytes,
                "scratchSizeBytes": telemetry_state.resources.peak_scratch_size_bytes,
                "outputSizeBytes": telemetry_state.resources.peak_output_size_bytes,
            },
            "modelRuntime": model_runtime,
            "averageThroughputFps": average_throughput,
            "segmentCount": len(interpolation_segments),
            "segmentFrameLimit": segment_frame_limit,
            "log": log + [
                f"Model: {model_name} ({model_id})",
                f"Interpolation mode: {interpolation_mode}",
                f"Resolved output fps: {encode_fps}",
                f"Resolved output canvas: {encode_width}x{encode_height}",
                f"Chunked interpolation segments: {len(interpolation_segments)} at up to {segment_frame_limit} source frames each with one-frame boundary overlap",
                f"Preview start offset: {preview_window_start_seconds:.2f}s",
                f"Processed duration: {effective_duration:.2f}s",
                f"Average throughput: {average_throughput:.2f} fps",
                f"Rolling throughput: {(telemetry_state.resources.rolling_frames_per_second or 0.0):.2f} fps",
                f"Stage timings: extract {telemetry_state.extract_stage_seconds:.2f}s, upscale {telemetry_state.upscale_stage_seconds:.2f}s, interpolate {telemetry_state.interpolate_stage_seconds:.2f}s, encode {telemetry_state.encode_stage_seconds:.2f}s, remux {telemetry_state.remux_stage_seconds:.2f}s",
            ],
        }

    progress_state = PipelineProgressState()
    telemetry_state = PipelineTelemetryState(
        started_at=time.time(),
        source_path=source_path,
        scratch_path=work_dir,
        output_path=output_file,
        job_id=cache_key,
    )
    progress_lock = threading.Lock()
    _write_progress(
        progress_path,
        phase="queued",
        percent=0,
        message="Job queued",
        processed_frames=0,
        total_frames=total_frames,
        telemetry_state=telemetry_state,
    )

    stage_errors: queue.Queue[BaseException] = queue.Queue()
    stop_event = threading.Event()
    extract_queue: queue.Queue[PipelineSegment | object] = queue.Queue(maxsize=PIPELINE_STAGE_QUEUE_DEPTH)
    encode_queue: queue.Queue[tuple[PipelineSegment, Path] | object] = queue.Queue(maxsize=PIPELINE_STAGE_QUEUE_DEPTH)
    segment_outputs: list[Path | None] = [None] * len(segments)
    sentinel = object()

    def _record_error(error: BaseException) -> None:
        if stage_errors.empty():
            stage_errors.put(error)
        stop_event.set()

    def _queue_put(target_queue: queue.Queue, value: object) -> None:
        while not stop_event.is_set():
            try:
                target_queue.put(value, timeout=0.5)
                return
            except queue.Full:
                continue

    def _queue_get(target_queue: queue.Queue):
        while True:
            if stop_event.is_set() and target_queue.empty():
                return sentinel
            try:
                return target_queue.get(timeout=0.5)
            except queue.Empty:
                if stop_event.is_set():
                    return sentinel

    def _publish_progress(phase: str, message: str) -> None:
        with progress_lock:
            if not _should_publish_stage_progress(phase, progress_state):
                return
            _emit_pipeline_progress(
                progress_path,
                phase=phase,
                message=message,
                total_frames=total_frames,
                progress_state=progress_state,
                colorization_mode=colorization_mode,
                telemetry_state=telemetry_state,
            )

    def extractor_worker() -> None:
        try:
            for segment in segments:
                if stop_event.is_set():
                    break
                with progress_lock:
                    _set_segment_progress(
                        telemetry_state,
                        segment_index=segment.index + 1,
                        segment_count=len(segments),
                        segment_processed_frames=0,
                        segment_total_frames=segment.frame_count,
                    )
                _publish_progress(
                    "extracting",
                    f"Extracting segment {segment.index + 1}/{len(segments)} (0/{segment.frame_count} frames)",
                )
                segment_input_dir = segment_root / f"segment_{segment.index:04d}" / "in"
                stage_started_at = time.time()
                extracted_count = _extract_segment_frames(
                    ffmpeg=str(ffmpeg),
                    source_path=source_path,
                    segment=segment,
                    source_start_seconds=preview_window_start_seconds,
                    input_dir=segment_input_dir,
                    log=log,
                    cancel_path=cancel_path,
                    pause_path=pause_path,
                )
                with progress_lock:
                    _record_stage_duration(telemetry_state, "extract", time.time() - stage_started_at)
                    progress_state.extracted_frames = min(total_frames, progress_state.extracted_frames + extracted_count)
                    _set_segment_progress(
                        telemetry_state,
                        segment_index=segment.index + 1,
                        segment_count=len(segments),
                        segment_processed_frames=extracted_count,
                        segment_total_frames=segment.frame_count,
                    )
                _publish_progress(
                    "extracting",
                    f"Extracted segment {segment.index + 1}/{len(segments)} ({progress_state.extracted_frames}/{total_frames} frames)",
                )
                _queue_put(extract_queue, segment)
        except BaseException as error:  # noqa: BLE001
            _record_error(error)
        finally:
            _queue_put(extract_queue, sentinel)

    def upscaler_worker() -> None:
        loaded_model = None
        loaded_colorizer = None
        colorizer_runtime: dict[str, object] | None = None
        try:
            if colorization_enabled:
                from upscaler_worker.models.colorizers import colorize_directory, load_runtime_colorizer

                loaded_colorizer = load_runtime_colorizer(
                    colorizer_model_id,
                    gpu_id,
                    selected_precision_mode,
                    log,
                    reference_image_paths=selected_color_reference_images,
                    deepremaster_processing_mode=deepremaster_processing_mode,
                )
                colorizer_model_ref = getattr(loaded_colorizer, "repo_id", None)
                if colorizer_model_ref is None:
                    colorizer_model_ref = str(getattr(loaded_colorizer, "checkpoint_path", "")) or None
                colorizer_runtime = {
                    "runner": "torch",
                    "precision": loaded_colorizer.precision_mode,
                    "repoId": colorizer_model_ref,
                    "inputSize": getattr(loaded_colorizer, "input_size", None),
                }
                if colorization_mode == "colorizeOnly":
                    model_runtime = dict(colorizer_runtime)
                    effective_precision_mode = loaded_colorizer.precision_mode

            if backend_id == "pytorch-image-sr":
                if resolved_pytorch_execution_path == PYTORCH_EXECUTION_PATH_STREAMING:
                    raise NotImplementedError(
                        "PyTorch streaming execution path is not implemented yet. Use file-io for now."
                    )
                from upscaler_worker.models.pytorch_sr import load_runtime_model

                loaded_model = load_runtime_model(
                    model_id,
                    gpu_id,
                    False,
                    effective_tile,
                    log,
                    preset=preset,
                    torch_compile_enabled=torch_compile_enabled,
                    torch_compile_mode=torch_compile_mode,
                    torch_compile_cudagraphs=torch_compile_cudagraphs,
                    bf16=False,
                    precision=selected_precision_mode,
                    pytorch_runner=pytorch_runner,
                    channels_last_enabled=channels_last,
                )
                log.append(f"Loaded PyTorch model checkpoint: {loaded_model.checkpoint_path}")
                effective_precision_mode = loaded_model.precision_mode
                model_runtime = {
                    "runner": loaded_model.runner,
                    "precision": loaded_model.precision_mode,
                    "dtype": str(loaded_model.dtype).replace("torch.", ""),
                    "frameBatchSize": loaded_model.frame_batch_size,
                    "channelsLast": loaded_model.channels_last,
                    "torchCompileRequested": loaded_model.torch_compile_requested,
                    "torchCompileEnabled": loaded_model.torch_compile_enabled,
                    "torchCompileMode": loaded_model.torch_compile_mode,
                    "torchCompileCudagraphs": loaded_model.torch_compile_cudagraphs,
                }
                if colorizer_runtime is not None:
                    model_runtime["colorizer"] = colorizer_runtime
            elif backend_id == "pytorch-video-sr":
                model_runtime = {
                    "runner": "external-command",
                    "precision": selected_precision_mode,
                }
                effective_precision_mode = selected_precision_mode
                if colorizer_runtime is not None:
                    model_runtime["colorizer"] = colorizer_runtime

            while True:
                item = _queue_get(extract_queue)
                if item is sentinel:
                    break
                segment = item
                if not isinstance(segment, PipelineSegment):
                    continue

                segment_dir = segment_root / f"segment_{segment.index:04d}"
                segment_input_dir = segment_dir / "in"
                segment_colorized_dir = segment_dir / "colorized"
                segment_output_dir = segment_dir / "out"
                stage_input_dir = segment_input_dir
                stage_started_at = time.time()

                if colorization_enabled:
                    with progress_lock:
                        colorized_before_segment = progress_state.colorized_frames
                        _set_segment_progress(
                            telemetry_state,
                            segment_index=segment.index + 1,
                            segment_count=len(segments),
                            segment_processed_frames=0,
                            segment_total_frames=segment.frame_count,
                        )
                    _publish_progress(
                        "colorizing",
                        f"Colorizing segment {segment.index + 1}/{len(segments)} (0/{segment.frame_count} frames)",
                    )

                    def report_colorize_progress(processed_in_segment: int, total_in_segment: int) -> None:
                        with progress_lock:
                            progress_state.colorized_frames = min(total_frames, colorized_before_segment + processed_in_segment)
                            _set_segment_progress(
                                telemetry_state,
                                segment_index=segment.index + 1,
                                segment_count=len(segments),
                                segment_processed_frames=processed_in_segment,
                                segment_total_frames=total_in_segment,
                            )
                        _publish_progress(
                            "colorizing",
                            f"Colorizing segment {segment.index + 1}/{len(segments)} ({processed_in_segment}/{total_in_segment} frames)",
                        )

                    colorize_stage_started_at = time.time()
                    colorized_count = colorize_directory(
                        loaded_model=loaded_colorizer,
                        input_dir=segment_input_dir,
                        output_dir=segment_colorized_dir,
                        cancel_path=cancel_path,
                        pause_path=pause_path,
                        progress_callback=report_colorize_progress,
                    )
                    with progress_lock:
                        _record_stage_duration(telemetry_state, "colorize", time.time() - colorize_stage_started_at)
                        progress_state.colorized_frames = min(total_frames, colorized_before_segment + colorized_count)
                        _set_segment_progress(
                            telemetry_state,
                            segment_index=segment.index + 1,
                            segment_count=len(segments),
                            segment_processed_frames=colorized_count,
                            segment_total_frames=segment.frame_count,
                        )
                    _publish_progress(
                        "colorizing",
                        f"Colorized segment {segment.index + 1}/{len(segments)} ({progress_state.colorized_frames}/{total_frames} frames)",
                    )
                    stage_input_dir = segment_colorized_dir

                if colorization_mode != "colorizeOnly":
                    upscale_stage_started_at = time.time()
                    with progress_lock:
                        upscaled_before_segment = progress_state.upscaled_frames
                        _set_segment_progress(
                            telemetry_state,
                            segment_index=segment.index + 1,
                            segment_count=len(segments),
                            segment_processed_frames=0,
                            segment_total_frames=segment.frame_count,
                        )
                    _publish_progress(
                        "upscaling",
                        f"Upscaling segment {segment.index + 1}/{len(segments)} (0/{segment.frame_count} frames)",
                    )

                    if backend_id == "realesrgan-ncnn":
                        def report_upscale_progress(processed_in_segment: int, total_in_segment: int, batch_index: int | None = None, batch_count: int | None = None) -> None:
                            with progress_lock:
                                progress_state.upscaled_frames = min(total_frames, upscaled_before_segment + processed_in_segment)
                                _set_segment_progress(
                                    telemetry_state,
                                    segment_index=segment.index + 1,
                                    segment_count=len(segments),
                                    segment_processed_frames=processed_in_segment,
                                    segment_total_frames=total_in_segment,
                                    batch_index=batch_index,
                                    batch_count=batch_count,
                                )
                            batch_label = f" batch {batch_index}/{batch_count}" if batch_index is not None and batch_count is not None else ""
                            _publish_progress(
                                "upscaling",
                                f"Upscaling segment {segment.index + 1}/{len(segments)}{batch_label} ({processed_in_segment}/{total_in_segment} frames)",
                            )

                        upscaled_count = _upscale_ncnn_segment(
                            runtime=runtime,
                            input_dir=stage_input_dir,
                            output_dir=segment_output_dir,
                            model_id=model_id,
                            gpu_id=gpu_id,
                            effective_tile=effective_tile,
                            log=log,
                            cancel_path=cancel_path,
                            pause_path=pause_path,
                            progress_callback=report_upscale_progress,
                        )
                    elif backend_id == "pytorch-image-sr":
                        def report_upscale_progress(processed_in_segment: int, total_in_segment: int, batch_index: int | None = None, batch_count: int | None = None) -> None:
                            with progress_lock:
                                progress_state.upscaled_frames = min(total_frames, upscaled_before_segment + processed_in_segment)
                                _set_segment_progress(
                                    telemetry_state,
                                    segment_index=segment.index + 1,
                                    segment_count=len(segments),
                                    segment_processed_frames=processed_in_segment,
                                    segment_total_frames=total_in_segment,
                                    batch_index=batch_index,
                                    batch_count=batch_count,
                                )
                            batch_label = f" batch {batch_index}/{batch_count}" if batch_index is not None and batch_count is not None else ""
                            _publish_progress(
                                "upscaling",
                                f"Upscaling segment {segment.index + 1}/{len(segments)}{batch_label} ({processed_in_segment}/{total_in_segment} frames)",
                            )

                        def report_upscale_batch(batch_index: int, batch_count: int, processed_in_segment: int, total_in_segment: int) -> None:
                            report_upscale_progress(processed_in_segment, total_in_segment, batch_index, batch_count)

                        upscaled_count = _upscale_pytorch_segment(
                            loaded_model=loaded_model,
                            input_dir=stage_input_dir,
                            output_dir=segment_output_dir,
                            effective_tile=effective_tile,
                            cancel_path=cancel_path,
                            pause_path=pause_path,
                            progress_callback=report_upscale_batch,
                        )
                    elif backend_id == "pytorch-video-sr":
                        def report_upscale_progress(processed_in_segment: int, total_in_segment: int, batch_index: int | None = None, batch_count: int | None = None) -> None:
                            with progress_lock:
                                progress_state.upscaled_frames = min(total_frames, upscaled_before_segment + processed_in_segment)
                                _set_segment_progress(
                                    telemetry_state,
                                    segment_index=segment.index + 1,
                                    segment_count=len(segments),
                                    segment_processed_frames=processed_in_segment,
                                    segment_total_frames=total_in_segment,
                                    batch_index=batch_index,
                                    batch_count=batch_count,
                                )
                            batch_label = f" batch {batch_index}/{batch_count}" if batch_index is not None and batch_count is not None else ""
                            _publish_progress(
                                "upscaling",
                                f"Upscaling segment {segment.index + 1}/{len(segments)}{batch_label} ({processed_in_segment}/{total_in_segment} frames)",
                            )

                        upscaled_count, external_runtime = _upscale_external_video_segment(
                            input_dir=stage_input_dir,
                            output_dir=segment_output_dir,
                            model_id=model_id,
                            effective_tile=effective_tile,
                            gpu_id=gpu_id,
                            precision_mode=selected_precision_mode,
                            log=log,
                            cancel_path=cancel_path,
                            pause_path=pause_path,
                            progress_callback=report_upscale_progress,
                        )
                        model_runtime = external_runtime
                        effective_precision_mode = str(external_runtime.get("precision", selected_precision_mode))
                        if colorizer_runtime is not None:
                            model_runtime["colorizer"] = colorizer_runtime
                    else:
                        raise RuntimeError(f"Backend '{backend_id}' is cataloged but not runnable in the current app build")

                    with progress_lock:
                        _record_stage_duration(telemetry_state, "upscale", time.time() - upscale_stage_started_at)
                        progress_state.upscaled_frames = min(total_frames, upscaled_before_segment + upscaled_count)
                        _set_segment_progress(
                            telemetry_state,
                            segment_index=segment.index + 1,
                            segment_count=len(segments),
                            segment_processed_frames=upscaled_count,
                            segment_total_frames=segment.frame_count,
                            batch_index=None,
                            batch_count=None,
                        )
                    _publish_progress(
                        "upscaling",
                        f"Upscaled segment {segment.index + 1}/{len(segments)} ({progress_state.upscaled_frames}/{total_frames} frames)",
                    )
                else:
                    segment_output_dir = stage_input_dir

                shutil.rmtree(segment_input_dir, ignore_errors=True)
                if stage_input_dir != segment_input_dir and colorization_mode != "colorizeOnly":
                    shutil.rmtree(stage_input_dir, ignore_errors=True)
                _queue_put(encode_queue, (segment, segment_output_dir))
        except BaseException as error:  # noqa: BLE001
            _record_error(error)
        finally:
            if loaded_model is not None and getattr(loaded_model.device, "type", "cpu") == "cuda":
                import torch

                torch.cuda.synchronize(loaded_model.device)
            if loaded_colorizer is not None and getattr(loaded_colorizer.device, "type", "cpu") == "cuda":
                import torch

                torch.cuda.synchronize(loaded_colorizer.device)
            _queue_put(encode_queue, sentinel)

    def encoder_worker() -> None:
        try:
            while True:
                item = _queue_get(encode_queue)
                if item is sentinel:
                    break
                if not isinstance(item, tuple):
                    continue
                segment, segment_output_dir = item
                segment_file = encoded_dir / f"segment_{segment.index:04d}.{PIPELINE_INTERMEDIATE_CONTAINER}"
                stage_started_at = time.time()
                with progress_lock:
                    _set_segment_progress(
                        telemetry_state,
                        segment_index=segment.index + 1,
                        segment_count=len(segments),
                        segment_processed_frames=0,
                        segment_total_frames=segment.frame_count,
                    )
                encoded_count = _encode_segment_video(
                    ffmpeg=str(ffmpeg),
                    upscaled_dir=segment_output_dir,
                    output_file=segment_file,
                    fps=fps,
                    codec=codec,
                    crf=crf,
                    video_encoder_config=video_encoder_config,
                    filter_chain=filter_chain,
                    model_name=model_name,
                    output_mode=output_mode,
                    aspect_ratio_preset=aspect_ratio_preset,
                    resolution_basis=resolution_basis,
                    resolved_width=resolved_width,
                    resolved_height=resolved_height,
                    crop_left=crop_left,
                    crop_top=crop_top,
                    crop_width=crop_width,
                    crop_height=crop_height,
                    container=container,
                    log=log,
                    progress_path=progress_path,
                    cancel_path=cancel_path,
                        pause_path=pause_path,
                    total_frames=total_frames,
                    extracted_frames=progress_state.extracted_frames,
                    colorized_frames=progress_state.colorized_frames,
                    upscaled_frames=progress_state.upscaled_frames,
                    interpolated_frames=progress_state.interpolated_frames,
                    encoded_frames_before_segment=progress_state.encoded_frames,
                    telemetry_state=telemetry_state,
                )
                shutil.rmtree(segment_output_dir, ignore_errors=True)
                segment_outputs[segment.index] = segment_file
                with progress_lock:
                    _record_stage_duration(telemetry_state, "encode", time.time() - stage_started_at)
                    progress_state.encoded_frames = min(total_frames, progress_state.encoded_frames + max(encoded_count, segment.frame_count))
                    _set_segment_progress(
                        telemetry_state,
                        segment_index=segment.index + 1,
                        segment_count=len(segments),
                        segment_processed_frames=max(encoded_count, segment.frame_count),
                        segment_total_frames=segment.frame_count,
                    )
                _publish_progress(
                    "encoding",
                    f"Encoded segment {segment.index + 1}/{len(segments)} ({progress_state.encoded_frames}/{total_frames} frames)",
                )
        except BaseException as error:  # noqa: BLE001
            _record_error(error)

    if colorization_mode == "colorizeOnly":
        _publish_progress("extracting", "Starting overlapped extract/colorize/encode pipeline")
    elif colorization_enabled:
        _publish_progress("extracting", "Starting overlapped extract/colorize/upscale/encode pipeline")
    else:
        _publish_progress("extracting", "Starting overlapped extract/upscale/encode pipeline")

    concat_manifest: Path | None = None
    if backend_id == "pytorch-image-sr" and resolved_pytorch_execution_path == PYTORCH_EXECUTION_PATH_STREAMING:
        loaded_model = preloaded_pytorch_model
        if loaded_model is None:
            from upscaler_worker.models.pytorch_sr import load_runtime_model

            loaded_model = load_runtime_model(
                model_id,
                gpu_id,
                False,
                effective_tile,
                log,
                preset=preset,
                torch_compile_enabled=torch_compile_enabled,
                torch_compile_mode=torch_compile_mode,
                torch_compile_cudagraphs=torch_compile_cudagraphs,
                bf16=False,
                precision=selected_precision_mode,
                pytorch_runner=pytorch_runner,
                channels_last_enabled=channels_last,
            )
            log.append(f"Loaded PyTorch model checkpoint: {loaded_model.checkpoint_path}")
        effective_precision_mode = loaded_model.precision_mode
        model_runtime = {
            "runner": loaded_model.runner,
            "precision": loaded_model.precision_mode,
            "dtype": str(loaded_model.dtype).replace("torch.", ""),
            "frameBatchSize": loaded_model.frame_batch_size,
            "channelsLast": loaded_model.channels_last,
            "torchCompileRequested": loaded_model.torch_compile_requested,
            "torchCompileEnabled": loaded_model.torch_compile_enabled,
            "torchCompileMode": loaded_model.torch_compile_mode,
            "torchCompileCudagraphs": loaded_model.torch_compile_cudagraphs,
        }
        _set_segment_progress(
            telemetry_state,
            segment_index=1,
            segment_count=1,
            segment_processed_frames=0,
            segment_total_frames=total_frames,
        )
        _publish_progress("extracting", "Starting streaming decode/upscale/encode pipeline")
        encoded_frame_count = _run_streaming_pytorch_pipeline(
            ffmpeg=str(ffmpeg),
            source_path=source_path,
            source_width=int(metadata["width"]),
            source_height=int(metadata["height"]),
            fps=fps,
            total_frames=total_frames,
            effective_duration=effective_duration,
            source_start_seconds=preview_window_start_seconds,
            silent_video=silent_video,
            codec=codec,
            crf=crf,
            video_encoder_config=video_encoder_config,
            filter_chain=filter_chain,
            loaded_model=loaded_model,
            effective_tile=effective_tile,
            log=log,
            progress_path=progress_path,
            cancel_path=cancel_path,
            pause_path=pause_path,
            progress_state=progress_state,
            telemetry_state=telemetry_state,
        )
        progress_state.remuxed_frames = min(total_frames, encoded_frame_count)
        if getattr(loaded_model.device, "type", "cpu") == "cuda":
            import torch

            torch.cuda.synchronize(loaded_model.device)
        _publish_progress("remuxing", "Streaming video encoded; preparing final remux")
    else:
        workers = [
            threading.Thread(target=extractor_worker, name="extractor", daemon=True),
            threading.Thread(target=upscaler_worker, name="upscaler", daemon=True),
            threading.Thread(target=encoder_worker, name="encoder", daemon=True),
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        if not stage_errors.empty():
            raise stage_errors.get()

        segment_files = [segment_file for segment_file in segment_outputs if segment_file is not None]
        if len(segment_files) != len(segments):
            raise RuntimeError("Failed to encode one or more pipeline segments")

        _publish_progress("remuxing", "Finalizing encoded segments")
        concat_manifest = encoded_dir / "segments.txt"
        remux_stage_started_at = time.time()
        concatenated_frame_count = _concat_segment_videos(
            ffmpeg=str(ffmpeg),
            segment_files=segment_files,
            concat_manifest=concat_manifest,
            output_file=silent_video,
            progress_path=progress_path,
            cancel_path=cancel_path,
            pause_path=pause_path,
            total_frames=total_frames,
            extracted_frames=progress_state.extracted_frames,
            colorized_frames=progress_state.colorized_frames,
            upscaled_frames=progress_state.upscaled_frames,
            interpolated_frames=progress_state.interpolated_frames,
            encoded_frames=progress_state.encoded_frames,
            log=log,
            telemetry_state=telemetry_state,
        )
        with progress_lock:
            _record_stage_duration(telemetry_state, "remux", time.time() - remux_stage_started_at)
            progress_state.remuxed_frames = min(total_frames, concatenated_frame_count)
        _publish_progress("remuxing", "Concatenated silent segments")

    if bool(metadata["hasAudio"]):
        _publish_progress("remuxing", "Re-syncing original audio")
        remux_command = [
                ffmpeg,
                "-y",
                "-i",
                str(silent_video),
            *( ["-ss", f"{preview_window_start_seconds:.6f}"] if preview_mode and preview_window_start_seconds > 0 else [] ),
            *( ["-ss", f"{preview_window_start_seconds:.6f}"] if preview_mode and preview_window_start_seconds > 0 else [] ),
                "-i",
                source_path,
                *( ["-t", f"{effective_duration:.3f}"] if preview_mode else [] ),
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                "-metadata",
                "upscaler_audio_source=original",
                "-metadata",
                f"upscaler_model={active_model_id}",
                "-metadata",
                f"upscaler_codec={codec}",
                "-metadata",
                f"upscaler_container={container}",
                str(output_file),
            ]
        remux_stage_started_at = time.time()
        remuxed_frame_count, _ = _run_ffmpeg_with_frame_progress(
            remux_command,
            log,
            progress_path,
            cancel_path,
            pause_path,
            phase="remuxing",
            percent_base=97,
            percent_span=3,
            total_frames=total_frames,
            message_prefix="Remuxing original audio",
            extracted_frames=progress_state.extracted_frames,
            colorized_frames=progress_state.colorized_frames,
            upscaled_frames=progress_state.upscaled_frames,
            interpolated_frames=progress_state.interpolated_frames,
            encoded_frames=progress_state.encoded_frames,
            remuxed_frames=progress_state.remuxed_frames,
            telemetry_state=telemetry_state,
        )
        with progress_lock:
            _record_stage_duration(telemetry_state, "remux", time.time() - remux_stage_started_at)
    else:
        finalize_command = [
            ffmpeg,
            "-y",
            "-i",
            str(silent_video),
            "-c",
            "copy",
            str(output_file),
        ]
        remux_stage_started_at = time.time()
        remuxed_frame_count, _ = _run_ffmpeg_with_frame_progress(
            finalize_command,
            log,
            progress_path,
            cancel_path,
            pause_path,
            phase="remuxing",
            percent_base=97,
            percent_span=3,
            total_frames=total_frames,
            message_prefix="Finalizing video output",
            extracted_frames=progress_state.extracted_frames,
            colorized_frames=progress_state.colorized_frames,
            upscaled_frames=progress_state.upscaled_frames,
            interpolated_frames=progress_state.interpolated_frames,
            encoded_frames=progress_state.encoded_frames,
            remuxed_frames=progress_state.remuxed_frames,
            telemetry_state=telemetry_state,
        )
        with progress_lock:
            _record_stage_duration(telemetry_state, "remux", time.time() - remux_stage_started_at)

    progress_state.remuxed_frames = min(total_frames, remuxed_frame_count)
    _write_progress(
        progress_path,
        phase="completed",
        percent=100,
        message="Pipeline completed",
        processed_frames=total_frames,
        total_frames=total_frames,
        extracted_frames=progress_state.extracted_frames,
        colorized_frames=progress_state.colorized_frames,
        upscaled_frames=progress_state.upscaled_frames,
        encoded_frames=progress_state.encoded_frames,
        remuxed_frames=progress_state.remuxed_frames,
        telemetry_state=telemetry_state,
    )

    if concat_manifest is not None:
        for segment_file in segment_files:
            Path(segment_file).unlink(missing_ok=True)
        concat_manifest.unlink(missing_ok=True)
    silent_video.unlink(missing_ok=True)

    total_elapsed_seconds = max(0.0, time.time() - telemetry_state.started_at)
    resolved_frame_count = max(1, progress_state.remuxed_frames or progress_state.encoded_frames or total_frames)
    average_throughput = resolved_frame_count / max(0.001, total_elapsed_seconds)
    output_media = _build_pipeline_media_summary(
        width=resolved_width,
        height=resolved_height,
        frame_rate=float(metadata["frameRate"]),
        duration_seconds=effective_duration,
        frame_count=resolved_frame_count,
        has_audio=bool(metadata["hasAudio"]),
        container=container,
        video_codec=codec,
    )
    effective_settings = _build_pipeline_effective_settings(
        backend_id=backend_id,
        quality_preset=preset,
        requested_tile_size=tile_size,
        effective_tile_size=effective_tile,
        requested_precision=requested_precision_mode,
        selected_precision=selected_precision_mode,
        effective_precision=effective_precision_mode,
        precision_source=precision_source,
        processed_duration_seconds=effective_duration,
        preview_start_offset_seconds=preview_window_start_seconds if preview_mode else None,
        segment_frame_limit=segment_frame_limit,
        preview_mode=preview_mode,
        preview_duration_seconds=preview_duration_seconds,
        segment_duration_seconds=segment_duration_seconds,
    )

    return {
        "outputPath": str(output_file),
        "workDir": str(work_dir),
        "executionPath": resolved_pytorch_execution_path or ("realesrgan-ncnn-vulkan" if backend_id == "realesrgan-ncnn" else "file-io" if backend_id == "pytorch-image-colorization" else "external-command"),
        "videoEncoder": video_encoder_config.encoder,
        "videoEncoderLabel": video_encoder_config.label,
        "runner": str(model_runtime.get("runner")) if isinstance(model_runtime, dict) and model_runtime.get("runner") else ("ncnn-vulkan" if backend_id == "realesrgan-ncnn" else "torch" if backend_id == "pytorch-image-colorization" else pytorch_runner or "torch"),
        "precision": effective_precision_mode,
        "torchCompileEnabled": torch_compile_enabled,
        "torchCompileMode": torch_compile_mode,
        "torchCompileCudagraphs": torch_compile_cudagraphs,
        "frameCount": resolved_frame_count,
        "hadAudio": bool(metadata["hasAudio"]),
        "codec": codec,
        "container": container,
        "sourceMedia": source_media,
        "outputMedia": output_media,
        "effectiveSettings": effective_settings,
        "runtime": runtime,
        "stageTimings": {
            "extractSeconds": telemetry_state.extract_stage_seconds,
            "colorizeSeconds": telemetry_state.colorize_stage_seconds,
            "upscaleSeconds": telemetry_state.upscale_stage_seconds,
            "interpolateSeconds": telemetry_state.interpolate_stage_seconds,
            "encodeSeconds": telemetry_state.encode_stage_seconds,
            "remuxSeconds": telemetry_state.remux_stage_seconds,
        },
        "resourcePeaks": {
            "processRssBytes": telemetry_state.resources.peak_process_rss_bytes,
            "gpuMemoryUsedBytes": telemetry_state.resources.peak_gpu_memory_used_bytes,
            "gpuMemoryTotalBytes": telemetry_state.resources.gpu_memory_total_bytes,
            "scratchSizeBytes": telemetry_state.resources.peak_scratch_size_bytes,
            "outputSizeBytes": telemetry_state.resources.peak_output_size_bytes,
        },
        "modelRuntime": model_runtime,
        "averageThroughputFps": average_throughput,
        "segmentCount": len(segments),
        "segmentFrameLimit": segment_frame_limit,
        "log": log + [
            f"Model: {model_name} ({active_model_id})",
            *( [f"Colorizer: {model_label(colorizer_model_id)} ({colorizer_model_id})"] if colorization_enabled and colorizer_model_id else [] ),
            f"Execution path: {resolved_pytorch_execution_path or ('realesrgan-ncnn-vulkan' if backend_id == 'realesrgan-ncnn' else 'file-io' if backend_id == 'pytorch-image-colorization' else 'external-command')}",
            f"Runner: {str(model_runtime.get('runner')) if isinstance(model_runtime, dict) and model_runtime.get('runner') else ('ncnn-vulkan' if backend_id == 'realesrgan-ncnn' else 'torch' if backend_id == 'pytorch-image-colorization' else pytorch_runner or 'torch')}",
            f"Precision: {effective_precision_mode}",
            f"Quality policy: tile {effective_tile}, precision {effective_precision_mode} ({precision_source})",
            f"Resolved output canvas: {resolved_width}x{resolved_height} ({resolved_aspect_ratio:.4f}:1)",
            f"Chunked pipeline segments: {len(segments)} at up to {segment_frame_limit} frames each",
            f"Preview mode: {'on' if preview_mode else 'off'}",
            f"Preview start offset: {preview_window_start_seconds:.2f}s",
            f"Segment duration target: {'single preview segment' if preview_mode else f'{segment_duration_seconds or PIPELINE_SEGMENT_TARGET_SECONDS:.2f}s'}",
            f"Processed duration: {effective_duration:.2f}s",
            f"Average throughput: {average_throughput:.2f} fps",
            f"Rolling throughput: {(telemetry_state.resources.rolling_frames_per_second or 0.0):.2f} fps",
            f"Stage timings: extract {telemetry_state.extract_stage_seconds:.2f}s, colorize {telemetry_state.colorize_stage_seconds:.2f}s, upscale {telemetry_state.upscale_stage_seconds:.2f}s, interpolate {telemetry_state.interpolate_stage_seconds:.2f}s, encode {telemetry_state.encode_stage_seconds:.2f}s, remux {telemetry_state.remux_stage_seconds:.2f}s",
        ],
    }
