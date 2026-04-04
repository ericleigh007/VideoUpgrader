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

from upscaler_worker.cancellation import JobCancelledError, cancellation_requested, ensure_not_cancelled, terminate_process
from upscaler_worker.media import probe_video
from upscaler_worker.model_catalog import ensure_runnable_model, model_backend_id
from upscaler_worker.models.pytorch_sr import resolve_precision_mode
from upscaler_worker.models.realesrgan import model_label
from upscaler_worker.runtime import ensure_runtime_assets, repo_root


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
class StreamingFrameBatch:
    batch_index: int
    frames: list[np.ndarray]


@dataclass(frozen=True)
class StreamingPixelBatch:
    batch_index: int
    pixels: list[np.ndarray]


@dataclass
class PipelineProgressState:
    extracted_frames: int = 0
    upscaled_frames: int = 0
    encoded_frames: int = 0
    remuxed_frames: int = 0


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
    scratch_path: Path
    output_path: Path
    resources: PipelineTelemetryResources = field(default_factory=PipelineTelemetryResources)
    segment_index: int | None = None
    segment_count: int | None = None
    segment_processed_frames: int | None = None
    segment_total_frames: int | None = None
    batch_index: int | None = None
    batch_count: int | None = None
    extract_stage_seconds: float = 0.0
    upscale_stage_seconds: float = 0.0
    encode_stage_seconds: float = 0.0
    remux_stage_seconds: float = 0.0


def _round_dimension(value: float) -> int:
    rounded = max(2, int(round(value)))
    return rounded if rounded % 2 == 0 else rounded + 1


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


def _run(command: list[str], log: list[str], cancel_path: str | None = None) -> None:
    log.append("$ " + " ".join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
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
            if cancellation_requested(cancel_path):
                terminate_process(process)
                raise JobCancelledError("Job cancelled by user")
            time.sleep(0.1)
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
        "upscaleStageSeconds": telemetry_state.upscale_stage_seconds,
        "encodeStageSeconds": telemetry_state.encode_stage_seconds,
        "remuxStageSeconds": telemetry_state.remux_stage_seconds,
    }


def _record_stage_duration(telemetry_state: PipelineTelemetryState, stage: str, duration_seconds: float) -> None:
    if duration_seconds <= 0:
        return
    if stage == "extract":
        telemetry_state.extract_stage_seconds += duration_seconds
        return
    if stage == "upscale":
        telemetry_state.upscale_stage_seconds += duration_seconds
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
    upscaled_frames: int = 0,
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
        "upscaledFrames": upscaled_frames,
        "encodedFrames": encoded_frames,
        "remuxedFrames": remuxed_frames,
    }
    if telemetry_state is not None:
        payload.update(_sample_progress_telemetry(telemetry_state, processed_frames=processed_frames, total_frames=total_frames))
    target.write_text(json.dumps(payload), encoding="utf-8")


def _run_ffmpeg_with_frame_progress(
    command: list[str],
    log: list[str],
    progress_path: str | None,
    cancel_path: str | None,
    *,
    phase: str,
    percent_base: int,
    percent_span: int,
    total_frames: int,
    message_prefix: str,
    extracted_frames: int,
    upscaled_frames: int,
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
                upscaled_frames=upscaled_frames,
                encoded_frames=stage_progress_frames if phase == "encoding" else encoded_frames,
                remuxed_frames=stage_progress_frames if phase == "remuxing" else remuxed_frames,
                telemetry_state=telemetry_state,
            )

    reader_thread = threading.Thread(target=reader, name="ffmpeg-frame-progress-reader", daemon=True)
    reader_thread.start()

    try:
        while process.poll() is None:
            if cancellation_requested(cancel_path):
                terminate_process(process)
                raise JobCancelledError("Job cancelled by user")
            reader_thread.join(timeout=0.1)
        reader_thread.join(timeout=1)
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
) -> None:
    _run(command, log, cancel_path)


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


def _pipeline_ratio(processed_frames: int, total_frames: int) -> float:
    if total_frames <= 0:
        return 0.0
    return max(0.0, min(1.0, processed_frames / total_frames))


def _pipeline_percent(total_frames: int, progress_state: PipelineProgressState) -> int:
    extract_weight = 10
    upscale_weight = 70
    encode_weight = 15
    remux_weight = 5
    aggregate = (
        _pipeline_ratio(progress_state.extracted_frames, total_frames) * extract_weight
        + _pipeline_ratio(progress_state.upscaled_frames, total_frames) * upscale_weight
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
    telemetry_state: PipelineTelemetryState | None = None,
) -> None:
    _write_progress(
        progress_path,
        phase=phase,
        percent=_pipeline_percent(total_frames, progress_state),
        message=message,
        processed_frames=max(
            progress_state.extracted_frames,
            progress_state.upscaled_frames,
            progress_state.encoded_frames,
            progress_state.remuxed_frames,
        ),
        total_frames=total_frames,
        extracted_frames=progress_state.extracted_frames,
        upscaled_frames=progress_state.upscaled_frames,
        encoded_frames=progress_state.encoded_frames,
        remuxed_frames=progress_state.remuxed_frames,
        telemetry_state=telemetry_state,
    )


def _extract_segment_frames(
    *,
    ffmpeg: str,
    source_path: str,
    segment: PipelineSegment,
    input_dir: Path,
    log: list[str],
    cancel_path: str | None,
) -> int:
    input_dir.mkdir(parents=True, exist_ok=True)
    extract_command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{segment.start_seconds:.6f}",
        "-i",
        source_path,
        "-map",
        "0:v:0",
        "-frames:v",
        str(segment.frame_count),
        str(input_dir / "frame_%08d.png"),
    ]
    _run(extract_command, log, cancel_path)
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
    _run_realesrgan_batch(realesrgan_command, log, cancel_path)
    return len(extracted_frames)


def _upscale_pytorch_segment(
    *,
    loaded_model,
    input_dir: Path,
    output_dir: Path,
    effective_tile: int,
    cancel_path: str | None,
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


def _run_streaming_pytorch_pipeline(
    *,
    ffmpeg: str,
    source_path: str,
    source_width: int,
    source_height: int,
    fps: str,
    total_frames: int,
    effective_duration: float,
    silent_video: Path,
    codec: str,
    crf: int,
    filter_chain: str | None,
    loaded_model,
    effective_tile: int,
    log: list[str],
    progress_path: str | None,
    cancel_path: str | None,
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
    if effective_duration > 0:
        decode_command[6:6] = ["-t", f"{effective_duration:.6f}"]

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

    video_encoder = "libx265" if codec == "h265" else "libx264"
    encode_command.extend(
        [
            "-c:v",
            video_encoder,
            "-preset",
            "medium",
            "-crf",
            str(crf),
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
    log: list[str],
    progress_path: str | None,
    cancel_path: str | None,
    total_frames: int,
    extracted_frames: int,
    upscaled_frames: int,
    encoded_frames_before_segment: int,
    telemetry_state: PipelineTelemetryState | None,
) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    encode_command = [
        ffmpeg,
        "-y",
        "-framerate",
        fps,
        "-i",
        str(upscaled_dir / "frame_%08d.png"),
    ]
    if filter_chain is not None:
        encode_command.extend(["-vf", filter_chain])

    video_encoder = "libx265" if codec == "h265" else "libx264"
    encode_command.extend(
        [
            "-c:v",
            video_encoder,
            "-preset",
            "medium",
            "-crf",
            str(crf),
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
        phase="encoding",
        percent_base=80,
        percent_span=15,
        total_frames=total_frames,
        message_prefix=f"Encoding segment video {output_file.name}",
        extracted_frames=extracted_frames,
        upscaled_frames=upscaled_frames,
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
    total_frames: int,
    extracted_frames: int,
    upscaled_frames: int,
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
        phase="remuxing",
        percent_base=95,
        percent_span=2,
        total_frames=total_frames,
        message_prefix="Concatenating encoded segments",
        extracted_frames=extracted_frames,
        upscaled_frames=upscaled_frames,
        encoded_frames=encoded_frames,
        remuxed_frames=0,
        telemetry_state=telemetry_state,
    )
    return concatenated_frames


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
    if preset == "qualityMax":
        return 0
    if preset == "qualityBalanced":
        return 256
    return 128


def run_realesrgan_pipeline(
    *,
    source_path: str,
    model_id: str,
    output_mode: str,
    preset: str,
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
    progress_path: str | None,
    cancel_path: str | None,
    preview_mode: bool,
    preview_duration_seconds: float | None,
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
    precision_mode = resolve_precision_mode(fp16=fp16, bf16=bf16, precision=precision)
    fp16_enabled = precision_mode == "fp16"
    bf16_enabled = precision_mode == "bf16"

    ensure_not_cancelled(cancel_path)
    ensure_runnable_model(model_id)
    backend_id = model_backend_id(model_id)
    resolved_pytorch_execution_path = _resolve_pytorch_execution_path(model_id, pytorch_execution_path)
    runtime = ensure_runtime_assets()
    metadata = probe_video(source_path)
    requested_output = Path(output_path)
    if not requested_output.is_absolute():
        requested_output = repo_root() / requested_output
    normalized_output = requested_output.with_suffix(f".{container}").resolve()
    normalized_output.parent.mkdir(parents=True, exist_ok=True)

    jobs_root = repo_root() / "artifacts" / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)

    cache_key = hashlib.sha256(
        "|".join(
            [
                source_path,
                model_id,
                output_mode,
                preset,
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
                precision_mode,
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
    model_name = model_label(model_id)

    log: list[str] = []
    model_runtime: dict[str, object] | None = None
    ffmpeg = runtime["ffmpegPath"]
    fps = f"{metadata['frameRate']:.6f}".rstrip("0").rstrip(".")
    effective_duration = float(metadata["durationSeconds"])
    if preview_mode:
        requested_preview_duration = preview_duration_seconds if preview_duration_seconds and preview_duration_seconds > 0 else 8.0
        effective_duration = min(effective_duration, requested_preview_duration)
    total_frames = max(1, int(round(float(metadata["frameRate"]) * effective_duration)))
    force_single_stream_segment = backend_id == "pytorch-image-sr" and resolved_pytorch_execution_path == PYTORCH_EXECUTION_PATH_STREAMING
    segment_frame_limit = total_frames if preview_mode or force_single_stream_segment else _segment_frame_limit(float(metadata["frameRate"]), segment_duration_seconds)
    segments = _plan_pipeline_segments(
        total_frames,
        float(metadata["frameRate"]),
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

    effective_tile = _effective_tile_size(model_id, preset, tile_size)

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

    progress_state = PipelineProgressState()
    telemetry_state = PipelineTelemetryState(started_at=time.time(), scratch_path=work_dir, output_path=output_file)
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
            _emit_pipeline_progress(
                progress_path,
                phase=phase,
                message=message,
                total_frames=total_frames,
                progress_state=progress_state,
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
                    input_dir=segment_input_dir,
                    log=log,
                    cancel_path=cancel_path,
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
        try:
            if backend_id == "pytorch-image-sr":
                if resolved_pytorch_execution_path == PYTORCH_EXECUTION_PATH_STREAMING:
                    raise NotImplementedError(
                        "PyTorch streaming execution path is not implemented yet. Use file-io for now."
                    )
                from upscaler_worker.models.pytorch_sr import load_runtime_model

                loaded_model = load_runtime_model(
                    model_id,
                    gpu_id,
                    fp16_enabled,
                    effective_tile,
                    log,
                    preset=preset,
                    torch_compile_enabled=torch_compile_enabled,
                    torch_compile_mode=torch_compile_mode,
                    torch_compile_cudagraphs=torch_compile_cudagraphs,
                    bf16=bf16_enabled,
                    precision=precision_mode,
                    pytorch_runner=pytorch_runner,
                    channels_last_enabled=channels_last,
                )
                log.append(f"Loaded PyTorch model checkpoint: {loaded_model.checkpoint_path}")
                model_runtime = {
                    "runner": loaded_model.runner,
                    "precision": precision_mode,
                    "dtype": str(loaded_model.dtype).replace("torch.", ""),
                    "frameBatchSize": loaded_model.frame_batch_size,
                    "channelsLast": loaded_model.channels_last,
                    "torchCompileRequested": loaded_model.torch_compile_requested,
                    "torchCompileEnabled": loaded_model.torch_compile_enabled,
                    "torchCompileMode": loaded_model.torch_compile_mode,
                    "torchCompileCudagraphs": loaded_model.torch_compile_cudagraphs,
                }

            while True:
                item = _queue_get(extract_queue)
                if item is sentinel:
                    break
                segment = item
                if not isinstance(segment, PipelineSegment):
                    continue

                segment_dir = segment_root / f"segment_{segment.index:04d}"
                segment_input_dir = segment_dir / "in"
                segment_output_dir = segment_dir / "out"
                stage_started_at = time.time()
                with progress_lock:
                    upscaled_before_segment = progress_state.upscaled_frames
                with progress_lock:
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
                    upscaled_count = _upscale_ncnn_segment(
                        runtime=runtime,
                        input_dir=segment_input_dir,
                        output_dir=segment_output_dir,
                        model_id=model_id,
                        gpu_id=gpu_id,
                        effective_tile=effective_tile,
                        log=log,
                        cancel_path=cancel_path,
                    )
                    with progress_lock:
                        progress_state.upscaled_frames = min(total_frames, upscaled_before_segment + upscaled_count)
                        _set_segment_progress(
                            telemetry_state,
                            segment_index=segment.index + 1,
                            segment_count=len(segments),
                            segment_processed_frames=upscaled_count,
                            segment_total_frames=segment.frame_count,
                        )
                elif backend_id == "pytorch-image-sr":
                    def report_upscale_batch(batch_index: int, batch_count: int, processed_in_segment: int, total_in_segment: int) -> None:
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
                        _publish_progress(
                            "upscaling",
                            f"Upscaling segment {segment.index + 1}/{len(segments)} batch {batch_index}/{batch_count} ({processed_in_segment}/{total_in_segment} frames)",
                        )

                    upscaled_count = _upscale_pytorch_segment(
                        loaded_model=loaded_model,
                        input_dir=segment_input_dir,
                        output_dir=segment_output_dir,
                        effective_tile=effective_tile,
                        cancel_path=cancel_path,
                        progress_callback=report_upscale_batch,
                    )
                else:
                    raise RuntimeError(f"Backend '{backend_id}' is cataloged but not runnable in the current app build")

                shutil.rmtree(segment_input_dir, ignore_errors=True)
                with progress_lock:
                    _record_stage_duration(telemetry_state, "upscale", time.time() - stage_started_at)
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
                _queue_put(encode_queue, (segment, segment_output_dir))
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
                    total_frames=total_frames,
                    extracted_frames=progress_state.extracted_frames,
                    upscaled_frames=progress_state.upscaled_frames,
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

    _publish_progress("extracting", "Starting overlapped extract/upscale/encode pipeline")

    concat_manifest: Path | None = None
    if backend_id == "pytorch-image-sr" and resolved_pytorch_execution_path == PYTORCH_EXECUTION_PATH_STREAMING:
        loaded_model = preloaded_pytorch_model
        if loaded_model is None:
            from upscaler_worker.models.pytorch_sr import load_runtime_model

            loaded_model = load_runtime_model(
                model_id,
                gpu_id,
                fp16_enabled,
                effective_tile,
                log,
                preset=preset,
                torch_compile_enabled=torch_compile_enabled,
                torch_compile_mode=torch_compile_mode,
                torch_compile_cudagraphs=torch_compile_cudagraphs,
                bf16=bf16_enabled,
                precision=precision_mode,
                pytorch_runner=pytorch_runner,
                channels_last_enabled=channels_last,
            )
            log.append(f"Loaded PyTorch model checkpoint: {loaded_model.checkpoint_path}")
        model_runtime = {
            "runner": loaded_model.runner,
            "precision": precision_mode,
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
            silent_video=silent_video,
            codec=codec,
            crf=crf,
            filter_chain=filter_chain,
            loaded_model=loaded_model,
            effective_tile=effective_tile,
            log=log,
            progress_path=progress_path,
            cancel_path=cancel_path,
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
            total_frames=total_frames,
            extracted_frames=progress_state.extracted_frames,
            upscaled_frames=progress_state.upscaled_frames,
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
                f"upscaler_model={model_id}",
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
            phase="remuxing",
            percent_base=97,
            percent_span=3,
            total_frames=total_frames,
            message_prefix="Remuxing original audio",
            extracted_frames=progress_state.extracted_frames,
            upscaled_frames=progress_state.upscaled_frames,
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
            phase="remuxing",
            percent_base=97,
            percent_span=3,
            total_frames=total_frames,
            message_prefix="Finalizing video output",
            extracted_frames=progress_state.extracted_frames,
            upscaled_frames=progress_state.upscaled_frames,
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
    average_throughput = total_frames / max(0.001, total_elapsed_seconds)

    return {
        "outputPath": str(output_file),
        "workDir": str(work_dir),
        "executionPath": resolved_pytorch_execution_path or "external-executable",
        "runner": pytorch_runner or "torch",
        "precision": precision_mode,
        "torchCompileEnabled": torch_compile_enabled,
        "torchCompileMode": torch_compile_mode,
        "torchCompileCudagraphs": torch_compile_cudagraphs,
        "frameCount": total_frames,
        "hadAudio": bool(metadata["hasAudio"]),
        "codec": codec,
        "container": container,
        "runtime": runtime,
        "stageTimings": {
            "extractSeconds": telemetry_state.extract_stage_seconds,
            "upscaleSeconds": telemetry_state.upscale_stage_seconds,
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
            f"Model: {model_name} ({model_id})",
            f"Execution path: {resolved_pytorch_execution_path or 'external-executable'}",
            f"Runner: {pytorch_runner or 'torch'}",
            f"Precision: {precision_mode}",
            f"Resolved output canvas: {resolved_width}x{resolved_height} ({resolved_aspect_ratio:.4f}:1)",
            f"Chunked pipeline segments: {len(segments)} at up to {segment_frame_limit} frames each",
            f"Preview mode: {'on' if preview_mode else 'off'}",
            f"Segment duration target: {'single preview segment' if preview_mode else f'{segment_duration_seconds or PIPELINE_SEGMENT_TARGET_SECONDS:.2f}s'}",
            f"Processed duration: {effective_duration:.2f}s",
            f"Average throughput: {average_throughput:.2f} fps",
            f"Rolling throughput: {(telemetry_state.resources.rolling_frames_per_second or 0.0):.2f} fps",
            f"Stage timings: extract {telemetry_state.extract_stage_seconds:.2f}s, upscale {telemetry_state.upscale_stage_seconds:.2f}s, encode {telemetry_state.encode_stage_seconds:.2f}s, remux {telemetry_state.remux_stage_seconds:.2f}s",
        ],
    }
