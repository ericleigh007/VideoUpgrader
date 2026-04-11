from __future__ import annotations

import hashlib
import json
import re
import subprocess
import threading
from pathlib import Path

from upscaler_worker.cancellation import JobCancelledError, cancellation_requested, ensure_not_cancelled, terminate_process, terminate_process_tree, wait_if_paused
from upscaler_worker.runtime import ensure_runtime_assets
from upscaler_worker.video_encoding import VideoEncoderConfig, probe_video_encoder, select_runtime_gpu


PREVIEW_COMPATIBLE_CONTAINERS = {"mp4"}
PREVIEW_CLIP_SECONDS = 30
PREVIEW_MAX_WIDTH = 960
FAST_MP4_VERSION = "fast-mp4-v2"
STREAM_LINE_PATTERN = r"^\s*Stream\s+#\d+:\d+(?:\[[^\]]+\])?(?:\([^)]+\))?:\s*{kind}:\s+.+$"


def _preview_root() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts" / "runtime" / "source-previews"


def _converted_source_root() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts" / "runtime" / "converted-sources"


def _preview_key(source_path: str) -> str:
    source = Path(source_path)
    stat = source.stat()
    material = f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|preview-v2"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _converted_source_key(source_path: str) -> str:
    source = Path(source_path)
    stat = source.stat()
    material = f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{FAST_MP4_VERSION}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _fast_mp4_output_path(source_path: str) -> Path:
    source = Path(source_path)
    return _converted_source_root() / f"{source.stem}_{_converted_source_key(source_path)}.mp4"


def _run_ffmpeg(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _find_stream_line(stderr: str, kind: str) -> str | None:
    match = re.search(STREAM_LINE_PATTERN.format(kind=re.escape(kind)), stderr, flags=re.MULTILINE)
    return match.group(0) if match else None


def _split_stream_descriptor(line: str, marker: str) -> list[str]:
    if f"{marker}:" not in line:
        return []
    descriptor = line.split(f"{marker}:", maxsplit=1)[1].strip()
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in descriptor:
        if char == "," and depth == 0:
            value = "".join(current).strip()
            if value:
                parts.append(value)
            current = []
            continue
        current.append(char)
        if char in "([":
            depth += 1
        elif char in ")]" and depth > 0:
            depth -= 1
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_profile(descriptor: str | None) -> str | None:
    if not descriptor:
        return None
    matches = re.findall(r"\(([^)]+)\)", descriptor)
    for match in matches:
        candidate = match.strip()
        if candidate and "/" not in candidate:
            return candidate
    return None


def _parse_int_kbps(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"(\d+)\s*kb/s", value)
    if not match:
        return None
    return int(match.group(1))


def _write_progress(
    progress_path: str | None,
    *,
    phase: str,
    percent: int,
    message: str,
    processed_frames: int,
    total_frames: int,
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
        "extractedFrames": 0,
        "upscaledFrames": 0,
        "encodedFrames": 0,
        "remuxedFrames": 0,
    }
    target.write_text(json.dumps(payload), encoding="utf-8")


def _run_ffmpeg_with_time_progress(
    command: list[str],
    *,
    progress_path: str | None,
    cancel_path: str | None,
    pause_path: str | None,
    phase: str,
    duration_seconds: float,
    message_prefix: str,
) -> None:
    progress_command = command[:-1] + ["-progress", "pipe:1", "-nostats", command[-1]]
    process = subprocess.Popen(
        progress_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    processed_units = 0
    total_units = max(1, int(round(duration_seconds * 1000)))
    output_lines: list[str] = []
    state = {"processed_units": 0}
    _write_progress(
        progress_path,
        phase=phase,
        percent=0,
        message=message_prefix,
        processed_frames=0,
        total_frames=total_units,
    )

    def reader() -> None:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            output_lines.append(line)
            if not line.startswith("out_time_ms="):
                continue
            try:
                out_time_ms = int(line.split("=", maxsplit=1)[1])
            except ValueError:
                continue

            current_units = min(total_units, max(0, out_time_ms // 1000))
            state["processed_units"] = current_units
            percent = min(99, int((current_units / max(total_units, 1)) * 100))
            _write_progress(
                progress_path,
                phase=phase,
                percent=percent,
                message=f"{message_prefix} ({current_units / 1000:.1f}s / {duration_seconds:.1f}s)",
                processed_frames=current_units,
                total_frames=total_units,
            )

    reader_thread = threading.Thread(target=reader, name="ffmpeg-time-progress-reader", daemon=True)
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
                    percent=min(99, int((state["processed_units"] / max(total_units, 1)) * 100)),
                    message=f"Paused: {message_prefix} ({state['processed_units'] / 1000:.1f}s / {duration_seconds:.1f}s)",
                    processed_frames=state["processed_units"],
                    total_frames=total_units,
                ),
                on_resume=lambda: _write_progress(
                    progress_path,
                    phase=phase,
                    percent=min(99, int((state["processed_units"] / max(total_units, 1)) * 100)),
                    message=f"Resumed: {message_prefix} ({state['processed_units'] / 1000:.1f}s / {duration_seconds:.1f}s)",
                    processed_frames=state["processed_units"],
                    total_frames=total_units,
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
    if return_code != 0:
        output_text = "\n".join(output_lines)
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(progress_command)}\n{output_text}")


def convert_source_to_mp4(
    source_path: str,
    progress_path: str | None = None,
    cancel_path: str | None = None,
    pause_path: str | None = None,
) -> dict[str, object]:
    source = Path(source_path)
    ensure_not_cancelled(cancel_path)
    wait_if_paused(pause_path, cancel_path=cancel_path)
    if source.suffix.lower() == ".mp4":
        _write_progress(
            progress_path,
            phase="completed",
            percent=100,
            message="Source is already MP4",
            processed_frames=1,
            total_frames=1,
        )
        return probe_video(source_path)

    runtime = ensure_runtime_assets()
    converted_root = _converted_source_root()
    converted_root.mkdir(parents=True, exist_ok=True)
    output_path = _fast_mp4_output_path(source_path)
    if output_path.exists():
        ensure_not_cancelled(cancel_path)
        wait_if_paused(pause_path, cancel_path=cancel_path)
        cached_metadata = probe_video(str(output_path))
        cached_duration_units = max(1, int(round(float(cached_metadata["durationSeconds"]) * 1000)))
        _write_progress(
            progress_path,
            phase="completed",
            percent=100,
            message="Using cached MP4 conversion",
            processed_frames=cached_duration_units,
            total_frames=cached_duration_units,
        )
        return cached_metadata

    ffmpeg_path = str(runtime["ffmpegPath"])
    source_metadata = probe_video(source_path)
    duration_seconds = float(source_metadata["durationSeconds"])
    selected_gpu = select_runtime_gpu(runtime, None)
    gpu_name = str(selected_gpu.get("name", "")).strip() if selected_gpu is not None else ""
    prefers_nvidia = any(token in gpu_name.lower() for token in ["nvidia", "geforce", "rtx", "quadro"])
    common_args = [
        ffmpeg_path,
        "-y",
        "-i",
        source_path,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-sn",
        "-dn",
        "-vf",
        "scale=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
    ]

    if prefers_nvidia:
        nvenc_config = VideoEncoderConfig(
            encoder="h264_nvenc",
            quality_args=("-preset", "p1", "-cq", "19", "-b:v", "0"),
            label=f"nvidia-nvenc ({gpu_name})",
            hardware_accelerated=True,
        )
        nvenc_available, nvenc_probe_details = probe_video_encoder(ffmpeg_path, nvenc_config)
    else:
        nvenc_config = None
        nvenc_available = False
        nvenc_probe_details = None

    if nvenc_config is not None and nvenc_available:
        nvenc_command = common_args + [
            "-c:v",
            nvenc_config.encoder,
            *nvenc_config.quality_args,
            str(output_path),
        ]
        try:
            _run_ffmpeg_with_time_progress(
                nvenc_command,
                progress_path=progress_path,
                cancel_path=cancel_path,
                pause_path=pause_path,
                phase="encoding",
                duration_seconds=duration_seconds,
                message_prefix="Fast converting source with NVIDIA encoder",
            )
            return probe_video(str(output_path))
        except RuntimeError as error:
            nvenc_stderr = str(error)
    else:
        nvenc_stderr = ""
        if nvenc_config is not None and nvenc_probe_details:
            nvenc_stderr = f"NVENC probe failed for {nvenc_config.encoder} on {gpu_name}: {nvenc_probe_details}"

    software_config = VideoEncoderConfig(
        encoder="libx264",
        quality_args=("-preset", "ultrafast", "-crf", "18"),
        label="software-cpu",
        hardware_accelerated=False,
    )
    x264_command = common_args + [
        "-c:v",
        software_config.encoder,
        *software_config.quality_args,
        str(output_path),
    ]
    try:
        _run_ffmpeg_with_time_progress(
            x264_command,
            progress_path=progress_path,
            cancel_path=cancel_path,
            pause_path=pause_path,
            phase="encoding",
            duration_seconds=duration_seconds,
            message_prefix="Fast converting source with x264 fallback",
        )
    except RuntimeError as error:
        stderr = str(error)
        if nvenc_stderr:
            stderr = f"NVENC attempt failed first:\n{nvenc_stderr}\n\nlibx264 fallback failed:\n{stderr}"
        raise RuntimeError(f"Could not convert {source_path} to MP4\n{stderr}")

    _write_progress(
        progress_path,
        phase="completed",
        percent=100,
        message="Source conversion completed",
        processed_frames=max(1, int(round(duration_seconds * 1000))),
        total_frames=max(1, int(round(duration_seconds * 1000))),
    )

    return probe_video(str(output_path))


def ensure_browser_preview(source_path: str, ffmpeg_path: str, container: str) -> str:
    if container in PREVIEW_COMPATIBLE_CONTAINERS:
        return source_path

    preview_root = _preview_root()
    preview_root.mkdir(parents=True, exist_ok=True)
    source = Path(source_path)
    preview_path = preview_root / f"{source.stem}_{_preview_key(source_path)}.mp4"
    if preview_path.exists() and preview_path.stat().st_size > 0:
        return str(preview_path)
    if preview_path.exists():
        preview_path.unlink(missing_ok=True)

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        source_path,
        "-map",
        "0:v:0",
        "-t",
        str(PREVIEW_CLIP_SECONDS),
        "-vf",
        f"scale=trunc(min(iw\\,{PREVIEW_MAX_WIDTH})/2)*2:-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(preview_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        preview_path.unlink(missing_ok=True)
        stderr = completed.stderr.strip()
        raise RuntimeError(f"Could not create browser preview for {source_path}\n{stderr}")

    return str(preview_path)


def probe_video(source_path: str) -> dict[str, object]:
    runtime = ensure_runtime_assets()
    process = subprocess.run(
        [runtime["ffmpegPath"], "-hide_banner", "-i", source_path],
        capture_output=True,
        text=True,
        check=False,
    )
    stderr = process.stderr
    duration_match = re.search(r"Duration:\s+(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    duration_bitrate_match = re.search(r"bitrate:\s*(\d+)\s*kb/s", stderr)
    video_line = _find_stream_line(stderr, "Video")
    audio_line = _find_stream_line(stderr, "Audio")
    video_match = re.search(r"Video:\s*([^,]+),.*?(\d{2,5})x(\d{2,5}).*?(\d+(?:\.\d+)?)\s+fps", video_line or "")
    audio_present = audio_line is not None

    if duration_match is None or video_match is None:
        raise RuntimeError(f"Could not parse video metadata for {source_path}\n{stderr}")

    hours = int(duration_match.group(1))
    minutes = int(duration_match.group(2))
    seconds = float(duration_match.group(3))
    duration_seconds = hours * 3600 + minutes * 60 + seconds

    container = source_path.rsplit(".", maxsplit=1)[-1].lower() if "." in source_path else "unknown"
    video_parts = _split_stream_descriptor(video_line or "", "Video")
    audio_parts = _split_stream_descriptor(audio_line or "", "Audio")
    video_codec = video_match.group(1).split()[0].lower()
    video_profile = _extract_profile(video_parts[0] if video_parts else video_match.group(1))
    pixel_format = video_parts[1] if len(video_parts) > 1 else None
    audio_codec = audio_parts[0].split()[0].lower() if audio_parts else None
    audio_profile = _extract_profile(audio_parts[0] if audio_parts else None)
    audio_sample_rate_match = re.search(r"(\d+)\s+Hz", audio_line or "")
    audio_sample_rate = int(audio_sample_rate_match.group(1)) if audio_sample_rate_match else None
    audio_channels = audio_parts[2] if len(audio_parts) > 2 else None
    source_bitrate_kbps = int(duration_bitrate_match.group(1)) if duration_bitrate_match else None
    audio_bitrate_kbps = _parse_int_kbps(audio_line)
    preview_path = ensure_browser_preview(source_path, str(runtime["ffmpegPath"]), container)
    return {
        "path": source_path,
        "previewPath": preview_path,
        "width": int(video_match.group(2)),
        "height": int(video_match.group(3)),
        "durationSeconds": duration_seconds,
        "frameRate": float(video_match.group(4)),
        "hasAudio": audio_present,
        "container": container,
        "videoCodec": video_codec,
        "sourceBitrateKbps": source_bitrate_kbps,
        "videoProfile": video_profile,
        "pixelFormat": pixel_format,
        "audioCodec": audio_codec,
        "audioProfile": audio_profile,
        "audioSampleRate": audio_sample_rate,
        "audioChannels": audio_channels,
        "audioBitrateKbps": audio_bitrate_kbps,
    }
