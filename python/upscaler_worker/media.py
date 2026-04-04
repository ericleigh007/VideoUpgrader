from __future__ import annotations

import hashlib
import json
import re
import subprocess
import threading
from pathlib import Path

from upscaler_worker.cancellation import JobCancelledError, cancellation_requested, ensure_not_cancelled, terminate_process
from upscaler_worker.runtime import ensure_runtime_assets


PREVIEW_COMPATIBLE_CONTAINERS = {"mp4"}
PREVIEW_CLIP_SECONDS = 30
PREVIEW_MAX_WIDTH = 960
FAST_MP4_VERSION = "fast-mp4-v1"


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


def _prefers_nvidia_encoder(runtime: dict[str, object]) -> bool:
    available_gpus = runtime.get("availableGpus", [])
    if not isinstance(available_gpus, list):
        return False
    return any("nvidia" in str(device.get("name", "")).lower() for device in available_gpus if isinstance(device, dict))


def _run_ffmpeg(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


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
            if cancellation_requested(cancel_path):
                terminate_process(process)
                raise JobCancelledError("Job cancelled by user")
            reader_thread.join(timeout=0.1)
        reader_thread.join(timeout=1)
    finally:
        if process.stdout:
            process.stdout.close()

    return_code = process.wait()
    if return_code != 0:
        output_text = "\n".join(output_lines)
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(progress_command)}\n{output_text}")


def convert_source_to_mp4(source_path: str, progress_path: str | None = None, cancel_path: str | None = None) -> dict[str, object]:
    source = Path(source_path)
    ensure_not_cancelled(cancel_path)
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

    if _prefers_nvidia_encoder(runtime):
        nvenc_command = common_args + [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p1",
            "-cq",
            "19",
            "-b:v",
            "0",
            str(output_path),
        ]
        try:
            _run_ffmpeg_with_time_progress(
                nvenc_command,
                progress_path=progress_path,
                cancel_path=cancel_path,
                phase="encoding",
                duration_seconds=duration_seconds,
                message_prefix="Fast converting source with NVIDIA encoder",
            )
            return probe_video(str(output_path))
        except RuntimeError as error:
            nvenc_stderr = str(error)
    else:
        nvenc_stderr = ""

    x264_command = common_args + [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        str(output_path),
    ]
    try:
        _run_ffmpeg_with_time_progress(
            x264_command,
            progress_path=progress_path,
            cancel_path=cancel_path,
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
    video_match = re.search(r"Video:.*?(\d{2,5})x(\d{2,5}).*?(\d+(?:\.\d+)?)\s+fps", stderr)
    audio_present = "Audio:" in stderr

    if duration_match is None or video_match is None:
        raise RuntimeError(f"Could not parse video metadata for {source_path}\n{stderr}")

    hours = int(duration_match.group(1))
    minutes = int(duration_match.group(2))
    seconds = float(duration_match.group(3))
    duration_seconds = hours * 3600 + minutes * 60 + seconds

    container = source_path.rsplit(".", maxsplit=1)[-1].lower() if "." in source_path else "unknown"
    preview_path = ensure_browser_preview(source_path, str(runtime["ffmpegPath"]), container)
    return {
        "path": source_path,
        "previewPath": preview_path,
        "width": int(video_match.group(1)),
        "height": int(video_match.group(2)),
        "durationSeconds": duration_seconds,
        "frameRate": float(video_match.group(3)),
        "hasAudio": audio_present,
        "container": container,
    }
