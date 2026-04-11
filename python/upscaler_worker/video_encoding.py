from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable

ENCODER_PROBE_FRAME_SIZE = "256x256"


@dataclass(frozen=True)
class VideoEncoderConfig:
    encoder: str
    quality_args: tuple[str, ...]
    label: str
    hardware_accelerated: bool


ProbeEncoder = Callable[[str, VideoEncoderConfig], bool | tuple[bool, str | None]]


def software_video_encoder_config(codec: str, crf: int) -> VideoEncoderConfig:
    return VideoEncoderConfig(
        encoder="libx265" if codec == "h265" else "libx264",
        quality_args=("-preset", "medium", "-crf", str(crf)),
        label="software-cpu",
        hardware_accelerated=False,
    )


def select_runtime_gpu(runtime: dict[str, object], gpu_id: int | None) -> dict[str, object] | None:
    available_gpus = runtime.get("availableGpus", [])
    if not isinstance(available_gpus, list) or not available_gpus:
        return None

    if gpu_id is not None:
        selected = next(
            (
                device
                for device in available_gpus
                if isinstance(device, dict) and device.get("id") == gpu_id
            ),
            None,
        )
        if selected is not None:
            return selected

    discrete = next(
        (
            device
            for device in available_gpus
            if isinstance(device, dict) and device.get("kind") == "discrete"
        ),
        None,
    )
    if discrete is not None:
        return discrete

    return next((device for device in available_gpus if isinstance(device, dict)), None)


def hardware_video_encoder_config(codec: str, crf: int, gpu_name: str) -> VideoEncoderConfig | None:
    normalized_name = gpu_name.lower()
    if any(token in normalized_name for token in ["nvidia", "geforce", "rtx", "quadro"]):
        return VideoEncoderConfig(
            encoder="hevc_nvenc" if codec == "h265" else "h264_nvenc",
            quality_args=("-preset", "p5", "-cq", str(crf), "-b:v", "0"),
            label=f"nvidia-nvenc ({gpu_name})",
            hardware_accelerated=True,
        )
    return None


def probe_video_encoder(ffmpeg: str, config: VideoEncoderConfig) -> tuple[bool, str | None]:
    probe_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={ENCODER_PROBE_FRAME_SIZE}:d=0.1:r=1",
        "-frames:v",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        config.encoder,
        *config.quality_args,
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            probe_command,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)

    stderr = completed.stderr.strip() or None
    return completed.returncode == 0, stderr


def resolve_video_encoder_config(
    *,
    ffmpeg: str,
    runtime: dict[str, object],
    gpu_id: int | None,
    codec: str,
    crf: int,
    log: list[str],
    probe_encoder: ProbeEncoder | None = None,
) -> VideoEncoderConfig:
    software_config = software_video_encoder_config(codec, crf)
    selected_gpu = select_runtime_gpu(runtime, gpu_id)
    if selected_gpu is None:
        log.append(f"Video encoder: {software_config.encoder} ({software_config.label})")
        return software_config

    gpu_name = str(selected_gpu.get("name", "")).strip()
    hardware_config = hardware_video_encoder_config(codec, crf, gpu_name)
    if hardware_config is None:
        log.append(f"Video encoder: {software_config.encoder} ({software_config.label})")
        return software_config

    if probe_encoder is None:
        hardware_available, probe_details = probe_video_encoder(ffmpeg, hardware_config)
    else:
        raw_probe_result = probe_encoder(ffmpeg, hardware_config)
        if isinstance(raw_probe_result, tuple):
            hardware_available, probe_details = raw_probe_result
        else:
            hardware_available = bool(raw_probe_result)
            probe_details = None

    if hardware_available:
        log.append(f"Video encoder: {hardware_config.encoder} ({hardware_config.label})")
        return hardware_config

    probe_suffix = ""
    if probe_details:
        probe_line = probe_details.splitlines()[0].strip()
        if probe_line:
            probe_suffix = f" Details: {probe_line}"
    log.append(
        f"Hardware encoder probe failed for {hardware_config.encoder} on {gpu_name}; "
        f"falling back to {software_config.encoder}.{probe_suffix}"
    )
    log.append(f"Video encoder: {software_config.encoder} ({software_config.label})")
    return software_config
