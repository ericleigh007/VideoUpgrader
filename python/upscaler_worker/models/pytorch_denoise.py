from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from upscaler_worker.model_catalog import ensure_runnable_model, model_label, model_research_runtime


@dataclass(frozen=True)
class ExternalDenoiseCommand:
    model_id: str
    model_label: str
    command_env_var: str
    command_source: str
    command: list[str]
    environment: dict[str, str]


def _sequence_frames(root: Path) -> list[Path]:
    return sorted(root.glob("frame_*.png"))


def _default_external_denoise_command(model_id: str) -> str | None:
    if model_id not in {
        "fastdvdnet",
        "swinir-denoise-real",
        "scunet-real-denoise",
        "drunet-gray-color-denoise",
    }:
        return None

    return subprocess.list2cmdline([
        sys.executable,
        "-m",
        "upscaler_worker.ai_denoise_runner",
        "--input",
        "{input_dir}",
        "--output",
        "{output_dir}",
        "--model",
        "{model_id}",
        "--gpu-id",
        "{gpu_id}",
        "--precision",
        "{precision}",
    ])


def resolve_external_denoise_command_template(model_id: str, command_env_var: str) -> tuple[str | None, str]:
    env_command = os.environ.get(command_env_var, "").strip()
    if env_command:
        return env_command, "environment"

    default_command = _default_external_denoise_command(model_id)
    if default_command:
        return default_command, "repo-default"

    return None, "missing"


def build_external_denoise_command(
    *,
    model_id: str,
    input_dir: Path,
    output_dir: Path,
    gpu_id: int | None = None,
    precision: str = "fp32",
) -> ExternalDenoiseCommand:
    ensure_runnable_model(model_id)
    research_runtime = model_research_runtime(model_id)
    if research_runtime is None or str(research_runtime.get("kind")) != "external-command":
        raise RuntimeError(f"Denoiser '{model_id}' does not declare an external AI denoise runtime")

    command_env_var = str(research_runtime.get("commandEnvVar", "")).strip()
    if not command_env_var:
        raise RuntimeError(f"Denoiser '{model_id}' does not declare the environment variable used to launch its AI runtime")

    raw_command, command_source = resolve_external_denoise_command_template(model_id, command_env_var)
    if not raw_command:
        raise RuntimeError(
            f"Set {command_env_var} to an AI denoiser runner command before using {model_label(model_id)}. "
            "The command may use placeholders: {input_dir}, {output_dir}, {model_id}, {frame_count}, {gpu_id}, and {precision}. "
            "It must write frame_*.png outputs matching the input frame count."
        )

    input_frames = _sequence_frames(input_dir)
    if not input_frames:
        raise RuntimeError(f"No input frames were found in '{input_dir}' for AI denoise execution")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = len(input_frames)
    replacements = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model_id": model_id,
        "frame_count": str(frame_count),
        "gpu_id": str(gpu_id if gpu_id is not None else -1),
        "precision": precision,
    }
    command = [part.format(**replacements) for part in shlex.split(raw_command, posix=False)]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID"),
            "UPSCALER_DENOISE_MODEL_ID": model_id,
            "UPSCALER_DENOISE_INPUT_DIR": str(input_dir),
            "UPSCALER_DENOISE_OUTPUT_DIR": str(output_dir),
            "UPSCALER_DENOISE_FRAME_COUNT": str(frame_count),
            "UPSCALER_DENOISE_GPU_ID": str(gpu_id) if gpu_id is not None else "",
            "UPSCALER_DENOISE_PRECISION": precision,
            "UPSCALER_DENOISE_COMMAND_ENV_VAR": command_env_var,
            "UPSCALER_DENOISE_COMMAND_SOURCE": command_source,
        }
    )
    return ExternalDenoiseCommand(
        model_id=model_id,
        model_label=model_label(model_id),
        command_env_var=command_env_var,
        command_source=command_source,
        command=command,
        environment=environment,
    )


def validate_external_denoise_outputs(*, input_dir: Path, output_dir: Path) -> int:
    input_count = len(_sequence_frames(input_dir))
    output_count = len(_sequence_frames(output_dir))
    if output_count <= 0:
        raise RuntimeError(f"The external AI denoiser did not produce any frame_*.png outputs in '{output_dir}'")
    if input_count != output_count:
        raise RuntimeError(f"Expected {input_count} output frames from the external AI denoiser, received {output_count}")
    return output_count