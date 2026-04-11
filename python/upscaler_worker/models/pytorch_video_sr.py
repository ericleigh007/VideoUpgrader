from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from upscaler_worker.model_catalog import ensure_benchmarkable_model, model_label, model_research_runtime


@dataclass(frozen=True)
class ExternalVideoSrCommand:
    model_id: str
    model_label: str
    command_env_var: str
    command: list[str]
    environment: dict[str, str]


def _sequence_frames(root: Path) -> list[Path]:
    return sorted(root.glob("frame_*.png"))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_external_video_sr_command(model_id: str) -> str | None:
    if model_id != "rvrt-x4":
        return None

    rvrt_root = _repo_root() / "tmp" / "RVRT"
    if not rvrt_root.exists():
        return None

    return subprocess.list2cmdline([
        sys.executable,
        "-m",
        "upscaler_worker.rvrt_external_runner",
        "--input",
        "{input_dir}",
        "--output",
        "{output_dir}",
        "--model",
        "{model_id}",
        "--tile",
        "{tile_size}",
        "--precision",
        "{precision}",
    ])


def resolve_external_video_sr_command_template(model_id: str, command_env_var: str) -> tuple[str | None, str]:
    env_command = os.environ.get(command_env_var, "").strip()
    if env_command:
        return env_command, "environment"

    default_command = _default_external_video_sr_command(model_id)
    if default_command:
        return default_command, "repo-default"

    return None, "missing"


def build_external_video_sr_command(
    *,
    model_id: str,
    input_dir: Path,
    output_dir: Path,
    tile_size: int,
    gpu_id: int | None = None,
    precision: str = "fp32",
) -> ExternalVideoSrCommand:
    ensure_benchmarkable_model(model_id)
    research_runtime = model_research_runtime(model_id)
    if research_runtime is None or str(research_runtime.get("kind")) != "external-command":
        raise RuntimeError(f"Model '{model_id}' does not declare an external research runtime")

    command_env_var = str(research_runtime.get("commandEnvVar", "")).strip()
    if not command_env_var:
        raise RuntimeError(f"Model '{model_id}' does not declare the environment variable used to launch its research runtime")

    raw_command, command_source = resolve_external_video_sr_command_template(model_id, command_env_var)
    if not raw_command:
        raise RuntimeError(
            f"Set {command_env_var} to an external runner command before benchmarking {model_label(model_id)}, "
            f"or place the official RVRT repo at '{_repo_root() / 'tmp' / 'RVRT'}' so the built-in runner can be used automatically. "
            f"The command may use placeholders like {{input_dir}}, {{output_dir}}, {{model_id}}, {{tile_size}}, and {{frame_count}}."
        )

    input_frames = _sequence_frames(input_dir)
    if not input_frames:
        raise RuntimeError(f"No input frames were found in '{input_dir}' for external video SR execution")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = len(input_frames)
    replacements = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model_id": model_id,
        "tile_size": str(tile_size),
        "frame_count": str(frame_count),
        "gpu_id": str(gpu_id if gpu_id is not None else -1),
        "precision": precision,
    }
    command = [part.format(**replacements) for part in shlex.split(raw_command, posix=False)]
    environment = os.environ.copy()
    environment.update(
        {
            "UPSCALER_VIDEO_SR_MODEL_ID": model_id,
            "UPSCALER_VIDEO_SR_INPUT_DIR": str(input_dir),
            "UPSCALER_VIDEO_SR_OUTPUT_DIR": str(output_dir),
            "UPSCALER_VIDEO_SR_TILE_SIZE": str(tile_size),
            "UPSCALER_VIDEO_SR_FRAME_COUNT": str(frame_count),
            "UPSCALER_VIDEO_SR_GPU_ID": str(gpu_id) if gpu_id is not None else "",
            "UPSCALER_VIDEO_SR_PRECISION": precision,
            "UPSCALER_VIDEO_SR_COMMAND_SOURCE": command_source,
        }
    )
    return ExternalVideoSrCommand(
        model_id=model_id,
        model_label=model_label(model_id),
        command_env_var=command_env_var,
        command=command,
        environment=environment,
    )


def validate_external_video_sr_outputs(*, input_dir: Path, output_dir: Path) -> int:
    input_count = len(_sequence_frames(input_dir))
    output_count = len(_sequence_frames(output_dir))
    if output_count <= 0:
        raise RuntimeError(f"The external video SR runner did not produce any frame_*.png outputs in '{output_dir}'")
    if input_count != output_count:
        raise RuntimeError(
            f"Expected {input_count} output frames from the external video SR runner, received {output_count}"
        )
    return output_count