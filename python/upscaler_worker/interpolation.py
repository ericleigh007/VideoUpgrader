from __future__ import annotations

from pathlib import Path


SUPPORTED_INTERPOLATION_MODES = {"off", "afterUpscale", "interpolateOnly"}
SUPPORTED_INTERPOLATION_TARGET_FPS = {30, 60}
RIFE_MODEL_DIRECTORY = "rife-v4.6"


def validate_interpolation_request(interpolation_mode: str, interpolation_target_fps: int | None) -> None:
    if interpolation_mode not in SUPPORTED_INTERPOLATION_MODES:
        supported_modes = ", ".join(sorted(SUPPORTED_INTERPOLATION_MODES))
        raise ValueError(f"Unsupported interpolation mode '{interpolation_mode}'. Expected one of: {supported_modes}")
    if interpolation_mode == "off":
        return
    if interpolation_target_fps not in SUPPORTED_INTERPOLATION_TARGET_FPS:
        supported_fps = ", ".join(str(value) for value in sorted(SUPPORTED_INTERPOLATION_TARGET_FPS))
        raise ValueError(f"Interpolation currently requires an explicit target fps of {supported_fps}")


def resolve_output_fps(source_fps: float, interpolation_mode: str, interpolation_target_fps: int | None) -> float:
    validate_interpolation_request(interpolation_mode, interpolation_target_fps)
    if interpolation_mode == "off" or interpolation_target_fps is None:
        return float(source_fps)
    return float(interpolation_target_fps)


def resolve_segment_output_frame_count(*, start_frame: int, frame_count: int, source_fps: float, output_fps: float) -> int:
    if frame_count <= 0:
        return 0
    if source_fps <= 0 or output_fps <= 0:
        raise ValueError("source_fps and output_fps must be positive")
    start_output_frame = round(start_frame * output_fps / source_fps)
    end_output_frame = round((start_frame + frame_count) * output_fps / source_fps)
    return max(frame_count, end_output_frame - start_output_frame)


def should_skip_interpolation(*, input_frame_count: int, target_frame_count: int) -> bool:
    return target_frame_count <= input_frame_count


def build_rife_command(
    *,
    executable_path: str,
    model_root: str,
    input_dir: Path,
    output_dir: Path,
    target_frame_count: int,
    gpu_id: int | None,
    uhd_mode: bool,
) -> list[str]:
    command = [
        executable_path,
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-n",
        str(target_frame_count),
        "-m",
        str(Path(model_root) / RIFE_MODEL_DIRECTORY),
        "-f",
        "frame_%08d.png",
    ]
    if gpu_id is not None:
        command.extend(["-g", str(gpu_id)])
    if uhd_mode:
        command.append("-u")
    return command