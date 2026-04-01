from __future__ import annotations

import hashlib

from upscaler_worker.model_catalog import (
    ensure_runnable_model,
    model_backend_id,
    model_label,
    model_runtime_name,
)


def build_realesrgan_job_plan(
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
    preview_mode: bool = False,
    preview_duration_seconds: float | None = None,
    output_path: str,
    codec: str,
    container: str,
    tile_size: int,
    fp16: bool,
    crf: int,
) -> dict[str, object]:
    ensure_runnable_model(model_id)
    cache_material = "|".join(
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
            str(int(preview_mode)),
            str(preview_duration_seconds or 0),
            output_path,
            codec,
            container,
            str(tile_size),
            str(int(fp16)),
            str(crf),
        ]
    )
    cache_key = hashlib.sha256(cache_material.encode("utf-8")).hexdigest()

    command = [
        "python",
        "-m",
        "upscaler_worker.cli",
        "run-realesrgan-pipeline",
        "--source",
        source_path,
        "--model-id",
        model_id,
        "--output-mode",
        output_mode,
        "--preset",
        preset,
        "--aspect-ratio-preset",
        aspect_ratio_preset,
        "--resolution-basis",
        resolution_basis,
        "--output-path",
        output_path,
        "--codec",
        codec,
        "--container",
        container,
        "--tile-size",
        str(tile_size),
        "--crf",
        str(crf),
    ]

    if gpu_id is not None:
        command.extend(["--gpu-id", str(gpu_id)])
    if custom_aspect_width:
        command.extend(["--custom-aspect-width", str(custom_aspect_width)])
    if custom_aspect_height:
        command.extend(["--custom-aspect-height", str(custom_aspect_height)])
    if target_width:
        command.extend(["--target-width", str(target_width)])
    if target_height:
        command.extend(["--target-height", str(target_height)])
    if crop_left is not None:
        command.extend(["--crop-left", str(crop_left)])
    if crop_top is not None:
        command.extend(["--crop-top", str(crop_top)])
    if crop_width is not None:
        command.extend(["--crop-width", str(crop_width)])
    if crop_height is not None:
        command.extend(["--crop-height", str(crop_height)])
    if preview_mode:
        command.append("--preview-mode")
    if preview_duration_seconds is not None:
        command.extend(["--preview-duration-seconds", str(preview_duration_seconds)])
    if fp16:
        command.append("--fp16")

    notes = [
        "Prepared for the current multi-backend Python pipeline entrypoint.",
        f"Model: {model_id}",
        f"Backend: {model_backend_id(model_id)}",
        f"Runtime name: {model_runtime_name(model_id)}",
        f"Output mode: {output_mode}",
        f"Preset: {preset}",
        f"GPU: {gpu_id if gpu_id is not None else 'auto'}",
        f"Aspect ratio: {aspect_ratio_preset}",
        f"Resolution basis: {resolution_basis}",
        f"Target width: {target_width or 'auto'}",
        f"Target height: {target_height or 'auto'}",
        f"Crop rect: {crop_left or 0:.2f}, {crop_top or 0:.2f}, {crop_width or 0:.2f}, {crop_height or 0:.2f}",
        f"Codec: {codec}",
        f"Container: {container}",
        f"CRF: {crf}",
    ]
    if custom_aspect_width and custom_aspect_height:
        notes.append(f"Custom aspect ratio: {custom_aspect_width}:{custom_aspect_height}")
    if preview_mode:
        notes.append(f"Preview mode: {preview_duration_seconds or 8:.1f}s")
    if fp16:
        notes.append("FP16 requested for model paths that support half precision.")

    return {
        "model": model_label(model_id),
        "cacheKey": cache_key,
        "command": command,
        "notes": notes,
    }
