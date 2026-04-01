from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from upscaler_worker.media import probe_video
from upscaler_worker.model_catalog import ensure_runnable_model, model_backend_id
from upscaler_worker.models.realesrgan import model_label
from upscaler_worker.runtime import ensure_runtime_assets, repo_root


BATCH_FRAME_COUNT = 12


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
        return _round_dimension(source_width * 4), _round_dimension(source_height * 4), aspect_ratio

    if aspect_ratio >= 1:
        return 3840, _round_dimension(3840 / aspect_ratio), aspect_ratio

    return _round_dimension(2160 * aspect_ratio), 2160, aspect_ratio


def _run(command: list[str], log: list[str]) -> None:
    log.append("$ " + " ".join(command))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.stdout.strip():
        log.append(completed.stdout.strip())
    if completed.stderr.strip():
        log.append(completed.stderr.strip())
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


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
    target.write_text(json.dumps(payload), encoding="utf-8")


def _run_ffmpeg_with_frame_progress(
    command: list[str],
    log: list[str],
    progress_path: str | None,
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

    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        output_lines.append(line)
        if line.startswith("frame="):
            try:
                current_frame = int(line.split("=", maxsplit=1)[1])
            except ValueError:
                continue

            percent = percent_base if total_frames <= 0 else min(percent_base + percent_span, percent_base + int((current_frame / max(total_frames, 1)) * percent_span))
            _write_progress(
                progress_path,
                phase=phase,
                percent=percent,
                message=f"{message_prefix} ({current_frame}/{total_frames})",
                processed_frames=current_frame,
                total_frames=total_frames,
                extracted_frames=extracted_frames,
                upscaled_frames=upscaled_frames,
                encoded_frames=current_frame if phase == "encoding" else encoded_frames,
                remuxed_frames=current_frame if phase == "remuxing" else remuxed_frames,
            )

    return_code = process.wait()
    if output_lines:
        log.append("\n".join(output_lines))
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(progress_command)}")
    return current_frame, total_frames


def _run_realesrgan_batch(
    command: list[str],
    log: list[str],
) -> None:
    _run(command, log)


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
) -> int:
    from upscaler_worker.models.pytorch_sr import load_runtime_model, upscale_frames

    extracted_frames = sorted(input_frames.glob("frame_*.png"))
    if not extracted_frames:
        raise RuntimeError("No extracted frames were found for upscaling.")

    loaded_model = load_runtime_model(model_id, gpu_id, fp16, effective_tile, log)
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


def _effective_tile_size(preset: str, tile_size: int) -> int:
    if tile_size > 0:
        return tile_size
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
    preview_mode: bool,
    preview_duration_seconds: float | None,
    output_path: str,
    codec: str,
    container: str,
    tile_size: int,
    fp16: bool,
    crf: int,
) -> dict[str, object]:
    ensure_runnable_model(model_id)
    runtime = ensure_runtime_assets()
    metadata = probe_video(source_path)
    source = Path(source_path)
    requested_output = Path(output_path)
    normalized_output = requested_output.with_suffix(f".{container}")
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
                str(crf),
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]

    work_dir = jobs_root / f"job_{cache_key}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    input_frames = work_dir / "in"
    output_frames = work_dir / "out"
    encoded_dir = work_dir / "enc"
    input_frames.mkdir(parents=True, exist_ok=True)
    output_frames.mkdir(parents=True, exist_ok=True)
    encoded_dir.mkdir(parents=True, exist_ok=True)

    output_file = normalized_output
    silent_video = encoded_dir / f"video_no_audio.{container}"
    model_name = model_label(model_id)

    log: list[str] = []
    ffmpeg = runtime["ffmpegPath"]
    fps = f"{metadata['frameRate']:.6f}".rstrip("0").rstrip(".")
    effective_duration = float(metadata["durationSeconds"])
    if preview_mode:
        requested_preview_duration = preview_duration_seconds if preview_duration_seconds and preview_duration_seconds > 0 else 8.0
        effective_duration = min(effective_duration, requested_preview_duration)
    total_frames = max(1, int(round(float(metadata["frameRate"]) * effective_duration)))
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

    _write_progress(
        progress_path,
        phase="queued",
        percent=0,
        message="Job queued",
        processed_frames=0,
        total_frames=total_frames,
    )

    extract_command = [
        ffmpeg,
        "-y",
        "-i",
        source_path,
    ]
    if preview_mode:
        extract_command.extend(["-t", f"{effective_duration:.3f}"])
    extract_command.extend([
        "-map",
        "0:v:0",
        str(input_frames / "frame_%08d.png"),
    ])

    _write_progress(
        progress_path,
        phase="extracting",
        percent=5,
        message="Extracting source frames",
        processed_frames=0,
        total_frames=total_frames,
    )
    _run(
        extract_command,
        log,
    )
    extracted_frame_count = len(list(input_frames.glob("frame_*.png")))
    _write_progress(
        progress_path,
        phase="extracting",
        percent=12,
        message=f"Extracted source frames ({extracted_frame_count}/{total_frames})",
        processed_frames=extracted_frame_count,
        total_frames=total_frames,
        extracted_frames=extracted_frame_count,
    )

    effective_tile = _effective_tile_size(preset, tile_size)

    _write_progress(
        progress_path,
        phase="upscaling",
        percent=15,
        message=f"Upscaling extracted frames with {model_name}",
        processed_frames=0,
        total_frames=total_frames,
    )
    upscaled_frame_count = _upscale_frames(
        runtime=runtime,
        input_frames=input_frames,
        output_frames=output_frames,
        model_id=model_id,
        gpu_id=gpu_id,
        effective_tile=effective_tile,
        progress_path=progress_path,
        total_frames=total_frames,
        log=log,
        work_dir=work_dir,
        fp16=fp16,
    )

    _write_progress(
        progress_path,
        phase="encoding",
        percent=88,
        message="Encoding upscaled frames",
        processed_frames=0,
        total_frames=total_frames,
        extracted_frames=extracted_frame_count,
        upscaled_frames=upscaled_frame_count,
    )
    encode_command = [
        ffmpeg,
        "-y",
        "-framerate",
        fps,
        "-i",
        str(output_frames / "frame_%08d.png"),
    ]

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
        ]
    )
    if container == "mp4":
        encode_command.extend(["-movflags", "+faststart"])
    encode_command.append(str(silent_video))
    encoded_frame_count, _ = _run_ffmpeg_with_frame_progress(
        encode_command,
        log,
        progress_path,
        phase="encoding",
        percent_base=88,
        percent_span=7,
        total_frames=total_frames,
        message_prefix="Encoding video frames",
        extracted_frames=extracted_frame_count,
        upscaled_frames=upscaled_frame_count,
        encoded_frames=0,
        remuxed_frames=0,
    )

    if bool(metadata["hasAudio"]):
        _write_progress(
            progress_path,
            phase="remuxing",
            percent=95,
            message="Re-syncing original audio",
            processed_frames=0,
            total_frames=total_frames,
            extracted_frames=extracted_frame_count,
            upscaled_frames=upscaled_frame_count,
            encoded_frames=encoded_frame_count,
        )
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
        remuxed_frame_count, _ = _run_ffmpeg_with_frame_progress(
            remux_command,
            log,
            progress_path,
            phase="remuxing",
            percent_base=95,
            percent_span=4,
            total_frames=total_frames,
            message_prefix="Remuxing original audio",
            extracted_frames=extracted_frame_count,
            upscaled_frames=upscaled_frame_count,
            encoded_frames=encoded_frame_count,
            remuxed_frames=0,
        )
    else:
        shutil.copy2(silent_video, output_file)
        remuxed_frame_count = encoded_frame_count

    frame_count = len(list(output_frames.glob("*.png")))
    _write_progress(
        progress_path,
        phase="completed",
        percent=100,
        message="Pipeline completed",
        processed_frames=frame_count,
        total_frames=total_frames,
        extracted_frames=extracted_frame_count,
        upscaled_frames=upscaled_frame_count,
        encoded_frames=encoded_frame_count,
        remuxed_frames=remuxed_frame_count,
    )
    return {
        "outputPath": str(output_file),
        "workDir": str(work_dir),
        "frameCount": frame_count,
        "hadAudio": bool(metadata["hasAudio"]),
        "codec": codec,
        "container": container,
        "runtime": runtime,
        "log": log + [
            f"Model: {model_name} ({model_id})",
            f"Resolved output canvas: {resolved_width}x{resolved_height} ({resolved_aspect_ratio:.4f}:1)",
            f"Preview mode: {'on' if preview_mode else 'off'}",
            f"Processed duration: {effective_duration:.2f}s",
        ],
    }
