from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from upscaler_worker.model_catalog import model_backend_id, model_catalog, model_label
from upscaler_worker.pipeline import _denoise_segment


DEFAULT_MODELS = [
    "ffmpeg-hqdn3d-balanced",
    "ffmpeg-hqdn3d-strong",
    "ffmpeg-nlmeans-quality",
]
DEFAULT_CONTROL_SAFE_BOTTOM_PIXELS = 120


def _comparison_models(include_ai: bool, requested_models: list[str] | None) -> list[str]:
    if requested_models:
        return requested_models
    if not include_ai:
        return list(DEFAULT_MODELS)
    return [
        str(model["id"])
        for model in sorted(model_catalog(), key=lambda entry: int(entry.get("qualityRank", 999)))
        if str(model.get("task", "upscale")) == "denoise"
        and bool(model.get("comparisonEligible"))
        and str(model.get("executionStatus")) == "runnable"
    ]


def _model_set_suffix(model_ids: list[str]) -> str:
    if not model_ids:
        return ""
    joined = "-".join(model_ids[:3])
    suffix = "".join(character if character.isalnum() or character in "._-" else "-" for character in joined).strip("-._")
    if len(model_ids) > 3:
        suffix += f"-plus{len(model_ids) - 3}"
    return suffix


def _extract_sample_frames(
    *,
    ffmpeg: str,
    source: Path,
    input_dir: Path,
    start_seconds: float,
    duration_seconds: float,
    fps: int,
) -> int:
    input_dir.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        f"fps={fps}",
        str(input_dir / "frame_%08d.png"),
    ]
    subprocess.run(command, check=True)
    return len(list(input_dir.glob("frame_*.png")))


def _render_quad_preview(
    *,
    ffmpeg: str,
    roots: list[Path],
    output_video: Path,
    output_frame: Path,
    fps: int,
    frame_index: int,
    control_safe_bottom_pixels: int,
) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    video_command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(roots[0] / "frame_%08d.png"),
        "-framerate",
        str(fps),
        "-i",
        str(roots[1] / "frame_%08d.png"),
        "-framerate",
        str(fps),
        "-i",
        str(roots[2] / "frame_%08d.png"),
        "-framerate",
        str(fps),
        "-i",
        str(roots[3] / "frame_%08d.png"),
        "-filter_complex",
        f"[0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[quad];[quad]pad=iw:ih+{max(0, control_safe_bottom_pixels)}:0:0:color=black,format=yuv420p[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        str(output_video),
    ]
    subprocess.run(video_command, check=True)

    frame_name = f"frame_{frame_index:08d}.png"
    frame_command = [
        ffmpeg,
        "-y",
        "-i",
        str(roots[0] / frame_name),
        "-i",
        str(roots[1] / frame_name),
        "-i",
        str(roots[2] / frame_name),
        "-i",
        str(roots[3] / frame_name),
        "-filter_complex",
        "[0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2",
        "-frames:v",
        "1",
        "-update",
        "1",
        str(output_frame),
    ]
    subprocess.run(frame_command, check=True)


def compare_denoisers(
    *,
    source: Path,
    output_dir: Path,
    work_dir: Path,
    start_seconds: float = 120.0,
    duration_seconds: float = 4.0,
    fps: int = 12,
    models: list[str] | None = None,
    include_ai: bool = False,
    gpu_id: int | None = None,
    precision: str = "fp32",
    keep_work_dir: bool = False,
    control_safe_bottom_pixels: int = DEFAULT_CONTROL_SAFE_BOTTOM_PIXELS,
) -> dict[str, object]:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    selected_models = _comparison_models(include_ai, models)
    if len(selected_models) < 3:
        raise ValueError("At least three denoiser models are required so the quad preview can include original plus three outputs")

    if work_dir.exists():
        shutil.rmtree(work_dir)
    input_dir = work_dir / "input"
    extracted_count = _extract_sample_frames(
        ffmpeg=ffmpeg,
        source=source,
        input_dir=input_dir,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        fps=fps,
    )
    if extracted_count <= 0:
        raise RuntimeError("Sample extraction did not produce any frames")

    runtime = {"ffmpegPath": ffmpeg}
    log: list[str] = []
    results: list[dict[str, object]] = []
    successful_roots = [input_dir]
    started_at = time.time()
    for model_id in selected_models:
        output_root = work_dir / model_id
        model_started_at = time.time()
        try:
            output_count = _denoise_segment(
                runtime=runtime,
                input_dir=input_dir,
                output_dir=output_root,
                model_id=model_id,
                gpu_id=gpu_id,
                precision=precision,
                log=log,
                cancel_path=None,
                pause_path=None,
            )
            elapsed = time.time() - model_started_at
            results.append(
                {
                    "modelId": model_id,
                    "label": model_label(model_id),
                    "backendId": model_backend_id(model_id),
                    "status": "succeeded",
                    "frames": output_count,
                    "seconds": elapsed,
                    "framesPerSecond": output_count / elapsed if elapsed > 0 else None,
                    "outputDir": str(output_root),
                }
            )
            successful_roots.append(output_root)
        except Exception as error:  # noqa: BLE001
            results.append(
                {
                    "modelId": model_id,
                    "label": model_label(model_id),
                    "backendId": model_backend_id(model_id),
                    "status": "skipped",
                    "reason": str(error),
                    "outputDir": str(output_root),
                }
            )

    successful_models = [result for result in results if result["status"] == "succeeded"]
    if len(successful_roots) < 4:
        missing = 4 - len(successful_roots)
        raise RuntimeError(f"Only {len(successful_models)} denoisers succeeded; need {missing} more for a quad preview")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem.replace(" ", "-").replace("[", "").replace("]", "")[:80]
    sample_suffix = f"s{start_seconds:g}-d{duration_seconds:g}-fps{fps}"
    if include_ai:
        sample_suffix += "-ai"
    requested_suffix = _model_set_suffix(selected_models) if models else ""
    if requested_suffix:
        sample_suffix += f"-{requested_suffix}"
    output_video = output_dir / f"{stem}-{sample_suffix}-denoise-comparison.mp4"
    output_frame = output_dir / f"{stem}-{sample_suffix}-denoise-comparison-frame.png"
    _render_quad_preview(
        ffmpeg=ffmpeg,
        roots=successful_roots[:4],
        output_video=output_video,
        output_frame=output_frame,
        fps=fps,
        frame_index=max(1, min(extracted_count, extracted_count // 2)),
        control_safe_bottom_pixels=control_safe_bottom_pixels,
    )

    summary = {
        "source": str(source),
        "sample": {
            "startSeconds": start_seconds,
            "durationSeconds": duration_seconds,
            "fps": fps,
            "frames": extracted_count,
        },
        "includeAi": include_ai,
        "models": results,
        "quadLayout": {
            "topLeft": "original",
            "topRight": successful_models[0]["modelId"],
            "bottomLeft": successful_models[1]["modelId"],
            "bottomRight": successful_models[2]["modelId"],
        },
        "playerSafeArea": {
            "bottomPixels": control_safe_bottom_pixels,
            "appliesTo": "video",
        },
        "outputs": {
            "video": str(output_video),
            "frame": str(output_frame),
            "workDir": str(work_dir) if keep_work_dir else None,
        },
        "seconds": time.time() - started_at,
    }
    manifest_path = output_dir / f"{stem}-{sample_suffix}-denoise-comparison.json"
    summary["outputs"]["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
    return summary