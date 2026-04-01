from __future__ import annotations

import argparse
import json
from pathlib import Path

from upscaler_worker.media import probe_video
from upscaler_worker.models.realesrgan import build_realesrgan_job_plan
from upscaler_worker.pipeline import run_realesrgan_pipeline
from upscaler_worker.runtime import ensure_runtime_assets
from upscaler_worker.synthetic.generate_benchmarks import generate_benchmark_fixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="upscaler_worker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-realesrgan-job")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--model-id", default="realesrgan-x4plus")
    prepare.add_argument("--output-mode", required=True)
    prepare.add_argument("--preset", required=True)
    prepare.add_argument("--gpu-id", type=int)
    prepare.add_argument("--aspect-ratio-preset", default="16:9")
    prepare.add_argument("--custom-aspect-width", type=int)
    prepare.add_argument("--custom-aspect-height", type=int)
    prepare.add_argument("--resolution-basis", choices=["exact", "width", "height"], default="exact")
    prepare.add_argument("--target-width", type=int)
    prepare.add_argument("--target-height", type=int)
    prepare.add_argument("--crop-left", type=float)
    prepare.add_argument("--crop-top", type=float)
    prepare.add_argument("--crop-width", type=float)
    prepare.add_argument("--crop-height", type=float)
    prepare.add_argument("--preview-mode", action="store_true")
    prepare.add_argument("--preview-duration-seconds", type=float)
    prepare.add_argument("--output-path", required=True)
    prepare.add_argument("--codec", choices=["h264", "h265"], default="h264")
    prepare.add_argument("--container", choices=["mp4", "mkv"], default="mp4")
    prepare.add_argument("--tile-size", type=int, default=0)
    prepare.add_argument("--fp16", action="store_true")
    prepare.add_argument("--crf", type=int, default=18)

    subparsers.add_parser("ensure-runtime")

    probe = subparsers.add_parser("probe-video")
    probe.add_argument("--source", required=True)

    run_job = subparsers.add_parser("run-realesrgan-pipeline")
    run_job.add_argument("--source", required=True)
    run_job.add_argument("--model-id", default="realesrgan-x4plus")
    run_job.add_argument("--output-mode", required=True)
    run_job.add_argument("--preset", required=True)
    run_job.add_argument("--gpu-id", type=int)
    run_job.add_argument("--aspect-ratio-preset", default="16:9")
    run_job.add_argument("--custom-aspect-width", type=int)
    run_job.add_argument("--custom-aspect-height", type=int)
    run_job.add_argument("--resolution-basis", choices=["exact", "width", "height"], default="exact")
    run_job.add_argument("--target-width", type=int)
    run_job.add_argument("--target-height", type=int)
    run_job.add_argument("--crop-left", type=float)
    run_job.add_argument("--crop-top", type=float)
    run_job.add_argument("--crop-width", type=float)
    run_job.add_argument("--crop-height", type=float)
    run_job.add_argument("--progress-path")
    run_job.add_argument("--preview-mode", action="store_true")
    run_job.add_argument("--preview-duration-seconds", type=float)
    run_job.add_argument("--output-path", required=True)
    run_job.add_argument("--codec", choices=["h264", "h265"], default="h264")
    run_job.add_argument("--container", choices=["mp4", "mkv"], default="mp4")
    run_job.add_argument("--tile-size", type=int, default=0)
    run_job.add_argument("--fp16", action="store_true")
    run_job.add_argument("--crf", type=int, default=18)

    benchmark = subparsers.add_parser("generate-benchmark")
    benchmark.add_argument("--output-dir", required=True)
    benchmark.add_argument("--name", default="default_fixture")
    benchmark.add_argument("--frames", type=int, default=12)
    benchmark.add_argument("--width", type=int, default=3840)
    benchmark.add_argument("--height", type=int, default=2160)
    benchmark.add_argument("--downscale-width", type=int, default=1280)
    benchmark.add_argument("--downscale-height", type=int, default=720)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-realesrgan-job":
        result = build_realesrgan_job_plan(
            source_path=args.source,
            model_id=args.model_id,
            output_mode=args.output_mode,
            preset=args.preset,
            gpu_id=args.gpu_id,
            aspect_ratio_preset=args.aspect_ratio_preset,
            custom_aspect_width=args.custom_aspect_width,
            custom_aspect_height=args.custom_aspect_height,
            resolution_basis=args.resolution_basis,
            target_width=args.target_width,
            target_height=args.target_height,
            crop_left=args.crop_left,
            crop_top=args.crop_top,
            crop_width=args.crop_width,
            crop_height=args.crop_height,
            preview_mode=args.preview_mode,
            preview_duration_seconds=args.preview_duration_seconds,
            output_path=args.output_path,
            codec=args.codec,
            container=args.container,
            tile_size=args.tile_size,
            fp16=args.fp16,
            crf=args.crf,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "ensure-runtime":
        result = ensure_runtime_assets()
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "probe-video":
        result = probe_video(args.source)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "run-realesrgan-pipeline":
        result = run_realesrgan_pipeline(
            source_path=args.source,
            model_id=args.model_id,
            output_mode=args.output_mode,
            preset=args.preset,
            gpu_id=args.gpu_id,
            aspect_ratio_preset=args.aspect_ratio_preset,
            custom_aspect_width=args.custom_aspect_width,
            custom_aspect_height=args.custom_aspect_height,
            resolution_basis=args.resolution_basis,
            target_width=args.target_width,
            target_height=args.target_height,
            crop_left=args.crop_left,
            crop_top=args.crop_top,
            crop_width=args.crop_width,
            crop_height=args.crop_height,
            progress_path=args.progress_path,
            preview_mode=args.preview_mode,
            preview_duration_seconds=args.preview_duration_seconds,
            output_path=args.output_path,
            codec=args.codec,
            container=args.container,
            tile_size=args.tile_size,
            fp16=args.fp16,
            crf=args.crf,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "generate-benchmark":
        result = generate_benchmark_fixture(
            output_dir=Path(args.output_dir),
            name=args.name,
            frames=args.frames,
            width=args.width,
            height=args.height,
            downscale_width=args.downscale_width,
            downscale_height=args.downscale_height,
        )
        print(json.dumps(result, indent=2))
        return 0

    parser.error("Unsupported command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
