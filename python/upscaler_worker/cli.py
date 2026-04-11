from __future__ import annotations

import argparse
import json
from pathlib import Path

from upscaler_worker.benchmark import benchmark_fixture, compare_precision_quality, _parse_tile_sizes
from upscaler_worker.benchmark_pytorch_pipeline_paths import benchmark_pytorch_pipeline_paths, _parse_execution_paths
from upscaler_worker.media import convert_source_to_mp4, probe_video
from upscaler_worker.precision import resolve_precision_mode
from upscaler_worker.models.realesrgan import build_realesrgan_job_plan
from upscaler_worker.pipeline import run_realesrgan_pipeline
from upscaler_worker.runtime import ensure_runtime_assets
from upscaler_worker.synthetic.av_sync import generate_av_sync_fixture, validate_av_sync
from upscaler_worker.synthetic.generate_benchmarks import generate_benchmark_fixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="upscaler_worker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-realesrgan-job")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--model-id", default="realesrgan-x4plus")
    prepare.add_argument("--output-mode", required=True)
    prepare.add_argument("--preset", required=True)
    prepare.add_argument("--interpolation-mode", choices=["off", "afterUpscale", "interpolateOnly"], default="off")
    prepare.add_argument("--interpolation-target-fps", choices=[30, 60], type=int)
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
    prepare.add_argument("--segment-duration-seconds", type=float)
    prepare.add_argument("--output-path", required=True)
    prepare.add_argument("--codec", choices=["h264", "h265"], default="h264")
    prepare.add_argument("--container", choices=["mp4", "mkv"], default="mp4")
    prepare.add_argument("--tile-size", type=int, default=0)
    prepare.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
    prepare.add_argument("--fp16", action="store_true")
    prepare.add_argument("--bf16", action="store_true")
    prepare.add_argument("--torch-compile", action="store_true")
    prepare.add_argument("--torch-compile-mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    prepare.add_argument("--torch-compile-cudagraphs", action="store_true")
    prepare.add_argument("--channels-last", action="store_true")
    prepare.add_argument("--pytorch-execution-path")
    prepare.add_argument("--pytorch-runner", choices=["torch", "tensorrt"], default="torch")
    prepare.add_argument("--crf", type=int, default=18)

    subparsers.add_parser("ensure-runtime")

    probe = subparsers.add_parser("probe-video")
    probe.add_argument("--source", required=True)

    convert = subparsers.add_parser("convert-source-to-mp4")
    convert.add_argument("--source", required=True)
    convert.add_argument("--progress-path")
    convert.add_argument("--cancel-path")
    convert.add_argument("--pause-path")

    run_job = subparsers.add_parser("run-realesrgan-pipeline")
    run_job.add_argument("--source", required=True)
    run_job.add_argument("--model-id", default="realesrgan-x4plus")
    run_job.add_argument("--output-mode", required=True)
    run_job.add_argument("--preset", required=True)
    run_job.add_argument("--interpolation-mode", choices=["off", "afterUpscale", "interpolateOnly"], default="off")
    run_job.add_argument("--interpolation-target-fps", choices=[30, 60], type=int)
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
    run_job.add_argument("--job-id")
    run_job.add_argument("--progress-path")
    run_job.add_argument("--cancel-path")
    run_job.add_argument("--pause-path")
    run_job.add_argument("--preview-mode", action="store_true")
    run_job.add_argument("--preview-duration-seconds", type=float)
    run_job.add_argument("--preview-start-offset-seconds", type=float)
    run_job.add_argument("--segment-duration-seconds", type=float)
    run_job.add_argument("--output-path", required=True)
    run_job.add_argument("--codec", choices=["h264", "h265"], default="h264")
    run_job.add_argument("--container", choices=["mp4", "mkv"], default="mp4")
    run_job.add_argument("--tile-size", type=int, default=0)
    run_job.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
    run_job.add_argument("--fp16", action="store_true")
    run_job.add_argument("--bf16", action="store_true")
    run_job.add_argument("--torch-compile", action="store_true")
    run_job.add_argument("--torch-compile-mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    run_job.add_argument("--torch-compile-cudagraphs", action="store_true")
    run_job.add_argument("--channels-last", action="store_true")
    run_job.add_argument("--pytorch-execution-path")
    run_job.add_argument("--pytorch-runner", choices=["torch", "tensorrt"], default="torch")
    run_job.add_argument("--crf", type=int, default=18)

    benchmark = subparsers.add_parser("generate-benchmark")
    benchmark.add_argument("--output-dir", required=True)
    benchmark.add_argument("--name", default="default_fixture")
    benchmark.add_argument("--frames", type=int, default=12)
    benchmark.add_argument("--width", type=int, default=3840)
    benchmark.add_argument("--height", type=int, default=2160)
    benchmark.add_argument("--downscale-width", type=int, default=1280)
    benchmark.add_argument("--downscale-height", type=int, default=720)

    perf_benchmark = subparsers.add_parser("benchmark-upscaler")
    perf_benchmark.add_argument("--manifest-path", required=True)
    perf_benchmark.add_argument("--model-id", required=True)
    perf_benchmark.add_argument("--tile-sizes", default="128,256,384,512")
    perf_benchmark.add_argument("--repeats", type=int, default=2)
    perf_benchmark.add_argument("--gpu-id", type=int)
    perf_benchmark.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
    perf_benchmark.add_argument("--fp16", action="store_true")
    perf_benchmark.add_argument("--bf16", action="store_true")
    perf_benchmark.add_argument("--pytorch-runner", choices=["torch", "tensorrt"], default="torch")

    precision_compare = subparsers.add_parser("compare-precision-quality")
    precision_compare.add_argument("--manifest-path", required=True)
    precision_compare.add_argument("--model-id", required=True)
    precision_compare.add_argument("--tile-size", type=int, default=128)
    precision_compare.add_argument("--gpu-id", type=int)
    precision_compare.add_argument("--reference-precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    precision_compare.add_argument("--candidate-precision", default="bf16", choices=["fp32", "fp16", "bf16"])
    precision_compare.add_argument("--max-frames", type=int)
    precision_compare.add_argument("--pytorch-runner", choices=["torch", "tensorrt"], default="torch")

    path_benchmark = subparsers.add_parser("benchmark-pytorch-pipeline-paths")
    path_benchmark.add_argument("--execution-paths", default="file-io,streaming")
    path_benchmark.add_argument("--repeats", type=int, default=1)
    path_benchmark.add_argument("--duration-seconds", type=float, default=4.0)
    path_benchmark.add_argument("--width", type=int, default=320)
    path_benchmark.add_argument("--height", type=int, default=180)
    path_benchmark.add_argument("--fps", type=int, default=12)
    path_benchmark.add_argument("--tile-size", type=int, default=128)
    path_benchmark.add_argument("--model-id", default="realesrnet-x4plus")
    path_benchmark.add_argument("--preset", default="qualityBalanced", choices=["qualityMax", "qualityBalanced", "vramSafe"])
    path_benchmark.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
    path_benchmark.add_argument("--fp16", action="store_true")
    path_benchmark.add_argument("--bf16", action="store_true")
    path_benchmark.add_argument("--torch-compile", action="store_true")
    path_benchmark.add_argument("--torch-compile-mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    path_benchmark.add_argument("--torch-compile-cudagraphs", action="store_true")
    path_benchmark.add_argument("--channels-last", action="store_true")
    path_benchmark.add_argument("--pytorch-runner", choices=["torch", "tensorrt"], default="torch")
    path_benchmark.add_argument("--output-mode", default="preserveAspect4k")
    path_benchmark.add_argument("--resolution-basis", default="exact", choices=["exact", "width", "height"])
    path_benchmark.add_argument("--target-width", type=int, default=640)
    path_benchmark.add_argument("--target-height", type=int, default=360)

    av_fixture = subparsers.add_parser("generate-av-sync-fixture")
    av_fixture.add_argument("--output-path", required=True)
    av_fixture.add_argument("--duration-seconds", type=float, default=7200.0)
    av_fixture.add_argument("--width", type=int, default=1280)
    av_fixture.add_argument("--height", type=int, default=720)
    av_fixture.add_argument("--fps", type=int, default=30)
    av_fixture.add_argument("--flash-interval-seconds", type=float, default=1.0)
    av_fixture.add_argument("--flash-duration-seconds", type=float, default=0.08)

    av_validate = subparsers.add_parser("validate-av-sync")
    av_validate.add_argument("--media-path", required=True)
    av_validate.add_argument("--manifest-path", required=True)
    av_validate.add_argument("--tolerance-ms", type=float, default=45.0)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "fp16") and hasattr(args, "bf16"):
        try:
            selected_precision = resolve_precision_mode(
                fp16=getattr(args, "fp16", False),
                bf16=getattr(args, "bf16", False),
                precision=getattr(args, "precision", None),
            )
        except ValueError as error:
            parser.error(str(error))
    else:
        selected_precision = "fp32"

    if args.command == "prepare-realesrgan-job":
        result = build_realesrgan_job_plan(
            source_path=args.source,
            model_id=args.model_id,
            output_mode=args.output_mode,
            preset=args.preset,
            interpolation_mode=args.interpolation_mode,
            interpolation_target_fps=args.interpolation_target_fps,
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
            segment_duration_seconds=args.segment_duration_seconds,
            output_path=args.output_path,
            codec=args.codec,
            container=args.container,
            tile_size=args.tile_size,
            fp16=selected_precision == "fp16",
            bf16=selected_precision == "bf16",
            precision=selected_precision,
            torch_compile_enabled=args.torch_compile,
            torch_compile_mode=args.torch_compile_mode,
            torch_compile_cudagraphs=args.torch_compile_cudagraphs,
            channels_last=args.channels_last,
            pytorch_execution_path=args.pytorch_execution_path,
            pytorch_runner=args.pytorch_runner,
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

    if args.command == "convert-source-to-mp4":
        result = convert_source_to_mp4(args.source, progress_path=args.progress_path, cancel_path=args.cancel_path, pause_path=args.pause_path)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "run-realesrgan-pipeline":
        result = run_realesrgan_pipeline(
            source_path=args.source,
            model_id=args.model_id,
            output_mode=args.output_mode,
            preset=args.preset,
            interpolation_mode=args.interpolation_mode,
            interpolation_target_fps=args.interpolation_target_fps,
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
            job_id=args.job_id,
            progress_path=args.progress_path,
            cancel_path=args.cancel_path,
            pause_path=args.pause_path,
            preview_mode=args.preview_mode,
            preview_duration_seconds=args.preview_duration_seconds,
            preview_start_offset_seconds=args.preview_start_offset_seconds,
            segment_duration_seconds=args.segment_duration_seconds,
            output_path=args.output_path,
            codec=args.codec,
            container=args.container,
            tile_size=args.tile_size,
            fp16=selected_precision == "fp16",
            bf16=selected_precision == "bf16",
            precision=selected_precision,
            torch_compile_enabled=args.torch_compile,
            torch_compile_mode=args.torch_compile_mode,
            torch_compile_cudagraphs=args.torch_compile_cudagraphs,
            channels_last=args.channels_last,
            crf=args.crf,
            pytorch_execution_path=args.pytorch_execution_path,
            pytorch_runner=args.pytorch_runner,
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

    if args.command == "benchmark-upscaler":
        result = benchmark_fixture(
            manifest_path=Path(args.manifest_path),
            model_id=args.model_id,
            tile_sizes=_parse_tile_sizes(args.tile_sizes),
            repeats=args.repeats,
            gpu_id=args.gpu_id,
            fp16=selected_precision == "fp16",
            bf16=selected_precision == "bf16",
            precision=selected_precision,
            pytorch_runner=args.pytorch_runner,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "compare-precision-quality":
        result = compare_precision_quality(
            manifest_path=Path(args.manifest_path),
            model_id=args.model_id,
            tile_size=args.tile_size,
            gpu_id=args.gpu_id,
            reference_precision=args.reference_precision,
            candidate_precision=args.candidate_precision,
            max_frames=args.max_frames,
            pytorch_runner=args.pytorch_runner,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "benchmark-pytorch-pipeline-paths":
        result = benchmark_pytorch_pipeline_paths(
            model_id=args.model_id,
            execution_paths=_parse_execution_paths(args.execution_paths),
            repeats=args.repeats,
            duration_seconds=args.duration_seconds,
            width=args.width,
            height=args.height,
            fps=args.fps,
            tile_size=args.tile_size,
            output_mode=args.output_mode,
            resolution_basis=args.resolution_basis,
            target_width=args.target_width,
            target_height=args.target_height,
            preset=args.preset,
            fp16=selected_precision == "fp16",
            bf16=selected_precision == "bf16",
            precision=selected_precision,
            torch_compile_enabled=args.torch_compile,
            torch_compile_mode=args.torch_compile_mode,
            torch_compile_cudagraphs=args.torch_compile_cudagraphs,
            pytorch_runner=args.pytorch_runner,
            channels_last=args.channels_last,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "generate-av-sync-fixture":
        result = generate_av_sync_fixture(
            output_path=Path(args.output_path),
            duration_seconds=args.duration_seconds,
            width=args.width,
            height=args.height,
            fps=args.fps,
            flash_interval_seconds=args.flash_interval_seconds,
            flash_duration_seconds=args.flash_duration_seconds,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "validate-av-sync":
        result = validate_av_sync(
            media_path=Path(args.media_path),
            manifest_path=Path(args.manifest_path),
            tolerance_ms=args.tolerance_ms,
        )
        print(json.dumps(result, indent=2))
        return 0

    parser.error("Unsupported command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
