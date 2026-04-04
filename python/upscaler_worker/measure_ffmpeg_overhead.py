from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from upscaler_worker.runtime import ensure_runtime_assets


@dataclass(frozen=True)
class BenchResult:
    name: str
    seconds: float


def _timed_run(command: list[str]) -> float:
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return elapsed


def _summarize(name: str, samples: list[float]) -> dict[str, float | str | list[float]]:
    median_seconds = statistics.median(samples)
    average_seconds = statistics.fmean(samples)
    return {
        "name": name,
        "samples": [round(sample, 6) for sample in samples],
        "minSeconds": round(min(samples), 6),
        "maxSeconds": round(max(samples), 6),
        "medianSeconds": round(median_seconds, 6),
        "averageSeconds": round(average_seconds, 6),
    }


def _build_fixture(ffmpeg_path: str, source_path: Path, duration_seconds: float, width: int, height: int, fps: int) -> None:
    source_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={width}x{height}:rate={fps}",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=880:sample_rate=48000",
        "-t",
        f"{duration_seconds:.3f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(source_path),
    ]
    _timed_run(command)


def _launch_baseline_command(ffmpeg_path: str) -> list[str]:
    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=size=16x16:rate=1:duration=0.04",
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]


def _decode_null_command(ffmpeg_path: str, source_path: Path) -> list[str]:
    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]


def _extract_png_command(ffmpeg_path: str, source_path: Path, output_dir: Path) -> list[str]:
    return [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        str(output_dir / "frame_%08d.png"),
    ]


def _encode_png_command(ffmpeg_path: str, png_dir: Path, fps: int, output_path: Path) -> list[str]:
    return [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(png_dir / "frame_%08d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]


def _remux_command(ffmpeg_path: str, encoded_video_path: Path, source_path: Path, output_path: Path) -> list[str]:
    return [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(encoded_video_path),
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c",
        "copy",
        "-shortest",
        str(output_path),
    ]


def benchmark_ffmpeg_overhead(
    *,
    repeats: int,
    duration_seconds: float,
    width: int,
    height: int,
    fps: int,
) -> dict[str, object]:
    runtime = ensure_runtime_assets()
    ffmpeg_path = str(runtime["ffmpegPath"])

    with tempfile.TemporaryDirectory(prefix="upscaler-ffmpeg-bench-") as temp_dir:
        root = Path(temp_dir)
        source_path = root / "fixture.mp4"
        extracted_dir = root / "extracted"
        encoded_video_path = root / "encoded.mkv"
        remuxed_output_path = root / "remuxed.mp4"

        _build_fixture(ffmpeg_path, source_path, duration_seconds, width, height, fps)

        extract_seed_dir = root / "extract-seed"
        extract_seed_dir.mkdir(parents=True, exist_ok=True)
        _timed_run(_extract_png_command(ffmpeg_path, source_path, extract_seed_dir))

        scenarios: list[tuple[str, callable]] = []

        def run_launch_baseline() -> float:
            return _timed_run(_launch_baseline_command(ffmpeg_path))

        def run_decode_null() -> float:
            return _timed_run(_decode_null_command(ffmpeg_path, source_path))

        def run_extract_png() -> float:
            if extracted_dir.exists():
                shutil.rmtree(extracted_dir)
            extracted_dir.mkdir(parents=True, exist_ok=True)
            return _timed_run(_extract_png_command(ffmpeg_path, source_path, extracted_dir))

        def run_encode_png() -> float:
            if encoded_video_path.exists():
                encoded_video_path.unlink()
            return _timed_run(_encode_png_command(ffmpeg_path, extract_seed_dir, fps, encoded_video_path))

        def run_remux() -> float:
            if remuxed_output_path.exists():
                remuxed_output_path.unlink()
            if not encoded_video_path.exists():
                _timed_run(_encode_png_command(ffmpeg_path, extract_seed_dir, fps, encoded_video_path))
            return _timed_run(_remux_command(ffmpeg_path, encoded_video_path, source_path, remuxed_output_path))

        scenarios = [
            ("launchBaseline", run_launch_baseline),
            ("decodeToNull", run_decode_null),
            ("extractToPng", run_extract_png),
            ("encodeFromPng", run_encode_png),
            ("remuxAudio", run_remux),
        ]

        measurements: dict[str, list[float]] = {}
        for name, runner in scenarios:
            runner()
            measurements[name] = [runner() for _ in range(repeats)]

    summaries = {name: _summarize(name, samples) for name, samples in measurements.items()}
    launch_median = float(summaries["launchBaseline"]["medianSeconds"])

    def share_of_launch(name: str) -> float:
        denominator = float(summaries[name]["medianSeconds"])
        if denominator <= 0:
            return 0.0
        return round((launch_median / denominator) * 100.0, 2)

    derived = {
        "launchSharePercent": {
            "decodeToNull": share_of_launch("decodeToNull"),
            "extractToPng": share_of_launch("extractToPng"),
            "encodeFromPng": share_of_launch("encodeFromPng"),
            "remuxAudio": share_of_launch("remuxAudio"),
        },
        "medianSecondsMinusLaunch": {
            "decodeToNull": round(max(0.0, float(summaries["decodeToNull"]["medianSeconds"]) - launch_median), 6),
            "extractToPng": round(max(0.0, float(summaries["extractToPng"]["medianSeconds"]) - launch_median), 6),
            "encodeFromPng": round(max(0.0, float(summaries["encodeFromPng"]["medianSeconds"]) - launch_median), 6),
            "remuxAudio": round(max(0.0, float(summaries["remuxAudio"]["medianSeconds"]) - launch_median), 6),
        },
        "estimatedPngWriteOverDecodeSeconds": round(
            max(0.0, float(summaries["extractToPng"]["medianSeconds"]) - float(summaries["decodeToNull"]["medianSeconds"])),
            6,
        ),
    }

    return {
        "ffmpegPath": ffmpeg_path,
        "repeats": repeats,
        "fixture": {
            "durationSeconds": duration_seconds,
            "width": width,
            "height": height,
            "fps": fps,
        },
        "measurements": summaries,
        "derived": derived,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure FFmpeg launch overhead against actual media work.")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--duration-seconds", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = benchmark_ffmpeg_overhead(
        repeats=args.repeats,
        duration_seconds=args.duration_seconds,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    payload = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())