from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upscaler_worker.measure_ffmpeg_overhead import _build_fixture
from upscaler_worker.runtime import ensure_runtime_assets


def _timed_run(command: list[str], env: dict[str, str] | None = None, allowed_returncodes: set[int] | None = None) -> float:
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    elapsed = time.perf_counter() - started
    accepted_codes = allowed_returncodes or {0}
    if completed.returncode not in accepted_codes:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return elapsed


def _summarize(name: str, samples: list[float]) -> dict[str, object]:
    return {
        "name": name,
        "samples": [round(sample, 6) for sample in samples],
        "minSeconds": round(min(samples), 6),
        "maxSeconds": round(max(samples), 6),
        "medianSeconds": round(statistics.median(samples), 6),
        "averageSeconds": round(statistics.fmean(samples), 6),
    }


def _direct_probe_command(ffmpeg_path: str, source_path: Path) -> list[str]:
    return [ffmpeg_path, "-hide_banner", "-i", str(source_path)]


def _direct_convert_command(ffmpeg_path: str, source_path: Path, output_path: Path, use_nvenc: bool) -> list[str]:
    common = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-sn",
        "-dn",
        "-vf",
        "scale=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
    ]
    if use_nvenc:
        return common + [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p1",
            "-cq",
            "19",
            "-b:v",
            "0",
            str(output_path),
        ]
    return common + [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        str(output_path),
    ]


def benchmark_worker_cli_overhead(
    *,
    python_executable: str,
    repeats: int,
    duration_seconds: float,
    width: int,
    height: int,
    fps: int,
) -> dict[str, object]:
    runtime = ensure_runtime_assets()
    ffmpeg_path = str(runtime["ffmpegPath"])
    use_nvenc = any("nvidia" in str(device.get("name", "")).lower() for device in runtime.get("availableGpus", []) if isinstance(device, dict))

    with tempfile.TemporaryDirectory(prefix="upscaler-worker-cli-bench-") as temp_dir:
        root = Path(temp_dir)
        mp4_source = root / "fixture.mp4"
        avi_seed = root / "fixture.avi"
        _build_fixture(ffmpeg_path, mp4_source, duration_seconds, width, height, fps)

        avi_build_command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(mp4_source),
            "-c:v",
            "mpeg4",
            "-q:v",
            "3",
            "-c:a",
            "pcm_s16le",
            str(avi_seed),
        ]
        _timed_run(avi_build_command)

        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

        measurements: dict[str, list[float]] = {
            "pythonStartOnly": [],
            "workerEnsureRuntime": [],
            "directProbeVideo": [],
            "workerProbeVideo": [],
            "directConvertToMp4": [],
            "workerConvertToMp4": [],
        }

        for repeat_index in range(repeats):
            direct_output = root / f"direct_{repeat_index:02d}.mp4"
            worker_source = root / f"worker_source_{repeat_index:02d}.avi"
            worker_source.write_bytes(avi_seed.read_bytes())

            measurements["pythonStartOnly"].append(_timed_run([python_executable, "-c", "pass"], env=env))
            measurements["workerEnsureRuntime"].append(
                _timed_run([python_executable, "-m", "upscaler_worker.cli", "ensure-runtime"], env=env)
            )
            measurements["directProbeVideo"].append(
                _timed_run(_direct_probe_command(ffmpeg_path, mp4_source), env=env, allowed_returncodes={0, 1})
            )
            measurements["workerProbeVideo"].append(
                _timed_run([python_executable, "-m", "upscaler_worker.cli", "probe-video", "--source", str(mp4_source)], env=env)
            )
            measurements["directConvertToMp4"].append(
                _timed_run(_direct_convert_command(ffmpeg_path, worker_source, direct_output, use_nvenc), env=env)
            )
            measurements["workerConvertToMp4"].append(
                _timed_run([python_executable, "-m", "upscaler_worker.cli", "convert-source-to-mp4", "--source", str(worker_source)], env=env)
            )

            cached_outputs = list((Path(__file__).resolve().parents[2] / "artifacts" / "runtime" / "converted-sources").glob(f"{worker_source.stem}_*.mp4"))
            if cached_outputs:
                cached_outputs[0].unlink(missing_ok=True)

        summaries = {name: _summarize(name, samples) for name, samples in measurements.items()}
        derived = {
            "workerProbeMinusDirectProbeSeconds": round(
                float(summaries["workerProbeVideo"]["medianSeconds"]) - float(summaries["directProbeVideo"]["medianSeconds"]),
                6,
            ),
            "workerConvertMinusDirectConvertSeconds": round(
                float(summaries["workerConvertToMp4"]["medianSeconds"]) - float(summaries["directConvertToMp4"]["medianSeconds"]),
                6,
            ),
            "workerConvertMinusDirectConvertMinusTwoDirectProbesSeconds": round(
                float(summaries["workerConvertToMp4"]["medianSeconds"])
                - float(summaries["directConvertToMp4"]["medianSeconds"])
                - 2.0 * float(summaries["directProbeVideo"]["medianSeconds"]),
                6,
            ),
            "workerEnsureRuntimeMinusPythonStartSeconds": round(
                float(summaries["workerEnsureRuntime"]["medianSeconds"]) - float(summaries["pythonStartOnly"]["medianSeconds"]),
                6,
            ),
        }

    return {
        "pythonExecutable": python_executable,
        "ffmpegPath": ffmpeg_path,
        "repeats": repeats,
        "fixture": {
            "durationSeconds": duration_seconds,
            "width": width,
            "height": height,
            "fps": fps,
            "conversionSourceContainer": "avi",
            "encoderPath": "h264_nvenc" if use_nvenc else "libx264",
        },
        "measurements": summaries,
        "derived": derived,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure Python worker startup plus FFmpeg launch overhead.")
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--duration-seconds", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = benchmark_worker_cli_overhead(
        python_executable=args.python_executable,
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