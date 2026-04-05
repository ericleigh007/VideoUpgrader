import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from upscaler_worker.interpolation import resolve_output_fps
from upscaler_worker.runtime import ensure_runtime_assets
from upscaler_worker.synthetic.av_sync import generate_av_sync_fixture, validate_av_sync


def _run_ffmpeg(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg command failed: {' '.join(command)}\n{completed.stderr.strip()}")


class InterpolationAvSyncTests(unittest.TestCase):
    def test_simulated_2x_interpolation_pipeline_preserves_av_sync(self) -> None:
        runtime = ensure_runtime_assets()
        ffmpeg_path = str(runtime["ffmpegPath"])

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_path = temp_root / "source_fixture.mp4"
            source_manifest = generate_av_sync_fixture(
                output_path=source_path,
                duration_seconds=3.2,
                width=320,
                height=180,
                fps=30,
                flash_interval_seconds=0.5,
                flash_duration_seconds=0.08,
            )

            extracted_dir = temp_root / "extracted"
            extracted_dir.mkdir(parents=True, exist_ok=True)
            _run_ffmpeg([
                ffmpeg_path,
                "-y",
                "-i",
                str(source_path),
                "-map",
                "0:v:0",
                str(extracted_dir / "%08d.png"),
            ])

            source_frames = sorted(extracted_dir.glob("*.png"))
            self.assertGreater(len(source_frames), 0)

            interpolated_dir = temp_root / "interpolated"
            interpolated_dir.mkdir(parents=True, exist_ok=True)
            for index, frame_path in enumerate(source_frames, start=1):
                shutil.copy2(frame_path, interpolated_dir / f"{(index - 1) * 2 + 1:08d}.png")
                shutil.copy2(frame_path, interpolated_dir / f"{(index - 1) * 2 + 2:08d}.png")

            output_fps = resolve_output_fps(float(source_manifest["fps"]), "interpolateOnly", 60)
            silent_interpolated_path = temp_root / "interpolated_silent.mp4"
            _run_ffmpeg([
                ffmpeg_path,
                "-y",
                "-framerate",
                f"{output_fps:.6f}".rstrip("0").rstrip("."),
                "-i",
                str(interpolated_dir / "%08d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(silent_interpolated_path),
            ])

            remuxed_output_path = temp_root / "interpolated_with_audio.mp4"
            _run_ffmpeg([
                ffmpeg_path,
                "-y",
                "-i",
                str(silent_interpolated_path),
                "-i",
                str(source_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-shortest",
                str(remuxed_output_path),
            ])

            remux_manifest_path = temp_root / "interpolated_with_audio.sync.json"
            remux_manifest = dict(source_manifest)
            remux_manifest["outputPath"] = str(remuxed_output_path.resolve())
            remux_manifest["fps"] = int(output_fps)
            remux_manifest_path.write_text(json.dumps(remux_manifest, indent=2), encoding="utf-8")

            validation = validate_av_sync(
                media_path=remuxed_output_path,
                manifest_path=remux_manifest_path,
                tolerance_ms=80.0,
            )

            self.assertTrue(validation["passed"])
            self.assertGreaterEqual(validation["comparedEventCount"], 4)
            self.assertLessEqual(abs(float(validation["maxAvOffsetMs"])), 80.0)


if __name__ == "__main__":
    unittest.main()