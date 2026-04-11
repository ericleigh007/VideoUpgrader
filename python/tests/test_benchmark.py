import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from upscaler_worker.benchmark import (
    LiveRunMonitorState,
    _assess_gpu_activity,
    _compare_frame_pair,
    _parse_tile_sizes,
    _summarize_frame_metrics,
    benchmark_fixture,
)
from upscaler_worker.synthetic.generate_benchmarks import generate_benchmark_fixture


class BenchmarkHarnessTests(unittest.TestCase):
    def test_gpu_activity_detected_from_utilization(self) -> None:
        state = LiveRunMonitorState(sample_count=2, active_sample_count=1, max_utilization_percent=81)

        activity = _assess_gpu_activity(state)

        self.assertTrue(activity["activityDetected"])
        self.assertIsNone(activity["warning"])

    def test_gpu_activity_warns_when_run_stays_idle(self) -> None:
        state = LiveRunMonitorState(
            sample_count=3,
            active_sample_count=0,
            max_utilization_percent=0,
            first_memory_used_bytes=10 * 1024 * 1024 * 1024,
            peak_memory_used_bytes=10 * 1024 * 1024 * 1024,
        )

        activity = _assess_gpu_activity(state)

        self.assertFalse(activity["activityDetected"])
        self.assertEqual(activity["observedMemoryDeltaBytes"], 0)
        self.assertIn("No meaningful GPU activity", activity["warning"])

    def test_parse_tile_sizes_sorts_and_deduplicates(self) -> None:
        self.assertEqual(_parse_tile_sizes("384,128,256,256"), [128, 256, 384])

    def test_parse_tile_sizes_requires_values(self) -> None:
        with self.assertRaises(ValueError):
            _parse_tile_sizes("  ")

    def test_benchmark_fixture_rejects_empty_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            manifest_path.write_text(json.dumps({"entries": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                benchmark_fixture(
                    manifest_path=manifest_path,
                    model_id="realesrgan-x4plus",
                    tile_sizes=[128],
                    repeats=1,
                    gpu_id=None,
                    fp16=False,
                )

    def test_benchmark_fixture_reports_metadata_for_runnable_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = generate_benchmark_fixture(
                output_dir=Path(temp_dir),
                name="fixture_a",
                frames=2,
                width=640,
                height=360,
                downscale_width=320,
                downscale_height=180,
            )
            with patch("upscaler_worker.benchmark._benchmark_ncnn_fixture", return_value={"tileSize": 128, "summary": {}}):
                result = benchmark_fixture(
                    manifest_path=Path(temp_dir) / "fixture_a" / "manifest.json",
                    model_id="realesrgan-x4plus",
                    tile_sizes=[128],
                    repeats=1,
                    gpu_id=None,
                    fp16=False,
                )
            self.assertEqual(result["fixtureName"], manifest["name"])
            self.assertEqual(result["frameCount"], 2)
            self.assertEqual(result["modelId"], "realesrgan-x4plus")
            self.assertEqual(result["backendId"], "realesrgan-ncnn")
            self.assertEqual(len(result["results"]), 1)

    def test_benchmark_fixture_allows_research_video_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            generate_benchmark_fixture(
                output_dir=Path(temp_dir),
                name="fixture_video",
                frames=2,
                width=640,
                height=360,
                downscale_width=320,
                downscale_height=180,
            )
            with patch("upscaler_worker.benchmark._benchmark_pytorch_video_fixture", return_value={"tileSize": 128, "summary": {}}):
                result = benchmark_fixture(
                    manifest_path=Path(temp_dir) / "fixture_video" / "manifest.json",
                    model_id="rvrt-x4",
                    tile_sizes=[128],
                    repeats=1,
                    gpu_id=None,
                    fp16=False,
                )

            self.assertEqual(result["modelId"], "rvrt-x4")
            self.assertEqual(result["backendId"], "pytorch-video-sr")
            self.assertEqual(result["runner"], "external-executable")
            self.assertEqual(len(result["results"]), 1)

    def test_compare_frame_pair_reports_identical_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            left = Path(temp_dir) / "left.png"
            right = Path(temp_dir) / "right.png"
            pixels = np.full((8, 8, 3), 127, dtype=np.uint8)
            from PIL import Image

            Image.fromarray(pixels, mode="RGB").save(left)
            Image.fromarray(pixels, mode="RGB").save(right)

            metrics = _compare_frame_pair(left, right)
            self.assertEqual(metrics["meanAbsoluteError"], 0.0)
            self.assertEqual(metrics["maxAbsoluteError"], 0)
            self.assertEqual(metrics["ssim"], 1.0)
            self.assertTrue(float(metrics["psnr"]) > 1000.0 or metrics["psnr"] == float("inf"))

    def test_summarize_frame_metrics_aggregates_values(self) -> None:
        summary = _summarize_frame_metrics(
            [
                {"frame": "a", "meanAbsoluteError": 1.0, "rootMeanSquaredError": 2.0, "maxAbsoluteError": 3, "psnr": 40.0, "ssim": 0.99},
                {"frame": "b", "meanAbsoluteError": 2.0, "rootMeanSquaredError": 4.0, "maxAbsoluteError": 5, "psnr": 38.0, "ssim": 0.97},
            ]
        )
        self.assertEqual(summary["frameCount"], 2)
        self.assertEqual(summary["averageMeanAbsoluteError"], 1.5)
        self.assertEqual(summary["maxAbsoluteError"], 5)
        self.assertEqual(summary["averagePsnr"], 39.0)


if __name__ == "__main__":
    unittest.main()