import unittest
from pathlib import Path
from unittest.mock import patch

from upscaler_worker.benchmark_pytorch_pipeline_paths import (
    benchmark_pytorch_pipeline_paths,
    LiveRunMonitorState,
    _assess_gpu_activity,
)


class BenchmarkPytorchPipelinePathsTests(unittest.TestCase):
    def test_gpu_activity_detected_from_utilization(self) -> None:
        state = LiveRunMonitorState(sample_count=4, active_sample_count=2, max_utilization_percent=72)

        activity = _assess_gpu_activity(state)

        self.assertTrue(activity["activityDetected"])
        self.assertIsNone(activity["warning"])

    def test_gpu_activity_warns_when_utilization_and_memory_stay_flat(self) -> None:
        state = LiveRunMonitorState(
            sample_count=3,
            active_sample_count=0,
            max_utilization_percent=0,
            first_memory_used_bytes=8 * 1024 * 1024 * 1024,
            peak_memory_used_bytes=8 * 1024 * 1024 * 1024,
        )

        activity = _assess_gpu_activity(state)

        self.assertFalse(activity["activityDetected"])
        self.assertEqual(activity["observedMemoryDeltaBytes"], 0)
        self.assertIn("No meaningful GPU activity", activity["warning"])

    @patch("upscaler_worker.benchmark_pytorch_pipeline_paths.ensure_runtime_assets", return_value={"ffmpegPath": "ffmpeg"})
    @patch("upscaler_worker.benchmark_pytorch_pipeline_paths._build_fixture")
    @patch("upscaler_worker.benchmark_pytorch_pipeline_paths.model_backend_id", return_value="pytorch-image-sr")
    @patch("upscaler_worker.benchmark_pytorch_pipeline_paths.run_realesrgan_pipeline")
    @patch("upscaler_worker.models.pytorch_sr.load_runtime_model")
    def test_reuses_loaded_model_across_repeats(
        self,
        load_runtime_model_mock,
        run_pipeline_mock,
        _model_backend_id_mock,
        build_fixture_mock,
        _ensure_runtime_assets_mock,
    ) -> None:
        loaded_model = object()
        load_runtime_model_mock.return_value = loaded_model
        run_pipeline_mock.return_value = {
            "executionPath": "streaming",
            "averageThroughputFps": 1.0,
            "segmentCount": 1,
            "segmentFrameLimit": 24,
            "stageTimings": {},
            "resourcePeaks": {},
            "modelRuntime": {},
        }

        def fake_build_fixture(_ffmpeg_path: str, output_path: Path, *_args) -> None:
            output_path.write_bytes(b"fixture")

        build_fixture_mock.side_effect = fake_build_fixture

        result = benchmark_pytorch_pipeline_paths(
            model_id="swinir-realworld-x4",
            execution_paths=["streaming"],
            repeats=2,
            duration_seconds=1.0,
            width=320,
            height=180,
            fps=24,
            tile_size=128,
            output_mode="preserveAspect4k",
            resolution_basis="exact",
            target_width=3840,
            target_height=2160,
            preset="qualityBalanced",
            fp16=False,
            torch_compile_enabled=False,
            torch_compile_mode="reduce-overhead",
            torch_compile_cudagraphs=False,
            bf16=False,
            precision="fp32",
            pytorch_runner="tensorrt",
            reuse_loaded_model=True,
        )

        self.assertTrue(result["reuseLoadedModel"])
        load_runtime_model_mock.assert_called_once()
        self.assertEqual(run_pipeline_mock.call_count, 2)
        self.assertIs(run_pipeline_mock.call_args_list[0].kwargs["preloaded_pytorch_model"], loaded_model)
        self.assertIs(run_pipeline_mock.call_args_list[1].kwargs["preloaded_pytorch_model"], loaded_model)
        self.assertIn("gpuActivity", result["results"][0]["runs"][0])

    @patch("upscaler_worker.benchmark_pytorch_pipeline_paths.ensure_runtime_assets", return_value={"ffmpegPath": "ffmpeg"})
    @patch("upscaler_worker.benchmark_pytorch_pipeline_paths._build_fixture")
    @patch("upscaler_worker.benchmark_pytorch_pipeline_paths.model_backend_id", return_value="pytorch-image-sr")
    @patch("upscaler_worker.benchmark_pytorch_pipeline_paths.run_realesrgan_pipeline")
    @patch("upscaler_worker.models.pytorch_sr.load_runtime_model")
    def test_can_disable_loaded_model_reuse(
        self,
        load_runtime_model_mock,
        run_pipeline_mock,
        _model_backend_id_mock,
        build_fixture_mock,
        _ensure_runtime_assets_mock,
    ) -> None:
        run_pipeline_mock.return_value = {
            "executionPath": "streaming",
            "averageThroughputFps": 1.0,
            "segmentCount": 1,
            "segmentFrameLimit": 24,
            "stageTimings": {},
            "resourcePeaks": {},
            "modelRuntime": {},
        }

        def fake_build_fixture(_ffmpeg_path: str, output_path: Path, *_args) -> None:
            output_path.write_bytes(b"fixture")

        build_fixture_mock.side_effect = fake_build_fixture

        result = benchmark_pytorch_pipeline_paths(
            model_id="swinir-realworld-x4",
            execution_paths=["streaming"],
            repeats=2,
            duration_seconds=1.0,
            width=320,
            height=180,
            fps=24,
            tile_size=128,
            output_mode="preserveAspect4k",
            resolution_basis="exact",
            target_width=3840,
            target_height=2160,
            preset="qualityBalanced",
            fp16=False,
            torch_compile_enabled=False,
            torch_compile_mode="reduce-overhead",
            torch_compile_cudagraphs=False,
            bf16=False,
            precision="fp32",
            pytorch_runner="tensorrt",
            reuse_loaded_model=False,
        )

        self.assertFalse(result["reuseLoadedModel"])
        load_runtime_model_mock.assert_not_called()
        self.assertEqual(run_pipeline_mock.call_count, 2)
        self.assertIsNone(run_pipeline_mock.call_args_list[0].kwargs["preloaded_pytorch_model"])
        self.assertIsNone(run_pipeline_mock.call_args_list[1].kwargs["preloaded_pytorch_model"])


if __name__ == "__main__":
    unittest.main()