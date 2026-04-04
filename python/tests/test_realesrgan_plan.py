import unittest

from upscaler_worker.models.realesrgan import build_realesrgan_job_plan


class RealesrganPlanTests(unittest.TestCase):
    def test_builds_portable_windows_command(self) -> None:
        plan = build_realesrgan_job_plan(
            source_path="C:/videos/input.mp4",
            model_id="realesrgan-x4plus",
            output_mode="preserveAspect4k",
            preset="qualityBalanced",
            interpolation_mode="afterUpscale",
            interpolation_target_fps=60,
            gpu_id=1,
            aspect_ratio_preset="1:1",
            custom_aspect_width=None,
            custom_aspect_height=None,
            resolution_basis="width",
            target_width=2048,
            target_height=None,
            crop_left=0.10,
            crop_top=0.20,
            crop_width=0.75,
            crop_height=0.75,
            segment_duration_seconds=20,
            output_path="artifacts/outputs/output.mp4",
            codec="h264",
            container="mp4",
            tile_size=256,
            fp16=False,
            precision="bf16",
            torch_compile_enabled=True,
            pytorch_execution_path="streaming",
            pytorch_runner="tensorrt",
            crf=18,
        )

        self.assertEqual(plan["model"], "Real-ESRGAN x4 Plus")
        self.assertEqual(plan["command"][:4], ["python", "-m", "upscaler_worker.cli", "run-realesrgan-pipeline"])
        self.assertIn("realesrgan-x4plus", plan["command"])
        self.assertIn("--gpu-id", plan["command"])
        self.assertIn("1", plan["command"])
        self.assertIn("--tile-size", plan["command"])
        self.assertIn("--segment-duration-seconds", plan["command"])
        self.assertIn("20", plan["command"])
        self.assertIn("--interpolation-mode", plan["command"])
        self.assertIn("afterUpscale", plan["command"])
        self.assertIn("--interpolation-target-fps", plan["command"])
        self.assertIn("60", plan["command"])
        self.assertIn("--precision", plan["command"])
        self.assertIn("bf16", plan["command"])
        self.assertIn("--torch-compile", plan["command"])
        self.assertIn("--pytorch-execution-path", plan["command"])
        self.assertIn("streaming", plan["command"])
        self.assertIn("--pytorch-runner", plan["command"])
        self.assertIn("tensorrt", plan["command"])
        self.assertIn("Backend: realesrgan-ncnn", plan["notes"])
        self.assertIn("Runtime name: realesrgan-x4plus", plan["notes"])
        self.assertIn("GPU: 1", plan["notes"])
        self.assertIn("Aspect ratio: 1:1", plan["notes"])
        self.assertIn("Target width: 2048", plan["notes"])
        self.assertIn("Crop rect: 0.10, 0.20, 0.75, 0.75", plan["notes"])
        self.assertIn("Interpolation mode: afterUpscale", plan["notes"])
        self.assertIn("Interpolation target fps: 60", plan["notes"])
        self.assertIn("Segment duration: 20.0s", plan["notes"])
        self.assertIn("Codec: h264", plan["notes"])
        self.assertIn("Precision: bf16", plan["notes"])
        self.assertIn("torch.compile requested for PyTorch image SR execution.", plan["notes"])
        self.assertIn("PyTorch execution path: streaming", plan["notes"])
        self.assertIn("PyTorch runner: tensorrt", plan["notes"])
        self.assertEqual(len(plan["cacheKey"]), 64)


if __name__ == "__main__":
    unittest.main()
