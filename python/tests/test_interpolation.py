import unittest
from pathlib import Path
from unittest.mock import patch

from upscaler_worker.interpolation import (
    build_rife_command,
    resolve_output_fps,
    resolve_segment_output_frame_count,
    should_skip_interpolation,
    validate_interpolation_request,
)


class InterpolationTests(unittest.TestCase):
    def test_validate_interpolation_request_rejects_missing_target(self) -> None:
        with self.assertRaises(ValueError):
            validate_interpolation_request("afterUpscale", None)

    def test_resolve_output_fps_uses_target_when_enabled(self) -> None:
        self.assertEqual(resolve_output_fps(24.0, "afterUpscale", 60), 60.0)
        self.assertEqual(resolve_output_fps(24.0, "off", None), 24.0)

    def test_resolve_segment_output_frame_count_matches_segment_boundaries(self) -> None:
        self.assertEqual(
            resolve_segment_output_frame_count(start_frame=0, frame_count=24, source_fps=24.0, output_fps=60.0),
            60,
        )
        self.assertEqual(
            resolve_segment_output_frame_count(start_frame=24, frame_count=24, source_fps=24.0, output_fps=60.0),
            60,
        )

    def test_should_skip_interpolation_when_target_is_not_higher(self) -> None:
        self.assertTrue(should_skip_interpolation(input_frame_count=300, target_frame_count=300))
        self.assertFalse(should_skip_interpolation(input_frame_count=300, target_frame_count=750))

    def test_build_rife_command_includes_model_target_and_gpu(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            command = build_rife_command(
                executable_path="C:/tools/rife-ncnn-vulkan.exe",
                model_root="C:/tools/rife/models",
                input_dir=Path("C:/frames/in"),
                output_dir=Path("C:/frames/out"),
                target_frame_count=750,
                gpu_id=1,
                uhd_mode=True,
            )

        self.assertIn("-n", command)
        self.assertIn("750", command)
        self.assertIn("-g", command)
        self.assertIn("1", command)
        self.assertIn("-u", command)
        self.assertIn("-j", command)
        self.assertIn("8:4:8", command)
        self.assertIn("rife-v4.6", " ".join(command))

    def test_build_rife_command_allows_default_threading_override(self) -> None:
        with patch.dict("os.environ", {"UPSCALER_RIFE_THREADS": "default"}, clear=True):
            command = build_rife_command(
                executable_path="C:/tools/rife-ncnn-vulkan.exe",
                model_root="C:/tools/rife/models",
                input_dir=Path("C:/frames/in"),
                output_dir=Path("C:/frames/out"),
                target_frame_count=750,
                gpu_id=1,
                uhd_mode=True,
            )

        self.assertIn("-u", command)
        self.assertNotIn("-j", command)


if __name__ == "__main__":
    unittest.main()