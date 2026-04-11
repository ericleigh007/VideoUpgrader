from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from upscaler_worker.runtime import ensure_runtime_assets


class RuntimeStatusTests(unittest.TestCase):
    @patch("upscaler_worker.runtime.detect_available_gpus", return_value=([], None))
    @patch("upscaler_worker.runtime.ensure_realesrgan_runtime", return_value={"realesrganPath": "realesrgan.exe", "modelDir": "models"})
    @patch("imageio_ffmpeg.get_ffmpeg_exe", return_value="ffmpeg")
    @patch("upscaler_worker.runtime.resolve_external_video_sr_command_template", return_value=(None, "missing"))
    def test_ensure_runtime_reports_unconfigured_external_research_runtimes(self, _resolver, _ffmpeg, _realesrgan, _gpus) -> None:
        previous = os.environ.pop("UPSCALER_RVRT_COMMAND", None)
        try:
            runtime = ensure_runtime_assets()
        finally:
            if previous is not None:
                os.environ["UPSCALER_RVRT_COMMAND"] = previous

        rvrt = runtime["externalResearchRuntimes"]["rvrt-x4"]
        self.assertEqual(rvrt["kind"], "external-command")
        self.assertEqual(rvrt["commandEnvVar"], "UPSCALER_RVRT_COMMAND")
        self.assertFalse(rvrt["configured"])
        self.assertEqual(rvrt["source"], "missing")

    @patch("upscaler_worker.runtime.detect_available_gpus", return_value=([], None))
    @patch("upscaler_worker.runtime.ensure_realesrgan_runtime", return_value={"realesrganPath": "realesrgan.exe", "modelDir": "models"})
    @patch("imageio_ffmpeg.get_ffmpeg_exe", return_value="ffmpeg")
    @patch("upscaler_worker.runtime.resolve_external_video_sr_command_template", return_value=("python runner.py --input {input_dir} --output {output_dir}", "environment"))
    def test_ensure_runtime_reports_configured_external_research_runtimes(self, _resolver, _ffmpeg, _realesrgan, _gpus) -> None:
        runtime = ensure_runtime_assets()

        rvrt = runtime["externalResearchRuntimes"]["rvrt-x4"]
        self.assertTrue(rvrt["configured"])
        self.assertEqual(rvrt["source"], "environment")


if __name__ == "__main__":
    unittest.main()