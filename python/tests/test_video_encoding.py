import unittest
from unittest.mock import patch

from upscaler_worker.video_encoding import (
    ENCODER_PROBE_FRAME_SIZE,
    VideoEncoderConfig,
    probe_video_encoder,
    resolve_video_encoder_config,
)


class VideoEncodingTests(unittest.TestCase):
    def test_probe_video_encoder_uses_workstation_safe_probe_frame_size(self) -> None:
        with patch("upscaler_worker.video_encoding.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 0
            run_mock.return_value.stderr = ""

            available, details = probe_video_encoder(
                "ffmpeg",
                VideoEncoderConfig(
                    encoder="h264_nvenc",
                    quality_args=("-preset", "p5", "-cq", "18", "-b:v", "0"),
                    label="nvenc",
                    hardware_accelerated=True,
                ),
            )

        self.assertTrue(available)
        self.assertIsNone(details)
        probe_command = run_mock.call_args.args[0]
        self.assertIn(f"color=c=black:s={ENCODER_PROBE_FRAME_SIZE}:d=0.1:r=1", probe_command)
        self.assertEqual(ENCODER_PROBE_FRAME_SIZE, "256x256")

    def test_resolve_video_encoder_config_prefers_h264_nvenc_for_h264(self) -> None:
        log: list[str] = []
        config = resolve_video_encoder_config(
            ffmpeg="ffmpeg",
            runtime={
                "availableGpus": [{"id": 1, "name": "NVIDIA RTX PRO 6000", "kind": "discrete"}],
            },
            gpu_id=1,
            codec="h264",
            crf=18,
            log=log,
            probe_encoder=lambda *_args: (True, None),
        )

        self.assertEqual(config.encoder, "h264_nvenc")
        self.assertTrue(config.hardware_accelerated)
        self.assertIn("Video encoder: h264_nvenc", log[0])

    def test_resolve_video_encoder_config_prefers_hevc_nvenc_for_h265(self) -> None:
        log: list[str] = []
        config = resolve_video_encoder_config(
            ffmpeg="ffmpeg",
            runtime={
                "availableGpus": [{"id": 1, "name": "NVIDIA RTX PRO 6000", "kind": "discrete"}],
            },
            gpu_id=1,
            codec="h265",
            crf=18,
            log=log,
            probe_encoder=lambda *_args: (True, None),
        )

        self.assertEqual(config.encoder, "hevc_nvenc")
        self.assertTrue(config.hardware_accelerated)
        self.assertIn("Video encoder: hevc_nvenc", log[0])

    def test_resolve_video_encoder_config_logs_probe_details_before_cpu_fallback(self) -> None:
        log: list[str] = []
        config = resolve_video_encoder_config(
            ffmpeg="ffmpeg",
            runtime={
                "availableGpus": [{"id": 1, "name": "NVIDIA RTX PRO 6000", "kind": "discrete"}],
            },
            gpu_id=1,
            codec="h265",
            crf=20,
            log=log,
            probe_encoder=lambda *_args: (False, "Frame Dimension less than the minimum supported value.\nextra detail"),
        )

        self.assertEqual(config.encoder, "libx265")
        self.assertFalse(config.hardware_accelerated)
        self.assertIn("Hardware encoder probe failed for hevc_nvenc", log[0])
        self.assertIn("Frame Dimension less than the minimum supported value.", log[0])
        self.assertIn("Video encoder: libx265", log[1])


if __name__ == "__main__":
    unittest.main()