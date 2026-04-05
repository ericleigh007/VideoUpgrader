import unittest
from unittest.mock import patch

from upscaler_worker.media import probe_video


class MediaProbeTests(unittest.TestCase):
    @patch("upscaler_worker.media.ensure_browser_preview", return_value="C:/fixtures/sample-preview.mp4")
    @patch("upscaler_worker.media.ensure_runtime_assets", return_value={"ffmpegPath": "C:/tools/ffmpeg.exe"})
    @patch("upscaler_worker.media.subprocess.run")
    def test_probe_video_parses_hevc_codec_without_breaking_dimensions(self, mock_run, *_mocks) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = """
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'sample.mp4':
  Duration: 00:00:12.50, start: 0.000000, bitrate: 4123 kb/s
  Stream #0:0: Video: hevc (Main) (hev1 / 0x31766568), yuv420p(tv, bt709), 1920x1080, 23.976 fps, 23.98 tbr, 24k tbn
  Stream #0:1: Audio: aac (LC), 48000 Hz, stereo, fltp, 128 kb/s
"""

        result = probe_video("C:/fixtures/sample.mp4")

        self.assertEqual(result["videoCodec"], "hevc")
        self.assertEqual(result["width"], 1920)
        self.assertEqual(result["height"], 1080)
        self.assertAlmostEqual(result["frameRate"], 23.976)
        self.assertEqual(result["container"], "mp4")
        self.assertTrue(result["hasAudio"])
        self.assertEqual(result["sourceBitrateKbps"], 4123)
        self.assertEqual(result["videoProfile"], "Main")
        self.assertEqual(result["pixelFormat"], "yuv420p(tv, bt709)")
        self.assertEqual(result["audioCodec"], "aac")
        self.assertEqual(result["audioProfile"], "LC")
        self.assertEqual(result["audioSampleRate"], 48000)
        self.assertEqual(result["audioChannels"], "stereo")
        self.assertEqual(result["audioBitrateKbps"], 128)


if __name__ == "__main__":
    unittest.main()
