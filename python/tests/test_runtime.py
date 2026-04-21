from __future__ import annotations

import os
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import patch

from upscaler_worker.runtime import download_file_with_retries, ensure_colormnet_runtime, ensure_runtime_assets, ensure_rvrt_model_weights, ensure_rvrt_repo


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

    def test_ensure_rvrt_model_weights_downloads_expected_release_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)

            def fake_urlretrieve(url: str, destination: Path) -> tuple[str, object]:
                Path(destination).write_bytes(b"rvrt-weight")
                return str(destination), None

            with patch("upscaler_worker.runtime.repo_root", return_value=temp_root), patch(
                "upscaler_worker.runtime.urllib.request.urlretrieve",
                side_effect=fake_urlretrieve,
            ) as urlretrieve_mock:
                result = ensure_rvrt_model_weights()

            self.assertTrue(Path(result["modelPath"]).exists())
            self.assertEqual(Path(result["modelPath"]).read_bytes(), b"rvrt-weight")
            self.assertIn("002_RVRT_videosr_bi_Vimeo_14frames.pth", urlretrieve_mock.call_args.args[0])

    def test_ensure_rvrt_repo_extracts_repo_archive_into_tmp_rvrt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            temp_runtime = temp_root / "artifacts" / "runtime"

            def fake_urlretrieve(_url: str, destination: Path) -> tuple[str, object]:
                archive_path = Path(destination)
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr("RVRT-main/main_test_rvrt.py", "print('rvrt')\n")
                    archive.writestr("RVRT-main/requirements.txt", "einops\n")
                return str(archive_path), None

            with patch("upscaler_worker.runtime.repo_root", return_value=temp_root), patch(
                "upscaler_worker.runtime.runtime_root",
                return_value=temp_runtime,
            ), patch(
                "upscaler_worker.runtime.urllib.request.urlretrieve",
                side_effect=fake_urlretrieve,
            ):
                result = ensure_rvrt_repo()

            self.assertTrue((temp_root / "tmp" / "RVRT" / "main_test_rvrt.py").exists())
            self.assertTrue((temp_root / "tmp" / "RVRT" / "requirements.txt").exists())
            self.assertEqual(result["rvrtRoot"], str(temp_root / "tmp" / "RVRT"))

    def test_ensure_colormnet_runtime_extracts_test_app(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            temp_runtime = temp_root / "artifacts" / "runtime"

            def fake_urlretrieve(_url: str, destination: Path) -> tuple[str, object]:
                archive_path = Path(destination)
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr("colormnet-main/test_app.py", "def run_cli(args_list=None):\n    return args_list\n")
                    archive.writestr("colormnet-main/model/network.py", "class ColorMNet: pass\n")
                return str(archive_path), None

            with patch("upscaler_worker.runtime.repo_root", return_value=temp_root), patch(
                "upscaler_worker.runtime.runtime_root",
                return_value=temp_runtime,
            ), patch(
                "upscaler_worker.runtime.urllib.request.urlretrieve",
                side_effect=fake_urlretrieve,
            ):
                result = ensure_colormnet_runtime()

            self.assertTrue((temp_runtime / "colormnet" / "src" / "test_app.py").exists())
            self.assertEqual(result["colormnetSourceRoot"], str(temp_runtime / "colormnet" / "src"))

    def test_download_file_with_retries_retries_transient_http_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "download.bin"
            attempts: list[int] = []

            def flaky_urlretrieve(_url: str, target: Path) -> tuple[str, object]:
                attempts.append(len(attempts))
                if len(attempts) < 3:
                    raise urllib.error.HTTPError("https://example.invalid/model.bin", 502, "Bad Gateway", hdrs=None, fp=None)
                Path(target).write_bytes(b"ok")
                return str(target), None

            with patch("upscaler_worker.runtime.urllib.request.urlretrieve", side_effect=flaky_urlretrieve), patch(
                "upscaler_worker.runtime.time.sleep",
                return_value=None,
            ):
                result = download_file_with_retries("https://example.invalid/model.bin", destination)

            self.assertEqual(result, destination)
            self.assertTrue(destination.exists())
            self.assertEqual(destination.read_bytes(), b"ok")
            self.assertEqual(len(attempts), 3)


if __name__ == "__main__":
    unittest.main()