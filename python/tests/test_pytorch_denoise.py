import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from upscaler_worker.models.pytorch_denoise import build_external_denoise_command


class PytorchDenoiseCommandTests(unittest.TestCase):
    @patch("upscaler_worker.models.pytorch_denoise.model_label", return_value="DRUNet Controllable Denoise")
    @patch("upscaler_worker.models.pytorch_denoise.model_research_runtime")
    @patch("upscaler_worker.models.pytorch_denoise.ensure_runnable_model")
    def test_repo_default_command_propagates_gpu_without_cuda_visibility_poisoning(
        self,
        _ensure_runnable_model_mock,
        model_research_runtime_mock,
        _model_label_mock,
    ) -> None:
        model_research_runtime_mock.return_value = {
            "kind": "external-command",
            "commandEnvVar": "UPSCALER_DRUNET_DENOISE_COMMAND",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            Image.new("RGB", (8, 8), color=(16, 16, 16)).save(input_dir / "frame_00000001.png")

            command = build_external_denoise_command(
                model_id="drunet-gray-color-denoise",
                input_dir=input_dir,
                output_dir=output_dir,
                gpu_id=1,
                precision="bf16",
            )

        self.assertEqual(command.environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", command.environment)
        self.assertEqual(command.environment["UPSCALER_DENOISE_GPU_ID"], "1")
        self.assertEqual(command.environment["UPSCALER_DENOISE_PRECISION"], "bf16")
        self.assertIn("--gpu-id", command.command)
        self.assertIn("1", command.command)
        self.assertIn("--precision", command.command)


if __name__ == "__main__":
    unittest.main()
