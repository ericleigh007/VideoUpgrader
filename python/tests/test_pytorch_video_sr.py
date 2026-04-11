import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from upscaler_worker.models.pytorch_video_sr import (
    build_external_video_sr_command,
    resolve_external_video_sr_command_template,
    validate_external_video_sr_outputs,
)


class PytorchVideoSrTests(unittest.TestCase):
    def test_build_external_video_sr_command_expands_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir(parents=True)
            (input_dir / "frame_000001.png").touch()

            previous = os.environ.get("UPSCALER_RVRT_COMMAND")
            os.environ["UPSCALER_RVRT_COMMAND"] = "runner --input {input_dir} --output {output_dir} --model {model_id} --tile {tile_size} --frames {frame_count}"
            try:
                command = build_external_video_sr_command(
                    model_id="rvrt-x4",
                    input_dir=input_dir,
                    output_dir=output_dir,
                    tile_size=128,
                    precision="fp16",
                )
            finally:
                if previous is None:
                    os.environ.pop("UPSCALER_RVRT_COMMAND", None)
                else:
                    os.environ["UPSCALER_RVRT_COMMAND"] = previous

            self.assertEqual(
                command.command,
                [
                    "runner",
                    "--input",
                    str(input_dir),
                    "--output",
                    str(output_dir),
                    "--model",
                    "rvrt-x4",
                    "--tile",
                    "128",
                    "--frames",
                    "1",
                ],
            )
            self.assertEqual(command.command_env_var, "UPSCALER_RVRT_COMMAND")
            self.assertEqual(command.environment["UPSCALER_VIDEO_SR_MODEL_ID"], "rvrt-x4")
            self.assertEqual(command.environment["UPSCALER_VIDEO_SR_PRECISION"], "fp16")
            self.assertEqual(command.environment["UPSCALER_VIDEO_SR_COMMAND_SOURCE"], "environment")

    @patch("upscaler_worker.models.pytorch_video_sr._default_external_video_sr_command", return_value=None)
    def test_build_external_video_sr_command_requires_env_var(self, _default_command) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir(parents=True)
            (input_dir / "frame_000001.png").touch()

            previous = os.environ.pop("UPSCALER_RVRT_COMMAND", None)
            try:
                with self.assertRaises(RuntimeError):
                    build_external_video_sr_command(
                        model_id="rvrt-x4",
                        input_dir=input_dir,
                        output_dir=output_dir,
                        tile_size=128,
                        precision="fp32",
                    )
            finally:
                if previous is not None:
                    os.environ["UPSCALER_RVRT_COMMAND"] = previous

    @patch("upscaler_worker.models.pytorch_video_sr._default_external_video_sr_command", return_value="python -m upscaler_worker.rvrt_external_runner --input {input_dir} --output {output_dir} --model {model_id} --tile {tile_size} --precision {precision}")
    def test_build_external_video_sr_command_uses_repo_default_when_env_var_missing(self, _default_command) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir(parents=True)
            (input_dir / "frame_000001.png").touch()

            previous = os.environ.pop("UPSCALER_RVRT_COMMAND", None)
            try:
                command = build_external_video_sr_command(
                    model_id="rvrt-x4",
                    input_dir=input_dir,
                    output_dir=output_dir,
                    tile_size=192,
                    precision="bf16",
                )
            finally:
                if previous is not None:
                    os.environ["UPSCALER_RVRT_COMMAND"] = previous

            self.assertEqual(command.command_env_var, "UPSCALER_RVRT_COMMAND")
            self.assertEqual(command.environment["UPSCALER_VIDEO_SR_COMMAND_SOURCE"], "repo-default")
            self.assertIn("upscaler_worker.rvrt_external_runner", " ".join(command.command))
            self.assertIn("bf16", command.command)

    @patch("upscaler_worker.models.pytorch_video_sr._default_external_video_sr_command", return_value="python runner.py")
    def test_resolve_external_video_sr_command_template_prefers_environment(self, _default_command) -> None:
        previous = os.environ.get("UPSCALER_RVRT_COMMAND")
        os.environ["UPSCALER_RVRT_COMMAND"] = "custom runner"
        try:
            command, source = resolve_external_video_sr_command_template("rvrt-x4", "UPSCALER_RVRT_COMMAND")
        finally:
            if previous is None:
                os.environ.pop("UPSCALER_RVRT_COMMAND", None)
            else:
                os.environ["UPSCALER_RVRT_COMMAND"] = previous

        self.assertEqual(command, "custom runner")
        self.assertEqual(source, "environment")

    def test_validate_external_video_sr_outputs_requires_matching_frame_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            (input_dir / "frame_000001.png").touch()
            (input_dir / "frame_000002.png").touch()
            (output_dir / "frame_000001.png").touch()

            with self.assertRaises(RuntimeError):
                validate_external_video_sr_outputs(input_dir=input_dir, output_dir=output_dir)

    def test_validate_external_video_sr_outputs_accepts_matching_frame_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            (input_dir / "frame_000001.png").touch()
            (output_dir / "frame_000001.png").touch()

            self.assertEqual(validate_external_video_sr_outputs(input_dir=input_dir, output_dir=output_dir), 1)


if __name__ == "__main__":
    unittest.main()