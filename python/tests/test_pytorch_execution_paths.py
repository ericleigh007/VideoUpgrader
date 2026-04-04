import unittest

from upscaler_worker.benchmark_pytorch_pipeline_paths import _parse_execution_paths
from upscaler_worker.pipeline import _resolve_pytorch_execution_path


class PytorchExecutionPathTests(unittest.TestCase):
    def test_parse_execution_paths_deduplicates_and_preserves_order(self) -> None:
        self.assertEqual(_parse_execution_paths("file-io, streaming, file-io"), ["file-io", "streaming"])

    def test_parse_execution_paths_rejects_unknown_path(self) -> None:
        with self.assertRaises(ValueError):
            _parse_execution_paths("file-io,wat")

    def test_resolve_pytorch_execution_path_defaults_to_file_io(self) -> None:
        self.assertEqual(_resolve_pytorch_execution_path("realesrnet-x4plus", None), "file-io")

    def test_resolve_pytorch_execution_path_ignores_non_pytorch_models(self) -> None:
        self.assertIsNone(_resolve_pytorch_execution_path("realesrgan-x4plus", None))


if __name__ == "__main__":
    unittest.main()