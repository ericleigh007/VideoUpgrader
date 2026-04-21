import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from upscaler_worker.models import colormnet
from upscaler_worker.models.colormnet import _build_args_list, _ensure_progressbar_module, _resolve_reference_image_path


class ColorMNetTests(unittest.TestCase):
    def test_ensure_progressbar_module_installs_identity_shim_when_missing(self) -> None:
        with patch("upscaler_worker.models.colormnet.importlib.import_module", side_effect=ModuleNotFoundError()):
            with patch.dict(colormnet.sys.modules, {}, clear=False):
                _ensure_progressbar_module()

                self.assertIn("progressbar", colormnet.sys.modules)
                shim = colormnet.sys.modules["progressbar"]
                self.assertEqual(list(shim.progressbar([1, 2, 3])), [1, 2, 3])

    def test_resolve_reference_image_path_requires_a_reference(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_reference_image_path([])

    def test_resolve_reference_image_path_uses_first_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.png"
            reference = Path(temp_dir) / "reference.png"
            reference.write_bytes(b"reference")

            resolved = _resolve_reference_image_path([str(missing), str(reference)])

        self.assertEqual(resolved, reference)

    def test_build_args_list_matches_upstream_test_app_contract(self) -> None:
        checkpoint_path = Path("C:/models/colormnet.pth")
        input_root = Path("C:/work/input_video")
        output_root = Path("C:/work/output")
        ref_root = Path("C:/work/ref")

        args = _build_args_list(
            checkpoint_path=checkpoint_path,
            input_root=input_root,
            output_root=output_root,
            ref_root=ref_root,
        )

        self.assertEqual(args[:8], [
            "--model", str(checkpoint_path),
            "--d16_batch_path", str(input_root),
            "--ref_path", str(ref_root),
            "--output", str(output_root),
        ])
        self.assertIn("--FirstFrameIsNotExemplar", args)
        self.assertIn("--save_all", args)
        self.assertIn("--size", args)


if __name__ == "__main__":
    unittest.main()