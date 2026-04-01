import json
import tempfile
import unittest
from pathlib import Path

from upscaler_worker.synthetic.generate_benchmarks import generate_benchmark_fixture


class GenerateBenchmarksTests(unittest.TestCase):
    def test_generates_fixture_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = generate_benchmark_fixture(
                output_dir=Path(temp_dir),
                name="fixture_a",
                frames=3,
                width=640,
                height=360,
                downscale_width=320,
                downscale_height=180,
            )

            self.assertEqual(manifest["frames"], 3)
            fixture_dir = Path(temp_dir) / "fixture_a"
            manifest_path = fixture_dir / "manifest.json"
            self.assertTrue(manifest_path.exists())
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(loaded["entries"]), 3)
            self.assertTrue((fixture_dir / "master" / "frame_0000.png").exists())
            self.assertTrue((fixture_dir / "degraded" / "frame_0000.png").exists())


if __name__ == "__main__":
    unittest.main()
