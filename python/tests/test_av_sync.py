import tempfile
import unittest
from pathlib import Path

from upscaler_worker.synthetic.av_sync import generate_av_sync_fixture, validate_av_sync


class AvSyncTests(unittest.TestCase):
    def test_generates_and_validates_short_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "av_sync_fixture.mp4"
            manifest = generate_av_sync_fixture(
                output_path=output_path,
                duration_seconds=3.2,
                width=320,
                height=180,
                fps=12,
                flash_interval_seconds=0.5,
                flash_duration_seconds=0.08,
            )

            manifest_path = output_path.with_suffix(output_path.suffix + ".sync.json")
            self.assertTrue(output_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertGreater(len(manifest["eventTimes"]), 0)

            result = validate_av_sync(media_path=output_path, manifest_path=manifest_path, tolerance_ms=80.0)
            self.assertTrue(result["passed"])
            self.assertGreaterEqual(result["comparedEventCount"], 3)


if __name__ == "__main__":
    unittest.main()