import tempfile
import threading
import time
import unittest
from pathlib import Path

from upscaler_worker.cancellation import JobCancelledError, wait_if_paused


class CancellationControlTests(unittest.TestCase):
    def test_wait_if_paused_blocks_until_pause_signal_is_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pause_path = Path(temp_dir) / "pause.signal"
            pause_path.write_text("paused", encoding="utf-8")
            callbacks: list[str] = []

            def clear_pause() -> None:
                time.sleep(0.15)
                pause_path.unlink(missing_ok=True)

            thread = threading.Thread(target=clear_pause, daemon=True)
            thread.start()

            started_at = time.monotonic()
            did_pause = wait_if_paused(
                str(pause_path),
                on_pause=lambda: callbacks.append("pause"),
                on_resume=lambda: callbacks.append("resume"),
            )
            elapsed = time.monotonic() - started_at
            thread.join(timeout=1)

            self.assertTrue(did_pause)
            self.assertGreaterEqual(elapsed, 0.1)
            self.assertEqual(callbacks, ["pause", "resume"])

    def test_wait_if_paused_raises_when_cancelled_during_pause(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pause_path = Path(temp_dir) / "pause.signal"
            cancel_path = Path(temp_dir) / "cancel.signal"
            pause_path.write_text("paused", encoding="utf-8")

            def cancel_job() -> None:
                time.sleep(0.1)
                cancel_path.write_text("cancelled", encoding="utf-8")

            thread = threading.Thread(target=cancel_job, daemon=True)
            thread.start()

            with self.assertRaises(JobCancelledError):
                wait_if_paused(str(pause_path), cancel_path=str(cancel_path))

            pause_path.unlink(missing_ok=True)
            thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()