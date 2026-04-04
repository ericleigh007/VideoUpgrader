import sys
import unittest

from upscaler_worker.pipeline import _run


class PipelineProcessRunnerTests(unittest.TestCase):
    def test_run_drains_chatty_process_output(self) -> None:
        log: list[str] = []
        command = [
            sys.executable,
            "-c",
            (
                "import sys, time; "
                "[sys.stdout.write('x'*2048 + '\\n') or sys.stdout.flush() or time.sleep(0.001) for _ in range(80)]"
            ),
        ]

        _run(command, log)

        self.assertTrue(log)
        self.assertIn("$ ", log[0])
        self.assertGreaterEqual(len(log), 2)


if __name__ == "__main__":
    unittest.main()