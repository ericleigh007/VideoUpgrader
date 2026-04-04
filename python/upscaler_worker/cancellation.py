from __future__ import annotations

import subprocess
import time
from pathlib import Path


class JobCancelledError(RuntimeError):
    pass


def cancellation_requested(cancel_path: str | None) -> bool:
    return bool(cancel_path) and Path(cancel_path).exists()


def ensure_not_cancelled(cancel_path: str | None) -> None:
    if cancellation_requested(cancel_path):
        raise JobCancelledError("Job cancelled by user")


def terminate_process(process: subprocess.Popen[object], timeout_seconds: float = 3.0) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    deadline = time.monotonic() + timeout_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)

    if process.poll() is None:
        process.kill()