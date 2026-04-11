from __future__ import annotations

import subprocess
import time
from pathlib import Path


class JobCancelledError(RuntimeError):
    pass


def cancellation_requested(cancel_path: str | None) -> bool:
    return bool(cancel_path) and Path(cancel_path).exists()


def pause_requested(pause_path: str | None) -> bool:
    return bool(pause_path) and Path(pause_path).exists()


def ensure_not_cancelled(cancel_path: str | None) -> None:
    if cancellation_requested(cancel_path):
        raise JobCancelledError("Job cancelled by user")


def _load_psutil():
    try:
        import psutil  # type: ignore
    except ImportError as error:  # pragma: no cover - exercised only when pause is requested without the dependency installed
        raise RuntimeError("Pause/resume support requires psutil to be installed in the worker environment") from error
    return psutil


def _iter_process_tree(process: subprocess.Popen[object]):
    psutil = _load_psutil()

    try:
        root = psutil.Process(process.pid)
    except psutil.Error:
        return []

    descendants = root.children(recursive=True)
    return [*descendants, root]


def suspend_process_tree(process: subprocess.Popen[object]) -> None:
    if process.poll() is not None:
        return

    for entry in _iter_process_tree(process):
        try:
            entry.suspend()
        except Exception:  # noqa: BLE001
            continue


def resume_process_tree(process: subprocess.Popen[object]) -> None:
    if process.poll() is not None:
        return

    for entry in reversed(_iter_process_tree(process)):
        try:
            entry.resume()
        except Exception:  # noqa: BLE001
            continue


def wait_if_paused(
    pause_path: str | None,
    *,
    cancel_path: str | None = None,
    process: subprocess.Popen[object] | None = None,
    on_pause=None,
    on_resume=None,
    poll_interval_seconds: float = 0.1,
) -> bool:
    if not pause_requested(pause_path):
        return False

    if process is not None:
        suspend_process_tree(process)
    if on_pause is not None:
        on_pause()

    try:
        while pause_requested(pause_path):
            if cancellation_requested(cancel_path):
                if process is not None and process.poll() is None:
                    resume_process_tree(process)
                    terminate_process(process)
                raise JobCancelledError("Job cancelled by user")
            time.sleep(poll_interval_seconds)
    finally:
        if process is not None and process.poll() is None:
            resume_process_tree(process)

    if on_resume is not None:
        on_resume()
    return True


def terminate_process(process: subprocess.Popen[object], timeout_seconds: float = 3.0) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    deadline = time.monotonic() + timeout_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)

    if process.poll() is None:
        process.kill()


def terminate_process_tree(process: subprocess.Popen[object], timeout_seconds: float = 3.0) -> None:
    if process.poll() is not None:
        return

    try:
        entries = _iter_process_tree(process)
    except RuntimeError:
        entries = []

    if entries:
        for entry in entries:
            try:
                entry.terminate()
            except Exception:  # noqa: BLE001
                continue

        deadline = time.monotonic() + timeout_seconds
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)

        if process.poll() is None:
            for entry in entries:
                try:
                    entry.kill()
                except Exception:  # noqa: BLE001
                    continue
            time.sleep(0.1)
        return

    terminate_process(process, timeout_seconds=timeout_seconds)