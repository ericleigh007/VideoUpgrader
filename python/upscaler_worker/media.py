from __future__ import annotations

import re
import subprocess

from upscaler_worker.runtime import ensure_runtime_assets


def probe_video(source_path: str) -> dict[str, object]:
    runtime = ensure_runtime_assets()
    process = subprocess.run(
        [runtime["ffmpegPath"], "-hide_banner", "-i", source_path],
        capture_output=True,
        text=True,
        check=False,
    )
    stderr = process.stderr
    duration_match = re.search(r"Duration:\s+(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    video_match = re.search(r"Video:.*?(\d{2,5})x(\d{2,5}).*?(\d+(?:\.\d+)?)\s+fps", stderr)
    audio_present = "Audio:" in stderr

    if duration_match is None or video_match is None:
        raise RuntimeError(f"Could not parse video metadata for {source_path}\n{stderr}")

    hours = int(duration_match.group(1))
    minutes = int(duration_match.group(2))
    seconds = float(duration_match.group(3))
    duration_seconds = hours * 3600 + minutes * 60 + seconds

    container = source_path.rsplit(".", maxsplit=1)[-1].lower() if "." in source_path else "unknown"
    return {
        "path": source_path,
        "width": int(video_match.group(1)),
        "height": int(video_match.group(2)),
        "durationSeconds": duration_seconds,
        "frameRate": float(video_match.group(3)),
        "hasAudio": audio_present,
        "container": container,
    }
