from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from upscaler_worker.runtime import ensure_runtime_assets


DEFAULT_FLASH_INTERVAL_SECONDS = 1.0
DEFAULT_FLASH_DURATION_SECONDS = 0.08
DEFAULT_POP_FREQUENCY_HZ = 1000
DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_VIDEO_FPS = 30
DEFAULT_VIDEO_WIDTH = 1280
DEFAULT_VIDEO_HEIGHT = 720
DEFAULT_VIDEO_THRESHOLD = 180.0
DEFAULT_AUDIO_THRESHOLD = 5000
DEFAULT_AUDIO_WINDOW_MS = 5
DEFAULT_SYNC_TOLERANCE_MS = 45.0


def _event_times(duration_seconds: float, interval_seconds: float) -> list[float]:
    if duration_seconds <= 0 or interval_seconds <= 0:
        return []

    event_count = int(duration_seconds // interval_seconds) + 1
    times = [round(index * interval_seconds, 6) for index in range(event_count)]
    return [time for time in times if time < duration_seconds]


def generate_av_sync_fixture(
    *,
    output_path: Path,
    duration_seconds: float,
    width: int = DEFAULT_VIDEO_WIDTH,
    height: int = DEFAULT_VIDEO_HEIGHT,
    fps: int = DEFAULT_VIDEO_FPS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    flash_interval_seconds: float = DEFAULT_FLASH_INTERVAL_SECONDS,
    flash_duration_seconds: float = DEFAULT_FLASH_DURATION_SECONDS,
    pop_frequency_hz: int = DEFAULT_POP_FREQUENCY_HZ,
) -> dict[str, object]:
    runtime = ensure_runtime_assets()
    ffmpeg_path = str(runtime["ffmpegPath"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flash_enable = f"lt(mod(t\\,{flash_interval_seconds:.6f})\\,{flash_duration_seconds:.6f})"
    video_filter = (
        f"drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='{flash_enable}',"
        f"drawbox=x=0:y=0:w=iw:h=8:color=red@0.8:t=fill:enable='{flash_enable}',"
        f"drawbox=x=0:y=ih-8:w=iw:h=8:color=red@0.8:t=fill:enable='{flash_enable}'"
    )
    audio_expression = (
        f"if(lt(mod(t\\,{flash_interval_seconds:.6f})\\,{flash_duration_seconds:.6f})\\,"
        f"0.85*sin(2*PI*{pop_frequency_hz}*t)\\,0)"
    )

    command = [
        ffmpeg_path,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={width}x{height}:r={fps}:d={duration_seconds:.6f}",
        "-f",
        "lavfi",
        "-i",
        f"aevalsrc={audio_expression}:s={sample_rate}:d={duration_seconds:.6f}",
        "-filter:v",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Could not generate AV sync fixture\n{completed.stderr.strip()}")

    manifest = {
        "outputPath": str(output_path.resolve()),
        "durationSeconds": duration_seconds,
        "width": width,
        "height": height,
        "fps": fps,
        "sampleRate": sample_rate,
        "flashIntervalSeconds": flash_interval_seconds,
        "flashDurationSeconds": flash_duration_seconds,
        "popFrequencyHz": pop_frequency_hz,
        "eventTimes": _event_times(duration_seconds, flash_interval_seconds),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".sync.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _read_manifest(manifest_path: Path) -> dict[str, object]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _detect_video_flash_times(media_path: Path, fps: int, threshold: float) -> list[float]:
    runtime = ensure_runtime_assets()
    ffmpeg_path = str(runtime["ffmpegPath"])
    frame_width = 32
    frame_height = 18
    frame_bytes = frame_width * frame_height
    command = [
        ffmpeg_path,
        "-v",
        "error",
        "-i",
        str(media_path),
        "-map",
        "0:v:0",
        "-vf",
        f"fps={fps},scale={frame_width}:{frame_height},format=gray",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    assert process.stderr is not None
    flash_times: list[float] = []
    frame_index = 0
    active = False
    try:
        while True:
            frame = process.stdout.read(frame_bytes)
            if len(frame) < frame_bytes:
                break
            mean_luma = float(np.frombuffer(frame, dtype=np.uint8).mean())
            is_flash = mean_luma >= threshold
            if is_flash and not active:
                flash_times.append(frame_index / fps)
            active = is_flash
            frame_index += 1
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"Could not detect video flash timings\n{stderr.strip()}")
    finally:
        process.stdout.close()
        process.stderr.close()
    return flash_times


def _detect_audio_pop_times(media_path: Path, sample_rate: int, threshold: int, window_ms: int) -> list[float]:
    runtime = ensure_runtime_assets()
    ffmpeg_path = str(runtime["ffmpegPath"])
    window_samples = max(1, int(sample_rate * (window_ms / 1000.0)))
    command = [
        ffmpeg_path,
        "-v",
        "error",
        "-i",
        str(media_path),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    assert process.stderr is not None
    audio_times: list[float] = []
    processed_samples = 0
    remainder = np.array([], dtype=np.int16)
    active = False
    try:
        while True:
            chunk = process.stdout.read(window_samples * 200 * 2)
            if not chunk:
                break
            samples = np.frombuffer(chunk, dtype=np.int16)
            if remainder.size > 0:
                samples = np.concatenate([remainder, samples])
            usable = (samples.size // window_samples) * window_samples
            if usable <= 0:
                remainder = samples
                continue
            windows = samples[:usable].reshape(-1, window_samples)
            levels = np.max(np.abs(windows.astype(np.int32)), axis=1)
            for index, level in enumerate(levels):
                is_pop = int(level) >= threshold
                if is_pop and not active:
                    audio_times.append((processed_samples + index * window_samples) / sample_rate)
                active = is_pop
            processed_samples += usable
            remainder = samples[usable:]
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"Could not detect audio pop timings\n{stderr.strip()}")
    finally:
        process.stdout.close()
        process.stderr.close()
    return audio_times


def validate_av_sync(
    *,
    media_path: Path,
    manifest_path: Path,
    video_threshold: float = DEFAULT_VIDEO_THRESHOLD,
    audio_threshold: int = DEFAULT_AUDIO_THRESHOLD,
    audio_window_ms: int = DEFAULT_AUDIO_WINDOW_MS,
    tolerance_ms: float = DEFAULT_SYNC_TOLERANCE_MS,
) -> dict[str, object]:
    manifest = _read_manifest(manifest_path)
    expected_times = [float(value) for value in manifest["eventTimes"]]
    fps = int(manifest["fps"])
    sample_rate = int(manifest["sampleRate"])

    video_times = _detect_video_flash_times(media_path, fps=fps, threshold=video_threshold)
    audio_times = _detect_audio_pop_times(media_path, sample_rate=sample_rate, threshold=audio_threshold, window_ms=audio_window_ms)

    compared_events = min(len(expected_times), len(video_times), len(audio_times))
    if compared_events == 0:
        raise RuntimeError("No AV sync events were detected in the media")

    event_pairs = []
    for index in range(compared_events):
        expected_time = expected_times[index]
        video_time = video_times[index]
        audio_time = audio_times[index]
        event_pairs.append(
            {
                "index": index,
                "expectedTime": expected_time,
                "videoTime": video_time,
                "audioTime": audio_time,
                "videoOffsetMs": round((video_time - expected_time) * 1000.0, 3),
                "audioOffsetMs": round((audio_time - expected_time) * 1000.0, 3),
                "avOffsetMs": round((audio_time - video_time) * 1000.0, 3),
            }
        )

    max_av_offset_ms = max(abs(pair["avOffsetMs"]) for pair in event_pairs)
    max_video_offset_ms = max(abs(pair["videoOffsetMs"]) for pair in event_pairs)
    max_audio_offset_ms = max(abs(pair["audioOffsetMs"]) for pair in event_pairs)
    passed = max_av_offset_ms <= tolerance_ms
    return {
        "mediaPath": str(media_path.resolve()),
        "manifestPath": str(manifest_path.resolve()),
        "expectedEventCount": len(expected_times),
        "detectedVideoEventCount": len(video_times),
        "detectedAudioEventCount": len(audio_times),
        "comparedEventCount": compared_events,
        "maxVideoOffsetMs": round(max_video_offset_ms, 3),
        "maxAudioOffsetMs": round(max_audio_offset_ms, 3),
        "maxAvOffsetMs": round(max_av_offset_ms, 3),
        "toleranceMs": tolerance_ms,
        "passed": passed,
        "eventPairs": event_pairs,
    }