from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import imageio_ffmpeg

from upscaler_worker.model_catalog import model_catalog
from upscaler_worker.models.pytorch_video_sr import resolve_external_video_sr_command_template


REALESRGAN_ZIP_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
    "realesrgan-ncnn-vulkan-20220424-windows.zip"
)
RIFE_ZIP_URL = (
    "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/"
    "rife-ncnn-vulkan-20221029-windows.zip"
)
GPU_LINE_PATTERN = re.compile(r"^\[(\d+)\s+([^\]]+)\]")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def runtime_root() -> Path:
    return repo_root() / "artifacts" / "runtime"


def _flatten_runtime_install(install_dir: Path, executable_name: str) -> Path:
    nested_exe = next(install_dir.rglob(executable_name), None)
    if nested_exe is None:
        raise RuntimeError(f"Downloaded runtime package did not contain {executable_name}")
    if nested_exe != install_dir / executable_name:
        for child in nested_exe.parent.iterdir():
            destination = install_dir / child.name
            if child == destination:
                continue
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(child), str(destination))
    return install_dir / executable_name


def _ensure_portable_runtime(*, install_dir: Path, zip_name: str, zip_url: str, executable_name: str) -> Path:
    exe_path = install_dir / executable_name
    if exe_path.exists():
        return exe_path

    install_dir.mkdir(parents=True, exist_ok=True)
    zip_path = runtime_root() / zip_name
    urllib.request.urlretrieve(zip_url, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(install_dir)
    zip_path.unlink(missing_ok=True)
    return _flatten_runtime_install(install_dir, executable_name)


def ensure_realesrgan_runtime() -> dict[str, str]:
    install_dir = runtime_root() / "realesrgan-ncnn-vulkan"
    exe_path = _ensure_portable_runtime(
        install_dir=install_dir,
        zip_name="realesrgan-ncnn-vulkan.zip",
        zip_url=REALESRGAN_ZIP_URL,
        executable_name="realesrgan-ncnn-vulkan.exe",
    )

    return {
        "realesrganPath": str(exe_path),
        "modelDir": str(install_dir / "models"),
    }


def ensure_rife_runtime() -> dict[str, str]:
    install_dir = runtime_root() / "rife-ncnn-vulkan"
    exe_path = _ensure_portable_runtime(
        install_dir=install_dir,
        zip_name="rife-ncnn-vulkan.zip",
        zip_url=RIFE_ZIP_URL,
        executable_name="rife-ncnn-vulkan.exe",
    )

    return {
        "rifePath": str(exe_path),
        "rifeModelRoot": str(install_dir),
    }


def _gpu_kind_from_name(name: str) -> str:
    normalized = name.lower()
    if "nvidia" in normalized or "geforce" in normalized or "rtx" in normalized or "radeon" in normalized or "quadro" in normalized:
        return "discrete"
    if "intel" in normalized or "uhd" in normalized or "iris" in normalized:
        return "integrated"
    return "unknown"


@functools.lru_cache(maxsize=1)
def detect_available_gpus() -> tuple[list[dict[str, object]], int | None]:
    realesrgan = ensure_realesrgan_runtime()
    probe_root = runtime_root() / "gpu_probe"
    input_dir = probe_root / "in"
    output_dir = probe_root / "out"
    devices: list[dict[str, object]] = []

    try:
        if probe_root.exists():
            shutil.rmtree(probe_root)
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        completed = subprocess.run(
            [
                realesrgan["realesrganPath"],
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-m",
                realesrgan["modelDir"],
                "-n",
                "realesrgan-x4plus",
                "-f",
                "png",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        combined_output = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
        seen_ids: set[int] = set()
        for raw_line in combined_output.splitlines():
            match = GPU_LINE_PATTERN.match(raw_line.strip())
            if not match:
                continue
            device_id = int(match.group(1))
            if device_id in seen_ids:
                continue
            seen_ids.add(device_id)
            device_name = match.group(2).strip()
            devices.append({
                "id": device_id,
                "name": device_name,
                "kind": _gpu_kind_from_name(device_name),
            })
    except (OSError, subprocess.TimeoutExpired):
        devices = []
    finally:
        shutil.rmtree(probe_root, ignore_errors=True)

    default_gpu_id = next((device["id"] for device in devices if device["kind"] == "discrete"), None)
    if default_gpu_id is None and devices:
        default_gpu_id = devices[0]["id"]
    return devices, default_gpu_id


def detect_external_research_runtimes() -> dict[str, object]:
    statuses: dict[str, object] = {}
    for model in model_catalog():
        model_id = str(model.get("id", "")).strip()
        research_runtime = model.get("researchRuntime")
        if not model_id or not isinstance(research_runtime, dict):
            continue
        if str(research_runtime.get("kind", "")).strip() != "external-command":
            continue

        command_env_var = str(research_runtime.get("commandEnvVar", "")).strip()
        command_template, command_source = resolve_external_video_sr_command_template(model_id, command_env_var)
        statuses[model_id] = {
            "kind": "external-command",
            "commandEnvVar": command_env_var,
            "configured": bool(command_template),
            "source": command_source,
        }
    return statuses


def ensure_runtime_assets() -> dict[str, object]:
    runtime_root().mkdir(parents=True, exist_ok=True)
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    realesrgan = ensure_realesrgan_runtime()
    available_gpus, default_gpu_id = detect_available_gpus()
    external_research_runtimes = detect_external_research_runtimes()
    return {
        "ffmpegPath": str(ffmpeg_path),
        "realesrganPath": realesrgan["realesrganPath"],
        "modelDir": realesrgan["modelDir"],
        "availableGpus": available_gpus,
        "defaultGpuId": default_gpu_id,
        "externalResearchRuntimes": external_research_runtimes,
    }
