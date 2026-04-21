from __future__ import annotations

import functools
import shutil
import os
import re
import ssl
import subprocess
import time
import urllib.error
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
RVRT_REPO_ZIP_URL = "https://codeload.github.com/JingyunLiang/RVRT/zip/refs/heads/main"
RVRT_RELEASE_BASE_URL = "https://github.com/JingyunLiang/RVRT/releases/download/v0.0"
RVRT_DEFAULT_TASK = "002_RVRT_videosr_bi_Vimeo_14frames"
RIFE_ZIP_URL = (
    "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/"
    "rife-ncnn-vulkan-20221029-windows.zip"
)
DEOLDIFY_REPO_ZIP_URL = "https://codeload.github.com/jantic/DeOldify/zip/refs/heads/master"
DEEPREMASTER_REPO_ZIP_URL = "https://codeload.github.com/satoshiiizuka/siggraphasia2019_remastering/zip/refs/heads/master"
COLORMNET_REPO_ZIP_URL = "https://codeload.github.com/yyang181/colormnet/zip/refs/heads/main"
DEEPREMASTER_MODEL_URL = "http://iizuka.cs.tsukuba.ac.jp/data/remasternet.pth.tar"
GPU_LINE_PATTERN = re.compile(r"^\[(\d+)\s+([^\]]+)\]")
TRANSIENT_DOWNLOAD_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
DOWNLOAD_RETRY_ATTEMPTS = 4
DOWNLOAD_RETRY_BASE_DELAY_SECONDS = 1.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def runtime_root() -> Path:
    return repo_root() / "artifacts" / "runtime"


def _is_transient_download_error(error: BaseException) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return int(getattr(error, "code", 0)) in TRANSIENT_DOWNLOAD_STATUS_CODES
    return isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError, OSError))


def download_file_with_retries(url: str, destination: Path, *, attempts: int = DOWNLOAD_RETRY_ATTEMPTS) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    last_error: BaseException | None = None

    for attempt_index in range(max(1, attempts)):
        partial_path.unlink(missing_ok=True)
        try:
            try:
                urllib.request.urlretrieve(url, partial_path)
            except urllib.error.URLError as error:
                reason = getattr(error, "reason", None)
                if not isinstance(reason, ssl.SSLCertVerificationError):
                    raise
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                with urllib.request.urlopen(url, context=ssl_context) as response, partial_path.open("wb") as destination_handle:
                    shutil.copyfileobj(response, destination_handle)
            partial_path.replace(destination)
            return destination
        except BaseException as error:
            last_error = error
            partial_path.unlink(missing_ok=True)
            is_last_attempt = attempt_index >= max(1, attempts) - 1
            if is_last_attempt or not _is_transient_download_error(error):
                raise
            time.sleep(DOWNLOAD_RETRY_BASE_DELAY_SECONDS * (attempt_index + 1))

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Could not download {url}")


def _download_file(url: str, destination: Path) -> Path:
    return download_file_with_retries(url, destination)


def _extract_archive_with_optional_root_strip(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        names = [member.filename.replace("\\", "/") for member in archive.infolist() if member.filename]
        top_levels = {name.split("/", 1)[0] for name in names if name}
        strip_root = len(top_levels) == 1
        root_prefix = f"{next(iter(top_levels))}/" if strip_root else ""

        for member in archive.infolist():
            raw_name = member.filename.replace("\\", "/")
            if not raw_name or raw_name.endswith("/"):
                continue
            relative_name = raw_name[len(root_prefix):] if strip_root and raw_name.startswith(root_prefix) else raw_name
            if not relative_name:
                continue
            target_path = destination / relative_name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source_handle, target_path.open("wb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)


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
    _download_file(zip_url, zip_path)
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


def ensure_rvrt_repo() -> dict[str, str]:
    install_dir = repo_root() / "tmp" / "RVRT"
    entrypoint = install_dir / "main_test_rvrt.py"
    if entrypoint.exists():
        return {
            "rvrtRoot": str(install_dir),
            "entryPoint": str(entrypoint),
        }

    archive_path = runtime_root() / "rvrt-main.zip"
    _download_file(RVRT_REPO_ZIP_URL, archive_path)
    if install_dir.exists():
        shutil.rmtree(install_dir)
    install_dir.mkdir(parents=True, exist_ok=True)
    _extract_archive_with_optional_root_strip(archive_path, install_dir)
    archive_path.unlink(missing_ok=True)
    if not entrypoint.exists():
        raise RuntimeError(f"RVRT bootstrap archive did not contain {entrypoint.name}")
    return {
        "rvrtRoot": str(install_dir),
        "entryPoint": str(entrypoint),
    }


def ensure_rvrt_model_weights(task_name: str = RVRT_DEFAULT_TASK) -> dict[str, str]:
    model_path = repo_root() / "model_zoo" / "rvrt" / f"{task_name}.pth"
    if not model_path.exists():
        model_url = f"{RVRT_RELEASE_BASE_URL}/{task_name}.pth"
        _download_file(model_url, model_path)
    return {
        "task": task_name,
        "modelPath": str(model_path),
    }


def _ensure_deoldify_dummy_assets(dummy_root: Path) -> Path:
    dummy_root.mkdir(parents=True, exist_ok=True)
    for image_index in range(10):
        placeholder = dummy_root / f"placeholder_{image_index:02d}.png"
        if placeholder.exists():
            continue
        placeholder.write_bytes(
            bytes.fromhex(
                "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE"
                "0000000C49444154789C636060000000040001F61738550000000049454E44AE426082"
            )
        )
    return dummy_root


def ensure_deoldify_runtime() -> dict[str, str]:
    install_dir = runtime_root() / "deoldify"
    source_root = install_dir / "src"
    entrypoint = source_root / "deoldify" / "__init__.py"
    if not entrypoint.exists():
        archive_path = runtime_root() / "deoldify-master.zip"
        _download_file(DEOLDIFY_REPO_ZIP_URL, archive_path)
        if source_root.exists():
            shutil.rmtree(source_root)
        source_root.mkdir(parents=True, exist_ok=True)
        _extract_archive_with_optional_root_strip(archive_path, source_root)
        archive_path.unlink(missing_ok=True)
        if not entrypoint.exists():
            raise RuntimeError("DeOldify bootstrap archive did not contain the expected source tree")

    dummy_root = _ensure_deoldify_dummy_assets(repo_root() / "dummy")
    model_root = install_dir / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    return {
        "deoldifySourceRoot": str(source_root),
        "deoldifyModelRoot": str(model_root),
        "deoldifyDummyRoot": str(dummy_root),
    }


def ensure_deepremaster_runtime() -> dict[str, str]:
    install_dir = runtime_root() / "deepremaster"
    source_root = install_dir / "src"
    model_root = install_dir / "model"
    entrypoint = source_root / "model" / "remasternet.py"
    checkpoint_path = model_root / "remasternet.pth.tar"
    if not entrypoint.exists():
        archive_path = runtime_root() / "deepremaster-master.zip"
        _download_file(DEEPREMASTER_REPO_ZIP_URL, archive_path)
        if source_root.exists():
            shutil.rmtree(source_root)
        source_root.mkdir(parents=True, exist_ok=True)
        _extract_archive_with_optional_root_strip(archive_path, source_root)
        archive_path.unlink(missing_ok=True)
        if not entrypoint.exists():
            raise RuntimeError("DeepRemaster bootstrap archive did not contain model/remasternet.py")

    model_root.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        _download_file(DEEPREMASTER_MODEL_URL, checkpoint_path)

    return {
        "deepremasterSourceRoot": str(source_root),
        "deepremasterModelRoot": str(model_root),
        "deepremasterCheckpointPath": str(checkpoint_path),
    }


def ensure_colormnet_runtime() -> dict[str, str]:
    install_dir = runtime_root() / "colormnet"
    source_root = install_dir / "src"
    entrypoint = source_root / "test_app.py"
    if not entrypoint.exists():
        archive_path = runtime_root() / "colormnet-main.zip"
        _download_file(COLORMNET_REPO_ZIP_URL, archive_path)
        if source_root.exists():
            shutil.rmtree(source_root)
        source_root.mkdir(parents=True, exist_ok=True)
        _extract_archive_with_optional_root_strip(archive_path, source_root)
        archive_path.unlink(missing_ok=True)
        if not entrypoint.exists():
            raise RuntimeError("ColorMNet bootstrap archive did not contain test_app.py")

    return {
        "colormnetSourceRoot": str(source_root),
        "colormnetEntryPoint": str(entrypoint),
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
