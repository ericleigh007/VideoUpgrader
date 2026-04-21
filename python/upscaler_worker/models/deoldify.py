from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from upscaler_worker.cancellation import ensure_not_cancelled, wait_if_paused
from upscaler_worker.model_catalog import ensure_runnable_model, model_backend_id, model_label
from upscaler_worker.models.pytorch_sr import ensure_model_checkpoint
from upscaler_worker.runtime import ensure_deoldify_runtime


SUPPORTED_DEOLDIFY_MODELS = {
    "deoldify-stable": {
        "weightsName": "ColorizeStable_gen",
        "renderFactor": 21,
        "deviceId": "GPU0",
    },
    "deoldify-video": {
        "weightsName": "ColorizeVideo_gen",
        "renderFactor": 21,
        "deviceId": "GPU0",
    },
}
SUPPORTED_FRAME_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class LoadedDeoldifyColorizer:
    model_id: str
    model_label: str
    repo_id: str
    checkpoint_path: Path
    source_root: Path
    model_root: Path
    input_size: int
    render_factor: int
    device: torch.device
    precision_mode: str
    colorizer_filter: object


def _resolve_device(gpu_id: int | None) -> tuple[torch.device, str, list[str]]:
    notes: list[str] = []
    if not torch.cuda.is_available():
        notes.append("PyTorch CUDA runtime unavailable. Falling back to CPU inference.")
        return torch.device("cpu"), "CPU", notes

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        notes.append("No CUDA devices reported by PyTorch. Falling back to CPU inference.")
        return torch.device("cpu"), "CPU", notes

    resolved_gpu_id = 0 if gpu_id is None else gpu_id
    if 0 <= resolved_gpu_id < device_count:
        return torch.device(f"cuda:{resolved_gpu_id}"), f"GPU{resolved_gpu_id}", notes

    if device_count == 1 and resolved_gpu_id >= 0:
        notes.append(f"Mapped app GPU id {resolved_gpu_id} to PyTorch cuda:0.")
        return torch.device("cuda:0"), "GPU0", notes

    notes.append(f"Requested GPU {resolved_gpu_id} is not available to PyTorch. Using cuda:0 instead.")
    return torch.device("cuda:0"), "GPU0", notes


def load_runtime_colorizer(
    model_id: str,
    gpu_id: int | None,
    precision: str | None,
    log: list[str],
    reference_image_paths: list[str] | None = None,
) -> LoadedDeoldifyColorizer:
    del precision
    del reference_image_paths
    ensure_runnable_model(model_id)
    if model_backend_id(model_id) != "pytorch-image-colorization":
        raise ValueError(f"Model '{model_id}' is not a DeOldify-compatible colorizer")

    runtime_config = SUPPORTED_DEOLDIFY_MODELS.get(model_id)
    if runtime_config is None:
        supported = ", ".join(sorted(SUPPORTED_DEOLDIFY_MODELS))
        raise NotImplementedError(f"Colorizer '{model_id}' is not implemented. Supported DeOldify colorizers: {supported}")

    runtime_assets = ensure_deoldify_runtime()
    source_root = Path(runtime_assets["deoldifySourceRoot"])
    model_root = Path(runtime_assets["deoldifyModelRoot"])
    checkpoint_path = ensure_model_checkpoint(model_id)
    model_target = model_root / checkpoint_path.name
    if not model_target.exists():
        model_target.write_bytes(checkpoint_path.read_bytes())

    device, deoldify_device_id, notes = _resolve_device(gpu_id)
    for note in notes:
        log.append(note)

    sys.path.insert(0, str(source_root))
    from deoldify import device as device_settings  # type: ignore[import-not-found]
    from deoldify.device_id import DeviceId  # type: ignore[import-not-found]
    device_settings.set(getattr(DeviceId, deoldify_device_id))
    from deoldify.filters import ColorizerFilter  # type: ignore[import-not-found]
    from deoldify.generators import gen_inference_wide  # type: ignore[import-not-found]

    original_torch_load = torch.load

    def _legacy_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        kwargs["map_location"] = "cpu"
        return original_torch_load(*args, **kwargs)

    with patch("torch.load", _legacy_torch_load):
        learner = gen_inference_wide(
            root_folder=model_root.parent,
            weights_name=str(runtime_config["weightsName"]),
        )
    learner.model = learner.model.to(device)
    learner.model.eval()
    colorizer_filter = ColorizerFilter(learn=learner)
    log.append(f"Loaded DeOldify checkpoint from {model_target}")

    return LoadedDeoldifyColorizer(
        model_id=model_id,
        model_label=model_label(model_id),
        repo_id="jantic/DeOldify",
        checkpoint_path=model_target,
        source_root=source_root,
        model_root=model_root,
        input_size=int(runtime_config["renderFactor"]),
        render_factor=int(runtime_config["renderFactor"]),
        device=device,
        precision_mode="fp32",
        colorizer_filter=colorizer_filter,
    )


def colorize_directory(
    *,
    loaded_model: LoadedDeoldifyColorizer,
    input_dir: Path,
    output_dir: Path,
    cancel_path: str | None,
    pause_path: str | None,
    progress_callback=None,
) -> int:
    frame_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_FRAME_SUFFIXES
    )
    if not frame_paths:
        raise RuntimeError("No extracted frames were found for colorization.")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_frames = len(frame_paths)
    for index, frame_path in enumerate(frame_paths, start=1):
        ensure_not_cancelled(cancel_path)
        wait_if_paused(pause_path, cancel_path=cancel_path)
        image = Image.open(frame_path).convert("RGB")
        colorized_image = loaded_model.colorizer_filter.filter(
            image,
            image,
            render_factor=loaded_model.render_factor,
            post_process=True,
        )
        output_path = output_dir / frame_path.name
        colorized_array = np.asarray(colorized_image)
        Image.fromarray(colorized_array).save(output_path)
        image.close()
        colorized_image.close()
        if progress_callback is not None:
            progress_callback(index, total_frames)

    if loaded_model.device.type == "cuda":
        torch.cuda.synchronize(loaded_model.device)
    return total_frames