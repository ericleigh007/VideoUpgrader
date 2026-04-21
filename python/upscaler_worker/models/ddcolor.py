from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from upscaler_worker.cancellation import ensure_not_cancelled, wait_if_paused
from upscaler_worker.model_catalog import ensure_runnable_model, model_backend_id, model_label


COLORIZER_INPUT_SIZE = 512
SUPPORTED_DDCOLOR_MODELS = {
    "ddcolor-modelscope": "piddnad/ddcolor_modelscope",
    "ddcolor-paper": "piddnad/ddcolor_paper",
}
SUPPORTED_FRAME_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class LoadedDdcolorModel:
    model_id: str
    model_label: str
    repo_id: str
    model: torch.nn.Module
    device: torch.device
    precision_mode: str
    autocast_dtype: torch.dtype | None
    input_size: int


def _resolve_device(gpu_id: int | None) -> tuple[torch.device, list[str]]:
    notes: list[str] = []
    if not torch.cuda.is_available():
        notes.append("PyTorch CUDA runtime unavailable. Falling back to CPU inference.")
        return torch.device("cpu"), notes

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        notes.append("No CUDA devices reported by PyTorch. Falling back to CPU inference.")
        return torch.device("cpu"), notes

    if gpu_id is None:
        return torch.device("cuda:0"), notes

    if 0 <= gpu_id < device_count:
        return torch.device(f"cuda:{gpu_id}"), notes

    if device_count == 1 and gpu_id >= 0:
        notes.append(f"Mapped app GPU id {gpu_id} to PyTorch cuda:0.")
        return torch.device("cuda:0"), notes

    notes.append(f"Requested GPU {gpu_id} is not available to PyTorch. Using cuda:0 instead.")
    return torch.device("cuda:0"), notes


def _resolve_precision(precision: str | None, device: torch.device) -> tuple[str, torch.dtype | None]:
    if device.type != "cuda":
        return "fp32", None

    selected = (precision or "fp32").strip().lower()
    if selected == "fp16":
        return selected, torch.float16
    if selected == "bf16":
        return selected, torch.bfloat16
    return "fp32", None


def _ddcolor_hf_type():
    try:
        from ddcolor import DDColor
        from huggingface_hub import PyTorchModelHubMixin
    except ImportError as error:
        raise RuntimeError(
            "DDColor runtime dependencies are missing. Install python/requirements.txt to enable local colorization."
        ) from error

    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)

    return DDColorHF


def load_runtime_colorizer(
    model_id: str,
    gpu_id: int | None,
    precision: str | None,
    log: list[str],
    reference_image_paths: list[str] | None = None,
) -> LoadedDdcolorModel:
    del reference_image_paths
    ensure_runnable_model(model_id)
    if model_backend_id(model_id) != "pytorch-image-colorization":
        raise ValueError(f"Model '{model_id}' is not a DDColor-compatible colorizer")

    repo_id = SUPPORTED_DDCOLOR_MODELS.get(model_id)
    if repo_id is None:
        supported = ", ".join(sorted(SUPPORTED_DDCOLOR_MODELS))
        raise NotImplementedError(f"Colorizer '{model_id}' is not implemented. Supported colorizers: {supported}")

    device, notes = _resolve_device(gpu_id)
    precision_mode, autocast_dtype = _resolve_precision(precision, device)
    for note in notes:
        log.append(note)

    DDColorHF = _ddcolor_hf_type()
    model = DDColorHF.from_pretrained(repo_id)
    model = model.to(device)
    model.eval()
    log.append(f"Loaded DDColor checkpoint from Hugging Face: {repo_id}")

    return LoadedDdcolorModel(
        model_id=model_id,
        model_label=model_label(model_id),
        repo_id=repo_id,
        model=model,
        device=device,
        precision_mode=precision_mode,
        autocast_dtype=autocast_dtype,
        input_size=COLORIZER_INPUT_SIZE,
    )


def _colorize_image(loaded_model: LoadedDdcolorModel, img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None:
        raise ValueError("img is None (cv2.imread failed?)")

    height, width = img_bgr.shape[:2]
    img = (img_bgr / 255.0).astype(np.float32)
    orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
    img_resized = cv2.resize(img, (loaded_model.input_size, loaded_model.input_size))
    img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

    tensor_gray_rgb = (
        torch.from_numpy(img_gray_rgb.transpose((2, 0, 1)))
        .float()
        .unsqueeze(0)
        .to(loaded_model.device)
    )

    inference_context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=loaded_model.autocast_dtype)
        if loaded_model.device.type == "cuda" and loaded_model.autocast_dtype is not None
        else nullcontext()
    )
    with inference_context():
        with autocast_context:
            output_ab = loaded_model.model(tensor_gray_rgb)
        output_ab = output_ab.float().cpu()

    output_ab_resized = F.interpolate(output_ab, size=(height, width))[0].float().numpy().transpose(1, 2, 0)
    output_lab = np.concatenate((orig_l, output_ab_resized), axis=-1)
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    return (output_bgr * 255.0).round().astype(np.uint8)


def colorize_directory(
    *,
    loaded_model: LoadedDdcolorModel,
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
        pixels = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        colorized_pixels = _colorize_image(loaded_model, pixels)
        output_path = output_dir / frame_path.name
        if not cv2.imwrite(str(output_path), colorized_pixels):
            raise RuntimeError(f"Failed to write colorized frame '{output_path.name}'")
        if progress_callback is not None:
            progress_callback(index, total_frames)

    if loaded_model.device.type == "cuda":
        torch.cuda.synchronize(loaded_model.device)
    return total_frames