from __future__ import annotations

from dataclasses import dataclass
import importlib
import shutil
import sys
import tempfile
import types
from pathlib import Path

import torch
from PIL import Image

from upscaler_worker.cancellation import ensure_not_cancelled, wait_if_paused
from upscaler_worker.model_catalog import ensure_runnable_model, model_backend_id, model_label
from upscaler_worker.models.pytorch_sr import ensure_model_checkpoint
from upscaler_worker.runtime import ensure_colormnet_runtime


SUPPORTED_FRAME_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
COLORMNET_MODULE_PREFIXES = ("test_app", "model", "inference", "dataset", "util")


@dataclass
class LoadedColorMNetColorizer:
    model_id: str
    model_label: str
    checkpoint_path: Path
    source_root: Path
    entrypoint: Path
    reference_image_path: Path
    device: torch.device
    precision_mode: str
    test_module: object


class _ProgressTrackingLoader:
    def __init__(self, loader, callback=None):
        self._loader = loader
        self._callback = callback
        self._processed = 0
        try:
            self._total = len(loader)
        except TypeError:
            self._total = 0

    def __iter__(self):
        for item in self._loader:
            self._processed += 1
            if self._callback is not None:
                self._callback(self._processed, self._total)
            yield item

    def __len__(self) -> int:
        return self._total

    def __getattr__(self, name: str):
        return getattr(self._loader, name)


def _resolve_device(gpu_id: int | None) -> tuple[torch.device, list[str]]:
    notes: list[str] = []
    if not torch.cuda.is_available():
        raise RuntimeError("ColorMNet currently requires a CUDA-capable PyTorch runtime.")

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        raise RuntimeError("PyTorch did not report any CUDA devices for ColorMNet.")

    if gpu_id is None:
        return torch.device("cuda:0"), notes

    if 0 <= gpu_id < device_count:
        return torch.device(f"cuda:{gpu_id}"), notes

    if device_count == 1 and gpu_id >= 0:
        notes.append(f"Mapped app GPU id {gpu_id} to PyTorch cuda:0.")
        return torch.device("cuda:0"), notes

    notes.append(f"Requested GPU {gpu_id} is not available to PyTorch. Using cuda:0 instead.")
    return torch.device("cuda:0"), notes


def _resolve_reference_image_path(reference_image_paths: list[str] | None) -> Path:
    if not reference_image_paths:
        raise ValueError("ColorMNet requires exactly one selected reference image.")

    for raw_path in reference_image_paths:
        candidate = Path(raw_path)
        if candidate.exists():
            return candidate

    raise ValueError("ColorMNet could not find the selected reference image on disk.")


def _reset_colormnet_module_cache() -> None:
    prefixes = tuple(f"{prefix}." for prefix in COLORMNET_MODULE_PREFIXES)
    for module_name in list(sys.modules):
        if module_name in COLORMNET_MODULE_PREFIXES or module_name.startswith(prefixes):
            sys.modules.pop(module_name, None)


def _ensure_progressbar_module() -> None:
    if "progressbar" in sys.modules:
        return

    try:
        importlib.import_module("progressbar")
        return
    except ModuleNotFoundError:
        pass

    shim = types.ModuleType("progressbar")

    def progressbar(iterable, *args, **kwargs):
        del args, kwargs
        return iterable

    shim.progressbar = progressbar
    sys.modules["progressbar"] = shim


def _load_test_module(source_root: Path):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    _ensure_progressbar_module()
    _reset_colormnet_module_cache()
    return importlib.import_module("test_app")


def _frame_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_FRAME_SUFFIXES
    )


def _copy_frame_sequence(frame_paths: list[Path], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in frame_paths:
        shutil.copy2(frame_path, target_dir / frame_path.name)


def _resize_reference_image(reference_path: Path, output_path: Path, size: tuple[int, int]) -> None:
    with Image.open(reference_path) as image:
        resized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        resized.save(output_path)


def _build_args_list(*, checkpoint_path: Path, input_root: Path, output_root: Path, ref_root: Path) -> list[str]:
    return [
        "--model", str(checkpoint_path),
        "--d16_batch_path", str(input_root),
        "--ref_path", str(ref_root),
        "--output", str(output_root),
        "--dataset", "D16_batch",
        "--split", "val",
        "--save_all",
        "--FirstFrameIsNotExemplar",
        "--max_mid_term_frames", "10",
        "--min_mid_term_frames", "5",
        "--max_long_term_elements", "10000",
        "--num_prototypes", "128",
        "--top_k", "30",
        "--mem_every", "5",
        "--deep_update_every", "-1",
        "--size", "-1",
    ]


def load_runtime_colorizer(
    model_id: str,
    gpu_id: int | None,
    precision: str | None,
    log: list[str],
    reference_image_paths: list[str] | None = None,
) -> LoadedColorMNetColorizer:
    del precision
    ensure_runnable_model(model_id)
    if model_backend_id(model_id) != "pytorch-image-colorization":
        raise ValueError(f"Model '{model_id}' is not a ColorMNet-compatible colorizer")
    if model_id != "colormnet":
        raise NotImplementedError(f"Colorizer '{model_id}' is not implemented")

    runtime_assets = ensure_colormnet_runtime()
    source_root = Path(runtime_assets["colormnetSourceRoot"])
    entrypoint = Path(runtime_assets["colormnetEntryPoint"])
    checkpoint_path = ensure_model_checkpoint(model_id)
    reference_image_path = _resolve_reference_image_path(reference_image_paths)
    device, notes = _resolve_device(gpu_id)
    for note in notes:
        log.append(note)

    test_module = _load_test_module(source_root)
    log.append(f"Loaded ColorMNet checkpoint from {checkpoint_path}")
    log.append(f"Using ColorMNet exemplar {reference_image_path.name}")

    return LoadedColorMNetColorizer(
        model_id=model_id,
        model_label=model_label(model_id),
        checkpoint_path=checkpoint_path,
        source_root=source_root,
        entrypoint=entrypoint,
        reference_image_path=reference_image_path,
        device=device,
        precision_mode="fp32",
        test_module=test_module,
    )


def colorize_directory(
    *,
    loaded_model: LoadedColorMNetColorizer,
    input_dir: Path,
    output_dir: Path,
    cancel_path: str | None,
    pause_path: str | None,
    progress_callback=None,
) -> int:
    ensure_not_cancelled(cancel_path)
    wait_if_paused(pause_path, cancel_path=cancel_path)

    frame_paths = _frame_paths(input_dir)
    if not frame_paths:
        raise RuntimeError("No extracted frames were found for ColorMNet colorization.")

    output_dir.mkdir(parents=True, exist_ok=True)
    clip_name = input_dir.name
    with Image.open(frame_paths[0]) as first_frame:
        frame_size = first_frame.size

    with tempfile.TemporaryDirectory(prefix="colormnet-", dir=output_dir.parent) as temp_dir:
        work_root = Path(temp_dir)
        input_root = work_root / "input_video"
        ref_root = work_root / "ref"
        output_root = work_root / "output"
        clip_input_dir = input_root / clip_name
        clip_ref_dir = ref_root / clip_name
        clip_output_dir = output_root / clip_name
        clip_ref_dir.mkdir(parents=True, exist_ok=True)
        _copy_frame_sequence(frame_paths, clip_input_dir)
        _resize_reference_image(loaded_model.reference_image_path, clip_ref_dir / "ref.png", frame_size)

        original_loader = loaded_model.test_module.DataLoader

        def patched_dataloader(*args, **kwargs):
            kwargs["num_workers"] = 0
            loader = original_loader(*args, **kwargs)
            return _ProgressTrackingLoader(loader, progress_callback)

        args_list = _build_args_list(
            checkpoint_path=loaded_model.checkpoint_path,
            input_root=input_root,
            output_root=output_root,
            ref_root=ref_root,
        )

        if loaded_model.device.index is not None:
            torch.cuda.set_device(loaded_model.device.index)

        loaded_model.test_module.DataLoader = patched_dataloader
        try:
            loaded_model.test_module.run_cli(args_list)
        finally:
            loaded_model.test_module.DataLoader = original_loader

        generated_frames = _frame_paths(clip_output_dir)
        if not generated_frames:
            raise RuntimeError("ColorMNet completed without writing any output frames.")

        for output_frame in generated_frames:
            ensure_not_cancelled(cancel_path)
            wait_if_paused(pause_path, cancel_path=cancel_path)
            shutil.copy2(output_frame, output_dir / output_frame.name)

    if progress_callback is not None:
        progress_callback(len(frame_paths), len(frame_paths))
    torch.cuda.synchronize(loaded_model.device)
    torch.cuda.empty_cache()
    return len(frame_paths)